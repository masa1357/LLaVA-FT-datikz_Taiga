# =====================================================================
# gradepred_data.py
# date: 2025/07/09
# description:
#   - Reflection データセット内の任意のテキストから成績を予測するデータセット
#   - cross validationに対応
# =====================================================================
import logging
from pathlib import Path
import re, unicodedata
import pandas as pd
from torch.utils.data import Dataset, Subset 
from typing import Any, Iterable, Sequence
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

WS_RE = re.compile(r"[ \u3000\r\n\t]+")  # \u3000 = 全角スペース
# 句読点変換テーブル
PUNCT_TABLE = str.maketrans(
    {
        "，": "、",  # 全角カンマ → 読点
        ",": "、",  # 半角カンマ → 読点
        "．": "。",  # 全角ピリオド → 句点
        ".": "。",  # 半角ピリオド → 句点
    }
)

label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}


class GradePredictionDataset(Dataset):
    """
    Reflection データセット内の任意のテキストから成績を予測するデータセット型
    構造：
        {
            "userid": "user_id",
            "L1": {
                "Q1": "Q1の回答",
                "Q2": "Q2の回答",
                "Q3": "Q3の回答",
                "Q4": "Q4の回答",
                "Q5": "Q5の回答",
            }
            ...
            "L15": {
                "Q1": "Q1の回答",
                "Q2": "Q2の回答",
                "Q3": "Q3の回答",
                "Q4": "Q4の回答",
                "Q5": "Q5の回答",
            }
            "labels": labels (0~4)
            "grades": grades (A,B,C,D,F)
        }
    """

    # -------- Dataset インタフェース --------
    def __init__(self,
                 dataset_path: Path,
                 logger: logging.Logger | None = None,
                 fill_token: str = "NaN",
                 answer_col: str = "answer_content",
                 question_filter: Sequence[int] | None = None,
                 merge_key: str = "userid",
                 ):
        """
        ファイルを読み込み，データセットを構成する
        """
        self.logger = logger or logging.getLogger(__name__)
        self.fill_token = fill_token

        self.answer_col = answer_col
        self.dataset_path = dataset_path

        # フィルタが None→全質問(1-5)、指定がある→重複排除 + ソート
        self.q_filter = (
            sorted(set(question_filter)) if question_filter else list(range(1, 6))
        )

        self.reflection_path = Path(dataset_path) / "Reflection"
        self.grade_path = Path(dataset_path) / "Grade"
        for p in (self.reflection_path, self.grade_path):
            if not p.is_dir():
                raise FileNotFoundError(f"{p} が存在しません")

        left = self._read_folder(self.reflection_path)
        right = self._read_folder(self.grade_path)

        required_cols = {"userid", "course_number", "question_number", answer_col}

        if not required_cols.issubset(left.columns):
            raise KeyError(f"reflection csv に {required_cols} がありません")
        df = pd.merge(left, right, on=merge_key, how="inner")
        df["label"] = df["grade"].map(label_map)
        # df["userid"]内のユニーク値をリストにして保持（昇順）
        user_ids = (
            df["userid"]
            .dropna()            # 欠損を除外
            .unique()            # 重複排除
        )
        self.user_ids = sorted(user_ids.tolist())  # list 化して昇順ソート

        # ----------- 前処理 & ネスト構築 -----------
        df[answer_col] = df[answer_col].apply(self._preprocess)
        self.raw_dataset = self._build_nested(df)
        self.dataset = self.raw_dataset.copy()

        self.logger.info(
            f"Dataset init done. users={len(self.dataset)}, qs={self.q_filter}, "
        )

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx:int):
        return self.dataset[idx]
    


    # -------- 内部ユーティリティ --------
    def reset(self):
        """データセットの形状をリセット"""
        self.logger.info("reset dataset!")
        self.dataset = self.raw_dataset


    def concat(self):
        # 連結テキストモード
        self.logger.info("simple sentence mode...")
        self.dataset: list[dict[str, Any]] = []
        sep = "\n"
        for sample in self.raw_dataset:
            lines: list[str] = []
            for c in range(1, 16):
                for qn in self.q_filter:
                    ans = sample[f"L{c}"][f"Q{qn}"]
                    lines.append(f"L{c:02d}-Q{qn}: {ans}")
            self.dataset.append(
                {
                    "userid": sample["userid"],
                    "labels": sample["labels"],
                    "grades": sample["grades"],
                    "input_text": sep.join(lines),
                }
            )

    def unzip(self):
        # 分割テキストモード
        self.logger.info("15 * 5 sentence mode...")
        self.dataset = []  # 元のデータセットをクリア
        for sample in self.raw_dataset:
            userid = sample["userid"]
            grades = sample["grades"]
            labels = sample["labels"]
            for c in range(1, 16):
                for q in self.q_filter:
                    course_key = f"L{c}"
                    question_key = f"Q{q}"
                    ans = sample[course_key][question_key]
                    self.dataset.append(
                        {
                            "userid": userid,
                            "labels": labels,
                            "grades": grades,
                            "input_text": f"{course_key}-{question_key}: {ans}",
                        }
                    )

    def extention(self):
        # 学習用データセットに拡張データを追加
        self.logger.info("use extention data...")
        ext_df = pd.read_csv(Path(self.dataset_path) / "extdata.csv")
        for _, row in ext_df.iterrows():
            # 拡張データの行を追加
            
            # raw["grade"]に，(0~4)のラベルが含まれている
            # このとき，4:A, 3:B, 2:C, 1:D, 0:F に対応しているため，
            # label : 4->0 , 3->1, 2->2, 1->3, 0->4と変換，
            ext_label_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
            label = ext_label_map.get(row["grade"], 4)  # デフォルトは4 (F)
            ext_graded_map = {
                0: "F",
                1: "D",
                2: "C",
                3: "B",
                4: "A",
            }
            # grades : 4->A, 3->B, 2->C, 1->D, 0->Fと変換
            grade = ext_graded_map.get(row["grade"], "F")  # デフォルトはF

            self.dataset.append(
                {
                    "userid": "ext_user",  # 拡張データのユーザID
                    "labels": label,
                    "grades": grade,
                    "input_text": row["answer"],
                }
            )


    def _read_folder(self, path: Path) -> pd.DataFrame:
        """
        指定されたパス内の CSV ファイル群を読み込み，縦にconcatする
        Parameters
        ----------
        path : Path
            読み込むフォルダのパス

        Returns
        -------
        pd.DataFrame
            読み込んだ CSV ファイルを縦に結合した DataFrame
        """
        files = list(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"{path} に CSV がありません")
        return pd.concat(map(pd.read_csv, files), ignore_index=True)

    def _preprocess(self, text: str | None) -> str:
        """
        前処理を行う
        前処理のプロセスは，LLMに入力することを考慮し，表記ゆれの修正に留める（ノイズの除去）
        1．Unicode正規化(NFKC正規化)
            -> 全角・半角や濁点付きの結合文字を統一
        2．句読点の統一
            -> 句点は「。」，読点は「、」
        3．スペース，改行の統一
            -> スペース，改行は半角スペースに統一

        Parameters
        ----------
        text : str
            前処理を行うテキスト
        Returns
        -------
        text : str
            前処理済みテキスト

        """

        # 欠損チェック（pandas 系 NaN も含む）
        if pd.isna(text) or text == "":
            return self.fill_token

        # 1. Unicode正規化
        text = unicodedata.normalize("NFKC", text)

        # 2. 句読点の統一
        text = text.translate(PUNCT_TABLE)

        # 3. スペース，改行の統一
        # 空白・改行系をまとめて空白1つにする正規表現
        WS_RE = re.compile(r"[ \u3000\r\n\t]+")  # \u3000 = 全角スペース
        text = WS_RE.sub(" ", text).strip()

        return text

    def _build_nested(self, df: pd.DataFrame) -> list[dict]:
        """
        DataFrame をユーザ単位のネスト辞書リストに変換する
        Parameters
        ----------
        df : pd.DataFrame
            ユーザのアンケート回答データ
            必須カラム：
                - userid: ユーザID
                - question_number: 質問番号 (1-5)
                - course_number: 講義回 (1-15)
                - answer_content: 回答内容
                - grade: 成績 (A, B, C, D, F)
                - label: ラベル (0~4)
        Returns
        -------
        samples : list[dict]
            ユーザ単位のネスト辞書リスト
            各辞書は以下の形式：
            {
                "userid": str,      # ユーザID
                "labels": int,      # ラベル (0~4)
                "grades": str,      # 成績 (A, B, C, D, F)
                "L1": {             # 講義回1の回答
                    "Q1": str,     # Q1の回答
                    "Q2": str,     # Q2の回答
                    ...
                },
                ...
            }

        """

        # フィルタ掛け
        df = df[df["question_number"].isin(self.q_filter)].copy()

        # 欠損行を補完する MultiIndex
        idx = pd.MultiIndex.from_product(
            [df["userid"].unique(), range(1, 16), self.q_filter],
            names=["userid", "course_number", "question_number"],
        )
        df_full = (
            df.set_index(["userid", "course_number", "question_number"])
            .reindex(idx)
            .reset_index()
        )

        # pivot (course, question → 列 Qx)
        pivot = (
            df_full.pivot_table(
                index=["userid", "course_number"],
                columns="question_number",
                values=self.answer_col,
                aggfunc="first",
            )
            .rename(columns=lambda q: f"Q{int(q)}")
            .fillna(self.fill_token)
        )

        samples: list[dict[str, Any]] = []
        qs_cols = [f"Q{n}" for n in self.q_filter]

        for uid, block in pivot.groupby(level=0):
            grade_val = df.loc[df["userid"] == uid, "grade"].iloc[0]
            label_val = df.loc[df["userid"] == uid, "label"].iloc[0]

            entry: dict[str, Any] = {
                "userid": uid,
                "labels": int(label_val),
                "grades": str(grade_val),
            }
            for c in range(1, 16):
                key = f"L{c}"
                if (uid, c) in block.index:
                    entry[key] = block.loc[(uid, c), qs_cols].to_dict()
                else:
                    entry[key] = {q: self.fill_token for q in qs_cols}
            samples.append(entry)
        return samples


class ext_GPDataset(GradePredictionDataset):
    """
    拡張データだけ入手（あとでtrain datasetに結合）
    """
    def __init__(self, dataset_path: Path):
        # super().__init__()
        self.dataset_path = dataset_path
        self.dataset: list[dict[str, Any]] = []
        self.extention()
        

class GradePredictionCollator:
    """
    GradePredictionDataset 用の collate_fn を作成
    huggingface Trainer の DataCollator 互換のインタフェース
    huggingface>DataCollatorForSeq2Seq
    十分条件；
    features    : Dict[

    ]
    return      : Dict[
        "input_ids": torch.Tensor,
        "attention_mask": torch.Tensor,
        "labels": torch.Tensor,
    ]
    """

    def __init__(
        self,
        tokenizer,
        max_tokens: int = 4096,
        logger: logging.Logger | None = None,
        question_filter: list[int] | None = [1, 2, 3, 4, 5],
        include_target: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            トークナイザ
        max_tokens : int, optional
            最大トークン数（デフォルトは4096）
        logger : logging.Logger, optional
            ロガー（デフォルトはNone）
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.include_target = include_target
        self.logger = logger or logging.getLogger(__name__)

        Q_TEXT = {
            1: "Q1:今日の内容を自分なりの言葉で説明してみてください\n",
            2: "Q2:今日の内容で、分かったこと・できたことを書いてください\n",
            3: "Q3:今日の内容で、分からなかったこと・できなかったことを書いてください\n",
            4: "Q4:質問があれば書いてください\n",
            5: "Q5:今日の授業の感想や反省を書いてください\n",
        }
        question = "".join(Q_TEXT[q] for q in question_filter)
        logger.info(f"Questions: {question_filter}, {question}")

        self.preamble = (
            "あなたは大学の教授であり、学生の成績を決定する役割を担っています。"
            "以下に示す学生の講義後アンケートを読み、成績を高い順に A、B、C、D、F のいずれかに分類してください。\n"
            "成績は出力例のような形式で出力してください。\n"
            "入力文のL は講義回、Q は質問番号を示します（例: L1-Q1）。\n"
            f"アンケートの質問文は、\n{question}です。"
            "回答が NaN の場合は未回答です。\n"
            "出力には A、B、C、D、F のいずれかを含めてください。\n"
            "出力例:\n"
            "この学生の成績は、Aです。\n"
            "アンケート内容："
        )
        logger.info(
            f"preamble set:\n================\n{self.preamble}\n================\n"
        )

    def __call__(self, features: Iterable[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        LLMに入力するため，データセットの整形，トークナイズを行う

        Parameters
        ----------
        features : Iterable[dict[str, Any]]
            データセットのサンプル（辞書型）のイテラブル
            {
                "userid": str,      # ユーザID
                "input_text": str,  # アンケート内容
                "grades": str,      # 成績（A, B, C, D, F）
                "labels": int,      # ラベル（0~4）
            }
        Returns
        -------
        Dict[str, torch.Tensor]
            トークナイズされた入力データとラベル
            {
                "input_ids": torch.Tensor,
                "attention_mask": torch.Tensor,
                "labels": torch.Tensor,
            }
        """
        prompts = [
            f"[INST] {self.preamble}\n\n{ex['input_text']}\n[/INST]\n"
            for ex in features
        ]
        grades = [ex["grades"] for ex in features]
        tgt_str = [f" この学生の成績は、{g}です。" for g in grades]

        self.logger.debug("prompt sample:\n%s", prompts[0][:500])
        self.logger.debug("grade sample: %s", tgt_str[0][:50])

        enc_prompt = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_tokens,
            padding=False,
        )

        enc_tgt = self.tokenizer(tgt_str, add_special_tokens=False, padding=False)

        self.logger.debug("Tokenization Done")

        input_ids, labels = [], []
        for p_ids, t_ids in zip(enc_prompt["input_ids"], enc_tgt["input_ids"]):
            if len(p_ids) + len(t_ids) + 1 > self.max_tokens:
                p_ids = p_ids[: self.max_tokens - len(t_ids) - 1]
            eos = [self.tokenizer.eos_token_id]
            if self.include_target:
                ids = p_ids + t_ids + eos
                lbl = [-100] * len(p_ids) + t_ids + eos
            else:
                ids = p_ids + eos
                lbl = [-100] * len(ids)
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            labels.append(torch.tensor(lbl, dtype=torch.long))

        # もし左パディングなら
        if self.tokenizer.padding_side == "left":
            max_len = max(len(ids) for ids in input_ids)
            batch_ids = torch.full(
                (len(input_ids), max_len), self.tokenizer.pad_token_id, dtype=torch.long
            )
            batch_lbl = torch.full((len(labels), max_len), -100, dtype=torch.long)

            for i, (ids, lbl) in enumerate(zip(input_ids, labels)):
                batch_ids[i, -len(ids) :] = torch.tensor(ids)
                batch_lbl[i, -len(lbl) :] = torch.tensor(lbl)

            attn_mask = batch_ids.ne(self.tokenizer.pad_token_id).long()

        else:
            batch_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch_lbl = pad_sequence(labels, batch_first=True, padding_value=-100)
            attn_mask = batch_ids.ne(self.tokenizer.pad_token_id).long()

        self.logger.debug("Collating Done")

        self.logger.debug(
            "Collate: %d samples, max_tokens=%d",
            len(features),
            self.max_tokens,
        )

        # enc_tgt["input_ids"] をtorch.tensorに変換して保持
        target_ids = torch.tensor(enc_tgt["input_ids"], dtype=torch.long)

        return {
            "input_ids": batch_ids,
            "attention_mask": attn_mask,
            "labels": batch_lbl,
            "target_ids": target_ids,  #!<追加> ★ 真値を保持
        }

    def get_prompt(self, features: dict[str, Any]) -> str:
        """
        プロンプトを確認
        """
        prompts = [
            f"[INST] {self.preamble}\n\n{ex['input_text']}\n[/INST]\n"
            for ex in features
        ]

        return prompts
