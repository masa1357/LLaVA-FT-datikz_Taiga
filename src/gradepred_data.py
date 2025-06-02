# =====================================================================
# gradepred_data.py
# date: 2025/05/05
# description:
#   - Reflection データセット内の任意のテキストから成績を予測するデータセット
# =====================================================================
import logging
from pathlib import Path
import re, unicodedata
import pandas as pd
from torch.utils.data import Dataset
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
            "userid" : "user_id",
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

    def __init__(
        self,
        dataset_path: Path,
        logger: logging.Logger | None = None,
        merge_key: str = "userid",
        fill_token: str = "NaN",
        answer_col: str = "answer_content",
        question_filter: Sequence[int] | None = None,
        concatenate: bool = False,
        valid_ratio: float = 0.2,
        random_state: int = 42,
        mode: str = "all",
        testcase: bool = False,
    ):
        """
        ファイルを読み込み，データセットを構成する
        """
        self.logger = logger or logging.getLogger(__name__)
        self.fill_token = fill_token

        self.answer_col = answer_col
        self.concat = concatenate
        # フィルタが None→全質問(1-5)、指定がある→重複排除 + ソート
        self.q_filter = (
            sorted(set(question_filter)) if question_filter else list(range(1, 6))
        )

        if testcase:
            # ----------------- テストデータセット -----------------
            # JSQuAD のテストデータセットを使用
            from datasets import load_dataset

            ds = load_dataset("shunk031/JGLUE", name="JSQuAD", trust_remote_code=True)
            ds = ds["validation"]
            # 実際に使うデータと同じ形式にするために，データを調整
            # 1. id -> userid
            # 2. context + question -> input_text
            # 3. answers["text"] -> grades, labels
            dummy_samples = []
            for uid, ctx, q, a in zip(
                ds["id"], ds["context"], ds["question"], ds["answers"]
            ):
                dummy_samples.append(
                    {
                        "userid": uid,
                        "input_text": f"{ctx}\n{q}",
                        "grades": str(a["text"][0]),
                        "labels": 0,
                    }
                )
            self.dataset = dummy_samples

            # datasetsの要素数を削減 (500sample)
            self.dataset = self.dataset[:500]

        else:
            # ----------------- データ読込 -----------------
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

            # ----------- 前処理 & ネスト構築 -----------
            df[answer_col] = df[answer_col].apply(self._preprocess)
            self.dataset = self._build_nested(df)

        self.logger.info(
            f"Dataset init done. users={len(self.dataset)}, qs={self.q_filter}, "
            f"concat={self.concat}"
        )

        # -------------------------------------------------
        # 8:2 の層化分割
        # -------------------------------------------------
        labels = [s["labels"] for s in self.dataset]

        train_idx, valid_idx = train_test_split(
            range(len(self.dataset)),
            test_size=valid_ratio,
            stratify=labels,
            random_state=random_state,
        )

        match mode:
            case "train":
                self.train_dataset = [self.dataset[i] for i in train_idx]
                self.dataset = self.train_dataset
            case "valid":
                self.valid_dataset = [self.dataset[i] for i in valid_idx]
                self.dataset = self.valid_dataset
            case "all":
                pass
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        self.logger.info(f"Dataset build done: total={len(self.dataset)}, ")

        # df = pd.merge(left, right, on=merge_key, how="inner")
        # df["label"] = df["grade"].map(label_map)

        # # 前処理
        # df[answer_col] = df[answer_col].apply(self._preprocess)

        # # データセットを辞書型として構築，より構造的に，呼び出しやすい形式にする
        # self.dataset = self._build_nested(df, answer_col, fill_token)
        # self.logger.info(
        #     f"GradePredictionDataset initialized with {len(self.dataset)} samples"
        # )

    # -------- Dataset インタフェース --------
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[idx]

    # ──────────────────────────────────────────────
    # 特定の質問番号を抽出する関数
    # ──────────────────────────────────────────────
    # def subset_by_questions(
    #     self,
    #     q_numbers: list[int] | tuple[int, ...],
    #     *,
    #     concatenate: bool = False,
    #     sep: str = " ",
    # ) -> list[dict]:
    #     """
    #     指定した質問番号 (1-5) だけを残した新しいサンプル集合を返す。

    #     Parameters
    #     ----------
    #     q_numbers    : 取得したい質問番号のリスト/タプル  e.g. [2,4]
    #     concatenate  : True にすると L1〜L15 の回答を串刺しで連結し
    #                    'input_text' キー１本にまとめて返す
    #     sep          : concatenate=True のとき回答間をつなぐ区切り文字列

    #     Returns
    #     -------
    #     samples      : list[dict]
    #     """
    #     qs = {f"Q{n}" for n in q_numbers}
    #     out = []

    #     for sample in self.dataset:
    #         new_sample = {"userid": sample["userid"], "labels": sample["labels"]}

    #         if concatenate:
    #             parts = []
    #             for c in range(1, 16):
    #                 ldict = sample[f"L{c}"]
    #                 # Qx が欠損なら fill_token が入っているのでそのまま使う
    #                 for q in qs:
    #                     parts.append(ldict[q])
    #             new_sample["input_text"] = sep.join(parts)
    #         else:
    #             # ネスト構造を維持して部分的に残す
    #             for c in range(1, 16):
    #                 new_sample[f"L{c}"] = {q: sample[f"L{c}"][q] for q in qs}
    #         out.append(new_sample)
    #     return out

    # -------- 内部ユーティリティ --------
    def _read_folder(self, path: Path) -> pd.DataFrame:
        files = list(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"{path} に CSV がありません")
        return pd.concat(map(pd.read_csv, files), ignore_index=True)

    # def read_folder(self, path: Path, rules: str = "*.csv") -> pd.DataFrame:
    #     """
    #     フォルダ内のファイルを全て読み込み，縦方向に結合する

    #     Parameters
    #     ----------
    #     path  : str
    #         フォルダパス
    #     rules : str
    #         読み込むフォルダの形式（デフォルトではcsvファイルすべて）

    #     Returns
    #     -------
    #     df : DataFrame
    #         読み込んだ全てのcsvを縦に結合したdf.
    #     """
    #     self.logger.info(f"Read {path} {rules} data...")
    #     file_list = list(path.glob(rules))
    #     self.logger.info(f"Found files: {file_list}")

    #     df = pd.DataFrame()
    #     if file_list:
    #         for file in file_list:
    #             temp_df = pd.read_csv(file)
    #             df = pd.concat([df, temp_df], axis=0, ignore_index=True)
    #         self.logger.info(f"Total rows: {len(df)}")
    #     else:
    #         self.logger.error(f"No {rules} files in {path}")

    #     return df

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
        DataFrame（userid, question_number, course_number, answer_content, label）
        ──▶ ユーザ単位のネスト辞書リストに変換
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
            if self.concat:
                # 連結テキストモード
                sep = "\n"
                lines: list[str] = []
                for c in range(1, 16):
                    for qn in self.q_filter:
                        ans = (
                            block.loc[(uid, c), f"Q{qn}"]
                            if (uid, c) in block.index
                            else self.fill_token
                        )
                        lines.append(f"L{c:02d}-Q{qn}: {ans}")
                samples.append(
                    {
                        "userid": uid,
                        "labels": int(label_val),
                        "grades": str(grade_val),
                        "input_text": sep.join(lines),
                    }
                )
            else:
                # ネスト保持モード
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

        # # ── ① 期待される全組み合わせを用意し欠損を明示 ──
        # idx = pd.MultiIndex.from_product(
        #     [df["userid"].unique(), range(1, 16), range(1, 6)],
        #     names=["userid", "course_number", "question_number"],
        # )
        # df_full = (
        #     df.set_index(["userid", "course_number", "question_number"])
        #     .reindex(idx)  # 存在しない行を補完
        #     .reset_index()
        # )

        # # ── ② pivot で Lx–Qy テーブル化 ──
        # pivot = (
        #     df_full.pivot_table(
        #         index=["userid", "course_number"],
        #         columns="question_number",
        #         values=answer_col,
        #         aggfunc="first",
        #     )
        #     .rename(columns=lambda q: f"Q{int(q)}")
        #     .fillna(fill_token)
        # )

        # # ── ③ ユーザごとにネスト辞書を構築 ──
        # result = []
        # for userid, user_block in pivot.groupby(level=0):
        #     user_dict = {"userid": userid}

        #     # 各コース L1〜L15
        #     for c in range(1, 16):
        #         key = f"L{c}"
        #         if (userid, c) in user_block.index:
        #             user_dict[key] = user_block.loc[(userid, c)].to_dict()
        #         else:  # そのコースまるごと欠損
        #             user_dict[key] = {f"Q{q}": fill_token for q in range(1, 6)}

        #     # labels（userid ごとに同じと仮定）
        #     label_val = df.loc[df["userid"] == userid, "label"].iloc[
        #         0
        #     ]  # ない場合は KeyError。必要なら try/except で None を許容
        #     user_dict["labels"] = int(label_val)

        #     result.append(user_dict)

        # return result


def collate_fn(
    batch,
    tokenizer,
    max_tokens: int = 4096,
    question_filter: list[int] | None = [1, 2, 3, 4, 5],
    logger: logging.Logger | None = None,
    include_target=True,
    testcase=False,
):
    logger = logger or logging.getLogger(__name__)

    Q_TEXT = {
        1: "Q1:今日の内容を自分なりの言葉で説明してみてください\n",
        2: "Q2:今日の内容で、分かったこと・できたことを書いてください\n",
        3: "Q3:今日の内容で、分からなかったこと・できなかったことを書いてください\n",
        4: "Q4:質問があれば書いてください\n",
        5: "Q5:今日の授業の感想や反省を書いてください\n",
    }
    question = "".join(Q_TEXT[q] for q in question_filter)

    if testcase:
        preamble = ("")
    else:
        preamble = (
            "あなたは大学の教授であり，学生の成績を決定する役割を担っています。"
            "以下に示す学生の講義後アンケートを読み，成績を A, B, C, D, F のいずれかに分類してください。\n"
            "L は講義回，Q は質問番号を示します（例: L1-Q1）。\n"
            f"アンケートの質問文は，\n{question}\nです．"
            "回答が NaN の場合は未回答です。\n"
            "上記を踏まえ，出力には A/B/C/D/F のいずれか **1 文字のみ** を返してください。\n"
            "アンケート内容："
        )


    # ----- 入力文とターゲットを作成 -----
    sources, targets = [], []
    for sample in batch:
        survey = sample["input_text"]
        grade = sample["grades"]  # 'A'~'F' (str)

        # Llama-3 のチャット書式: <s>[INST] system + user [/INST] assistant
        #   - BOS (<s>) は tokenizer.bos_token で自動付与されるので add_special_tokens=True で任せる
        prompt = f"[INST] {preamble}\n\n{survey}\n[/INST]\n"
        answer = f" {grade}"  # 先頭スペースは BPE で単語境界を作るため

        sources.append(prompt)
        targets.append(answer)

    logger.info("Collate: %d samples, max_tokens=%d", len(batch), max_tokens)
    logger.debug("prompt sample:\n%s", sources[0][:500])

    # ----- トークナイズ＆教師ラベル作成 -----
    input_ids, label_ids, attn_masks = [], [], []
    for src, tgt in zip(sources, targets):
        # add_special_tokens=False → 自前で eos を足す
        prompt_ids = tokenizer.encode(src, add_special_tokens=False)
        target_ids = tokenizer.encode(tgt, add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]

        # 長さ制御（truncate は “プロンプト側” を先に削る）
        if len(prompt_ids) + len(target_ids) > max_tokens:
            logger.debug(
                "Truncating prompt: %d + %d > %d",
                len(prompt_ids),
                len(target_ids),
                max_tokens,
            )
            prompt_ids = prompt_ids[: max_tokens - len(target_ids)]

        if include_target:
            ids = prompt_ids + target_ids
        else:
            ids = prompt_ids + [tokenizer.eos_token_id]

        labels = [-100] * len(prompt_ids) + target_ids  # prompt は損失計算しない
        attn_mask = [1] * len(ids)

        input_ids.append(torch.tensor(ids, dtype=torch.long))
        label_ids.append(torch.tensor(labels, dtype=torch.long))
        attn_masks.append(torch.tensor(attn_mask, dtype=torch.long))

    # ----- Padding -----
    pad_id = tokenizer.pad_token_id
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    attention_mask = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    logger.info("Tokenize done: ")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "grades": targets,
    }
