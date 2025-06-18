# =====================================================================
# gradeexplanation_data.py
# date: 2025/06/17
# description:
#   - Reflection データセット内の任意のテキストから成績とその根拠を予測するデータセットクラス
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


class GradeExplanationDataset(Dataset):
    """
    Reflection データセット内の任意のテキストから成績+根拠を予測するデータセット型
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
            "grades": grades (A,B,C,D,F)
            "labels": text(grades + 根拠説明)
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
        division: bool = False,
        add_extended: bool = False,
        tokenizer=None,
        max_tokens: int = 4096 - 330,  # プロンプト長(330)を減算
        trim: bool = True,
    ):
        """
        ファイルを読み込み，データセットを構成する
        """
        self.logger = logger or logging.getLogger(__name__)
        self.fill_token = fill_token

        self.answer_col = answer_col
        self.concat = concatenate
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

        # フィルタが None→全質問(1-5)、指定がある→重複排除 + ソート
        self.q_filter = (
            sorted(set(question_filter)) if question_filter else list(range(1, 6))
        )

        # ================================================================
        # デフォルトのデータセット読み込み
        # dataset_path/Reflection   : 生徒のアンケート回答
        # dataset_path/Grade        : 成績データ
        # ================================================================
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
        if trim:
            self.trim_dataset()  # 各回答を最大トークン数に収まるように切り詰める

        self.logger.info(
            f"Dataset init done. users={len(self.dataset)}, qs={self.q_filter}, "
            f"concat={self.concat}"
        )

        # ================================================================
        # 層化分割（デフォルト 8:2）
        # ================================================================
        labels = [s["labels"] for s in self.dataset]

        train_idx, valid_idx = train_test_split(
            range(len(self.dataset)),
            test_size=valid_ratio,
            stratify=labels,  #! ラベル分布を保持
            random_state=random_state,
        )

        # ================================================================
        # モードに応じたデータセットを出力
        # mode: "train", "valid", "all"
        #   - "train"  : 学習用データセット
        #   - "valid"  : 検証用データセット
        #   - "all"    : 全データセット（学習・検証両方）
        # 呼び出し側で各モードに応じてインスタンス化
        #  - 例:
        #       `GradePredictionDataset(..., mode="train")`
        #       `GradePredictionDataset(..., mode="valid")`
        #   -> random_stateを統一しないとリークするので注意
        # ================================================================
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

        if division:
            # 各データセット内に含まれるL,Qをそれぞれの行として変換する
            # 例:
            #   {
            # "userid"      : "user_id",
            # "labels"      : int,      # ラベル (0~4)
            # "grades"      : str,      # 成績 (A, B, C, D, F)
            # "input_text"  : str,      # アンケート内容 (Lx-Qy: 回答内容)
            #   }
            # -> 各カラムを15(L1~L15) * 5(Q1) の行に展開する
            self.logger.info("Dividing dataset into Lx-Qy format...")
            tmp = self.dataset.copy()
            self.dataset = []  # 元のデータセットをクリア
            for sample in tmp:
                userid = sample["userid"]
                grades = sample["grades"]
                labels = sample["labels"]
                for c in range(1, 16):
                    for q in self.q_filter:
                        cource_key = f"L{c}"
                        question_key = f"Q{q}"
                        ans = sample[cource_key][question_key]
                        self.dataset.append(
                            {
                                "userid": userid,
                                "labels": labels,
                                "grades": grades,
                                "input_text": f"{cource_key}-{question_key}: {ans}",
                            }
                        )
            if mode == "train" and add_extended:
                # 学習用データセットに拡張データを追加
                ext_df = pd.read_csv(Path(dataset_path) / "extdata.csv")
                for _, row in ext_df.iterrows():
                    # 拡張データの行を追加
                    # raw["grade"]に，(0~4)のラベルが含まれている
                    # このとき，4:A, 3:B, 2:C, 1:D, 0:F に対応しているため，
                    # label : 4->0 , 3->1, 2->2, 1->3, 0->4と変換，
                    # grades : 4->A, 3->B, 2->C, 1->D, 0->Fと変換
                    ext_label_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
                    label = ext_label_map.get(row["grade"], 4)  # デフォルトは4 (F)
                    ext_graded_map = {
                        0: "F",
                        1: "D",
                        2: "C",
                        3: "B",
                        4: "A",
                    }
                    grade = ext_graded_map.get(row["grade"], "F")  # デフォルトはF

                    self.dataset.append(
                        {
                            "userid": "ext_user",  # 拡張データのユーザID
                            "labels": label,
                            "grades": grade,
                            "input_text": row["answer"],
                        }
                    )

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

    # -------- 内部ユーティリティ --------
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
        if concat=True:
            samples : list[dict]
                ユーザ単位の辞書リスト
                各辞書は以下の形式：
                {
                    "userid": str,      # ユーザID
                    "labels": int,      # ラベル (0~4)
                    "grades": str,      # 成績 (A, B, C, D, F)
                    "input_text": str,
                        # アンケート内容を連結したテキスト
                        # 各講義回の回答を "L1-Q1: 回答内容" の形式で連結
                }

        else:
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

            # ================================================================
            # 連結するか，ネスト構造を保持するか
            # concat=True なら，テキストを連結して1つの文字列にする
            # concat=False なら，ネスト構造を保持して辞書型で返す
            # LLMへの入力のため，concat=Trueにしてプロンプト化する
            # ================================================================
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

    def trim_dataset(self) -> None:
        """
        データセット内の各回答を最大トークン数に収まるように切り詰める
        L1-Q1 - L15-Q5までのそれぞれの文章長を取得し，各文章を最大トークン数に収まるように切り詰める
        Parameters
        ----------
        max_tokens : int
            切り詰める最大トークン数
        dataset : list[dict[str, Any]]
            対象データセット
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
        Returns
        -------
        dataset : list[dict[str, Any]]
            切り詰めたデータセット

        """
        dataset = self.dataset
        max_tokens = self.max_tokens
        truncate_end = "right"  # 切り詰める方向（"right" or "left"）

        for sample in dataset:
            # --- 1) 回答ごとの token 列を収集 -----------------
            token_info: list[tuple[tuple[str, str], list[int]]] = (
                []
            )  # ((L?,Q?), tokens)
            for c in range(1, 16):
                c_key = f"L{c}"
                if c_key not in sample:
                    continue
                for qn in self.q_filter:
                    q_key = f"Q{qn}"
                    ans = sample[c_key].get(q_key, "")
                    # トークン数の取得
                    tokens = self.tokenizer.encode(
                        ans, add_special_tokens=False, truncation=False
                    )
                    token_info.append(((c_key, q_key), tokens))

            # 各回答のトークン数の合計を取得
            total_tokens = sum(len(tokens) for _, tokens in token_info)
            if total_tokens <= max_tokens:
                continue

            trim = max_tokens - total_tokens
            self.logger.debug(
                f"Sample {sample['userid']} exceeds max_tokens ({total_tokens} > {max_tokens}). Trimm {trim} tokens."
            )
            # --- 2) 長い順に削減 -------------------------------
            token_info.sort(key=lambda x: len(x[1]), reverse=True)
            # idx = 0
            # while total_tokens > max_tokens and idx < len(token_info):
            #     entry = token_info[idx][1]
            #     if len(entry) > 1:
            #         if truncate_end == "right":
            #             entry.pop()
            #         else:
            #             entry.pop(0)
            #         total_tokens -= 1
            #     else:
            #         idx += 1  # 次へ

            ELLIPSIS_TOKENS = self.tokenizer.encode("...", add_special_tokens=False)
            ELLIPSIS_LEN = len(ELLIPSIS_TOKENS)

            while total_tokens > max_tokens:
                # 今回だけの長さ順
                token_info.sort(key=lambda x: len(x[1]), reverse=True)
                entry_tokens = token_info[0][1]  # 現在最長

                # これ以上削れない（エリプシスだけ残っている or 長さ不足）
                if len(entry_tokens) <= ELLIPSIS_LEN:
                    break

                # ① 末尾に ... が 既に 付いているか確認
                if entry_tokens[-ELLIPSIS_LEN:] == ELLIPSIS_TOKENS:
                    # ② 既に ... がある → その直前を削る
                    del entry_tokens[-ELLIPSIS_LEN - 1]
                    total_tokens -= 1
                else:
                    # ③ まだ無ければ 末尾1トークンを ... に置換
                    entry_tokens.pop()  # 1 トークン削除
                    entry_tokens.extend(ELLIPSIS_TOKENS)  # ... を追加
                    # pop と extend で ( -1 + ELLIPSIS_LEN ) だけ総トークンが増減
                    total_tokens += ELLIPSIS_LEN - 1

            #! ライブラリ追加したらこっちのほうが計算量が少なくなる
            # import heapq

            # # token_info: [((c_key,q_key), tokens), ...]  # 前段で生成済み
            # heap = [(-len(toks), i) for i, (_, toks) in enumerate(token_info)]
            # heapq.heapify(heap)

            # while total_tokens > max_tokens:
            #     neg_len, i = heapq.heappop(heap)          # 最長を取得
            #     toks = token_info[i][1]
            #     if len(toks) == 1:                        # これ以上削れない
            #         continue
            #     pop_idx = -1 if truncate_end == "right" else 0
            #     toks.pop(pop_idx)                         # 1トークン削除
            #     total_tokens -= 1
            #     heapq.heappush(heap, (-len(toks), i))     # 更新して再投入

            # --- 3) 文章を戻す --------------------------------
            for (c_key, q_key), toks in token_info:
                sample[c_key][q_key] = self.tokenizer.decode(
                    toks, skip_special_tokens=True
                )

        for sample in dataset:
            # 各回答を連結，使わないキーを削除[仮コード]
            lines = []
            for c in range(1, 16):
                c_key = f"L{c}"
                if c_key not in sample:
                    continue
                for qn in self.q_filter:
                    q_key = f"Q{qn}"
                    ans = sample[c_key].get(q_key, "")
                    lines.append(f"{c_key}-{q_key}: {ans}")
            sample["input_text"] = "\n".join(lines)

        self.dataset = dataset
        del dataset  # メモリ解放


class GradeExplanationCollator:
    """
    GradeExplanationDataset 用の collate_fn を作成
    huggingface Trainer の DataCollator 互換のインタフェース
    huggingface>DataCollator
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
            "さらに、その成績を決定した根拠を簡潔に説明してください。\n"
            "成績と根拠は出力例のような形式で出力してください。\n"
            "入力文のL は講義回、Q は質問番号を示します（例: L1-Q1）。\n"
            f"アンケートの質問文は、\n{question}です。"
            "回答が NaN の場合は未回答であり、回答文字数が一定以上ならば切り捨てています。\n"
            "出力には、必ず A、B、C、D、F のいずれかを含めてください。\n"
            "出力例:\n"
            "この学生の成績は、Aです。理由は、質問１において、講義内容を数式を用いて詳細に説明しており、講義内容の理解度が高いためです。\n"
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
        # tgt_str は features内のtargetに保存される
        tgt_str = [ex["target"] for ex in features]

        self.logger.debug("prompt sample:\n%s", prompts[0][:500])
        self.logger.debug("target sample: %s", tgt_str[0][:50])

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
        # target_ids = torch.tensor(enc_tgt["input_ids"], dtype=torch.long)
        seq_list = [torch.tensor(ids, dtype=torch.long) for ids in enc_tgt["input_ids"]]

        target_ids = pad_sequence(
            seq_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

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
