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
    def __init__(self):
        """
        ファイルを読み込み，データセットを構成する
        """
        self.logger = logger or logging.getLogger(__name__)
        self.fill_token = fill_token

        self.answer_col = answer_col

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
    def concat(self):
        # 連結テキストモード
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

