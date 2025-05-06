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
from typing import Any

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
        }
    """

    def __init__(
        self,
        dataset_path: Path,
        logger: logging.Logger | None = None,
        merge_key: str = "userid",
        fill_token: str = "NaN",
        answer_col: str = "answer_content",
    ):
        """
        ファイルを読み込み，データセットを構成する
        """
        self.logger = logger or logging.getLogger(__name__)
        self.fill_token = fill_token

        self.reflection_path = Path(dataset_path) / "Reflection"
        self.grade_path = Path(dataset_path) / "Grade"
        for p in (self.reflection_path, self.grade_path):
            if not p.is_dir():
                raise FileNotFoundError(f"{p} が存在しません")

        left = self._read_folder(self.reflection_path)
        right = self._read_folder(self.grade_path)

        df = pd.merge(left, right, on=merge_key, how="inner")
        df["label"] = df["grade"].map(label_map)

        # 前処理
        df[answer_col] = df[answer_col].apply(self._preprocess)

        # データセットを辞書型として構築，より構造的に，呼び出しやすい形式にする
        self.dataset = self._build_nested(df, answer_col, fill_token)
        self.logger.info(
            f"GradePredictionDataset initialized with {len(self.dataset)} samples"
        )

    # -------- Dataset インタフェース --------
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[idx]

    

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

    @staticmethod
    def _build_nested(
        df: pd.DataFrame, answer_col: str = "answer_content", fill_token: str = "NaN"
    ) -> list[dict]:
        """
        DataFrame（userid, question_number, course_number, answer_content, label）
        ──▶ ユーザ単位のネスト辞書リストに変換
        """
        # ── ① 期待される全組み合わせを用意し欠損を明示 ──
        idx = pd.MultiIndex.from_product(
            [df["userid"].unique(), range(1, 16), range(1, 6)],
            names=["userid", "course_number", "question_number"],
        )
        df_full = (
            df.set_index(["userid", "course_number", "question_number"])
            .reindex(idx)  # 存在しない行を補完
            .reset_index()
        )

        # ── ② pivot で Lx–Qy テーブル化 ──
        pivot = (
            df_full.pivot_table(
                index=["userid", "course_number"],
                columns="question_number",
                values=answer_col,
                aggfunc="first",
            )
            .rename(columns=lambda q: f"Q{int(q)}")
            .fillna(fill_token)
        )

        # ── ③ ユーザごとにネスト辞書を構築 ──
        result = []
        for userid, user_block in pivot.groupby(level=0):
            user_dict = {"userid": userid}

            # 各コース L1〜L15
            for c in range(1, 16):
                key = f"L{c}"
                if (userid, c) in user_block.index:
                    user_dict[key] = user_block.loc[(userid, c)].to_dict()
                else:  # そのコースまるごと欠損
                    user_dict[key] = {f"Q{q}": fill_token for q in range(1, 6)}

            # labels（userid ごとに同じと仮定）
            label_val = df.loc[df["userid"] == userid, "label"].iloc[
                0
            ]  # ない場合は KeyError。必要なら try/except で None を許容
            user_dict["labels"] = int(label_val)

            result.append(user_dict)

        return result
