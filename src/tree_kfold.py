# =====================================================================
# tree_kfold.py
# date: 2025/07/18
# description:
#   - LightGBMを用いた成績予測モデルの構築
#   - データセットの事件列変化量を特徴量として使用
#   - k-foldを採用
#   - shapを用いて可視化
# =====================================================================

from logging import INFO, DEBUG
import argparse
import sudachipy 
# ================ プロジェクト内（ローカル） ================
from util import load_model, set_seed, set_logger
from gradepred_data_forkfold import GradePredictionDataset


class GetSentenceVector(GradePredictionDataset):

    def __init__(
        self,
        dataset_path,
        logger=None,
        fill_token="NaN",
        answer_col="answer_content",
        question_filter=None,
        merge_key="userid",
        max_tokens=3072,
    ):
        super().__init__(
            dataset_path,
            logger,
            fill_token,
            answer_col,
            question_filter,
            merge_key,
            max_tokens,
        )
    

    def getvector(self):
        """
        文章のベクトル化を行う
        構造：
        {
            "userid": "user_id",
            "sentences":{
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
            }
            "labels": labels (0~4)
            "grades": grades (A,B,C,D,F)
        }        
        """
        dict = sudachipy.Dictionary()
        tokenizer = dict.create() # 辞書から分割器を作る

        # dataset["L1"]-["L15"]を["sentence"]内に持っていく
        for c in range(1, 16):
            self.dataset["sentence"][f"L{c}"] = self.raw_dataset[f"L{c}"]

        # 単語分かち書き
        for sample in self.dataset:
            self.logger.info("make wakati")
            for c in range(1,16):
                c_key = f"L{c}"
                for q in self.question_filter:
                    q_key = f"Q{q}"
                    # 分かち書き
                    wakati = tokenizer.tokenize(self.dataset["sentence"][c_key][q_key])
                    self.logger.debug(wakati)
                
                    # 埋め込み変換
                    
            


    def geteucdist(self):
        pass

    def getcosdist(self):
        pass

    def wordcount(self):
        pass


def main():
    # ? logger設定
    print("set logger")
    logger = set_logger(level=INFO)

    # ================================================================
    # パラメータの取得
    # ================================================================

    parser = argparse.ArgumentParser(
        description="Fine-tune LLama with LoRA on Reflection dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/",
        help="data path",
    )
    args = parser.parse_args()
    set_seed(42)

    # ================================================================
    # データセットの取得
    # ================================================================
    dataset = GradePredictionDataset(dataset_path=args.data_path, logger=logger)
    dataset.reset()


if __name__ == "__main__":
    main()
