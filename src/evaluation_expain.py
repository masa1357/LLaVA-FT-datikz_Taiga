# =====================================================================
# evaluation_expain.py (create 2025/06/20)
# description:
#   - 文章生成タスクの評価を行う
# =====================================================================


# libraries
import json
from pathlib import Path
import logging
from typing import List, Dict

# evaluation libraries
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score  # nltk.download('wordnet') が初回のみ必要
from moverscore_v2 import word_mover_score, get_idf_dict


# own libraries
from src.util import set_logger

DATA_DIR = Path("results/GradeExplanation/connect")

# =============================================================
# evaluation function
# =============================================================
def evaluate_record(record: Dict[str, List[str]]) -> Dict[str, float]:

    preds: List[str] = record["output_sentence"]
    refs:  List[str] = record["label_sentence"]

    # ---------- BLEU ----------
    bleu_score = sacrebleu.corpus_bleu(preds, [refs]).score

    # ---------- ROUGE ----------
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougel = 0.0, 0.0, 0.0
    for p, r in zip(preds, refs):
        scores = rouge.score(r, p)
        rouge1  += scores["rouge1"].fmeasure
        rouge2  += scores["rouge2"].fmeasure
        rougel += scores["rougeL"].fmeasure
    n = len(preds)
    rouge1, rouge2, rougel = [s / n * 100 for s in (rouge1, rouge2, rougel)]

    # ---------- METEOR ----------
    meteor = sum(meteor_score([r], p) for p, r in zip(preds, refs)) / n * 100

    # ---------- MoverScore ----------
    idf_pred = get_idf_dict(preds)
    idf_ref  = get_idf_dict(refs)
    ms_list  = word_mover_score(
        refs, preds,
        idf_ref, idf_pred,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=16,
    )
    moverscore = sum(ms_list) / n * 100

    return {
        "BLEU":   round(bleu_score, 2),
        "ROUGE-1": round(rouge1, 2),
        "ROUGE-2": round(rouge2, 2),
        "ROUGE-L": round(rougel, 2),
        "METEOR": round(meteor, 2),
        "MoverScore": round(moverscore, 2),
    }


# =============================================================
# main function
# =============================================================

def main() -> None:
    print("setup logger")
    logger = set_logger(logging.INFO)

    # =============================================================
    # load data
    # =============================================================
    before_file = "pred_result_before_training.json",
    after_file  = "pred_result_after_training.json",
    
    with open(DATA_DIR / before_file, encoding="utf-8") as f:
        before_data = json.load(f)
    with open(DATA_DIR / after_file, encoding="utf-8") as f:
        after_data = json.load(f)

    # データ構造確認と必要に応じてラップ
    if isinstance(before_data, dict):
        before_data = [before_data]
    if isinstance(after_data, dict):
        after_data = [after_data]

    # データ構造：
    # {
    #   "input_sentence": [
    #     "プロンプト1",
    #     ...
    #     "プロンプトn",
    # ]
    #   "output_sentence": [
    #     "出力1",
    #     ...
    #     "出力n",
    #   ]
    #   "label_sentence": [
    #     "正解1",
    #     ...
    #     "正解n",
    #   ]
    # }

    
    # =============================================================
    # Check data
    # =============================================================

    logger.info("Check data : before_training")
    for i, data in enumerate(before_data):
        logger.info(f"Data {i+1}:")
        logger.info(f"Input: {data['input_sentence']}")
        logger.info(f"Output: {data['output_sentence']}")
        logger.info(f"Label: {data['label_sentence']}")

        if i >= 10:  # 最初の10件だけ表示
            break
    logger.info("Check data : after_training")
    for i, data in enumerate(after_data):
        logger.info(f"Data {i+1}:")
        logger.info(f"Input: {data['input_sentence']}")
        logger.info(f"Output: {data['output_sentence']}")
        logger.info(f"Label: {data['label_sentence']}")

        if i >= 10:  # 最初の10件だけ表示
            break
    # =============================================================
    # evaluation
    # metrics 1: BLEU
    # metrics 2: ROUGE
    # metrics 3: METEOR
    # metrics 4: moverscore
    # =============================================================

    def eval_and_log(dataset, tag):
        logger.info(f"✅  Start Evaluation ({tag}) training data ...")
        agg = {k: 0.0 for k in
               ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "MoverScore"]}
        for i, rec in enumerate(dataset, 1):
            scores = evaluate_record(rec)
            logger.info(f"[{tag} #{i}] {scores}")
            for k in agg:
                agg[k] += scores[k]

        m = len(dataset)
        mean_scores = {k: round(v / m, 2) for k, v in agg.items()}
        logger.info(f"📊  Mean metrics ({tag}) → {mean_scores}")

    eval_and_log(before_data, "before")
    eval_and_log(after_data,  "after")


if __name__ == "__main__":
    main()