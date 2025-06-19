# =====================================================================
# call_GPTAPI.py
# date: 2025/06/18
# description:
#   - GPT APIを呼び出す
# =====================================================================
from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

# --- 外部ライブラリ ----------------------------------------------------
from openai import OpenAI, error as oai_err
from transformers import AutoTokenizer
import torch

# --- 自作モジュール ----------------------------------------------------
from src.gradeexplanation_data import GradeExplanationDataset
from src.util import set_logger

# =====================================================================
# 定数設定
# =====================================================================

# https://chatgpt.com/g/g-JuG2vFjqa-organization
# OpenAI APIキーを設定
# ../data/APIkey.txtに記載
API_PATH: Path = Path("./data/APIkey.txt")
DATA_PATH: Path = Path("./data")
OUT_FILE: Path = DATA_PATH / "GradeExplanationDataset_Extend.jsonl"
OPENAI_MODEL_NAME = "gpt-4o-2024-08-06" # "o1-2024-12-17"

MAX_RETRY = 5           # GPT 呼び出しの最大リトライ回数
BACKOFF_BASE = 2        # 2^n 秒で指数バックオフ
SEED = 42               # 乱数シード（再現用）

# ----------------------------------------------------------------------
# Few‑Shot 例（固定）
# ----------------------------------------------------------------------
GOOD_EXAMPLES = (
    "入力:\n"
    "携帯やpcは音声、文章、写真を伝える。現在の通信技術に至るまでに、さまざまな技術が生み出され用いられた。\n"
    "情報の伝達は古来より様々な手法で行われている。現代はコンピューターを活用したものが主体である。\n"
    "情報科学の重要性を歴史を通して学んだ。\n"
    "出力:\n"
    "情報科学は重要であり，歴史を通して様々な技術が発展してきたことを学んだ。\n"
)

BAD_EXAMPLES = (
    "入力: コンピュータの仕組みを学んだ。\n"
    "誤出力: コンピュータの仕組みを学んだ。  # 💀 同語句繰り返し\n"
)

SYSTEM_PROMPT = (
    "あなたは大学講義『情報科学』のアンケート生成アシスタントです。"
    "回答はすべて日本語で書き、個人情報と不適切表現を含めてはいけません。"
)


# ----------------------------------------------------------------------
# ユーティリティ
# ----------------------------------------------------------------------

def load_openai_client(file_path: Path) -> OpenAI:
    api_key = file_path.read_text().strip()
    return OpenAI(api_key=api_key)



def call_gpt_api(
    client: OpenAI,
    model: str,
    prompt_user: str,
    *,
    seed: int | None = None,
    temperature: float = 0.5,
    top_p: float = 0.9,
    logger: logging.Logger | None = None,
) -> str | None:
    """system+user の 2 ロール構成で呼び出し。リトライ付き。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_user},
    ]

    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                n=1,
                **({"seed": seed} if seed is not None else {}),
            )
            return resp.choices[0].message.content
        except (oai_err.RateLimitError, oai_err.APIError, oai_err.APIConnectionError) as exc:
            wait = BACKOFF_BASE ** (attempt - 1) + random.random()
            if logger:
                logger.warning("%s: retry %d/%d after %.1fs", exc.__class__.__name__, attempt, MAX_RETRY, wait)
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.error("Unexpected error: %s", exc)
            return None
    return None


# ================================================================
# ヘルパ
# ================================================================
def extract_grade(text: str) -> str:
    # 1. 文字列から[/INST]以前の部分を削除
    text = text.split("[/INST]")[-1]

    # 2. 文字列から成績を抽出
    # 「成績は、Xです」の X を正規表現で抜く
    m = re.search(r"成績は、([A-D]|F)です", text)
    if m:
        return m.group(1)
    else:
        for grade in ["A", "B", "C", "D", "F"]:
            if grade in text:
                return grade
    return "F"  # デフォルトは F

# ================================================================
# main
# ================================================================
def main() -> None:
    # loggerの設定
    print("set logger")
    logger = set_logger(level=INFO)
    logger.info("Logger ready ✅")


    # tokenizerの設定
    tokenizer = AutoTokenizer.from_pretrained(
        "elyza/Llama-3-ELYZA-JP-8B", use_fast=True
    )

    question_filter = [1, 2, 3, 4, 5]  # フィルタリングする質問番号

    data = GradeExplanationDataset(
        dataset_path=DATA_PATH,
        logger=logger,
        question_filter=question_filter,
        concatenate=False,
        mode="all",
        division=False,
        tokenizer=tokenizer,
    )

    # プロンプトの設定
    Q_TEXT = {
        1: "今日の内容を自分なりの言葉で説明してみてください\n",
        2: "今日の内容で、分かったこと・できたことを書いてください\n",
        3: "今日の内容で、分からなかったこと・できなかったことを書いてください\n",
        4: "質問があれば書いてください\n",
        5: "今日の授業の感想や反省を書いてください\n",
    }
    # EXTEND_PREAMBLE = (
    #     "これから提示する文章は，大学の講義「情報科学」における学生のアンケート回答です。\n"
    #     "この講義では、学生に対して講義終了時にアンケートを実施し、学生の理解度や講義内容に対する意見を収集しています。\n"
    #     "講義は15回実施され，アンケートも同様に15回収集されました。\n"
    #     "このように収集されたアンケートの回答文を元に，似たような形式のアンケート回答を生成してください。\n"
    #     "入力するアンケートは，同じ講義回の，同じ質問に対する回答を10件提示します。\n"
    #     "このとき，以下のルールを遵守してください。\n"
    #     "1. 生成する文章は1~2文程度で簡潔にまとめてください。\n"
    #     "2. 生成する文章は，講義内容の理解度や意見を反映した内容にしてください。\n"
    #     "3. 生成する文章は，入力されたアンケートと一致してはいけません。\n"
    #     "4. 表現が異なるアンケートを5件生成してください。\n"
    #     "以下に，アンケートと，そこから生成された回答の一例を示します。\n"
    #     "## 例：\n"
    #     "### 質問内容：今日の内容を自分なりの言葉で説明してみてください\n"
    #     "### アンケート例:\n"
    #     "- 携帯やpcは音声、文章、写真を伝える。現在の通信技術に至るまでに、さまざまな技術が生み出され用いられた。\n"
    #     "- 情報の伝達は古来より様々な手法で行われている。現代はコンピューターを活用したものが主体である。\n"
    #     "- 情報科学の重要性を歴史を通して学んだ。\n"
    #     "### 生成された回答例:\n"
    #     "- 情報科学は重要であり，歴史を通して様々な技術が発展してきたことを学んだ。\n"
    #     "例と同じような形式で、以下のアンケート回答と類似する回答を10件生成してください。\n"
    #     "### 質問内容：{question}\n"
    #     "### アンケート回答:\n{answers}\n"
    # )

    BASE_PROMPT = (
        "以下の 10 件は、同一講義回・同一質問に対する学生の回答です。"
        "これらを参考に **新しい回答を 5 行** 生成してください。\n\n"
        "■ ルール\n"
        "1. 各行は最大 100 文字、1〜2 文で簡潔に。\n"
        "2. 講義内容への理解・意見を必ず含める。\n"
        "3. 入力回答と 30% 以上語句が一致しないよう語彙を変える。\n"
        "4. 個人情報・不適切表現は禁止。\n"
        "5. 出力は改行区切り 5 行のみ。番号や記号は付けない。\n\n"
        "### 良い例\n{good_examples}\n"
        "### NG 例\n{bad_examples}\n"
        "---\n"
        "### 質問\n{question}\n\n"
        "### 参考アンケート (10 件)\n{answers}\n"
        "--- ここから出力 (5 行) ---"
    )



    # OpenAIクライアントのロード
    client = load_openai_client(API_PATH)

    # データの要素を確認
    logger.info(f"data keys: {data[0].keys() if data else 'No data available'}")

    # --- Grade 分布計算
    total = len(data)
    grade_rate: Dict[str, float] = {
        g: sum(d["grades"] == g for d in data) / total for g in ["A", "B", "C", "D", "F"]
    }
    logger.info("Grade distribution → %s", grade_rate)

    # 4000 呼出しを 75 (15×5) ペアに分配
    base_calls = 4000
    per_pair = {g: max(1, int(rate * base_calls) // 75) for g, rate in grade_rate.items()}
    logger.info("Calls per (lecture, question) pair → %s", per_pair)

    # --- 出力ファイルの準備
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 追記モードで開く（既存内容を活かす）

    rng = random.Random(SEED)

    # =============================================================
    # 生成ループ
    # =============================================================
    generated_cnt = 0
    for grade, num in per_pair.items():
        logger.info("Grade %s: %d×75 generate", grade, num)
        subset = [d for d in data if d["grades"] == grade]
        for lec in range(1, 16):
            for q in range(1, 6):
                grp = [d for d in subset if d["lecture"] == lec and d["question"] == q]
                if len(grp) < num:
                    continue
                sampled = rng.sample(grp, num)
                answers = [d["input_text"] for d in sampled]

                prompt_user = BASE_PROMPT.format(
                    good_examples=GOOD_EXAMPLES,
                    bad_examples=BAD_EXAMPLES,
                    question=Q_TEXT[q],
                    answers="\n".join(f"- {a}" for a in answers),
                )

                reply = call_gpt_api(
                    client,
                    OPENAI_MODEL_NAME,
                    prompt_user,
                    seed=SEED,
                    logger=logger,
                )
                if not reply:
                    continue

                lines = [ln.lstrip("- ").strip() for ln in reply.splitlines() if ln.strip()][:5]
                with OUT_FILE.open("a", encoding="utf-8") as fout:
                    for line in lines:
                        fout.write(json.dumps({
                            "lecture": lec,
                            "question": q,
                            "grades": grade,
                            "input_text": "\n".join(answers),
                            "target": line,
                        }, ensure_ascii=False) + "\n")
                        generated_cnt += 1

    logger.info("▶︎ 完了: %d 件を %s に書き込みました", generated_cnt, OUT_FILE)

if __name__ == "__main__":
    main()
