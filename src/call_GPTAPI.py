# =====================================================================
# call_GPTAPI.py
# date: 2025/06/18
# description:
#   - GPT APIを呼び出す
# =====================================================================

import json
from logging import DEBUG, INFO
from pathlib import Path

# 外部ライブラリ
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import re

# 自作モジュール／クラス（正しいパスに合わせて修正）
from src.gradeexplanation_data import GradeExplanationDataset
from src.util import set_logger

# 型ヒント用 (任意)
from typing import Any, Dict, List

# https://chatgpt.com/g/g-JuG2vFjqa-organization
# OpenAI APIキーを設定
# ../data/APIkey.txtに記載

API_PATH = "./data/APIkey.txt"
DATA_PATH = "./data/"

DEFAULT_PROMPT = (
    "あなたは大学講義『情報科学』で収集されたアンケートの根拠説明アシスタントです。"
    "回答はすべて日本語で書き、個人情報と不適切表現を含めてはいけません。"
)


def load_openai_client(file_path):
    """
    OpenAIクライアントをロードする関数
    Parameters
    ----------
    file_path : str
        OpenAI APIキーのpath
    model : str
        使用モデル名
    Returns
    -------
    openai_client
        OpenAIクライアントのインスタンス
    """

    with open(API_PATH) as f:
        api_key = f.read()

    client = OpenAI(
        api_key=api_key,
    )

    return client


def call_gpt_api(
    client, model, prompt, seed=None, temperature=0, top_p=0.5, logger=None, system_prompt:str = DEFAULT_PROMPT
):
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            n=1,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing prompt: {prompt}\n{e}")
        return "ERROR"


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


def main():
    # loggerの設定
    print("set logger")
    logger = set_logger(level=INFO)

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
        trim=True,
    )

    # プロンプトの設定
    Q_TEXT = {
        1: "Q1:今日の内容を自分なりの言葉で説明してみてください\n",
        2: "Q2:今日の内容で、分かったこと・できたことを書いてください\n",
        3: "Q3:今日の内容で、分からなかったこと・できなかったことを書いてください\n",
        4: "Q4:質問があれば書いてください\n",
        5: "Q5:今日の授業の感想や反省を書いてください\n",
    }

    question = "".join(Q_TEXT[q] for q in question_filter)

    preamble = (
        "あなたは大学の教授であり、「情報科学」の講義を担当しています。\n"
        "この講義では、学生に対して講義終了時にアンケートを実施し、学生の理解度や講義内容に対する意見を収集しています。\n"
        "アンケートは以下の質問で構成されています。\n"
        f"{question}"
        "また，学生の成績は高い順にA, B, C, D, Fの5段階で評価されます。\n"
        "このとき、学生のアンケートの回答文と成績から、なぜその学生がこの成績を取ったのか、根拠を回答形式に基づいて提示してください。\n"
        "アンケート内容のL は講義回、Q は質問番号を示します（例: L1-Q1）。\n"
        "アンケート内容が NaN の場合は未回答であり、回答文字数が一定以上ならば切り捨てています。\n"
        "根拠は簡潔に、具体的な取り組みや意見に注目して提示してください。\n"
        "【出力フォーマット】\n"
        "この学生の成績は、<成績>です。理由は、<質問内容の要約>ためです。\n"
        "出力例:\n"
        "この学生の成績は、Aです。理由は、質問１において、講義内容を数式を用いて詳細に説明しており、講義内容の理解度が高いためです。\n"
        "アンケート内容："
    )
    preamble2 = "成績:"

  

    # モデル設定
    model = "gpt-4o-2024-08-06"  # "o1-2024-12-17"
    # OpenAIクライアントのロード
    client = load_openai_client(API_PATH)

    # test(logger levelが DEBUG の場合のみ)
    if logger.level == DEBUG:
        logger.debug("Debug mode: Using a small subset of data for testing.")
        # デバッグ用にデータの最初の10件を使用
        data = data[:10]

    # データの要素を確認
    logger.info(f"data keys: {data[0].keys() if data else 'No data available'}")

    # 生成
    for sample in tqdm(data):
        prompt = f"{preamble}\n{sample['input_text']}\n{preamble2} {sample['grades']}\n"
        logger.debug(f"Prompt: {prompt}")
        out = call_gpt_api(
            client, model, prompt, seed=42, temperature=0, top_p=0.5, logger=logger
        )
        sample["target"] = out
        logger.info(f"Output: {out}")

    # 結果の保存
    output_path = (
        f"{DATA_PATH}/GradeExplanationDataset_Qs{len(question_filter)}_trimmed.pt"
    )
    torch.save(data, output_path)

    # targetとgradesが一致しているか確認
    error_count = 0
    for sample in data:
        if sample["target"] == "ERROR":
            logger.error("Error in generating response for a sample.")
        else:
            # targetから成績を抽出
            extracted_grade = extract_grade(sample["target"])
            if extracted_grade != sample["grades"]:
                error_count += 1
                logger.error(
                    f"Grade mismatch on {sample['userid']}: Expected {sample['grades']}, but got {extracted_grade}."
                )

    logger.info(f"Total errors found: {error_count}")

def main2():
    """
    ChatGPT APIを呼び出して、成績説明を生成するメイン関数
    main()との違いは，拡張データセットに対する処理であること
    -> 読み込み先，プロンプトが異なる
    """
    print("set logger")
    logger = set_logger(level=DEBUG)

    # データの読み込み
    file = Path("data/GradeExplanationDataset_Extend.jsonl")
    with file.open("r", encoding="utf-8") as f:
        generated_data = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(generated_data)} samples from {file}")

    # プロンプトの設定
    Q_TEXT = {
        1: "今日の内容を自分なりの言葉で説明してみてください\n",
        2: "今日の内容で、分かったこと・できたことを書いてください\n",
        3: "今日の内容で、分からなかったこと・できなかったことを書いてください\n",
        4: "質問があれば書いてください\n",
        5: "今日の授業の感想や反省を書いてください\n",
    }

    preamble = (
        "以下の文章は、大学の講義『情報科学』における学生のアンケート回答と、最終成績です。\n"
        "これを参考に、なぜ学生がこの成績を取ったのか、根拠を回答形式に基づいて提示してください。\n"
        "アンケート内容のL は講義回を示します（例: L1）。\n"
        "根拠は簡潔に、具体的な取り組みや意見に注目して提示してください。\n"
        "■ ルール\n"
        "1. 回答はすべて日本語で書き、個人情報と不適切表現を含めてはいけません。\n"
        "2. 成績は高い順にA, B, C, D, Fの5段階で評価され、Fは単位不認定を意味します。\n"
        "3. 回答文字数は300文字、1〜2 文までに控えてください。\n"
        "4. 回答は以下に提示するフォーマットに従ってください。\n"
        "出力例:\n"
        "この学生の成績は、Aです。理由は、質問１において、講義内容を数式を用いて詳細に説明しており、講義内容の理解度が高いためです。\n"
        "### 出力フォーマット\n"
        "この学生の成績は、<成績>です。理由は、<成績の根拠>ためです。\n"
        "---\n"
        "以下はアンケートの質問と学生のアンケート回答，成績です。\n"
        "質問内容:{question}\n"
        "アンケート内容：{Answer}\n"
        "成績: {grades}\n"
    )

    # モデル設定
    model = "gpt-4o-2024-08-06"  # "o1-2024-12-17"
    # OpenAIクライアントのロード
    client = load_openai_client(API_PATH)

    # test(logger levelが DEBUG の場合のみ)
    if logger.level == DEBUG:
        logger.debug("Debug mode: Using a small subset of data for testing.")
        # デバッグ用にデータの最初の10件を使用
        data = data[:10]

    # データの要素を確認
    logger.info(f"data keys: {data[0].keys() if data else 'No data available'}")

    # 生成
    for sample in tqdm(data):
        question = Q_TEXT[sample["question"]]
        lec = sample["lecture"]
        grades = sample["grades"]
        answer = sample["target"]
        
        answer = lec + answer
        prompt = preamble.format(
            question=question, Answer=answer, grades=grades
        )

        logger.debug(f"Prompt: {prompt}")
        out = call_gpt_api(
            client, model, prompt, seed=42, temperature=0, top_p=0.5, logger=logger
        )
        sample["target"] = out
        logger.info(f"Output: {out}")

    # 結果の保存
    output_path = (
        f"{DATA_PATH}/GradeExplanationDataset_Extended_data.pt"
    )
    torch.save(data, output_path)

    # targetとgradesが一致しているか確認
    error_count = 0
    for i, sample in enumerate(data):
        if sample["target"] == "ERROR":
            logger.error("Error in generating response for a sample.")
        else:
            # targetから成績を抽出
            extracted_grade = extract_grade(sample["target"])
            if extracted_grade != sample["grades"]:
                error_count += 1
                logger.error(
                    f"Grade mismatch {i}th case: Expected {sample['grades']}, but got {extracted_grade}."
                )

    logger.info(f"Total errors found: {error_count}")




if __name__ == "__main__":
    main2()
