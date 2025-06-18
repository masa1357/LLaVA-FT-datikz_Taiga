# =====================================================================
# call_GPTAPI.py
# date: 2025/06/18
# description:
#   - GPT APIを呼び出す
# =====================================================================

import json                               
from logging import DEBUG                
from pathlib import Path                  

# 外部ライブラリ
from openai import OpenAI                 
from transformers import AutoTokenizer    

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

    with open('./APIkey.txt') as f:
        api_key = f.read()

    client = OpenAI(
        api_key=api_key,
    )

    return client

def call_gpt_api(client, model, prompt, seed=None, temperature=0, top_p=0.5, logger=None):
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                seed = seed,
                temperature = temperature,
                top_p = top_p,
                n = 1,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error processing prompt: {prompt}\n{e}")
            return "ERROR"
        
def main():
    # loggerの設定
    print("set logger")
    logger = set_logger(level=DEBUG)


    # tokenizerの設定
    tokenizer = AutoTokenizer.from_pretrained("elyza/Llama-3-ELYZA-JP-8B", use_fast=True)

    question_filter = [1, 2, 3, 4, 5]  # フィルタリングする質問番号

    data = GradeExplanationDataset(
        dataset_path=DATA_PATH,
        logger=logger,
        question_filter= question_filter,
        cancatenate=False,
        mode="all"
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
    
    # モデル設定
    model = "gpt-4o-2024-08-06" #"o1-2024-12-17"
    # OpenAIクライアントのロード
    client = load_openai_client(API_PATH)

    # test(logger levelが DEBUG の場合のみ)
    if logger.level == DEBUG:
        logger.debug("Debug mode: Using a small subset of data for testing.")
        # デバッグ用にデータの最初の10件を使用
        data = data[:10]

    # 生成
    for sample in data:
        prompt = f"{preamble}\n{sample['input_text']}\n"
        logger.debug(f"Prompt: {prompt}")
        out = call_gpt_api(
            client, 
            model, 
            prompt, 
            seed=42, 
            temperature=0.5, 
            top_p=0.5, 
            logger=logger
        )
        sample["target"] = out
        logger.info(f"Output: {out}")

    # 結果の保存
    output_path = f"{DAATA_PATH}/GradeExplanationDataset_Qs{len(question_filter)}_trimmed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    main()
        
