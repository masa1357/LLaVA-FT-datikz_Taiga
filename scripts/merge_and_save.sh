#!/bin/bash
set -e

# モデル情報
BASE_MODEL="llava-hf/llava-1.5-7b-hf"
LORA_CHECKPOINT="./outputs/llava-datikz-lora/checkpoint-1812"
SAVE_PATH="./outputs/llava-datikz-lora/llava-datikz-full"

# 実行コマンド
python3 src/merge_and_save_model.py \
    --base_model_name "${BASE_MODEL}" \
    --lora_checkpoint "${LORA_CHECKPOINT}" \
    --save_path "${SAVE_PATH}"
