#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export TMPDIR=/tmp

datetime=$(date +%Y%m%d_%H%M%S)
LOGFILE="./log/eval_log_${datetime}.txt"
MODEL_PATH="./outputs/llava-datikz-lora/llava-datikz-full"    #"llava-hf/llava-1.5-7b-hf"  #"./output/llava-datikz-lora/llava-datikz-full"
OUTPUT_PATH="./results/llava-datikz-full_testdata.json"

# 依存ライブラリをインストール（ログにも保存）
echo "🔧 Installing dependencies..." | tee -a "$LOGFILE"
pip install --user absl-py nltk rouge-score >> "$LOGFILE" 2>&1


CMD="CUDA_VISIBLE_DEVICES=3 \
    python3 src/evaluate_captioning.py \
    --model_path ${MODEL_PATH} \
    --output_json ${OUTPUT_PATH}"

echo "Running evaluation at $(date):" | tee "$LOGFILE"
echo "$CMD" | tee -a "$LOGFILE"
eval $CMD 2>&1 | tee -a "$LOGFILE"
