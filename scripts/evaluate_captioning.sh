#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export TMPDIR=/tmp

datetime=$(date +%Y%m%d_%H%M%S)
LOGFILE="./log/eval_log_${datetime}.txt"
MODEL_PATH="./outputs/llava-datikz-lora/llava-datikz-full"    #"llava-hf/llava-1.5-7b-hf"  #"./outputs/llava-datikz-lora/llava-datikz-full"
OUTPUT_PATH="./results/llava-datikz-base_testdata.json"

#CUDA_VISIBLE_DEVICES=0 \
CMD="
    python3 src/evaluate_captioning.py \
    --model_path ${MODEL_PATH} \
    --output_json ${OUTPUT_PATH}"

echo "Running evaluation at $(date):" | tee "$LOGFILE"
echo "$CMD" | tee -a "$LOGFILE"
eval $CMD 2>&1 | tee -a "$LOGFILE"
