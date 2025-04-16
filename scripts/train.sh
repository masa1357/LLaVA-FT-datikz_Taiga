#!/bin/bash
set -e

export PYTHONPATH=$(pwd)

LOGFILE="./log/train_log_$(date +%Y%m%d_%H%M%S).txt"

CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISABLE_DYNAMO=1 TORCH_COMPILE=0 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3 \
        torchrun \
        --nproc_per_node=4 \
        --master_port=29501 \
        src/train.py \
        --output_dir ./outputs/llava-datikz-lora\
        --epochs 2 \
        --batch_size 4 \
        --run_name llava-datikz-lora_test \
        "

echo "Running command at $(date):" | tee "$LOGFILE"
echo "$CMD" | tee -a "$LOGFILE"
eval $CMD 2>&1 | tee -a "$LOGFILE"
