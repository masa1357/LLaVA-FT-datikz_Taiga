#!/bin/bash
set -e

export PYTHONPATH=$(pwd)

LOGFILE="./log/train_log_$(date +%Y%m%d_%H%M%S).txt"

CMD = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TORCH_DISABLE_DYNAMO=1 \
        TORCH_COMPILE=0 \
        TOKENIZERS_PARALLELISM=false \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        \
        # ─── accelerate option ──────────────────────────────────────
        accelerate launch \
        --config_file ds_zero3.yaml \                # ← num_processes=4, zero_stage=3 等を記述
        --main_process_port 29501 \                  # ← master_port を固定したい場合だけ付ける
        src/train_DDP.py \                           # ← python script
        --output_dir ./outputs/llama3-elyza-8b-lora \
        --epochs 5 \
        --batch_size 4 \                             # ← global batch
        --micro_batch_size 1 \                       # ← 4GPU ×1 = global4 と一致
        --run_name llama3-elyza-8b-lora_test \
        --max_words 4096
        "

echo "Running command at $(date):" | tee "$LOGFILE"
echo "$CMD " | tee -a "$LOGFILE"
eval $CMD 2>&1 | tee -a "$LOGFILE"
