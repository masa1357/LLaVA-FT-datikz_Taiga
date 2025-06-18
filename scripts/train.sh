#!/bin/bash
set -e

export PYTHONPATH=$(pwd)

LOGFILE="./log/train_log_$(date +%Y%m%d_%H%M%S).log"

CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TORCH_DISABLE_DYNAMO=1 \
        TORCH_COMPILE=0 \
        TOKENIZERS_PARALLELISM=false \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        accelerate launch \
        --config_file ds_zero3.yaml \
        src/train_DDP.py \
        --output_dir ./outputs/llama3-elyza-8b-lora \
        --epochs 3 \
        --batch_size 4 \
        --run_name llama3-elyza-8b-lora_test \
        --max_words 3072
        --logfile $LOGFILE \
        "

echo "Running command at $(date):" | tee "$LOGFILE"
pip list | grep -E 'torch|deepspeed|transformers|accelerate|safetensors' | tee -a "$LOGFILE"
echo "$CMD " | tee -a "$LOGFILE"
eval $CMD 2>&1 | tee -a "$LOGFILE"
