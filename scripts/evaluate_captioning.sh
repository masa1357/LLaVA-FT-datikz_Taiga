#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export TMPDIR=/tmp

datetime=$(date +%Y%m%d_%H%M%S)
LOGFILE="./log/eval_log_${datetime}.log"

#CUDA_VISIBLE_DEVICES=0 \
CMD="python3 src/evaluation_expain.py"

echo "Running evaluation at $(date):" | tee "$LOGFILE"
echo "$CMD" | tee -a "$LOGFILE"
eval $CMD 2>&1 | tee -a "$LOGFILE"
