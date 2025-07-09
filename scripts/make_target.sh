# exacute ./src/call_GPTAPI.py

#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
LOGFILE="./log/call_GPT_$(date +%Y%m%d_%H%M%S).log"

CMD="python3 ./src/call_GPTAPI.py"

echo "Running command at $(date):" | tee "$LOGFILE"
echo "$CMD " | tee -a "$LOGFILE"
eval $CMD 2>&1 | tee -a "$LOGFILE"

        