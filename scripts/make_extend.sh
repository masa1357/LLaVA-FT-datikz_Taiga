# exacute ./src/call_GPTAPI_genextend.py

#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
LOGFILE="./log/call_Extend$(date +%Y%m%d_%H%M%S).log"

CMD="python3 ./src/call_GPTAPI_genextend.py"

echo "Running command at $(date):" | tee "$LOGFILE"
echo "$CMD " | tee -a "$LOGFILE"
eval $CMD 2>&1 | tee -a "$LOGFILE"

        