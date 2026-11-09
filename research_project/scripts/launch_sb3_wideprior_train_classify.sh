#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
nohup nice -n 5 caffeinate -i .venv/bin/python scripts/run_sb3_wideprior_training_and_classification.py \
  > logs/sb3_wideprior_train_classify.log 2>&1 < /dev/null &
disown
echo "launched pid $!"
