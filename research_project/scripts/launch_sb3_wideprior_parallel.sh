#!/usr/bin/env bash
# Launch the parallel workers for the S-B3 wide-prior bank (fully detached, niced,
# caffeinated). Safe to re-run: each worker resumes from its own .partial.npz and
# skips if its final .npz already exists.
#
# Layout (8 workers; the original sequential process keeps running as a 9th until
# assembly):
#   train chunks 1-6 : 280 rows each (1680)  -> data/..._chunk{1..6}.npz
#   worker 7         : sbc_reduced 120 rows (canonical file, original seed 9102),
#                      then sbc_confirm part A 130 rows
#   worker 8         : sbc_confirm part B 270 rows
#
# Follow progress: tail -f data/wu2003_sbi_train_sb3_n5000_sb3_wideprior_stage1bfix.progress.log
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CHUNK=scripts/generate_sb3_wideprior_chunk.py
SUF=_sb3_wideprior_stage1bfix
mkdir -p logs

launch() {  # launch <logname> <cmd...>
  local name=$1; shift
  nohup nice -n 10 caffeinate -i "$@" > "logs/sb3_wideprior_${name}.log" 2>&1 < /dev/null &
  disown
  echo "launched ${name} (pid $!)"
}

for w in 1 2 3 4 5 6; do
  launch "train_chunk${w}" $PY $CHUNK \
    --out "data/wu2003_sbi_train_sb3_n5000${SUF}_chunk${w}.npz" \
    --n-rows 280 --seed $((910100 + w)) --label "train_chunk${w}"
done

# worker 7: two sequential jobs in one detached shell
nohup nice -n 10 caffeinate -i bash -c "
  $PY $CHUNK --out results/sb3_sbc_set_n120${SUF}.npz --n-rows 120 --seed 9102 --label sbc_reduced &&
  $PY $CHUNK --out results/sb3_sbc_set_n400${SUF}_partA.npz --n-rows 130 --seed 910301 --label sbc_confirm_A
" > logs/sb3_wideprior_sbc_worker7.log 2>&1 < /dev/null &
disown
echo "launched sbc worker 7 (pid $!)"

launch "sbc_confirm_B" $PY $CHUNK \
  --out "results/sb3_sbc_set_n400${SUF}_partB.npz" \
  --n-rows 270 --seed 910302 --label sbc_confirm_B
