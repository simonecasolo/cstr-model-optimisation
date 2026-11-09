#!/usr/bin/env bash
# Resume the S-B3 wide-prior parallel generation after the 2026-10-17 disk-full crash
# (OSError ENOSPC killed chunks 1,3,4, the SBC-reduced worker, and the sequential run;
# all .partial.npz checkpoints were verified intact).
#
# Differences vs launch_sb3_wideprior_parallel.sh:
#  * every worker runs inside an auto-restart loop (up to 30 retries, 10 min apart), so a
#    transient disk-full / crash resumes from its own checkpoint instead of dying;
#  * rows are redistributed over the idle cores. Row accounting for exactly 5000 train rows:
#      sequential partial 3400 + chunk1 140 + chunk2 280 + chunk3 280 + chunk4 140
#      + chunk5 280 + chunk6 280 + chunk7 120 + chunk8 120 = 5040  (assembly trims to 5000)
#  * the SBC-reduced job and SBC-confirm part A become separate workers.
# chunk2 and chunk6 (still alive from the first launch) and chunk5 / partB (finished) are untouched.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CHUNK=scripts/generate_sb3_wideprior_chunk.py
SUF=_sb3_wideprior_stage1bfix
mkdir -p logs

launch() {  # launch <name> <out> <n_rows> <seed>
  local name=$1 out=$2 n=$3 seed=$4
  nohup nice -n 10 caffeinate -i bash -c "
    for i in \$(seq 1 30); do
      $PY $CHUNK --out '$out' --n-rows $n --seed $seed --label '$name' && exit 0
      echo \"[retry \$i] worker $name exited non-zero at \$(date); resuming in 600s\"
      sleep 600
    done
    exit 1
  " > "logs/sb3_wideprior_resume_${name}.log" 2>&1 < /dev/null &
  disown
  echo "launched ${name} -> ${out} (n_rows=${n}, pid $!)"
}

launch train_chunk1 "data/wu2003_sbi_train_sb3_n5000${SUF}_chunk1.npz" 140 910101
launch train_chunk3 "data/wu2003_sbi_train_sb3_n5000${SUF}_chunk3.npz" 280 910103
launch train_chunk4 "data/wu2003_sbi_train_sb3_n5000${SUF}_chunk4.npz" 140 910104
launch train_chunk7 "data/wu2003_sbi_train_sb3_n5000${SUF}_chunk7.npz" 120 910107
launch train_chunk8 "data/wu2003_sbi_train_sb3_n5000${SUF}_chunk8.npz" 120 910108
launch sbc_reduced  "results/sb3_sbc_set_n120${SUF}.npz"               120 9102
launch sbc_confirm_A "results/sb3_sbc_set_n400${SUF}_partA.npz"        130 910301
