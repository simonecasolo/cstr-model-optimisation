"""Assemble the parallel-generated S-B3 wide-prior chunks into the canonical files.

Writes (same names/keys `generate_sb3_wideprior_bank.py` would have produced):
  data/wu2003_sbi_train_sb3_n5000_sb3_wideprior_stage1bfix.npz        (exactly 5000 rows)
  results/sb3_sbc_set_n400_sb3_wideprior_stage1bfix.npz               (partA + partB = 400 rows)
(results/sb3_sbc_set_n120_... is already written directly by worker 7.)

Train rows = the sequential run's .partial.npz first (earliest rows, in order), then
chunk1..chunk6, truncated to N_TRAIN. STOP the sequential process (PID in the log) before
running this so its last checkpoint is final. Refuses to run if anything is missing or
if it finds duplicated rows.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
SUF = "_sb3_wideprior_stage1bfix"
N_TRAIN, N_CONFIRM = 5000, 400
STRUCT = "S-B"
PARAMS = ["alpha", "beta_r", "eta_col", "xi_reb"]


def load(p: pathlib.Path):
    if not p.exists():
        sys.exit(f"MISSING: {p.name} (generation not finished?)")
    d = np.load(p)
    return np.asarray(d["thetas"], np.float32), np.asarray(d["summaries"], np.float32), int(np.asarray(d["attempts"]).item())


def check_clean(th, sm, name):
    assert np.isfinite(th).all() and np.isfinite(sm).all(), f"{name}: non-finite values"
    assert len(np.unique(th, axis=0)) == len(th), f"{name}: duplicated theta rows"


def save(path, th, sm, attempts, note):
    np.savez(path, thetas=th, summaries=sm, attempts=attempts, structure=STRUCT,
             prior="box_uniform_4d_sb3", param_names=PARAMS, scenario_specific_warm_start=True, note=note)
    print(f"saved {path.name}: {len(th)} rows")


if __name__ == "__main__":
    # ---- train bank
    import re
    seq = DATA / f"wu2003_sbi_train_sb3_n{N_TRAIN}{SUF}.partial.npz"
    prefix = f"wu2003_sbi_train_sb3_n{N_TRAIN}{SUF}_chunk"
    unfinished = sorted(p.name for p in DATA.glob(prefix + "*.partial.npz"))
    if unfinished:
        sys.exit(f"UNFINISHED chunks still have .partial.npz files: {unfinished}")
    chunk_files = sorted((p for p in DATA.glob(prefix + "*.npz") if not p.name.endswith(".partial.npz")),
                         key=lambda p: int(re.search(r"chunk(\d+)\.npz$", p.name).group(1)))
    if not chunk_files:
        sys.exit("no finished chunk files found")
    parts = [load(seq)] + [load(p) for p in chunk_files]
    th = np.concatenate([p[0] for p in parts]); sm = np.concatenate([p[1] for p in parts])
    att = sum(p[2] for p in parts)
    print(f"train: sequential={len(parts[0][0])} + chunks={[len(p[0]) for p in parts[1:]]} = {len(th)} rows")
    if len(th) < N_TRAIN:
        sys.exit(f"only {len(th)} < {N_TRAIN} train rows; run more chunk rows first")
    th, sm = th[:N_TRAIN], sm[:N_TRAIN]
    check_clean(th, sm, "train")
    save(DATA / f"wu2003_sbi_train_sb3_n{N_TRAIN}{SUF}.npz", th, sm, att,
         "assembled from sequential partial + 6 parallel chunks; clean stage1bfix matched-protocol, S-B3 box-uniform")

    # ---- sbc confirm
    a, b = load(RESULTS / f"sb3_sbc_set_n400{SUF}_partA.npz"), load(RESULTS / f"sb3_sbc_set_n400{SUF}_partB.npz")
    th, sm = np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])
    assert len(th) == N_CONFIRM, f"sbc_confirm has {len(th)} rows, expected {N_CONFIRM}"
    check_clean(th, sm, "sbc_confirm")
    save(RESULTS / f"sb3_sbc_set_n400{SUF}.npz", th, sm, a[2] + b[2],
         "assembled from 2 parallel parts; clean stage1bfix matched-protocol, S-B3 box-uniform")
    print("done.")
