"""Generate an S-C.2 training bank + 2 SBC sets under the WIDE UNIFORM prior
(`box_uniform_5d()`), not the narrow lognormal prior every S-C/S-C.2 posterior
this session has used so far.

Motivation (2026-09-04): the S-C.2 seed-0 posterior (lognormal-prior-trained)
passed every SBC check but failed fault classification catastrophically
(macro-F1~0.07) -- diagnosed as severe shrinkage-toward-prior-mean, because
the Wu2003 named fault scenarios sit 1-3.4 log-sigma into the lognormal
prior's tail, a region the training bank barely covers. The ORIGINAL,
well-performing nb31 S-B posterior (`sbi-logs/wu2003_posterior_sb.pkl`) was
confirmed (checked directly: flat histogram over [0.4,1.2], std matching the
uniform-distribution theoretical value exactly) to have been trained on the
WIDE UNIFORM prior. A later attempt to retrain that same uniform prior under
the matched protocol failed SBC catastrophically (`wu2003_posterior_sb_
matched_full.pkl`, the failure that triggered nb35-38's long unsuccessful
debugging and the eventual switch to the narrow lognormal prior) -- but that
attempt PREDATES this session's warm-start bug fix (Stage 1b). This script
tests whether combining the wide uniform prior with everything now fixed
(corrected warm start, S-C.2 xi_reb features) finally achieves both
calibration AND classification usefulness simultaneously.

Mirrors scripts/regenerate_sc_bank_stage1bfix.py's structure/checkpointing
exactly, just: (a) box_uniform_5d() instead of truncated_lognormal_5d(), (b)
structure="S-C.2" directly (includes the xi_reb-sharpening features).
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.priors import box_uniform_5d
from cstr_sbi.recycle.summaries import N_SUMMARIES_SC2
from sbi_pipeline import _structure_config, _simulate_summary

DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-C.2"
NOISE_PCT = 0.003
N_TRAIN = 1000
N_SBC_REDUCED = 120
N_SBC_CONFIRM = 400

SUFFIX = "_uniform_stage1bfix"
SC2_BANK_PATH = DATA / f"wu2003_sbi_train_sc2_true_n{N_TRAIN}{SUFFIX}.npz"
SC2_SBC_REDUCED_PATH = RESULTS / f"sc2_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
SC2_SBC_CONFIRM_PATH = RESULTS / f"sc2_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"

torch.manual_seed(20260904)
prior_uniform = box_uniform_5d()


def generate_bank(out_path, n_rows, seed, label):
    if out_path.exists():
        bank = np.load(out_path)
        print(f"[{label}] loading existing: {out_path}")
        return np.asarray(bank["thetas"], dtype=np.float32), np.asarray(bank["summaries"], dtype=np.float32)
    ctrl, y0 = _structure_config(STRUCTURE)
    rng = np.random.default_rng(seed)
    theta_rows, summary_rows = [], []
    t0 = time.time()
    attempts = 0
    partial_path = out_path.with_suffix(".partial.npz")
    if partial_path.exists():
        partial = np.load(partial_path)
        theta_rows = list(np.asarray(partial["thetas"], dtype=np.float32))
        summary_rows = list(np.asarray(partial["summaries"], dtype=np.float32))
        attempts = int(np.asarray(partial["attempts"]).item()) if "attempts" in partial.files else len(theta_rows)
        print(f"[{label}] resuming from partial: {len(theta_rows)}/{n_rows} rows, {attempts} attempts")
    while len(theta_rows) < n_rows:
        attempts += 1
        th = prior_uniform.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th)
        summary_rows.append(s)
        if len(theta_rows) % 25 == 0:
            elapsed = time.time() - t0
            rate = elapsed / len(theta_rows)
            eta = rate * (n_rows - len(theta_rows))
            accept_rate = len(theta_rows) / attempts
            print(f"[{label}] {len(theta_rows)}/{n_rows}, attempts={attempts} "
                  f"(accept_rate={accept_rate:.3f}), elapsed={elapsed:.0f}s, "
                  f"~{rate:.2f}s/draw, ETA={eta/60:.1f}min", flush=True)
        if len(theta_rows) % 250 == 0:
            np.savez(partial_path,
                     thetas=np.asarray(theta_rows, dtype=np.float32),
                     summaries=np.asarray(summary_rows, dtype=np.float32), attempts=attempts)
    theta_arr = np.asarray(theta_rows, dtype=np.float32)
    summary_arr = np.asarray(summary_rows, dtype=np.float32)
    assert summary_arr.shape[1] == N_SUMMARIES_SC2
    np.savez(out_path, thetas=theta_arr, summaries=summary_arr, attempts=attempts, seed=seed,
             structure=STRUCTURE, prior="box_uniform_5d",
             scenario_specific_warm_start=True,
             note="wide uniform prior, Stage 1b fix v2 warm start, S-C.2 features")
    print(f"[{label}] saved: {out_path}, theta={theta_arr.shape}, summaries={summary_arr.shape}, "
          f"wall_s={time.time()-t0:.0f}, accept_rate={len(theta_arr)/attempts:.3f}")
    return theta_arr, summary_arr


def main():
    t_all = time.time()
    generate_bank(SC2_BANK_PATH, N_TRAIN, seed=20260901, label="train n=1000")
    generate_bank(SC2_SBC_REDUCED_PATH, N_SBC_REDUCED, seed=20260902, label="sbc n=120")
    generate_bank(SC2_SBC_CONFIRM_PATH, N_SBC_CONFIRM, seed=20260903, label="sbc n=400")
    print(f"\nAll done in {time.time()-t_all:.0f}s total.")


if __name__ == "__main__":
    main()
