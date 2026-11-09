"""Generate an S-C.2 training bank + 2 SBC sets under the MIXED prior:
`alpha ~ Uniform(0.4, 1.2)`, the other 4 parameters their existing
truncated-lognormal marginals (`cstr_sbi.recycle.priors.mixed_alpha_uniform_prior`).

Rationale (2026-09-05): `alpha` was found to be the ONLY parameter with a
catastrophic, seed-dependent shrinkage-toward-prior-mean failure across every
S-C.2 variant tested this session. A full 5-D uniform-prior bank attempt was
tried and aborted -- accept rate collapsed to ~37% (vs ~91-95% for the
all-lognormal bank) because multiple parameters going simultaneously extreme
at once makes the matched-protocol warm-start solve hard to converge. This
mixed prior targets the diagnosed problem directly (only alpha's coverage was
ever shown to be the issue) while keeping the other four parameters in their
well-behaved narrow range, expected to avoid that blowup.

Mirrors scripts/regenerate_sc2_uniform_bank.py's structure/checkpointing
exactly, just swapping the prior.
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

from cstr_sbi.recycle.priors import mixed_alpha_uniform_prior
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

SUFFIX = "_mixedprior_stage1bfix"
SC2_BANK_PATH = DATA / f"wu2003_sbi_train_sc2_true_n{N_TRAIN}{SUFFIX}.npz"
SC2_SBC_REDUCED_PATH = RESULTS / f"sc2_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
SC2_SBC_CONFIRM_PATH = RESULTS / f"sc2_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"

torch.manual_seed(20260905)
prior_mixed = mixed_alpha_uniform_prior()


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
        th = prior_mixed.sample((1,)).numpy()[0]
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
             structure=STRUCTURE, prior="mixed_alpha_uniform_prior",
             scenario_specific_warm_start=True,
             note="alpha uniform(0.4,1.2), rest lognormal; Stage 1b fix v2 warm start, S-C.2 features")
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
