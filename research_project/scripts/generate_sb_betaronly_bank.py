"""Isolation test: widen ONLY beta_r's training prior to Uniform(0.4,1.2)
(mixed_wide_prior(uniform_idx=(1,))), keeping alpha/eta_col/xi_reb/z_A0_eff on
their existing narrow lognormal marginals. Same n_train=200 pilot scale and
checkpointing convention as `scripts/generate_sb_mixedalpha_bank.py` (the
alpha-only isolation test).

Purpose: alpha showed a persistent, seed-dependent, never-fully-explained SNPE
training instability under every widened-prior variant tested
(`system_ii_sbi_solidification_plan.md`). beta_r has been the single most
robustly-calibrating parameter in every experiment so far, but only ever
tested under the NARROW prior -- this checks whether that robustness survives
widening, before committing ~105h to the full 4-parameter
(`scripts/generate_sb_wideprior_bank.py`) campaign that includes beta_r.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.priors import mixed_wide_prior
from sbi_pipeline import _structure_config, _simulate_summary

DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-B"
NOISE_PCT = 0.003
SUFFIX = "_betaronly_stage1bfix"
N_TRAIN = 200
N_SBC_REDUCED = 40
N_SBC_CONFIRM = 100
CHECKPOINT_EVERY = 25

UNIFORM_IDX = (1,)  # beta_r only

prior = mixed_wide_prior(uniform_idx=UNIFORM_IDX)
ctrl, y0 = _structure_config(STRUCTURE)


def generate_bank(out_path, n_rows, seed, label):
    if out_path.exists():
        bank = np.load(out_path)
        print(f"[{label}] loading existing: {out_path}")
        return np.asarray(bank["thetas"], dtype=np.float32), np.asarray(bank["summaries"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    theta_rows, summary_rows = [], []
    attempts = 0
    partial_path = out_path.with_suffix(".partial.npz")
    if partial_path.exists():
        partial = np.load(partial_path)
        theta_rows = list(np.asarray(partial["thetas"], dtype=np.float32))
        summary_rows = list(np.asarray(partial["summaries"], dtype=np.float32))
        attempts = int(np.asarray(partial["attempts"]).item()) if "attempts" in partial.files else len(theta_rows)
        print(f"[{label}] resuming from partial: {len(theta_rows)}/{n_rows} rows, {attempts} attempts")
    t0 = time.time()
    while len(theta_rows) < n_rows:
        attempts += 1
        th = prior.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th.astype(np.float32))
        summary_rows.append(s.astype(np.float32))
        if len(theta_rows) % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            print(f"[{label}] {len(theta_rows)}/{n_rows} rows, {attempts} attempts, "
                  f"{elapsed:.0f}s elapsed, accept_rate={len(theta_rows)/attempts:.2f}")
            np.savez(partial_path, thetas=np.stack(theta_rows), summaries=np.stack(summary_rows), attempts=attempts)
    np.savez(out_path, thetas=np.stack(theta_rows), summaries=np.stack(summary_rows),
             attempts=attempts, structure=STRUCTURE, prior="mixed_wide_uniform_idx_1_betar_only",
             scenario_specific_warm_start=True,
             note="clean stage1bfix matched-protocol warm start, beta_r widened to uniform only")
    if partial_path.exists():
        partial_path.unlink()
    print(f"[{label}] DONE: {len(theta_rows)} rows, {attempts} attempts, "
          f"final accept_rate={len(theta_rows)/attempts:.2f}, {time.time()-t0:.0f}s total")
    return np.stack(theta_rows), np.stack(summary_rows)


if __name__ == "__main__":
    train_path = DATA / f"wu2003_sbi_train_sb_n{N_TRAIN}{SUFFIX}.npz"
    sbc_reduced_path = RESULTS / f"sb_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
    sbc_confirm_path = RESULTS / f"sb_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"

    generate_bank(train_path, N_TRAIN, seed=9001, label="train")
    generate_bank(sbc_reduced_path, N_SBC_REDUCED, seed=9002, label="sbc_reduced")
    generate_bank(sbc_confirm_path, N_SBC_CONFIRM, seed=9003, label="sbc_confirm")
    print("\nAll three artifacts done.")
