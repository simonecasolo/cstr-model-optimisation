"""Generate a clean (post-warm-start-fix) S-B training bank + two SBC sets
under `mixed_alpha_uniform_prior()` (alpha ~ Uniform(0.4,1.2), the other 4
parameters keep their existing narrow lognormal marginals) -- the
already-validated-for-S-C, cheaper partial-widening fix (accept rate ~75%,
measured directly for S-B before launching this: see chat log), rather than
the full `mixed_wide_prior()` (all 4 widened, ~40% accept rate, ~8-9h for a
useful bank -- too expensive to run in this session).

Tests the hypothesis that the pure-lognormal-prior S-B posterior's
catastrophic classification failure (`scripts/eval_sb_armc_classification.py`:
macro-F1=0.089, severe faults shrunk toward healthy) is a training-prior-
coverage problem, not a genuine identifiability limit -- exactly the
mechanism already confirmed for S-C
(`docs/Manuscript/system_ii_sbi_solidification_plan.md`, "Mixed-prior fix").

Reduced sizes relative to the S-C precedent (n_train=1000) to fit a
~2h background budget at this prior's measured ~20s/accepted-row cost:
n_train=200, SBC reduced=40, SBC confirm=100. This is a directional pilot,
not a publication-grade bank -- scale up only if this gives a clear signal.

Checkpointed every 25 accepted rows so it is safe to interrupt and resume.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.priors import mixed_alpha_uniform_prior
from sbi_pipeline import _structure_config, _simulate_summary

DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-B"
NOISE_PCT = 0.003
SUFFIX = "_mixedalpha_stage1bfix"
N_TRAIN = 200
N_SBC_REDUCED = 40
N_SBC_CONFIRM = 100
CHECKPOINT_EVERY = 25

prior = mixed_alpha_uniform_prior()
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
             attempts=attempts, structure=STRUCTURE, prior="mixed_alpha_uniform",
             scenario_specific_warm_start=True,
             note="clean stage1bfix matched-protocol warm start, alpha-widened-uniform prior")
    if partial_path.exists():
        partial_path.unlink()
    print(f"[{label}] DONE: {len(theta_rows)} rows, {attempts} attempts, "
          f"final accept_rate={len(theta_rows)/attempts:.2f}, {time.time()-t0:.0f}s total")
    return np.stack(theta_rows), np.stack(summary_rows)


train_path = DATA / f"wu2003_sbi_train_sb_n{N_TRAIN}{SUFFIX}.npz"
sbc_reduced_path = RESULTS / f"sb_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
sbc_confirm_path = RESULTS / f"sb_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"

generate_bank(train_path, N_TRAIN, seed=7001, label="train")
generate_bank(sbc_reduced_path, N_SBC_REDUCED, seed=7002, label="sbc_reduced")
generate_bank(sbc_confirm_path, N_SBC_CONFIRM, seed=7003, label="sbc_confirm")
print("\nAll three artifacts done.")
