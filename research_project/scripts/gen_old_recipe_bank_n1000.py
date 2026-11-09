"""Diagnostic side-experiment: generate an n=1000 S-C training bank under the
OLD (pre-Stage-1b-fix) matched-protocol warm start, to isolate whether the
per-parameter differences seen in the n_train=1000-on-fixed-bank run
(scripts/run_nb24a_stage1bfix_training.py) are attributable to the warm-start
fix itself, or simply to n_train dropping from 5000 to 1000.

Reimplements the OLD recipe explicitly (t_final=100, rtol=1e-4, atol=1e-6,
max_steps=100_000, unconditional sol.ys[-1], no convergence check) rather than
calling the now-fixed `sbi_pipeline._matched_y0` -- this script intentionally
reproduces the bug for comparison purposes only. Do not reuse this pattern
anywhere else.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax
from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB, NOMINAL_INLET, NOMINAL_Y0_EXPLICIT,
    simulate_trajectory_explicit_jit, extract_observations_explicit,
    simulate_to_steady_state_explicit,
)
from cstr_sbi.recycle.priors import truncated_lognormal_5d
from cstr_sbi.recycle.summaries import compute_summaries, N_SUMMARIES_SC

DATA = ROOT / "data"
STRUCTURE = "S-C"
NOISE_PCT = 0.003
N_TRAIN = 1000
OUT_PATH = DATA / f"wu2003_sbi_train_sc_lognormal_true_n{N_TRAIN}_oldrecipe.npz"

_old_ss_jit = jax.jit(
    lambda theta, ctrl: simulate_to_steady_state_explicit(
        theta, NOMINAL_INLET, ctrl, NOMINAL_Y0_EXPLICIT,
        t_final=100.0, rtol=1e-4, atol=1e-6, max_steps=100_000,
    )
)


def old_matched_y0(theta_np, ctrl):
    theta_jnp = jnp.array(theta_np, dtype=jnp.float32)
    return np.asarray(_old_ss_jit(theta_jnp, ctrl))  # unconditional, no convergence check -- the bug


def old_simulate_summary(theta_np, ctrl, rng):
    theta_jnp = jnp.array(theta_np, dtype=jnp.float32)
    y0 = old_matched_y0(theta_np, ctrl)
    ts, ys = simulate_trajectory_explicit_jit(
        theta_jnp, NOMINAL_INLET, ctrl, jnp.array(y0), t_final=2.0, n_save=120, rtol=1e-3, atol=1e-5,
    )
    raw = np.asarray(extract_observations_explicit(ys, theta_jnp, ctrl))
    t_np = np.asarray(ts)
    if np.isnan(raw).any() or np.isinf(raw).any():
        return None
    scale = np.maximum(np.max(np.abs(raw), axis=0), 1e-6)
    noise = rng.normal(0, NOISE_PCT * scale, raw.shape)
    s = compute_summaries(raw + noise, STRUCTURE, t_np, theta=theta_np, rng=rng)
    if np.isnan(s).any():
        return None
    return s


def main():
    if OUT_PATH.exists():
        print(f"already exists: {OUT_PATH}")
        return
    prior_logn = truncated_lognormal_5d()
    rng = np.random.default_rng(20260901)
    theta_rows, summary_rows = [], []
    t0 = time.time()
    attempts = 0
    while len(theta_rows) < N_TRAIN:
        attempts += 1
        th = prior_logn.sample((1,)).numpy()[0]
        s = old_simulate_summary(th, NOMINAL_CTRL_SB, rng)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th)
        summary_rows.append(s)
        if len(theta_rows) % 100 == 0:
            elapsed = time.time() - t0
            print(f"[oldrecipe n=1000] {len(theta_rows)}/{N_TRAIN}, attempts={attempts}, "
                  f"elapsed={elapsed:.0f}s, ~{elapsed/len(theta_rows):.3f}s/draw", flush=True)
    theta_arr = np.asarray(theta_rows, dtype=np.float32)
    summary_arr = np.asarray(summary_rows, dtype=np.float32)
    assert summary_arr.shape[1] == N_SUMMARIES_SC
    np.savez(OUT_PATH, thetas=theta_arr, summaries=summary_arr, attempts=attempts, seed=20260901,
             structure=STRUCTURE, prior="truncated_lognormal_centered_nominal",
             scenario_specific_warm_start=True,
             note="OLD (pre-Stage-1b-fix) matched-protocol warm start -- diagnostic comparison only")
    print(f"saved: {OUT_PATH}, theta={theta_arr.shape}, summaries={summary_arr.shape}, "
          f"wall_s={time.time()-t0:.0f}, accept_rate={len(theta_arr)/attempts:.3f}")


if __name__ == "__main__":
    main()
