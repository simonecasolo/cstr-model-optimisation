"""Generate an S-C.2 training bank + 2 SBC sets under the WIDE MIXED prior:
`alpha, beta_r, eta_col, xi_reb` all ~ Uniform over their prior box,
`z_A0_eff` ~ its existing truncated-lognormal marginal
(`cstr_sbi.recycle.priors.mixed_wide_prior`).

Rationale (2026-09-05, `system_ii_sbi_solidification_plan.md` priority-0
follow-up): widening only `alpha` (`regenerate_sc2_mixedprior_bank.py`) fixed
that parameter's coverage mismatch and roughly doubled classification
macro-F1 (0.07->0.11), but `beta_r`/`eta_col`/`xi_reb` plausibly have the same
narrow-lognormal/Wu2003-fault-catalogue coverage mismatch (their own fault
values sit 0.60-0.90, multiple log-sigma into the narrow core). `z_A0_eff` is
kept narrow -- its near-exact dedicated feed-analyzer signal makes it largely
prior-width-insensitive. Only run this AFTER `smoketest_sc2_wideprior.py`
confirms the accept rate is viable (widening 4/5 dims is a new,
not-yet-quantified point between the alpha-only success (~75%) and the full
5-D failure (~37%)).

Mirrors scripts/regenerate_sc2_mixedprior_bank.py's structure/checkpointing
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

from cstr_sbi.recycle.priors import mixed_wide_prior
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

SUFFIX = "_wideprior_stage1bfix"
SC2_BANK_PATH = DATA / f"wu2003_sbi_train_sc2_true_n{N_TRAIN}{SUFFIX}.npz"
SC2_SBC_REDUCED_PATH = RESULTS / f"sc2_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
SC2_SBC_CONFIRM_PATH = RESULTS / f"sc2_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"

torch.manual_seed(20260905)
prior_wide = mixed_wide_prior()
PROGRESS_EVERY = 100
CHECKPOINT_EVERY = 250


def _emit_progress(progress_path, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    progress_path.parent.mkdir(exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def generate_bank(out_path, n_rows, seed, label):
    progress_path = out_path.with_suffix(".progress.log")
    if out_path.exists():
        bank = np.load(out_path)
        _emit_progress(progress_path, f"[{label}] loading existing: {out_path}")
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
        _emit_progress(progress_path, f"[{label}] resuming from partial: {len(theta_rows)}/{n_rows} rows, {attempts} attempts")
    else:
        _emit_progress(progress_path, f"[{label}] starting fresh: target={n_rows}, seed={seed}, out={out_path}")
    while len(theta_rows) < n_rows:
        attempts += 1
        th = prior_wide.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th)
        summary_rows.append(s)
        if len(theta_rows) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = elapsed / len(theta_rows)
            eta = rate * (n_rows - len(theta_rows))
            accept_rate = len(theta_rows) / attempts
            _emit_progress(
                progress_path,
                f"[{label}] accepted={len(theta_rows)}/{n_rows}, attempts={attempts} "
                f"(accept_rate={accept_rate:.3f}), elapsed={elapsed:.0f}s, "
                f"~{rate:.2f}s/draw, ETA={eta/60:.1f}min",
            )
        if len(theta_rows) % CHECKPOINT_EVERY == 0:
            np.savez(partial_path,
                     thetas=np.asarray(theta_rows, dtype=np.float32),
                     summaries=np.asarray(summary_rows, dtype=np.float32), attempts=attempts)
            _emit_progress(progress_path, f"[{label}] checkpoint saved: {partial_path} ({len(theta_rows)}/{n_rows})")
    theta_arr = np.asarray(theta_rows, dtype=np.float32)
    summary_arr = np.asarray(summary_rows, dtype=np.float32)
    assert summary_arr.shape[1] == N_SUMMARIES_SC2
    np.savez(out_path, thetas=theta_arr, summaries=summary_arr, attempts=attempts, seed=seed,
             structure=STRUCTURE, prior="mixed_wide_prior",
             scenario_specific_warm_start=True,
             note="alpha/beta_r/eta_col/xi_reb uniform over prior box, z_A0_eff lognormal; "
                  "Stage 1b fix v2 warm start, S-C.2 features")
    _emit_progress(progress_path,
                   f"[{label}] saved: {out_path}, theta={theta_arr.shape}, summaries={summary_arr.shape}, "
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
