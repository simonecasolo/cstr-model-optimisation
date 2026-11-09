"""Regenerate the S-C (feed-analyzer) training bank and SBC validation sets
under the Stage 1b-fixed matched protocol (`sbi_pipeline._matched_y0`, fix v2:
downstream-summary-convergence-checked steady-state warm start).

Mirrors `notebooks/24a_wu2003_sbi_training_sc.ipynb` Sections 1-3 exactly (same
prior, same noise, same checkpointing convention every 250 draws) but:
  - imports the prior from the library (`cstr_sbi.recycle.priors.
    truncated_lognormal_5d`) instead of nb24a's inline copy, per Stage 0.2;
  - uses N_TRAIN=5000, not nb24a's stale N_TRAIN=15000 (n=15000 was tested
    2026-09-02 and found evidence-worse; the solidification plan's Stage 2
    explicitly says not to scale past 5000 until it passes the gate);
  - writes to NEW filenames (`_stage1bfix` suffix) rather than overwriting the
    old, now-known-contaminated artifacts -- both are kept for direct
    before/after comparison, per this project's "never silently change
    something under an existing name" convention.

Run standalone (not via nbconvert) because nb24a's own bank-generation cell
previously hung under nbconvert for 2+ hours for unrelated reasons (see
HANDOFF.md); this project's established fallback is to run the identical code
as a script and let the notebook load the resulting files afterward.

Expect this to be SLOW: the Stage 1b fix makes each steady-state warm-start
draw ~10-20x more expensive than the old (broken) recipe (median ~9-11s/draw
vs ~0.1-1s/draw), because it is now actually integrating to a checked
convergence criterion instead of silently accepting an under-integrated
state. Checkpoints every 250 draws so this is safe to interrupt and resume.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.priors import truncated_lognormal_5d
from cstr_sbi.recycle.summaries import N_SUMMARIES_SC, N_SUMMARIES_SB
from sbi_pipeline import _structure_config, _simulate_summary

DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-C"
NOISE_PCT = 0.003
# 2026-09-03: user asked for a smaller n first, given the ~14-16h estimate at n=5000 --
# N_TRAIN=1000 gives a real, trainable bank (nb38's fixed_y0 experiments used n=3000; 1000
# is a reasonable reduced-first-pass size) at ~1000*10s =~ 2.8h instead of ~14-16h. Bump to
# 5000 (or beyond) once this smaller pass gives a usable initial signal.
N_TRAIN = 1000
N_SBC_REDUCED = 120
N_SBC_CONFIRM = 400

SUFFIX = "_stage1bfix"
SC_BANK_PATH = DATA / f"wu2003_sbi_train_sc_lognormal_true_n{N_TRAIN}{SUFFIX}.npz"
SC_SBC_REDUCED_PATH = RESULTS / f"24a_sysII_sc_lognormal_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
SC_SBC_CONFIRM_PATH = RESULTS / f"24a_sysII_sc_lognormal_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"
SB_PAIRED_BANK_PATH = DATA / f"wu2003_sbi_train_sb_paired_lognormal_true_n{N_TRAIN}{SUFFIX}.npz"

prior_logn = truncated_lognormal_5d()


def generate_lognormal_bank_sc(out_path, n_rows, seed, label):
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
        th = prior_logn.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th)
        summary_rows.append(s)
        if len(theta_rows) % 50 == 0:
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
    np.savez(out_path, thetas=theta_arr, summaries=summary_arr, attempts=attempts, seed=seed,
              structure=STRUCTURE, prior="truncated_lognormal_centered_nominal",
              scenario_specific_warm_start=True,
              note="Stage 1b fix v2 (summary-convergence-checked matched-protocol warm start)")
    print(f"[{label}] saved: {out_path}, theta={theta_arr.shape}, summaries={summary_arr.shape}, "
          f"wall_s={time.time()-t0:.0f}, accept_rate={len(theta_arr)/attempts:.3f}")
    return theta_arr, summary_arr


def main():
    t_all = time.time()

    theta_train, summary_train_sc = generate_lognormal_bank_sc(
        SC_BANK_PATH, N_TRAIN, seed=20260901, label="train n=5000")
    assert summary_train_sc.shape[1] == N_SUMMARIES_SC

    summary_train_sb_paired = summary_train_sc[:, :N_SUMMARIES_SB]
    if not SB_PAIRED_BANK_PATH.exists():
        np.savez(SB_PAIRED_BANK_PATH, thetas=theta_train, summaries=summary_train_sb_paired,
                  structure="S-B", prior="truncated_lognormal_centered_nominal",
                  scenario_specific_warm_start=True,
                  note="sliced from the Stage-1b-fixed paired S-C bank")
        print(f"saved paired S-B bank: {SB_PAIRED_BANK_PATH}")

    theta_sbc, summary_sbc_sc = generate_lognormal_bank_sc(
        SC_SBC_REDUCED_PATH, N_SBC_REDUCED, seed=20260902, label="sbc n=120")

    theta_sbc_c, summary_sbc_c = generate_lognormal_bank_sc(
        SC_SBC_CONFIRM_PATH, N_SBC_CONFIRM, seed=20260903, label="sbc n=400")

    print(f"\nAll done in {time.time()-t_all:.0f}s total.")


if __name__ == "__main__":
    main()
