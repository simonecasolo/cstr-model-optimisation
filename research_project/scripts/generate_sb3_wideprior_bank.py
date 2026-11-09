"""Generate a clean (post-warm-start-fix) S-B3 training bank + two SBC sets
under `box_uniform_4d_sb3()` -- all 4 endogenous parameters (alpha, beta_r,
eta_col, xi_reb) widened to Uniform over their prior box, `z_A0_eff` fixed
exogenous at its nominal value (the S-B3 framing: same S-B instrumentation
and simulator, narrower fault taxonomy, feed-fault scenarios excluded).

This is the honest counterpart to `notebooks/legacy_systemII/31e`'s
seed-cherry-picked S-B3 result: multi-seed, `baseline` (default) z-scoring
(the lesson from the full S-B wide-prior run -- `z_score_x='none'` silently
collapses posterior informativeness in a full-uniform-prior context), proper
SBC with the pre-registered anti-shrinkage-slope check, no seed selection by
classification outcome alone.

Reuses the exact same clean, matched-protocol simulation pipeline already
validated for the full S-B run (`scripts/sbi_pipeline._simulate_summary`,
`scenario_specific_warm_start=True`) -- the only difference is `z_A0_eff` is
fixed to nominal in the 5-D theta vector before simulating
(`cstr_sbi.recycle.sb3.theta4_to_theta5`), and the prior draws only cover the
4 free dimensions.

Checkpointed every 50 accepted rows, resumable (loads `.partial.npz` and
continues) -- same convention as `scripts/generate_sb_wideprior_bank.py`.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.sb3 import box_uniform_4d_sb3, theta4_to_theta5, SB3_PARAM_NAMES
from sbi_pipeline import _structure_config, _simulate_summary

DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-B"
NOISE_PCT = 0.003
SUFFIX = "_sb3_wideprior_stage1bfix"
N_TRAIN = 5000
N_SBC_REDUCED = 120
N_SBC_CONFIRM = 400
CHECKPOINT_EVERY = 50

prior4 = box_uniform_4d_sb3()
ctrl, y0 = _structure_config(STRUCTURE)

LOG_PATH = DATA / f"wu2003_sbi_train_sb3_n{N_TRAIN}{SUFFIX}.progress.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def generate_bank(out_path, n_rows, seed, label):
    """Generate (or resume generating) `n_rows` accepted (theta4, summary) pairs.
    `theta4` is the 4-D S-B3 parameter vector (z_A0_eff NOT included -- fixed
    to nominal internally before simulating).
    """
    if out_path.exists():
        bank = np.load(out_path)
        log(f"[{label}] loading existing: {out_path}")
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
        log(f"[{label}] resuming from partial: {len(theta_rows)}/{n_rows} rows, {attempts} attempts")
    t0 = time.time()
    while len(theta_rows) < n_rows:
        attempts += 1
        th4 = prior4.sample((1,)).numpy()[0]
        th5 = theta4_to_theta5(th4)
        s = _simulate_summary(th5, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th4.astype(np.float32))
        summary_rows.append(s.astype(np.float32))
        if len(theta_rows) % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            eta_s = elapsed / len(theta_rows) * (n_rows - len(theta_rows))
            log(f"[{label}] {len(theta_rows)}/{n_rows} rows, {attempts} attempts, "
                f"{elapsed:.0f}s elapsed, accept_rate={len(theta_rows)/attempts:.2f}, "
                f"ETA={eta_s/3600:.1f}h")
            np.savez(partial_path, thetas=np.stack(theta_rows), summaries=np.stack(summary_rows), attempts=attempts)
    np.savez(out_path, thetas=np.stack(theta_rows), summaries=np.stack(summary_rows),
             attempts=attempts, structure=STRUCTURE, prior="box_uniform_4d_sb3",
             param_names=list(SB3_PARAM_NAMES), scenario_specific_warm_start=True,
             note="clean stage1bfix matched-protocol warm start, S-B3: alpha/beta_r/eta_col/xi_reb "
                  "widened to uniform, z_A0_eff fixed exogenous at nominal")
    if partial_path.exists():
        partial_path.unlink()
    log(f"[{label}] DONE: {len(theta_rows)} rows, {attempts} attempts, "
        f"final accept_rate={len(theta_rows)/attempts:.2f}, {time.time()-t0:.0f}s total")
    return np.stack(theta_rows), np.stack(summary_rows)


if __name__ == "__main__":
    train_path = DATA / f"wu2003_sbi_train_sb3_n{N_TRAIN}{SUFFIX}.npz"
    sbc_reduced_path = RESULTS / f"sb3_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
    sbc_confirm_path = RESULTS / f"sb3_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"

    log(f"=== (re)started at {time.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"pid={__import__('os').getpid()} === tail -f {LOG_PATH} to follow progress")

    generate_bank(train_path, N_TRAIN, seed=9101, label="train")
    generate_bank(sbc_reduced_path, N_SBC_REDUCED, seed=9102, label="sbc_reduced")
    generate_bank(sbc_confirm_path, N_SBC_CONFIRM, seed=9103, label="sbc_confirm")
    log("\nAll three artifacts done.")
