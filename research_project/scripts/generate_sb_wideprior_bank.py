"""Generate a clean (post-warm-start-fix) S-B training bank + two SBC sets
under `mixed_wide_prior(uniform_idx=(0,1,2,3,4))` -- ALL FIVE of alpha,
beta_r, eta_col, xi_reb, z_A0_eff widened to Uniform over their prior box
(this is, by construction, equivalent to `box_uniform_5d()` once every index
is uniform -- no lognormal component remains).

z_A0_eff was initially excluded on the grounds that its fault-scenario
deviations were "proportionally small" -- that claim did not survive its own
quantitative check (notebook 35, Section 3): z_A0_eff's max deviation from
the narrow lognormal's median across the 14-scenario catalogue is 2.60
log-sigma, *larger* than eta_col's and xi_reb's (2.23 each), which were
widened on exactly that magnitude argument. z_A0_eff is now included for
consistency. The real, still-valid reason to expect z_A0_eff's OWN behaviour
to improve less than alpha/beta_r/eta_col/xi_reb's is mechanistic, not
magnitude-based: z_A0_eff has a confirmed, representation-robust confound
with alpha and eta_col (Artefact 3 -- checked 5 independent ways, survives
the raw-trajectory FIM test that collapses genuine representation artefacts).
That is an information-theoretic limit of S-B's instrumentation (no
feed-composition sensor), not a training-prior-coverage problem -- widening
z_A0_eff's prior does not resolve it, so `feed`-class classification is
expected to remain unreliable regardless. What widening DOES get us: honest,
non-shrunk z_A0_eff point estimates and correct SBC behaviour for it,
instead of the same prior-mean-shrinkage artifact alpha showed before being
widened -- worth having even though it does not fix the confound itself.

This is "Step 1" of `notebooks/legacy_systemII/44_wu2003_wideprior_multiround_classification.ipynb`'s
recipe, adapted from S-C.2 to plain S-B, WITHOUT that notebook's Step 4
per-scenario round-2 multi-round refinement (out of scope here -- that is an
expensive, still-experimental extra on top of the bank itself, never
completed, and not what was asked for). See
`notebooks/35_wu2003_sb_wideprior_bank_generation.ipynb` for the full
reasoning and the prior-mass-coverage check against the 14-scenario
catalogue.

Checkpointed every 50 accepted rows (tighter than nb44's every-250 convention,
per explicit instruction -- this run is expensive enough, and the machine
uptime uncertain enough, that losing 250 rows of a run estimated to take
multiple days is a real risk) -- safe to interrupt (Ctrl-C, process kill, or
machine sleep/restart) and resume by just re-running this script; it loads
and continues from the `.partial.npz` checkpoint automatically.

Cost: a 10-row smoke test with the 4-parameter-widened version of this recipe
(S-B, alpha/beta_r/eta_col/xi_reb) measured 68.4s/accepted row at 43% accept
rate (notebook 35, Section 4) -- projecting ~105h for the full
n_train=5000 + SBC campaign. Widening z_A0_eff too is expected to lower the
accept rate somewhat further (one more dimension that can land in a
hard-to-converge combination); re-measure with a fresh smoke test before
trusting a revised ETA. Run under `caffeinate -i -s` (macOS) if running
interactively so the machine does not sleep mid-run; the background-process
launcher already handles this project's convention of running standalone
(not via `jupyter nbconvert`, which has hung for multi-hour jobs before in
this project, see HANDOFF.md).
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
SUFFIX = "_wideprior_stage1bfix"
N_TRAIN = 5000
N_SBC_REDUCED = 120
N_SBC_CONFIRM = 400
CHECKPOINT_EVERY = 50

# Widen all 5: alpha(0), beta_r(1), eta_col(2), xi_reb(3), z_A0_eff(4).
UNIFORM_IDX = (0, 1, 2, 3, 4)

prior = mixed_wide_prior(uniform_idx=UNIFORM_IDX)
ctrl, y0 = _structure_config(STRUCTURE)

# Single shared, human-tailable progress log for the whole campaign (train +
# both SBC sets, in sequence) -- separate from stdout/the harness's own
# background-task output capture, so progress can be tracked with a plain
# `tail -f` from any terminal regardless of how this process was launched.
LOG_PATH = DATA / f"wu2003_sbi_train_sb_n{N_TRAIN}{SUFFIX}.progress.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def generate_bank(out_path, n_rows, seed, label):
    """Generate (or resume generating) `n_rows` accepted (theta, summary) pairs,
    checkpointing to `out_path.with_suffix('.partial.npz')` every
    `CHECKPOINT_EVERY` accepted rows. Returns (thetas, summaries) arrays.
    Safe to call again after an interruption -- picks up from the checkpoint.
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
        th = prior.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th.astype(np.float32))
        summary_rows.append(s.astype(np.float32))
        if len(theta_rows) % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            eta_s = elapsed / len(theta_rows) * (n_rows - len(theta_rows))
            log(f"[{label}] {len(theta_rows)}/{n_rows} rows, {attempts} attempts, "
                f"{elapsed:.0f}s elapsed, accept_rate={len(theta_rows)/attempts:.2f}, "
                f"ETA={eta_s/3600:.1f}h")
            np.savez(partial_path, thetas=np.stack(theta_rows), summaries=np.stack(summary_rows), attempts=attempts)
    np.savez(out_path, thetas=np.stack(theta_rows), summaries=np.stack(summary_rows),
             attempts=attempts, structure=STRUCTURE, prior="mixed_wide_uniform_idx_01234",
             scenario_specific_warm_start=True,
             note="clean stage1bfix matched-protocol warm start, alpha/beta_r/eta_col/xi_reb widened to uniform")
    if partial_path.exists():
        partial_path.unlink()
    log(f"[{label}] DONE: {len(theta_rows)} rows, {attempts} attempts, "
        f"final accept_rate={len(theta_rows)/attempts:.2f}, {time.time()-t0:.0f}s total")
    return np.stack(theta_rows), np.stack(summary_rows)


if __name__ == "__main__":
    train_path = DATA / f"wu2003_sbi_train_sb_n{N_TRAIN}{SUFFIX}.npz"
    sbc_reduced_path = RESULTS / f"sb_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
    sbc_confirm_path = RESULTS / f"sb_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"

    log(f"=== (re)started at {time.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"pid={__import__('os').getpid()} === tail -f {LOG_PATH} to follow progress")

    generate_bank(train_path, N_TRAIN, seed=8001, label="train")
    generate_bank(sbc_reduced_path, N_SBC_REDUCED, seed=8002, label="sbc_reduced")
    generate_bank(sbc_confirm_path, N_SBC_CONFIRM, seed=8003, label="sbc_confirm")
    print("\nAll three artifacts done.")
