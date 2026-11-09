"""Solidification-plan Stage 1c: generate a second, independent N=400 lognormal
S-C SBC set (different seed from `results/24a_sysII_sc_lognormal_sbc_set_n400.npz`,
which was generated with seed=20260903 in notebooks/24a).

Every 24a/n15000 conclusion to date rests on that ONE fixed SBC-set realization.
This script produces a second one so any future confirmation run can report
pass/fail on both, per `system_ii_sbi_solidification_plan.md` Stage 0.3's pass gate
("two independent SBC sets").

Reuses the exact bank-generation recipe from notebooks/24a (matched protocol,
scenario_specific_warm_start=True, S-C structure, truncated-lognormal prior --
now imported from the library instead of re-pasted, per Stage 0.2).
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
from sbi_pipeline import _structure_config, _simulate_summary
from cstr_sbi.recycle.summaries import N_SUMMARIES_SC

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-C"
NOISE_PCT = 0.003
N_ROWS = 400
SEED = 20260904  # distinct from 24a's 20260902 (n=120) and 20260903 (n=400)
OUT_PATH = RESULTS / f"44_sysII_sc_lognormal_sbc_set_n{N_ROWS}_setB.npz"


def main():
    if OUT_PATH.exists():
        print(f"already exists, not regenerating: {OUT_PATH}")
        return
    prior_logn = truncated_lognormal_5d()
    ctrl, y0 = _structure_config(STRUCTURE)
    rng = np.random.default_rng(SEED)
    theta_rows, summary_rows = [], []
    t0 = time.time()
    attempts = 0
    while len(theta_rows) < N_ROWS:
        attempts += 1
        th = prior_logn.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th)
        summary_rows.append(s)
        if len(theta_rows) % 100 == 0:
            print(f"[bank] {len(theta_rows)}/{N_ROWS}, attempts={attempts}, elapsed={time.time()-t0:.0f}s")

    theta_arr = np.asarray(theta_rows, dtype=np.float32)
    summary_arr = np.asarray(summary_rows, dtype=np.float32)
    assert summary_arr.shape[1] == N_SUMMARIES_SC
    np.savez(
        OUT_PATH,
        thetas=theta_arr,
        summaries=summary_arr,
        attempts=attempts,
        seed=SEED,
        structure=STRUCTURE,
        prior="truncated_lognormal_centered_nominal",
        scenario_specific_warm_start=True,
        note="Stage-1c second independent SBC set, distinct seed from 24a's n=400 set (20260903)",
    )
    print(f"saved: {OUT_PATH}, theta={theta_arr.shape}, summaries={summary_arr.shape}, "
          f"wall_s={time.time()-t0:.0f}, attempts={attempts}")


if __name__ == "__main__":
    main()
