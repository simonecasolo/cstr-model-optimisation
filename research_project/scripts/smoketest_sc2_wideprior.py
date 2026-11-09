"""Smoke test for `cstr_sbi.recycle.priors.mixed_wide_prior()` (alpha, beta_r,
eta_col, xi_reb all ~ Uniform over their prior box; z_A0_eff kept narrow
lognormal) BEFORE committing to a full bank generation.

Per `system_ii_sbi_solidification_plan.md`'s priority-0 next step: widening
`alpha` alone held accept rate at ~75% (`regenerate_sc2_mixedprior_bank.py`);
the full 5-D uniform prior collapsed to ~37% (`regenerate_sc2_uniform_bank.py`)
because multiple simultaneously-extreme parameters make the matched-protocol
warm-start solve hard to converge. Widening 4/5 dims (this script) is a new,
not-yet-quantified intermediate point -- check the accept rate on a small
number of draws here before spending ~hours on a full n=1000+SBC-sets bank.

Mirrors `regenerate_sc2_mixedprior_bank.py`'s simulate call exactly, just
capped at N_SMOKE draws and not saved as a reusable bank.
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
from cstr_sbi.recycle.physics import PARAM_NAMES
from sbi_pipeline import _structure_config, _simulate_summary

STRUCTURE = "S-C.2"
NOISE_PCT = 0.003
N_SMOKE = 40  # target *accepted* draws

torch.manual_seed(20260905)
prior_wide = mixed_wide_prior()


def main():
    ctrl, y0 = _structure_config(STRUCTURE)
    rng = np.random.default_rng(20260905)
    theta_rows = []
    t0 = time.time()
    attempts = 0
    fail_thetas = []
    while len(theta_rows) < N_SMOKE:
        attempts += 1
        th = prior_wide.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        ok = s is not None and np.isfinite(s).all()
        if ok:
            theta_rows.append(th)
        else:
            fail_thetas.append(th)
        elapsed = time.time() - t0
        accept_rate = len(theta_rows) / attempts
        print(f"attempt {attempts}: {'OK' if ok else 'FAIL'}  "
              f"theta={np.round(th, 3)}  accept_rate={accept_rate:.3f}  elapsed={elapsed:.0f}s",
              flush=True)
        if attempts >= 200:
            print("Hit 200-attempt cap before reaching N_SMOKE accepted draws -- stopping early.")
            break

    accept_rate = len(theta_rows) / attempts
    print(f"\n=== SMOKE TEST RESULT ===")
    print(f"accepted={len(theta_rows)}/{attempts} attempts, accept_rate={accept_rate:.3f}, "
          f"wall_s={time.time()-t0:.0f}, s/draw={ (time.time()-t0)/attempts:.2f}")
    if fail_thetas:
        print(f"\n{len(fail_thetas)} failed draws (theta = {PARAM_NAMES}):")
        for th in fail_thetas:
            print(f"  {np.round(th, 3)}")
    if accept_rate < 0.5:
        print("\n*** accept_rate < 0.5: DO NOT proceed to full bank generation without investigating. ***")
    else:
        print(f"\naccept_rate looks viable for a full bank generation (compare: alpha-only ~0.75, full-5D ~0.37).")


if __name__ == "__main__":
    main()
