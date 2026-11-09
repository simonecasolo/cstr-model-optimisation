"""Smoke test for the FULL 5-D `box_uniform_5d()` prior under the CURRENT
(Stage-1b warm-start fix) protocol -- the one clean test never actually
completed. History (per 2026-09-05/06 research): the original nb31 S-B
posterior was trained on this exact prior and gave the best classification
result ever seen in this project (macro-F1=0.694), but that predates a
physics bug fix (Item 6) that invalidated it. A "matched protocol" retrain
(2026-08-27) failed SBC catastrophically (0/8 seeds) -- but that retrain
PREDATES the Stage-1b warm-start fix (2026-09-03/04), so the failure is
confounded with a since-fixed bug, not a clean read on the prior itself. A
post-fix re-test this session (2026-09-04) was aborted purely on accept-rate
grounds (~37%) before ever reaching SBC/classification. This script measures
that accept rate/timing directly and freshly, under the CURRENT protocol, on
a small number of draws -- before committing to any full bank generation.
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

from cstr_sbi.recycle.priors import box_uniform_5d
from cstr_sbi.recycle.physics import PARAM_NAMES
from sbi_pipeline import _structure_config, _simulate_summary

STRUCTURE = "S-C.2"
NOISE_PCT = 0.003
N_SMOKE = 40  # target *accepted* draws

torch.manual_seed(20260906)
prior_uniform = box_uniform_5d()


def main():
    ctrl, y0 = _structure_config(STRUCTURE)
    rng = np.random.default_rng(20260906)
    theta_rows = []
    t0 = time.time()
    attempts = 0
    fail_thetas = []
    while len(theta_rows) < N_SMOKE:
        attempts += 1
        th = prior_uniform.sample((1,)).numpy()[0]
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
    print(f"\n=== SMOKE TEST RESULT (full 5-D box_uniform_5d, Stage-1b-fixed protocol) ===")
    print(f"accepted={len(theta_rows)}/{attempts} attempts, accept_rate={accept_rate:.3f}, "
          f"wall_s={time.time()-t0:.0f}, s/draw={(time.time()-t0)/attempts:.2f}")
    if fail_thetas:
        print(f"\n{len(fail_thetas)} failed draws (theta = {PARAM_NAMES}):")
        for th in fail_thetas:
            print(f"  {np.round(th, 3)}")
    print(f"\n(compare: lognormal ~0.91-0.95, alpha-only-mixed ~0.75, "
          f"4/5-wide-mixed ~0.46-0.60, this session's earlier aborted full-uniform attempt ~0.37)")


if __name__ == "__main__":
    main()
