"""Final confirmatory check for the Stage 1b fix v2 (summary-convergence-based
matched-protocol warm start). Picks 5 draws that passed fix v2's own check
(`results/44_stage1b_fix_validation.csv`, converged=True) and compares their
y0 against an independent, high-budget reference solve (t_final=500,
max_steps=1_000_000 -- large enough to genuinely converge, per the earlier
refcheck script), using the ROBUST feature subset (not the full 66-D vector,
which is dominated by known-noisy slope/std/corr features on near-flat
signals -- see the sbi_pipeline.py module comment above `_robust_sb_idx`).
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import sbi_pipeline as sp  # noqa: E402
from cstr_sbi.recycle.physics import (  # noqa: E402
    NOMINAL_INLET, NOMINAL_Y0_EXPLICIT, recycle_rhs_explicit, _diffrax_solve,
)

_reference_jit = jax.jit(
    lambda theta, ctrl: _diffrax_solve(
        recycle_rhs_explicit, NOMINAL_Y0_EXPLICIT, (theta, NOMINAL_INLET, ctrl),
        t_final=500.0, rtol=1e-7, atol=1e-9, max_steps=1_000_000,
    )
)


def reference_y0(theta_jnp, ctrl):
    sol = _reference_jit(theta_jnp, ctrl)
    y0 = np.asarray(sol.ys[-1])
    converged = bool(sol.result == diffrax.RESULTS.successful)
    return y0, converged


def main():
    val = pd.read_csv("results/44_stage1b_fix_validation.csv")
    conv = val[val["converged"] == True].reset_index(drop=True)
    sample = conv.sample(n=5, random_state=0) if len(conv) > 5 else conv
    ctrl, _ = sp._structure_config("S-C")
    robust_idx = sp._robust_sb_idx()

    warm_theta = jnp.array(sample.iloc[0][["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]].values, dtype=jnp.float32)
    reference_y0(warm_theta, ctrl)

    print(f"{'draw':>5} {'alpha':>7} {'ref_conv':>8} {'full_shift':>10} {'robust_shift':>12}")
    for _, row in sample.iterrows():
        theta_np = row[["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]].values.astype(np.float32)
        theta_jnp = jnp.array(theta_np, dtype=jnp.float32)

        y0_new, _ = sp._scenario_steady_state_checked(theta_jnp, ctrl)
        t0 = time.time()
        y0_ref, ref_converged = reference_y0(theta_jnp, ctrl)
        t_ref = time.time() - t0

        s_new = sp._simulate_summary(theta_np, ctrl, np.asarray(y0_new), "S-C", noise_pct=0.0,
                                      rng=np.random.default_rng(0), scenario_specific_warm_start=False)
        s_ref = sp._simulate_summary(theta_np, ctrl, y0_ref, "S-C", noise_pct=0.0,
                                      rng=np.random.default_rng(0), scenario_specific_warm_start=False)
        full_shift = np.linalg.norm(s_new - s_ref) / max(np.linalg.norm(s_ref), 1e-12)
        robust_shift = (np.linalg.norm(s_new[robust_idx] - s_ref[robust_idx])
                        / max(np.linalg.norm(s_ref[robust_idx]), 1e-12))
        print(f"{int(row['draw']):>5} {theta_np[0]:>7.3f} {str(ref_converged):>8} "
              f"{full_shift:>10.4f} {robust_shift:>12.6f}  (t_ref={t_ref:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
