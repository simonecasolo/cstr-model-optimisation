"""Follow-up to stage1b_fix_validation.py: that script's reference re-solve
(rtol=1e-7, atol=1e-9, t_final=300, max_steps=100_000) failed to converge
within its own budget for every one of the 16 draws that passed the new
`_matched_y0` check -- so the reported `new_shift_vs_ref` numbers are not
trustworthy (comparing against an unconverged reference). This script redoes
the reference solve for just those 16 draws with a much larger max_steps
budget (bounded, since it's now only 16 draws, not 45).
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

RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "44_stage1b_fix_validation_refcheck.csv"

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
    rhs = np.asarray(recycle_rhs_explicit(0.0, jnp.array(y0), (theta_jnp, NOMINAL_INLET, ctrl)))
    return y0, converged, float(np.linalg.norm(rhs))


def main():
    val = pd.read_csv("results/44_stage1b_fix_validation.csv")
    conv = val[val["converged"] == True].reset_index(drop=True)
    ctrl, _ = sp._structure_config("S-C")

    warm_theta = jnp.array(conv.iloc[0][["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]].values, dtype=jnp.float32)
    reference_y0(warm_theta, ctrl)

    rows = []
    t_start = time.time()
    for i in range(len(conv)):
        row = conv.iloc[i]
        theta_np = row[["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]].values.astype(np.float32)
        theta_jnp = jnp.array(theta_np, dtype=jnp.float32)

        y0_new, _ = sp._scenario_steady_state_checked(theta_jnp, ctrl)

        t0 = time.time()
        y0_ref, ref_converged, ref_res_norm = reference_y0(theta_jnp, ctrl)
        t_ref = time.time() - t0

        s_new = sp._simulate_summary(theta_np, ctrl, np.asarray(y0_new), "S-C", noise_pct=0.0,
                                      rng=np.random.default_rng(0), scenario_specific_warm_start=False)
        s_ref = sp._simulate_summary(theta_np, ctrl, y0_ref, "S-C", noise_pct=0.0,
                                      rng=np.random.default_rng(0), scenario_specific_warm_start=False)
        shift = np.nan
        if s_new is not None and s_ref is not None:
            shift = float(np.linalg.norm(s_new - s_ref) / max(np.linalg.norm(s_ref), 1e-12))

        print(f"draw {int(row['draw'])}: alpha={theta_np[0]:.3f}  ref_converged={ref_converged}  "
              f"ref_res_norm={ref_res_norm:.4g}  t_ref={t_ref:.1f}s  new_shift_vs_ref2={shift:.4f}  "
              f"old_shift(orig audit)={row['old_shift']:.4f}  [{time.time()-t_start:.0f}s elapsed]", flush=True)

        rows.append({
            "draw": int(row["draw"]), "alpha": theta_np[0], "ref_converged": ref_converged,
            "ref_res_norm": ref_res_norm, "t_ref": t_ref, "new_shift_vs_ref2": shift,
            "old_shift": row["old_shift"], "res_norm_new": row["res_norm_new"],
        })
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    n_ref_conv = int(out["ref_converged"].sum())
    print(f"\nDone: {n_ref_conv}/{len(out)} references converged.")
    good = out[out["ref_converged"] == True]["new_shift_vs_ref2"].dropna()
    if len(good):
        print(f"new_shift_vs_ref2 (only where reference itself converged, n={len(good)}): "
              f"mean={good.mean():.4f} median={good.median():.4f} max={good.max():.4f}")
        print(f"fraction > 0.003 noise floor: {(good > 0.003).mean():.3f}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
