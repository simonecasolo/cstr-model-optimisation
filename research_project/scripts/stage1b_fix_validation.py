"""Validate the Stage 1b warm-start fix (`_matched_y0` in `sbi_pipeline.py`,
tightened tolerance + explicit convergence check, 2026-09-03) against the same
45 theta draws the original audit found 100% contaminated.

For each draw:
  1. Run the NEW `_matched_y0` -- record (converged, wall_time, residual_norm).
  2. If converged, compute the downstream S-C summary and compare against a
     tightly-resolved reference solve (rtol=1e-7, atol=1e-9, t_final=300,
     max_steps=100_000 -- bounded, so a pathological reference computation
     can't itself hang; if the reference doesn't converge either, skip the
     comparison for that draw rather than block).

Outputs: results/44_stage1b_fix_validation.csv
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
OUT_CSV = RESULTS / "44_stage1b_fix_validation.csv"

_reference_jit = jax.jit(
    lambda theta, ctrl: _diffrax_solve(
        recycle_rhs_explicit, NOMINAL_Y0_EXPLICIT, (theta, NOMINAL_INLET, ctrl),
        t_final=300.0, rtol=1e-7, atol=1e-9, max_steps=100_000,
    )
)


def reference_y0(theta_jnp, ctrl):
    sol = _reference_jit(theta_jnp, ctrl)
    y0 = np.asarray(sol.ys[-1])
    converged = bool(sol.result == diffrax.RESULTS.successful)
    return y0, converged


def main():
    df = pd.read_csv("results/44_sc_warmstart_convergence_audit.csv")
    ctrl, _ = sp._structure_config("S-C")

    # warm up both JIT paths once (compile cost excluded from per-draw timing)
    warm_theta = jnp.array(df.iloc[0][["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]].values, dtype=jnp.float32)
    sp._scenario_steady_state_checked(warm_theta, ctrl)
    reference_y0(warm_theta, ctrl)

    rows = []
    t_start = time.time()
    for i in range(len(df)):
        row = df.iloc[i]
        theta_np = row[["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]].values.astype(np.float32)
        theta_jnp = jnp.array(theta_np, dtype=jnp.float32)

        t0 = time.time()
        y0_new, converged = sp._scenario_steady_state_checked(theta_jnp, ctrl)
        t_new = time.time() - t0
        rhs_new = np.asarray(recycle_rhs_explicit(0.0, y0_new, (theta_jnp, NOMINAL_INLET, ctrl)))
        res_norm_new = float(np.linalg.norm(rhs_new))

        shift = np.nan
        ref_converged = None
        if converged:
            t0 = time.time()
            y0_ref, ref_converged = reference_y0(theta_jnp, ctrl)
            t_ref = time.time() - t0
            s_new = sp._simulate_summary(theta_np, ctrl, np.asarray(y0_new), "S-C", noise_pct=0.0,
                                          rng=np.random.default_rng(0), scenario_specific_warm_start=False)
            s_ref = sp._simulate_summary(theta_np, ctrl, y0_ref, "S-C", noise_pct=0.0,
                                          rng=np.random.default_rng(0), scenario_specific_warm_start=False)
            if s_new is not None and s_ref is not None:
                shift = float(np.linalg.norm(s_new - s_ref) / max(np.linalg.norm(s_ref), 1e-12))
        else:
            t_ref = np.nan

        old_shift = row["summary_shift_rel_l2"]
        print(f"draw {i}: alpha={theta_np[0]:.3f}  converged={converged}  "
              f"res_norm_new={res_norm_new:.4g}  t_new={t_new:.1f}s  "
              f"old_shift={old_shift:.4f}  new_shift(vs ref)={shift:.4f}  "
              f"ref_converged={ref_converged}  t_ref={t_ref:.1f}s  "
              f"[{time.time()-t_start:.0f}s elapsed total]", flush=True)

        rows.append({
            "draw": i, "alpha": theta_np[0], "beta_r": theta_np[1], "eta_col": theta_np[2],
            "xi_reb": theta_np[3], "z_A0_eff": theta_np[4],
            "converged": converged, "res_norm_new": res_norm_new, "t_new": t_new,
            "old_shift": old_shift, "new_shift_vs_ref": shift,
            "ref_converged": ref_converged, "t_ref": t_ref,
        })
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    n_converged = int(out["converged"].sum())
    print(f"\nDone: {n_converged}/{len(out)} draws converged under the new check.")
    valid_shift = out["new_shift_vs_ref"].dropna()
    if len(valid_shift):
        print(f"new_shift_vs_ref (n={len(valid_shift)}): mean={valid_shift.mean():.4f} "
              f"median={valid_shift.median():.4f} max={valid_shift.max():.4f}")
        print(f"fraction of converged draws with new_shift > 0.003 (noise floor): "
              f"{(valid_shift > 0.003).mean():.3f}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
