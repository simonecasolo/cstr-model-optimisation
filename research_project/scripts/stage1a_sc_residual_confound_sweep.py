"""Stage 1a residual-confound FIM sweep under the S-C instrumentation structure.

Checks the six pairs NOT already covered by the confirmed-collapsed
`(alpha, z_A0_eff)` / `(eta_col, z_A0_eff)` results and the `xi_reb`-vs-`z_A0_eff`
clearance: `(alpha,beta_r)`, `(alpha,eta_col)`, `(alpha,xi_reb)`,
`(xi_reb,eta_col)`, `(xi_reb,beta_r)`, `(beta_r,eta_col)`.

Methodology copied verbatim from `notebooks/23a_wu2003_summary_statistics_sc.ipynb`
cell 11 (72-D S-C summary FIM) and the raw-trajectory cross-check pattern from
`notebooks/42_wu2003_manuscript_fim_reaudit.ipynb` cell 2 (`feat_raw_fim`), adapted
to append the S-C feed-analyzer channel to the raw representation so the
representation-collapse check is apples-to-apples with the 72-D summary (both
"see" the feed analyzer; only the aggregation into 6 per-channel stats differs).

Outputs `results/44_sc_residual_confound_sweep.csv` and prints the full table plus
the raw-trajectory cross-check for any (point, pair) with |corr| > 0.3.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB, NOMINAL_INLET,
    simulate_trajectory_explicit_jit, extract_observations_explicit,
)
from cstr_sbi.recycle.simulator import nominal_warm_start
from cstr_sbi.recycle.summaries import (
    compute_summaries, feed_analyzer_reading, SB_INDICES, FEED_ANALYZER_NOISE_PCT,
)
from fim_utils import compute_fim, cov_offdiag_corr  # NEVER offdiag_ratio -- sign-buggy, see docstring

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

Y0_FIM = np.asarray(nominal_warm_start("S-B"))
EPS_FIM = np.array([0.02, 0.02, 0.02, 0.02, 0.01])
PARAM_IDX = {"alpha": 0, "beta_r": 1, "eta_col": 2, "xi_reb": 3, "z_A0_eff": 4}

POINTS = {
    "nominal":          np.array([1.00, 1.00, 1.00, 1.00, 0.90], dtype=np.float32),
    "alpha_displaced":  np.array([0.82, 1.00, 1.00, 1.00, 0.90], dtype=np.float32),
    "eta_col_interior": np.array([1.00, 1.00, 0.85, 1.00, 0.90], dtype=np.float32),
    "W2_like":          np.array([0.85, 1.00, 1.00, 1.00, 0.90], dtype=np.float32),
}

PAIRS = [
    ("alpha", "beta_r"),
    ("alpha", "eta_col"),
    ("alpha", "xi_reb"),
    ("xi_reb", "eta_col"),
    ("xi_reb", "beta_r"),
    ("beta_r", "eta_col"),
]


def sim_raw_full(theta_np, seed=None, noise_pct=0.003):
    th = jnp.array(theta_np, dtype=jnp.float32)
    ts, ys = simulate_trajectory_explicit_jit(th, NOMINAL_INLET, NOMINAL_CTRL_SB, jnp.array(Y0_FIM),
                                               t_final=2.0, n_save=120, rtol=1e-3, atol=1e-5)
    raw = np.asarray(extract_observations_explicit(ys, th, NOMINAL_CTRL_SB))
    t_hh = np.asarray(ts)
    if seed is not None:
        rng = np.random.default_rng(seed)
        scale = np.maximum(np.max(np.abs(raw), axis=0), 1e-6)
        raw = raw + rng.normal(0, noise_pct * scale, raw.shape)
    return raw, t_hh


def feat_sc(theta, seed):
    """72-D S-C summary representation (9 SB channels x 6 stats + 12 physics + 6 feed-analyzer stats)."""
    raw, t_hh = sim_raw_full(theta, seed)
    rng = np.random.default_rng(seed if seed is not None else 0)
    return compute_summaries(raw, "S-C", t_hh, theta=theta, rng=rng)


def feat_sc_raw(theta, seed):
    """Unaggregated raw-trajectory S-C representation: 9 SB raw channels x 120 timesteps,
    plus the feed-analyzer's own raw (constant + noise) trace x 120 timesteps -- i.e. the
    same information the 72-D summary sees, but without the per-channel mean/std/slope/
    min/max/final-quarter-mean aggregation. Flattened to 1-D.
    """
    raw, t_hh = sim_raw_full(theta, seed)
    sb_raw = raw[:, SB_INDICES]
    rng = np.random.default_rng(seed if seed is not None else 0)
    feed_raw = feed_analyzer_reading(theta, t_hh, rng, FEED_ANALYZER_NOISE_PCT)
    full = np.concatenate([sb_raw, feed_raw[:, None]], axis=1)
    return full.reshape(-1)


def main():
    rows = []
    summary_fims = {}
    for i, (point_name, theta_pt) in enumerate(POINTS.items()):
        print(f"\n=== computing S-C summary FIM at {point_name}: {theta_pt.tolist()} ===")
        FIM_sc = compute_fim(feat_sc, theta_pt, EPS_FIM, seed_offset=1000 * i)
        summary_fims[point_name] = (FIM_sc, theta_pt, i)
        for a, b in PAIRS:
            corr = cov_offdiag_corr(FIM_sc, PARAM_IDX[a], PARAM_IDX[b])
            rows.append({"point": point_name, "pair": f"{a}/{b}", "corr": corr})

    df = pd.DataFrame(rows)
    print("\n=== Full 4-points x 6-pairs correlation table (72-D S-C summary) ===")
    print(df.pivot(index="pair", columns="point", values="corr").round(4).to_string())
    print()
    print(df.round(4).to_string(index=False))

    # Raw-trajectory cross-check for any (point, pair) with |corr| > 0.3
    print("\n=== Raw-trajectory cross-check (|corr| > 0.3 triggers) ===")
    crosscheck_rows = []
    triggered = df[df["corr"].abs() > 0.3]
    if triggered.empty:
        print("(none triggered: no |corr| > 0.3 among the 6 pairs x 4 points)")
    else:
        # Group by point so we only recompute the raw FIM once per point that has any trigger
        for point_name in triggered["point"].unique():
            FIM_sc, theta_pt, i = summary_fims[point_name]
            print(f"\n-- computing raw-trajectory S-C FIM at {point_name} --")
            FIM_raw = compute_fim(feat_sc_raw, theta_pt, EPS_FIM, seed_offset=1000 * i)
            sub = triggered[triggered["point"] == point_name]
            for _, r in sub.iterrows():
                a, b = r["pair"].split("/")
                corr_summary = r["corr"]
                corr_raw = cov_offdiag_corr(FIM_raw, PARAM_IDX[a], PARAM_IDX[b])
                survives = abs(corr_raw) > 0.5
                print(f"    {point_name:18s} {r['pair']:20s} summary corr={corr_summary:+.4f}  "
                      f"raw corr={corr_raw:+.4f}  {'SURVIVES' if survives else 'collapses'}")
                crosscheck_rows.append({
                    "point": point_name, "pair": r["pair"],
                    "corr_summary_72d": corr_summary, "corr_raw_traj": corr_raw,
                    "survives_gate_a": survives,
                })

    df.to_csv(RESULTS / "44_sc_residual_confound_sweep.csv", index=False)
    print(f"\nSaved 72-D summary sweep table to {RESULTS / '44_sc_residual_confound_sweep.csv'}")

    if crosscheck_rows:
        cross_df = pd.DataFrame(crosscheck_rows)
        cross_path = RESULTS / "44_sc_residual_confound_sweep_rawcheck.csv"
        cross_df.to_csv(cross_path, index=False)
        print(f"Saved raw-trajectory cross-check table to {cross_path}")
        print("\n=== Cross-check summary ===")
        print(cross_df.round(4).to_string(index=False))

    # Gate A verdict: |corr| > 0.5 in the 72-D summary AND survives (|raw corr| > 0.5)
    gate_a_hits = []
    if crosscheck_rows:
        for r in crosscheck_rows:
            if abs(r["corr_summary_72d"]) > 0.5 and r["survives_gate_a"]:
                gate_a_hits.append(r)

    print("\n=== Gate A verdict ===")
    if gate_a_hits:
        for r in gate_a_hits:
            print(f"  GATE A TRIGGERED: {r['pair']} at {r['point']} "
                  f"(summary corr={r['corr_summary_72d']:+.4f}, raw corr={r['corr_raw_traj']:+.4f})")
    else:
        print("  Gate A NOT triggered: no pair/point combination exceeds |corr|>0.5 in the "
              "72-D summary AND survives (|raw corr|>0.5) under the raw-trajectory representation.")

    return df, crosscheck_rows


if __name__ == "__main__":
    main()
