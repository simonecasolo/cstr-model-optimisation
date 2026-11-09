"""Gate A tiebreaker: deterministic grid-scan cross-check of Stage 1a's
`(alpha, eta_col)` FIM finding, at Stage 1a's OWN trigger points this time
(not W12, which `stage1a_gridscan_crosscheck.py` already checked and found
still collapses under S-C). Stage 1a's FIM sweep
(`results/44_sc_residual_confound_sweep.csv`) found `|corr| = 0.64-0.66` at
`alpha_displaced` ([0.82,1,1,1,0.90]) and `eta_col_interior`
([1,1,0.85,1,0.90]), surviving a raw-trajectory cross-check -- but never a
deterministic grid-scan (the tiebreaker method this project trusts most,
since it depends on neither a linearisation nor a trained network -- see
notebooks/40-41's precedent for `(alpha, z_A0_eff)`).

Runs the same F_R-only / robust-combined / full-72D methodology as
`stage1a_gridscan_crosscheck.py`, centered at each of the two points.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB, NOMINAL_INLET, PARAM_NAMES,
    simulate_trajectory_explicit_jit, extract_observations_explicit,
)
from cstr_sbi.recycle.simulator import nominal_warm_start
from cstr_sbi.recycle.summaries import compute_summaries, summary_names

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CTRL = NOMINAL_CTRL_SB
Y0 = nominal_warm_start("S-B")
SC_BANK = np.load(DATA / "wu2003_sbi_train_sc_lognormal_true_n5000.npz")
SC_SUMMARY_NAMES = summary_names("S-C")
SC_BANK_STD = np.maximum(SC_BANK["summaries"].std(axis=0), 1e-4)


def sim_summary_sc(theta_arr, seed=0):
    theta_jnp = jnp.array(theta_arr, dtype=jnp.float32)
    ts, ys = simulate_trajectory_explicit_jit(
        theta_jnp, NOMINAL_INLET, CTRL, jnp.array(Y0), t_final=2.0, n_save=120, rtol=1e-3, atol=1e-5,
    )
    raw = np.asarray(extract_observations_explicit(ys, theta_jnp, CTRL))
    t_np = np.asarray(ts)
    if np.isnan(raw).any() or np.isinf(raw).any():
        return None
    rng = np.random.default_rng(seed)
    return compute_summaries(raw, "S-C", t_np, theta=theta_arr, rng=rng)


def robust_feature_subset_sc(names):
    return [
        i for i, n in enumerate(names)
        if n.endswith("_mean") or n.endswith("_q_mean")
        or n in ("recycle_ratio", "col_recovery", "reb_per_boilup", "recycle_excess",
                  "Vn_final", "Rn_final", "UA_proxy", "Tr_Tj_ratio")
        or n.startswith("feed_analyzer")
    ]


def run_scan_sc(param1, param2, truth, grid1_range, grid2_range, feature_idx):
    p1_idx = PARAM_NAMES.index(param1)
    p2_idx = PARAM_NAMES.index(param2)
    s_true = sim_summary_sc(truth, seed=0)
    if s_true is None:
        raise RuntimeError("Truth simulation invalid")
    g1 = np.linspace(*grid1_range)
    g2 = np.linspace(*grid2_range)
    dist = np.full((len(g1), len(g2)), np.nan)
    for i, v1 in enumerate(g1):
        for j, v2 in enumerate(g2):
            theta = truth.copy()
            theta[p1_idx] = v1
            theta[p2_idx] = v2
            s = sim_summary_sc(theta, seed=0)
            if s is None:
                continue
            d = (s[feature_idx] - s_true[feature_idx]) / SC_BANK_STD[feature_idx]
            dist[i, j] = np.sqrt(np.sum(d ** 2))
    return g1, g2, dist


def run_point(label, truth, grid1, grid2, idx_full, idx_robust, idx_fr):
    print(f"\n########## {label}: truth={truth.tolist()} ##########")
    g1, g2, dist_fr = run_scan_sc("alpha", "eta_col", truth, grid1, grid2, idx_fr)
    print(f"F_R-only:      min={np.nanmin(dist_fr):.3f} max={np.nanmax(dist_fr):.3f}")
    _, _, dist_rb = run_scan_sc("alpha", "eta_col", truth, grid1, grid2, idx_robust)
    print(f"robust:        min={np.nanmin(dist_rb):.3f} max={np.nanmax(dist_rb):.3f}")
    _, _, dist_full = run_scan_sc("alpha", "eta_col", truth, grid1, grid2, idx_full)
    print(f"full 72-D:     min={np.nanmin(dist_full):.3f} max={np.nanmax(dist_full):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    panels = [
        (axes[0], dist_fr, "F_R-only\n(classic banana, reference)"),
        (axes[1], dist_rb, "S-C robust combined\n(incl. feed-analyzer feature)"),
        (axes[2], dist_full, "S-C full 72-D"),
    ]
    alpha_idx, eta_idx = PARAM_NAMES.index("alpha"), PARAM_NAMES.index("eta_col")
    for ax, dist, title in panels:
        im = ax.pcolormesh(g2, g1, dist, shading="auto", cmap="viridis", vmax=3.0)
        ax.scatter([truth[eta_idx]], [truth[alpha_idx]], color="red", marker="*", s=250, zorder=10, label=f"Truth ({label})")
        ax.set_xlabel("eta_col"); ax.set_ylabel("alpha")
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=9, loc="lower left")
        fig.colorbar(im, ax=ax, label="normalised distance to truth")
    plt.suptitle(f"(alpha, eta_col) at {label} under S-C -- Gate A tiebreaker", fontsize=12)
    plt.tight_layout()
    out_png = FIGURES / f"44_alpha_etacol_gate_a_{label}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")

    for name, dist in [("F_R-only", dist_fr), ("robust", dist_rb), ("full", dist_full)]:
        low = dist < 1.0
        frac = low.sum() / low.size
        print(f"  {name:12s}: fraction of grid with normalised distance < 1.0 = {frac:.3f} "
              f"({low.sum()}/{low.size} cells)")
    return dist_fr, dist_rb, dist_full


def main():
    grid1 = (0.55, 1.15, 25)   # alpha
    grid2 = (0.55, 0.98, 25)   # eta_col

    idx_full = list(range(len(SC_SUMMARY_NAMES)))
    idx_robust = robust_feature_subset_sc(SC_SUMMARY_NAMES)
    idx_fr = [i for i, n in enumerate(SC_SUMMARY_NAMES) if n.startswith("F_R_norm_")]
    print(f"robust feature count (incl. feed_analyzer): {len(idx_robust)} / {len(SC_SUMMARY_NAMES)}")

    points = {
        "eta_col_interior": np.array([1.00, 1.00, 0.85, 1.00, 0.90], dtype=np.float32),
        "W2_like":          np.array([0.85, 1.00, 1.00, 1.00, 0.90], dtype=np.float32),
        "alpha_displaced":  np.array([0.82, 1.00, 1.00, 1.00, 0.90], dtype=np.float32),
    }
    results = {}
    for label, truth in points.items():
        results[label] = run_point(label, truth, grid1, grid2, idx_full, idx_robust, idx_fr)

    np.savez(RESULTS / "44_alpha_etacol_gate_a_gridscan.npz",
             **{f"{label}_{kind}": arr for label, dists in results.items()
                for kind, arr in zip(["fr", "robust", "full"], dists)})
    print("\nSaved results/44_alpha_etacol_gate_a_gridscan.npz")


if __name__ == "__main__":
    main()
