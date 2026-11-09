"""Cross-check Stage 1a's FIM-based `(alpha, eta_col)` residual-confound finding
(`results/44_sc_residual_confound_sweep.csv`, `..._rawcheck.csv`) against the
deterministic grid-scan methodology `scripts/identifiability_scan.py::run_scan`
already used to RETRACT `(alpha, eta_col)` under S-B at scenario W12
(`notebooks/29b_identifiability_scan.ipynb` Section 1, `finding_sa_headline_retracted`
memory). That retraction used truth=[0.75, 1.0, 0.80, 1.0, 0.90] and found the
combined 26-feature S-B set collapses the single-channel banana to a small region near
truth. Stage 1a's FIM sweep never tested that exact point (it used nominal/
alpha_displaced=0.82/eta_col_interior=0.85/W2_like=0.85, none of which jointly displace
BOTH alpha and eta_col the way W12 does) -- this script closes that gap by running the
identical grid-scan methodology, at the identical point, under S-C (S-B + feed analyzer)
instead of S-B, to see whether adding the analyzer channel reopens a ridge that S-B's own
combined features already closed, or whether it stays closed (meaning Stage 1a's FIM
correlation is a genuinely different, point-dependent effect that appears away from W12,
not a contradiction of the W12 retraction).
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


def sim_summary_sc(theta_arr: np.ndarray, seed: int = 0) -> np.ndarray | None:
    """Deterministic-in-expectation S-C summary (feed-analyzer noise uses a fixed
    seed so repeated evaluations at the same theta are reproducible, matching the
    no-observation-noise convention `run_scan` uses for the SB channels)."""
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


def robust_feature_subset_sc(names: list[str]) -> list[int]:
    return [
        i for i, n in enumerate(names)
        if n.endswith("_mean") or n.endswith("_q_mean")
        or n in ("recycle_ratio", "col_recovery", "reb_per_boilup", "recycle_excess",
                  "Vn_final", "Rn_final", "UA_proxy", "Tr_Tj_ratio")
        or n.startswith("feed_analyzer")
    ]


def run_scan_sc(param1, param2, truth, grid1_range, grid2_range, feature_idx, label):
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
    print(f"\n=== {label}: min={np.nanmin(dist):.3f}, max={np.nanmax(dist):.3f} ===")
    return g1, g2, dist


def main():
    truth_w12 = np.array([0.75, 1.0, 0.80, 1.0, 0.90], dtype=np.float32)
    grid1 = (0.55, 1.15, 25)
    grid2 = (0.55, 0.98, 25)

    idx_full = list(range(len(SC_SUMMARY_NAMES)))
    idx_robust = robust_feature_subset_sc(SC_SUMMARY_NAMES)
    idx_fr = [i for i, n in enumerate(SC_SUMMARY_NAMES) if n.startswith("F_R_norm_")]
    print(f"robust feature count (incl. feed_analyzer): {len(idx_robust)} / {len(SC_SUMMARY_NAMES)}")

    g1_fr, g2_fr, dist_fr = run_scan_sc("alpha", "eta_col", truth_w12, grid1, grid2, idx_fr,
                                         "F_R-only (S-C, same as S-B baseline)")
    g1_rb, g2_rb, dist_rb = run_scan_sc("alpha", "eta_col", truth_w12, grid1, grid2, idx_robust,
                                         "S-C robust combined (incl. feed analyzer)")
    g1_full, g2_full, dist_full = run_scan_sc("alpha", "eta_col", truth_w12, grid1, grid2, idx_full,
                                                "S-C full 72-D")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    panels = [
        (axes[0], dist_fr, "F_R-only\n(classic banana, reference)"),
        (axes[1], dist_rb, "S-C robust combined\n(incl. feed-analyzer feature)"),
        (axes[2], dist_full, "S-C full 72-D"),
    ]
    for ax, dist, title in panels:
        im = ax.pcolormesh(g2_fr, g1_fr, dist, shading="auto", cmap="viridis", vmax=3.0)
        ax.scatter([0.80], [0.75], color="red", marker="*", s=250, zorder=10, label="Truth (W12)")
        ax.set_xlabel("eta_col"); ax.set_ylabel("alpha")
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=9, loc="lower left")
        fig.colorbar(im, ax=ax, label="normalised distance to truth")
    plt.suptitle("(alpha, eta_col) at W12 under S-C -- does the feed analyzer reopen the ridge nb29b closed?",
                 fontsize=12)
    plt.tight_layout()
    out_png = FIGURES / "44_alpha_etacol_w12_sc_gridscan.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")

    np.savez(RESULTS / "44_alpha_etacol_w12_sc_gridscan.npz",
             g1=g1_full, g2=g2_full, dist_fr=dist_fr, dist_robust=dist_rb, dist_full=dist_full)

    # Quantify "ridge extent": fraction of grid within a low-distance band (<1.0) away from
    # the truth cell itself -- a real degenerate ridge shows many low-distance cells forming
    # an extended diagonal; a resolved/collapsed case shows only the truth cell itself.
    for name, dist in [("F_R-only", dist_fr), ("robust", dist_rb), ("full", dist_full)]:
        low = dist < 1.0
        frac = low.sum() / low.size
        print(f"{name:12s}: fraction of grid with normalised distance < 1.0 = {frac:.3f} "
              f"({low.sum()}/{low.size} cells)")


if __name__ == "__main__":
    main()
