"""Deterministic-simulator identifiability scan for parameter-pair degeneracies.

Motivated by the discovery (2026-07-03) that the article's original (alpha, eta_col)
"banana" headline under S-B did not survive scrutiny: it was an artifact of restricting
attention to F_R-derived summary features alone. This script implements the scan
methodology used to find that out, and to confirm the real degeneracy is (alpha, beta_r):

For a fixed "truth" theta and a grid over two of its components (others held fixed),
compute the deterministic (no-noise) S-B summary vector at every grid point and compare
it to the truth's summary vector, normalised by the per-feature std of the training bank
(a robust, bank-wide scale -- avoids the pathological near-zero variance seen when
normalising by noise-only std, which blows up on a few numerically-sensitive features).

Restricting the feature comparison to a single channel (e.g. F_R_norm only, or T_reb
only) reproduces the classic banana/degenerate-ridge shape when the two parameters are
truly confounded only through that one channel. Using the combined feature set tests
whether *other* already-available channels resolve it.

Usage example (reproduces the (alpha, beta_r) banana at W11):
    python scripts/identifiability_scan.py --param1 alpha --param2 beta_r \
        --truth 0.80 0.80 1.0 1.0 0.90 --grid1 0.55 1.15 21 --grid2 0.55 1.15 21

Usage example (reproduces the (alpha, eta_col) F_R-only vs combined comparison at W12):
    python scripts/identifiability_scan.py --param1 alpha --param2 eta_col \
        --truth 0.75 1.0 0.80 1.0 0.90 --grid1 0.55 1.15 25 --grid2 0.55 0.98 25 \
        --channel-subset F_R_norm
    python scripts/identifiability_scan.py --param1 alpha --param2 eta_col \
        --truth 0.75 1.0 0.80 1.0 0.90 --grid1 0.55 1.15 25 --grid2 0.55 0.98 25 \
        --channel-subset robust
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB, extract_observations_explicit, simulate_trajectory_explicit_jit,
    NOMINAL_INLET, PARAM_NAMES,
)
from cstr_sbi.recycle.simulator import nominal_warm_start
from cstr_sbi.recycle.summaries import compute_summaries, summary_names

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
BANK_PATH = DATA / "wu2003_sbi_train_sb_n03_v2.npz"

CTRL = NOMINAL_CTRL_SB
Y0 = nominal_warm_start("S-B")


def sim_summary(theta_arr: np.ndarray) -> np.ndarray | None:
    """Deterministic (no observation noise) S-B summary vector for one theta."""
    theta_jnp = jnp.array(theta_arr, dtype=jnp.float32)
    ts, ys = simulate_trajectory_explicit_jit(
        theta_jnp, NOMINAL_INLET, CTRL, Y0, t_final=2.0, n_save=120, rtol=1e-3, atol=1e-5,
    )
    raw = np.asarray(extract_observations_explicit(ys, theta_jnp, CTRL))
    t_np = np.asarray(ts)
    if np.isnan(raw).any() or np.isinf(raw).any():
        return None
    return compute_summaries(raw, "S-B", t_np)


def robust_feature_subset(names: list[str]) -> list[int]:
    """Mean/q_mean stats + non-correlation physics features -- avoids numerically
    sensitive slope/min/max/corr features that spike on near-flat signals."""
    return [
        i for i, n in enumerate(names)
        if n.endswith("_mean") or n.endswith("_q_mean")
        or n in ("recycle_ratio", "col_recovery", "reb_per_boilup", "recycle_excess",
                  "Vn_final", "Rn_final", "UA_proxy", "Tr_Tj_ratio")
    ]


def channel_subset(names: list[str], channel: str) -> list[int]:
    return [i for i, n in enumerate(names) if n.startswith(f"{channel}_")]


def run_scan(param1: str, param2: str, truth: np.ndarray,
             grid1_range: tuple[float, float, int], grid2_range: tuple[float, float, int],
             channel_subset_name: str = "robust"):
    names = summary_names("S-B")
    bank = np.load(BANK_PATH)
    bank_std = np.maximum(bank["summaries"].std(axis=0), 1e-4)

    if channel_subset_name == "robust":
        idx = robust_feature_subset(names)
    elif channel_subset_name == "full":
        idx = list(range(len(names)))
    else:
        idx = channel_subset(names, channel_subset_name)
    print(f"Using {len(idx)} features ({channel_subset_name}): "
          f"{[names[i] for i in idx[:8]]}{'...' if len(idx) > 8 else ''}")

    p1_idx = PARAM_NAMES.index(param1)
    p2_idx = PARAM_NAMES.index(param2)

    s_true = sim_summary(truth)
    if s_true is None:
        raise RuntimeError("Truth simulation is invalid (NaN/Inf) -- check truth theta.")

    g1 = np.linspace(*grid1_range)
    g2 = np.linspace(*grid2_range)
    dist = np.full((len(g1), len(g2)), np.nan)

    for i, v1 in enumerate(g1):
        for j, v2 in enumerate(g2):
            theta = truth.copy()
            theta[p1_idx] = v1
            theta[p2_idx] = v2
            s = sim_summary(theta)
            if s is None:
                continue
            d = (s[idx] - s_true[idx]) / bank_std[idx]
            dist[i, j] = np.sqrt(np.sum(d**2))

    print(f"\n{param1}\\{param2} distance grid ({channel_subset_name} features):")
    header = "".join(f"{v:5.2f}" for v in g2)
    print(f"{'':9s}{header}")
    for i in range(len(g1)):
        row = "".join(f"{dist[i,j]:5.1f}" for j in range(len(g2)))
        print(f"{g1[i]:.3f}    {row}")

    return g1, g2, dist


def sim_raw_trajectory(theta_arr: np.ndarray, channels: list[str]) -> np.ndarray | None:
    """Deterministic (no-noise) raw trajectory for the given RAW_CHANNELS, flattened.

    Unlike sim_summary, this returns the full (n_t,) time series per channel rather than
    6 aggregated per-channel statistics -- i.e. the same time-resolution information an
    EKF processing 120 sequential sub-steps would see, not the compressed representation
    SBI/the standard identifiability scan trains and compares on.
    """
    from cstr_sbi.recycle.summaries import RAW_INDEX

    theta_jnp = jnp.array(theta_arr, dtype=jnp.float32)
    ts, ys = simulate_trajectory_explicit_jit(
        theta_jnp, NOMINAL_INLET, CTRL, Y0, t_final=2.0, n_save=120, rtol=1e-3, atol=1e-5,
    )
    raw = np.asarray(extract_observations_explicit(ys, theta_jnp, CTRL))
    if np.isnan(raw).any() or np.isinf(raw).any():
        return None
    idx = [RAW_INDEX[c] for c in channels]
    return raw[:, idx].reshape(-1)


def sim_subwindow_summary(theta_arr: np.ndarray, channels: list[str], n_sub: int = 4) -> np.ndarray | None:
    """Deterministic per-sub-window (mean, std) per channel -- a richer-but-still-hand-
    -crafted feature set that preserves within-window transient shape at n_sub points
    instead of collapsing the whole 2h window to a single mean/std/slope/min/max/q_mean,
    like compute_summaries does. A practical middle ground between compute_summaries (66-D,
    whole-window) and the full raw trajectory (very high-D, not a realistic SBI feature set).
    """
    from cstr_sbi.recycle.summaries import RAW_INDEX

    theta_jnp = jnp.array(theta_arr, dtype=jnp.float32)
    ts, ys = simulate_trajectory_explicit_jit(
        theta_jnp, NOMINAL_INLET, CTRL, Y0, t_final=2.0, n_save=120, rtol=1e-3, atol=1e-5,
    )
    raw = np.asarray(extract_observations_explicit(ys, theta_jnp, CTRL))
    if np.isnan(raw).any() or np.isinf(raw).any():
        return None
    idx = [RAW_INDEX[c] for c in channels]
    obs = raw[:, idx]  # (120, n_ch)
    n_t = obs.shape[0]
    edges = np.linspace(0, n_t, n_sub + 1).astype(int)
    feats = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        chunk = obs[lo:hi]
        feats.append(chunk.mean(axis=0))
        feats.append(chunk.std(axis=0))
    return np.concatenate(feats)


def compute_norm_std(feature_fn, n_samples: int = 300, seed: int = 0) -> np.ndarray:
    """Per-element std of feature_fn(theta) over a fresh uniform-prior sample.

    Plays the same normalising role as the training-bank std does for compute_summaries
    (scaling each feature element by its typical spread across the parameter space before
    computing an L2 distance), but computed on the fly since no bank of raw trajectories or
    sub-window features is cached anywhere in this project.
    """
    from cstr_sbi.recycle.priors import PRIOR_LOW_5D, PRIOR_HIGH_5D

    rng = np.random.default_rng(seed)
    samples = []
    thetas = rng.uniform(PRIOR_LOW_5D, PRIOR_HIGH_5D, size=(n_samples, 5)).astype(np.float32)
    for th in thetas:
        f = feature_fn(th)
        if f is not None and not (np.isnan(f).any() or np.isinf(f).any()):
            samples.append(f)
    arr = np.stack(samples)
    print(f"  norm_std computed from {arr.shape[0]}/{n_samples} valid prior draws")
    return np.maximum(arr.std(axis=0), 1e-6)


def run_scan_custom(param1: str, param2: str, truth: np.ndarray,
                     grid1_range: tuple[float, float, int], grid2_range: tuple[float, float, int],
                     feature_fn, norm_std: np.ndarray, label: str = "custom"):
    """Same grid-scan logic as run_scan, but for an arbitrary feature_fn(theta) -> vector
    and a precomputed norm_std, instead of compute_summaries + the training-bank std.

    Kept fully separate from run_scan (which nb26/nb29b already depend on and which must
    keep behaving identically) so this is purely additive.
    """
    p1_idx = PARAM_NAMES.index(param1)
    p2_idx = PARAM_NAMES.index(param2)

    s_true = feature_fn(truth)
    if s_true is None:
        raise RuntimeError("Truth simulation is invalid (NaN/Inf) -- check truth theta.")

    g1 = np.linspace(*grid1_range)
    g2 = np.linspace(*grid2_range)
    dist = np.full((len(g1), len(g2)), np.nan)

    for i, v1 in enumerate(g1):
        for j, v2 in enumerate(g2):
            theta = truth.copy()
            theta[p1_idx] = v1
            theta[p2_idx] = v2
            s = feature_fn(theta)
            if s is None:
                continue
            d = (s - s_true) / norm_std
            dist[i, j] = np.sqrt(np.mean(d ** 2))

    print(f"\n{param1}\\{param2} distance grid ({label}, {len(s_true)}-D features):")
    header = "".join(f"{v:5.2f}" for v in g2)
    print(f"{'':9s}{header}")
    for i in range(len(g1)):
        row = "".join(f"{dist[i,j]:5.1f}" for j in range(len(g2)))
        print(f"{g1[i]:.3f}    {row}")

    return g1, g2, dist


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param1", required=True, choices=PARAM_NAMES)
    p.add_argument("--param2", required=True, choices=PARAM_NAMES)
    p.add_argument("--truth", nargs=5, type=float, required=True,
                    metavar=("alpha", "beta_r", "eta_col", "xi_reb", "z_A0"))
    p.add_argument("--grid1", nargs=3, type=float, required=True, metavar=("LOW", "HIGH", "N"))
    p.add_argument("--grid2", nargs=3, type=float, required=True, metavar=("LOW", "HIGH", "N"))
    p.add_argument("--channel-subset", default="robust",
                    help="'robust', 'full', or a channel prefix like 'F_R_norm' or 'T_reb'")
    args = p.parse_args()

    truth = np.array(args.truth, dtype=np.float32)
    grid1 = (args.grid1[0], args.grid1[1], int(args.grid1[2]))
    grid2 = (args.grid2[0], args.grid2[1], int(args.grid2[2]))
    run_scan(args.param1, args.param2, truth, grid1, grid2, args.channel_subset)


if __name__ == "__main__":
    main()
