"""S-A calibration last-effort: PCA feature reduction + architecture sweep.

Hypothesis: The S-A 72-D summary space has ~36 near-zero-variance features that,
after sbi's z-scoring, become amplified noise. Combined with a network that may
be too small (zuko_nsf 60/3), this creates a noisy optimization landscape where
no single seed can learn all 5 conditional relationships simultaneously.

This script tests:
  - StandardScaler + PCA to reduce 72-D to K effective dimensions
  - Multiple NSF architectures (wider/deeper than current 60/3)

Uses the existing 15k S-A bank (data/wu2003_sbi_train_sa_n03_v2.npz).
Reuses sbi_pipeline.py for training and SBC.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from cstr_sbi.recycle.priors import box_uniform_5d
from cstr_sbi.recycle.physics import PARAM_NAMES
from sbi_pipeline import (
    _structure_config, _simulate_summary, run_sbc, DATA, SBI_LOGS,
)

RESULTS_DIR = SBI_LOGS / "sa_pca_sweep"
RESULTS_DIR.mkdir(exist_ok=True)

BANK_PATH = DATA / "wu2003_sbi_train_sa_n03_v2.npz"

# ──────────────────────────────────────────────────────────────
# PCA preprocessing
# ──────────────────────────────────────────────────────────────

def fit_pca_pipeline(summaries: np.ndarray, n_components: int):
    """Fit StandardScaler + PCA on training summaries."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(summaries)
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    print(f"[pca] {n_components} components, "
          f"explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    return scaler, pca


def transform_pca(summaries: np.ndarray, scaler: StandardScaler, pca: PCA):
    """Apply fitted StandardScaler + PCA."""
    return pca.transform(scaler.transform(summaries)).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Training with PCA-projected features
# ──────────────────────────────────────────────────────────────

def train_posterior_pca(
    thetas: np.ndarray,
    summaries_pca: np.ndarray,
    arch: str,
    hidden_features: int,
    num_transforms: int,
    torch_seed: int = 0,
    max_num_epochs: int = 200,
    stop_after_epochs: int = 20,
):
    """Train SNPE posterior on PCA-projected summaries."""
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn

    torch.manual_seed(torch_seed)
    thetas_t = torch.tensor(thetas, dtype=torch.float32)
    summaries_t = torch.tensor(summaries_pca, dtype=torch.float32)
    print(f"[train] {thetas_t.shape[0]} samples, {summaries_t.shape[1]}-D summaries, "
          f"arch={arch} hidden={hidden_features} transforms={num_transforms} seed={torch_seed}")

    prior = box_uniform_5d()
    density_estimator = posterior_nn(
        model=arch, hidden_features=hidden_features, num_transforms=num_transforms,
    )
    inference = SNPE(prior=prior, density_estimator=density_estimator)
    inference.append_simulations(thetas_t, summaries_t)

    t0 = time.time()
    density_estimator_trained = inference.train(
        max_num_epochs=max_num_epochs,
        validation_fraction=0.1,
        stop_after_epochs=stop_after_epochs,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator_trained)
    elapsed = time.time() - t0
    print(f"[train] Completed in {elapsed:.0f}s")
    return posterior, elapsed


# ──────────────────────────────────────────────────────────────
# SBC with PCA projection
# ──────────────────────────────────────────────────────────────

def run_sbc_pca(
    posterior,
    scaler: StandardScaler,
    pca: PCA,
    structure: str = "S-A",
    noise_pct: float = 0.003,
    n_sbc: int = 400,
    n_post: int = 200,
    seed: int = 12345,
) -> dict:
    """Run SBC using PCA-projected summaries."""
    from scipy import stats as scipy_stats
    ctrl, y0 = _structure_config(structure)
    prior = box_uniform_5d()
    rng = np.random.default_rng(seed)

    ranks = {name: [] for name in PARAM_NAMES}
    t0 = time.time()
    n_valid = 0
    for i in range(n_sbc):
        th = prior.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, structure, noise_pct, rng)
        if s is None:
            continue
        n_valid += 1
        s_pca = transform_pca(s.reshape(1, -1), scaler, pca)[0]
        with torch.no_grad():
            samp = posterior.sample(
                (n_post,), x=torch.tensor(s_pca, dtype=torch.float32),
                show_progress_bars=False,
            ).detach().numpy()
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
        if (i + 1) % 100 == 0:
            print(f"[sbc]   {i+1}/{n_sbc}  ({time.time()-t0:.0f}s, {n_valid} valid)")

    results = {"n_sbc": n_sbc, "n_post": n_post, "n_valid": n_valid}
    for name in PARAM_NAMES:
        r = np.array(ranks[name])
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        results[name] = {
            "ranks": r.tolist(),
            "ks_pvalue": float(ks.pvalue),
            "mean_rank_frac": float(r.mean() / n_post),
        }
    print(f"[sbc] Completed in {time.time()-t0:.0f}s, n_valid={n_valid}/{n_sbc}")
    for name in PARAM_NAMES:
        p = results[name]["ks_pvalue"]
        flag = " ***FAIL***" if p < 0.05 else ""
        print(f"[sbc]   {name:10s} KS p={p:.4f}  mrk={results[name]['mean_rank_frac']:.3f}{flag}")
    return results


# ──────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────

ARCHITECTURES = {
    "30_2":  (30, 2),
    "40_2":  (40, 2),
    "40_3":  (40, 3),
    "60_3":  (60, 3),
    "80_3":  (80, 3),
    "60_5":  (60, 5),
    "80_5":  (80, 5),
    "128_5": (128, 5),
}

PCA_DIMS = {
    "pca15": 15,
    "pca25": 25,
    "pca40": 40,
    "raw":   None,
}


def run_single(pca_key: str, arch_key: str, seed: int, n_sbc: int = 400):
    """Run one (pca, arch, seed) combination. Returns result dict."""
    label = f"{pca_key}_{arch_key}_seed{seed}"
    result_path = RESULTS_DIR / f"{label}.json"
    if result_path.exists():
        print(f"\n=== {label} already exists, skipping ===")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"=== {label} ===")
    print(f"{'='*60}")

    # Load bank
    bank = np.load(BANK_PATH)
    thetas = bank["thetas"]
    summaries = bank["summaries"]

    # PCA preprocessing
    pca_dim = PCA_DIMS[pca_key]
    if pca_dim is not None:
        scaler, pca = fit_pca_pipeline(summaries, pca_dim)
        summaries_train = transform_pca(summaries, scaler, pca)
    else:
        scaler, pca = None, None
        summaries_train = summaries.astype(np.float32)

    # Train
    hidden, transforms = ARCHITECTURES[arch_key]
    posterior, train_time = train_posterior_pca(
        thetas, summaries_train,
        arch="zuko_nsf", hidden_features=hidden, num_transforms=transforms,
        torch_seed=seed,
    )

    # SBC
    if pca_dim is not None:
        sbc = run_sbc_pca(
            posterior, scaler, pca,
            structure="S-A", noise_pct=0.003,
            n_sbc=n_sbc, seed=12345 + seed,
        )
    else:
        sbc = run_sbc(
            posterior, structure="S-A", noise_pct=0.003,
            n_sbc=n_sbc, seed=12345 + seed,
        )

    min_ks = min(sbc[name]["ks_pvalue"] for name in PARAM_NAMES)
    passed = min_ks > 0.05

    result = {
        "label": label,
        "pca_key": pca_key,
        "pca_dim": pca_dim,
        "arch_key": arch_key,
        "hidden_features": hidden,
        "num_transforms": transforms,
        "seed": seed,
        "train_time_s": train_time,
        "n_sbc": n_sbc,
        "min_ks_pvalue": min_ks,
        "passed": passed,
        "sbc": {name: {k: v for k, v in sbc[name].items() if k != "ranks"}
                for name in PARAM_NAMES},
    }

    # Save result
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[result] {label}: min_ks={min_ks:.4f} {'PASS' if passed else 'FAIL'}")

    # Save posterior + PCA pipeline if it passed
    if passed:
        pkl_path = RESULTS_DIR / f"{label}_posterior.pkl"
        save_data = {"posterior": posterior, "meta": result}
        if pca_dim is not None:
            save_data["scaler"] = scaler
            save_data["pca"] = pca
        with open(pkl_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"[save] Posterior saved to {pkl_path}")

    return result


def print_summary(results: list[dict]):
    """Print a summary table of all completed runs."""
    print(f"\n{'='*80}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Label':<35} {'min KS p':>10} {'Pass':>6}  eta_col   xi_reb")
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x["min_ks_pvalue"]):
        ec = r["sbc"]["eta_col"]["ks_pvalue"]
        xr = r["sbc"]["xi_reb"]["ks_pvalue"]
        tag = "PASS" if r["passed"] else "fail"
        print(f"{r['label']:<35} {r['min_ks_pvalue']:>10.4f} {tag:>6}  {ec:.4f}    {xr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="S-A PCA + architecture sweep")
    parser.add_argument("--pca", nargs="+", default=["pca25", "pca15"],
                        choices=list(PCA_DIMS.keys()),
                        help="PCA settings to test")
    parser.add_argument("--arch", nargs="+", default=["80_5", "128_5"],
                        choices=list(ARCHITECTURES.keys()),
                        help="Architectures to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3],
                        help="Torch seeds")
    parser.add_argument("--n-sbc", type=int, default=400,
                        help="Number of SBC samples")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: pca25 × 128_5 × seed 0 only")
    args = parser.parse_args()

    if args.quick:
        args.pca = ["pca25"]
        args.arch = ["128_5"]
        args.seeds = [0]

    total = len(args.pca) * len(args.arch) * len(args.seeds)
    print(f"Sweep: {len(args.pca)} PCA × {len(args.arch)} arch × {len(args.seeds)} seeds = {total} runs")
    print(f"Estimated time: ~{total * 30} min ({total * 0.5:.1f} hours)")

    results = []
    for pca_key in args.pca:
        for arch_key in args.arch:
            for seed in args.seeds:
                r = run_single(pca_key, arch_key, seed, n_sbc=args.n_sbc)
                results.append(r)

    print_summary(results)

    # Report best result
    best = max(results, key=lambda r: r["min_ks_pvalue"])
    print(f"\nBest: {best['label']} (min KS p={best['min_ks_pvalue']:.4f})")
    if best["passed"]:
        print(">>> At least one combination PASSED SBC at N=400!")
        print(">>> Next step: confirm at N=800 with independent RNG draw.")
        print(f">>>   python scripts/sa_pca_arch_sweep.py --pca {best['pca_key']} "
              f"--arch {best['arch_key']} --seeds {best['seed']} --n-sbc 800")
    else:
        print(">>> No combination passed. See sweep summary for patterns.")


if __name__ == "__main__":
    main()
