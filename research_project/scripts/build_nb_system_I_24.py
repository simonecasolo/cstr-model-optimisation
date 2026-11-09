"""nb24: investigate WHY the hand-crafted-summary (HC) and raw-trajectory-CNN
(Theory F) SBI posteriors disagree so persistently on Corr(alpha,beta),
even after the warm-start fix and at the nb22-determined optimal budget
n_sims=50,000 (see notebooks/system_I/TODO.md and HANDOFF.md's 2026-11-08
THREE-METHOD-CROSS-CHECK entries, and
finding_system1_no_alpha_beta_banana.md).

This is a standalone script, not an nbclient-executed notebook, per this
project's established house rule (nb20's multi-hour nbclient stalls on long
runs) -- a plain script with disk-flushed progress sentinels, run
self-caffeinated in the background.

Four hypotheses tested (see module docstring sections below for each):
  H1. CNN-side calibration check: SBC (KS test on rank statistics, same
      N_SBC=500 / num_posterior_samples=1000 convention nb22 used for HC) +
      anti-shrinkage slope diagnostic, on the ALREADY-CACHED CNN posteriors
      at n=50,000 (results/22_theoryF_cnn_posterior_seed{0,1,2}_n50000.pkl)
      -- no retraining needed for this part.
  H2. Architecture/capacity check: held-out predictive log-density
      (posterior.log_prob on a held-out set NOT used for training, same
      theta draws probed through both the summary-stat pipeline and the
      raw-trajectory pipeline) comparing HC posteriors (results/22_budget50000_seed{0-4}.pkl)
      vs CNN posteriors (results/22_theoryF_cnn_posterior_seed{0,1,2}_n50000.pkl).
      A representation that is "worse at compressing the data" should show
      lower held-out log-density on average.
  H3. Multi-seed robustness: train 2 MORE CNN seeds (seed3, seed4) at
      n=50,000 (same architecture/training convention as
      nb22_cnn_theoryF_recheck.py, cost ~21-31 min/seed per that script's own
      measured wall times) to bring CNN up to 5 seeds, matching HC's 5-seed
      nb22 budget study, and check whether CNN's own Corr(alpha,beta)@Sc4
      estimate is itself seed-stable or seed-dependent.
  H4. Qualitative representation comparison: for a few fixed theta points,
      multiple noise realizations each, compare the 29-D hand-crafted
      summary vector to the CNN's learned embedding-net output on the SAME
      raw trajectories -- a cheap, illustrative check, not a formal
      statistical test. Skipped if it would cost disproportionate effort
      for the payoff (decided inline after seeing H1-H3's results).

Outputs:
  results/24_cnn_sbc_metrics.csv          H1
  results/24_cnn_grid_diagnostics.csv     H1 (slope/coverage/std)
  results/24_heldout_logprob_comparison.csv   H2
  results/24_cnn_5seed_corr_by_scenario.csv   H3
  figures/24_cnn_sbc_rank_histograms.png  H1
  figures/24_heldout_logprob_comparison.png   H2
  figures/24_cnn_5seed_corr_summary.png   H3
  figures/24_feature_comparison.png       H4 (if run)
  results/24_verdict.txt                  final honest summary

Progress sentinels: results/24_progress_*.txt (disk-flushed, not print()).
"""

from __future__ import annotations

import csv
import datetime as _dt
import pickle
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cstr_sbi.inference import (
    train_sbi_posterior,
    simulation_wrapper_sbi,
    simulation_wrapper_sbi_raw,
)
from cstr_sbi.priors import box_uniform_2d
from cstr_sbi.physics import NOMINAL_INLET_CL, NOMINAL_CTRL
from cstr_sbi.scenarios import SCENARIO_CONFIGS
from sbi.neural_nets.embedding_nets import CNNEmbedding

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

N_SIMULATIONS = 50_000
N_SBC = 500
NUM_POSTERIOR_SAMPLES = 1000
N_HELDOUT = 500
HC_SEEDS = [0, 1, 2, 3, 4]        # existing cached HC posteriors (nb22)
CNN_EXISTING_SEEDS = [0, 1, 2]    # existing cached CNN posteriors (nb22 recheck)
CNN_NEW_SEEDS = [3, 4]            # new seeds trained in this script (H3)
GRID_SCENARIOS = [
    "Sc1_closed_healthy", "Sc2_closed_fouling", "Sc3_closed_decay",
    "Sc4_closed_combined", "Sc5_closed_saturated", "Sc7_closed_drift",
]


def mark(stage: str) -> None:
    (RESULTS / f"24_progress_{stage}.txt").write_text(
        f"{time.time()}\n{_dt.datetime.now().isoformat()}\n"
    )


def build_embedding_net():
    """IDENTICAL architecture to 04b_embedding_net_study.ipynb / nb20's / nb22's
    Theory F CNN recheck -- deliberately unchanged, so H3's new seeds are
    directly comparable to the 3 already-cached ones."""
    return CNNEmbedding(
        input_shape=(120,), in_channels=4, out_channels_per_layer=[16, 32],
        num_conv_layers=2, num_linear_layers=2, num_linear_units=64,
        output_dim=20, kernel_size=5, pool_kernel_size=2,
    )


def load_hc_posterior(seed: int):
    path = RESULTS / f"22_budget{N_SIMULATIONS}_seed{seed}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)["posterior"]


def load_cnn_posterior(seed: int):
    path = RESULTS / f"22_theoryF_cnn_posterior_seed{seed}_n{N_SIMULATIONS}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)["posterior"]


def obs_x_summary(sc_name: str, seed: int = 999):
    sc = SCENARIO_CONFIGS[sc_name]
    theta = torch.tensor([[sc.alpha, sc.beta]], dtype=torch.float32)
    return simulation_wrapper_sbi(
        theta, inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
        scenario_specific_warm_start=True, seed=seed,
    )


def obs_x_raw(sc_name: str, seed: int = 999):
    sc = SCENARIO_CONFIGS[sc_name]
    theta = torch.tensor([[sc.alpha, sc.beta]], dtype=torch.float32)
    return simulation_wrapper_sbi_raw(
        theta, inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
        scenario_specific_warm_start=True, seed=seed,
    )


def corr_ab(posterior, x, n_draw: int = 5000, seed: int = 0):
    torch.manual_seed(seed)
    samp = posterior.sample((n_draw,), x=x, show_progress_bars=False).numpy()
    return float(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1]), samp


# ---------------------------------------------------------------------------
# H1: CNN-side SBC + anti-shrinkage slope, at n=50,000, reusing cached
# posteriors (no retraining).
# ---------------------------------------------------------------------------

def run_sbc_for_posterior(posterior, sim_fn, seed_offset: int) -> dict:
    """Same convention as nb22's run_sbc_for_posterior, parameterized over
    which simulator (summary vs raw) generates the SBC calibration set."""
    from sbi.diagnostics import run_sbc, check_sbc

    prior = box_uniform_2d()
    torch.manual_seed(seed_offset)
    theta_sbc = prior.sample((N_SBC,))

    x_list = []
    for i in range(N_SBC):
        x_i = sim_fn(
            theta_sbc[i:i + 1], inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
            scenario_specific_warm_start=True, seed=seed_offset + i,
        )
        x_list.append(x_i)
    x_sbc = torch.cat(x_list)

    ranks, dap_samples = run_sbc(
        theta_sbc, x_sbc, posterior,
        num_posterior_samples=NUM_POSTERIOR_SAMPLES,
        reduce_fns="marginals",
    )
    checks = check_sbc(ranks, theta_sbc, dap_samples,
                        num_posterior_samples=NUM_POSTERIOR_SAMPLES)
    ranks_np = ranks.numpy()

    def shape_diag(col: int) -> dict:
        r = ranks_np[:, col]
        n_bins_edge = NUM_POSTERIOR_SAMPLES * 0.20
        frac_edge = float(np.mean(
            (r < n_bins_edge) | (r > NUM_POSTERIOR_SAMPLES - n_bins_edge)
        ))
        frac_mid = float(np.mean(
            (r >= (NUM_POSTERIOR_SAMPLES * 0.4)) & (r <= (NUM_POSTERIOR_SAMPLES * 0.6))
        ))
        return {"mean_rank": float(r.mean()), "frac_edge_20pct": frac_edge,
                "frac_mid_20pct": frac_mid, "ranks": r}

    diag_a = shape_diag(0)
    diag_b = shape_diag(1)
    return {
        "ks_p_alpha": float(checks["ks_pvals"][0]),
        "ks_p_beta": float(checks["ks_pvals"][1]),
        "mean_rank_a": diag_a["mean_rank"], "mean_rank_b": diag_b["mean_rank"],
        "frac_edge_20pct_a": diag_a["frac_edge_20pct"],
        "frac_edge_20pct_b": diag_b["frac_edge_20pct"],
        "frac_mid_20pct_a": diag_a["frac_mid_20pct"],
        "frac_mid_20pct_b": diag_b["frac_mid_20pct"],
        "_ranks_a": diag_a["ranks"], "_ranks_b": diag_b["ranks"],
    }


def grid_diagnostics(posterior, obs_fn, n_draw: int = 2000) -> dict:
    """Posterior-mean-vs-truth slope (anti-shrinkage), 90% CI coverage, and
    posterior std, across the Sc1,2,3,4,5,7 closed-loop truth grid. Same
    convention as nb22's grid_diagnostics, parameterized over obs_fn (summary
    vs raw observation generator)."""
    truths_a, truths_b = [], []
    means_a, means_b = [], []
    cover_a, cover_b = [], []
    stds_a, stds_b = [], []
    for sc_name in GRID_SCENARIOS:
        sc = SCENARIO_CONFIGS[sc_name]
        x = obs_fn(sc_name, seed=777 + sc.id)
        torch.manual_seed(42 + sc.id)
        samp = posterior.sample((n_draw,), x=x, show_progress_bars=False).numpy()
        a, b = samp[:, 0], samp[:, 1]
        truths_a.append(sc.alpha); truths_b.append(sc.beta)
        means_a.append(float(a.mean())); means_b.append(float(b.mean()))
        stds_a.append(float(a.std())); stds_b.append(float(b.std()))
        lo_a, hi_a = np.percentile(a, [5, 95])
        lo_b, hi_b = np.percentile(b, [5, 95])
        cover_a.append(bool(lo_a <= sc.alpha <= hi_a))
        cover_b.append(bool(lo_b <= sc.beta <= hi_b))

    def slope(truths, means):
        truths = np.asarray(truths); means = np.asarray(means)
        if truths.std() == 0:
            return float("nan")
        A = np.vstack([truths, np.ones_like(truths)]).T
        m, _ = np.linalg.lstsq(A, means, rcond=None)[0]
        return float(m)

    return {
        "slope_alpha": slope(truths_a, means_a),
        "slope_beta": slope(truths_b, means_b),
        "coverage90_alpha": float(np.mean(cover_a)),
        "coverage90_beta": float(np.mean(cover_b)),
        "std_alpha_mean": float(np.mean(stds_a)),
        "std_beta_mean": float(np.mean(stds_b)),
    }


def run_h1():
    mark("h1_start")
    print("H1: CNN SBC + anti-shrinkage slope at n=50,000 (cached posteriors)", flush=True)
    sbc_csv = RESULTS / "24_cnn_sbc_metrics.csv"
    grid_csv = RESULTS / "24_cnn_grid_diagnostics.csv"
    sbc_rows, grid_rows, rank_cache = [], [], {}

    for seed in CNN_EXISTING_SEEDS:
        tag = f"cnn_seed{seed}"
        mark(f"h1_sbc_{tag}")
        print(f"  {tag}: SBC...", flush=True)
        post = load_cnn_posterior(seed)
        t0 = time.time()
        sbc = run_sbc_for_posterior(post, simulation_wrapper_sbi_raw,
                                     seed_offset=200_000 + seed * 1000)
        rank_cache[seed] = (sbc.pop("_ranks_a"), sbc.pop("_ranks_b"))
        sbc_rows.append({"seed": seed, **sbc})
        print(f"  {tag}: SBC done in {time.time()-t0:.0f}s "
              f"ks_p_a={sbc['ks_p_alpha']:.3f} ks_p_b={sbc['ks_p_beta']:.3f}", flush=True)

        mark(f"h1_grid_{tag}")
        grid = grid_diagnostics(post, obs_x_raw)
        grid_rows.append({"seed": seed, **grid})
        print(f"  {tag}: slope_a={grid['slope_alpha']:.3f} "
              f"slope_b={grid['slope_beta']:.3f}", flush=True)

    with open(sbc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sbc_rows[0].keys()))
        w.writeheader(); w.writerows(sbc_rows)
    with open(grid_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader(); w.writerows(grid_rows)

    # rank histograms, same visual convention project uses elsewhere
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(CNN_EXISTING_SEEDS), 2, figsize=(9, 3 * len(CNN_EXISTING_SEEDS)))
    axes = np.atleast_2d(axes)
    for row, seed in enumerate(CNN_EXISTING_SEEDS):
        ra, rb = rank_cache[seed]
        for col, (r, name) in enumerate([(ra, "alpha"), (rb, "beta")]):
            ax = axes[row, col]
            ax.hist(r, bins=20, range=(0, NUM_POSTERIOR_SAMPLES), color="steelblue", alpha=0.8)
            ax.axhline(N_SBC / 20, color="red", ls="--", lw=1)
            ax.set_title(f"CNN seed{seed}, {name} (SBC rank hist)", fontsize=9)
    fig.suptitle("nb24 H1: CNN-embedding posterior SBC rank histograms (n=50,000)\n"
                 "flat = calibrated, U/hump-shaped = miscalibrated", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "24_cnn_sbc_rank_histograms.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    mark("h1_done")
    print("H1 done.", flush=True)
    return sbc_rows, grid_rows


# ---------------------------------------------------------------------------
# H2: held-out predictive log-density, HC vs CNN, same held-out theta set.
# ---------------------------------------------------------------------------

def run_h2():
    mark("h2_start")
    print(f"H2: held-out log-density comparison, N_HELDOUT={N_HELDOUT}", flush=True)
    prior = box_uniform_2d()
    torch.manual_seed(55_555)
    theta_heldout = prior.sample((N_HELDOUT,))

    mark("h2_simulate_summary")
    print("  generating held-out summary-stat observations...", flush=True)
    t0 = time.time()
    x_summary_list = []
    for i in range(N_HELDOUT):
        x_summary_list.append(simulation_wrapper_sbi(
            theta_heldout[i:i + 1], inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
            scenario_specific_warm_start=True, seed=300_000 + i,
        ))
    x_summary = torch.cat(x_summary_list)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    mark("h2_simulate_raw")
    print("  generating held-out raw-trajectory observations (SAME theta, SAME noise seeds)...", flush=True)
    t0 = time.time()
    x_raw_list = []
    for i in range(N_HELDOUT):
        x_raw_list.append(simulation_wrapper_sbi_raw(
            theta_heldout[i:i + 1], inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
            scenario_specific_warm_start=True, seed=300_000 + i,
        ))
    x_raw = torch.cat(x_raw_list)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for seed in HC_SEEDS:
        mark(f"h2_hc_logprob_seed{seed}")
        post = load_hc_posterior(seed)
        lp = []
        with torch.no_grad():
            for i in range(N_HELDOUT):
                try:
                    v = post.log_prob(theta_heldout[i:i + 1], x=x_summary[i:i + 1],
                                       norm_posterior=False)
                    lp.append(float(v))
                except Exception:
                    continue
        lp = np.asarray(lp)
        rows.append({"representation": "hand_crafted", "seed": seed,
                     "n_valid": len(lp), "mean_logprob": float(np.mean(lp)),
                     "median_logprob": float(np.median(lp)),
                     "std_logprob": float(np.std(lp))})
        print(f"  HC seed{seed}: mean_logprob={np.mean(lp):+.2f} (n_valid={len(lp)})", flush=True)

    for seed in CNN_EXISTING_SEEDS:
        mark(f"h2_cnn_logprob_seed{seed}")
        post = load_cnn_posterior(seed)
        lp = []
        with torch.no_grad():
            for i in range(N_HELDOUT):
                try:
                    v = post.log_prob(theta_heldout[i:i + 1], x=x_raw[i:i + 1],
                                       norm_posterior=False)
                    lp.append(float(v))
                except Exception:
                    continue
        lp = np.asarray(lp)
        rows.append({"representation": "cnn_raw", "seed": seed,
                     "n_valid": len(lp), "mean_logprob": float(np.mean(lp)),
                     "median_logprob": float(np.median(lp)),
                     "std_logprob": float(np.std(lp))})
        print(f"  CNN seed{seed}: mean_logprob={np.mean(lp):+.2f} (n_valid={len(lp)})", flush=True)

    csv_path = RESULTS / "24_heldout_logprob_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for rep, color in [("hand_crafted", "steelblue"), ("cnn_raw", "darkorange")]:
        vals = [r["mean_logprob"] for r in rows if r["representation"] == rep]
        ax.scatter([rep] * len(vals), vals, color=color, s=60, alpha=0.8)
        ax.scatter([rep], [np.mean(vals)], color="black", marker="_", s=400, linewidths=2)
    ax.set_ylabel("mean held-out log p(theta | x)  [higher = better fit]")
    ax.set_title(f"nb24 H2: held-out predictive log-density\n"
                 f"(N_heldout={N_HELDOUT}, same theta/noise draws for both representations)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "24_heldout_logprob_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    mark("h2_done")
    print("H2 done.", flush=True)
    return rows


# ---------------------------------------------------------------------------
# H3: train 2 more CNN seeds, bring CNN to 5-seed parity with HC's nb22 study.
# ---------------------------------------------------------------------------

def run_h3():
    mark("h3_start")
    print("H3: training 2 more CNN seeds (seed3, seed4) at n=50,000...", flush=True)
    prior = box_uniform_2d()

    for seed in CNN_NEW_SEEDS:
        tag = f"seed{seed}_n{N_SIMULATIONS}"
        cache_path = RESULTS / f"22_theoryF_cnn_posterior_{tag}.pkl"
        if cache_path.exists():
            mark(f"h3_skip_{tag}")
            print(f"  {tag}: already cached, skipping training", flush=True)
            continue
        mark(f"h3_training_{tag}")
        print(f"  {tag}: training...", flush=True)
        t0 = time.time()
        try:
            emb = build_embedding_net()
            post, meta = train_sbi_posterior(
                prior, n_simulations=N_SIMULATIONS, density_estimator="nsf",
                hidden_features=128, num_transforms=5, training_batch_size=256,
                max_num_epochs=300, seed=seed, embedding_net=emb, use_raw_obs=True,
                scenario_specific_warm_start=True,
            )
            wall = time.time() - t0
            with open(cache_path, "wb") as f:
                pickle.dump({"posterior": post, "meta": meta, "wall_time_s": wall}, f)
            mark(f"h3_done_training_{tag}")
            print(f"  {tag}: trained in {wall:.0f}s", flush=True)
        except Exception as e:
            mark(f"h3_ERROR_{tag}")
            (RESULTS / f"24_error_cnn_{tag}.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
            print(f"  {tag}: ERROR -- {e} -- continuing", flush=True)
            continue

    mark("h3_training_all_done")
    print("H3: all CNN seed training done (or already cached). Computing Corr(alpha,beta)...", flush=True)

    all_cnn_seeds = CNN_EXISTING_SEEDS + [s for s in CNN_NEW_SEEDS
                                           if (RESULTS / f"22_theoryF_cnn_posterior_seed{s}_n{N_SIMULATIONS}.pkl").exists()]
    rows = []
    for sc_name in GRID_SCENARIOS:
        mark(f"h3_corr_{sc_name}")
        x_raw = obs_x_raw(sc_name, seed=999)
        corrs = {}
        for seed in all_cnn_seeds:
            post = load_cnn_posterior(seed)
            c, _ = corr_ab(post, x_raw, n_draw=5000, seed=600_000 + seed)
            corrs[seed] = c
        corr_vals = list(corrs.values())
        rows.append({
            "scenario": sc_name, "n_seeds": len(corr_vals),
            "corr_mean": float(np.mean(corr_vals)), "corr_std": float(np.std(corr_vals)),
            "corr_min": float(np.min(corr_vals)), "corr_max": float(np.max(corr_vals)),
            **{f"corr_seed{seed}": c for seed, c in corrs.items()},
        })
        print(f"  {sc_name:24s} CNN (n_seeds={len(corr_vals)}) "
              f"mean={np.mean(corr_vals):+.3f} std={np.std(corr_vals):.3f} "
              f"range=[{np.min(corr_vals):+.3f},{np.max(corr_vals):+.3f}]", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "24_cnn_5seed_corr_by_scenario.csv", index=False)

    # compare CNN's 5-seed std to HC's 5-seed std (from nb22, budget=50000)
    hc_summary_path = RESULTS / "22_metrics_per_budget_seed.csv"
    hc_std_sc4 = None
    if hc_summary_path.exists():
        import pandas as pd
        hc_df = pd.read_csv(hc_summary_path)
        hc_50k = hc_df[hc_df["budget"] == N_SIMULATIONS]
        if len(hc_50k) > 0:
            hc_std_sc4 = float(hc_50k["corr_ab_sc4"].std())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos = np.arange(len(GRID_SCENARIOS))
    means = df["corr_mean"].values
    stds = df["corr_std"].values
    ax.errorbar(xpos, means, yerr=stds, marker="o", capsize=4, color="darkorange",
                label=f"CNN ({len(all_cnn_seeds)} seeds)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([s.split("_")[0] for s in GRID_SCENARIOS])
    ax.set_ylabel("Corr(alpha,beta)")
    title = "nb24 H3: CNN-embedding Corr(alpha,beta) seed-robustness"
    if hc_std_sc4 is not None:
        title += f"\n(HC 5-seed std @ Sc4 for comparison: {hc_std_sc4:.3f})"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "24_cnn_5seed_corr_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    mark("h3_done")
    print("H3 done.", flush=True)
    return rows, hc_std_sc4


# ---------------------------------------------------------------------------
# H4: qualitative feature comparison (cheap, illustrative only).
# ---------------------------------------------------------------------------

def run_h4():
    mark("h4_start")
    print("H4: qualitative HC-summary vs CNN-embedding feature comparison...", flush=True)
    from cstr_sbi.simulator import apply_sensor_layer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick 3 fixed theta points (healthy, Sc4, Sc2), 20 noise realizations each.
    points = {
        "Sc1_healthy": (1.0, 1.0),
        "Sc4_combined": (0.85, 0.85),
        "Sc2_fouling": (1.0, 0.7),
    }
    n_noise = 20
    cnn_post = load_cnn_posterior(0)
    # Extract the embedding net directly from the trained density estimator
    # (sbi 0.26.1: posterior.posterior_estimator.embedding_net), confirmed
    # working via an isolated check before writing this function.
    try:
        net = cnn_post.posterior_estimator
        emb_module = net.embedding_net
    except Exception as e:
        mark("h4_ERROR_embedding_extract")
        print(f"  could not extract embedding net from posterior: {e} -- skipping H4", flush=True)
        return None

    summary_feats = {}
    cnn_feats = {}
    for name, (a, b) in points.items():
        theta = torch.tensor([[a, b]], dtype=torch.float32)
        s_list, e_list = [], []
        for n in range(n_noise):
            x_s = simulation_wrapper_sbi(theta, inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
                                          scenario_specific_warm_start=True, seed=900_000 + n)
            x_r = simulation_wrapper_sbi_raw(theta, inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
                                              scenario_specific_warm_start=True, seed=900_000 + n)
            s_list.append(x_s.numpy()[0])
            with torch.no_grad():
                e = emb_module(x_r).numpy()[0]
            e_list.append(e)
        summary_feats[name] = np.stack(s_list)
        cnn_feats[name] = np.stack(e_list)

    # Within-point vs between-point variance ratio for each representation
    # (a crude "discriminability" proxy: do embeddings separate the 3 points
    # more or less cleanly than the hand-crafted summary, relative to their
    # own noise floor?).
    def discriminability(feats_dict):
        all_pts = list(feats_dict.values())
        within_var = np.mean([np.var(p, axis=0).mean() for p in all_pts])
        means = np.stack([p.mean(axis=0) for p in all_pts])
        between_var = np.var(means, axis=0).mean()
        return float(between_var / (within_var + 1e-12))

    disc_summary = discriminability(summary_feats)
    disc_cnn = discriminability(cnn_feats)
    print(f"  discriminability (between/within variance ratio): "
          f"hand-crafted={disc_summary:.2f}  CNN-embedding={disc_cnn:.2f}", flush=True)

    # PCA-2D visualization of each representation's 3-point x 20-noise cloud
    from numpy.linalg import svd
    def pca2(all_feats):
        X = np.concatenate(list(all_feats.values()), axis=0)
        X = X - X.mean(axis=0)
        U, S, Vt = svd(X, full_matrices=False)
        return X @ Vt[:2].T

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, feats_dict, title in [
        (axes[0], summary_feats, f"hand-crafted 29-D summary\n(disc={disc_summary:.2f})"),
        (axes[1], cnn_feats, f"CNN embedding (20-D)\n(disc={disc_cnn:.2f})"),
    ]:
        proj = pca2(feats_dict)
        offset = 0
        for name, p in feats_dict.items():
            n = p.shape[0]
            ax.scatter(proj[offset:offset + n, 0], proj[offset:offset + n, 1],
                       label=name, alpha=0.7)
            offset += n
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("nb24 H4: qualitative feature-space comparison\n"
                 "(same theta points, same noise seeds, both representations)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "24_feature_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    mark("h4_done")
    print("H4 done.", flush=True)
    return {"disc_summary": disc_summary, "disc_cnn": disc_cnn}


# ---------------------------------------------------------------------------
# Final honest verdict
# ---------------------------------------------------------------------------

def write_verdict(h1_sbc, h1_grid, h2_rows, h3_rows, hc_std_sc4, h4_result):
    lines = []
    lines.append("nb24: why do hand-crafted-summary and CNN-raw-trajectory SBI")
    lines.append("posteriors disagree on Corr(alpha,beta)? -- verdict")
    lines.append("=" * 70)
    lines.append("")

    lines.append("H1 (CNN calibration -- SBC + anti-shrinkage slope at n=50,000):")
    for row in h1_sbc:
        lines.append(f"  seed{row['seed']}: ks_p_alpha={row['ks_p_alpha']:.3f} "
                      f"ks_p_beta={row['ks_p_beta']:.3f} "
                      f"mean_rank_a={row['mean_rank_a']:.1f} mean_rank_b={row['mean_rank_b']:.1f} "
                      f"frac_edge_a={row['frac_edge_20pct_a']:.2f} frac_edge_b={row['frac_edge_20pct_b']:.2f}")
    for row in h1_grid:
        lines.append(f"  seed{row['seed']}: slope_alpha={row['slope_alpha']:.3f} "
                      f"slope_beta={row['slope_beta']:.3f} "
                      f"coverage90_a={row['coverage90_alpha']:.2f} coverage90_b={row['coverage90_beta']:.2f}")
    sbc_fail = [r for r in h1_sbc if r["ks_p_alpha"] < 0.05 or r["ks_p_beta"] < 0.05]
    slope_fail = [r for r in h1_grid if r["slope_alpha"] < 0.8 or r["slope_beta"] < 0.8]
    lines.append(f"  -> SBC outright failures (p<0.05 on either param): {len(sbc_fail)}/{len(h1_sbc)} seeds")
    lines.append(f"  -> Slope<0.8 (shrinkage) on either param: {len(slope_fail)}/{len(h1_grid)} seeds")
    lines.append("")

    lines.append("H2 (held-out predictive log-density, HC vs CNN, same theta/noise):")
    hc_vals = [r["mean_logprob"] for r in h2_rows if r["representation"] == "hand_crafted"]
    cnn_vals = [r["mean_logprob"] for r in h2_rows if r["representation"] == "cnn_raw"]
    lines.append(f"  hand-crafted: mean_logprob={np.mean(hc_vals):+.2f} (across {len(hc_vals)} seeds, "
                 f"std={np.std(hc_vals):.2f})")
    lines.append(f"  CNN-raw:      mean_logprob={np.mean(cnn_vals):+.2f} (across {len(cnn_vals)} seeds, "
                 f"std={np.std(cnn_vals):.2f})")
    lines.append(f"  -> delta (HC - CNN) = {np.mean(hc_vals) - np.mean(cnn_vals):+.2f} "
                 "(positive means hand-crafted fits held-out data better)")
    lines.append("")

    lines.append("H3 (CNN multi-seed robustness, 5 seeds total):")
    for row in h3_rows:
        lines.append(f"  {row['scenario']}: mean={row['corr_mean']:+.3f} std={row['corr_std']:.3f} "
                      f"range=[{row['corr_min']:+.3f},{row['corr_max']:+.3f}] (n_seeds={row['n_seeds']})")
    sc4_row = next((r for r in h3_rows if r["scenario"] == "Sc4_closed_combined"), None)
    if sc4_row is not None:
        lines.append(f"  -> CNN 5-seed std @ Sc4 = {sc4_row['corr_std']:.3f}" +
                      (f"  vs HC 5-seed std @ Sc4 = {hc_std_sc4:.3f}" if hc_std_sc4 is not None else ""))
    lines.append("")

    if h4_result is not None:
        lines.append("H4 (qualitative feature-space comparison):")
        lines.append(f"  discriminability (between/within variance ratio across 3 theta points): "
                      f"hand-crafted={h4_result['disc_summary']:.2f}, "
                      f"CNN-embedding={h4_result['disc_cnn']:.2f}")
        lines.append("  See figures/24_feature_comparison.png for the PCA-2D projections.")
    else:
        lines.append("H4: skipped or failed (see log above) -- embedding extraction from the")
        lines.append("  trained posterior's internals was not available via the attempted API.")
    lines.append("")

    lines.append("-" * 70)
    lines.append("OVERALL ASSESSMENT (filled in manually after reviewing the numbers above")
    lines.append("-- do not auto-generate a conclusion the data doesn't support).")
    (RESULTS / "24_verdict.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def load_h1_from_csv():
    """Resume support: if results/24_cnn_sbc_metrics.csv and
    24_cnn_grid_diagnostics.csv already exist (from a prior run that
    completed H1 before crashing later), load them instead of
    recomputing -- H1 takes ~10 min and its output is already on disk."""
    sbc_csv = RESULTS / "24_cnn_sbc_metrics.csv"
    grid_csv = RESULTS / "24_cnn_grid_diagnostics.csv"
    if not (sbc_csv.exists() and grid_csv.exists()):
        return None
    import pandas as pd
    sbc_rows = pd.read_csv(sbc_csv).to_dict("records")
    grid_rows = pd.read_csv(grid_csv).to_dict("records")
    if len(sbc_rows) == len(CNN_EXISTING_SEEDS) and len(grid_rows) == len(CNN_EXISTING_SEEDS):
        return sbc_rows, grid_rows
    return None


def load_h2_from_csv():
    """Resume support, same rationale as load_h1_from_csv."""
    csv_path = RESULTS / "24_heldout_logprob_comparison.csv"
    if not csv_path.exists():
        return None
    import pandas as pd
    rows = pd.read_csv(csv_path).to_dict("records")
    expected = len(HC_SEEDS) + len(CNN_EXISTING_SEEDS)
    if len(rows) == expected:
        return rows
    return None


def main() -> None:
    mark("start")

    h1_cached = load_h1_from_csv()
    if h1_cached is not None:
        print("H1: found complete cached results on disk, skipping recomputation.", flush=True)
        h1_sbc, h1_grid = h1_cached
    else:
        h1_sbc, h1_grid = run_h1()

    h2_cached = load_h2_from_csv()
    if h2_cached is not None:
        print("H2: found complete cached results on disk, skipping recomputation.", flush=True)
        h2_rows = h2_cached
    else:
        h2_rows = run_h2()

    h3_rows, hc_std_sc4 = run_h3()
    h4_result = None
    try:
        h4_result = run_h4()
    except Exception as e:
        mark("h4_ERROR")
        (RESULTS / "24_error_h4.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
        print(f"H4 failed: {e} -- continuing without it", flush=True)
    write_verdict(h1_sbc, h1_grid, h2_rows, h3_rows, hc_std_sc4, h4_result)
    mark("finished")
    print("nb24 build script finished.", flush=True)


if __name__ == "__main__":
    main()
