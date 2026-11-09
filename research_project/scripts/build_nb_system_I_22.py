"""nb22: SBI training-budget convergence + full calibration study (System I),
run on the CORRECTED warm-start (closed_loop_steady_state_exact, fixed
2026-11-07 -- see notebooks/system_I/TODO.md / HANDOFF.md's dedicated
warm-start-fix entry for the derivation and verification).

This is a standalone script, not an nbclient-executed notebook, per this
project's own established lesson (nb20's multi-hour nbclient stalls on long
runs -- see build_nb_system_I_20.py's docstring): anything that runs for
hours should be a plain script with disk-flushed progress sentinels, run
self-caffeinated in the background, not a notebook cell whose stdout is
buffered until the whole run finishes.

Why every previously-cached posterior (n=10k/20k/40k from nb19/nb20) is NOT
reused here: those were trained before the warm-start fix (and some, per a
direct check of their saved metadata, did not even use
scenario_specific_warm_start=True in the first place), so they are not
comparable to a corrected-warm-start budget sweep. Everything below is
trained fresh.

Budgets: [1000, 5000, 10000, 20000, 40000, 50000], 5 seeds each (30 runs).
Cost-checked before launch (see HANDOFF.md): ~20ms/sample + ~2s fixed
overhead per training run (SNPE_C + NSF), extrapolated to ~3.5h serial total
for the 30 training runs; SBC/coverage/slope evaluation is comparatively
cheap (sampling-only, no retraining).

For each (budget, seed) this script computes and appends to
results/22_metrics_per_budget_seed.csv:
  - corr_ab_sc4            Corr(alpha,beta) at Sc4 (continuity w/ nb19/nb20)
  - ks_p_alpha, ks_p_beta  SBC KS p-values (N_SBC=500, num_posterior_samples
                           =1000, sbi.diagnostics.run_sbc/check_sbc -- EXACT
                           same convention as 04_sbi_training.ipynb)
  - mean_rank_a/b, frac_edge_20pct_a/b, frac_mid_20pct_a/b
                           rank-histogram shape diagnostics (nb19 Sec 10a
                           convention) to separate bias-type miscalibration
                           from over/under-dispersion
  - slope_alpha, slope_beta
                           posterior-mean-vs-truth regression slope across
                           the Sc1,2,3,4,5,7 closed-loop truth grid
                           (anti-shrinkage check)
  - coverage90_alpha, coverage90_beta
                           90% credible-interval coverage across the same
                           grid
  - std_alpha_mean, std_beta_mean
                           mean posterior std across the same grid
                           (precision stability)
  - wall_time_s            training wall time (from train_sbi_posterior meta)

Progress sentinels: results/22_progress_*.txt (disk-flushed, not print()).
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
import jax
import jax.numpy as jnp

from cstr_sbi.inference import train_sbi_posterior, simulation_wrapper_sbi
from cstr_sbi.priors import box_uniform_2d
from cstr_sbi.physics import NOMINAL_INLET_CL, NOMINAL_CTRL
from cstr_sbi.scenarios import SCENARIO_CONFIGS

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

BUDGETS = [1000, 5000, 10000, 20000, 40000, 50000]
SEEDS = [0, 1, 2, 3, 4]
N_SBC = 500
NUM_POSTERIOR_SAMPLES = 1000
GRID_SCENARIOS = [
    "Sc1_closed_healthy", "Sc2_closed_fouling", "Sc3_closed_decay",
    "Sc4_closed_combined", "Sc5_closed_saturated", "Sc7_closed_drift",
]

CSV_PATH = RESULTS / "22_metrics_per_budget_seed.csv"
CSV_FIELDS = [
    "budget", "seed", "wall_time_s", "corr_ab_sc4",
    "ks_p_alpha", "ks_p_beta",
    "mean_rank_a", "mean_rank_b",
    "frac_edge_20pct_a", "frac_edge_20pct_b",
    "frac_mid_20pct_a", "frac_mid_20pct_b",
    "slope_alpha", "slope_beta",
    "coverage90_alpha", "coverage90_beta",
    "std_alpha_mean", "std_beta_mean",
]


def mark(stage: str) -> None:
    """Disk-flushed progress sentinel -- see module docstring for why this
    is not print() (nbclient's stdout-buffering lesson applies to any long
    run, notebook or not; keeping the same convention here for a uniform
    external-monitoring experience across nb20/21/22)."""
    (RESULTS / f"22_progress_{stage}.txt").write_text(
        f"{time.time()}\n{_dt.datetime.now().isoformat()}\n"
    )


def ensure_csv_header() -> None:
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(row: dict) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def already_done(budget: int, seed: int) -> bool:
    if not CSV_PATH.exists():
        return False
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if int(row["budget"]) == budget and int(row["seed"]) == seed:
                return True
    return False


def obs_x_for_scenario(sc_name: str, seed: int = 999) -> torch.Tensor:
    """29-D summary observation at a scenario's own ground-truth (alpha,beta),
    generated with the SAME matched/corrected warm-start protocol used for
    training (scenario_specific_warm_start=True) -- avoids re-introducing a
    train/eval protocol mismatch while evaluating the budget sweep."""
    sc = SCENARIO_CONFIGS[sc_name]
    theta = torch.tensor([[sc.alpha, sc.beta]], dtype=torch.float32)
    x = simulation_wrapper_sbi(
        theta, inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
        scenario_specific_warm_start=True, seed=seed,
    )
    return x


def corr_ab(posterior, x, n_draw: int = 5000, seed: int = 0):
    torch.manual_seed(seed)
    samp = posterior.sample((n_draw,), x=x, show_progress_bars=False).numpy()
    return float(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1]), samp


def run_sbc_for_posterior(posterior, seed_offset: int = 100_000) -> dict:
    """Exact same convention as 04_sbi_training.ipynb: N_SBC=500,
    num_posterior_samples=1000, sbi.diagnostics.run_sbc/check_sbc, plus the
    nb19 Sec 10a rank-histogram shape diagnostics (mean_rank,
    frac_edge_20pct, frac_mid_20pct) to separate bias from dispersion."""
    from sbi.diagnostics import run_sbc, check_sbc

    prior = box_uniform_2d()
    torch.manual_seed(seed_offset)
    theta_sbc = prior.sample((N_SBC,))

    x_list = []
    for i in range(N_SBC):
        x_i = simulation_wrapper_sbi(
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
        n_bins_edge = NUM_POSTERIOR_SAMPLES * 0.20  # bottom/top 20% of [0, num_posterior_samples]
        frac_edge = float(np.mean(
            (r < n_bins_edge) | (r > NUM_POSTERIOR_SAMPLES - n_bins_edge)
        ))
        frac_mid = float(np.mean(
            (r >= (NUM_POSTERIOR_SAMPLES * 0.4)) & (r <= (NUM_POSTERIOR_SAMPLES * 0.6))
        ))
        return {"mean_rank": float(r.mean()), "frac_edge_20pct": frac_edge,
                "frac_mid_20pct": frac_mid}

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
    }


def grid_diagnostics(posterior, n_draw: int = 2000) -> dict:
    """Posterior-mean-vs-truth slope (anti-shrinkage), 90% CI coverage, and
    posterior std, across the Sc1,2,3,4,5,7 closed-loop truth grid."""
    truths_a, truths_b = [], []
    means_a, means_b = [], []
    cover_a, cover_b = [], []
    stds_a, stds_b = [], []
    for sc_name in GRID_SCENARIOS:
        sc = SCENARIO_CONFIGS[sc_name]
        x = obs_x_for_scenario(sc_name, seed=777 + sc.id)
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


def main() -> None:
    ensure_csv_header()
    mark("start")
    prior = box_uniform_2d()
    sc4_x = obs_x_for_scenario("Sc4_closed_combined", seed=999)

    total = len(BUDGETS) * len(SEEDS)
    done_count = 0
    for budget in BUDGETS:
        for seed in SEEDS:
            done_count += 1
            tag = f"budget{budget}_seed{seed}"
            if already_done(budget, seed):
                mark(f"skip_{tag}")
                print(f"[{done_count}/{total}] {tag}: already in CSV, skipping")
                continue

            mark(f"training_{tag}")
            print(f"[{done_count}/{total}] {tag}: training...", flush=True)
            t0 = time.time()
            try:
                save_to = str(RESULTS / f"22_{tag}.pkl")
                posterior, meta = train_sbi_posterior(
                    prior, n_simulations=budget, seed=seed,
                    scenario_specific_warm_start=True, save_to=save_to,
                )
                wall_time_s = meta.get("wall_time_s", time.time() - t0)

                mark(f"corr_{tag}")
                corr, _ = corr_ab(posterior, sc4_x, seed=seed)

                mark(f"sbc_{tag}")
                sbc_metrics = run_sbc_for_posterior(posterior, seed_offset=100_000 + seed * 1000)

                mark(f"grid_{tag}")
                grid_metrics = grid_diagnostics(posterior)

                row = {
                    "budget": budget, "seed": seed, "wall_time_s": wall_time_s,
                    "corr_ab_sc4": corr,
                    **sbc_metrics, **grid_metrics,
                }
                append_csv_row(row)
                mark(f"done_{tag}")
                print(f"[{done_count}/{total}] {tag}: DONE "
                      f"corr={corr:+.3f} ks_p_a={sbc_metrics['ks_p_alpha']:.3f} "
                      f"ks_p_b={sbc_metrics['ks_p_beta']:.3f} "
                      f"slope_a={grid_metrics['slope_alpha']:.3f} "
                      f"slope_b={grid_metrics['slope_beta']:.3f} "
                      f"wall={wall_time_s:.0f}s", flush=True)
            except Exception as e:
                mark(f"ERROR_{tag}")
                err_path = RESULTS / f"22_error_{tag}.txt"
                err_path.write_text(f"{e}\n\n{traceback.format_exc()}")
                print(f"[{done_count}/{total}] {tag}: ERROR -- {e} "
                      f"(see {err_path}) -- continuing to next run", flush=True)
                continue

    mark("all_training_done")
    print("All training+evaluation runs complete. Writing summary...", flush=True)
    write_summary()
    mark("finished")
    print("nb22 build script finished.", flush=True)


def write_summary() -> None:
    """Aggregate per-seed metrics into per-budget distributions (mean/std/
    range), save a summary CSV, render summary figures, and write an
    explicit optimal-n_sims recommendation -- NOT forcing a single clean
    number if the metrics disagree (per this project's own house style)."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(CSV_PATH)
    metrics = [
        "corr_ab_sc4", "ks_p_alpha", "ks_p_beta",
        "slope_alpha", "slope_beta",
        "coverage90_alpha", "coverage90_beta",
        "std_alpha_mean", "std_beta_mean",
    ]
    summary_rows = []
    for budget, g in df.groupby("budget"):
        row = {"budget": budget, "n_seeds": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std()
            row[f"{m}_min"] = g[m].min()
            row[f"{m}_max"] = g[m].max()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("budget")
    summary_df.to_csv(RESULTS / "22_summary_per_budget.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    panels = [
        ("corr_ab_sc4", "Corr(alpha,beta) @ Sc4"),
        ("ks_p_alpha", "SBC KS p-value (alpha)"),
        ("ks_p_beta", "SBC KS p-value (beta)"),
        ("slope_alpha", "mean-vs-truth slope (alpha)"),
        ("slope_beta", "mean-vs-truth slope (beta)"),
        ("coverage90_alpha", "90% CI coverage (alpha)"),
        ("coverage90_beta", "90% CI coverage (beta)"),
        ("std_alpha_mean", "mean posterior std (alpha)"),
        ("std_beta_mean", "mean posterior std (beta)"),
    ]
    for ax, (m, title) in zip(axes.flat, panels):
        g = df.groupby("budget")[m]
        budgets = sorted(df["budget"].unique())
        means = [g.get_group(b).mean() if b in g.groups else np.nan for b in budgets]
        stds = [g.get_group(b).std() if b in g.groups else np.nan for b in budgets]
        ax.errorbar(budgets, means, yerr=stds, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("n_simulations/seed")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if "ks_p" in m:
            ax.axhline(0.05, color="red", ls="--", lw=1, label="p=0.05")
            ax.legend(fontsize=8)
        if "slope" in m:
            ax.axhline(1.0, color="green", ls="--", lw=1, label="slope=1 (no shrinkage)")
            ax.legend(fontsize=8)
        if "coverage" in m:
            ax.axhline(0.90, color="green", ls="--", lw=1, label="nominal 90%")
            ax.legend(fontsize=8)
    fig.suptitle("nb22: SBI training-budget convergence study (System I, corrected warm-start)\n"
                 "candidate SI content, clearly labelled, not yet promoted",
                 fontsize=12)
    fig.savefig(FIGS / "22_budget_convergence_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    write_recommendation(summary_df)


def write_recommendation(summary_df) -> None:
    """Explicit, honest optimal-n_sims recommendation. Does NOT force a
    single clean number if calibration, anti-shrinkage, and correlation
    convergence disagree -- states the disagreement plainly if present."""
    lines = []
    lines.append("nb22 Part 3: optimal n_sims recommendation")
    lines.append("=" * 60)
    lines.append("")
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    sbc_pass = summary_df[
        (summary_df["ks_p_alpha_mean"] > 0.05) & (summary_df["ks_p_beta_mean"] > 0.05)
    ]
    slope_ok = summary_df[
        (summary_df["slope_alpha_mean"] > 0.8) & (summary_df["slope_beta_mean"] > 0.8)
    ]
    corr_deltas = summary_df["corr_ab_sc4_mean"].diff().abs()
    corr_converged = summary_df[corr_deltas.fillna(1.0) < 0.05]

    lines.append(f"Budgets passing SBC (both params, mean KS p>0.05): "
                 f"{sbc_pass['budget'].tolist()}")
    lines.append(f"Budgets with slope>0.8 (both params, anti-shrinkage OK): "
                 f"{slope_ok['budget'].tolist()}")
    lines.append(f"Budgets where Corr(alpha,beta)@Sc4 step-to-step delta < 0.05 "
                 f"(convergence criterion from nb20): {corr_converged['budget'].tolist()}")
    lines.append("")

    candidates = set(sbc_pass["budget"]) & set(slope_ok["budget"]) & set(corr_converged["budget"])
    if candidates:
        rec = min(candidates)
        lines.append(f"RECOMMENDATION: n_sims = {rec} is the smallest budget satisfying "
                     "SBC calibration, anti-shrinkage slope, and Corr(alpha,beta)@Sc4 "
                     "convergence simultaneously.")
    else:
        lines.append("RECOMMENDATION: NO budget in the tested range "
                     "[1000,5000,10000,20000,40000,50000] satisfies all three criteria "
                     "(SBC calibration, anti-shrinkage slope>0.8, and Corr(alpha,beta)@Sc4 "
                     "convergence) simultaneously. This is an honest non-finding, not a "
                     "forced choice -- see the per-metric budget lists above for where "
                     "each criterion is individually satisfied, and the full "
                     "results/22_summary_per_budget.csv / "
                     "results/22_metrics_per_budget_seed.csv for the raw distributions "
                     "across seeds. If the metrics disagree about the direction (e.g. "
                     "SBC passes at a LOWER budget than where the slope/correlation "
                     "stabilizes), that disagreement is itself the finding: calibration "
                     "and anti-shrinkage/correlation-convergence are not the same "
                     "property, and this codebase's recipe may need more than raw budget "
                     "escalation (e.g. an architecture change, more MCMC-based "
                     "cross-validation, or an even higher budget beyond 50000) to satisfy "
                     "both at once.")
    out = "\n".join(lines)
    (RESULTS / "22_final_recommendation.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
