"""Diagnostic figure requested directly: histograms of (1) prior theta samples
under both the lognormal and uniform priors, (2) the most significant summary
statistics under the lognormal bank (uniform-prior bank generation was
stopped before any usable data existed -- flagged honestly, not faked), and
(3) the seed-0 trained posterior at a representative severe-shrinkage
scenario (W4_cat_threshold), overlaid against the lognormal prior and the
true value, to visualize the shrinkage failure directly.
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.priors import box_uniform_5d, truncated_lognormal_5d
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.summaries import summary_names
from cstr_sbi.recycle.scenarios import list_closed_loop
from cstr_sbi.recycle.simulator import nominal_warm_start, deterministic_window, noisy_replicates
from cstr_sbi.recycle.summaries import compute_summaries

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SBI_LOGS = ROOT / "sbi-logs"

# Okabe-Ito colorblind-safe palette
C_LOGN = "#0072B2"   # blue
C_UNIF = "#E69F00"   # orange
C_TRUE = "#D55E00"   # vermillion
C_POST = "#009E73"   # green

# --- Row 1: prior theta samples, lognormal (real bank) vs uniform (fresh sample) ---
bank = np.load(DATA / "wu2003_sbi_train_sc2_lognormal_true_n1000_stage1bfix.npz")
theta_logn = np.asarray(bank["thetas"], dtype=np.float32)

torch.manual_seed(0)
prior_unif = box_uniform_5d()
theta_unif = prior_unif.sample((1000,)).numpy()

# --- Row 2: summary statistics, lognormal bank only ---
summary_logn = np.asarray(bank["summaries"], dtype=np.float32)
names = summary_names("S-C.2")
SUMMARY_FEATS = ["feed_analyzer_mean", "xi_reb_qreb_feat", "xi_reb_treb_feat", "T_r_mean"]
feat_idx = [names.index(f) for f in SUMMARY_FEATS]

# --- Row 3: seed-0 posterior at W4_cat_threshold vs lognormal prior vs truth ---
with open(SBI_LOGS / "wu2003_posterior_sc2_lognormal_true_n1000_zscorenone_seed0_stage1bfix.pkl", "rb") as f:
    posterior = pickle.load(f)["posterior"]

scenarios = {sc.name: sc for sc in list_closed_loop()}
sc = scenarios["W4_cat_threshold"]
theta_true = np.asarray(sc.theta(), dtype=np.float32)
y0_sb = nominal_warm_start("S-B")
t_h, raw_obs = deterministic_window(sc, structure="S-B", y0=y0_sb)
t_h_arr = np.asarray(t_h)
rng = np.random.default_rng(20260705)
reps = noisy_replicates(raw_obs, n_replicates=1, rng=rng, noise_pct=0.003)
s_obs = compute_summaries(reps[0], "S-C.2", t_h_arr, theta=theta_true, rng=rng)

torch.manual_seed(0)
with torch.no_grad():
    post_samples = posterior.sample((2000,), x=torch.tensor(s_obs, dtype=torch.float32),
                                     show_progress_bars=False, reject_outside_prior=False).numpy()

torch.manual_seed(1)
prior_logn_dist = truncated_lognormal_5d()
prior_logn_samples = prior_logn_dist.sample((2000,)).numpy()

# ============================================================================
fig, axes = plt.subplots(3, 5, figsize=(22, 12))

for j, name in enumerate(PARAM_NAMES):
    ax = axes[0, j]
    ax.hist(theta_logn[:, j], bins=30, color=C_LOGN, alpha=0.6, density=True, label="lognormal (n=1000)")
    ax.hist(theta_unif[:, j], bins=30, color=C_UNIF, alpha=0.5, density=True, label="uniform (n=1000, fresh sample)")
    ax.set_title(f"prior: {name}", fontsize=11)
    if j == 0:
        ax.set_ylabel("Row 1: PRIOR theta samples", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

for j in range(5):
    ax = axes[1, j]
    if j < len(SUMMARY_FEATS):
        feat = SUMMARY_FEATS[j]
        ax.hist(summary_logn[:, feat_idx[j]], bins=30, color=C_LOGN, alpha=0.7, density=True)
        ax.set_title(f"summary (lognormal bank only): {feat}", fontsize=10.5)
        ax.text(0.03, 0.95, "uniform-prior bank:\nstopped, no data yet",
                transform=ax.transAxes, fontsize=8.5, va="top", color=C_UNIF,
                bbox=dict(boxstyle="round", fc="white", ec=C_UNIF, alpha=0.9))
    else:
        ax.axis("off")
    if j == 0:
        ax.set_ylabel("Row 2: SUMMARY STATISTICS", fontsize=10)

for j, name in enumerate(PARAM_NAMES):
    ax = axes[2, j]
    ax.hist(prior_logn_samples[:, j], bins=30, color=C_LOGN, alpha=0.35, density=True, label="lognormal prior")
    ax.hist(post_samples[:, j], bins=30, color=C_POST, alpha=0.7, density=True, label="seed-0 posterior | W4 obs")
    ax.axvline(theta_true[j], color=C_TRUE, lw=2.5, ls="--", label=f"truth={theta_true[j]:.2f}")
    ax.set_title(f"posterior: {name}\n(W4_cat_threshold, severe fault)", fontsize=10)
    if j == 0:
        ax.set_ylabel("Row 3: POSTERIOR vs prior vs truth", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

fig.suptitle("Prior / summary-statistic / posterior diagnostics -- lognormal vs uniform prior, "
             "and the shrinkage failure at W4 (severe alpha fault)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_path = FIGURES / "44_prior_summary_posterior_diagnostics.png"
fig.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"saved {out_path}")
