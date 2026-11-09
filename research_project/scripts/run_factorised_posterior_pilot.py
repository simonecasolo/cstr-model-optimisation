"""Pilot: factorised S-C posterior on the EXISTING Stage-1b-fixed n=1000 bank
(no new bank generation). Per `system_ii_sbi_solidification_plan.md` Stage 2:

    p(theta | x) = p(z_A0_eff | x_analyzer) * p(alpha, beta_r, eta_col, xi_reb | x)

- `z_A0_eff`: EXACT analytic posterior (truncated-lognormal prior x Gaussian
  likelihood from the feed-analyzer window-mean feature), via 1-D numerical
  grid integration -- no training, calibrated by construction.
- `(alpha, beta_r, eta_col, xi_reb)`: 4-D SNPE flow in logit space (exact
  induced prior, not a Gaussian surrogate), conditioned on the full 72-D S-C
  summary vector (the analyzer features stay in `x` as conditioning, per the
  plan -- near-equivalent to conditioning on z_A0_eff itself).

Purpose: test whether removing the near-delta z_A0_eff dimension from the
flow's training objective (the diagnosed root cause of the alpha/xi_reb
shrinkage) helps at n_train=1000, BEFORE paying for a n=5000 bank regen.
"""
from __future__ import annotations

import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.priors import (
    truncated_lognormal_5d, induced_logit_prior, TruncatedLogNormalPrior,
    theta_to_u, u_to_theta, PRIOR_LOW_5D, PRIOR_HIGH_5D, NOMINAL_THETA_5D, LOGN_SIGMA_5D,
    EPS_LOGIT_BOX,
)
from cstr_sbi.recycle.summaries import summary_names, FEED_ANALYZER_NOISE_PCT

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SBI_LOGS = ROOT / "sbi-logs"
for p in [DATA, RESULTS, SBI_LOGS]:
    p.mkdir(exist_ok=True)

N_TRAIN = 1000
SUFFIX = "_stage1bfix"
ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
N_POST = 120
SEEDS = [0, 1, 2]

# --- 4-D subspace indices (alpha, beta_r, eta_col, xi_reb -- exclude z_A0_eff) ---
FLOW_IDX = [0, 1, 2, 3]
Z_IDX = 4
FLOW_NAMES = [PARAM_NAMES[i] for i in FLOW_IDX]
LOW4, HIGH4 = PRIOR_LOW_5D[FLOW_IDX], PRIOR_HIGH_5D[FLOW_IDX]
FEED_ANALYZER_MEAN_COL = summary_names("S-C").index("feed_analyzer_mean")

# --- Load data ---
bank = np.load(DATA / f"wu2003_sbi_train_sc_lognormal_true_n{N_TRAIN}{SUFFIX}.npz")
theta_train = np.asarray(bank["thetas"], dtype=np.float32)
summary_train = np.asarray(bank["summaries"], dtype=np.float32)
r120 = np.load(RESULTS / f"24a_sysII_sc_lognormal_sbc_set_n120{SUFFIX}.npz")
theta_sbc, summary_sbc = np.asarray(r120["thetas"], dtype=np.float32), np.asarray(r120["summaries"], dtype=np.float32)
r400 = np.load(RESULTS / f"24a_sysII_sc_lognormal_sbc_set_n400{SUFFIX}.npz")
theta_sbc_c, summary_sbc_c = np.asarray(r400["thetas"], dtype=np.float32), np.asarray(r400["summaries"], dtype=np.float32)
print(f"theta_train={theta_train.shape} theta_sbc={theta_sbc.shape} theta_sbc_c={theta_sbc_c.shape}")


# ===========================================================================
# Part A: exact analytic z_A0_eff posterior
# ===========================================================================

# Marginal prior on z_A0_eff alone: the 5-D truncated-lognormal prior's box
# truncation is separable across dimensions (a hyper-rectangle indicator
# factorises), so the marginal of an independent-in-log-space truncated
# lognormal IS itself exactly a 1-D truncated lognormal with the same
# median/sigma/bounds on that one dimension -- no approximation here.
Z_MEDIAN, Z_SIGMA = float(NOMINAL_THETA_5D[Z_IDX]), float(LOGN_SIGMA_5D[Z_IDX])
Z_LOW, Z_HIGH = float(PRIOR_LOW_5D[Z_IDX]), float(PRIOR_HIGH_5D[Z_IDX])

_Z_GRID = np.linspace(Z_LOW, Z_HIGH, 20001, dtype=np.float64)
_Z_LOGPDF_PRIOR = None  # filled by _z_prior_logpdf on first use


def _z_prior_logpdf(z_grid):
    """log-density of the exact marginal truncated-lognormal prior on z_A0_eff, on `z_grid`."""
    mu = np.log(Z_MEDIAN)
    lo_z = (np.log(Z_LOW) - mu) / Z_SIGMA
    hi_z = (np.log(Z_HIGH) - mu) / Z_SIGMA
    log_norm = np.log(scipy_stats.norm.cdf(hi_z) - scipy_stats.norm.cdf(lo_z))
    logpdf = scipy_stats.norm.logpdf(np.log(z_grid), loc=mu, scale=Z_SIGMA) - np.log(z_grid) - log_norm
    return logpdf


_Z_LOGPDF_PRIOR = _z_prior_logpdf(_Z_GRID)


def z_a0_eff_posterior_grid(analyzer_mean_reading: float, n_readings: int = 120):
    """Exact posterior p(z_A0_eff | analyzer window-mean reading), via 1-D grid
    integration (truncated-lognormal prior x Gaussian likelihood). Returns
    (grid, pmf) -- a discrete approximation to the exact posterior, fine enough
    (20001 points) that grid error is negligible next to the likelihood's own
    ~1e-5-scale width. `n_readings` matches the window length used to average
    the analyzer's per-timestep noise (see `feed_analyzer_reading`)."""
    # likelihood sd for the window-MEAN (not a single reading): noise averages
    # down by sqrt(n_readings), matching how the "feed_analyzer_mean" summary
    # feature is computed.
    lik_sd = FEED_ANALYZER_NOISE_PCT * max(abs(analyzer_mean_reading), 1e-3) / np.sqrt(n_readings)
    loglik = scipy_stats.norm.logpdf(_Z_GRID, loc=analyzer_mean_reading, scale=lik_sd)
    logpost = _Z_LOGPDF_PRIOR + loglik
    logpost -= logpost.max()
    pmf = np.exp(logpost)
    pmf /= pmf.sum()
    return _Z_GRID, pmf


def sample_z_a0_eff_posterior(analyzer_mean_reading: float, n_samples: int, rng: np.random.Generator):
    grid, pmf = z_a0_eff_posterior_grid(analyzer_mean_reading)
    idx = rng.choice(len(grid), size=n_samples, p=pmf)
    # de-discretise: jitter within the grid cell to avoid exact ties at rank time
    cell_width = grid[1] - grid[0]
    jitter = rng.uniform(-0.5, 0.5, n_samples) * cell_width
    return np.clip(grid[idx] + jitter, Z_LOW, Z_HIGH)


def sbc_z_a0_eff(theta_sbc_set, summary_sbc_set, seed, n_post=N_POST):
    rng = np.random.default_rng(seed)
    ranks = []
    for th, x_obs in zip(theta_sbc_set, summary_sbc_set):
        true_z = th[Z_IDX]
        # compute_summaries applies _compress_heavy_tails = sign(x)*log1p(|x|)
        # to the WHOLE summary vector uniformly, including feed_analyzer_mean --
        # invert it (expm1, exact inverse of log1p for positive values) to
        # recover the raw noisy reading before using it as the likelihood mean.
        reading = np.expm1(x_obs[FEED_ANALYZER_MEAN_COL])
        samp = sample_z_a0_eff_posterior(reading, n_post, rng)
        ranks.append(int(np.sum(samp < true_z)))
    r = np.asarray(ranks, dtype=float)
    ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
    return {"ks_pvalue": float(ks.pvalue), "mean_rank_frac": float(r.mean() / n_post), "n_sbc": len(theta_sbc_set)}


# ===========================================================================
# Part B: 4-D flow in logit space over (alpha, beta_r, eta_col, xi_reb)
# ===========================================================================

theta_prior_4d = TruncatedLogNormalPrior(
    torch.as_tensor(NOMINAL_THETA_5D[FLOW_IDX]), torch.as_tensor(LOGN_SIGMA_5D[FLOW_IDX]),
    torch.as_tensor(LOW4), torch.as_tensor(HIGH4),
)
prior_u_4d = induced_logit_prior(theta_prior_4d, low=LOW4, high=HIGH4)


def train_4d_posterior(torch_seed, z_score_x="none"):
    torch.manual_seed(torch_seed)
    u_train = theta_to_u(theta_train[:, FLOW_IDX], LOW4, HIGH4)
    kwargs = dict(model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS)
    if z_score_x is not None:
        kwargs["z_score_x"] = z_score_x
    inference = SNPE(prior=prior_u_4d, density_estimator=posterior_nn(**kwargs))
    inference.append_simulations(torch.tensor(u_train, dtype=torch.float32), torch.tensor(summary_train, dtype=torch.float32))
    t0 = time.time()
    trained = inference.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1, stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    return inference.build_posterior(trained), time.time() - t0


def sbc_4d(posterior, theta_sbc_set, summary_sbc_set, seed, n_post=N_POST):
    torch.manual_seed(seed)
    ranks = {name: [] for name in FLOW_NAMES}
    for th, x_obs in zip(theta_sbc_set, summary_sbc_set):
        with torch.no_grad():
            u_samp = posterior.sample((n_post,), x=torch.tensor(x_obs, dtype=torch.float32),
                                       show_progress_bars=False, reject_outside_prior=False).detach().cpu().numpy()
        theta_samp = u_to_theta(u_samp, LOW4, HIGH4)
        for k, name in enumerate(FLOW_NAMES):
            ranks[name].append(int(np.sum(theta_samp[:, k] < th[FLOW_IDX[k]])))
    results = {"n_sbc": int(len(theta_sbc_set)), "n_post": int(n_post)}
    for name in FLOW_NAMES:
        r = np.asarray(ranks[name], dtype=float)
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        results[name] = {"ks_pvalue": float(ks.pvalue), "mean_rank_frac": float(r.mean() / n_post)}
    results["min_ks_pvalue"] = min(results[name]["ks_pvalue"] for name in FLOW_NAMES)
    return results


def main():
    t0 = time.time()

    print("=== Part A: z_A0_eff exact analytic posterior, SBC (unit test) ===")
    z_reduced = sbc_z_a0_eff(theta_sbc, summary_sbc, seed=500001)
    z_confirm = sbc_z_a0_eff(theta_sbc_c, summary_sbc_c, seed=500002)
    print(f"  reduced (N=120): ks_pvalue={z_reduced['ks_pvalue']:.4g}  mean_rank_frac={z_reduced['mean_rank_frac']:.4f}")
    print(f"  confirm (N=400): ks_pvalue={z_confirm['ks_pvalue']:.4g}  mean_rank_frac={z_confirm['mean_rank_frac']:.4f}")

    print("\n=== Part B: 4-D logit-space flow, 3 seeds ===")
    all_reduced, all_confirm = [], []
    for seed in SEEDS:
        posterior_path = SBI_LOGS / f"wu2003_posterior_sc_factorised4d_seed{seed}_n{N_TRAIN}{SUFFIX}.pkl"
        if posterior_path.exists():
            with open(posterior_path, "rb") as f:
                payload = pickle.load(f)
            posterior, train_wall_s = payload["posterior"], float(payload.get("train_wall_s", 0.0))
        else:
            posterior, train_wall_s = train_4d_posterior(seed)
            with open(posterior_path, "wb") as f:
                pickle.dump({"posterior": posterior, "n_train": N_TRAIN, "torch_seed": seed,
                             "flow_idx": FLOW_IDX, "z_score_x": "none", "train_wall_s": train_wall_s,
                             "note": "factorised-posterior pilot, 4-D logit-space flow"}, f)
        sbc_r = sbc_4d(posterior, theta_sbc, summary_sbc, seed=510000 + seed)
        sbc_c = sbc_4d(posterior, theta_sbc_c, summary_sbc_c, seed=520000 + seed)
        print(f"  seed {seed}: train_wall_s={train_wall_s:.1f}  reduced min_ks={sbc_r['min_ks_pvalue']:.4g}  "
              f"confirm min_ks={sbc_c['min_ks_pvalue']:.4g}")
        print(f"    reduced per-param: " + "  ".join(f"{n}={sbc_r[n]['ks_pvalue']:.3g}" for n in FLOW_NAMES))
        print(f"    confirm per-param: " + "  ".join(f"{n}={sbc_c[n]['ks_pvalue']:.3g}" for n in FLOW_NAMES))
        all_reduced.append({"seed": seed, "train_wall_s": train_wall_s, **{f"{n}_ks": sbc_r[n]["ks_pvalue"] for n in FLOW_NAMES}, "min_ks": sbc_r["min_ks_pvalue"]})
        all_confirm.append({"seed": seed, **{f"{n}_ks": sbc_c[n]["ks_pvalue"] for n in FLOW_NAMES}, "min_ks": sbc_c["min_ks_pvalue"]})

    reduced_df = pd.DataFrame(all_reduced)
    confirm_df = pd.DataFrame(all_confirm)
    reduced_df.to_csv(RESULTS / f"factorised4d_reduced_sbc_n120_ntrain{N_TRAIN}{SUFFIX}.csv", index=False)
    confirm_df.to_csv(RESULTS / f"factorised4d_confirm_sbc_n400_ntrain{N_TRAIN}{SUFFIX}.csv", index=False)

    n_pass_r = int((reduced_df["min_ks"] > 0.05).sum())
    n_pass_c = int((confirm_df["min_ks"] > 0.05).sum())
    print(f"\n[factorised 4D] {n_pass_r}/{len(SEEDS)} pass reduced, {n_pass_c}/{len(SEEDS)} pass confirm")
    print(f"\nAll done in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
