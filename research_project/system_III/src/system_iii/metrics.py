"""Pure-numpy posterior-quality metrics for the System I warm-start (nb29) and
budget-convergence (nb31) studies.

No JAX / torch / cstr_sbi imports on purpose: this module is unit-testable on its own
(tests/test_ws_metrics.py) and is shared by scripts/ws_eval.py.

Conventions
-----------
- ``samples`` has shape (B, n, 2): n posterior draws of (alpha, beta) for each of B
  test observations. ``theta`` has shape (B, 2): the true parameters.
- The prior is the box [ALPHA_LOW, ALPHA_HIGH] x [BETA_LOW, BETA_HIGH] (passed in as
  ``lo``/``hi`` so this module needs no project import).
- Fault classes and the 0.90 threshold follow ``cstr_sbi.metrics.classify_fault``
  (class = quadrant with the largest posterior mass).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

CLASSES = ("healthy", "fouling_dominant", "decay_dominant", "combined")


# --------------------------------------------------------------------------- SBC
def sbc_ranks(samples: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Marginal SBC ranks: number of posterior draws below the true value. (B, 2) ints in [0, n]."""
    samples = np.asarray(samples)
    theta = np.asarray(theta)
    return (samples < theta[:, None, :]).sum(axis=1)


def sbc_ks_pvalues(ranks: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """KS test of randomised ranks against U(0,1), per marginal. Returns (2,) p-values.

    Ranks are made continuous by adding U(0,1) jitter before dividing by n+1, which makes
    the null distribution exactly uniform (avoids the conservative bias of a discrete KS)."""
    rng = np.random.default_rng(seed)
    u = (ranks + rng.uniform(size=ranks.shape)) / (n + 1)
    return np.array([stats.kstest(u[:, j], "uniform").pvalue for j in range(ranks.shape[1])])


def marginal_coverage(ranks: np.ndarray, n: int, level: float) -> np.ndarray:
    """Empirical coverage of the central ``level`` credible interval, per marginal. (2,)."""
    q = ranks / n
    lo, hi = (1 - level) / 2, (1 + level) / 2
    return ((q >= lo) & (q <= hi)).mean(axis=0)


def rank_shape(ranks: np.ndarray, n: int) -> dict:
    """Rank-histogram shape: mean rank (bias) and edge/middle mass (dispersion), as in nb22."""
    q = ranks / n
    out = {}
    for j, p in enumerate(("a", "b")):
        out[f"mean_rank_{p}"] = float(q[:, j].mean())          # 0.5 if unbiased
        out[f"frac_edge20_{p}"] = float(((q[:, j] < 0.2) | (q[:, j] > 0.8)).mean())  # 0.4 if calibrated
        out[f"frac_mid20_{p}"] = float(((q[:, j] >= 0.4) & (q[:, j] <= 0.6)).mean())  # 0.2 if calibrated
    return out


# -------------------------------------------------------------------------- TARP
def tarp(samples: np.ndarray, theta: np.ndarray, lo, hi, seed: int = 0, n_grid: int = 99) -> dict:
    """TARP joint-coverage test (Lemos et al. 2023) in prior-normalised coordinates.

    For each test point a reference point is drawn uniformly in the prior box; f_i is the
    fraction of posterior draws closer to the reference than the true theta. Under a
    correct posterior f ~ U(0,1), so the expected-coverage curve ECP(a) = mean(f < a) is
    the diagonal. Sensitive to errors in the JOINT shape, i.e. to a wrong (alpha,beta)
    correlation, which marginal SBC cannot see.

    Returns ``tarp_dev`` = max_a |ECP(a) - a| and ``tarp_ks_p`` (KS of f against U(0,1))."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    s = (np.asarray(samples) - lo) / (hi - lo)
    t = (np.asarray(theta) - lo) / (hi - lo)
    rng = np.random.default_rng(seed)
    ref = rng.uniform(size=t.shape)
    d_s = np.linalg.norm(s - ref[:, None, :], axis=2)
    d_t = np.linalg.norm(t - ref, axis=1)
    f = (d_s < d_t[:, None]).mean(axis=1)
    a = np.linspace(0.01, 0.99, n_grid)
    ecp = (f[:, None] < a[None, :]).mean(axis=0)
    return {"tarp_dev": float(np.max(np.abs(ecp - a))),
            "tarp_ks_p": float(stats.kstest(f, "uniform").pvalue)}


# --------------------------------------------------------------- point accuracy
def point_metrics(samples: np.ndarray, theta: np.ndarray, lo, hi) -> dict:
    """Posterior-mean RMSE (prior-width normalised, mean over the two parameters),
    per-parameter bias, posterior std and posterior Corr(alpha, beta)."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    mean = samples.mean(axis=1)
    err = (mean - theta) / (hi - lo)
    sd = samples.std(axis=1)
    corr = np.array([_corr(samples[i]) for i in range(samples.shape[0])])
    return {
        "rmse_norm": float(np.sqrt((err ** 2).mean())),
        "bias_a": float((mean[:, 0] - theta[:, 0]).mean()),
        "bias_b": float((mean[:, 1] - theta[:, 1]).mean()),
        "sd_a": float(sd[:, 0].mean()), "sd_b": float(sd[:, 1].mean()),
        "corr_ab_mean": float(np.nanmean(corr)),
    }


def _corr(x: np.ndarray) -> float:
    if x[:, 0].std() == 0 or x[:, 1].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])


def per_obs_corr(samples: np.ndarray) -> np.ndarray:
    return np.array([_corr(samples[i]) for i in range(samples.shape[0])])


# ---------------------------------------------------------------- classification
def class_of_theta(theta: np.ndarray, thr: float = 0.90) -> np.ndarray:
    """True fault class of each (alpha, beta), same quadrant rule as classify_fault."""
    a_ok = theta[:, 0] >= thr
    b_ok = theta[:, 1] >= thr
    out = np.empty(len(theta), dtype=object)
    out[a_ok & b_ok] = "healthy"
    out[a_ok & ~b_ok] = "fouling_dominant"
    out[~a_ok & b_ok] = "decay_dominant"
    out[~a_ok & ~b_ok] = "combined"
    return out


def class_of_samples(samples: np.ndarray, thr: float = 0.90) -> np.ndarray:
    """Predicted class = quadrant with the largest posterior mass (classify_fault rule)."""
    a_ok = samples[:, :, 0] >= thr
    b_ok = samples[:, :, 1] >= thr
    mass = np.stack([(a_ok & b_ok).mean(1), (a_ok & ~b_ok).mean(1),
                     (~a_ok & b_ok).mean(1), (~a_ok & ~b_ok).mean(1)], axis=1)
    return np.array(CLASSES, dtype=object)[mass.argmax(axis=1)]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the classes present in y_true (classes absent from the truth are
    not scored, matching the convention of reporting F1 over the evaluated scenarios)."""
    f1s = []
    for c in sorted(set(y_true)):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def credible_coverage_at_truth(samples: np.ndarray, theta: np.ndarray, level: float = 0.90) -> np.ndarray:
    """Per-observation, per-parameter: is the truth inside the central ``level`` interval? (B, 2) bool."""
    q_lo = np.quantile(samples, (1 - level) / 2, axis=1)
    q_hi = np.quantile(samples, (1 + level) / 2, axis=1)
    return (theta >= q_lo) & (theta <= q_hi)


# --------------------------------------------------------------- full evaluation
def interior_mask(theta: np.ndarray, lo, hi, margin: float = 0.05) -> np.ndarray:
    """True for draws at least ``margin`` (fraction of prior width) from every box edge.
    Near an edge the true posterior is truncated and NPE is known to degrade
    (Tirapongprasert & Ho 2026), so SBC is also reported on the interior subset."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    m = margin * (hi - lo)
    return np.all((theta >= lo + m) & (theta <= hi - m), axis=1)


def evaluate_sbc_set(samples, theta, lo, hi, seed=0) -> dict:
    """All prior-predictive (SBC-set) metrics for one posterior on one test set."""
    n = samples.shape[1]
    r = sbc_ranks(samples, theta)
    ks = sbc_ks_pvalues(r, n, seed=seed)
    inner = interior_mask(theta, lo, hi)
    ks_in = sbc_ks_pvalues(r[inner], n, seed=seed) if inner.sum() >= 50 else np.array([np.nan, np.nan])
    cov90 = marginal_coverage(r, n, 0.90)
    cov50 = marginal_coverage(r, n, 0.50)
    out = {"ks_p_a": float(ks[0]), "ks_p_b": float(ks[1]),
           "ks_p_a_interior": float(ks_in[0]), "ks_p_b_interior": float(ks_in[1]),
           "n_interior": int(inner.sum()),
           "cov90_a": float(cov90[0]), "cov90_b": float(cov90[1]),
           "cov50_a": float(cov50[0]), "cov50_b": float(cov50[1])}
    out.update(rank_shape(r, n))
    out.update(tarp(samples, theta, lo, hi, seed=seed))
    out.update(point_metrics(samples, theta, lo, hi))
    return out


def evaluate_scenario_set(samples, theta, sc_name, lo, hi, thr=0.90) -> tuple[dict, list[dict]]:
    """Scenario-set metrics: overall macro-F1 plus one row per scenario
    (bias, 90% coverage at truth, mean/sd of posterior Corr(alpha,beta), accuracy)."""
    sc_name = np.asarray(sc_name)
    y_true = class_of_theta(theta, thr)
    y_pred = class_of_samples(samples, thr)
    cover = credible_coverage_at_truth(samples, theta, 0.90)
    corr = per_obs_corr(samples)
    mean = samples.mean(axis=1)
    sd = samples.std(axis=1)
    rows = []
    for s in sorted(set(sc_name.tolist())):
        m = sc_name == s
        rows.append({
            "scenario": s,
            "alpha_true": float(theta[m, 0][0]), "beta_true": float(theta[m, 1][0]),
            "alpha_hat": float(mean[m, 0].mean()), "beta_hat": float(mean[m, 1].mean()),
            "bias_a": float((mean[m, 0] - theta[m, 0]).mean()),
            "bias_b": float((mean[m, 1] - theta[m, 1]).mean()),
            "sd_a": float(sd[m, 0].mean()), "sd_b": float(sd[m, 1].mean()),
            "cov90_truth_a": float(cover[m, 0].mean()), "cov90_truth_b": float(cover[m, 1].mean()),
            "corr_ab_mean": float(np.nanmean(corr[m])), "corr_ab_sd": float(np.nanstd(corr[m])),
            "class_acc": float((y_pred[m] == y_true[m]).mean()),
        })
    overall = {"macro_f1": macro_f1(y_true, y_pred),
               "accuracy": float((y_pred == y_true).mean())}
    return overall, rows
