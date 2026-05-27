"""Posterior comparison metrics — M5 / M7 deliverable.

Direct port of ``sbi_mcmc_heat_exchanger/src/hx_models/metrics.py``,
adapted for the 4-D CSTR parameter space ``[UA, k0, alpha, beta]``.

Requires ``properscoring`` (for CRPS) and ``scipy``.

Public API
----------
compute_crps(samples, true_value) -> float
    Continuous Ranked Probability Score (lower is better).

compute_wasserstein(samples_a, samples_b) -> float
    Wasserstein-1 distance between two 1-D sample sets.

compute_kl_divergence(samples_a, samples_b) -> float
    KL(P || Q) via Gaussian KDE, averaged with the reverse direction.

coverage_check(samples, true_value, levels) -> dict
    Boolean coverage + CI width at the requested credible levels.

compute_all_continuous_metrics(samples, true_value, *, other_samples) -> dict
    Convenience aggregator used by notebook 05 and 06.

r_hat(chains) -> np.ndarray
    Gelman-Rubin R̂ per parameter.  ``chains`` has shape ``(n_chains, n_draws, n_params)``.

effective_sample_size(chains) -> np.ndarray
    Bulk ESS per parameter (ArviZ implementation).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde, entropy


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def compute_crps(samples: np.ndarray, true_value: float) -> float:
    """CRPS for a set of posterior samples vs. a scalar true value.

    Lower is better.  Uses ``properscoring.crps_ensemble``.
    """
    from properscoring import crps_ensemble
    return float(crps_ensemble(true_value, np.asarray(samples).ravel()))


# ---------------------------------------------------------------------------
# Wasserstein-1
# ---------------------------------------------------------------------------

def compute_wasserstein(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Wasserstein-1 (earth-mover's) distance between two 1-D sample sets."""
    return float(wasserstein_distance(
        np.asarray(samples_a).ravel(),
        np.asarray(samples_b).ravel(),
    ))


# ---------------------------------------------------------------------------
# KL divergence via KDE
# ---------------------------------------------------------------------------

def compute_kl_divergence(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_points: int = 1000,
) -> float:
    """KL(P_a || P_b) estimated via Gaussian KDE on a shared grid."""
    a = np.asarray(samples_a).ravel()
    b = np.asarray(samples_b).ravel()
    x_min = min(a.min(), b.min())
    x_max = max(a.max(), b.max())
    pad = 0.1 * (x_max - x_min) + 1e-8
    grid = np.linspace(x_min - pad, x_max + pad, n_points)
    p = gaussian_kde(a)(grid) + 1e-10
    q = gaussian_kde(b)(grid) + 1e-10
    p /= p.sum(); q /= q.sum()
    return float(entropy(p, q))


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def coverage_check(
    samples: np.ndarray,
    true_value: float,
    levels: tuple[float, ...] = (0.5, 0.9, 0.95),
) -> dict:
    """Check whether ``true_value`` lies within each credible interval.

    Returns a flat dict with keys like ``coverage_90``, ``ci_width_90``,
    ``ci_lower_90``, ``ci_upper_90`` for each level.
    """
    samples = np.asarray(samples).ravel()
    out: dict = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lo = float(np.percentile(samples, 100 * alpha))
        hi = float(np.percentile(samples, 100 * (1.0 - alpha)))
        pct = int(round(level * 100))
        out[f"coverage_{pct}"] = bool(lo <= true_value <= hi)
        out[f"ci_width_{pct}"] = hi - lo
        out[f"ci_lower_{pct}"] = lo
        out[f"ci_upper_{pct}"] = hi
    return out


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def compute_all_continuous_metrics(
    samples: np.ndarray,
    true_value: float,
    *,
    other_samples: np.ndarray | None = None,
    levels: tuple[float, ...] = (0.5, 0.9, 0.95),
) -> dict:
    """All metrics for a single continuous parameter.

    Parameters
    ----------
    samples
        Posterior samples (1-D array).
    true_value
        Ground-truth scalar.
    other_samples
        Optional second sample set (e.g. MCMC vs. SBI) for W1/KL comparison.
    """
    s = np.asarray(samples).ravel()
    out: dict = {
        "crps":   compute_crps(s, true_value),
        "mean":   float(np.mean(s)),
        "median": float(np.median(s)),
        "std":    float(np.std(s)),
    }
    out.update(coverage_check(s, true_value, levels))
    if other_samples is not None:
        other = np.asarray(other_samples).ravel()
        out["wasserstein"]  = compute_wasserstein(s, other)
        out["kl_forward"]   = compute_kl_divergence(s, other)
        out["kl_reverse"]   = compute_kl_divergence(other, s)
    return out


# ---------------------------------------------------------------------------
# MCMC diagnostics: R-hat and ESS
# ---------------------------------------------------------------------------

def r_hat(chains: np.ndarray) -> np.ndarray:
    """Gelman-Rubin R̂ per parameter using ArviZ.

    Parameters
    ----------
    chains
        Shape ``(n_chains, n_draws, n_params)``.

    Returns
    -------
    ``np.ndarray`` of shape ``(n_params,)``.
    """
    import arviz as az
    # ArviZ expects dict of arrays with shape (n_chains, n_draws).
    n_chains, n_draws, n_params = chains.shape
    data = {f"p{i}": chains[:, :, i] for i in range(n_params)}
    idata = az.from_dict(posterior=data)
    rhat_vals = az.rhat(idata)
    return np.array([float(rhat_vals[f"p{i}"].values) for i in range(n_params)])


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Bulk ESS per parameter using ArviZ.

    Parameters
    ----------
    chains
        Shape ``(n_chains, n_draws, n_params)``.

    Returns
    -------
    ``np.ndarray`` of shape ``(n_params,)``.
    """
    import arviz as az
    n_chains, n_draws, n_params = chains.shape
    data = {f"p{i}": chains[:, :, i] for i in range(n_params)}
    idata = az.from_dict(posterior=data)
    ess_vals = az.ess(idata, method="bulk")
    return np.array([float(ess_vals[f"p{i}"].values) for i in range(n_params)])


# ---------------------------------------------------------------------------
# Full-parameter-vector convenience wrapper (used by notebooks 05 and 06)
# ---------------------------------------------------------------------------

PARAM_NAMES_2D = ("alpha", "beta")
PARAM_NAMES_4D = ("UA", "k0", "alpha", "beta")  # ODE internal; kept for compatibility


def summarise_posterior(
    samples: np.ndarray,
    true_theta: np.ndarray,
    param_names: tuple[str, ...] = PARAM_NAMES_2D,
    *,
    levels: tuple[float, ...] = (0.5, 0.9, 0.95),
) -> dict:
    """Per-parameter metrics for a single observation.

    Parameters
    ----------
    samples
        Shape ``(n_samples, n_params)``.
    true_theta
        Shape ``(n_params,)``.

    Returns
    -------
    Nested dict: ``{param_name: {metric_name: value}}``.
    """
    out: dict = {}
    for i, name in enumerate(param_names):
        out[name] = compute_all_continuous_metrics(
            samples[:, i], float(true_theta[i]), levels=levels,
        )
    return out


# ---------------------------------------------------------------------------
# Fault classification from 2-D [alpha, beta] posterior samples (M6c)
# ---------------------------------------------------------------------------

FAULT_CLASSES = ("healthy", "fouling_dominant", "decay_dominant", "combined")


def classify_fault(
    samples: np.ndarray,
    *,
    alpha_threshold: float = 0.85,
    beta_threshold: float = 0.85,
) -> dict:
    """Classify the active fault from 2-D ``[alpha, beta]`` posterior samples.

    Uses posterior mass in each quadrant of the (alpha, beta) unit square
    to assign probabilistic fault labels — no supervised labels needed.

    Fault classes:

    * ``healthy``          — alpha ≥ threshold AND beta ≥ threshold
    * ``fouling_dominant`` — beta  < threshold AND alpha ≥ threshold
    * ``decay_dominant``   — alpha < threshold AND beta  ≥ threshold
    * ``combined``         — alpha < threshold AND beta  < threshold

    Parameters
    ----------
    samples
        Shape ``(n_samples, 2)`` array of ``[alpha, beta]`` posterior samples.
    alpha_threshold, beta_threshold
        Boundary between healthy and degraded for each parameter.
        Default **0.85** — calibrated against the M5 finding that the
        closed-loop posterior mean for β sits ~0.10–0.15 below the true value
        due to the UA–β compensation effect.  A threshold of 0.95 (prior edge)
        causes most Sc2 replicates (β_true=0.70, β_post_mean≈0.54) to be
        classified as ``healthy`` because the posterior straddles 0.95.
        With threshold 0.85 the fault boundary sits within the posterior body
        for moderate faults (β_true ≥ 0.70).

    Returns
    -------
    dict with keys:
        ``"class"``       — name of the most probable fault class
        ``"probs"``       — dict mapping each class name to its posterior probability
        ``"alpha_mean"``  — posterior mean of alpha
        ``"beta_mean"``   — posterior mean of beta
    """
    samples = np.asarray(samples)
    alpha, beta = samples[:, 0], samples[:, 1]
    n = len(alpha)

    probs = {
        "healthy":          float(np.mean((alpha >= alpha_threshold) & (beta >= beta_threshold))),
        "fouling_dominant": float(np.mean((alpha >= alpha_threshold) & (beta <  beta_threshold))),
        "decay_dominant":   float(np.mean((alpha <  alpha_threshold) & (beta >= beta_threshold))),
        "combined":         float(np.mean((alpha <  alpha_threshold) & (beta <  beta_threshold))),
    }
    predicted_class = max(probs, key=probs.__getitem__)
    return {
        "class":      predicted_class,
        "probs":      probs,
        "alpha_mean": float(np.mean(alpha)),
        "beta_mean":  float(np.mean(beta)),
    }


def compute_classification_metrics(
    predicted_classes: list[str],
    true_classes: list[str],
) -> dict:
    """Confusion matrix and per-class F1 for the 4-class fault taxonomy.

    Parameters
    ----------
    predicted_classes
        List of predicted fault class names (from ``classify_fault``).
    true_classes
        List of ground-truth fault class names.

    Returns
    -------
    dict with ``"confusion_matrix"`` (4×4 numpy array, rows=true, cols=predicted),
    ``"per_class_f1"`` (dict), ``"macro_f1"`` (float), ``"accuracy"`` (float).
    """
    classes = list(FAULT_CLASSES)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    c2i = {c: i for i, c in enumerate(classes)}

    for true, pred in zip(true_classes, predicted_classes):
        cm[c2i[true], c2i[pred]] += 1

    f1_per_class: dict = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class[cls] = float(f1)

    macro_f1 = float(np.mean(list(f1_per_class.values())))
    accuracy = float(np.diag(cm).sum() / cm.sum()) if cm.sum() > 0 else 0.0

    return {
        "confusion_matrix": cm,
        "class_names":      classes,
        "per_class_f1":     f1_per_class,
        "macro_f1":         macro_f1,
        "accuracy":         accuracy,
    }
