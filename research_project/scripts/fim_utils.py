"""Shared Fisher-information-matrix (FIM) utilities.

Generalises the FIM = J^T Sigma^-1 J methodology introduced in
`notebooks/23_wu2003_summary_statistics.ipynb` Section 7 and refined in
`notebooks/29b_identifiability_scan.ipynb` Section 4 / `notebooks/31_wu2003_fault_classification.ipynb`
Section 6b. Those two notebooks each carried an independent copy of this logic hardcoded to
Wu 2003's 5-parameter theta (`J = np.zeros((n_feat, 5))`, a 5-entry `EPS_FIM`); this module
makes the parameter count implicit in `theta_point`/`eps` so the same code path can be reused
for System I's 2-parameter (alpha, beta) and for cross-system validation
(`notebooks/33_fim_cross_system_validation.ipynb`).

`compute_fim` returns the full FIM matrix (not just a single reported pair) so that a
downstream Cramer-Rao-bound report (`crb_report`) and an off-diagonal/ratio report
(`offdiag_ratio`) are guaranteed to be computed from the *same* noise draws, rather than by
independently recomputing the FIM with different random seeds.
"""

import numpy as np


def compute_fim(feat_fn, theta_point, eps, n_reps_sigma=60, seed_offset=0):
    """FIM = J^T Sigma^-1 J.

    `feat_fn(theta, seed)` maps a parameter vector to a feature vector (whole-window
    summary statistics, raw trajectory, or any other representation) with a given noise
    seed. `Sigma` is the diagonal feature-noise covariance, estimated from the variance of
    `n_reps_sigma` independent noisy replicates at `theta_point` (i.e. from real
    replicate/sensor noise, not an assumed analytic form). `J` is the Jacobian of the
    (noise-free-in-expectation) feature map with respect to theta, via central finite
    differences with per-parameter step sizes `eps` (`len(eps) == len(theta_point)`).

    Returns the full `(n_params, n_params)` FIM matrix.
    """
    theta_point = np.asarray(theta_point, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    n_params = len(theta_point)
    assert len(eps) == n_params, "eps must have one entry per parameter in theta_point"

    reps = np.stack([feat_fn(theta_point, seed=seed_offset + 2000 + i) for i in range(n_reps_sigma)])
    Sigma_diag = np.var(reps, axis=0) + 1e-12
    n_feat = reps.shape[1]

    J = np.zeros((n_feat, n_params))
    for j, e in enumerate(eps):
        th_p = theta_point.copy(); th_p[j] += e
        th_m = theta_point.copy(); th_m[j] -= e
        s_p = feat_fn(th_p, seed=seed_offset + j * 10 + 1)
        s_m = feat_fn(th_m, seed=seed_offset + j * 10 + 2)
        J[:, j] = (s_p - s_m) / (2 * e)

    return J.T @ (J / Sigma_diag[:, None])


def offdiag_ratio(FIM, idx_a, idx_b):
    """Normalised off-diagonal entry and diagonal ratio, from the RAW information matrix
    (from a full FIM matrix, e.g. as returned by `compute_fim`).

    Returns `(FIM_norm[idx_a, idx_b], FIM[idx_a, idx_a] / FIM[idx_b, idx_b])` — identical
    convention to the original `compute_fim_offdiag`/`compute_fim_offdiag_pair` functions in
    nb29b/nb31, so migrating those notebooks to call this reproduces the same numbers.

    **Warning (found 2026-09-01, notebooks 39/40):** the first return value is a normalised
    entry of the raw *information* matrix, not a posterior *correlation* coefficient. For a
    multi-parameter FIM these can and do have OPPOSITE SIGNS once the other parameters are
    properly accounted for (confirmed numerically: raw off-diagonal `+0.40` vs. the actual
    inverted-covariance correlation `-0.60` at one Wu2003 operating point, and the reverse at
    another). This function is still useful for *representation-collapse* checks where you
    only care whether the entry goes to ~0 (an artifact-vs-real test, as in `nb29b`), but do
    **not** compare its sign or magnitude directly against an actual trained posterior's
    empirical correlation -- use `cov_offdiag_corr` for that.
    """
    D = np.sqrt(np.diag(FIM))
    D_safe = np.where(D > 0, D, 1.0)
    FIM_norm = FIM / np.outer(D_safe, D_safe)
    ratio = FIM[idx_a, idx_a] / max(FIM[idx_b, idx_b], 1e-30)
    return FIM_norm[idx_a, idx_b], ratio


def cov_offdiag_corr(FIM, idx_a, idx_b, ridge=1e-10):
    """Marginal posterior correlation between parameters `idx_a`/`idx_b` implied by the
    Laplace/Gaussian approximation at the point `FIM` was computed at -- i.e. from the
    INVERTED FIM (posterior covariance), not the raw information matrix.

    This is the quantity that is actually comparable to an SNPE posterior's empirical
    sample correlation (`np.corrcoef`). `offdiag_ratio`'s raw, normalised information-matrix
    entry is a different quantity and can have the opposite sign -- see its docstring.
    Whenever cross-checking a FIM-based identifiability claim against a real trained
    posterior (or reporting a "does this coupling survive representation X" result), use
    this function.
    """
    n = FIM.shape[0]
    cov = np.linalg.inv(FIM + ridge * np.eye(n))
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    return float(cov[idx_a, idx_b] / (d[idx_a] * d[idx_b]))


def crb_report(FIM, ridge=1e-10, names=None):
    """Cramer-Rao bound report: inverts the FIM (with a small ridge for numerical safety
    against near-singular matrices) and returns per-parameter standard deviations and the
    FIM's condition number.

    This is the quantity a normalised off-diagonal alone cannot show: whether a
    representation change that shrinks the *correlation coefficient* between two parameters
    also shrinks their *achievable estimator variance*, or merely redistributes correlation
    across a higher-dimensional feature space without tightening the bound at all.

    Returns a dict with keys `cov` (the full inverse-FIM/covariance matrix), `sd`
    (per-parameter Cramer-Rao standard deviations), and `cond` (condition number of the raw
    FIM, a diagnostic for how close the matrix is to singular before regularisation).
    """
    n = FIM.shape[0]
    cov = np.linalg.inv(FIM + ridge * np.eye(n))
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    cond = np.linalg.cond(FIM)
    out = {"cov": cov, "sd": sd, "cond": cond}
    if names is not None:
        out["sd_by_name"] = dict(zip(names, sd))
    return out
