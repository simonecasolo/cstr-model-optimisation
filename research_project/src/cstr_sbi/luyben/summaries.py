"""65-D summary statistics for the Luyben recycle plant.

8 observable channels: [T_r, Tc, Qc, T_s, Q_s, F_R, F_P, F_prod]
2-hour windows at 1 min resolution -> (120, 8) input.

Feature groups (65-D total):
    Per-channel base (8 × 5 = 40): mean, std, slope, min, max
    Final-window means (8):        last-25% mean per channel
    Control aggregates (6):        integrated errors + saturation fractions
    Physics-informed (11):         physics-derived cross-channel features
"""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp

from cstr_sbi.luyben.physics import (
    TSP_R, TSP_S, NSP_L,
    QC_MIN_R, QC_MAX_R, QS_MIN, QS_MAX, FR_MIN, FR_MAX, FP_MIN, FP_MAX,
    F_R_NOM, F_P_NOM, N_L_NOM,
)


# ---------------------------------------------------------------------------
# Feature inventory
# ---------------------------------------------------------------------------

CHANNEL_NAMES = ("T_r", "Tc", "Qc", "T_s", "Q_s", "F_R", "F_P", "F_prod")
BASE_STATS = ("mean", "std", "slope", "min", "max")
FINAL_WINDOW_FRACTION: float = 0.25

_PER_CHANNEL = tuple(f"{ch}_{stat}" for ch in CHANNEL_NAMES for stat in BASE_STATS)
_FINAL_WINDOW = tuple(f"{ch}_final_mean" for ch in CHANNEL_NAMES)
_AGGREGATES = (
    "int_abs_Tr_err",   # integral |T_r - TSP_R|
    "int_abs_Ts_err",   # integral |T_s - TSP_S|
    "Qc_sat_low_frac",  # fraction of time Qc at lower limit
    "Qc_sat_high_frac", # fraction of time Qc at upper limit
    "F_R_std",          # recycle flow variability (snowball indicator)
    "F_P_std",          # purge flow variability
)
_PHYSICS = (
    "UA_r_eff_proxy",    # (T_r - Tc) / max(Qc, eps)  ~ 1/(beta_r * UA_r)
    "UA_s_eff_proxy",    # (T_s - T_s_ref) / max(Q_s, eps)  ~ 1/(beta_s * UA_s)
    "recycle_load",      # F_R / F_R_nom  (snowball indicator; encodes alpha, eta_sep)
    "purge_deviation",   # F_P / F_P_nom  (encodes xi)
    "pump_head_proxy",   # F_R / F_R_nom  - same as recycle_load but kept separate
    "conversion_proxy",  # F_prod / (F_prod + F_R)  (encodes alpha, eta_sep)
    "recycle_richness",  # T_s / T_r  (encodes eta_sep via vapour composition effect)
    "feed_preheat_proxy",# T_r - T_r_mean_last_window  (encodes kappa)
    "corr_Qc_FR",        # corr(Qc, F_R)  (snowball coupling signal)
    "corr_Qs_Ts",        # corr(Q_s, T_s)  (separator loop coupling)
    "corr_FR_FP",        # corr(F_R, F_P)  (recycle-purge coupling)
)

FEATURE_NAMES = _PER_CHANNEL + _FINAL_WINDOW + _AGGREGATES + _PHYSICS
N_FEATURES: int = len(FEATURE_NAMES)
assert N_FEATURES == 65, f"Expected 65 features, got {N_FEATURES}"

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "all": FEATURE_NAMES,
    "per_channel": _PER_CHANNEL,
    "final_window": _FINAL_WINDOW,
    "aggregates": _AGGREGATES,
    "physics": _PHYSICS,
    "compact": (
        tuple(f"{ch}_mean" for ch in CHANNEL_NAMES)
        + _FINAL_WINDOW
        + ("F_R_std", "F_P_std")
        + _PHYSICS
    ),
}

T_S_REF: float = 290.0  # separator coolant reference temperature


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_slope(t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    t_mean = jnp.mean(t)
    x_mean = jnp.mean(x)
    dt = t - t_mean
    denom = jnp.sum(dt * dt)
    slope = jnp.sum(dt * (x - x_mean)) / jnp.where(denom > 0, denom, 1.0)
    return jnp.where(denom > 0, slope, jnp.nan)


def _safe_corr(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Pearson correlation, returns 0.0 if either series is constant."""
    x_c = x - jnp.mean(x)
    y_c = y - jnp.mean(y)
    num = jnp.sum(x_c * y_c)
    denom = jnp.sqrt(jnp.sum(x_c ** 2) * jnp.sum(y_c ** 2))
    return jnp.where(denom > 1e-10, num / denom, 0.0)


# ---------------------------------------------------------------------------
# Main summary function
# ---------------------------------------------------------------------------

def compute_summary_statistics(
    obs: jnp.ndarray,
    t: jnp.ndarray | None = None,
    *,
    tsp_r: float = TSP_R,
    tsp_s: float = TSP_S,
    qc_min: float = QC_MIN_R,
    qc_max: float = QC_MAX_R,
    sat_tol: float = 1.0,
    final_window_fraction: float = FINAL_WINDOW_FRACTION,
    f_r_nom: float = F_R_NOM,
    f_p_nom: float = F_P_NOM,
) -> jnp.ndarray:
    """Compute the 65-D summary statistics for a single (120, 8) observation window.

    Parameters
    ----------
    obs
        ``(n_t, 8)`` array: [T_r, Tc, Qc, T_s, Q_s, F_R, F_P, F_prod]
    t
        Optional ``(n_t,)`` time grid in minutes.

    Returns
    -------
    ``jnp.ndarray`` of shape ``(65,)``, ordered as in ``FEATURE_NAMES``.
    """
    n_t = obs.shape[0]
    if t is None:
        t = jnp.arange(n_t, dtype=obs.dtype)

    # Per-channel base stats (8 × 5 = 40)
    means  = jnp.mean(obs, axis=0)
    stds   = jnp.std(obs, axis=0)
    mins   = jnp.min(obs, axis=0)
    maxs   = jnp.max(obs, axis=0)
    slopes = jax.vmap(lambda col: _safe_slope(t, col), in_axes=1)(obs)
    per_channel = jnp.stack([means, stds, slopes, mins, maxs], axis=1).reshape(-1)

    # Final-window means (8)
    n_final = jnp.maximum(jnp.int32(jnp.ceil(final_window_fraction * n_t)), 1)
    idx = jnp.arange(n_t)
    mask = idx >= (n_t - n_final)
    final_means = jnp.sum(obs * mask[:, None], axis=0) / jnp.sum(mask)

    # Control aggregates (6)
    T_r_col = obs[:, 0]
    T_s_col = obs[:, 3]
    Qc_col  = obs[:, 2]
    F_R_col = obs[:, 5]
    F_P_col = obs[:, 6]

    dt_grid = t[1] - t[0] if n_t > 1 else jnp.asarray(1.0)
    int_abs_Tr_err = jnp.sum(jnp.abs(T_r_col - tsp_r)) * dt_grid
    int_abs_Ts_err = jnp.sum(jnp.abs(T_s_col - tsp_s)) * dt_grid
    qc_sat_low  = jnp.mean((Qc_col <= qc_min + sat_tol).astype(obs.dtype))
    qc_sat_high = jnp.mean((Qc_col >= qc_max - sat_tol).astype(obs.dtype))
    f_r_std = jnp.std(F_R_col)
    f_p_std = jnp.std(F_P_col)

    # Physics-informed features (11)
    _eps = jnp.asarray(1e-6, dtype=obs.dtype)
    T_r_mean  = means[0]
    Tc_mean   = means[1]
    Qc_mean   = means[2]
    T_s_mean  = means[3]
    Q_s_mean  = means[4]
    F_R_mean  = means[5]
    F_P_mean  = means[6]
    F_prod_mean = means[7]

    UA_r_eff_proxy  = (T_r_mean - Tc_mean)  / jnp.maximum(jnp.abs(Qc_mean),  _eps)
    UA_s_eff_proxy  = (T_s_mean - T_S_REF)  / jnp.maximum(jnp.abs(Q_s_mean), _eps)
    recycle_load    = F_R_mean / jnp.maximum(f_r_nom, _eps)
    purge_deviation = F_P_mean / jnp.maximum(f_p_nom, _eps)
    pump_head_proxy = F_R_mean / jnp.maximum(f_r_nom, _eps)
    conversion_proxy = F_prod_mean / jnp.maximum(F_prod_mean + F_R_mean, _eps)
    recycle_richness = T_s_mean / jnp.maximum(T_r_mean, _eps)
    feed_preheat_proxy = T_r_mean - final_means[0]  # temp deviation from end of window

    corr_Qc_FR = _safe_corr(Qc_col,  F_R_col)
    corr_Qs_Ts = _safe_corr(obs[:, 4], T_s_col)
    corr_FR_FP = _safe_corr(F_R_col,  F_P_col)

    return jnp.concatenate([
        per_channel,
        final_means,
        jnp.array([int_abs_Tr_err, int_abs_Ts_err,
                   qc_sat_low, qc_sat_high, f_r_std, f_p_std]),
        jnp.array([UA_r_eff_proxy, UA_s_eff_proxy, recycle_load, purge_deviation,
                   pump_head_proxy, conversion_proxy, recycle_richness,
                   feed_preheat_proxy, corr_Qc_FR, corr_Qs_Ts, corr_FR_FP]),
    ])


# ---------------------------------------------------------------------------
# Batched variant
# ---------------------------------------------------------------------------

def compute_summary_statistics_batch(
    obs: jnp.ndarray,
    t: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Vectorised summary over batch axis.

    ``obs`` has shape ``(n_batch, n_t, 8)``; output has shape ``(n_batch, 65)``.
    """
    fn = lambda x: compute_summary_statistics(x, t)  # noqa: E731
    return jax.jit(jax.vmap(fn))(obs)


def list_features() -> list[str]:
    return list(FEATURE_NAMES)


def feature_indices(names: Iterable[str]) -> jnp.ndarray:
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    return jnp.array([name_to_idx[n] for n in names], dtype=jnp.int32)
