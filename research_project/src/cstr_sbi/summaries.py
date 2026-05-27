"""Summary statistics for SBI / MCMC observations -- M3 deliverable, revised M5-2026-05-18.

Each observation is a window of length ``n_t`` over the four observable
channels ``[C, T, Tc, Qc]`` (see ``cstr_sbi.simulator``). The summary
statistics compress that ``(n_t, 4)`` window into a low-dimensional feature
vector that is robust to sensor noise and to small differences in the
window length.

Design (see research_spec Section 3.2 and execution_plan M3/M5):

* **Per-channel base features (5 × 4 = 20):** for each of the four channels
  ``[C, T, Tc, Qc]``, compute ``mean``, ``std``, ``slope`` (least-squares
  linear fit vs time), ``min``, ``max``.

* **Final-window steady-state means (4):** mean over the last 25 % of the
  window for each channel.

* **Control / process aggregates (3):** time-integrated ``|T - Tsp|``
  (degree-minutes off-setpoint), fraction of time ``Qc`` saturates at
  the lower bound, fraction of time it saturates at the upper bound.

* **Physics-informed cross-channel features (2) — added M5-2026-05-18:**
  M5 MCMC analysis revealed that (UA, β) and (k0, α) are structurally
  non-identifiable individually: the ODE only constrains the products
  ``β·UA`` and ``α·k0``. Two features that directly encode those products
  from observable trajectories are added:

  - ``UA_eff_proxy``: ``(T_mean − Tc_mean) / max(Qc_mean, ε)``.
    From the jacket steady-state energy balance
    ``β·UA·(T−Tc) ≈ ρc·Cpc·Qc·ΔTc``, this ratio is proportional to
    ``1/(β·UA)``, the dominant fault signal in jacket-fouling scenarios.

  - ``k0_eff_proxy``: ``log(C_mean / max(Ci_nominal − C_mean, ε))``.
    From the CSTR component balance at pseudo-steady state,
    ``C / (Ci − C) ∝ 1 / (α·k0_eff)``, so the log-ratio linearises
    the dependence on the effective rate constant.

Total: **29 features**.

Features are returned as a flat ``jnp.ndarray`` of length ``N_FEATURES``;
``FEATURE_NAMES`` and ``FEATURE_GROUPS`` provide names and ablation subsets.

All routines are pure-JAX, support ``jax.vmap`` over the batch axis, and
are NaN-tolerant (failed simulations propagate ``NaN`` rather than
corrupting the dataset silently).
"""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp

from cstr_sbi.physics import CI_NOMINAL, QC_MAX, QC_MIN, TSP


# ---------------------------------------------------------------------------
# Feature inventory
# ---------------------------------------------------------------------------

CHANNEL_NAMES: tuple[str, ...] = ("C", "T", "Tc", "Qc")
BASE_STATS: tuple[str, ...] = ("mean", "std", "slope", "min", "max")
FINAL_WINDOW_FRACTION: float = 0.25  # last 25 % of the window

_PER_CHANNEL = tuple(f"{ch}_{stat}" for ch in CHANNEL_NAMES for stat in BASE_STATS)
_FINAL_WINDOW = tuple(f"{ch}_final_mean" for ch in CHANNEL_NAMES)
_AGGREGATES = ("int_abs_T_err", "Qc_sat_low_frac", "Qc_sat_high_frac")
# Physics-informed features added M5-2026-05-18 (see module docstring).
_PHYSICS = ("UA_eff_proxy", "k0_eff_proxy")

FEATURE_NAMES: tuple[str, ...] = _PER_CHANNEL + _FINAL_WINDOW + _AGGREGATES + _PHYSICS
N_FEATURES: int = len(FEATURE_NAMES)
assert N_FEATURES == 29

#: Ablation subsets used by notebook 03 and downstream sensitivity work.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "all": FEATURE_NAMES,
    "per_channel": _PER_CHANNEL,
    "final_window": _FINAL_WINDOW,
    "aggregates": _AGGREGATES,
    "physics": _PHYSICS,
    # Compact set: mean and final-window mean per channel plus saturation + physics.
    "compact": tuple(f"{ch}_mean" for ch in CHANNEL_NAMES)
    + _FINAL_WINDOW
    + ("Qc_sat_low_frac", "Qc_sat_high_frac")
    + _PHYSICS,
    # Means only (the simplest possible baseline).
    "means_only": tuple(f"{ch}_mean" for ch in CHANNEL_NAMES),
}


def list_features() -> list[str]:
    """Canonical, ordered list of feature names."""
    return list(FEATURE_NAMES)


def feature_indices(names: Iterable[str]) -> jnp.ndarray:
    """Return the integer indices of ``names`` into the full feature vector."""
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    return jnp.array([name_to_idx[n] for n in names], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Single-observation summary
# ---------------------------------------------------------------------------

def _safe_linear_slope(t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Least-squares slope of ``x`` vs ``t``; NaN if variance(t) is zero."""
    t_mean = jnp.mean(t)
    x_mean = jnp.mean(x)
    dt = t - t_mean
    denom = jnp.sum(dt * dt)
    slope = jnp.sum(dt * (x - x_mean)) / jnp.where(denom > 0, denom, 1.0)
    return jnp.where(denom > 0, slope, jnp.nan)


def compute_summary_statistics(
    obs: jnp.ndarray,
    t: jnp.ndarray | None = None,
    *,
    tsp: float = TSP,
    qc_min: float = QC_MIN,
    qc_max: float = QC_MAX,
    sat_tol: float = 1.0,
    final_window_fraction: float = FINAL_WINDOW_FRACTION,
    ci_nominal: float = CI_NOMINAL,
) -> jnp.ndarray:
    """Compute the 29-D summary statistics for a single observation window.

    Parameters
    ----------
    obs
        ``(n_t, 4)`` array of ``[C, T, Tc, Qc]`` samples.
    t
        Optional ``(n_t,)`` time grid in minutes. If ``None``, a uniformly
        spaced grid is used (the slope feature is then in units of
        ``[channel] / sample``).
    tsp
        Temperature setpoint used for the ``int_abs_T_err`` feature.
    qc_min, qc_max
        Valve clamp limits used to detect controller saturation.
    sat_tol
        Numerical tolerance (L/min) for the saturation test.
    ci_nominal
        Nominal inlet concentration ``Ci`` used by the ``k0_eff_proxy``
        feature (default: ``CI_NOMINAL`` from ``physics.py``).

    Returns
    -------
    ``jnp.ndarray`` of shape ``(29,)``, ordered as in ``FEATURE_NAMES``.
    """
    n_t = obs.shape[0]
    if t is None:
        t = jnp.arange(n_t, dtype=obs.dtype)

    # Per-channel base stats (4 channels x 5 stats = 20)
    means = jnp.mean(obs, axis=0)
    stds = jnp.std(obs, axis=0)
    mins = jnp.min(obs, axis=0)
    maxs = jnp.max(obs, axis=0)
    slopes = jax.vmap(lambda col: _safe_linear_slope(t, col), in_axes=1)(obs)

    per_channel = jnp.stack([means, stds, slopes, mins, maxs], axis=1).reshape(-1)

    # Final-window steady-state means
    n_final = jnp.maximum(jnp.int32(jnp.ceil(final_window_fraction * n_t)), 1)
    # JAX-friendly: use a boolean mask rather than dynamic slicing
    idx = jnp.arange(n_t)
    mask = idx >= (n_t - n_final)
    final_means = jnp.sum(obs * mask[:, None], axis=0) / jnp.sum(mask)

    # Aggregates
    T_err = jnp.abs(obs[:, 1] - tsp)
    if n_t > 1:
        dt = t[1] - t[0]
    else:
        dt = jnp.asarray(1.0, dtype=obs.dtype)
    int_abs_T_err = jnp.sum(T_err) * dt

    qc = obs[:, 3]
    sat_low = jnp.mean((qc <= qc_min + sat_tol).astype(obs.dtype))
    sat_high = jnp.mean((qc >= qc_max - sat_tol).astype(obs.dtype))

    # Physics-informed features (M5 revision):
    # UA_eff_proxy ∝ 1/(β·UA): from jacket steady-state ΔT/Qc balance.
    T_mean  = means[1]
    Tc_mean = means[2]
    Qc_mean = means[3]
    C_mean  = means[0]
    _eps = jnp.asarray(1e-6, dtype=obs.dtype)
    UA_eff_proxy = (T_mean - Tc_mean) / jnp.maximum(jnp.abs(Qc_mean), _eps)
    # k0_eff_proxy ∝ 1/(α·k0): log-ratio from CSTR component balance.
    # Clipped to [-8, 2] to prevent z-scoring collapse during SBI training
    # when extreme prior draws push C_mean → Ci (denominator → 0).
    k0_eff_proxy = jnp.clip(
        jnp.log(
            jnp.maximum(C_mean, _eps) /
            jnp.maximum(ci_nominal - C_mean, _eps)
        ),
        a_min=-8.0, a_max=2.0,
    )

    return jnp.concatenate(
        [
            per_channel,
            final_means,
            jnp.array([int_abs_T_err, sat_low, sat_high]),
            jnp.array([UA_eff_proxy, k0_eff_proxy]),
        ]
    )


# ---------------------------------------------------------------------------
# Batched variant -- the workhorse for notebook 03 and M4 SBI training
# ---------------------------------------------------------------------------

_compute_summary_statistics_jit = jax.jit(
    compute_summary_statistics,
    static_argnames=("final_window_fraction",),
)


def compute_summary_statistics_batch(
    obs: jnp.ndarray,
    t: jnp.ndarray | None = None,
    *,
    tsp: float = TSP,
    qc_min: float = QC_MIN,
    qc_max: float = QC_MAX,
    sat_tol: float = 1.0,
    final_window_fraction: float = FINAL_WINDOW_FRACTION,
    ci_nominal: float = CI_NOMINAL,
) -> jnp.ndarray:
    """Vectorised summary computation over a batch axis.

    ``obs`` has shape ``(n_batch, n_t, 4)``; output has shape
    ``(n_batch, 29)``.
    """
    fn = lambda x: compute_summary_statistics(  # noqa: E731
        x, t,
        tsp=tsp, qc_min=qc_min, qc_max=qc_max,
        sat_tol=sat_tol, final_window_fraction=final_window_fraction,
        ci_nominal=ci_nominal,
    )
    return jax.jit(jax.vmap(fn))(obs)


# ---------------------------------------------------------------------------
# Convenience: subset selection by name
# ---------------------------------------------------------------------------

def select_features(
    summaries: jnp.ndarray,
    names: Iterable[str],
) -> jnp.ndarray:
    """Return the columns of ``summaries`` corresponding to ``names``.

    ``summaries`` is ``(..., N_FEATURES)``; output is ``(..., len(names))``.
    """
    idx = feature_indices(names)
    return jnp.take(summaries, idx, axis=-1)
