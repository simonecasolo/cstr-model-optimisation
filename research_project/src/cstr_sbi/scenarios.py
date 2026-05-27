"""Scenario data generators for the eight scenarios in research_spec Section 4.

This module is the single source of truth for *what* each scenario means.
Notebook 02 (M2 data-generation) and the M6 batch generators consume the
``SCENARIO_CONFIGS`` table below to produce labelled observation datasets.

Each scenario carries:

* ``alpha``, ``beta`` -- ground-truth catalyst-decay and jacket-fouling
  factors (the inference targets in the M4 SBI study).
* ``mode``            -- ``"closed_loop"`` for Sc 1-5 and 7; ``"open_loop"``
  for Sc 0 and 6 (where ``Qc`` is held fixed at ``Qc0`` and the PI controller
  is bypassed).
* ``drift_T``         -- additive sensor-drift offset on the T channel
  (only Sc 7).
* ``description``     -- one-liner summary used in plots and the saved CSV.

Scenario truths follow the Fogler-grounded nominal operating point of
``cstr_parameters_recommended.md`` (and ``cstr_sbi.physics``):
``[UA = 1.25e4, k0 = 1.696e13]`` with ``alpha`` and ``beta`` perturbed away
from 1 in the faulty cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from cstr_sbi.physics import (
    K0_NOMINAL,
    QC0,
    UA_NOMINAL,
)


@dataclass(frozen=True)
class ScenarioConfig:
    """One row of the scenario truth table."""

    id: int
    name: str
    alpha: float
    beta: float
    mode: str           # "closed_loop" | "open_loop"
    drift_T: float      # K, additive sensor offset on T (post-simulation)
    description: str

    def params(self) -> jnp.ndarray:
        """Return the 4-D parameter vector ``[UA, k0, alpha, beta]`` (Fogler nominal)."""
        return jnp.array([UA_NOMINAL, K0_NOMINAL, self.alpha, self.beta])

    def open_loop_params(self) -> jnp.ndarray:
        """Return the 2-D open-loop parameter vector ``[UA, k0]``."""
        # Sc 0 / Sc 6 still treat alpha and beta as ground truth, but the
        # open-loop simulator uses only [UA, k0]. Fault is encoded by scaling
        # those values directly when the open-loop simulator is invoked.
        return jnp.array([self.beta * UA_NOMINAL, self.alpha * K0_NOMINAL])


# ---------------------------------------------------------------------------
# The eight scenarios (research_spec Section 4)
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS: dict[str, ScenarioConfig] = {
    "Sc0_open_healthy": ScenarioConfig(
        id=0, name="Sc0_open_healthy",
        alpha=1.0, beta=1.0, mode="open_loop", drift_T=0.0,
        description="Healthy reactor, open-loop (Qc fixed). Validation against spec Section 8.",
    ),
    "Sc1_closed_healthy": ScenarioConfig(
        id=1, name="Sc1_closed_healthy",
        alpha=1.0, beta=1.0, mode="closed_loop", drift_T=0.0,
        description="Healthy reactor, closed-loop with PI control. Baseline.",
    ),
    "Sc2_closed_fouling": ScenarioConfig(
        id=2, name="Sc2_closed_fouling",
        alpha=1.0, beta=0.7, mode="closed_loop", drift_T=0.0,
        description="Jacket fouling (beta=0.7) -- primary fault scenario.",
    ),
    "Sc3_closed_decay": ScenarioConfig(
        id=3, name="Sc3_closed_decay",
        alpha=0.7, beta=1.0, mode="closed_loop", drift_T=0.0,
        description="Catalyst decay (alpha=0.7).",
    ),
    "Sc4_closed_combined": ScenarioConfig(
        id=4, name="Sc4_closed_combined",
        alpha=0.85, beta=0.85, mode="closed_loop", drift_T=0.0,
        description="Combined moderate fouling and decay.",
    ),
    "Sc5_closed_saturated": ScenarioConfig(
        id=5, name="Sc5_closed_saturated",
        alpha=1.0, beta=0.4, mode="closed_loop", drift_T=0.0,
        description="Severe fouling -- controller saturates at Qc_max.",
    ),
    "Sc6_open_with_fault": ScenarioConfig(
        id=6, name="Sc6_open_with_fault",
        alpha=1.0, beta=0.7, mode="open_loop", drift_T=0.0,
        description=(
            "Open-loop simulator applied to a fault -- the failure baseline "
            "(SBI trained on this is what the paper refutes)."
        ),
    ),
    "Sc7_closed_drift": ScenarioConfig(
        id=7, name="Sc7_closed_drift",
        alpha=1.0, beta=0.85, mode="closed_loop", drift_T=2.0,
        description=(
            "Mild fouling plus +2 K sensor drift on T -- substudy of "
            "drift-vs-fouling identifiability."
        ),
    ),
}


def list_configs() -> list[ScenarioConfig]:
    """Return scenarios in numerical order."""
    return sorted(SCENARIO_CONFIGS.values(), key=lambda s: s.id)


# ---------------------------------------------------------------------------
# Inlet perturbation (used by M6's 30-day stream and as a per-replicate hook
# in M2's data generation).  See research_spec Section 3.5.
# ---------------------------------------------------------------------------

def generate_degradation_stream(
    *,
    Tcrit: float = 43200.0,
    dt_window: float = 60.0,
    n_replicates_per_window: int = 1,
    seed: int = 0,
    inlet: jnp.ndarray | None = None,
    ctrl: jnp.ndarray | None = None,
    alpha_threshold: float = 0.85,
    beta_threshold: float = 0.85,
) -> list[dict]:
    """Generate a 30-day degradation trajectory sliced into 60-min windows.

    Implements Scenario 8 (sequential degradation tracking) from the spec.
    Both α(t) and β(t) decay linearly from 1.0 to 0.9 over ``Tcrit`` minutes
    (10% degradation per 30 days, linearised Kern-Seaton / acid-neutralisation).

    Parameters
    ----------
    Tcrit
        Critical time (minutes) at which degradation reaches 10%.
        Default: 43 200 min = 30 days.
    dt_window
        Window size in minutes (default 60).
    n_replicates_per_window
        Number of independent noise realisations per window (default 1;
        use > 1 for uncertainty-in-data studies).
    seed
        Base JAX random seed.
    inlet, ctrl
        Override nominal inlet / controller arrays if needed.

    Returns
    -------
    List of dicts, one per window, each containing:

    * ``"t_start"``   — window start time in minutes
    * ``"alpha_true"`` — ground-truth α at this window
    * ``"beta_true"``  — ground-truth β at this window
    * ``"fault_class"`` — one of "healthy", "fouling_dominant", "decay_dominant", "combined"
    * ``"obs"``        — ``(n_replicates_per_window, n_t, 4)`` array of noisy observations
    * ``"t"``          — ``(n_t,)`` time grid in minutes within the window

    Usage::

        stream = generate_degradation_stream()
        for w in stream:
            x_summary = compute_summary_statistics(w["obs"][0], w["t"])
            samples = posterior.sample(5000, x=x_summary)
    """
    import jax
    from cstr_sbi.physics import NOMINAL_CTRL, NOMINAL_INLET_CL, K0_NOMINAL, UA_NOMINAL
    from cstr_sbi.simulator import generate_replicates, warm_start_ic

    if inlet is None:
        inlet = NOMINAL_INLET_CL
    if ctrl is None:
        ctrl = NOMINAL_CTRL

    n_windows = int(Tcrit / dt_window)
    master_key = jax.random.PRNGKey(seed)

    stream = []
    for win_idx in range(n_windows):
        t_start = win_idx * dt_window
        # Linear degradation model.
        alpha_true = float(1.0 - 0.1 * t_start / Tcrit)
        beta_true  = float(1.0 - 0.1 * t_start / Tcrit)
        alpha_true = max(alpha_true, 0.4)
        beta_true  = max(beta_true,  0.4)

        params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha_true, beta_true])

        y0 = warm_start_ic(params, inlet, ctrl)
        win_key = jax.random.fold_in(master_key, win_idx)
        t_out, obs = generate_replicates(
            params, inlet, ctrl, y0,
            n_replicates=n_replicates_per_window,
            master_key=win_key,
            t_window=dt_window,
        )

        # Classification thresholds — default 0.85 aligned with metrics.classify_fault.
        alpha_thresh = alpha_threshold
        beta_thresh  = beta_threshold
        if alpha_true >= alpha_thresh and beta_true >= beta_thresh:
            fault_class = "healthy"
        elif alpha_true >= alpha_thresh and beta_true < beta_thresh:
            fault_class = "fouling_dominant"
        elif alpha_true < alpha_thresh and beta_true >= beta_thresh:
            fault_class = "decay_dominant"
        else:
            fault_class = "combined"

        stream.append({
            "t_start":    t_start,
            "alpha_true": alpha_true,
            "beta_true":  beta_true,
            "fault_class": fault_class,
            "obs":        jnp.asarray(obs),
            "t":          jnp.asarray(t_out),
        })

    return stream


def perturb_inlet(
    base_inlet: jnp.ndarray,
    key,
    *,
    T_amp: float = 2.0,
    Ci_min: float = 0.9,
    Ci_max: float = 1.0,
) -> jnp.ndarray:
    """Sample a perturbed inlet around ``base_inlet = [Ci, Ti, Tci]``.

    ``Ti`` and ``Tci`` get a uniform ``+/- T_amp`` perturbation;
    ``Ci`` is drawn uniformly from ``[Ci_min, Ci_max]``.
    """
    import jax
    import jax.numpy as jnp

    k1, k2, k3 = jax.random.split(key, 3)
    dTi = jax.random.uniform(k1, (), minval=-T_amp, maxval=T_amp)
    dTci = jax.random.uniform(k2, (), minval=-T_amp, maxval=T_amp)
    Ci = jax.random.uniform(k3, (), minval=Ci_min, maxval=Ci_max)
    return jnp.array([Ci, base_inlet[1] + dTi, base_inlet[2] + dTci])
