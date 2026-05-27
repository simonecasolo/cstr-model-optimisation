"""Stochastic CSTR simulator -- M2 deliverable.

Provides a JAX-vectorised Euler-Maruyama integrator for the closed-loop CSTR
plus a thin sensor layer that applies Gaussian measurement noise and additive
sensor drift to the simulated trajectory. Used by the data-generation notebook
(02), the SBI training-set builder (M4), and the scenario generators in
``scenarios.py``.

Conventions follow ``cstr_sbi.physics`` (4-state closed-loop, 3-state
open-loop).  Noise terms are interpreted as continuous-time SDE diffusion
coefficients with units ``[variable] / sqrt(min)``: a step of size ``dt``
adds a perturbation of std ``sigma * sqrt(dt)``.

The default noise levels in ``DEFAULT_PROCESS_SIGMA`` are scaled down from
the nominal spec values (research_spec.md Section 3.5) so that the
realised noise is visible but does not drown the fault signal at the
Fogler-grounded operating point. The literal spec amplitudes
(``0.1 mol/L/min`` on the C balance, ``10 K/min`` on the energy balances)
are too aggressive when applied as proper SDE diffusion -- accumulated
variance over a 60 min window would dominate the signal -- and almost
certainly refer to a discrete-time fixed-step interpretation in the
legacy code where the increment per step equals ``sigma * dt`` rather
than ``sigma * sqrt(dt)``. See the docstring of ``simulate_em_window``
for how to override.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from cstr_sbi.physics import (
    NOMINAL_CTRL,
    cstr_closed_loop_rhs,
    cstr_open_loop_rhs,
    compute_qc,
    simulate_closed_loop_to_steady_state,
)


# ---------------------------------------------------------------------------
# Defaults (see module docstring for the rationale on the noise levels)
# ---------------------------------------------------------------------------

# Process noise diffusion coefficients [C, T, Tc, I]; integrator state I has
# zero noise (the integrator is part of the controller, not the plant).
# sigma_T = 0.1 K/sqrt(min) gives a steady-state OU std of approx 0.5 K with
# the Fogler PI tuning -- visible but well below the fault signal magnitude.
DEFAULT_PROCESS_SIGMA = jnp.array([0.0005, 0.1, 0.1, 0.0])  # /sqrt(min)

# Sensor noise as a fraction of each channel's running maximum value
# (research_spec.md Section 3.5: "amplitude = 0.5% of the maximum value of
# each variable").
DEFAULT_SENSOR_NOISE_PCT = 0.005

# Sensor drift (additive offsets, applied post-simulation; spec Section 3.5).
DEFAULT_DRIFT_T = 0.0    # K
DEFAULT_DRIFT_CI = 0.0   # mol/L  (applied to inlet during simulation, not here)

# Default integrator settings.
DEFAULT_DT_INT = 0.01    # min, internal Euler-Maruyama step
DEFAULT_DT_OUT = 0.5     # min, observation-grid resolution


# ---------------------------------------------------------------------------
# Closed-loop Euler-Maruyama
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_steps", "stride"))
def _em_scan_closed_loop(y0, key, params, inlet, ctrl, dt, sigma, n_steps, stride):
    """Inner scan -- returns trajectory subsampled at every ``stride`` steps."""
    keys = jax.random.split(key, n_steps)
    sqrt_dt = jnp.sqrt(dt)

    def step(y, k):
        drift = cstr_closed_loop_rhs(0.0, y, (params, inlet, ctrl))
        xi = jax.random.normal(k, shape=y.shape)
        y_next = y + drift * dt + sigma * sqrt_dt * xi
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, keys)
    return ys[stride - 1::stride]


def simulate_em_window(
    params: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray,
    y0: jnp.ndarray,
    *,
    key: jax.Array,
    t_window: float = 60.0,
    dt: float = DEFAULT_DT_INT,
    dt_out: float = DEFAULT_DT_OUT,
    sigma: jnp.ndarray = DEFAULT_PROCESS_SIGMA,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Euler-Maruyama integration of the closed-loop CSTR over ``t_window``.

    Returns ``(t, ys, qc)``:
        t   -- shape ``(n_out,)``, output times in minutes (excluding 0).
        ys  -- shape ``(n_out, 4)``, state ``[C, T, Tc, I]``.
        qc  -- shape ``(n_out,)``, the realised coolant flow at each output time.

    ``params`` is the 4-D ``[UA, k0, alpha, beta]``; ``inlet`` is the 3-D
    closed-loop ``[Ci, Ti, Tci]``; ``ctrl`` is the 6-D PI controller vector.
    """
    n_steps = int(round(t_window / dt))
    stride = int(round(dt_out / dt))
    ys = _em_scan_closed_loop(y0, key, params, inlet, ctrl, dt, sigma, n_steps, stride)
    t_out = jnp.arange(1, ys.shape[0] + 1) * dt_out
    qc_out = jax.vmap(compute_qc, in_axes=(0, 0, None))(ys[:, 1], ys[:, 3], ctrl)
    return t_out, ys, qc_out


# ---------------------------------------------------------------------------
# Open-loop Euler-Maruyama (used for Scenario 0 and Scenario 6)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_steps", "stride"))
def _em_scan_open_loop(y0, key, params, inlet_ol, dt, sigma, n_steps, stride):
    """Open-loop EM scan -- ``inlet_ol`` is 4-D (includes fixed Qc)."""
    keys = jax.random.split(key, n_steps)
    sqrt_dt = jnp.sqrt(dt)
    sigma3 = sigma[:3]  # only C, T, Tc carry noise

    def step(y, k):
        drift = cstr_open_loop_rhs(0.0, y, (params, inlet_ol))
        xi = jax.random.normal(k, shape=y.shape)
        y_next = y + drift * dt + sigma3 * sqrt_dt * xi
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, keys)
    return ys[stride - 1::stride]


def simulate_em_window_open_loop(
    params: jnp.ndarray,
    inlet_ol: jnp.ndarray,
    y0: jnp.ndarray,
    *,
    key: jax.Array,
    t_window: float = 60.0,
    dt: float = DEFAULT_DT_INT,
    dt_out: float = DEFAULT_DT_OUT,
    sigma: jnp.ndarray = DEFAULT_PROCESS_SIGMA,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Open-loop EM integrator with fixed Qc.

    ``params = [UA, k0]`` (2-D, no degradation in M2's Sc 0/6 baseline);
    ``inlet_ol = [Ci, Ti, Tci, Qc]`` (4-D); state vector is ``[C, T, Tc]``.
    Returns ``(t, ys, qc_const)`` where ``qc_const`` is the constant input Qc
    repeated for shape parity with the closed-loop output.
    """
    n_steps = int(round(t_window / dt))
    stride = int(round(dt_out / dt))
    ys = _em_scan_open_loop(y0, key, params, inlet_ol, dt, sigma, n_steps, stride)
    t_out = jnp.arange(1, ys.shape[0] + 1) * dt_out
    qc_const = jnp.full((ys.shape[0],), inlet_ol[3])
    return t_out, ys, qc_const


# ---------------------------------------------------------------------------
# Sensor layer
# ---------------------------------------------------------------------------

def apply_sensor_layer(
    ys_obs: jnp.ndarray,
    *,
    key: jax.Array,
    noise_pct: float = DEFAULT_SENSOR_NOISE_PCT,
    drift_T: float = DEFAULT_DRIFT_T,
) -> jnp.ndarray:
    """Apply Gaussian sensor noise (fraction of channel max) and T-sensor drift.

    ``ys_obs`` is the ``(n_t, 4)`` array of observable channels
    ``[C, T, Tc, Qc]``. Returns the same shape.
    """
    max_per_channel = jnp.maximum(jnp.max(jnp.abs(ys_obs), axis=0, keepdims=True), 1e-12)
    sigma_obs = noise_pct * max_per_channel
    noise = jax.random.normal(key, shape=ys_obs.shape) * sigma_obs
    obs = ys_obs + noise
    obs = obs.at[:, 1].add(drift_T)
    return obs


# ---------------------------------------------------------------------------
# Replicate generator (vmap over seeds)
# ---------------------------------------------------------------------------

def _stack_obs(t, ys, qc):
    """Pack ``(t, ys, qc)`` into a ``(n_t, 4)`` ``[C, T, Tc, Qc]`` array."""
    return jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)


@partial(jax.jit, static_argnames=("n_steps_int", "stride", "n_replicates"))
def _generate_replicates_closed_loop(
    params, inlet, ctrl, y0, master_key, dt, sigma_proc, sigma_noise_pct,
    drift_T, n_steps_int, stride, n_replicates,
):
    """JIT-compiled batched generator for ``n_replicates`` closed-loop windows."""
    keys_proc = jax.random.split(master_key, 2 * n_replicates)
    proc_keys = keys_proc[:n_replicates]
    sens_keys = keys_proc[n_replicates:]

    def one_replicate(proc_key, sens_key):
        ys = _em_scan_closed_loop(
            y0, proc_key, params, inlet, ctrl, dt, sigma_proc,
            n_steps_int, stride,
        )
        qc = jax.vmap(compute_qc, in_axes=(0, 0, None))(ys[:, 1], ys[:, 3], ctrl)
        ys_obs = _stack_obs(None, ys, qc)
        return apply_sensor_layer(
            ys_obs, key=sens_key,
            noise_pct=sigma_noise_pct, drift_T=drift_T,
        )

    return jax.vmap(one_replicate)(proc_keys, sens_keys)


def generate_replicates(
    params: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray,
    y0: jnp.ndarray,
    n_replicates: int,
    master_key: jax.Array,
    *,
    t_window: float = 60.0,
    dt: float = DEFAULT_DT_INT,
    dt_out: float = DEFAULT_DT_OUT,
    sigma_proc: jnp.ndarray = DEFAULT_PROCESS_SIGMA,
    noise_pct: float = DEFAULT_SENSOR_NOISE_PCT,
    drift_T: float = DEFAULT_DRIFT_T,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorised closed-loop replicate generator.

    Returns ``(t_out, observations)`` where ``observations`` has shape
    ``(n_replicates, n_t, 4)`` containing ``[C, T, Tc, Qc]`` with both
    process and sensor noise applied.
    """
    n_steps_int = int(round(t_window / dt))
    stride = int(round(dt_out / dt))
    obs = _generate_replicates_closed_loop(
        params, inlet, ctrl, y0, master_key,
        dt, sigma_proc, noise_pct, drift_T,
        n_steps_int, stride, n_replicates,
    )
    t_out = jnp.arange(1, obs.shape[1] + 1) * dt_out
    return t_out, obs


# ---------------------------------------------------------------------------
# Convenience: deterministic warm-start IC for a given (params, inlet, ctrl)
# ---------------------------------------------------------------------------

def warm_start_ic(
    params: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    t_warm: float = 1500.0,
) -> jnp.ndarray:
    """Run the deterministic closed-loop simulator to (near) steady state and
    return the resulting ``[C, T, Tc, I]`` as an EM warm-start IC.

    This is the M2-side answer to the M1 finding that a cold IC
    ``[0.5, 300, 297, 0]`` causes Tsit5 to break: each scenario gets its
    own warm-start, computed once and reused across all replicates.
    """
    from cstr_sbi.physics import NOMINAL_Y0_CL

    return simulate_closed_loop_to_steady_state(
        params, inlet, ctrl, NOMINAL_Y0_CL, t_final=t_warm,
    )
