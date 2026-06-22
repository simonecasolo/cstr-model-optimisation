"""Stochastic Luyben plant simulator.

Euler-Maruyama integrator for the 13-state closed-loop plant,
plus sensor layer, replicate generator, and warm-start helper.
Mirrors the structure of cstr_sbi.simulator for the propylene oxide system.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from cstr_sbi.luyben.physics import (
    NOMINAL_CTRL_ALL,
    NOMINAL_INLET,
    NOMINAL_THETA,
    NOMINAL_Y0,
    N_OBS,
    luyben_rhs,
    extract_observations,
    simulate_to_steady_state,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Process noise: additive on [Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s, + 5 integrators=0]
DEFAULT_PROCESS_SIGMA = jnp.array(
    [0.002, 0.002, 0.2, 0.1, 0.5, 0.005, 0.005, 0.2,  # plant states
     0.0, 0.0, 0.0, 0.0, 0.0],                          # PI integrators
    dtype=jnp.float32,
)

DEFAULT_SENSOR_NOISE_PCT: float = 0.005  # 0.5% of channel max
DEFAULT_DT_INT: float = 0.02             # min, internal EM step
DEFAULT_DT_OUT: float = 1.0              # min, 2-hour window -> 120 obs


# ---------------------------------------------------------------------------
# EM scan (closed-loop, 13-state)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_steps", "stride"))
def _em_scan(y0, key, theta, inlet, ctrl, dt, sigma, n_steps, stride):
    """Inner JAX scan for the Luyben EM integrator."""
    keys = jax.random.split(key, n_steps)
    sqrt_dt = jnp.sqrt(dt)

    def step(y, k):
        drift = luyben_rhs(0.0, y, (theta, inlet, ctrl))
        xi = jax.random.normal(k, shape=y.shape)
        # Clip states to physical bounds after each step
        y_next = y + drift * dt + sigma * sqrt_dt * xi
        y_next = y_next.at[0].set(jnp.maximum(y_next[0], 0.0))   # Ca >= 0
        y_next = y_next.at[1].set(jnp.maximum(y_next[1], 0.0))   # Cb >= 0
        y_next = y_next.at[4].set(jnp.maximum(y_next[4], 1.0))   # n_L >= 1
        y_next = y_next.at[5].set(jnp.clip(y_next[5], 0.0, 1.0)) # x_A in [0,1]
        y_next = y_next.at[6].set(jnp.clip(y_next[6], 0.0, 1.0)) # x_B in [0,1]
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, keys)
    return ys[stride - 1::stride]


def simulate_em_window(
    theta: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray,
    y0: jnp.ndarray,
    *,
    key: jax.Array,
    t_window: float = 120.0,
    dt: float = DEFAULT_DT_INT,
    dt_out: float = DEFAULT_DT_OUT,
    sigma: jnp.ndarray = DEFAULT_PROCESS_SIGMA,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """EM integration of the Luyben plant over ``t_window`` minutes.

    Returns ``(t, ys, obs)`` where:
        t   -- shape (n_out,), output times in minutes
        ys  -- shape (n_out, 13), all states
        obs -- shape (n_out, 8), observable channels [T_r, Tc, Qc, T_s, Q_s, F_R, F_P, F_prod]
    """
    n_steps = int(round(t_window / dt))
    stride  = int(round(dt_out / dt))
    ys = _em_scan(y0, key, theta, inlet, ctrl, dt, sigma, n_steps, stride)
    t_out = jnp.arange(1, ys.shape[0] + 1) * dt_out
    obs = extract_observations(ys, theta, ctrl)
    return t_out, ys, obs


# ---------------------------------------------------------------------------
# Sensor layer (8 observable channels)
# ---------------------------------------------------------------------------

def apply_sensor_layer(
    obs: jnp.ndarray,
    *,
    key: jax.Array,
    noise_pct: float = DEFAULT_SENSOR_NOISE_PCT,
) -> jnp.ndarray:
    """Apply Gaussian sensor noise (fraction of channel max) to (n_t, 8) obs array."""
    max_per_channel = jnp.maximum(jnp.max(jnp.abs(obs), axis=0, keepdims=True), 1e-6)
    sigma_obs = noise_pct * max_per_channel
    noise = jax.random.normal(key, shape=obs.shape) * sigma_obs
    return obs + noise


# ---------------------------------------------------------------------------
# Replicate generator
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_steps_int", "stride", "n_replicates"))
def _generate_replicates_jit(
    theta, inlet, ctrl, y0, master_key, dt, sigma_proc, noise_pct,
    n_steps_int, stride, n_replicates,
):
    keys = jax.random.split(master_key, 2 * n_replicates)
    proc_keys = keys[:n_replicates]
    sens_keys = keys[n_replicates:]

    def one_replicate(proc_key, sens_key):
        ys = _em_scan(y0, proc_key, theta, inlet, ctrl, dt, sigma_proc, n_steps_int, stride)
        obs = extract_observations(ys, theta, ctrl)
        return apply_sensor_layer(obs, key=sens_key, noise_pct=noise_pct)

    return jax.vmap(one_replicate)(proc_keys, sens_keys)


def generate_replicates(
    theta: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray,
    y0: jnp.ndarray,
    n_replicates: int,
    master_key: jax.Array,
    *,
    t_window: float = 120.0,
    dt: float = DEFAULT_DT_INT,
    dt_out: float = DEFAULT_DT_OUT,
    sigma_proc: jnp.ndarray = DEFAULT_PROCESS_SIGMA,
    noise_pct: float = DEFAULT_SENSOR_NOISE_PCT,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate ``n_replicates`` noisy 2-hour observation windows.

    Returns ``(t_out, observations)`` where observations has shape
    ``(n_replicates, n_t, 8)``.
    """
    n_steps_int = int(round(t_window / dt))
    stride = int(round(dt_out / dt))
    obs = _generate_replicates_jit(
        theta, inlet, ctrl, y0, master_key,
        dt, sigma_proc, noise_pct,
        n_steps_int, stride, n_replicates,
    )
    t_out = jnp.arange(1, obs.shape[1] + 1) * dt_out
    return t_out, obs


# ---------------------------------------------------------------------------
# Warm-start IC
# ---------------------------------------------------------------------------

def warm_start_ic(
    theta: jnp.ndarray,
    inlet: jnp.ndarray = None,
    ctrl: jnp.ndarray = None,
    t_warm: float = 2000.0,
) -> jnp.ndarray:
    """Run the deterministic simulator to (near) steady state and return IC."""
    if inlet is None:
        inlet = NOMINAL_INLET
    if ctrl is None:
        ctrl = NOMINAL_CTRL_ALL
    return simulate_to_steady_state(theta, inlet, ctrl, NOMINAL_Y0, t_final=t_warm)
