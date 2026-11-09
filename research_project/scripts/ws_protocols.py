"""Warm-start / initial-state protocols for the System I generative model.

Self-contained, vectorised (jit+vmap) alternative to the per-sample Python loop in
``cstr_sbi.inference.simulation_wrapper_sbi``. Imports physics/simulator READ-ONLY;
edits nothing shared. Used by nb29 (warm-start protocol study).

Protocols (initial state of each simulated observation window):
  P0   fixed healthy steady state for every theta        (abrupt fault at window start)
  P1   deterministic closed-loop steady state at theta   (current production protocol)
  P1b  P1 followed by a noisy burn-in of ``burn_min`` (default 150 min, ~6 of the slowest
       closed-loop time constants, 10-26 min after the 2026-11-08 controller-form fix), so the
       window starts from the noisy plant's typical spread. ``p1b_mode="lna"`` instead draws
       y0 ~ N(ss, C_LNA) from the Lyapunov equation (cross-check / cheap alternative).
  P2   theta ramps to its current value over ``hist_min`` (default 300 min) at the
       30-day-stream degradation rates (rate_mult scales them), start = steady state of the
       earlier theta, noise on throughout (plant lags a slowly moving steady state)
  P3   steady state at theta' = theta + N(0, sigma_nuis^2) (clipped to prior), a
       nuisance initial-state perturbation

theta = (alpha, beta); UA and k0 fixed at nominal.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from cstr_sbi.physics import (
    K0_NOMINAL, UA_NOMINAL, NOMINAL_INLET_CL, NOMINAL_CTRL, cstr_closed_loop_rhs,
    compute_qc, closed_loop_steady_state_exact,
)
from cstr_sbi.simulator import DEFAULT_PROCESS_SIGMA, DEFAULT_SENSOR_NOISE_PCT, DEFAULT_DT_INT, DEFAULT_DT_OUT
from cstr_sbi.summaries import compute_summary_statistics
from cstr_sbi.priors import ALPHA_LOW, ALPHA_HIGH, BETA_LOW, BETA_HIGH

PROTOCOLS = ("P0", "P1", "P1b", "P2", "P3")

# Realistic degradation rates from scenarios.generate_degradation_stream defaults
# (30-day horizon, alpha_end=beta_end=0.85, beta Kern-Seaton tau=200 h):
#   alpha: linear, 0.15 per 720 h        -> 2.08e-4 /h
#   beta : initial slope 0.15/(1-e^-3.6) / 200 h -> 7.7e-4 /h (decreasing later)
RATE_ALPHA_PER_MIN = 0.15 / 720.0 / 60.0
RATE_BETA_PER_MIN = 0.15 / (1.0 - np.exp(-720.0 / 200.0)) / 200.0 / 60.0

_inlet = NOMINAL_INLET_CL
_ctrl = NOMINAL_CTRL


def _params(alpha, beta):
    return jnp.stack([jnp.asarray(UA_NOMINAL, jnp.float32), jnp.asarray(K0_NOMINAL, jnp.float32),
                      jnp.asarray(alpha, jnp.float32), jnp.asarray(beta, jnp.float32)])


def steady_state(alpha, beta):
    return closed_loop_steady_state_exact(_params(alpha, beta), _inlet, _ctrl)


_ss_batch = jax.jit(jax.vmap(steady_state))


@partial(jax.jit, static_argnames=("n_steps",))
def _em_ramp(y0, key, theta_start, theta_end, n_steps, dt=DEFAULT_DT_INT, sigma=DEFAULT_PROCESS_SIGMA):
    """EM with (alpha,beta) interpolated linearly theta_start -> theta_end over the run.
    Returns the full state path (n_steps, 4)."""
    keys = jax.random.split(key, n_steps)
    sqrt_dt = jnp.sqrt(dt)

    def step(carry, inp):
        y, = carry
        k, frac = inp
        th = theta_start + (theta_end - theta_start) * frac
        p = _params(th[0], th[1])
        drift = cstr_closed_loop_rhs(0.0, y, (p, _inlet, _ctrl))
        y1 = y + drift * dt + sigma * sqrt_dt * jax.random.normal(k, shape=y.shape)
        return (y1,), y1

    fracs = (jnp.arange(n_steps) + 1.0) / n_steps
    _, ys = jax.lax.scan(step, (y0,), (keys, fracs))
    return ys


def _observe(ys_window, sens_key, t_out, noise_pct=DEFAULT_SENSOR_NOISE_PCT, noise_abs=None):
    """ys_window (n_out,4)=[C,T,Tc,I] subsampled -> obs (n_out,4)=[C,T,Tc,Qc] with sensor noise.

    Default: sigma = noise_pct * window maximum of each channel (the project convention).
    ``noise_abs`` = (sigma_C, sigma_T, sigma_Tc, sigma_Qc) replaces it with absolute,
    per-channel standard deviations; an entry of None keeps the default convention for that
    channel (nb33 sensor-realism study, 2026-11-08)."""
    qc = jax.vmap(compute_qc, in_axes=(0, 0, None))(ys_window[:, 1], ys_window[:, 3], _ctrl)
    obs = jnp.stack([ys_window[:, 0], ys_window[:, 1], ys_window[:, 2], qc], axis=1)
    mx = jnp.maximum(jnp.max(jnp.abs(obs), axis=0, keepdims=True), 1e-12)
    if noise_abs is None:
        # exactly the original expression: keeps earlier datasets bit-reproducible
        return obs + jax.random.normal(sens_key, obs.shape) * noise_pct * mx
    na = np.array([np.nan if v is None else float(v) for v in noise_abs], dtype=np.float32)
    sig = jnp.where(jnp.isnan(na)[None, :], noise_pct * mx, jnp.asarray(np.nan_to_num(na))[None, :])
    return obs + jax.random.normal(sens_key, obs.shape) * sig



_jac_batch = jax.jit(jax.vmap(lambda a, b, y: jax.jacfwd(
    lambda yy: cstr_closed_loop_rhs(0.0, yy, (_params(a, b), _inlet, _ctrl)))(y)))


def lna_chol(alpha, beta, y_ss):
    """Cholesky factors L (B,4,4) of the stationary LNA covariance C = L L^T at each theta.
    Saturated points (zero integrator row/col) drop the frozen state(s)."""
    from scipy.linalg import solve_continuous_lyapunov
    J = np.asarray(_jac_batch(alpha, beta, y_ss), dtype=np.float64)
    q = np.asarray(DEFAULT_PROCESS_SIGMA, dtype=np.float64) ** 2
    out = np.zeros_like(J)
    for i in range(J.shape[0]):
        act = [k for k in range(4) if np.abs(J[i, k, :]).max() > 1e-9 or np.abs(J[i, :, k]).max() > 1e-9]
        Ji = J[i][np.ix_(act, act)]
        Ci = solve_continuous_lyapunov(Ji, -np.diag(q[act]))
        Ci = 0.5 * (Ci + Ci.T)
        w, V = np.linalg.eigh(Ci); w = np.clip(w, 0.0, None)
        Li = V @ np.diag(np.sqrt(w))
        out[i][np.ix_(act, act)] = Li
    return jnp.asarray(out, jnp.float32)


def make_batch_simulator(protocol: str, *, t_window=60.0, burn_min=150.0, hist_min=300.0, p1b_mode="burnin",
                         rate_mult=1.0, sigma_nuis=0.03, dt=DEFAULT_DT_INT, dt_out=DEFAULT_DT_OUT,
                         noise_abs=None):
    """Return ``sim(theta_np (B,2), seed:int) -> (obs (B,n_t,4), t (n_t,))`` for a protocol."""
    stride = int(round(dt_out / dt))
    n_win = int(round(t_window / dt))
    pre = {"P0": 0, "P1": 0, "P3": 0, "P1b": (int(round(burn_min / dt)) if p1b_mode == "burnin" else 0),
           "P2": int(round(hist_min / dt))}[protocol]
    n_tot = pre + n_win
    t_out = jnp.arange(1, n_win // stride + 1) * dt_out

    def one(theta, y0, th_start, key):
        proc_key, sens_key = jax.random.split(key)
        ys = _em_ramp(y0, proc_key, th_start, theta, n_steps=n_tot, dt=dt)
        ys_w = ys[pre:][stride - 1::stride]
        return _observe(ys_w, sens_key, t_out, noise_abs=noise_abs)

    # theta is held fixed inside the window; for P2 the ramp reaches theta at the END of the
    # history and stays there -> ramp only over `pre` steps. Implement by using ramp over whole run
    # for P2 would keep drifting through the window; so handle P2 with explicit two-segment run.
    if protocol == "P2":
        def one(theta, y0, th_start, key):  # noqa: F811
            proc_key, sens_key = jax.random.split(key)
            k1, k2 = jax.random.split(proc_key)
            ys_h = _em_ramp(y0, k1, th_start, theta, n_steps=pre, dt=dt)
            ys_w = _em_ramp(ys_h[-1], k2, theta, theta, n_steps=n_win, dt=dt)[stride - 1::stride]
            return _observe(ys_w, sens_key, t_out, noise_abs=noise_abs)

    one_v = jax.jit(jax.vmap(one))

    def sim(theta_np, seed: int):
        theta = jnp.asarray(theta_np, jnp.float32)
        B = theta.shape[0]
        rng = np.random.default_rng(seed)
        if protocol == "P0":
            y0 = jnp.tile(steady_state(1.0, 1.0)[None], (B, 1)); th_start = theta
        elif protocol == "P1":
            y0 = _ss_batch(theta[:, 0], theta[:, 1]); th_start = theta
        elif protocol == "P1b" and p1b_mode == "burnin":
            y0 = _ss_batch(theta[:, 0], theta[:, 1]); th_start = theta   # noisy burn-in run inside the scan
        elif protocol == "P1b":
            yss = _ss_batch(theta[:, 0], theta[:, 1])
            L = lna_chol(theta[:, 0], theta[:, 1], yss)
            z = jnp.asarray(rng.normal(size=(B, 4)), jnp.float32)
            y0 = yss + jnp.einsum("bij,bj->bi", L, z); th_start = theta
        elif protocol == "P3":
            eps = jnp.asarray(rng.normal(0, sigma_nuis, size=(B, 2)), jnp.float32)
            tp = jnp.clip(theta + eps, jnp.array([ALPHA_LOW, BETA_LOW]), jnp.array([ALPHA_HIGH, BETA_HIGH]))
            y0 = _ss_batch(tp[:, 0], tp[:, 1]); th_start = theta
        elif protocol == "P2":
            # earlier theta = healthier: theta + rate * hist  (random per-sample rate in [0, rate_mult*max])
            u = jnp.asarray(rng.uniform(0, 1, size=(B, 2)), jnp.float32)
            d = u * rate_mult * hist_min * jnp.array([RATE_ALPHA_PER_MIN, RATE_BETA_PER_MIN], jnp.float32)
            th_start = jnp.clip(theta + d, jnp.array([ALPHA_LOW, BETA_LOW]), jnp.array([ALPHA_HIGH, BETA_HIGH]))
            y0 = _ss_batch(th_start[:, 0], th_start[:, 1])
        else:
            raise ValueError(protocol)
        keys = jax.random.split(jax.random.PRNGKey(seed), B)
        return one_v(theta, y0, th_start, keys), t_out

    return sim


_summ = jax.jit(jax.vmap(lambda o, t: compute_summary_statistics(o, t), in_axes=(0, None)))


def summaries(obs, t):
    return np.asarray(_summ(obs, t))


def simulate_summaries(theta_np, protocol: str, seed: int, *, batch: int = 100, **kw):
    """Chunked: (n,2) theta -> (n,29) summaries under ``protocol``."""
    sim = make_batch_simulator(protocol, **kw)
    out = []
    for i in range(0, len(theta_np), batch):
        obs, t = sim(theta_np[i:i + batch], seed + i)
        out.append(summaries(obs, t))
    return np.concatenate(out)
