"""Closed-loop window simulator with a PER-SAMPLE inlet (notebook 34, study R2).

Same physics, protocols (P1b, P2), random-number consumption and sensor layer as
``ws_protocols.make_batch_simulator``, except that every simulated window carries its own inlet
vector [Ci, Ti, Tci] instead of the global nominal one. Used to

  (A) simulate a plant whose true inlet differs from the nominal values the posterior assumes
      (misspecification test), and
  (B) train posteriors with the inlet drawn as a nuisance variable (robust training).

``ws_protocols`` is NOT modified (its outputs stay bit-identical for every earlier study). With
the nominal inlet this module reproduces ``ws_protocols`` to floating-point accuracy; the check
is ``nominal_identity_check()`` and tests/test_ws_inlet.py. Exact bit-identity is not expected:
the inlet becomes a traced array instead of a compile-time constant, which can change the order
of floating-point operations.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import ws_protocols as ws
from cstr_sbi.physics import (CI_NOMINAL, TI_NOMINAL, TCI_NOMINAL, NOMINAL_CTRL, cstr_closed_loop_rhs,
                              closed_loop_steady_state_exact)
from cstr_sbi.simulator import DEFAULT_PROCESS_SIGMA, DEFAULT_DT_INT, DEFAULT_DT_OUT
from cstr_sbi.priors import ALPHA_LOW, ALPHA_HIGH, BETA_LOW, BETA_HIGH

NOMINAL = np.array([CI_NOMINAL, TI_NOMINAL, TCI_NOMINAL], np.float32)
_ctrl = NOMINAL_CTRL


def _ss(alpha, beta, inlet):
    return closed_loop_steady_state_exact(ws._params(alpha, beta), inlet, _ctrl)


_ss_batch = jax.jit(jax.vmap(_ss))


@partial(jax.jit, static_argnames=("n_steps",))
def _em_ramp(y0, key, theta_start, theta_end, inlet, n_steps, dt=DEFAULT_DT_INT, sigma=DEFAULT_PROCESS_SIGMA):
    """ws_protocols._em_ramp with the inlet as an argument."""
    keys = jax.random.split(key, n_steps)
    sqrt_dt = jnp.sqrt(dt)

    def step(carry, inp):
        y, = carry
        k, frac = inp
        th = theta_start + (theta_end - theta_start) * frac
        p = ws._params(th[0], th[1])
        drift = cstr_closed_loop_rhs(0.0, y, (p, inlet, _ctrl))
        y1 = y + drift * dt + sigma * sqrt_dt * jax.random.normal(k, shape=y.shape)
        return (y1,), y1

    fracs = (jnp.arange(n_steps) + 1.0) / n_steps
    _, ys = jax.lax.scan(step, (y0,), (keys, fracs))
    return ys


def make_batch_simulator(protocol: str, *, t_window=60.0, burn_min=150.0, hist_min=300.0, rate_mult=1.0,
                         dt=DEFAULT_DT_INT, dt_out=DEFAULT_DT_OUT, noise_abs=None):
    """Return ``sim(theta (B,2), inlet (B,3), seed) -> (obs (B,n_t,4), t)`` for P1b (burn-in) or P2.

    The inlet is the TRUE plant inlet of each window (it acts on the dynamics and on the steady
    state the window starts from). The sensor layer and the controller are unchanged."""
    if protocol not in ("P1b", "P2"):
        raise ValueError("ws_inlet supports P1b (burn-in) and P2")
    stride = int(round(dt_out / dt))
    n_win = int(round(t_window / dt))
    pre = int(round((burn_min if protocol == "P1b" else hist_min) / dt))
    n_tot = pre + n_win
    t_out = jnp.arange(1, n_win // stride + 1) * dt_out

    if protocol == "P1b":
        def one(theta, inlet, y0, th_start, key):
            proc_key, sens_key = jax.random.split(key)
            ys = _em_ramp(y0, proc_key, th_start, theta, inlet, n_steps=n_tot, dt=dt)
            ys_w = ys[pre:][stride - 1::stride]
            return ws._observe(ys_w, sens_key, t_out, noise_abs=noise_abs)
    else:
        def one(theta, inlet, y0, th_start, key):
            proc_key, sens_key = jax.random.split(key)
            k1, k2 = jax.random.split(proc_key)
            ys_h = _em_ramp(y0, k1, th_start, theta, inlet, n_steps=pre, dt=dt)
            ys_w = _em_ramp(ys_h[-1], k2, theta, theta, inlet, n_steps=n_win, dt=dt)[stride - 1::stride]
            return ws._observe(ys_w, sens_key, t_out, noise_abs=noise_abs)

    one_v = jax.jit(jax.vmap(one))

    def sim(theta_np, inlet_np, seed: int):
        theta = jnp.asarray(theta_np, jnp.float32)
        inlet = jnp.asarray(inlet_np, jnp.float32)
        B = theta.shape[0]
        assert inlet.shape == (B, 3), inlet.shape
        rng = np.random.default_rng(seed)            # same draws, same order as ws_protocols
        if protocol == "P1b":
            th_start = theta
        else:
            u = jnp.asarray(rng.uniform(0, 1, size=(B, 2)), jnp.float32)
            d = u * rate_mult * hist_min * jnp.array([ws.RATE_ALPHA_PER_MIN, ws.RATE_BETA_PER_MIN], jnp.float32)
            th_start = jnp.clip(theta + d, jnp.array([ALPHA_LOW, BETA_LOW]), jnp.array([ALPHA_HIGH, BETA_HIGH]))
        y0 = _ss_batch(th_start[:, 0], th_start[:, 1], inlet)
        keys = jax.random.split(jax.random.PRNGKey(seed), B)
        return one_v(theta, inlet, y0, th_start, keys), t_out

    return sim


_CACHE: dict = {}


def simulate(theta, inlet, protocol: str, seed: int, *, raw: bool = False, features: str = "S0_base29",
             batch: int = 200, **kw) -> np.ndarray:
    """ws_eval.simulate with a per-sample true inlet (n,3) [Ci, Ti, Tci]. Chunk i uses seed + i,
    exactly as ws_eval.simulate, so with the nominal inlet the two give the same windows (to
    floating-point accuracy). Features use the NOMINAL inlet constants (the analyst's assumption)."""
    import ws_eval as we
    key = (protocol, tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = make_batch_simulator(protocol, **kw)
    sim = _CACHE[key]
    theta = np.asarray(theta, np.float32)
    inlet = np.broadcast_to(np.asarray(inlet, np.float32), (len(theta), 3))
    out = []
    for i in range(0, len(theta), batch):
        obs, t = sim(theta[i:i + batch], inlet[i:i + batch], seed + i)
        obs = np.asarray(obs)
        out.append(obs if raw else we.features(obs, t, features))
    return np.concatenate(out)


def nominal_identity_check(protocol: str = "P2", n: int = 64, seed: int = 4242, **kw) -> dict:
    """Compare this simulator at the nominal inlet with ws_eval.simulate (raw windows)."""
    import ws_eval as we
    rng = np.random.default_rng(seed)
    th = rng.uniform([ALPHA_LOW, BETA_LOW], [ALPHA_HIGH, BETA_HIGH], size=(n, 2)).astype(np.float32)
    a = we.simulate(th, protocol, seed, raw=True, **kw)
    b = simulate(th, NOMINAL, protocol, seed, raw=True, **kw)
    d = np.abs(a - b).max(axis=(0, 1))
    return {"max_abs_diff_per_channel": d.tolist(), "bit_identical": bool(np.array_equal(a, b))}
