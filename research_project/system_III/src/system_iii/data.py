"""Window generation for training, testing and the stress tests: simulate + measure inlet + featurise.

``make_windows`` is the single entry point used by every notebook, so training and test data always
come from the same simulator and feature code; only the arguments differ.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .physics import PLANT
from .simulator import simulate, prior_draws
from .sensors import measured_inlet, realistic_conditions, INLET_EPS
from .features import features, nominal_inlet_features
from .scenarios import SCENARIOS

CHUNK = 5_000


def make_windows(theta, seed, *, conditions="production", sensors="S3", protocol="P2", feature_set="S1m",
                 inlet_eps=INLET_EPS, assume_nominal_inlet=False, return_raw=False, plant=PLANT, **sim_kw):
    """Simulate windows for theta (n, 2) and return dict(x, theta, [obs, inlet_true, inlet_meas]).

    conditions: "production" = inlet varies (+-2 K, Ci, Q) and every sensor carries a calibration offset;
                "nominal" = nominal inlet, no calibration offsets (System I-like idealisation);
                "inlet_only" = inlet varies, no sensor calibration offsets.
    assume_nominal_inlet: the NAIVE model's features (inlet assumed nominal instead of measured).
    Extra keyword arguments go to the simulator (e.g. control, oT, drift_T_monitor, Tci)."""
    theta = np.asarray(theta, float)
    xs, obs_all, inl_all, meas_all = [], [], [], []
    for c in range(0, len(theta), CHUNK):
        th = theta[c:c + CHUNK]; n = len(th)
        rng = np.random.default_rng(seed * 7919 + c)
        kw = {}
        if conditions in ("production", "inlet_only"):
            kw = realistic_conditions(n, rng, plant, calibration=(conditions == "production"))
        kw.update({k: (v[c:c + CHUNK] if isinstance(v, np.ndarray) and v.shape[:1] == (len(theta),) else v)
                   for k, v in sim_kw.items()})
        out = simulate(th[:, 0], th[:, 1], seed=seed + c, plant=plant, sensors=sensors, protocol=protocol, **kw)
        Ti_m, Tci_m = measured_inlet(out["inlet"][:, 1], out["inlet"][:, 2], rng, eps=inlet_eps)
        if assume_nominal_inlet:
            x = nominal_inlet_features(out["obs"], out["t"], feature_set, plant)
        else:
            x = features(out["obs"], out["t"], Ti_m, Tci_m, feature_set, plant)
        xs.append(x)
        if return_raw:
            obs_all.append(out["obs"].astype(np.float32)); inl_all.append(out["inlet"]); meas_all.append(np.stack([Ti_m, Tci_m], 1))
    res = dict(x=np.concatenate(xs), theta=theta.astype(np.float32))
    if return_raw:
        res.update(obs=np.concatenate(obs_all), inlet_true=np.concatenate(inl_all), inlet_meas=np.concatenate(meas_all))
    return res


def scenario_set(names, reps, seed, return_raw=False, **kw):
    """Windows for a list of scenarios (each with its own control mode, drift and control-sensor offset)."""
    xs, th, nm, raws = [], [], [], []
    for i, name in enumerate(names):
        sc = SCENARIOS[name]
        theta = np.tile([sc.alpha, sc.beta], (reps, 1))
        extra = dict(kw)
        if sc.control == "fixed":
            extra["control"] = "fixed"
        if sc.drift_T_monitor:
            extra["drift_T_monitor"] = sc.drift_T_monitor
        if sc.oT:
            extra["oT"] = sc.oT
        d = make_windows(theta, seed + 1000 * i, return_raw=return_raw, **extra)
        xs.append(d["x"]); th.append(d["theta"]); nm += [name] * reps
        if return_raw:
            raws.append({k: d[k] for k in ("obs", "inlet_true", "inlet_meas")})
    out = dict(x=np.concatenate(xs), theta=np.concatenate(th), sc_name=np.array(nm))
    if return_raw:
        for k in ("obs", "inlet_true", "inlet_meas"):
            out[k] = np.concatenate([r[k] for r in raws])
    return out


def test_set(seed, n_sbc=1000, sc_reps=50, scenarios=None, **kw):
    """SBC set (prior draws) + scenario set, in the layout inference.evaluate expects."""
    from .scenarios import CLOSED
    sbc = make_windows(prior_draws(n_sbc, seed), seed + 1, **kw)
    sc = scenario_set(scenarios or CLOSED, sc_reps, seed + 2, **kw)
    return dict(theta_sbc=sbc["theta"], x_sbc=sbc["x"], theta_sc=sc["theta"], x_sc=sc["x"], sc_name=sc["sc_name"])


def save_npz(path, d):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
