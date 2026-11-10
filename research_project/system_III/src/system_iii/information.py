"""Gaussian Fisher information of the summary features and Bayesian Cramer-Rao bounds (common random numbers).

phi = (alpha, beta, nuisances...). Nuisances are calibration offsets / unmeasured inlet deviations with
Gaussian priors; the bound for (alpha, beta) marginalises them: inv(F + diag(prior precisions)).
"""
from __future__ import annotations

import numpy as np

from .physics import PLANT
from .simulator import simulate
from .features import features
from .sensors import CAL_SD, INLET_EPS, INLET_WHITE, CI_REL_SD, Q_REL_SD

NUISANCES = {   # name: (prior sd, finite-difference step)
    "oTi": (INLET_EPS, 0.1), "oTci": (INLET_EPS, 0.1), "uCi": (CI_REL_SD, 0.005), "uQ": (Q_REL_SD, 0.0025),
    "oT": (CAL_SD["oT"], 0.1), "oTc": (CAL_SD["oTc"], 0.1), "uQc": (CAL_SD["uQc"], 0.005), "uC": (CAL_SD["uC"], 0.005),
}


def _features(alpha, beta, n, seed, sensors, protocol, feature_set, plant, d, sim_kw=None):
    p = plant
    out = simulate(np.full(n, alpha + d.get("alpha", 0.0)), np.full(n, beta + d.get("beta", 0.0)), seed=seed,
                   plant=p, sensors=sensors, protocol=protocol,
                   Ti=p.Ti - d.get("oTi", 0.0), Tci=p.Tci - d.get("oTci", 0.0),
                   Ci=p.Ci * (1 + d.get("uCi", 0.0)), Q=p.Q * (1 + d.get("uQ", 0.0)),
                   oT=d.get("oT", 0.0), oTc=d.get("oTc", 0.0), uQc=d.get("uQc", 0.0), uC=d.get("uC", 0.0),
                   **(sim_kw or {}))
    rng = np.random.default_rng(seed + 11)
    w = INLET_WHITE / np.sqrt(len(out["t"]))
    Ti_m = p.Ti + w * rng.standard_normal(n); Tci_m = p.Tci + w * rng.standard_normal(n)
    return features(out["obs"], out["t"], Ti_m, Tci_m, feature_set, p).astype(float)


def bayes_crb(alpha=1.0, beta=1.0, *, nuisances=("oTi", "oTci", "uCi", "uQ", "oT", "oTc", "uQc", "uC"),
              n=1500, seed=4000, h=0.01, sensors="S3", protocol="P2", feature_set="S1m", plant=PLANT, **sim_kw):
    """Extra keyword arguments go to the simulator (e.g. control="cascade")."""
    """Returns dict(sd_alpha, sd_beta, corr_ab, sd_*_oracle, dropped features)."""
    X0 = _features(alpha, beta, n, seed, sensors, protocol, feature_set, plant, {}, sim_kw)
    cols, prec = [], [0.0, 0.0]
    for name in ("alpha", "beta") + tuple(nuisances):
        s = h if name in ("alpha", "beta") else NUISANCES[name][1]
        xp = _features(alpha, beta, n, seed, sensors, protocol, feature_set, plant, {name: s}, sim_kw).mean(0)
        xm = _features(alpha, beta, n, seed, sensors, protocol, feature_set, plant, {name: -s}, sim_kw).mean(0)
        cols.append((xp - xm) / (2 * s))
        if name not in ("alpha", "beta"):
            prec.append(1.0 / NUISANCES[name][0] ** 2)
    G = np.stack(cols, 1)
    keep = X0.std(0) > 1e-9 * (np.abs(X0.mean(0)) + 1)
    S = np.cov(X0[:, keep], rowvar=False); S += np.eye(len(S)) * 1e-9 * np.trace(S) / len(S)
    F = G[keep].T @ np.linalg.solve(S, G[keep])
    Fi = np.linalg.inv(F + np.diag(prec)); Fo = np.linalg.inv(F[:2, :2])
    return dict(sd_alpha=float(np.sqrt(Fi[0, 0])), sd_beta=float(np.sqrt(Fi[1, 1])),
                corr_ab=float(Fi[0, 1] / np.sqrt(Fi[0, 0] * Fi[1, 1])),
                sd_alpha_oracle=float(np.sqrt(Fo[0, 0])), sd_beta_oracle=float(np.sqrt(Fo[1, 1])),
                n_dropped=int((~keep).sum()), nuisances=list(nuisances))
