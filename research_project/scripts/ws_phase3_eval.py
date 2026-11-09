"""Shared evaluation helpers for the Phase 3 step-g notebooks (04, 06, ...): posterior draws for D0 / D1 windows and per-scenario metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

import ws_eval as we
import ws_metrics as wm
import ws_phase3 as P3

SCN = {1: "Sc1", 2: "Sc2", 3: "Sc3", 4: "Sc4", 5: "Sc5", 7: "Sc7"}
CHI2_90_2D = 4.605170185988092


def draws(which: str, data: str, n: int = 1000, seed: int = 0):
    """Posterior draws (B, n, 2) for 'prod' (nb36 ensemble, S1m features, needs D1) or 'naive' (nb31 ensemble, S1 features, D0 or D1).
    Returns (samples, dataset_dict, mask of the evaluated scenarios)."""
    d = P3.load_d1() if data == "D1" else P3.load_d0()
    keep = np.isin(d["scenario_id"], list(SCN))
    if which == "prod":
        assert data == "D1", "the production posterior needs the measured inlet (D1)"
        post, _ = P3.load_production_posterior(); X = P3.production_features(d)
    else:
        post, _ = P3.load_naive_posterior(); X = P3.naive_features(d)
    s, leak = we.sample_within_prior(post, X[keep], n, rng_seed=seed)
    sub = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(keep) else v) for k, v in d.items()}
    return s, sub, leak


def per_scenario(s, d, extra=True):
    """Bias, 90% coverage, sd, W1, CRPS (alpha, beta), 2-D coverage (Gaussian 90% region) and class accuracy per scenario."""
    from cstr_sbi.metrics import compute_crps, compute_wasserstein
    th = d["theta"][:, 2:].astype(float); sid = d["scenario_id"]
    yt = wm.class_of_theta(th); yp = wm.class_of_samples(s)
    q05 = np.quantile(s, .05, axis=1); q95 = np.quantile(s, .95, axis=1); mean = s.mean(1); sd = s.std(1)
    rows = []
    for sc, nm in SCN.items():
        m = np.where(sid == sc)[0]
        if len(m) == 0:
            continue
        r = dict(scenario=nm, n=len(m), alpha_true=float(th[m[0], 0]), beta_true=float(th[m[0], 1]), class_acc=float((yt[m] == yp[m]).mean()))
        for j, p in enumerate(("alpha", "beta")):
            r[f"{p}_mean"] = float(mean[m, j].mean()); r[f"{p}_bias"] = float((mean[m, j] - th[m, j]).mean()); r[f"{p}_sd"] = float(sd[m, j].mean())
            r[f"{p}_cov90"] = float(((th[m, j] >= q05[m, j]) & (th[m, j] <= q95[m, j])).mean())
            if extra:
                r[f"{p}_W1"] = float(np.mean([compute_wasserstein(s[i, :, j], np.full(s.shape[1], th[i, j])) for i in m]))
                r[f"{p}_CRPS"] = float(np.mean([compute_crps(s[i, :, j], th[i, j]) for i in m]))
        c = 0
        for i in m:
            cov = np.cov(s[i].T); dv = th[i] - mean[i]
            try:
                c += float(dv @ np.linalg.solve(cov, dv) <= CHI2_90_2D)
            except np.linalg.LinAlgError:
                pass
        r["joint_cov90_gauss"] = c / len(m)
        rows.append(r)
    return pd.DataFrame(rows)
