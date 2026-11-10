"""Stochastic checks C6 (information), C8 (proxy separation), C9 (summary robustness)."""
from __future__ import annotations

import numpy as np

from design_lib import simulate, observe, features, plugins, base29, FEATURE_NAMES

T_GRID = np.arange(1, 121) * 0.5
WHITE = 0.2 / np.sqrt(120.0)
NUIS_SD = {"oTi": 0.1, "oTci": 0.1, "uCi": 0.01, "uQ": 0.005}
NUIS_SD_EXT = {**NUIS_SD, "oT_loop": 0.1, "oTc": 0.1, "uQc": 0.01, "uC": 0.01}


def windows(d, alpha, beta, n, seed, *, Ti=None, Tci=None, Ci=None, Q=None, protocol="P2", Ti_m=None, Tci_m=None,
            drift_T=0.0, oT_loop=0.0, oTc=0.0, uQc=0.0, uC=0.0):
    """Simulate n windows and return (features (n,34), obs).
    drift_T: offset on a MONITORING T reading (outside the loop, System I Sc7 convention).
    oT_loop: offset on the CONTROL T sensor (in the loop: true T is held at Tsp - oT_loop, reading = Tsp).
    oTc: jacket-outlet sensor offset [K]; uQc: coolant flowmeter relative bias; uC: analyser relative bias."""
    from dataclasses import replace as _rep
    a = np.full(n, alpha) if np.isscalar(alpha) else np.asarray(alpha)
    b = np.full(n, beta) if np.isscalar(beta) else np.asarray(beta)
    d_true = _rep(d, Tsp=d.Tsp - oT_loop) if oT_loop else d
    ys, qs = simulate(d_true, a, b, Ci=Ci, Ti=Ti, Tci=Tci, Q=Q, seed=seed, protocol=protocol)
    obs = observe(d, ys, qs, seed=seed + 7)
    obs[:, :, 1] += drift_T + oT_loop
    obs[:, :, 2] += oTc
    obs[:, :, 3] *= (1 + uQc)
    obs[:, :, 0] *= (1 + uC)
    rng = np.random.default_rng(seed + 11)
    Ti_m = (np.full(n, d.Ti) if Ti_m is None else Ti_m) + WHITE * rng.standard_normal(n)
    Tci_m = (np.full(n, d.Tci) if Tci_m is None else Tci_m) + WHITE * rng.standard_normal(n)
    return features(d, obs, T_GRID, Ti_m, Tci_m), obs


def fim(d, alpha=1.0, beta=1.0, n=1500, seed=4000, h=0.01, extended=False):
    """Gaussian FIM of the 34-D features over phi = (alpha, beta, oTi, oTci, uCi, uQ) with common random numbers.
    Offsets: the measured inlet stays at nominal while the true inlet moves (true = measured - offset)."""
    steps = {"alpha": h, "beta": h, "oTi": 0.1, "oTci": 0.1, "uCi": 0.005, "uQ": 0.0025}
    sds = dict(NUIS_SD)
    if extended:
        steps.update({"oT_loop": 0.1, "oTc": 0.1, "uQc": 0.005, "uC": 0.005}); sds = dict(NUIS_SD_EXT)

    def run(da=0.0, db=0.0, oTi=0.0, oTci=0.0, uCi=0.0, uQ=0.0, **ext):
        X, _ = windows(d, alpha + da, beta + db, n, seed, Ti=d.Ti - oTi, Tci=d.Tci - oTci,
                       Ci=d.Ci * (1 + uCi), Q=d.Q * (1 + uQ), **ext)
        return X

    X0 = run()
    G = []
    for k, s in steps.items():
        kw_p = {"alpha": dict(da=s), "beta": dict(db=s)}.get(k, {k: s})
        kw_m = {"alpha": dict(da=-s), "beta": dict(db=-s)}.get(k, {k: -s})
        G.append((run(**kw_p).mean(0) - run(**kw_m).mean(0)) / (2 * s))
    G = np.stack(G, 1)
    keep = X0.std(0) > 1e-9 * (np.abs(X0.mean(0)) + 1)
    Xk, Gk = X0[:, keep], G[keep]
    S = np.cov(Xk, rowvar=False)
    S += np.eye(len(S)) * 1e-9 * np.trace(S) / len(S)
    F = Gk.T @ np.linalg.solve(S, Gk)
    prior = np.diag([0, 0] + [1 / sds[k] ** 2 for k in list(steps)[2:]])
    Fi_bayes = np.linalg.inv(F + prior)
    Fi_oracle = np.linalg.inv(F[:2, :2])
    out = dict(sd_alpha=float(np.sqrt(Fi_bayes[0, 0])), sd_beta=float(np.sqrt(Fi_bayes[1, 1])),
               corr_ab=float(Fi_bayes[0, 1] / np.sqrt(Fi_bayes[0, 0] * Fi_bayes[1, 1])),
               sd_alpha_oracle=float(np.sqrt(Fi_oracle[0, 0])), sd_beta_oracle=float(np.sqrt(Fi_oracle[1, 1])),
               cond_cov=float(np.linalg.cond(S)), n_dropped_constant=int((~keep).sum()),
               dropped=[FEATURE_NAMES[i] for i in np.flatnonzero(~keep)])
    # proxy-only information: alpha_hat, beta_hat_jac
    idx = [FEATURE_NAMES.index("ln_alpha_hat"), FEATURE_NAMES.index("ln_beta_hat_jac")]
    Gp, Sp = G[idx], np.cov(X0[:, idx], rowvar=False)
    Fp = Gp.T @ np.linalg.solve(Sp, Gp)
    Fpi = np.linalg.inv(Fp + prior)
    out.update(proxy_sd_alpha=float(np.sqrt(Fpi[0, 0])), proxy_sd_beta=float(np.sqrt(Fpi[1, 1])))
    return out


def realistic_inlets(d, n, seed):
    rng = np.random.default_rng(seed)
    Ti = d.Ti + rng.uniform(-2, 2, n); Tci = d.Tci + rng.uniform(-2, 2, n)
    Ti_m = Ti + NUIS_SD["oTi"] * rng.standard_normal(n)
    Tci_m = Tci + NUIS_SD["oTci"] * rng.standard_normal(n)
    Ci = d.Ci * (1 + NUIS_SD["uCi"] * rng.standard_normal(n))
    Q = d.Q * (1 + NUIS_SD["uQ"] * rng.standard_normal(n))
    return dict(Ti=Ti, Tci=Tci, Ci=Ci, Q=Q, Ti_m=Ti_m, Tci_m=Tci_m)


def proxy_accuracy(d, points, n=400, seed=5000, thr=0.90):
    """Plain threshold rule on the two physics proxies under the full realistic variation."""
    rows = []
    iU = FEATURE_NAMES.index("UA_eff_proxy"); iK = FEATURE_NAMES.index("k0_eff_proxy")
    for j, (a, b) in enumerate(points):
        inl = realistic_inlets(d, n, seed + j)
        X, obs = windows(d, a, b, n, seed + 100 + j, **inl)
        ah = np.exp(X[:, FEATURE_NAMES.index("ln_alpha_hat")]); bh = np.exp(X[:, FEATURE_NAMES.index("ln_beta_hat_jac")])
        bh_rx = np.exp(X[:, FEATURE_NAMES.index("ln_beta_hat_rx")])
        true = (a < thr, b < thr)
        pred_ok = ((ah < thr) == true[0]) & ((bh < thr) == true[1])
        rows.append(dict(alpha=a, beta=b, acc=float(pred_ok.mean()), alpha_hat_mean=float(ah.mean()),
                         alpha_hat_sd=float(ah.std()), beta_hat_mean=float(bh.mean()), beta_hat_sd=float(bh.std()),
                         beta_rx_mean=float(bh_rx.mean()), beta_rx_sd=float(bh_rx.std()),
                         UA_proxy_mean=float(X[:, iU].mean()), UA_proxy_sd=float(X[:, iU].std()),
                         k0_proxy_mean=float(X[:, iK].mean()), k0_proxy_sd=float(X[:, iK].std()),
                         sat_high_frac=float((obs[:, :, 3] >= d.Qcmax - 1).mean())))
    return rows


def cross_sensitivity(d, n=300, seed=6000):
    """Slopes of the proxy means w.r.t. alpha and beta over an unsaturated 4x4 grid (realistic variation)."""
    A = np.array([0.6, 0.8, 1.0, 1.15]); B = np.array([0.6, 0.8, 1.0, 1.15])
    M = {}
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            inl = realistic_inlets(d, n, seed + 10 * i + j)
            X, _ = windows(d, a, b, n, seed + 500 + 10 * i + j, **inl)
            M[(i, j)] = (np.exp(X[:, FEATURE_NAMES.index("ln_alpha_hat")]).mean(),
                         np.exp(X[:, FEATURE_NAMES.index("ln_beta_hat_jac")]).mean(),
                         X[:, FEATURE_NAMES.index("UA_eff_proxy")].mean(), X[:, FEATURE_NAMES.index("k0_eff_proxy")].mean())
    out = {"d_alphahat_d_beta": [], "d_betahat_d_alpha": [], "d_alphahat_d_alpha": [], "d_betahat_d_beta": [],
           "UAproxy_d_alpha_over_d_beta": [], "k0proxy_d_beta_over_d_alpha": []}
    for i in range(len(A)):
        for j in range(len(B) - 1):
            out["d_alphahat_d_beta"].append((M[(i, j + 1)][0] - M[(i, j)][0]) / (B[j + 1] - B[j]))
            out["d_betahat_d_beta"].append((M[(i, j + 1)][1] - M[(i, j)][1]) / (B[j + 1] - B[j]))
    for j in range(len(B)):
        for i in range(len(A) - 1):
            out["d_betahat_d_alpha"].append((M[(i + 1, j)][1] - M[(i, j)][1]) / (A[i + 1] - A[i]))
            out["d_alphahat_d_alpha"].append((M[(i + 1, j)][0] - M[(i, j)][0]) / (A[i + 1] - A[i]))
    # legacy proxies: ratio of cross- to own-sensitivity
    for j in range(len(B)):
        for i in range(len(A) - 1):
            dUa = (M[(i + 1, j)][2] - M[(i, j)][2]) / (A[i + 1] - A[i])
            dUb = (M[(i, min(j + 1, 3))][2] - M[(i, max(j - 1, 0))][2]) / (B[min(j + 1, 3)] - B[max(j - 1, 0)])
            out["UAproxy_d_alpha_over_d_beta"].append(dUa / dUb)
    for i in range(len(A)):
        for j in range(len(B) - 1):
            dKb = (M[(i, j + 1)][3] - M[(i, j)][3]) / (B[j + 1] - B[j])
            dKa = (M[(min(i + 1, 3), j)][3] - M[(max(i - 1, 0), j)][3]) / (A[min(i + 1, 3)] - A[max(i - 1, 0)])
            out["k0proxy_d_beta_over_d_alpha"].append(dKb / dKa)
    return {k: (float(np.min(v)), float(np.max(v))) for k, v in out.items()}


def summary_robustness(d, n=800, seed=7000):
    """Feature shift P1b vs P2 at nominal, in units of the P2 sd; constant features; clipping."""
    X2, _ = windows(d, 1.0, 1.0, n, seed, protocol="P2")
    X1, _ = windows(d, 1.0, 1.0, n, seed, protocol="P1b")
    sd = X2.std(0)
    shift = np.where(sd > 0, (X1.mean(0) - X2.mean(0)) / np.where(sd > 0, sd, 1), 0)
    kidx = FEATURE_NAMES.index("k0_eff_proxy")
    return dict(max_abs_shift_sd=float(np.abs(shift).max()), worst_feature=FEATURE_NAMES[int(np.abs(shift).argmax())],
                shifts={FEATURE_NAMES[i]: round(float(shift[i]), 3) for i in np.argsort(-np.abs(shift))[:6]},
                constant_features=[FEATURE_NAMES[i] for i in np.flatnonzero(sd < 1e-12)],
                k0_proxy_clip_frac=float(((X2[:, kidx] <= -8) | (X2[:, kidx] >= 2)).mean()))
