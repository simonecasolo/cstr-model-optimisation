"""System R design toolkit (numpy/scipy). Jacketed CSTR, first-order exothermic reaction, PI on coolant flow.

Units: L, min, mol, K, J. States y = [C, T, Tc, I].
Fault parameters: alpha (k multiplier), beta (UA multiplier).
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict

import numpy as np
from scipy.optimize import brentq

R_GAS = 8.314


@dataclass(frozen=True)
class Design:
    name: str
    V: float          # L
    Vc: float         # L jacket volume
    Q: float          # L/min feed
    Ci: float         # mol/L reactant in feed
    Ti: float         # K feed temperature
    Tci: float        # K coolant inlet
    k0: float         # 1/min
    Ea: float         # J/mol
    dH: float         # J/mol (negative = exothermic)
    rhoCp: float      # J/(L K) reacting mixture
    rhocCpc: float    # J/(L K) coolant
    UA: float         # J/(min K)
    Tsp: float        # K
    Kp: float = 0.0   # (L/min)/K
    taui: float = 10.0  # min
    Qc0: float = 0.0  # L/min bias
    Qcmin: float = 0.0
    Qcmax: float = 1.0
    Tmax_safe: float = 355.0
    sig_proc: tuple = (0.0005, 0.1, 0.1, 0.0)   # per sqrt(min), [C, T, Tc, I]
    sig_meas: tuple = (0.01, 0.2, 0.2, 2.0)     # sensor sd [C, T, Tc, Qc]

    @property
    def tau(self):
        return self.V / self.Q

    @property
    def dT_ad(self):
        return -self.dH * self.Ci / self.rhoCp


def k_of(d: Design, T, alpha=1.0):
    return alpha * d.k0 * np.exp(-d.Ea / (R_GAS * T))


# --------------------------------------------------------------------- steady states
def ss_unsat(d: Design, alpha, beta, Ci=None, Ti=None, Tci=None, Q=None):
    """Closed-loop SS with T = Tsp. Returns dict (may be infeasible -> flag)."""
    Ci = d.Ci if Ci is None else Ci; Ti = d.Ti if Ti is None else Ti
    Tci = d.Tci if Tci is None else Tci; Q = d.Q if Q is None else Q
    T = d.Tsp
    k = k_of(d, T, alpha)
    tau = d.V / Q
    C = Ci / (1 + tau * k)
    qgen = -d.dH * k * C * d.V
    qfeed = d.rhoCp * Q * (T - Ti)
    qj = qgen - qfeed                         # J/min to remove through the jacket
    UAe = beta * d.UA
    Tc = T - qj / UAe
    if Tc - Tci <= 1e-9:
        Qc = np.inf if qj > 0 else -np.inf
    else:
        Qc = qj / (d.rhocCpc * (Tc - Tci))
    return dict(C=C, T=T, Tc=Tc, Qc=Qc, qgen=qgen, qfeed=qfeed, qj=qj, X=1 - C / Ci)


def _f_open(d, T, alpha, beta, Qc, Ci, Ti, Tci, Q):
    tau = d.V / Q
    k = k_of(d, T, alpha)
    C = Ci / (1 + tau * k)
    m = Qc * d.rhocCpc
    UAe = beta * d.UA
    Tc = (UAe * T + m * Tci) / (UAe + m)
    return -d.dH * k * C * d.V + d.rhoCp * Q * (Ti - T) - UAe * (T - Tc)


def ss_open_all(d: Design, alpha, beta, Qc, Ci=None, Ti=None, Tci=None, Q=None, Tlo=250.0, Thi=520.0, n=4000):
    """All open-loop steady states at a fixed coolant flow (root scan + brentq)."""
    Ci = d.Ci if Ci is None else Ci; Ti = d.Ti if Ti is None else Ti
    Tci = d.Tci if Tci is None else Tci; Q = d.Q if Q is None else Q
    Ts = np.linspace(Tlo, Thi, n)
    f = _f_open(d, Ts, alpha, beta, Qc, Ci, Ti, Tci, Q)
    roots = []
    for i in range(n - 1):
        if f[i] == 0:
            roots.append(Ts[i])
        elif f[i] * f[i + 1] < 0:
            roots.append(brentq(lambda TT: _f_open(d, TT, alpha, beta, Qc, Ci, Ti, Tci, Q), Ts[i], Ts[i + 1]))
    out = []
    for T in roots:
        k = k_of(d, T, alpha); C = Ci / (1 + d.V / Q * k)
        m = Qc * d.rhocCpc; UAe = beta * d.UA
        Tc = (UAe * T + m * Tci) / (UAe + m)
        out.append(dict(C=C, T=T, Tc=Tc, Qc=Qc, X=1 - C / Ci))
    return out


def ss_closed(d: Design, alpha, beta, **inl):
    """Closed-loop steady state(s): unsaturated if feasible, else open-loop SS at the clamp."""
    s = ss_unsat(d, alpha, beta, **inl)
    if d.Qcmin <= s["Qc"] <= d.Qcmax:
        s["sat"] = 0
        return [s]
    Qc = d.Qcmax if s["Qc"] > d.Qcmax else d.Qcmin
    sols = ss_open_all(d, alpha, beta, Qc, **inl)
    for x in sols:
        x["sat"] = 1 if Qc == d.Qcmax else -1
    return sols


# --------------------------------------------------------------------- dynamics
def rhs(d: Design, y, alpha, beta, Ci, Ti, Tci, Q):
    """Vectorised RHS. y: (n, 4); parameters broadcastable (n,)."""
    C, T, Tc, I = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    k = alpha * d.k0 * np.exp(-d.Ea / (R_GAS * T))
    qun = d.Qc0 + d.Kp * ((T - d.Tsp) + I / d.taui)
    Qc = np.clip(qun, d.Qcmin, d.Qcmax)
    unsat = (qun > d.Qcmin) & (qun < d.Qcmax)
    UAe = beta * d.UA
    dC = Q / d.V * (Ci - C) - k * C
    dT = Q / d.V * (Ti - T) + (-d.dH) * k * C / d.rhoCp - UAe * (T - Tc) / (d.rhoCp * d.V)
    dTc = Qc / d.Vc * (Tci - Tc) + UAe * (T - Tc) / (d.rhocCpc * d.Vc)
    dI = np.where(unsat, T - d.Tsp, 0.0)
    return np.stack([dC, dT, dTc, dI], axis=1), Qc


def init_state(d, alpha, beta, Ci, Ti, Tci, Q):
    """Closed-loop steady state per sample (unsaturated branch, else the saturated root closest to Tsp)."""
    n = len(alpha)
    y = np.zeros((n, 4))
    for i in range(n):
        sols = ss_closed(d, alpha[i], beta[i], Ci=Ci[i], Ti=Ti[i], Tci=Tci[i], Q=Q[i])
        s = min(sols, key=lambda x: abs(x["T"] - d.Tsp))
        if s["sat"] == 0:
            I = (s["Qc"] - d.Qc0) * d.taui / d.Kp
        else:
            I = 0.0
        y[i] = [s["C"], s["T"], s["Tc"], I]
    return y


def simulate(d: Design, alpha, beta, *, Ci=None, Ti=None, Tci=None, Q=None, seed=0, protocol="P2",
             hist_min=300.0, burn_min=150.0, t_window=60.0, dt=0.02, dt_out=0.5,
             rate_a=None, rate_b=None, start=None):
    """Euler-Maruyama window simulation. Returns true states (n, n_t, 4) and Qc (n, n_t).

    P2: theta ramps linearly from theta_start (healthier, drawn as in ws_protocols) to theta over hist_min, then
    a window at constant theta. P1b: start at the SS of theta, noisy burn-in, then the window. Inlet constant.
    """
    rng = np.random.default_rng(seed)
    alpha = np.asarray(alpha, float); beta = np.asarray(beta, float); n = len(alpha)
    bc = lambda v, dflt: np.full(n, dflt) if v is None else np.broadcast_to(np.asarray(v, float), (n,)).copy()
    Ci, Ti, Tci, Q = bc(Ci, d.Ci), bc(Ti, d.Ti), bc(Tci, d.Tci), bc(Q, d.Q)
    if protocol == "P2":
        ra = 0.0005 if rate_a is None else rate_a      # per min (ws_protocols RATE_* order of magnitude)
        rb = 0.0005 if rate_b is None else rate_b
        u = rng.uniform(0, 1, size=(n, 2))
        a0 = np.clip(alpha + u[:, 0] * ra * hist_min, 0.4, 1.2)
        b0 = np.clip(beta + u[:, 1] * rb * hist_min, 0.4, 1.2)
        pre = int(round(hist_min / dt))
    else:
        a0, b0 = alpha.copy(), beta.copy()
        pre = int(round(burn_min / dt))
    y = init_state(d, a0, b0, Ci, Ti, Tci, Q) if start is None else start.copy()
    nwin = int(round(t_window / dt)); stride = int(round(dt_out / dt))
    sig = np.asarray(d.sig_proc) * np.sqrt(dt)
    out_y, out_q = [], []
    for s in range(pre + nwin):
        frac = min((s + 1) / pre, 1.0) if pre else 1.0
        a = a0 + (alpha - a0) * frac
        b = b0 + (beta - b0) * frac
        f, Qc = rhs(d, y, a, b, Ci, Ti, Tci, Q)
        y = y + f * dt + sig * rng.standard_normal((n, 4))
        if s >= pre and (s - pre + 1) % stride == 0:
            _, Qc_now = rhs(d, y, a, b, Ci, Ti, Tci, Q)
            out_y.append(y.copy()); out_q.append(Qc_now)
    ys = np.stack(out_y, axis=1); qs = np.stack(out_q, axis=1)
    return ys, qs


def observe(d: Design, ys, qs, seed=1, sig_meas=None):
    sm = np.asarray(d.sig_meas if sig_meas is None else sig_meas)
    rng = np.random.default_rng(seed)
    obs = np.concatenate([ys[:, :, :3], qs[:, :, None]], axis=2)
    return obs + rng.standard_normal(obs.shape) * sm


# --------------------------------------------------------------------- features
def base29(d: Design, obs, t):
    n, nt, _ = obs.shape
    means = obs.mean(1); stds = obs.std(1); mins = obs.min(1); maxs = obs.max(1)
    tc = t - t.mean()
    slopes = (tc[None, :, None] * (obs - means[:, None, :])).sum(1) / (tc ** 2).sum()
    per = np.stack([means, stds, slopes, mins, maxs], axis=2).reshape(n, -1)
    nf = int(np.ceil(0.25 * nt))
    fin = obs[:, -nf:, :].mean(1)
    dtt = t[1] - t[0]
    ierr = np.abs(obs[:, :, 1] - d.Tsp).sum(1) * dtt
    qc = obs[:, :, 3]
    sl = (qc <= d.Qcmin + 1.0).mean(1); sh = (qc >= d.Qcmax - 1.0).mean(1)
    ua_proxy = (means[:, 1] - means[:, 2]) / np.maximum(np.abs(means[:, 3]), 1e-6)
    k_proxy = np.clip(np.log(np.maximum(means[:, 0], 1e-6) / np.maximum(d.Ci - means[:, 0], 1e-6)), -8, 2)
    return np.concatenate([per, fin, np.stack([ierr, sl, sh, ua_proxy, k_proxy], 1)], axis=1)


def plugins(d: Design, obs, Ti_m, Tci_m, Ci=None, Q=None):
    """ln alpha_hat (component balance), ln beta_hat_jac (coolant balance), ln beta_hat_rx (reactor balance)."""
    Ci = d.Ci if Ci is None else Ci; Q = d.Q if Q is None else Q
    m = obs.mean(1)
    Cm, Tm, Tcm, Qcm = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    k1 = d.k0 * np.exp(-d.Ea / (R_GAS * Tm))
    a = (Q / d.V) * (Ci - Cm) / (k1 * np.maximum(Cm, 1e-9))
    dT = np.maximum(Tm - Tcm, 1e-3)
    bj = d.rhocCpc * Qcm * (Tcm - Tci_m) / (d.UA * dT)
    qgen = -d.dH * Q * (Ci - Cm)
    qflow = d.rhoCp * Q * (Ti_m - Tm)
    br = (qflow + qgen) / (d.UA * dT)
    lg = lambda v: np.log(np.clip(v, 1e-3, 1e3))
    return np.stack([lg(a), lg(bj), lg(br)], axis=1)


def features(d, obs, t, Ti_m, Tci_m):
    """Production-type 34-D: 29 summaries + 3 plug-ins (measured inlet) + measured Ti, Tci (deviation)."""
    return np.concatenate([base29(d, obs, t), plugins(d, obs, Ti_m, Tci_m),
                           np.stack([Ti_m - d.Ti, Tci_m - d.Tci], 1)], axis=1)


FEATURE_NAMES = ([f"{c}_{s}" for c in ("C", "T", "Tc", "Qc") for s in ("mean", "std", "slope", "min", "max")]
                 + [f"{c}_final_mean" for c in ("C", "T", "Tc", "Qc")]
                 + ["int_abs_T_err", "Qc_sat_low", "Qc_sat_high", "UA_eff_proxy", "k0_eff_proxy",
                    "ln_alpha_hat", "ln_beta_hat_jac", "ln_beta_hat_rx", "Ti_meas", "Tci_meas"])


# --------------------------------------------------------------------- controller
def jacobian(d, y0, alpha, beta, Ci, Ti, Tci, Q, h=1e-6):
    J = np.zeros((4, 4))
    f0, _ = rhs(d, y0[None], np.array([alpha]), np.array([beta]), np.array([Ci]), np.array([Ti]), np.array([Tci]), np.array([Q]))
    for j in range(4):
        yp = y0.copy(); yp[j] += h * max(1.0, abs(y0[j]))
        fp, _ = rhs(d, yp[None], np.array([alpha]), np.array([beta]), np.array([Ci]), np.array([Ti]), np.array([Tci]), np.array([Q]))
        J[:, j] = (fp[0] - f0[0]) / (yp[j] - y0[j])
    return J


def tune_pi(d: Design, lam_frac=1.0):
    """SIMC-type tuning from the open-loop response T(Qc) at the nominal SS (no delay, first-order fit from the
    linearisation): Kc = tau/(K*lam), tau_I = min(tau, 4 lam), lam = lam_frac * tau."""
    s = ss_unsat(d, 1.0, 1.0)
    dd = replace(d, Kp=0.0, Qc0=s["Qc"])   # open loop at nominal Qc
    y0 = np.array([s["C"], s["T"], s["Tc"], 0.0])
    J = jacobian(dd, y0, 1, 1, d.Ci, d.Ti, d.Tci, d.Q)[:3, :3]
    # input vector dQc
    B = np.array([0.0, 0.0, (d.Tci - s["Tc"]) / d.Vc])
    K = (-np.linalg.solve(J, B))[1]          # steady-state gain dT/dQc (negative)
    ev = np.linalg.eigvals(J)
    tau = 1.0 / np.min(np.abs(ev.real))      # slowest open-loop mode
    lam = lam_frac * tau
    Kc = tau / (abs(K) * lam)
    ti = min(tau, 4 * lam)
    return dict(K=K, tau=tau, Kp=Kc, taui=ti, Qc0=s["Qc"], open_loop_eigs=ev)
