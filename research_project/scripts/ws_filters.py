"""EKF / UKF baselines for System I (Phase 3 step a3 = Phase 0 item F1).

Replaces the NumPy copy embedded in notebook 16, which used the OLD controller law
(Qc = Qc0 + Kp (T - Tsp) + I/tau_i, dQc/dI = 1/tau_i, I0 = (Qc - Qc0 - Kp (T - Tsp)) tau_i). Here the constants come from
``cstr_sbi.physics`` and the controller is the standard PI law of the fixed model,

    Qc = clip(Qc0 + Kp [ (T - Tsp) + I / tau_i ]),   dI/dt = (T - Tsp) * gate   (I in K min),

so dQc/dI = Kp / tau_i and I0 = tau_i ((Qc - Qc0) / Kp - (T - Tsp)). Tests: tests/test_ws_filters.py (dynamics and
measurement equal physics.py at random states; analytic Jacobians equal finite differences).

State  x = [C, T, Tc, I, alpha, beta]            (6, "known inlet": the inlet is a fixed input)
       x = [C, T, Tc, I, alpha, beta, dTi, dTci]  (8, "augmented": constant calibration offsets of the measured inlet
                                                   temperatures, prior sd 0.1 K; the plant sees Ti = Ti_meas - dTi,
                                                   Tci = Tci_meas - dTci)
The inlet temperatures (measured, or nominal for the naive variant) and Ci enter as constants of the ``Plant``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
from scipy.linalg import cholesky, expm

import cstr_sbi.physics as ph

K0, EA, HR, RG = float(ph.K0_NOMINAL), float(ph.E_A), float(ph.H_R), float(ph.R_GAS)
RHO, CP, RHOC, CPC = float(ph.RHO), float(ph.C_P), float(ph.RHO_C), float(ph.C_PC)
VOL, VC, QF, UA0 = float(ph.V), float(ph.V_C), float(ph.Q), float(ph.UA_NOMINAL)
CI0, TI0, TCI0 = float(ph.CI_NOMINAL), float(ph.TI_NOMINAL), float(ph.TCI_NOMINAL)
TSP, KP, TAUI, QC0, QCMIN, QCMAX = (float(v) for v in (ph.TSP, ph.KP, ph.TAU_I, ph.QC0, ph.QC_MIN, ph.QC_MAX))
N3_SIGMA = np.array([0.01, 0.2, 0.2, 2.0])         # sensor sd [C, T, Tc, Qc] (N3)
PROCESS_SIGMA = np.array([0.0005, 0.1, 0.1])       # per sqrt(min), cstr_sbi.simulator.DEFAULT_PROCESS_SIGMA[:3]
EPS_PRIOR_K = 0.1                                   # prior sd of the Ti, Tci calibration offsets (nb36 production model)
N_SUB = 10                                          # RK4 substeps per observation interval


class Plant:
    """Closed-loop CSTR with known (measured) inlet constants; optionally augmented with two offset states."""

    def __init__(self, ti: float = TI0, tci: float = TCI0, ci: float = CI0, augment: bool = False):
        self.ti, self.tci, self.ci, self.augment = float(ti), float(tci), float(ci), bool(augment)
        self.n = 8 if augment else 6

    def _inlet_temps(self, x):
        if self.augment:
            return self.ti - x[6], self.tci - x[7]
        return self.ti, self.tci

    def rhs(self, x):
        C, T, Tc, Ig, al, be = x[:6]
        Ti, Tci = self._inlet_temps(x)
        k = al * K0 * np.exp(-EA / (RG * T)); UAe = be * UA0
        qc_raw = QC0 + KP * ((T - TSP) + Ig / TAUI)
        Qc = np.clip(qc_raw, QCMIN, QCMAX); gate = 1.0 if QCMIN < qc_raw < QCMAX else 0.0
        d = np.zeros(self.n)
        d[0] = (QF / VOL) * (self.ci - C) - k * C
        d[1] = (QF / VOL) * (Ti - T) - HR * k * C / (RHO * CP) - UAe * (T - Tc) / (RHO * CP * VOL)
        d[2] = (Qc / VC) * (Tci - Tc) + UAe * (T - Tc) / (RHOC * CPC * VC)
        d[3] = (T - TSP) * gate
        return d

    def h(self, x):
        C, T, Tc, Ig = x[:4]
        return np.array([C, T, Tc, np.clip(QC0 + KP * ((T - TSP) + Ig / TAUI), QCMIN, QCMAX)])

    def jac_A(self, x):
        C, T, Tc, Ig, al, be = x[:6]
        _, Tci = self._inlet_temps(x)
        ex = np.exp(-EA / (RG * T)); k = al * K0 * ex; dk_dT = k * EA / (RG * T ** 2); UAe = be * UA0
        qc_raw = QC0 + KP * ((T - TSP) + Ig / TAUI)
        sat = not (QCMIN < qc_raw < QCMAX)
        Qc = np.clip(qc_raw, QCMIN, QCMAX)
        dQc_dT = 0.0 if sat else KP
        dQc_dI = 0.0 if sat else KP / TAUI
        gate = 0.0 if sat else 1.0
        A = np.zeros((self.n, self.n))
        A[0, 0] = -QF / VOL - k
        A[0, 1] = -dk_dT * C
        A[0, 4] = -K0 * ex * C
        A[1, 0] = -HR * k / (RHO * CP)
        A[1, 1] = -QF / VOL - HR * dk_dT * C / (RHO * CP) - UAe / (RHO * CP * VOL)
        A[1, 2] = UAe / (RHO * CP * VOL)
        A[1, 4] = -HR * K0 * ex * C / (RHO * CP)
        A[1, 5] = -UA0 * (T - Tc) / (RHO * CP * VOL)
        A[2, 1] = dQc_dT * (Tci - Tc) / VC + UAe / (RHOC * CPC * VC)
        A[2, 2] = -Qc / VC - UAe / (RHOC * CPC * VC)
        A[2, 3] = dQc_dI * (Tci - Tc) / VC
        A[2, 5] = UA0 * (T - Tc) / (RHOC * CPC * VC)
        A[3, 1] = gate
        if self.augment:
            A[1, 6] = -QF / VOL            # Ti = ti - dTi
            A[2, 7] = -Qc / VC             # Tci = tci - dTci
        return A

    def jac_H(self, x):
        T, Ig = x[1], x[3]
        sat = not (QCMIN < QC0 + KP * ((T - TSP) + Ig / TAUI) < QCMAX)
        H = np.zeros((4, self.n))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        if not sat:
            H[3, 1] = KP
            H[3, 3] = KP / TAUI
        return H


def _rk4(f, x, dt):
    k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def propagate(plant: Plant, x, dt_obs: float = 0.5, n_sub: int = N_SUB):
    dt = dt_obs / n_sub
    for _ in range(n_sub):
        x = _rk4(plant.rhs, x, dt)
    return x


def make_x0(obs_row, augment: bool = False):
    """Initial state from the first observation; I0 inverts the controller; (alpha, beta) at the prior mean."""
    C0, T0, Tc0, Qc0 = (float(v) for v in obs_row)
    I0 = float(np.clip(TAUI * ((Qc0 - QC0) / KP - (T0 - TSP)), -100.0, 100.0))
    x0 = [C0, T0, Tc0, I0, 0.80, 0.80]
    return np.array(x0 + ([0.0, 0.0] if augment else []))


def default_noise(augment: bool = False, dt: float = 0.5, sigma_obs=N3_SIGMA, eps_prior: float = EPS_PRIOR_K):
    """(Q, R, P0): process noise per observation step, sensor noise, initial covariance.
    The two offset states are constants (no process noise) with prior sd ``eps_prior``."""
    q = [PROCESS_SIGMA[0] ** 2 * dt, PROCESS_SIGMA[1] ** 2 * dt, PROCESS_SIGMA[2] ** 2 * dt, 0.05 ** 2 * dt, 1e-6, 1e-6]
    p0 = [0.005 ** 2, 2.0 ** 2, 5.0 ** 2, 10.0 ** 2, 0.25 ** 2, 0.25 ** 2]
    if augment:
        q += [0.0, 0.0]; p0 += [eps_prior ** 2, eps_prior ** 2]
    return np.diag(q), np.diag(np.asarray(sigma_obs, float) ** 2), np.diag(p0)


def _bound(x):
    x[4] = np.clip(x[4], 0.3, 1.3); x[5] = np.clip(x[5], 0.3, 1.3)
    return x


def _ll_term(y, S):
    """Gaussian log-density of the innovation y with covariance S."""
    sign, logdet = np.linalg.slogdet(S)
    return -0.5 * (len(y) * np.log(2 * np.pi) + logdet + float(y @ np.linalg.solve(S, y)))


def run_ekf(plant: Plant, obs, x0, P0, Q, R, dt_obs: float = 0.5, return_ll: bool = False):
    n_t, n = len(obs), plant.n
    x, P = x0.copy(), P0.copy()
    xh = np.zeros((n_t, n)); Ph = np.zeros((n_t, n, n)); I = np.eye(n); ll = 0.0
    for k in range(n_t):
        xp = propagate(plant, x, dt_obs)
        F = expm(plant.jac_A(x) * dt_obs)
        Pp = F @ P @ F.T + Q
        H = plant.jac_H(xp)
        y = obs[k] - plant.h(xp)
        S = H @ Pp @ H.T + R
        if return_ll:
            ll += _ll_term(y, S)
        K = Pp @ H.T @ np.linalg.inv(S)
        x = _bound(xp + K @ y)
        P = (I - K @ H) @ Pp
        P = 0.5 * (P + P.T)
        xh[k], Ph[k] = x, P
    return (xh, Ph, ll) if return_ll else (xh, Ph)


def _sigma_points(x, P, kappa=0.0):
    n = len(x); c = n + kappa
    try:
        L = cholesky(c * P, lower=True)
    except np.linalg.LinAlgError:
        L = cholesky(c * (P + 1e-9 * np.eye(n)), lower=True)
    s = np.zeros((2 * n + 1, n)); s[0] = x
    for i in range(n):
        s[i + 1] = x + L[:, i]; s[n + i + 1] = x - L[:, i]
    Wm = np.full(2 * n + 1, 0.5 / c); Wm[0] = kappa / c
    return s, Wm, Wm.copy()


def run_ukf(plant: Plant, obs, x0, P0, Q, R, dt_obs: float = 0.5, return_ll: bool = False):
    n_t, n = len(obs), plant.n
    x, P = x0.copy(), P0.copy()
    xh = np.zeros((n_t, n)); Ph = np.zeros((n_t, n, n)); ll = 0.0
    for k in range(n_t):
        sg, Wm, Wc = _sigma_points(x, P)
        sp = np.array([propagate(plant, s, dt_obs) for s in sg])
        xp = Wm @ sp
        Pp = Q.copy()
        for i in range(len(sp)):
            d = sp[i] - xp; Pp += Wc[i] * np.outer(d, d)
        zs = np.array([plant.h(s) for s in sp]); zp = Wm @ zs
        Pzz = R.copy(); Pxz = np.zeros((n, 4))
        for i in range(len(sp)):
            dz = zs[i] - zp; Pzz += Wc[i] * np.outer(dz, dz); Pxz += Wc[i] * np.outer(sp[i] - xp, dz)
        if return_ll:
            ll += _ll_term(obs[k] - zp, Pzz)
        K = Pxz @ np.linalg.inv(Pzz)
        x = _bound(xp + K @ (obs[k] - zp))
        P = Pp - K @ Pzz @ K.T
        P = 0.5 * (P + P.T)
        xh[k], Ph[k] = x, P
    return (xh, Ph, ll) if return_ll else (xh, Ph)


def estimate_window(obs, method: str = "ekf", ti: float = TI0, tci: float = TCI0, ci: float = CI0, augment: bool = False,
                    sigma_obs=N3_SIGMA, eps_prior: float = EPS_PRIOR_K, q_scale: float = 1.0, r_scale: float = 1.0):
    """Filter one (n_t, 4) window [C, T, Tc, Qc] with the inlet temperatures (ti, tci) as known constants.
    Returns dict(alpha, beta, sd_alpha, sd_beta, [dTi, dTci], wall_s)."""
    plant = Plant(ti, tci, ci, augment)
    Q, R, P0 = default_noise(augment, sigma_obs=sigma_obs, eps_prior=eps_prior)
    Q, R = Q * q_scale, R * r_scale                     # tuning scale factors (notebook 16, c3); 1.0 = the untuned default
    x0 = make_x0(obs[0], augment)
    t0 = time.perf_counter()
    xh, Ph = (run_ekf if method == "ekf" else run_ukf)(plant, np.asarray(obs, float), x0, P0, Q, R)
    out = dict(alpha=xh[-1, 4], beta=xh[-1, 5], sd_alpha=float(np.sqrt(Ph[-1, 4, 4])), sd_beta=float(np.sqrt(Ph[-1, 5, 5])),
               wall_s=time.perf_counter() - t0)
    if augment:
        out.update(dTi=xh[-1, 6], dTci=xh[-1, 7], sd_dTci=float(np.sqrt(Ph[-1, 7, 7])))
    return out


def loglik_window(args):
    """Predictive log-likelihood of one window (module-level so a process pool can pickle it).
    args = (obs, method, ti, tci, augment, q_scale, r_scale). Returns the summed innovation log-density."""
    obs, method, ti, tci, augment, qs, rs = args
    plant = Plant(ti, tci, CI0, augment)
    Q, R, P0 = default_noise(augment)
    x0 = make_x0(obs[0], augment)
    try:
        out = (run_ekf if method == "ekf" else run_ukf)(plant, np.asarray(obs, float), x0, P0, Q * qs, R * rs, return_ll=True)
        return float(out[2])
    except Exception:
        return float("nan")
