"""System III physics: jacketed CSTR for propylene-glycol production (acid-catalysed hydrolysis of
propylene oxide, first order in PO), PI control of the reactor temperature with the coolant flow.

Source of the plant data: Fogler, Elements of Chemical Reaction Engineering, 3rd ed., Example 9-4, as
reproduced in COMSOL's "Startup of a Continuous Stirred Tank Reactor" model: V = 1.89 m3,
v = 3.47e-3 m3/s, c_PO = 2903 mol/m3 (with water 36291 and methanol 3629 mol/m3), Tf = 297 K,
A = 4.71e9 1/s, E = 75.36 kJ/mol, UA = 8441 W/K, coolant water at 289 K; steady state 337 K.
Design changes (docs/system_III_design.md): UA = 2 x Fogler (fouling allowance; jacket + coil), coolant
valve sized so the nominal flow is 35% of maximum, PI controller re-tuned (Kp = 50 (L/min)/K, tau_I = 8 min).

Units: L, min, mol, K, J. States y = [C, T, Tc, I]. Faults: alpha (k multiplier, catalyst activity),
beta (UA multiplier, jacket fouling). All functions are numpy and vectorised where it matters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

R_GAS = 8.314
RHOCP_FEED = 2.903 * 146.5 + 36.291 * 75.4 + 3.629 * 81.6     # J/(L K), molar heat capacities x feed composition


@dataclass(frozen=True)
class Plant:
    V: float = 1890.0                 # L
    Vc: float = 190.0                 # L jacket (+ coil) holdup
    Q: float = 3.47e-3 * 60_000       # L/min feed (208.2)
    Ci: float = 2.903                 # mol/L PO in the feed
    Ti: float = 297.0                 # K feed temperature
    Tci: float = 289.0                # K coolant (chilled water) inlet
    k0: float = 4.71e9 * 60.0         # 1/min (Fogler / Furusawa 1969: 1.696e13 1/h)
    Ea: float = 75362.0               # J/mol
    dH: float = -84666.0              # J/mol (-36 400 Btu/lbmol)
    rhoCp: float = RHOCP_FEED         # J/(L K)
    rhocCpc: float = 4184.0           # J/(L K) water
    UA: float = 2.0 * 8441.0 * 60.0   # J/(min K): 2 x Fogler's 8441 W/K
    Tsp: float = 337.0                # K setpoint (Fogler's steady state)
    Kp: float = 50.0                  # (L/min)/K
    taui: float = 8.0                 # min
    Qcmin: float = 0.0                # L/min
    Qcmax: float = 288.42             # L/min (nominal / 0.35)
    Qc0: float = 100.95               # L/min bias = nominal steady-state flow
    Tmax_safe: float = 355.0          # K (Fogler)

    @property
    def tau(self):
        return self.V / self.Q


PLANT = Plant()


def k_of(p: Plant, T, alpha=1.0):
    return alpha * p.k0 * np.exp(-p.Ea / (R_GAS * T))


def ss_unsat(p: Plant, alpha, beta, Ci=None, Ti=None, Tci=None, Q=None):
    """Closed-loop steady state with T = Tsp (vectorised). Qc may fall outside the valve range:
    callers check. Returns dict of arrays."""
    Ci = p.Ci if Ci is None else Ci; Ti = p.Ti if Ti is None else Ti
    Tci = p.Tci if Tci is None else Tci; Q = p.Q if Q is None else Q
    alpha = np.asarray(alpha, float); beta = np.asarray(beta, float)
    T = np.full(np.broadcast(alpha, beta, np.asarray(Ci), np.asarray(Ti), np.asarray(Tci), np.asarray(Q)).shape, p.Tsp)
    k = k_of(p, T, alpha)
    C = Ci / (1 + p.V / Q * k)
    qgen = -p.dH * k * C * p.V
    qj = qgen - p.rhoCp * Q * (T - Ti)
    Tc = T - qj / (beta * p.UA)
    with np.errstate(divide="ignore", invalid="ignore"):
        Qc = np.where(Tc - Tci > 1e-9, qj / (p.rhocCpc * (Tc - Tci)), np.inf)
    return dict(C=C, T=T, Tc=Tc, Qc=Qc, qj=qj, qgen=qgen, X=1 - C / Ci)


def _f_open(p, T, alpha, beta, Qc, Ci, Ti, Tci, Q):
    k = k_of(p, T, alpha)
    C = Ci / (1 + p.V / Q * k)
    m = Qc * p.rhocCpc; UAe = beta * p.UA
    Tc = (UAe * T + m * Tci) / (UAe + m)
    return -p.dH * k * C * p.V + p.rhoCp * Q * (Ti - T) - UAe * (T - Tc)


def ss_open_all(p: Plant, alpha, beta, Qc, Ci=None, Ti=None, Tci=None, Q=None, Tlo=250.0, Thi=520.0, n=4000):
    """All steady states at a fixed coolant flow (scalar inputs)."""
    Ci = p.Ci if Ci is None else Ci; Ti = p.Ti if Ti is None else Ti
    Tci = p.Tci if Tci is None else Tci; Q = p.Q if Q is None else Q
    Ts = np.linspace(Tlo, Thi, n)
    f = _f_open(p, Ts, alpha, beta, Qc, Ci, Ti, Tci, Q)
    out = []
    for i in np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0):
        T = brentq(lambda TT: _f_open(p, TT, alpha, beta, Qc, Ci, Ti, Tci, Q), Ts[i], Ts[i + 1])
        k = k_of(p, T, alpha); C = Ci / (1 + p.V / Q * k)
        m = Qc * p.rhocCpc; UAe = beta * p.UA
        out.append(dict(C=C, T=T, Tc=(UAe * T + m * Tci) / (UAe + m), Qc=Qc, X=1 - C / Ci))
    return out


def ss_closed(p: Plant, alpha, beta, **inl):
    """Closed-loop steady state(s) for scalar inputs: unsaturated if feasible, else the open-loop
    steady states at the valve clamp (sat = +1 high, -1 low)."""
    s = {k: float(v) for k, v in ss_unsat(p, alpha, beta, **inl).items()}
    if p.Qcmin <= s["Qc"] <= p.Qcmax:
        s["sat"] = 0
        return [s]
    Qc = p.Qcmax if s["Qc"] > p.Qcmax else p.Qcmin
    sols = ss_open_all(p, alpha, beta, Qc, **inl)
    for x in sols:
        x["sat"] = 1 if Qc == p.Qcmax else -1
    return sols


def beta_saturation(p: Plant, alpha=1.0, **inl):
    """Fouling level at which the valve reaches Qc_max (unsaturated branch)."""
    g = lambda b: float(ss_unsat(p, alpha, b, **inl)["Qc"]) - p.Qcmax
    return brentq(g, 0.05, 1.5) if g(1.5) < 0 < g(0.05) else np.nan
