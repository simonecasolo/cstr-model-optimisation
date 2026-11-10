"""Linear closed-loop check (C3) with the realistic sensor chain: thermowell (tw) + PV filter (tf) + PI on the
lagged reading. Augmented state [C, T, Tc, Tw, Tf, I]; DCS execution (1.2 s) treated as continuous."""
import sys, itertools, json, numpy as np
from dataclasses import replace
sys.path.insert(0, "src")
from system_iii.physics import PLANT, k_of, ss_unsat

def rhs(p, y, a, b, Ci, Ti, Tci, Q, tw, tf):
    C, T, Tc, Tw, Tf, I = y
    qc = p.Qc0 + p.Kp * ((Tf - p.Tsp) + I / p.taui)
    k = k_of(p, T, a); UAe = b * p.UA
    return np.array([Q / p.V * (Ci - C) - k * C,
                     Q / p.V * (Ti - T) + (-p.dH) * k * C / p.rhoCp - UAe * (T - Tc) / (p.rhoCp * p.V),
                     qc / p.Vc * (Tci - Tc) + UAe * (T - Tc) / (p.rhocCpc * p.Vc),
                     (T - Tw) / tw, (Tw - Tf) / tf, Tf - p.Tsp])

def eigs(p, a, b, tw, tf, Ci, Ti, Tci, Q):
    s = ss_unsat(p, np.array([a]), np.array([b]), Ci=np.array([Ci]), Ti=np.array([Ti]), Tci=np.array([Tci]), Q=np.array([Q]))
    qc = float(s["Qc"][0])
    if not (p.Qcmin < qc < p.Qcmax): return None
    y0 = np.array([s["C"][0], s["T"][0], s["Tc"][0], p.Tsp, p.Tsp, (qc - p.Qc0) * p.taui / p.Kp])
    J = np.zeros((6, 6)); f0 = rhs(p, y0, a, b, Ci, Ti, Tci, Q, tw, tf)
    for j in range(6):
        h = 1e-6 * max(1, abs(y0[j])); yp = y0.copy(); yp[j] += h
        J[:, j] = (rhs(p, yp, a, b, Ci, Ti, Tci, Q, tw, tf) - f0) / h
    return np.linalg.eigvals(J)

def check(Kp, ti, tw, tf):
    p = replace(PLANT, Kp=Kp, taui=ti); g = np.linspace(0.4, 1.2, 17)
    mx, slow, wd, n = -np.inf, 0, 1, 0
    for dTi, dTci in [(0, 0)] + list(itertools.product([-2, 2], [-2, 2])):
        for a in g:
            for b in g:
                ev = eigs(p, a, b, tw, tf, p.Ci, p.Ti + dTi, p.Tci + dTci, p.Q)
                if ev is None: continue
                n += 1; mx = max(mx, ev.real.max()); slow = max(slow, 1 / np.abs(ev.real).min())
                wd = min(wd, np.min(-ev.real / np.abs(ev)))
    return dict(Kp=Kp, taui=ti, tw=tw, pv_filter=tf, n_unsat=n, max_real_eig=mx, slowest_tau_min=slow, min_damping=wd)

if __name__ == "__main__":
    out = [check(*c) for c in [(50, 8, 1e-3, 0.2), (50, 8, 0.5, 0.2), (35, 10, 0.5, 0.1), (25, 10, 0.5, 0.1), (20, 12, 0.5, 0.1)]]
    for r in out: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(out, open("design/ctrl_checks_s5.json", "w"), indent=1, default=float)
