"""Linear check of a cascade loop (outer: reactor T -> jacket-outlet setpoint; inner: jacket-outlet Tc -> coolant
flow), with thermowells on T and Tc and a PV filter on the outer loop. State [C, T, Tc, Tw, Tcw, Tf, Io, Ii]."""
import sys, itertools, json, numpy as np
sys.path.insert(0, "src")
from system_iii.physics import PLANT, k_of, ss_unsat

def jac_eigs(p, a, b, Ko, To, Ki, Ti_, tw, tf, dTi=0.0, dTci=0.0):
    Ci, Tin, Tci, Q = p.Ci, p.Ti + dTi, p.Tci + dTci, p.Q
    s = ss_unsat(p, np.array([a]), np.array([b]), Ci=np.array([Ci]), Ti=np.array([Tin]), Tci=np.array([Tci]), Q=np.array([Q]))
    qc0 = float(s["Qc"][0]); tc0 = float(s["Tc"][0])
    if not (p.Qcmin < qc0 < p.Qcmax): return None, qc0
    def f(y):
        C, T, Tc, Tw, Tcw, Tf, Io, Ii = y
        tcsp = tc0 - Ko * ((Tf - p.Tsp) + Io / To)
        qc = qc0 + Ki * ((Tcw - tcsp) + Ii / Ti_)
        k = k_of(p, T, a); UAe = b * p.UA
        return np.array([Q / p.V * (Ci - C) - k * C,
                         Q / p.V * (Tin - T) + (-p.dH) * k * C / p.rhoCp - UAe * (T - Tc) / (p.rhoCp * p.V),
                         qc / p.Vc * (Tci - Tc) + UAe * (T - Tc) / (p.rhocCpc * p.Vc),
                         (T - Tw) / tw, (Tc - Tcw) / tw, (Tw - Tf) / tf, Tf - p.Tsp, Tcw - tcsp])
    y0 = np.array([s["C"][0], s["T"][0], tc0, p.Tsp, tc0, p.Tsp, 0.0, 0.0])
    f0 = f(y0); J = np.zeros((8, 8))
    for j in range(8):
        h = 1e-6 * max(1, abs(y0[j])); yp = y0.copy(); yp[j] += h; J[:, j] = (f(yp) - f0) / h
    return np.linalg.eigvals(J), qc0

def check(Ko, To, Ki, Ti_, tw=0.5, tf=0.1, ngrid=9):
    g = np.linspace(0.4, 1.2, ngrid); slow, wd, mx, worst = 0, 1, -9, None
    for dTi, dTci in [(0, 0)] + list(itertools.product([-2, 2], [-2, 2])):
        for a in g:
            for b in g:
                ev, _ = jac_eigs(PLANT, a, b, Ko, To, Ki, Ti_, tw, tf, dTi, dTci)
                if ev is None: continue
                t = 1 / np.abs(ev.real).min()
                if t > slow: slow, worst = t, (float(a), float(b), dTi, dTci)
                wd = min(wd, float(np.min(-ev.real / np.abs(ev)))); mx = max(mx, float(ev.real.max()))
    return dict(Ko=Ko, To=To, Ki=Ki, Ti=Ti_, tw=tw, tf=tf, slowest_tau_min=float(slow), min_damping=wd, max_real=mx, worst=worst)

if __name__ == "__main__":
    print("Vc", PLANT.Vc)
    res = []
    for Ko, To, Ki, Ti_ in itertools.product([1, 2, 4], [8, 12], [5, 10, 20], [1, 2]):
        r = check(Ko, To, Ki, Ti_); res.append(r)
    res.sort(key=lambda r: r["slowest_tau_min"] if r["max_real"] < 0 else 1e9)
    for r in res[:12]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
