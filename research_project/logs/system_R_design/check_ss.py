import numpy as np, itertools, json
from scipy.optimize import brentq
from design_lib import *
from candidates import *

def beta_sat(d, alpha=1.0, **inl):
    g = lambda b: ss_unsat(d, alpha, b, **inl)["Qc"] - d.Qcmax
    lo, hi = 0.05, 1.5
    if g(hi) > 0: return np.nan
    if g(lo) < 0: return lo
    return brentq(g, lo, hi)

def scan(d):
    grid = np.linspace(0.4, 1.2, 17)
    corners = [dict()] + [dict(Ti=d.Ti+a, Tci=d.Tci+b, Ci=d.Ci*(1+c), Q=d.Q*(1+q))
                          for a, b, c, q in itertools.product([-2, 2], [-2, 2], [-0.02, 0.02], [-0.01, 0.01])]
    nmult = 0; Tmax = 0; nsat_hi = 0; nsat_lo = 0; worst = None; n = 0
    for inl in corners:
        for a in grid:
            for b in grid:
                sols = ss_closed(d, a, b, **inl); n += 1
                if len(sols) != 1:
                    nmult += 1; worst = (a, b, inl, [round(s["T"], 1) for s in sols])
                for s in sols:
                    Tmax = max(Tmax, s["T"])
                    nsat_hi += s["sat"] == 1; nsat_lo += s["sat"] == -1
    lowsat_any = False
    for b in grid:
        for inl in corners:
            if ss_unsat(d, 0.4, b, **inl)["Qc"] < d.Qcmin: lowsat_any = True
    return dict(n_points=n, n_multiple=nmult, example_multiple=str(worst), Tmax_all_ss=round(Tmax, 1),
                frac_sat_high=round(nsat_hi / n, 3), lowsat_alpha04=lowsat_any,
                beta_sat_nominal=round(beta_sat(d), 3),
                beta_sat_worst_inlet=round(max(beta_sat(d, **c) for c in corners[1:]), 3))

def sens(d):
    """Deterministic error of the plug-in beta estimates (relative, ~ abs at beta=1) for inlet errors."""
    s = ss_unsat(d, 1.0, 1.0)
    obs = np.array([[[s["C"], s["T"], s["Tc"], s["Qc"]]] * 2])
    base = plugins(d, obs, np.array([d.Ti]), np.array([d.Tci]))[0]
    out = {}
    # true Tci shifted by +0.5, analyst assumes nominal: plant SS at true inlet, estimator with nominal
    for lab, inl, meas in [("Tci+0.5", dict(Tci=d.Tci + 0.5), None), ("Ti+0.5", dict(Ti=d.Ti + 0.5), None),
                           ("Ci+1%", dict(Ci=d.Ci * 1.01), None), ("Q+0.5%", dict(Q=d.Q * 1.005), None)]:
        s2 = ss_unsat(d, 1.0, 1.0, **inl)
        o2 = np.array([[[s2["C"], s2["T"], s2["Tc"], s2["Qc"]]] * 2])
        e = plugins(d, o2, np.array([d.Ti]), np.array([d.Tci]))[0] - base
        out[lab] = dict(dln_alpha=round(e[0], 4), dln_beta_jac=round(e[1], 4), dln_beta_rx=round(e[2], 4))
    return out

res = {}
for d in [SYS_I] + list(CANDIDATES.values()):
    r = scan(d); r["sens"] = sens(d); res[d.name] = r
    print(d.name, json.dumps({k: v for k, v in r.items() if k != "sens"}))
    print("   ", r["sens"])
json.dump(res, open("ss_checks.json", "w"), indent=1, default=str)
