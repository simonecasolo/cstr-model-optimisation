import numpy as np, itertools, json
from dataclasses import replace
from design_lib import *
from candidates import *

def closed_loop_eigs(d, a, b, **inl):
    s = ss_unsat(d, a, b, **inl)
    if not (d.Qcmin < s["Qc"] < d.Qcmax): return None
    I = (s["Qc"] - d.Qc0) * d.taui / d.Kp
    y0 = np.array([s["C"], s["T"], s["Tc"], I])
    Ci = inl.get("Ci", d.Ci); Ti = inl.get("Ti", d.Ti); Tci = inl.get("Tci", d.Tci); Q = inl.get("Q", d.Q)
    return np.linalg.eigvals(jacobian(d, y0, a, b, Ci, Ti, Tci, Q))

def check(d):
    grid = np.linspace(0.4, 1.2, 17)
    corners = [dict()] + [dict(Ti=d.Ti+x, Tci=d.Tci+y) for x, y in itertools.product([-2, 2], [-2, 2])]
    maxre, slow, osc, n = -np.inf, 0, 0, 0
    for inl in corners:
        for a in grid:
            for b in grid:
                ev = closed_loop_eigs(d, a, b, **inl)
                if ev is None: continue
                n += 1
                maxre = max(maxre, ev.real.max())
                slow = max(slow, 1 / np.abs(ev.real).min())
                damp = np.min(-ev.real / np.abs(ev))
                osc = max(osc, 1 - damp)
    return dict(n_unsat=n, max_real_eig=maxre, slowest_tau_min=slow, worst_1_minus_damping=osc)

if __name__ == "__main__":
  out = {}
  for name in ["R3_fogler_UA2.0", "R5_fogler_UA2.5"]:
      d0 = CANDIDATES[name]
      tp = tune_pi(d0)
      print(name, "open-loop: K=%.4f K/(L/min), slowest tau=%.2f min, eigs" % (tp["K"], tp["tau"]), np.round(tp["open_loop_eigs"], 3))
      for lam in (0.5, 1.0, 2.0):
          t = tune_pi(d0, lam_frac=lam)
          d = replace(d0, Kp=t["Kp"], taui=t["taui"])
          r = check(d); r.update(lam_frac=lam, Kp=t["Kp"], taui=t["taui"])
          print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
          out[f"{name}_lam{lam}"] = r
  json.dump(out, open("ctrl_checks.json", "w"), indent=1, default=float)
