"""Realism upgrade (2026-11-10): S5 = S3 noise + thermowells (0.5 min) + analyser (5 min sample-and-hold, 2 min dead time).
Checks: (1) closed-loop control over the prior corners, S3 vs S5; (2) Bayesian CRB (C6) S3 vs S5;
(3) proxy-rule accuracy (C8) under S5."""
import sys, json, time, numpy as np
sys.path.insert(0, "src")
from system_iii.simulator import simulate
from system_iii.physics import PLANT
from system_iii.information import bayes_crb
from system_iii.data import make_windows
from system_iii.features import NAMES
out = {"ctrl": {}, "crb": {}, "proxy_rule": {}}
corners = [(1, 1), (0.4, 0.4), (0.4, 1.2), (1.2, 0.4), (1.2, 1.2), (0.7, 0.7)]
for sens in ("S3", "S5"):
    r = []
    for a, b in corners:
        o = simulate(np.full(200, a), np.full(200, b), seed=11, sensors=sens)
        Tt = o["true"][:, :, 1]
        r.append(dict(alpha=a, beta=b, T_true_sd=float(Tt.std(1).mean()), T_bias=float(Tt.mean() - PLANT.Tsp),
                      Qc_sd=float(o["qc_true"].std(1).mean()), Qc_mean=float(o["qc_true"].mean()),
                      n_C_levels=float(np.mean([len(np.unique(row)) for row in o["obs"][:, :, 0]]))))
    out["ctrl"][sens] = r
    print(sens, [(d["alpha"], d["beta"], round(d["T_true_sd"], 3), round(d["T_bias"], 3), round(d["Qc_sd"], 2), d["n_C_levels"]) for d in r], flush=True)
t = time.time()
for sens in ("S3", "S5"):
    out["crb"][sens] = {"inlet_only": bayes_crb(sensors=sens, nuisances=("oTi", "oTci", "uCi", "uQ")),
                        "all_offsets": bayes_crb(sensors=sens)}
    print(sens, {k: (round(v["sd_alpha"], 4), round(v["sd_beta"], 4), round(v["corr_ab"], 3), round(v["sd_alpha_oracle"], 4), round(v["sd_beta_oracle"], 4))
                 for k, v in out["crb"][sens].items()}, round(time.time() - t), "s", flush=True)
ia, ib = NAMES["S1m"].index("ln_alpha_hat"), NAMES["S1m"].index("ln_beta_hat_jac")
pts = {"Sc1": (1, 1), "Sc2": (1, .7), "Sc3": (.7, 1), "Sc4": (.75, .75), "Sc5": (1, .87), "Sc6": (.87, 1), "Sc7": (1.1, .4)}
for i, (k, (a, b)) in enumerate(pts.items()):
    d = make_windows(np.tile([a, b], (400, 1)), 900 + i, conditions="production", sensors="S5")
    ah, bh = np.exp(d["x"][:, ia]), np.exp(d["x"][:, ib])
    ok = ((ah < .9) == (a < .9)) & ((bh < .9) == (b < .9))
    out["proxy_rule"][k] = dict(acc=float(ok.mean()), a_mean=float(ah.mean()), a_sd=float(ah.std()), b_mean=float(bh.mean()), b_sd=float(bh.std()))
    print(k, {kk: round(vv, 4) for kk, vv in out["proxy_rule"][k].items()}, flush=True)
json.dump(out, open("design/verify_s5.json", "w"), indent=1)
