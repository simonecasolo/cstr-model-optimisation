"""S5 sensors + cascade 4/8/20/1 (PV filter 0.1): C3 linear check on the full grid, C6 CRB, C8 proxy rule."""
import sys, json, time, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "design")
from check_cascade import check
from system_iii.information import bayes_crb
from system_iii.data import make_windows
from system_iii.features import NAMES
KW = dict(control="cascade", cascade=(4, 8, 20, 1), pv_filter_min=0.1)
out = {"C3": check(4, 8, 20, 1, tw=0.5, tf=0.1, ngrid=17)}
print("C3", out["C3"], flush=True)
out["crb"] = {"inlet_only": bayes_crb(sensors="S5", nuisances=("oTi", "oTci", "uCi", "uQ"), **KW), "all_offsets": bayes_crb(sensors="S5", **KW),
              "T_only": bayes_crb(sensors="S5", feature_set="T_only", nuisances=("oTi", "oTci", "uCi", "uQ"), **KW)}
for k, v in out["crb"].items(): print(k, round(v["sd_alpha"], 4), round(v["sd_beta"], 4), round(v["corr_ab"], 3), flush=True)
ia, ib = NAMES["S1m"].index("ln_alpha_hat"), NAMES["S1m"].index("ln_beta_hat_jac")
pts = {"Sc1": (1, 1), "Sc2": (1, .7), "Sc3": (.7, 1), "Sc4": (.75, .75), "Sc5": (1, .87), "Sc6": (.87, 1), "Sc7": (1.1, .4)}
out["proxy_rule"] = {}
for i, (k, (a, b)) in enumerate(pts.items()):
    d = make_windows(np.tile([a, b], (400, 1)), 900 + i, conditions="production", sensors="S5", **KW)
    ah, bh = np.exp(d["x"][:, ia]), np.exp(d["x"][:, ib])
    ok = ((ah < .9) == (a < .9)) & ((bh < .9) == (b < .9))
    out["proxy_rule"][k] = dict(acc=float(ok.mean()), a_mean=float(ah.mean()), a_sd=float(ah.std()), b_mean=float(bh.mean()), b_sd=float(bh.std()))
    print(k, {kk: round(vv, 4) for kk, vv in out["proxy_rule"][k].items()}, flush=True)
json.dump(out, open("design/verify_s5_cascade.json", "w"), indent=1, default=float)
