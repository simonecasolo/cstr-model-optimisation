"""Re-verify the design headline numbers with the System III package simulator (digital PI on measured T)."""
import sys, json, time, numpy as np
sys.path.insert(0, "src")
from system_iii.information import bayes_crb
from system_iii.data import make_windows
from system_iii.features import NAMES
out = {}
t = time.time()
out["crb_inlet_only"] = bayes_crb(nuisances=("oTi", "oTci", "uCi", "uQ"))
out["crb_all_offsets"] = bayes_crb()
print({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items()}, round(time.time() - t), "s", flush=True)
ia, ib = NAMES["S1m"].index("ln_alpha_hat"), NAMES["S1m"].index("ln_beta_hat_jac")
pts = {"Sc1": (1, 1), "Sc2": (1, .7), "Sc3": (.7, 1), "Sc4": (.75, .75), "Sc5": (1, .87), "Sc6": (.87, 1), "Sc7": (1.1, .4)}
acc = {}
for k, (a, b) in pts.items():
    d = make_windows(np.tile([a, b], (400, 1)), 900 + len(acc), conditions="production")
    ah, bh = np.exp(d["x"][:, ia]), np.exp(d["x"][:, ib])
    ok = ((ah < .9) == (a < .9)) & ((bh < .9) == (b < .9))
    acc[k] = dict(acc=float(ok.mean()), a_mean=float(ah.mean()), a_sd=float(ah.std()), b_mean=float(bh.mean()), b_sd=float(bh.std()))
    print(k, {kk: round(vv, 4) for kk, vv in acc[k].items()}, flush=True)
out["proxy_rule"] = acc
json.dump(out, open("design/verify_package.json", "w"), indent=1)
