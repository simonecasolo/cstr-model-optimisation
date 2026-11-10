import json, sys
from dataclasses import replace
from candidates import *
from stoch import proxy_accuracy, cross_sensitivity, summary_robustness
name = sys.argv[1]
d = {"R3": replace(CANDIDATES["R3_fogler_UA2.0"], Kp=50, taui=8),
     "R5": replace(CANDIDATES["R5_fogler_UA2.5"], Kp=40, taui=10), "R0": SYS_I}[name]
pts = [(1.05, 1.05), (1.05, 0.70), (0.70, 1.05), (0.75, 0.75), (1.0, 0.88), (0.88, 1.0), (1.0, 0.45)]
acc = proxy_accuracy(d, pts)
for r in acc: print(name, {k: round(v, 4) for k, v in r.items()}, flush=True)
cs = cross_sensitivity(d); print(name, "cross", {k: (round(a, 3), round(b, 3)) for k, (a, b) in cs.items()}, flush=True)
sr = summary_robustness(d); print(name, "robust", sr, flush=True)
json.dump(dict(acc=acc, cross=cs, robust=sr), open(f"c8_{name}.json", "w"), indent=1, default=float)
