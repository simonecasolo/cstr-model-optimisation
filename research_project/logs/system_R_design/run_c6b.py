import json, sys
from dataclasses import replace
from candidates import *
from stoch import fim
designs = {"R0_SystemI": SYS_I, "R3_fogler_UA2.0": replace(CANDIDATES["R3_fogler_UA2.0"], Kp=50, taui=8),
           "R5_fogler_UA2.5": replace(CANDIDATES["R5_fogler_UA2.5"], Kp=40, taui=10)}
res = {}
for name, d in designs.items():
    res[name] = {}
    for pt in [(1.0, 1.0), (0.75, 0.75), (1.0, 0.6)]:
        r = fim(d, *pt, extended=True)
        res[name][str(pt)] = r
        print(name, pt, {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and k != "cond_cov"}, flush=True)
json.dump(res, open("c6_fim_extended.json", "w"), indent=1)
