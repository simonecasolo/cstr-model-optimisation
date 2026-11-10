import json, time
from dataclasses import replace
from candidates import *
from stoch import fim
designs = {"R0_SystemI": SYS_I,
           "R3_fogler_UA2.0": replace(CANDIDATES["R3_fogler_UA2.0"], Kp=50, taui=8),
           "R5_fogler_UA2.5": replace(CANDIDATES["R5_fogler_UA2.5"], Kp=40, taui=10)}
res = {}
for name, d in designs.items():
    t = time.time(); r = fim(d); r["seconds"] = round(time.time() - t)
    res[name] = r
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(res, open("c6_fim.json", "w"), indent=1)
