"""Tune the digital PI (DCS execution every 1.2 s, PV filter, measured T with 0.2 K white noise) by simulation."""
import sys, itertools, numpy as np
from dataclasses import replace
sys.path.insert(0, "src")
from system_iii.physics import PLANT
from system_iii.simulator import simulate
SENS = sys.argv[2] if len(sys.argv) > 2 else "S3"
pts = [(1.0, 1.0), (0.75, 0.75), (1.2, 0.45), (0.4, 1.2), (0.4, 0.45), (0.5, 0.45)]
grid = [(50, 8, 0.2), (50, 8, 0.5), (30, 10, 0.2), (30, 10, 0.5), (80, 8, 0.2), (20, 12, 0.2)] if len(sys.argv) < 2 else eval(sys.argv[1])
for Kp, ti, tf in grid:
    p = replace(PLANT, Kp=Kp, taui=ti)
    res = []
    for a, b in pts:
        o = simulate(np.full(150, a), np.full(150, b), seed=7, plant=p, pv_filter_min=tf, protocol="P1b", sensors=SENS)
        Tt = o["true"][:, :, 1]
        res.append((a, b, Tt.std(1).mean(), Tt.mean() - p.Tsp, o["qc_true"].std(1).mean(), o["qc_true"].mean()))
    print(f"Kp={Kp} ti={ti} filt={tf}: " + "  ".join(f"({a},{b}) Tsd={sd:.2f} dT={m:+.2f} Qsd={qs:.1f}/{qm:.0f}" for a, b, sd, m, qs, qm in res), flush=True)
