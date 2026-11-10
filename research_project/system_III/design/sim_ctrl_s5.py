"""Simulation check of the control options under the realistic sensor chain (S5), P2 warm start, production inlet corners."""
import sys, json, numpy as np
from dataclasses import replace
sys.path.insert(0, "src")
from system_iii.physics import PLANT
from system_iii.simulator import simulate
pts = [(1.0, 1.0), (0.75, 0.75), (0.4, 0.4), (0.5, 0.45), (0.4, 1.2), (1.0, 0.45), (1.2, 0.45), (1.2, 1.2)]
opts = {"S3 PI 50/8 f0.2 (as built)": dict(sensors="S3"),
        "S5 PI 50/8 f0.2": dict(sensors="S5"),
        "S5 PI 20/12 f0.1": dict(sensors="S5", plant=replace(PLANT, Kp=20, taui=12), pv_filter_min=0.1),
        "S5 cascade 6/6/30/1 f0.1": dict(sensors="S5", control="cascade", cascade=(6, 6, 30, 1), pv_filter_min=0.1),
        "S5 cascade 4/8/20/1 f0.1": dict(sensors="S5", control="cascade", cascade=(4, 8, 20, 1), pv_filter_min=0.1)}
out = {}
for name, kw in opts.items():
    r = []
    for dTi, dTci in [(0, 0), (-2, -2), (2, 2)]:
        for a, b in pts:
            o = simulate(np.full(100, a), np.full(100, b), seed=21, Ti=PLANT.Ti + dTi, Tci=PLANT.Tci + dTci, **kw)
            Tt = o["true"][:, :, 1]
            r.append(dict(a=a, b=b, dTi=dTi, dTci=dTci, Tsd=float(Tt.std(1).mean()), bias=float(Tt.mean() - PLANT.Tsp),
                          Qsd=float(o["qc_true"].std(1).mean()), Qm=float(o["qc_true"].mean()),
                          sat=float((o["qc_true"] >= PLANT.Qcmax - 1e-6).mean())))
    out[name] = r
    uns = [x for x in r if x["sat"] < 0.01]
    w = max(uns, key=lambda x: x["Tsd"])
    print(f"{name:28s} max Tsd(unsat)={w['Tsd']:.2f} at {w['a'],w['b'],w['dTi']}  max|bias|(unsat)={max(abs(x['bias']) for x in uns):.2f}  "
          f"max Qsd={max(x['Qsd'] for x in uns):.1f}  nominal Tsd={r[0]['Tsd']:.2f} Qsd={r[0]['Qsd']:.1f}", flush=True)
json.dump(out, open("design/sim_ctrl_s5.json", "w"), indent=1)
