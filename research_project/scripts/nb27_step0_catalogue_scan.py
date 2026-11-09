"""Step 0 of docs/Manuscript/system_i_k0_fix_and_correlation_plan.md.

Scan Tsp at k0 = K0_NOMINAL/60 across the full Sc0-Sc7 catalogue AND the prior box
corners (alpha,beta in [0.4,1.2]^2), reporting valve margin / saturation per point.
Read-only: imports physics, edits nothing.
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from cstr_sbi.physics import *
from cstr_sbi.physics import K0_NOMINAL, E_A, R_GAS, Q, V, H_R, RHO, C_P, RHO_C, C_PC
from cstr_sbi.scenarios import list_configs
from cstr_sbi.priors import ALPHA_LOW, ALPHA_HIGH, BETA_LOW, BETA_HIGH

K0C = float(K0_NOMINAL) / 60.0
inlet = NOMINAL_INLET_CL

def cl(alpha, beta, tsp, k0):
    ctrl = jnp.array([KP, TAU_I, tsp, QC0, QC_MIN, QC_MAX], dtype=jnp.float64)
    p = jnp.array([UA_NOMINAL, k0, alpha, beta], dtype=jnp.float64)
    C, T, Tc, I = [float(v) for v in closed_loop_steady_state_exact(p, inlet, ctrl)]
    Qc = float(compute_qc(T, I, ctrl))
    return T, Qc

def ol_T(alpha, beta, k0, Qc=QC0):
    UA = beta * UA_NOMINAL
    def f(T):
        ke = alpha * k0 * np.exp(-E_A / (R_GAS * T)); C = (Q/V)*CI_NOMINAL/(Q/V+ke)
        Tc = (Qc*RHO_C*C_PC*TCI_NOMINAL + UA*T)/(Qc*RHO_C*C_PC + UA)
        return (Q/V)*(TI_NOMINAL-T) - H_R*ke*C/(RHO*C_P) - UA*(T-Tc)/(RHO*C_P*V)
    Ts = np.linspace(280, 400, 2401); fs = [f(t) for t in Ts]
    roots = [brentq(f, Ts[i], Ts[i+1]) for i in range(len(Ts)-1) if fs[i]*fs[i+1] < 0]
    return roots

pts = {c.name: (c.alpha, c.beta, c.mode) for c in list_configs()}
corners = {f"corner_a{a}_b{b}": (a, b, "closed_loop")
           for a, b in itertools.product([ALPHA_LOW, ALPHA_HIGH], [BETA_LOW, BETA_HIGH])}
rows = []
for tsp in np.arange(296.0, 313.0, 0.5):
    for name, (a, b, mode) in {**pts, **corners}.items():
        if mode != "closed_loop": continue
        T, Qc = cl(a, b, tsp, K0C)
        rows.append(dict(Tsp=tsp, point=name, T=T, Qc=Qc, dT=T - tsp,
                         margin=min(Qc - QC_MIN, QC_MAX - Qc),
                         saturated=abs(T - tsp) > 1e-6))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/27_step0_catalogue_scan.csv", index=False)

for label, names in [("Sc1-Sc7 closed-loop catalogue", [n for n,(a,b,m) in pts.items() if m=="closed_loop"]),
                     ("catalogue minus Sc5 (designed to saturate)", [n for n,(a,b,m) in pts.items() if m=="closed_loop" and not n.startswith("Sc5")]),
                     ("prior-box corners", list(corners)),
                     ("catalogue minus Sc5 + corners", [n for n,(a,b,m) in pts.items() if m=="closed_loop" and not n.startswith("Sc5")] + list(corners))]:
    sub = df[df.point.isin(names)]
    g = sub.groupby("Tsp").agg(worst_margin=("margin", "min"), n_sat=("saturated", "sum"))
    ok = g[g.n_sat == 0]
    print(f"\n== {label}: {len(names)} points")
    print("  Tsp with zero saturated points:", "none" if ok.empty else f"{ok.index.min()}..{ok.index.max()}")
    best = g.loc[g.worst_margin.idxmax()]
    print(f"  best worst-margin Tsp={g.worst_margin.idxmax()} margin={best.worst_margin:.2f}, n_sat={int(best.n_sat)}")
    print(g.iloc[::4].to_string())

print("\nPer-point detail at Tsp=300.5 and 300.0, 305:")
for t in (300.0, 300.5, 305.0):
    print(df[np.isclose(df.Tsp, t)][["point","T","Qc","margin","saturated"]].to_string(index=False))

print("\nOpen-loop (Qc=80 fixed) steady states, buggy vs corrected k0:")
for n in ("Sc0_open_healthy", "Sc6_open_with_fault"):
    a, b, _ = pts[n]
    print(n, "buggy T roots:", np.round(ol_T(a, b, float(K0_NOMINAL)), 2),
          " corrected T roots:", np.round(ol_T(a, b, K0C), 2))
