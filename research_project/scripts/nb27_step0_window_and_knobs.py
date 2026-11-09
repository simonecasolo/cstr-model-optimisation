"""Step 0b: reachable steady-state T window [T(Qc_max), T(Qc_min)] per point at k0/60,
and a single-knob search (Ci, Ti, UA, Tci) for a common non-saturated Tsp. Read-only."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from cstr_sbi.physics import *
from cstr_sbi.physics import K0_NOMINAL
from cstr_sbi.scenarios import list_configs
from cstr_sbi.priors import ALPHA_LOW, ALPHA_HIGH, BETA_LOW, BETA_HIGH
K0C = float(K0_NOMINAL) / 60.0

def window(a, b, k0, Ci=CI_NOMINAL, Ti=TI_NOMINAL, Tci=TCI_NOMINAL, UA=UA_NOMINAL):
    inlet = jnp.array([Ci, Ti, Tci]); out = []
    for qc in (QC_MAX, QC_MIN):  # force saturation at each bound via Tsp far beyond reach
        tsp = 200.0 if qc == QC_MAX else 450.0
        ctrl = jnp.array([KP, TAU_I, tsp, QC0, QC_MIN, QC_MAX])
        p = jnp.array([UA, k0, a, b])
        out.append(float(closed_loop_steady_state_exact(p, inlet, ctrl)[1]))
    return out  # [T_lo (max cooling), T_hi (no cooling)]

cat = [(c.name, c.alpha, c.beta) for c in list_configs() if c.mode == "closed_loop"]
corners = [(f"corner a{a} b{b}", a, b) for a, b in itertools.product([ALPHA_LOW, ALPHA_HIGH], [BETA_LOW, BETA_HIGH])]

def report(label, pts, **kw):
    ws = {n: window(a, b, K0C, **kw) for n, a, b in pts}
    lo = max(w[0] for w in ws.values()); hi = min(w[1] for w in ws.values())
    return ws, lo, hi

print("== Reachable T window per point, k0/60, nominal feed/UA ==")
ws, lo, hi = report("base", cat + corners)
for n, w in ws.items(): print(f"  {n:24s} [{w[0]:.2f}, {w[1]:.2f}]  width {w[1]-w[0]:.2f} K")
print(f"  common intersection: [{lo:.2f}, {hi:.2f}] -> {'EMPTY' if lo>=hi else 'width %.2f K'%(hi-lo)}")
ws, lo, hi = report("cat-only", cat)
print(f"  catalogue-only (incl. Sc5) intersection: [{lo:.2f}, {hi:.2f}]")
ws, lo, hi = report("cat-no-sc5", [p for p in cat if not p[0].startswith('Sc5')])
print(f"  catalogue minus Sc5 intersection: [{lo:.2f}, {hi:.2f}]")

print("\n== Buggy k0 for reference ==")
wsb = {n: window(a, b, float(K0_NOMINAL)) for n, a, b in cat + corners}
lob = max(w[0] for w in wsb.values()); hib = min(w[1] for w in wsb.values())
print(f"  common intersection (all pts): [{lob:.2f}, {hib:.2f}]; Sc5 window {wsb['Sc5_closed_saturated']}")

print("\n== Single-knob search at k0/60 (all 6 catalogue incl. Sc5 + 4 corners; want intersection width >= 6 K) ==")
res = []
for knob, vals in [("Ci", [0.97, 1.5, 2.0, 2.5, 3.0]), ("Ti", [297, 300, 303, 306, 309]),
                   ("UA", [1.25e4, 2e4, 3e4, 5e4]), ("Tci", [290, 293, 297, 300])]:
    for v in vals:
        kw = {knob: v}
        ws, lo, hi = report("k", cat + corners, **kw)
        ws2, lo2, hi2 = report("k", [p for p in cat if not p[0].startswith('Sc5')], **kw)
        res.append((knob, v, lo, hi, hi - lo, lo2, hi2, hi2 - lo2))
        print(f"  {knob}={v:<8}: all-pts [{lo:.1f},{hi:.1f}] w={hi-lo:6.2f} | no-Sc5,no-corners [{lo2:.1f},{hi2:.1f}] w={hi2-lo2:6.2f}")
