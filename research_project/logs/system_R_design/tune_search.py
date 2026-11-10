import numpy as np, itertools, sys
from dataclasses import replace
from design_lib import *
from candidates import *
from check_ctrl import closed_loop_eigs

def worst(d, alo=0.4, grid_n=9, corners=True):
    grid_a = np.linspace(alo, 1.2, grid_n); grid_b = np.linspace(0.4, 1.2, grid_n)
    cs = [dict()] + ([dict(Ti=d.Ti+x, Tci=d.Tci+y) for x, y in itertools.product([-2, 2], [-2, 2])] if corners else [])
    mx, mind, arg = -np.inf, 1.0, None
    for inl in cs:
        for a in grid_a:
            for b in grid_b:
                ev = closed_loop_eigs(d, a, b, **inl)
                if ev is None: continue
                r = ev.real.max()
                if r > mx: mx, arg = r, (a, b, inl)
                mind = min(mind, np.min(-ev.real / np.abs(ev)))
    return mx, mind, arg

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "R3_fogler_UA2.0"
    d0 = CANDIDATES[name]
    best = []
    for Kp in [50, 65, 80, 100, 130]:
        for ti in [6, 8, 10, 13, 16]:
            d = replace(d0, Kp=Kp, taui=ti)
            for alo in (0.4, 0.5):
                mx, md, arg = worst(d, alo)
                best.append((alo, Kp, ti, mx, md))
    for alo in (0.4, 0.5):
        rows = sorted([b for b in best if b[0] == alo], key=lambda r: r[3])[:6]
        print("alpha_low", alo)
        for r in rows: print(f"   Kp={r[1]:5.1f} taui={r[2]:5.1f}  max Re(eig)={r[3]:.4f} (slowest tau {1/abs(r[3]):.1f} min)  min damping={r[4]:.3f}")
