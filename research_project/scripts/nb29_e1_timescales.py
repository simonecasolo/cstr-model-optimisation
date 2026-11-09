"""nb29 / E1: sanity + time scales (no training). Writes results/29_e1_*.csv, figures/29_e1_*.png"""
import sys, numpy as np, pandas as pd, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ws_protocols as ws
from cstr_sbi.physics import cstr_closed_loop_rhs, NOMINAL_INLET_CL as inlet, NOMINAL_CTRL as ctrl, QC_MIN, QC_MAX
from cstr_sbi.scenarios import SCENARIO_CONFIGS
R, F = ROOT/"results", ROOT/"figures"
ZERO = jnp.zeros(4)

def rhs64(y, a, b):
    """float64 NumPy copy of cstr_closed_loop_rhs (standard-form PI, conditional-integration anti-windup)."""
    import cstr_sbi.physics as ph
    Kp, ti, Tsp, Qc0, Qmin, Qmax = [float(x) for x in ctrl]
    C, T, Tc, I = y
    qu = Qc0 + Kp*((T - Tsp) + I/ti); Qc = min(max(qu, Qmin), Qmax)
    ke = a*float(ph.K0_NOMINAL)*np.exp(-ph.E_A/(ph.R_GAS*T)); UAe = b*ph.UA_NOMINAL
    dC = (ph.Q/ph.V)*(float(inlet[0])-C) - ke*C
    dT = (ph.Q/ph.V)*(float(inlet[1])-T) - ph.H_R*ke*C/(ph.RHO*ph.C_P) - UAe*(T-Tc)/(ph.RHO*ph.C_P*ph.V)
    dTc = (Qc/ph.V_C)*(float(inlet[2])-Tc) + UAe*(T-Tc)/(ph.RHO_C*ph.C_PC*ph.V_C)
    return [dC, dT, dTc, (T - Tsp) if (qu > Qmin and qu < Qmax) else 0.0]

pts = {n: (c.alpha, c.beta) for n, c in SCENARIO_CONFIGS.items() if c.mode == "closed_loop"}
pts.update({f"corner a{a} b{b}": (a, b) for a in (0.4, 1.2) for b in (0.4, 1.2)})

rows = []
for name, (a, b) in pts.items():
    p = ws._params(a, b); y = ws.steady_state(a, b)
    rhs = np.asarray(cstr_closed_loop_rhs(0.0, y, (p, inlet, ctrl)))
    J = np.asarray(jax.jacfwd(lambda yy: cstr_closed_loop_rhs(0.0, yy, (p, inlet, ctrl)))(y))
    # saturated points: integrator gate pins dI=0 -> row/col of I is zero; drop zero modes
    ev = np.linalg.eigvals(J)
    ev = ev[np.abs(ev) > 1e-9]
    slow = float(np.min(np.abs(ev.real))) if len(ev) else np.nan
    tau_slow = 1.0 / slow if slow > 0 else np.nan
    # deterministic relaxation from P0 (healthy ss) to ss(theta) -- float64 LSODA (float32 Euler stalls near the
    # flat fixed point: increments < 1.5e-5 K are lost at T~312.5, which would fake a "slow settling")
    from scipy.integrate import solve_ivp
    y0h = np.asarray(ws.steady_state(1.0, 1.0), np.float64)
    sol = solve_ivp(lambda t_, yy: rhs64(yy, a, b), (0, 600), y0h, method="LSODA",
                    t_eval=np.arange(0.5, 600.01, 0.5), rtol=1e-9, atol=1e-11)
    path = sol.y.T; t = sol.t
    from cstr_sbi.physics import compute_qc
    Kp_, ti_, Tsp_, Qc0_, Qmin_, Qmax_ = [float(x) for x in ctrl]
    qc = np.clip(Qc0_ + Kp_*((path[:,1]-Tsp_) + path[:,3]/ti_), Qmin_, Qmax_)
    qc_ss = float(compute_qc(y[1], y[3], ctrl))
    def settle(x, xs, x0):
        dev0 = abs(x0 - xs)
        if dev0 < 1e-3: return np.nan          # nothing to settle
        out = np.where(np.abs(x - xs) > 0.02*dev0)[0]
        return float(t[out[-1]]) if len(out) else 0.0
    t_settle_qc = settle(qc, qc_ss, float(compute_qc(y0h[1], y0h[3], ctrl)))
    t_settle_T = settle(path[:,1], float(y[1]), float(y0h[1]))
    rows.append(dict(point=name, alpha=a, beta=b, max_abs_rhs=float(np.abs(rhs).max()),
                     I_rhs=float(rhs[3]), qc_ss=qc_ss, saturated=bool(qc_ss <= QC_MIN+1e-3 or qc_ss >= QC_MAX-1e-3),
                     slowest_eig_timescale_min=tau_slow, P0_settle_Qc_2pct_min=t_settle_qc, P0_settle_T_2pct_min=t_settle_T,
                     P0_Qc_initial_dev=float(compute_qc(y0h[1], y0h[3], ctrl)) - qc_ss))
e1a = pd.DataFrame(rows); e1a.to_csv(R/"29_e1_fixedpoint_settling.csv", index=False)
print(e1a.round(4).to_string())

# (c) burn-in length for P1b: noise-free LNA variance transient  dC/dt = J C + C J^T + G G^T, C(0)=0  (start = deterministic ss)
from scipy.integrate import solve_ivp
from cstr_sbi.simulator import DEFAULT_PROCESS_SIGMA
qd = np.asarray(DEFAULT_PROCESS_SIGMA, float)**2
sc = {"Sc1 healthy": (1.0, 1.0), "Sc4 combined": (0.85, 0.85), "Sc7 (1,.85)": (1.0, 0.85), "Sc2 fouling": (1.0, 0.7)}
res = []
fig, axes = plt.subplots(1, len(sc), figsize=(16, 3.8), constrained_layout=True, sharey=True)
for ax, (nm, (a, b)) in zip(axes, sc.items()):
    p = ws._params(a, b); y = ws.steady_state(a, b)
    J = np.asarray(jax.jacfwd(lambda yy: cstr_closed_loop_rhs(0.0, yy, (p, inlet, ctrl)))(y), float)
    act = [k for k in range(4) if np.abs(J[k]).max() > 1e-9 or np.abs(J[:, k]).max() > 1e-9]
    Ja, Q = J[np.ix_(act, act)], np.diag(qd[act]); m = len(act)
    f = lambda t_, c: (Ja @ c.reshape(m, m) + c.reshape(m, m) @ Ja.T + Q).ravel()
    ts = np.arange(0, 400.01, 1.0)
    sol = solve_ivp(f, (0, 400), np.zeros(m*m), t_eval=ts, rtol=1e-8, atol=1e-14, method="LSODA")
    Cs = sol.y.T.reshape(len(ts), m, m)
    var = np.stack([np.diagonal(c) for c in Cs]); fin = var[-1]
    for j, k in enumerate(act):
        nmj = ["C", "T", "Tc", "I"][k]; frac = np.sqrt(var[:, j]/fin[j])
        t95 = float(ts[np.where(frac < 0.95)[0][-1] + 1]) if np.any(frac < 0.95) else 0.0
        res.append(dict(point=nm, state=nmj, t95_min=t95, stationary_std=float(np.sqrt(fin[j]))))
        ax.plot(ts, frac, label=nmj)
    ax.axhline(0.95, color="k", ls=":"); ax.axvline(150, color="r", ls="--", lw=1); ax.axvline(60, color="grey", ls="--", lw=1)
    ax.set_title(nm); ax.set_xlabel("min after deterministic start"); ax.set_xlim(0, 300)
axes[0].set_ylabel("std / stationary std (LNA)"); axes[0].legend()
fig.suptitle("P1b burn-in: time for the noise-induced spread to reach 95% of stationary (red = 150 min burn-in, grey = 60-min window)")
fig.savefig(F/"29_e1_burnin.png", dpi=130)
e1b = pd.DataFrame(res); e1b.to_csv(R/"29_e1_burnin.csv", index=False); print(e1b.round(3).to_string())

# (d) time-scale comparison
ra, rb = ws.RATE_ALPHA_PER_MIN, ws.RATE_BETA_PER_MIN
ts = pd.DataFrame([
  dict(quantity="window length (min)", value=60.0),
  dict(quantity="alpha drift over one window", value=ra*60), dict(quantity="beta drift (initial) over one window", value=rb*60),
  dict(quantity="time for alpha to move 0.01 (min)", value=0.01/ra), dict(quantity="time for beta to move 0.01 (min)", value=0.01/rb),
  dict(quantity="30-day horizon (min)", value=43200.0)])
ts.to_csv(R/"29_e1_timescales.csv", index=False); print(ts.to_string(index=False))
