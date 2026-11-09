"""Controller-form counterfactual (planning the fix; NOT the decision to fix -- that was made on physical grounds).
Standard PI:  Qc = Qc0 + Kp*[(T-Tsp) + I/tau_i],  dI/dt = (T-Tsp)*gate   (I in K*min).
Coded (buggy): Qc = Qc0 + Kp*(T-Tsp) + I/tau_i.  Edits nothing; imports physics read-only.
Checks: (1) damping of ALL closed-loop eigenvalues, (2) saturation / anti-windup behaviour in a 60-min window and
after an abrupt fault, (3) process-noise amplification into T and Qc (stationary, Lyapunov)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
import jax, jax.numpy as jnp
from scipy.linalg import solve_continuous_lyapunov
import ws_protocols as ws
import cstr_sbi.physics as ph
from cstr_sbi.simulator import DEFAULT_PROCESS_SIGMA
inlet, ctrl = ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL
Kp, tau_i, Tsp, Qc0, Qmin, Qmax = [float(x) for x in ctrl]

def rhs_std(y, p, ctrl=ctrl):
    Kp, tau_i, Tsp, Qc0, Qmin, Qmax = ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5]
    T, I = y[1], y[3]
    qc_un = Qc0 + Kp*((T - Tsp) + I/tau_i)
    d = ph.cstr_closed_loop_rhs(0.0, y.at[3].set(0.0), (p, inlet, ctrl))   # plant part; override Qc/dI below
    # recompute plant with std-form Qc (rhs above used I=0 -> wrong Qc); rebuild explicitly:
    Qc = jnp.clip(qc_un, Qmin, Qmax)
    UA, k0, a, b = p[0], p[1], p[2], p[3]; C, Tc = y[0], y[2]
    ke = a*k0*jnp.exp(-ph.E_A/(ph.R_GAS*T)); UAe = b*UA
    dC = (ph.Q/ph.V)*(inlet[0]-C) - ke*C
    dT = (ph.Q/ph.V)*(inlet[1]-T) - ph.H_R*ke*C/(ph.RHO*ph.C_P) - UAe*(T-Tc)/(ph.RHO*ph.C_P*ph.V)
    dTc = (Qc/ph.V_C)*(inlet[2]-Tc) + UAe*(T-Tc)/(ph.RHO_C*ph.C_PC*ph.V_C)
    gate = (qc_un > Qmin) & (qc_un < Qmax)
    return jnp.array([dC, dT, dTc, jnp.where(gate, T - Tsp, 0.0)])

def ss_std(a, b):
    y = ws.steady_state(a, b)           # coded-law ss: same C,T,Tc,Qc; I_coded/tau_i = Kp*I_std/tau_i
    return y.at[3].set(y[3]/Kp)

def qc_std(y): return float(jnp.clip(Qc0 + Kp*((y[1]-Tsp) + y[3]/tau_i), Qmin, Qmax))

pts = {"Sc1 healthy":(1,1),"Sc2 fouling":(1,.7),"Sc3 decay":(.7,1),"Sc4 combined":(.85,.85),"Sc5 saturated":(1,.4),
       "Sc7 (1,.85)":(1,.85),"corner (.4,.4)":(.4,.4),"corner (.4,1.2)":(.4,1.2),"corner (1.2,.4)":(1.2,.4),"corner (1.2,1.2)":(1.2,1.2)}
q = np.asarray(DEFAULT_PROCESS_SIGMA, float)**2
rows = []
for nm,(a,b) in pts.items():
    p = ws._params(a,b); y = ss_std(a,b)
    res_ss = float(jnp.abs(rhs_std(y,p)).max())
    J = np.asarray(jax.jacfwd(lambda yy: rhs_std(yy,p))(y), float)
    act = [k for k in range(4) if np.abs(J[k]).max()>1e-9 or np.abs(J[:,k]).max()>1e-9]
    Ja = J[np.ix_(act,act)]; ev = np.linalg.eigvals(Ja)
    zeta = [(-e.real/abs(e)) if abs(e.imag)>1e-9 else 1.0 for e in ev]
    cplx = [e for e in ev if abs(e.imag)>1e-9]
    # stationary std of T and Qc (unclamped linearised) from Lyapunov
    C = solve_continuous_lyapunov(Ja, -np.diag(q[act]))
    sT = float(np.sqrt(C[act.index(1),act.index(1)]))
    if 3 in act:
        w = np.zeros(len(act)); w[act.index(1)] = Kp; w[act.index(3)] = Kp/tau_i
        sQ = float(np.sqrt(w@C@w))
    else: sQ = 0.0
    rows.append(dict(point=nm, sat_ss=bool(abs(qc_std(y)-Qmax)<1e-2 or abs(qc_std(y)-Qmin)<1e-2), max_abs_rhs=res_ss,
        slowest_tau_min=float(1/np.min(np.abs(ev.real))), fastest_tau_min=float(1/np.max(np.abs(ev.real))),
        n_complex_pairs=len(cplx)//2, min_zeta=float(min(zeta)), osc_period_min=(float(2*np.pi/abs(cplx[0].imag)) if cplx else np.nan),
        std_T_K=sT, std_Qc_Lmin=sQ))
df = pd.DataFrame(rows); df.to_csv(ROOT/"results/30_counterfactual_linear.csv", index=False)
print(df.round(4).to_string())

# (2) nonlinear EM windows: fraction of samples at the rails; abrupt fault recovery from healthy start (P0 analogue)
from functools import partial
@partial(jax.jit, static_argnames=("n",))
def em_std(y0, key, th, n):
    p = ws._params(th[0], th[1]); keys = jax.random.split(key, n); sq = jnp.sqrt(0.01)
    def step(y,k):
        y1 = y + rhs_std(y,p)*0.01 + DEFAULT_PROCESS_SIGMA*sq*jax.random.normal(k,(4,))
        return y1,y1
    return jax.lax.scan(step,y0,keys)[1]
print("\nNonlinear EM, standard form, 60-min windows, 40 reps each: fraction of window at valve rails; abrupt-fault recovery")
rows2=[]
for nm,(a,b) in {"Sc1 healthy":(1,1),"Sc4 combined":(.85,.85),"Sc5 saturated":(1,.4),"Sc2 fouling":(1,.7)}.items():
    for start,(a0,b0) in {"own ss":(a,b),"abrupt from healthy":(1,1)}.items():
        y0 = ss_std(a0,b0); th = jnp.array([a,b])
        paths = np.stack([np.asarray(em_std(y0, jax.random.PRNGKey(i), th, 6000)[49::50]) for i in range(40)])
        qc = np.clip(Qc0+Kp*((paths[...,1]-Tsp)+paths[...,3]/tau_i), Qmin, Qmax)
        lo, hi = (qc<=Qmin+1e-6).mean(), (qc>=Qmax-1e-6).mean()
        Imax = np.abs(paths[...,3]).max()
        rows2.append(dict(point=nm, start=start, frac_at_Qmin=lo, frac_at_Qmax=hi, T_end_minus_Tsp=float(paths[:,-1,1].mean()-Tsp),
                          Qc_end_mean=float(qc[:,-1].mean()), Qc_ss=qc_std(ss_std(a,b)), max_abs_I=float(Imax), any_nan=bool(np.isnan(paths).any())))
d2 = pd.DataFrame(rows2); d2.to_csv(ROOT/"results/30_counterfactual_nonlinear.csv", index=False); print(d2.round(3).to_string())
