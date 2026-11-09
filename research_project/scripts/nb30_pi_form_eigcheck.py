import sys, numpy as np
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import jax, jax.numpy as jnp
import ws_protocols as ws
from cstr_sbi.physics import cstr_closed_loop_rhs, NOMINAL_INLET_CL as inlet, NOMINAL_CTRL as ctrl
import cstr_sbi.physics as ph
Kp, tau_i, Tsp, Qc0, Qmin, Qmax = [float(x) for x in ctrl]
# Standard form: Qc = Qc0 + Kp*[(T-Tsp) + I/tau_i], dI=T-Tsp.  Equivalent to the coded law with
# I_coded = Kp * I_std  (so I_coded/tau_i = Kp*I_std/tau_i), but dI_coded = Kp*(T-Tsp).
# => implement by scaling the coded dI by Kp.
def rhs_std(y, p):
    d = cstr_closed_loop_rhs(0.0, y, (p, inlet, ctrl))
    return d.at[3].multiply(Kp)           # integral gain Kp/tau_i instead of 1/tau_i
print("Kp=%g tau_i=%g -> coded integral gain 1/tau_i=%.3g ; standard Kp/tau_i=%.3g (x%g)"%(Kp,tau_i,1/tau_i,Kp/tau_i,Kp))
for nm,(a,b) in {"Sc1 healthy":(1,1),"Sc3 decay":(0.7,1),"Sc4 combined":(0.85,0.85),"Sc7 drift-ish (1,0.85)":(1,0.85),"corner (1.2,1.2)":(1.2,1.2)}.items():
    p=ws._params(a,b); y=ws.steady_state(a,b)   # steady state identical (T=Tsp, same Qc); only I's scale differs, irrelevant to J's other entries
    out=[]
    for f in (lambda yy: cstr_closed_loop_rhs(0.0,yy,(p,inlet,ctrl)), lambda yy: rhs_std(yy,p)):
        J=np.asarray(jax.jacfwd(f)(y)); ev=np.linalg.eigvals(J); ev=ev[np.abs(ev)>1e-9]
        out.append(ev)
    sl=lambda ev: ev[np.argmin(np.abs(ev.real))]
    a1,a2=sl(out[0]),sl(out[1])
    print(f"{nm:24s} coded slowest {a1.real:+.2e} (tau {1/abs(a1.real):7.0f} min) | standard-form slowest {a2.real:+.2e} (tau {1/abs(a2.real):6.1f} min) [Im {a2.imag:+.2e}]")
