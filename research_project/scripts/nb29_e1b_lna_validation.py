"""nb29 / E1b: validate the LNA stationary covariance (p1b_mode="lna") against a long EM run from the steady state (= the P1b burn-in),
and quantify the plant lag behind a moving steady state (P2 realism)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
import jax, jax.numpy as jnp
import ws_protocols as ws
R = ROOT/"results"
rows = []
for nm, (a, b) in {"Sc1 healthy": (1.0, 1.0), "Sc4 combined": (0.85, 0.85), "Sc3 decay": (0.7, 1.0)}.items():
    yss = ws.steady_state(a, b)[None]
    L = np.asarray(ws.lna_chol(jnp.array([a]), jnp.array([b]), yss))[0]
    C_lna = L @ L.T
    th = jnp.array([a, b]); T_run = 800.0; n = int(T_run/0.01)
    keys = jax.random.split(jax.random.PRNGKey(11), 300)
    last = jax.jit(jax.vmap(lambda k: ws._em_ramp(yss[0], k, th, th, n_steps=n)[-1]))
    Y = np.asarray(last(keys))
    C_em = np.cov(Y.T)
    for j, s in enumerate(["C", "T", "Tc", "I"]):
        rows.append(dict(point=nm, state=s, std_LNA=float(np.sqrt(C_lna[j, j])), std_EM_16000min=float(np.sqrt(C_em[j, j])),
                         ratio=float(np.sqrt(C_em[j, j])/max(np.sqrt(C_lna[j, j]), 1e-30))))
    corr = lambda C: C[1, 3]/np.sqrt(C[1, 1]*C[3, 3])
    rows.append(dict(point=nm, state="corr(T,I)", std_LNA=float(corr(C_lna)), std_EM_16000min=float(corr(C_em)), ratio=np.nan))
df = pd.DataFrame(rows); df.to_csv(R/"29_e1b_lna_validation.csv", index=False); print(df.round(4).to_string())

# plant lag behind a ramping steady state: deterministic, ramp at the max realistic rate, report Qc & T lag
ra, rb = ws.RATE_ALPHA_PER_MIN, ws.RATE_BETA_PER_MIN
lag = []
from cstr_sbi.physics import compute_qc, NOMINAL_CTRL
for nm, (a, b) in {"Sc1 healthy": (1.0, 1.0), "Sc4 combined": (0.85, 0.85)}.items():
    H = 2000.0; n = int(H/0.01)
    for tag, (da, db) in {"alpha ramp": (ra*H, 0.0), "beta ramp": (0.0, rb*H)}.items():
        th_s = jnp.array([a + da, b + db]); th_e = jnp.array([a, b])
        y0 = ws.steady_state(*map(float, th_s))
        path = np.asarray(ws._em_ramp(y0, jax.random.PRNGKey(0), th_s, th_e, n_steps=n, sigma=jnp.zeros(4))[-1])
        yss = np.asarray(ws.steady_state(a, b))
        qc_p = float(compute_qc(path[1], path[3], NOMINAL_CTRL)); qc_s = float(compute_qc(yss[1], yss[3], NOMINAL_CTRL))
        lag.append(dict(point=nm, ramp=tag, theta_change=f"{da or db:.3f}", T_lag_K=float(path[1]-yss[1]), Qc_lag_Lmin=qc_p-qc_s,
                        C_lag_molL=float(path[0]-yss[0]), I_lag=float(path[3]-yss[3])))
lagdf = pd.DataFrame(lag); lagdf.to_csv(R/"29_e1b_lag.csv", index=False); print(lagdf.round(4).to_string())
