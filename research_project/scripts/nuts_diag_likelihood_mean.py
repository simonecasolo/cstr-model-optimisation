"""Diagnostic for step e (NUTS): is the NUTS likelihood mean (deterministic summary at theta, nominal inlet, matched warm
start) consistent with the N3 + P2 observation process? Compares E[S | theta] from replicates (D0's generator) with the
deterministic summary, per feature in units of the replicate sd, at the closed-loop scenarios. Also the conditioning of
the replicate covariance. Read-only; writes results/nuts_diag_likelihood_mean.csv."""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
import numpy as np, pandas as pd, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import ws_eval as we
from cstr_sbi.physics import UA_NOMINAL, K0_NOMINAL, NOMINAL_INLET_CL, NOMINAL_CTRL, simulate_closed_loop_trajectory
from cstr_sbi.simulator import warm_start_ic
from cstr_sbi.summaries import compute_summary_statistics
th2, names, drift = we.scenario_thetas(1)
rows = []
for sc, (a, b) in zip(names, th2):
    if sc == "Sc7":
        continue
    reps = we.simulate(np.tile([a, b], (400, 1)).astype(np.float32), "P2", 5_550_000, features="S0_base29")
    mu, sd = reps.mean(0), reps.std(0)
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, a, b])
    y0 = warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
    ts, ys, qc = simulate_closed_loop_trajectory(params, NOMINAL_INLET_CL, NOMINAL_CTRL, y0, t_final=60.0, n_save=120)
    det = np.asarray(compute_summary_statistics(jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1), ts))
    z = (det - mu) / np.maximum(sd, 1e-12)
    ev = np.linalg.eigvalsh(np.cov(reps.T))
    print(f"{sc}: |z| > 3 for {(np.abs(z) > 3).sum()} of 29 features; median |z| {np.median(np.abs(z)):.2f}; max {np.abs(z).max():.1f}; cov eig min/max {ev.min():.2e}/{ev.max():.2e}, cond {ev.max() / max(ev.min(), 1e-300):.1e}", flush=True)
    for j in range(29):
        rows.append(dict(scenario=sc, feature=j, det=det[j], rep_mean=mu[j], rep_sd=sd[j], z=z[j]))
pd.DataFrame(rows).to_csv(ROOT / "results" / "nuts_diag_likelihood_mean.csv", index=False)
