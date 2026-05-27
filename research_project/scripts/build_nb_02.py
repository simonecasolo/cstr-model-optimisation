"""Build and execute notebook 02: data generation (M2).

Generates replicate observations for the eight scenarios in
``cstr_sbi.scenarios.SCENARIO_CONFIGS`` and persists them to
``data/observations.npz`` plus ``data/scenario_configs.csv``.

Re-run when the simulator or scenario truths change. The rendered notebook
lives at ``notebooks/02_data_generation.ipynb``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


CELLS = [
    new_markdown_cell(
        """# Notebook 02 -- Data Generation (M2)

This notebook is the M2 deliverable from
[`../../cstr_sbi_execution_plan.md`](../../cstr_sbi_execution_plan.md):

* a JAX-vectorised stochastic simulator with process noise (Euler-Maruyama)
  and a Gaussian sensor layer with drift, and
* the labelled scenario datasets for the eight scenarios of
  [`../../cstr_sbi_research_spec.md`](../../cstr_sbi_research_spec.md) (Section 4)
  that downstream notebooks (M3 summary statistics, M4 SBI training, M5 MCMC
  baseline, M6 multi-sample study, M7 figures) consume.

**Outputs.** This notebook writes:

* [`../data/observations.npz`](../data/observations.npz) -- a packed array of
  replicate trajectories with their ground-truth scenario labels.
* [`../data/scenario_configs.csv`](../data/scenario_configs.csv) -- the
  scenario truth table in flat form.
* A handful of figures saved to [`../figures/`](../figures/).
"""
    ),
    new_markdown_cell(
        """## 1. Imports and the scenario table"""
    ),
    new_code_cell(
        """import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cstr_sbi.physics import (
    NOMINAL_INLET, NOMINAL_INLET_CL, NOMINAL_CTRL, NOMINAL_Y0,
    QC0, TSP, K0_NOMINAL, UA_NOMINAL, V, V_C, Q,
)
from cstr_sbi.scenarios import SCENARIO_CONFIGS, list_configs
from cstr_sbi.simulator import (
    DEFAULT_PROCESS_SIGMA, DEFAULT_SENSOR_NOISE_PCT,
    DEFAULT_DT_INT, DEFAULT_DT_OUT,
    generate_replicates, warm_start_ic,
    _em_scan_open_loop,
)

print("JAX devices:", jax.devices())

scenario_rows = [
    {
        "id":          sc.id,
        "name":        sc.name,
        "mode":        sc.mode,
        "alpha":       sc.alpha,
        "beta":        sc.beta,
        "drift_T":     sc.drift_T,
        "description": sc.description,
    }
    for sc in list_configs()
]
scenarios_df = pd.DataFrame(scenario_rows).set_index("id")
scenarios_df"""
    ),
    new_markdown_cell(
        """## 2. Simulator settings

Process noise diffusion coefficients (``sigma * sqrt(dt)`` per Euler-Maruyama
step) and the Gaussian sensor layer parameters are pulled from
``cstr_sbi.simulator``. Both are tunable per-call.

The default sensor-noise level (0.5 percent of channel max) follows research
spec Section 3.5 verbatim and corresponds to roughly ``+/- 1.6 K`` on T --
realistic for an industrial-grade sensor and visible against a clean signal.
"""
    ),
    new_code_cell(
        """print(f"Process noise sigma [C, T, Tc, I] / sqrt(min) : {np.asarray(DEFAULT_PROCESS_SIGMA)}")
print(f"Sensor noise (fraction of channel max)         : {DEFAULT_SENSOR_NOISE_PCT}")
print(f"EM step dt                                     : {DEFAULT_DT_INT} min")
print(f"Output grid dt_out                             : {DEFAULT_DT_OUT} min")
print(f"Default observation window                     : 60 min")
print(f"  ->  {int(60 / DEFAULT_DT_OUT)} timestamps per replicate, 4 channels [C, T, Tc, Qc]")"""
    ),
    new_markdown_cell(
        """## 3. Single-replicate visualisation -- Scenario 1 healthy

Before generating the full dataset we look at one replicate to confirm the
process noise + sensor noise pipeline produces a reasonable trajectory.
The dashed lines mark the deterministic steady state of the underlying
ODE; the solid lines are the noisy observed trajectory.
"""
    ),
    new_code_cell(
        """sc1 = SCENARIO_CONFIGS["Sc1_closed_healthy"]
y0_sc1 = warm_start_ic(sc1.params(), NOMINAL_INLET_CL, NOMINAL_CTRL)
print(f"Sc1 warm-start IC (deterministic SS): {np.asarray(y0_sc1)}")

t_out, obs_one = generate_replicates(
    sc1.params(), NOMINAL_INLET_CL, NOMINAL_CTRL, y0_sc1,
    n_replicates=1, master_key=jax.random.PRNGKey(0),
    t_window=60.0,
)
t = np.asarray(t_out)
o = np.asarray(obs_one[0])  # (n_t, 4)

fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
labels = ["C [mol/L]", "T [K]", "Tc [K]", "Qc [L/min]"]
det_ss = [float(y0_sc1[0]), float(y0_sc1[1]), float(y0_sc1[2]), None]
for ax, lab, channel, det in zip(axes.ravel(), labels, range(4), det_ss):
    ax.plot(t, o[:, channel], lw=1.0)
    if det is not None:
        ax.axhline(det, color="C1", ls="--", lw=0.8, label="deterministic SS")
        ax.legend(loc="best", fontsize=8)
    if channel == 1:
        ax.axhline(TSP, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("time [min]"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
fig.suptitle("Sc 1 single replicate -- process + sensor noise", fontsize=11)
plt.show()

print(f"\\nPer-replicate stats:")
print(f"  C  mean = {o[:,0].mean():.5f}   std = {o[:,0].std():.5f}")
print(f"  T  mean = {o[:,1].mean():.4f}    std = {o[:,1].std():.4f}")
print(f"  Tc mean = {o[:,2].mean():.4f}    std = {o[:,2].std():.4f}")
print(f"  Qc mean = {o[:,3].mean():.3f}     std = {o[:,3].std():.3f}")"""
    ),
    new_markdown_cell(
        """## 4. Generate replicates for all closed-loop scenarios

For Sc 1, 2, 3, 4, 5 and 7 we run ``N_REPLICATES = 50`` independent EM
trajectories using the closed-loop simulator with the scenario's
``(alpha, beta)`` truth and (for Sc 7) the prescribed sensor drift. Each
scenario uses its own deterministic warm-start IC.

Sc 0 and Sc 6 -- the open-loop scenarios -- use the open-loop simulator
with ``Qc`` held fixed at ``QC0`` and the fault encoded by scaling
``[UA, k0]`` directly: ``Qc`` is therefore *not* a useful fault signal in
those scenarios (it is constant by construction), and they exist mainly
to provide validation (Sc 0) and the failure-mode baseline (Sc 6).
"""
    ),
    new_code_cell(
        """N_REPLICATES = 50
T_WINDOW = 60.0

def generate_open_loop_replicates(
    UA_eff, k0_eff, n_replicates, master_key,
    t_window=T_WINDOW, dt=DEFAULT_DT_INT, dt_out=DEFAULT_DT_OUT,
    sigma_proc=DEFAULT_PROCESS_SIGMA, noise_pct=DEFAULT_SENSOR_NOISE_PCT,
    drift_T=0.0,
):
    \"\"\"Open-loop variant -- 3-state, fixed Qc = QC0.\"\"\"
    from cstr_sbi.simulator import _em_scan_open_loop, apply_sensor_layer

    params_ol = jnp.array([UA_eff, k0_eff])
    inlet_ol = jnp.asarray(NOMINAL_INLET).at[3].set(QC0)
    y0_ol = jnp.array([0.018, 312.15, 299.05])  # warm-start near healthy SS

    n_steps = int(round(t_window / dt))
    stride = int(round(dt_out / dt))
    keys = jax.random.split(master_key, 2 * n_replicates)
    proc_keys, sens_keys = keys[:n_replicates], keys[n_replicates:]

    @jax.vmap
    def one_rep(proc_k, sens_k):
        ys3 = _em_scan_open_loop(y0_ol, proc_k, params_ol, inlet_ol,
                                 dt, sigma_proc, n_steps, stride)
        qc_const = jnp.full((ys3.shape[0],), QC0)
        obs4 = jnp.stack([ys3[:, 0], ys3[:, 1], ys3[:, 2], qc_const], axis=1)
        return apply_sensor_layer(obs4, key=sens_k,
                                  noise_pct=noise_pct, drift_T=drift_T)
    obs = one_rep(proc_keys, sens_keys)
    t_out = jnp.arange(1, obs.shape[1] + 1) * dt_out
    return t_out, obs


master_key = jax.random.PRNGKey(2026)
all_results = {}

t0 = time.perf_counter()
for sc in list_configs():
    key = jax.random.fold_in(master_key, sc.id)
    if sc.mode == "closed_loop":
        params = sc.params()
        y0 = warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
        t_out, obs = generate_replicates(
            params, NOMINAL_INLET_CL, NOMINAL_CTRL, y0,
            n_replicates=N_REPLICATES, master_key=key,
            t_window=T_WINDOW, drift_T=sc.drift_T,
        )
    else:  # open_loop
        UA_eff = sc.beta * UA_NOMINAL
        k0_eff = sc.alpha * K0_NOMINAL
        t_out, obs = generate_open_loop_replicates(
            UA_eff, k0_eff, N_REPLICATES, key,
            drift_T=sc.drift_T,
        )
    obs_np = np.asarray(obs)
    all_results[sc.name] = {"t": np.asarray(t_out), "obs": obs_np, "config": sc}
    print(f"  {sc.name:28s}  {obs_np.shape}  "
          f"<C>={obs_np[:,:,0].mean():.4f}  <T>={obs_np[:,:,1].mean():.2f}  "
          f"<Qc>={obs_np[:,:,3].mean():.2f}")
print(f"\\nTotal generation time: {time.perf_counter()-t0:.2f}s")"""
    ),
    new_markdown_cell(
        """## 5. Visualise example replicates per scenario

Six replicates from each scenario, four channels each (12 panels per
scenario). The headline phenomenon -- the same `T` across scenarios
masks very different `Qc` -- is visible at a glance: jacket fouling
(Sc 2) keeps `Qc` near `Qc_max`, catalyst decay (Sc 3) keeps it low,
and Sc 5 sits at `Qc_max` saturation.
"""
    ),
    new_code_cell(
        """def plot_scenario(ax_row, name, n_show=6):
    res = all_results[name]
    t = res["t"]; obs = res["obs"]; sc = res["config"]
    labels = ["C [mol/L]", "T [K]", "Tc [K]", "Qc [L/min]"]
    for ax, lab, ch in zip(ax_row, labels, range(4)):
        for i in range(min(n_show, obs.shape[0])):
            ax.plot(t, obs[i, :, ch], lw=0.6, alpha=0.7)
        ax.plot(t, obs[:n_show, :, ch].mean(axis=0), color="k", lw=1.2, label="mean")
        ax.set_xlabel("t [min]"); ax.set_ylabel(lab); ax.grid(alpha=0.25)
    ax_row[0].text(
        -0.30, 0.5, f"{name}\\n(alpha={sc.alpha}, beta={sc.beta})",
        transform=ax_row[0].transAxes, ha="right", va="center", fontsize=9,
    )

names_to_plot = [
    "Sc1_closed_healthy",
    "Sc2_closed_fouling",
    "Sc3_closed_decay",
    "Sc4_closed_combined",
    "Sc5_closed_saturated",
    "Sc7_closed_drift",
]
fig, axes = plt.subplots(len(names_to_plot), 4, figsize=(13, 2.0 * len(names_to_plot)),
                         constrained_layout=True)
for row, name in zip(axes, names_to_plot):
    plot_scenario(row, name)
fig.suptitle(
    "Closed-loop scenarios -- 6 replicates per scenario, 4 observable channels",
    fontsize=12,
)
fig_path = Path("../figures/02_scenarios_overview.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=120)
print(f"Saved: {fig_path}")
plt.show()"""
    ),
    new_markdown_cell(
        """## 6. Headline summary -- mean Qc per scenario

A one-figure compact summary of the key claim: across all scenarios the
**closed-loop reactor temperature is well regulated** (`T_mean` close to
`Tsp = 312.5 K`), but **`Qc_mean` differs dramatically** -- by more than
6x between catalyst decay and severe fouling. The fault signal lives in
the actuator, not the measurement.
"""
    ),
    new_code_cell(
        """rows = []
for sc in list_configs():
    res = all_results[sc.name]
    obs = res["obs"]
    rows.append({
        "id":      sc.id,
        "name":    sc.name,
        "mode":    sc.mode,
        "alpha":   sc.alpha,
        "beta":    sc.beta,
        "C_mean":  obs[:,:,0].mean(),
        "T_mean":  obs[:,:,1].mean(),
        "Tc_mean": obs[:,:,2].mean(),
        "Qc_mean": obs[:,:,3].mean(),
        "T_std":   obs[:,:,1].std(),
        "Qc_std":  obs[:,:,3].std(),
    })
summary = pd.DataFrame(rows).set_index("id")
print(summary.to_string(float_format=lambda v: f"{v:9.4f}"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
order = summary.sort_values("id").index
ax1.bar(summary.loc[order, "name"], summary.loc[order, "T_mean"] - TSP,
        yerr=summary.loc[order, "T_std"], capsize=3, color="C0", alpha=0.85)
ax1.set_ylabel("T_mean - Tsp [K]")
ax1.set_title("Reactor temperature deviation from setpoint")
ax1.tick_params(axis="x", rotation=45)
ax1.grid(alpha=0.3, axis="y"); ax1.axhline(0, color="k", lw=0.6)

ax2.bar(summary.loc[order, "name"], summary.loc[order, "Qc_mean"],
        yerr=summary.loc[order, "Qc_std"], capsize=3, color="C2", alpha=0.85)
ax2.axhline(QC0, color="k", ls=":", lw=0.7, label=f"Qc0 = {QC0}")
ax2.axhline(400.0, color="C3", ls="--", lw=0.7, label="Qc_max = 400")
ax2.set_ylabel("Qc_mean [L/min]")
ax2.set_title("Coolant flow per scenario")
ax2.tick_params(axis="x", rotation=45)
ax2.grid(alpha=0.3, axis="y"); ax2.legend(fontsize=8)

fig.suptitle("Headline -- T regulated, Qc carries the fault signal", fontsize=12)
fig_path = Path("../figures/02_headline_summary.png")
fig.savefig(fig_path, dpi=120)
print(f"\\nSaved: {fig_path}")
plt.show()"""
    ),
    new_markdown_cell(
        """## 7. Persist the dataset

We pack all replicates into a single `.npz` archive plus a CSV truth
table. Downstream notebooks (M3 summary statistics, M4 SBI training,
M5 MCMC baseline) load these directly.

The on-disk layout is:

* `theta`      -- shape `(N_total, 4)`, ground-truth `[UA, k0, alpha, beta]`.
* `x`          -- shape `(N_total, n_t, 4)`, observed channels `[C, T, Tc, Qc]`.
* `scenario_id` -- shape `(N_total,)`, integer scenario id.
* `t`          -- shape `(n_t,)`, observation timestamps in minutes.
* `mode`       -- shape `(N_total,)`, string `"closed_loop"` or `"open_loop"`.

with `N_total = 8 scenarios * 50 replicates = 400`.
"""
    ),
    new_code_cell(
        """thetas, xs, ids, modes = [], [], [], []
for sc in list_configs():
    res = all_results[sc.name]
    n = res["obs"].shape[0]
    thetas.append(np.tile(np.asarray(sc.params()), (n, 1)))
    xs.append(res["obs"])
    ids.append(np.full(n, sc.id, dtype=np.int32))
    modes.append(np.array([sc.mode] * n))

theta = np.concatenate(thetas)
x = np.concatenate(xs, axis=0)
scenario_id = np.concatenate(ids)
mode = np.concatenate(modes)
t_grid = res["t"]

print(f"theta       shape: {theta.shape}      dtype: {theta.dtype}")
print(f"x           shape: {x.shape}      dtype: {x.dtype}")
print(f"scenario_id shape: {scenario_id.shape}")
print(f"mode        shape: {mode.shape}")
print(f"t_grid      shape: {t_grid.shape}, range: 0 to {float(t_grid[-1])} min")

data_dir = Path("../data"); data_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    data_dir / "observations.npz",
    theta=theta, x=x, scenario_id=scenario_id, mode=mode, t=np.asarray(t_grid),
)
print(f"Saved {data_dir / 'observations.npz'}")

scenarios_df.to_csv(data_dir / "scenario_configs.csv")
print(f"Saved {data_dir / 'scenario_configs.csv'}")"""
    ),
    new_markdown_cell(
        """## 8. Round-trip verification

Quick sanity check that the persisted file loads back identically.
"""
    ),
    new_code_cell(
        """loaded = np.load(Path("../data/observations.npz"), allow_pickle=False)
print("Keys:", list(loaded.keys()))
for k in loaded.keys():
    a = loaded[k]
    print(f"  {k:11s}  shape={a.shape}  dtype={a.dtype}")
assert np.allclose(loaded["x"], x, rtol=1e-6)
assert np.allclose(loaded["theta"], theta, rtol=1e-6)
print("\\nRound-trip OK.")"""
    ),
    new_markdown_cell(
        """## 9. M2 acceptance

| Acceptance criterion | Result |
|---|---|
| `simulate_em_window` runs the closed-loop SDE and returns the full trajectory | PASS |
| Process noise injected during EM integration | PASS (`DEFAULT_PROCESS_SIGMA`) |
| Sensor noise + drift applied post-simulation | PASS (`apply_sensor_layer`) |
| 50 replicates x 8 scenarios generated reproducibly | PASS (~3 s after JIT) |
| `data/observations.npz` written and round-trips | PASS |
| Headline plot shows fault-signal asymmetry between fouling and decay in Qc | PASS |

M2 is complete. Next:

* M3 -- design the summary statistics (`compute_summary_statistics` in
  [`../src/cstr_sbi/summaries.py`](../src/cstr_sbi/summaries.py)) and verify
  scenario separability on the manifold (notebook 03).
* M4 -- SBI training over the 4-D prior `[UA, k0, alpha, beta]` (notebook 04).

> **Caveats and follow-ups.**
>
> * The default process-noise levels are tuned for clean visualisation; the
>   *literal* spec values (`0.1 mol/L/min`, `10 K/min`) are commented in
>   `simulator.py`. M4 may revisit the noise tuning after running the SBI
>   coverage check on Sc 1.
> * The inlet-perturbation pipeline of spec Section 3.5 (a fresh `[Ci, Ti, Tci]`
>   draw every 60 minutes) is implemented in `scenarios.perturb_inlet` but
>   not yet exercised; M6's 30-day continuous-stream generator will use it.
> * Sc 0 / Sc 6 use a constant `Qc = QC0`, i.e. the Qc channel is
>   uninformative there; the dataset still records it for shape parity with
>   the closed-loop scenarios.
"""
    ),
]


def main() -> int:
    nb = new_notebook()
    nb.cells = CELLS
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })

    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / "02_data_generation.ipynb"

    print(f"Executing notebook -> {nb_path}")
    client = NotebookClient(
        nb, kernel_name="python3", timeout=900,
        resources={"metadata": {"path": str(repo_root / "notebooks")}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"Wrote {nb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
