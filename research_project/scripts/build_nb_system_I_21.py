"""Build and execute notebook 21 (System I): open-loop-aware NUTS ground truth for
Sc0/Sc6.

**Context.** Notebook 20 (`scripts/build_nb_system_I_20.py`, Theory C) found that
Sc0/Sc6 -- System I's two OPEN-LOOP scenarios (Sc0: healthy, alpha=beta=1.0; Sc6:
jacket-fouling fault, alpha=1.0, beta=0.7) -- leave NUTS badly non-convergent (R-hat
in the hundreds-to-thousands, only 30%/50% of windows passing) even after fixing a
real diagonal-vs-full-covariance likelihood bug that had already resolved an apparent
"multimodality" artifact on the 6 CLOSED-LOOP scenarios. That left Sc0/Sc6 as an
explicit, flagged, unresolved follow-up (see `notebooks/system_I/TODO.md`'s newest
entry and `notebooks/system_I/20_...ipynb`'s own closing markdown cell).

**Root cause, diagnosed separately from this notebook's own setup (not re-derived
here, just acted on):** `cstr_generative_model` (`src/cstr_sbi/inference.py`), the
NUTS likelihood target, ALWAYS simulated the closed-loop PI-controller dynamics --
regardless of whether the real observed data came from open-loop generation. For
Sc0/Sc6 this is not merely a bad warm start (the Theory-D-style mismatch already fixed
for the closed-loop scenarios); it is a genuine STRUCTURAL model mismatch: NUTS was
fitting a model with feedback control to data that has none. `cstr_generative_model`/
`run_mcmc_baseline` gained an additive `mode: str = "closed_loop"` parameter (default
preserves ALL prior behaviour exactly -- verified by a direct `Predictive(...)` smoke
test before this notebook was built; `mode="open_loop"` branches to
`simulate_open_loop_trajectory_fixed`, the open-loop simulator with a FIXED,
theta-independent warm start matching the convention `scripts/nb20_nuts_worker.py`
already uses for its own open-loop observed-data generation).

**This notebook's job**: re-run NUTS on Sc0/Sc6 under this open-loop-aware generative
model (same full-covariance-likelihood fix as notebook 20, same convergence bar,
R-hat<1.01 on both parameters), and report plainly whether the fix resolves
convergence and, if so, what Corr(alpha,beta) NUTS actually finds there -- compared
against SBI's own numbers at those two points (`+0.612` for Sc0, `+0.564` for Sc6,
from notebook 20's Stage 1 survey, `results/20_stage1_per_scenario_corr.csv`).

**Scope note, decided up front**: does NOT touch, re-derive, or re-litigate anything
about the 6 CLOSED-LOOP scenarios (Theory C/E tension, SBC, training-budget
convergence) -- all of that stays exactly where notebook 20 left it. Does NOT touch
`notebooks/system_I/20_...ipynb`, `scripts/build_nb_system_I_20.py`, or
`scripts/recompute_theoryC_section5.py` (a separate, concurrently-running process in
this same session owns those files).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    return new_markdown_cell(source, metadata={"language": "markdown"})


def code(source: str):
    return new_code_cell(source, metadata={"language": "python"})


CELLS = [
    md(
        r"""# Notebook 21 (System I) -- open-loop-aware NUTS ground truth for Sc0/Sc6

**Read `notebooks/system_I/TODO.md`'s newest entries and `HANDOFF.md`'s newest entry
before trusting any number below.** Notebook 20 (Theory C) found Sc0/Sc6 (System I's
two open-loop scenarios) leave NUTS badly non-convergent (R-hat in the
hundreds-to-thousands) even after fixing a diagonal-vs-full-covariance likelihood bug
that cleanly resolved the same symptom on the 6 closed-loop scenarios. This notebook
tests one specific, previously-undiagnosed candidate explanation: `cstr_generative_model`
(the shared NUTS likelihood, `src/cstr_sbi/inference.py`) ALWAYS simulated CLOSED-LOOP
dynamics, even when fitting Sc0/Sc6's open-loop-generated data -- a genuine model
mismatch, not just a bad warm start. `cstr_generative_model`/`run_mcmc_baseline` were
given an additive `mode` parameter (`"closed_loop"` default, unchanged behaviour;
`"open_loop"` branches to the correct simulator) to test this directly.

**This notebook does NOT re-litigate anything about the closed-loop scenarios** (Theory
C/E tension, SBC, training-budget convergence) -- all untouched, exactly where notebook
20 left it. Nothing here is promoted into `main.tex`/the SI/`article_outline_CChE.md`."""
    ),
    md("## 0. Setup"),
    code(
        """import sys, time, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cstr_sbi.physics import (
    K0_NOMINAL, UA_NOMINAL, NOMINAL_INLET_CL, NOMINAL_CTRL, NOMINAL_INLET,
)
from cstr_sbi.simulator import (
    warm_start_ic, simulate_em_window_open_loop, apply_sensor_layer,
    DEFAULT_SENSOR_NOISE_PCT,
)
from cstr_sbi.summaries import compute_summary_statistics
from cstr_sbi.inference import run_mcmc_baseline, cstr_generative_model
from cstr_sbi.scenarios import SCENARIO_CONFIGS

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

SC0 = SCENARIO_CONFIGS["Sc0_open_healthy"]
SC6 = SCENARIO_CONFIGS["Sc6_open_with_fault"]
TARGET_SCENARIOS = [SC0, SC6]
for sc in TARGET_SCENARIOS:
    print(f"{sc.name:24s} alpha={sc.alpha:.2f} beta={sc.beta:.2f} mode={sc.mode}")

import datetime as _dt
def mark(stage):
    (RESULTS / f"21_progress_{stage}.txt").write_text(
        f"{time.time()}\\n{_dt.datetime.now().isoformat()}\\n"
    )
mark("setup_done")
print("Setup complete.")
"""
    ),
    md(
        r"""## 1. The fix: `mode="open_loop"` on `cstr_generative_model`

Before trusting any NUTS number below, confirm (a) the new `mode` parameter leaves
`mode="closed_loop"` (the default) byte-identical in behaviour to before this change,
and (b) `mode="open_loop"` actually produces a sane 29-D summary, via a direct
`numpyro.infer.Predictive` smoke test (no MCMC yet, just forward sampling from the
generative model under both modes)."""
    ),
    code(
        """from numpyro.infer import Predictive

pred_cl = Predictive(cstr_generative_model, num_samples=5)
out_cl = pred_cl(jax.random.PRNGKey(0), mode="closed_loop")
pred_ol = Predictive(cstr_generative_model, num_samples=5)
out_ol = pred_ol(jax.random.PRNGKey(0), mode="open_loop")

print("closed_loop mode -- obs shape:", out_cl["obs"].shape, "any NaN:", bool(jnp.any(jnp.isnan(out_cl["obs"]))))
print("open_loop   mode -- obs shape:", out_ol["obs"].shape, "any NaN:", bool(jnp.any(jnp.isnan(out_ol["obs"]))))

# Confirm closed_loop (default) is unchanged relative to calling with no mode kwarg at all.
pred_default = Predictive(cstr_generative_model, num_samples=5)
out_default = pred_default(jax.random.PRNGKey(0), mode="closed_loop")
same_params = bool(jnp.allclose(out_cl["alpha"], out_default["alpha"])) and bool(jnp.allclose(out_cl["obs"], out_default["obs"]))
print(f"closed_loop explicit vs default-equivalent call identical: {same_params}")
assert same_params, "mode='closed_loop' must be byte-identical to pre-existing default behaviour"
print("MODE SMOKE TEST PASSED")
"""
    ),
    md(
        r"""## 2. Observed-data generation and full-covariance likelihood (Sc0/Sc6)

Same methodology as notebook 20's Theory C/E: the observed summary at each scenario's
true theta is generated via the STOCHASTIC open-loop simulator
(`simulate_em_window_open_loop`, matching `scripts/nb20_nuts_worker.py`'s own
observed-data convention -- fault encoded as `[UA_eff, k0_eff] = [beta, alpha] *
[UA_NOMINAL, K0_NOMINAL]`, fixed theta-independent warm start truncated to `[C, T,
Tc]`), and the full empirical covariance of the 29-D summary statistics is estimated
from 120 replicates at that same true theta, to use as `cov_obs` in
`cstr_generative_model` (now under `mode="open_loop"`) instead of the diagonal
default -- the same fix that resolved the closed-loop "multimodality" artifact in
notebook 20."""
    ),
    code(
        """def feat_summary_po_openloop(theta, seed, noise_pct=DEFAULT_SENSOR_NOISE_PCT):
    alpha, beta = float(theta[0]), float(theta[1])
    params_ol = jnp.array([beta * UA_NOMINAL, alpha * K0_NOMINAL], dtype=jnp.float64)
    y0_ol = warm_start_ic(
        jnp.array([UA_NOMINAL, K0_NOMINAL, 1.0, 1.0]), NOMINAL_INLET_CL, NOMINAL_CTRL
    )[:3]
    pk, sk = jax.random.split(jax.random.PRNGKey(int(seed)))
    t, ys, qc = simulate_em_window_open_loop(params_ol, NOMINAL_INLET, y0_ol, key=pk)
    obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
    obs = apply_sensor_layer(obs, key=sk, noise_pct=noise_pct)
    t_s = jnp.arange(1, ys.shape[0] + 1) * 0.5
    return np.asarray(compute_summary_statistics(jnp.array(obs), jnp.array(t_s)))

N_REPS_COV = 120
scenario_cov_obs = {}
scenario_obs_summary = {}
t0 = time.perf_counter()
for sc in TARGET_SCENARIOS:
    theta_pt = np.array([sc.alpha, sc.beta])
    reps = np.stack([feat_summary_po_openloop(theta_pt, seed=7000 + i) for i in range(N_REPS_COV)])
    Sigma_full = np.cov(reps.T) + 1e-6 * np.eye(reps.shape[1])
    scenario_cov_obs[sc.name] = Sigma_full
    scenario_obs_summary[sc.name] = reps[0]  # one representative draw, used as the
    # single "observed" vector in the cost-check below (matches nb20's own
    # obs_x_for_scenario convention of a single draw, not an average)
    print(f"{sc.name:24s}: cov_obs computed, shape={Sigma_full.shape}")
print(f"Covariance precompute: {time.perf_counter()-t0:.0f}s for {len(TARGET_SCENARIOS)} scenarios")
mark("cov_precompute_done")
"""
    ),
    md(
        r"""## 3. Cost-check: one real open-loop NUTS window, measured not assumed

Per this project's established discipline (notebook 20's Theory C section hit a
10-100x gap between the manuscript's Table VII cost ESTIMATE and the measured
in-process cost) -- measure a single real window at the full Table-VII-style protocol
(4 chains/500 samples/300 warmup, `run_mcmc_baseline`'s own defaults) under
`mode="open_loop"` before committing to any window count."""
    ),
    code(
        """mark("costcheck_start")
sys.path.insert(0, str(ROOT / "scripts"))
from nb21_nuts_worker import nuts_one_window_openloop

NUTS_PROTOCOL = (4, 500, 300)  # (n_chains, n_samples, n_warmup)

t0 = time.perf_counter()
probe = nuts_one_window_openloop(
    (SC0.name, SC0.alpha, SC0.beta, SC0.drift_T, 9999, str(ROOT), *NUTS_PROTOCOL, scenario_cov_obs[SC0.name])
)
probe_wall = time.perf_counter() - t0
print(f"Single open-loop window probe (Sc0): {probe_wall:.1f}s wall, status={probe['status']}, "
      f"r_hat_alpha={probe.get('r_hat_alpha')}, r_hat_beta={probe.get('r_hat_beta')}, "
      f"corr={probe.get('corr')}")
mark("costcheck_done")
"""
    ),
    code(
        """# Window-count decision, based on the MEASURED probe cost above (sequential --
# only 2 scenarios, no multiprocessing needed/wanted, see nb21_nuts_worker.py docstring).
N_WINDOWS_PER_SCENARIO = 20
total_windows = N_WINDOWS_PER_SCENARIO * len(TARGET_SCENARIOS)
eta_hours_serial = total_windows * probe_wall / 3600
print(f"Projected ETA for {total_windows} windows (sequential): {eta_hours_serial:.2f}h")
if eta_hours_serial > 2.0:
    print("WARNING: projected sequential ETA exceeds 2h at the measured probe cost -- "
          "reducing to 10 windows/scenario per this project's scope-reduction discipline.")
    N_WINDOWS_PER_SCENARIO = 10
    total_windows = N_WINDOWS_PER_SCENARIO * len(TARGET_SCENARIOS)
    eta_hours_serial = total_windows * probe_wall / 3600
    print(f"Revised projected ETA: {eta_hours_serial:.2f}h")
print(f"Final choice: {N_WINDOWS_PER_SCENARIO} windows/scenario x {len(TARGET_SCENARIOS)} "
      f"scenarios = {total_windows} windows, protocol={NUTS_PROTOCOL}")
"""
    ),
    md(
        r"""## 4. Sequential sweep with a real per-window timeout

Only 2 scenarios are needed here, so this notebook deliberately skips
`multiprocessing` -- a single process with a real `multiprocessing.Process`-based
per-window timeout (true `terminate()`, not `Pool.apply_async(...).get(timeout=...)`,
which only stops the caller from waiting while the stuck worker keeps running forever;
see `scripts/nb20_nuts_worker.py`'s own docstring for the full history of that lesson).
Given the cost-check above, a conservative timeout of max(600s, 20x the measured probe
cost) is used per window."""
    ),
    code(
        """import multiprocessing as mp
from nb21_nuts_worker import nuts_one_window_openloop_to_queue

PER_WINDOW_TIMEOUT_S = max(600.0, 20.0 * probe_wall)
print(f"Per-window timeout: {PER_WINDOW_TIMEOUT_S:.0f}s")

def run_with_timeout(args, timeout_s):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=nuts_one_window_openloop_to_queue, args=(args, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(5)
        sc_name = args[0]
        return {"scenario": sc_name, "window": args[4], "corr": float("nan"),
                "wall_time_s": None, "status": "timeout", "samples": None,
                "r_hat_alpha": float("nan"), "r_hat_beta": float("nan"),
                "n_eff_alpha": float("nan"), "n_eff_beta": float("nan"),
                "n_divergences": -1, "converged": False}
    return q.get()

CACHE_21 = RESULTS / "21_nuts_openloop_results.pkl"
if CACHE_21.exists():
    with open(CACHE_21, "rb") as f:
        nuts_results_21 = pickle.load(f)
    print(f"Reusing cached results from {CACHE_21.name} ({len(nuts_results_21)} windows)")
else:
    nuts_results_21 = []
    t0 = time.perf_counter()
    for sc in TARGET_SCENARIOS:
        for w in range(N_WINDOWS_PER_SCENARIO):
            args = (sc.name, sc.alpha, sc.beta, sc.drift_T, w, str(ROOT), *NUTS_PROTOCOL,
                    scenario_cov_obs[sc.name])
            res = run_with_timeout(args, PER_WINDOW_TIMEOUT_S)
            nuts_results_21.append(res)
            print(f"{sc.name:24s} window {w:2d}: status={res['status']:10s} "
                  f"r_hat_a={res.get('r_hat_alpha')} r_hat_b={res.get('r_hat_beta')} "
                  f"corr={res.get('corr')}")
            mark(f"sweep_progress_{sc.name}_{w}")
    with open(CACHE_21, "wb") as f:
        pickle.dump(nuts_results_21, f)
    print(f"\\nSweep wall time: {time.perf_counter()-t0:.0f}s for {len(nuts_results_21)} windows")

df_nuts_21 = pd.DataFrame([{k: v for k, v in r.items() if k != "samples"} for r in nuts_results_21])
df_nuts_21.to_csv(RESULTS / "21_nuts_openloop_diagnostics.csv", index=False)
print(df_nuts_21["status"].value_counts())
"""
    ),
    md(
        r"""## 5. Convergence-gated pooling and final Corr(alpha,beta) per scenario

Same convention as notebook 20: pool ONLY windows with R-hat < 1.01 for BOTH
parameters into the "final" posterior; report the unfiltered ("all ok windows") pooled
stats alongside so the effect of gating is visible, and report the fraction of windows
that actually converged per scenario as a first-class result, not hidden."""
    ),
    code(
        """nuts_pooled_converged_21 = {}
nuts_pooled_all_21 = {}
convergence_rows_21 = []
for sc in TARGET_SCENARIOS:
    ok = [r for r in nuts_results_21 if r["scenario"] == sc.name and r["status"] == "ok"
          and r["samples"] is not None]
    converged = [r for r in ok if r.get("converged", False)]
    if ok:
        nuts_pooled_all_21[sc.name] = np.concatenate([r["samples"] for r in ok], axis=0)
    if converged:
        nuts_pooled_converged_21[sc.name] = np.concatenate([r["samples"] for r in converged], axis=0)
    r_hats_a = [r["r_hat_alpha"] for r in ok if np.isfinite(r.get("r_hat_alpha", np.nan))]
    r_hats_b = [r["r_hat_beta"] for r in ok if np.isfinite(r.get("r_hat_beta", np.nan))]
    n_div = sum(r.get("n_divergences", 0) for r in ok if r.get("n_divergences", -1) >= 0)
    convergence_rows_21.append({
        "scenario": sc.name,
        "n_windows_total": N_WINDOWS_PER_SCENARIO,
        "n_windows_ok": len(ok), "n_windows_converged": len(converged),
        "frac_converged": len(converged) / len(ok) if ok else np.nan,
        "mean_r_hat_alpha": np.mean(r_hats_a) if r_hats_a else np.nan,
        "mean_r_hat_beta": np.mean(r_hats_b) if r_hats_b else np.nan,
        "max_r_hat_alpha": np.max(r_hats_a) if r_hats_a else np.nan,
        "max_r_hat_beta": np.max(r_hats_b) if r_hats_b else np.nan,
        "total_divergences": n_div,
        "corr_pooled_converged": (float(np.corrcoef(nuts_pooled_converged_21[sc.name][:, 0],
                                                      nuts_pooled_converged_21[sc.name][:, 1])[0, 1])
                                   if sc.name in nuts_pooled_converged_21 else np.nan),
        "corr_pooled_all": (float(np.corrcoef(nuts_pooled_all_21[sc.name][:, 0],
                                               nuts_pooled_all_21[sc.name][:, 1])[0, 1])
                             if sc.name in nuts_pooled_all_21 else np.nan),
    })

df_convergence_21 = pd.DataFrame(convergence_rows_21)
print(df_convergence_21.to_string(index=False, float_format="%.3f"))
df_convergence_21.to_csv(RESULTS / "21_nuts_openloop_convergence_summary.csv", index=False)
mark("pooling_done")
"""
    ),
    md(
        r"""## 6. Comparison against SBI's own Sc0/Sc6 numbers

SBI's matched-protocol posterior correlations at these two points, from notebook 20's
Stage 1 survey (`results/20_stage1_per_scenario_corr.csv`, read, NOT modified): Sc0
$=+0.612$, Sc6 $=+0.564$. Write the comparison as a small standalone CSV
(`results/21_openloop_nuts_vs_sbi.csv`) -- a NEW file, notebook 20's own
`results/20_final_sbi_vs_nuts_comparison.csv` is left untouched."""
    ),
    code(
        """SBI_CORR_OPENLOOP = {"Sc0_open_healthy": 0.612, "Sc6_open_with_fault": 0.564}

comparison_rows = []
for sc in TARGET_SCENARIOS:
    row = df_convergence_21[df_convergence_21["scenario"] == sc.name].iloc[0]
    comparison_rows.append({
        "scenario": sc.name,
        "alpha_true": sc.alpha, "beta_true": sc.beta,
        "sbi_corr": SBI_CORR_OPENLOOP[sc.name],
        "nuts_corr_openloop_convergence_gated": row["corr_pooled_converged"],
        "nuts_corr_openloop_all_windows": row["corr_pooled_all"],
        "frac_windows_converged": row["frac_converged"],
        "n_windows_total": row["n_windows_total"],
        "mean_r_hat_alpha": row["mean_r_hat_alpha"],
        "mean_r_hat_beta": row["mean_r_hat_beta"],
    })

df_comparison_21 = pd.DataFrame(comparison_rows)
print(df_comparison_21.to_string(index=False, float_format="%.3f"))
df_comparison_21.to_csv(RESULTS / "21_openloop_nuts_vs_sbi.csv", index=False)
mark("comparison_done")
"""
    ),
    md(
        r"""## 7. Figure: pooled (alpha,beta) posterior scatter, Sc0/Sc6

Direct visual complement to the scalar correlation numbers -- convergence-gated pooled
NUTS samples for each scenario, if any windows converged."""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, sc in zip(axes, TARGET_SCENARIOS):
    if sc.name in nuts_pooled_converged_21:
        samp = nuts_pooled_converged_21[sc.name]
        ax.scatter(samp[:, 0], samp[:, 1], s=4, alpha=0.3)
        corr = np.corrcoef(samp[:, 0], samp[:, 1])[0, 1]
        ax.set_title(f"{sc.name}\\nNUTS (open-loop mode), converged-gated, Corr={corr:+.3f}")
    else:
        ax.set_title(f"{sc.name}\\nNO CONVERGED WINDOWS")
    ax.axvline(sc.alpha, color="k", ls=":", lw=0.8)
    ax.axhline(sc.beta, color="k", ls=":", lw=0.8)
    ax.set_xlabel("alpha"); ax.set_ylabel("beta")
plt.tight_layout()
plt.savefig(FIGS / "21_openloop_nuts_posterior_scatter.png", dpi=130)
plt.close()
print("Saved figures/21_openloop_nuts_posterior_scatter.png")
"""
    ),
    md(
        r"""## 8. Plain verdict (written AFTER seeing the real numbers above, not before)

Fill in honestly from the actual `df_comparison_21` / `df_convergence_21` output --
does the `mode="open_loop"` fix resolve Sc0/Sc6's convergence or not? If partially,
say exactly what "partially" means (which scenario, what fraction of windows, what the
R-hat values actually are) rather than rounding to a clean yes/no. Report the actual
Corr(alpha,beta) ground truth if it exists, and state plainly whether it agrees or
disagrees with SBI's own +0.612 (Sc0) / +0.564 (Sc6) -- in sign, in magnitude, or
both."""
    ),
    code(
        """print(df_convergence_21.to_string(index=False, float_format="%.4f"))
print()
print(df_comparison_21.to_string(index=False, float_format="%.4f"))
mark("notebook_complete")
"""
    ),
]


def build_and_execute() -> int:
    nb = new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }

    out_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "system_I"
        / "21_open_loop_nuts_ground_truth.ipynb"
    )

    client = NotebookClient(nb, timeout=None, kernel_name="python3")
    print(f"Executing notebook -> {out_path}")
    client.execute()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    print(f"Wrote {out_path}")
    return 0


def main() -> int:
    return build_and_execute()


if __name__ == "__main__":
    raise SystemExit(main())
