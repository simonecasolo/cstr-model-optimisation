"""Standalone recompute of notebook 20's Section 5 (Theory-D-mechanism investigation)
and Theory C (SNPE vs NUTS, now under the matched-warm-start fix), run OUTSIDE
nbclient/the Jupyter kernel.

Why this script exists: three separate multi-hour/multi-ten-minute stalls occurred when
re-running the FULL notebook end-to-end through nbclient to pick up the matched-warm-
start fix (same symptom each time: sustained torch CPU activity, visible via `sample`,
but no new output for far longer than the same stage took earlier in the session).
Rather than keep fighting an unreliable nbclient rebuild for the one stage that changed,
this script recomputes ONLY Section 5 + Theory C + the Section 3b comparison directly,
reusing Stage 0/1/A/B/D/E/F's already-cached, UNAFFECTED results from disk
(`results/20_stage1_per_scenario_corr.csv` etc.) -- those stages were not touched by the
matched-warm-start fix and do not need to be redone. Results are written to the exact
same file paths `scripts/build_nb_system_I_20.py` and the executed notebook expect, so
the notebook can be patched directly with real numbers (the same precedent this project
already used for nb19's Section 10d: a markdown-only patch after compute finished,
documented as such, rather than re-running an expensive rebuild a second time).
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from cstr_sbi.physics import (  # noqa: E402
    UA_NOMINAL, K0_NOMINAL, NOMINAL_INLET_CL, NOMINAL_CTRL,
)
from cstr_sbi.simulator import (  # noqa: E402
    warm_start_ic, simulate_em_window, apply_sensor_layer, DEFAULT_SENSOR_NOISE_PCT,
)
from cstr_sbi.summaries import compute_summary_statistics  # noqa: E402
from cstr_sbi.inference import run_mcmc_baseline  # noqa: E402
from cstr_sbi.scenarios import SCENARIO_CONFIGS, list_configs  # noqa: E402

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mark(stage: str) -> None:
    (RESULTS / f"20_progress_{stage}.txt").write_text(f"{time.time()}\n{stage}\n")


SCENARIOS = list_configs()


def feat_summary_po(theta, seed, drift_T=0.0, noise_pct=DEFAULT_SENSOR_NOISE_PCT):
    alpha, beta = float(theta[0]), float(theta[1])
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float64)
    y0 = warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
    proc_key, sens_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    t, ys, qc = simulate_em_window(params, NOMINAL_INLET_CL, NOMINAL_CTRL, y0, key=proc_key)
    obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
    obs = apply_sensor_layer(obs, key=sens_key, noise_pct=noise_pct)
    if drift_T != 0.0:
        obs = obs.at[:, 1].add(drift_T)
    t_s = jnp.arange(1, ys.shape[0] + 1) * 0.5
    return np.asarray(compute_summary_statistics(jnp.array(obs), jnp.array(t_s)))


def feat_summary_po_openloop(theta, seed, noise_pct=DEFAULT_SENSOR_NOISE_PCT):
    from cstr_sbi.physics import NOMINAL_INLET
    from cstr_sbi.simulator import simulate_em_window_open_loop
    alpha, beta = float(theta[0]), float(theta[1])
    params_ol = jnp.array([beta * UA_NOMINAL, alpha * K0_NOMINAL], dtype=jnp.float64)
    y0_ol = warm_start_ic(jnp.array([UA_NOMINAL, K0_NOMINAL, 1.0, 1.0]), NOMINAL_INLET_CL, NOMINAL_CTRL)[:3]
    proc_key, sens_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    t, ys, qc = simulate_em_window_open_loop(params_ol, NOMINAL_INLET, y0_ol, key=proc_key)
    obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
    obs = apply_sensor_layer(obs, key=sens_key, noise_pct=noise_pct)
    t_s = jnp.arange(1, ys.shape[0] + 1) * 0.5
    return np.asarray(compute_summary_statistics(jnp.array(obs), jnp.array(t_s)))


def step0_covariance_precompute():
    mark("recompute_covariance_start")
    log("Precomputing per-scenario full covariance (120 reps each)...")
    N_REPS_COV = 120
    scenario_cov_obs = {}
    t0 = time.perf_counter()
    for sc in SCENARIOS:
        theta_pt = np.array([sc.alpha, sc.beta])
        if sc.mode == "open_loop":
            reps = np.stack([feat_summary_po_openloop(theta_pt, seed=7000 + i) for i in range(N_REPS_COV)])
        else:
            reps = np.stack([feat_summary_po(theta_pt, seed=7000 + i, drift_T=sc.drift_T) for i in range(N_REPS_COV)])
        Sigma_full = np.cov(reps.T) + 1e-6 * np.eye(reps.shape[1])
        scenario_cov_obs[sc.name] = Sigma_full
        log(f"  {sc.name}: cov_obs computed")
    log(f"Covariance precompute: {time.perf_counter()-t0:.0f}s")
    with open(RESULTS / "20_scenario_cov_obs.pkl", "wb") as f:
        pickle.dump(scenario_cov_obs, f)
    mark("recompute_covariance_done")
    return scenario_cov_obs


def step1_section5():
    """Section 5, Steps 2-3 -- real numbers already obtained once manually earlier in
    this session (matched_warm_start decisive test + warm-start sensitivity check);
    recomputed here fresh for a clean, reproducible on-disk artifact rather than reusing
    hand-run numbers."""
    mark("recompute_section5_step2_start")
    log("Section 5 Step 2: decisive matched_warm_start test at Sc4...")
    with open(RESULTS / "20_scenario_cov_obs.pkl", "rb") as f:
        scenario_cov_obs = pickle.load(f)
    sc4 = SCENARIO_CONFIGS["Sc4_closed_combined"]
    theta_pt = np.array([sc4.alpha, sc4.beta])
    s_sc4 = feat_summary_po(theta_pt, seed=777)
    cov_sc4 = scenario_cov_obs[sc4.name]

    import numpyro.diagnostics as npd
    step2_rows = []
    for label, mws in [("matched_warm_start=False (old)", False), ("matched_warm_start=True (fix)", True)]:
        t0 = time.perf_counter()
        mcmc, samp, wall = run_mcmc_baseline(
            s_sc4, seed=0, n_chains=4, n_samples=500, n_warmup=300, progress_bar=False,
            cov_obs=cov_sc4, matched_warm_start=mws,
        )
        elapsed = time.perf_counter() - t0
        by_chain = mcmc.get_samples(group_by_chain=True)
        res = npd.summary(by_chain)
        corr = float(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1])
        step2_rows.append({
            "setting": label, "corr": corr,
            "r_hat_alpha": float(res["alpha"]["r_hat"]), "r_hat_beta": float(res["beta"]["r_hat"]),
            "wall_s": elapsed,
        })
        log(f"  {label}: corr={corr:+.3f} r_hat a={res['alpha']['r_hat']:.3f} "
            f"b={res['beta']['r_hat']:.3f} wall={elapsed:.0f}s")
    pd.DataFrame(step2_rows).to_csv(RESULTS / "20_section5_step2_decisive_test.csv", index=False)
    mark("recompute_section5_step2_done")

    mark("recompute_section5_step3_start")
    log("Section 5 Step 3: warm-start steady-state sensitivity at Sc4...")
    EPS_WS = 0.02
    a0, b0 = 0.85, 0.85

    def warm_start_state(alpha, beta):
        params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float64)
        return np.asarray(warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL))

    d_da = (warm_start_state(a0 + EPS_WS, b0) - warm_start_state(a0 - EPS_WS, b0)) / (2 * EPS_WS)
    d_db = (warm_start_state(a0, b0 + EPS_WS) - warm_start_state(a0, b0 - EPS_WS)) / (2 * EPS_WS)
    state_names = ["C", "T", "Tc", "I"]
    warmstart_rows = []
    for i, name in enumerate(state_names):
        iso_slope = -d_da[i] / d_db[i] if d_db[i] != 0 else float("inf")
        warmstart_rows.append({"state": name, "d_dalpha": d_da[i], "d_dbeta": d_db[i], "iso_slope": iso_slope})
        log(f"  {name}: d/dalpha={d_da[i]:+.4f} d/dbeta={d_db[i]:+.4f} iso_slope={iso_slope:+.4f}")
    pd.DataFrame(warmstart_rows).to_csv(RESULTS / "20_section5_step3_warmstart_sensitivity.csv", index=False)
    mark("recompute_section5_step3_done")


def step2_theoryC_sweep(scenario_cov_obs, n_windows_per_scenario: int = 20,
                         sweep_hard_deadline_s: int = 3 * 3600,
                         scenario_priority_order: list | None = None,
                         resume_from_pickle: Path | None = None):
    """
    RESUMABLE VERSION (added 2026-11-07, coordinator-directed): the first full run of
    this sweep hit its original 3h hard deadline just as Sc4 -- the scenario this entire
    corrected-protocol rerun exists to resolve -- was starting, with Sc0-3 already fully
    attempted (Sc0 ~100% timeout, as expected for the open-loop model mismatch; Sc1-3
    a real mix of successes and timeouts). The hard deadline could not be extended live
    on the already-running process (it's a plain in-memory `time.perf_counter()`
    comparison inside that process's own loop, with no polling of a file/env var/signal
    that a second process could use to change it) -- the only safe option was to let that
    run finish naturally (which writes a real, usable `20_theoryC_nuts_results.pkl` with
    everything collected so far) and relaunch a resumable version. `resume_from_pickle`
    loads that file, skips re-dispatching any (scenario, window) pair already present
    (success OR timeout -- a timeout is still "attempted", not retried automatically,
    to avoid spending another 20 minutes confirming what's already known), and
    `scenario_priority_order` controls which scenarios get dispatched first with the
    FRESH deadline budget -- set to put Sc4 first per the coordinator's explicit
    instruction ("Sc4 matters more than Sc5-7"), ahead of any remaining untried scenario.
    """
    import multiprocessing as mp
    from nb20_nuts_worker import nuts_one_window

    NUTS_PROTOCOL = (4, 500, 300)
    N_CPU = __import__("os").cpu_count()

    already_done = set()  # set of (scenario_name, window_idx) already attempted
    prior_results = []
    if resume_from_pickle is not None and Path(resume_from_pickle).exists():
        with open(resume_from_pickle, "rb") as f:
            prior_results = pickle.load(f)
        # BUGFIX (2026-11-07, caught before it caused silent data loss): the hard-
        # deadline branch below writes a placeholder result (status
        # "timeout_hard_deadline_not_dispatched") for every task that was in the
        # ORIGINAL run's full task list but never actually started -- these must NOT
        # count as "already attempted" on resume, or a relaunch would silently skip
        # entire scenarios that were never really run (caught live: a first resume
        # attempt reported all 8 scenarios, including untouched Sc5-7, as "20/20
        # already attempted" purely because of these placeholders). Only genuine
        # outcomes -- "ok", or a real per-window timeout/hard-deadline kill where a
        # worker process was actually started and running -- count as attempted.
        # "timeout_hard_deadline" (a worker that was still running when the OVERALL
        # sweep ran out of budget, killed mid-flight) is similarly inconclusive -- it is
        # NOT a confirmed per-window timeout verdict (that's the plain "timeout" status,
        # from the 20-min per-window cap), just an artifact of when the deadline
        # happened to land relative to that window's progress. Both placeholder
        # statuses are excluded from "already attempted" so a resume gives these
        # windows a genuine, fair retry rather than silently skipping them.
        NOT_GENUINELY_ATTEMPTED = {"timeout_hard_deadline_not_dispatched", "timeout_hard_deadline"}
        already_done = {(r["scenario"], r["window"]) for r in prior_results
                         if r["status"] not in NOT_GENUINELY_ATTEMPTED}
        # Drop the inconclusive placeholders from what gets carried forward into
        # results_list below -- they're about to be retried as fresh tasks, so keeping
        # them here would double-count that (scenario,window) once the real result
        # comes back from this run.
        prior_results = [r for r in prior_results if (r["scenario"], r["window"]) in already_done]
        log(f"Resuming from {resume_from_pickle}: {len(already_done)} (scenario,window) "
            f"pairs genuinely already attempted, will NOT be re-dispatched.")
        per_scenario_counts = {}
        for sc_name, _w in already_done:
            per_scenario_counts[sc_name] = per_scenario_counts.get(sc_name, 0) + 1
        for sc in SCENARIOS:
            log(f"  {sc.name:24s}: {per_scenario_counts.get(sc.name, 0)}/{n_windows_per_scenario} "
                f"genuinely already attempted")

    mark("recompute_theoryC_costcheck_start")
    log("Cost-check: one real window under the corrected (matched-warm-start) protocol...")
    sc4 = SCENARIO_CONFIGS["Sc4_closed_combined"]
    t0 = time.perf_counter()
    probe = nuts_one_window((sc4.name, sc4.alpha, sc4.beta, sc4.mode, sc4.drift_T,
                              9997, str(ROOT), *NUTS_PROTOCOL, scenario_cov_obs[sc4.name]))
    probe_wall = time.perf_counter() - t0
    log(f"  probe: {probe_wall:.0f}s, status={probe['status']}")

    total_windows = n_windows_per_scenario * len(SCENARIOS)
    n_workers = max(min(N_CPU - 1, total_windows), 1)
    log(f"(Cost-check is informational only on a resumed run -- the deadline and window "
        f"count are set explicitly by the caller using real per-scenario timing from the "
        f"prior run, not re-derived from this one probe.)")

    # PER-WINDOW TIMEOUT WITH REAL KILL (added 2026-11-07, after direct diagnosis): NUTS's
    # per-window cost has a wide, heavy-tailed distribution under matched_warm_start=True
    # -- most windows run in ~300s, but some seeds/scenarios hit much deeper NUTS trees
    # (each leapfrog step re-solves the nested matched-warm-start ODE under
    # dense_mass=True/max_tree_depth=10, so a deep tree is far more expensive here than in
    # a plain model), observed directly (via progress_bar=True on a reproduction of one
    # such window) to still be genuinely progressing, just at ~1-2s/iteration instead of
    # the typical ~0.4s/iteration -- confirmed NOT a hang, NOT a JIT cold-start issue
    # (compile time measured directly via JAX_LOG_COMPILES=1 was ~15s, not the issue), NOT
    # a CPU-oversubscription/Pool-pickling bug (reproduced identically via a clean,
    # uncontended Pool(1) and via a direct, Pool-free call). This is a real, expected NUTS
    # performance characteristic for this specific nested-ODE model, not a bug to fix in
    # the sampler. Uses `multiprocessing.Process` (NOT `Pool`) specifically so a timed-out
    # window's worker can be `.terminate()`d outright and its CPU slot immediately given to
    # a fresh task -- `Pool.apply_async(...).get(timeout=...)` was tried first and rejected:
    # it only stops the MAIN process from waiting, the pool worker stays permanently stuck
    # on the abandoned task and never rejoins the pool, so slow windows would monotonically
    # collapse effective parallelism over the course of the sweep.
    TIMEOUT_PER_WINDOW_S = 20 * 60  # 20 min: generous vs. the ~300s typical case, well
                                     # below the 60+ min outliers actually observed.
    SWEEP_HARD_DEADLINE_S = sweep_hard_deadline_s

    mark("recompute_theoryC_sweep_start")
    ordered_scenarios = SCENARIOS
    if scenario_priority_order:
        name_to_sc = {sc.name: sc for sc in SCENARIOS}
        ordered_names = list(scenario_priority_order) + [
            sc.name for sc in SCENARIOS if sc.name not in scenario_priority_order]
        ordered_scenarios = [name_to_sc[n] for n in ordered_names]
        log(f"Scenario dispatch priority order: {[sc.name for sc in ordered_scenarios]}")

    tasks = []
    skipped = 0
    for sc in ordered_scenarios:
        for w in range(n_windows_per_scenario):
            if (sc.name, w) in already_done:
                skipped += 1
                continue
            tasks.append((sc.name, sc.alpha, sc.beta, sc.mode, sc.drift_T, w, str(ROOT),
                          *NUTS_PROTOCOL, scenario_cov_obs[sc.name]))
    log(f"Launching {len(tasks)} NUTS windows ({skipped} skipped as already-attempted), "
        f"{n_workers} concurrent slots (matched_warm_start=True, full covariance, "
        f"{TIMEOUT_PER_WINDOW_S}s/window kill-timeout, {SWEEP_HARD_DEADLINE_S}s hard deadline)...")

    ctx = mp.get_context("spawn")
    from nb20_nuts_worker import nuts_one_window_to_queue

    def _timeout_result(task, status):
        return {"scenario": task[0], "window": task[5], "corr": float("nan"),
                "wall_time_s": None, "status": status, "samples": None,
                "r_hat_alpha": float("nan"), "r_hat_beta": float("nan"),
                "n_eff_alpha": float("nan"), "n_eff_beta": float("nan"),
                "n_divergences": -1, "converged": False}

    def _launch(task):
        q = ctx.Queue()
        p = ctx.Process(target=nuts_one_window_to_queue, args=(task, q))
        p.start()
        return {"task": task, "process": p, "queue": q, "submit_t": time.perf_counter()}

    t0 = time.perf_counter()
    results_list = list(prior_results)  # seed with resumed results so the final pickle
                                         # written below contains everything, not just
                                         # this run's newly-dispatched tasks.
    pending_tasks = list(tasks)
    slots = [_launch(pending_tasks.pop(0)) for _ in range(min(n_workers, len(pending_tasks)))]
    n_done = 0
    while slots and (time.perf_counter() - t0) < SWEEP_HARD_DEADLINE_S:
        still_active = []
        for slot in slots:
            task, p, q = slot["task"], slot["process"], slot["queue"]
            if not q.empty():
                r = q.get()
                results_list.append(r)
                p.join(timeout=5)
                n_done += 1
                if n_done % 10 == 0 or n_done == len(tasks):
                    log(f"  {n_done}/{len(tasks)} windows done ({time.perf_counter()-t0:.0f}s elapsed)")
                    mark(f"recompute_theoryC_sweep_progress_{n_done}_of_{len(tasks)}")
                if pending_tasks:
                    still_active.append(_launch(pending_tasks.pop(0)))
            elif (time.perf_counter() - slot["submit_t"]) > TIMEOUT_PER_WINDOW_S:
                log(f"  TIMEOUT: {task[0]} window {task[5]} exceeded {TIMEOUT_PER_WINDOW_S}s "
                    f"-- killing this worker and freeing its slot.")
                p.terminate()
                p.join(timeout=5)
                results_list.append(_timeout_result(task, "timeout"))
                n_done += 1
                if pending_tasks:
                    still_active.append(_launch(pending_tasks.pop(0)))
            else:
                still_active.append(slot)
        slots = still_active
        if slots:
            time.sleep(10)
    if slots:
        log(f"Hit the {SWEEP_HARD_DEADLINE_S}s sweep hard deadline with {len(slots)} windows "
            f"still outstanding -- killing remaining workers and recording them as timeout.")
        for slot in slots:
            slot["process"].terminate()
            slot["process"].join(timeout=5)
            results_list.append(_timeout_result(slot["task"], "timeout_hard_deadline"))
    for task in pending_tasks:  # only non-empty if the hard deadline hit before dispatch
        results_list.append(_timeout_result(task, "timeout_hard_deadline_not_dispatched"))

    log(f"Sweep done: {time.perf_counter()-t0:.0f}s total this run "
        f"({sum(1 for r in results_list if r['status'] == 'ok')}/{len(results_list)} ok overall, "
        f"{sum(1 for r in results_list if 'timeout' in r['status'])}/{len(results_list)} timed out overall)")

    with open(RESULTS / "20_theoryC_nuts_results.pkl", "wb") as f:
        pickle.dump(results_list, f)
    df_nuts = pd.DataFrame([{k: v for k, v in r.items() if k != "samples"} for r in results_list])
    df_nuts.to_csv(RESULTS / "20_theoryC_nuts_results.csv", index=False)
    log(df_nuts["status"].value_counts().to_string())
    mark("recompute_theoryC_sweep_done")
    return results_list, n_windows_per_scenario


def step3_pool_and_compare(results_list, n_windows_per_scenario):
    mark("recompute_pooling_start")
    log("Pooling convergence-gated NUTS samples per scenario...")
    df_stage1 = pd.read_csv(RESULTS / "20_stage1_per_scenario_corr.csv")

    nuts_pooled_converged = {}
    nuts_pooled_all = {}
    convergence_summary_rows = []
    for sc in SCENARIOS:
        ok = [r for r in results_list if r["scenario"] == sc.name and r["status"] == "ok"
              and r["samples"] is not None]
        converged = [r for r in ok if r.get("converged", False)]
        if ok:
            nuts_pooled_all[sc.name] = np.concatenate([r["samples"] for r in ok], axis=0)
        if converged:
            nuts_pooled_converged[sc.name] = np.concatenate([r["samples"] for r in converged], axis=0)
        r_hats_a = [r["r_hat_alpha"] for r in ok if np.isfinite(r.get("r_hat_alpha", np.nan))]
        r_hats_b = [r["r_hat_beta"] for r in ok if np.isfinite(r.get("r_hat_beta", np.nan))]
        n_div = sum(r.get("n_divergences", 0) for r in ok if r.get("n_divergences", -1) >= 0)
        convergence_summary_rows.append({
            "scenario": sc.name, "n_windows_ok": len(ok), "n_windows_converged": len(converged),
            "frac_converged": len(converged) / len(ok) if ok else np.nan,
            "mean_r_hat_alpha": np.mean(r_hats_a) if r_hats_a else np.nan,
            "mean_r_hat_beta": np.mean(r_hats_b) if r_hats_b else np.nan,
            "max_r_hat_alpha": np.max(r_hats_a) if r_hats_a else np.nan,
            "max_r_hat_beta": np.max(r_hats_b) if r_hats_b else np.nan,
            "total_divergences": n_div,
        })
        log(f"  {sc.name}: {len(converged)}/{len(ok)} converged")
    df_convergence_summary = pd.DataFrame(convergence_summary_rows)
    df_convergence_summary.to_csv(RESULTS / "20_theoryC_nuts_convergence_summary.csv", index=False)

    def _stats(pooled):
        if pooled is not None and len(pooled) > 10:
            return (float(np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1]),
                    pooled[:, 0].mean(), pooled[:, 0].std(),
                    pooled[:, 1].mean(), pooled[:, 1].std(), len(pooled))
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    theoryC_rows = []
    for sc in SCENARIOS:
        snpe_row = df_stage1.loc[df_stage1["scenario"] == sc.name].iloc[0]
        nuts_corr, nuts_a_mean, nuts_a_std, nuts_b_mean, nuts_b_std, n_pooled = _stats(
            nuts_pooled_converged.get(sc.name))
        nuts_corr_unf, *_rest, n_pooled_unf = _stats(nuts_pooled_all.get(sc.name))
        theoryC_rows.append({
            "scenario": sc.name,
            "snpe_corr_mean": snpe_row["corr_mean"],
            "nuts_corr": nuts_corr, "nuts_alpha_mean": nuts_a_mean, "nuts_alpha_std": nuts_a_std,
            "nuts_beta_mean": nuts_b_mean, "nuts_beta_std": nuts_b_std,
            "nuts_n_pooled_samples": n_pooled,
            "nuts_corr_unfiltered_all_windows": nuts_corr_unf,
            "nuts_n_pooled_samples_unfiltered": n_pooled_unf,
        })
    df_theoryC = pd.DataFrame(theoryC_rows)
    df_theoryC.to_csv(RESULTS / "20_theoryC_snpe_vs_nuts.csv", index=False)
    log(df_theoryC.to_string(index=False, float_format="%.3f"))

    comparison_rows = []
    for sc in SCENARIOS:
        s1 = df_stage1.loc[df_stage1["scenario"] == sc.name].iloc[0]
        c = df_theoryC.loc[df_theoryC["scenario"] == sc.name].iloc[0]
        cs = df_convergence_summary.loc[df_convergence_summary["scenario"] == sc.name].iloc[0]
        comparison_rows.append({
            "scenario": sc.name, "alpha_true": sc.alpha, "beta_true": sc.beta, "mode": sc.mode,
            "sbi_alpha_mean": s1["sbi_alpha_mean"], "sbi_alpha_std": s1["sbi_alpha_std"],
            "sbi_beta_mean": s1["sbi_beta_mean"], "sbi_beta_std": s1["sbi_beta_std"],
            "sbi_corr": s1["corr_mean"], "sbi_corr_std_across_seeds": s1["corr_std"],
            "nuts_alpha_mean": c["nuts_alpha_mean"], "nuts_alpha_std": c["nuts_alpha_std"],
            "nuts_beta_mean": c["nuts_beta_mean"], "nuts_beta_std": c["nuts_beta_std"],
            "nuts_corr": c["nuts_corr"], "nuts_n_pooled_samples": c["nuts_n_pooled_samples"],
            "nuts_corr_unfiltered_all_windows": c["nuts_corr_unfiltered_all_windows"],
            "nuts_frac_converged": cs["frac_converged"],
            "nuts_mean_r_hat_alpha": cs["mean_r_hat_alpha"],
            "nuts_mean_r_hat_beta": cs["mean_r_hat_beta"],
        })
    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(RESULTS / "20_final_sbi_vs_nuts_comparison.csv", index=False)
    log(df_comparison.to_string(index=False, float_format="%.3f"))
    mark("recompute_pooling_done")
    log(f"n_windows_per_scenario used: {n_windows_per_scenario}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--resume-from", type=str, default=None,
                    help="Path to a prior 20_theoryC_nuts_results.pkl to resume from "
                         "(skips already-attempted (scenario,window) pairs).")
    p.add_argument("--deadline-hours", type=float, default=3.0,
                    help="Sweep hard deadline in hours from THIS run's own start.")
    p.add_argument("--priority", type=str, default=None,
                    help="Comma-separated scenario names to dispatch first, e.g. "
                         "'Sc4_closed_combined,Sc5_closed_saturated'.")
    args = p.parse_args()

    scenario_cov_obs = step0_covariance_precompute()
    step1_section5()
    priority_order = args.priority.split(",") if args.priority else None
    resume_path = Path(args.resume_from) if args.resume_from else None
    results_list, n_wps = step2_theoryC_sweep(
        scenario_cov_obs, n_windows_per_scenario=20,
        sweep_hard_deadline_s=int(args.deadline_hours * 3600),
        scenario_priority_order=priority_order,
        resume_from_pickle=resume_path,
    )
    step3_pool_and_compare(results_list, n_wps)
    mark("recompute_ALL_DONE")
    log("ALL DONE")


if __name__ == "__main__":
    main()
