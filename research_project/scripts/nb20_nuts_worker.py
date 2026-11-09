"""Standalone, importable NUTS-window worker for notebook 20's Theory C
(SNPE vs. NUTS ground truth). Must live in a real module file, NOT be defined
inside a notebook cell: `multiprocessing`'s default `spawn` start method (the
only safe one alongside JAX/torch on macOS) pickles the worker function by
import path, and a function defined interactively inside a Jupyter kernel's
`__main__` namespace cannot be re-imported by a spawned child process
(`AttributeError: Can't get attribute '_nuts_one_window' on <module '__main__'>`
-- the exact crash this module fixes, found when `build_nb_system_I_20.py`'s
Theory C cell first ran this as a notebook-local closure).

UPDATED (2026-11-06, coordinator-flagged): `cstr_generative_model`'s likelihood
defaults to a DIAGONAL Gaussian over the 29-D summary statistics -- the SAME
independence assumption Theory E found to be wrong for the local-FIM diagnostic (the
full empirical covariance has large off-diagonal terms that close most of the
local-FIM-vs-trained-posterior gap). Reproduced directly: a diagonal-likelihood NUTS
run at the healthy point gave R-hat(alpha)=3.4-5.2, R-hat(beta)=8.5-11.1 across 3 seeds,
with chains visibly settling into DIFFERENT (alpha,beta) basins -- re-running the exact
same window with the full empirical covariance as `cov_obs` instead collapses this to
R-hat<=1.002 for both parameters, n_eff in the hundreds-to-low-thousands, across all 3
seeds. The original "multimodality" finding was therefore a likelihood-misspecification
artifact of the diagonal assumption, not genuine ground-truth multimodality -- this
worker now always uses the full-covariance likelihood (`cov_obs`, precomputed ONCE per
scenario by the caller, not per window, since it only depends on theta) so NUTS is
internally consistent with Theory E's own finding rather than silently reproducing the
same bug elsewhere in this notebook.

UPDATED (2026-11-07, CPU-oversubscription lesson -- read before launching ANY parallel
NUTS pool with this worker, not just for this one script). A full 8-scenario/20-window
sweep (9-worker `multiprocessing.Pool`) was launched without thread limits and went
~20+ minutes with ZERO completed windows, despite a solo cost-check measuring
~316s/window (which predicts several completions per worker in that time). Diagnosis:
all 9 worker processes were genuinely busy (confirmed via `ps`, 90-98% CPU each,
multiple threads per process via `ps -M`), not hung -- but JAX/BLAS/MKL each default to
spawning an internal thread pool sized to the VISIBLE core count on EVERY worker
process, so 9 worker processes x each worker's own multi-threaded math library,
on a 10-physical-core machine, massively oversubscribes the CPU: 9 single-threaded jobs
would fit; 9 jobs each ALSO trying to use ~10 threads internally do not, and every
worker's real per-window wall time inflates far past the solo-process measurement as a
result. **Fix: pin every thread-pool-spawning library to exactly 1 thread per worker**,
set as environment variables BEFORE `jax`/`numpy` are imported in this function (each
spawned worker process imports them fresh, so setting `os.environ` at the very top of
this function, before the `import jax` line below, is early enough) -- this is the
standard fix for any CPU-parallel JAX/NumPy workload. Re-confirm with a real cost-check
after applying this (do not assume it returns to exactly the pre-oversubscription
number) before committing to a full sweep.

CORRECTION (2026-11-07, same day, read this AFTER the entry above -- the oversubscription
fix above is still good practice and was kept, but it was NOT the actual root cause of
the zero-completions symptom). After applying the thread-limiting fix above, the exact
same symptom recurred under a CLEAN, UNCONTENDED `Pool(1)` (one worker, no competing
processes at all) -- ruling out oversubscription as the explanation. Direct, targeted
diagnosis (not guessing): (a) `JAX_LOG_COMPILES=1` on a fresh direct call showed total
XLA compile time of ~15s, ruling out cold-start JIT compilation as the cause; (b)
`progress_bar=True` on a reproduction of one of the slow windows showed NUTS was
genuinely, continuously progressing through its 800 warmup+sample iterations, just at
~1-2s/iteration instead of the typical ~0.4s/iteration. **The real cause: NUTS's
per-window cost has a wide, heavy-tailed distribution under `matched_warm_start=True`.**
Every leapfrog step inside a NUTS trajectory re-solves the nested matched-warm-start ODE
(an expensive differentiable `diffeqsolve`), so when NUTS's adaptive tree-doubling (up to
`max_tree_depth=10`, i.e. up to 1023 leapfrog steps per iteration) happens to go deep for
a given seed/scenario -- an ordinary, expected NUTS behavior, not a bug -- the per-
iteration cost multiplies accordingly, compounding into some windows taking 60+ minutes
against a single-window-probe-based ETA of ~300s. The practical fix is NOT a sampler
change but bounding exposure to the slow tail: `scripts/recompute_theoryC_section5.py`'s
sweep now uses a per-window timeout with a REAL kill (via `multiprocessing.Process.
terminate()`, not `Pool.apply_async(...).get(timeout=...)`, which only stops the caller
from waiting -- the underlying pool worker stays stuck on the abandoned task forever and
never rejoins the pool, silently collapsing effective parallelism over the course of a
sweep). Timed-out windows are recorded with `status="timeout"` and excluded from
convergence-gated pooling downstream, the same as a non-converged window.
"""

from __future__ import annotations

import os
from pathlib import Path


def nuts_one_window(args):
    (sc_name, alpha, beta, mode, drift_T, window_idx, root_str,
     n_chains, n_samples, n_warmup, cov_obs) = args
    # cov_obs: a (29,29) np.ndarray (full empirical summary-statistic covariance at this
    # scenario's true theta, precomputed ONCE per scenario by the caller -- see module
    # docstring addendum below) or None to fall back to the diagonal default.

    # CPU-oversubscription fix (2026-11-07, see module docstring): pin every thread-
    # pool-spawning math library to exactly 1 thread per worker, BEFORE jax/numpy are
    # imported -- each spawned worker process imports them fresh under "spawn", so this
    # is early enough to take effect.
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = "1"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
    ).strip()

    import sys
    root = Path(root_str)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))

    import numpy as np
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    from cstr_sbi.physics import (
        UA_NOMINAL, K0_NOMINAL, NOMINAL_INLET_CL, NOMINAL_CTRL, NOMINAL_INLET,
    )
    from cstr_sbi.simulator import (
        warm_start_ic, simulate_em_window, simulate_em_window_open_loop,
        apply_sensor_layer, DEFAULT_SENSOR_NOISE_PCT,
    )
    from cstr_sbi.summaries import compute_summary_statistics
    from cstr_sbi.inference import run_mcmc_baseline

    seed = 900_000 + (abs(hash(sc_name)) % 10_000) + window_idx
    if mode == "open_loop":
        params_ol = jnp.array([beta * UA_NOMINAL, alpha * K0_NOMINAL], dtype=jnp.float64)
        y0_ol = warm_start_ic(
            jnp.array([UA_NOMINAL, K0_NOMINAL, 1.0, 1.0]), NOMINAL_INLET_CL, NOMINAL_CTRL
        )[:3]
        pk, sk = jax.random.split(jax.random.PRNGKey(seed))
        t, ys, qc = simulate_em_window_open_loop(params_ol, NOMINAL_INLET, y0_ol, key=pk)
    else:
        params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float64)
        y0 = warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
        pk, sk = jax.random.split(jax.random.PRNGKey(seed))
        t, ys, qc = simulate_em_window(params, NOMINAL_INLET_CL, NOMINAL_CTRL, y0, key=pk)

    obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
    obs = apply_sensor_layer(obs, key=sk, noise_pct=DEFAULT_SENSOR_NOISE_PCT)
    if drift_T != 0.0:
        obs = obs.at[:, 1].add(drift_T)
    t_s = jnp.arange(1, ys.shape[0] + 1) * 0.5
    s = np.asarray(compute_summary_statistics(jnp.array(obs), jnp.array(t_s)))

    try:
        import numpyro.diagnostics as numpyro_diag

        # matched_warm_start=True ADDED 2026-11-07 (decisive test confirmed the mechanism):
        # cstr_generative_model previously used a FIXED (healthy-nominal) warm start for
        # every posterior draw regardless of the sampled (alpha,beta), while the observed
        # data being fit here was generated via a per-draw MATCHED warm start (sim_raw_po)
        # -- the same train/eval protocol mismatch already found and fixed for SBI's own
        # beta bias, never ported to this NUTS model. A single-window decisive test at Sc4
        # showed Corr(alpha,beta) moving from +0.026 (mismatched) to -0.740 (matched) --
        # close to SBI's own -0.893 at that point -- so this is now the standard setting,
        # not an optional toggle, for every NUTS call in this notebook's Theory C.
        _mcmc, samp, wall = run_mcmc_baseline(
            s, seed=window_idx, n_chains=n_chains, n_samples=n_samples, n_warmup=n_warmup,
            progress_bar=False, cov_obs=cov_obs, matched_warm_start=True,
        )
        # cstr_generative_model samples only [alpha, beta] (UA/k0 fixed, not sampled) --
        # samp is (n, 2), NOT (n, 4) despite run_mcmc_baseline's docstring claiming
        # [UA,k0,alpha,beta]; confirmed by inspecting cstr_generative_model directly
        # after an IndexError here during this notebook's own live Theory C run.
        corr = float(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1])  # columns [alpha, beta]

        # Convergence diagnostics (added after the coordinator flagged that pooling
        # NUTS samples without checking r_hat/ESS/divergences would silently
        # contaminate the one result meant to be this investigation's ground truth).
        # Per-chain, per-parameter R-hat and ESS via numpyro's own diagnostics.summary,
        # same convention as notebooks/system_I/05_mcmc_baseline.ipynb's R-hat/ESS table.
        by_chain = _mcmc.get_samples(group_by_chain=True)
        diag_summary = numpyro_diag.summary(by_chain)
        r_hat_alpha = float(diag_summary["alpha"]["r_hat"])
        r_hat_beta = float(diag_summary["beta"]["r_hat"])
        n_eff_alpha = float(diag_summary["alpha"]["n_eff"])
        n_eff_beta = float(diag_summary["beta"]["n_eff"])
        extra = _mcmc.get_extra_fields()
        n_divergences = int(np.sum(np.asarray(extra["diverging"]))) if "diverging" in extra else -1
        # Convention: r_hat < 1.01 for BOTH parameters (standard Vehtari et al. threshold,
        # matching the "R-hat ~= 1.00 [check]" convention already used in 05_mcmc_baseline.ipynb).
        converged = bool(r_hat_alpha < 1.01 and r_hat_beta < 1.01)

        return {"scenario": sc_name, "window": window_idx, "corr": corr, "wall_time_s": wall,
                "status": "ok", "samples": np.asarray(samp),
                "r_hat_alpha": r_hat_alpha, "r_hat_beta": r_hat_beta,
                "n_eff_alpha": n_eff_alpha, "n_eff_beta": n_eff_beta,
                "n_divergences": n_divergences, "converged": converged}
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a probe
        return {"scenario": sc_name, "window": window_idx, "corr": float("nan"),
                "wall_time_s": None, "status": f"failed: {exc!r}", "samples": None,
                "r_hat_alpha": float("nan"), "r_hat_beta": float("nan"),
                "n_eff_alpha": float("nan"), "n_eff_beta": float("nan"),
                "n_divergences": -1, "converged": False}


def nuts_one_window_to_queue(args, result_queue):
    """Thin wrapper for use with `multiprocessing.Process` (not `Pool`) so a per-window
    timeout can actually TERMINATE the specific slow worker and free its CPU slot for a
    new task, instead of `Pool.apply_async`'s timeout, which only stops the main process
    from WAITING on a slow result -- the underlying pool worker stays permanently busy on
    the abandoned task, never picking up new work. Added 2026-11-07 after confirming (via
    `progress_bar=True` on a reproduced slow window) that some (scenario, window_idx)
    combinations genuinely take 60+ minutes under `matched_warm_start=True` -- a real,
    seed-dependent NUTS performance tail (deep trees are far more expensive here since
    every leapfrog step re-solves the nested matched-warm-start ODE), not a bug -- so the
    practical fix is being able to kill and move on, not just stop waiting.
    """
    result_queue.put(nuts_one_window(args))
