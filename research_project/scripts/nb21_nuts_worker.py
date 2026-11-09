"""Standalone, importable NUTS-window worker for notebook 21 (System I):
open-loop-aware NUTS ground truth for Sc0/Sc6.

Must live in a real module file, not be defined inside a notebook cell --
``multiprocessing``'s ``spawn`` start method (the only one safe alongside
JAX/torch on macOS) pickles the worker function by import path, and a
function defined interactively inside a Jupyter kernel's ``__main__``
namespace cannot be re-imported by a spawned child process. This is the
exact lesson already learned and documented in
``scripts/nb20_nuts_worker.py`` -- read that file's docstring in full before
touching this one, it is NOT duplicated here beyond what's load-bearing for
this notebook.

**Why this worker exists, separately from nb20_nuts_worker.py**: notebook 20
(Theory C) found that Sc0/Sc6 (System I's two OPEN-LOOP scenarios) leave NUTS
badly non-convergent (R-hat in the hundreds-to-thousands) even after fixing a
real diagonal-vs-full-covariance likelihood bug. The root cause, diagnosed
separately (notebook 21's own setup): ``cstr_generative_model``
(``src/cstr_sbi/inference.py``) ALWAYS simulated the closed-loop PI-controller
dynamics as its likelihood target, regardless of whether the real observed
data came from open-loop generation -- for Sc0/Sc6, NUTS was fitting a
structurally WRONG generative model to the data, not merely suffering a bad
warm start. ``cstr_generative_model``/``run_mcmc_baseline`` were given an
additive ``mode: str = "closed_loop"`` parameter (default preserves ALL prior
behaviour exactly; ``mode="open_loop"`` branches to the open-loop simulator)
-- this worker always passes ``mode="open_loop"`` for both scenarios tested
here, so NUTS's internal model now matches the data-generating process.

**Sequential, not parallel, by design.** Only 2 scenarios x a modest window
count are needed here (not nb20's 8-scenario/160-window sweep), so this
notebook deliberately skips ``multiprocessing`` altogether -- avoiding the
entire class of bug nb20/`recompute_theoryC_section5.py` hit (oversubscription,
``Pool.apply_async`` timeouts that don't free a stuck worker, spawn-guard
pitfalls). A single long-running process with a real per-window
``multiprocessing.Process``-based timeout (started lazily only if a window
is observed to run away) is enough; see the notebook's own Theory-C-mirror
cell for how the timeout is applied at the call site, not inside this
function.
"""

from __future__ import annotations

import os
from pathlib import Path


def nuts_one_window_openloop(args):
    (sc_name, alpha, beta, drift_T, window_idx, root_str,
     n_chains, n_samples, n_warmup, cov_obs) = args
    # cov_obs: a (29,29) np.ndarray (full empirical summary-statistic
    # covariance at this scenario's true theta, precomputed ONCE per
    # scenario by the caller via the open-loop data-gen path) or None to
    # fall back to the diagonal default.

    # CPU-oversubscription fix (same lesson as nb20_nuts_worker.py): pin every
    # thread-pool-spawning math library to exactly 1 thread, before jax/numpy
    # are imported. Harmless even though this notebook runs sequentially (no
    # competing sibling processes) -- kept for parity/safety in case a future
    # caller does parallelise this worker.
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
        warm_start_ic, simulate_em_window_open_loop,
        apply_sensor_layer, DEFAULT_SENSOR_NOISE_PCT,
    )
    from cstr_sbi.summaries import compute_summary_statistics
    from cstr_sbi.inference import run_mcmc_baseline

    seed = 910_000 + (abs(hash(sc_name)) % 10_000) + window_idx

    # Observed-data generation -- IDENTICAL to nb20_nuts_worker.py's open_loop
    # branch: [UA_eff, k0_eff] = [beta, alpha] * [UA_NOMINAL, K0_NOMINAL], and
    # a FIXED (theta-independent) warm start at the healthy closed-loop
    # steady state, truncated to [C, T, Tc] (no controller state exists in
    # open loop).
    params_ol = jnp.array([beta * UA_NOMINAL, alpha * K0_NOMINAL], dtype=jnp.float64)
    y0_ol = warm_start_ic(
        jnp.array([UA_NOMINAL, K0_NOMINAL, 1.0, 1.0]), NOMINAL_INLET_CL, NOMINAL_CTRL
    )[:3]
    pk, sk = jax.random.split(jax.random.PRNGKey(seed))
    t, ys, qc = simulate_em_window_open_loop(params_ol, NOMINAL_INLET, y0_ol, key=pk)

    obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
    obs = apply_sensor_layer(obs, key=sk, noise_pct=DEFAULT_SENSOR_NOISE_PCT)
    if drift_T != 0.0:
        obs = obs.at[:, 1].add(drift_T)
    t_s = jnp.arange(1, ys.shape[0] + 1) * 0.5
    s = np.asarray(compute_summary_statistics(jnp.array(obs), jnp.array(t_s)))

    try:
        import numpyro.diagnostics as numpyro_diag

        # mode="open_loop" -- THE fix under test: NUTS's own generative model
        # now simulates the same open-loop physics the observed data above
        # was generated from, instead of (as before this notebook) silently
        # always fitting the closed-loop model.
        _mcmc, samp, wall = run_mcmc_baseline(
            s, seed=window_idx, n_chains=n_chains, n_samples=n_samples, n_warmup=n_warmup,
            progress_bar=False, cov_obs=cov_obs, mode="open_loop",
        )
        corr = float(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1])  # columns [alpha, beta]

        by_chain = _mcmc.get_samples(group_by_chain=True)
        diag_summary = numpyro_diag.summary(by_chain)
        r_hat_alpha = float(diag_summary["alpha"]["r_hat"])
        r_hat_beta = float(diag_summary["beta"]["r_hat"])
        n_eff_alpha = float(diag_summary["alpha"]["n_eff"])
        n_eff_beta = float(diag_summary["beta"]["n_eff"])
        extra = _mcmc.get_extra_fields()
        n_divergences = int(np.sum(np.asarray(extra["diverging"]))) if "diverging" in extra else -1
        # Same convention as nb20_nuts_worker.py / 05_mcmc_baseline.ipynb:
        # r_hat < 1.01 for BOTH parameters.
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


def nuts_one_window_openloop_to_queue(args, result_queue):
    """Thin wrapper for ``multiprocessing.Process``-based per-window timeout
    (real ``terminate()``, not ``Pool.apply_async(...).get(timeout=...)``,
    which only stops the caller from waiting -- see nb20_nuts_worker.py's
    docstring for the full story of why that distinction matters here)."""
    result_queue.put(nuts_one_window_openloop(args))
