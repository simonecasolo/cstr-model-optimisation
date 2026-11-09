"""SBI training and MCMC baseline — M4 + M5 deliverable.

This module provides two complementary inference engines that share the
same simulator and summary-statistics pipeline:

M5 — MCMC baseline (implemented here)
--------------------------------------
``cstr_generative_model`` is a NumPyro probabilistic model that:
  1. Samples ``[UA, k0, alpha, beta]`` from the 4-D BoxUniform prior.
  2. Runs ``simulate_em_window`` (Euler-Maruyama, JAX) to get one
     60-min trajectory.
  3. Computes the 27-D summary statistics via ``compute_summary_statistics``.
  4. Places a Gaussian likelihood on those summaries conditioned on the
     observed summary vector.

``run_mcmc_baseline`` wraps NumPyro NUTS around this model and returns
the raw ``MCMC`` object plus derived samples, timing, and chain arrays
suitable for R̂ / ESS diagnostics.

M4 — SBI training (stub, requires sbi + torch)
-----------------------------------------------
``simulation_wrapper_sbi`` bridges the JAX simulator to sbi's torch-tensor
API.  ``train_sbi_posterior`` runs SNPE_C with an NSF density estimator.
Both are implemented as lightweight wrappers that defer the heavy
``import sbi / import torch`` to call time so the module can be imported
without those packages (useful for M5-only runs).

Design notes
------------
* The Gaussian likelihood scale ``sigma_obs`` is estimated from the
  empirical std of the observed summary vector across the 50 healthy-
  scenario replicates in ``data/observations.npz``.  Alternatively, the
  caller can pass an explicit ``sigma_obs``.  We use a diagonal
  covariance (i.e., features are treated as independent) — a common
  approximation in ABC-style MCMC on summary statistics.

* NUTS is run with ``chain_method="vectorized"`` (single-device JAX
  vectorisation over chains) which is faster than ``"parallel"`` on a
  single CPU machine.

* The module is intentionally *stateless*: all state (trained posterior,
  MCMC object, timing) is returned by value so notebooks can cache them
  with standard pickle/npz.
"""

from __future__ import annotations

import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_sample

from cstr_sbi.physics import (
    NOMINAL_CTRL, NOMINAL_INLET_CL, NOMINAL_Y0_CL,
    simulate_closed_loop_trajectory,
    compute_qc,
)
from cstr_sbi.priors import (
    ALPHA_HIGH, ALPHA_LOW,
    BETA_HIGH, BETA_LOW,
    PARAM_NAMES_2D,
    PARAM_NAMES_4D,
)
from cstr_sbi.simulator import simulate_em_window, warm_start_ic
from cstr_sbi.summaries import compute_summary_statistics
from cstr_sbi.physics import simulate_open_loop_trajectory_fixed


# ---------------------------------------------------------------------------
# NumPyro generative model
# ---------------------------------------------------------------------------

def cstr_generative_model(
    obs_summary: jnp.ndarray | None = None,
    *,
    sigma_obs: jnp.ndarray | float = 1.0,
    cov_obs: jnp.ndarray | None = None,
    inlet: jnp.ndarray = NOMINAL_INLET_CL,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray = NOMINAL_Y0_CL,
    t_window: float = 60.0,
    dt_out: float = 0.5,
    n_save: int | None = None,
    matched_warm_start: bool = False,
    mode: str = "closed_loop",
):
    """NumPyro generative model for the **2-D** CSTR inference problem.

    Prior: independent Uniform on ``[alpha, beta]``.
    UA and k0 are fixed at their nominal design constants (UA_NOMINAL,
    K0_NOMINAL) — they are NOT sampled.  See spec §3.3 and notebook 05 §5a
    for why UA and k0 must be fixed (structural non-identifiability).

    Likelihood: diagonal Gaussian on the 29-D summary statistics by default
    (``cov_obs=None``), OR a full-covariance multivariate Gaussian if
    ``cov_obs`` (a ``(29, 29)`` matrix) is given.

    ADDED (2026-11-06, coordinator-flagged): the diagonal default assumes the
    29 engineered summary statistics are independent given theta, which is
    the SAME assumption `notebooks/system_I/20_...ipynb`'s Theory E found to
    be wrong for the local-FIM diagnostic (the full empirical covariance has
    large off-diagonal terms; using it instead of the diagonal approximation
    closed most of the local-FIM-vs-trained-posterior gap at several
    scenarios). Passing the same full empirical covariance here as `cov_obs`
    makes NUTS's own likelihood internally consistent with that finding,
    rather than NUTS silently inheriting the same misspecification Theory E
    diagnosed elsewhere. `sigma_obs` is ignored when `cov_obs` is given.

    Uses the **deterministic diffrax Tsit5** integrator (not Euler-Maruyama)
    so that the model is fully differentiable and JIT-compiles end-to-end
    inside NUTS.  See notebook 05 §3 for the integrator choice rationale.

    Parameters
    ----------
    obs_summary
        The observed 29-D summary vector.  Pass ``None`` for prior-
        predictive sampling (``numpyro.infer.Predictive``).
    sigma_obs
        Likelihood noise scale(s) — either a scalar or a ``(29,)`` array.
        Estimated from data if not provided; see ``estimate_sigma_obs``.
    inlet, ctrl, y0
        Simulator inputs (fixed at the nominal values by default).
    t_window
        Observation window length in minutes (default 60).
    dt_out
        Output grid spacing in minutes (default 0.5).
    n_save
        Number of saved time points.  If ``None``, computed from
        ``t_window / dt_out``.
    matched_warm_start
        ADDED 2026-11-07 (coordinator-flagged mismatch, notebook 20 Section 5).
        If ``False`` (default, UNCHANGED prior behaviour): every posterior
        draw's trajectory starts from the SAME fixed ``y0`` (healthy nominal
        steady state by default) regardless of the sampled ``alpha,beta`` --
        this is the "fixed warm start" protocol. If ``True``: ``y0`` is
        instead recomputed PER DRAW as ``warm_start_ic(params, inlet, ctrl)``
        at that draw's own sampled ``(alpha, beta)`` -- the exact same
        function/call SBI's "matched protocol" training
        (``scenario_specific_warm_start=True`` in ``simulation_wrapper_sbi``)
        uses to generate its training data. The external ``y0`` argument is
        ignored when ``matched_warm_start=True``. This matters because the
        OBSERVED data NUTS is asked to fit (when generated via this
        project's own matched-protocol simulator wrappers, e.g.
        ``sim_raw_po`` in notebook 20) is itself generated under the matched
        convention -- with ``matched_warm_start=False``, NUTS's internal
        model and the data-generating process it is fitting are literally
        different generative models (a train/eval protocol mismatch
        structurally identical to the one already found and fixed for SBI's
        beta-bias, apparently never ported to this NUTS model).
    mode
        ADDED 2026-11-07 (notebook 21, open-loop model-mismatch diagnosis). If
        ``"closed_loop"`` (default, UNCHANGED prior behaviour), this function
        behaves EXACTLY as before -- it always simulates the PI-controller
        closed-loop dynamics regardless of what generated the observed data.
        If ``"open_loop"``: branches to simulate the 3-state open-loop CSTR
        (``simulate_open_loop_trajectory_fixed``, fixed-step Tsit5, NUTS/JIT-
        safe) with ``[UA, k0]`` scaled directly by ``[beta, alpha]`` (same
        fault-encoding convention as ``ScenarioConfig.open_loop_params`` and
        ``scripts/nb20_nuts_worker.py``'s open-loop data-gen branch) and a
        FIXED, theta-independent warm start (the healthy-nominal closed-loop
        steady state truncated to its first 3 states, ``[C, T, Tc]`` --
        matching ``nb20_nuts_worker``'s own open-loop convention, since the
        open-loop simulator has no feedback to re-settle to a new steady
        state under a fault, by construction). ``matched_warm_start`` is
        ignored when ``mode="open_loop"`` (there is nothing to match: the
        open-loop warm start was never theta-dependent in the first place,
        for either the model or the data it is fitting). This exists because
        ``cstr_generative_model`` was found (notebook 20, Theory C) to ALWAYS
        use the closed-loop simulator even when the real observed data came
        from Sc0/Sc6's open-loop generation -- a genuine model mismatch (not
        just a bad warm start), diagnosed as the likely cause of those two
        scenarios' badly non-converging NUTS chains (R-hat in the
        hundreds-to-thousands) even after the diagonal-vs-full-covariance
        likelihood fix. ``inlet``/``ctrl`` keep their closed-loop-shaped
        defaults (``NOMINAL_INLET_CL`` is 3-D, ``[Ci,Ti,Tci]``) for backward
        compatibility; in open-loop mode the constant coolant flow ``Qc`` is
        taken from ``ctrl[3]`` (``NOMINAL_CTRL``'s ``Qc0`` entry, identical
        value to ``NOMINAL_INLET``'s own 4th entry) so the same default
        arguments work for both modes without the caller having to pass a
        separate 4-D open-loop inlet.
    """
    from cstr_sbi.physics import K0_NOMINAL, UA_NOMINAL
    from cstr_sbi.simulator import warm_start_ic

    alpha = numpyro.sample("alpha", dist.Uniform(ALPHA_LOW, ALPHA_HIGH))
    beta  = numpyro.sample("beta",  dist.Uniform(BETA_LOW,  BETA_HIGH))

    if n_save is None:
        n_save = int(round(t_window / dt_out))

    if mode == "open_loop":
        # [UA_eff, k0_eff] = [beta, alpha] * [UA_NOMINAL, K0_NOMINAL] -- same
        # fault-encoding convention as ScenarioConfig.open_loop_params() and
        # nb20_nuts_worker's open-loop data-gen branch.
        params_ol = jnp.array([beta * UA_NOMINAL, alpha * K0_NOMINAL])
        inlet_ol = jnp.concatenate([jnp.asarray(inlet)[:3], jnp.asarray(ctrl)[3:4]])
        y0_ol = jnp.asarray(y0)[:3]

        _ts, ys_ol = simulate_open_loop_trajectory_fixed(
            params_ol, inlet_ol, y0_ol,
            t_final=t_window, n_save=n_save,
        )
        qc_const = jnp.full((ys_ol.shape[0],), inlet_ol[3])
        obs_packed = jnp.stack([ys_ol[:, 0], ys_ol[:, 1], ys_ol[:, 2], qc_const], axis=1)
        summary = compute_summary_statistics(obs_packed, _ts)
    else:
        # UA and k0 are fixed design constants — not sampled.
        params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta])

        y0_used = warm_start_ic(params, inlet, ctrl) if matched_warm_start else y0

        ts, ys, qc = simulate_closed_loop_trajectory(
            params, inlet, ctrl, y0_used,
            t_final=t_window, n_save=n_save,
        )
        obs_packed = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
        summary = compute_summary_statistics(obs_packed, ts)

    if cov_obs is not None:
        likelihood = dist.MultivariateNormal(loc=summary, covariance_matrix=cov_obs)
    else:
        likelihood = dist.Normal(summary, sigma_obs).to_event(1)

    numpyro.sample(
        "obs",
        likelihood,
        obs=obs_summary,
    )


# ---------------------------------------------------------------------------
# Estimate observation noise scale from data
# ---------------------------------------------------------------------------

def estimate_sigma_obs(
    observations_npz: str | None = None,
    scenario_id_filter: int | None = 1,
    floor: float = 1e-3,
) -> np.ndarray:
    """Estimate per-feature std from healthy-scenario replicates.

    Uses the 50 Sc1 (healthy closed-loop) replicates in
    ``data/observations.npz`` by default.  The floor prevents zero
    sigma on perfectly constant features.

    Parameters
    ----------
    observations_npz
        Path to the .npz file.  If ``None``, resolves to
        ``../data/observations.npz`` relative to this file.
    scenario_id_filter
        Restrict to this scenario id (default: 1 = healthy closed-loop).
        Pass ``None`` to use all scenarios.
    """
    from pathlib import Path
    if observations_npz is None:
        observations_npz = str(
            Path(__file__).parent.parent.parent / "data" / "observations.npz"
        )
    d = np.load(observations_npz, allow_pickle=False)
    x = jnp.asarray(d["x"])
    t = jnp.asarray(d["t"])
    sid = d["scenario_id"]

    mask = slice(None) if scenario_id_filter is None else (sid == scenario_id_filter)
    x_sel = x[mask]

    from cstr_sbi.summaries import compute_summary_statistics_batch
    S = np.asarray(compute_summary_statistics_batch(x_sel, t))
    sigma = np.std(S, axis=0)
    sigma = np.maximum(sigma, floor)
    return sigma.astype(np.float32)


# ---------------------------------------------------------------------------
# NUTS runner
# ---------------------------------------------------------------------------

def run_mcmc_baseline(
    obs_summary: np.ndarray,
    *,
    sigma_obs: np.ndarray | float | None = None,
    cov_obs: np.ndarray | None = None,
    inlet: jnp.ndarray = NOMINAL_INLET_CL,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray | None = None,
    matched_warm_start: bool = False,
    mode: str = "closed_loop",
    n_chains: int = 4,
    n_samples: int = 500,
    n_warmup: int = 300,
    seed: int = 0,
    observations_npz: str | None = None,
    progress_bar: bool = True,
) -> Tuple[MCMC, np.ndarray, float]:
    """Run NUTS on ``cstr_generative_model`` for a single observed summary.

    Parameters
    ----------
    obs_summary
        27-D observed summary vector (output of
        ``compute_summary_statistics``).
    sigma_obs
        Likelihood scale.  If ``None``, estimated from the Sc1 replicates
        in ``data/observations.npz`` via ``estimate_sigma_obs``. Ignored if
        ``cov_obs`` is given.
    cov_obs
        ADDED 2026-11-06: optional full ``(29, 29)`` likelihood covariance
        matrix. If given, ``cstr_generative_model`` uses a full-covariance
        multivariate Gaussian likelihood instead of the diagonal default --
        see that function's own docstring for why (Theory E's full-vs-diagonal
        finding in notebook 20).
    y0
        Initial state for the Euler-Maruyama integrator.  If ``None``,
        uses the warm-start IC computed from the nominal closed-loop
        parameters (truncated to the first 3 states, ``[C, T, Tc]``, when
        ``mode="open_loop"`` -- see ``cstr_generative_model``'s docstring).
    mode
        ADDED 2026-11-07 (notebook 21): passed straight through to
        ``cstr_generative_model``. ``"closed_loop"`` (default) preserves all
        existing behaviour; ``"open_loop"`` fits the open-loop generative
        model instead -- see that function's own docstring for the full
        rationale and the ``matched_warm_start``-is-ignored caveat in that
        mode.
    n_chains, n_samples, n_warmup
        NUTS budget.  Defaults give ~30–60 min wall time on a single CPU
        for a 27-D summary likelihood.
    seed
        JAX random seed.
    progress_bar
        Show tqdm progress bar during sampling.

    Returns
    -------
    mcmc
        The finished ``numpyro.infer.MCMC`` object (call
        ``mcmc.print_summary()`` for diagnostics).
    samples_arr
        Shape ``(n_chains * n_samples, 4)`` flat sample array
        ``[UA, k0, alpha, beta]``.
    wall_time_s
        Total wall-clock time in seconds (includes warmup).
    """
    if cov_obs is not None:
        cov_obs = jnp.asarray(cov_obs, dtype=jnp.float32)
        sigma_obs = jnp.ones(cov_obs.shape[0], dtype=jnp.float32)  # unused when cov_obs is set
    elif sigma_obs is None:
        sigma_obs = jnp.asarray(estimate_sigma_obs(observations_npz))
    else:
        sigma_obs = jnp.asarray(sigma_obs, dtype=jnp.float32)

    obs_jnp = jnp.asarray(obs_summary, dtype=jnp.float32)

    if y0 is None:
        from cstr_sbi.physics import NOMINAL_PARAMS_CL
        y0 = warm_start_ic(NOMINAL_PARAMS_CL, inlet, ctrl)
        if mode == "open_loop":
            # Open-loop state is [C, T, Tc] -- no controller/integral state.
            # Fixed, theta-independent warm start (see cstr_generative_model's
            # docstring): mirrors nb20_nuts_worker's open-loop data-gen branch.
            y0 = y0[:3]

    kernel = NUTS(
        cstr_generative_model,
        init_strategy=init_to_sample,
        target_accept_prob=0.80,
        max_tree_depth=10,
        dense_mass=True,   # learns full covariance; essential for the UA-β ridge
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        chain_method="vectorized",
        progress_bar=progress_bar,
    )

    t0 = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(seed),
        obs_summary=obs_jnp,
        sigma_obs=sigma_obs,
        cov_obs=cov_obs,
        inlet=jnp.asarray(inlet),
        ctrl=jnp.asarray(ctrl),
        y0=jnp.asarray(y0),
        matched_warm_start=matched_warm_start,
        mode=mode,
    )
    wall_time_s = time.perf_counter() - t0

    raw = mcmc.get_samples()
    samples_arr = np.stack(
        [np.asarray(raw[p]) for p in PARAM_NAMES_2D], axis=-1
    )  # (n_chains * n_samples, 2)

    return mcmc, samples_arr, wall_time_s


def get_chain_array(mcmc: MCMC) -> np.ndarray:
    """Return samples reshaped as ``(n_chains, n_draws, 2)`` for R̂/ESS.

    Returns the 2-D ``[alpha, beta]`` chain array.
    Requires that the MCMC was run with ``chain_method='vectorized'``
    or ``'parallel'``.
    """
    raw = mcmc.get_samples(group_by_chain=True)
    return np.stack(
        [np.asarray(raw[p]) for p in PARAM_NAMES_2D], axis=-1
    )  # (n_chains, n_draws, 2)


# ---------------------------------------------------------------------------
# SBI training (M4 — requires sbi + torch; deferred import)
# ---------------------------------------------------------------------------

def simulation_wrapper_sbi(
    theta_torch,
    inlet: jnp.ndarray = NOMINAL_INLET_CL,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray = NOMINAL_Y0_CL,
    t_window: float = 60.0,
    dt: float = 0.01,
    dt_out: float = 0.5,
    seed: int = 0,
    scenario_specific_warm_start: bool = False,
):
    """Bridge between sbi's torch-tensor parameter batch and the JAX simulator.

    Parameters
    ----------
    theta_torch
        ``torch.Tensor`` of shape ``(n_batch, 2)`` with rows ``[alpha, beta]``.
        UA and k0 are fixed at their nominal values internally.
    y0
        Warm-start initial condition ``[C, T, Tc, I]``. Defaults to
        ``NOMINAL_Y0_CL`` (the healthy closed-loop steady-state IC). Ignored
        when ``scenario_specific_warm_start=True`` (see below).
        **Pass a pre-computed warm IC here rather than calling
        ``warm_start_ic`` inside the loop** — per-sample warm-starting adds
        O(n_batch) expensive ODE solves and may fail for extreme prior draws.
    scenario_specific_warm_start
        2026-08-13, reviewer_response_plan.md Major Comment 7 / Stage 3 item 8:
        if True, compute a per-sample warm start ``warm_start_ic(params, ...)``
        for *this training draw's own* (alpha, beta) instead of reusing the
        single fixed ``y0`` for every prior draw. This matches the protocol
        already used for the *evaluation* data in ``notebooks/02_data_generation.ipynb``
        (each scenario is warm-started at its own steady state), closing the
        train/test initial-condition mismatch this investigation found: with
        the default ``False``, every training window contains an implicit
        healthy-onset transient regardless of how faulted the sampled theta
        is, while evaluation windows never do. Costs ~0.35s/call extra (one
        additional `diffrax` steady-state solve per training sample) — for
        ``n_simulations=10_000`` this adds roughly an hour; use a reduced
        budget for a diagnostic run.

    Returns
    -------
    ``torch.Tensor`` of shape ``(n_batch, 29)`` — summary statistics.
    """
    import torch
    from cstr_sbi.physics import K0_NOMINAL, UA_NOMINAL
    from cstr_sbi.simulator import apply_sensor_layer, DEFAULT_SENSOR_NOISE_PCT
    # warm_start_ic already imported at module level (line 69)

    theta_np = theta_torch.detach().cpu().numpy()
    n_batch = theta_np.shape[0]

    summaries = []
    for i in range(n_batch):
        alpha, beta = float(theta_np[i, 0]), float(theta_np[i, 1])
        # Build full 4-D params with fixed UA and k0.
        params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float32)
        y0_i = warm_start_ic(params, inlet, ctrl) if scenario_specific_warm_start else y0
        proc_key, sens_key = jax.random.split(jax.random.PRNGKey(seed + i))
        _, ys, qc = simulate_em_window(
            params, inlet, ctrl, y0_i,
            key=proc_key, t_window=t_window, dt=dt, dt_out=dt_out,
        )
        t_out = jnp.arange(1, ys.shape[0] + 1) * dt_out
        obs_packed = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
        # Apply sensor noise to match the distribution of observations.npz
        obs_packed = apply_sensor_layer(obs_packed, key=sens_key,
                                        noise_pct=DEFAULT_SENSOR_NOISE_PCT)
        s = compute_summary_statistics(obs_packed, t_out)
        summaries.append(np.asarray(s))

    return torch.tensor(np.stack(summaries), dtype=torch.float32)


def simulation_wrapper_sbi_raw(
    theta_torch,
    inlet: jnp.ndarray = NOMINAL_INLET_CL,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray = NOMINAL_Y0_CL,
    t_window: float = 60.0,
    dt: float = 0.01,
    dt_out: float = 0.5,
    seed: int = 0,
    scenario_specific_warm_start: bool = False,
):
    """Like ``simulation_wrapper_sbi`` but returns raw flattened observations.

    See ``simulation_wrapper_sbi``'s docstring for ``scenario_specific_warm_start``.

    Returns
    -------
    ``torch.Tensor`` of shape ``(n_batch, n_timesteps * 4)`` — raw [C, T, Tc, Qc].
    """
    import torch
    from cstr_sbi.physics import K0_NOMINAL, UA_NOMINAL
    from cstr_sbi.simulator import apply_sensor_layer, DEFAULT_SENSOR_NOISE_PCT
    # warm_start_ic already imported at module level (line 69)

    theta_np = theta_torch.detach().cpu().numpy()
    n_batch = theta_np.shape[0]

    raw_obs = []
    for i in range(n_batch):
        alpha, beta = float(theta_np[i, 0]), float(theta_np[i, 1])
        params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float32)
        y0_i = warm_start_ic(params, inlet, ctrl) if scenario_specific_warm_start else y0
        proc_key, sens_key = jax.random.split(jax.random.PRNGKey(seed + i))
        _, ys, qc = simulate_em_window(
            params, inlet, ctrl, y0_i,
            key=proc_key, t_window=t_window, dt=dt, dt_out=dt_out,
        )
        obs_packed = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
        obs_packed = apply_sensor_layer(obs_packed, key=sens_key,
                                        noise_pct=DEFAULT_SENSOR_NOISE_PCT)
        raw_obs.append(np.asarray(obs_packed).reshape(-1))

    return torch.tensor(np.stack(raw_obs), dtype=torch.float32)


def train_sbi_posterior(
    prior,
    n_simulations: int = 10_000,
    *,
    density_estimator: str = "nsf",
    hidden_features: int = 128,
    num_transforms: int = 5,
    training_batch_size: int = 256,
    max_num_epochs: int = 200,
    save_to: str | None = None,
    seed: int = 0,
    inlet: jnp.ndarray = NOMINAL_INLET_CL,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray = NOMINAL_Y0_CL,
    embedding_net: "torch.nn.Module | None" = None,
    use_raw_obs: bool = False,
    scenario_specific_warm_start: bool = False,
):
    """Train an SNPE_C posterior with an NSF density estimator (M4).

    Parameters
    ----------
    prior
        A ``sbi``-compatible prior from ``box_uniform_2d()`` — 2-D ``[α, β]``.
    n_simulations
        Number of prior draws used to build the training set.
    density_estimator
        ``"nsf"`` (Neural Spline Flow, recommended) or ``"maf"``.
    save_to
        If given, pickle the trained posterior to this path.
    embedding_net
        Optional ``torch.nn.Module`` that maps raw observations to a learned
        embedding. When provided, ``z_score_x="structured"`` is used.
        See ``sbi.neural_nets.embedding_nets`` for built-in options.
    use_raw_obs
        If True, pass raw flattened ``(n_timesteps * 4,)`` observations to the
        density estimator instead of 29-D summary statistics. Requires
        ``embedding_net`` to be set.
    scenario_specific_warm_start
        2026-08-13, matched-initialization-protocol experiment (Major Comment 7):
        forwarded to ``simulation_wrapper_sbi`` — if True, every training
        simulation is warm-started at its own sampled (alpha, beta)'s steady
        state, matching the evaluation-data protocol, instead of the single
        fixed healthy warm-start ``y0`` used for every prior draw by default.

    Returns
    -------
    posterior
        A trained ``sbi`` posterior object with a ``.sample()`` method.
    metadata : dict
        Training metadata (n_simulations, method, timing, etc.).
    """
    try:
        import torch
        from sbi.inference import SNPE, simulate_for_sbi
        from sbi.neural_nets import posterior_nn
    except ImportError as e:
        raise ImportError(
            "sbi and torch must be installed. Run: pip install sbi torch"
        ) from e

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    z_score_x = "structured" if embedding_net is not None else "independent"
    density_estimator_build_fun = posterior_nn(
        model=density_estimator,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        embedding_net=embedding_net or torch.nn.Identity(),
        z_score_x=z_score_x,
    )
    inference_obj = SNPE(
        prior=prior,
        density_estimator=density_estimator_build_fun,
        show_progress_bars=False,
    )

    _counter = [int(rng.integers(0, 2**31))]
    sim_fn = simulation_wrapper_sbi_raw if use_raw_obs else simulation_wrapper_sbi

    def wrapper(theta: "torch.Tensor") -> "torch.Tensor":
        _counter[0] += 1
        return sim_fn(theta, inlet=inlet, ctrl=ctrl, y0=y0,
                      seed=_counter[0],
                      scenario_specific_warm_start=scenario_specific_warm_start)

    t0 = time.perf_counter()
    theta, x = simulate_for_sbi(
        simulator=wrapper,
        proposal=prior,
        num_simulations=n_simulations,
        num_workers=1,
        show_progress_bar=True,
    )
    inference_obj.append_simulations(theta, x)

    import io, sys
    _captured = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = _captured
    try:
        density_estimator_trained = inference_obj.train(
            training_batch_size=training_batch_size,
            max_num_epochs=max_num_epochs,
            show_train_summary=False,
        )
    finally:
        sys.stdout = _orig_stdout

    posterior = inference_obj.build_posterior(density_estimator_trained)
    wall_time_s = time.perf_counter() - t0

    summary = getattr(inference_obj, "_summary", {})
    n_epochs = summary.get("epochs_trained", ["?"])[-1]
    best_val = summary.get("best_validation_loss", [None])[-1]
    best_val_str = f"{best_val:.4f}" if best_val is not None else "?"
    print(f"Training complete: {n_epochs} epochs, "
          f"best val loss = {best_val_str}, "
          f"wall time = {wall_time_s:.0f}s")

    metadata = {
        "n_simulations": n_simulations,
        "density_estimator": density_estimator,
        "hidden_features": hidden_features,
        "num_transforms": num_transforms,
        "training_batch_size": training_batch_size,
        "max_num_epochs": max_num_epochs,
        "wall_time_s": wall_time_s,
    }

    if save_to is not None:
        import pickle
        with open(save_to, "wb") as f:
            pickle.dump({"posterior": posterior, "metadata": metadata}, f)

    return posterior, metadata


def sample_posterior(
    posterior,
    obs_summary: np.ndarray,
    n_samples: int = 10_000,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Draw samples from a trained 2-D SBI posterior ``[alpha, beta]``.

    Returns shape ``(n_samples, 2)`` array.
    """
    import torch
    x_obs = torch.tensor(obs_summary, dtype=torch.float32)
    samples = posterior.sample(
        (n_samples,), x=x_obs, show_progress_bars=False,
        # Disable rejection sampling at the prior boundary — the trained
        # posterior may place some mass outside [0.4,1.0]^2 due to the
        # IC mismatch documented in M4/M6. We sample the raw neural density
        # and interpret the marginals; re-training with correct ICs will fix
        # the out-of-prior mass.
        reject_outside_prior=False,
    )
    return samples.detach().cpu().numpy()
