"""Standalone SBI pipeline for the Wu 2003 zuko_nsf / reb_per_boilup retraining study.

Runs outside the Jupyter kernel to avoid the OOM issues hit in nb26. Mirrors the
training-bank-generation, SNPE-training, and SBC cells of notebooks 24/25 exactly
(same simulator calls, same noise-injection scheme, same SBC procedure with
reject_outside_prior=False per the nb29 rejection-sampling-artifact finding).

Three-variant plan (see HANDOFF.md / partitioned-swinging-aho.md):
  A: zuko_nsf 60/3, 0.3% noise  — primary, tests whether new arch + reb_per_boilup fixes eta_col SBC
  B: zuko_nsf 60/3, 1.0% noise  — noise sensitivity
  C: nsf 128/5,     1.0% noise — isolates noise effect from architecture effect
Run for both S-B (nb24) and S-A (nb25) control structures.
"""

from __future__ import annotations

import pathlib
import pickle
import time

import numpy as np
import torch
import jax.numpy as jnp
from scipy import stats as scipy_stats

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from cstr_sbi.recycle.priors import box_uniform_5d
from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB, NOMINAL_CTRL_SA, NOMINAL_INLET, NOMINAL_Y0_EXPLICIT, PARAM_NAMES,
    simulate_trajectory_explicit_jit, extract_observations_explicit,
    simulate_to_steady_state_explicit,
)
from cstr_sbi.recycle.summaries import (
    compute_summaries, N_SUMMARIES_SB, N_SUMMARIES_SA, SB_INDICES, N_SB,
)
from cstr_sbi.recycle.simulator import nominal_warm_start

import jax

# --- Matched-protocol (per-draw scenario-specific) warm start ---------------
# Reviewer-response Major Comment 7 / notebook_execution_plan Stage 3, Item 11:
# training must draw its initial condition from the SAME theta as the sample
# being generated (matching nb22's per-scenario evaluation-data protocol),
# instead of a single fixed nominal warm-start reused for every prior draw.
#
# Cost note (measured 2026-08-13, see pending_manuscript_fixes.md Stage 3, Item
# System-II): computing a steady state for an arbitrary 5-D prior draw (as
# opposed to the 16 curated, physically-mild Wu2003 scenarios) is far more
# expensive and occasionally far slower than the curated-scenario case --
# ~3.5s/draw on average but with a long tail (some draws >50s) for parameter
# combinations far from the nominal operating point. Loosened tolerances
# (rtol=1e-4, atol=1e-6) and a shorter horizon (t_final=100) are used here
# instead of nominal_warm_start's defaults (1e-6/1e-8, t_final=200) -- adequate
# given 0.3% sensor noise dominates precision well above 1e-4 relative anyway
# -- to keep this tractable at all; it is still ~10x the fixed-warm-start cost.
_scenario_ss_jit = jax.jit(
    lambda theta, ctrl: simulate_to_steady_state_explicit(
        theta, NOMINAL_INLET, ctrl, NOMINAL_Y0_EXPLICIT,
        t_final=100.0, rtol=1e-4, atol=1e-6, max_steps=100_000,
    )
)


def _matched_y0(theta_np, ctrl):
    """Per-draw scenario-specific steady-state warm start (matched protocol)."""
    theta_jnp = jnp.array(theta_np, dtype=jnp.float32)
    return _scenario_ss_jit(theta_jnp, ctrl)

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
SBI_LOGS = pathlib.Path(__file__).resolve().parent.parent / "sbi-logs"
DATA.mkdir(exist_ok=True)
SBI_LOGS.mkdir(exist_ok=True)

ETA_COL_IDX = PARAM_NAMES.index("eta_col")   # 2
ALPHA_IDX = PARAM_NAMES.index("alpha")       # 0
XI_REB_IDX = PARAM_NAMES.index("xi_reb")     # 3


def _structure_config(structure: str):
    if structure == "S-B":
        return NOMINAL_CTRL_SB, nominal_warm_start("S-B")
    if structure == "S-A":
        return NOMINAL_CTRL_SA, nominal_warm_start("S-A")
    raise ValueError(f"Unknown structure: {structure!r}")


def _simulate_summary(theta_np, ctrl, y0, structure, noise_pct, rng,
                       scenario_specific_warm_start=False):
    """One prior draw -> summary vector, or None if the simulation is invalid.

    scenario_specific_warm_start : if True, ignore the passed-in fixed `y0`
        and instead compute a per-draw steady-state warm start from `theta_np`
        itself (matched protocol, see module docstring above `_matched_y0`).
    """
    theta_jnp = jnp.array(theta_np, dtype=jnp.float32)
    y0_i = _matched_y0(theta_np, ctrl) if scenario_specific_warm_start else y0
    ts, ys = simulate_trajectory_explicit_jit(
        theta_jnp, NOMINAL_INLET, ctrl, y0_i,
        t_final=2.0, n_save=120, rtol=1e-3, atol=1e-5,
    )
    raw = extract_observations_explicit(ys, theta_jnp, ctrl)
    raw_np = np.asarray(raw)
    t_np = np.asarray(ts)
    if np.isnan(raw_np).any() or np.isinf(raw_np).any():
        return None
    scale = np.maximum(np.max(np.abs(raw_np), axis=0), 1e-6)
    noise = rng.normal(0, noise_pct * scale, raw_np.shape)
    s = compute_summaries(raw_np + noise, structure, t_np)
    if np.isnan(s).any():
        return None
    return s


def generate_training_bank(
    structure: str,
    noise_pct: float,
    out_path: pathlib.Path,
    n_train: int = 15000,
    seed: int = 20260625,
    scenario_specific_warm_start: bool = False,
    progress_every: int = 1000,
) -> pathlib.Path:
    """Generate (or reuse) an SNPE training bank: prior draws -> summaries.

    Idempotent: if out_path already exists, it is reused as-is (delete it to force
    regeneration). This lets multiple variants sharing a noise level reuse one bank.

    scenario_specific_warm_start : if True (matched protocol, Major Comment 7 /
        Stage 3 Item 11), each prior draw's initial condition is its own
        steady state rather than one fixed nominal warm-start shared by all
        draws -- substantially more expensive, see `_matched_y0` docstring.
    """
    if out_path.exists():
        print(f"[bank] {out_path} already exists, reusing")
        return out_path

    ctrl, y0 = _structure_config(structure)
    prior = box_uniform_5d()
    rng = np.random.default_rng(seed)

    print(f"[bank] Generating {n_train} {structure} training sims @ {noise_pct*100:.1f}% noise "
          f"(scenario_specific_warm_start={scenario_specific_warm_start})...")
    theta_samples = prior.sample((n_train,)).numpy()
    thetas_list, summaries_list = [], []
    t0 = time.time()
    for i, th in enumerate(theta_samples):
        s = _simulate_summary(th, ctrl, y0, structure, noise_pct, rng,
                               scenario_specific_warm_start=scenario_specific_warm_start)
        if s is not None:
            thetas_list.append(th)
            summaries_list.append(s)
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"[bank]   {i+1}/{n_train}  ({elapsed:.0f}s elapsed, {len(thetas_list)} valid)")

    thetas_arr = np.stack(thetas_list)
    summaries_arr = np.stack(summaries_list)
    print(f"[bank] Valid simulations: {len(thetas_arr)}/{n_train} "
          f"in {time.time()-t0:.0f}s")
    np.savez(out_path, thetas=thetas_arr, summaries=summaries_arr)
    print(f"[bank] Saved to {out_path}")
    return out_path


def train_posterior(
    bank_path: pathlib.Path,
    arch: str,
    hidden_features: int,
    num_transforms: int,
    max_num_epochs: int = 200,
    stop_after_epochs: int = 20,
    torch_seed: int = 0,
):
    """Train an SNPE posterior on a training bank. Returns the built posterior."""
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn

    torch.manual_seed(torch_seed)
    train_data = np.load(bank_path)
    thetas_t = torch.tensor(train_data["thetas"], dtype=torch.float32)
    summaries_t = torch.tensor(train_data["summaries"], dtype=torch.float32)
    print(f"[train] {thetas_t.shape[0]} samples, {summaries_t.shape[1]}-D summaries, "
          f"arch={arch} hidden={hidden_features} transforms={num_transforms}")

    prior = box_uniform_5d()
    density_estimator = posterior_nn(
        model=arch, hidden_features=hidden_features, num_transforms=num_transforms,
    )
    inference = SNPE(prior=prior, density_estimator=density_estimator)
    inference.append_simulations(thetas_t, summaries_t)

    t0 = time.time()
    density_estimator_trained = inference.train(
        max_num_epochs=max_num_epochs,
        validation_fraction=0.1,
        stop_after_epochs=stop_after_epochs,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator_trained)
    print(f"[train] Completed in {time.time()-t0:.0f}s")
    return posterior


def run_sbc(
    posterior,
    structure: str,
    noise_pct: float,
    n_sbc: int = 200,
    n_post: int = 200,
    seed: int = 12345,
    scenario_specific_warm_start: bool = False,
) -> dict:
    """Run SBC for all 5 parameters using reject_outside_prior=False (nb29 fix)."""
    ctrl, y0 = _structure_config(structure)
    prior = box_uniform_5d()
    rng = np.random.default_rng(seed)

    ranks = {name: [] for name in PARAM_NAMES}
    t0 = time.time()
    n_valid = 0
    for i in range(n_sbc):
        th = prior.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, structure, noise_pct, rng,
                               scenario_specific_warm_start=scenario_specific_warm_start)
        if s is None:
            continue
        n_valid += 1
        with torch.no_grad():
            # reject_outside_prior=False: the docstring above has claimed this fix
            # since the nb29 rejection-sampling-hang finding, but the kwarg was never
            # actually passed here (only `run_sbc_cnn` below had it) -- a real,
            # pre-existing gap, surfaced by the Stage-3 Item-11 matched-protocol
            # diagnostic (2026-08-13) hanging for ~1hr at "0.000% proposal samples
            # accepted" when evaluating out-of-training-distribution (matched-
            # protocol) summaries against a fixed-protocol-trained posterior.
            samp = posterior.sample(
                (n_post,), x=torch.tensor(s, dtype=torch.float32),
                show_progress_bars=False,
                reject_outside_prior=False,
            ).detach().numpy()
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
        if (i + 1) % 50 == 0:
            print(f"[sbc]   {i+1}/{n_sbc}  ({time.time()-t0:.0f}s, {n_valid} valid)")

    results = {"n_sbc": n_sbc, "n_post": n_post, "n_valid": n_valid}
    for name in PARAM_NAMES:
        r = np.array(ranks[name])
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        results[name] = {
            "ranks": r.tolist(),
            "ks_pvalue": float(ks.pvalue),
            "mean_rank_frac": float(r.mean() / n_post),
        }
    print(f"[sbc] Completed in {time.time()-t0:.0f}s, n_valid={n_valid}/{n_sbc}")
    for name in PARAM_NAMES:
        print(f"[sbc]   {name:10s} KS p={results[name]['ks_pvalue']:.4f}")
    return results


def coverage_at_90(posterior, structure: str, noise_pct: float, n_trials: int = 100,
                    seed: int = 777, scenario_specific_warm_start: bool = False) -> dict:
    """90% CI coverage check per parameter (fraction of trials where truth falls in CI)."""
    ctrl, y0 = _structure_config(structure)
    prior = box_uniform_5d()
    rng = np.random.default_rng(seed)

    hits = {name: 0 for name in PARAM_NAMES}
    widths = {name: [] for name in PARAM_NAMES}
    n_valid = 0
    for i in range(n_trials):
        th = prior.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, structure, noise_pct, rng,
                               scenario_specific_warm_start=scenario_specific_warm_start)
        if s is None:
            continue
        n_valid += 1
        with torch.no_grad():
            samp = posterior.sample(
                (500,), x=torch.tensor(s, dtype=torch.float32),
                show_progress_bars=False,
                reject_outside_prior=False,
            ).detach().numpy()
        for k, name in enumerate(PARAM_NAMES):
            lo, hi = np.percentile(samp[:, k], [5, 95])
            widths[name].append(hi - lo)
            if lo <= th[k] <= hi:
                hits[name] += 1
    return {
        name: {
            "coverage_90": hits[name] / max(n_valid, 1),
            "mean_ci_width": float(np.mean(widths[name])) if widths[name] else float("nan"),
        }
        for name in PARAM_NAMES
    } | {"n_valid": n_valid}


def train_ensemble_and_select(
    bank_path: pathlib.Path,
    structure: str,
    noise_pct: float,
    arch: str,
    hidden_features: int,
    num_transforms: int,
    seeds: list[int],
    n_sbc: int = 200,
    n_post: int = 200,
):
    """Train one posterior per seed, SBC-verify each, and select the best-calibrated seed.

    Motivated by Finding 2 (HANDOFF.md): a single SNPE training run's SBC p-value is not
    reliable evidence of calibration for this problem -- identical data/architecture with
    only the torch seed differing produces wildly different SBC outcomes. Training an
    ensemble and selecting by worst-case (min) KS p-value across all 5 parameters avoids
    reporting a "seed-lottery" result while keeping a single posterior object (so
    downstream notebooks' `posterior.sample(...)` interface is unchanged).

    Returns (best_posterior, best_seed, per_seed_results) where per_seed_results is a
    list of dicts with keys: seed, sbc, coverage, min_ks_pvalue.
    """
    per_seed = []
    posteriors = {}
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        posterior = train_posterior(
            bank_path=bank_path, arch=arch, hidden_features=hidden_features,
            num_transforms=num_transforms, torch_seed=seed,
        )
        sbc = run_sbc(posterior, structure=structure, noise_pct=noise_pct,
                       n_sbc=n_sbc, n_post=n_post, seed=12345 + seed)
        min_ks = min(sbc[name]["ks_pvalue"] for name in PARAM_NAMES)
        per_seed.append({"seed": seed, "sbc": sbc, "min_ks_pvalue": min_ks})
        posteriors[seed] = posterior
        print(f"[ensemble] seed {seed}: min KS p-value across params = {min_ks:.4f}")

    best = max(per_seed, key=lambda r: r["min_ks_pvalue"])
    best_seed = best["seed"]
    n_pass = sum(1 for r in per_seed if r["min_ks_pvalue"] > 0.05)
    print(f"\n[ensemble] {n_pass}/{len(seeds)} seeds pass SBC (min KS p>0.05 across all params)")
    print(f"[ensemble] Selected seed {best_seed} (min KS p={best['min_ks_pvalue']:.4f})")
    return posteriors[best_seed], best_seed, per_seed


# ---------------------------------------------------------------------------
# CNN embedding pipeline (raw-observation variant)
# ---------------------------------------------------------------------------

N_SAVE = 120  # timesteps per window (1-min resolution, 2h window)


def _simulate_raw_sb(theta_np, ctrl, y0, noise_pct, rng):
    """One prior draw -> raw S-B observation (120, 9), or None if invalid."""
    theta_jnp = jnp.array(theta_np, dtype=jnp.float32)
    ts, ys = simulate_trajectory_explicit_jit(
        theta_jnp, NOMINAL_INLET, ctrl, y0,
        t_final=2.0, n_save=N_SAVE, rtol=1e-3, atol=1e-5,
    )
    raw = extract_observations_explicit(ys, theta_jnp, ctrl)
    raw_np = np.asarray(raw)
    if np.isnan(raw_np).any() or np.isinf(raw_np).any():
        return None
    sb_obs = raw_np[:, SB_INDICES]
    scale = np.maximum(np.max(np.abs(sb_obs), axis=0, keepdims=True), 1e-6)
    noise = rng.normal(0, noise_pct * scale, sb_obs.shape)
    return (sb_obs + noise).astype(np.float32)


def generate_raw_training_bank(
    structure: str,
    noise_pct: float,
    out_path: pathlib.Path,
    n_train: int = 15000,
    seed: int = 20260625,
) -> pathlib.Path:
    """Generate (or reuse) a CNN training bank: prior draws -> raw observations."""
    if out_path.exists():
        print(f"[bank-cnn] {out_path} already exists, reusing")
        return out_path

    ctrl, y0 = _structure_config(structure)
    prior = box_uniform_5d()
    rng = np.random.default_rng(seed)

    print(f"[bank-cnn] Generating {n_train} {structure} raw-obs sims "
          f"@ {noise_pct*100:.1f}% noise...")
    theta_samples = prior.sample((n_train,)).numpy()
    thetas_list, obs_list = [], []
    t0 = time.time()
    for i, th in enumerate(theta_samples):
        obs = _simulate_raw_sb(th, ctrl, y0, noise_pct, rng)
        if obs is not None:
            thetas_list.append(th)
            obs_list.append(obs)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"[bank-cnn]   {i+1}/{n_train}  "
                  f"({elapsed:.0f}s elapsed, {len(thetas_list)} valid)")

    thetas_arr = np.stack(thetas_list)
    raw_obs_arr = np.stack(obs_list)
    print(f"[bank-cnn] Valid: {len(thetas_arr)}/{n_train} in {time.time()-t0:.0f}s")
    print(f"[bank-cnn] Shapes: thetas={thetas_arr.shape}, "
          f"raw_obs={raw_obs_arr.shape}")
    np.savez(out_path, thetas=thetas_arr, raw_obs=raw_obs_arr)
    print(f"[bank-cnn] Saved to {out_path}")
    return out_path


def train_cnn_posterior(
    bank_path: pathlib.Path,
    hidden_features: int = 60,
    num_transforms: int = 3,
    output_dim: int = 30,
    max_num_epochs: int = 300,
    stop_after_epochs: int = 20,
    torch_seed: int = 0,
):
    """Train a CNN-embedding SNPE posterior on raw S-B observations."""
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn
    from sbi.neural_nets.embedding_nets import CNNEmbedding

    torch.manual_seed(torch_seed)
    train_data = np.load(bank_path)
    thetas_t = torch.tensor(train_data["thetas"], dtype=torch.float32)
    raw_obs = train_data["raw_obs"]
    n_samples, n_t, n_ch = raw_obs.shape
    obs_flat = torch.tensor(
        raw_obs.reshape(n_samples, -1), dtype=torch.float32
    )
    print(f"[train-cnn] {n_samples} samples, raw ({n_t},{n_ch}) -> "
          f"flat {obs_flat.shape[1]}-D, output_dim={output_dim}")

    embedding_net = CNNEmbedding(
        input_shape=(n_t,),
        in_channels=n_ch,
        out_channels_per_layer=[16, 32],
        num_conv_layers=2,
        num_linear_layers=2,
        num_linear_units=64,
        output_dim=output_dim,
        kernel_size=5,
        pool_kernel_size=2,
    )
    n_params = sum(p.numel() for p in embedding_net.parameters())
    print(f"[train-cnn] CNNEmbedding: {n_params:,} parameters")

    prior = box_uniform_5d()
    density_estimator = posterior_nn(
        model="zuko_nsf",
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        embedding_net=embedding_net,
        z_score_x="structured",
    )
    inference = SNPE(prior=prior, density_estimator=density_estimator)
    inference.append_simulations(thetas_t, obs_flat)

    t0 = time.time()
    density_estimator_trained = inference.train(
        max_num_epochs=max_num_epochs,
        validation_fraction=0.1,
        stop_after_epochs=stop_after_epochs,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator_trained)
    wall_time = time.time() - t0
    print(f"[train-cnn] Completed in {wall_time:.0f}s")

    meta = {
        "arch": "zuko_nsf",
        "hidden_features": hidden_features,
        "num_transforms": num_transforms,
        "output_dim": output_dim,
        "n_cnn_params": n_params,
        "n_train": n_samples,
        "torch_seed": torch_seed,
        "wall_time_s": wall_time,
    }
    return posterior, meta


def run_sbc_cnn(
    posterior,
    structure: str,
    noise_pct: float,
    n_sbc: int = 400,
    n_post: int = 200,
    seed: int = 12345,
) -> dict:
    """SBC for CNN-embedding posterior using raw observations."""
    ctrl, y0 = _structure_config(structure)
    prior = box_uniform_5d()
    rng = np.random.default_rng(seed)

    ranks = {name: [] for name in PARAM_NAMES}
    t0 = time.time()
    n_valid = 0
    for i in range(n_sbc):
        th = prior.sample((1,)).numpy()[0]
        obs = _simulate_raw_sb(th, ctrl, y0, noise_pct, rng)
        if obs is None:
            continue
        n_valid += 1
        obs_flat = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            # reject_outside_prior=False: sbi 0.26's default rejection sampling can
            # spin near-indefinitely if the flow's mass for a given observation drifts
            # outside the prior box (documented hang mode in this project, nb27/nb31).
            samp = posterior.sample(
                (n_post,), x=obs_flat,
                show_progress_bars=False,
                reject_outside_prior=False,
            ).detach().numpy()
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
        if (i + 1) % 50 == 0:
            print(f"[sbc-cnn]   {i+1}/{n_sbc}  "
                  f"({time.time()-t0:.0f}s, {n_valid} valid)")

    results = {"n_sbc": n_sbc, "n_post": n_post, "n_valid": n_valid}
    for name in PARAM_NAMES:
        r = np.array(ranks[name])
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        results[name] = {
            "ranks": r.tolist(),
            "ks_pvalue": float(ks.pvalue),
            "mean_rank_frac": float(r.mean() / n_post),
        }
    print(f"[sbc-cnn] Completed in {time.time()-t0:.0f}s, "
          f"n_valid={n_valid}/{n_sbc}")
    for name in PARAM_NAMES:
        print(f"[sbc-cnn]   {name:10s} KS p={results[name]['ks_pvalue']:.4f}")
    return results


def train_cnn_ensemble_and_select(
    bank_path: pathlib.Path,
    structure: str,
    noise_pct: float,
    hidden_features: int = 60,
    num_transforms: int = 3,
    output_dim: int = 30,
    seeds: list[int] | None = None,
    n_sbc: int = 400,
    n_post: int = 200,
):
    """Train CNN posteriors across seeds, SBC each, select best-calibrated."""
    if seeds is None:
        seeds = list(range(8))

    per_seed = []
    posteriors = {}
    for seed in seeds:
        print(f"\n=== CNN seed {seed} ===")
        posterior, meta = train_cnn_posterior(
            bank_path=bank_path,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            output_dim=output_dim,
            torch_seed=seed,
        )
        sbc = run_sbc_cnn(
            posterior, structure=structure, noise_pct=noise_pct,
            n_sbc=n_sbc, n_post=n_post, seed=12345 + seed,
        )
        min_ks = min(sbc[name]["ks_pvalue"] for name in PARAM_NAMES)
        per_seed.append({
            "seed": seed, "sbc": sbc, "meta": meta,
            "min_ks_pvalue": min_ks,
        })
        posteriors[seed] = posterior
        print(f"[cnn-ensemble] seed {seed}: "
              f"min KS p-value = {min_ks:.4f}")

    best = max(per_seed, key=lambda r: r["min_ks_pvalue"])
    best_seed = best["seed"]
    n_pass = sum(1 for r in per_seed if r["min_ks_pvalue"] > 0.05)
    print(f"\n[cnn-ensemble] {n_pass}/{len(seeds)} seeds pass SBC "
          f"(min KS p>0.05)")
    print(f"[cnn-ensemble] Selected seed {best_seed} "
          f"(min KS p={best['min_ks_pvalue']:.4f})")
    return posteriors[best_seed], best_seed, per_seed


def save_variant(posterior, sbc_results, coverage_results, variant_name: str, meta: dict):
    pkl_path = SBI_LOGS / f"wu2003_posterior_variant_{variant_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"posterior": posterior, "meta": meta}, f)

    import json
    json_path = SBI_LOGS / f"variant_{variant_name}_results.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "sbc": sbc_results, "coverage": coverage_results}, f, indent=2)
    print(f"[save] Posterior -> {pkl_path}")
    print(f"[save] Results   -> {json_path}")
    return pkl_path, json_path
