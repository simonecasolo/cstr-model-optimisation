"""SBI training and inference for the Luyben recycle plant.

Mirrors cstr_sbi.inference for the propylene oxide system.
Provides:
    simulation_wrapper_sbi  -- bridges JAX simulator to sbi torch API
    train_sbi_posterior     -- SNPE_C with NSF, 8-D theta, 65-D summaries
    sample_posterior        -- draw samples from trained posterior
"""

from __future__ import annotations

import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from cstr_sbi.luyben.physics import (
    NOMINAL_CTRL_ALL,
    NOMINAL_INLET,
    NOMINAL_THETA,
    NOMINAL_Y0,
    PARAM_NAMES,
)
from cstr_sbi.luyben.simulator import simulate_em_window, warm_start_ic, DEFAULT_SENSOR_NOISE_PCT
from cstr_sbi.luyben.summaries import compute_summary_statistics


# ---------------------------------------------------------------------------
# sbi simulation wrapper
# ---------------------------------------------------------------------------

def simulation_wrapper_sbi(
    theta_torch,
    inlet: jnp.ndarray = None,
    ctrl: jnp.ndarray = None,
    y0: jnp.ndarray = None,
    t_window: float = 120.0,
    dt: float = 0.02,
    dt_out: float = 1.0,
    seed: int = 0,
):
    """Bridge between sbi's torch-tensor theta batch and the JAX Luyben simulator.

    Parameters
    ----------
    theta_torch
        ``torch.Tensor`` of shape ``(n_batch, 8)`` with rows
        [alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta].
    y0
        Warm-start IC ``(13,)``. Pass a pre-computed warm IC for efficiency.

    Returns
    -------
    ``torch.Tensor`` of shape ``(n_batch, 65)`` -- summary statistics.
    """
    import torch
    from cstr_sbi.luyben.simulator import apply_sensor_layer

    if inlet is None:
        inlet = NOMINAL_INLET
    if ctrl is None:
        ctrl = NOMINAL_CTRL_ALL
    if y0 is None:
        y0 = NOMINAL_Y0

    theta_np = theta_torch.detach().cpu().numpy()
    n_batch = theta_np.shape[0]

    summaries = []
    for i in range(n_batch):
        theta_i = jnp.array(theta_np[i], dtype=jnp.float32)
        proc_key, sens_key = jax.random.split(jax.random.PRNGKey(seed + i))
        _, _, obs = simulate_em_window(
            theta_i, inlet, ctrl, y0,
            key=proc_key, t_window=t_window, dt=dt, dt_out=dt_out,
        )
        t_out = jnp.arange(1, obs.shape[0] + 1) * dt_out
        obs_noisy = apply_sensor_layer(obs, key=sens_key, noise_pct=DEFAULT_SENSOR_NOISE_PCT)
        s = compute_summary_statistics(obs_noisy, t_out)
        summaries.append(np.asarray(s))

    return torch.tensor(np.stack(summaries), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sbi_posterior(
    prior,
    n_simulations: int = 10_000,
    *,
    density_estimator: str = "nsf",
    hidden_features: int = 192,
    num_transforms: int = 7,
    training_batch_size: int = 256,
    max_num_epochs: int = 300,
    save_to: str | None = None,
    seed: int = 0,
    inlet: jnp.ndarray = None,
    ctrl: jnp.ndarray = None,
    y0: jnp.ndarray = None,
):
    """Train SNPE_C with an NSF density estimator for the 8-D Luyben problem.

    Parameters
    ----------
    prior
        A ``sbi``-compatible prior from ``box_uniform_8d()``.
    n_simulations
        Number of prior draws for the training set. Use 10k-20k; scale up
        if SBC shows poor calibration for more than 2 of the 8 parameters.
    hidden_features, num_transforms
        NSF architecture. Larger than the propylene oxide system (8-D vs 2-D).
    save_to
        If given, pickle the trained posterior to this path.

    Returns
    -------
    posterior
        Trained sbi posterior with a ``.sample()`` method.
    metadata : dict
        Training metadata (n_simulations, timing, etc.).
    """
    try:
        import torch
        from sbi.inference import SNPE, simulate_for_sbi
        from sbi.neural_nets import posterior_nn
    except ImportError as e:
        raise ImportError("sbi and torch must be installed.") from e

    if inlet is None:
        inlet = NOMINAL_INLET
    if ctrl is None:
        ctrl = NOMINAL_CTRL_ALL
    if y0 is None:
        y0 = NOMINAL_Y0

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    density_estimator_build_fun = posterior_nn(
        model=density_estimator,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        z_score_x="independent",
    )
    inference_obj = SNPE(
        prior=prior,
        density_estimator=density_estimator_build_fun,
        show_progress_bars=False,
    )

    _counter = [int(rng.integers(0, 2**31))]

    def wrapper(theta: "torch.Tensor") -> "torch.Tensor":
        _counter[0] += 1
        return simulation_wrapper_sbi(theta, inlet=inlet, ctrl=ctrl, y0=y0,
                                      seed=_counter[0])

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
        "n_params": 8,
        "n_features": 65,
    }

    if save_to is not None:
        import pickle
        with open(save_to, "wb") as f:
            pickle.dump({"posterior": posterior, "metadata": metadata}, f)

    return posterior, metadata


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_posterior(
    posterior,
    obs_summary: np.ndarray,
    n_samples: int = 10_000,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Draw samples from a trained 8-D SBI posterior.

    Returns shape ``(n_samples, 8)`` array with columns
    [alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta].
    """
    import torch
    x_obs = torch.tensor(obs_summary, dtype=torch.float32)
    samples = posterior.sample(
        (n_samples,), x=x_obs, show_progress_bars=False,
        reject_outside_prior=False,
    )
    return samples.detach().cpu().numpy()
