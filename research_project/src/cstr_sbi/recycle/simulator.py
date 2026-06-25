"""Dataset generation utilities for the Wu 2003 recycle plant.

The simulator uses deterministic diffrax trajectories from ``recycle.physics``
and adds a Gaussian sensor layer to create replicate observation windows. This
keeps nb22 fast and reproducible while preserving the closed-loop compensation,
recycle snowball, and partial-observability effects needed for the SBI study.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import numpy as np
import pandas as pd

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL,
    NOMINAL_INLET,
    NOMINAL_THETA,
    NOMINAL_Y0,
    QJ_NOM,
    T_SP,
    extract_observations,
    simulate_to_steady_state,
    simulate_trajectory,
)
from cstr_sbi.recycle.scenarios import RecycleScenarioConfig, list_configs


RAW_CHANNELS = [
    "z_A",
    "T_r",
    "T_j",
    "Q_j",
    "x_D",
    "x_B",
    "F_R_norm",
    "T_reb",
    "Q_reb",
    "F_B_norm",
]
RAW_INDEX = {name: i for i, name in enumerate(RAW_CHANNELS)}

SA_CHANNELS = ["T_r", "T_j", "Q_j", "x_D", "T_reb", "Q_reb", "F_R_norm", "F_B_norm"]
SB_CHANNELS = ["T_r", "T_j", "Q_j", "T_reb", "Q_reb", "F_R_norm", "F_B_norm"]
SA_INDICES = [RAW_INDEX[name] for name in SA_CHANNELS]
SB_INDICES = [RAW_INDEX[name] for name in SB_CHANNELS]

DEFAULT_N_REPLICATES = 30
DEFAULT_T_FINAL_H = 2.0
DEFAULT_N_SAVE = 120
DEFAULT_SENSOR_NOISE_PCT = 0.003


def open_loop_ctrl() -> jnp.ndarray:
    """Return a fixed-cooling open-loop proxy controller vector."""
    return NOMINAL_CTRL.at[0].set(0.0).at[4].set(QJ_NOM).at[5].set(QJ_NOM)


CTRL_OPEN_PROXY = open_loop_ctrl()


def scenario_ctrl(scenario: RecycleScenarioConfig) -> jnp.ndarray:
    """Controller vector for a scenario's declared mode."""
    return CTRL_OPEN_PROXY if scenario.mode == "open_loop" else NOMINAL_CTRL


def nominal_warm_start() -> jnp.ndarray:
    """Nominal closed-loop steady state used as the common window initial state."""
    return simulate_to_steady_state(
        NOMINAL_THETA,
        NOMINAL_INLET,
        NOMINAL_CTRL,
        NOMINAL_Y0,
        t_final=200.0,
    )


def deterministic_window(
    scenario: RecycleScenarioConfig,
    *,
    y0: jnp.ndarray | None = None,
    t_final_h: float = DEFAULT_T_FINAL_H,
    n_save: int = DEFAULT_N_SAVE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic raw observations for one scenario.

    Returns ``(t_h, raw_obs)`` where ``raw_obs`` has columns ``RAW_CHANNELS``.
    """
    if y0 is None:
        y0 = nominal_warm_start()
    ctrl = scenario_ctrl(scenario)
    ts, ys = simulate_trajectory(
        scenario.theta(),
        NOMINAL_INLET,
        ctrl,
        y0,
        t_final=t_final_h,
        n_save=n_save,
    )
    raw_obs = extract_observations(ys, scenario.theta(), ctrl)
    return np.asarray(ts), np.asarray(raw_obs)


def _sensor_noise_sigma(obs: np.ndarray, noise_pct: float) -> np.ndarray:
    scale = np.maximum(np.max(np.abs(obs), axis=0, keepdims=True), 1e-12)
    return noise_pct * scale


def noisy_replicates(
    base_obs: np.ndarray,
    *,
    n_replicates: int,
    rng: np.random.Generator,
    noise_pct: float = DEFAULT_SENSOR_NOISE_PCT,
) -> np.ndarray:
    """Generate Gaussian sensor-noise replicates around one base trajectory."""
    sigma = _sensor_noise_sigma(base_obs, noise_pct)
    noise = rng.normal(loc=0.0, scale=sigma, size=(n_replicates, *base_obs.shape))
    return base_obs[None, :, :] + noise


def structure_view(raw_obs: np.ndarray, structure: str) -> np.ndarray:
    """Project raw observations to the S-A or S-B channel set."""
    if structure == "S-A":
        return raw_obs[..., SA_INDICES]
    if structure == "S-B":
        return raw_obs[..., SB_INDICES]
    raise ValueError(f"Unknown structure: {structure}")


def scenario_rows(configs: Iterable[RecycleScenarioConfig]) -> pd.DataFrame:
    """Return a flat scenario truth table."""
    rows = []
    for scenario in configs:
        row = asdict(scenario)
        row["theta"] = np.asarray(scenario.theta(), dtype=float).tolist()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def generate_dataset(
    *,
    n_replicates: int = DEFAULT_N_REPLICATES,
    t_final_h: float = DEFAULT_T_FINAL_H,
    n_save: int = DEFAULT_N_SAVE,
    noise_pct: float = DEFAULT_SENSOR_NOISE_PCT,
    seed: int = 20260625,
    configs: list[RecycleScenarioConfig] | None = None,
) -> dict[str, object]:
    """Generate the planned WU2003 S-A/S-B observation dataset.

    Returns a dictionary containing S-A and S-B arrays with shapes
    ``(n_scenarios * n_replicates, n_t, n_channels)`` plus labels and metadata.
    """
    configs = list_configs() if configs is None else sorted(configs, key=lambda s: s.id)
    y0 = nominal_warm_start()
    rng = np.random.default_rng(seed)

    sa_windows = []
    sb_windows = []
    labels = []
    deterministic = {}
    t_grid = None

    for scenario in configs:
        t_h, raw = deterministic_window(
            scenario,
            y0=y0,
            t_final_h=t_final_h,
            n_save=n_save,
        )
        if t_grid is None:
            t_grid = t_h
        deterministic[scenario.name] = raw
        raw_reps = noisy_replicates(raw, n_replicates=n_replicates, rng=rng, noise_pct=noise_pct)
        sa_reps = structure_view(raw_reps, "S-A")
        sb_reps = structure_view(raw_reps, "S-B")
        sa_windows.append(sa_reps)
        sb_windows.append(sb_reps)
        for replicate in range(n_replicates):
            labels.append(
                {
                    "scenario_id": scenario.id,
                    "scenario_name": scenario.name,
                    "mode": scenario.mode,
                    "replicate": replicate,
                    "alpha": scenario.alpha,
                    "beta_r": scenario.beta_r,
                    "eta_col": scenario.eta_col,
                    "xi_reb": scenario.xi_reb,
                    "z_A0_eff": scenario.z_A0_eff,
                }
            )

    labels_df = pd.DataFrame(labels)
    return {
        "t_h": np.asarray(t_grid),
        "observations_sa": np.concatenate(sa_windows, axis=0),
        "observations_sb": np.concatenate(sb_windows, axis=0),
        "labels": labels_df,
        "scenario_table": scenario_rows(configs),
        "deterministic_raw": deterministic,
        "raw_channels": RAW_CHANNELS,
        "sa_channels": SA_CHANNELS,
        "sb_channels": SB_CHANNELS,
        "n_replicates": n_replicates,
        "noise_pct": noise_pct,
        "seed": seed,
    }


def save_dataset(dataset: dict[str, object], path: str | Path) -> Path:
    """Persist a generated dataset to ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = dataset["labels"]
    scenario_table = dataset["scenario_table"]
    np.savez_compressed(
        path,
        t_h=dataset["t_h"],
        observations_sa=dataset["observations_sa"],
        observations_sb=dataset["observations_sb"],
        labels=labels.to_records(index=False),
        scenario_table=scenario_table.to_records(index=False),
        raw_channels=np.asarray(dataset["raw_channels"], dtype=object),
        sa_channels=np.asarray(dataset["sa_channels"], dtype=object),
        sb_channels=np.asarray(dataset["sb_channels"], dtype=object),
        n_replicates=np.asarray(dataset["n_replicates"]),
        noise_pct=np.asarray(dataset["noise_pct"]),
        seed=np.asarray(dataset["seed"]),
    )
    return path


def physical_effect_metrics(dataset: dict[str, object]) -> pd.DataFrame:
    """Compute deterministic physical-effect checks from raw scenario windows."""
    raw = dataset["deterministic_raw"]

    def get(name: str, channel: str) -> np.ndarray:
        return raw[name][:, RAW_INDEX[channel]]

    rows = []
    if "W2_cat_decay" in raw:
        fr = get("W2_cat_decay", "F_R_norm")
        fr_change = 100.0 * (fr[-1] / fr[0] - 1.0)
        rows.append({
            "effect": "snowball onset",
            "scenario": "W2_cat_decay",
            "metric": "F_R change [%]",
            "value": fr_change,
            "passes": fr_change > 3.0,
        })
    if "W11_snowball" in raw:
        fr = get("W11_snowball", "F_R_norm")
        fr_change = 100.0 * (fr[-1] / fr[0] - 1.0)
        rows.append({
            "effect": "compound snowball",
            "scenario": "W11_snowball",
            "metric": "F_R change [%]",
            "value": fr_change,
            "passes": fr_change > 3.0,
        })
    if "W3_rxr_fouling" in raw:
        t_error = np.max(np.abs(get("W3_rxr_fouling", "T_r") - T_SP))
        qj_change = get("W3_rxr_fouling", "Q_j")[-1] / get("W3_rxr_fouling", "Q_j")[0] - 1.0
        rows.extend([
            {
                "effect": "masking",
                "scenario": "W3_rxr_fouling",
                "metric": "max |T_r - T_sp| [K]",
                "value": t_error,
                "passes": t_error < 2.0,
            },
            {
                "effect": "compensation",
                "scenario": "W3_rxr_fouling",
                "metric": "Q_j change [%]",
                "value": 100.0 * qj_change,
                "passes": abs(qj_change) > 0.05,
            },
        ])
    if "W4_col_tray_eff" in raw and "W1_healthy" in raw:
        trebf = get("W4_col_tray_eff", "T_reb")[-1] - get("W1_healthy", "T_reb")[-1]
        rows.append({
            "effect": "column degradation",
            "scenario": "W4_col_tray_eff",
            "metric": "T_reb final shift [K] vs W1",
            "value": trebf,
            "passes": abs(trebf) > 1.0,
        })
    if "W3_rxr_fouling_ol" in raw and "W3_rxr_fouling" in raw:
        cl = np.max(np.abs(get("W3_rxr_fouling", "T_r") - T_SP))
        ol = np.max(np.abs(get("W3_rxr_fouling_ol", "T_r") - T_SP))
        rows.append({
            "effect": "open-loop contrast",
            "scenario": "W3_rxr_fouling_ol",
            "metric": "OL/CL max T excursion ratio",
            "value": ol / max(cl, 1e-12),
            "passes": ol > cl,
        })

    return pd.DataFrame(rows)
