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
    NOMINAL_CTRL_SA,
    NOMINAL_CTRL_SB,
    NOMINAL_INLET,
    NOMINAL_THETA,
    NOMINAL_Y0_EXPLICIT,
    QJ_NOM,
    T_SP,
    extract_observations_explicit,
    simulate_to_steady_state_explicit,
    simulate_trajectory_explicit,
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
    "R_norm",
    "V_norm",
]
RAW_INDEX = {name: i for i, name in enumerate(RAW_CHANNELS)}

SA_CHANNELS = [
    "T_r", "T_j", "Q_j", "x_D", "T_reb", "Q_reb",
    "F_R_norm", "F_B_norm", "R_norm", "V_norm",
]
SB_CHANNELS = [
    "T_r", "T_j", "Q_j", "T_reb", "Q_reb",
    "F_R_norm", "F_B_norm", "R_norm", "V_norm",
]
SA_INDICES = [RAW_INDEX[name] for name in SA_CHANNELS]
SB_INDICES = [RAW_INDEX[name] for name in SB_CHANNELS]

DEFAULT_N_REPLICATES = 30
DEFAULT_T_FINAL_H = 2.0
DEFAULT_N_SAVE = 120
DEFAULT_SENSOR_NOISE_PCT = 0.003


def _nominal_ctrl(structure: str) -> jnp.ndarray:
    if structure == "S-A":
        return NOMINAL_CTRL_SA
    if structure == "S-B":
        return NOMINAL_CTRL_SB
    raise ValueError(f"Unknown structure: {structure}")


def open_loop_ctrl(structure: str) -> jnp.ndarray:
    """Return fixed reactor/column actuator settings for open-loop diagnostics."""
    ctrl = _nominal_ctrl(structure)
    ctrl = ctrl.at[0].set(0.0).at[4].set(QJ_NOM).at[5].set(QJ_NOM)
    ctrl = ctrl.at[7].set(0.0).at[10].set(0.0).at[17].set(0.0).at[20].set(0.0)
    ctrl = ctrl.at[14].set(ctrl[13]).at[15].set(ctrl[13])
    ctrl = ctrl.at[24].set(ctrl[23]).at[25].set(ctrl[23])
    return ctrl


def scenario_ctrl(scenario: RecycleScenarioConfig, structure: str) -> jnp.ndarray:
    """Controller vector for a scenario's declared mode."""
    return open_loop_ctrl(structure) if scenario.mode == "open_loop" else _nominal_ctrl(structure)


def nominal_warm_start(structure: str = "S-B") -> jnp.ndarray:
    """Nominal closed-loop steady state used as a structure-specific warm start."""
    return simulate_to_steady_state_explicit(
        NOMINAL_THETA,
        NOMINAL_INLET,
        _nominal_ctrl(structure),
        NOMINAL_Y0_EXPLICIT,
        t_final=200.0,
    )


def deterministic_window(
    scenario: RecycleScenarioConfig,
    *,
    structure: str = "S-B",
    y0: jnp.ndarray | None = None,
    t_final_h: float = DEFAULT_T_FINAL_H,
    n_save: int = DEFAULT_N_SAVE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic raw observations for one scenario.

    Returns ``(t_h, raw_obs)`` where ``raw_obs`` has columns ``RAW_CHANNELS``.
    """
    if y0 is None:
        y0 = nominal_warm_start(structure)
    ctrl = scenario_ctrl(scenario, structure)
    ts, ys = simulate_trajectory_explicit(
        scenario.theta(),
        NOMINAL_INLET,
        ctrl,
        y0,
        t_final=t_final_h,
        n_save=n_save,
    )
    raw_obs = extract_observations_explicit(ys, scenario.theta(), ctrl)
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
    y0_sa = nominal_warm_start("S-A")
    y0_sb = nominal_warm_start("S-B")
    rng = np.random.default_rng(seed)

    sa_windows = []
    sb_windows = []
    labels = []
    deterministic_sa = {}
    deterministic_sb = {}
    t_grid = None

    for scenario in configs:
        t_h, raw_sa = deterministic_window(
            scenario,
            structure="S-A",
            y0=y0_sa,
            t_final_h=t_final_h,
            n_save=n_save,
        )
        _, raw_sb = deterministic_window(
            scenario,
            structure="S-B",
            y0=y0_sb,
            t_final_h=t_final_h,
            n_save=n_save,
        )
        if t_grid is None:
            t_grid = t_h
        deterministic_sa[scenario.name] = raw_sa
        deterministic_sb[scenario.name] = raw_sb
        sa_raw_reps = noisy_replicates(raw_sa, n_replicates=n_replicates, rng=rng, noise_pct=noise_pct)
        sb_raw_reps = noisy_replicates(raw_sb, n_replicates=n_replicates, rng=rng, noise_pct=noise_pct)
        sa_reps = structure_view(sa_raw_reps, "S-A")
        sb_reps = structure_view(sb_raw_reps, "S-B")
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
        "deterministic_raw": deterministic_sb,
        "deterministic_raw_sa": deterministic_sa,
        "deterministic_raw_sb": deterministic_sb,
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
    raw_sa = dataset.get("deterministic_raw_sa", dataset["deterministic_raw"])
    raw_sb = dataset.get("deterministic_raw_sb", dataset["deterministic_raw"])

    def get(raw: dict[str, np.ndarray], name: str, channel: str) -> np.ndarray:
        return raw[name][:, RAW_INDEX[channel]]

    rows = []
    if "W2_cat_decay" in raw_sb:
        fr = get(raw_sb, "W2_cat_decay", "F_R_norm")
        fr_change = 100.0 * (fr[-1] / fr[0] - 1.0)
        rows.append({
            "effect": "snowball onset",
            "scenario": "W2_cat_decay/S-B",
            "metric": "F_R change [%]",
            "value": fr_change,
            "passes": fr_change > 3.0,
        })
    if "W11_snowball" in raw_sb:
        fr = get(raw_sb, "W11_snowball", "F_R_norm")
        fr_change = 100.0 * (fr[-1] / fr[0] - 1.0)
        v = get(raw_sb, "W11_snowball", "V_norm")
        v_change = 100.0 * (v[-1] / v[0] - 1.0)
        rows.append({
            "effect": "compound loop response",
            "scenario": "W11_snowball/S-B",
            "metric": "max(|F_R change|, |V_norm change|) [%]",
            "value": max(abs(fr_change), abs(v_change)),
            "passes": max(abs(fr_change), abs(v_change)) > 10.0,
        })
    if "W3_rxr_fouling" in raw_sb:
        t_error = np.max(np.abs(get(raw_sb, "W3_rxr_fouling", "T_r") - T_SP))
        qj = get(raw_sb, "W3_rxr_fouling", "Q_j")
        qj_change = qj[-1] / qj[0] - 1.0
        rows.extend([
            {
                "effect": "masking",
                "scenario": "W3_rxr_fouling/S-B",
                "metric": "max |T_r - T_sp| [K]",
                "value": t_error,
                "passes": t_error < 2.0,
            },
            {
                "effect": "compensation",
                "scenario": "W3_rxr_fouling/S-B",
                "metric": "Q_j change [%]",
                "value": 100.0 * qj_change,
                "passes": abs(qj_change) > 0.05,
            },
        ])
    if "W4_col_tray_eff" in raw_sa and "W1_healthy" in raw_sa:
        trebf = get(raw_sa, "W4_col_tray_eff", "T_reb")[-1] - get(raw_sa, "W1_healthy", "T_reb")[-1]
        r_shift = get(raw_sa, "W4_col_tray_eff", "R_norm")[-1] - get(raw_sa, "W1_healthy", "R_norm")[-1]
        v_shift = get(raw_sa, "W4_col_tray_eff", "V_norm")[-1] - get(raw_sa, "W1_healthy", "V_norm")[-1]
        rows.append({
            "effect": "column degradation",
            "scenario": "W4_col_tray_eff/S-A",
            "metric": "max(|R_norm shift|, |V_norm shift|)",
            "value": max(abs(r_shift), abs(v_shift)),
            "passes": max(abs(r_shift), abs(v_shift)) > 0.02 or abs(trebf) > 1.0,
        })
    if "W4_col_tray_eff" in raw_sa and "W4_col_tray_eff" in raw_sb:
        xD_delta = np.max(np.abs(get(raw_sa, "W4_col_tray_eff", "x_D") - get(raw_sb, "W4_col_tray_eff", "x_D")))
        rows.append({
            "effect": "control-structure split",
            "scenario": "W4_col_tray_eff",
            "metric": "max |x_D(S-A) - x_D(S-B)|",
            "value": xD_delta,
            "passes": xD_delta > 1e-3,
        })
    if "W3_rxr_fouling_ol" in raw_sb and "W3_rxr_fouling" in raw_sb:
        cl = np.max(np.abs(get(raw_sb, "W3_rxr_fouling", "T_r") - T_SP))
        ol = np.max(np.abs(get(raw_sb, "W3_rxr_fouling_ol", "T_r") - T_SP))
        rows.append({
            "effect": "open-loop contrast",
            "scenario": "W3_rxr_fouling_ol/S-B",
            "metric": "OL/CL max T excursion ratio",
            "value": ol / max(cl, 1e-12),
            "passes": ol > cl,
        })

    return pd.DataFrame(rows)
