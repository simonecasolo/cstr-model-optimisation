"""Build and execute notebook 22: Wu 2003 data generation."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    """Markdown cell with explicit language metadata."""
    return new_markdown_cell(source, metadata={"language": "markdown"})


def code(source: str):
    """Python cell with explicit language metadata."""
    return new_code_cell(source, metadata={"language": "python"})


CELLS = [
    md(
        """# Notebook 22 -- Wu 2003 Data Generation

This notebook is the nb22 checkpoint from
[`../project_wu2003_sbi.md`](../project_wu2003_sbi.md): generate the WU2003
observation windows that downstream summary-statistics and SBI notebooks will
consume.

Plan contract:

- `23` scenarios = `16` closed-loop + `7` open-loop diagnostic counterparts
- `30` replicates per scenario
- `2` explicit control structures: S-A and S-B
- `2 h` observation window at `1 min` resolution = `120` time points
- S-A: `10` channels, with `x_D`, reflux effort, and boilup effort
- S-B: `9` channels, without `x_D`, with conventional controller effort

The generated files are:

- [`../data/wu2003_observations.npz`](../data/wu2003_observations.npz)
- [`../data/wu2003_scenario_configs.csv`](../data/wu2003_scenario_configs.csv)
"""
    ),
    md("""## 1. Imports and generator settings"""),
    code(
        """from pathlib import Path
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cstr_sbi.recycle.simulator import (
    DEFAULT_N_REPLICATES,
    DEFAULT_N_SAVE,
    DEFAULT_SENSOR_NOISE_PCT,
    DEFAULT_T_FINAL_H,
    RAW_CHANNELS,
    SA_CHANNELS,
    SB_CHANNELS,
    generate_dataset,
    physical_effect_metrics,
    save_dataset,
)
from cstr_sbi.recycle.scenarios import list_closed_loop_configs, list_open_loop_configs

print("JAX devices:", jax.devices())
print(f"Closed-loop scenarios: {len(list_closed_loop_configs())}")
print(f"Open-loop diagnostic counterparts: {len(list_open_loop_configs())}")
print(f"Replicates per scenario: {DEFAULT_N_REPLICATES}")
print(f"Window: {DEFAULT_T_FINAL_H} h, time points: {DEFAULT_N_SAVE}")
print(f"Sensor noise fraction: {DEFAULT_SENSOR_NOISE_PCT}")
print("Raw channels:", RAW_CHANNELS)
print("S-A channels:", SA_CHANNELS)
print("S-B channels:", SB_CHANNELS)
"""
    ),
    md(
        """## 2. Generate the dataset

The generator integrates one deterministic S-A trajectory and one deterministic
S-B trajectory per scenario, then adds independent Gaussian sensor noise for each
replicate. This keeps the stochastic layer focused on measurement-window
variation while making the two structures genuine closed-loop simulations.
"""
    ),
    code(
        """N_REPLICATES = 30
N_SAVE = 120
T_FINAL_H = 2.0
SEED = 20260625
NOISE_PCT = 0.003

start = time.perf_counter()
dataset = generate_dataset(
    n_replicates=N_REPLICATES,
    t_final_h=T_FINAL_H,
    n_save=N_SAVE,
    noise_pct=NOISE_PCT,
    seed=SEED,
)
elapsed = time.perf_counter() - start

obs_sa = dataset["observations_sa"]
obs_sb = dataset["observations_sb"]
labels = dataset["labels"]
scenario_table = dataset["scenario_table"]

print(f"Generation time: {elapsed:.2f} s")
print(f"S-A observations: {obs_sa.shape}")
print(f"S-B observations: {obs_sb.shape}")
print(f"Labels: {labels.shape}")
print(f"Scenario table: {scenario_table.shape}")
"""
    ),
    md("""## 3. Scenario table and replicate counts"""),
    code(
        """display(scenario_table)

count_table = (
    labels.groupby(["mode", "scenario_name"])
    .size()
    .rename("n_windows_per_structure")
    .reset_index()
)
display(count_table)
"""
    ),
    md(
        """## 4. Persist outputs

The `.npz` stores both S-A and S-B arrays plus labels/channel metadata. The CSV
is a human-readable scenario truth table.
"""
    ),
    code(
        """data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
npz_path = save_dataset(dataset, data_dir / "wu2003_observations.npz")
csv_path = data_dir / "wu2003_scenario_configs.csv"
scenario_table.to_csv(csv_path, index=False)

print(f"Wrote {npz_path}")
print(f"Wrote {csv_path}")
print(f"NPZ size: {npz_path.stat().st_size / 1024**2:.2f} MiB")
"""
    ),
    md(
        """## 5. Basic data-quality checks

These checks make sure the dataset is shaped as planned and does not contain
NaNs or infinities. They are deliberately mechanical and should pass before any
physical interpretation is trusted.
"""
    ),
    code(
        """expected_scenarios = 23
expected_windows_per_structure = expected_scenarios * N_REPLICATES
expected_total_windows = expected_windows_per_structure * 2

nan_rate_sa = 1.0 - np.isfinite(obs_sa).mean()
nan_rate_sb = 1.0 - np.isfinite(obs_sb).mean()

shape_checks = pd.DataFrame([
    {
        "check": "scenario count",
        "expected": expected_scenarios,
        "observed": len(scenario_table),
        "passes": len(scenario_table) == expected_scenarios,
    },
    {
        "check": "S-A window shape",
        "expected": (expected_windows_per_structure, N_SAVE, len(SA_CHANNELS)),
        "observed": obs_sa.shape,
        "passes": obs_sa.shape == (expected_windows_per_structure, N_SAVE, len(SA_CHANNELS)),
    },
    {
        "check": "S-B window shape",
        "expected": (expected_windows_per_structure, N_SAVE, len(SB_CHANNELS)),
        "observed": obs_sb.shape,
        "passes": obs_sb.shape == (expected_windows_per_structure, N_SAVE, len(SB_CHANNELS)),
    },
    {
        "check": "total windows",
        "expected": expected_total_windows,
        "observed": obs_sa.shape[0] + obs_sb.shape[0],
        "passes": obs_sa.shape[0] + obs_sb.shape[0] == expected_total_windows,
    },
    {
        "check": "S-A finite values",
        "expected": "nan/inf rate = 0",
        "observed": nan_rate_sa,
        "passes": nan_rate_sa == 0.0,
    },
    {
        "check": "S-B finite values",
        "expected": "nan/inf rate = 0",
        "observed": nan_rate_sb,
        "passes": nan_rate_sb == 0.0,
    },
    {
        "check": "S-A includes x_D",
        "expected": True,
        "observed": "x_D" in SA_CHANNELS,
        "passes": "x_D" in SA_CHANNELS,
    },
    {
        "check": "S-B excludes x_D",
        "expected": True,
        "observed": "x_D" not in SB_CHANNELS,
        "passes": "x_D" not in SB_CHANNELS,
    },
])

shape_checks
"""
    ),
    md(
        """## 6. Physical-effect checks

These checks verify that the generated deterministic scenario windows contain
the effects the paper needs:

- **Snowball effect:** catalyst decay increases recycle flow.
- **Compound loop response:** combined reactor/column degradation moves the
    explicit boilup/reboiler compensation channel.
- **Masking:** jacket fouling is mostly hidden in `T_r` under feedback.
- **Compensation:** the controller output `Q_j` moves when jacket fouling occurs.
- **Column degradation:** S-A reflux/boilup compensation moves under tray-efficiency loss.
- **Open-loop contrast:** open-loop diagnostic windows expose larger temperature
  excursions than closed-loop windows.
"""
    ),
    code(
        """effect_checks = physical_effect_metrics(dataset)
effect_checks
"""
    ),
    md("""## 7. Visual inspection: key effects"""),
    code(
        """raw_sb = dataset["deterministic_raw_sb"]
raw_sa = dataset["deterministic_raw_sa"]
raw_idx = {name: i for i, name in enumerate(RAW_CHANNELS)}

def series(raw, scenario, channel):
    return raw[scenario][:, raw_idx[channel]]

t = dataset["t_h"]
fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.5), constrained_layout=True)

axes[0, 0].plot(t, series(raw_sb, "W1_healthy", "F_R_norm"), label="W1 S-B")
axes[0, 0].plot(t, series(raw_sb, "W2_cat_decay", "F_R_norm"), label="W2 S-B catalyst decay")
axes[0, 0].set_ylabel("F_R/F_R_nom [-]")
axes[0, 0].set_title("Snowball: recycle buildup")

axes[0, 1].plot(t, series(raw_sb, "W3_rxr_fouling", "T_r") - series(raw_sb, "W1_healthy", "T_r")[0], label="W3 S-B closed-loop")
axes[0, 1].plot(t, series(raw_sb, "W3_rxr_fouling_ol", "T_r") - series(raw_sb, "W1_healthy", "T_r")[0], label="W3 S-B open-loop")
axes[0, 1].set_ylabel("T_r - nominal T_r [K]")
axes[0, 1].set_title("Masking: feedback suppresses T_r excursion")

axes[1, 0].plot(t, series(raw_sb, "W1_healthy", "Q_j") / series(raw_sb, "W1_healthy", "Q_j")[0], label="W1 S-B")
axes[1, 0].plot(t, series(raw_sb, "W3_rxr_fouling", "Q_j") / series(raw_sb, "W1_healthy", "Q_j")[0], label="W3 S-B fouling")
axes[1, 0].set_ylabel("Q_j/Q_j_nom [-]")
axes[1, 0].set_title("Compensation: controller output moves")

axes[1, 1].plot(t, series(raw_sa, "W1_healthy", "R_norm"), label="W1 S-A R")
axes[1, 1].plot(t, series(raw_sa, "W4_col_tray_eff", "R_norm"), label="W4 S-A R")
axes[1, 1].plot(t, series(raw_sa, "W4_col_tray_eff", "V_norm"), label="W4 S-A V")
axes[1, 1].set_ylabel("normalized controller effort [-]")
axes[1, 1].set_title("Column effect: explicit reflux/boilup compensation")

for ax in axes.ravel():
    ax.set_xlabel("time [h]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.show()
"""
    ),
    md("""## 8. Replicate noise sanity check"""),
    code(
        """# Pull W1 S-A replicate block and inspect channel standard deviations.
w1_mask = labels["scenario_name"].to_numpy() == "W1_healthy"
w1_sa = obs_sa[w1_mask]
noise_summary = pd.DataFrame({
    "channel": SA_CHANNELS,
    "mean_overall": w1_sa.mean(axis=(0, 1)),
    "std_overall": w1_sa.std(axis=(0, 1)),
    "mean_replicate_std": w1_sa.std(axis=1).mean(axis=0),
+})
noise_summary
""".replace("\n+})", "\n})")
    ),
    md("""## 9. Final nb22 acceptance check"""),
    code(
        """acceptance = pd.DataFrame([
    {
        "check": "mechanical data contract",
        "expected": "all shape/finite/channel checks pass",
        "observed": bool(shape_checks["passes"].all()),
        "status": "PASS" if bool(shape_checks["passes"].all()) else "FAIL",
    },
    {
        "check": "physical effects present",
        "expected": "snowball, masking, compensation, column, open-loop checks pass",
        "observed": bool(effect_checks["passes"].all()),
        "status": "PASS" if bool(effect_checks["passes"].all()) else "FAIL",
    },
    {
        "check": "saved npz exists",
        "expected": str(npz_path),
        "observed": npz_path.exists(),
        "status": "PASS" if npz_path.exists() else "FAIL",
    },
    {
        "check": "saved scenario csv exists",
        "expected": str(csv_path),
        "observed": csv_path.exists(),
        "status": "PASS" if csv_path.exists() else "FAIL",
    },
])
acceptance
"""
    ),
    md(
        """## 10. Interpretation

The generated nb22 dataset satisfies the current plan-level data contract:
`23` scenarios, `30` replicates, `2` explicit control structures, and the planned
channel asymmetry between S-A and S-B.

The important physical effects are visible in the deterministic windows before
sensor noise is added. That matters because downstream summary statistics and
SBI training should learn from the intended process signatures rather than from
accidental artefacts of the data generator.

The current implementation is the recommended minimal explicit-loop model: reactor
and jacket dynamics are integrated, the column remains QSS, and S-A/S-B differ by
their reflux/reboiler control policies as well as by measured channels. A full
dynamic tray model remains a possible later validation extension, not a prerequisite
for nb23 summary-statistics prototyping.
"""
    ),
]


def main() -> int:
    nb = new_notebook()
    nb.cells = CELLS
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
    )

    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / "22_wu2003_data_generation.ipynb"

    print(f"Executing notebook -> {nb_path}")
    client = NotebookClient(
        nb,
        kernel_name="python3",
        timeout=1800,
        resources={"metadata": {"path": str(repo_root)}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"Wrote {nb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
