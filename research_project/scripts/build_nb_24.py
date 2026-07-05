"""Build and execute notebook 24: Wu 2003 S-B SNPE training."""

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


def code_intro(source: str):
    """Markdown explanation inserted before every generated code cell."""
    compact = " ".join(source.strip().split())
    if "from pathlib import Path" in source and "torch.manual_seed" in source:
        text = (
            "This code imports the numerical, plotting, and SBI libraries, defines paths and priors, "
            "loads the nb23 S-B evaluation summaries, and reports the training configuration."
        )
    elif "def _safe_corr" in source and "def simulate_sb_summary" in source:
        text = (
            "This code defines the S-B summary-statistic pipeline used by nb24: robust correlations, slopes, "
            "66-D window summaries, sensor noise, and one prior-simulation-to-summary wrapper."
        )
    elif "def sample_prior_numpy" in source and "generate_training_bank" in source:
        text = (
            "This code creates or loads the continuous-prior S-B training bank for SNPE, rejecting invalid "
            "simulations and saving the accepted 5-D parameter draws with their 66-D summaries."
        )
    elif "show_features =" in source and "24_simulator_sanity.png" in source:
        text = (
            "This code performs the prior-predictive sanity check by comparing selected S-B summary features "
            "from prior simulations against the closed-loop scenario observations."
        )
    elif "def make_prior" in source and "train_snpe_from_bank" in source:
        text = (
            "This code defines the SBI helper functions: the 5-D BoxUniform prior, cached SNPE training from "
            "the simulation bank, posterior sampling for one observation, and scenario lookup."
        )
    elif "sensitivity_results = []" in source:
        text = (
            "This code runs or loads the MAF simulation-budget sensitivity study, samples W1 and W11 posteriors, "
            "and saves the alpha/eta_col sensitivity scatter plot."
        )
    elif "FINAL_PATH" in source and "density_estimator=\"nsf\"" in source:
        text = (
            "This code trains or loads the final NSF posterior on the full S-B simulation bank and writes the "
            "training metadata used by downstream diagnostics."
        )
    elif "posterior_cases = []" in source:
        text = (
            "This code samples the final posterior for W1 and W11 and plots the primary alpha and eta_col "
            "recovery marginals with 90% intervals."
        )
    elif "scenario_sample_cache = {}" in source:
        text = (
            "This code samples the final posterior for every closed-loop scenario and builds the all-scenario "
            "marginal posterior density grid."
        )
    elif "def classify_wu_samples" in source:
        text = (
            "This code converts posterior samples into coarse fault classes and plots the W1/W11 joint "
            "alpha-eta_col posterior with threshold guides."
        )
    elif "recovery_rows = []" in source:
        text = (
            "This code computes posterior mean recovery, absolute errors, interval coverage, and recovery "
            "scatter plots across all closed-loop scenarios."
        )
    elif "N_SBC =" in source and "sbc_ranks_rows" in source:
        text = (
            "This code performs simulation-based calibration on held-out prior-bank cases and plots rank "
            "histograms for all five inferred parameters."
        )
    elif "metrics_rows = []" in source and "acceptance = pd.DataFrame" in source:
        text = (
            "This code writes posterior metrics, scenario recovery, SBC ranks, metadata, and the nb24 "
            "acceptance table that checks shapes, finiteness, figures, and output files."
        )
    elif "NB24B_FACTORS_PATH" in source and "calibration_addendum" in source:
        text = (
            "This code appends the nb24b calibration repair evidence to nb24 without replacing the raw "
            "SNPE outputs, then saves a reporting table for raw versus calibrated eta/xi intervals."
        )
    else:
        text = f"This code computes the next nb24 analysis step starting with: `{compact[:120]}`."
    return md(f"**What This Code Computes**\n\n{text}")


def add_code_explanations(cells):
    explained = []
    for cell in cells:
        if cell.cell_type == "code":
            explained.append(code_intro(cell.source))
        explained.append(cell)
    return explained


CELLS = [
    md(
        """# Notebook 24 -- Wu 2003 S-B SBI Training

This notebook mirrors the original nb04 workflow for the Wu 2003 recycle plant's
S-B conventional measurement structure. It does not train on the hand-picked
scenario catalogue. Instead it generates a continuous-prior S-B simulation bank,
summarises each 2 h trajectory to the 66-D S-B feature vector, trains SNPE-C, and
then evaluates the trained posterior on the closed-loop scenario windows from
nb22/nb23.

Main outputs:

- `data/wu2003_sbi_train_sb.npz`: continuous-prior S-B SBI training bank.
- `results/wu2003_nb24_sb_sbi_posterior_final.pkl`: trained final NSF posterior.
- `results/wu2003_nb24_sb_sbi_training_metadata.json`: training metadata.
- `figures/24_*.png`: nb04-style prior predictive, sensitivity, recovery,
  marginal, joint-posterior, and SBC figures.
"""
    ),
    md("""## 1. Imports, constants, and scenario summaries"""),
    code(
        """from pathlib import Path
import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kstest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB,
    NOMINAL_INLET,
    extract_observations_explicit,
    simulate_trajectory_explicit,
)
from cstr_sbi.recycle.simulator import SB_INDICES, nominal_warm_start

pd.set_option("display.precision", 5)

torch.manual_seed(20260625)

DATA_DIR = Path("data")
RESULTS = Path("results")
FIGS = Path("figures")
DATA_DIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

SCENARIO_FEATURE_PATH = DATA_DIR / "wu2003_summary_features.npz"
TRAIN_BANK_PATH = DATA_DIR / "wu2003_sbi_train_sb.npz"
assert SCENARIO_FEATURE_PATH.exists(), "Run nb23 first to create data/wu2003_summary_features.npz"

PARAMETER_NAMES = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]
PRIOR_LOW = np.array([0.40, 0.40, 0.50, 0.40, 0.70], dtype=np.float32)
PRIOR_HIGH = np.array([1.20, 1.20, 1.00, 1.20, 0.95], dtype=np.float32)
N_SBI_SIMULATIONS = int(os.environ.get("WU2003_NB24_N_SBI_SIMULATIONS", "15000"))
SENSITIVITY_BUDGETS = sorted(set(
    min(n, N_SBI_SIMULATIONS) for n in [1_000, 5_000, N_SBI_SIMULATIONS]
))
FINAL_N_SIMULATIONS = N_SBI_SIMULATIONS
MAX_NUM_EPOCHS = int(os.environ.get("WU2003_NB24_MAX_EPOCHS", "200"))
TRAINING_BATCH_SIZE = 256
POSTERIOR_SAMPLES = int(os.environ.get("WU2003_NB24_POSTERIOR_SAMPLES", "5000"))
POSTERIOR_SAMPLES_BIG = int(os.environ.get("WU2003_NB24_POSTERIOR_SAMPLES_BIG", "10000"))
PROGRESS_INTERVAL = int(os.environ.get("WU2003_NB24_PROGRESS_INTERVAL", "100"))


def log_status(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[nb24 {stamp}] {message}", flush=True)

with np.load(SCENARIO_FEATURE_PATH, allow_pickle=True) as data:
    X_sb_all = data["X_sb"].astype(np.float32)
    features_sb = [str(x) for x in data["features_sb"]]
    labels_all = pd.DataFrame.from_records(data["labels"])

closed_mask = labels_all["mode"].eq("closed_loop").to_numpy()
X_eval = X_sb_all[closed_mask]
labels_eval = labels_all.loc[closed_mask].reset_index(drop=True)
theta_eval = labels_eval[PARAMETER_NAMES].to_numpy(dtype=np.float32)
scenario_id = labels_eval["scenario_id"].to_numpy()
scenario_name = labels_eval["scenario_name"].to_numpy()

log_status(f"S-B evaluation summaries: {X_eval.shape}")
log_status(f"Evaluation scenarios: {len(np.unique(scenario_id))}")
log_status(f"SBI target training simulations: {N_SBI_SIMULATIONS}")
log_status(f"Sensitivity budgets: {SENSITIVITY_BUDGETS}")
log_status(f"S-B feature dimension: {len(features_sb)}")
log_status(f"Posterior samples: {POSTERIOR_SAMPLES}; big samples: {POSTERIOR_SAMPLES_BIG}")
log_status(f"Progress interval: {PROGRESS_INTERVAL} accepted simulations")
log_status(f"torch: {torch.__version__}")
"""
    ),
    md("""## 2. Summary-statistic and simulator functions"""),
    code(
        """def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _slope(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t0 = t - t.mean()
    denom = np.sum(t0**2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(t0 * (y - y.mean())) / denom)


def summarize_windows(windows, channels, t):
    feature_names = []
    rows = []
    channel_index = {name: i for i, name in enumerate(channels)}
    final_start = int(np.floor(0.75 * len(t)))

    for window in windows:
        values = []
        for channel in channels:
            y = window[:, channel_index[channel]]
            values.extend([
                float(np.mean(y)),
                float(np.std(y)),
                _slope(t, y),
                float(np.min(y)),
                float(np.max(y)),
                float(np.mean(y[final_start:])),
            ])
        rows.append(values)

    for channel in channels:
        feature_names.extend([
            f"{channel}__mean",
            f"{channel}__std",
            f"{channel}__slope",
            f"{channel}__min",
            f"{channel}__max",
            f"{channel}__final25_mean",
        ])

    physics_names = [
        "UA_proxy_final",
        "recycle_ratio_final",
        "col_recovery_proxy_final",
        "reb_intensity_final",
        "reactor_conversion_proxy_final",
        "recycle_excess_final",
        "Tr_Tj_ratio_final",
        "Qj_slope",
        "corr_Qj_FR",
        "corr_Qreb_FR",
        "R_effort_final",
        "V_effort_final",
    ]

    physics_rows = []
    for window in windows:
        idx = channel_index
        tr = window[:, idx["T_r"]]
        tj = window[:, idx["T_j"]]
        qj = window[:, idx["Q_j"]]
        qreb = window[:, idx["Q_reb"]]
        fr = window[:, idx["F_R_norm"]]
        fb = window[:, idx["F_B_norm"]]
        r_effort = window[:, idx["R_norm"]]
        v_effort = window[:, idx["V_norm"]]

        physics_rows.append([
            qj[-1] / max(abs(tr[-1] - tj[-1]), 1e-6),
            fr[-1],
            fb[-1] / max(1.0 + fr[-1], 1e-6),
            qreb[-1] / max(fr[-1], 1e-6),
            fb[-1] * 0.0105 / 460.0,
            fr[-1] - 1.0,
            tr[-1] / max(tj[-1], 1e-6),
            _slope(t, qj),
            _safe_corr(qj, fr),
            _safe_corr(qreb, fr),
            r_effort[-1],
            v_effort[-1],
        ])

    return np.hstack([np.asarray(rows), np.asarray(physics_rows)]).astype(np.float32), feature_names + physics_names


SB_CHANNELS = [
    "T_r", "T_j", "Q_j", "T_reb", "Q_reb",
    "F_R_norm", "F_B_norm", "R_norm", "V_norm",
]


def noisy_sensor_layer(obs, rng, noise_pct=0.003):
    scale = np.maximum(np.max(np.abs(obs), axis=0, keepdims=True), 1e-12)
    return obs + rng.normal(0.0, noise_pct * scale, size=obs.shape)


def simulate_sb_summary(theta, y0, rng):
    ts, ys = simulate_trajectory_explicit(
        theta,
        NOMINAL_INLET,
        NOMINAL_CTRL_SB,
        y0,
        t_final=2.0,
        n_save=120,
    )
    raw = np.asarray(extract_observations_explicit(ys, theta, NOMINAL_CTRL_SB))
    sb_obs = noisy_sensor_layer(raw[..., SB_INDICES], rng)
    X, names = summarize_windows(sb_obs[None, :, :], SB_CHANNELS, np.asarray(ts))
    return X[0], names


assert X_eval.shape[1] == len(features_sb) == 66
"""
    ),
    md("""## 3. Generate or load the continuous-prior S-B SBI bank

This is the missing Wu analogue of nb04's `simulate_for_sbi` call. The scenario
bank remains the evaluation set; this cell creates the continuous 5-D prior bank
used to train SNPE-C.
"""),
    code(
        """def sample_prior_numpy(n, rng):
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * rng.random((n, len(PARAMETER_NAMES)), dtype=np.float32)


def generate_training_bank(n_simulations, path):
    log_status(f"Starting S-B continuous-prior bank generation: target={n_simulations:,}, path={path}")
    rng = np.random.default_rng(20260625)
    theta_train = np.empty((n_simulations, len(PARAMETER_NAMES)), dtype=np.float32)
    X_train = np.empty((n_simulations, len(features_sb)), dtype=np.float32)
    y0_sb = nominal_warm_start("S-B")
    t0 = time.perf_counter()
    feature_names_ref = None
    accepted = 0
    attempted = 0
    rejected = 0
    while accepted < n_simulations:
        theta_i = sample_prior_numpy(1, rng)[0]
        attempted += 1
        try:
            summary_i, feature_names_ref = simulate_sb_summary(theta_i, y0_sb, rng)
        except Exception:
            rejected += 1
            continue
        if not np.isfinite(summary_i).all():
            rejected += 1
            continue
        theta_train[accepted] = theta_i
        X_train[accepted] = summary_i
        accepted += 1
        if accepted % PROGRESS_INTERVAL == 0 or accepted == n_simulations:
            elapsed = time.perf_counter() - t0
            rate = accepted / max(elapsed, 1e-9)
            remaining = (n_simulations - accepted) / max(rate, 1e-9)
            reject_pct = 100.0 * rejected / max(attempted, 1)
            log_status(
                f"bank {accepted:>6}/{n_simulations} finite; attempted={attempted}; "
                f"rejected={rejected} ({reject_pct:.1f}%); elapsed={elapsed/60:.1f} min; "
                f"rate={rate:.2f}/s; eta={remaining/60:.1f} min"
            )
    np.savez_compressed(
        path,
        theta=theta_train,
        X=X_train,
        feature_names=np.asarray(feature_names_ref, dtype=object),
        parameter_names=np.asarray(PARAMETER_NAMES, dtype=object),
        prior_low=PRIOR_LOW,
        prior_high=PRIOR_HIGH,
        n_simulations=np.asarray(n_simulations),
        n_attempted=np.asarray(attempted),
        n_rejected=np.asarray(rejected),
        wall_time_s=np.asarray(time.perf_counter() - t0),
    )
    log_status(f"Saved S-B training bank: theta={theta_train.shape}, X={X_train.shape}, path={path}")
    return theta_train, X_train, feature_names_ref


if TRAIN_BANK_PATH.exists():
    log_status(f"Found cached bank at {TRAIN_BANK_PATH}; validating shape and finiteness")
    with np.load(TRAIN_BANK_PATH, allow_pickle=True) as bank_data:
        theta_train = bank_data["theta"].astype(np.float32)
        X_train = bank_data["X"].astype(np.float32)
        train_feature_names = [str(x) for x in bank_data["feature_names"]]
    if theta_train.shape[0] != N_SBI_SIMULATIONS or X_train.shape[1] != len(features_sb) or not np.isfinite(X_train).all():
        log_status(
            f"Cached bank unsuitable: theta={theta_train.shape}, X={X_train.shape}, "
            f"finite_X={np.isfinite(X_train).all()}; regenerating"
        )
        theta_train, X_train, train_feature_names = generate_training_bank(N_SBI_SIMULATIONS, TRAIN_BANK_PATH)
    else:
        theta_train = theta_train[:N_SBI_SIMULATIONS]
        X_train = X_train[:N_SBI_SIMULATIONS]
        log_status(f"Loaded cached S-B SBI bank: theta={theta_train.shape}, X={X_train.shape}")
else:
    log_status(f"No cached bank found at {TRAIN_BANK_PATH}; generating from prior")
    theta_train, X_train, train_feature_names = generate_training_bank(N_SBI_SIMULATIONS, TRAIN_BANK_PATH)

assert theta_train.shape == (N_SBI_SIMULATIONS, 5)
assert X_train.shape == (N_SBI_SIMULATIONS, 66)
assert np.isfinite(theta_train).all() and np.isfinite(X_train).all()
log_status(f"SBI bank ready: theta={theta_train.shape}, X={X_train.shape}")
"""
    ),
    md("""## 4. Prior-predictive sanity check"""),
    code(
        """show_features = [
    "recycle_ratio_final",
    "reb_intensity_final",
    "R_effort_final",
    "V_effort_final",
]
feature_index = {name: i for i, name in enumerate(features_sb)}
scenario_ids_sorted = sorted(np.unique(scenario_id))
scenario_colors = dict(zip(scenario_ids_sorted, plt.cm.tab20(np.linspace(0, 1, len(scenario_ids_sorted)))))

fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
for col, feature in enumerate(show_features):
    idx = feature_index[feature]
    sim_vals = X_train[:, idx]
    obs_vals_all = X_eval[:, idx]
    sim_lo, sim_hi = np.percentile(sim_vals, [2.5, 97.5])
    span = max(np.ptp(np.r_[sim_vals, obs_vals_all]), 1e-9)
    xs = np.linspace(min(sim_vals.min(), obs_vals_all.min()) - 0.05 * span, max(sim_vals.max(), obs_vals_all.max()) + 0.05 * span, 300)

    ax = axes[0, col]
    kde = gaussian_kde(sim_vals)
    density = kde(xs)
    ax.fill_between(xs, density, alpha=0.30, color="steelblue", label="prior bank KDE")
    for sid in scenario_ids_sorted:
        vals = X_eval[scenario_id == sid, idx]
        ax.plot(vals, np.full_like(vals, -0.025 * density.max()), "|", color=scenario_colors[sid], alpha=0.45, markersize=8)
    ax.axvspan(sim_lo, sim_hi, alpha=0.10, color="steelblue")
    ax.set_title(feature, fontsize=9)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)

    ax = axes[1, col]
    ax.hist(sim_vals, bins=45, alpha=0.45, color="steelblue", density=True, label="prior bank")
    for sid in scenario_ids_sorted:
        vals = X_eval[scenario_id == sid, idx]
        ax.axvline(vals.mean(), color=scenario_colors[sid], linewidth=1.1, linestyle="--", alpha=0.75)
    ax.axvline(sim_lo, color="steelblue", linewidth=1, linestyle=":", alpha=0.8)
    ax.axvline(sim_hi, color="steelblue", linewidth=1, linestyle=":", alpha=0.8)
    ax.set_title(f"{feature} scenario means", fontsize=8)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)

fig.suptitle(f"S-B prior-predictive sanity check: {N_SBI_SIMULATIONS:,} prior simulations vs closed-loop scenarios")
fig.savefig(FIGS / "24_simulator_sanity.png", dpi=130, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 5. SNPE training helpers"""),
    code(
        """def make_prior():
    return BoxUniform(
        low=torch.as_tensor(PRIOR_LOW, dtype=torch.float32),
        high=torch.as_tensor(PRIOR_HIGH, dtype=torch.float32),
    )


def train_snpe_from_bank(theta_bank, X_bank, *, n_simulations, density_estimator, seed, cache_path):
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        log_status(f"Loaded cached {density_estimator} posterior: {cache_path}")
        return cached["posterior"], cached["scaler"], cached["metadata"]

    log_status(
        f"Training {density_estimator} SNPE posterior: n={n_simulations:,}, "
        f"max_epochs={MAX_NUM_EPOCHS}, batch={TRAINING_BATCH_SIZE}, cache={cache_path}"
    )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(theta_bank), size=n_simulations, replace=False)
    scaler = StandardScaler().fit(X_bank[idx])
    theta_tensor = torch.as_tensor(theta_bank[idx], dtype=torch.float32)
    x_tensor = torch.as_tensor(scaler.transform(X_bank[idx]), dtype=torch.float32)
    log_status(f"Prepared tensors for {density_estimator}: theta={tuple(theta_tensor.shape)}, x={tuple(x_tensor.shape)}")

    density = posterior_nn(
        model=density_estimator,
        hidden_features=128,
        num_transforms=5,
        z_score_x="independent",
    )
    inference = SNPE(prior=make_prior(), density_estimator=density, show_progress_bars=True)
    t0 = time.perf_counter()
    log_status(f"Appending simulations for {density_estimator}")
    inference.append_simulations(theta_tensor, x_tensor)
    log_status(f"Starting neural-density training for {density_estimator}")
    estimator = inference.train(
        training_batch_size=TRAINING_BATCH_SIZE,
        max_num_epochs=MAX_NUM_EPOCHS,
        show_train_summary=False,
    )
    log_status(f"Building posterior for {density_estimator}")
    posterior = inference.build_posterior(estimator)
    wall_time_s = time.perf_counter() - t0
    metadata = {
        "n_simulations": int(n_simulations),
        "density_estimator": density_estimator,
        "hidden_features": 128,
        "num_transforms": 5,
        "training_batch_size": TRAINING_BATCH_SIZE,
        "max_num_epochs": MAX_NUM_EPOCHS,
        "wall_time_s": wall_time_s,
        "cache_path": str(cache_path),
    }
    with open(cache_path, "wb") as f:
        pickle.dump({"posterior": posterior, "scaler": scaler, "metadata": metadata}, f)
    log_status(f"Trained {density_estimator} posterior on {n_simulations:,} sims in {wall_time_s/60:.1f} min")
    return posterior, scaler, metadata


def sample_posterior_for_x(posterior, scaler, x, *, n_samples=5000, seed=0):
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(scaler.transform(np.asarray(x, dtype=np.float32)[None, :])[0], dtype=torch.float32)
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_tensor, show_progress_bars=False)
    return samples.cpu().numpy().astype(np.float32)


def pick_eval(sid, replicate=0):
    matches = np.where(scenario_id == sid)[0]
    idx = int(matches[min(replicate, len(matches) - 1)])
    return X_eval[idx], theta_eval[idx], scenario_name[idx]
"""
    ),
    md("""## 6. Sensitivity study: simulation budget"""),
    code(
        """sensitivity_results = []
for n_sims in SENSITIVITY_BUDGETS:
    log_status(f"Sensitivity run starting: MAF with {n_sims:,} simulations")
    cache = RESULTS / f"wu2003_nb24_sb_sensitivity_maf_{n_sims}.pkl"
    posterior, scaler, meta = train_snpe_from_bank(
        theta_train,
        X_train,
        n_simulations=n_sims,
        density_estimator="maf",
        seed=42 + int(n_sims),
        cache_path=cache,
    )
    result = {"n_sims": n_sims, "posterior": posterior, "scaler": scaler, "metadata": meta, "samples": {}}
    for sid, label, color in [(1, "W1 healthy", "steelblue"), (11, "W11 snowball", "tomato")]:
        x_obs, theta_true, _ = pick_eval(sid)
        log_status(f"Sampling sensitivity posterior: n_sims={n_sims:,}, scenario={label}, samples={POSTERIOR_SAMPLES}")
        result["samples"][sid] = {
            "label": label,
            "color": color,
            "true": theta_true,
            "samples": sample_posterior_for_x(posterior, scaler, x_obs, n_samples=POSTERIOR_SAMPLES, seed=20260625 + sid + n_sims),
        }
    sensitivity_results.append(result)
    log_status(f"Sensitivity run complete: MAF with {n_sims:,} simulations")

log_status("Building sensitivity scatter figure")
fig, axes = plt.subplots(2, len(sensitivity_results), figsize=(14, 7), constrained_layout=True)
axes = np.asarray(axes).reshape(2, len(sensitivity_results))
for col, result in enumerate(sensitivity_results):
    for row, sid in enumerate([1, 11]):
        item = result["samples"][sid]
        samples = item["samples"]
        truth = item["true"]
        ax = axes[row, col]
        ax.scatter(samples[:, 0], samples[:, 2], s=4, alpha=0.18, color=item["color"])
        ax.axvline(truth[0], color="black", linewidth=1.5, linestyle="--", label=f"alpha={truth[0]:.2f}")
        ax.axhline(truth[2], color="purple", linewidth=1.5, linestyle="--", label=f"eta_col={truth[2]:.2f}")
        ax.set_xlim(PRIOR_LOW[0] - 0.02, PRIOR_HIGH[0] + 0.02)
        ax.set_ylim(PRIOR_LOW[2] - 0.02, PRIOR_HIGH[2] + 0.02)
        ax.set_xlabel("alpha")
        ax.set_ylabel("eta_col")
        ax.set_title(f"{item['label']}\\nn_sims={result['n_sims']:,}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
fig.suptitle("SNPE S-B posterior [alpha, eta_col]: sensitivity over n_simulations (MAF)")
fig.savefig(FIGS / "24_sensitivity_scatter.png", dpi=130, bbox_inches="tight")
log_status(f"Saved {FIGS / '24_sensitivity_scatter.png'}")
plt.show()
"""
    ),
    md("""## 7. Final model: NSF posterior"""),
    code(
        """FINAL_PATH = RESULTS / "wu2003_nb24_sb_sbi_posterior_final.pkl"
log_status("Final NSF posterior phase starting")
posterior_final, scaler_final, meta_final = train_snpe_from_bank(
    theta_train,
    X_train,
    n_simulations=FINAL_N_SIMULATIONS,
    density_estimator="nsf",
    seed=20260625,
    cache_path=FINAL_PATH,
)
(RESULTS / "wu2003_nb24_sb_sbi_training_metadata.json").write_text(json.dumps(meta_final, indent=2))
log_status(f"Saved final training metadata: {RESULTS / 'wu2003_nb24_sb_sbi_training_metadata.json'}")
print(json.dumps(meta_final, indent=2), flush=True)
"""
    ),
    md("""## 8. Posterior recovery: W1 healthy and W11 snowball"""),
    code(
        """N_SAMPLES = POSTERIOR_SAMPLES_BIG
log_status(f"Sampling W1/W11 posterior recovery cases: samples={N_SAMPLES}")
posterior_cases = []
for sid, label, color in [(1, "W1 -- healthy", "steelblue"), (11, "W11 -- snowball", "tomato")]:
    x_obs, truth, _ = pick_eval(sid)
    log_status(f"Sampling recovery posterior: scenario={label}")
    posterior_cases.append({
        "scenario_id": sid,
        "label": label,
        "color": color,
        "truth": truth,
        "samples": sample_posterior_for_x(posterior_final, scaler_final, x_obs, n_samples=N_SAMPLES, seed=20260625 + sid),
    })

fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for row, case in enumerate(posterior_cases):
    samples = case["samples"]
    truth = case["truth"]
    color = case["color"]
    for col, (param_idx, param_name) in enumerate([(0, "alpha"), (2, "eta_col")]):
        ax = axes[row, col]
        vals = samples[:, param_idx]
        ci5, ci95 = np.percentile(vals, [5, 95])
        covered = ci5 <= truth[param_idx] <= ci95
        ax.hist(vals, bins=60, color=color, alpha=0.82)
        ax.axvline(truth[param_idx], color="black", linewidth=2, linestyle="--", label=f"true {param_name}={truth[param_idx]:.2f}")
        ax.axvspan(ci5, ci95, alpha=0.20, color=color, label="90% interval")
        ax.set_xlabel(param_name)
        ax.set_ylabel("count")
        ax.set_title(f"{case['label']}\\n{param_name} mean={vals.mean():.3f}, covered={covered}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
fig.suptitle("SNPE S-B posterior: alpha and eta_col marginals -- M4 recovery check")
fig.savefig(FIGS / "24_posterior_recovery.png", dpi=130, bbox_inches="tight")
log_status(f"Saved {FIGS / '24_posterior_recovery.png'}")
plt.show()
"""
    ),
    md("""## 9. Marginal posterior densities across scenarios"""),
    code(
        """scenario_sample_cache = {}
for sid in sorted(np.unique(scenario_id)):
    x_obs, truth, name = pick_eval(int(sid))
    log_status(f"Sampling all-scenario marginal posterior: scenario_id={int(sid)}, name={name}, samples={POSTERIOR_SAMPLES}")
    samples = sample_posterior_for_x(posterior_final, scaler_final, x_obs, n_samples=POSTERIOR_SAMPLES, seed=20260625 + int(sid))
    scenario_sample_cache[int(sid)] = {"name": name, "truth": truth, "samples": samples}

fig, axes = plt.subplots(len(scenario_sample_cache), len(PARAMETER_NAMES), figsize=(16, 2.1 * len(scenario_sample_cache)), constrained_layout=True)
scenario_colors = plt.cm.tab20(np.linspace(0, 1, len(scenario_sample_cache)))
for row, (sid, item) in enumerate(scenario_sample_cache.items()):
    color = scenario_colors[row]
    for col, name in enumerate(PARAMETER_NAMES):
        ax = axes[row, col]
        vals = item["samples"][:, col]
        lo, hi = PRIOR_LOW[col], PRIOR_HIGH[col]
        grid = np.linspace(lo, hi, 250)
        if np.std(vals) > 1e-12:
            try:
                density = gaussian_kde(vals)(grid)
                ax.fill_between(grid, density, alpha=0.30, color=color)
                ax.plot(grid, density, color=color, linewidth=1.4)
            except Exception:
                ax.hist(vals, bins=30, density=True, color=color, alpha=0.35)
        ci5, ci95 = np.percentile(vals, [5, 95])
        ax.axvspan(ci5, ci95, color=color, alpha=0.12)
        ax.axvline(item["truth"][col], color="black", linewidth=1.3, linestyle="--")
        ax.set_xlim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))
        if row == 0:
            ax.set_title(name, fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(item["name"].replace("_", "\\n"), fontsize=8)
        ax.grid(alpha=0.18)
fig.suptitle("SNPE S-B marginal posterior densities -- all closed-loop scenarios")
fig.savefig(FIGS / "24_marginal_posteriors.png", dpi=130, bbox_inches="tight")
log_status(f"Saved {FIGS / '24_marginal_posteriors.png'}")
plt.show()
"""
    ),
    md("""## 10. 2-D joint posterior and fault classification"""),
    code(
        """def classify_wu_samples(samples):
    reactor = (samples[:, 0] < 0.85) | (samples[:, 1] < 0.85)
    column = (samples[:, 2] < 0.80) | (samples[:, 3] < 0.85) | (samples[:, 3] > 1.15)
    feed = np.abs(samples[:, 4] - 0.90) > 0.05
    labels_pred = np.where(~reactor & ~column & ~feed, "healthy", "fault")
    labels_pred = np.where(reactor & ~column & ~feed, "reactor", labels_pred)
    labels_pred = np.where(~reactor & column & ~feed, "column", labels_pred)
    labels_pred = np.where(~reactor & ~column & feed, "feed", labels_pred)
    labels_pred = np.where(reactor & column, "reactor+column", labels_pred)
    labels_pred = np.where(reactor & feed, "reactor+feed", labels_pred)
    labels_pred = np.where(column & feed, "column+feed", labels_pred)
    values, counts = np.unique(labels_pred, return_counts=True)
    probs = {value: count / len(labels_pred) for value, count in zip(values, counts)}
    return max(probs, key=probs.get), probs

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
log_status("Building joint posterior/classification figure")
for ax, sid, color in zip(axes, [1, 11], ["steelblue", "tomato"]):
    item = scenario_sample_cache[int(sid)]
    samples = item["samples"]
    truth = item["truth"]
    class_name, probs = classify_wu_samples(samples)
    ax.scatter(samples[:, 0], samples[:, 2], s=4, alpha=0.18, color=color)
    ax.axvline(truth[0], color="black", linewidth=1.5, linestyle="--", label=f"alpha={truth[0]:.2f}")
    ax.axhline(truth[2], color="purple", linewidth=1.5, linestyle="--", label=f"eta_col={truth[2]:.2f}")
    ax.axvline(0.85, color="gray", linewidth=1, linestyle=":")
    ax.axhline(0.80, color="gray", linewidth=1, linestyle=":")
    ax.text(0.03, 0.03, f"Class: {class_name}\\nP={probs[class_name]:.2f}", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_xlim(PRIOR_LOW[0] - 0.02, PRIOR_HIGH[0] + 0.02)
    ax.set_ylim(PRIOR_LOW[2] - 0.02, PRIOR_HIGH[2] + 0.02)
    ax.set_xlabel("alpha (catalyst activity)")
    ax.set_ylabel("eta_col (column efficiency)")
    ax.set_title(item["name"].replace("_", " "))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
fig.suptitle("SNPE S-B 2-D posterior [alpha, eta_col] with fault thresholds")
fig.savefig(FIGS / "24_joint_posterior_2d.png", dpi=130, bbox_inches="tight")
log_status(f"Saved {FIGS / '24_joint_posterior_2d.png'}")
plt.show()
"""
    ),
    md("""## 11. Posterior recovery across all evaluation scenarios"""),
    code(
        """recovery_rows = []
rank_rows = []
log_status("Computing all-scenario posterior recovery metrics")
for sid, item in scenario_sample_cache.items():
    samples = item["samples"]
    truth = item["truth"]
    row = {"scenario_id": sid, "scenario_name": item["name"]}
    for j, name in enumerate(PARAMETER_NAMES):
        vals = samples[:, j]
        ci5, ci95 = np.percentile(vals, [5, 95])
        row[f"true_{name}"] = float(truth[j])
        row[f"mean_{name}"] = float(vals.mean())
        row[f"abs_error_{name}"] = abs(row[f"mean_{name}"] - row[f"true_{name}"])
        row[f"covered90_{name}"] = bool(ci5 <= truth[j] <= ci95)
        rank_rows.append({"parameter": name, "scaled_rank": float(np.mean(vals < truth[j]))})
    recovery_rows.append(row)

scenario_recovery = pd.DataFrame(recovery_rows)
sbc_ranks = pd.DataFrame(rank_rows)
display(scenario_recovery)

fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(16, 3.2), constrained_layout=True)
for ax, name in zip(axes, PARAMETER_NAMES):
    ax.scatter(scenario_recovery[f"true_{name}"], scenario_recovery[f"mean_{name}"], s=35, alpha=0.8)
    lo, hi = PRIOR_LOW[PARAMETER_NAMES.index(name)], PRIOR_HIGH[PARAMETER_NAMES.index(name)]
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title(name)
    ax.set_xlabel("true")
    ax.set_ylabel("posterior mean")
    ax.grid(alpha=0.25)
fig.suptitle("SNPE S-B posterior recovery across closed-loop scenarios")
fig.savefig(FIGS / "24_posterior_recovery_all_scenarios.png", dpi=130, bbox_inches="tight")
log_status(f"Saved {FIGS / '24_posterior_recovery_all_scenarios.png'}")
plt.show()
"""
    ),
    md("""## 12. Simulation-based calibration"""),
    code(
        """N_SBC = min(int(os.environ.get("WU2003_NB24_N_SBC", "500")), len(theta_train) // 5)
log_status(f"Starting SBC rank diagnostics: N_SBC={N_SBC}, samples_per_case={min(1000, POSTERIOR_SAMPLES)}")
rng = np.random.default_rng(20260626)
sbc_idx = rng.choice(len(theta_train), size=N_SBC, replace=False)
sbc_samples = []
sbc_ranks_rows = []
for count, idx in enumerate(sbc_idx):
    if count % 25 == 0 or count == N_SBC - 1:
        log_status(f"SBC sampling progress: {count + 1}/{N_SBC}")
    samples = sample_posterior_for_x(posterior_final, scaler_final, X_train[idx], n_samples=min(1000, POSTERIOR_SAMPLES), seed=300000 + count)
    truth = theta_train[idx]
    sbc_samples.append(samples)
    for j, name in enumerate(PARAMETER_NAMES):
        sbc_ranks_rows.append({"parameter": name, "scaled_rank": float(np.mean(samples[:, j] < truth[j]))})

sbc_ranks = pd.DataFrame(sbc_ranks_rows)
sbc_checks = []
fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(16, 3.2), constrained_layout=True)
for ax, name in zip(axes, PARAMETER_NAMES):
    vals = sbc_ranks.loc[sbc_ranks["parameter"].eq(name), "scaled_rank"].to_numpy()
    ks = kstest(vals, "uniform")
    sbc_checks.append({"parameter": name, "ks_p": float(ks.pvalue)})
    ax.hist(vals, bins=np.linspace(0, 1, 11), density=True, color="#4C78A8", edgecolor="white", alpha=0.8)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"{name}\\nKS p={ks.pvalue:.3f}")
    ax.set_xlabel("rank / n_samples")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)
axes[0].set_ylabel("density")
fig.suptitle(f"SNPE S-B SBC rank histograms ({N_SBC} prior test cases)")
fig.savefig(FIGS / "24_sbc_ranks.png", dpi=130, bbox_inches="tight")
log_status(f"Saved {FIGS / '24_sbc_ranks.png'}")
plt.show()
sbc_checks = pd.DataFrame(sbc_checks)
display(sbc_checks)
"""
    ),
    md("""## 13. Persist metrics and acceptance checks"""),
    code(
        """metrics_rows = []
log_status("Persisting nb24 metrics and acceptance outputs")
for name in PARAMETER_NAMES:
    errors = scenario_recovery[f"mean_{name}"] - scenario_recovery[f"true_{name}"]
    metrics_rows.append({
        "parameter": name,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "scenario_coverage_90": float(np.mean(scenario_recovery[f"covered90_{name}"])),
        "sbc_rank_ks_p": float(sbc_checks.loc[sbc_checks["parameter"].eq(name), "ks_p"].iloc[0]),
    })
posterior_metrics = pd.DataFrame(metrics_rows)

metrics_path = RESULTS / "wu2003_nb24_sb_sbi_posterior_metrics.csv"
scenario_path = RESULTS / "wu2003_nb24_sb_sbi_scenario_recovery.csv"
ranks_path = RESULTS / "wu2003_nb24_sb_sbi_sbc_ranks.csv"
metadata_path = RESULTS / "wu2003_nb24_sb_sbi_training_metadata.json"

posterior_metrics.to_csv(metrics_path, index=False)
scenario_recovery.to_csv(scenario_path, index=False)
sbc_ranks.to_csv(ranks_path, index=False)
metadata = dict(meta_final)
metadata.update({
    "train_bank_path": str(TRAIN_BANK_PATH),
    "train_bank_shape": list(X_train.shape),
    "theta_train_shape": list(theta_train.shape),
    "n_sbc": int(N_SBC),
    "figures": [
        "24_simulator_sanity.png",
        "24_sensitivity_scatter.png",
        "24_posterior_recovery.png",
        "24_marginal_posteriors.png",
        "24_joint_posterior_2d.png",
        "24_posterior_recovery_all_scenarios.png",
        "24_sbc_ranks.png",
    ],
})
metadata_path.write_text(json.dumps(metadata, indent=2))
log_status(f"Saved metrics: {metrics_path}, {scenario_path}, {ranks_path}, {metadata_path}")

display(posterior_metrics)

figure_paths = [FIGS / name for name in metadata["figures"]]
acceptance = pd.DataFrame([
    {
        "check": "continuous-prior S-B training bank shape",
        "expected": (N_SBI_SIMULATIONS, 66),
        "observed": X_train.shape,
        "status": "PASS" if X_train.shape == (N_SBI_SIMULATIONS, 66) else "FAIL",
    },
    {
        "check": "finite training bank",
        "expected": "theta and summaries finite",
        "observed": bool(np.isfinite(theta_train).all() and np.isfinite(X_train).all()),
        "status": "PASS" if bool(np.isfinite(theta_train).all() and np.isfinite(X_train).all()) else "FAIL",
    },
    {
        "check": "final SNPE posterior trained",
        "expected": f"NSF posterior on {FINAL_N_SIMULATIONS:,} simulations",
        "observed": {"n_simulations": meta_final["n_simulations"], "density": meta_final["density_estimator"]},
        "status": "PASS" if meta_final["n_simulations"] == FINAL_N_SIMULATIONS and meta_final["density_estimator"] == "nsf" else "FAIL",
    },
    {
        "check": "SBC executed",
        "expected": "rank diagnostics for all five parameters",
        "observed": sbc_checks.shape,
        "status": "PASS" if sbc_checks.shape == (5, 2) else "FAIL",
    },
    {
        "check": "nb04-style figures exist",
        "expected": [p.name for p in figure_paths],
        "observed": [p.name for p in figure_paths if p.exists()],
        "status": "PASS" if all(p.exists() for p in figure_paths) else "FAIL",
    },
    {
        "check": "metrics files exist",
        "expected": "metrics, recovery, ranks, metadata",
        "observed": all(p.exists() for p in [metrics_path, scenario_path, ranks_path, metadata_path]),
        "status": "PASS" if all(p.exists() for p in [metrics_path, scenario_path, ranks_path, metadata_path]) else "FAIL",
    },
])
acceptance
"""
    ),
    md(
        """## 14. Interpretation

nb24 now follows nb04's actual SBI pattern. The scenario bank is used for
evaluation, while the SNPE posterior is trained on a fresh continuous-prior
simulation bank over `[alpha, beta_r, eta_col, xi_reb, z_A0_eff]`. S-B uses the
66-D conventional summary vector, so the trained posterior is the correct nb24
handoff for the later S-B/S-A comparison and banana-posterior experiments.
"""
    ),
    md(
        """## 15. Calibration addendum from nb24b

This section is appended rather than inserted into the original nb24 workflow so
the raw SNPE training, sampling, recovery plots, and SBC outputs remain intact.
nb24b adds a separate held-out calibration check: it does not change the learned
posterior density, but it shows how the reported 90% intervals should be treated
for parameters whose point recovery is good but uncertainty is overconfident.

The motivation is the eta/xi result from nb24a and nb24b. `eta_col` and `xi_reb`
have small posterior mean errors, but the raw intervals under-cover the closed-loop
scenario truths. Therefore nb24 should report both the raw posterior and a
calibrated reporting interval, and should avoid describing eta/xi as solved solely
because their means are accurate.
"""
    ),
    code(
        """NB24B_FACTORS_PATH = RESULTS / "wu2003_nb24b_calibration_factors.csv"
NB24B_SCENARIO_PATH = RESULTS / "wu2003_nb24b_scenario_repair_summary.csv"
NB24B_METRICS_PATH = RESULTS / "wu2003_nb24b_raw_vs_repaired_metrics.csv"

required_nb24b = [NB24B_FACTORS_PATH, NB24B_SCENARIO_PATH, NB24B_METRICS_PATH]
missing_nb24b = [path for path in required_nb24b if not path.exists()]
if missing_nb24b:
    raise FileNotFoundError(
        "Run scripts/build_nb_24b.py before this addendum. Missing: "
        + ", ".join(str(path) for path in missing_nb24b)
    )

calibration_factors = pd.read_csv(NB24B_FACTORS_PATH)
scenario_repair_summary = pd.read_csv(NB24B_SCENARIO_PATH)
heldout_calibration = pd.read_csv(NB24B_METRICS_PATH)

raw_scenario = scenario_repair_summary[scenario_repair_summary["posterior"].eq("raw")].copy()
repaired_scenario = scenario_repair_summary[scenario_repair_summary["posterior"].eq("repaired")].copy()
scenario_delta = raw_scenario[["parameter", "coverage90", "mean_width90_prior_frac"]].merge(
    repaired_scenario[["parameter", "coverage90", "mean_width90_prior_frac"]],
    on="parameter",
    suffixes=("_raw", "_calibrated"),
)
scenario_delta = scenario_delta.merge(
    calibration_factors[["parameter", "interval_scale", "minimum_repair_scale"]],
    on="parameter",
    how="left",
)
scenario_delta["coverage90_gain"] = scenario_delta["coverage90_calibrated"] - scenario_delta["coverage90_raw"]
scenario_delta["calibration_note"] = np.select(
    [
        scenario_delta["parameter"].eq("eta_col"),
        scenario_delta["parameter"].eq("xi_reb"),
        scenario_delta["coverage90_gain"] > 0,
    ],
    [
        "use calibrated reporting interval; raw NSF interval under-covers",
        "use conservative boundary-aware reporting interval; raw NSF interval under-covers near xi=1",
        "minor interval calibration improvement",
    ],
    default="raw interval retained",
)

calibration_addendum = scenario_delta[[
    "parameter",
    "interval_scale",
    "coverage90_raw",
    "coverage90_calibrated",
    "coverage90_gain",
    "mean_width90_prior_frac_raw",
    "mean_width90_prior_frac_calibrated",
    "calibration_note",
]]
calibration_addendum.to_csv(RESULTS / "wu2003_nb24_sb_sbi_calibration_addendum.csv", index=False)

display(calibration_addendum.round(5))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
x = np.arange(len(PARAMETER_NAMES))
width = 0.36
plot_table = calibration_addendum.set_index("parameter").loc[PARAMETER_NAMES]
axes[0].bar(x - width / 2, plot_table["coverage90_raw"], width=width, label="raw", color="#4C78A8", alpha=0.85)
axes[0].bar(x + width / 2, plot_table["coverage90_calibrated"], width=width, label="calibrated", color="#E45756", alpha=0.85)
axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("scenario 90% coverage")
axes[0].set_title("nb24 scenario coverage before/after nb24b repair")
axes[0].legend(fontsize=8)

axes[1].bar(x, plot_table["interval_scale"], color="#72B7B2", alpha=0.9)
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
axes[1].set_ylabel("interval scale")
axes[1].set_title("reporting interval inflation from nb24b")
for ax in axes:
    ax.set_xticks(x, PARAMETER_NAMES, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
fig.suptitle("nb24 addendum: keep raw posterior, report calibrated eta/xi uncertainty")
fig.savefig(FIGS / "24_calibration_addendum.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        """### Why this changes the nb24 interpretation

The appended calibration addendum changes the reporting interpretation, not the
SNPE training code. The raw nb24 posterior remains the object produced by the
nb04-style continuous-prior workflow. The added nb24b evidence says how to report
its uncertainty:

- `alpha`, `beta_r`, and `z_A0_eff` can still be discussed mainly through the
  closed-loop information-loss and broad-but-mostly-calibrated framing.
- `eta_col` has accurate point recovery but raw intervals that are too narrow;
  nb24 should report the nb24b calibrated interval scale.
- `xi_reb` needs an even more conservative boundary-aware reporting interval,
  because several closed-loop scenarios sit near the prior upper boundary where
  the raw neural posterior under-covers.

Therefore nb24 should not replace its existing posterior figures. It should append
this calibration table and state that eta/xi point estimates are useful, while raw
eta/xi posterior intervals are overconfident unless calibrated against independent
prior-predictive simulations.
"""
    ),
]


def main() -> int:
    nb = new_notebook()
    nb.cells = add_code_explanations(CELLS)
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
    nb_path = repo_root / "notebooks" / "24_wu2003_sbi_training_sb.ipynb"

    print(f"Executing notebook -> {nb_path}", flush=True)

    def _cell_label(cell) -> str:
        first = cell.source.strip().splitlines()[0] if cell.source.strip() else "empty cell"
        return first[:100]

    def _on_cell_start(cell, cell_index, **kwargs):
        print(f"[nb24-run] start cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    def _on_cell_complete(cell, cell_index, **kwargs):
        print(f"[nb24-run] done  cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    def _on_cell_error(cell, cell_index, execute_reply, **kwargs):
        print(f"[nb24-run] ERROR cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    class StreamingNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            content = msg.get("content", {})
            if msg.get("msg_type") == "stream":
                text = content.get("text", "")
                if text:
                    prefix = f"[nb24-cell {cell_index + 1} {content.get('name', 'stream')}] "
                    for line in text.rstrip().splitlines():
                        print(prefix + line, flush=True)
            return super().process_message(msg, cell, cell_index)

    client = StreamingNotebookClient(
        nb,
        kernel_name="python3",
        timeout=None,
        resources={"metadata": {"path": str(repo_root)}},
        on_cell_start=_on_cell_start,
        on_cell_complete=_on_cell_complete,
        on_cell_error=_on_cell_error,
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"Wrote {nb_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
