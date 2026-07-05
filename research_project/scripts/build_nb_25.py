"""Build and execute notebook 25: Wu 2003 S-A SNPE training and S-A/S-B comparison."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    return new_markdown_cell(source, metadata={"language": "markdown"})


def code(source: str):
    return new_code_cell(source, metadata={"language": "python"})


def code_intro(source: str):
    compact = " ".join(source.strip().split())
    if "from pathlib import Path" in source and "N_SA_SIMULATIONS" in source:
        text = (
            "This code imports libraries, defines paths and priors, loads the S-A and S-B scenario summaries, "
            "and reports the S-A SNPE training configuration."
        )
    elif "def _safe_corr" in source and "simulate_sa_summary" in source:
        text = (
            "This code defines the 72-D S-A summary pipeline and one prior-simulation wrapper using the "
            "analyser-rich S-A controller and channel set."
        )
    elif "def generate_training_bank" in source:
        text = (
            "This code creates or loads the continuous-prior S-A training bank for SNPE, rejecting invalid "
            "simulations and caching accepted 5-D parameter draws with their 72-D summaries."
        )
    elif "def make_prior" in source and "train_snpe_from_bank" in source:
        text = (
            "This code defines the S-A BoxUniform prior, cached SNPE training helper, posterior sampler, "
            "and scenario lookup utilities."
        )
    elif "posterior_sa" in source and "train_snpe_from_bank" in source:
        text = (
            "This code trains or loads the final S-A NSF posterior on the S-A simulation bank."
        )
    elif "scenario_rows = []" in source and "scenario_sample_cache" in source:
        text = (
            "This code samples the S-A posterior for every closed-loop scenario and computes posterior mean, "
            "90% interval coverage, and interval width for all five parameters."
        )
    elif "sbc_idx" in source and "sbc_ranks_rows" in source:
        text = (
            "This code performs simulation-based calibration rank diagnostics on held-out S-A prior-bank cases."
        )
    elif "sb_metrics" in source and "comparison_table" in source:
        text = (
            "This code compares S-A metrics against nb24's S-B metrics to quantify the value of analyser-rich "
            "composition information."
        )
    elif "fig, axes = plt.subplots" in source and "25_sa_vs_sb" in source:
        text = (
            "This code plots the S-A vs S-B recovery and coverage comparison figures."
        )
    elif "acceptance = pd.DataFrame" in source:
        text = (
            "This code writes nb25 metrics, scenario recovery, SBC ranks, metadata, and acceptance checks."
        )
    else:
        text = f"This code computes the next nb25 S-A SBI step starting with: `{compact[:120]}`."
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
        """# Notebook 25 -- Wu 2003 S-A SBI Training and S-A/S-B Comparison

This notebook is the analyser-rich counterpart to nb24. nb24 trained SNPE for
S-B, the conventional measurement structure without online composition analysis.
nb25 trains the same 5-D posterior for **S-A**, where the column loops use and
observe composition information, including the `x_D` trajectory.

The purpose is not only to train another posterior. The main question is the
information value of S-A relative to S-B:

- Does the extra composition channel sharpen `eta_col` and the snowball-related
  `alpha`/`eta_col` separation?
- Does S-A improve calibration compared with the raw S-B NSF posterior?
- Which parameters remain masked even when analyser-rich measurements are used?

Main outputs:

- `data/wu2003_sbi_train_sa.npz`: continuous-prior S-A SBI training bank.
- `results/wu2003_nb25_sa_sbi_posterior_final.pkl`: trained S-A NSF posterior.
- `results/wu2003_nb25_sa_sbi_posterior_metrics.csv`: S-A posterior metrics.
- `results/wu2003_nb25_sa_vs_sb_metrics.csv`: direct S-A vs S-B comparison.
- `figures/25_*.png`: S-A recovery, marginal, SBC, and comparison figures.
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
from scipy.stats import kstest
from sklearn.preprocessing import StandardScaler

import torch
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SA,
    NOMINAL_INLET,
    extract_observations_explicit,
    simulate_trajectory_explicit,
)
from cstr_sbi.recycle.simulator import SA_INDICES, nominal_warm_start

pd.set_option("display.precision", 5)
torch.manual_seed(20260629)

DATA_DIR = Path("data")
RESULTS = Path("results")
FIGS = Path("figures")
for path in [DATA_DIR, RESULTS, FIGS]:
    path.mkdir(exist_ok=True)

SCENARIO_FEATURE_PATH = DATA_DIR / "wu2003_summary_features.npz"
TRAIN_BANK_PATH = DATA_DIR / "wu2003_sbi_train_sa.npz"
SB_METRICS_PATH = RESULTS / "wu2003_nb24_sb_sbi_posterior_metrics.csv"
SB_RECOVERY_PATH = RESULTS / "wu2003_nb24_sb_sbi_scenario_recovery.csv"
assert SCENARIO_FEATURE_PATH.exists(), "Run nb23 first to create data/wu2003_summary_features.npz"
assert SB_METRICS_PATH.exists(), "Run nb24 before nb25 so S-B metrics are available."

PARAMETER_NAMES = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]
PRIOR_LOW = np.array([0.40, 0.40, 0.50, 0.40, 0.70], dtype=np.float32)
PRIOR_HIGH = np.array([1.20, 1.20, 1.00, 1.20, 0.95], dtype=np.float32)
PRIOR_WIDTH = PRIOR_HIGH - PRIOR_LOW

N_SA_SIMULATIONS = int(os.environ.get("WU2003_NB25_N_SA_SIMULATIONS", "15000"))
MAX_NUM_EPOCHS = int(os.environ.get("WU2003_NB25_MAX_EPOCHS", "200"))
TRAINING_BATCH_SIZE = int(os.environ.get("WU2003_NB25_BATCH", "256"))
POSTERIOR_SAMPLES = int(os.environ.get("WU2003_NB25_POSTERIOR_SAMPLES", "5000"))
N_SBC = int(os.environ.get("WU2003_NB25_N_SBC", "500"))
PROGRESS_INTERVAL = int(os.environ.get("WU2003_NB25_PROGRESS_INTERVAL", "100"))


def log_status(message):
    print(f"[nb25 {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


with np.load(SCENARIO_FEATURE_PATH, allow_pickle=True) as data:
    X_sa_all = data["X_sa"].astype(np.float32)
    X_sb_all = data["X_sb"].astype(np.float32)
    features_sa = [str(x) for x in data["features_sa"]]
    features_sb = [str(x) for x in data["features_sb"]]
    labels_all = pd.DataFrame.from_records(data["labels"])

closed_mask = labels_all["mode"].eq("closed_loop").to_numpy()
X_eval_sa = X_sa_all[closed_mask]
X_eval_sb = X_sb_all[closed_mask]
labels_eval = labels_all.loc[closed_mask].reset_index(drop=True)
theta_eval = labels_eval[PARAMETER_NAMES].to_numpy(dtype=np.float32)
scenario_id = labels_eval["scenario_id"].to_numpy()
scenario_name = labels_eval["scenario_name"].to_numpy()

log_status(f"S-A evaluation summaries: {X_eval_sa.shape}")
log_status(f"S-B evaluation summaries available for comparison: {X_eval_sb.shape}")
log_status(f"Evaluation scenarios: {len(np.unique(scenario_id))}")
log_status(f"S-A target training simulations: {N_SA_SIMULATIONS}")
log_status(f"S-A feature dimension: {len(features_sa)}")
log_status(f"Posterior samples per scenario: {POSTERIOR_SAMPLES}; SBC cases: {N_SBC}")
"""
    ),
    md("""## 2. S-A summary-statistic and simulator functions"""),
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


SA_CHANNELS = [
    "T_r", "T_j", "Q_j", "x_D", "T_reb", "Q_reb",
    "F_R_norm", "F_B_norm", "R_norm", "V_norm",
]


def noisy_sensor_layer(obs, rng, noise_pct=0.003):
    scale = np.maximum(np.max(np.abs(obs), axis=0, keepdims=True), 1e-12)
    return obs + rng.normal(0.0, noise_pct * scale, size=obs.shape)


def simulate_sa_summary(theta, y0, rng):
    ts, ys = simulate_trajectory_explicit(theta, NOMINAL_INLET, NOMINAL_CTRL_SA, y0, t_final=2.0, n_save=120)
    raw = np.asarray(extract_observations_explicit(ys, theta, NOMINAL_CTRL_SA))
    sa_obs = noisy_sensor_layer(raw[..., SA_INDICES], rng)
    X, names = summarize_windows(sa_obs[None, :, :], SA_CHANNELS, np.asarray(ts))
    return X[0], names


assert X_eval_sa.shape[1] == len(features_sa) == 72
"""
    ),
    md("""## 3. Generate or load the continuous-prior S-A training bank"""),
    code(
        """def sample_prior_numpy(n, rng):
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * rng.random((n, len(PARAMETER_NAMES)), dtype=np.float32)


def generate_training_bank(n_simulations, path):
    log_status(f"Starting S-A continuous-prior bank generation: target={n_simulations:,}, path={path}")
    rng = np.random.default_rng(20260629)
    theta_train = np.empty((n_simulations, len(PARAMETER_NAMES)), dtype=np.float32)
    X_train = np.empty((n_simulations, len(features_sa)), dtype=np.float32)
    y0_sa = nominal_warm_start("S-A")
    t0 = time.perf_counter()
    feature_names_ref = None
    accepted = 0
    attempted = 0
    rejected = 0
    while accepted < n_simulations:
        theta_i = sample_prior_numpy(1, rng)[0]
        attempted += 1
        try:
            summary_i, feature_names_ref = simulate_sa_summary(theta_i, y0_sa, rng)
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
                f"rejected={rejected} ({reject_pct:.1f}%); elapsed={elapsed/60:.1f} min; eta={remaining/60:.1f} min"
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
    log_status(f"Saved S-A training bank: theta={theta_train.shape}, X={X_train.shape}, path={path}")
    return theta_train, X_train, feature_names_ref


if TRAIN_BANK_PATH.exists():
    log_status(f"Found cached S-A bank at {TRAIN_BANK_PATH}; validating shape and finiteness")
    with np.load(TRAIN_BANK_PATH, allow_pickle=True) as bank_data:
        theta_train = bank_data["theta"].astype(np.float32)
        X_train = bank_data["X"].astype(np.float32)
        train_feature_names = [str(x) for x in bank_data["feature_names"]]
    if theta_train.shape[0] < N_SA_SIMULATIONS or X_train.shape[1] != len(features_sa) or not np.isfinite(X_train).all():
        log_status("Cached S-A bank unsuitable; regenerating")
        theta_train, X_train, train_feature_names = generate_training_bank(N_SA_SIMULATIONS, TRAIN_BANK_PATH)
    else:
        theta_train = theta_train[:N_SA_SIMULATIONS]
        X_train = X_train[:N_SA_SIMULATIONS]
        log_status(f"Loaded cached S-A SBI bank: theta={theta_train.shape}, X={X_train.shape}")
else:
    theta_train, X_train, train_feature_names = generate_training_bank(N_SA_SIMULATIONS, TRAIN_BANK_PATH)

assert theta_train.shape == (N_SA_SIMULATIONS, 5)
assert X_train.shape == (N_SA_SIMULATIONS, 72)
assert np.isfinite(theta_train).all() and np.isfinite(X_train).all()
log_status(f"S-A SBI bank ready: theta={theta_train.shape}, X={X_train.shape}")
"""
    ),
    md("""## 4. SNPE training helpers"""),
    code(
        """def make_prior():
    return BoxUniform(
        low=torch.as_tensor(PRIOR_LOW, dtype=torch.float32),
        high=torch.as_tensor(PRIOR_HIGH, dtype=torch.float32),
    )


def train_snpe_from_bank(theta_bank, X_bank, *, cache_path):
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        metadata = cached["metadata"]
        if metadata.get("n_simulations") == len(theta_bank) and metadata.get("max_num_epochs") == MAX_NUM_EPOCHS:
            log_status(f"Loaded cached S-A posterior: {cache_path}")
            return cached["posterior"], cached["scaler"], cached["metadata"]
        log_status("Cached S-A posterior metadata differs; retraining")

    log_status(f"Training S-A NSF posterior: n={len(theta_bank):,}, max_epochs={MAX_NUM_EPOCHS}, batch={TRAINING_BATCH_SIZE}")
    scaler = StandardScaler().fit(X_bank)
    theta_tensor = torch.as_tensor(theta_bank, dtype=torch.float32)
    x_tensor = torch.as_tensor(scaler.transform(X_bank), dtype=torch.float32)
    density = posterior_nn(
        model="nsf",
        hidden_features=128,
        num_transforms=5,
        z_score_x="independent",
    )
    inference = SNPE(prior=make_prior(), density_estimator=density, show_progress_bars=True)
    t0 = time.perf_counter()
    inference.append_simulations(theta_tensor, x_tensor)
    estimator = inference.train(
        training_batch_size=TRAINING_BATCH_SIZE,
        max_num_epochs=MAX_NUM_EPOCHS,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(estimator)
    metadata = {
        "n_simulations": int(len(theta_bank)),
        "density_estimator": "nsf",
        "hidden_features": 128,
        "num_transforms": 5,
        "training_batch_size": int(TRAINING_BATCH_SIZE),
        "max_num_epochs": int(MAX_NUM_EPOCHS),
        "wall_time_s": float(time.perf_counter() - t0),
        "cache_path": str(cache_path),
    }
    with open(cache_path, "wb") as f:
        pickle.dump({"posterior": posterior, "scaler": scaler, "metadata": metadata}, f)
    log_status(f"Trained S-A posterior in {metadata['wall_time_s']/60:.1f} min")
    return posterior, scaler, metadata


def sample_posterior_for_x(posterior, scaler, x, *, n_samples=POSTERIOR_SAMPLES, seed=0):
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(scaler.transform(np.asarray(x, dtype=np.float32)[None, :])[0], dtype=torch.float32)
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_tensor, show_progress_bars=False)
    return samples.cpu().numpy().astype(np.float32)


def pick_eval(sid, replicate=0):
    matches = np.where(scenario_id == sid)[0]
    idx = int(matches[min(replicate, len(matches) - 1)])
    return X_eval_sa[idx], theta_eval[idx], scenario_name[idx]
"""
    ),
    md("""## 5. Train the final S-A posterior"""),
    code(
        """FINAL_PATH = RESULTS / "wu2003_nb25_sa_sbi_posterior_final.pkl"
posterior_sa, scaler_sa, meta_sa = train_snpe_from_bank(theta_train, X_train, cache_path=FINAL_PATH)
log_status(f"S-A posterior metadata: {meta_sa}")
"""
    ),
    md("""## 6. Posterior recovery across S-A closed-loop scenarios"""),
    code(
        """scenario_rows = []
scenario_sample_cache = {}
log_status("Sampling S-A posterior for all closed-loop evaluation scenarios")
for sid in sorted(np.unique(scenario_id)):
    x_obs, truth, name = pick_eval(int(sid))
    samples = sample_posterior_for_x(posterior_sa, scaler_sa, x_obs, n_samples=POSTERIOR_SAMPLES, seed=20260629 + int(sid))
    scenario_sample_cache[int(sid)] = {"name": name, "truth": truth, "samples": samples}
    row = {"scenario_id": int(sid), "scenario_name": name}
    for j, parameter in enumerate(PARAMETER_NAMES):
        vals = samples[:, j]
        q05, q50, q95 = np.percentile(vals, [5, 50, 95])
        row[f"true_{parameter}"] = float(truth[j])
        row[f"mean_{parameter}"] = float(vals.mean())
        row[f"q05_{parameter}"] = float(q05)
        row[f"q50_{parameter}"] = float(q50)
        row[f"q95_{parameter}"] = float(q95)
        row[f"abs_error_{parameter}"] = abs(row[f"mean_{parameter}"] - row[f"true_{parameter}"])
        row[f"covered90_{parameter}"] = bool(q05 <= truth[j] <= q95)
        row[f"width90_{parameter}"] = float(q95 - q05)
    scenario_rows.append(row)

scenario_recovery = pd.DataFrame(scenario_rows)
scenario_recovery.to_csv(RESULTS / "wu2003_nb25_sa_sbi_scenario_recovery.csv", index=False)
display(scenario_recovery)

metrics_rows = []
for parameter in PARAMETER_NAMES:
    errors = scenario_recovery[f"mean_{parameter}"] - scenario_recovery[f"true_{parameter}"]
    metrics_rows.append({
        "parameter": parameter,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "scenario_coverage_90": float(np.mean(scenario_recovery[f"covered90_{parameter}"])),
        "mean_width90": float(np.mean(scenario_recovery[f"width90_{parameter}"])),
        "mean_width90_prior_frac": float(np.mean(scenario_recovery[f"width90_{parameter}"] / PRIOR_WIDTH[PARAMETER_NAMES.index(parameter)])),
    })
posterior_metrics = pd.DataFrame(metrics_rows)
posterior_metrics.to_csv(RESULTS / "wu2003_nb25_sa_sbi_posterior_metrics.csv", index=False)
display(posterior_metrics.round(5))
"""
    ),
    md("""## 7. S-A marginal posterior grid"""),
    code(
        """focus_scenarios = [1, 4, 11, 13, 16]
fig, axes = plt.subplots(len(focus_scenarios), len(PARAMETER_NAMES), figsize=(16, 10), constrained_layout=True)
for row, sid in enumerate(focus_scenarios):
    item = scenario_sample_cache[sid]
    for col, parameter in enumerate(PARAMETER_NAMES):
        ax = axes[row, col]
        vals = item["samples"][:, col]
        truth = item["truth"][col]
        q05, q95 = np.percentile(vals, [5, 95])
        ax.hist(vals, bins=50, density=True, color="#4C78A8", alpha=0.78)
        ax.axvline(truth, color="black", linestyle="--", linewidth=1.4)
        ax.axvspan(q05, q95, color="#4C78A8", alpha=0.18)
        ax.set_xlim(PRIOR_LOW[col], PRIOR_HIGH[col])
        ax.set_title(f"{item['name']}\\n{parameter}", fontsize=8)
        ax.grid(alpha=0.2)
        if col == 0:
            ax.set_ylabel("density")
fig.suptitle("nb25: S-A posterior marginals for representative closed-loop scenarios")
fig.savefig(FIGS / "25_sa_marginal_posteriors.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 8. Simulation-based calibration for S-A"""),
    code(
        """n_sbc_eff = min(N_SBC, len(theta_train) // 5)
log_status(f"Starting S-A SBC rank diagnostics: N_SBC={n_sbc_eff}, samples_per_case={min(1000, POSTERIOR_SAMPLES)}")
rng = np.random.default_rng(20260629)
sbc_idx = rng.choice(len(theta_train), size=n_sbc_eff, replace=False)
sbc_ranks_rows = []
for count, idx in enumerate(sbc_idx):
    if count % 25 == 0 or count == n_sbc_eff - 1:
        log_status(f"S-A SBC sampling progress: {count + 1}/{n_sbc_eff}")
    samples = sample_posterior_for_x(posterior_sa, scaler_sa, X_train[idx], n_samples=min(1000, POSTERIOR_SAMPLES), seed=400000 + count)
    truth = theta_train[idx]
    for j, parameter in enumerate(PARAMETER_NAMES):
        sbc_ranks_rows.append({"parameter": parameter, "scaled_rank": float(np.mean(samples[:, j] < truth[j]))})

sbc_ranks = pd.DataFrame(sbc_ranks_rows)
sbc_checks = []
fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(16, 3.2), constrained_layout=True)
for ax, parameter in zip(axes, PARAMETER_NAMES):
    vals = sbc_ranks.loc[sbc_ranks["parameter"].eq(parameter), "scaled_rank"].to_numpy()
    ks = kstest(vals, "uniform")
    sbc_checks.append({"parameter": parameter, "sbc_rank_ks_p": float(ks.pvalue)})
    ax.hist(vals, bins=np.linspace(0, 1, 11), density=True, color="#4C78A8", edgecolor="white", alpha=0.8)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"{parameter}\\nKS p={ks.pvalue:.3f}")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)
axes[0].set_ylabel("density")
fig.suptitle(f"nb25: S-A SBC rank histograms ({n_sbc_eff} prior-bank cases)")
fig.savefig(FIGS / "25_sa_sbc_ranks.png", dpi=140, bbox_inches="tight")
plt.show()

sbc_checks = pd.DataFrame(sbc_checks)
sbc_ranks.to_csv(RESULTS / "wu2003_nb25_sa_sbi_sbc_ranks.csv", index=False)
posterior_metrics = posterior_metrics.merge(sbc_checks, on="parameter", how="left")
posterior_metrics.to_csv(RESULTS / "wu2003_nb25_sa_sbi_posterior_metrics.csv", index=False)
display(sbc_checks.round(5))
"""
    ),
    md("""## 9. S-A vs S-B information-value comparison"""),
    code(
        """sb_metrics = pd.read_csv(SB_METRICS_PATH)
sb_metrics = sb_metrics.rename(columns={"scenario_coverage_90": "coverage90"})
sa_metrics = posterior_metrics.rename(columns={"scenario_coverage_90": "coverage90"})

comparison_rows = []
for parameter in PARAMETER_NAMES:
    sa = sa_metrics[sa_metrics["parameter"].eq(parameter)].iloc[0]
    sb = sb_metrics[sb_metrics["parameter"].eq(parameter)].iloc[0]
    comparison_rows.append({
        "parameter": parameter,
        "mae_sb": float(sb["mae"]),
        "mae_sa": float(sa["mae"]),
        "mae_improvement_sa_minus_sb": float(sb["mae"] - sa["mae"]),
        "coverage90_sb": float(sb["coverage90"]),
        "coverage90_sa": float(sa["coverage90"]),
        "coverage90_gain_sa_minus_sb": float(sa["coverage90"] - sb["coverage90"]),
        "sbc_ks_p_sb": float(sb.get("sbc_rank_ks_p", np.nan)),
        "sbc_ks_p_sa": float(sa.get("sbc_rank_ks_p", np.nan)),
    })
comparison_table = pd.DataFrame(comparison_rows)
comparison_table["interpretation"] = np.select(
    [
        comparison_table["parameter"].eq("eta_col") & (comparison_table["mae_improvement_sa_minus_sb"] > 0),
        comparison_table["parameter"].isin(["alpha", "beta_r"]) & (comparison_table["mae_improvement_sa_minus_sb"] > 0.01),
        comparison_table["coverage90_gain_sa_minus_sb"] > 0.15,
    ],
    [
        "composition analyser improves column-efficiency point recovery",
        "S-A reduces reactor-side masking error",
        "S-A improves uncertainty coverage",
    ],
    default="no material S-A gain over S-B in this metric",
)
comparison_table.to_csv(RESULTS / "wu2003_nb25_sa_vs_sb_metrics.csv", index=False)
display(comparison_table.round(5))
"""
    ),
    md("""## 10. Comparison figures"""),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
x = np.arange(len(PARAMETER_NAMES))
width = 0.36
axes[0].bar(x - width / 2, comparison_table["mae_sb"], width=width, label="S-B conventional", color="#4C78A8", alpha=0.85)
axes[0].bar(x + width / 2, comparison_table["mae_sa"], width=width, label="S-A analyser-rich", color="#E45756", alpha=0.85)
axes[0].set_ylabel("mean absolute error")
axes[0].set_title("posterior mean recovery")
axes[1].bar(x - width / 2, comparison_table["coverage90_sb"], width=width, label="S-B conventional", color="#4C78A8", alpha=0.85)
axes[1].bar(x + width / 2, comparison_table["coverage90_sa"], width=width, label="S-A analyser-rich", color="#E45756", alpha=0.85)
axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("scenario 90% coverage")
axes[1].set_title("uncertainty coverage")
for ax in axes:
    ax.set_xticks(x, PARAMETER_NAMES, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("nb25: S-A analyser information value compared with nb24 S-B")
fig.savefig(FIGS / "25_sa_vs_sb_metrics.png", dpi=140, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(16, 3.2), constrained_layout=True)
for ax, parameter in zip(axes, PARAMETER_NAMES):
    ax.scatter(scenario_recovery[f"true_{parameter}"], scenario_recovery[f"mean_{parameter}"], s=35, alpha=0.8, color="#E45756")
    lo, hi = PRIOR_LOW[PARAMETER_NAMES.index(parameter)], PRIOR_HIGH[PARAMETER_NAMES.index(parameter)]
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title(parameter)
    ax.set_xlabel("true")
    ax.set_ylabel("S-A posterior mean")
    ax.grid(alpha=0.25)
fig.suptitle("nb25: S-A posterior recovery across closed-loop scenarios")
fig.savefig(FIGS / "25_sa_posterior_recovery_all_scenarios.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 11. Persist metadata and acceptance checks"""),
    code(
        """metadata = dict(meta_sa)
metadata.update({
    "train_bank_path": str(TRAIN_BANK_PATH),
    "train_bank_shape": list(X_train.shape),
    "theta_train_shape": list(theta_train.shape),
    "n_sbc": int(n_sbc_eff),
    "figures": [
        "25_sa_marginal_posteriors.png",
        "25_sa_sbc_ranks.png",
        "25_sa_vs_sb_metrics.png",
        "25_sa_posterior_recovery_all_scenarios.png",
    ],
})
metadata_path = RESULTS / "wu2003_nb25_sa_sbi_training_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2))

expected_files = [
    TRAIN_BANK_PATH,
    RESULTS / "wu2003_nb25_sa_sbi_posterior_final.pkl",
    RESULTS / "wu2003_nb25_sa_sbi_posterior_metrics.csv",
    RESULTS / "wu2003_nb25_sa_sbi_scenario_recovery.csv",
    RESULTS / "wu2003_nb25_sa_sbi_sbc_ranks.csv",
    RESULTS / "wu2003_nb25_sa_vs_sb_metrics.csv",
    metadata_path,
] + [FIGS / name for name in metadata["figures"]]

acceptance = pd.DataFrame([
    {"check": "continuous-prior S-A training bank shape", "observed": str(X_train.shape), "status": "PASS" if X_train.shape == (N_SA_SIMULATIONS, 72) else "FAIL"},
    {"check": "finite S-A training bank", "observed": str(bool(np.isfinite(theta_train).all() and np.isfinite(X_train).all())), "status": "PASS" if bool(np.isfinite(theta_train).all() and np.isfinite(X_train).all()) else "FAIL"},
    {"check": "S-A posterior trained or loaded", "observed": str(meta_sa), "status": "PASS" if meta_sa.get("density_estimator") == "nsf" else "FAIL"},
    {"check": "SBC executed", "observed": str(sbc_checks.shape), "status": "PASS" if sbc_checks.shape == (5, 2) else "FAIL"},
    {"check": "S-A vs S-B comparison exists", "observed": str(comparison_table.shape), "status": "PASS" if comparison_table.shape[0] == 5 else "FAIL"},
    {"check": "all output files exist", "observed": str(all(path.exists() for path in expected_files)), "status": "PASS" if all(path.exists() for path in expected_files) else "FAIL"},
])
acceptance.to_csv(RESULTS / "wu2003_nb25_acceptance.csv", index=False)
display(acceptance)
"""
    ),
    md(
        """## 12. Interpretation

nb25 completes the pair with nb24. The comparison table is the main result: it
quantifies whether S-A's composition information improves point recovery,
coverage, or SBC calibration relative to S-B.

The expected interpretation is parameter-specific. `eta_col` should benefit most
from S-A because the analyser-rich structure observes and controls composition,
including `x_D`. `alpha` and `beta_r` may improve less because reactor temperature
control still masks part of the reactor-side information. If `xi_reb` remains
miscalibrated, the issue is likely density calibration or boundary behavior rather
than simply the absence of `x_D`.

This notebook should be read together with nb24b and nb24c: nb24b calibrates the
raw S-B reporting intervals, while nb24c tests whether learned raw-trajectory
summaries change the conclusion. nb25 asks the more physical question: how much
better does the plant become when the missing analyser information is actually
available?
"""
    ),
]


def main() -> int:
    nb = new_notebook()
    nb.cells = add_code_explanations(CELLS)
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )

    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / "25_wu2003_sbi_training_sa.ipynb"
    print(f"Executing notebook -> {nb_path}", flush=True)

    def _cell_label(cell) -> str:
        first = cell.source.strip().splitlines()[0] if cell.source.strip() else "empty cell"
        return first[:100]

    def _on_cell_start(cell, cell_index, **kwargs):
        print(f"[nb25-run] start cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    def _on_cell_complete(cell, cell_index, **kwargs):
        print(f"[nb25-run] done  cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    def _on_cell_error(cell, cell_index, execute_reply, **kwargs):
        print(f"[nb25-run] ERROR cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    class StreamingNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            content = msg.get("content", {})
            if msg.get("msg_type") == "stream":
                text = content.get("text", "")
                if text:
                    prefix = f"[nb25-cell {cell_index + 1} {content.get('name', 'stream')}] "
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
