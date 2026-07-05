"""Build and execute notebook 26: Wu 2003 headline banana posterior and EKF failure."""

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
    if "from pathlib import Path" in source and "POSTERIOR_PATH" in source:
        text = (
            "This code imports the numerical, plotting, and SBI libraries, defines paths and priors, "
            "loads the nb23 S-B summaries, and checks that the nb24 S-B posterior is available."
        )
    elif "def _safe_corr" in source and "simulate_sb_summary" in source:
        text = (
            "This code defines the same 66-D S-B summary function used for nb24 training, plus a "
            "deterministic simulator wrapper for local sensitivity calculations."
        )
    elif "def sample_sb_posterior" in source:
        text = (
            "This code loads the trained nb24 S-B posterior, samples W11, W12, and W15, and saves "
            "the posterior draws used by the headline banana figures."
        )
    elif "fig, axes = plt.subplots" in source and "26_sb_alpha_eta_banana" in source:
        text = (
            "This code plots the S-B joint alpha-eta_col posterior for the headline scenarios and "
            "computes simple shape diagnostics for the banana geometry."
        )
    elif "def finite_difference_jacobian" in source:
        text = (
            "This code builds a local linear-Gaussian EKF-style update in summary space by finite-differencing "
            "the Wu simulator and combining the resulting Jacobian with replicate measurement covariance."
        )
    elif "ekf_rows = []" in source:
        text = (
            "This code runs the local Gaussian baseline on W11, W12, and W15 replicates, computes interval "
            "coverage, and contrasts it with the sampled SBI posterior."
        )
    elif "fig, axes = plt.subplots" in source and "26_ekf_failure" in source:
        text = (
            "This code draws the nb26 paper-facing comparison: SBI curved uncertainty versus the local "
            "Gaussian EKF-style approximation and its coverage failure."
        )
    elif "metadata =" in source and "acceptance" in source:
        text = (
            "This code writes nb26 metadata and acceptance checks for generated tables, figures, posterior "
            "samples, and notebook evidence."
        )
    else:
        text = f"This code computes the next nb26 headline-experiment step starting with: `{compact[:120]}`."
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
        """# Notebook 26 -- Wu 2003 Headline Banana Posterior and EKF Failure

This notebook turns the nb24/nb25 training results into the headline Wu 2003
experiment. It focuses on the conventional **S-B** measurement structure because
that is where closed-loop masking and recycle snowballing are strongest.

The nb26 contract is the roadmap headline: **W12 banana posterior** and **W15
snowball EKF failure**. The generated scenario catalogue used earlier in the Wu
workflow drifted from that written roadmap, so this notebook constructs the two
headline observations directly from the roadmap parameter vectors instead of
reusing the catalogue labels.

Main outputs:

- `results/wu2003_nb26_headline_posterior_summary.csv`: SBI posterior shape and interval diagnostics.
- `results/wu2003_nb26_ekf_gaussian_failure.csv`: local Gaussian/EKF-style coverage diagnostics.
- `results/wu2003_nb26_acceptance.csv`: reproducibility and output checks.
- `figures/26_sb_alpha_eta_banana.png`: S-B alpha--eta_col headline posterior.
- `figures/26_ekf_failure_summary.png`: SBI versus Gaussian baseline failure summary.
"""
    ),
    md("""## 1. Imports, paths, priors, and scenario summaries"""),
    code(
        """from pathlib import Path
import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB,
    NOMINAL_INLET,
    extract_observations_explicit,
    simulate_trajectory_explicit,
)
from cstr_sbi.recycle.simulator import DEFAULT_SENSOR_NOISE_PCT, SB_INDICES, nominal_warm_start

pd.set_option("display.precision", 5)
torch.manual_seed(20260630)

DATA = Path("data")
RESULTS = Path("results")
FIGS = Path("figures")
for path in [DATA, RESULTS, FIGS]:
    path.mkdir(exist_ok=True)

SCENARIO_FEATURE_PATH = DATA / "wu2003_summary_features.npz"
POSTERIOR_PATH = RESULTS / "wu2003_nb24_sb_sbi_posterior_final.pkl"
TRAIN_BANK_PATH = DATA / "wu2003_sbi_train_sb.npz"
assert SCENARIO_FEATURE_PATH.exists(), "Run nb23 before nb26."
assert POSTERIOR_PATH.exists(), "Run nb24 before nb26."
assert TRAIN_BANK_PATH.exists(), "Run nb24 before nb26 so the S-B bank and scaler context exist."

PARAMETER_NAMES = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]
PRIOR_LOW = np.array([0.40, 0.40, 0.50, 0.40, 0.70], dtype=np.float32)
PRIOR_HIGH = np.array([1.20, 1.20, 1.00, 1.20, 0.95], dtype=np.float32)
PRIOR_WIDTH = PRIOR_HIGH - PRIOR_LOW
PRIOR_COV = np.diag((PRIOR_WIDTH.astype(float) ** 2) / 12.0)
POSTERIOR_SAMPLES = int(os.environ.get("WU2003_NB26_POSTERIOR_SAMPLES", "8000"))
N_HEADLINE_REPLICATES = int(os.environ.get("WU2003_NB26_N_REPLICATES", "30"))
HEADLINE_SCENARIOS = [
    {
        "scenario_id": 12,
        "scenario_name": "W12_snowball_compound_plan",
        "description": "Roadmap W12: alpha and eta_col jointly degraded; target banana posterior under S-B.",
        "theta": np.array([0.75, 1.00, 0.80, 1.00, 0.90], dtype=np.float32),
    },
    {
        "scenario_id": 15,
        "scenario_name": "W15_snowball_threshold_plan",
        "description": "Roadmap W15: near snowball tipping point; target EKF-style overconfidence failure.",
        "theta": np.array([0.58, 1.00, 0.90, 1.00, 0.90], dtype=np.float32),
    },
]
TARGET_SCENARIOS = [item["scenario_id"] for item in HEADLINE_SCENARIOS]


def log_status(message):
    print(f"[nb26 {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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

catalogue_truth = labels_eval[["scenario_id", "scenario_name", *PARAMETER_NAMES]].drop_duplicates().reset_index(drop=True)
headline_truth = pd.DataFrame([
    {"scenario_id": item["scenario_id"], "scenario_name": item["scenario_name"], **dict(zip(PARAMETER_NAMES, item["theta"]))}
    for item in HEADLINE_SCENARIOS
])
display(headline_truth)
log_status(f"Loaded S-B summaries: X_eval={X_eval.shape}, features={len(features_sb)}")
log_status(f"Roadmap headline scenarios: {TARGET_SCENARIOS}; posterior samples per scenario={POSTERIOR_SAMPLES}")
log_status(f"Noisy replicates per headline scenario for Gaussian/EKF-style coverage: {N_HEADLINE_REPLICATES}")
"""
    ),
    md("""## 2. S-B summary simulator for local sensitivities"""),
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


def simulate_sb_summary(theta, y0):
    ts, ys = simulate_trajectory_explicit(theta, NOMINAL_INLET, NOMINAL_CTRL_SB, y0, t_final=2.0, n_save=120)
    raw = np.asarray(extract_observations_explicit(ys, theta, NOMINAL_CTRL_SB))
    X, names = summarize_windows(raw[None, :, SB_INDICES], SB_CHANNELS, np.asarray(ts))
    return X[0], names


def simulate_sb_replicate_summaries(theta, y0, n_replicates, seed):
    rng = np.random.default_rng(seed)
    ts, ys = simulate_trajectory_explicit(theta, NOMINAL_INLET, NOMINAL_CTRL_SB, y0, t_final=2.0, n_save=120)
    raw = np.asarray(extract_observations_explicit(ys, theta, NOMINAL_CTRL_SB))[:, SB_INDICES]
    scale = np.maximum(np.max(np.abs(raw), axis=0, keepdims=True), 1e-12)
    noise = rng.normal(0.0, DEFAULT_SENSOR_NOISE_PCT * scale, size=(n_replicates, *raw.shape))
    noisy = raw[None, :, :] + noise
    X_reps, names = summarize_windows(noisy, SB_CHANNELS, np.asarray(ts))
    X_det, _ = summarize_windows(raw[None, :, :], SB_CHANNELS, np.asarray(ts))
    return X_det[0], X_reps, names


y0_sb = nominal_warm_start("S-B")
test_summary, test_names = simulate_sb_summary(np.array([1.0, 1.0, 1.0, 1.0, 0.90], dtype=np.float32), y0_sb)
assert test_summary.shape == (66,)
assert len(test_names) == len(features_sb) == 66
log_status("S-B deterministic summary simulator is ready")

headline_cases = {}
for item in HEADLINE_SCENARIOS:
    x_det, x_reps, names = simulate_sb_replicate_summaries(
        item["theta"], y0_sb, N_HEADLINE_REPLICATES, seed=20260630 + item["scenario_id"]
    )
    headline_cases[item["scenario_id"]] = {
        "name": item["scenario_name"],
        "description": item["description"],
        "truth": item["theta"],
        "x_obs": x_det,
        "X_reps": x_reps,
    }
    log_status(f"Built roadmap {item['scenario_name']}: x_obs={x_det.shape}, reps={x_reps.shape}")
"""
    ),
    md("""## 3. Sample the nb24 S-B posterior for headline scenarios"""),
    code(
        """def sample_sb_posterior(posterior, scaler, x, n_samples, seed):
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(scaler.transform(np.asarray(x, dtype=np.float32)[None, :])[0], dtype=torch.float32)
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_tensor, show_progress_bars=False, reject_outside_prior=False)
    return samples.cpu().numpy().astype(np.float32)

def out_of_prior_fraction(samples):
    samples = np.asarray(samples)
    outside = (samples < PRIOR_LOW[None, :]) | (samples > PRIOR_HIGH[None, :])
    return outside.mean(axis=0), outside.any(axis=1).mean()

with open(POSTERIOR_PATH, "rb") as f:
    payload = pickle.load(f)
posterior_sb = payload["posterior"]
scaler_sb = payload["scaler"]
meta_sb = payload["metadata"]
log_status(f"Loaded nb24 S-B posterior metadata: {meta_sb}")

posterior_cases = {}
posterior_rows = []
for sid in TARGET_SCENARIOS:
    x_obs = headline_cases[sid]["x_obs"]
    truth = headline_cases[sid]["truth"]
    name = headline_cases[sid]["name"]
    samples = sample_sb_posterior(posterior_sb, scaler_sb, x_obs, POSTERIOR_SAMPLES, seed=20260630 + sid)
    outside_by_parameter, outside_any = out_of_prior_fraction(samples)
    log_status(f"Sampled {name}: any_outside_prior={outside_any:.3f}, by_parameter={outside_by_parameter.round(3).tolist()}")
    posterior_cases[sid] = {"name": name, "truth": truth, "samples": samples, "x_obs": x_obs, "X_reps": headline_cases[sid]["X_reps"], "outside_any": outside_any}
    for j, parameter in enumerate(PARAMETER_NAMES):
        vals = samples[:, j]
        q05, q50, q95 = np.percentile(vals, [5, 50, 95])
        posterior_rows.append({
            "scenario_id": sid,
            "scenario_name": name,
            "parameter": parameter,
            "truth": float(truth[j]),
            "mean": float(vals.mean()),
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "width90": float(q95 - q05),
            "covered90": bool(q05 <= truth[j] <= q95),
            "prior_width_fraction": float((q95 - q05) / PRIOR_WIDTH[j]),
            "out_of_prior_fraction": float(outside_by_parameter[j]),
        })

posterior_summary = pd.DataFrame(posterior_rows)
posterior_summary.to_csv(RESULTS / "wu2003_nb26_headline_posterior_summary.csv", index=False)
np.savez_compressed(
    RESULTS / "wu2003_nb26_headline_posterior_samples.npz",
    scenario_ids=np.asarray(TARGET_SCENARIOS),
    parameter_names=np.asarray(PARAMETER_NAMES, dtype=object),
    samples=np.stack([posterior_cases[sid]["samples"] for sid in TARGET_SCENARIOS]),
    truths=np.stack([posterior_cases[sid]["truth"] for sid in TARGET_SCENARIOS]),
    scenario_names=np.asarray([posterior_cases[sid]["name"] for sid in TARGET_SCENARIOS], dtype=object),
)
display(posterior_summary.pivot(index=["scenario_id", "scenario_name"], columns="parameter", values="prior_width_fraction").round(3))
"""
    ),
    md("""## 4. Alpha--eta_col posterior geometry"""),
    code(
        """shape_rows = []
fig, axes = plt.subplots(1, len(TARGET_SCENARIOS), figsize=(14, 4.2), constrained_layout=True)
for ax, sid in zip(axes, TARGET_SCENARIOS):
    item = posterior_cases[sid]
    samples = item["samples"]
    truth = item["truth"]
    alpha = samples[:, 0]
    eta = samples[:, 2]
    corr = float(np.corrcoef(alpha, eta)[0, 1])
    cov = np.cov(np.column_stack([alpha, eta]).T)
    eigvals = np.linalg.eigvalsh(cov)
    elongation = float(np.sqrt(eigvals[-1] / max(eigvals[0], 1e-12)))
    shape_rows.append({
        "scenario_id": sid,
        "scenario_name": item["name"],
        "alpha_eta_corr": corr,
        "alpha_eta_elongation": elongation,
        "alpha_width90": float(np.percentile(alpha, 95) - np.percentile(alpha, 5)),
        "eta_col_width90": float(np.percentile(eta, 95) - np.percentile(eta, 5)),
        "roadmap_note": "roadmap headline alpha-eta snowball case",
    })
    ax.hexbin(alpha, eta, gridsize=38, cmap="Blues", mincnt=1, linewidths=0.0)
    ax.scatter([truth[0]], [truth[2]], marker="x", color="black", s=70, linewidth=2, label="truth")
    ax.set_xlim(PRIOR_LOW[0], PRIOR_HIGH[0])
    ax.set_ylim(PRIOR_LOW[2], PRIOR_HIGH[2])
    ax.set_xlabel("alpha")
    ax.set_ylabel("eta_col")
    ax.set_title(f"{item['name']}\\nr={corr:.2f}, elong={elongation:.1f}")
    ax.grid(alpha=0.2)
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("nb26: S-B alpha--eta_col posterior geometry from nb24 posterior")
fig.savefig(FIGS / "26_sb_alpha_eta_banana.png", dpi=160, bbox_inches="tight")
plt.show()

shape_diagnostics = pd.DataFrame(shape_rows)
shape_diagnostics.to_csv(RESULTS / "wu2003_nb26_banana_shape_diagnostics.csv", index=False)
display(shape_diagnostics.round(4))
"""
    ),
    md("""## 5. Local Gaussian/EKF-style baseline in summary space"""),
    code(
        """def finite_difference_jacobian(theta_center, y0, scaler, relative_step=0.015):
    theta_center = np.asarray(theta_center, dtype=np.float32)
    x0, _ = simulate_sb_summary(theta_center, y0)
    x0_scaled = scaler.transform(x0[None, :])[0]
    H = np.zeros((len(x0_scaled), len(theta_center)), dtype=float)
    for j in range(len(theta_center)):
        step = max(float(PRIOR_WIDTH[j] * relative_step), 1e-4)
        lo = max(float(PRIOR_LOW[j]), float(theta_center[j] - step))
        hi = min(float(PRIOR_HIGH[j]), float(theta_center[j] + step))
        if hi <= lo:
            continue
        theta_lo = theta_center.copy(); theta_lo[j] = lo
        theta_hi = theta_center.copy(); theta_hi[j] = hi
        x_lo, _ = simulate_sb_summary(theta_lo, y0)
        x_hi, _ = simulate_sb_summary(theta_hi, y0)
        H[:, j] = (scaler.transform(x_hi[None, :])[0] - scaler.transform(x_lo[None, :])[0]) / (hi - lo)
    return x0_scaled, H


def local_gaussian_update(x_obs, x0_scaled, H, R_diag, theta_center):
    x_scaled = scaler_sb.transform(np.asarray(x_obs, dtype=np.float32)[None, :])[0]
    residual = x_scaled - x0_scaled
    R_inv = 1.0 / np.maximum(R_diag, 1e-8)
    prior_inv = np.linalg.inv(PRIOR_COV)
    precision = prior_inv + H.T @ (R_inv[:, None] * H)
    cov = np.linalg.pinv(precision)
    gain_rhs = H.T @ (R_inv * residual)
    mean = np.asarray(theta_center, dtype=float) + cov @ gain_rhs
    mean = np.clip(mean, PRIOR_LOW, PRIOR_HIGH)
    return mean, cov


local_models = {}
for sid in TARGET_SCENARIOS:
    truth = headline_cases[sid]["truth"]
    X_rep_scaled = scaler_sb.transform(headline_cases[sid]["X_reps"])
    R_diag = np.var(X_rep_scaled, axis=0, ddof=1) + 1e-5
    x0_scaled, H = finite_difference_jacobian(truth, y0_sb, scaler_sb)
    local_models[sid] = {"truth": truth, "x0_scaled": x0_scaled, "H": H, "R_diag": R_diag}
    log_status(f"Built local Gaussian model for scenario {sid}: H={H.shape}, median R={np.median(R_diag):.3e}")
"""
    ),
    md("""## 6. Coverage failure versus SBI"""),
    code(
        """ekf_rows = []
for sid in TARGET_SCENARIOS:
    model = local_models[sid]
    for replicate_index, x_obs in enumerate(posterior_cases[sid]["X_reps"]):
        mean, cov = local_gaussian_update(x_obs, model["x0_scaled"], model["H"], model["R_diag"], model["truth"])
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        q05 = np.maximum(PRIOR_LOW, mean - 1.64485 * std)
        q95 = np.minimum(PRIOR_HIGH, mean + 1.64485 * std)
        for j, parameter in enumerate(PARAMETER_NAMES):
            ekf_rows.append({
                "scenario_id": sid,
                "scenario_name": posterior_cases[sid]["name"],
                "replicate": replicate_index,
                "parameter": parameter,
                "truth": float(model["truth"][j]),
                "mean_gaussian": float(mean[j]),
                "q05_gaussian": float(q05[j]),
                "q95_gaussian": float(q95[j]),
                "width90_gaussian": float(q95[j] - q05[j]),
                "covered90_gaussian": bool(q05[j] <= model["truth"][j] <= q95[j]),
            })

ekf_detail = pd.DataFrame(ekf_rows)
ekf_metrics = (
    ekf_detail.groupby(["scenario_id", "scenario_name", "parameter"], as_index=False)
    .agg(
        gaussian_coverage90=("covered90_gaussian", "mean"),
        gaussian_mean_width90=("width90_gaussian", "mean"),
        gaussian_mae=("mean_gaussian", lambda s: float(np.mean(np.abs(s - ekf_detail.loc[s.index, "truth"])))),
    )
)
sbi_metrics = posterior_summary[["scenario_id", "scenario_name", "parameter", "covered90", "width90"]].rename(
    columns={"covered90": "sbi_covered90", "width90": "sbi_width90"}
)
comparison = ekf_metrics.merge(sbi_metrics, on=["scenario_id", "scenario_name", "parameter"], how="left")
comparison["gaussian_width_prior_frac"] = comparison["gaussian_mean_width90"] / comparison["parameter"].map(dict(zip(PARAMETER_NAMES, PRIOR_WIDTH)))
comparison["sbi_width_prior_frac"] = comparison["sbi_width90"] / comparison["parameter"].map(dict(zip(PARAMETER_NAMES, PRIOR_WIDTH)))
comparison.to_csv(RESULTS / "wu2003_nb26_ekf_gaussian_failure.csv", index=False)
ekf_detail.to_csv(RESULTS / "wu2003_nb26_ekf_gaussian_detail.csv", index=False)
display(comparison[comparison["parameter"].isin(["alpha", "eta_col"])].round(4))
"""
    ),
    md("""## 7. Headline failure figures"""),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
sub = comparison[comparison["parameter"].isin(["alpha", "eta_col"])]
labels = [f"W{int(r.scenario_id)}\\n{r.parameter}" for r in sub.itertuples()]
x = np.arange(len(labels))
width = 0.36
axes[0].bar(x - width / 2, sub["sbi_width_prior_frac"], width=width, label="SBI posterior", color="#4C78A8", alpha=0.85)
axes[0].bar(x + width / 2, sub["gaussian_width_prior_frac"], width=width, label="local Gaussian", color="#E45756", alpha=0.85)
axes[0].set_ylabel("90% interval width / prior width")
axes[0].set_title("uncertainty width")
axes[0].set_xticks(x, labels, rotation=0)
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend(fontsize=8)

axes[1].bar(x - width / 2, sub["sbi_covered90"].astype(float), width=width, label="SBI scenario truth", color="#4C78A8", alpha=0.85)
axes[1].bar(x + width / 2, sub["gaussian_coverage90"], width=width, label="local Gaussian replicates", color="#E45756", alpha=0.85)
axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1)
axes[1].axhline(0.65, color="black", linestyle=":", linewidth=1)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("90% interval coverage")
axes[1].set_title("coverage failure")
axes[1].set_xticks(x, labels, rotation=0)
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend(fontsize=8)
fig.suptitle("nb26: S-B curved posterior versus local Gaussian/EKF-style approximation")
fig.savefig(FIGS / "26_ekf_failure_summary.png", dpi=160, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 8. Persist metadata and acceptance checks"""),
    code(
        """metadata = {
    "notebook": "26_wu2003_headline_banana_and_ekf_failure.ipynb",
    "posterior_path": str(POSTERIOR_PATH),
    "posterior_metadata": meta_sb,
    "target_scenarios": TARGET_SCENARIOS,
    "posterior_samples": POSTERIOR_SAMPLES,
    "note": "W12 and W15 are generated directly from the roadmap headline parameter vectors, bypassing the older catalogue drift.",
    "outputs": [
        "wu2003_nb26_headline_posterior_summary.csv",
        "wu2003_nb26_headline_posterior_samples.npz",
        "wu2003_nb26_banana_shape_diagnostics.csv",
        "wu2003_nb26_ekf_gaussian_failure.csv",
        "wu2003_nb26_ekf_gaussian_detail.csv",
        "26_sb_alpha_eta_banana.png",
        "26_ekf_failure_summary.png",
    ],
}
metadata_path = RESULTS / "wu2003_nb26_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2))

expected_files = [
    RESULTS / "wu2003_nb26_headline_posterior_summary.csv",
    RESULTS / "wu2003_nb26_headline_posterior_samples.npz",
    RESULTS / "wu2003_nb26_banana_shape_diagnostics.csv",
    RESULTS / "wu2003_nb26_ekf_gaussian_failure.csv",
    RESULTS / "wu2003_nb26_ekf_gaussian_detail.csv",
    metadata_path,
    FIGS / "26_sb_alpha_eta_banana.png",
    FIGS / "26_ekf_failure_summary.png",
]

alpha_eta = comparison[comparison["parameter"].isin(["alpha", "eta_col"])]
acceptance = pd.DataFrame([
    {"check": "nb24 S-B posterior loaded", "observed": str(meta_sb), "status": "PASS" if meta_sb.get("n_simulations") == 15000 else "WARN"},
    {"check": "roadmap headline scenarios sampled", "observed": str(sorted(posterior_cases)), "status": "PASS" if sorted(posterior_cases) == [12, 15] else "FAIL"},
    {"check": "W12 roadmap banana theta used", "observed": str(posterior_cases[12]["truth"].tolist()), "status": "PASS" if np.allclose(posterior_cases[12]["truth"], [0.75, 1.0, 0.80, 1.0, 0.90]) else "FAIL"},
    {"check": "W15 roadmap threshold theta used", "observed": str(posterior_cases[15]["truth"].tolist()), "status": "PASS" if np.allclose(posterior_cases[15]["truth"], [0.58, 1.0, 0.90, 1.0, 0.90]) else "FAIL"},
    {"check": "local Gaussian diagnostics produced", "observed": str(comparison.shape), "status": "PASS" if comparison.shape[0] == 10 else "FAIL"},
    {"check": "sub-65 Gaussian coverage checked", "observed": str(float(alpha_eta["gaussian_coverage90"].min())), "status": "PASS" if float(alpha_eta["gaussian_coverage90"].min()) < 0.65 else "WARN"},
    {"check": "all output files exist", "observed": str(all(path.exists() for path in expected_files)), "status": "PASS" if all(path.exists() for path in expected_files) else "FAIL"},
])
acceptance.to_csv(RESULTS / "wu2003_nb26_acceptance.csv", index=False)
display(acceptance)
"""
    ),
    md(
        """## 9. Interpretation

nb26 is the paper-facing counterpart to nb24 and nb25, but the exact roadmap
stress test is not automatically confirmed by the current trained posterior. The
notebook intentionally bypasses the older generated scenario catalogue and builds
W12/W15 from the written roadmap parameter vectors. That makes this a direct test
of the selected plan row rather than a relabelled catalogue case.

The result should be read critically. For the exact roadmap W12/W15 observations,
the nb24 S-B posterior has to be sampled with prior rejection disabled; otherwise
sampling stalls because the observations sit outside the accepted proposal region.
The saved posterior summary reports this through `out_of_prior_fraction`. This is
evidence that the current nb24 training/data contract and the written roadmap
headline scenarios are not yet aligned.

The alpha--eta_col posterior shapes are narrow rather than banana-shaped, and the
local Gaussian/EKF-style coverage diagnostic does not reach the planned <65%
failure threshold. Therefore nb26, in its roadmap-aligned form, is a falsifying
diagnostic: the selected headline claim needs either an aligned scenario catalogue,
a new S-B posterior trained on the roadmap scenario regime, or a full time-domain
augmented EKF before it can be used as Figures 10 and 12.
"""
    ),
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    nb = new_notebook(cells=add_code_explanations(CELLS), metadata={"language_info": {"name": "python"}})
    out_path = root / "notebooks" / "26_wu2003_headline_banana_and_ekf_failure.ipynb"
    nbformat.write(nb, out_path)

    def _cell_label(cell) -> str:
        first = cell.source.strip().splitlines()[0] if cell.source.strip() else "empty cell"
        return first[:100]

    def _on_cell_start(cell, cell_index, **kwargs):
        print(f"[nb26-run] start cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    def _on_cell_complete(cell, cell_index, **kwargs):
        print(f"[nb26-run] done  cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    def _on_cell_error(cell, cell_index, execute_reply, **kwargs):
        print(f"[nb26-run] ERROR cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    class StreamingNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            content = msg.get("content", {})
            if msg.get("msg_type") == "stream":
                text = content.get("text", "")
                if text:
                    prefix = f"[nb26-cell {cell_index + 1} {content.get('name', 'stream')}] "
                    for line in text.rstrip().splitlines():
                        print(prefix + line, flush=True)
            return super().process_message(msg, cell, cell_index)

    client = StreamingNotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        on_cell_start=_on_cell_start,
        on_cell_complete=_on_cell_complete,
        on_cell_error=_on_cell_error,
    )
    client.execute()
    nbformat.write(nb, out_path)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())