"""Build and execute notebook 24b: Wu 2003 posterior calibration repair."""

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
    if "from __future__ import annotations" in source and "N_CALIBRATION" in source:
        text = (
            "This code imports libraries, defines paths, parameter names, prior bounds, and the "
            "calibration sample-count controls for nb24b."
        )
    elif "def _safe_corr" in source and "simulate_sb_summary" in source:
        text = (
            "This code reconstructs nb24's S-B prior-to-summary simulator so nb24b can generate an "
            "independent calibration bank with the same 66-D feature contract."
        )
    elif "def generate_calibration_bank" in source:
        text = (
            "This code creates or loads a fresh prior-predictive calibration bank that is independent "
            "of the nb24 SNPE training bank."
        )
    elif "def sample_posterior_for_x" in source and "posterior_sample_rows" in source:
        text = (
            "This code samples the raw nb24 posterior for each held-out calibration case and stores "
            "posterior draws for downstream rank and coverage checks."
        )
    elif "def interval_metrics" in source and "raw_metrics" in source:
        text = (
            "This code computes raw held-out calibration metrics: posterior mean error, 90% coverage, "
            "interval width, and SBC-style rank uniformity."
        )
    elif "scale_grid" in source and "calibration_factors" in source:
        text = (
            "This code fits simple per-parameter interval inflation factors on the calibration split "
            "and evaluates them on the held-out test split, with conservative minimum inflation for "
            "the eta/xi parameters flagged by nb24a."
        )
    elif "scenario_rows = []" in source and "scenario_comparison" in source:
        text = (
            "This code applies the learned calibration factors to the original nb24 closed-loop scenario "
            "set and compares raw versus calibrated scenario coverage."
        )
    elif "fig, axes = plt.subplots" in source and "24b_calibration_repair" in source:
        text = (
            "This code plots the before/after calibration repair summary and saves the nb24b figure."
        )
    elif "acceptance = pd.DataFrame" in source:
        text = (
            "This code writes the nb24b acceptance table, calibration metadata, and output-file checks."
        )
    else:
        text = f"This code computes the next nb24b calibration-repair step starting with: `{compact[:120]}`."
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
        """# Notebook 24b -- Calibrating the Wu S-B eta/xi posteriors

nb24a showed that `eta_col` and `xi_reb` have excellent posterior mean recovery
but poor uncertainty calibration: their intervals are too narrow or displaced for
the advertised 90% coverage. This notebook tests a direct repair.

The repair is deliberately modest. It does not retrain the SNPE density and it
does not reinterpret poor coverage as new physical information. Instead, it uses
an independent prior-predictive calibration bank to learn per-parameter interval
inflation factors, then checks whether the raw nb24 posterior can be made honest
about its uncertainty.

The notebook outputs:

- `data/wu2003_nb24b_calibration_bank_sb.npz`: independent S-B calibration bank.
- `results/wu2003_nb24b_raw_calibration_metrics.csv`: raw held-out metrics.
- `results/wu2003_nb24b_calibration_factors.csv`: learned interval scales.
- `results/wu2003_nb24b_repaired_calibration_metrics.csv`: repaired held-out metrics.
- `results/wu2003_nb24b_scenario_repair.csv`: nb24 scenario before/after table.
- `figures/24b_calibration_repair.png`: raw vs repaired calibration summary.
"""
    ),
    md("""## 1. Imports, paths, and controls"""),
    code(
        """from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB,
    NOMINAL_INLET,
    extract_observations_explicit,
    simulate_trajectory_explicit,
)
from cstr_sbi.recycle.simulator import SB_INDICES, nominal_warm_start

warnings.filterwarnings("ignore", category=UserWarning)

DATA = Path("data")
RESULTS = Path("results")
FIGS = Path("figures")
for path in [DATA, RESULTS, FIGS]:
    path.mkdir(exist_ok=True)

PARAMETER_NAMES = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]
PRIOR_LOW = np.array([0.40, 0.40, 0.50, 0.40, 0.70], dtype=np.float32)
PRIOR_HIGH = np.array([1.20, 1.20, 1.00, 1.20, 0.95], dtype=np.float32)
PRIOR_WIDTH = PRIOR_HIGH - PRIOR_LOW

N_CALIBRATION = int(os.environ.get("WU2003_NB24B_N_CALIBRATION", "240"))
POSTERIOR_SAMPLES = int(os.environ.get("WU2003_NB24B_POSTERIOR_SAMPLES", "500"))
PROGRESS_INTERVAL = int(os.environ.get("WU2003_NB24B_PROGRESS_INTERVAL", "40"))
CALIBRATION_BANK_PATH = DATA / "wu2003_nb24b_calibration_bank_sb.npz"


def log_status(message):
    print(f"[nb24b {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


print("nb24b calibration cases:", N_CALIBRATION)
print("nb24b posterior samples per case:", POSTERIOR_SAMPLES)
"""
    ),
    md("""## 2. Load the trained nb24 posterior and evaluation summaries"""),
    code(
        """with open(RESULTS / "wu2003_nb24_sb_sbi_posterior_final.pkl", "rb") as f:
    posterior_payload = pickle.load(f)
posterior = posterior_payload["posterior"]
scaler = posterior_payload["scaler"]
posterior_metadata = posterior_payload["metadata"]

with np.load(DATA / "wu2003_sbi_train_sb.npz", allow_pickle=True) as bank:
    train_feature_names = [str(x) for x in bank["feature_names"]]

with np.load(DATA / "wu2003_summary_features.npz", allow_pickle=True) as data:
    X_sb_all = data["X_sb"].astype(np.float32)
    labels_all = pd.DataFrame.from_records(data["labels"])

closed_mask = labels_all["mode"].eq("closed_loop").to_numpy()
X_eval = X_sb_all[closed_mask]
labels_eval = labels_all.loc[closed_mask].reset_index(drop=True)
theta_eval = labels_eval[PARAMETER_NAMES].to_numpy(dtype=np.float32)
scenario_id = labels_eval["scenario_id"].to_numpy()
scenario_name = labels_eval["scenario_name"].to_numpy()

print("loaded posterior metadata:", posterior_metadata)
print("nb24 feature count:", len(train_feature_names))
print("closed-loop scenario summaries:", X_eval.shape)
assert len(train_feature_names) == 66
"""
    ),
    md("""## 3. Reconstruct the nb24 S-B summary simulator"""),
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


SB_CHANNELS = ["T_r", "T_j", "Q_j", "T_reb", "Q_reb", "F_R_norm", "F_B_norm", "R_norm", "V_norm"]


def summarize_windows(windows, channels, t):
    feature_names = []
    rows = []
    channel_index = {name: i for i, name in enumerate(channels)}
    final_start = int(np.floor(0.75 * len(t)))

    for window in windows:
        values = []
        for channel in channels:
            y = window[:, channel_index[channel]]
            values.extend([float(np.mean(y)), float(np.std(y)), _slope(t, y), float(np.min(y)), float(np.max(y)), float(np.mean(y[final_start:]))])
        rows.append(values)

    for channel in channels:
        feature_names.extend([f"{channel}__mean", f"{channel}__std", f"{channel}__slope", f"{channel}__min", f"{channel}__max", f"{channel}__final25_mean"])

    physics_names = [
        "UA_proxy_final", "recycle_ratio_final", "col_recovery_proxy_final", "reb_intensity_final",
        "reactor_conversion_proxy_final", "recycle_excess_final", "Tr_Tj_ratio_final", "Qj_slope",
        "corr_Qj_FR", "corr_Qreb_FR", "R_effort_final", "V_effort_final",
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


def noisy_sensor_layer(obs, rng, noise_pct=0.003):
    scale = np.maximum(np.max(np.abs(obs), axis=0, keepdims=True), 1e-12)
    return obs + rng.normal(0.0, noise_pct * scale, size=obs.shape)


def simulate_sb_summary(theta, y0, rng):
    ts, ys = simulate_trajectory_explicit(theta, NOMINAL_INLET, NOMINAL_CTRL_SB, y0, t_final=2.0, n_save=120)
    raw = np.asarray(extract_observations_explicit(ys, theta, NOMINAL_CTRL_SB))
    sb_obs = noisy_sensor_layer(raw[..., SB_INDICES], rng)
    X, names = summarize_windows(sb_obs[None, :, :], SB_CHANNELS, np.asarray(ts))
    return X[0], names


def sample_prior_numpy(n, rng):
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * rng.random((n, len(PARAMETER_NAMES)), dtype=np.float32)
"""
    ),
    md("""## 4. Generate an independent calibration bank"""),
    code(
        """def generate_calibration_bank(n_cases, path):
    log_status(f"Starting independent S-B calibration bank: target={n_cases:,}, path={path}")
    rng = np.random.default_rng(20260627)
    theta_cal = np.empty((n_cases, len(PARAMETER_NAMES)), dtype=np.float32)
    X_cal = np.empty((n_cases, len(train_feature_names)), dtype=np.float32)
    y0_sb = nominal_warm_start("S-B")
    accepted = 0
    attempted = 0
    rejected = 0
    feature_names_ref = None
    t0 = time.perf_counter()
    while accepted < n_cases:
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
        theta_cal[accepted] = theta_i
        X_cal[accepted] = summary_i
        accepted += 1
        if accepted % PROGRESS_INTERVAL == 0 or accepted == n_cases:
            elapsed = time.perf_counter() - t0
            log_status(
                f"calibration bank {accepted:>5}/{n_cases}; attempted={attempted}; "
                f"rejected={rejected}; elapsed={elapsed/60:.1f} min"
            )
    np.savez_compressed(
        path,
        theta=theta_cal,
        X=X_cal,
        feature_names=np.asarray(feature_names_ref, dtype=object),
        parameter_names=np.asarray(PARAMETER_NAMES, dtype=object),
        prior_low=PRIOR_LOW,
        prior_high=PRIOR_HIGH,
        n_cases=np.asarray(n_cases),
        n_attempted=np.asarray(attempted),
        n_rejected=np.asarray(rejected),
        wall_time_s=np.asarray(time.perf_counter() - t0),
    )
    log_status(f"Saved independent calibration bank: theta={theta_cal.shape}, X={X_cal.shape}")
    return theta_cal, X_cal


if CALIBRATION_BANK_PATH.exists():
    with np.load(CALIBRATION_BANK_PATH, allow_pickle=True) as bank:
        theta_cal = bank["theta"].astype(np.float32)
        X_cal = bank["X"].astype(np.float32)
    if theta_cal.shape[0] < N_CALIBRATION or X_cal.shape[1] != 66 or not np.isfinite(X_cal).all():
        log_status("Cached calibration bank is too small or invalid; regenerating")
        theta_cal, X_cal = generate_calibration_bank(N_CALIBRATION, CALIBRATION_BANK_PATH)
    else:
        theta_cal = theta_cal[:N_CALIBRATION]
        X_cal = X_cal[:N_CALIBRATION]
        log_status(f"Loaded cached independent calibration bank: theta={theta_cal.shape}, X={X_cal.shape}")
else:
    theta_cal, X_cal = generate_calibration_bank(N_CALIBRATION, CALIBRATION_BANK_PATH)

assert theta_cal.shape == (N_CALIBRATION, 5)
assert X_cal.shape == (N_CALIBRATION, 66)
"""
    ),
    md("""## 5. Sample raw posterior on held-out calibration cases"""),
    code(
        """def sample_posterior_for_x(x, n_samples=POSTERIOR_SAMPLES, seed=0):
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(scaler.transform(np.asarray(x, dtype=np.float32)[None, :])[0], dtype=torch.float32)
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_tensor, show_progress_bars=False)
    return samples.cpu().numpy().astype(np.float32)


posterior_samples = np.empty((len(theta_cal), POSTERIOR_SAMPLES, len(PARAMETER_NAMES)), dtype=np.float32)
posterior_sample_rows = []
log_status(f"Sampling raw posterior for {len(theta_cal)} independent calibration cases")
for i, x in enumerate(X_cal):
    if i % max(PROGRESS_INTERVAL, 1) == 0 or i == len(theta_cal) - 1:
        log_status(f"posterior sampling {i + 1}/{len(theta_cal)}")
    samples = sample_posterior_for_x(x, n_samples=POSTERIOR_SAMPLES, seed=410000 + i)
    posterior_samples[i] = samples
    for j, parameter in enumerate(PARAMETER_NAMES):
        q05, q50, q95 = np.percentile(samples[:, j], [5, 50, 95])
        posterior_sample_rows.append({
            "case": i,
            "parameter": parameter,
            "truth": float(theta_cal[i, j]),
            "mean": float(samples[:, j].mean()),
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "rank": float(np.mean(samples[:, j] < theta_cal[i, j])),
        })

posterior_case_summary = pd.DataFrame(posterior_sample_rows)
posterior_case_summary.to_csv(RESULTS / "wu2003_nb24b_raw_case_summary.csv", index=False)
display(posterior_case_summary.groupby("parameter")[["truth", "mean", "rank"]].mean().round(4))
"""
    ),
    md("""## 6. Raw independent calibration metrics"""),
    code(
        """def interval_metrics(samples, truths, mask, label):
    rows = []
    for j, parameter in enumerate(PARAMETER_NAMES):
        vals = samples[mask, :, j]
        truth = truths[mask, j]
        q05 = np.percentile(vals, 5, axis=1)
        q95 = np.percentile(vals, 95, axis=1)
        means = vals.mean(axis=1)
        ranks = np.mean(vals < truth[:, None], axis=1)
        ks = kstest(ranks, "uniform")
        rows.append({
            "split": label,
            "parameter": parameter,
            "mae": float(np.mean(np.abs(means - truth))),
            "bias": float(np.mean(means - truth)),
            "coverage90": float(np.mean((q05 <= truth) & (truth <= q95))),
            "mean_width90": float(np.mean(q95 - q05)),
            "mean_width90_prior_frac": float(np.mean((q95 - q05) / PRIOR_WIDTH[j])),
            "sbc_rank_ks_p": float(ks.pvalue),
        })
    return pd.DataFrame(rows)


rng = np.random.default_rng(20260627)
perm = rng.permutation(len(theta_cal))
fit_mask = np.zeros(len(theta_cal), dtype=bool)
fit_mask[perm[: len(theta_cal) // 2]] = True
test_mask = ~fit_mask

raw_metrics = pd.concat([
    interval_metrics(posterior_samples, theta_cal, fit_mask, "calibration_fit_raw"),
    interval_metrics(posterior_samples, theta_cal, test_mask, "heldout_test_raw"),
], ignore_index=True)
raw_metrics.to_csv(RESULTS / "wu2003_nb24b_raw_calibration_metrics.csv", index=False)
display(raw_metrics.round(5))
"""
    ),
    md("""## 7. Fit and evaluate posterior interval calibration"""),
    code(
        """def apply_centered_scale(samples, scales):
    means = samples.mean(axis=1, keepdims=True)
    repaired = means + (samples - means) * scales[None, None, :]
    return np.clip(repaired, PRIOR_LOW[None, None, :], PRIOR_HIGH[None, None, :]).astype(np.float32)


def coverage_for_scale(samples, truths, mask, parameter_index, scale):
    vals = samples[mask, :, parameter_index]
    truth = truths[mask, parameter_index]
    means = vals.mean(axis=1, keepdims=True)
    repaired = np.clip(means + (vals - means) * scale, PRIOR_LOW[parameter_index], PRIOR_HIGH[parameter_index])
    q05 = np.percentile(repaired, 5, axis=1)
    q95 = np.percentile(repaired, 95, axis=1)
    return float(np.mean((q05 <= truth) & (truth <= q95)))


scale_grid = np.concatenate([np.linspace(1.0, 3.0, 21), np.linspace(3.25, 10.0, 28)])
minimum_repair_scales = {
    "eta_col": 1.5,
    "xi_reb": 3.0,
}
scale_rows = []
best_scales = np.ones(len(PARAMETER_NAMES), dtype=np.float32)
for j, parameter in enumerate(PARAMETER_NAMES):
    candidates = []
    for scale in scale_grid:
        coverage = coverage_for_scale(posterior_samples, theta_cal, fit_mask, j, float(scale))
        candidates.append((abs(coverage - 0.90), -coverage, float(scale), coverage))
    _, _, best_scale, _ = sorted(candidates)[0]
    best_scale = max(best_scale, minimum_repair_scales.get(parameter, 1.0))
    best_coverage = coverage_for_scale(posterior_samples, theta_cal, fit_mask, j, best_scale)
    best_scales[j] = best_scale
    scale_rows.append({
        "parameter": parameter,
        "interval_scale": float(best_scale),
        "minimum_repair_scale": float(minimum_repair_scales.get(parameter, 1.0)),
        "fit_coverage90": float(best_coverage),
        "target_coverage90": 0.90,
    })

calibration_factors = pd.DataFrame(scale_rows)
calibration_factors.to_csv(RESULTS / "wu2003_nb24b_calibration_factors.csv", index=False)

repaired_samples = apply_centered_scale(posterior_samples, best_scales)
repaired_metrics = pd.concat([
    interval_metrics(repaired_samples, theta_cal, fit_mask, "calibration_fit_repaired"),
    interval_metrics(repaired_samples, theta_cal, test_mask, "heldout_test_repaired"),
], ignore_index=True)
repaired_metrics.to_csv(RESULTS / "wu2003_nb24b_repaired_calibration_metrics.csv", index=False)

comparison = pd.concat([raw_metrics, repaired_metrics], ignore_index=True)
comparison.to_csv(RESULTS / "wu2003_nb24b_raw_vs_repaired_metrics.csv", index=False)
display(calibration_factors.round(4))
display(comparison[comparison["split"].str.contains("heldout")].round(5))
"""
    ),
    md("""## 8. Apply calibration factors to nb24 scenario posteriors"""),
    code(
        """def pick_eval(sid, replicate=0):
    matches = np.where(scenario_id == sid)[0]
    idx = int(matches[min(replicate, len(matches) - 1)])
    return X_eval[idx], theta_eval[idx], scenario_name[idx]


scenario_rows = []
for sid in sorted(np.unique(scenario_id)):
    x_obs, truth, name = pick_eval(int(sid))
    raw = sample_posterior_for_x(x_obs, n_samples=POSTERIOR_SAMPLES, seed=510000 + int(sid))
    repaired = apply_centered_scale(raw[None, :, :], best_scales)[0]
    for label, samples in [("raw", raw), ("repaired", repaired)]:
        for j, parameter in enumerate(PARAMETER_NAMES):
            q05, q50, q95 = np.percentile(samples[:, j], [5, 50, 95])
            scenario_rows.append({
                "scenario_id": int(sid),
                "scenario_name": name,
                "posterior": label,
                "parameter": parameter,
                "truth": float(truth[j]),
                "mean": float(samples[:, j].mean()),
                "q05": float(q05),
                "q50": float(q50),
                "q95": float(q95),
                "covered90": bool(q05 <= truth[j] <= q95),
                "width90": float(q95 - q05),
                "width90_prior_frac": float((q95 - q05) / PRIOR_WIDTH[j]),
            })

scenario_comparison = pd.DataFrame(scenario_rows)
scenario_summary = scenario_comparison.groupby(["posterior", "parameter"], as_index=False).agg(
    coverage90=("covered90", "mean"),
    mean_width90=("width90", "mean"),
    mean_width90_prior_frac=("width90_prior_frac", "mean"),
    mean_abs_error=("mean", lambda s: np.nan),
)
# Compute mean absolute error explicitly because groupby aggregation above only sees the mean column.
mae_rows = []
for (posterior_label, parameter), sub in scenario_comparison.groupby(["posterior", "parameter"]):
    mae_rows.append({"posterior": posterior_label, "parameter": parameter, "mean_abs_error": float(np.mean(np.abs(sub["mean"] - sub["truth"])))})
mae_table = pd.DataFrame(mae_rows)
scenario_summary = scenario_summary.drop(columns=["mean_abs_error"]).merge(mae_table, on=["posterior", "parameter"], how="left")
scenario_comparison.to_csv(RESULTS / "wu2003_nb24b_scenario_repair.csv", index=False)
scenario_summary.to_csv(RESULTS / "wu2003_nb24b_scenario_repair_summary.csv", index=False)
display(scenario_summary.round(5))
"""
    ),
    md("""## 9. Before/after calibration figure"""),
    code(
        """heldout = comparison[comparison["split"].isin(["heldout_test_raw", "heldout_test_repaired"])].copy()
heldout["method"] = heldout["split"].map({"heldout_test_raw": "raw", "heldout_test_repaired": "repaired"})
scenario_plot = scenario_summary.copy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
width = 0.36
x = np.arange(len(PARAMETER_NAMES))
colors = {"raw": "#4C78A8", "repaired": "#E45756"}
for offset, method in [(-width / 2, "raw"), (width / 2, "repaired")]:
    sub = heldout[heldout["method"].eq(method)].set_index("parameter").loc[PARAMETER_NAMES]
    axes[0].bar(x + offset, sub["coverage90"], width=width, label=method, color=colors[method], alpha=0.85)
    axes[1].bar(x + offset, -np.log10(np.maximum(sub["sbc_rank_ks_p"], 1e-12)), width=width, label=method, color=colors[method], alpha=0.85)

for offset, method in [(-width / 2, "raw"), (width / 2, "repaired")]:
    sub = scenario_plot[scenario_plot["posterior"].eq(method)].set_index("parameter").loc[PARAMETER_NAMES]
    axes[2].bar(x + offset, sub["coverage90"], width=width, label=method, color=colors[method], alpha=0.85)

axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1)
axes[0].set_title("Held-out 90% coverage")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("coverage")
axes[1].axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)
axes[1].axhline(-np.log10(0.01), color="gray", linestyle=":", linewidth=1)
axes[1].set_title("Held-out SBC rank test")
axes[1].set_ylabel("-log10 KS p")
axes[2].axhline(0.90, color="black", linestyle="--", linewidth=1)
axes[2].set_title("nb24 scenario 90% coverage")
axes[2].set_ylim(0, 1.05)
for ax in axes:
    ax.set_xticks(x, PARAMETER_NAMES, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("nb24b: posterior interval calibration repair for Wu S-B")
fig.savefig(FIGS / "24b_calibration_repair.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 10. Acceptance checks and saved outputs"""),
    code(
        """metadata = {
    "n_calibration": int(N_CALIBRATION),
    "posterior_samples_per_case": int(POSTERIOR_SAMPLES),
    "calibration_bank_path": str(CALIBRATION_BANK_PATH),
    "posterior_cache": str(RESULTS / "wu2003_nb24_sb_sbi_posterior_final.pkl"),
    "calibration_method": "centered per-parameter interval inflation with prior clipping",
    "target_coverage90": 0.90,
}
with open(RESULTS / "wu2003_nb24b_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

expected_files = [
    DATA / "wu2003_nb24b_calibration_bank_sb.npz",
    RESULTS / "wu2003_nb24b_raw_case_summary.csv",
    RESULTS / "wu2003_nb24b_raw_calibration_metrics.csv",
    RESULTS / "wu2003_nb24b_calibration_factors.csv",
    RESULTS / "wu2003_nb24b_repaired_calibration_metrics.csv",
    RESULTS / "wu2003_nb24b_raw_vs_repaired_metrics.csv",
    RESULTS / "wu2003_nb24b_scenario_repair.csv",
    RESULTS / "wu2003_nb24b_scenario_repair_summary.csv",
    RESULTS / "wu2003_nb24b_metadata.json",
    FIGS / "24b_calibration_repair.png",
]
acceptance = pd.DataFrame([
    {"check": "independent calibration bank shape", "observed": str(theta_cal.shape), "status": "PASS" if theta_cal.shape == (N_CALIBRATION, 5) else "FAIL"},
    {"check": "posterior samples finite", "observed": str(np.isfinite(posterior_samples).all()), "status": "PASS" if np.isfinite(posterior_samples).all() else "FAIL"},
    {"check": "calibration factors finite", "observed": str(np.isfinite(best_scales).all()), "status": "PASS" if np.isfinite(best_scales).all() else "FAIL"},
    {"check": "all output files exist", "observed": str(all(path.exists() for path in expected_files)), "status": "PASS" if all(path.exists() for path in expected_files) else "FAIL"},
])
acceptance.to_csv(RESULTS / "wu2003_nb24b_acceptance.csv", index=False)
display(acceptance)
"""
    ),
    md(
        """## 11. How nb24 should be modified

nb24 should be modified, but the modification should be explicit rather than
silently replacing the posterior. nb24a showed the problem; nb24b shows the
repair strategy.

1. **Add a held-out calibration bank.** nb24 should generate an independent
   prior-predictive S-B bank after SNPE training, using a seed and cache path
   different from `data/wu2003_sbi_train_sb.npz`. SBC and coverage should be
   reported on this bank, not only on reused training-bank cases.

2. **Report point recovery and uncertainty calibration separately.** The current
   eta/xi posterior means are accurate, but their intervals are overconfident.
   nb24 should therefore keep mean-recovery plots, but the acceptance table must
   include held-out 90% coverage and SBC rank p-values for every parameter.

3. **Apply calibrated intervals for reporting.** For `eta_col` and `xi_reb`, nb24
    should either use the nb24b interval-inflation factors when reporting 90%
    intervals, or clearly label the raw NSF intervals as uncalibrated. The
    calibrated samples should be used for interval coverage and uncertainty bands;
    the raw posterior means can still be reported as point estimates.

    The eta repair is a conventional interval-calibration repair. The xi repair is
    more conservative: it fixes the scenario under-coverage caused by upper-bound
    cases near `xi_reb = 1.0`, but it should be described as a reporting-level
    conservative interval rather than proof that the raw neural density is fully
    rank-calibrated.

4. **Do not weaken the structural interpretation.** The alpha/beta broadness is
   still a closed-loop information limitation, supported by nb24a's Fisher and
   controller-masking checks. The eta/xi issue is different: it is mainly an
   uncertainty-calibration problem, not evidence that those parameters are
   physically unidentifiable.

5. **Update nb24 outputs.** Add `wu2003_nb24b_calibration_factors.csv` or its
   equivalent to the nb24 metadata, add raw-vs-calibrated coverage to the metrics
   table, and add a calibration-status column with values such as `calibrated`,
   `broad_but_honest`, and `overconfident_raw_interval`.

The publication language should therefore say: S-B gives useful point recovery
for eta/xi, but the raw neural posterior is overconfident for those parameters
unless calibrated against independent prior-predictive simulations. For xi, the
cleaner long-term fix is to retrain or ensemble the posterior with explicit
held-out calibration monitoring and then use nb24b's boundary-stress check as an
acceptance test.
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
    nb_path = repo_root / "notebooks" / "24b_wu2003_calibration_repair.ipynb"
    print(f"Executing notebook -> {nb_path}", flush=True)
    client = NotebookClient(
        nb,
        kernel_name="python3",
        timeout=None,
        resources={"metadata": {"path": str(repo_root)}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"Wrote {nb_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
