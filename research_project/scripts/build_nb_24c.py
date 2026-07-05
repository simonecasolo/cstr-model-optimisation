"""Build and execute notebook 24c: Wu 2003 S-B CNN embedding summary study."""

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
    if "from __future__ import annotations" in source and "N_CNN_SIMULATIONS" in source:
        text = (
            "This code imports libraries, defines paths, prior bounds, and the environment controls for "
            "the Wu S-B CNN embedding study."
        )
    elif "wu2003_observations.npz" in source and "posterior_summary_payload" in source:
        text = (
            "This code loads the existing nb24 hand-crafted-summary posterior, Wu S-B raw scenario windows, "
            "and closed-loop evaluation labels used for baseline-vs-CNN comparison."
        )
    elif "def simulate_sb_raw_window" in source and "raw_windows_to_flat" in source:
        text = (
            "This code defines the raw-window simulator and channel-first flattening convention used to feed "
            "S-B trajectories into the CNN embedding network."
        )
    elif "def generate_raw_training_bank" in source:
        text = (
            "This code creates or loads a continuous-prior raw S-B training bank for the CNNEmbedding SNPE "
            "posterior, rejecting invalid simulated windows."
        )
    elif "CNNEmbedding" in source and "train_cnn_posterior" in source:
        text = (
            "This code trains or loads the nb24c CNNEmbedding posterior from raw S-B windows, mirroring the "
            "nb04b learned-summary workflow."
        )
    elif "def sample_summary_posterior" in source and "comparison_rows" in source:
        text = (
            "This code samples both the original hand-crafted-summary posterior and the CNNEmbedding posterior "
            "for each closed-loop scenario and computes recovery/coverage metrics."
        )
    elif "focused_cases" in source and "24c_w1_w11" in source:
        text = (
            "This code plots W1 and W11 marginal posterior comparisons for all five parameters, showing whether "
            "the CNN changes the nb24 conclusions."
        )
    elif "fig, axes = plt.subplots" in source and "24c_recovery" in source:
        text = (
            "This code summarizes all-scenario baseline-vs-CNN recovery and coverage in a compact comparison "
            "figure."
        )
    elif "verdict_rows" in source:
        text = (
            "This code writes nb24c outputs and records the interpretation of whether learned raw summaries "
            "repair or confirm the nb24/nb24a limitations."
        )
    else:
        text = f"This code computes the next nb24c CNN-embedding step starting with: `{compact[:120]}`."
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
        """# Notebook 24c -- CNN learned summaries for Wu S-B raw trajectories

This notebook mirrors the idea of nb04b for the Wu 2003 S-B problem. nb04b asked
whether a posterior defect was caused by hand-crafted summary statistics or by a
structural closed-loop information limitation. It trained an `sbi` posterior with
`CNNEmbedding` on raw time-series observations and compared that posterior with
the hand-crafted-summary posterior.

nb24c asks the same question for Wu S-B:

- baseline: nb24's NSF posterior trained on the 66-D hand-crafted S-B summary;
- learned summary: a new NSF posterior whose `CNNEmbedding` reads raw `(120, 9)`
  S-B trajectories flattened in channel-first order.

If the CNN posterior materially improves eta/xi calibration or alpha/beta
recovery, the 66-D summary is losing useful information. If it shows the same
broadness or overconfidence, the limitation is more likely structural or a
neural-density calibration issue rather than just a hand-crafted-summary problem.
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

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.utils import BoxUniform

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB,
    NOMINAL_INLET,
    extract_observations_explicit,
    simulate_trajectory_explicit,
)
from cstr_sbi.recycle.simulator import SB_INDICES, nominal_warm_start

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.precision", 5)

torch.manual_seed(20260628)

DATA = Path("data")
RESULTS = Path("results")
FIGS = Path("figures")
for path in [DATA, RESULTS, FIGS]:
    path.mkdir(exist_ok=True)

PARAMETER_NAMES = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]
PRIOR_LOW = np.array([0.40, 0.40, 0.50, 0.40, 0.70], dtype=np.float32)
PRIOR_HIGH = np.array([1.20, 1.20, 1.00, 1.20, 0.95], dtype=np.float32)
PRIOR_WIDTH = PRIOR_HIGH - PRIOR_LOW

RAW_BANK_PATH = DATA / "wu2003_nb24c_raw_cnn_train_sb.npz"
CNN_POSTERIOR_PATH = RESULTS / "wu2003_nb24c_cnn_embedding_posterior.pkl"
SUMMARY_POSTERIOR_PATH = RESULTS / "wu2003_nb24_sb_sbi_posterior_final.pkl"
SCENARIO_FEATURE_PATH = DATA / "wu2003_summary_features.npz"
RAW_OBS_PATH = DATA / "wu2003_observations.npz"

N_CNN_SIMULATIONS = int(os.environ.get("WU2003_NB24C_N_CNN_SIMULATIONS", "1200"))
MAX_NUM_EPOCHS = int(os.environ.get("WU2003_NB24C_MAX_EPOCHS", "120"))
TRAINING_BATCH_SIZE = int(os.environ.get("WU2003_NB24C_BATCH", "256"))
POSTERIOR_SAMPLES = int(os.environ.get("WU2003_NB24C_POSTERIOR_SAMPLES", "2000"))
PROGRESS_INTERVAL = int(os.environ.get("WU2003_NB24C_PROGRESS_INTERVAL", "100"))


def log_status(message):
    print(f"[nb24c {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


print("nb24c raw CNN simulations:", N_CNN_SIMULATIONS)
print("nb24c max epochs:", MAX_NUM_EPOCHS)
print("nb24c posterior samples:", POSTERIOR_SAMPLES)
"""
    ),
    md("""## 2. Load nb24 baseline posterior and raw S-B observations"""),
    code(
        """assert SUMMARY_POSTERIOR_PATH.exists(), "Run nb24 before nb24c."
assert SCENARIO_FEATURE_PATH.exists(), "Run nb23 before nb24c."
assert RAW_OBS_PATH.exists(), "Run nb22 before nb24c."

with open(SUMMARY_POSTERIOR_PATH, "rb") as f:
    posterior_summary_payload = pickle.load(f)
summary_posterior = posterior_summary_payload["posterior"]
summary_scaler = posterior_summary_payload["scaler"]
summary_meta = posterior_summary_payload["metadata"]

with np.load(SCENARIO_FEATURE_PATH, allow_pickle=True) as data:
    X_sb_all = data["X_sb"].astype(np.float32)
    labels_summary = pd.DataFrame.from_records(data["labels"])

with np.load(RAW_OBS_PATH, allow_pickle=True) as obs:
    t_h = obs["t_h"].astype(np.float32)
    raw_sb_all = obs["observations_sb"].astype(np.float32)
    labels_raw = pd.DataFrame.from_records(obs["labels"])
    sb_channels = [str(x) for x in obs["sb_channels"]]

closed_mask = labels_summary["mode"].eq("closed_loop").to_numpy()
X_eval_summary = X_sb_all[closed_mask]
raw_eval = raw_sb_all[closed_mask]
labels_eval = labels_summary.loc[closed_mask].reset_index(drop=True)
theta_eval = labels_eval[PARAMETER_NAMES].to_numpy(dtype=np.float32)
scenario_id = labels_eval["scenario_id"].to_numpy()
scenario_name = labels_eval["scenario_name"].to_numpy()

assert raw_eval.shape[1:] == (120, 9)
assert X_eval_summary.shape[0] == raw_eval.shape[0]
print("baseline summary posterior:", summary_meta)
print("raw S-B evaluation windows:", raw_eval.shape)
print("summary evaluation matrix:", X_eval_summary.shape)
print("S-B channels:", sb_channels)
"""
    ),
    md("""## 3. Raw-window simulator and CNN input convention"""),
    code(
        """def make_prior():
    return BoxUniform(
        low=torch.as_tensor(PRIOR_LOW, dtype=torch.float32),
        high=torch.as_tensor(PRIOR_HIGH, dtype=torch.float32),
    )


def sample_prior_numpy(n, rng):
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * rng.random((n, len(PARAMETER_NAMES)), dtype=np.float32)


def noisy_sensor_layer(obs, rng, noise_pct=0.003):
    scale = np.maximum(np.max(np.abs(obs), axis=0, keepdims=True), 1e-12)
    return obs + rng.normal(0.0, noise_pct * scale, size=obs.shape)


def simulate_sb_raw_window(theta, y0, rng):
    ts, ys = simulate_trajectory_explicit(theta, NOMINAL_INLET, NOMINAL_CTRL_SB, y0, t_final=2.0, n_save=120)
    raw = np.asarray(extract_observations_explicit(ys, theta, NOMINAL_CTRL_SB), dtype=np.float32)
    sb = raw[:, SB_INDICES]
    sb = noisy_sensor_layer(sb, rng).astype(np.float32)
    if not np.isfinite(sb).all():
        raise ValueError("non-finite raw S-B window")
    return sb


def fit_channel_normalizer(windows):
    mean = windows.mean(axis=(0, 1)).astype(np.float32)
    std = np.maximum(windows.std(axis=(0, 1)).astype(np.float32), 1e-6)
    return mean, std


def raw_windows_to_flat(windows, channel_mean, channel_std):
    normalized = (windows - channel_mean[None, None, :]) / channel_std[None, None, :]
    channel_first = np.transpose(normalized, (0, 2, 1))
    return channel_first.reshape(channel_first.shape[0], -1).astype(np.float32)


print("CNN flattened S-B input dimension:", 9 * len(t_h))
"""
    ),
    md("""## 4. Generate or load continuous-prior raw S-B training bank"""),
    code(
        """def generate_raw_training_bank(n_simulations, path):
    log_status(f"Starting raw S-B CNN bank: target={n_simulations:,}, path={path}")
    rng = np.random.default_rng(20260628)
    theta_train = np.empty((n_simulations, len(PARAMETER_NAMES)), dtype=np.float32)
    raw_train = np.empty((n_simulations, len(t_h), len(sb_channels)), dtype=np.float32)
    y0_sb = nominal_warm_start("S-B")
    accepted = 0
    attempted = 0
    rejected = 0
    t0 = time.perf_counter()
    while accepted < n_simulations:
        theta_i = sample_prior_numpy(1, rng)[0]
        attempted += 1
        try:
            window_i = simulate_sb_raw_window(theta_i, y0_sb, rng)
        except Exception:
            rejected += 1
            continue
        theta_train[accepted] = theta_i
        raw_train[accepted] = window_i
        accepted += 1
        if accepted % PROGRESS_INTERVAL == 0 or accepted == n_simulations:
            elapsed = time.perf_counter() - t0
            rate = accepted / max(elapsed, 1e-9)
            remaining = (n_simulations - accepted) / max(rate, 1e-9)
            log_status(
                f"raw bank {accepted:>5}/{n_simulations}; attempted={attempted}; "
                f"rejected={rejected}; elapsed={elapsed/60:.1f} min; eta={remaining/60:.1f} min"
            )
    channel_mean, channel_std = fit_channel_normalizer(raw_train)
    np.savez_compressed(
        path,
        theta=theta_train,
        raw_windows=raw_train,
        t_h=t_h,
        sb_channels=np.asarray(sb_channels, dtype=object),
        parameter_names=np.asarray(PARAMETER_NAMES, dtype=object),
        prior_low=PRIOR_LOW,
        prior_high=PRIOR_HIGH,
        channel_mean=channel_mean,
        channel_std=channel_std,
        n_simulations=np.asarray(n_simulations),
        n_attempted=np.asarray(attempted),
        n_rejected=np.asarray(rejected),
        wall_time_s=np.asarray(time.perf_counter() - t0),
    )
    log_status(f"Saved raw CNN bank: theta={theta_train.shape}, raw={raw_train.shape}")
    return theta_train, raw_train, channel_mean, channel_std


if RAW_BANK_PATH.exists():
    with np.load(RAW_BANK_PATH, allow_pickle=True) as bank:
        theta_cnn = bank["theta"].astype(np.float32)
        raw_cnn = bank["raw_windows"].astype(np.float32)
        channel_mean = bank["channel_mean"].astype(np.float32)
        channel_std = bank["channel_std"].astype(np.float32)
    if theta_cnn.shape[0] < N_CNN_SIMULATIONS or raw_cnn.shape[1:] != (120, 9) or not np.isfinite(raw_cnn).all():
        log_status("Cached raw CNN bank is too small or invalid; regenerating")
        theta_cnn, raw_cnn, channel_mean, channel_std = generate_raw_training_bank(N_CNN_SIMULATIONS, RAW_BANK_PATH)
    else:
        theta_cnn = theta_cnn[:N_CNN_SIMULATIONS]
        raw_cnn = raw_cnn[:N_CNN_SIMULATIONS]
        log_status(f"Loaded cached raw CNN bank: theta={theta_cnn.shape}, raw={raw_cnn.shape}")
else:
    theta_cnn, raw_cnn, channel_mean, channel_std = generate_raw_training_bank(N_CNN_SIMULATIONS, RAW_BANK_PATH)

x_cnn = raw_windows_to_flat(raw_cnn, channel_mean, channel_std)
assert theta_cnn.shape == (N_CNN_SIMULATIONS, 5)
assert x_cnn.shape == (N_CNN_SIMULATIONS, 1080)
assert np.isfinite(x_cnn).all()
print("CNN training tensors:", theta_cnn.shape, x_cnn.shape)
"""
    ),
    md("""## 5. Train or load the CNNEmbedding posterior"""),
    code(
        """def train_cnn_posterior(theta_bank, x_bank, cache_path):
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        meta = payload["metadata"]
        if meta.get("n_simulations") == len(theta_bank) and meta.get("max_num_epochs") == MAX_NUM_EPOCHS:
            log_status(f"Loaded cached CNNEmbedding posterior: {cache_path}")
            return payload["posterior"], payload["metadata"]
        log_status("Cached CNNEmbedding posterior metadata differs; retraining")

    embedding_net = CNNEmbedding(
        input_shape=(120,),
        in_channels=9,
        out_channels_per_layer=[16, 32],
        num_conv_layers=2,
        num_linear_layers=2,
        num_linear_units=64,
        output_dim=32,
        kernel_size=5,
        pool_kernel_size=2,
    )
    n_parameters = sum(p.numel() for p in embedding_net.parameters())
    print(f"CNNEmbedding parameters: {n_parameters}")
    density = posterior_nn(
        model="nsf",
        hidden_features=128,
        num_transforms=5,
        embedding_net=embedding_net,
        z_score_x="structured",
    )
    inference = SNPE(prior=make_prior(), density_estimator=density, show_progress_bars=True)
    theta_tensor = torch.as_tensor(theta_bank, dtype=torch.float32)
    x_tensor = torch.as_tensor(x_bank, dtype=torch.float32)
    log_status(f"Appending CNN simulations: theta={tuple(theta_tensor.shape)}, x={tuple(x_tensor.shape)}")
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
        "embedding": "CNNEmbedding",
        "input_shape": [120],
        "in_channels": 9,
        "output_dim": 32,
        "hidden_features": 128,
        "num_transforms": 5,
        "training_batch_size": int(TRAINING_BATCH_SIZE),
        "max_num_epochs": int(MAX_NUM_EPOCHS),
        "embedding_parameters": int(n_parameters),
        "wall_time_s": float(time.perf_counter() - t0),
    }
    with open(cache_path, "wb") as f:
        pickle.dump({"posterior": posterior, "metadata": metadata}, f)
    log_status(f"Trained CNNEmbedding posterior in {metadata['wall_time_s']/60:.1f} min")
    return posterior, metadata


posterior_cnn, cnn_meta = train_cnn_posterior(theta_cnn, x_cnn, CNN_POSTERIOR_PATH)
print(cnn_meta)
"""
    ),
    md("""## 6. Compare hand-crafted summary posterior with CNN posterior"""),
    code(
        """def sample_summary_posterior(x_summary, n_samples=POSTERIOR_SAMPLES, seed=0):
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(summary_scaler.transform(np.asarray(x_summary, dtype=np.float32)[None, :])[0], dtype=torch.float32)
    with torch.no_grad():
        samples = summary_posterior.sample((n_samples,), x=x_tensor, show_progress_bars=False)
    return samples.cpu().numpy().astype(np.float32)


def sample_cnn_posterior(raw_window, n_samples=POSTERIOR_SAMPLES, seed=0):
    torch.manual_seed(seed)
    x_flat = raw_windows_to_flat(raw_window[None, :, :].astype(np.float32), channel_mean, channel_std)[0]
    x_tensor = torch.as_tensor(x_flat, dtype=torch.float32)
    with torch.no_grad():
        samples = posterior_cnn.sample((n_samples,), x=x_tensor, show_progress_bars=False)
    return samples.cpu().numpy().astype(np.float32)


def pick_eval(sid, replicate=0):
    matches = np.where(scenario_id == sid)[0]
    idx = int(matches[min(replicate, len(matches) - 1)])
    return X_eval_summary[idx], raw_eval[idx], theta_eval[idx], scenario_name[idx]


def summarize_samples(samples, truth):
    rows = []
    for j, parameter in enumerate(PARAMETER_NAMES):
        vals = samples[:, j]
        q05, q50, q95 = np.percentile(vals, [5, 50, 95])
        rows.append({
            "parameter": parameter,
            "truth": float(truth[j]),
            "mean": float(vals.mean()),
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "covered90": bool(q05 <= truth[j] <= q95),
            "abs_error": float(abs(vals.mean() - truth[j])),
            "width90": float(q95 - q05),
            "width90_prior_frac": float((q95 - q05) / PRIOR_WIDTH[j]),
        })
    return rows


comparison_rows = []
scenario_sample_cache = {}
for sid in sorted(np.unique(scenario_id)):
    x_summary, raw_window, truth, name = pick_eval(int(sid))
    log_status(f"Sampling baseline and CNN posteriors for scenario {sid}: {name}")
    summary_samples = sample_summary_posterior(x_summary, seed=620000 + int(sid))
    cnn_samples = sample_cnn_posterior(raw_window, seed=630000 + int(sid))
    scenario_sample_cache[int(sid)] = {"name": name, "truth": truth, "summary": summary_samples, "cnn": cnn_samples}
    for posterior_label, samples in [("summary_66d", summary_samples), ("cnn_raw", cnn_samples)]:
        for row in summarize_samples(samples, truth):
            row.update({"scenario_id": int(sid), "scenario_name": name, "posterior": posterior_label})
            comparison_rows.append(row)

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(RESULTS / "wu2003_nb24c_cnn_vs_summary_scenario_metrics.csv", index=False)
metrics = comparison.groupby(["posterior", "parameter"], as_index=False).agg(
    mae=("abs_error", "mean"),
    coverage90=("covered90", "mean"),
    mean_width90_prior_frac=("width90_prior_frac", "mean"),
)
metrics.to_csv(RESULTS / "wu2003_nb24c_cnn_vs_summary_metrics.csv", index=False)
display(metrics.round(5))
"""
    ),
    md("""## 7. W1/W11 all-parameter marginal comparison"""),
    code(
        """focused_cases = [1, 11]
fig, axes = plt.subplots(len(focused_cases), len(PARAMETER_NAMES), figsize=(16, 5.8), constrained_layout=True)
for row, sid in enumerate(focused_cases):
    item = scenario_sample_cache[sid]
    for col, parameter in enumerate(PARAMETER_NAMES):
        ax = axes[row, col]
        truth = item["truth"][col]
        vals_summary = item["summary"][:, col]
        vals_cnn = item["cnn"][:, col]
        ax.hist(vals_summary, bins=45, density=True, alpha=0.45, color="#4C78A8", label="66-D summary")
        ax.hist(vals_cnn, bins=45, density=True, alpha=0.45, color="#E45756", label="CNN raw")
        ax.axvline(truth, color="black", linestyle="--", linewidth=1.4, label="truth")
        ax.set_xlim(PRIOR_LOW[col], PRIOR_HIGH[col])
        ax.set_title(f"{item['name']}\\n{parameter}", fontsize=9)
        ax.grid(alpha=0.2)
        if col == 0:
            ax.set_ylabel("density")
        if row == 0 and col == len(PARAMETER_NAMES) - 1:
            ax.legend(fontsize=7)
fig.suptitle("nb24c: hand-crafted 66-D summary posterior vs CNN raw-window posterior")
fig.savefig(FIGS / "24c_w1_w11_cnn_vs_summary_marginals.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 8. All-scenario recovery and coverage comparison"""),
    code(
        """plot_metrics = metrics.copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
x = np.arange(len(PARAMETER_NAMES))
width = 0.36
colors = {"summary_66d": "#4C78A8", "cnn_raw": "#E45756"}
for offset, label in [(-width / 2, "summary_66d"), (width / 2, "cnn_raw")]:
    sub = plot_metrics[plot_metrics["posterior"].eq(label)].set_index("parameter").loc[PARAMETER_NAMES]
    axes[0].bar(x + offset, sub["mae"], width=width, label=label, color=colors[label], alpha=0.85)
    axes[1].bar(x + offset, sub["coverage90"], width=width, label=label, color=colors[label], alpha=0.85)
axes[0].set_ylabel("mean absolute error")
axes[0].set_title("posterior mean recovery")
axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("scenario 90% coverage")
axes[1].set_title("interval coverage")
for ax in axes:
    ax.set_xticks(x, PARAMETER_NAMES, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("nb24c: does a CNN learned summary improve Wu S-B inference?")
fig.savefig(FIGS / "24c_recovery_cnn_vs_summary.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 9. Verdict and outputs"""),
    code(
        """summary_metrics = metrics[metrics["posterior"].eq("summary_66d")].set_index("parameter")
cnn_metrics = metrics[metrics["posterior"].eq("cnn_raw")].set_index("parameter")
verdict_rows = []
for parameter in PARAMETER_NAMES:
    mae_delta = float(cnn_metrics.loc[parameter, "mae"] - summary_metrics.loc[parameter, "mae"])
    cov_delta = float(cnn_metrics.loc[parameter, "coverage90"] - summary_metrics.loc[parameter, "coverage90"])
    verdict_rows.append({
        "parameter": parameter,
        "cnn_minus_summary_mae": mae_delta,
        "cnn_minus_summary_coverage90": cov_delta,
        "interpretation": "CNN improves this parameter" if (mae_delta < -0.01 or cov_delta > 0.15) else "CNN does not materially improve this parameter",
    })

verdict = pd.DataFrame(verdict_rows)
verdict.to_csv(RESULTS / "wu2003_nb24c_verdict.csv", index=False)
metadata = {
    "n_cnn_simulations": int(N_CNN_SIMULATIONS),
    "posterior_samples": int(POSTERIOR_SAMPLES),
    "max_num_epochs": int(MAX_NUM_EPOCHS),
    "raw_bank_path": str(RAW_BANK_PATH),
    "cnn_posterior_path": str(CNN_POSTERIOR_PATH),
    "cnn_metadata": cnn_meta,
    "figures": [
        "24c_w1_w11_cnn_vs_summary_marginals.png",
        "24c_recovery_cnn_vs_summary.png",
    ],
}
with open(RESULTS / "wu2003_nb24c_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

acceptance = pd.DataFrame([
    {"check": "raw CNN bank shape", "observed": str(raw_cnn.shape), "status": "PASS" if raw_cnn.shape == (N_CNN_SIMULATIONS, 120, 9) else "FAIL"},
    {"check": "CNN posterior trained or loaded", "observed": cnn_meta.get("embedding"), "status": "PASS" if cnn_meta.get("embedding") == "CNNEmbedding" else "FAIL"},
    {"check": "comparison metrics exist", "observed": metrics.shape, "status": "PASS" if metrics.shape[0] == 10 else "FAIL"},
    {"check": "figures exist", "observed": str(all((FIGS / name).exists() for name in metadata["figures"])), "status": "PASS" if all((FIGS / name).exists() for name in metadata["figures"]) else "FAIL"},
])
acceptance.to_csv(RESULTS / "wu2003_nb24c_acceptance.csv", index=False)
display(verdict.round(5))
display(acceptance)
"""
    ),
    md(
        """## 10. Interpretation

nb24c is the Wu analogue of nb04b. It asks whether the conclusions from nb24 and
nb24a are artifacts of the 66-D hand-crafted summary or persist when the summary
statistics are learned from raw S-B trajectories.

The comparison should be read parameter by parameter:

- If the CNN posterior substantially improves `alpha` or `beta_r`, then the
  hand-crafted S-B summary is discarding useful transient information.
- If the CNN posterior leaves `alpha`/`beta_r` broad or similarly biased, that
  supports the controller-masking interpretation from nb24a.
- If the CNN posterior fixes `eta_col`/`xi_reb` coverage without nb24b-style
  interval calibration, then the hand-crafted summary was part of the eta/xi
  calibration problem.
- If the CNN posterior remains overconfident for eta/xi, then the issue is more
  likely neural-density calibration, boundary behavior, or lack of enough raw
  prior simulations, rather than only the 66-D summary design.

Because this notebook trains a new posterior from raw windows, its conclusion is
not meant to replace nb24 immediately. It is a stress test: nb24 should keep its
66-D summary posterior as the reproducible baseline, use nb24b for calibrated
reporting intervals, and use nb24c to decide whether a learned raw-trajectory
summary deserves to become the future primary SBI pipeline.
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
    nb_path = repo_root / "notebooks" / "24c_wu2003_cnn_embedding_summary.ipynb"
    print(f"Executing notebook -> {nb_path}", flush=True)

    def _cell_label(cell) -> str:
        first = cell.source.strip().splitlines()[0] if cell.source.strip() else "empty cell"
        return first[:100]

    def _on_cell_start(cell, cell_index, **kwargs):
        print(f"[nb24c-run] start cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    def _on_cell_complete(cell, cell_index, **kwargs):
        print(f"[nb24c-run] done  cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    def _on_cell_error(cell, cell_index, execute_reply, **kwargs):
        print(f"[nb24c-run] ERROR cell {cell_index + 1}/{len(nb.cells)}: {_cell_label(cell)}", flush=True)

    class StreamingNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            content = msg.get("content", {})
            if msg.get("msg_type") == "stream":
                text = content.get("text", "")
                if text:
                    prefix = f"[nb24c-cell {cell_index + 1} {content.get('name', 'stream')}] "
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
