"""Build and execute notebook 24a: Wu 2003 S-B posterior diagnostic audit."""

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
    if "from __future__ import annotations" in source and "POSTERIOR_SAMPLES" in source:
        text = (
            "This code imports the diagnostic libraries, defines paths, parameter names, prior bounds, and "
            "sample-count controls for the nb24a audit."
        )
    elif "KEYWORDS =" in source and "notebook_scan" in source:
        text = (
            "This code scans the project notebooks for the diagnostic arguments already used in the study, "
            "including masking, identifiability, posterior calibration, and summary-statistic checks."
        )
    elif "wu2003_sbi_train_sb.npz" in source and "posterior_payload" in source:
        text = (
            "This code loads the nb24 simulation bank, closed-loop S-B evaluation summaries, trained final "
            "posterior, scaler, and saved nb24 calibration/recovery tables."
        )
    elif "def sample_posterior_for_x" in source and "focused_cases" in source:
        text = (
            "This code samples W1 and W11 posteriors for all five parameters, computes 90% intervals, and "
            "plots focused marginal distributions that nb24 did not show explicitly."
        )
    elif "sbc_checks = []" in source and "calibration_audit" in source:
        text = (
            "This code recomputes calibration diagnostics from nb24's SBC ranks and scenario recovery table, "
            "then classifies parameters as broad-but-calibrated or miscalibrated/overconfident."
        )
    elif "corr_rows = []" in source and "posterior_corr" in source:
        text = (
            "This code computes posterior correlations for W1 and W11 to test whether parameter confounding "
            "contributes to broad or tilted marginal posteriors."
        )
    elif "def feature_group_mask" in source and "feature_predictability" in source:
        text = (
            "This code compares how well different S-B feature groups predict each parameter, separating "
            "temperature-only information from controller-effort and physics-proxy information."
        )
    elif "mutual_info_regression" in source and "feature_mi" in source:
        text = (
            "This code estimates feature-level mutual information for each parameter and aggregates it by "
            "channel to identify which S-B measurements carry parameter information."
        )
    elif "def deterministic_summary" in source and "fisher_proxy" in source:
        text = (
            "This code implements the nb15-style local Fisher/sensitivity proxy for Wu S-B by finite-differencing "
            "the deterministic 66-D summary with respect to each inferred parameter."
        )
    elif "RAW_CHANNELS =" in source and "masking_index" in source:
        text = (
            "This code runs deterministic low/nominal/high parameter sweeps to measure whether controlled "
            "variables stay masked while manipulated variables and controller efforts move."
        )
    elif "verdict_rows =" in source:
        text = (
            "This code combines the calibration, feature-predictability, Fisher-proxy, masking, and posterior "
            "correlation evidence into the final nb24a verdict table."
        )
    else:
        text = f"This code computes the next nb24a diagnostic step starting with: `{compact[:120]}`."
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
        """# Notebook 24a -- Why are the Wu S-B posteriors broad?

This notebook audits the nb24 S-B posterior using the same style of arguments
used across the project notebooks: feedback masking and controller effort
(nb01/nb21/nb22), summary-statistic separability (nb03/nb23), prior-predictive
and posterior recovery checks (nb04/nb24), saturation/identifiability logic
(nb08), claim/limitation framing (nb14), the β-bias Fisher-information argument
(nb15), and SBC/calibration diagnostics (nb24).

The goal is to separate four explanations:

1. **Controller masking:** controlled variables barely move while manipulated
   variables compensate.
2. **Parameter confounding:** different parameters produce similar S-B summaries.
3. **Summary/statistic loss:** the 66-D S-B summary omits useful information.
4. **SBI calibration artifact:** the posterior density is numerically trained but
   over- or under-confident.
"""
    ),
    md("""## 1. Imports and paths"""),
    code(
        """from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB,
    NOMINAL_INLET,
    NOMINAL_THETA,
    extract_observations_explicit,
    simulate_trajectory_explicit,
)
from cstr_sbi.recycle.simulator import SB_INDICES, nominal_warm_start

warnings.filterwarnings("ignore", category=UserWarning)

DATA = Path("data")
RESULTS = Path("results")
FIGS = Path("figures")
NOTEBOOKS = Path("notebooks")
for path in [RESULTS, FIGS]:
    path.mkdir(exist_ok=True)

PARAMETER_NAMES = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]
PRIOR_LOW = np.array([0.40, 0.40, 0.50, 0.40, 0.70], dtype=np.float32)
PRIOR_HIGH = np.array([1.20, 1.20, 1.00, 1.20, 0.95], dtype=np.float32)
PRIOR_WIDTH = PRIOR_HIGH - PRIOR_LOW

POSTERIOR_SAMPLES = int(os.environ.get("WU2003_NB24A_POSTERIOR_SAMPLES", "2000"))
MI_SUBSAMPLE = int(os.environ.get("WU2003_NB24A_MI_SUBSAMPLE", "6000"))

print("nb24a posterior samples per scenario:", POSTERIOR_SAMPLES)
print("nb24a MI subsample:", MI_SUBSAMPLE)
"""
    ),
    md("""## 2. What diagnostic arguments already exist in the notebooks?"""),
    code(
        """KEYWORDS = [
    "posterior", "sbc", "lda", "summary", "pca", "tsne", "controller",
    "control", "masking", "identifiability", "sensitivity",
    "prior predictive", "coverage", "saturation", "compensation",
]

scan_rows = []
for path in sorted(NOTEBOOKS.glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    headings = []
    hits = []
    for cell_number, cell in enumerate(nb.cells, start=1):
        source = cell.source
        if cell.cell_type == "markdown":
            headings.extend(line.strip() for line in source.splitlines() if line.startswith("#"))
        low = source.lower()
        matched = [keyword for keyword in KEYWORDS if keyword in low]
        if matched:
            first = source.strip().split("\\n")[0][:120] if source.strip() else ""
            hits.append({"cell": cell_number, "type": cell.cell_type, "matched": ", ".join(matched[:5]), "first_line": first})
    scan_rows.append({
        "notebook": path.name,
        "n_cells": len(nb.cells),
        "n_diagnostic_hits": len(hits),
        "headings": " | ".join(headings[:6]),
        "diagnostic_examples": hits[:4],
    })

notebook_scan = pd.DataFrame(scan_rows)
notebook_scan.to_json(RESULTS / "wu2003_nb24a_notebook_scan.json", orient="records", indent=2)
display(notebook_scan[["notebook", "n_cells", "n_diagnostic_hits", "headings"]])
"""
    ),
    md(
        """### Read-through summary, including nb14/nb15

The earlier notebooks use a consistent diagnostic grammar:

- **nb01/nb21/nb22:** feedback suppresses controlled-variable movement, while
  controller outputs carry the compensation signal.
- **nb03/nb23:** summary statistics are judged by separability, dimensionality,
  MI ranking, and LDA probes before being handed to SBI.
- **nb04/nb24:** SBI is judged by prior predictive checks, simulation-budget
  sensitivity, posterior recovery, and SBC.
- **nb08/nb09:** broad or biased posteriors are interpreted as identifiability,
  saturation, or nuisance-parameter confounding rather than as plotting defects.
- **nb14:** a posterior limitation is publication-relevant only when it is tied
    to a concrete information mechanism and not simply to a weak plot or a single
    estimator. The useful framing is: identify the operating envelope, report the
    limitation, and distinguish method artefacts from structural closed-loop loss.
- **nb15:** the strongest argument is a Fisher-information / sensitivity one:
    if a parameter's column in the local summary Jacobian is small or collinear
    after feedback, the broad or biased posterior is expected. Nonlinearity or a
    single posterior mean is not enough evidence by itself.

nb24a therefore tests both the plant/control mechanism and the statistical
posterior quality.
"""
    ),
    md("""## 3. Load nb24 artifacts"""),
    code(
        """with np.load(DATA / "wu2003_sbi_train_sb.npz", allow_pickle=True) as bank:
    theta_train = bank["theta"].astype(np.float32)
    X_train = bank["X"].astype(np.float32)
    features_sb = [str(x) for x in bank["feature_names"]]
    bank_meta = {key: np.asarray(bank[key]).item() for key in bank.files if key in ["n_attempted", "n_rejected", "wall_time_s"]}

with np.load(DATA / "wu2003_summary_features.npz", allow_pickle=True) as data:
    X_sb_all = data["X_sb"].astype(np.float32)
    labels_all = pd.DataFrame.from_records(data["labels"])

closed_mask = labels_all["mode"].eq("closed_loop").to_numpy()
X_eval = X_sb_all[closed_mask]
labels_eval = labels_all.loc[closed_mask].reset_index(drop=True)
theta_eval = labels_eval[PARAMETER_NAMES].to_numpy(dtype=np.float32)
scenario_id = labels_eval["scenario_id"].to_numpy()
scenario_name = labels_eval["scenario_name"].to_numpy()

with open(RESULTS / "wu2003_nb24_sb_sbi_posterior_final.pkl", "rb") as f:
    posterior_payload = pickle.load(f)
posterior = posterior_payload["posterior"]
scaler = posterior_payload["scaler"]
meta_final = posterior_payload["metadata"]

posterior_metrics = pd.read_csv(RESULTS / "wu2003_nb24_sb_sbi_posterior_metrics.csv")
scenario_recovery_nb24 = pd.read_csv(RESULTS / "wu2003_nb24_sb_sbi_scenario_recovery.csv")
sbc_ranks_nb24 = pd.read_csv(RESULTS / "wu2003_nb24_sb_sbi_sbc_ranks.csv")

print("train bank:", theta_train.shape, X_train.shape, bank_meta)
print("evaluation summaries:", X_eval.shape, "scenarios:", len(np.unique(scenario_id)))
print("posterior metadata:", meta_final)
display(posterior_metrics)
"""
    ),
    md("""## 4. All five posterior marginals for W1 and W11"""),
    code(
        """def sample_posterior_for_x(x, n_samples=POSTERIOR_SAMPLES, seed=0):
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(scaler.transform(np.asarray(x, dtype=np.float32)[None, :])[0], dtype=torch.float32)
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_tensor, show_progress_bars=False)
    return samples.cpu().numpy().astype(np.float32)


def pick_eval(sid, replicate=0):
    matches = np.where(scenario_id == sid)[0]
    idx = int(matches[min(replicate, len(matches) - 1)])
    return X_eval[idx], theta_eval[idx], scenario_name[idx]


focused_cases = []
for sid, label, color in [(1, "W1 healthy", "steelblue"), (11, "W11 snowball", "tomato")]:
    x_obs, truth, name = pick_eval(sid)
    samples = sample_posterior_for_x(x_obs, n_samples=POSTERIOR_SAMPLES, seed=24000 + sid)
    focused_cases.append({"sid": sid, "label": label, "name": name, "truth": truth, "samples": samples, "color": color})

posterior_rows = []
for case in focused_cases:
    for j, parameter in enumerate(PARAMETER_NAMES):
        vals = case["samples"][:, j]
        q05, q50, q95 = np.percentile(vals, [5, 50, 95])
        posterior_rows.append({
            "scenario_id": case["sid"],
            "scenario": case["name"],
            "parameter": parameter,
            "truth": float(case["truth"][j]),
            "mean": float(vals.mean()),
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "width90": float(q95 - q05),
            "width90_prior_frac": float((q95 - q05) / PRIOR_WIDTH[j]),
            "covered90": bool(q05 <= case["truth"][j] <= q95),
        })

focused_posterior_summary = pd.DataFrame(posterior_rows)
focused_posterior_summary.to_csv(RESULTS / "wu2003_nb24a_w1_w11_all_parameter_intervals.csv", index=False)
display(focused_posterior_summary.round(4))

fig, axes = plt.subplots(len(focused_cases), len(PARAMETER_NAMES), figsize=(16, 5.8), constrained_layout=True)
for row, case in enumerate(focused_cases):
    for col, parameter in enumerate(PARAMETER_NAMES):
        vals = case["samples"][:, col]
        truth = case["truth"][col]
        q05, q95 = np.percentile(vals, [5, 95])
        ax = axes[row, col]
        ax.hist(vals, bins=55, color=case["color"], alpha=0.78, density=True)
        ax.axvline(truth, color="black", linewidth=1.8, linestyle="--", label="truth")
        ax.axvline(vals.mean(), color="white", linewidth=1.5, linestyle="-", label="mean")
        ax.axvspan(q05, q95, color=case["color"], alpha=0.18, label="90% interval")
        ax.set_xlim(PRIOR_LOW[col], PRIOR_HIGH[col])
        ax.set_title(f"{case['label']}\\n{parameter}", fontsize=9)
        if col == 0:
            ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        if row == 0 and col == len(PARAMETER_NAMES) - 1:
            ax.legend(fontsize=7)
fig.suptitle("nb24a: W1/W11 posterior marginals for all five parameters")
fig.savefig(FIGS / "24a_w1_w11_all_parameter_marginals.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 5. Calibration audit: broad but honest vs narrow and wrong"""),
    code(
        """sbc_checks = []
for parameter in PARAMETER_NAMES:
    vals = sbc_ranks_nb24.loc[sbc_ranks_nb24["parameter"].eq(parameter), "scaled_rank"].to_numpy()
    ks = kstest(vals, "uniform")
    sbc_checks.append({"parameter": parameter, "sbc_ks_p": float(ks.pvalue)})
sbc_checks = pd.DataFrame(sbc_checks)

coverage_rows = []
for parameter in PARAMETER_NAMES:
    coverage_rows.append({
        "parameter": parameter,
        "scenario_coverage_90": float(scenario_recovery_nb24[f"covered90_{parameter}"].mean()),
        "mean_abs_error": float(scenario_recovery_nb24[f"abs_error_{parameter}"].mean()),
    })
coverage = pd.DataFrame(coverage_rows)

calibration_audit = posterior_metrics.merge(sbc_checks, on="parameter", how="left", suffixes=("", "_recomputed"))
calibration_audit = calibration_audit.merge(
    coverage[["parameter", "mean_abs_error"]].rename(columns={"mean_abs_error": "mean_abs_error_recomputed"}),
    on="parameter",
    how="left",
)
calibration_audit["interpretation"] = np.select(
    [
        (calibration_audit["sbc_rank_ks_p"] >= 0.05) & (calibration_audit["scenario_coverage_90"] >= 0.8),
        (calibration_audit["sbc_rank_ks_p"] < 0.01) | (calibration_audit["scenario_coverage_90"] < 0.8),
    ],
    ["broad/uncertain but reasonably calibrated", "miscalibrated or overconfident"],
    default="borderline",
)
calibration_audit.to_csv(RESULTS / "wu2003_nb24a_calibration_audit.csv", index=False)
display(calibration_audit.round(5))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].bar(calibration_audit["parameter"], calibration_audit["scenario_coverage_90"], color="#4C78A8")
axes[0].axhline(0.9, color="black", linestyle="--", linewidth=1, label="nominal 90%")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("scenario 90% coverage")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=0.2)

axes[1].bar(calibration_audit["parameter"], -np.log10(np.maximum(calibration_audit["sbc_rank_ks_p"], 1e-12)), color="#F58518")
axes[1].axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1, label="p=0.05")
axes[1].axhline(-np.log10(0.01), color="gray", linestyle=":", linewidth=1, label="p=0.01")
axes[1].set_ylabel("-log10 SBC KS p")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=0.2)
fig.suptitle("nb24a: calibration distinguishes broad posteriors from overconfident posteriors")
fig.savefig(FIGS / "24a_calibration_audit.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 6. Posterior confounding: are broad reactor parameters compensating each other?"""),
    code(
        """corr_rows = []
for case in focused_cases:
    corr = np.corrcoef(case["samples"].T)
    for i, p1 in enumerate(PARAMETER_NAMES):
        for j, p2 in enumerate(PARAMETER_NAMES):
            if j <= i:
                continue
            corr_rows.append({"scenario_id": case["sid"], "scenario": case["name"], "pair": f"{p1} vs {p2}", "corr": float(corr[i, j])})
posterior_corr = pd.DataFrame(corr_rows)
posterior_corr.to_csv(RESULTS / "wu2003_nb24a_w1_w11_posterior_correlations.csv", index=False)
display(posterior_corr.reindex(posterior_corr["corr"].abs().sort_values(ascending=False).index).head(20).round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for ax, case in zip(axes, focused_cases):
    corr = np.corrcoef(case["samples"].T)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(PARAMETER_NAMES)), PARAMETER_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(PARAMETER_NAMES)), PARAMETER_NAMES)
    ax.set_title(case["label"])
    for i in range(len(PARAMETER_NAMES)):
        for j in range(len(PARAMETER_NAMES)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.8, label="posterior correlation")
fig.suptitle("nb24a: posterior parameter confounding")
fig.savefig(FIGS / "24a_w1_w11_posterior_correlation.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 7. Feature-group predictability: controlled variables vs controller effort"""),
    code(
        """def feature_group_mask(names, prefixes):
    return np.array([any(name.startswith(prefix) for prefix in prefixes) for name in names])


temperature_mask = feature_group_mask(features_sb, ["T_r", "T_j", "T_reb"])
effort_mask = feature_group_mask(features_sb, ["Q_j", "Q_reb", "F_R_norm", "F_B_norm", "R_norm", "V_norm"])
physics_mask = ~(temperature_mask | effort_mask)

feature_sets = {
    "all_S-B_features": np.ones(len(features_sb), dtype=bool),
    "temperatures_only": temperature_mask,
    "controller_effort_only": effort_mask,
    "physics_proxy_only": physics_mask,
    "no_controller_effort": ~effort_mask,
}

cv = KFold(n_splits=5, shuffle=True, random_state=20260626)
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 13)))

r2_rows = []
for parameter_index, parameter in enumerate(PARAMETER_NAMES):
    y = theta_train[:, parameter_index]
    for set_name, mask in feature_sets.items():
        if mask.sum() == 0:
            continue
        scores = cross_val_score(ridge, X_train[:, mask], y, cv=cv, scoring="r2", n_jobs=None)
        r2_rows.append({
            "parameter": parameter,
            "feature_set": set_name,
            "n_features": int(mask.sum()),
            "r2_mean": float(scores.mean()),
            "r2_std": float(scores.std()),
        })

feature_predictability = pd.DataFrame(r2_rows)
feature_predictability.to_csv(RESULTS / "wu2003_nb24a_feature_group_r2.csv", index=False)
display(feature_predictability.pivot(index="parameter", columns="feature_set", values="r2_mean").round(3))

fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
pivot = feature_predictability.pivot(index="parameter", columns="feature_set", values="r2_mean").loc[PARAMETER_NAMES]
im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
ax.set_yticks(range(len(pivot.index)), pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white" if pivot.iloc[i, j] < 0.45 else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="5-fold CV R2")
ax.set_title("Can S-B summary groups predict the true prior parameters?")
fig.savefig(FIGS / "24a_feature_group_predictability.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 8. Feature mutual information: which summaries carry each parameter?"""),
    code(
        """rng = np.random.default_rng(20260626)
mi_idx = rng.choice(len(theta_train), size=min(MI_SUBSAMPLE, len(theta_train)), replace=False)

mi_rows = []
for parameter_index, parameter in enumerate(PARAMETER_NAMES):
    mi = mutual_info_regression(X_train[mi_idx], theta_train[mi_idx, parameter_index], random_state=20260626, n_neighbors=5)
    for feature, value in zip(features_sb, mi):
        if "__" in feature:
            channel, statistic = feature.split("__", 1)
        else:
            channel, statistic = "physics_proxy", feature
        mi_rows.append({"parameter": parameter, "feature": feature, "channel": channel, "statistic": statistic, "mi": float(value)})

feature_mi = pd.DataFrame(mi_rows)
feature_mi.to_csv(RESULTS / "wu2003_nb24a_feature_mi.csv", index=False)

top_mi = feature_mi.sort_values(["parameter", "mi"], ascending=[True, False]).groupby("parameter").head(12)
display(top_mi.round(4))

channel_mi = feature_mi.groupby(["parameter", "channel"], as_index=False)["mi"].sum()
channel_mi.to_csv(RESULTS / "wu2003_nb24a_channel_mi.csv", index=False)
pivot = channel_mi.pivot(index="parameter", columns="channel", values="mi").fillna(0).loc[PARAMETER_NAMES]

fig, ax = plt.subplots(figsize=(13, 4.6), constrained_layout=True)
im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot.index)), pivot.index)
fig.colorbar(im, ax=ax, label="sum mutual information")
ax.set_title("nb24a: S-B feature-channel information by parameter")
fig.savefig(FIGS / "24a_channel_mutual_information.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 9. Local Fisher-information proxy, following nb15"""),
    code(
        """# nb15 showed that closed-loop bias is best diagnosed by local sensitivity / Fisher information.
# Here we compute the Wu S-B analogue using finite differences of the 66-D summary vector.

def _safe_corr(a, b):
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


def deterministic_summary(theta):
    y0 = nominal_warm_start("S-B")
    ts, ys = simulate_trajectory_explicit(theta, NOMINAL_INLET, NOMINAL_CTRL_SB, y0, t_final=2.0, n_save=120)
    raw = np.asarray(extract_observations_explicit(ys, theta, NOMINAL_CTRL_SB))
    sb = raw[:, SB_INDICES]
    X, names = summarize_windows(sb[None, :, :], SB_CHANNELS, np.asarray(ts))
    return X[0], names


feature_scale = np.maximum(X_train.std(axis=0), 1e-6)
fisher_cases = {
    "W1_healthy": np.array([1.0, 1.0, 1.0, 1.0, 0.90], dtype=np.float32),
    "W11_snowball": np.array([0.65, 1.0, 0.75, 1.0, 0.90], dtype=np.float32),
    "nominal_prior_center": np.asarray(NOMINAL_THETA, dtype=np.float32),
}

fisher_rows = []
jacobian_corr_rows = []
for case_name, theta0 in fisher_cases.items():
    base, summary_names = deterministic_summary(theta0)
    J = np.zeros((len(base), len(PARAMETER_NAMES)), dtype=float)
    for j, parameter in enumerate(PARAMETER_NAMES):
        step = 0.01 * PRIOR_WIDTH[j]
        lo = max(PRIOR_LOW[j], theta0[j] - step)
        hi = min(PRIOR_HIGH[j], theta0[j] + step)
        if hi == lo:
            continue
        theta_lo = theta0.copy(); theta_hi = theta0.copy()
        theta_lo[j] = lo; theta_hi[j] = hi
        f_lo, _ = deterministic_summary(theta_lo)
        f_hi, _ = deterministic_summary(theta_hi)
        J[:, j] = (f_hi - f_lo) / (hi - lo)
    Jz = J / feature_scale[:, None]
    fim_proxy = Jz.T @ Jz
    col_norms = np.sqrt(np.diag(fim_proxy))
    for j, parameter in enumerate(PARAMETER_NAMES):
        fisher_rows.append({
            "case": case_name,
            "parameter": parameter,
            "scaled_jacobian_norm": float(col_norms[j]),
            "relative_to_max": float(col_norms[j] / max(col_norms.max(), 1e-12)),
        })
    for i, p1 in enumerate(PARAMETER_NAMES):
        for j, p2 in enumerate(PARAMETER_NAMES):
            if j <= i:
                continue
            denom = max(col_norms[i] * col_norms[j], 1e-12)
            jacobian_corr_rows.append({"case": case_name, "pair": f"{p1} vs {p2}", "cosine": float(fim_proxy[i, j] / denom)})

fisher_proxy = pd.DataFrame(fisher_rows)
jacobian_cosines = pd.DataFrame(jacobian_corr_rows)
fisher_proxy.to_csv(RESULTS / "wu2003_nb24a_fisher_proxy.csv", index=False)
jacobian_cosines.to_csv(RESULTS / "wu2003_nb24a_jacobian_cosines.csv", index=False)
display(fisher_proxy.pivot(index="parameter", columns="case", values="relative_to_max").loc[PARAMETER_NAMES].round(3))
display(jacobian_cosines.reindex(jacobian_cosines["cosine"].abs().sort_values(ascending=False).index).head(15).round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
pivot = fisher_proxy.pivot(index="parameter", columns="case", values="relative_to_max").loc[PARAMETER_NAMES]
im = axes[0].imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
axes[0].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
axes[0].set_yticks(range(len(pivot.index)), pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        axes[0].text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if pivot.iloc[i, j] < 0.45 else "black")
fig.colorbar(im, ax=axes[0], label="relative scaled sensitivity")
axes[0].set_title("Local Fisher proxy: column norm")

cos_pivot = jacobian_cosines[jacobian_cosines["case"].eq("W11_snowball")].pivot_table(index="pair", values="cosine")
cos_pivot = cos_pivot.reindex(cos_pivot["cosine"].abs().sort_values(ascending=True).index)
axes[1].barh(cos_pivot.index, cos_pivot["cosine"], color=np.where(cos_pivot["cosine"] >= 0, "#4C78A8", "#E45756"))
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_xlabel("standardized Jacobian cosine")
axes[1].set_title("W11 local parameter-direction collinearity")
axes[1].grid(axis="x", alpha=0.2)
fig.suptitle("nb24a: nb15-style Fisher/sensitivity audit for Wu S-B")
fig.savefig(FIGS / "24a_fisher_proxy.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 10. Deterministic parameter sweeps: does S-B feedback mask state movement?"""),
    code(
        """RAW_CHANNELS = [
    "z_A", "T_r", "T_j", "Q_j", "x_D", "x_B", "F_R_norm", "T_reb", "Q_reb", "F_B_norm", "R_norm", "V_norm",
]
SB_CHANNELS = ["T_r", "T_j", "Q_j", "T_reb", "Q_reb", "F_R_norm", "F_B_norm", "R_norm", "V_norm"]
CONTROLLED_CHANNELS = ["T_r", "T_j", "T_reb", "F_B_norm"]
EFFORT_CHANNELS = ["Q_j", "Q_reb", "F_R_norm", "R_norm", "V_norm"]


def deterministic_sb_window(theta):
    y0 = nominal_warm_start("S-B")
    ts, ys = simulate_trajectory_explicit(theta, NOMINAL_INLET, NOMINAL_CTRL_SB, y0, t_final=2.0, n_save=120)
    raw = np.asarray(extract_observations_explicit(ys, theta, NOMINAL_CTRL_SB))
    return np.asarray(ts), raw[:, SB_INDICES]


nominal_theta = np.asarray(NOMINAL_THETA, dtype=np.float32)
ts_nom, sb_nom = deterministic_sb_window(nominal_theta)
final_start = int(0.75 * len(ts_nom))
nominal_final = sb_nom[final_start:].mean(axis=0)

sweep_values = {
    "low": PRIOR_LOW,
    "nominal": nominal_theta,
    "high": PRIOR_HIGH,
}

sweep_rows = []
for parameter_index, parameter in enumerate(PARAMETER_NAMES):
    for level, values in sweep_values.items():
        theta = nominal_theta.copy()
        theta[parameter_index] = values[parameter_index]
        ts, sb = deterministic_sb_window(theta)
        final_mean = sb[final_start:].mean(axis=0)
        delta = final_mean - nominal_final
        rel_delta = delta / np.maximum(np.abs(nominal_final), 1e-9)
        for channel, value, delta_value, rel in zip(SB_CHANNELS, final_mean, delta, rel_delta):
            sweep_rows.append({
                "parameter": parameter,
                "level": level,
                "theta_value": float(theta[parameter_index]),
                "channel": channel,
                "final25_mean": float(value),
                "delta_vs_nominal": float(delta_value),
                "relative_delta_vs_nominal": float(rel),
            })

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(RESULTS / "wu2003_nb24a_deterministic_sweep_channels.csv", index=False)

masking_rows = []
for parameter in PARAMETER_NAMES:
    for level in ["low", "high"]:
        sub = sweep_df[(sweep_df["parameter"].eq(parameter)) & (sweep_df["level"].eq(level))]
        controlled_values = sub[sub["channel"].isin(CONTROLLED_CHANNELS)]["relative_delta_vs_nominal"].to_numpy(dtype=float)
        effort_values = sub[sub["channel"].isin(EFFORT_CHANNELS)]["relative_delta_vs_nominal"].to_numpy(dtype=float)
        nonfinite_count = int((~np.isfinite(np.r_[controlled_values, effort_values])).sum())
        controlled_values = controlled_values[np.isfinite(controlled_values)]
        effort_values = effort_values[np.isfinite(effort_values)]
        controlled_norm = float(np.linalg.norm(controlled_values)) if len(controlled_values) else np.nan
        effort_norm = float(np.linalg.norm(effort_values)) if len(effort_values) else np.nan
        masking_rows.append({
            "parameter": parameter,
            "level": level,
            "controlled_relative_norm": controlled_norm,
            "effort_relative_norm": effort_norm,
            "effort_to_controlled_ratio": effort_norm / max(controlled_norm, 1e-9),
            "nonfinite_channel_count": nonfinite_count,
        })
masking_index = pd.DataFrame(masking_rows)
masking_index.to_csv(RESULTS / "wu2003_nb24a_masking_index.csv", index=False)
display(masking_index.round(4))

heat = sweep_df[sweep_df["level"].isin(["low", "high"])].copy()
heat["case"] = heat["parameter"] + "=" + heat["level"]
pivot = heat.pivot(index="case", columns="channel", values="relative_delta_vs_nominal")

fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
limit = np.nanpercentile(np.abs(pivot.to_numpy()), 95)
im = ax.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot.index)), pivot.index)
fig.colorbar(im, ax=ax, label="relative final-window change vs nominal")
ax.set_title("nb24a: deterministic S-B sweep shows which channels move when each parameter changes")
fig.savefig(FIGS / "24a_deterministic_parameter_sweep_heatmap.png", dpi=140, bbox_inches="tight")
plt.show()
"""
    ),
    md("""## 11. Verdict table"""),
    code(
        """alpha_beta = calibration_audit[calibration_audit["parameter"].isin(["alpha", "beta_r"])]
eta_xi = calibration_audit[calibration_audit["parameter"].isin(["eta_col", "xi_reb"])]
r2_pivot = feature_predictability.pivot(index="parameter", columns="feature_set", values="r2_mean")

verdict_rows = [
    {
        "hypothesis": "nb14/nb15 closed-loop information-loss framing transfers to Wu S-B",
        "evidence": "local Fisher proxy plus deterministic masking audit, not just posterior visual width",
        "status": "SUPPORTED" if fisher_proxy["relative_to_max"].min() < 0.35 else "MIXED",
    },
    {
        "hypothesis": "controller masking broadens reactor-side posteriors",
        "evidence": "temperature-only feature R2 and deterministic sweep compared with controller-effort movement",
        "status": "SUPPORTED" if (r2_pivot.loc[["alpha", "beta_r"], "controller_effort_only"].mean() > r2_pivot.loc[["alpha", "beta_r"], "temperatures_only"].mean()) else "MIXED",
    },
    {
        "hypothesis": "alpha/beta_r broadness is mainly an SBI calibration failure",
        "evidence": "alpha and beta_r have high scenario coverage and non-significant SBC KS tests",
        "status": "NOT PRIMARY" if (alpha_beta["scenario_coverage_90"].min() >= 0.8 and alpha_beta["sbc_rank_ks_p"].min() >= 0.05) else "POSSIBLE",
    },
    {
        "hypothesis": "eta_col and xi_reb are trustworthy because their means are accurate",
        "evidence": "coverage and SBC fail despite tiny mean errors",
        "status": "REJECTED" if (eta_xi["sbc_rank_ks_p"].min() < 0.01 or eta_xi["scenario_coverage_90"].min() < 0.8) else "SUPPORTED",
    },
    {
        "hypothesis": "nb24 hides important posterior dimensions",
        "evidence": "nb24 W1/W11 figure only shows alpha and eta_col; nb24a adds all five marginals",
        "status": "SUPPORTED",
    },
    {
        "hypothesis": "parameter confounding contributes to broadness",
        "evidence": "posterior correlations among alpha, beta_r, and z_A0_eff in W1/W11",
        "status": "SUPPORTED" if posterior_corr[posterior_corr["pair"].str.contains("alpha")]["corr"].abs().max() > 0.35 else "WEAK",
    },
]

verdict = pd.DataFrame(verdict_rows)
verdict.to_csv(RESULTS / "wu2003_nb24a_verdict.csv", index=False)
display(verdict)
"""
    ),
    md(
        """## 12. Interpretation

The broad reactor-side posterior should not be treated as just a plotting or
training failure. nb14 says limitations should be tied to a concrete information
mechanism, and nb15 shows how to do that: compare local sensitivities/Fisher
information, not only posterior means. If controller-effort summaries are more
predictive than temperature-only summaries, deterministic sweeps show small
controlled-state movement with larger manipulated-variable movement, and the
Fisher proxy shows weak or collinear parameter directions, the broadness is a
plantwide feedback effect: S-B observes the compensation signal only indirectly.

The stronger warning is different: parameters such as `eta_col` and `xi_reb` can
look excellent by posterior mean while failing coverage/SBC. That is a calibration
problem or overconfident density-estimator problem, not evidence that those
parameters are solved. nb24 should therefore distinguish **broad but calibrated**
from **narrow but miscalibrated**.

The immediate reporting fix is to keep nb24's all-scenario grid, but add the
focused all-five-parameter W1/W11 marginal plot from nb24a and make calibration
warnings explicit in the acceptance table.
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
    nb_path = repo_root / "notebooks" / "24a_wu2003_posterior_diagnostics.ipynb"
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