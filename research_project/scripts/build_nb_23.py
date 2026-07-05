"""Build and execute notebook 23: Wu 2003 summary statistics and discriminability."""

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
        """# Notebook 23 -- Wu 2003 Summary Statistics and Discriminability

This notebook is the nb23 checkpoint from the Wu 2003 SBI extension plan. It
turns the nb22 observation windows into fixed-length feature vectors and checks
whether those features preserve the fault-discriminating structure needed by the
SBI notebooks.

The original plan described 55-D/61-D summaries for a 7/8-channel shortcut data
contract. The current explicit-loop dataset has more information because reflux
and boilup controller efforts are explicit channels:

- S-A: 10 channels, including `x_D`, `R_norm`, and `V_norm`.
- S-B: 9 channels, excluding `x_D` but retaining conventional controller effort.

Accordingly, nb23 uses the same summary design but updated dimensions:

- S-B: `9 channels x 6 per-channel features + 12 physics features = 66` features.
- S-A: `10 channels x 6 per-channel features + 12 physics features = 72` features.

The key outputs are saved for nb24/nb25:

- `data/wu2003_summary_features.npz`
- `data/wu2003_summary_feature_rankings.csv`
"""
    ),
    md("""## 1. Imports and data loading"""),
    code(
        """from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.precision", 5)

DATA_PATH = Path("data/wu2003_observations.npz")
assert DATA_PATH.exists(), "Run notebook 22 first to create data/wu2003_observations.npz"

with np.load(DATA_PATH, allow_pickle=True) as data:
    obs_sa = data["observations_sa"]
    obs_sb = data["observations_sb"]
    labels_raw = data["labels"]
    scenario_table_raw = data["scenario_table"]
    sa_channels = [str(x) for x in data["sa_channels"]]
    sb_channels = [str(x) for x in data["sb_channels"]]
    t_h = data["t_h"]
    n_replicates = int(np.asarray(data["n_replicates"]))
    noise_pct = float(np.asarray(data["noise_pct"]))

labels = pd.DataFrame.from_records(labels_raw)
scenario_table = pd.DataFrame.from_records(scenario_table_raw)

print("S-A observations:", obs_sa.shape)
print("S-B observations:", obs_sb.shape)
print("Labels:", labels.shape)
print("Scenario table:", scenario_table.shape)
print("S-A channels:", sa_channels)
print("S-B channels:", sb_channels)
print(f"Replicates per scenario: {n_replicates}; sensor noise fraction: {noise_pct}")
"""
    ),
    md("""## 2. Label taxonomy"""),
    code(
        """def fault_family(row):
    reactor = (row["alpha"] < 0.95) or (row["beta_r"] < 0.95)
    column = (row["eta_col"] < 0.95) or (row["xi_reb"] < 0.95)
    feed = abs(row["z_A0_eff"] - 0.90) > 0.03
    if not (reactor or column or feed):
        return "healthy"
    parts = []
    if reactor:
        parts.append("reactor")
    if column:
        parts.append("column")
    if feed:
        parts.append("feed")
    return "+".join(parts)


labels = labels.copy()
labels["fault_family"] = labels.apply(fault_family, axis=1)
labels["is_open_loop"] = labels["mode"].eq("open_loop")

label_summary = (
    labels.groupby(["mode", "fault_family", "scenario_name"])
    .size()
    .rename("n_windows_per_structure")
    .reset_index()
)
display(label_summary)
"""
    ),
    md("""## 3. Summary-statistic implementation"""),
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
            if not feature_names:
                pass
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
        treb = window[:, idx["T_reb"]]
        qreb = window[:, idx["Q_reb"]]
        fr = window[:, idx["F_R_norm"]]
        fb = window[:, idx["F_B_norm"]]
        r_effort = window[:, idx["R_norm"]]
        v_effort = window[:, idx["V_norm"]]

        ua_proxy = qj[-1] / max(abs(tr[-1] - tj[-1]), 1e-6)
        recycle_ratio = fr[-1]
        col_recovery = fb[-1] / max(1.0 + fr[-1], 1e-6)
        reb_intensity = qreb[-1] / max(fr[-1], 1e-6)
        reactor_conversion = fb[-1] * 0.0105 / 460.0
        recycle_excess = fr[-1] - 1.0
        tr_tj_ratio = tr[-1] / max(tj[-1], 1e-6)
        qj_slope = _slope(t, qj)
        corr_qj_fr = _safe_corr(qj, fr)
        corr_qreb_fr = _safe_corr(qreb, fr)

        physics_rows.append([
            ua_proxy,
            recycle_ratio,
            col_recovery,
            reb_intensity,
            reactor_conversion,
            recycle_excess,
            tr_tj_ratio,
            qj_slope,
            corr_qj_fr,
            corr_qreb_fr,
            r_effort[-1],
            v_effort[-1],
        ])

    X = np.hstack([np.asarray(rows), np.asarray(physics_rows)])
    names = feature_names + physics_names
    return X, names


X_sa, features_sa = summarize_windows(obs_sa, sa_channels, t_h)
X_sb, features_sb = summarize_windows(obs_sb, sb_channels, t_h)

print("S-A summary matrix:", X_sa.shape)
print("S-B summary matrix:", X_sb.shape)
print("S-A finite:", np.isfinite(X_sa).all())
print("S-B finite:", np.isfinite(X_sb).all())
"""
    ),
    md("""## 4. Standardisation and PCA"""),
    code(
        """def embed_pca(X, n_components=8):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=20260625)
    scores = pca.fit_transform(Xz)
    return scaler, pca, scores


scaler_sa, pca_sa, pca_scores_sa = embed_pca(X_sa)
scaler_sb, pca_sb, pca_scores_sb = embed_pca(X_sb)

pca_table = pd.DataFrame([
    {
        "structure": "S-A",
        "n_features": X_sa.shape[1],
        "PC1_var": pca_sa.explained_variance_ratio_[0],
        "PC2_var": pca_sa.explained_variance_ratio_[1],
        "PC1_PC2_cumulative": pca_sa.explained_variance_ratio_[:2].sum(),
        "PC1_to_PC5_cumulative": pca_sa.explained_variance_ratio_[:5].sum(),
    },
    {
        "structure": "S-B",
        "n_features": X_sb.shape[1],
        "PC1_var": pca_sb.explained_variance_ratio_[0],
        "PC2_var": pca_sb.explained_variance_ratio_[1],
        "PC1_PC2_cumulative": pca_sb.explained_variance_ratio_[:2].sum(),
        "PC1_to_PC5_cumulative": pca_sb.explained_variance_ratio_[:5].sum(),
    },
])
pca_table
"""
    ),
    md("""## 5. PCA scenario maps"""),
    code(
        """def plot_embedding(scores, labels, title):
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    families = sorted(labels["fault_family"].unique())
    for family in families:
        mask = labels["fault_family"].to_numpy() == family
        ax.scatter(scores[mask, 0], scores[mask, 1], s=18, alpha=0.65, label=family)
    open_mask = labels["is_open_loop"].to_numpy()
    ax.scatter(scores[open_mask, 0], scores[open_mask, 1], s=52, facecolors="none", edgecolors="k", linewidths=0.8, label="open-loop diagnostic")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    return fig, ax


plot_embedding(pca_scores_sa, labels, "S-A summary PCA")
plot_embedding(pca_scores_sb, labels, "S-B summary PCA")
plt.show()
"""
    ),
    md("""## 6. t-SNE nonlinear maps"""),
    code(
        """def run_tsne(X, seed=20260625):
    Xz = StandardScaler().fit_transform(X)
    try:
        return TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=seed,
        ).fit_transform(Xz)
    except TypeError:
        return TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            n_iter=1000,
            random_state=seed,
        ).fit_transform(Xz)


tsne_sa = run_tsne(X_sa)
tsne_sb = run_tsne(X_sb)
plot_embedding(tsne_sa, labels, "S-A summary t-SNE")
plot_embedding(tsne_sb, labels, "S-B summary t-SNE")
plt.show()
"""
    ),
    md("""## 7. Mutual-information ranking"""),
    code(
        """family_codes = labels["fault_family"].astype("category").cat.codes.to_numpy()
scenario_codes = labels["scenario_id"].astype("category").cat.codes.to_numpy()
parameter_targets = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]


def mi_table(X, feature_names, structure):
    Xz = StandardScaler().fit_transform(X)
    mi_family = mutual_info_classif(Xz, family_codes, random_state=20260625)
    mi_scenario = mutual_info_classif(Xz, scenario_codes, random_state=20260625)
    rows = []
    for i, feature in enumerate(feature_names):
        row = {
            "structure": structure,
            "feature": feature,
            "mi_fault_family": mi_family[i],
            "mi_scenario": mi_scenario[i],
        }
        for target in parameter_targets:
            row[f"mi_{target}"] = mutual_info_regression(
                Xz[:, [i]], labels[target].to_numpy(), random_state=20260625
            )[0]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["mi_total"] = df[[c for c in df.columns if c.startswith("mi_")]].sum(axis=1)
    return df.sort_values("mi_total", ascending=False).reset_index(drop=True)


mi_sa = mi_table(X_sa, features_sa, "S-A")
mi_sb = mi_table(X_sb, features_sb, "S-B")
mi_all = pd.concat([mi_sa, mi_sb], ignore_index=True)

display(mi_sa.head(20))
display(mi_sb.head(20))
"""
    ),
    md(
        """## 8. nb03-style LDA separability probe

Notebook 03 used cross-validated LDA accuracy as a compact supervised check that
the summary statistics separate operating scenarios. The same probe is useful
here because nb23 already has scenario and fault-family labels. LDA is not used
as an embedding for SBI; it is a cheap discriminability diagnostic for comparing
S-A and S-B feature sets.
"""
    ),
    code(
        """def lda_cv_accuracy(X, y, feature_indices=None, n_splits=5):
    if feature_indices is None:
        X_sub = X
        n_features = X.shape[1]
    else:
        feature_indices = np.asarray(feature_indices, dtype=int)
        X_sub = X[:, feature_indices]
        n_features = len(feature_indices)
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260625)
    scores = cross_val_score(clf, X_sub, y, cv=skf)
    return n_features, float(scores.mean()), float(scores.std(ddof=1))


def lda_probe_table(X, mi_df, structure):
    scenario_order = mi_df.sort_values("mi_scenario", ascending=False).index.to_numpy()
    family_order = mi_df.sort_values("mi_fault_family", ascending=False).index.to_numpy()
    total_order = mi_df.sort_values("mi_total", ascending=False).index.to_numpy()
    rows = []
    for target_name, y, order in [
        ("scenario", scenario_codes, scenario_order),
        ("fault_family", family_codes, family_order),
    ]:
        for subset, indices in [
            ("full", None),
            ("top-10 target MI", order[:10]),
            ("top-20 target MI", order[:20]),
            ("top-40 total MI", total_order[:40]),
        ]:
            n_features, mean_acc, std_acc = lda_cv_accuracy(X, y, indices)
            rows.append({
                "structure": structure,
                "target": target_name,
                "subset": subset,
                "n_features": n_features,
                "cv_accuracy_mean": mean_acc,
                "cv_accuracy_std": std_acc,
                "chance_accuracy": 1.0 / len(np.unique(y)),
            })
    return pd.DataFrame(rows)


lda_sa = lda_probe_table(X_sa, mi_sa, "S-A")
lda_sb = lda_probe_table(X_sb, mi_sb, "S-B")
lda_results = pd.concat([lda_sa, lda_sb], ignore_index=True)
lda_results.sort_values(["target", "cv_accuracy_mean", "structure"], ascending=[True, False, True])
"""
    ),
    md("""## 9. Feature shortlist"""),
    code(
        """TOP_K = 40
shortlist_sa = mi_sa.head(TOP_K).copy()
shortlist_sb = mi_sb.head(TOP_K).copy()

shortlist = pd.concat([shortlist_sa, shortlist_sb], ignore_index=True)
shortlist_summary = pd.DataFrame([
    {
        "structure": "S-A",
        "n_features_total": len(features_sa),
        "n_shortlisted": len(shortlist_sa),
        "top_feature": shortlist_sa.iloc[0]["feature"],
        "top_feature_mi_total": shortlist_sa.iloc[0]["mi_total"],
    },
    {
        "structure": "S-B",
        "n_features_total": len(features_sb),
        "n_shortlisted": len(shortlist_sb),
        "top_feature": shortlist_sb.iloc[0]["feature"],
        "top_feature_mi_total": shortlist_sb.iloc[0]["mi_total"],
    },
])
shortlist_summary
"""
    ),
    md("""## 10. Persist summary features"""),
    code(
        """out_npz = Path("data/wu2003_summary_features.npz")
out_csv = Path("data/wu2003_summary_feature_rankings.csv")
out_lda_csv = Path("data/wu2003_summary_lda_results.csv")

np.savez_compressed(
    out_npz,
    X_sa=X_sa,
    X_sb=X_sb,
    features_sa=np.asarray(features_sa, dtype=object),
    features_sb=np.asarray(features_sb, dtype=object),
    labels=labels.to_records(index=False),
    pca_sa=pca_scores_sa,
    pca_sb=pca_scores_sb,
    tsne_sa=tsne_sa,
    tsne_sb=tsne_sb,
    lda_results=lda_results.to_records(index=False),
)
mi_all.to_csv(out_csv, index=False)
lda_results.to_csv(out_lda_csv, index=False)

print(f"Wrote {out_npz} ({out_npz.stat().st_size / 1024**2:.2f} MiB)")
print(f"Wrote {out_csv}")
print(f"Wrote {out_lda_csv}")
"""
    ),
    md("""## 11. Acceptance checks"""),
    code(
        """expected_sa_features = len(sa_channels) * 6 + 12
expected_sb_features = len(sb_channels) * 6 + 12

acceptance = pd.DataFrame([
    {
        "check": "S-A summary shape",
        "expected": (obs_sa.shape[0], expected_sa_features),
        "observed": X_sa.shape,
        "status": "PASS" if X_sa.shape == (obs_sa.shape[0], expected_sa_features) else "FAIL",
    },
    {
        "check": "S-B summary shape",
        "expected": (obs_sb.shape[0], expected_sb_features),
        "observed": X_sb.shape,
        "status": "PASS" if X_sb.shape == (obs_sb.shape[0], expected_sb_features) else "FAIL",
    },
    {
        "check": "finite summaries",
        "expected": "all finite",
        "observed": bool(np.isfinite(X_sa).all() and np.isfinite(X_sb).all()),
        "status": "PASS" if bool(np.isfinite(X_sa).all() and np.isfinite(X_sb).all()) else "FAIL",
    },
    {
        "check": "PCA separates at least some fault variance",
        "expected": "PC1+PC2 cumulative > 0.25 for both structures",
        "observed": f"S-A={pca_sa.explained_variance_ratio_[:2].sum():.3f}, S-B={pca_sb.explained_variance_ratio_[:2].sum():.3f}",
        "status": "PASS" if min(pca_sa.explained_variance_ratio_[:2].sum(), pca_sb.explained_variance_ratio_[:2].sum()) > 0.25 else "FAIL",
    },
    {
        "check": "MI ranking non-degenerate",
        "expected": "positive MI in both structures",
        "observed": f"S-A top={mi_sa.iloc[0]['mi_total']:.3f}, S-B top={mi_sb.iloc[0]['mi_total']:.3f}",
        "status": "PASS" if mi_sa.iloc[0]["mi_total"] > 0.1 and mi_sb.iloc[0]["mi_total"] > 0.1 else "FAIL",
    },
    {
        "check": "LDA scenario probe beats chance",
        "expected": "full-feature scenario accuracy > 5x chance for both structures",
        "observed": lda_results[(lda_results["target"] == "scenario") & (lda_results["subset"] == "full")][["structure", "cv_accuracy_mean", "chance_accuracy"]].to_dict("records"),
        "status": "PASS" if all(
            row["cv_accuracy_mean"] > 5.0 * row["chance_accuracy"]
            for row in lda_results[(lda_results["target"] == "scenario") & (lda_results["subset"] == "full")].to_dict("records")
        ) else "FAIL",
    },
    {
        "check": "saved summary files exist",
        "expected": "npz, ranking csv, and LDA csv outputs exist",
        "observed": out_npz.exists() and out_csv.exists() and out_lda_csv.exists(),
        "status": "PASS" if out_npz.exists() and out_csv.exists() and out_lda_csv.exists() else "FAIL",
    },
])
acceptance
"""
    ),
    md(
        """## 12. Additional nb03-style physics proxies

Notebook 03 showed that hand-crafted inverse-physics features such as
`UA_eff_proxy` and `k0_eff_proxy` can carry more parameter information than generic
moments alone. This section tests the same idea for the Wu recycle plant without
changing the baseline nb23 handoff above.

Two additional features are appended diagnostically:

- `snowball_conversion_proxy = log(F_R_norm / F_B_norm)`, a compact proxy for recycle
  buildup relative to product withdrawal. It targets catalyst activity `alpha` and feed
  effects through the snowball mechanism.
- `column_effort_proxy = R_norm * V_norm / F_R_norm`, a separation-effort-per-recycle
  proxy. It targets `eta_col` and `xi_reb` through reflux/boilup compensation.

If they improve the MI or PCA diagnostics, they should be promoted into the formal
summary contract for nb24/nb25.
"""
    ),
    code(
        """def crafted_proxy_features(windows, channels):
    channel_index = {name: i for i, name in enumerate(channels)}
    rows = []
    for window in windows:
        fr = window[:, channel_index["F_R_norm"]]
        fb = window[:, channel_index["F_B_norm"]]
        r_effort = window[:, channel_index["R_norm"]]
        v_effort = window[:, channel_index["V_norm"]]
        snowball_conversion_proxy = np.log(max(fr[-1], 1e-8) / max(fb[-1], 1e-8))
        column_effort_proxy = r_effort[-1] * v_effort[-1] / max(fr[-1], 1e-8)
        rows.append([snowball_conversion_proxy, column_effort_proxy])
    return np.asarray(rows), ["snowball_conversion_proxy", "column_effort_proxy"]


crafted_sa, crafted_names = crafted_proxy_features(obs_sa, sa_channels)
crafted_sb, _ = crafted_proxy_features(obs_sb, sb_channels)

X_sa_aug = np.hstack([X_sa, crafted_sa])
X_sb_aug = np.hstack([X_sb, crafted_sb])
features_sa_aug = features_sa + crafted_names
features_sb_aug = features_sb + crafted_names

mi_sa_aug = mi_table(X_sa_aug, features_sa_aug, "S-A")
mi_sb_aug = mi_table(X_sb_aug, features_sb_aug, "S-B")

_, pca_sa_aug, pca_scores_sa_aug = embed_pca(X_sa_aug)
_, pca_sb_aug, pca_scores_sb_aug = embed_pca(X_sb_aug)

def max_target_mi(mi_df, target):
    return float(mi_df[f"mi_{target}"].max())


comparison_rows = []
for structure, X_base, X_aug, mi_base, mi_aug, pca_base, pca_aug in [
    ("S-A", X_sa, X_sa_aug, mi_sa, mi_sa_aug, pca_sa, pca_sa_aug),
    ("S-B", X_sb, X_sb_aug, mi_sb, mi_sb_aug, pca_sb, pca_sb_aug),
]:
    row = {
        "structure": structure,
        "baseline_features": X_base.shape[1],
        "augmented_features": X_aug.shape[1],
        "baseline_top_mi": mi_base.iloc[0]["mi_total"],
        "augmented_top_mi": mi_aug.iloc[0]["mi_total"],
        "delta_top_mi": mi_aug.iloc[0]["mi_total"] - mi_base.iloc[0]["mi_total"],
        "baseline_top10_mean_mi": mi_base.head(10)["mi_total"].mean(),
        "augmented_top10_mean_mi": mi_aug.head(10)["mi_total"].mean(),
        "delta_top10_mean_mi": mi_aug.head(10)["mi_total"].mean() - mi_base.head(10)["mi_total"].mean(),
        "baseline_PC1_PC2": pca_base.explained_variance_ratio_[:2].sum(),
        "augmented_PC1_PC2": pca_aug.explained_variance_ratio_[:2].sum(),
        "delta_PC1_PC2": pca_aug.explained_variance_ratio_[:2].sum() - pca_base.explained_variance_ratio_[:2].sum(),
    }
    for target in parameter_targets:
        row[f"delta_max_mi_{target}"] = max_target_mi(mi_aug, target) - max_target_mi(mi_base, target)
    comparison_rows.append(row)

augmentation_comparison = pd.DataFrame(comparison_rows)
crafted_rankings = pd.concat([
    mi_sa_aug[mi_sa_aug["feature"].isin(crafted_names)],
    mi_sb_aug[mi_sb_aug["feature"].isin(crafted_names)],
], ignore_index=True)

display(augmentation_comparison)
display(crafted_rankings.sort_values(["structure", "mi_total"], ascending=[True, False]))
"""
    ),
    md("""## 13. Augmented feature persistence and acceptance"""),
    code(
        """out_aug_npz = Path("data/wu2003_summary_features_augmented.npz")
out_aug_csv = Path("data/wu2003_summary_feature_rankings_augmented.csv")
out_aug_lda_csv = Path("data/wu2003_summary_lda_results_augmented.csv")

lda_sa_aug = lda_probe_table(X_sa_aug, mi_sa_aug, "S-A")
lda_sb_aug = lda_probe_table(X_sb_aug, mi_sb_aug, "S-B")
lda_results_aug = pd.concat([lda_sa_aug, lda_sb_aug], ignore_index=True)

mi_aug_all = pd.concat([mi_sa_aug, mi_sb_aug], ignore_index=True)
np.savez_compressed(
    out_aug_npz,
    X_sa=X_sa_aug,
    X_sb=X_sb_aug,
    features_sa=np.asarray(features_sa_aug, dtype=object),
    features_sb=np.asarray(features_sb_aug, dtype=object),
    crafted_feature_names=np.asarray(crafted_names, dtype=object),
    augmentation_comparison=augmentation_comparison.to_records(index=False),
    labels=labels.to_records(index=False),
    pca_sa=pca_scores_sa_aug,
    pca_sb=pca_scores_sb_aug,
    tsne_sa=tsne_sa,
    tsne_sb=tsne_sb,
    lda_results=lda_results_aug.to_records(index=False),
)
mi_aug_all.to_csv(out_aug_csv, index=False)
lda_results_aug.to_csv(out_aug_lda_csv, index=False)

improvement_flags = []
for _, row in augmentation_comparison.iterrows():
    improved = (
        row["delta_top_mi"] > 1e-9
        or row["delta_top10_mean_mi"] > 1e-9
        or max(row[f"delta_max_mi_{target}"] for target in parameter_targets) > 1e-9
    )
    improvement_flags.append(bool(improved))

augmentation_acceptance = pd.DataFrame([
    {
        "check": "S-A augmented summary shape",
        "expected": (obs_sa.shape[0], len(features_sa) + len(crafted_names)),
        "observed": X_sa_aug.shape,
        "status": "PASS" if X_sa_aug.shape == (obs_sa.shape[0], len(features_sa) + len(crafted_names)) else "FAIL",
    },
    {
        "check": "S-B augmented summary shape",
        "expected": (obs_sb.shape[0], len(features_sb) + len(crafted_names)),
        "observed": X_sb_aug.shape,
        "status": "PASS" if X_sb_aug.shape == (obs_sb.shape[0], len(features_sb) + len(crafted_names)) else "FAIL",
    },
    {
        "check": "finite augmented summaries",
        "expected": "all finite",
        "observed": bool(np.isfinite(X_sa_aug).all() and np.isfinite(X_sb_aug).all()),
        "status": "PASS" if bool(np.isfinite(X_sa_aug).all() and np.isfinite(X_sb_aug).all()) else "FAIL",
    },
    {
        "check": "crafted proxies tested for improvement",
        "expected": "at least one MI diagnostic improves in each structure",
        "observed": improvement_flags,
        "status": "PASS" if all(improvement_flags) else "DIAGNOSTIC",
    },
    {
        "check": "saved augmented summary files exist",
        "expected": "augmented npz, ranking csv, and LDA csv outputs exist",
        "observed": out_aug_npz.exists() and out_aug_csv.exists() and out_aug_lda_csv.exists(),
        "status": "PASS" if out_aug_npz.exists() and out_aug_csv.exists() and out_aug_lda_csv.exists() else "FAIL",
    },
])

print(f"Wrote {out_aug_npz} ({out_aug_npz.stat().st_size / 1024**2:.2f} MiB)")
print(f"Wrote {out_aug_csv}")
print(f"Wrote {out_aug_lda_csv}")
augmentation_acceptance
"""
    ),
    md(
        """## 14. Interpretation

The nb23 feature layer is consistent with the current explicit-loop data contract.
The summary dimensions are intentionally updated from the original 55-D/61-D plan
because nb22 now exposes reflux and boilup controller efforts. These features are
not nuisance additions: `R_norm` and `V_norm` are exactly the compensation channels
that make S-A and S-B real plant-wide control structures rather than measurement
projections of the same trajectory.

The saved feature matrices and MI ranking provide the handoff to nb24/nb25. nb24
should train the S-B posterior from the conventional 66-D summary set; nb25 should
train the S-A posterior from the analyzer-rich 72-D summary set and quantify the
value of `x_D` plus richer column compensation information.

The final diagnostic section also tests two additional nb03-style crafted features.
Those augmented 68-D/74-D matrices are saved separately so they can be promoted only
if the improvement diagnostics justify changing the formal nb24/nb25 contract.
The LDA tables mirror nb03's supervised separability check and should be read as
diagnostics, not as a proposed inference model.
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
    nb_path = repo_root / "notebooks" / "23_wu2003_summary_statistics.ipynb"

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
