"""Build / execute notebooks/system_I/03a_summary_statistics_N3.ipynb (System I, Phase 3 step g: the descriptive part of notebook 03).

Re-runs sections 1-9 of the old notebook 03 (heatmap, PCA, t-SNE, mutual information, LDA ablation by feature group) on the regenerated
D0 (29 base summaries, nominal inlet) and adds the production feature set S1m on D1 (34 features). Notebook 03 itself (with section 10, the
summary-set decision) is left untouched. Rules in docs/Manuscript/system_i_N3_runbook.md section 2f (row g-03).
A written "## 9. Interpretation" is kept on rebuild.

    python scripts/build_nb_system_I_03a.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "03a_summary_statistics_N3.ipynb"

HEADER = """# 03a — Summary statistics on the regenerated data (System I, N3)

Sections 1–9 of the old notebook 03 (M3) on the data of the N3 re-run; notebook 03 itself, with its section 10 (the summary-set decision S0 → S1), is left untouched. **D0** (`observations.npz`: Sc0–Sc7 × 50, nominal inlet, P2, N3 sensors) gives the 29 base summaries and,
with the three plug-in estimators, the naive set S1 (32). **D1** (Sc1–Sc5 and Sc7 × 50, measured inlet) gives the production set **S1m** (34: the 29 base summaries, the three plug-in estimators computed with the *measured* inlet, and the two measured temperatures minus 297 K).

**Content (descriptive; no decision rule; runbook §2f row g-03):** per-scenario feature heatmap; PCA and t-SNE of the standardised D0 features; mutual-information ranking with the scenario label (D0 29 features; D1 34); 5-fold LDA accuracy by feature group (the ablation of the old §6); the saturation-fraction features
(the old text said only Sc5 saturates; with the inlet varying more scenarios do); the plug-in estimators against the truth. **Checks:** finite; 50 windows per scenario; the standardised features have no constant column among the 29 base features of D0 except those listed.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
warnings.filterwarnings("ignore")
import importlib, ws_eval, ws_phase3, summary_variants
for _m in (ws_eval, ws_phase3, summary_variants):
    importlib.reload(_m)
we, P3, sv = ws_eval, ws_phase3, summary_variants
from cstr_sbi.scenarios import SCENARIO_CONFIGS
from cstr_sbi.summaries import FEATURE_GROUPS, feature_indices
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
R = ROOT / "results"; FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True); T0 = time.time()
BASE = list(sv.BASE29); EST = list(sv.EST); S1M_NAMES = BASE + EST + ["Ti_meas_minus_297", "Tci_meas_minus_297"]
d0, d1 = P3.load_d0(), P3.load_d1()
t = np.asarray(d0["t"], np.float32)
X0 = we.features(d0["x"], t, "S1_base_est")            # (400, 32): 29 base + 3 estimators, nominal inlet
X1 = P3.production_features(d1)                        # (300, 34): S1m
sid0, sid1 = d0["scenario_id"], d1["scenario_id"]
assert np.isfinite(X0).all() and np.isfinite(X1).all() and X0.shape == (400, 32) and X1.shape == (300, 34)
assert all((sid0 == s).sum() == 50 for s in range(8)) and all((sid1 == s).sum() == 50 for s in (1, 2, 3, 4, 5, 7))
name = {s.id: s.name for s in SCENARIO_CONFIGS.values()}
const0 = [BASE[i] for i in range(29) if X0[:, i].std() < 1e-12]
print("D0", X0.shape, "| D1", X1.shape, "| constant base features in D0:", const0)
"""),
    ("md", "## 2. Per-scenario feature heatmap (D0, 29 base features, z-scored over the dataset)"),
    ("code", """B0 = X0[:, :29]; keep0 = B0.std(0) > 1e-12
Z0 = np.zeros_like(B0); Z0[:, keep0] = StandardScaler().fit_transform(B0[:, keep0])
Zs = pd.DataFrame(Z0, columns=BASE).groupby(sid0).mean()
fig, ax = plt.subplots(figsize=(13, 4.5)); im = ax.imshow(Zs.values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
ax.set_yticks(range(len(Zs))); ax.set_yticklabels([f"Sc{i} {name[i]}" for i in Zs.index], fontsize=8); ax.set_xticks(range(29)); ax.set_xticklabels(BASE, rotation=90, fontsize=8)
fig.colorbar(im, ax=ax, label="z-score"); fig.tight_layout(); fig.savefig(FIGS / "03a_N3_feature_heatmap.png", dpi=140); plt.show()
"""),
    ("md", "## 3. PCA and t-SNE of the standardised D0 features"),
    ("code", """pca = PCA(n_components=5).fit(Z0); Y = pca.transform(Z0)
print("explained variance ratio PC1..5:", np.round(pca.explained_variance_ratio_, 3), "| cumulative 2:", round(float(pca.explained_variance_ratio_[:2].sum()), 3))
for k in range(2):
    o = np.argsort(-np.abs(pca.components_[k]))[:6]; print(f"PC{k+1} top loadings:", [(BASE[i], round(float(pca.components_[k, i]), 2)) for i in o])
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0, learning_rate="auto").fit_transform(Z0)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5)); cm = plt.get_cmap("tab10")
for k, (E_, ttl) in enumerate(((Y, "PCA"), (tsne, "t-SNE"))):
    for s in sorted(set(sid0)):
        m = sid0 == s; ax[k].scatter(E_[m, 0], E_[m, 1], s=20, alpha=.7, color=cm(s), marker="o" if d0["mode"][m][0] == "closed_loop" else "s", label=f"Sc{s} {name[s]}")
    ax[k].set_title(ttl + " of the 29 base summaries (D0)"); ax[k].grid(alpha=.3)
ax[0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(FIGS / "03a_N3_pca_tsne.png", dpi=140); plt.show()
"""),
    ("md", "## 4. Mutual information with the scenario label: D0 (29 base features) and D1 (34 production features)"),
    ("code", """mi0 = mutual_info_classif(Z0[:, keep0], sid0, random_state=0); n0 = [BASE[i] for i in np.where(keep0)[0]]
Z1 = StandardScaler().fit_transform(X1); mi1 = mutual_info_classif(Z1, sid1, random_state=0)
t0 = pd.DataFrame(dict(feature=n0, mi=mi0)).sort_values("mi", ascending=False); t1 = pd.DataFrame(dict(feature=S1M_NAMES, mi=mi1)).sort_values("mi", ascending=False)
print("D0, top 10:"); display(t0.head(10).round(3).reset_index(drop=True)); print("D1 (S1m), top 10:"); display(t1.head(10).round(3).reset_index(drop=True))
t0.to_csv(R / "03a_N3_mi_D0.csv", index=False); t1.to_csv(R / "03a_N3_mi_D1_S1m.csv", index=False)
"""),
    ("md", "## 5. LDA ablation by feature group (5-fold, scenario label)"),
    ("code", """def cv(Z, y, idx):
    sk = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(np.mean([LinearDiscriminantAnalysis().fit(Z[tr][:, idx], y[tr]).score(Z[te][:, idx], y[te]) for tr, te in sk.split(Z, y)]))
rows = [("D0: all 29 base", 29, cv(Z0[:, keep0], sid0, np.arange(keep0.sum())))]
names0 = np.array(BASE)[keep0]
for g, nm in FEATURE_GROUPS.items():
    if g == "all": continue
    idx = [i for i, n in enumerate(names0) if n in nm]
    if idx: rows.append((f"D0: group {g}", len(idx), cv(Z0[:, keep0], sid0, np.array(idx))))
for K in (5, 10, 15):
    o = [n0.index(f) for f in t0.feature.head(K)]; rows.append((f"D0: top-{K} MI", K, cv(Z0[:, keep0], sid0, np.array(o))))
nm1 = np.array(S1M_NAMES)
for lab, sel in (("D1: base 29", BASE), ("D1: base + 3 estimators", BASE + EST), ("D1: S1m (34)", S1M_NAMES), ("D1: estimators + measured temperatures only", EST + S1M_NAMES[-2:])):
    rows.append((lab, len(sel), cv(Z1, sid1, np.array([list(nm1).index(f) for f in sel]))))
abl = pd.DataFrame(rows, columns=["subset", "n_features", "lda_cv_accuracy"]); display(abl.round(3)); abl.to_csv(R / "03a_N3_lda_ablation.csv", index=False)
"""),
    ("md", "## 6. Saturation fractions per scenario (D0 nominal inlet against D1 inlet varying)"),
    ("code", """rows = []
for s in range(8):
    m0 = sid0 == s; r = dict(scenario=f"Sc{s}", D0_high=float(X0[m0, BASE.index("Qc_sat_high_frac")].mean()), D0_low=float(X0[m0, BASE.index("Qc_sat_low_frac")].mean()))
    m1 = sid1 == s
    if m1.any(): r.update(D1_high=float(X1[m1, BASE.index("Qc_sat_high_frac")].mean()), D1_low=float(X1[m1, BASE.index("Qc_sat_low_frac")].mean()))
    rows.append(r)
sat = pd.DataFrame(rows); display(sat.round(3)); sat.to_csv(R / "03a_N3_saturation_fractions.csv", index=False)
"""),
    ("md", "## 7. The plug-in estimators against the truth (window means: exp of the log-estimate)"),
    ("code", """rows = []
for lab, X, sid, d in (("D0 (nominal inlet assumed; true inlet nominal)", X0, sid0, d0), ("D1 (S1m: measured inlet in the estimators)", X1, sid1, d1)):
    th = d["theta"]
    for s in sorted(set(sid)):
        m = sid == s
        if d["mode"][m][0] != "closed_loop": continue
        rows.append(dict(data=lab, scenario=f"Sc{s}", alpha_true=float(th[m][0, 2]), beta_true=float(th[m][0, 3]), alpha_hat=float(np.exp(X[m, 29]).mean()),
                         beta_hat_jac=float(np.exp(X[m, 30]).mean()), beta_hat_rx=float(np.exp(X[m, 31]).mean()), sd_ln_alpha=float(X[m, 29].std()), sd_ln_beta_jac=float(X[m, 30].std()), sd_ln_beta_rx=float(X[m, 31].std())))
est = pd.DataFrame(rows); display(est.round(3)); est.to_csv(R / "03a_N3_plugin_estimators.csv", index=False)
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 9. Interpretation — to be written after the run
Toward the storyline: what the summaries separate (and what they do not), what the plug-in estimators add on D0 and D1, how the saturation features change with the inlet varying, and what this says about why the production set is S1m."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 9. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 9. Interpretation"):
                    c.source = kept[-1]
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    if execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print(build(execute="--execute" in sys.argv))
