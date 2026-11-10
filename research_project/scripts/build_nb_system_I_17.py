"""Build / execute notebooks/system_I/17_supplementary_figures_N3.ipynb (System I, user decision 4, 2026-11-10).

The SI figures regenerated from STORED data and results only. Writes ONLY to figures/si_N3/; never to data/ or scenario_configs.csv
(SHA-256 of every file in data/ is recorded before and compared after). Rules: docs/Manuscript/system_i_N3_runbook.md section 2f (row g-17b).
The old notebook 17 (which rewrites data/observations.npz) is left untouched.

    python scripts/build_nb_system_I_17.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "17_supplementary_figures_N3.ipynb"

HEADER = """# 17 — Supplementary-information figures from the stored N3 data and results (System I)

The old notebook 17 regenerated the SI figures by re-simulating the dataset and **rewriting `data/observations.npz` and `data/scenario_configs.csv`**; run now it would overwrite the regenerated D0. This notebook only **reads** stored data and results and writes only to `figures/si_N3/`
(the SI keeps citing the files in `figures/` until the swap is decided). **Safety checks (fixed before running; runbook §2f row g-17b):** the SHA-256 of every file in `data/` is recorded first and compared at the end (any difference stops the notebook); `save()` refuses any path outside `figures/si_N3/`.

| Figure | File | SI section | Source (stored) |
|---|---|---|---|
| 1 | `02_scenarios_overview.png` | S1 | D0 raw windows, Sc1–Sc5, Sc7 |
| 2 | `02_headline_summary.png` | S1 | D0, T_mean − T_sp and Qc_mean per scenario |
| 3 | `04_simulator_sanity.png` | S2 | prior-predictive cloud of the notebook-36 SBC test set against the D1 windows |
| 4 | `03_pca.png` | S4 | D0, PCA of the 29 base features |
| 5 | `04_sensitivity_scatter.png` | S3 | β posterior means of D1 Sc1/Sc2 for the stored notebook-36 budget ensembles |
| 6 | `18_ablation_study.png` | S? | notebook 18's feature-group ablation |
| 7 | `04_sbc_ranks.png` | S6 | rank histograms of the production ensemble on the stored notebook-36 SBC test set |
"""

CELLS = [
    ("md", HEADER),
    ("code", """import hashlib, json, os, sys, time, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
warnings.filterwarnings("ignore")
DATA, R = ROOT / "data", ROOT / "results"; FIGS = ROOT / "figures" / "si_N3"; FIGS.mkdir(parents=True, exist_ok=True); T0 = time.time()
def sha_all(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}
SHA_BEFORE = sha_all(DATA); print(f"recorded SHA-256 of {len(SHA_BEFORE)} files in data/ (incl. scenario_configs.csv: {'scenario_configs.csv' in SHA_BEFORE})")
def save(fig, name):
    path = (FIGS / name).resolve(); assert FIGS.resolve() in path.parents, f"refusing to write outside {FIGS}: {path}"
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", path.relative_to(ROOT))
import importlib, ws_eval, ws_metrics, ws_phase3, nb36_production_measured_inlet, summary_variants
for _m in (ws_eval, ws_metrics, ws_phase3, nb36_production_measured_inlet, summary_variants):
    importlib.reload(_m)
we, wm, P3, P36, sv = ws_eval, ws_metrics, ws_phase3, nb36_production_measured_inlet, summary_variants
from cstr_sbi.scenarios import SCENARIO_CONFIGS
d0, d1 = P3.load_d0(), P3.load_d1(); t = d0["t"]; sid0 = d0["scenario_id"]
name = {s.id: s.name for s in SCENARIO_CONFIGS.values()}
"""),
    ("md", "## Figure 1 — scenarios overview (D0 raw windows)"),
    ("code", """fig, axes = plt.subplots(6, 4, figsize=(13, 12), constrained_layout=True); labs = ["C [mol/L]", "T [K]", "Tc [K]", "Qc [L/min]"]
for row, sc in zip(axes, (1, 2, 3, 4, 5, 7)):
    x = d0["x"][sid0 == sc]; a_, b_ = d0["theta"][sid0 == sc][0, 2:]
    for ch, ax in enumerate(row):
        for i in range(6): ax.plot(t, x[i, :, ch], lw=.6, alpha=.7)
        ax.plot(t, x[:6, :, ch].mean(0), "k", lw=1.2); ax.set_ylabel(labs[ch]); ax.grid(alpha=.25)
    row[0].text(-0.32, .5, f"Sc{sc} {name[sc]}\\n(α={a_:.2f}, β={b_:.2f})", transform=row[0].transAxes, ha="right", va="center", fontsize=8)
for ax in axes[-1]: ax.set_xlabel("t [min]")
fig.suptitle("Closed-loop scenarios (D0: nominal inlet, P2, N3 sensors) — 6 replicates each"); save(fig, "02_scenarios_overview.png")
"""),
    ("md", "## Figure 2 — headline summary (D0)"),
    ("code", """rows = []
for sc in sorted(set(sid0)):
    x = d0["x"][sid0 == sc]; rows.append(dict(id=int(sc), name=name[sc], T_dev=float(x[:, :, 1].mean() - 312.5), T_std=float(x[:, :, 1].std()), Qc_mean=float(x[:, :, 3].mean()), Qc_std=float(x[:, :, 3].std())))
sm = pd.DataFrame(rows); fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
a1.bar(sm.name, sm.T_dev, yerr=sm.T_std, capsize=3, color="C0", alpha=.85); a1.axhline(0, color="k", lw=.6); a1.set_ylabel("T_mean − T_sp [K]"); a1.tick_params(axis="x", rotation=45); a1.grid(alpha=.3, axis="y")
a2.bar(sm.name, sm.Qc_mean, yerr=sm.Qc_std, capsize=3, color="C2", alpha=.85); a2.axhline(400, color="C3", ls="--", lw=.7, label="Qc_max = 400"); a2.set_ylabel("Qc_mean [L/min]"); a2.tick_params(axis="x", rotation=45); a2.legend(fontsize=8); a2.grid(alpha=.3, axis="y")
fig.suptitle("Headline (D0): T regulated, Qc carries the fault signal"); save(fig, "02_headline_summary.png"); display(sm.round(3))
"""),
    ("md", "## Figure 3 — prior-predictive check (stored notebook-36 SBC test set against the D1 windows)"),
    ("code", """test = we.load_npz(P36.OUT / "test.npz"); Xs = test["x_sbc"]; Xd = P3.production_features(d1)
names = sv.BASE29 + sv.EST + ("Ti_meas", "Tci_meas"); show = ["UA_eff_proxy", "k0_eff_proxy", "C_mean", "Qc_mean"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True); cols = {1: "#ff7f0e", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd", 5: "#8c564b", 7: "#e377c2"}
for ax, f in zip(axes, show):
    j = names.index(f); ax.hist(Xs[:, j], bins=50, density=True, color="lightgrey", label="prior predictive (1,000 SBC draws)")
    for sc, c in cols.items(): ax.hist(Xd[d1["scenario_id"] == sc, j], bins=15, density=True, histtype="step", color=c, lw=1.3, label=f"D1 Sc{sc}")
    ax.set_title(f); ax.grid(alpha=.3)
axes[0].legend(fontsize=6); fig.suptitle("Prior-predictive check (production model: P2, N3, measured inlet)"); save(fig, "04_simulator_sanity.png")
inside = {f: float(np.mean((Xd[:, names.index(f)] >= Xs[:, names.index(f)].min()) & (Xd[:, names.index(f)] <= Xs[:, names.index(f)].max()))) for f in show}
print("share of D1 windows inside the range of the prior-predictive sample:", {k: round(v, 3) for k, v in inside.items()})
"""),
    ("md", "## Figure 4 — PCA of the 29 base features (D0)"),
    ("code", """from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
B = we.features(d0["x"], np.asarray(t, np.float32), "S0_base29"); keep = B.std(0) > 1e-12; Z = StandardScaler().fit_transform(B[:, keep]); pca = PCA(4).fit(Z); Y = pca.transform(Z); cm = plt.get_cmap("tab10")
fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for sc in sorted(set(sid0)):
    m = sid0 == sc; mk = "o" if d0["mode"][m][0] == "closed_loop" else "s"
    ax[0].scatter(Y[m, 0], Y[m, 1], s=22, alpha=.7, color=cm(sc % 10), marker=mk, label=f"Sc{sc} {name[sc]}"); ax[1].scatter(Y[m, 2], Y[m, 3], s=22, alpha=.7, color=cm(sc % 10), marker=mk)
ax[0].set(xlabel=f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)", ylabel=f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)"); ax[1].set(xlabel=f"PC3 ({100 * pca.explained_variance_ratio_[2]:.1f}%)", ylabel=f"PC4 ({100 * pca.explained_variance_ratio_[3]:.1f}%)")
ax[0].legend(fontsize=7, ncol=2); [a.grid(alpha=.3) for a in ax]; fig.suptitle("PCA of the standardised 29 base summaries (D0)"); save(fig, "03_pca.png")
print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 3), "| PC1+PC2:", round(100 * pca.explained_variance_ratio_[:2].sum(), 1), "%")
"""),
    ("md", "## Figure 5 — budget sensitivity (stored notebook-36 posteriors at each budget, 5-seed ensembles)"),
    ("code", """Xp = P3.production_features(d1); sid1 = d1["scenario_id"]; BUD = [5000, 10000, 20000, 40000, 80000, 160000]; res = {}
for N in BUD:
    mem = [pickle.load(open(P36._post_path(N, s), "rb"))["posterior"] for s in P36.SEEDS]; post = we.EnsemblePosterior(mem)
    smp, _ = we.sample_within_prior(post, Xp[np.isin(sid1, [1, 2])], 500, rng_seed=N); res[N] = (smp.mean(1)[:, 1], smp.std(1)[:, 1])
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)
for k, (sc, bt) in enumerate(((1, 1.0), (2, 0.85))):
    m = sid1[np.isin(sid1, [1, 2])] == sc
    for bi, N in enumerate(BUD): ax[k].plot(np.full(m.sum(), bi) + np.random.default_rng(bi).uniform(-.15, .15, m.sum()), res[N][0][m], ".", color="C0", alpha=.4)
    ax[k].plot(range(len(BUD)), [res[N][0][m].mean() for N in BUD], "k-o", ms=4); ax[k].axhline(bt, color="g", ls="--", label=f"true β = {bt}"); ax[k].set_xticks(range(len(BUD))); ax[k].set_xticklabels([f"{N // 1000}k" for N in BUD])
    ax[k].set_title(f"D1 Sc{sc}: β̂ (posterior mean) per window"); ax[k].set_xlabel("training simulations per member"); ax[k].legend(); ax[k].grid(alpha=.3)
fig.suptitle("Budget sensitivity of the production posterior (5-seed ensembles, stored)"); save(fig, "04_sensitivity_scatter.png")
print({N: (round(float(res[N][0][sid1[np.isin(sid1, [1, 2])] == 2].mean()), 3), round(float(res[N][1].mean()), 3)) for N in BUD}, "(mean β̂ in Sc2, mean posterior sd)")
"""),
    ("md", "## Figure 6 — feature-group ablation (notebook 18, stored)"),
    ("code", """ab = pd.read_csv(R / "18_N3_ablation.csv").set_index("set"); fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
ab[["sd_a", "sd_b"]].plot.barh(ax=ax[0]); ax[0].set_title("posterior sd by feature set"); ab[["macro_f1", "accuracy"]].plot.barh(ax=ax[1]); ax[1].set_xlim(.4, 1); ax[1].set_title("scenario-grid macro-F1 / accuracy"); [a.grid(alpha=.3) for a in ax]
save(fig, "18_ablation_study.png")
"""),
    ("md", "## Figure 7 — SBC rank histograms of the production ensemble (stored notebook-36 SBC test set)"),
    ("code", """post, prod = P3.load_production_posterior(); th_s = test["theta_sbc"]; smp, _ = we.sample_within_prior(post, Xs, 1000, rng_seed=0)
r = wm.sbc_ranks(smp, th_s); inner = wm.interior_mask(th_s, we.LO, we.HI); ks_all = wm.sbc_ks_pvalues(r, 1000, seed=0); ks_in = wm.sbc_ks_pvalues(r[inner], 1000, seed=0)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
for j, nm in enumerate(("α", "β")):
    ax[j].hist(r[:, j] / 1000, bins=20, color="C0"); ax[j].axhline(len(r) / 20, color="k", ls="--", lw=.8); ax[j].set_title(f"{nm}: KS p = {ks_all[j]:.3f} (all), {ks_in[j]:.3f} (interior, n = {int(inner.sum())})"); ax[j].set_xlabel("rank / 1000")
fig.suptitle("SBC of the production ensemble (20k × 5 seeds; 1,000 test draws)"); save(fig, "04_sbc_ranks.png")
m36 = pd.read_csv(R / "36_N3_metrics.csv"); ref = m36[(m36.budget == 20000) & (m36.member == "ensemble")].iloc[0]
print(f"KS p (interior): alpha {ks_in[0]:.3f}, beta {ks_in[1]:.3f} | stored nb36 metrics: {ref.ks_p_a_interior:.3f}, {ref.ks_p_b_interior:.3f}")
"""),
    ("md", "## Safety check: nothing in data/ changed"),
    ("code", """SHA_AFTER = sha_all(DATA)
changed = sorted(k for k in set(SHA_BEFORE) | set(SHA_AFTER) if SHA_BEFORE.get(k) != SHA_AFTER.get(k))
assert not changed, f"data/ changed: {changed}"
print(f"data/ unchanged: {len(SHA_AFTER)} files, all SHA-256 identical before and after (scenario_configs.csv included)")
json.dump({k: v for k, v in SHA_AFTER.items()}, open(R / "17_N3_data_sha256_after.json", "w"), indent=0)
print("figures written:", sorted(p.name for p in FIGS.iterdir())); print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
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
