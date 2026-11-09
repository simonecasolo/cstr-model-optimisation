"""Build / execute notebooks/system_I/18_feature_ablation_N3.ipynb (System I, Phase 3 step g).

Ablation of the production feature groups (S1m, 34 features): which of the summaries, the plug-in estimators and the measured temperatures
carry the information. Re-uses the notebook-36 bank, test set and training recipe; SBC / TARP / coverage / width / accuracy per set.
Rules in docs/Manuscript/system_i_N3_runbook.md section 2f (row g-18). A written "## 6. Interpretation" is kept on rebuild.
The old notebook 18 (budget, architecture and density-estimator-family ablation on the pre-fix simulator) is not re-run.

    python scripts/build_nb_system_I_18.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "18_feature_ablation_N3.ipynb"

HEADER = """# 18 — Which feature groups carry the information? Ablation of the production features (System I, N3)

The old notebook 18 ablated the training budget, the network and the density-estimator family on the pre-fix simulator. Under the N3 re-run the budget study is notebook 31/36 and the architecture is fixed (NSF 128/5); what is open is **which of the production features matter**.
The production set S1m has 34 features: the 29 base summaries (per-channel statistics, final-window means, aggregates, two physics proxies), three plug-in estimators of ln α and ln β computed with the measured inlet, and the two measured temperatures.

**Design (fixed before running; runbook §2f row g-18).** Same data as notebook 36: its 160k bank, its 20 000-simulation training recipe (the first 20 000 rows of the seed's permutation), its held-out test set (1,000 SBC draws and the scenario grid). Each feature set is trained with **3 seeds** and evaluated as the 3-member **ensemble** (single members are unreliable, notebooks 31/36).
Sets: **A** base 29 only (no inlet information); **B** base 29 + the two measured temperatures; **C** base 29 + the three estimators; **D** S1m (34, production); **E** estimators + measured temperatures only (5); and, from D, **drop one base group**: per-channel statistics (20), final-window means (4), aggregates (3), physics proxies (2).
Reported per set: SBC interior KS p and TARP p, 90% coverage, mean posterior sd of α and β, macro-F1 and accuracy on the scenario grid, β coverage and sd per scenario. **Checks:** finite; set D reproduces the notebook-36 ensemble's β sd within 15% (a different member count). No decision rule: a descriptive ablation.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import os, sys, time, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
warnings.filterwarnings("ignore")
import importlib, ws_eval, nb36_production_measured_inlet, summary_variants
for _m in (ws_eval, nb36_production_measured_inlet, summary_variants):
    importlib.reload(_m)
we, P36, sv = ws_eval, nb36_production_measured_inlet, summary_variants
from cstr_sbi.summaries import FEATURE_GROUPS, FEATURE_NAMES
R = ROOT / "results"; OUTD = R / "18_N3"; OUTD.mkdir(exist_ok=True); FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True); T0 = time.time()
torch.set_num_threads(4)
BUDGET, SEEDS = 20_000, (0, 1, 2)
th, X = P36.load_bank(); test = we.load_npz(P36.OUT / "test.npz")
assert X.shape[1] == 34 and np.isfinite(X).all()
BASE = list(sv.BASE29); NAMES = BASE + list(sv.EST) + ["Ti_meas", "Tci_meas"]
ix = lambda sel: [NAMES.index(n) for n in sel]
g = lambda name: [BASE[i] for i in range(29) if BASE[i] in FEATURE_GROUPS[name]]
def drop(group): return [n for n in NAMES if n not in g(group)]
SETS = {"A base 29 (no inlet information)": BASE, "B base 29 + measured temperatures": BASE + ["Ti_meas", "Tci_meas"], "C base 29 + 3 estimators": BASE + list(sv.EST),
        "D S1m (34, production)": NAMES, "E estimators + measured temperatures (5)": list(sv.EST) + ["Ti_meas", "Tci_meas"],
        "D minus per-channel statistics": drop("per_channel"), "D minus final-window means": drop("final_window"), "D minus aggregates": drop("aggregates"), "D minus physics proxies": drop("physics")}
for k, v in SETS.items(): print(f"{k}: {len(v)} features")
"""),
    ("md", "## 1. Train (3 seeds per set, 20,000 simulations; cached) and evaluate the 3-member ensembles"),
    ("code", """def path(si, s): return OUTD / f"post_set{si}_s{s}.pkl"
rows, sc_rows = [], []
for si, (name, sel) in enumerate(SETS.items()):
    cols = np.array(ix(sel)); members = []
    for s in SEEDS:
        f = path(si, s)
        if not f.exists():
            perm = np.random.default_rng(36_000 + s).permutation(len(th)); idx = perm[:BUDGET]
            post, meta = we.train_npe(th[idx], X[idx][:, cols], seed=s)
            pickle.dump({"posterior": post, "metadata": meta}, open(f, "wb"))
        members.append(pickle.load(open(f, "rb"))["posterior"])
    tsub = dict(test); tsub["x_sbc"] = test["x_sbc"][:, cols]; tsub["x_sc"] = test["x_sc"][:, cols]
    row, scr = we.evaluate(we.EnsemblePosterior(members), tsub, seed=0)
    rows.append(dict(set=name, n_features=len(cols), **{k: row[k] for k in ("ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p", "cov90_a", "cov90_b", "sd_a", "sd_b", "rmse_norm", "macro_f1", "accuracy", "leak_sbc_mean")}))
    sc_rows += [dict(set=name, **r) for r in scr]
    print(f"{name}: done {time.time() - T0:.0f}s", flush=True)
abl = pd.DataFrame(rows); display(abl.round(3)); abl.to_csv(R / "18_N3_ablation.csv", index=False)
pd.DataFrame(sc_rows).to_csv(R / "18_N3_ablation_scenarios.csv", index=False)
"""),
    ("md", "## 2. Check against notebook 36 and per-scenario β"),
    ("code", """m36 = pd.read_csv(R / "36_N3_metrics.csv"); ref = m36[(m36.budget == 20000) & (m36.member == "ensemble")].iloc[0]
d_row = abl[abl.set == "D S1m (34, production)"].iloc[0]
print(f"set D (3-member ensemble): sd_b {d_row.sd_b:.4f}, sd_a {d_row.sd_a:.4f} | notebook 36 (5-member ensemble): sd_b {ref.sd_b:.4f}, sd_a {ref.sd_a:.4f}")
assert abs(d_row.sd_b / ref.sd_b - 1) < 0.15, "reproduction check failed"
print("check passed: set D reproduces the notebook-36 beta sd within 15%")
sc = pd.DataFrame(sc_rows); sc["sc"] = sc.scenario.str.split("_").str[0]
display(sc.pivot_table(index="set", columns="sc", values="sd_b").round(3)); display(sc.pivot_table(index="set", columns="sc", values="cov90_truth_b").round(2))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5)); a = abl.set_index("set")
a[["sd_a", "sd_b"]].plot.barh(ax=ax[0]); ax[0].set_title("posterior sd by feature set"); a[["macro_f1", "accuracy"]].plot.barh(ax=ax[1]); ax[1].set_title("scenario-grid macro-F1 / accuracy"); ax[1].set_xlim(0.4, 1.0)
for a_ in ax: a_.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIGS / "18_N3_feature_ablation.png", dpi=150); plt.show()
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 6. Interpretation — to be written after the run
Toward the storyline: which feature groups matter (the measured temperatures? the plug-in estimators?), what is redundant, and what the base summaries alone cannot do when the inlet varies."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 6. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 6. Interpretation"):
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
