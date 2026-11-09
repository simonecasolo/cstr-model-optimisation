"""Build / execute notebooks/system_I/11_fault_classification_N3.ipynb (System I, Phase 3 step b).

Production posterior (nb36, measured inlet) on D1 and on a larger evaluation set L1; naive posterior in a sensitivity section.
Rules and checks fixed in docs/Manuscript/system_i_N3_runbook.md section 2f (row b). A written section-"6. Interpretation"
is kept when the notebook is rebuilt.

    python scripts/build_nb_system_I_11.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "11_fault_classification_N3.ipynb"

HEADER = """# 11 — Fault classification with the production posterior (System I, N3, measured inlet)

Replaces the classification parts of the old notebooks 05a and 11. **Production posterior** = notebook 36 (20k, 5-seed ensemble,
measured inlet temperatures as inputs). **Data:** D1 (`observations_measured_inlet.npz`, 300 windows, notebook 02b) and a larger
evaluation set **L1** (1,000 windows per scenario, same generator and inlet distribution as D1, new seeds, not stored) for
standard errors of about 0.01. The **naive** posterior (nb31, nominal inlet assumed) appears only in the sensitivity section.

**Rules (fixed before running; runbook §2f row b).**
* Class = quadrant of (α, β) with threshold 0.90; predicted class = the quadrant holding most of the posterior mass.
* **Three headline numbers per scenario:** class accuracy, β 90% coverage at the truth, β sd (mean posterior sd), with binomial
  standard errors; the headline figures come from L1, D1 is the stored-dataset result.
* **Macro-F1 headline over Sc1–Sc5** (well-specified scenarios); all six scenarios and Sc7 (+2 K sensor drift, misspecified by design)
  are reported separately; per-class F1 and the confusion matrix.
* **Saturation split:** a window is "saturated" if more than 50% of its samples have Qc ≥ 395 L/min (with the inlet varying over ±2 K
  the plant often reaches the jacket limit, notebook 02b); the three numbers per group.
* **Checks (stop if violated):** β coverage at the truth ≥ 0.80 in each of Sc1–Sc4 on L1; overall prior leakage < 5%; everything finite.
* Not here: the LDA baseline and the pre-computed NUTS classification of 05a (the NUTS part waits for the NUTS baseline); the 30-day
  classification timeline of the old 11 moves to notebook 10.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
import importlib, system_I_config, ws_protocols, ws_eval, ws_metrics, ws_inlet, nb31_budget_convergence, nb35_measured_inlet, nb36_production_measured_inlet, ws_phase3
for _m in (system_I_config, ws_protocols, ws_eval, ws_metrics, ws_inlet, nb31_budget_convergence, nb35_measured_inlet, nb36_production_measured_inlet, ws_phase3):
    importlib.reload(_m)
we, wm, P3 = ws_eval, ws_metrics, ws_phase3
T0 = time.time(); R = ROOT / "results"
SCN = {1: "Sc1", 2: "Sc2", 3: "Sc3", 4: "Sc4", 5: "Sc5", 7: "Sc7"}
CLS = list(wm.CLASSES); N_DRAW = 2000
post_pm, prod = P3.load_production_posterior()
print("Production posterior:", prod["production_budget"], "simulations,", "ensemble of %d" % len(prod["members"]), "| summary set", prod["summary_set"], "| eps", prod["inlet_model"]["eps_K"], "K")

def evaluate(post, X, theta2, names, seed=0):
    smp, leak = we.sample_within_prior(post, X, N_DRAW, rng_seed=seed)
    return smp, leak

def three(smp, theta2, names, mask=None):
    \"\"\"Per scenario: n, class accuracy (+-SE), beta coverage (+-SE), beta sd, beta bias, alpha sd.\"\"\"
    if mask is None: mask = np.ones(len(theta2), bool)
    yt = wm.class_of_theta(theta2); yp = wm.class_of_samples(smp)
    cov = wm.credible_coverage_at_truth(smp, theta2, 0.90)[:, 1]; sd = smp.std(axis=1); mean = smp.mean(axis=1)
    rows = []
    for s in sorted(set(names)):
        m = (names == s) & mask
        if m.sum() == 0: continue
        n = int(m.sum()); acc = float((yp[m] == yt[m]).mean()); c = float(cov[m].mean())
        rows.append(dict(scenario=s, n=n, class_acc=acc, acc_se=(acc * (1 - acc) / n) ** 0.5, cov90_b=c, cov_se=(c * (1 - c) / n) ** 0.5,
                         sd_b=float(sd[m, 1].mean()), bias_b=float((mean[m, 1] - theta2[m, 1]).mean()), sd_a=float(sd[m, 0].mean())))
    return pd.DataFrame(rows)

def names_of(d): return np.array([SCN[int(i)] for i in d["scenario_id"]])
"""),
    ("md", "## 1. Production posterior on D1 (the stored dataset, 300 windows)"),
    ("code", """d1 = P3.load_d1(); n1 = names_of(d1); th1 = d1["theta"][:, 2:]
X1 = P3.production_features(d1)
smp1, leak1 = evaluate(post_pm, X1, th1, n1)
tab1 = three(smp1, th1, n1); display(tab1.round(3))
print("prior leakage (mean fraction of raw flow mass outside the prior): %.4f" % leak1.mean())
tab1.to_csv(R / "11_N3_D1_three_numbers.csv", index=False)
"""),
    ("md", "## 2. Larger evaluation set L1 (1,000 windows per scenario): the headline figures"),
    ("code", """t0 = time.time()
dL = P3.generate_d1(n_rep=1000, seed_base=3_800_000)
nL = names_of(dL); thL = dL["theta"][:, 2:]
XL = P3.production_features(dL)
smpL, leakL = evaluate(post_pm, XL, thL, nL, seed=1)
ytL, ypL = wm.class_of_theta(thL), wm.class_of_samples(smpL)
tabL = three(smpL, thL, nL); display(tabL.round(3))
print(f"L1: {len(thL)} windows simulated and evaluated in {time.time() - t0:.0f}s | prior leakage {leakL.mean():.4f}")
tabL.to_csv(R / "11_N3_L1_three_numbers.csv", index=False)
"""),
    ("md", "## 3. Checks (stop if violated)"),
    ("code", """assert np.isfinite(smp1).all() and np.isfinite(smpL).all() and np.isfinite(X1).all() and np.isfinite(XL).all()
bad = tabL[tabL.scenario.isin(["Sc1", "Sc2", "Sc3", "Sc4"]) & (tabL.cov90_b < 0.80)]
if len(bad):
    raise RuntimeError("STOP: beta coverage at the truth < 0.80 in " + ", ".join(f"{r.scenario} ({r.cov90_b:.2f})" for r in bad.itertuples()))
if leakL.mean() >= 0.05 or leak1.mean() >= 0.05:
    raise RuntimeError(f"STOP: prior leakage {max(leakL.mean(), leak1.mean()):.3f} >= 0.05")
print("checks passed: beta coverage >= 0.80 in Sc1-Sc4 (L1):", tabL[tabL.scenario.isin(["Sc1", "Sc2", "Sc3", "Sc4"])][["scenario", "cov90_b"]].round(3).values.tolist(),
      "| leakage %.4f / %.4f" % (leak1.mean(), leakL.mean()))
"""),
    ("md", "## 4. Macro-F1, per-class F1 and the confusion matrix"),
    ("code", """def per_class(yt, yp):
    rows = []
    for c in CLS:
        tp = int(((yp == c) & (yt == c)).sum()); fp = int(((yp == c) & (yt != c)).sum()); fn = int(((yp != c) & (yt == c)).sum())
        rows.append(dict(cls=c, n_true=int((yt == c).sum()), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1)))
    return pd.DataFrame(rows)
out = []
for lab, d_th, d_n, d_smp in (("L1 (1,000 per scenario)", thL, nL, smpL), ("D1 (50 per scenario)", th1, n1, smp1)):
    yt, yp = wm.class_of_theta(d_th), wm.class_of_samples(d_smp)
    for grp, m in (("Sc1-Sc5 (headline)", np.isin(d_n, ["Sc1", "Sc2", "Sc3", "Sc4", "Sc5"])), ("all six", np.ones(len(d_n), bool)), ("Sc7 only", d_n == "Sc7")):
        out.append(dict(set=lab, scenarios=grp, n=int(m.sum()), accuracy=float((yp[m] == yt[m]).mean()), macro_f1=wm.macro_f1(yt[m], yp[m])))
f1tab = pd.DataFrame(out); display(f1tab.round(3)); f1tab.to_csv(R / "11_N3_macro_f1.csv", index=False)
m5 = np.isin(nL, ["Sc1", "Sc2", "Sc3", "Sc4", "Sc5"]); ytL5, ypL5 = ytL[m5], ypL[m5]
print("Per-class F1, L1, Sc1-Sc5:"); pc = per_class(ytL5, ypL5); display(pc.round(3)); pc.to_csv(R / "11_N3_per_class_f1.csv", index=False)
cm = pd.crosstab(pd.Series(ytL5, name="true"), pd.Series(ypL5, name="predicted")).reindex(index=CLS, columns=CLS, fill_value=0)
print("Confusion matrix (counts), L1, Sc1-Sc5:"); display(cm); print("row-normalised:"); display((cm.div(cm.sum(1).replace(0, 1), axis=0)).round(3))
cm.to_csv(R / "11_N3_confusion_L1.csv")
"""),
    ("md", "## 5. Saturation split (L1): the same three numbers for unsaturated and saturated windows"),
    ("code", """satfrac = (dL["x"][:, :, 3] >= 395).mean(axis=1); sat = satfrac > 0.5
parts = []
for lab, mk in (("unsaturated", ~sat), ("saturated (>50% of samples at Qc >= 395)", sat)):
    t = three(smpL, thL, nL, mask=mk); t.insert(0, "group", lab); parts.append(t)
split = pd.concat(parts); display(split.round(3)); split.to_csv(R / "11_N3_L1_saturation_split.csv", index=False)
print("fraction of saturated windows per scenario:", {s: round(float(sat[nL == s].mean()), 3) for s in sorted(set(nL))})
"""),
    ("md", "## 6. Sensitivity: the naive posterior (nominal inlet assumed) on D1 and on D0"),
    ("code", """post_pn, prod_n = P3.load_naive_posterior()
print("Naive posterior:", prod_n["production_budget"], "simulations, ensemble of %d" % len(prod_n["members"]), "| summary set", prod_n["summary_set"])
smp_n1, _ = evaluate(post_pn, P3.naive_features(d1), th1, n1)
naive_d1 = three(smp_n1, th1, n1); naive_d1.insert(0, "data", "D1 (inlet varies, naive posterior assumes nominal)")
d0 = P3.load_d0(); cl = np.isin(d0["scenario_id"], [1, 2, 3, 4, 5, 7])
d0c = {k: (v[cl] if hasattr(v, "shape") and v.shape[:1] == cl.shape else v) for k, v in d0.items()}
n0 = names_of(d0c); th0 = d0c["theta"][:, 2:]
smp_n0, _ = evaluate(post_pn, P3.naive_features(d0c), th0, n0)
naive_d0 = three(smp_n0, th0, n0); naive_d0.insert(0, "data", "D0 (nominal inlet, naive posterior valid)")
sens = pd.concat([naive_d0, naive_d1]); display(sens.round(3)); sens.to_csv(R / "11_N3_naive_sensitivity.csv", index=False)
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 6b. Interpretation — to be written after the run
Toward the storyline: the production posterior's three numbers per scenario (L1) and what limits them (the class threshold distance, the β sd at ε = 0.1 K,
the saturated mixture); Sc7 as the silent failure of an unmodelled sensor drift; the naive posterior on D1 as the inlet silent failure on the paper's own data
(sensitivity only); what supports the "reliable" half and what documents a failure mode."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 6b. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 6b. Interpretation"):
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
