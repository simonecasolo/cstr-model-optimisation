"""Build (and optionally execute) notebooks/system_I/36_production_measured_inlet.ipynb.

Production posterior with measured inlet temperatures (eps = 0.1 K), budget study with nb31's
rule v2.1, and acceptance on the nb35 inlet grid. Stage functions in
scripts/nb36_production_measured_inlet.py (resumable). Needs nb31 (rule code), nb35 (test grid,
oracle) and the nb34/nb31 results for comparison.

Usage:  python scripts/build_nb_system_I_36.py            # build only
        python scripts/build_nb_system_I_36.py --execute  # build + execute headless (nbclient)
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "36_production_measured_inlet.ipynb"

HEADER = r"""# 36 — Production posterior with measured inlet temperatures (System I, N3, P2)

**Why.** Notebook 34 showed the nominal-inlet production posterior (nb31) is confidently wrong when
the inlet is not nominal (0.5 K on the coolant inlet → β biased by 7–13 posterior sds). Notebook 35
showed that giving the posterior the **measured** inlet temperatures removes that failure, with an
instrument requirement of a calibration error ≤ 0.1 K. **Design decision (user, 2026-11-09):** the
paper's production model measures Ti and Tci with class-AA RTDs (ε = 0.1 K); 0.05 and 0.2 K are
sensitivity results (nb35). The nb31 posterior stays as the *naive* nominal-inlet model (nb34, and
Claim 1 in nb07 at the nominal inlet).

**Design.** Plant and sensors as nb35 (inlet varies ±2 K per window; Ci ±1% unmeasured; measured
Ti, Tci with a 0.1 K calibration offset + averaged sensor noise); 34-D features (29 summaries,
plug-in estimators with the measured inlet, measured Ti, Tci). Bank 160k; budgets 5k–160k; 5 seeds;
test: SBC (1000) + scenario grid (Sc1–Sc5, Sc7 × 50) with the inlet varying.

## Rules — fixed before running
* **Budget and deployed object:** nb31 rule **v2.1**, unchanged (persistent convergence incl. the
  precision criterion; the deployed posterior must itself pass calibration (d)).
* **Acceptance:** on the nb35 inlet grid (true Tci, Ti each ±0.5, ±1, ±2 K; Sc1–Sc4), nb35's
  deployability criteria (β cov90 ≥ 0.80; class accuracy ≥ nb35 oracle − 0.10). At most one
  violating cell, and no v2.1 flag; otherwise **STOP** for the user.

**Running.** ≈ 4–5 h (bank, 30 trainings, evaluation). Re-running resumes.
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
import importlib, system_I_config, ws_protocols, ws_eval, ws_metrics, ws_inlet, nb29_ws_study, nb31_budget_convergence, nb35_measured_inlet, nb36_production_measured_inlet
for _m in (system_I_config, ws_protocols, ws_eval, ws_metrics, ws_inlet, nb29_ws_study, nb31_budget_convergence, nb35_measured_inlet, nb36_production_measured_inlet):
    importlib.reload(_m)
P = nb36_production_measured_inlet
assert P.SENSOR == "N3" and P.EPS == 0.1, (P.SENSOR, P.EPS)
R, FIGS = ROOT / "results", ROOT / "figures"
THREADS = 4
T0 = time.time()
print("sensor", P.SENSOR, "| protocol", P.PROTOCOL, "| features", P.FEATURES, "| eps", P.EPS, "K | outputs", P.TAG)"""),
    ("md", "## 1. Bank, test set, training (resumable)"),
    ("code", """P.stage_bank()
P.stage_test()
P.stage_train(THREADS)
print(f"{(time.time() - T0) / 60:.1f} min")"""),
    ("md", "## 2. Evaluation across budgets"),
    ("code", """P.stage_eval()
m = pd.read_csv(R / f"{P.TAG}_metrics.csv"); m["member"] = m["member"].astype(str)
ens = m[m.member == "ensemble"].set_index("budget").sort_index()
display(ens[["ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p", "tarp_dev", "cov90_a", "cov90_b", "rmse_norm",
             "macro_f1", "sda_Sc1", "sdb_Sc1", "sda_Sc4", "sdb_Sc4", "corr_Sc4"]].round(4))
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for s in ("Sc1", "Sc2", "Sc3", "Sc4"):
    axes[0].plot(ens.index, ens[f"sdb_{s}"], "o-", label=s); axes[1].plot(ens.index, ens[f"sda_{s}"], "o-", label=s)
for ax, t in zip(axes[:2], ("β posterior sd", "α posterior sd")):
    ax.set_xscale("log"); ax.set_xlabel("simulation budget"); ax.set_title(t); ax.legend(fontsize=8)
for k in ("ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p"):
    axes[2].plot(ens.index, ens[k], "o-", label=k)
axes[2].axhline(0.01, color="C3", ls="--"); axes[2].set_xscale("log"); axes[2].set_yscale("log")
axes[2].set_title("calibration (ensemble)"); axes[2].legend(fontsize=8)
fig.savefig(FIGS / f"{P.TAG}_budget_convergence.png", dpi=150); plt.show()"""),
    ("md", "## 3. Production decision (nb31 rule v2.1, unchanged)"),
    ("code", """res = P.stage_decide()
if res["flag"]:
    raise RuntimeError("STOP: v2.1 decision flagged -- user review: " + str(res["flag"]))"""),
    ("md", """## 4. Acceptance on the nb35 inlet grid, and comparison with the naive nominal-inlet model
The same windows as nb35 (true Tci, Ti each ±0.5, ±1, ±2 K). Reference rows: nb35's oracle and the
nb31 nominal-inlet production posterior on those windows (from `35_N3_metrics.csv`)."""),
    ("code", """acc, acc_res = P.stage_accept()
acc_g = acc[acc.short.isin(["Sc1", "Sc2", "Sc3", "Sc4"])]
for k in ("bias_b", "z_bias_b", "cov90_truth_b", "sd_b", "class_acc"):
    print(k)
    display(acc_g.pivot_table(index="condition", columns="short", values=k).reindex(list(P.M.CONDITIONS)).round(3))
m35 = pd.read_csv(R / "35_N3_metrics.csv"); m35["short"] = m35.scenario.str.split("_").str[0]
ref = m35[m35.variant.isin(["oracle", "eps0.1", "production"]) & m35.short.isin(["Sc1", "Sc2", "Sc3", "Sc4"])]
comp = pd.concat([acc_g.assign(variant="nb36 production (eps 0.1, 160k study)"), ref])
summ = comp.groupby(["variant"]).agg(cov90_b_min=("cov90_truth_b", "min"), cov90_b_mean=("cov90_truth_b", "mean"),
                                     abs_bias_b=("bias_b", lambda v: np.abs(v).mean()), sd_b=("sd_b", "median"),
                                     class_acc=("class_acc", "mean"))
display(summ.round(3))
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")"""),
    ("md", """## 5. Interpretation — to be written after the run
Write toward the storyline (`docs/Manuscript/system_i_rerun_plan.md`):
1. **Budget and calibration** of the measured-inlet model (rule v2.1 verdict; does the extra nuisance
   need a larger budget than nb31's 40k? does the ensemble pass TARP where nb35's 40k oracle failed?).
2. **Precision**: β and α sd at production, against nb35 (eps0.1 at 40k, oracle) and the naive
   nominal-inlet model (0.015 at the nominal inlet only).
3. **Robustness**: coverage, bias and class accuracy over ±2 K of true inlet error (acceptance), next
   to the naive model's collapse (nb34/nb35 production rows).
4. **For the paper**: this is the production model; the headline β precision and accuracy figures
   come from here; the nominal-inlet posterior is the naive model of the silent-failure section.
5. **Follow-ups** (not run here): Phase 3 notebooks use this posterior; Claim 1 (nb07) remains a
   nominal-inlet comparison (both models assume the nominal inlet), which should be stated."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    # Keep the written section-5 commentary when the notebook already exists (a rebuild would otherwise overwrite it
    # with the placeholder; added 2026-11-09). The placeholder text is never "kept".
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 5. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 5. Interpretation"):
                    c.source = kept[-1]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    if execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=None, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}})   # cwd = notebooks/system_I
        try:
            client.execute()
        finally:
            nbformat.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print(build(execute="--execute" in sys.argv))
