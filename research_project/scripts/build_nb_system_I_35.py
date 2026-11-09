"""Build (and optionally execute) notebooks/system_I/35_measured_inlet.ipynb (study R2, Part C).

The notebook IS the experiment: every stage runs in a notebook cell with the kernel's working
directory set to notebooks/system_I. Stage functions live in scripts/nb35_measured_inlet.py and
skip work already on disk (resumable). Needs the N3 production posterior (nb31).

Usage:  python scripts/build_nb_system_I_35.py            # build only
        python scripts/build_nb_system_I_35.py --execute  # build + execute headless (nbclient)
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "35_measured_inlet.ipynb"

HEADER = r"""# 35 — R2 Part C: give the posterior the measured inlet temperatures (System I)

**Why.** Notebook 34: the production posterior assumes the nominal inlet and is confidently wrong
when it is not — a 0.5 K error in the coolant inlet temperature Tci biases β by 7–13 posterior sds,
because the coolant warms by only ≈ 3 K in the jacket. Treating the inlet as unknown (nb34 Part B)
restores calibration but costs 5× in β precision. Real plants measure Tci and Ti, and their coolant
supply drifts by several K over a day or a season, so a fixed-inlet model would fail in practice
anyway. Here the posterior **receives the measured inlet temperatures**, and the question becomes:
**how accurate must those measurements be?** In the storyline: the fourth silent-failure source and
its remedy — a concrete instrument requirement for the practitioner checklist.

**Design** (N3 sensors, P2 protocol)
* Plant inlet varies per window: Ti, Tci ~ U(295, 299) K; Ci ± 1% (unmeasured).
* Inlet sensors: measured = true + calibration offset ε·N(0,1) (fixed per window) + window-mean
  white noise (0.2 K RTD averaged over 60 min).
* Features (34-D): the 29 summaries, the three plug-in estimators recomputed with the **measured**
  Ti and Tci, and the two measured temperatures.
* Variants, 3-seed ensembles at 40k, same draws and seeds: **oracle** (inlet known exactly),
  **ε = 0.05 K** (calibrated / matched pair), **ε = 0.1 K** (class-AA RTD), **ε = 0.2 K** (class-A RTD),
  **marginal** (no inlet measurement, inlet unknown). Reference: the N3 production posterior
  (assumes the nominal inlet).
* Tests: SBC set (1000 prior draws, inlet from the training distribution) and the scenario grid
  Sc1–Sc5 × 50 at fixed true inlets (nominal; Tci and Ti each ±0.5, ±1, ±2 K).

## Decision rule — fixed before running
A variant is **deployable** if it is calibrated on the SBC set (interior marginal KS p ≥ 0.01 for
α and β, TARP KS p ≥ 0.01) and, for every scenario Sc1–Sc4 and every inlet condition within ±2 K,
β 90% coverage at the truth ≥ 0.80 and class accuracy ≥ the oracle's − 0.10.
**Instrument requirement** = the largest deployable ε among {0.05, 0.1, 0.2} K.
**Production recommendation** (a user decision; the notebook changes nothing): ε = 0.1 K if
deployable, else ε = 0.05 K, else the marginal posterior with its precision cost.

**Running.** ≈ 1.5–2.5 h (120k simulations, 15 trainings, evaluation). Re-running resumes.
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
import importlib, system_I_config, ws_protocols, ws_eval, ws_metrics, ws_inlet, nb29_ws_study, nb31_budget_convergence, nb35_measured_inlet
for _m in (system_I_config, ws_protocols, ws_eval, ws_metrics, ws_inlet, nb29_ws_study, nb31_budget_convergence, nb35_measured_inlet):
    importlib.reload(_m)
M = nb35_measured_inlet
assert M.SENSOR == "N3", M.SENSOR
R, FIGS = ROOT / "results", ROOT / "figures"
THREADS = 4
T0 = time.time()
print("sensor", M.SENSOR, "| protocol", M.PROTOCOL, "| outputs", M.TAG, "| variants", list(M.VARIANTS))"""),
    ("md", "## 1. Simulate (training bank with varying inlet; test sets) and train the five variants"),
    ("code", """M.stage_bank()
M.stage_test()
M.stage_train(THREADS)
print(f"{(time.time() - T0) / 60:.1f} min")"""),
    ("md", "## 2. Evaluate (the five variants and the production posterior)"),
    ("code", """M.stage_eval()
sbc = pd.read_csv(R / f"{M.TAG}_sbc.csv")
display(sbc[["variant", "eps", "ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p", "cov90_a", "cov90_b",
             "sd_a_median", "sd_b_median", "rmse_norm", "leak_mean"]].round(4))
m = pd.read_csv(R / f"{M.TAG}_metrics.csv"); m["short"] = m.scenario.str.split("_").str[0]
order = list(M.CONDITIONS)
for k in ("bias_b", "z_bias_b", "cov90_truth_b", "sd_b", "class_acc"):
    print(k)
    display(m.pivot_table(index=["variant", "condition"], columns="short", values=k).reindex(order, level=1).round(3))"""),
    ("code", """vs = ["production"] + list(M.VARIANTS)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
for ax, fac in zip(axes[:2], ("Tci", "Ti")):
    conds = [f"{fac}-2.0", f"{fac}-1.0", f"{fac}-0.5", "nominal", f"{fac}+0.5", f"{fac}+1.0", f"{fac}+2.0"]
    xs = [-2, -1, -.5, 0, .5, 1, 2]
    for v in vs:
        d = m[(m.variant == v) & m.short.isin(M.GATED)].groupby("condition").cov90_truth_b.min().reindex(conds)
        ax.plot(xs, d.values, "o-", label=v)
    ax.axhline(0.8, color="C3", ls="--", lw=0.8); ax.axhline(0.9, color="grey", lw=0.8)
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel(f"true {fac} − 297 K"); ax.set_title(f"worst β 90% coverage over Sc1–Sc4 vs {fac}")
axes[0].legend(fontsize=8)
ax = axes[2]
s = sbc.set_index("variant")
eps_v = [v for v in M.VARIANTS if M.VARIANTS[v] is not None]
ax.plot([M.VARIANTS[v] for v in eps_v], [s.loc[v, "sd_b_median"] for v in eps_v], "o-", label="measured inlet (ε)")
ax.axhline(s.loc["marginal", "sd_b_median"], color="C3", ls="--", label="no inlet measurement (marginal)")
nom = m[(m.variant == "production") & (m.condition == "nominal") & m.short.isin(M.GATED)].sd_b.median()
ax.axhline(nom, color="grey", ls=":", label="production at the nominal inlet (assumed known)")
ax.set_xlabel("inlet-sensor calibration error ε [K]"); ax.set_ylabel("median β posterior sd")
ax.set_title("Price of not knowing the inlet"); ax.legend(fontsize=8)
fig.savefig(FIGS / f"{M.TAG}_coverage_and_precision.png", dpi=150); plt.show()"""),
    ("md", "## 3. Decision (pre-set rule)"),
    ("code", """dec = M.stage_decide()
for v, o in dec["variants"].items():
    if o["violations"]:
        print(v); display(pd.DataFrame(o["violations"]).round(3).head(12))
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")"""),
    ("md", """## 4. Interpretation — to be written after the run
Write toward the working storyline (`docs/Manuscript/system_i_rerun_plan.md`):
1. **Remedy.** Does conditioning on measured inlet temperatures remove the silent failure of nb34
   (coverage and bias against the production posterior at the same true inlets)?
2. **Instrument requirement.** The largest deployable ε and the β precision it buys, against the
   oracle (inlet known), the marginal posterior (inlet unknown) and the production posterior
   (inlet assumed nominal). State it as a checklist item, e.g. "measure the coolant inlet
   temperature to ≤ ε K (calibrated RTD or a matched pair with the jacket-outlet sensor)".
3. **Recommendation for production** (user decision): the rule's proposal and its cost.
4. **Limitations.** No offset on the jacket-outlet sensor Tc (what matters is the error on
   Tc − Tci); Ci unmeasured (±1%); feed flow assumed exact; ±2 K supply range."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
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
