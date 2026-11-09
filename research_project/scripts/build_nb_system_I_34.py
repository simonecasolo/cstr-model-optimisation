"""Build (and optionally execute) notebooks/system_I/34_inlet_uncertainty.ipynb (study R2).

The notebook IS the experiment: every stage runs in a notebook cell with the kernel's working
directory set to notebooks/system_I. Stage functions live in scripts/nb34_inlet_uncertainty.py
and skip work already on disk (resumable). Needs the N3 production posterior (nb31) and the
nb03 section-10b posteriors.

Usage:  python scripts/build_nb_system_I_34.py            # build only
        python scripts/build_nb_system_I_34.py --execute  # build + execute headless (nbclient)
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "34_inlet_uncertainty.ipynb"

HEADER = r"""# 34 — R2: is the diagnosis robust to uncertain inlet conditions? (System I)

**Why.** The jacket removes only ~19% of the reaction heat; the feed absorbs ~81%. The jacket
duty, and therefore β, is sensitive to the feed temperature: to first order a 1 K error in Ti
shifts the duty by ~27% (~0.2 in β). The generative model treats the inlet (Ci, Ti, Tci) as exactly
known, while the N3 production posterior reports β with sd ≈ 0.015. If realistic inlet errors bias
β by many posterior sds, the precision claim does not survive contact with a real plant. In the
paper's storyline this is the fourth source of silent failure (after the control law, the warm
start and the sensor model), or, if the posterior holds, evidence for the "reliable" half.

**A built-in alarm?** The production summary set S1 contains two independent plug-in estimates of
β: one from the jacket/coolant balance (uses Tci, Qc, Tc) and one from the reactor balance (uses
Ti, Ci). An error in Ti corrupts only the second, an error in Tci only the first, so their
disagreement d = ln β̂_jac − ln β̂_rx may flag inlet misspecification from the data alone.

**Design** (N3 sensors, P2 protocol, as production)
* **Check.** The per-sample-inlet simulator (`scripts/ws_inlet.py`) reproduces the production
  simulator at the nominal inlet to < 1% of the sensor σ; deterministic first-order sensitivity of
  the plug-in estimators to each inlet error.
* **Part A — misspecification.** True plant inlet perturbed one factor at a time: Ti ±0.5, ±1 K;
  Tci ±0.5, ±1 K; Ci ±1, ±2%. Scenarios Sc1–Sc5 × 50 windows with common random numbers. Posteriors
  (all assume the nominal inlet): the N3 production posterior (nb31) and, for S0 vs S1 at equal
  budget, the notebook 03 §10b ensembles at 20k.
* **Part B — robust training** (only if the rule below triggers, or `FORCE_B`). S1 posteriors at
  40k × 3 seeds, trained with the inlet fixed ("nominal") or drawn per sample as a nuisance
  ("robust": Ti, Tci ~ N(297, 0.5²) K, Ci ± 1% sd); compared on the Part A conditions and on SBC
  sets under both inlet distributions. Part B reports; it does not change production.

## Decision rule — fixed before running
Realistic inlet uncertainty = Ti ±0.5 K, Tci ±0.5 K, Ci ±1%. The production posterior is
**robust** if, for every realistic condition and every scenario Sc1–Sc4, the β 90% coverage at the
truth is ≥ 0.80 and the class accuracy is within 0.10 of the nominal condition. Otherwise Part B
is run. ±1 K and ±2% are stress tests (reported, not gated); Sc5 (saturated) is reported, not
gated. Adopting a robust posterior as production is a **user decision**.

**Running.** Execute top to bottom from `notebooks/system_I`. Check + Part A ≈ 20–30 min;
Part B (if run) ≈ 2–3 h. Re-running resumes.
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
import importlib, system_I_config, ws_protocols, ws_eval, ws_metrics, ws_inlet, nb29_ws_study, nb31_budget_convergence, nb34_inlet_uncertainty
for _m in (system_I_config, ws_protocols, ws_eval, ws_metrics, ws_inlet, nb29_ws_study, nb31_budget_convergence, nb34_inlet_uncertainty):
    importlib.reload(_m)
I = nb34_inlet_uncertainty
assert I.SENSOR == "N3", I.SENSOR
R, FIGS = ROOT / "results", ROOT / "figures"
FORCE_B = False      # True runs Part B even if the production posterior is robust (e.g. for the SI)
THREADS = 4
T0 = time.time()
print("sensor", I.SENSOR, "| protocol", I.PROTOCOL, "| summary set", I.SET, "| outputs", I.TAG)"""),
    ("md", """## 1. Check: simulator identity and deterministic sensitivities
Noise-free steady states at the perturbed inlet; the change of each plug-in estimate relative to
the nominal inlet (log units ≈ relative error). This is what an analyst who assumes the nominal
inlet would read off a perfect measurement."""),
    ("code", """ident, sens = I.stage_check()
show = sens[sens.scenario.isin(I.GATED)].pivot_table(index="condition", columns="scenario",
        values=["dln_beta_hat_jac_vs_nominal", "dln_beta_hat_rx_vs_nominal", "dln_alpha_hat_vs_nominal"])
display(show.reindex(list(I.CONDITIONS)).round(3))
display(sens[sens.scenario.isin(I.GATED)].pivot_table(index="condition", columns="scenario", values="Qc_ss_vs_nominal")
        .reindex(list(I.CONDITIONS)).round(1))"""),
    ("md", """## 2. Part A — the production posterior on a plant with the wrong inlet
Simulate the scenario grid under every inlet condition (common random numbers), then evaluate
the production posterior and the notebook 03 S0 / S1 ensembles (20k), all assuming the nominal inlet."""),
    ("code", """I.stage_sim_A()
I.stage_eval_A()
A = pd.read_csv(R / f"{I.TAG}_A_metrics.csv")
A["short"] = A.scenario.str.split("_").str[0]
for k in ("bias_b", "z_bias_b", "cov90_truth_b", "class_acc", "bias_a", "cov90_truth_a"):
    print(k)
    display(A.pivot_table(index=["posterior", "condition"], columns="short", values=k)
            .reindex(list(I.CONDITIONS), level=1).round(3))"""),
    ("code", """prod = [p for p in A.posterior.unique() if p.startswith("production_")][0]
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
for col, (fac, conds, xs, unit) in enumerate([("Ti", ["Ti-1.0", "Ti-0.5", "nominal", "Ti+0.5", "Ti+1.0"], [-1, -.5, 0, .5, 1], "K"),
                                              ("Tci", ["Tci-1.0", "Tci-0.5", "nominal", "Tci+0.5", "Tci+1.0"], [-1, -.5, 0, .5, 1], "K"),
                                              ("Ci", ["Ci-2%", "Ci-1%", "nominal", "Ci+1%", "Ci+2%"], [-2, -1, 0, 1, 2], "%")]):
    for sc in I.GATED + ["Sc5"]:
        d = A[(A.posterior == prod) & (A.short == sc)].set_index("condition").reindex(conds)
        axes[0, col].plot(xs, d.bias_b, "o-", label=sc)
        axes[1, col].plot(xs, d.cov90_truth_b, "o-", label=sc)
    axes[0, col].axhline(0, color="grey", lw=0.8)
    axes[0, col].set_title(f"β bias vs {fac} error (production posterior)"); axes[0, col].set_xlabel(f"{fac} error [{unit}]")
    axes[1, col].axhline(0.9, color="grey", lw=0.8); axes[1, col].axhline(0.8, color="C3", lw=0.8, ls="--")
    axes[1, col].set_ylim(-0.02, 1.02); axes[1, col].set_title("β 90% coverage at the truth"); axes[1, col].set_xlabel(f"{fac} error [{unit}]")
    axes[0, col].legend(fontsize=7)
fig.savefig(FIGS / f"{I.TAG}_A_beta_bias_coverage.png", dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for p in A.posterior.unique():
    d = A[(A.posterior == p) & A.short.isin(I.GATED)].groupby("condition").z_bias_b.apply(lambda v: np.abs(v).mean())
    ax.plot(list(I.CONDITIONS), d.reindex(list(I.CONDITIONS)).values, "o-", label=p)
ax.set_ylabel("mean |β bias| / posterior sd (Sc1–Sc4)"); ax.tick_params(axis="x", rotation=60); ax.legend(fontsize=8)
ax.set_title("Bias in units of the reported uncertainty: S0 vs S1 vs production")
fig.savefig(FIGS / f"{I.TAG}_A_zbias_by_set.png", dpi=150); plt.show()"""),
    ("md", """### 2b. Can the data flag the wrong inlet? Disagreement of the two β estimates
d = ln β̂_jac − ln β̂_rx per window, z-scored against a separate nominal calibration set (pooled
Sc1–Sc4). Detection rate at |z| > 3; the nominal row is the false-alarm rate."""),
    ("code", """al = I.stage_alarm_A()
display(al.pivot_table(index="condition", columns="scenario", values="shift_in_cal_sd").reindex(list(I.CONDITIONS)).round(2))
display(al.pivot_table(index="condition", columns="scenario", values="detect_rate_pooled").reindex(list(I.CONDITIONS)).round(2))"""),
    ("md", "## 3. Decision (pre-set rule)"),
    ("code", """decA = I.stage_decide_A()
RUN_B = decA["run_part_B"] or FORCE_B
print("Part B will run" if RUN_B else "Part B not needed (set FORCE_B = True to run it for the SI)")"""),
    ("md", """## 4. Part B — robust training with the inlet as a nuisance (runs only if needed)
Nominal vs robust S1 posteriors at 40k × 3 seeds (same θ draws and seeds). The price of robustness
is a wider β posterior; the gain is coverage under inlet errors."""),
    ("code", """if RUN_B:
    I.stage_bank_B(); I.stage_train_B(THREADS); I.stage_eval_B()
    Bm = pd.read_csv(R / f"{I.TAG}_B_metrics.csv"); Bm["short"] = Bm.scenario.str.split("_").str[0]
    for k in ("bias_b", "sd_b", "cov90_truth_b", "class_acc"):
        print(k)
        display(Bm.pivot_table(index=["posterior", "condition"], columns="short", values=k)
                .reindex(list(I.CONDITIONS), level=1).round(3))
    display(pd.read_csv(R / f"{I.TAG}_B_sbc.csv")[["posterior", "sbc_inlet", "ks_p_a_interior", "ks_p_b_interior",
                                                   "tarp_ks_p", "cov90_a", "cov90_b", "rmse_norm"]].round(4))
else:
    print("skipped")
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")"""),
    ("md", """## 5. Interpretation — to be written after the run
Write toward the working storyline (`docs/Manuscript/system_i_rerun_plan.md`):
1. **Sensitivity (section 1).** Which estimator each inlet error corrupts, and by how much per K / %.
2. **Production posterior (Part A).** β bias in units of its own sd, coverage and class accuracy at
   the realistic levels and the stress levels; the rule's verdict. Is the inlet a silent failure
   mode (confident and wrong) or is the posterior robust at realistic uncertainty?
3. **S0 vs S1.** Does the explicit plug-in estimator make S1 more fragile (it hard-codes the
   nominal inlet) or more robust (two redundant β estimates)?
4. **Alarm (2b).** Does the disagreement of the two β estimates detect the wrong inlet, at which
   level, with what false-alarm rate? If yes: a data-driven misspecification check for the checklist.
5. **Part B (if run).** Coverage regained and precision paid by training with the inlet as a
   nuisance; SBC under both inlet distributions. State that adopting it is a user decision.
6. **Checklist item** this study supports (e.g. "measure or marginalise the feed temperature")."""),
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
