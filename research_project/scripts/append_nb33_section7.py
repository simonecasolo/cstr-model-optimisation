"""Append (or replace) section 7 of notebooks/system_I/33_sensor_noise_realism.ipynb:
"What the sensor model does to the diagnosis" -- classification accuracy / F1 per sensor
model, artefact alpha-beta correlations, and alpha without a concentration reading.

Sections 1-6 (executed, with outputs) are kept untouched; only the new cells are executed
(kernel cwd = notebooks/system_I). The section re-uses the trained posteriors and test sets of
sections 2-4 (results/33/); its only new computation is posterior sampling (~10-20 min).
Descriptive: no decision rule.

    python scripts/append_nb33_section7.py            # write the cells only
    python scripts/append_nb33_section7.py --execute  # write and execute them
"""
import copy
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
NB = NB_DIR / "33_sensor_noise_realism.ipynb"
MARKER = "<!-- section7-diagnosis -->"

HEADER = MARKER + r"""
## 7. What the sensor model does to the diagnosis (added 2026-11-09)

Sections 4–6 measured posterior width and calibration. This section asks the operational
question: **does the plant get the right diagnosis?** Three points for the paper:

1. **Classification.** Fault class = the quadrant (threshold 0.90 on α and β) holding most of
   the posterior mass. Per-class F1, macro-F1 (with the spread over the 3 seed members) and
   confusion matrices, on the SBC set (1000 prior draws: every class over the whole prior) and on
   the scenario set (Sc1–Sc7 × 50).
2. **Artefact correlations.** An α–β posterior correlation is a property of the *information the
   sensors deliver*, not of the plant alone: the same Sc5 plant gives a strong ridge under one
   sensor model and none under another, and removing a measurement (N4) or an unmodelled drift
   (Sc7) creates correlations where the plant has none. The SBI correlation is compared with the
   correlation implied by the Fisher information of the same variant (independent of the estimator).
3. **α without a concentration reading (N4).** Catalyst activity enters the measurements almost
   only through C at ~98% conversion; without C the posterior on α should return to the prior,
   while β (read from T, Tc and Qc) is unaffected. Measured as posterior sd / prior sd
   (1 = nothing learned).

Same posteriors (40k × 3 seeds per variant) and test sets as sections 2–4.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import importlib, ws_protocols, ws_eval, ws_metrics, nb29_ws_study, nb33_sensor_noise
for _m in (ws_protocols, ws_eval, ws_metrics, nb29_ws_study, nb33_sensor_noise):
    importlib.reload(_m)
N, wm, we = nb33_sensor_noise, ws_metrics, ws_eval
R7, FIGS7 = ROOT / "results", ROOT / "figures"
V = list(N.VARIANTS)
DESC = {"N0": "N0  0.5% of max (inherited)", "N1": "N1  realistic C only", "N2": "N2  realistic T only",
        "N3": "N3  realistic, all channels", "N4": "N4  realistic, no C reading"}
T0_7 = time.time()
N.stage_classification()
print(f"sampling done in {(time.time() - T0_7) / 60:.1f} min")"""),
    ("md", """### 7.1 Classification accuracy and F1 per sensor model
Macro-F1 for the ensemble and mean ± sd over the three seed members; then per-class F1."""),
    ("code", """f1 = pd.read_csv(R7 / "33_class_f1.csv"); f1["member"] = f1["member"].astype(str)
mac = f1.drop_duplicates(["variant", "member", "set"])[["variant", "member", "set", "macro_f1", "accuracy"]]
ens = mac[mac.member == "ensemble"].pivot(index="variant", columns="set", values=["macro_f1", "accuracy"]).reindex(V)
seeds = mac[mac.member != "ensemble"].groupby(["variant", "set"]).macro_f1.agg(["mean", "std"]).unstack("set").reindex(V)
display(ens.round(3)); display(seeds.round(3))
per_cls = f1[f1.member == "ensemble"].pivot_table(index=["set", "cls"], columns="variant", values="f1")[V]
display(per_cls.round(3))
rec = f1[f1.member == "ensemble"].pivot_table(index=["set", "cls"], columns="variant", values="recall")[V]
print("recall (fraction of each true class recovered):"); display(rec.round(3))"""),
    ("code", """cm = pd.read_csv(R7 / "33_confusion.csv")
C = list(wm.CLASSES); short = ["healthy", "fouling", "decay", "combined"]
fig, axes = plt.subplots(1, len(V), figsize=(3.3 * len(V), 3.6), constrained_layout=True)
for ax, v in zip(axes, V):
    m = cm[(cm.variant == v) & (cm.set == "sbc")].pivot(index="true", columns="pred", values="n").reindex(index=C, columns=C)
    frac = m.div(m.sum(1), axis=0).values
    ax.imshow(frac, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.6 else "black")
    ax.set_xticks(range(4), short, rotation=45, fontsize=8); ax.set_yticks(range(4), short if v == V[0] else [], fontsize=8)
    mf = ens.loc[v, ("macro_f1", "sbc")]
    ax.set_title(f"{DESC[v]}\\nmacro-F1 {mf:.2f}", fontsize=9)
    ax.set_xlabel("predicted", fontsize=8)
axes[0].set_ylabel("true class (prior draws)", fontsize=8)
fig.suptitle("Fault classification under five sensor models (row-normalised confusion, SBC set)", fontsize=10)
fig.savefig(FIGS7 / "33_confusion_by_sensor_model.png", dpi=150); plt.show()

sc = pd.read_csv(R7 / "33_scenarios.csv"); sc["member"] = sc["member"].astype(str)
acc = sc[sc.member == "ensemble"].pivot_table(index="scenario", columns="variant", values="class_acc")[V]
fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
im = ax.imshow(acc.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
for i in range(acc.shape[0]):
    for j in range(acc.shape[1]):
        ax.text(j, i, f"{acc.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(V)), V); ax.set_yticks(range(len(acc)), acc.index, fontsize=8)
ax.set_title("Class accuracy per scenario (50 windows each, ensemble)", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(FIGS7 / "33_accuracy_by_scenario.png", dpi=150); plt.show()"""),
    ("md", """### 7.2 Artefact correlations: the α–β ridge depends on the sensors
Mean posterior Corr(α, β) per scenario (SBI ensemble) next to the correlation implied by the
Fisher information of the same variant (29 summaries, full covariance). Agreement between the
two means the correlation is in the data the sensors deliver, not created by the estimator.
Below: the joint posterior of the same Sc5 window (first replicate) under each sensor model."""),
    ("code", """inf = pd.read_csv(R7 / "33_information.csv")
corr_sbi = sc[sc.member == "ensemble"].pivot_table(index="scenario", columns="variant", values="corr_ab_mean")[V]
fim = pd.read_csv(R7 / "33_fim.csv")
corr_fim = fim[fim.representation == "summaries_fullcov"].pivot_table(index="point", columns="variant", values="corr_implied")[V]
print("SBI posterior Corr(alpha, beta), ensemble mean over windows:"); display(corr_sbi.round(2))
print("Fisher-information implied correlation (29 summaries):"); display(corr_fim.round(2))
pts = [(corr_fim.loc[p, v], corr_sbi.loc[[s for s in corr_sbi.index if s.startswith(p + "_")][0], v], v, p)
       for v in V for p in corr_fim.index]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
for v in V:
    q = [(a, b) for a, b, vv, _ in pts if vv == v]
    ax.scatter(*zip(*q), label=v, s=30)
for a, b, v, p in pts:
    if abs(b) > 0.2 or abs(a) > 0.2:
        ax.annotate(f"{p} {v}", (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.plot([-1, 1], [-1, 1], color="grey", lw=0.8); ax.axhspan(-0.2, 0.2, color="grey", alpha=0.1)
ax.set_xlim(-0.3, 1); ax.set_ylim(-0.3, 1)
ax.set_xlabel("Fisher-implied Corr(α, β)"); ax.set_ylabel("SBI posterior Corr(α, β)"); ax.legend(fontsize=8)
ax.set_title("Correlation follows the information, not the estimator", fontsize=10)
ax = axes[1]
sc5 = None
for k, v in enumerate(V):
    d = we.load_npz(N.OUT / f"cls_samples_{v}.npz")
    i = int(np.flatnonzero(d["sc_name"] == [s for s in set(d["sc_name"].tolist()) if s.startswith("Sc5_")][0])[0])
    s5 = d["samples"][i]; sc5 = d["theta"][i]
    ax.scatter(s5[:, 0], s5[:, 1], s=6, alpha=0.35, label=f"{v}  r = {np.corrcoef(s5.T)[0, 1]:+.2f}")
ax.plot(*sc5, "k*", ms=12, label="truth")
ax.set_xlabel("α"); ax.set_ylabel("β"); ax.legend(fontsize=7)
ax.set_title("Sc5 (saturated valve), one window, posterior under each sensor model", fontsize=10)
fig.savefig(FIGS7 / "33_artefact_correlation.png", dpi=150); plt.show()"""),
    ("md", """### 7.3 α without a concentration reading
Posterior sd divided by the prior sd (uniform prior on [0.4, 1.2]: sd 0.231); median over
windows. 1 means the data carry no information on that parameter. Below: pooled α and β
marginals for Sc1 (healthy, α = 1) and Sc3 (catalyst decay, α = 0.7) with and without C (N3 vs
N4), and for Sc1 vs Sc2 (fouling, β = 0.85)."""),
    ("code", """ratio = inf.pivot_table(index="group", columns="variant", values=["sd_ratio_a", "sd_ratio_b"])
display(ratio.round(3))
crb = fim[fim.representation == "summaries_fullcov"].pivot_table(index="point", columns="variant", values="crb_sd_a")[V]
print("Cramér–Rao sd of alpha (29 summaries); the prior sd is 0.231:"); display(crb.round(3))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
bins = np.linspace(0.4, 1.2, 81)
for v, ls in (("N3", "-"), ("N4", "--")):
    d = we.load_npz(N.OUT / f"cls_samples_{v}.npz")
    for pre, col in (("Sc1_", "C0"), ("Sc3_", "C3"), ("Sc2_", "C2")):
        m = np.char.startswith(d["sc_name"].astype(str), pre)
        if pre != "Sc2_":
            axes[0].hist(d["samples"][m][:, :, 0].ravel(), bins=bins, density=True, histtype="step", color=col, ls=ls,
                         lw=1.5, label=f"{pre[:-1]} {v}")
        if pre != "Sc3_":
            axes[1].hist(d["samples"][m][:, :, 1].ravel(), bins=bins, density=True, histtype="step", color=col, ls=ls,
                         lw=1.5, label=f"{pre[:-1]} {v}")
for ax, name, truths in ((axes[0], "α", (1.0, 0.7)), (axes[1], "β", (1.0, 0.85))):
    for t_ in truths:
        ax.axvline(t_, color="k", lw=0.8, ls=":")
    ax.axvline(0.90, color="grey", lw=1.2, label="class threshold 0.90")
    ax.set_xlabel(name); ax.legend(fontsize=8); ax.set_yticks([])
axes[0].set_title("α: healthy vs catalyst decay — with C (N3, solid) and without (N4, dashed)", fontsize=9)
axes[1].set_title("β: healthy vs fouling — unaffected by the loss of C", fontsize=9)
fig.savefig(FIGS7 / "33_alpha_without_C.png", dpi=150); plt.show()
print(f"section 7 wall time {(time.time() - T0_7) / 60:.1f} min")"""),
    ("md", """### 7.4 Interpretation — to be written after the run
State, with numbers from 7.1–7.3:
1. how macro-F1 and per-class F1 move along N0 → N1 → N2 → N3 → N4, which classes are confused
   (from the confusion matrices) and why (α sd against the distance of the true α from 0.90);
2. artefact correlations, three kinds visible in these tables: (a) the Sc5 ridge under N0/N1
   (imprecise T) that vanishes under N2/N3; (b) the correlation that appears under N4 in every
   scenario once α is no longer pinned by C (the posterior spreads along the direction the
   remaining measurements cannot resolve); (c) the Sc7 correlation under N0, produced by the
   unmodelled +2 K drift (misspecification). Where the SBI correlation tracks the Fisher-implied
   one, the correlation is in the data the sensors deliver, not created by the estimator; it is a
   property of the measurement set-up, not of the plant. Together with the controller bug and the
   noise-free start (nb29) this accounts for the old near-unity correlation;
3. that without C the α posterior is close to the prior (sd ratio ≈ 1), healthy and
   catalyst-decay plants become indistinguishable, while β and fouling diagnosis survive;
4. the practical statement for the paper (which instrumentation buys which diagnosis)."""),
]


def _cells():
    return [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]


def main(execute: bool):
    nb = nbformat.read(NB, as_version=4)
    keep = []
    for c in nb.cells:
        if c.cell_type == "markdown" and "".join(c.source).lstrip().startswith(MARKER):
            break
        keep.append(c)
    new = _cells()
    if execute:
        from nbclient import NotebookClient
        tmp = new_notebook(); tmp.cells = new
        tmp.metadata = copy.deepcopy(nb.metadata)
        client = NotebookClient(tmp, timeout=None, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}})   # cwd = notebooks/system_I
        try:
            client.execute()
        finally:
            nb.cells = keep + tmp.cells
            nbformat.write(nb, NB)
    else:
        nb.cells = keep + new
        nbformat.write(nb, NB)
    print(f"{NB}: {len(keep)} existing cells kept, {len(new)} section-7 cells written")


if __name__ == "__main__":
    main("--execute" in sys.argv)
