"""Build / execute notebooks/system_I/11b_drift_detector_N3.ipynb (System I, Phase 3 step c2).

z_T = (mean measured T - Tsp) / (0.2 K / sqrt(120)) per window: the residual on the controlled variable as a drift detector.
Read-only on D1 and L1 (same generator and seeds as notebook 11). Rules fixed in docs/Manuscript/system_i_N3_runbook.md
section 2f (row c2). A written "## 6. Interpretation" is kept on rebuild.

    python scripts/build_nb_system_I_11b.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "11b_drift_detector_N3.ipynb"

HEADER = """# 11b — A residual on the controlled variable as the drift detector (System I, N3)

**Why.** Notebook 11 shows that the production posterior is confidently wrong under an unmodelled +2 K temperature-sensor drift (Sc7: β̂ biased by −0.395, coverage 0.002):
the posterior extrapolates silently. The practitioner needs a check that does not depend on the posterior. In closed loop with integral action the controlled temperature sits at the setpoint,
so a measured mean temperature away from T_sp by more than the sensor noise is either a sensor offset or a loss of control (valve saturation).

**Statistic (fixed before running; runbook §2f row c2).** z_T = (mean over the 60-min window of the measured T − T_sp) / (0.2 K/√120) with the N3 sensor sd of 0.2 K (the window-mean noise bound, 0.0183 K).
Computed for Sc1–Sc5 and Sc7 on **D1** (50 per scenario) and **L1** (1,000 per scenario; same generator and seed base 3 800 000 as notebook 11). Reported per scenario: median, 5–95% range, and the
detection rate at |z_T| > 5; the same split by saturation (a window is saturated if more than 50% of its samples have Qc ≥ 395 L/min); a **guarded** rule (flag only unsaturated windows); the false-positive rate of the guarded rule in Sc1–Sc5 at |z_T| > 5
and at the 99th percentile of the unsaturated Sc1–Sc5 |z_T|; the detection rate of Sc7 at those thresholds; the AUC of |z_T| for Sc7 against Sc1–Sc5 (unsaturated windows).
**Checks:** all finite; D1 and L1 agree on the Sc7 detection rate within 3 binomial standard errors. No decision rule.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
import importlib, ws_phase3
importlib.reload(ws_phase3); P3 = ws_phase3
R = ROOT / "results"; T0 = time.time()
TSP, SD = 312.5, 0.2 / np.sqrt(120.0)
SCN = {1: "Sc1", 2: "Sc2", 3: "Sc3", 4: "Sc4", 5: "Sc5", 7: "Sc7"}
d1 = P3.load_d1(); dL = P3.generate_d1(n_rep=1000, seed_base=3_800_000)
def stat(d):
    x = d["x"]; z = (x[:, :, 1].mean(1) - TSP) / SD; sat = (x[:, :, 3] >= 395).mean(1) > 0.5
    return pd.DataFrame(dict(scenario=[SCN[int(s)] for s in d["scenario_id"]], z=z, sat=sat))
S = {"D1": stat(d1), "L1": stat(dL)}
for k, v in S.items():
    assert np.isfinite(v.z).all(); print(k, len(v), "windows")
"""),
    ("md", "## 1. Distribution of z_T per scenario and the detection rate at |z_T| > 5"),
    ("code", """rows = []
for dn, df in S.items():
    for sc in SCN.values():
        g = df[df.scenario == sc]; u = g[~g.sat]; s_ = g[g.sat]
        rows.append(dict(data=dn, scenario=sc, n=len(g), frac_saturated=g.sat.mean(), z_median=g.z.median(), z_p05=g.z.quantile(.05), z_p95=g.z.quantile(.95),
                         rate_gt5_all=(g.z.abs() > 5).mean(), rate_gt5_unsat=(u.z.abs() > 5).mean() if len(u) else np.nan,
                         rate_gt5_sat=(s_.z.abs() > 5).mean() if len(s_) else np.nan, n_unsat=len(u),
                         guarded_rate_gt5=((~g.sat) & (g.z.abs() > 5)).sum() / max((~g.sat).sum(), 1)))
tab = pd.DataFrame(rows); display(tab.round(3)); tab.to_csv(R / "11b_N3_zT_distribution.csv", index=False)
"""),
    ("md", "## 2. Guarded rule: false positives in Sc1–Sc5 and detection of Sc7 at two thresholds (L1)"),
    ("code", """L = S["L1"]; U = L[~L.sat]
ref = U[U.scenario != "Sc7"].z.abs()
thr99 = float(ref.quantile(0.99))
out = []
for name, thr in (("5", 5.0), ("99th pct of unsaturated Sc1-Sc5 |z|", thr99), ("10", 10.0), ("20", 20.0)):
    fp = {sc: float((U[U.scenario == sc].z.abs() > thr).mean()) for sc in SCN.values() if sc != "Sc7"}
    det_u = float((U[U.scenario == "Sc7"].z.abs() > thr).mean())
    det_all = float((L[L.scenario == "Sc7"].z.abs() > thr).mean())
    out.append(dict(threshold=name, value=round(thr, 2), **{f"FP_{k}": v for k, v in fp.items()}, FP_pooled_Sc1_5=float((ref > thr).mean()),
                    Sc7_detect_unsaturated=det_u, Sc7_detect_all_windows=det_all))
guard = pd.DataFrame(out); display(guard.round(3)); guard.to_csv(R / "11b_N3_zT_guarded_rule.csv", index=False)
# AUC of |z| for Sc7 against Sc1-Sc5 in unsaturated windows (Mann-Whitney)
from scipy.stats import mannwhitneyu
a = U[U.scenario == "Sc7"].z.abs().values; b = ref.values
auc = mannwhitneyu(a, b).statistic / (len(a) * len(b))
print(f"AUC of |z_T| (Sc7 vs Sc1-Sc5, unsaturated, L1): {auc:.4f} (n = {len(a)} vs {len(b)})")
# sign of z in Sc7 saturated windows
Sg = L[(L.scenario == "Sc7") & L.sat]
print(f"Sc7 saturated windows: n = {len(Sg)}; |z_T| > 5 in {(Sg.z.abs() > 5).mean():.3f}; median z_T {Sg.z.median():+.1f}")
"""),
    ("md", "## 3. Check: D1 and L1 agree on the Sc7 detection rate"),
    ("code", """r1 = (S["D1"][S["D1"].scenario == "Sc7"].z.abs() > 5); rL = (S["L1"][S["L1"].scenario == "Sc7"].z.abs() > 5)
p = float(rL.mean()); se = np.sqrt(p * (1 - p) / len(r1) + p * (1 - p) / len(rL)) if 0 < p < 1 else 1 / len(r1)
print(f"Sc7 detection (|z_T| > 5, all windows): D1 {r1.mean():.3f} (n={len(r1)}), L1 {rL.mean():.3f} (n={len(rL)})")
assert abs(r1.mean() - rL.mean()) <= 3 * max(se, 1e-9) + 1e-9, "D1 and L1 disagree"
print("check passed")
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 4. Figure"""),
    ("code", """import matplotlib.pyplot as plt
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a_, (lab, m) in zip(ax, (("unsaturated windows", ~L.sat), ("saturated windows", L.sat))):
    data = [L[(L.scenario == sc) & m].z.clip(-60, 150).values for sc in SCN.values()]
    a_.boxplot([d if len(d) else [np.nan] for d in data], labels=list(SCN.values()), flierprops=dict(ms=2, alpha=.3))
    a_.axhspan(-5, 5, color="grey", alpha=.15); a_.set_title(lab + " (L1; z clipped to [-60, 150])"); a_.set_ylabel("z_T")
fig.tight_layout(); fig.savefig(FIGS / "11b_N3_zT.png", dpi=150); plt.show()
"""),
    ("md", """## 6. Interpretation — to be written after the run
Toward the storyline: the posterior extrapolates silently; a residual on the controlled variable flags the drift. Report: what the detector catches (Sc7), what it falsely flags (saturation, transients) and the guarded rule's
false-positive and detection rates; which entry it becomes in the practitioner checklist."""),
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
