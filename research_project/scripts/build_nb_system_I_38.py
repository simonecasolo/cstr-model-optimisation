"""Build / execute notebooks/system_I/38_deployment_gate_N3.ipynb (System I; user decision after step f).

Deployment combination read-only on D1: gate (c2 detector, |z_T| > 10, saturation guard) -> drift-aware posterior (nb09) for flagged windows,
production 2-D posterior (nb36) for unflagged. Rules in docs/Manuscript/system_i_N3_runbook.md section 2f (row g2). A written
"## 6. Interpretation" is kept on rebuild.

    python scripts/build_nb_system_I_38.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "38_deployment_gate_N3.ipynb"

HEADER = """# 38 — The deployment combination: drift gate → drift-aware posterior (System I, N3, D1 only)

**Question.** Notebook 09 gives a posterior that remedies the Sc7 sensor drift at a price (wider β, macro-F1 −0.03); notebook 11b gives a detector that flags the drift without any posterior. What does a deployment that
uses the detector as a gate achieve? **Rule (user decision, fixed before running; runbook §2f row g2):** a window is *flagged* iff it is unsaturated (at most 50% of its samples at Qc ≥ 395 L/min) and |z_T| > 10, with
z_T = (mean measured T − T_sp)/(0.2 K/√120); **flagged windows use the drift-aware posterior (notebook 09 ensemble), unflagged windows the production 2-D posterior (notebook 36 ensemble).** The production model stays the 2-D posterior.
**Data: D1 only** (50 windows per scenario; the threshold 10 was chosen on L1). Per scenario Sc1–Sc5 and Sc7: class accuracy, β 90% coverage, β sd, β bias; the gate's flag rate (false-alarm rate in Sc1–Sc5, detection rate in Sc7) for all windows and for
unsaturated windows; the Sc7 split by saturation; the production-only and drift-aware-only posteriors on the same windows for reference; and a **labelled sensitivity** without the saturation guard (|z_T| > 10 on every window). Macro-F1 over Sc1–Sc5.
**Checks:** the gate's detection of Sc7 in unsaturated windows is 1.000 as in notebook 11b; everything finite; samples inside the prior. No decision rule. With 50 windows per scenario, coverage has a standard error of about 0.04.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
import importlib, ws_eval, ws_metrics, ws_phase3, nb09_drift_aware
for _m in (ws_eval, ws_metrics, ws_phase3, nb09_drift_aware):
    importlib.reload(_m)
we, wm, P3, D = ws_eval, ws_metrics, ws_phase3, nb09_drift_aware
R = ROOT / "results"; T0 = time.time()
SCN = {1: "Sc1", 2: "Sc2", 3: "Sc3", 4: "Sc4", 5: "Sc5", 7: "Sc7"}
d1 = P3.load_d1(); X = P3.production_features(d1)
post_prod, _ = P3.load_production_posterior(); post_da = D.ensemble()
s_prod, _ = we.sample_within_prior(post_prod, X, 1000, rng_seed=21)            # (300, 1000, 2)
s_da3, _ = D.sample_box(post_da, X, 1000, seed=22); s_da = s_da3[:, :, :2]
assert np.isfinite(s_prod).all() and np.isfinite(s_da).all()
sid = d1["scenario_id"]; th = d1["theta"][:, 2:].astype(float)
z = (d1["x"][:, :, 1].mean(1) - 312.5) / (0.2 / np.sqrt(120.0)); sat = (d1["x"][:, :, 3] >= 395).mean(1) > 0.5
flag = (~sat) & (np.abs(z) > 10.0); flag_ng = np.abs(z) > 10.0
print(f"D1: {len(X)} windows | flagged (guarded gate): {flag.sum()} | flagged without the guard: {flag_ng.sum()} | saturated windows: {sat.sum()}")
"""),
    ("md", "## 1. The gate: false-alarm and detection rates"),
    ("code", """rows = []
for sc, nm in SCN.items():
    m = sid == sc; u = m & ~sat
    rows.append(dict(scenario=nm, n=int(m.sum()), n_unsaturated=int(u.sum()), flag_rate_all_windows=float(flag[m].mean()), flag_rate_unsaturated=float(flag[u].mean()) if u.any() else np.nan,
                     flag_rate_no_guard_all=float(flag_ng[m].mean()), role="detection rate (drift present)" if sc == 7 else "false-alarm rate"))
gate = pd.DataFrame(rows); display(gate.round(3)); gate.to_csv(R / "38_N3_gate_rates.csv", index=False)
det_u = float(flag[(sid == 7) & ~sat].mean()); assert det_u == 1.0, det_u
print("check passed: Sc7 detection in unsaturated windows = 1.000 (as notebook 11b); Sc7 saturated windows:", int(((sid == 7) & sat).sum()), "(never flagged by the guarded gate, by construction)")
"""),
    ("md", "## 2. Per-scenario numbers for the combination (headline), with the references"),
    ("code", """def combo(f):
    return np.where(f[:, None, None], s_da, s_prod)
def table(s, label):
    yt = wm.class_of_theta(th); yp = wm.class_of_samples(s); q05 = np.quantile(s, .05, axis=1); q95 = np.quantile(s, .95, axis=1); mean = s.mean(1); sd = s.std(1)
    out = []
    for sc, nm in SCN.items():
        m = sid == sc
        out.append(dict(variant=label, scenario=nm, n=int(m.sum()), class_acc=float((yt[m] == yp[m]).mean()), cov90_b=float(((th[m, 1] >= q05[m, 1]) & (th[m, 1] <= q95[m, 1])).mean()),
                        sd_b=float(sd[m, 1].mean()), bias_b=float((mean[m, 1] - th[m, 1]).mean())))
    return pd.DataFrame(out), yt, yp
tabs, preds = {}, {}
for label, s in (("gate + drift-aware (guarded; headline)", combo(flag)), ("production 2-D only", s_prod), ("drift-aware only", s_da), ("sensitivity: gate without the saturation guard", combo(flag_ng))):
    tabs[label], yt, preds[label] = table(s, label)
allt = pd.concat(tabs.values(), ignore_index=True); allt["scenario"] = pd.Categorical(allt.scenario, list(SCN.values()))
display(allt.sort_values(["scenario", "variant"]).round(3).reset_index(drop=True)); allt.to_csv(R / "38_N3_D1_three_numbers.csv", index=False)
mf = []
for label, yp in preds.items():
    m15 = np.isin(sid, [1, 2, 3, 4, 5]); mf.append(dict(variant=label, macro_f1_Sc1_5=wm.macro_f1(yt[m15], yp[m15]), macro_f1_all_six=wm.macro_f1(yt, yp), accuracy_Sc1_5=float((yt[m15] == yp[m15]).mean())))
mf = pd.DataFrame(mf); display(mf.round(3)); mf.to_csv(R / "38_N3_macro_f1.csv", index=False)
"""),
    ("md", "## 3. Sc7 split by saturation (the guard exempts saturated windows)"),
    ("code", """s_c = combo(flag); rows = []
for lab, m in (("Sc7 unsaturated (flagged -> drift-aware)", (sid == 7) & ~sat), ("Sc7 saturated (unflagged -> production)", (sid == 7) & sat)):
    s = s_c[m]; q05 = np.quantile(s, .05, axis=1)[:, 1]; q95 = np.quantile(s, .95, axis=1)[:, 1]
    rows.append(dict(group=lab, n=int(m.sum()), cov90_b=float(((th[m, 1] >= q05) & (th[m, 1] <= q95)).mean()) if m.any() else np.nan,
                     bias_b=float((s.mean(1)[:, 1] - th[m, 1]).mean()) if m.any() else np.nan, sd_b=float(s.std(1)[:, 1].mean()) if m.any() else np.nan))
sp = pd.DataFrame(rows); display(sp.round(3)); sp.to_csv(R / "38_N3_sc7_by_saturation.csv", index=False)
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 6. Interpretation — to be written after the run
Toward the storyline: what the gate achieves (false-alarm and detection rates on D1), the performance of the combination per scenario against production-only and drift-aware-only, the Sc7 saturated windows that the guard leaves to the
production posterior, and the price/benefit statement for the deployment."""),
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
