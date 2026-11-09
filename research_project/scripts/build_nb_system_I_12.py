"""Build / execute notebooks/system_I/12_resource_analysis_N3.ipynb (System I, Phase 3 step g).

Per-window inference time of the production SBI posterior against the EKF / UKF; training cost; break-even number of windows. NUTS is not
part of the N3 re-run (decision 2026-11-09, option C): its historical cost is quoted as context only. Rules in
docs/Manuscript/system_i_N3_runbook.md section 2f (row g-12). A written "## 7. Interpretation" is kept on rebuild.
Run on a quiet machine: the notebook records the load average and refuses to proceed above 3.

    python scripts/build_nb_system_I_12.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "12_resource_analysis_N3.ipynb"

HEADER = """# 12 — Resource analysis: when does the amortised posterior pay off? (System I, N3)

The old notebook compared SBI with NUTS. In the N3 re-run there is no valid NUTS baseline (the old likelihood is invalid, plan SI-1; decision 2026-11-09), so the comparison is with the **classical estimators that were built and validated: the EKF and the UKF** (notebook 16), with the cost of NUTS quoted from the earlier implementation as
context only (a cost of the sampler, not of a validated baseline).

**Measurements (fixed before running; runbook §2f row g-12; a quiet machine: the notebook stops if the 1-minute load average exceeds 3).** (1) SBI inference per window with the production ensemble (5 members, notebook 36): feature computation plus 1,000 draws, a single window (median of 100 calls) and a batch of 300 windows; also 100 and 2,000 draws. (2) EKF and UKF (augmented, tuned as notebook 16) per window, 50 windows each, measured here.
(3) Training cost of the production posterior: simulating 2,000 windows (scaled ×10 to the 20,000-window bank) and the training times stored with the five members. (4) Break-even N* = T_train / (t_filter − t_SBI) windows, for single-window and batched SBI; and the cost of a **retraining** after a plant change (the amortised posterior must be retrained, a filter need not).
**Checks:** finite; load average below 3; the SBI single-window median within 40% of notebook 10's 53 ms. No decision rule.
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
load0 = os.getloadavg()[0]; print("1-minute load average at start:", round(load0, 2))
assert load0 < 3.0, f"machine not quiet (load {load0:.1f}): re-run when nothing else is running"
import importlib, ws_eval, ws_phase3, ws_filters, nb35_measured_inlet, nb36_production_measured_inlet, ws_inlet
for _m in (ws_eval, ws_phase3, ws_filters, nb35_measured_inlet, nb36_production_measured_inlet, ws_inlet):
    importlib.reload(_m)
we, P3, F, M, P36, wi = ws_eval, ws_phase3, ws_filters, nb35_measured_inlet, nb36_production_measured_inlet, ws_inlet
R = ROOT / "results"; FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True); T0 = time.time()
d1 = P3.load_d1(); post, prod = P3.load_production_posterior()
"""),
    ("md", "## 1. SBI inference time per window (production ensemble, 5 members)"),
    ("code", """def one(i, n):
    f = P36.features_from_raw(d1["x"][i:i + 1], d1["inlet_measured"][i:i + 1]); return we.sample_within_prior(post, f, n, rng_seed=i)
one(0, 1000)
rows = []
for n in (100, 1000, 2000):
    tt = []
    for i in range(100):
        t = time.perf_counter(); one(i, n); tt.append(time.perf_counter() - t)
    rows.append(dict(item=f"single window: features + {n} draws", ms_median=1000 * float(np.median(tt)), ms_p05=1000 * float(np.percentile(tt, 5)), ms_p95=1000 * float(np.percentile(tt, 95))))
t = time.perf_counter(); fb = P36.features_from_raw(d1["x"], d1["inlet_measured"]); t_f = time.perf_counter() - t
t = time.perf_counter(); we.sample_within_prior(post, fb, 1000, rng_seed=1); t_b = time.perf_counter() - t
rows.append(dict(item="batch of 300 windows, 1000 draws: per window", ms_median=1000 * t_b / 300, ms_p05=np.nan, ms_p95=np.nan))
rows.append(dict(item="batch of 300 windows: features per window", ms_median=1000 * t_f / 300, ms_p05=np.nan, ms_p95=np.nan))
sbi = pd.DataFrame(rows); display(sbi.round(3)); sbi.to_csv(R / "12_N3_sbi_timing.csv", index=False)
t_sbi_single = sbi.iloc[1].ms_median / 1000; t_sbi_batch = sbi.iloc[3].ms_median / 1000
assert np.isfinite(sbi.ms_median).all() and abs(t_sbi_single / 0.0529 - 1) < 0.40, ("single-window time differs from notebook 10 by more than 40%", t_sbi_single)
"""),
    ("md", "## 2. EKF and UKF per window (augmented, as notebook 16; 50 windows)"),
    ("code", """ids = np.arange(0, 300, 6)[:50]; ft = {}
for method in ("ekf", "ukf"):
    for qs, rs, lab in ((1.0, 1.0, "untuned"), ({"ekf": 2.0, "ukf": 0.25}[method], 2.0, "tuned")):
        tt = []
        for i in ids:
            ti, tci = (float(v) for v in d1["inlet_measured"][i])
            t = time.perf_counter(); F.estimate_window(d1["x"][i], method, ti, tci, F.CI0, augment=True, q_scale=qs, r_scale=rs); tt.append(time.perf_counter() - t)
        ft[(method, lab)] = float(np.median(tt))
fil = pd.DataFrame([dict(method=m.upper(), setting=l, ms_per_window=1000 * v) for (m, l), v in ft.items()]); display(fil.round(2)); fil.to_csv(R / "12_N3_filter_timing.csv", index=False)
"""),
    ("md", "## 3. Training cost of the production posterior and the break-even"),
    ("code", """t = time.perf_counter(); th = we.prior_draws(2000, 123); inl = M._inlets(2000, 124); b29 = wi.simulate(th, inl, M.PROTOCOL, 125, features="S0_base29", **M.NOISE); t_sim2k = time.perf_counter() - t
t_bank = t_sim2k * 10                                     # the 20,000-window bank
train_s = [pickle.load(open(m, "rb"))["metadata"]["train_s"] for m in prod["members"]]
T_train = t_bank + float(np.sum(train_s))
print(f"simulating 2,000 windows: {t_sim2k:.1f}s -> 20,000-window bank {t_bank:.0f}s | training of the 5 members (stored): {np.round(train_s).astype(int).tolist()} s, total {sum(train_s):.0f}s | T_train = {T_train:.0f}s = {T_train / 60:.1f} min (this machine, 10 cores, torch 4 threads)")
rows = []
for m in ("EKF", "UKF"):
    tf = fil[(fil.method == m) & (fil.setting == "tuned")].ms_per_window.iloc[0] / 1000
    for mode, ts in (("batched SBI", t_sbi_batch), ("single-window SBI", t_sbi_single)):
        rows.append(dict(filter=m, sbi_mode=mode, t_filter_ms=1000 * tf, t_sbi_ms=1000 * ts, break_even_windows=(T_train / (tf - ts)) if tf > ts else np.inf))
be = pd.DataFrame(rows); display(be.round(1)); be.to_csv(R / "12_N3_break_even.csv", index=False)
print("break-even N* = T_train / (t_filter - t_SBI); infinite where the SBI window is slower than the filter window.")
print("a retraining after a plant change costs another T_train = %.1f min (simulation + 5 trainings); a filter needs none (but its model constants must be right)." % (T_train / 60))
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min | 1-minute load average at the end: {os.getloadavg()[0]:.2f}")
"""),
    ("md", """## 7. Interpretation — to be written after the run
Toward the storyline: per-window cost of SBI against EKF and UKF, the one-off training cost and the break-even; what is not compared (NUTS; the filters' calibration and failure region, notebook 16); the retraining cost as the price of amortisation."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 7. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 7. Interpretation"):
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
