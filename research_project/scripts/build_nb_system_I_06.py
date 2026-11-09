"""Build / execute notebooks/system_I/06_multi_sample_study_N3.ipynb (System I, Phase 3 step g).

Per-scenario W1, CRPS, bias, coverage, 2-D coverage, accuracy: production on D1, naive on D0 (and on D1 as a labelled sensitivity).
Rules in docs/Manuscript/system_i_N3_runbook.md section 2f (row g-06). A written "## 6. Interpretation" is kept on rebuild.

    python scripts/build_nb_system_I_06.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "06_multi_sample_study_N3.ipynb"

HEADER = """# 06 — Multi-sample study across scenarios (System I, N3)

The old notebook evaluated the 2-D posterior on 50 replicates of each of the eight scenarios Sc0–Sc7 (400 windows) with the pre-fix simulator. Here: **Sc1–Sc5 and Sc7, 50 windows per scenario (300 windows)**; Sc0 and Sc6 (open loop) are not in the measured-inlet dataset D1
(user decision 1, 2026-11-09). **Production** posterior (measured inlet, notebook 36) on **D1**; the **naive** posterior (nominal inlet, notebook 31) on **D0** (its valid domain) and, as a labelled sensitivity only, on D1.

**Per scenario (rules fixed before running; runbook §2f row g-06):** W1 and CRPS of α and β against the truth (lower is better), bias, 90% coverage, the 2-D coverage (the truth inside the Gaussian-approximation 90% region of the posterior, a joint check that the marginals miss), class accuracy.
1,000 draws per window. **Checks:** finite; n = 50 per scenario; production β coverage ≥ 0.80 in Sc1–Sc4. No decision rule. Sc7 carries a +2 K sensor drift neither posterior models (misspecified by design).
"""

CELLS = [
    ("md", HEADER),
    ("code", """import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
import importlib, ws_eval, ws_metrics, ws_phase3, ws_phase3_eval
for _m in (ws_eval, ws_metrics, ws_phase3, ws_phase3_eval):
    importlib.reload(_m)
E = ws_phase3_eval; R = ROOT / "results"; FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True); T0 = time.time()
"""),
    ("md", "## 1. Posterior draws and per-scenario metrics"),
    ("code", """runs = {"production on D1": ("prod", "D1", 61), "naive on D0": ("naive", "D0", 62), "naive on D1 (sensitivity)": ("naive", "D1", 63)}
tabs, samp = {}, {}
for name, (w, dsn, seed) in runs.items():
    s, d, leak = E.draws(w, dsn, 1000, seed=seed)
    assert np.isfinite(s).all() and all((d["scenario_id"] == sc).sum() == 50 for sc in E.SCN), name
    t = E.per_scenario(s, d); t.insert(0, "posterior", name); tabs[name] = t; samp[name] = (s, d)
    print(f"{name}: leakage {leak.mean():.4f}")
allt = pd.concat(tabs.values(), ignore_index=True); allt.to_csv(R / "06_N3_per_scenario.csv", index=False)
prod = tabs["production on D1"]
assert (prod[prod.scenario.isin(["Sc1", "Sc2", "Sc3", "Sc4"])].beta_cov90 >= 0.80).all(), "check failed: production beta coverage"
print("check passed: production beta coverage >= 0.80 in Sc1-Sc4")
cols = ["scenario", "class_acc", "alpha_bias", "alpha_cov90", "alpha_W1", "alpha_CRPS", "beta_bias", "beta_cov90", "beta_W1", "beta_CRPS", "joint_cov90_gauss"]
for name, t in tabs.items():
    print(name); display(t[cols].round(3))
"""),
    ("md", "## 2. Figure: W1 and CRPS of β per scenario; posterior-mean scatter in (α, β)"),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
x = np.arange(len(E.SCN)); w = 0.27
for k, (name, t) in enumerate(tabs.items()):
    axes[0].bar(x + (k - 1) * w, t.beta_W1, w, label=name); axes[1].bar(x + (k - 1) * w, t.beta_CRPS, w, label=name)
for ax, ttl in ((axes[0], "W1(β posterior, truth)"), (axes[1], "CRPS(β)")):
    ax.set_xticks(x); ax.set_xticklabels(list(E.SCN.values())); ax.set_title(ttl); ax.grid(alpha=.3, axis="y")
axes[0].legend(fontsize=7)
s, d = samp["production on D1"]; mean = s.mean(1); sid = d["scenario_id"]
for sc, nm in E.SCN.items():
    m = sid == sc; axes[2].plot(mean[m, 0], mean[m, 1], ".", ms=4, label=nm); axes[2].plot(*d["theta"][m][0, 2:], "k+", ms=10)
axes[2].set_xlabel("α̂"); axes[2].set_ylabel("β̂"); axes[2].set_title("production on D1: posterior means (+ = truth)"); axes[2].legend(fontsize=7); axes[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIGS / "06_N3_multi_sample.png", dpi=150); plt.show()
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 6. Interpretation — to be written after the run
Toward the storyline: what the three metrics say per scenario (accuracy of the production posterior; the naive posterior on its valid and on the wrong data; Sc7 as the misspecified scenario); what the 2-D coverage adds to the marginals."""),
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
