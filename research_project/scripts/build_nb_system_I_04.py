"""Build / execute notebooks/system_I/04_posterior_recovery_N3.ipynb (System I, Phase 3 step g).

Loaders only: the production (nb36) and naive (nb31) posteriors are not retrained. Rules in docs/Manuscript/system_i_N3_runbook.md section 2f
(row g-04). A written "## 5. Interpretation" is kept on rebuild.

    python scripts/build_nb_system_I_04.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "04_posterior_recovery_N3.ipynb"

HEADER = """# 04 — The amortised posterior: training record and recovery of (α, β) (System I, N3)

The old notebook 04 trained the posterior (10 000 simulations, pre-fix simulator, 29 summaries). Under the N3 re-run the training lives elsewhere: the **naive** posterior (nominal inlet assumed, S1 features) in notebook 31 and the **production** posterior
(measured inlet temperatures as inputs, S1m features, 20 000 simulations × 5 seeds, ensemble) in notebook 36. This notebook **retrains nothing**; it (1) shows the training record — the budget study's SBC / TARP / coverage tables — read from the stored metrics, and (2) shows the recovery of (α, β) on the evaluation windows.

**Rules (fixed before running; runbook §2f row g-04).** Recovery: production on **D1** (measured inlet) and naive on **D0** (nominal inlet), Sc1–Sc5 and Sc7, 50 windows per scenario, 1,000 draws each; per scenario the mean and bias of α̂ and β̂, the mean posterior sd and the 90% coverage; a per-window figure.
**Checks:** finite; 50 windows per scenario; β 90% coverage ≥ 0.80 in Sc1–Sc4 for production on D1 and for naive on D0 (α coverage is reported, not gated: Sc5 saturates). Sc7 carries a +2 K sensor drift the posteriors do not model (a misspecified scenario by design). No decision rule.
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
    ("md", "## 1. Training record (read from the stored metrics; nothing recomputed)"),
    ("code", """cols = ["budget", "ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p", "cov90_a", "cov90_b", "sd_a", "sd_b", "accuracy"]
parts = []
for nm, f in (("naive (nb31: nominal inlet, S1)", "31_N3_metrics.csv"), ("production (nb36: measured inlet, S1m)", "36_N3_metrics.csv")):
    d = pd.read_csv(R / f); e = d[d.member == "ensemble"][cols].copy(); e.insert(0, "posterior", nm); parts.append(e)
rec = pd.concat(parts, ignore_index=True); display(rec.round(3)); rec.to_csv(R / "04_N3_training_record.csv", index=False)
import json
for nm, f in (("naive", "31_N3_production.json"), ("production", "36_N3_production.json")):
    j = json.loads((R / f).read_text()); print(nm, "-> deployed: budget", j["production_budget"], "| ensemble", j["use_ensemble"], "| summary set", j["summary_set"], "| members", len(j["members"]))
"""),
    ("md", "## 2. Recovery on the evaluation windows"),
    ("code", """s_p, d1, leak_p = E.draws("prod", "D1", 1000, seed=41)
s_n, d0, leak_n = E.draws("naive", "D0", 1000, seed=42)
for s_, d_, nm in ((s_p, d1, "production on D1"), (s_n, d0, "naive on D0")):
    assert np.isfinite(s_).all() and all((d_["scenario_id"] == sc).sum() == 50 for sc in E.SCN), nm
print(f"prior leakage: production {leak_p.mean():.4f}, naive {leak_n.mean():.4f}")
tp = E.per_scenario(s_p, d1, extra=False); tn = E.per_scenario(s_n, d0, extra=False)
show = ["scenario", "alpha_true", "beta_true", "alpha_mean", "alpha_bias", "alpha_sd", "alpha_cov90", "beta_mean", "beta_bias", "beta_sd", "beta_cov90", "class_acc"]
print("Production posterior on D1 (measured inlet)"); display(tp[show].round(3))
print("Naive posterior on D0 (nominal inlet; its valid domain)"); display(tn[show].round(3))
assert (tp[tp.scenario.isin(["Sc1", "Sc2", "Sc3", "Sc4"])].beta_cov90 >= 0.80).all() and (tn[tn.scenario.isin(["Sc1", "Sc2", "Sc3", "Sc4"])].beta_cov90 >= 0.80).all(), "beta coverage check failed"
print("check passed: beta coverage >= 0.80 in Sc1-Sc4 (production on D1, naive on D0)")
tp.assign(posterior="production on D1").to_csv(R / "04_N3_recovery_production_D1.csv", index=False); tn.assign(posterior="naive on D0").to_csv(R / "04_N3_recovery_naive_D0.csv", index=False)
"""),
    ("md", "## 3. Figure: posterior mean and 90% interval per window"),
    ("code", """fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), sharex="col")
for col, (s_, d_, ttl) in enumerate(((s_p, d1, "production on D1"), (s_n, d0, "naive on D0"))):
    th = d_["theta"][:, 2:]; mean = s_.mean(1); lo = np.quantile(s_, .05, axis=1); hi = np.quantile(s_, .95, axis=1); x = np.arange(len(th))
    for row, (j, nm) in enumerate(((0, "α"), (1, "β"))):
        ax = axes[row, col]; ax.fill_between(x, lo[:, j], hi[:, j], alpha=.3, color="steelblue"); ax.plot(x, mean[:, j], ".", ms=3, color="steelblue"); ax.plot(x, th[:, j], "k-", lw=1.2)
        ax.set_ylabel(nm); ax.set_title(f"{ttl}: {nm}"); ax.grid(alpha=.3)
        for b in range(0, len(th), 50): ax.axvline(b, color="grey", lw=.5)
    axes[1, col].set_xlabel("window (scenario blocks of 50: Sc1, Sc2, Sc3, Sc4, Sc5, Sc7)")
fig.tight_layout(); fig.savefig(FIGS / "04_N3_recovery.png", dpi=150); plt.show()
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 5. Interpretation — to be written after the run
Toward the storyline: what the training record shows (calibrated at every budget; ensemble required), what the recovery shows per scenario, and where the two posteriors differ (production on D1 against naive on D0 are different data; the naive posterior on D1 is notebook 11's sensitivity)."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 5. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 5. Interpretation"):
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
