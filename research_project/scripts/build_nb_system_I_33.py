"""Build (and optionally execute) notebooks/system_I/33_sensor_noise_realism.ipynb (study R1).

The notebook IS the experiment: every stage runs in a notebook cell with the kernel's
working directory set to notebooks/system_I. Stage functions live in
scripts/nb33_sensor_noise.py and skip work already on disk (resumable).

Usage:  python scripts/build_nb_system_I_33.py            # build only (run in Jupyter / VS Code)
        python scripts/build_nb_system_I_33.py --execute  # build + execute headless (nbclient)
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "33_sensor_noise_realism.ipynb"

HEADER = r"""# 33 — R1: does the sensor-noise model drive the results? (System I)

**Why.** The inherited sensor model sets σ = 0.5% of each channel's window maximum. That is
*pessimistic* for temperature (σ_T ≈ 1.56 K; industrial RTDs are ±0.1–0.5 K) and *optimistic*
for concentration (σ_C ≈ 9×10⁻⁵ mol/L; online analysers are typically 1–2% of full scale, one
to two orders of magnitude worse). At ~98% conversion α is inferred mainly from C, so its
precision is probably overstated. This study runs **before** the Phase 3 re-runs: if realistic
noise changes the conclusions, production should change first.

| Variant | σ_C [mol/L] | σ_T, σ_Tc [K] | σ_Qc [L/min] | Purpose |
|---|---|---|---|---|
| N0 | 0.5% of max (≈9×10⁻⁵) | 0.5% of max (≈1.56) | 0.5% of max | current production (reference) |
| N1 | 0.01 (1% of a 1 mol/L span) | as N0 | as N0 | realistic C only: isolates the α effect |
| N2 | as N0 | 0.2 | as N0 | realistic T only |
| N3 | 0.01 | 0.2 | 2 (0.5% of the 400 L/min span) | realistic on all channels (plant with an online analyser) |
| N4 | 1.0 (no usable C) | 0.2 | 2 | realistic, **no online analyser**: what temperatures and Qc alone identify |

All variants: P1b warm start, same θ draws and seeds, production NSF recipe, 40k simulations ×
3 seeds (each seed its own draws) and the 3-seed ensemble, each evaluated on a test set simulated
with the **same** variant (1000 SBC draws + the scenario grid × 50). Budget note: 40k, not the
production 160k, so compare variants with each other, not with nb31's 160k numbers.

## Decision rule — fixed before running
Compare the N3 ensemble with the N0 ensemble on Sc1–Sc5 (Sc7 is misspecified by design and
reported only). N3 changes a headline conclusion if, for any of these scenarios: (a) class
accuracy differs by more than 5 percentage points, (b) β 90% coverage at the truth falls outside
[0.85, 0.95], or (c) the α–β correlation changes regime (|r| < 0.2 uncorrelated, r ≥ 0.2
positive, r ≤ −0.2 negative). **If so, the user decides** whether production switches to N3
before Phase 3. If not, production stays N0 and N1–N3 go to the SI, with α's precision reported
under both. N4 is reported, not gated (it describes different instrumentation).

**Early result (precision limits, from the interrupted first run).** Realistic C noise (N1)
raises α's Cramér–Rao bound ~30–40× (Sc1 0.0014 → 0.054); β barely changes. Realistic T noise
(N2) tightens β's bound ~2.4× with the summaries, and removes most of the positive α–β
correlation at the saturated Sc5 (+0.76 → +0.07).

**Running.** Execute top to bottom from `notebooks/system_I` (≈1.5–2 h: FIM ~10 min, simulation
~20 min, 15 trainings of a few minutes each, evaluation ~10 min). Re-running resumes.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", \\
    f"execute this notebook from notebooks/system_I (cwd is {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import importlib
import ws_protocols, ws_eval, nb29_ws_study, nb33_sensor_noise
# reload the study code every time this cell runs: a kernel that imported an older version
# would otherwise keep using it (this is how N4 was silently skipped on 2026-11-08)
for _m in (ws_protocols, ws_eval, nb29_ws_study, nb33_sensor_noise):
    importlib.reload(_m)
N = nb33_sensor_noise
R, FIGS = ROOT / "results", ROOT / "figures"
THREADS = 4
T0 = time.time()
print({k: v for k, v in N.VARIANTS.items()})"""),
    ("md", """## 1. Information content per variant (Cramér–Rao bounds)
Gaussian Fisher information with common random numbers at the scenario points, for the 29-D
summaries (full covariance) and the raw trajectory (Ledoit–Wolf). This alone shows how much
α and β precision each sensor model allows, before any training."""),
    ("code", """N.stage_fim()
fim = pd.read_csv(R / "33_fim.csv")
for rep in ("summaries_fullcov", "raw_traj_lwcov"):
    f = fim[fim.representation == rep]
    print(rep)
    display(f.pivot_table(index="point", columns="variant", values=["crb_sd_a", "crb_sd_b"]).round(4))
display(fim[fim.representation == "summaries_fullcov"].pivot_table(index="point", columns="variant",
                                                                  values="corr_implied").round(3))"""),
    ("md", "## 2. Simulate training banks and test sets (all variants share θ draws and seeds)"),
    ("code", "N.stage_sim()"),
    ("md", "## 3. Train 5 variants × 3 seeds (40k each)"),
    ("code", "N.stage_train(THREADS)"),
    ("md", "## 4. Evaluate (singles and 3-seed ensembles, each on its own variant's test set)"),
    ("code", "N.stage_eval()"),
    ("code", """m = pd.read_csv(R / "33_metrics.csv"); m["member"] = m["member"].astype(str)
ens = m[m.member == "ensemble"].set_index("variant").reindex(list(N.VARIANTS))
display(ens[["ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p", "cov90_a", "cov90_b", "rmse_norm",
             "macro_f1", "leak_sbc_mean"]].round(3))
sc = pd.read_csv(R / "33_scenarios.csv"); sc["member"] = sc["member"].astype(str)
es = sc[sc.member == "ensemble"]
for k in ("class_acc", "cov90_truth_a", "cov90_truth_b", "sd_a", "sd_b", "corr_ab_mean", "alpha_hat", "beta_hat"):
    print(k)
    display(es.pivot_table(index="scenario", columns="variant", values=k).reindex(columns=list(N.VARIANTS)).round(4))"""),
    ("md", """### Posterior width against the information limit, per variant
Ensemble posterior sd divided by the 29-summary Cramér–Rao sd of the same variant. Values near
1 mean the estimator extracts what the summaries carry; the absolute sd (table above) is what
the paper's precision claims depend on."""),
    ("code", """crb = fim[fim.representation == "summaries_fullcov"].set_index(["variant", "point"])
es2 = es.assign(point=es.scenario.str.split("_").str[0])
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, par in zip(axes, ("a", "b")):
    for v in N.VARIANTS:
        d = es2[(es2.variant == v) & es2.point.isin(N.GATED)].set_index("point").reindex(N.GATED)
        ratio = [d.loc[p, f"sd_{par}"] / crb.loc[(v, p), f"crb_sd_{par}"] for p in N.GATED]
        ax.plot(N.GATED, ratio, "o-", label=v)
    ax.axhline(1, color="grey", lw=0.8); ax.set_yscale("log")
    ax.set_title(f"{'α' if par == 'a' else 'β'}: ensemble posterior sd / CRB (29 summaries)")
    ax.legend(fontsize=8)
fig.savefig(FIGS / "33_sd_over_crb.png", dpi=150); plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, par in zip(axes, ("a", "b")):
    for v in N.VARIANTS:
        d = es2[(es2.variant == v) & es2.point.isin(N.GATED)].set_index("point").reindex(N.GATED)
        ax.plot(N.GATED, d[f"sd_{par}"], "o-", label=v)
    ax.set_yscale("log"); ax.set_title(f"{'α' if par == 'a' else 'β'}: ensemble posterior sd (absolute)")
    ax.legend(fontsize=8)
fig.savefig(FIGS / "33_posterior_sd.png", dpi=150); plt.show()"""),
    ("md", "## 5. Decision (pre-set rule: N3 vs N0 on Sc1–Sc5)"),
    ("code", """res33 = N.stage_decide()
print("FLAG:", res33["flag"])
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")"""),
    ("md", """## 6. Interpretation
*(Written after the results.)* Report: how α's precision changes from N0 to N1 (realistic C) and
N3; whether β is affected (it should mainly depend on T, Tc and Qc, i.e. on N2); whether the
classification and the α–β correlation regimes survive; and the decision taken."""),
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
