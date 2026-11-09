"""Build / execute notebooks/system_I/08_identifiability_and_saturation_N3.ipynb (System I, Phase 3 step g).

Fisher information of the summary statistics at the scenario points, with the inlet nominal (S1, naive) and with the inlet as a nuisance
(S1m, measured inlet), and its change with fault severity and saturation. Rules in docs/Manuscript/system_i_N3_runbook.md section 2f
(row g-08). A written "## 7. Interpretation" is kept on rebuild.

    python scripts/build_nb_system_I_08.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "08_identifiability_and_saturation_N3.ipynb"

HEADER = """# 08 — Identifiability and saturation: Fisher information with the inlet nominal and as a nuisance (System I, N3)

**Question.** How much does a 60-minute window say about (α, β), how does that change with fault severity and with valve saturation, and what does an unknown inlet (within ±2 K, measured to 0.1 K) cost? The old notebook answered it with the pre-fix posterior;
the Fisher information of the summaries answers it without a posterior.

**Method (fixed before running; runbook §2f row g-08).** Gaussian Fisher information of the summary statistics, I = Gᵀ Σ⁻¹ G, with G the mean derivative of the features with respect to (α, β) (central finite differences, h = 0.01, **common random numbers** across the five
simulations per point) and Σ the full covariance of the standardised features at the point (the dΣ/dθ term is neglected, as in notebook 29 `_fim`; constant features are dropped); 1,000 replicates per simulation. Two modes at every point: **nominal** — the inlet fixed at its nominal value, the naive set S1 (32 features);
**nuisance** — the inlet drawn per replicate from the production distribution (Ti, Tci ~ U(295, 299) K, Ci ± 1%), features S1m (34: with the measured Ti, Tci, ε = 0.1 K, as in notebook 36). In the nuisance mode the full covariance includes the measured temperatures, so the information is the marginal one with the inlet *observed* through its measurement and the ε = 0.1 K error and the unmeasured Ci as the unavoidable nuisance.
**Points:** the scenario points Sc1 (1, 1), Sc2 (1, 0.85), Sc3 (0.7, 1), Sc4 (0.85, 0.85), Sc5 (1, 0.5) and the severity series at α = 1: β ∈ {1, 0.85, 0.7, 0.5, 0.4}. Per point: the Cramér–Rao standard deviations of α and β, the ratio I_αα/I_ββ, the implied correlation, and the fraction of samples at Qc ≥ 395 L/min.
**Checks:** the FIM is positive definite and the CRBs finite; the nominal-mode S1 CRBs at Sc1, Sc3, Sc4 and at β = 0.7 agree within 25% with those of notebook 03 section 10b (`03s_N3_fim.csv`, a different seed; its "Sc2" is the old point β = 0.7, not today's Sc2 at β = 0.85, so it is compared with the severity point β = 0.7 here). The raw-trajectory bound is not repeated: it is invalid for β under N3 (notebook 31, Ledoit–Wolf). No decision rule.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
warnings.filterwarnings("ignore")
import importlib, ws_eval, ws_inlet, nb35_measured_inlet, nb36_production_measured_inlet, nb29_ws_study, ws_phase3
for _m in (ws_eval, ws_inlet, nb35_measured_inlet, nb36_production_measured_inlet, nb29_ws_study, ws_phase3):
    importlib.reload(_m)
we, wi, M, P36, N29 = ws_eval, ws_inlet, nb35_measured_inlet, nb36_production_measured_inlet, nb29_ws_study
R = ROOT / "results"; FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True); T0 = time.time()
N_REP, H = 1000, 0.01
POINTS = {"Sc1": (1.0, 1.0), "Sc2": (1.0, 0.85), "Sc3": (0.7, 1.0), "Sc4": (0.85, 0.85), "Sc5": (1.0, 0.5), "sev beta=0.7": (1.0, 0.7), "sev beta=0.4": (1.0, 0.4)}
"""),
    ("md", "## 1. Fisher information at every point, both modes (common random numbers across the finite-difference variants)"),
    ("code", """rows = []
def variants(th0):
    th0 = np.array(th0, np.float32); return [th0, th0 + [H, 0], th0 - [H, 0], th0 + [0, H], th0 - [0, H]]
for pi, (name, th0) in enumerate(POINTS.items()):
    vs = variants(th0)
    # nominal inlet, naive features S1
    X_nom = [we.simulate(np.tile(v, (N_REP, 1)), M.PROTOCOL, 8_100_000 + pi, features="S1_base_est", **M.NOISE) for v in vs]
    # inlet as a nuisance (same inlet draws and the same measurement draws for the five variants), production features S1m, and the saturation fraction
    inl = M._inlets(N_REP, 8_200_000 + pi); X_nui, sat = [], []
    for v in vs:
        raw = np.array(wi.simulate(np.tile(v, (N_REP, 1)), inl, M.PROTOCOL, 8_300_000 + pi, raw=True, **M.NOISE), np.float32)
        t = np.arange(1, raw.shape[1] + 1) * 0.5
        X_nui.append(P36.features(we.features(raw, t, "S0_base29"), inl, 8_400_000 + pi)); sat.append(float((raw[:, :, 3] >= 395).mean()))
    for mode, arrs in (("nominal (S1)", X_nom), ("nuisance (S1m)", X_nui)):
        X0, ap, am, bp, bm = arrs
        G = np.stack([(ap - am).mean(0) / (2 * H), (bp - bm).mean(0) / (2 * H)], axis=1)
        F = N29._fim(G, X0, "full"); ev = np.linalg.eigvalsh(F); Fi = np.linalg.inv(F)
        rows.append(dict(point=name, alpha=th0[0], beta=th0[1], mode=mode, I_aa=F[0, 0], I_bb=F[1, 1], I_ab=F[0, 1], ratio_Iaa_Ibb=F[0, 0] / F[1, 1], corr_implied=float(-F[0, 1] / np.sqrt(F[0, 0] * F[1, 1])),
                         crb_sd_a=float(np.sqrt(Fi[0, 0])), crb_sd_b=float(np.sqrt(Fi[1, 1])), min_eig=float(ev.min()), frac_Qc_ge395=float(np.mean(sat))))
    print(name, "done", round(time.time() - T0), "s", flush=True)
fim = pd.DataFrame(rows); fim.to_csv(R / "08_N3_fim.csv", index=False)
assert (fim.min_eig > 0).all() and np.isfinite(fim[["crb_sd_a", "crb_sd_b"]]).all().all(), "FIM check failed"
display(fim.round(4))
"""),
    ("md", "## 2. Check against notebook 03 section 10b (nominal mode, S1, Sc1–Sc4)"),
    ("code", """ref = pd.read_csv(R / "03s_N3_fim.csv"); ref = ref[ref.set == "S1_base_est"].set_index("point")
nom = fim[fim["mode"] == "nominal (S1)"].set_index("point")
# notebook 03 section 10b (and notebook 29) label (alpha, beta) = (1, 0.7) "Sc2": the catalogue point was moved to beta = 0.85 on 2026-11-08 and those scripts still carry the old one (E2_POINTS in scripts/nb03_summary_study.py).
# Its counterpart here is the severity point beta = 0.7. (The first run of this notebook compared "Sc2" with "Sc2" and failed the check for exactly this reason.)
pair = {"Sc1": "Sc1", "Sc2": "sev beta=0.7", "Sc3": "Sc3", "Sc4": "Sc4"}
cmp_ = pd.DataFrame({"point_here": [pair[k] for k in ref.index], "crb_a_here": [nom.loc[pair[k], "crb_sd_a"] for k in ref.index], "crb_a_nb03": ref.crb_sd_a.values,
                     "crb_b_here": [nom.loc[pair[k], "crb_sd_b"] for k in ref.index], "crb_b_nb03": ref.crb_sd_b.values}, index=ref.index)
cmp_["ratio_a"] = cmp_.crb_a_here / cmp_.crb_a_nb03; cmp_["ratio_b"] = cmp_.crb_b_here / cmp_.crb_b_nb03; display(cmp_.round(3))
assert ((cmp_.ratio_a - 1).abs() < 0.25).all() and ((cmp_.ratio_b - 1).abs() < 0.25).all(), "agreement check failed"
print("check passed: nominal-mode CRBs agree with notebook 03 section 10b within 25%")
"""),
    ("md", "## 3. Comparison with the production posterior width and the figure (severity series at α = 1)"),
    ("code", """sev = ["sev beta=0.4", "Sc5", "sev beta=0.7", "Sc2", "Sc1"]; betas = [0.4, 0.5, 0.7, 0.85, 1.0]
d1 = ws_phase3.load_d1(); post, _ = ws_phase3.load_production_posterior(); Xo = ws_phase3.production_features(d1)
s, _ = we.sample_within_prior(post, Xo, 1000, rng_seed=5); sdb = {f"Sc{k}": float(s.std(1)[d1["scenario_id"] == k, 1].mean()) for k in (1, 2, 3, 4, 5)}
print("production posterior beta sd on D1 (mean over 50 windows):", {k: round(v, 3) for k, v in sdb.items()})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for mode, c in (("nominal (S1)", "tab:blue"), ("nuisance (S1m)", "tab:red")):
    f = fim[fim["mode"] == mode].set_index("point")
    ax[0].plot(betas, f.loc[sev, "crb_sd_b"], "o-", color=c, label=f"CRB β, {mode}"); ax[1].plot(betas, f.loc[sev, "ratio_Iaa_Ibb"], "o-", color=c, label=mode); ax[2].plot(betas, f.loc[sev, "corr_implied"], "o-", color=c, label=mode)
ax[0].plot([1.0, 0.85, 0.5], [sdb["Sc1"], sdb["Sc2"], sdb["Sc5"]], "k^", label="production posterior sd (D1)")
ax[0].set_ylabel("sd(β)"); ax[1].set_ylabel("I_αα / I_ββ"); ax[1].set_yscale("log"); ax[2].set_ylabel("implied corr(α, β)")
for a_, t_ in zip(ax, ("β resolution vs severity (α = 1)", "information ratio", "implied correlation")):
    a_.set_xlabel("true β"); a_.set_title(t_); a_.grid(alpha=.3); a_.legend(fontsize=7); a_.invert_xaxis()
fig.tight_layout(); fig.savefig(FIGS / "08_N3_fim_vs_severity.png", dpi=150); plt.show()
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 7. Interpretation — to be written after the run
Toward the storyline: the Cramér–Rao resolution of α and β at the scenario points with the inlet nominal and as a nuisance (the price of the inlet), how it changes with severity and saturation, whether α and β are confounded, and how close the production posterior sits to the bound."""),
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
