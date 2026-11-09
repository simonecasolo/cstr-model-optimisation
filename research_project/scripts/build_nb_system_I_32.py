"""Build / execute notebooks/system_I/32_alpha_beta_mechanism_N3.ipynb (System I, Phase 3 step g).

Consolidates the old notebooks 19, 20 and 26 under the fixed controller law, protocol P2 and the production posterior: the (alpha, beta) correlation
of the FIM (notebook 08 results) and of the posterior, by scenario and by saturation, and the effect of the unmeasured inlet concentration Ci.
Rules in docs/Manuscript/system_i_N3_runbook.md section 2f (row g-32). A written "## 6. Interpretation" is kept on rebuild.

    python scripts/build_nb_system_I_32.py [--execute]       (needs results/08_N3_fim.csv from notebook 08)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "32_alpha_beta_mechanism_N3.ipynb"

HEADER = """# 32 — Where does an (α, β) correlation come from? (System I, N3)

The old notebooks 19, 20 and 26 chased an (α, β) "banana" (a negative correlation of the posterior) that came from the malformed controller, the fixed-horizon warm start and the old sensor noise. This notebook asks the question again under the corrected model: **the fixed PI law, the P2 warm start, N3 sensors and the production posterior.**

**Content (fixed before running; runbook §2f row g-32).** (1) The correlation implied by the Fisher information at the scenario points, with the inlet nominal and as a nuisance (notebook 08's results), against the fraction of samples at the upper valve limit. (2) The posterior correlation of α and β per scenario: the production posterior on D1 (and, for reference, the naive posterior on D0),
overall and split by saturation (more than 50% of a window's samples at Qc ≥ 395 L/min). (3) The effect of the **unmeasured inlet concentration Ci** (±1%) on the information: the FIM at Sc1, Sc4 and Sc5 with Ti and Tci exactly nominal and Ci drawn per replicate, against the nominal FIM of notebook 08 (same method: central differences h = 0.01 with common random numbers, 1,000 replicates, full covariance).
**Checks:** finite; the FIM at the nominal inlet reproduced from notebook 08's table within the sampling noise (the Ci-only run has its own replicates: |Δ CRB| of the nominal run versus notebook 08 is not tested, the Ci run is compared with it). No decision rule.
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
import importlib, ws_eval, ws_inlet, nb35_measured_inlet, nb29_ws_study, ws_phase3, ws_phase3_eval
for _m in (ws_eval, ws_inlet, nb35_measured_inlet, nb29_ws_study, ws_phase3, ws_phase3_eval):
    importlib.reload(_m)
we, wi, M, N29, P3, E = ws_eval, ws_inlet, nb35_measured_inlet, nb29_ws_study, ws_phase3, ws_phase3_eval
R = ROOT / "results"; FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True); T0 = time.time()
fim = pd.read_csv(R / "08_N3_fim.csv"); assert np.isfinite(fim.corr_implied).all()
"""),
    ("md", "## 1. The correlation implied by the Fisher information (notebook 08)"),
    ("code", """t = fim.pivot_table(index="point", columns="mode", values=["corr_implied", "crb_sd_a", "crb_sd_b", "ratio_Iaa_Ibb"]); sat = fim.groupby("point").frac_Qc_ge395.first()
t[("frac_Qc_ge395", "")] = sat; display(t.round(3))
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for mode, c in (("nominal (S1)", "tab:blue"), ("nuisance (S1m)", "tab:red")):
    f = fim[fim["mode"] == mode]; ax.plot(f.frac_Qc_ge395, f.corr_implied, "o", color=c, label=mode)
    for _, r in f.iterrows(): ax.annotate(r.point, (r.frac_Qc_ge395, r.corr_implied), fontsize=6)
ax.set_xlabel("fraction of samples at Qc ≥ 395"); ax.set_ylabel("implied corr(α, β)"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "32_N3_fim_corr_vs_saturation.png", dpi=150); plt.show()
"""),
    ("md", "## 2. The posterior correlation of (α, β): per scenario and by saturation"),
    ("code", """rows = []
for lab, which, dn, seed in (("production on D1", "prod", "D1", 71), ("naive on D0", "naive", "D0", 72)):
    s, d, _ = E.draws(which, dn, 1000, seed=seed)
    c = np.array([np.corrcoef(s[i, :, 0], s[i, :, 1])[0, 1] for i in range(len(s))]); sid = d["scenario_id"]
    sat_ = (d["x"][:, :, 3] >= 395).mean(1) > 0.5
    for sc, nm in E.SCN.items():
        for grp, m in (("all", sid == sc), ("unsaturated", (sid == sc) & ~sat_), ("saturated", (sid == sc) & sat_)):
            if m.sum() >= 3: rows.append(dict(posterior=lab, scenario=nm, group=grp, n=int(m.sum()), corr_mean=float(np.nanmean(c[m])), corr_sd=float(np.nanstd(c[m]))))
pc = pd.DataFrame(rows); display(pc.pivot_table(index=["posterior", "scenario"], columns="group", values="corr_mean").round(3)); pc.to_csv(R / "32_N3_posterior_corr.csv", index=False)
"""),
    ("md", "## 3. The effect of the unmeasured inlet concentration Ci (Ti and Tci exactly nominal; Ci ± 1% per replicate)"),
    ("code", """N_REP, H = 1000, 0.01; rows = []
for pi, (name, th0) in enumerate({"Sc1": (1.0, 1.0), "Sc4": (0.85, 0.85), "Sc5": (1.0, 0.5)}.items()):
    th0 = np.array(th0, np.float32); vs = [th0, th0 + [H, 0], th0 - [H, 0], th0 + [0, H], th0 - [0, H]]
    rng = np.random.default_rng(8_500_000 + pi); nom = wi.NOMINAL.astype(np.float64)
    inl = np.tile(nom, (N_REP, 1)); inl[:, 0] = nom[0] * (1 + rng.normal(0, M.CI_REL_SD, N_REP)); inl = inl.astype(np.float32)
    arrs = [we.features(np.array(wi.simulate(np.tile(v, (N_REP, 1)), inl, M.PROTOCOL, 8_600_000 + pi, raw=True, **M.NOISE), np.float32), np.arange(1, 121) * 0.5, "S1_base_est") for v in vs]
    X0, ap, am, bp, bm = arrs; G = np.stack([(ap - am).mean(0) / (2 * H), (bp - bm).mean(0) / (2 * H)], axis=1)
    F = N29._fim(G, X0, "full"); Fi = np.linalg.inv(F)
    rows.append(dict(point=name, mode="Ci only (Ti, Tci nominal)", I_aa=F[0, 0], I_bb=F[1, 1], ratio_Iaa_Ibb=F[0, 0] / F[1, 1], corr_implied=float(-F[0, 1] / np.sqrt(F[0, 0] * F[1, 1])), crb_sd_a=float(np.sqrt(Fi[0, 0])), crb_sd_b=float(np.sqrt(Fi[1, 1]))))
ci = pd.DataFrame(rows); assert np.isfinite(ci[["crb_sd_a", "crb_sd_b"]]).all().all()
nom = fim[fim["mode"] == "nominal (S1)"].set_index("point"); nui = fim[fim["mode"] == "nuisance (S1m)"].set_index("point")
cmp_ = pd.DataFrame({"crb_b_nominal": nom.loc[ci.point, "crb_sd_b"].values, "crb_b_Ci_only": ci.crb_sd_b.values, "crb_b_all_inlet_nuisance": nui.loc[ci.point, "crb_sd_b"].values,
                     "crb_a_nominal": nom.loc[ci.point, "crb_sd_a"].values, "crb_a_Ci_only": ci.crb_sd_a.values, "crb_a_all_inlet_nuisance": nui.loc[ci.point, "crb_sd_a"].values,
                     "corr_nominal": nom.loc[ci.point, "corr_implied"].values, "corr_Ci_only": ci.corr_implied.values, "corr_all_inlet_nuisance": nui.loc[ci.point, "corr_implied"].values}, index=ci.point)
display(cmp_.round(4)); cmp_.to_csv(R / "32_N3_ci_effect.csv"); print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 6. Interpretation — to be written after the run
Toward the storyline: the old banana is gone; what correlation remains, where (saturation), how large, what the unmeasured Ci adds, and what the mechanism is."""),
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
