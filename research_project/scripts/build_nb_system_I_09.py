"""Build / execute notebooks/system_I/09_sensor_drift_aware_N3.ipynb (System I, Phase 3 step f).

Drift-aware production posterior theta = [alpha, beta, dT] (scripts/nb09_drift_aware.py). Rules and the pre-set decision rule are in
docs/Manuscript/system_i_N3_runbook.md section 2f (row f). A written "## 8. Interpretation" is kept on rebuild. The old notebook 09 is untouched.

    python scripts/build_nb_system_I_09.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "09_sensor_drift_aware_N3.ipynb"

HEADER = """# 09 — A drift-aware production posterior [α, β, δT] (System I, N3)

**Why.** The production posterior (notebook 36) knows nothing about a temperature-sensor offset. Under Sc7 (+2 K on the T sensor) it is confidently wrong:
β̂ biased by −0.395, 90% coverage 0.002 (notebook 11), and the residual detector z_T flags it (notebook 11b). The question here is whether making the offset a parameter of the posterior removes
the failure, and what it costs in the well-specified scenarios. The result decides whether Sc7 enters the main text as a remedied failure.

**Model (fixed before running; runbook §2f row f).** θ = [α, β, δT]; prior α, β ~ U(0.4, 1.2) (production), δT ~ U(−3, 3) K. Simulator = the production model of notebook 36 (protocol P2, N3 sensors, true inlet from the notebook-35
distribution, *measured* Ti, Tci with ε = 0.1 K) with δT added to the temperature channel after the sensor layer, as in the Sc7 generation; S1m features from the drifted window. 20,000 simulations × 5 seeds, ensemble (as notebook 36).
δCi of the old notebook 09 is dropped (Ci is the unmeasured ±1% nuisance of the simulator). **Evaluation:** an SBC set of 1,000 prior draws; D1 and L1 (1,000 windows per scenario, seed base 3 800 000, as notebook 11), with the three
numbers per scenario for β and the δT bias, sd and coverage; the 2-D production numbers from notebook 11; a descriptive extension at (α, β) = (0.85, 0.85) with δT ∈ {0, 0.25, 0.5, 1, 2} (200 windows each).

**Pre-set decision rule.** All of: (i) ensemble SBC, interior subset (5% margin per dimension): KS p ≥ 0.01 for α, β and δT, TARP KS p ≥ 0.01; (ii) L1 Sc7: β coverage ≥ 0.80 and |β bias| ≤ 0.05;
(iii) L1 Sc7: δT coverage ≥ 0.80 and |δT bias| ≤ 0.30 K; (iv) L1 Sc1–Sc5: β coverage ≥ 0.80 each and macro-F1 ≥ 0.877 (production 0.907 minus a price of 0.03) → **FIXED** (Sc7 enters the main text with the remedy); otherwise
**NOT FIXED** (Sc7 stays a documented limitation, the drift-aware result goes to the SI). A failed (i) stops the notebook's decision step and is reported, not retuned.
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
import importlib, ws_eval, ws_metrics, ws_phase3, nb09_drift_aware
for _m in (ws_eval, ws_metrics, ws_phase3, nb09_drift_aware):
    importlib.reload(_m)
we, wm, P3, D = ws_eval, ws_metrics, ws_phase3, nb09_drift_aware
T0 = time.time(); R = ROOT / "results"; FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)
SCN = {1: "Sc1", 2: "Sc2", 3: "Sc3", 4: "Sc4", 5: "Sc5", 7: "Sc7"}
"""),
    ("md", "## 1. Simulation bank and training (cached; stage functions of `scripts/nb09_drift_aware.py`)"),
    ("code", """t = time.time(); D.stage_bank(); th, x = D.load_bank()
assert th.shape == (20000, 3) and x.shape == (20000, 34) and np.isfinite(x).all()
print(f"bank {th.shape}, features {x.shape}, all finite | dT prior draws in [{th[:,2].min():.2f}, {th[:,2].max():.2f}] K | {time.time() - t:.0f}s")
t = time.time(); D.stage_train(); print(f"trained / loaded 5 members in {time.time() - t:.0f}s")
post = D.ensemble()
"""),
    ("md", "## 2. SBC and TARP on 1,000 prior draws (3-D; interior subset)"),
    ("code", """from scipy import stats
sb = D.stage_sbc_set(); N = 1000
def sbc_report(p, name):
    s, leak = D.sample_box(p, sb["x"], N, seed=5)
    r = wm.sbc_ranks(s, sb["theta"]); inner = wm.interior_mask(sb["theta"], D.LO, D.HI)
    ks = wm.sbc_ks_pvalues(r[inner], N, seed=0); cov = wm.marginal_coverage(r, N, 0.90)
    tr = wm.tarp(s[inner], sb["theta"][inner], D.LO, D.HI)
    return dict(posterior=name, n_interior=int(inner.sum()), ks_alpha=ks[0], ks_beta=ks[1], ks_dT=ks[2], tarp_ks_p=tr["tarp_ks_p"], tarp_dev=tr["tarp_dev"],
                cov90_alpha=cov[0], cov90_beta=cov[1], cov90_dT=cov[2], leakage=float(leak.mean())), s
rows = []
for k, m in enumerate(D.load_members()):
    rows.append(sbc_report(m, f"seed {k}")[0])
ens_row, s_sbc = sbc_report(post, "ensemble"); rows.append(ens_row)
sbc = pd.DataFrame(rows); display(sbc.round(3)); sbc.to_csv(R / "09_N3_sbc.csv", index=False)
rm = ((s_sbc.mean(1) - sb["theta"]) ** 2).mean(0) ** 0.5
print("ensemble posterior-mean RMSE per parameter (alpha, beta, dT):", np.round(rm, 3), "| posterior sd (mean):", np.round(s_sbc.std(1).mean(0), 3))
"""),
    ("md", "## 3. Scenarios: D1 and L1 (three numbers per scenario, δT row, and the 2-D production posterior for comparison)"),
    ("code", """d1 = P3.load_d1(); dL = P3.generate_d1(n_rep=1000, seed_base=3_800_000)
def evaluate(d, tag):
    X = P3.production_features(d); s, leak = D.sample_box(post, X, 1000, seed=11)
    th = d["theta"][:, 2:]; sid = d["scenario_id"]; sat = (d["x"][:, :, 3] >= 395).mean(1) > 0.5
    truth_dT = np.where(sid == 7, 2.0, 0.0)
    yt = wm.class_of_theta(th.astype(float)); yp = wm.class_of_samples(s[:, :, :2])
    q05 = np.quantile(s, 0.05, axis=1); q95 = np.quantile(s, 0.95, axis=1); mean = s.mean(1); sd = s.std(1)
    rows = []
    for sc in SCN:
        m = sid == sc
        eb = mean[m, 1] - th[m, 1]; ea = mean[m, 0] - th[m, 0]; ed = mean[m, 2] - truth_dT[m]
        rows.append(dict(data=tag, scenario=SCN[sc], n=int(m.sum()), class_acc=float((yt[m] == yp[m]).mean()),
                         cov90_b=float(((th[m, 1] >= q05[m, 1]) & (th[m, 1] <= q95[m, 1])).mean()), sd_b=float(sd[m, 1].mean()), bias_b=float(eb.mean()),
                         cov90_a=float(((th[m, 0] >= q05[m, 0]) & (th[m, 0] <= q95[m, 0])).mean()), sd_a=float(sd[m, 0].mean()), bias_a=float(ea.mean()),
                         dT_true=float(truth_dT[m][0]), dT_bias=float(ed.mean()), dT_sd=float(sd[m, 2].mean()),
                         cov90_dT=float(((truth_dT[m] >= q05[m, 2]) & (truth_dT[m] <= q95[m, 2])).mean())))
    return pd.DataFrame(rows), dict(s=s, leak=leak, th=th, sid=sid, yt=yt, yp=yp, sat=sat)
t1, e1 = evaluate(d1, "D1"); tL, eL = evaluate(dL, "L1")
assert np.isfinite(eL["s"]).all()
print("L1: prior leakage mean %.4f, max %.3f" % (eL["leak"].mean(), eL["leak"].max()))
prod = pd.read_csv(R / "11_N3_L1_three_numbers.csv")
cmp = tL.merge(prod[["scenario", "class_acc", "cov90_b", "sd_b", "bias_b"]], on="scenario", suffixes=("_drift_aware", "_2D_production"))
cols = ["scenario", "class_acc_drift_aware", "class_acc_2D_production", "cov90_b_drift_aware", "cov90_b_2D_production", "sd_b_drift_aware", "sd_b_2D_production", "bias_b_drift_aware", "bias_b_2D_production",
        "dT_true", "dT_bias", "dT_sd", "cov90_dT"]
display(cmp[cols].round(3)); display(t1[["scenario", "n", "class_acc", "cov90_b", "sd_b", "bias_b", "dT_true", "dT_bias", "dT_sd", "cov90_dT"]].round(3))
tL.to_csv(R / "09_N3_L1_three_numbers.csv", index=False); t1.to_csv(R / "09_N3_D1_three_numbers.csv", index=False); cmp[cols].to_csv(R / "09_N3_L1_vs_2D_production.csv", index=False)
"""),
    ("md", "## 4. Macro-F1 and the price in the well-specified scenarios"),
    ("code", """def macro(e, sel):
    m = np.isin(e["sid"], sel); return wm.macro_f1(e["yt"][m], e["yp"][m]), float((e["yt"][m] == e["yp"][m]).mean())
mf15, acc15 = macro(eL, [1, 2, 3, 4, 5]); mf_all, acc_all = macro(eL, [1, 2, 3, 4, 5, 7])
mf15_d1, acc15_d1 = macro(e1, [1, 2, 3, 4, 5])
print(f"L1 macro-F1 Sc1-Sc5: {mf15:.3f} (accuracy {acc15:.3f}) | all six: {mf_all:.3f} | production 2-D: 0.907 (Sc1-Sc5), 0.824 (all six) | D1 Sc1-Sc5: {mf15_d1:.3f}")
"""),
    ("md", "## 5. Decision rule (fixed before running)"),
    ("code", """sc7 = tL[tL.scenario == "Sc7"].iloc[0]; ens = sbc[sbc.posterior == "ensemble"].iloc[0]
crit = {
 "(i) SBC: interior KS p >= 0.01 (alpha, beta, dT) and TARP p >= 0.01": bool(min(ens.ks_alpha, ens.ks_beta, ens.ks_dT, ens.tarp_ks_p) >= 0.01),
 "(ii) Sc7: beta coverage >= 0.80 and |beta bias| <= 0.05": bool(sc7.cov90_b >= 0.80 and abs(sc7.bias_b) <= 0.05),
 "(iii) Sc7: dT coverage >= 0.80 and |dT bias| <= 0.30 K": bool(sc7.cov90_dT >= 0.80 and abs(sc7.dT_bias) <= 0.30),
 "(iv) Sc1-Sc5: beta coverage >= 0.80 each and macro-F1 >= 0.877": bool((tL[tL.scenario != "Sc7"].cov90_b >= 0.80).all() and mf15 >= 0.877)}
for k, v in crit.items():
    print("PASS" if v else "FAIL", k)
verdict = "FIXED" if all(crit.values()) else "NOT FIXED"
print("\\nDECISION:", verdict, "| Sc7 beta cov %.3f bias %+.3f | dT cov %.3f bias %+.3f | macro-F1 %.3f" % (sc7.cov90_b, sc7.bias_b, sc7.cov90_dT, sc7.dT_bias, mf15))
json.dump({"verdict": verdict, "criteria": crit, "macro_f1_Sc1_5": mf15, "sc7": sc7.to_dict()}, open(R / "09_N3_decision.json", "w"), indent=1, default=float)
assert crit["(i) SBC: interior KS p >= 0.01 (alpha, beta, dT) and TARP p >= 0.01"], "STOP: SBC failed (reported, no retuning)"
"""),
    ("md", "## 6. Descriptive extension: sensitivity to the size of the offset (α = β = 0.85)"),
    ("code", """import ws_inlet as wi, nb35_measured_inlet as M, nb36_production_measured_inlet as P36
rows = []
for k, dt in enumerate((0.0, 0.25, 0.5, 1.0, 2.0)):
    n = 200; th3 = np.column_stack([np.full(n, 0.85), np.full(n, 0.85), np.full(n, dt)]).astype(np.float32)
    inl = M._inlets(n, 9_400_000 + k); xs = D.simulate_drifted(th3, inl, 9_450_000 + 1000 * k, 9_460_000 + k)
    s, _ = D.sample_box(post, xs, 1000, seed=k); mean = s.mean(1); q05 = np.quantile(s, .05, axis=1); q95 = np.quantile(s, .95, axis=1)
    raw = np.array(wi.simulate(th3[:, :2], inl, D.PROTOCOL, 9_450_000 + 1000 * k, raw=True, **D.NOISE), np.float32); raw[:, :, 1] += dt
    zT = (raw[:, :, 1].mean(1) - 312.5) / (0.2 / np.sqrt(120)); sat = (raw[:, :, 3] >= 395).mean(1) > 0.5
    rows.append(dict(dT_true=dt, n=n, dT_hat_mean=float(mean[:, 2].mean()), dT_sd=float(s[:, :, 2].std(1).mean()), cov90_dT=float(((dt >= q05[:, 2]) & (dt <= q95[:, 2])).mean()),
                     beta_bias=float((mean[:, 1] - 0.85).mean()), cov90_b=float(((0.85 >= q05[:, 1]) & (0.85 <= q95[:, 1])).mean()),
                     frac_unsat=float((~sat).mean()), zT_gt10_unsat=float((np.abs(zT[~sat]) > 10).mean()) if (~sat).any() else np.nan))
ext = pd.DataFrame(rows); display(ext.round(3)); ext.to_csv(R / "09_N3_dT_sensitivity.csv", index=False)
"""),
    ("md", "## 7. Figure and timing"),
    ("code", """fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for a_, (k, nm) in zip(ax, ((1, "β"), (0, "α"), (2, "δT (K)"))):
    data = [eL["s"][eL["sid"] == sc][:, :, k].mean(1) for sc in SCN]
    a_.boxplot(data, tick_labels=list(SCN.values()), flierprops=dict(ms=2, alpha=.3)); a_.set_ylabel(f"posterior mean {nm}")
    if k == 1: a_.axhline(0.85, color="g", ls="--", lw=.8); a_.set_title("β̂ (L1): truth 0.85 in Sc2, Sc4(0.85/0.85) etc.")
    if k == 2: a_.axhline(2.0, color="g", ls="--", lw=.8); a_.axhline(0, color="g", ls="--", lw=.8); a_.set_title("δT̂ (L1): Sc7 truth +2 K, others 0")
fig.tight_layout(); fig.savefig(FIGS / "09_N3_drift_aware.png", dpi=150); plt.show()
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 8. Interpretation — to be written after the run
Toward the storyline: did the drift-aware posterior remove the Sc7 silent failure, at what price in β precision and classification of the well-specified scenarios, how large an offset does the posterior need to see
(the sensitivity table), and how does this relate to the z_T guard of notebook 11b. State the pre-set verdict and what it means for the main text."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 8. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 8. Interpretation"):
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
