"""Build / execute notebooks/system_I/16_ekf_ukf_baseline_N3.ipynb (System I, Phase 3 step c).

EKF / UKF baselines (scripts/ws_filters.py, fixed standard PI law) against the production SBI posterior on D1 and D0.
Rules and checks fixed in docs/Manuscript/system_i_N3_runbook.md section 2f (row c). A written section-"9. Interpretation"
is kept when the notebook is rebuilt. The old notebook 16 is left untouched (pre-fix record in logs/).

    python scripts/build_nb_system_I_16.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "16_ekf_ukf_baseline_N3.ipynb"

HEADER = """# 16 — EKF / UKF baselines against the production SBI posterior (System I, N3)

**Why.** The paper needs the industrial comparator: a joint state–parameter EKF / UKF with the same model, the same sensors and, where
the production posterior is given them, the same inlet information. The old notebook 16 used the malformed PI law (no Kp on the integral
term; fixed 2026-11-08), a 60-min window, the old sensor noise and the old summary-set SBI posterior, so none of its numbers carry over.
`scripts/ws_filters.py` (tests: `tests/test_ws_filters.py`) rebuilds the filters on the physics constants and the standard PI law.

**Variants (fixed before running; runbook §2f row c).**
* **aug** — 8-state: [C, T, Tc, I, α, β] plus two constant offset states (dTi, dTci, prior sd 0.1 K); the *measured* inlet temperatures
  are known inputs and the plant sees measured − offset. This is the filter counterpart of the production posterior's inlet model.
* **known** — 6-state, measured inlet temperatures as known inputs, no offset states.
* **naive** — 6-state, nominal inlet (297 K, 297 K) assumed: valid on D0 (nominal-inlet data), a sensitivity test on D1.

**Data.** D1 (`observations_measured_inlet.npz`, Sc1–Sc5 and Sc7 × 50, measured inlet) for aug / known / naive; D0
(`observations.npz`, same scenarios, nominal inlet) for naive. One configuration of each filter for every scenario (no tuning per
scenario); N3 sensor sd in R. The SBI production row is read from notebook 11's table (same D1 windows).

**Three numbers per method and scenario**, as in notebook 11: class accuracy (quadrant mass of independent Gaussians N(estimate, sd),
threshold 0.90, 2,000 draws), β 90% coverage (|estimate − truth| < 1.645 sd), β sd (mean reported sd); plus bias, RMSE and the
empirical sd of the estimates, and ms per window. **Checks:** (0) the filter tests pass; (1) all estimates finite with positive
variances (stop if > 5% of windows fail in a variant); (2) naive filter on D0 Sc1: |α bias|, |β bias| < 0.30; (3) mean aug dTci estimate on
D1 within ±0.3 K. No decision rule. Tracking on the new 30-day stream is added after notebook 10 is rebuilt.

**Tuning (c3; user rule, fixed before running; runbook §2f row c3).** The filters above use one hand-set Q, R, P0. To make the comparison fair they are also tuned,
once, on a **separate tuning set**: 50 windows per scenario for Sc1–Sc5 (250 windows), new seed base 4 100 000, generated like D1 (measured inlet, P2, N3). A 5 × 5 grid of scale
factors on the process-noise Q and on the measurement-noise R, each in {0.25, 0.5, 1, 2, 4}; objective = the mean predictive log-likelihood of the measurements (the summed Gaussian innovation
log-density of each window, averaged over windows), separately for the augmented EKF and UKF, one setting each for all scenarios. The tuned augmented filters are then re-evaluated on the D1
test windows (not used in tuning) and set next to the untuned numbers (coverage, RMSE, tail count: |β error| > 0.15). A boundary optimum is flagged, not extended. The tail windows of the untuned run are examined
(initialisation, divergence, observability) and reported.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
import importlib, ws_filters, ws_metrics, ws_phase3
for _m in (ws_filters, ws_metrics, ws_phase3):
    importlib.reload(_m)
F, wm, P3 = ws_filters, ws_metrics, ws_phase3
T0 = time.time(); R = ROOT / "results"
SCN = {1: "Sc1", 2: "Sc2", 3: "Sc3", 4: "Sc4", 5: "Sc5", 7: "Sc7"}
# check 0: the filter tests
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_ws_filters.py", "-q"], cwd=ROOT, capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1])
assert r.returncode == 0, r.stdout[-2000:]
d1, d0 = P3.load_d1(), P3.load_d0()
keep0 = np.isin(d0["scenario_id"], list(SCN))
print("D1:", d1["x"].shape, "| D0 (Sc1-5, Sc7):", int(keep0.sum()), "windows")
"""),
    ("md", "## 1. Run the filters (serial; UKF is the slow one)"),
    ("code", """def run(d, ids, method, mode):
    rows = []
    for i in ids:
        if mode == "naive":
            ti, tci, aug = F.TI0, F.TCI0, False
        else:
            ti, tci = (float(v) for v in d["inlet_measured"][i]); aug = mode == "aug"
        try:
            o = F.estimate_window(d["x"][i], method, ti, tci, F.CI0, augment=aug)
        except Exception as e:                      # a filter failure is counted, not hidden
            o = dict(alpha=np.nan, beta=np.nan, sd_alpha=np.nan, sd_beta=np.nan, wall_s=np.nan, dTi=np.nan, dTci=np.nan, err=str(e)[:80])
        rows.append(dict(win=int(i), sc=int(d["scenario_id"][i]), a_true=float(d["theta"][i, 2]), b_true=float(d["theta"][i, 3]),
                         a=o["alpha"], b=o["beta"], sd_a=o["sd_alpha"], sd_b=o["sd_beta"], ms=1000 * o["wall_s"],
                         dTci=o.get("dTci", np.nan), sat=float((d["x"][i][:, 3] >= 395).mean())))
    out = pd.DataFrame(rows); out["method"] = method.upper(); out["variant"] = mode
    return out

res = {}
ids1 = np.arange(len(d1["x"])); ids0 = np.where(keep0)[0]
for dname, d, ids, modes in (("D1", d1, ids1, ("aug", "known", "naive")), ("D0", d0, ids0, ("naive",))):
    for mode in modes:
        for method in ("ekf", "ukf"):
            t = time.time(); res[(dname, mode, method)] = run(d, ids, method, mode)
            print(f"{dname} {mode:5s} {method.upper()}: {time.time() - t:6.1f}s", flush=True)
allres = pd.concat([v.assign(data=k[0]) for k, v in res.items()], ignore_index=True)
print("total", len(allres), "estimates | wall %.1f min" % ((time.time() - T0) / 60))
"""),
    ("md", "## 2. Checks 1–3"),
    ("code", """bad = allres.groupby(["data", "variant", "method"]).apply(lambda g: float((~np.isfinite(g[["a", "b", "sd_a", "sd_b"]]).all(axis=1) | (g.sd_a <= 0) | (g.sd_b <= 0)).mean()))
print("fraction of failed windows per variant:\\n", bad.round(4).to_string())
assert (bad <= 0.05).all(), "check 1 failed"
s1 = allres[(allres.data == "D0") & (allres.sc == 1) & (allres.variant == "naive")].groupby("method")[["a", "b", "a_true", "b_true"]].mean()
print("\\nD0 Sc1 naive filter, mean estimate:\\n", s1.round(3).to_string())
for m, row in s1.iterrows():
    assert abs(row.a - row.a_true) < 0.30 and abs(row.b - row.b_true) < 0.30, f"check 2 failed ({m})"
dt = allres[(allres.data == "D1") & (allres.variant == "aug")].groupby("method").dTci.mean()
print("\\nmean aug dTci estimate on D1 (K):", dt.round(3).to_dict())
assert (dt.abs() < 0.3).all(), "check 3 failed"
print("checks 0-3 passed")
"""),
    ("md", "## 3. Three numbers per method and scenario (D1, headline) with the SBI production row"),
    ("code", """CLS = list(wm.CLASSES)
rng = np.random.default_rng(0)

def classify(g):
    z = rng.standard_normal((len(g), 2000, 2))
    smp = np.stack([g.a.values[:, None] + g.sd_a.values[:, None] * z[:, :, 0], g.b.values[:, None] + g.sd_b.values[:, None] * z[:, :, 1]], axis=2)
    return wm.class_of_samples(smp)

def summarise(df):
    out = []
    for (meth, var, sc), g in df.groupby(["method", "variant", "sc"]):
        g = g[np.isfinite(g[["a", "b", "sd_a", "sd_b"]]).all(axis=1)]
        yt = wm.class_of_theta(np.column_stack([g.a_true, g.b_true])); yp = classify(g)
        eb, ea = g.b - g.b_true, g.a - g.a_true
        out.append(dict(method=meth, variant=var, scenario=SCN[sc], n=len(g), class_acc=float((yt == yp).mean()),
                        cov90_b=float((eb.abs() < 1.645 * g.sd_b).mean()), sd_b=float(g.sd_b.mean()), bias_b=float(eb.mean()),
                        rmse_b=float(np.sqrt((eb ** 2).mean())), emp_sd_b=float(g.b.std()),
                        cov90_a=float((ea.abs() < 1.645 * g.sd_a).mean()), sd_a=float(g.sd_a.mean()), bias_a=float(ea.mean()),
                        rmse_a=float(np.sqrt((ea ** 2).mean())), ms_per_window=float(g.ms.mean())))
    return pd.DataFrame(out)

t1 = summarise(allres[(allres.data == "D1") & (allres.variant.isin(["aug", "known"]))])
sbi = pd.read_csv(R / "11_N3_D1_three_numbers.csv")
sbi_rows = pd.DataFrame(dict(method="SBI (production)", variant="measured inlet", scenario=sbi.scenario, n=sbi.n, class_acc=sbi.class_acc,
                             cov90_b=sbi.cov90_b, sd_b=sbi.sd_b, bias_b=sbi.bias_b, sd_a=sbi.sd_a))
cmp = pd.concat([t1, sbi_rows], ignore_index=True)
cmp["scenario"] = pd.Categorical(cmp.scenario, list(SCN.values()))
cols = ["method", "variant", "scenario", "n", "class_acc", "cov90_b", "sd_b", "bias_b", "rmse_b", "emp_sd_b", "sd_a", "bias_a", "rmse_a", "ms_per_window"]
cmp = cmp.sort_values(["scenario", "method", "variant"])[cols]
display(cmp.round(3).reset_index(drop=True))
cmp.to_csv(R / "16_N3_D1_comparison.csv", index=False)
"""),
    ("md", "## 4. Naive filters: valid on D0, sensitivity on D1 (nominal inlet assumed)"),
    ("code", """tn = pd.concat([summarise(allres[(allres.data == "D0")]).assign(data="D0 (nominal inlet, naive valid)"),
                summarise(allres[(allres.data == "D1") & (allres.variant == "naive")]).assign(data="D1 (inlet varies, naive assumes nominal)")], ignore_index=True)
tn = tn[["data", "method", "scenario", "n", "class_acc", "cov90_b", "sd_b", "bias_b", "rmse_b", "emp_sd_b", "bias_a", "rmse_a", "ms_per_window"]]
display(tn.round(3).sort_values(["data", "scenario", "method"]).reset_index(drop=True))
tn.to_csv(R / "16_N3_naive_D0_vs_D1.csv", index=False)
"""),
    ("md", "## 5. Saturation split (D1; saturated = more than 50% of the window's samples at Qc ≥ 395 L/min)"),
    ("code", """g = allres[(allres.data == "D1") & (allres.variant.isin(["aug", "known"]))].copy()
g["group"] = np.where(g.sat > 0.5, "saturated", "unsaturated")
rows = []
for (meth, var, grp), gg in g.groupby(["method", "variant", "group"]):
    gg = gg[gg.sc != 7]
    eb = gg.b - gg.b_true
    rows.append(dict(method=meth, variant=var, group=grp, n=len(gg), bias_b=float(eb.mean()), rmse_b=float(np.sqrt((eb ** 2).mean())),
                     cov90_b=float((eb.abs() < 1.645 * gg.sd_b).mean()), sd_b=float(gg.sd_b.mean())))
sat = pd.DataFrame(rows); display(sat.round(3)); sat.to_csv(R / "16_N3_saturation_split.csv", index=False)
print("(Sc7, the +2 K sensor drift, is excluded from the saturation split.)")
"""),
    ("md", "## 6. Figure: β̂ by scenario and filter variant"),
    ("code", """FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)
combo = [("D1", "aug", "ekf"), ("D1", "aug", "ukf"), ("D1", "known", "ekf"), ("D1", "known", "ukf"), ("D1", "naive", "ekf"), ("D1", "naive", "ukf")]
cols_ = ["#1f77b4", "#4c9be8", "#2ca02c", "#7fd37f", "#d62728", "#f08080"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
for ax, (sc, nm) in zip(axes.flat, SCN.items()):
    for k, (key, c) in enumerate(zip(combo, cols_)):
        v = res[key]; v = v[v.sc == sc]
        bp = ax.boxplot(v.b.dropna().values, positions=[k], widths=0.6, patch_artist=True, flierprops=dict(ms=3, alpha=0.4))
        bp["boxes"][0].set_facecolor(c); bp["medians"][0].set_color("black")
    ax.axhline(float(res[combo[0]][res[combo[0]].sc == sc].b_true.iloc[0]), color="green", ls="--", lw=1.3)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{k[2].upper()}\\n{k[1]}" for k in combo], fontsize=7); ax.set_title(nm)
axes[0, 0].set_ylabel("β̂"); axes[1, 0].set_ylabel("β̂")
fig.suptitle("D1: β̂ per window (green = truth) — aug / known (measured inlet) / naive (nominal inlet assumed)")
fig.tight_layout(); fig.savefig(FIGS / "16_N3_beta_by_variant.png", dpi=150); plt.show()
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 7. Timing

Reported as ms per window in the tables above (single process, NumPy; the production posterior is amortised: the cost is the 3-D
network forward pass and 2,000 draws per window, measured in notebook 12).

## 8. Per-window estimates

Stored in `results/16_N3_window_estimates.csv` for the later notebooks."""),
    ("code", """allres.to_csv(R / "16_N3_window_estimates.csv", index=False)
print("saved", (R / "16_N3_window_estimates.csv").relative_to(ROOT), len(allres), "rows")
"""),
    ("md", """## 8b. c3 — fair tuning of the augmented filters on a separate tuning set"""),
    ("code", """import concurrent.futures as cf
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
dT = P3.generate_d1(seed_base=4_100_000, n_rep=50)
mT = dT["scenario_id"] != 7
assert mT.sum() == 250 and not set(dT["seeds"].tolist()) & set(d1["seeds"].tolist())
GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
def grid_ll(method):
    tasks = [(dT["x"][i], method, float(dT["inlet_measured"][i, 0]), float(dT["inlet_measured"][i, 1]), True, qs, rs)
             for qs in GRID for rs in GRID for i in np.where(mT)[0]]
    with cf.ProcessPoolExecutor(max_workers=9) as ex:
        out = np.array(list(ex.map(F.loglik_window, tasks, chunksize=25)))
    return out.reshape(len(GRID), len(GRID), -1)
tun = {}
for method in ("ekf", "ukf"):
    t = time.time(); L = grid_ll(method)
    fin = np.isfinite(L)
    mean_ll = np.where(fin.all(axis=2), L.mean(axis=2), np.nan)       # a setting with any failed window is not eligible
    tun[method] = mean_ll
    qi, ri = np.unravel_index(np.nanargmax(mean_ll), mean_ll.shape)
    print(f"{method.upper()}: {time.time() - t:.0f}s | failed windows in the grid: {int((~fin).sum())} | best q_scale {GRID[qi]}, r_scale {GRID[ri]}, mean log-lik per window {mean_ll[qi, ri]:.1f}"
          + (" | BOUNDARY OPTIMUM" if qi in (0, len(GRID) - 1) or ri in (0, len(GRID) - 1) else " | interior"))
    display(pd.DataFrame(mean_ll, index=[f"q x{g}" for g in GRID], columns=[f"r x{g}" for g in GRID]).round(1))
    pd.DataFrame(mean_ll, index=GRID, columns=GRID).to_csv(R / f"16_N3_tuning_grid_{method}.csv")
BEST = {m: (GRID[np.unravel_index(np.nanargmax(v), v.shape)[0]], GRID[np.unravel_index(np.nanargmax(v), v.shape)[1]]) for m, v in tun.items()}
print("tuned (q_scale, r_scale):", BEST)
"""),
    ("md", "### Tuned versus untuned on the D1 test windows (augmented filters)"),
    ("code", """def run_tuned(method, qs, rs):
    rows = []
    for i in range(len(d1["x"])):
        ti, tci = (float(v) for v in d1["inlet_measured"][i])
        o = F.estimate_window(d1["x"][i], method, ti, tci, F.CI0, augment=True, q_scale=qs, r_scale=rs)
        rows.append(dict(win=i, sc=int(d1["scenario_id"][i]), a_true=float(d1["theta"][i, 2]), b_true=float(d1["theta"][i, 3]), a=o["alpha"], b=o["beta"],
                         sd_a=o["sd_alpha"], sd_b=o["sd_beta"], ms=1000 * o["wall_s"], dTci=o["dTci"], sat=float((d1["x"][i][:, 3] >= 395).mean())))
    out = pd.DataFrame(rows); out["method"] = method.upper(); out["variant"] = "aug_tuned"; return out
tuned = pd.concat([run_tuned(m, *BEST[m]) for m in ("ekf", "ukf")], ignore_index=True)
assert np.isfinite(tuned[["a", "b", "sd_a", "sd_b"]]).all().all()
base = allres[(allres.data == "D1") & (allres.variant == "aug")].copy()
def tail_stats(df):
    out = []
    for (m, sc), g in df.groupby(["method", "sc"]):
        eb = g.b - g.b_true
        out.append(dict(method=m, scenario=SCN[sc], n=len(g), cov90_b=float((eb.abs() < 1.645 * g.sd_b).mean()), rmse_b=float(np.sqrt((eb ** 2).mean())),
                        bias_b=float(eb.mean()), sd_b=float(g.sd_b.mean()), n_tail=int((eb.abs() > 0.15).sum())))
    return pd.DataFrame(out)
cmp3 = tail_stats(base).merge(tail_stats(tuned), on=["method", "scenario", "n"], suffixes=("_untuned", "_tuned"))
cmp3 = cmp3.sort_values(["method", "scenario"]); display(cmp3.round(3))
pool = []
for lab, df in (("untuned", base), ("tuned", tuned)):
    for m, g in df[df.sc != 7].groupby("method"):
        eb = g.b - g.b_true; ea = g.a - g.a_true
        pool.append(dict(variant=lab, method=m, n=len(g), cov90_b=float((eb.abs() < 1.645 * g.sd_b).mean()), rmse_b=float(np.sqrt((eb ** 2).mean())), n_tail_b=int((eb.abs() > 0.15).sum()),
                         cov90_a=float((ea.abs() < 1.645 * g.sd_a).mean()), rmse_a=float(np.sqrt((ea ** 2).mean())), median_abs_err_b=float(eb.abs().median()), mean_sd_b=float(g.sd_b.mean())))
pooled = pd.DataFrame(pool); display(pooled.round(3))
cmp3.to_csv(R / "16_N3_tuned_vs_untuned.csv", index=False); pooled.to_csv(R / "16_N3_tuned_vs_untuned_pooled_Sc1_5.csv", index=False)
tuned.to_csv(R / "16_N3_window_estimates_tuned.csv", index=False)
"""),
    ("md", "### Sensitivity (user add-on): one step beyond the UKF q boundary (q × 0.125). Reported only; the pre-set tuned result stays the headline"),
    ("code", """tasks = [(dT["x"][i], "ukf", float(dT["inlet_measured"][i, 0]), float(dT["inlet_measured"][i, 1]), True, 0.125, rs) for rs in GRID for i in np.where(mT)[0]]
with cf.ProcessPoolExecutor(max_workers=9) as ex:
    L125 = np.array(list(ex.map(F.loglik_window, tasks, chunksize=25))).reshape(len(GRID), -1)
ll125 = np.where(np.isfinite(L125).all(axis=1), L125.mean(axis=1), np.nan)
pre = float(tun["ukf"][np.unravel_index(np.nanargmax(tun["ukf"]), tun["ukf"].shape)])
best_r = GRID[int(np.nanargmax(ll125))]
print("UKF q x0.125, mean log-lik per window by r scale", dict(zip(GRID, np.round(ll125, 1))), "| pre-set optimum (q x0.25, r x2):", round(pre, 1))
print("best at q x0.125: r x%s, %.1f -> %s than the pre-set optimum" % (best_r, np.nanmax(ll125), "HIGHER (the optimum was not reached at the boundary)" if np.nanmax(ll125) > pre else "not higher (the boundary optimum stands)"))
x125 = run_tuned("ukf", 0.125, best_r); x125["variant"] = "aug_q0.125"
g_ = x125[x125.sc != 7]; eb = g_.b - g_.b_true; ea = g_.a - g_.a_true
sens = pd.DataFrame([dict(setting="UKF pre-set tuned (q x0.25, r x2)", **pooled[(pooled.variant == "tuned") & (pooled.method == "UKF")].iloc[0][["cov90_b", "rmse_b", "n_tail_b", "cov90_a", "mean_sd_b"]].to_dict()),
                     dict(setting=f"UKF q x0.125, r x{best_r}", cov90_b=float((eb.abs() < 1.645 * g_.sd_b).mean()), rmse_b=float(np.sqrt((eb ** 2).mean())), n_tail_b=int((eb.abs() > 0.15).sum()),
                          cov90_a=float((ea.abs() < 1.645 * g_.sd_a).mean()), mean_sd_b=float(g_.sd_b.mean()))])
display(sens.round(3)); sens.to_csv(R / "16_N3_ukf_q0125_sensitivity.csv", index=False)
pd.DataFrame(dict(r_scale=GRID, mean_loglik_q0125=ll125)).to_csv(R / "16_N3_tuning_grid_ukf_q0125.csv", index=False)
"""),
    ("md", "### The untuned tail windows: initialisation, divergence or observability?"),
    ("code", """tails = base[(base.sc != 7) & ((base.b - base.b_true).abs() > 0.15)].copy()
rows = []
for _, w in tails.iterrows():
    i = int(w.win); ti, tci = (float(v) for v in d1["inlet_measured"][i]); aug = True
    plant = F.Plant(ti, tci, F.CI0, aug); Q, R_, P0 = F.default_noise(aug)
    x0 = F.make_x0(d1["x"][i][0], aug)
    run_ = F.run_ekf if w.method == "EKF" else F.run_ukf
    xh, Ph = run_(plant, np.asarray(d1["x"][i], float), x0, P0, Q, R_)
    I0_raw = F.TAUI * ((d1["x"][i][0, 3] - F.QC0) / F.KP - (d1["x"][i][0, 1] - F.TSP))
    eps_ti, eps_tci = (d1["inlet_measured"][i] - d1["inlet_true"][i][1:3])
    rows.append(dict(method=w.method, window=i, scenario=SCN[int(w.sc)], beta_true=w.b_true, beta_hat=w.b, sd_b=w.sd_b,
                     beta_at_5min=xh[9, 5], beta_at_15min=xh[29, 5], beta_at_30min=xh[59, 5], sd_b_at_5min=float(np.sqrt(Ph[9, 5, 5])), sd_b_end=float(np.sqrt(Ph[-1, 5, 5])),
                     at_bound=bool(w.b < 0.31 or w.b > 1.29), I0_raw=float(I0_raw), I0_clipped=bool(abs(I0_raw) > 100),
                     dTci_hat=w.dTci, err_Tci_meas_minus_true=float(eps_tci), err_Ti_meas_minus_true=float(eps_ti), true_Ti=float(d1["inlet_true"][i][1]), true_Tci=float(d1["inlet_true"][i][2]),
                     frac_sat=w.sat, in_tuned_tail=bool(((tuned.method == w.method) & (tuned.win == i) & ((tuned.b - tuned.b_true).abs() > 0.15)).any())))
tl = pd.DataFrame(rows).sort_values(["method", "window"]); display(tl.round(3)); tl.to_csv(R / "16_N3_tail_windows.csv", index=False)
typ = base[(base.sc != 7) & ((base.b - base.b_true).abs() <= 0.15)]
eps_all = d1["inlet_measured"] - d1["inlet_true"][:, 1:3]
print("typical |measured - true| Tci / Ti over all D1 windows (K): median %.3f / %.3f ; 90th pct %.3f / %.3f" % (np.median(np.abs(eps_all[:, 1])), np.median(np.abs(eps_all[:, 0])), np.percentile(np.abs(eps_all[:, 1]), 90), np.percentile(np.abs(eps_all[:, 0]), 90)))
print("true Ti, Tci of the tail windows vs all: mean Ti %.2f vs %.2f, mean Tci %.2f vs %.2f" % (tl.true_Ti.mean(), d1["inlet_true"][:, 1].mean(), tl.true_Tci.mean(), d1["inlet_true"][:, 2].mean()))
print("tail windows with the same window in the other filter's tail:", int(tl.window.duplicated().sum()), "of", len(tl))
"""),
    ("md", """## 9. Interpretation — to be written after the run
Toward the storyline: what an industrial state estimator achieves with the same model, sensors and inlet information as the production
posterior; where it is overconfident (reported sd vs empirical sd, coverage); what the offset states buy; the naive filter as a further
inlet silent failure; the Sc7 drift for the filters; the cost comparison. State which comparisons are evidence for the "reliable" half."""),
    ("md", """## 10. Interpretation of the tuning and the tail windows (c3) — to be written after the run
Toward the storyline: whether a fair tuning closes the calibration gap to SBI (coverage, RMSE, tail count), whether the optimum is interior, what the tail windows are (initialisation, divergence, observability), what this means
for the comparison with the amortised posterior."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        for head in ("## 9. Interpretation", "## 10. Interpretation"):
            kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith(head)
                    and "to be written after the run" not in c.source.splitlines()[0]]
            if kept:
                for c in nb.cells:
                    if c.cell_type == "markdown" and c.source.startswith(head):
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
