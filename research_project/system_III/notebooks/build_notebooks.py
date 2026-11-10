"""Write the System III notebooks (not executed). Run from anywhere:  python system_III/notebooks/build_notebooks.py
Each notebook runs from system_III/notebooks (the setup cell asserts it). Rules marked PRE-REGISTER must be
fixed (and dated) in the notebook header before the notebook is executed."""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent

SETUP = '''import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "notebooks" and NB_DIR.parent.name == "system_III", f"run from system_III/notebooks (cwd {NB_DIR})"
ROOT = NB_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
import importlib, system_iii
from system_iii import physics as ph, simulator as sim, features as fe, data as da, scenarios as scn
from system_iii import sensors as sen, information as info, config as cfg, rules, metrics as wm
for _m in (ph, sim, fe, da, scn, sen, info, cfg, rules, wm):
    importlib.reload(_m)
R, FIGS, DATA = ROOT / "results", ROOT / "figures", ROOT / "data"
for _d in (R, FIGS, DATA):
    _d.mkdir(exist_ok=True)
P = ph.PLANT
T0 = time.time()'''

INTERP = "## Interpretation — to be written after the run\nWrite toward the storyline (`docs/system_III_plan.md` §1) and update the claims ledger (§2)."


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


NOTEBOOKS = {}

# ------------------------------------------------------------------------------------------ 00
NOTEBOOKS["00_design_scorecard"] = [
    md("""# 00 — Design scorecard (claim D1)
Reproduces the diagnosability scorecard fixed before any candidate was computed (`docs/SCORECARD.md`) and
checks that the System III package reproduces the design toolkit (`design/`). numpy only, ~5 min."""),
    code(SETUP),
    md("## 1. Steady states: uniqueness, safety, cooling margin, conversion"),
    code('''import itertools
grid = np.linspace(0.4, 1.2, 17)
corners = [dict()] + [dict(Ti=P.Ti + a, Tci=P.Tci + b, Ci=P.Ci * (1 + c), Q=P.Q * (1 + q))
                      for a, b, c, q in itertools.product([-2, 2], [-2, 2], [-0.02, 0.02], [-0.01, 0.01])]
n = mult = sat = 0; Tmax = 0
for inl in corners:
    for a in grid:
        for b in grid:
            sols = ph.ss_closed(P, a, b, **inl); n += 1
            mult += len(sols) != 1; sat += sols[0]["sat"] == 1; Tmax = max(Tmax, max(s["T"] for s in sols))
s = ph.ss_unsat(P, 1.0, 1.0)
c1 = dict(n_points=n, n_multiple=mult, frac_saturated=sat / n, Tmax=Tmax, X_nominal=float(s["X"]),
          jacket_share=float(s["qj"] / s["qgen"]), coolant_rise=float(s["Tc"] - P.Tci),
          qc_fraction=float(s["Qc"] / P.Qcmax), beta_sat=ph.beta_saturation(P),
          beta_sat_worst=max(ph.beta_saturation(P, **c) for c in corners[1:]))
display(pd.Series(c1).round(4))'''),
    md("## 2. Design toolkit results (stored) and package verification"),
    code('''for f in ["ss_checks.json", "ctrl_checks.json", "c6_fim.json", "c6_fim_extended.json", "c8_R3.json", "verify_package.json"]:
    p = ROOT / "design" / f
    print(f, "found" if p.exists() else "MISSING")
ver = json.loads((ROOT / "design" / "verify_package.json").read_text())
display(pd.DataFrame({k: v for k, v in ver.items() if k.startswith("crb")}).T)
display(pd.DataFrame(ver["proxy_rule"]).T.round(4))'''),
    code('''crb = info.bayes_crb()           # re-run with the package (~2 min): all calibration offsets marginalised
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in crb.items()})
da.save_json(R / "00_scorecard.json", dict(steady_state=c1, crb_all_offsets=crb))'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 01
NOTEBOOKS["01_model_demonstration"] = [
    md("""# 01 — Model demonstration and controller masking (claim C-M, motivation)
Trajectories for every scenario; the masking picture (T pinned at setpoint, the fault moves into Qc and Tc);
open-loop multistability (why this reactor needs control); control-sensor offset vs monitoring drift."""),
    code(SETUP),
    code('''names = scn.CLOSED + scn.OPEN
out = {}
for i, k in enumerate(names):
    s = scn.SCENARIOS[k]
    out[k] = sim.simulate(np.full(20, s.alpha), np.full(20, s.beta), seed=100 + i, control=s.control,
                          drift_T_monitor=s.drift_T_monitor, oT=s.oT)
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
for k in ["Sc1_healthy", "Sc2_fouling", "Sc3_decay", "Sc4_combined", "Sc5_severe_fouling"]:
    o = out[k]
    for j, lab in enumerate(["C [mol/L]", "T [K]", "Tc [K]", "Qc [L/min]"]):
        axes[j].plot(o["t"], o["obs"][0, :, j], lw=0.8, label=k); axes[j].set_ylabel(lab)
axes[0].legend(fontsize=7, ncol=3); axes[-1].set_xlabel("t [min] (window)")
fig.savefig(FIGS / "01_trajectories.png", dpi=150); plt.show()
summ = pd.DataFrame({k: {"T_true": o["true"][:, :, 1].mean(), "T_logged": o["obs"][:, :, 1].mean(),
                         "C": o["obs"][:, :, 0].mean(), "Tc": o["obs"][:, :, 2].mean(), "Qc": o["obs"][:, :, 3].mean(),
                         "Qc_sat_frac": (o["qc_true"] >= P.Qcmax - 1e-6).mean()} for k, o in out.items()}).T
display(summ.round(3)); summ.to_csv(R / "01_scenario_means.csv")'''),
    md("## Masking: what a temperature-only monitor sees"),
    code('''fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for k in ["Sc1_healthy", "Sc2_fouling", "Sc3_decay", "Sc4_combined"]:
    axes[0].hist(out[k]["obs"][:, :, 1].ravel(), bins=60, histtype="step", density=True, label=k)
    axes[1].hist(out[k]["obs"][:, :, 3].ravel(), bins=60, histtype="step", density=True, label=k)
axes[0].set_xlabel("logged T [K]"); axes[1].set_xlabel("logged Qc [L/min]"); axes[0].legend(fontsize=8)
axes[0].set_title("T: the controller hides the faults"); axes[1].set_title("Qc: where the fault information goes")
fig.savefig(FIGS / "01_masking.png", dpi=150); plt.show()'''),
    md("## Open-loop multistability"),
    code('''Ts = np.linspace(295, 360, 600)
fig, ax = plt.subplots(figsize=(7, 4))
for a, b in [(1, 1), (1, 0.7), (0.75, 0.75), (0.7, 1)]:
    ax.plot(Ts, ph._f_open(P, Ts, a, b, P.Qc0, P.Ci, P.Ti, P.Tci, P.Q) / 1e6, label=f"alpha={a}, beta={b}")
    print((a, b), "open-loop steady states at nominal Qc:", [round(s["T"], 1) for s in ph.ss_open_all(P, a, b, P.Qc0)])
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("T [K]"); ax.set_ylabel("net heat [MJ/min]"); ax.legend(fontsize=8)
fig.savefig(FIGS / "01_open_loop_multiplicity.png", dpi=150); plt.show()'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 02
NOTEBOOKS["02_data_generation"] = [
    md("""# 02 — Observation sets
* **D_prod**: every scenario × 50 windows under production conditions (inlet ±2 K measured with 0.1 K
  offsets, Ci 1%, Q 0.5%, every sensor calibration offset drawn), raw windows stored.
* **D_nom**: the same under nominal conditions (nominal inlet, no calibration offsets).
* **L_prod**: 1000 windows per closed-loop scenario under production conditions (headline accuracies).
Seeds fixed here; no posterior involved. ~5 min."""),
    code(SETUP),
    code('''names = scn.CLOSED + scn.OPEN
dprod = da.scenario_set(names, 50, seed=2000, conditions="production", return_raw=True)
dnom = da.scenario_set(names, 50, seed=2000, conditions="nominal", return_raw=True)
da.save_npz(DATA / "D_prod.npz", dprod); da.save_npz(DATA / "D_nom.npz", dnom)
lprod = da.scenario_set(scn.CLOSED, 1000, seed=2500, conditions="production", return_raw=False)
da.save_npz(DATA / "L_prod.npz", lprod)
print({k: v.shape for k, v in dprod.items()}, lprod["x"].shape, f"{(time.time() - T0) / 60:.1f} min")'''),
    code('''ia, ib = fe.NAMES["S1m"].index("ln_alpha_hat"), fe.NAMES["S1m"].index("ln_beta_hat_jac")
tab = pd.DataFrame({k: {"alpha_hat": np.exp(lprod["x"][lprod["sc_name"] == k, ia]).mean(),
                        "beta_hat_jac": np.exp(lprod["x"][lprod["sc_name"] == k, ib]).mean()} for k in scn.CLOSED}).T
display(tab.round(4)); tab.to_csv(R / "02_proxy_means_L.csv")'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 03
NOTEBOOKS["03_physics_proxies_and_summaries"] = [
    md("""# 03 — Physics proxies and summary statistics
1. Do the two physics proxies (ln α̂, ln β̂_jac) separate each fault category under realistic variation?
   Cross-sensitivity; the legacy UA_eff_proxy confound; robustness of the summaries.
2. **Summary-set selection** (S0 = 29 System I summaries; S1m = production; S1m_no_legacy).

**PRE-REGISTER (fill and date before running):** System I nb03 rule (`rules.summary_rule`), default S1m,
3 seeds × 20k from one bank, efficiency = mean posterior sd / Bayesian CRB of the same set at Sc1–Sc4."""),
    code(SETUP),
    md("## 1. Proxy separation (threshold rule) and cross-sensitivity"),
    code('''L = da.load_npz(DATA / "L_prod.npz")
ia, ib = fe.NAMES["S1m"].index("ln_alpha_hat"), fe.NAMES["S1m"].index("ln_beta_hat_jac")
rows = []
for k in scn.CLOSED:
    m = L["sc_name"] == k; s = scn.SCENARIOS[k]
    ah, bh = np.exp(L["x"][m, ia]), np.exp(L["x"][m, ib])
    ok = ((ah < scn.THRESHOLD) == (s.alpha < scn.THRESHOLD)) & ((bh < scn.THRESHOLD) == (s.beta < scn.THRESHOLD))
    rows.append(dict(scenario=k, alpha=s.alpha, beta=s.beta, proxy_rule_acc=ok.mean(), alpha_hat=ah.mean(),
                     alpha_hat_sd=ah.std(), beta_hat=bh.mean(), beta_hat_sd=bh.std()))
prox = pd.DataFrame(rows).set_index("scenario"); display(prox.round(4)); prox.to_csv(R / "03_proxy_rule.csv")
A = [0.6, 0.8, 1.0, 1.15]; M = {}
iu = fe.NAMES["S1m"].index("UA_eff_proxy")
for a in A:
    for b in A:
        d = da.make_windows(np.tile([a, b], (300, 1)), 3100 + int(100 * a + b * 10), conditions="production")
        M[(a, b)] = (np.exp(d["x"][:, ia]).mean(), np.exp(d["x"][:, ib]).mean(), d["x"][:, iu].mean())
cross = dict(
    d_alphahat_d_beta=[(M[(a, A[j + 1])][0] - M[(a, A[j])][0]) / (A[j + 1] - A[j]) for a in A for j in range(3)],
    d_betahat_d_alpha=[(M[(A[i + 1], b)][1] - M[(A[i], b)][1]) / (A[i + 1] - A[i]) for b in A for i in range(3)],
    UAproxy_alpha_sens=[(M[(A[i + 1], b)][2] - M[(A[i], b)][2]) / (A[i + 1] - A[i]) for b in A for i in range(3)],
    UAproxy_beta_sens=[(M[(a, A[j + 1])][2] - M[(a, A[j])][2]) / (A[j + 1] - A[j]) for a in A for j in range(3)])
display(pd.DataFrame({k: [np.min(v), np.max(v)] for k, v in cross.items()}, index=["min", "max"]).round(4))
da.save_json(R / "03_cross_sensitivity.json", {k: [float(x) for x in v] for k, v in cross.items()})'''),
    md("## 2. Summary robustness: warm start, constant features, clipping"),
    code('''x2 = da.make_windows(np.tile([1.0, 1.0], (800, 1)), 3300, protocol="P2")["x"]
x1 = da.make_windows(np.tile([1.0, 1.0], (800, 1)), 3300, protocol="P1b")["x"]
sd = x2.std(0); shift = np.where(sd > 0, (x1.mean(0) - x2.mean(0)) / np.where(sd > 0, sd, 1), 0)
rob = pd.DataFrame({"feature": fe.NAMES["S1m"], "shift_P1b_vs_P2_sd": shift, "sd": sd}).set_index("feature")
display(rob.sort_values("shift_P1b_vs_P2_sd", key=np.abs, ascending=False).head(8).round(3))
print("constant features at nominal:", list(rob.index[rob.sd < 1e-12]))
rob.to_csv(R / "03_summary_robustness.csv")'''),
    md("## 3. Summary-set selection (training)"),
    code('''from system_iii import inference as inf
SETS = {"S0": fe.NAMES["S0"], "S1m": fe.NAMES["S1m"], "S1m_no_legacy": fe.NAMES["S1m_no_legacy"]}
col = {k: [fe.NAMES["S1m"].index(n) for n in v] for k, v in SETS.items()}
bank_f = DATA / "03_bank_S1m_60k.npz"
if not bank_f.exists():
    da.save_npz(bank_f, da.make_windows(sim.prior_draws(60_000, 3400), 3400, conditions="production"))
bank = da.load_npz(bank_f)
test = da.test_set(3499, conditions="production")
rows = []
for sname, idx in col.items():
    for s in range(3):
        f = R / "03" / f"post_{sname}_s{s}.pkl"; f.parent.mkdir(exist_ok=True)
        if f.exists():
            post = __import__("pickle").load(open(f, "rb"))
        else:
            sl = slice(s * 20_000, (s + 1) * 20_000)
            post, meta = inf.train_npe(bank["theta"][sl], bank["x"][sl][:, idx], seed=s)
            __import__("pickle").dump(post, open(f, "wb"))
        sub = dict(test, x_sbc=test["x_sbc"][:, idx], x_sc=test["x_sc"][:, idx])
        row, _ = inf.evaluate(post, sub, seed=s)
        rows.append({"set": sname, "seed": s, "n_features": len(idx), **row})
met = pd.DataFrame(rows); met.to_csv(R / "03_summary_sets_metrics.csv", index=False)'''),
    code('''crb = {s: {pt: info.bayes_crb(scn.SCENARIOS[n].alpha, scn.SCENARIOS[n].beta, feature_set=s, n=1000)
           for pt, n in zip(["Sc1", "Sc2", "Sc3", "Sc4"], scn.GATED)} for s in SETS}
def eff_row(r):
    return np.mean([r[f"sd{p}_{pt}"] / crb[r["set"]][pt]["sd_" + ("alpha" if p == "a" else "beta")]
                    for pt in ["Sc1", "Sc2", "Sc3", "Sc4"] for p in ("a", "b")])
met["eff"] = met.apply(eff_row, axis=1)
se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
agg = met.groupby("set").agg(n_features=("n_features", "first"), ks_a=("ks_p_a_interior", "median"), ks_b=("ks_p_b_interior", "median"),
                             tarp=("tarp_ks_p", "median"), rmse=("rmse_norm", "mean"), rmse_se=("rmse_norm", se),
                             eff=("eff", "mean"), eff_se=("eff", se))
agg["calib_ok"] = (agg.ks_a >= 0.01) & (agg.ks_b >= 0.01) & (agg.tarp >= 0.01)
dec = rules.summary_rule(agg, default="S1m"); display(agg.round(4)); print(dec)
da.save_json(R / "03_summary_decision.json", dict(decision=dec, table=agg.reset_index().to_dict("records")))'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 04
NOTEBOOKS["04_information_limits"] = [
    md("""# 04 — Information limits (claims C3, C-M, I1)
Bayesian Cramér–Rao bounds of the production features at Sc1–Sc5 under three nuisance sets (inlet
known; inlet offsets + Ci + Q; plus every sensor calibration offset); which instrument limits β; what a
temperature-only monitor can know (masking); the sensor ladder S0–S4. numpy only, ~30 min."""),
    code(SETUP),
    code('''pts = {k: (scn.SCENARIOS[k].alpha, scn.SCENARIOS[k].beta) for k in scn.GATED + ["Sc5_severe_fouling"]}
NS = {"inlet_only": ("oTi", "oTci", "uCi", "uQ"), "all_offsets": tuple(info.NUISANCES)}
rows = []
for k, (a, b) in pts.items():
    for lab, nu in NS.items():
        r = info.bayes_crb(a, b, nuisances=nu, n=1500)
        rows.append({"scenario": k, "nuisances": lab, **{kk: v for kk, v in r.items() if isinstance(v, float)}})
crb = pd.DataFrame(rows); display(crb.round(4)); crb.to_csv(R / "04_crb.csv", index=False)'''),
    md("## Which instrument limits β? (each nuisance made known in turn, at Sc1)"),
    code('''base = info.bayes_crb(1, 1, n=1500)
brk = {"all unknown": (base["sd_alpha"], base["sd_beta"])}
for k in info.NUISANCES:
    r = info.bayes_crb(1, 1, nuisances=tuple(x for x in info.NUISANCES if x != k), n=1500)
    brk[f"{k} known"] = (r["sd_alpha"], r["sd_beta"])
brk = pd.DataFrame(brk, index=["sd_alpha", "sd_beta"]).T; display(brk.round(4)); brk.to_csv(R / "04_instrument_breakdown.csv")'''),
    md("## Masking (C-M): information in a temperature-only monitor"),
    code('''mask = {fs: info.bayes_crb(1, 1, feature_set=fs, n=1500, nuisances=("oTi", "oTci", "uCi", "uQ")) for fs in ["T_only", "C_T_only", "S1m"]}
display(pd.DataFrame(mask).T[["sd_alpha", "sd_beta"]].round(4)); da.save_json(R / "04_masking.json", mask)'''),
    md("## Sensor ladder (information side of notebook 10)"),
    code('''lad = {s: info.bayes_crb(1, 1, sensors=s, n=1500) for s in sen.SENSORS}
display(pd.DataFrame(lad).T[["sd_alpha", "sd_beta", "corr_ab"]].round(4)); da.save_json(R / "04_sensor_ladder_crb.json", lad)
print(f"{(time.time() - T0) / 60:.1f} min")'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 05
NOTEBOOKS["05_warm_start_protocols"] = [
    md("""# 05 — Warm-start protocols (claim F6)
Train × test matrix over P0, P1, P1b, P2 (3 seeds × 20k each, production conditions). Expected: much weaker
effects than System I (slowest closed-loop time constant ~12 min vs a 300-min history).

**PRE-REGISTER (fill and date before running):** `rules.protocol_rule` on the P2 test column (System I nb29
rule); production stays P2 unless the rule selects otherwise (a user decision then)."""),
    code(SETUP),
    code('''from system_iii import inference as inf
import pickle
PROTS = ["P0", "P1", "P1b", "P2"]
banks, tests = {}, {}
for p in PROTS:
    f = DATA / f"05_bank_{p}.npz"
    if not f.exists():
        da.save_npz(f, da.make_windows(sim.prior_draws(60_000, 5100), 5100, protocol=p))
    banks[p] = da.load_npz(f)
    g = DATA / f"05_test_{p}.npz"
    if not g.exists():
        da.save_npz(g, da.test_set(5199, protocol=p))
    tests[p] = da.load_npz(g)
rows = []
for p in PROTS:
    for s in range(3):
        f = R / "05" / f"post_{p}_s{s}.pkl"; f.parent.mkdir(exist_ok=True)
        if f.exists():
            post = pickle.load(open(f, "rb"))
        else:
            sl = slice(s * 20_000, (s + 1) * 20_000)
            post, _ = inf.train_npe(banks[p]["theta"][sl], banks[p]["x"][sl], seed=s); pickle.dump(post, open(f, "wb"))
        for q in PROTS:
            row, _ = inf.evaluate(post, tests[q], seed=s)
            rows.append({"train": p, "test": q, "seed": s, **row})
mat = pd.DataFrame(rows); mat.to_csv(R / "05_protocol_matrix.csv", index=False)
dec = rules.protocol_rule(mat, test_col="P2"); print(dec["choice"], dec["flag"])
da.save_json(R / "05_protocol_decision.json", dec)
display(mat.pivot_table(index="train", columns="test", values="cov90_b").round(3))'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 06
NOTEBOOKS["06_budget_and_production"] = [
    md("""# 06 — Budget convergence and the production posterior (claim C2)
160k bank (production conditions, protocol and features from `config.py`, or as decided in 03/05),
nested budgets 5k–160k × 5 seeds, evaluation on a fixed test set, **rule v2.1 unchanged** (`rules.budget_rule_v21`).
Writes `results/06_production.json`. ~2–4 h (training dominates)."""),
    code(SETUP),
    code('''from system_iii import inference as inf
import pickle
bank_f = DATA / "06_bank_160k.npz"
if not bank_f.exists():
    da.save_npz(bank_f, da.make_windows(sim.prior_draws(160_000, 6100), 6100, protocol=cfg.PROTOCOL, feature_set=cfg.FEATURE_SET))
bank = da.load_npz(bank_f)
test_f = DATA / "06_test.npz"
if not test_f.exists():
    da.save_npz(test_f, da.test_set(6199, protocol=cfg.PROTOCOL, feature_set=cfg.FEATURE_SET))
test = da.load_npz(test_f)
OUT = R / "06"; OUT.mkdir(exist_ok=True)
f_rows = R / "06_metrics.csv"
rows = pd.read_csv(f_rows).to_dict("records") if f_rows.exists() else []
done = {(int(r["budget"]), str(r["member"])) for r in rows}
for N in cfg.BUDGETS:
    members = []
    for s in cfg.SEEDS:
        f = OUT / f"post_n{N}_s{s}.pkl"
        if f.exists():
            post = pickle.load(open(f, "rb"))
        else:
            idx = np.random.default_rng(36_000 + s).permutation(len(bank["theta"]))[:N]
            post, meta = inf.train_npe(bank["theta"][idx], bank["x"][idx], seed=s); pickle.dump(post, open(f, "wb"))
        members.append(post)
        if (N, str(s)) not in done:
            row, _ = inf.evaluate(post, test, seed=0); rows.append({"budget": N, "member": str(s), **row})
    if (N, "ensemble") not in done:
        row, _ = inf.evaluate(inf.EnsemblePosterior(members), test, seed=0); rows.append({"budget": N, "member": "ensemble", **row})
    pd.DataFrame(rows).to_csv(f_rows, index=False); print(N, "done", f"{(time.time() - T0) / 60:.1f} min", flush=True)'''),
    code('''met = pd.read_csv(f_rows)
dec = rules.budget_rule_v21(met)
dec.update(protocol=cfg.PROTOCOL, feature_set=cfg.FEATURE_SET, sensors=cfg.SENSORS, conditions=cfg.CONDITIONS,
           member_files=[str(OUT / f"post_n{dec['production_budget']}_s{m}.pkl") for m in dec["members"]])
da.save_json(R / "06_production.json", dec)
display(pd.DataFrame(dec["steps"])); print({k: v for k, v in dec.items() if k != "steps"})
if dec["flag"]:
    raise RuntimeError("STOP: " + dec["flag"])'''),
    md(INTERP),
]

LOAD_PROD = '''from system_iii import inference as inf
import pickle
PROD = json.loads((R / "06_production.json").read_text())
_m = [pickle.load(open(f, "rb")) for f in PROD["member_files"]]
post = inf.EnsemblePosterior(_m) if PROD["use_ensemble"] else _m[0]
print("production:", PROD["production_budget"], "ensemble" if PROD["use_ensemble"] else "single", PROD["feature_set"])'''

# ------------------------------------------------------------------------------------------ 07
NOTEBOOKS["07_calibration_and_classification"] = [
    md("""# 07 — Calibration, precision and classification of the production posterior (C2, C3, C4, C5)
On the L-sets (1000 fresh windows per scenario, production conditions) and a fresh SBC set: SBC (interior),
TARP, coverage, bias, posterior sd vs the Bayesian CRB (notebook 04), four-class and three-way classification."""),
    code(SETUP), code(LOAD_PROD),
    code('''L = da.load_npz(DATA / "L_prod.npz")
sbc = da.make_windows(sim.prior_draws(2000, 7100), 7100, feature_set=PROD["feature_set"])
smp_sbc, leak = inf.sample_within_prior(post, sbc["x"], 1000, rng_seed=1)
cal = wm.evaluate_sbc_set(smp_sbc, sbc["theta"], inf.LO, inf.HI, seed=0); cal["leak_mean"] = float(leak.mean())
display(pd.Series(cal).round(4))
smp, leakL = inf.sample_within_prior(post, L["x"], 1000, rng_seed=2)
overall, sc_rows = wm.evaluate_scenario_set(smp, L["theta"], L["sc_name"], inf.LO, inf.HI)
tab = pd.DataFrame(sc_rows).set_index("scenario"); display(tab.round(4)); print(overall)
tw_a, tw_b = inf.three_way(smp)
three = pd.DataFrame({k: {"alpha_fault": (tw_a[L["sc_name"] == k] == 1).mean(), "alpha_undet": (tw_a[L["sc_name"] == k] == 0).mean(),
                          "beta_fault": (tw_b[L["sc_name"] == k] == 1).mean(), "beta_undet": (tw_b[L["sc_name"] == k] == 0).mean()}
                      for k in scn.CLOSED}).T
display(three.round(3))
crb = pd.read_csv(R / "04_crb.csv"); crb = crb[crb.nuisances == "all_offsets"].set_index("scenario")
tab["sd_b_over_crb"] = [tab.loc[k, "sd_b"] / crb.loc[k, "sd_beta"] if k in crb.index else np.nan for k in tab.index]
tab["sd_a_over_crb"] = [tab.loc[k, "sd_a"] / crb.loc[k, "sd_alpha"] if k in crb.index else np.nan for k in tab.index]
tab.to_csv(R / "07_scenarios.csv"); three.to_csv(R / "07_three_way.csv"); da.save_json(R / "07_calibration.json", dict(sbc=cal, overall=overall))'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 08
NOTEBOOKS["08_claim1_controller_modelling"] = [
    md("""# 08 — Claim 1 revised: what must be modelled about the control loop (C1r, C-M)
Arms (each 3 seeds × 40k, evaluated on the same closed-loop production test set):
* **A, controller-aware**: the production simulator (notebook 06 posterior).
* **B, controller-agnostic**: simulator driven by an exogenous coolant flow, no controller model; the
  posterior conditions on the observed Qc through the features.
* **C, open-loop-trained** (the System I form): coolant flow fixed at the nominal value.
* **M, masking**: production simulator, temperature-only features (T_only).

**PRE-REGISTER (user to fix and date before running):**
* B's exogenous flow: level ~ U(10, Qc_max) L/min per window, OU fluctuation 5% / 10 min (draft).
* Metric: per scenario β coverage, |bias|, sd; Claim 1 "controller modelling necessary" holds if B's β
  coverage < 0.80 or its sd > 1.5 × A's on any of Sc1–Sc4."""),
    code(SETUP), code(LOAD_PROD),
    code('''test = da.load_npz(DATA / "06_test.npz")
th = sim.prior_draws(120_000, 8100)
arms = {"B_agnostic": dict(control="exogenous", qc_level=np.random.default_rng(8101).uniform(10, P.Qcmax, 120_000)),
        "C_open_loop": dict(control="fixed", qc_level=np.full(120_000, P.Qc0))}
posts = {"A_aware": post}
for arm, kw in arms.items():
    f = DATA / f"08_bank_{arm}.npz"
    if not f.exists():
        da.save_npz(f, da.make_windows(th, 8100, **kw))
    b = da.load_npz(f); ms = []
    for s in range(3):
        g = R / "08" / f"post_{arm}_s{s}.pkl"; g.parent.mkdir(exist_ok=True)
        if g.exists():
            ms.append(pickle.load(open(g, "rb")))
        else:
            sl = slice(s * 40_000, (s + 1) * 40_000)
            p_, _ = inf.train_npe(b["theta"][sl], b["x"][sl], seed=s); pickle.dump(p_, open(g, "wb")); ms.append(p_)
    posts[arm] = inf.EnsemblePosterior(ms)
# masking arm: production bank, T-only features
bank = da.load_npz(DATA / "06_bank_160k.npz"); it = [fe.NAMES["S1m"].index(n) for n in fe.NAMES["T_only"]]
ms = []
for s in range(3):
    g = R / "08" / f"post_M_Tonly_s{s}.pkl"
    if g.exists():
        ms.append(pickle.load(open(g, "rb")))
    else:
        sl = slice(s * 40_000, (s + 1) * 40_000)
        p_, _ = inf.train_npe(bank["theta"][sl], bank["x"][sl][:, it], seed=s); pickle.dump(p_, open(g, "wb")); ms.append(p_)
posts["M_T_only"] = inf.EnsemblePosterior(ms)
rows = []
for arm, p_ in posts.items():
    sub = test if arm != "M_T_only" else dict(test, x_sbc=test["x_sbc"][:, it], x_sc=test["x_sc"][:, it])
    row, sc_rows = inf.evaluate(p_, sub, seed=0)
    rows += [{"arm": arm, **r} for r in sc_rows]
res = pd.DataFrame(rows); res.to_csv(R / "08_claim1.csv", index=False)
display(res.pivot_table(index="scenario", columns="arm", values=["cov90_truth_b", "bias_b", "sd_b", "class_acc"]).round(3))'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 09
NOTEBOOKS["09_baselines_ekf_ukf"] = [
    md("""# 09 — Baselines: EKF / UKF (claim B1) — **TO BUILD**
Specification (carry over the System I nb16-N3 design and its lessons):
* Augmented-state EKF and UKF on the 4-state model (C, T, Tc, I) with parameters (α, β) as random-walk
  states, the **measured inlet as a known input**, and constant augmented states for the Ti and Tci
  calibration offsets (prior sd 0.1 K) and the coolant-flowmeter bias (1%); also a known-input-only variant.
* Process/measurement noise from the simulator; tuning by a pre-set rule on a separate tuning set (grid of
  scale factors, objective = predictive log-likelihood), evaluated on D_prod.
* Report per scenario: β RMSE, coverage of the reported sd, tail count |error| > 0.15, cost per window.
Code to be written in `src/system_iii/filters.py` (numpy), then this notebook."""),
    code(SETUP),
    code('''raise NotImplementedError("notebook 09: implement src/system_iii/filters.py first (see the specification above)")'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 10
NOTEBOOKS["10_fragility_sensor_model"] = [
    md("""# 10 — Fragility: sensor-model mismatch (claim F3)
S0 (inherited 0.5%-of-value convention) … S4 (no usable concentration measurement); each trained 3 × 40k and
tested on its own sensor model; plus the cross case (trained on S0, tested on S3 data: the mismatch).
Report posterior sd, coverage, Corr(α, β) per scenario (artefact correlations), α information without C.

**PRE-REGISTER:** silent failure = β or α coverage < 0.5 with leakage < 1% and sd within 20% of the matched
posterior's."""),
    code(SETUP),
    code('''from system_iii import inference as inf
import pickle
rows = []; posts = {}
for s_model in sen.SENSORS:
    f = DATA / f"10_bank_{s_model}.npz"
    if not f.exists():
        da.save_npz(f, da.make_windows(sim.prior_draws(120_000, 10100), 10100, sensors=s_model))
    b = da.load_npz(f); ms = []
    for s in range(3):
        g = R / "10" / f"post_{s_model}_s{s}.pkl"; g.parent.mkdir(exist_ok=True)
        if g.exists():
            ms.append(pickle.load(open(g, "rb")))
        else:
            sl = slice(s * 40_000, (s + 1) * 40_000)
            p_, _ = inf.train_npe(b["theta"][sl], b["x"][sl], seed=s); pickle.dump(p_, open(g, "wb")); ms.append(p_)
    posts[s_model] = inf.EnsemblePosterior(ms)
for s_model in sen.SENSORS:
    test = da.test_set(10199, sensors=s_model)
    row, sc = inf.evaluate(posts[s_model], test, seed=0); rows += [{"trained": s_model, "tested": s_model, **r} for r in sc]
test3 = da.test_set(10199, sensors="S3")
row, sc = inf.evaluate(posts["S0"], test3, seed=0); rows += [{"trained": "S0", "tested": "S3", **r} for r in sc]
res = pd.DataFrame(rows); res.to_csv(R / "10_sensor_ladder.csv", index=False)
display(res.pivot_table(index="scenario", columns=["trained", "tested"], values=["sd_a", "sd_b", "cov90_truth_b", "corr_ab_mean"]).round(3))'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 11
NOTEBOOKS["11_fragility_inlet"] = [
    md("""# 11 — Fragility: inlet assumptions and the instrument requirement (F1, I1)
* **Naive model**: trained under nominal conditions with the inlet assumed nominal (System I's original
  assumption). Tested on windows whose true Tci or Ti deviates by 0, ±0.25, ±0.5, ±1, ±2 K.
* **Production model** (notebook 06): the same windows, with the measured inlet.
* **Alarm**: disagreement of the two plug-in β estimates (jacket vs reactor balance) with the nominal inlet
  assumed; threshold fixed on a separate nominal set.
* **ε sweep**: production-type posteriors trained with inlet calibration error ε = 0.05, 0.1, 0.2 K (3 × 40k).

**PRE-REGISTER:** silent failure as in notebook 10; alarm threshold |z| > 3 on the calibration set."""),
    code(SETUP), code(LOAD_PROD),
    code('''f = DATA / "11_bank_naive.npz"
if not f.exists():
    da.save_npz(f, da.make_windows(sim.prior_draws(120_000, 11100), 11100, conditions="nominal", assume_nominal_inlet=True))
b = da.load_npz(f); ms = []
for s in range(3):
    g = R / "11" / f"post_naive_s{s}.pkl"; g.parent.mkdir(exist_ok=True)
    if g.exists():
        ms.append(pickle.load(open(g, "rb")))
    else:
        sl = slice(s * 40_000, (s + 1) * 40_000)
        p_, _ = inf.train_npe(b["theta"][sl], b["x"][sl], seed=s); pickle.dump(p_, open(g, "wb")); ms.append(p_)
naive = inf.EnsemblePosterior(ms)
rows = []
for fac in ("Tci", "Ti"):
    for dv in (-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2):
        kw = {fac: (P.Tci if fac == "Tci" else P.Ti) + dv}
        for model, p_, naive_feat in (("naive", naive, True), ("production", post, False)):
            sc = da.scenario_set(scn.GATED, 100, seed=11200, conditions="nominal", assume_nominal_inlet=naive_feat, **kw)
            smp, leak = inf.sample_within_prior(p_, sc["x"], 1000, rng_seed=0)
            _, r = wm.evaluate_scenario_set(smp, sc["theta"], sc["sc_name"], inf.LO, inf.HI)
            rows += [{"factor": fac, "error_K": dv, "model": model, "leak": float(leak.mean()), **x} for x in r]
res = pd.DataFrame(rows); res.to_csv(R / "11_inlet_errors.csv", index=False)
display(res.pivot_table(index=["factor", "error_K"], columns=["model", "scenario"], values="cov90_truth_b").round(2))'''),
    md("## Alarm: disagreement of the two β estimates (inlet assumed nominal)"),
    code('''jb, rb = fe.NAMES["S1m"].index("ln_beta_hat_jac"), fe.NAMES["S1m"].index("ln_beta_hat_rx")
cal = da.scenario_set(scn.GATED, 200, seed=11300, conditions="nominal", assume_nominal_inlet=True)
dcal = cal["x"][:, jb] - cal["x"][:, rb]; mu, sd = dcal.mean(), dcal.std(ddof=1)
al = []
for fac in ("Tci", "Ti"):
    for dv in (0, 0.25, 0.5, 1, 2):
        kw = {fac: (P.Tci if fac == "Tci" else P.Ti) + dv}
        sc = da.scenario_set(scn.GATED, 100, seed=11400, conditions="nominal", assume_nominal_inlet=True, **kw)
        z = (sc["x"][:, jb] - sc["x"][:, rb] - mu) / sd
        al.append(dict(factor=fac, error_K=dv, detect_rate=float((np.abs(z) > 3).mean())))
display(pd.DataFrame(al)); pd.DataFrame(al).to_csv(R / "11_alarm.csv", index=False)'''),
    md("## ε sweep: what inlet-sensor accuracy buys"),
    code('''eps_rows = []
for eps in (0.05, 0.1, 0.2):
    f = DATA / f"11_bank_eps{eps}.npz"
    if not f.exists():
        da.save_npz(f, da.make_windows(sim.prior_draws(120_000, 11500), 11500, inlet_eps=eps))
    b = da.load_npz(f); ms = []
    for s in range(3):
        g = R / "11" / f"post_eps{eps}_s{s}.pkl"
        if g.exists():
            ms.append(pickle.load(open(g, "rb")))
        else:
            sl = slice(s * 40_000, (s + 1) * 40_000)
            p_, _ = inf.train_npe(b["theta"][sl], b["x"][sl], seed=s); pickle.dump(p_, open(g, "wb")); ms.append(p_)
    test = da.test_set(11599, inlet_eps=eps)
    row, sc = inf.evaluate(inf.EnsemblePosterior(ms), test, seed=0)
    eps_rows += [{"eps": eps, **r} for r in sc]
pd.DataFrame(eps_rows).to_csv(R / "11_eps_sweep.csv", index=False)
display(pd.DataFrame(eps_rows).pivot_table(index="scenario", columns="eps", values=["sd_b", "cov90_truth_b", "class_acc"]).round(3))'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 12
NOTEBOOKS["12_fragility_controller_and_sensors"] = [
    md("""# 12 — Fragility: controller retuned, control-sensor offset, monitoring drift, offsets beyond assumed (F2, F4, F5)
All tests use the production posterior (trained on the nominal controller and the assumed calibration sd).
* **Retuned controller**: Kp × {0.5, 2}, τI × {0.5, 2} in the plant only.
* **Control-sensor offset** oT ∈ {0.1, 0.25, 0.5, 1} K (in the loop: the plant moves, the reading stays at
  setpoint), and **monitoring drift** {0.5, 1, 2} K (outside the loop).
* **Calibration offsets beyond assumed**: every offset drawn at 2× and 3× its assumed sd.
* **Valve saturation** (decision 14; not a catalogue scenario): windows at α = 1.2, β = 0.41, valve at
  Qc_max. This sits at the prior's β edge, so coverage is reported with that caveat (System I nb29 lesson).
* **Residual detector** z_T = (mean logged T − Tsp)/(σ_T/√120): which of these it can see.

**PRE-REGISTER:** silent failure as in notebook 10; detector threshold fixed on a separate nominal set
(saturation guard as in System I)."""),
    code(SETUP), code(LOAD_PROD),
    code('''def assess(label, **kw):
    sc = da.scenario_set(scn.GATED, 100, seed=12100, **kw)
    smp, leak = inf.sample_within_prior(post, sc["x"], 1000, rng_seed=0)
    _, r = wm.evaluate_scenario_set(smp, sc["theta"], sc["sc_name"], inf.LO, inf.HI)
    return [{"case": label, "leak": float(leak.mean()), **x} for x in r]
rows = assess("nominal")
for fk, fv in (("Kp", 0.5), ("Kp", 2.0), ("taui", 0.5), ("taui", 2.0)):
    rows += assess(f"{fk}x{fv}", plant=replace(P, **{fk: getattr(P, fk) * fv}))
for o in (0.1, 0.25, 0.5, 1.0):
    rows += assess(f"control_offset_{o}K", oT=o)
for dr in (0.5, 1.0, 2.0):
    rows += assess(f"monitor_drift_{dr}K", drift_T_monitor=dr)
sat = da.make_windows(np.tile([1.2, 0.41], (100, 1)), 12200)                 # valve saturation (decision 14)
smp, leak = inf.sample_within_prior(post, sat["x"], 1000, rng_seed=0)
_, r = wm.evaluate_scenario_set(smp, sat["theta"], np.array(["saturated_a1.2_b0.41"] * 100), inf.LO, inf.HI)
rows += [{"case": "valve_saturation", "leak": float(leak.mean()), **x} for x in r]
res = pd.DataFrame(rows); res.to_csv(R / "12_controller_and_sensors.csv", index=False)
display(res.pivot_table(index="case", columns="scenario", values="cov90_truth_b").round(2))'''),
    code('''import system_iii.sensors as S
rows = []
for mult in (2.0, 3.0):
    old = dict(S.CAL_SD)
    try:
        for k in S.CAL_SD:
            S.CAL_SD[k] = old[k] * mult
        rows += assess(f"offsets_{mult}x_assumed")
    finally:
        S.CAL_SD.update(old)
pd.DataFrame(rows).to_csv(R / "12_offsets_beyond_assumed.csv", index=False)
display(pd.DataFrame(rows).pivot_table(index="case", columns="scenario", values=["cov90_truth_b", "bias_b"]).round(3))'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 13, 14
NOTEBOOKS["13_individual_posterior_checks"] = [
    md("""# 13 — Individual-posterior validity: L-C2ST (claim V1) — **TO BUILD**
Specification (System I nb37 lessons): standardised-input L-C2ST (sbi.diagnostics), calibration set 20,000
pairs from the production simulator, 100 null trials, 5 windows per scenario Sc1–Sc5 from D_prod; a power
check with deliberately wrong posteriors (shifted by +2 sd, halved width) run through the whole test;
report reproducibility across identical seeds. Fix the settings in this header before running."""),
    code(SETUP),
    code('''raise NotImplementedError("notebook 13: implement following the specification above")'''),
    md(INTERP),
]
NOTEBOOKS["14_tracking_30day"] = [
    md("""# 14 — 30-day tracking (claim T1) — **TO BUILD**
Specification: a continuous 30-day simulation (needs a streaming variant of `simulator.simulate` that carries
the state across windows) with slowly developing fouling and catalyst decay, a drifting coolant-supply
temperature (daily ±1 K plus a seasonal trend), persistent sensor calibration offsets for the month; the
production posterior applied window by window; report coverage, MAE, detection delay at the class thresholds,
and the trend-versus-level message (System I nb10)."""),
    code(SETUP),
    code('''raise NotImplementedError("notebook 14: add a streaming simulator first (see the specification above)")'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 15
NOTEBOOKS["15_cost_and_timing"] = [
    md("# 15 — Cost and timing\nSimulation, features, training (from notebook 06 metadata) and inference per window."),
    code(SETUP), code(LOAD_PROD),
    code('''t = time.time(); d = da.make_windows(sim.prior_draws(10_000, 15100), 15100); t_sim = (time.time() - t) / 10_000
x = d["x"][:720]
t = time.time(); inf.sample_within_prior(post, x[:1], 1000); t_one = time.time() - t
t = time.time(); inf.sample_within_prior(post, x, 1000); t_batch = (time.time() - t) / len(x)
cost = dict(sim_and_features_per_window_s=t_sim, sbi_single_window_s=t_one, sbi_batched_per_window_s=t_batch)
display(pd.Series(cost)); da.save_json(R / "15_cost.json", cost)'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 16
NOTEBOOKS["16_system_I_contrast"] = [
    md("""# 16 — System I contrast (claim X1)
Reads System I's stored results (read-only) next to System III's: the scorecard (design toolkit, both plants),
information limits, production precision and classification, and the inlet fragility."""),
    code(SETUP),
    code('''SYS1 = ROOT.parent / "results"
ss = json.loads((ROOT / "design" / "ss_checks.json").read_text())
c6 = json.loads((ROOT / "design" / "c6_fim_extended.json").read_text())
display(pd.DataFrame({k: {kk: v[kk] for kk in ("n_multiple", "Tmax_all_ss", "beta_sat_nominal", "beta_sat_worst_inlet")}
                      for k, v in ss.items() if k in ("R0_SystemI", "R3_fogler_UA2.0")}))
display(pd.DataFrame({k: v["(1.0, 1.0)"] for k, v in c6.items() if k in ("R0_SystemI", "R3_fogler_UA2.0")}).T[["sd_alpha", "sd_beta", "corr_ab"]].round(4))
for f in ["36_N3_production.json", "36_N3_acceptance.json", "34_N3_A_decision.json"]:
    p = SYS1 / f
    print(f, "->", json.loads(p.read_text()).get("production_budget", "") if p.exists() else "not found")'''),
    md(INTERP),
]

# ------------------------------------------------------------------------------------------ 17, 18
NOTEBOOKS["17_claims_and_conclusions"] = [
    md("# 17 — Claims and conclusions\nCollects every result file into the claims ledger table (`docs/system_III_plan.md` §2)."),
    code(SETUP),
    code('''files = sorted(p.name for p in R.glob("*.json")) + sorted(p.name for p in R.glob("*.csv"))
print("\\n".join(files))
ledger = {"D1": "00_scorecard.json", "C2": "07_calibration.json", "C3": "07_scenarios.csv", "C4": "07_three_way.csv",
          "C1r": "08_claim1.csv", "F1": "11_inlet_errors.csv", "F3": "10_sensor_ladder.csv", "F4/F5": "12_controller_and_sensors.csv",
          "F6": "05_protocol_decision.json", "I1": "11_eps_sweep.csv", "B1": "09 (to build)", "T1": "14 (to build)", "V1": "13 (to build)"}
display(pd.DataFrame({"result file": ledger, "present": {k: (R / v).exists() for k, v in ledger.items()}}))'''),
    md(INTERP),
]
NOTEBOOKS["18_si_figures"] = [
    md("# 18 — SI figures\nFigures from stored results only. Writes only to `figures/si/`; never to `data/`."),
    code(SETUP),
    code('''OUTF = FIGS / "si"; OUTF.mkdir(exist_ok=True)
def save(fig, name):
    p = (OUTF / name).resolve(); assert OUTF.resolve() in p.parents
    fig.savefig(p, dpi=200)
if (R / "06_metrics.csv").exists():
    m = pd.read_csv(R / "06_metrics.csv"); e = m[m.member.astype(str) == "ensemble"].set_index("budget")
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in ["Sc1", "Sc2", "Sc3", "Sc4"]:
        ax.plot(e.index, e[f"sdb_{s}"], "o-", label=s)
    ax.set_xscale("log"); ax.set_xlabel("budget"); ax.set_ylabel("posterior sd of beta"); ax.legend(); save(fig, "si_budget_sd.png")'''),
    md(INTERP),
]


# ------------------------------------------------------------------------------------------ 19
NOTEBOOKS["19_prior_sensitivity"] = [
    md("""# 19 — Effect of the prior choice (claim P1)
The production posterior assumes a uniform prior on the box α, β ∈ [0.4, 1.2]. This notebook measures how
much the results depend on that choice, and where.

**Priors compared**
| Label | Prior | How the posterior is obtained |
|---|---|---|
| U-box | uniform [0.4, 1.2]² (production) | production posterior (notebook 06) |
| U-narrow | uniform [0.6, 1.1]² — **excludes Sc5 (β = 0.50)** | retrained, 3 × 20k, training pairs = production-bank pairs inside the box (exact) |
| U-wide | uniform α ∈ [0.3, 1.3], β ∈ [0.4, 1.3] (largest box with a unique, unsaturated-low steady state; §1) | retrained, 3 × 20k, new 60k bank |
| H | truncated normal on the box, mean 1.0, sd 0.15 per axis ("faults are rare") | importance reweighting of U-box samples |
| D | truncated normal on the box, mean 0.7, sd 0.15 per axis ("degradation expected") | reweighting |
| LogU | log-uniform on the box, p ∝ 1/(αβ) (scale-invariant prior for multiplicative factors) | reweighting |

Reweighting is exact for priors on the same support (the posterior is ∝ likelihood × prior, the U-box
posterior ∝ likelihood); its quality is reported as the effective sample size (ESS).

**Questions**
1. Do the headline results (Sc1–Sc4) move when the prior changes? (data-dominated vs prior-dominated)
2. Where does the prior matter: near the thresholds (Sc8, Sc9), near the box edges (Sc5), and under a
   masked monitor (T-only features)?
3. What happens when the prior excludes the truth (U-narrow at Sc5), and does the raw NPE leakage outside
   the box flag it?
4. Calibration when the deployed population differs from the training prior: U-box posterior on truths drawn
   from H, vs the H-reweighted posterior.

**PRE-REGISTER (user to fix and date before running):**
* P1 "results are prior-robust" holds if, for Sc1–Sc4 and every prior whose support contains the truth:
  |posterior-mean shift| ≤ 0.5 U-box posterior sd (α and β), sd ratio within [0.8, 1.25], and the
  three-way decision unchanged in ≥ 95% of windows.
* Expected (reported, not gated): larger shifts at Sc8/Sc9; prior-dominated T-only posteriors;
  U-narrow coverage 0 at Sc5.
* ESS: a reweighted result is reported only where median ESS ≥ 200 of 2000 draws.
* Leakage detector for an excluding prior: flag if raw mass outside the box ≥ 0.2 (threshold fixed here).

Runtime: ~1–2 h (two retrained priors × 3 seeds; 60k new simulations). Needs notebook 06 and 02 (L-sets)."""),
    code(SETUP), code(LOAD_PROD),
    md("## 1. Check the wider box: unique steady state, no low-side saturation, cascade loop stable"),
    code('''WIDE = (np.array([0.3, 0.4]), np.array([1.3, 1.3])); NARROW = (np.array([0.6, 0.6]), np.array([1.1, 1.1]))
bad = []
for a in np.linspace(WIDE[0][0], WIDE[1][0], 25):
    for b in np.linspace(WIDE[0][1], WIDE[1][1], 25):
        for dTi, dTci in [(0, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)]:
            s = ph.ss_closed(P, a, b, Ti=P.Ti + dTi, Tci=P.Tci + dTci)
            if len(s) != 1 or s[0]["sat"] == -1:
                bad.append((a, b, dTi, dTci, len(s)))
print("wide box: grid points with multiple or low-saturated steady states:", len(bad))
assert not bad, "U-wide is not admissible; shrink it and log the change"
# controller check on the wide box (production control structure; see docs/realism_upgrade_2026-11-10.md)
sys.path.insert(0, str(ROOT / "design"))
if cfg.__dict__.get("CONTROL", "closed") == "cascade":
    from check_cascade import jac_eigs
    worst = max(1 / np.abs(ev.real).min() for a in np.linspace(0.3, 1.3, 11) for b in np.linspace(0.4, 1.3, 10)
                for ev in [jac_eigs(P, a, b, *cfg.CASCADE, tw=0.5, tf=0.1)[0]] if ev is not None)
    print("slowest closed-loop time constant on the wide box [min]:", round(worst, 1))'''),
    md("## 2. Helpers: box switching, priors, reweighting"),
    code('''import contextlib, pickle
from scipy.stats import truncnorm
OUT = R / "19"; OUT.mkdir(exist_ok=True)
L = da.load_npz(DATA / "L_prod.npz")
SC = [k for k in scn.CLOSED]                                   # closed-loop catalogue scenarios

@contextlib.contextmanager
def box(lo, hi):
    """Temporarily set the prior box used by training, prior-truncated sampling and the metrics."""
    old = inf.LO.copy(), inf.HI.copy()
    inf.LO[:] = lo; inf.HI[:] = hi
    try:
        yield
    finally:
        inf.LO[:] = old[0]; inf.HI[:] = old[1]

def tn_logpdf(v, m, s, lo, hi):
    return truncnorm.logpdf(v, (lo - m) / s, (hi - m) / s, loc=m, scale=s)

LO0, HI0 = inf.LO.copy(), inf.HI.copy()
PRIORS = {   # log-density on the production box (constants irrelevant)
    "H": lambda th: tn_logpdf(th[..., 0], 1.0, 0.15, LO0[0], HI0[0]) + tn_logpdf(th[..., 1], 1.0, 0.15, LO0[1], HI0[1]),
    "D": lambda th: tn_logpdf(th[..., 0], 0.7, 0.15, LO0[0], HI0[0]) + tn_logpdf(th[..., 1], 0.7, 0.15, LO0[1], HI0[1]),
    "LogU": lambda th: -np.log(th[..., 0]) - np.log(th[..., 1]),
}

def reweight(samples, logp, n_out=1000, seed=0):
    """Sampling-importance-resampling of U-box posterior samples to prior exp(logp); returns (samples, ESS)."""
    rng = np.random.default_rng(seed); B, n, _ = samples.shape
    out = np.empty((B, n_out, 2), np.float32); ess = np.empty(B)
    lw = logp(samples.astype(float)); lw -= lw.max(1, keepdims=True); w = np.exp(lw); w /= w.sum(1, keepdims=True)
    for i in range(B):
        out[i] = samples[i, rng.choice(n, n_out, p=w[i])]; ess[i] = 1.0 / (w[i] ** 2).sum()
    return out, ess

def summarise(label, smp, theta, names, lo, hi, extra=None):
    _, rows = wm.evaluate_scenario_set(smp, theta, names, lo, hi)
    ta, tb = inf.three_way(smp)
    for r in rows:
        m = names == r["scenario"]
        r.update(prior=label, mean_a=float(smp[m, :, 0].mean()), mean_b=float(smp[m, :, 1].mean()),
                 tw_a=ta[m], tw_b=tb[m], **({k: float(np.median(v[m])) for k, v in (extra or {}).items()}))
    return rows'''),
    md("## 3. U-box (production) and the reweighted priors H, D, LogU — full features"),
    code('''sel = np.isin(L["sc_name"], SC)
X, TH, NM = L["x"][sel], L["theta"][sel], L["sc_name"][sel]
base, leak = inf.sample_within_prior(post, X, 2000, rng_seed=0)
rows = summarise("U-box", base[:, :1000], TH, NM, LO0, HI0, {"leak": leak})
for name, lp in PRIORS.items():
    smp, ess = reweight(base, lp, seed=1)
    rows += summarise(name, smp, TH, NM, LO0, HI0, {"ess": ess})
full = pd.DataFrame(rows)'''),
    md("## 4. Retrained boxes: U-narrow (excludes Sc5) and U-wide"),
    code('''bank = da.load_npz(DATA / "06_bank_160k.npz")
def train_box(tag, lo, hi, theta, x):
    ms = []
    with box(lo, hi):
        for s in range(3):
            g = OUT / f"post_{tag}_s{s}.pkl"
            if g.exists():
                ms.append(pickle.load(open(g, "rb"))); continue
            sl = slice(s * 20_000, (s + 1) * 20_000)
            p_, _ = inf.train_npe(theta[sl], x[sl], seed=s); pickle.dump(p_, open(g, "wb")); ms.append(p_)
    return inf.EnsemblePosterior(ms)
inside = np.all((bank["theta"] >= NARROW[0]) & (bank["theta"] <= NARROW[1]), 1)
print("bank pairs inside U-narrow:", inside.sum())
post_narrow = train_box("narrow", *NARROW, bank["theta"][inside], bank["x"][inside])
fw = DATA / "19_bank_wide_60k.npz"
if not fw.exists():
    thw = np.random.default_rng(19_000).uniform(WIDE[0], WIDE[1], size=(60_000, 2))
    da.save_npz(fw, da.make_windows(thw, 19_000, protocol=cfg.PROTOCOL, feature_set=cfg.FEATURE_SET))
bw = da.load_npz(fw)
post_wide = train_box("wide", *WIDE, bw["theta"], bw["x"])
for tag, p_, (lo, hi) in (("U-narrow", post_narrow, NARROW), ("U-wide", post_wide, WIDE)):
    with box(lo, hi):
        smp, lk = inf.sample_within_prior(p_, X, 1000, rng_seed=2)
        full = pd.concat([full, pd.DataFrame(summarise(tag, smp, TH, NM, lo, hi, {"leak": lk}))])
full.drop(columns=["tw_a", "tw_b"]).to_csv(R / "19_prior_full.csv", index=False)'''),
    md("## 5. Sensitivity table and the P1 test"),
    code('''ref = full[full.prior == "U-box"].set_index("scenario")
def contains(prior, sc):
    a, b = scn.SCENARIOS[sc].alpha, scn.SCENARIOS[sc].beta
    lo, hi = {"U-narrow": NARROW, "U-wide": WIDE}.get(prior, (LO0, HI0))
    return lo[0] <= a <= hi[0] and lo[1] <= b <= hi[1]
out = []
for _, r in full[full.prior != "U-box"].iterrows():
    k = r.scenario; r0 = ref.loc[k]
    out.append(dict(prior=r.prior, scenario=k, contains_truth=contains(r.prior, k),
                    shift_a_sd=(r.mean_a - r0.mean_a) / r0.sd_a, shift_b_sd=(r.mean_b - r0.mean_b) / r0.sd_b,
                    sd_ratio_a=r.sd_a / r0.sd_a, sd_ratio_b=r.sd_b / r0.sd_b,
                    same_decision=float(np.mean((r.tw_a == r0.tw_a) & (r.tw_b == r0.tw_b))),
                    cov90_b=r.cov90_truth_b, class_acc=r.class_acc, ess=r.get("ess", np.nan), leak=r.get("leak", np.nan)))
sens = pd.DataFrame(out); sens.to_csv(R / "19_sensitivity.csv", index=False)
display(sens.pivot_table(index="scenario", columns="prior", values=["shift_b_sd", "shift_a_sd", "same_decision"]).round(2))
g = sens[sens.scenario.isin(scn.GATED) & sens.contains_truth & ~((sens.ess < 200) & sens.ess.notna())]
p1 = dict(max_abs_shift_sd=float(np.abs(g[["shift_a_sd", "shift_b_sd"]]).values.max()),
          sd_ratio_range=[float(g[["sd_ratio_a", "sd_ratio_b"]].values.min()), float(g[["sd_ratio_a", "sd_ratio_b"]].values.max())],
          min_same_decision=float(g.same_decision.min()))
p1["P1_holds"] = bool(p1["max_abs_shift_sd"] <= 0.5 and 0.8 <= p1["sd_ratio_range"][0] and p1["sd_ratio_range"][1] <= 1.25
                      and p1["min_same_decision"] >= 0.95)
print(p1); da.save_json(R / "19_P1.json", p1)'''),
    md("## 6. Prior excluding the truth (U-narrow at Sc5): does leakage flag it?"),
    code('''n5 = full[(full.prior == "U-narrow")].set_index("scenario")
display(n5[["mean_b", "sd_b", "cov90_truth_b", "leak"]].round(3))
print("Sc5 flagged by leakage ≥ 0.2:", float(n5.loc["Sc5_severe_fouling", "leak"]) >= 0.2,
      "| largest leak among scenarios whose truth is inside:", float(n5.drop("Sc5_severe_fouling").leak.max()))'''),
    md("## 7. Masked monitor: the same priors on the T-only posterior (prior-dominated?)"),
    code('''it = [fe.NAMES["S1m"].index(n) for n in fe.NAMES["T_only"]]
ms = []
for s in range(3):            # same members as notebook 08's masking arm (trained there if missing)
    g = R / "08" / f"post_M_Tonly_s{s}.pkl"; g.parent.mkdir(exist_ok=True)
    if g.exists():
        ms.append(pickle.load(open(g, "rb")))
    else:
        sl = slice(s * 40_000, (s + 1) * 40_000)
        p_, _ = inf.train_npe(bank["theta"][sl], bank["x"][sl][:, it], seed=s); pickle.dump(p_, open(g, "wb")); ms.append(p_)
post_T = inf.EnsemblePosterior(ms)
gs = np.isin(NM, scn.GATED + ["Sc5_severe_fouling"])
bT, _ = inf.sample_within_prior(post_T, X[gs][:, it], 2000, rng_seed=3)
prior_sd = (HI0 - LO0) / np.sqrt(12)
rowsT = []
for name, lp in [("U-box", None)] + list(PRIORS.items()):
    smp = bT[:, :1000] if lp is None else reweight(bT, lp, seed=4)[0]
    for k in np.unique(NM[gs]):
        m = NM[gs] == k
        rowsT.append(dict(features="T_only", prior=name, scenario=k, mean_b=float(smp[m, :, 1].mean()),
                          sd_b=float(smp[m, :, 1].std(1).mean()), contraction_b=1 - float(smp[m, :, 1].std(1).mean()) / prior_sd[1]))
for name in ["U-box"] + list(PRIORS):
    f_ = full[(full.prior == name) & full.scenario.isin(np.unique(NM[gs]))]
    rowsT += [dict(features="S1m", prior=name, scenario=r.scenario, mean_b=r.mean_b, sd_b=r.sd_b,
                   contraction_b=1 - r.sd_b / prior_sd[1]) for _, r in f_.iterrows()]
tT = pd.DataFrame(rowsT); tT.to_csv(R / "19_masked_vs_full.csv", index=False)
display(tT.pivot_table(index=["features", "scenario"], columns="prior", values="mean_b").round(3))
display(tT.pivot_table(index=["features", "scenario"], columns="prior", values="contraction_b").round(3))'''),
    md("## 8. Population different from the training prior (truths drawn from H)"),
    code('''rng = np.random.default_rng(19_500)
thH = np.stack([truncnorm.rvs((LO0[j] - 1) / 0.15, (HI0[j] - 1) / 0.15, loc=1, scale=0.15, size=1000, random_state=rng)
                for j in (0, 1)], 1)
dH = da.make_windows(thH, 19_500, protocol=cfg.PROTOCOL, feature_set=cfg.FEATURE_SET)
sH, _ = inf.sample_within_prior(post, dH["x"], 2000, rng_seed=5)
pop = {"U-box posterior (mismatched prior)": sH[:, :1000], "H-reweighted posterior (matched)": reweight(sH, PRIORS["H"], seed=6)[0]}
rows = []
for k, smp in pop.items():
    lo90, hi90 = np.quantile(smp, 0.05, 1), np.quantile(smp, 0.95, 1)
    cov = ((dH["theta"] >= lo90) & (dH["theta"] <= hi90)).mean(0)
    rows.append(dict(posterior=k, cov90_a=cov[0], cov90_b=cov[1],
                     rmse_a=float(np.sqrt(((smp[:, :, 0].mean(1) - dH["theta"][:, 0]) ** 2).mean())),
                     rmse_b=float(np.sqrt(((smp[:, :, 1].mean(1) - dH["theta"][:, 1]) ** 2).mean()))))
popt = pd.DataFrame(rows); popt.to_csv(R / "19_population_mismatch.csv", index=False); display(popt.round(4))'''),
    md("## 9. Figures"),
    code('''fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)
for ax, col in zip(axes, ["shift_a_sd", "shift_b_sd"]):
    pv = sens.pivot_table(index="scenario", columns="prior", values=col).reindex(SC)
    pv.plot.bar(ax=ax, width=0.8); ax.axhline(0.5, ls=":", c="k"); ax.axhline(-0.5, ls=":", c="k")
    ax.set_title(col.replace("_sd", "") + " (units of U-box posterior sd)"); ax.set_xlabel("")
fig.savefig(FIGS / "19_prior_shift.png", dpi=150); plt.show()
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for (feat, k), d in tT.groupby(["features", "scenario"]):
    ax.plot(d.prior, d.mean_b, "o-" if feat == "S1m" else "s--", label=f"{feat} {k}")
ax.set_ylabel("posterior mean of β"); ax.legend(fontsize=7, ncol=2); ax.set_title("full features follow the data; T-only follows the prior")
fig.savefig(FIGS / "19_masked_prior_dominance.png", dpi=150); plt.show()'''),
    md(INTERP),
]


def main():
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        (NB_DIR / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(NOTEBOOKS)} notebooks to {NB_DIR}")


if __name__ == "__main__":
    main()
