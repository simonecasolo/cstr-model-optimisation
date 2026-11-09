"""Build (and optionally execute) notebooks/system_I/31_budget_convergence.ipynb.

The notebook IS the experiment: every stage runs in a notebook cell with the kernel's
working directory set to notebooks/system_I. Stage functions live in
scripts/nb31_budget_convergence.py and skip work already on disk (resumable).

Usage:  python scripts/build_nb_system_I_31.py            # build only (run it in Jupyter / VS Code)
        python scripts/build_nb_system_I_31.py --execute  # build + execute headless (nbclient)
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "31_budget_convergence.ipynb"

HEADER = r"""# 31 — Training-budget convergence of the System I production posterior

Fixed controller law (2026-11-08); warm-start protocol chosen by notebook 29
(`results/29fx_selection.json`). This notebook refuses to start if that selection is flagged.

**Design.** One 160k simulation bank under the chosen protocol, simulated once. Budgets
5k, 10k, 20k, 40k, 80k, 160k × seeds 0–4 (nb29 measured ~20 s per 5k training run). Seed *s* trains on the first *N* rows
of its own permutation of the bank: nested across budgets for a fixed seed, different data
across seeds. A held-out test set (1000 SBC draws + 6 scenarios × 50 reps) from a reserved
seed. Each (N, seed) posterior and the 5-seed ensemble are scored with the nb29 suite.

## Convergence rule — fixed before results

Step N_k → N_{k+1} is **converged** if, for the ensemble at both budgets:
(a) |ΔCorr(α,β)| < 0.05 at each of Sc1, Sc2, Sc3, Sc4, Sc7 (Sc5 reported, not gated);
(b) |Δ 90% marginal coverage| < 0.02 for α and β;
(c) |ΔTARP deviation| < 0.01;
(d) at N_{k+1}: marginal KS p ≥ 0.01 for α and β on the **interior** SBC subset (true θ at
least 5% of the prior width from every edge), and TARP KS p ≥ 0.01 on the full set.
Full-set KS is reported alongside. *Amended after nb29, before any nb31 result:* in nb29 the
full-set α KS failed for every protocol at 5k while 90% coverage was ≈0.91; the interior gate
separates estimator bias from prior-edge truncation.

**Production budget** = N_{k+1} of the first converged step. If none converges: flag and review.
**Ensemble vs single seed**: ensemble if its TARP deviation ≤ the median single-seed TARP
deviation at that budget, else the single seed with the lowest TARP deviation.
Result: `results/31_production.json`, loaded downstream with `load_production_posterior()`.

**Running this notebook.** Execute top to bottom from `notebooks/system_I`. Training is the
long part (seconds at 5k, of order 10–30 min per seed at 160k). With `PARALLEL_TRAINING`,
seed groups train in subprocesses launched from this folder. Re-running resumes.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, sys, time, subprocess
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", \\
    f"execute this notebook from notebooks/system_I (cwd is {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import nb31_budget_convergence as B
R, FIGS = ROOT / "results", ROOT / "figures"

# ---- run switches -----------------------------------------------------------------
BUDGETS = B.DEFAULT_BUDGETS          # add 80_000 only if the decision below asks for it
PARALLEL_TRAINING = True
SEED_GROUPS = [[0, 1], [2, 3], [4]]  # one subprocess per group
THREADS = 3
T0 = time.time()

PROTOCOL = B.protocol()              # stops here if nb29's selection is missing or flagged
print("protocol from nb29:", PROTOCOL)"""),

    ("md", """## 0. Scenario catalogue used for the test set
The test set's scenario grid is read from `cstr_sbi.scenarios.SCENARIO_CONFIGS`. nb29 found
Sc5's β on the prior edge and Sc2 saturated; settle any catalogue change **before** section 1
(the test set is simulated once and cached)."""),
    ("code", """from cstr_sbi.scenarios import SCENARIO_CONFIGS
import ws_eval as we
display(pd.DataFrame([{"scenario": k, "alpha": SCENARIO_CONFIGS[k].alpha, "beta": SCENARIO_CONFIGS[k].beta,
                       "drift_T": SCENARIO_CONFIGS[k].drift_T,
                       "on_prior_edge": bool(np.isclose([SCENARIO_CONFIGS[k].alpha, SCENARIO_CONFIGS[k].beta],
                                                        np.r_[we.LO, we.HI].reshape(2, 2)).any())}
                      for k in we.EVAL_SCENARIOS]))"""),
    ("md", "## 1. Simulation bank (160k) and held-out test set"),
    ("code", """B.stage_bank(PROTOCOL)
B.stage_test(PROTOCOL)"""),

    ("md", "## 2. Train budgets × seeds"),
    ("code", """budgets_arg = ",".join(str(b) for b in BUDGETS)
if PARALLEL_TRAINING:
    procs = []
    for g in SEED_GROUPS:
        seeds = ",".join(str(s) for s in g)
        log = open(R / f"31_train_seeds{seeds.replace(',', '_')}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "nb31_budget_convergence.py"), "train",
             "--protocol", PROTOCOL, "--budgets", budgets_arg, "--seeds", seeds, "--threads", str(THREADS)],
            cwd=NB_DIR, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    assert all(c == 0 for c in codes), f"training subprocess failed: {codes} (see results/31_train_*.log)"
else:
    B.stage_train(PROTOCOL, BUDGETS, B.SEEDS, THREADS)
print(sorted(f.name for f in B.OUT.glob("post_*.pkl")))"""),

    ("md", "## 3. Evaluate every (budget, seed) posterior and the ensembles"),
    ("code", "B.stage_eval(PROTOCOL)"),
    ("code", """m = pd.read_csv(R / "31_metrics.csv"); m["member"] = m["member"].astype(str)
ens, single = m[m.member == "ensemble"].sort_values("budget"), m[m.member != "ensemble"]
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), constrained_layout=True)
for sc in ["Sc1", "Sc2", "Sc3", "Sc4", "Sc5", "Sc7"]:
    g = single.groupby("budget")[f"corr_{sc}"]
    axes[0].errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o", capsize=3, label=sc)
axes[0].set_title("posterior Corr(α,β), seed mean ± sd"); axes[0].legend(fontsize=7)
for k, ax, t in [("tarp_dev", axes[1], "TARP deviation"), ("cov90_b", axes[2], "90% coverage β (SBC set)"),
                 ("rmse_norm", axes[3], "RMSE / prior width")]:
    g = single.groupby("budget")[k]
    ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o", capsize=3, label="single seeds")
    ax.plot(ens.budget, ens[k], "k--s", label="5-seed ensemble"); ax.set_title(t); ax.legend(fontsize=7)
for ax in axes:
    ax.set_xscale("log"); ax.set_xlabel("training simulations")
axes[2].axhline(0.9, color="grey", lw=0.8)
fig.savefig(FIGS / "31_budget_convergence.png", dpi=150); plt.show()
display(ens.set_index("budget")[["ks_p_a", "ks_p_b", "ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p", "tarp_dev", "cov90_a", "cov90_b",
                                 "rmse_norm", "macro_f1", "leak_sbc_mean",
                                 "corr_Sc1", "corr_Sc2", "corr_Sc3", "corr_Sc4", "corr_Sc7"]].round(3))"""),

    ("md", """### Per-scenario classification accuracy and β coverage (ensemble)
Fraction of the 50 test windows per scenario assigned to the true fault class (threshold
0.90), and the 90% coverage of β at the truth. Sc2 (β = 0.85) sits at the midpoint of the
unsaturated fouling band, 0.05 below the threshold, so its accuracy is expected below ~100%."""),
    ("code", """scn = pd.read_csv(R / "31_scenarios.csv"); scn["member"] = scn["member"].astype(str)
e_sc = scn[scn.member == "ensemble"]
display(e_sc.pivot_table(index="scenario", columns="budget", values="class_acc").round(3))
display(e_sc.pivot_table(index="scenario", columns="budget", values="cov90_truth_b").round(3))
display(e_sc.pivot_table(index="scenario", columns="budget", values="beta_hat").round(3))"""),
    ("md", """### α calibration and efficiency against the Cramér–Rao bound
Full-set vs interior α SBC KS by budget, and the posterior sd at each scenario divided by the
29-D-summary Cramér–Rao bound from nb29 E2 (same protocol). A ratio well above 1 means the
estimator is not yet extracting the information in the summaries; it should approach ~1 as the
budget grows (exactly 1 only if the summaries' likelihood were Gaussian)."""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
for k, lab in [("ks_p_a", "α full"), ("ks_p_a_interior", "α interior"), ("ks_p_b", "β full"), ("ks_p_b_interior", "β interior")]:
    g = single.groupby("budget")[k].median()
    axes[0].plot(g.index, g.values, "o-", label=lab)
axes[0].axhline(0.01, color="grey", lw=0.8); axes[0].set_yscale("log"); axes[0].set_xscale("log")
axes[0].set_title("SBC KS p (seed median)"); axes[0].legend(fontsize=7); axes[0].set_xlabel("training simulations")
crb = pd.read_csv(R / "29fx_e2_fim.csv")
crb = crb[(crb.protocol == PROTOCOL) & (crb.representation == "summaries_fullcov")]
pts = {"Sc1": (1.0, 1.0), "Sc2": (1.0, 0.7), "Sc3": (0.7, 1.0), "Sc4": (0.85, 0.85)}
for sc, (a, b) in pts.items():
    cfg = [k for k in we.EVAL_SCENARIOS if k.startswith(sc + "_")][0]
    if not (np.isclose(SCENARIO_CONFIGS[cfg].alpha, a) and np.isclose(SCENARIO_CONFIGS[cfg].beta, b)):
        continue            # catalogue changed since nb29: no matching CRB point
    c = crb[crb.point == sc].iloc[0]
    for par, col, cr, ls in [("α", f"sda_{sc}", c.crb_sd_a, "-"), ("β", f"sdb_{sc}", c.crb_sd_b, "--")]:
        if col in ens:
            axes[1].plot(ens.budget, ens[col] / cr, "o" + ls, label=f"{sc} {par}")
axes[1].axhline(1, color="grey", lw=0.8); axes[1].set_xscale("log"); axes[1].set_xlabel("training simulations")
axes[1].set_title("ensemble posterior sd / CRB (29-D summaries)"); axes[1].legend(fontsize=7, ncol=2)
fig.savefig(FIGS / "31_alpha_calibration_efficiency.png", dpi=150); plt.show()"""),
    ("md", "## 4. Decision (applies the pre-set rule)"),
    ("code", """B.stage_decide(PROTOCOL)
res = json.loads((R / "31_production.json").read_text())
display(pd.DataFrame(res["steps"]).drop(columns=["dcorr"]))
print({k: v for k, v in res.items() if k != "steps"})
if res["flag"]:
    print("FLAG:", res["flag"])"""),

    ("md", """## 4b. Amended decision (rule v2) — written after seeing the results, approved by the user

The v1 rule above selected **20k**. Three flaws were found in it: (1) it took the first
converged step without requiring later steps to stay converged (20k→40k fails); (2) it gated on
Sc7, which is misspecified by design (+2 K sensor drift absent from the model), so its posterior
extrapolates and is not expected to converge; (3) it had no precision criterion, while α's
posterior sd at 20k was ~3.5× its Cramér–Rao bound and still halving per doubling.
Rule v2 drops Sc7 from the correlation gate and adds a precision criterion (< 10% change in
posterior sd) and persistence. No step qualifies (α precision still improves ~30% per doubling),
so v2 takes the **largest budget passing calibration: 160k, 5-seed ensemble**. Both choices are
kept in `results/31_production.json`, with the approval record. Note (not changed): criterion (c)
uses a TARP-deviation threshold (0.01) below that statistic's own sampling noise (~0.04).
Report both in the paper: the headline results do not depend on 20k vs 160k; only α precision does."""),
    ("code", """res_v2 = B.stage_decide_amended(
    PROTOCOL, approve=True, approved_by="user (Simone Casolo), in conversation with Claude",
    reason=("User approved 2026-11-08: v1 rule selected 20k but was flawed (no persistence, gated on the "
            "misspecified Sc7, no precision criterion); 160k 5-seed ensemble is calibrated and brings alpha "
            "within ~1.2x of its summary-statistic Cramer-Rao bound (beta at the bound)."))
display(pd.DataFrame(res_v2["rule_v2_steps"]))
print("production:", res_v2["production_budget"], "ensemble" if res_v2["use_ensemble"] else "single",
      "| v1 would have chosen:", res_v2["rule_v1_choice"])"""),
    ("md", "## 5. Smoke test of the production posterior as downstream notebooks will load it"),
    ("code", """import torch
post, info = B.load_production_posterior()
test = we.load_npz(B.OUT / f"test_{PROTOCOL}.npz")
x = test["x_sc"][test["sc_name"] == "Sc4_closed_combined"][:1]
s = post.sample((5000,), x=torch.as_tensor(x[0], dtype=torch.float32),
                show_progress_bars=False, reject_outside_prior=False).numpy()
print(info["protocol"], info["production_budget"], "ensemble" if info["use_ensemble"] else "single",
      "| Sc4 mean", s.mean(0).round(3), "corr", round(float(np.corrcoef(s.T)[0, 1]), 3))
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")"""),
]


N3_NOTE = """> **Sensor model: {sensor}** ({desc}). Budget study under the production sensor model adopted
> on 2026-11-09 after notebook 33. The N0 study is `31_budget_convergence.ipynb` (kept for the
> SI). Outputs carry the prefix `{tag}_`; the production record is `results/{tag}_production.json`.
> **Decision rule: v2 is pre-set here** (adopted by the user on 2026-11-08, before any {sensor}
> result): Sc7 excluded from the gate, < 10% change in posterior sd, persistence; if no step
> qualifies, the largest budget passing calibration. Section 4 still prints the v1 outcome for
> the record."""

V2_MD = """## 4b. Decision under the pre-set rule v2 ({sensor})
Rule v2 is the pre-set rule for this study (see the note at the top). If no budget passes
calibration (d), the record is FLAGGED and the pipeline must stop for user review."""

V2_CODE = """res_v2 = B.stage_decide_amended(
    PROTOCOL, approve=True,
    approved_by="pre-set rule v2 (adopted by the user on 2026-11-08 for all subsequent budget studies)",
    reason="{sensor} production budget study; v2 fixed before any {sensor} result")
display(pd.DataFrame(res_v2["rule_v2_steps"]))
print("production:", res_v2["production_budget"], "ensemble" if res_v2["use_ensemble"] else "single",
      "| v1 would have chosen:", res_v2["rule_v1_choice"], "| sensor model:", res_v2["sensor_model"])
if res_v2.get("flag"):
    raise RuntimeError("STOP: " + res_v2["flag"])"""

INTERP_MD = """## 6. Interpretation ({sensor} run) — to be written after the run
State for the paper, comparing with the N0 notebook: the production budget and ensemble choice;
whether α calibration (which failed at 40k in notebook 33) is reached by 160k; α and β posterior
sd against their Cramér–Rao bounds; per-scenario class accuracy (expect Sc4 below the others:
α is ~1σ from the threshold), β coverage, and α–β correlation (expect ≈ 0 everywhere, including
the saturated Sc5). Keep claims to what the tables show."""


def _for_sensor(cells, sensor):
    import system_I_config as CFG
    if sensor == "N0":                      # default model is now N3: pin N0 explicitly
        pin = ('os.environ["CSTR_SENSOR_MODEL"] = "N0"\n'
               "import importlib, ws_protocols, ws_eval, system_I_config, nb29_ws_study, nb31_budget_convergence\n"
               "for _m in (system_I_config, ws_protocols, ws_eval, nb29_ws_study, nb31_budget_convergence):\n"
               "    importlib.reload(_m)\n"
               "B = nb31_budget_convergence\n"
               'assert B.SENSOR == "N0", B.SENSOR\n')
        out = []
        for k, src in cells:
            if k == "code" and 'import nb31_budget_convergence as B\n' in src:
                src = src.replace('import nb31_budget_convergence as B\n', pin)
                if "import json, os," not in src:
                    src = src.replace("import json, ", "import json, os, ", 1)
            out.append((k, src))
        return out
    tag = "31" + CFG.suffix(sensor)
    out = []
    for k, src in cells:
        if k == "md" and src.startswith("## 4b. Amended decision (rule v2)"):
            out.append(("md", V2_MD.format(sensor=sensor)))
            continue
        if k == "code" and src.startswith("res_v2 = B.stage_decide_amended("):
            out.append(("code", V2_CODE.format(sensor=sensor)))
            continue
        if k == "code":
            src = (src.replace('R / "31_', 'R / f"{B.TAG}_').replace('R / f"31_train_', 'R / f"{B.TAG}_train_')
                      .replace('FIGS / "31_', 'FIGS / f"{B.TAG}_').replace('R / "29fx_e2_fim.csv"', 'R / f"{B.E2_TAG}_e2_fim.csv"'))
            if "import nb31_budget_convergence as B\n" in src:
                src = src.replace("import nb31_budget_convergence as B\n",
                                  f'os.environ["CSTR_SENSOR_MODEL"] = "{sensor}"\n'
                                  "import importlib, ws_protocols, ws_eval, system_I_config, nb31_budget_convergence\n"
                                  "for _m in (system_I_config, ws_protocols, ws_eval, nb31_budget_convergence):\n"
                                  "    importlib.reload(_m)\n"
                                  "B = nb31_budget_convergence\n"
                                  f'assert B.SENSOR == "{sensor}", B.SENSOR\n')
                src = src.replace("import json, sys, time, subprocess", "import json, os, sys, time, subprocess")
        out.append((k, src))
        if k == "md" and src is HEADER:
            out.append(("md", N3_NOTE.format(sensor=sensor, desc=CFG.describe(sensor), tag=tag)))
    out.append(("md", INTERP_MD.format(sensor=sensor)))
    return out


def build(execute: bool = False, sensor: str = "N0") -> Path:
    out_path = OUT if sensor == "N0" else OUT.with_name(OUT.stem + f"_{sensor}.ipynb")
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in _for_sensor(CELLS, sensor)]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    if execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=None, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}})   # cwd = notebooks/system_I
        try:
            client.execute()
        finally:
            nbformat.write(nb, out_path)
    return out_path


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    _sensor = sys.argv[sys.argv.index("--sensor") + 1] if "--sensor" in sys.argv else "N0"
    print(build(execute="--execute" in sys.argv, sensor=_sensor))
