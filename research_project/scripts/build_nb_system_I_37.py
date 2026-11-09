"""Build / execute notebooks/system_I/37_individual_posterior_lc2st_N3.ipynb (System I; replaces NUTS, user decision 2026-11-09).

L-C2ST (sbi.diagnostics.LC2ST, sbi defaults) of the production ensemble (nb36) on 5 windows per scenario Sc1-Sc5 of D1. Rules in
docs/Manuscript/system_i_N3_runbook.md section 2f (row e2). A written "## 6. Interpretation" is kept on rebuild.

    python scripts/build_nb_system_I_37.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "37_individual_posterior_lc2st_N3.ipynb"

HEADER = """# 37 — Is the production posterior right *for this window*? L-C2ST (System I, N3)

SBC and TARP test the posterior on average over the prior; the coverage tables test it per scenario. They cannot say whether the posterior at one particular observation is correct. The **local classifier two-sample test (L-C2ST**, Linhart et al. 2023;
`sbi.diagnostics.LC2ST`) can: a classifier is trained to tell apart (θ from the prior, x) from (θ from the posterior, x) on a calibration set; at a given x_o, the classifier's accuracy on posterior draws θ_o ~ q(θ | x_o) is compared with its distribution under the
null (a posterior that is exactly right). A small p-value says that the posterior at x_o differs from the truth in a way the classifier can see. This replaces the NUTS ground truth (decision 2026-11-09; the diagnosis is in the plan, SI-1).

**Settings (fixed before running; runbook §2f row e2).** sbi 0.26.1 `LC2ST` with its **defaults**: sklearn MLPClassifier with hidden layers (10·dim θ, 10·dim θ) = (20, 20), adam, early stopping, 1 fold, 1 classifier, seed 1, 100 null trials, permutation null, no z-scoring.
**Calibration set:** N_cal = 20,000 pairs (θ from the prior; x simulated under the production model: protocol P2, N3 sensors, true inlet from the notebook-35 distribution, *measured* Ti, Tci with ε = 0.1 K; new seeds 9 500 001–9 500 004) with one draw from the production ensemble (notebook 36) at each x.
**Test windows:** the first 5 windows of each scenario Sc1–Sc5 in D1 (25 windows, fixed by index), 2,000 ensemble draws per window as θ_o, x_o = the window's S1m features. **Reported:** the per-window p-value, the number below 0.05 (about 1.25 expected under a calibrated posterior) and the minimum; no decision rule.
**Positive controls (section 4; corrected after the first run showed the first design could not work):** the whole test is run for deliberately wrong posteriors (β shifted by +0.10, width halved), with the wrong posterior used for both the calibration draws and the test draws; and labelled variants with standardised inputs and with a compressed observation (section 4).
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
warnings.filterwarnings("ignore")
import importlib, ws_eval, ws_phase3, nb35_measured_inlet, nb36_production_measured_inlet, ws_inlet
for _m in (ws_eval, ws_phase3, nb35_measured_inlet, nb36_production_measured_inlet, ws_inlet):
    importlib.reload(_m)
we, P3, M, P36, wi = ws_eval, ws_phase3, nb35_measured_inlet, nb36_production_measured_inlet, ws_inlet
from sbi.diagnostics import LC2ST
R = ROOT / "results"; OUTD = R / "37_N3"; OUTD.mkdir(exist_ok=True); T0 = time.time()
N_CAL, N_THETA_O, N_NULL = 20_000, 2_000, 100
post, prod = P3.load_production_posterior()
print("Production posterior:", prod["production_budget"], "simulations, ensemble of", len(prod["members"]))
"""),
    ("md", "## 1. Calibration set under the production model (cached)"),
    ("code", """f = OUTD / "cal.npz"
if f.exists():
    c = we.load_npz(f); th, X = c["theta"], c["x"]
else:
    th = we.prior_draws(N_CAL, 9_500_001); inl = M._inlets(N_CAL, 9_500_002)
    b29 = wi.simulate(th, inl, M.PROTOCOL, 9_500_003, features="S0_base29", **M.NOISE)
    X = P36.features(b29, inl, 9_500_004); we.save_npz(f, {"theta": th, "x": X, "inlet": inl})
assert th.shape == (N_CAL, 2) and X.shape == (N_CAL, 34) and np.isfinite(X).all()
smp, leak = we.sample_within_prior(post, X, 1, rng_seed=1)
print(f"calibration set {X.shape}; one ensemble draw per x (prior leakage {leak.mean():.4f}) | {time.time() - T0:.0f}s")
T_ = torch.tensor(th, dtype=torch.float32); X_ = torch.tensor(X, dtype=torch.float32); S_ = torch.tensor(smp[:, 0, :], dtype=torch.float32)
"""),
    ("md", "## 2. Train the L-C2ST classifier and its null (sbi defaults)"),
    ("code", """t = time.time(); lc = LC2ST(T_, X_, S_, seed=1, num_trials_null=N_NULL)
lc.train_on_observed_data(); lc.train_under_null_hypothesis(verbosity=0)
print(f"classifier and {N_NULL} null classifiers trained in {time.time() - t:.0f}s")
"""),
    ("md", "## 3. The 25 test windows: per-window p-values"),
    ("code", """d1 = P3.load_d1(); Xo_all = P3.production_features(d1); sid = d1["scenario_id"]
SCN = {1: "Sc1", 2: "Sc2", 3: "Sc3", 4: "Sc4", 5: "Sc5"}
wins = np.concatenate([np.where(sid == s)[0][:5] for s in SCN])
so, _ = we.sample_within_prior(post, Xo_all[wins], N_THETA_O, rng_seed=2)
def pvals(samples):
    return np.array([lc.p_value(torch.tensor(samples[k], dtype=torch.float32), torch.tensor(Xo_all[w], dtype=torch.float32)) for k, w in enumerate(wins)])
p_main = pvals(so)
res = pd.DataFrame(dict(window=wins, scenario=[SCN[int(sid[w])] for w in wins], alpha_true=d1["theta"][wins, 2], beta_true=d1["theta"][wins, 3],
                        post_mean_alpha=so.mean(1)[:, 0], post_mean_beta=so.mean(1)[:, 1], p_value=p_main))
display(res.round(3)); res.to_csv(R / "37_N3_lc2st_pvalues.csv", index=False)
from scipy import stats
print(f"p < 0.05 in {(p_main < 0.05).sum()} of {len(p_main)} windows (expected about {0.05 * len(p_main):.2f} under a calibrated posterior) | minimum p = {p_main.min():.3f} | "
      f"median p = {np.median(p_main):.3f} | KS test of the 25 p-values against U(0,1): p = {stats.kstest(p_main, 'uniform').pvalue:.3f}")
print("the null is estimated with 100 trials: a printed p = 0.00 means fewer than 1 in 100 null statistics reach the observed one")
"""),
    ("md", """## 4. Positive controls, done correctly (added after the first run: the first power check was flawed)

The first version of the power check evaluated deliberately wrong posterior draws θ_o' with a classifier that had been trained on the *correct* posterior. That cannot work: a classifier trained to separate truth from a correct posterior has never seen the wrong one, so the shifted draws change nothing (the first run
rejected the production posterior and the shifted ones at the same rate, 0.52 and 0.52/0.48, exactly this symptom). **A valid positive control runs the whole L-C2ST for the wrong posterior q'**: the calibration posterior draws and the test draws both come from q'. Wrong posteriors: β shifted by +0.10 (2.3 posterior sd) and the width halved about the ensemble mean (applied to the one calibration draw per x and to the test draws).
Each variant (classifier input and size) is run with the production posterior and with the wrong ones, same 25 windows. **Variants (fixed before this run, after seeing that the pre-set default behaves oddly):** *default* (raw inputs, sbi default (20, 20), 100 null trials; the pre-set headline); *std* (inputs standardised with the calibration-set mean and sd, θ scaled to the prior, (20, 20), 100 null trials);
*comp* (std + x compressed to the ensemble's posterior mean and sd, 6 inputs, (20, 20), 100 null trials). A variant is *informative* if it flags the wrong posteriors in at least 80% of the windows and the production posterior in at most about 10% (the nominal 5% plus the sampling noise of 25 windows)."""),
    ("code", """c_mu, c_sd = X.mean(0), X.std(0); tx = lambda x: (np.asarray(x) - c_mu) / c_sd; tt = lambda t_: (np.asarray(t_) - 0.8) / 0.23
dr, _ = we.sample_within_prior(post, X, 500, rng_seed=7); dro, _ = we.sample_within_prior(post, Xo_all[wins], 500, rng_seed=8)
comp = np.concatenate([dr.mean(1), dr.std(1)], axis=1); c2_mu, c2_sd = comp.mean(0), comp.std(0)
comp_cal = (comp - c2_mu) / c2_sd; comp_obs = (np.concatenate([dro.mean(1), dro.std(1)], axis=1) - c2_mu) / c2_sd
m_cal = dr.mean(1); m_obs = so.mean(1)                                   # ensemble means: the centre of the width control
def make_q(kind, S_cal, S_obs):
    if kind == "production posterior":
        return S_cal, S_obs
    if kind == "beta shifted +0.10":
        a, b = S_cal.copy(), S_obs.copy(); a[:, 1] += 0.10; b[:, :, 1] += 0.10; return a, b
    if kind == "width x0.5":
        return m_cal + 0.5 * (S_cal - m_cal), m_obs[:, None, :] + 0.5 * (S_obs - m_obs[:, None, :])
def run_variant(vname, kind):
    S_cal, S_obs = make_q(kind, smp[:, 0, :], so)
    if vname == "default":
        Tc, Xc, Sc = th, X, S_cal; to = lambda s: s; xo = lambda k: Xo_all[wins[k]]; kw = {}
    elif vname == "std":
        Tc, Xc, Sc = tt(th), tx(X), tt(S_cal); to = tt; xo = lambda k: tx(Xo_all[wins[k]]); kw = {}
    else:
        Tc, Xc, Sc = tt(th), comp_cal, tt(S_cal); to = tt; xo = lambda k: comp_obs[k]; kw = {}
    t0_ = time.time()
    lcv = LC2ST(torch.tensor(Tc, dtype=torch.float32), torch.tensor(Xc, dtype=torch.float32), torch.tensor(Sc, dtype=torch.float32), seed=1, num_trials_null=N_NULL, **kw)
    lcv.train_on_observed_data(); lcv.train_under_null_hypothesis(verbosity=0)
    p = np.array([lcv.p_value(torch.tensor(to(S_obs[k]), dtype=torch.float32), torch.tensor(xo(k), dtype=torch.float32)) for k in range(len(wins))])
    print(f"{vname} | {kind}: {time.time() - t0_:.0f}s | p<0.05 in {(p < 0.05).mean():.2f} of windows, median p {np.median(p):.2f}", flush=True)
    return p
RUNS = [("default", "beta shifted +0.10"), ("std", "production posterior"), ("std", "beta shifted +0.10"), ("std", "width x0.5"), ("comp", "production posterior"), ("comp", "beta shifted +0.10")]
pv_runs = {("default", "production posterior"): p_main}
for vn, kind in RUNS:
    pv_runs[(vn, kind)] = run_variant(vn, kind)
rows = [dict(variant=v, posterior=k, frac_p_below_0_05=float((p < 0.05).mean()), median_p=float(np.median(p)), min_p=float(p.min()), n_windows=len(p)) for (v, k), p in pv_runs.items()]
pc = pd.DataFrame(rows).sort_values(["variant", "posterior"]); display(pc.round(3)); pc.to_csv(R / "37_N3_lc2st_positive_controls.csv", index=False)
pw = pd.DataFrame({f"{v} | {k}": p for (v, k), p in pv_runs.items()}); pw.insert(0, "window", wins); pw.insert(1, "scenario", res.scenario.values)
display(pw.round(2)); pw.to_csv(R / "37_N3_lc2st_pvalues_all_variants.csv", index=False)
print("posterior sd of beta in these windows (mean):", round(float(so.std(1)[:, 1].mean()), 3), "| a +0.10 shift is", round(0.10 / float(so.std(1)[:, 1].mean()), 1), "posterior sd")
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 6. Interpretation — to be written after the run
Toward the storyline: what the individual-posterior test says about the production ensemble per window, how much power the default classifier has (the positive control), what that licenses us to say, and what it does not (the NUTS replacement is a check against a
calibration set simulated by the same simulator, not an independent gold standard)."""),
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
