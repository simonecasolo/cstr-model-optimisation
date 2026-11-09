"""Build and execute notebook 20 (System I): six theories for the (alpha,beta)
posterior correlation at Sc4, tested against the real House-style open questions left
by notebook 19 (`notebooks/system_I/19_alpha_beta_banana_and_ekf_asymmetry.ipynb`).

nb19 established (full-budget n=10,000 sims/seed, 10 seeds + 1 cached posterior = 11
total): Corr(alpha,beta) at Sc4 (alpha=beta=0.85) mean=-0.482, std=0.134,
range=[-0.762,-0.269], every seed negative, none near zero -- seed-robust. SBC on the
same 11 posteriors: 7/11 clean pass, 2/11 outright FAIL on alpha (mechanistically
DISTINCT failure signatures, root cause unresolved), 2/11 ambiguous. The local-FIM/
Laplace diagnostic never predicted this correlation anywhere on the grid it was
checked ([-0.052, +0.209]) -- it is either a nonlinear/global phenomenon invisible to
linearisation, or the local-FIM's own diagonal-covariance assumption is itself wrong
(see Theory E below).

This notebook does NOT try to re-settle the SBC-failure-mechanism question (nb19
Section 10 already investigated that specifically and left it open; out of scope
here). Its job is the next-level question: assuming the correlation is real (nb19's
best-supported reading), WHY does it exist? Six theories (A-F), each with an explicit
falsification test (see `scripts/build_nb_system_I_20.py`'s own docstring history / the
originating design-interview spec for the full theory list); Stage 0 first checks
whether nb19's own "n=10,000 is the production budget" choice has actually converged.

**Explicit, loudly-flagged scope reductions relative to the full design-interview
spec** (both decided BEFORE execution, not silently applied -- see the corresponding
markdown cells in Section 5/Theory C for the reasoning in full):

1. **Theory C (NUTS ground truth) -- UPDATED 2026-11-06, additive scope from the
   coordinator mid-run.** The spec's literal "10 windows x 8 scenarios" plan, taken at
   the manuscript's own Table VII NUTS protocol cost ESTIMATE (median ~3e6 ms/window =
   50 min/window), projected to 66-240+ hours -- multiple days, not merely slow -- so
   the first pass through this notebook settled for a reduced, cheap qualitative probe
   instead (2 chains/300 samples/200 warmup, 4 windows/scenario). Directly TIMING both
   that reduced protocol and the full Table-VII-style protocol on this machine (not
   trusting the manuscript's estimate at face value) found both are far cheaper than
   expected in-process (~13s and ~25s/window respectively, vs. the ~50 min/window
   estimate) -- so this notebook now uses the FULL protocol at 20 windows/scenario
   across all 8 scenarios (160 windows total, parallelised across all cores) as the
   "final, reliable" per-scenario NUTS posterior the coordinator asked for, not the
   original cheap probe. See Section 3/Theory C's own markdown cell for the live
   cost-check numbers and the (now much smaller) fallback this cell still applies if
   the projected parallel ETA exceeds 3h.
2. **Theory F (raw-trajectory CNN embedding)**: reuses `04b_embedding_net_study.ipynb`'s
   architecture and training call almost verbatim, but only for the 6 CLOSED-LOOP
   scenarios (Sc1-5, Sc7) that `train_sbi_posterior(use_raw_obs=True, ...)` already
   supports out of the box. Sc0/Sc6 (open-loop) would need a second, bespoke
   raw-trajectory open-loop training path that does not exist anywhere in this
   codebase yet; building, debugging, and training it was judged not worth the time
   this session has, given Theory F's claim (summary-statistic compression) does not
   especially hinge on the open-loop scenarios, which Theory B already covers with a
   dedicated open-loop *summary-statistic* posterior. Flagged, not silently dropped.

Reuses directly: `cstr_sbi.inference.train_sbi_posterior` (including its existing
`use_raw_obs`/`embedding_net`/`scenario_specific_warm_start` arguments -- all of
Theory D and Theory F's training needs are already first-class options on this
function, no reimplementation needed); `cstr_sbi.inference.run_mcmc_baseline` (Theory
C); `scripts/fim_utils.py` (Theories A/E); nb19's `sim_raw_po`/`feat_summary_po`/
`compute_steady_state`/`cstr_rhs` helpers (copied here verbatim where nb19 itself
copied them from nb34, per this project's established convention of each notebook
carrying its own self-contained copy); `04b_embedding_net_study.ipynb`'s `CNNEmbedding`
construction (Theory F); `cstr_sbi.scenarios.SCENARIO_CONFIGS` (single source of truth
for all 8 scenario truths, used throughout instead of nb19's own ad hoc GRID_POINTS).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    return new_markdown_cell(source, metadata={"language": "markdown"})


def code(source: str):
    return new_code_cell(source, metadata={"language": "python"})


CELLS = [
    md(
        r"""# Notebook 20 (System I) -- six theories for the $(\alpha,\beta)$ posterior correlation, tested

**Context (read `HANDOFF.md` and `notebooks/system_I/TODO.md` top entries for the full
history before trusting any number below).** Notebook 19 found a real, seed-robust
posterior correlation Corr($\alpha,\beta$) at Sc4 ($\alpha=\beta=0.85$): across 10
independently-trained full-budget ($n=10{,}000$ sims/seed) SNPE posteriors,
mean$=-0.482$, std$=0.134$, range $[-0.762,-0.269]$ -- every seed negative, none near
zero. SBC on those 10 seeds + the 1 pre-existing cached posterior (11 total): 7/11 pass
cleanly, 2/11 fail outright on $\alpha$ (two DIFFERENT failure signatures, root cause
unresolved), 2/11 ambiguous. The local-FIM/Laplace diagnostic stayed small everywhere
checked ($[-0.052,+0.209]$) and never predicted this correlation.

**This notebook's job**, per the confirmed design-interview plan:
- **Stage 0**: check whether $n=10{,}000$ (nb19's working budget) has actually
  converged, by sweeping training budget and watching Corr($\alpha,\beta$) at Sc4.
  This specific result is flagged as **candidate SI content** (see its own markdown
  cell) -- everything else in this notebook stays internal/exploratory only.
- **Stage 1**: per-scenario Corr($\alpha,\beta$) survey across all of Sc0-Sc7, at the
  production budget Stage 0 settles on.
- **Stage 2**: six theories (A-F below), each with an explicit falsification test.
- **Stage 3**: an explicit CONFIRMED/FALSIFIED/PARTIALLY SUPPORTED/INCONCLUSIVE verdict
  per theory, plus a synthesis cell.

**Nothing in this notebook is promoted into `main.tex`/the SI/`article_outline_CChE.md`**
-- per the design interview, that stays a human decision, even for the Stage 0
convergence curve (which is only "candidate" content, formatting pending).

**Two scope reductions are flagged loudly up front** (full reasoning in
`scripts/build_nb_system_I_20.py`'s own docstring and in Section 5/Theory-C's own
markdown cell): Theory C (NUTS) runs a reduced protocol, not a Table-VII replication,
because the literal spec (10 windows x 8 scenarios at the full NUTS budget) projects to
multiple days of wall-clock even parallelised across every core on this machine. Theory
F (raw-trajectory CNN) covers the 6 closed-loop scenarios only, not Sc0/Sc6, because no
open-loop raw-trajectory training path exists anywhere in this codebase and building one
from scratch was judged not worth this session's remaining time given Theory B already
covers the open-vs-closed-loop axis directly."""
    ),
    md("## 0. Setup"),
    code(
        """import sys, time, pickle, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import torch

from fim_utils import compute_fim, cov_offdiag_corr, crb_report

from cstr_sbi.physics import (
    K0_NOMINAL, UA_NOMINAL, NOMINAL_INLET_CL, NOMINAL_CTRL, NOMINAL_Y0_CL,
    NOMINAL_INLET, QC0,
)
from cstr_sbi.simulator import (
    warm_start_ic, simulate_em_window, simulate_em_window_open_loop,
    apply_sensor_layer, DEFAULT_SENSOR_NOISE_PCT,
)
from cstr_sbi.summaries import compute_summary_statistics, FEATURE_NAMES
from cstr_sbi.priors import box_uniform_2d
from cstr_sbi.inference import train_sbi_posterior, run_mcmc_baseline
from cstr_sbi.scenarios import SCENARIO_CONFIGS, list_configs

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

prior = box_uniform_2d()
SCENARIOS = list_configs()
print(f"{len(SCENARIOS)} scenarios loaded:")
for sc in SCENARIOS:
    print(f"  {sc.name:24s} alpha={sc.alpha:.2f} beta={sc.beta:.2f} mode={sc.mode:12s} drift_T={sc.drift_T}")

N_CPU = os.cpu_count()
print(f"\\ncpu_count = {N_CPU}")


def sim_raw_po(theta, seed, noise_pct=DEFAULT_SENSOR_NOISE_PCT, drift_T=0.0):
    \"\"\"Closed-loop raw trajectory for [alpha,beta], matched-protocol warm start.\"\"\"
    alpha, beta = float(theta[0]), float(theta[1])
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float64)
    y0 = warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
    proc_key, sens_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    t, ys, qc = simulate_em_window(params, NOMINAL_INLET_CL, NOMINAL_CTRL, y0, key=proc_key)
    obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
    obs = apply_sensor_layer(obs, key=sens_key, noise_pct=noise_pct)
    if drift_T != 0.0:
        obs = obs.at[:, 1].add(drift_T)
    t_s = jnp.arange(1, ys.shape[0] + 1) * 0.5
    return np.asarray(obs), np.asarray(t_s)


def feat_summary_po(theta, seed, feature_mask=None, noise_pct=DEFAULT_SENSOR_NOISE_PCT, drift_T=0.0):
    obs, t_s = sim_raw_po(theta, seed, noise_pct=noise_pct, drift_T=drift_T)
    s = np.asarray(compute_summary_statistics(jnp.array(obs), jnp.array(t_s)))
    if feature_mask is not None:
        s = s[feature_mask]
    return s


def sim_raw_po_openloop(theta, seed, noise_pct=DEFAULT_SENSOR_NOISE_PCT):
    \"\"\"Open-loop raw trajectory for [alpha,beta]: Qc fixed, UA/k0 directly scaled by
    beta/alpha (same fault-encoding convention as `scenarios.ScenarioConfig.open_loop_params`).\"\"\"
    alpha, beta = float(theta[0]), float(theta[1])
    params_ol = jnp.array([beta * UA_NOMINAL, alpha * K0_NOMINAL], dtype=jnp.float64)
    # Open-loop steady state under HEALTHY conditions is used as warm start (matches
    # M2's own open-loop data-gen convention for Sc0/Sc6 -- the open-loop simulator has
    # no feedback to re-settle to a new steady state under a fault, by construction).
    y0_ol = warm_start_ic(jnp.array([UA_NOMINAL, K0_NOMINAL, 1.0, 1.0]), NOMINAL_INLET_CL, NOMINAL_CTRL)[:3]
    proc_key, sens_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    t, ys, qc = simulate_em_window_open_loop(params_ol, NOMINAL_INLET, y0_ol, key=proc_key)
    obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
    obs = apply_sensor_layer(obs, key=sens_key, noise_pct=noise_pct)
    t_s = jnp.arange(1, ys.shape[0] + 1) * 0.5
    return np.asarray(obs), np.asarray(t_s)


def feat_summary_po_openloop(theta, seed, noise_pct=DEFAULT_SENSOR_NOISE_PCT):
    obs, t_s = sim_raw_po_openloop(theta, seed, noise_pct=noise_pct)
    return np.asarray(compute_summary_statistics(jnp.array(obs), jnp.array(t_s)))


def obs_x_for_scenario(sc, seed=999):
    \"\"\"Return a torch tensor of the 29-D summary at a scenario's own true (alpha,beta),
    using the matching (open- vs closed-loop) simulator.\"\"\"
    theta = np.array([sc.alpha, sc.beta])
    if sc.mode == "open_loop":
        s = feat_summary_po_openloop(theta, seed=seed)
    else:
        s = feat_summary_po(theta, seed=seed, drift_T=sc.drift_T)
    return torch.tensor(s, dtype=torch.float32)


def corr_ab(posterior, x, n_draw=5000, seed=0):
    torch.manual_seed(seed)
    samp = posterior.sample((n_draw,), x=x, show_progress_bars=False).numpy()
    return float(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1]), samp

import datetime as _dt

def mark(stage):
    \"\"\"Write a disk-flushed progress sentinel -- NOT print(). nbclient captures cell
    stdout into the notebook's in-memory structure, invisible to any external log/file
    watcher until the whole run finishes (confirmed the hard way, twice, earlier in this
    session's multi-hour stall diagnoses). A file write is immediate and real, giving
    external monitoring a live view of which stage is actually running, not just which
    stage last finished.\"\"\"
    (RESULTS / f"20_progress_{stage}.txt").write_text(
        f"{time.time()}\\n{_dt.datetime.now().isoformat()}\\n"
    )

mark("setup_done")
print("Setup complete.")
"""
    ),
    md(
        """## 1. Stage 0 -- training-budget convergence check

Sweep `n_train in {500, 1000, 2500, 5000, 10000, 20000}`, 3 seeds per level except
$n=10{,}000$, which reuses nb19's already-cached 10 seeds
(`results/19_multiseed_posterior_seed{0..9}.pkl`) directly rather than retraining.
Compute Corr($\\alpha,\\beta$) at Sc4 for every (budget, seed) pair."""
    ),
    code(
        """# UPDATED (2026-11-06, additive scope from the coordinator mid-run): the first pass
# through this cell used only 3 seeds at n=20,000 and stopped there once the 10k->20k
# step exceeded the 0.05 convergence threshold, without actually confirming n=20,000
# itself is converged or escalating further. This version (a) matches n=20,000's seed
# count to n=10,000's (10 seeds, not 3) so the two levels are compared on equal footing,
# and (b) dynamically escalates to n=40,000 (5 seeds) if 10k->20k is still not converged,
# capping at one escalation level per the original spec's "go one level higher... flag
# prominently rather than silently stop" -- it does not silently keep climbing forever.
mark("stage0_training_start")
BUDGET_SEED_COUNTS = {500: 3, 1000: 3, 2500: 3, 5000: 3, 10000: 10, 20000: 10}
BUDGETS = [500, 1000, 2500, 5000, 10000, 20000]

x_sc4 = obs_x_for_scenario(SCENARIO_CONFIGS["Sc4_closed_combined"])

def train_or_load_budget_seed(budget, seed):
    if budget == 10000 and seed < 10:
        cache_path = RESULTS / f"19_multiseed_posterior_seed{seed}.pkl"
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        return d["posterior"], d["meta"], "reused-nb19"
    cache_path = RESULTS / f"20_budget{budget}_seed{seed}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        return d["posterior"], d["meta"], "cached-this-nb"
    t0 = time.perf_counter()
    post, meta = train_sbi_posterior(
        prior, n_simulations=budget, density_estimator="nsf",
        hidden_features=128, num_transforms=5, training_batch_size=256,
        max_num_epochs=200, seed=seed, scenario_specific_warm_start=True,
    )
    meta["seed"] = seed
    meta["wall_time_s"] = time.perf_counter() - t0
    with open(cache_path, "wb") as f:
        pickle.dump({"posterior": post, "meta": meta}, f)
    return post, meta, f"trained-now ({meta['wall_time_s']:.0f}s)"

budget_rows = []
t0_stage0 = time.perf_counter()
for budget in BUDGETS:
    for seed in range(BUDGET_SEED_COUNTS[budget]):
        post, meta, source = train_or_load_budget_seed(budget, seed)
        corr, _ = corr_ab(post, x_sc4, seed=100_000 + budget + seed)
        budget_rows.append({"budget": budget, "seed": seed, "corr": corr, "source": source})
        print(f"budget={budget:6d} seed={seed}: Corr(a,b)@Sc4={corr:+.3f}  [{source}]")

df_budget = pd.DataFrame(budget_rows)
print(f"\\nStage 0 (initial levels) wall time: {time.perf_counter()-t0_stage0:.0f}s")
print("STAGE 0 INITIAL TRAINING DONE")
"""
    ),
    code(
        """mark("stage0_escalation_start")
CONVERGENCE_THRESHOLD = 0.05
ESCALATION_BUDGETS = [40000]  # at most one escalation level, per the spec's own cap
ESCALATION_SEED_COUNT = 5

def mean_at(df, budget):
    return df.loc[df["budget"] == budget, "corr"].mean()

converged = False
escalation_log = []
prev_budget, cur_budget = 10000, 20000
delta = abs(mean_at(df_budget, cur_budget) - mean_at(df_budget, prev_budget))
print(f"n={prev_budget} -> n={cur_budget}: |delta mean| = {delta:.3f} (threshold {CONVERGENCE_THRESHOLD})")
converged = delta < CONVERGENCE_THRESHOLD
escalation_log.append((prev_budget, cur_budget, delta, converged))

for esc_budget in ESCALATION_BUDGETS:
    if converged:
        break
    print(f"NOT converged at n={cur_budget} -- escalating to n={esc_budget} "
          f"({ESCALATION_SEED_COUNT} seeds), per the spec's single-escalation-level plan.")
    BUDGETS.append(esc_budget)
    BUDGET_SEED_COUNTS[esc_budget] = ESCALATION_SEED_COUNT
    t0 = time.perf_counter()
    for seed in range(ESCALATION_SEED_COUNT):
        post, meta, source = train_or_load_budget_seed(esc_budget, seed)
        corr, _ = corr_ab(post, x_sc4, seed=100_000 + esc_budget + seed)
        budget_rows.append({"budget": esc_budget, "seed": seed, "corr": corr, "source": source})
        print(f"budget={esc_budget:6d} seed={seed}: Corr(a,b)@Sc4={corr:+.3f}  [{source}]")
    df_budget = pd.DataFrame(budget_rows)
    print(f"Escalation training ({esc_budget}): {time.perf_counter()-t0:.0f}s")
    prev_budget, cur_budget = cur_budget, esc_budget
    delta = abs(mean_at(df_budget, cur_budget) - mean_at(df_budget, prev_budget))
    print(f"n={prev_budget} -> n={cur_budget}: |delta mean| = {delta:.3f} (threshold {CONVERGENCE_THRESHOLD})")
    converged = delta < CONVERGENCE_THRESHOLD
    escalation_log.append((prev_budget, cur_budget, delta, converged))

df_budget.to_csv(RESULTS / "20_budget_convergence.csv", index=False)
df_escalation = pd.DataFrame(escalation_log, columns=["from_budget", "to_budget", "abs_delta_mean", "converged"])
df_escalation.to_csv(RESULTS / "20_budget_convergence_escalation_log.csv", index=False)
print(df_escalation.to_string(index=False))

FINAL_BUDGET_CONVERGED = converged
print(f"\\nFinal escalation level reached: n={cur_budget}. Converged per {CONVERGENCE_THRESHOLD} threshold? {converged}")
print("STAGE 0 TRAINING (incl. any escalation) DONE")
"""
    ),
    code(
        """summary0 = df_budget.groupby("budget")["corr"].agg(["mean", "std", "min", "max", "count"])
print(summary0)

fig, ax = plt.subplots(figsize=(7, 5))
for budget in BUDGETS:
    sub = df_budget[df_budget["budget"] == budget]
    ax.scatter([budget] * len(sub), sub["corr"], color="tab:blue", alpha=0.6, zorder=3)
    m, s = sub["corr"].mean(), sub["corr"].std()
    ax.errorbar([budget], [m], yerr=[s if not np.isnan(s) else 0], fmt="o", color="tab:red",
                capsize=4, zorder=4, markersize=7)
ax.set_xscale("log")
ax.set_xlabel("Training budget (simulations per seed)")
ax.set_ylabel(r"Corr($\\alpha,\\beta$) at Sc4 ($\\alpha=\\beta=0.85$)")
ax.set_title(r"System I: SNPE training-budget convergence of Corr($\\alpha,\\beta$) at Sc4 "
             "(blue = per-seed draws, red = mean +/- std per budget level)")
ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "20_budget_convergence.png", dpi=150)
plt.close(fig)
print("Saved figures/20_budget_convergence.png")

PRODUCTION_BUDGET = cur_budget
FINAL_SEED_COUNT = BUDGET_SEED_COUNTS[PRODUCTION_BUDGET]
print(f"\\nPRODUCTION_BUDGET selected: {PRODUCTION_BUDGET} ({FINAL_SEED_COUNT} seeds)")
print(f"Confirmed converged by this run's {CONVERGENCE_THRESHOLD}-threshold test? {FINAL_BUDGET_CONVERGED}")
if not FINAL_BUDGET_CONVERGED:
    print("WARNING -- NOT CONFIRMED CONVERGED even after the single permitted escalation level. "
          "Flagged explicitly (not silently accepted) in the markdown cell immediately below; "
          "PRODUCTION_BUDGET is the best available choice, not a confirmed-converged one.")
"""
    ),
    md(
        """**Candidate SI content, numbers stable, formatting pending.** The budget-convergence
curve above (`figures/20_budget_convergence.png`, `results/20_budget_convergence.csv`,
`results/20_budget_convergence_escalation_log.csv`) is the one result in this notebook
intended as candidate manuscript/SI content -- the rest stays internal/exploratory. It
has NOT been inserted into `main.tex`/the SI; that remains a human decision.

**Explicit convergence verdict (read `FINAL_BUDGET_CONVERGED` printed above before
trusting `PRODUCTION_BUDGET` as "the converged budget" rather than just "the budget we
stopped at").** This cell's own escalation logic allows exactly one escalation level
beyond $n=20{,}000$ (to $n=40{,}000$, 5 seeds) if the $n{=}10{,}000\\to n{=}20{,}000$ step
exceeds the 0.05 convergence threshold, per the original spec's cap ("go one level
higher... flag prominently rather than silently stop", not "keep climbing until
converged at any cost"). If the escalation log above still shows `converged=False` at
its last row, `PRODUCTION_BUDGET` is **the best available choice from this run, not a
confirmed-converged one** -- that distinction must be carried forward explicitly into
the final SBI-vs-NUTS comparison section and into `HANDOFF.md`, not smoothed over.
`PRODUCTION_BUDGET` (and the actual number of seeds trained at it, `FINAL_SEED_COUNT`)
is then used for every subsequent stage of this notebook, including the final
per-scenario SBI posterior deliverable."""
    ),
    md(
        """## 2. Stage 1 -- per-scenario Corr($\\alpha,\\beta$) survey, Sc0-Sc7

Uses `PRODUCTION_BUDGET`'s posteriors. For the 6 closed-loop scenarios (Sc1-5,7), reuse
the already-trained closed-loop posteriors directly (an amortised SNPE posterior can be
queried at any new observation without retraining). Sc0/Sc6 (open-loop) need a
DEDICATED open-loop-trained posterior, since the closed-loop training simulator cannot
coherently explain an open-loop observation -- trained here as a small new side task
(`FINAL_SEED_COUNT` seeds, same production budget), reusing the `sim_raw_po_openloop`/
`feat_summary_po_openloop` wrappers defined in Section 0 and a hand-rolled SNPE training
loop (the existing `train_sbi_posterior` only supports the closed-loop simulator)."""
    ),
    code(
        """mark("stage1_closedloop_load_start")
closed_loop_posts = {}
for seed in range(FINAL_SEED_COUNT):
    if PRODUCTION_BUDGET == 10000 and seed < 10:
        with open(RESULTS / f"19_multiseed_posterior_seed{seed}.pkl", "rb") as f:
            closed_loop_posts[seed] = pickle.load(f)["posterior"]
    else:
        with open(RESULTS / f"20_budget{PRODUCTION_BUDGET}_seed{seed}.pkl", "rb") as f:
            closed_loop_posts[seed] = pickle.load(f)["posterior"]
print(f"{len(closed_loop_posts)} closed-loop production-budget posteriors loaded (n={PRODUCTION_BUDGET}).")
"""
    ),
    code(
        """mark("stage1_openloop_train_start")
from sbi.inference import SNPE, simulate_for_sbi
from sbi.neural_nets import posterior_nn

def train_openloop(seed, n_simulations):
    cache_path = RESULTS / f"20_openloop_posterior_seed{seed}_n{n_simulations}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)["posterior"]
    torch.manual_seed(seed)
    de_build = posterior_nn(model="nsf", hidden_features=128, num_transforms=5,
                             z_score_x="independent")
    inf = SNPE(prior=prior, density_estimator=de_build, show_progress_bars=False)
    counter = [seed * 100_000]
    def wrapper(th_batch):
        th_np = th_batch.detach().cpu().numpy()
        out = []
        for row in th_np:
            counter[0] += 1
            out.append(feat_summary_po_openloop(row, seed=counter[0]))
        return torch.tensor(np.stack(out), dtype=torch.float32)
    t0 = time.perf_counter()
    theta_tr, x_tr = simulate_for_sbi(simulator=wrapper, proposal=prior,
                                       num_simulations=n_simulations, num_workers=1,
                                       show_progress_bar=False)
    inf.append_simulations(theta_tr, x_tr)
    import io, sys as _sys
    _cap = io.StringIO(); _orig = _sys.stdout; _sys.stdout = _cap
    try:
        de = inf.train(training_batch_size=256, max_num_epochs=200, show_train_summary=False)
    finally:
        _sys.stdout = _orig
    post = inf.build_posterior(de)
    wall = time.perf_counter() - t0
    with open(cache_path, "wb") as f:
        pickle.dump({"posterior": post, "wall_time_s": wall}, f)
    print(f"  open-loop seed={seed}: wall={wall:.0f}s -- cached to {cache_path.name}")
    return post

# UPDATED (2026-11-06, additive scope): bumped from a fixed 3 seeds to FINAL_SEED_COUNT
# (matching the closed-loop side) so the open-loop posteriors backing Sc0/Sc6 in the
# "final, reliable" per-scenario SBI deliverable have the same seed-robustness standard
# as the closed-loop posteriors, not a thinner diagnostic-only sample.
N_OPENLOOP_SEEDS = FINAL_SEED_COUNT
t0 = time.perf_counter()
openloop_posts = {s: train_openloop(s, PRODUCTION_BUDGET) for s in range(N_OPENLOOP_SEEDS)}
print(f"Open-loop training total: {time.perf_counter()-t0:.0f}s for {N_OPENLOOP_SEEDS} seeds")
print("STAGE 1 OPEN-LOOP TRAINING DONE")
"""
    ),
    code(
        """# EXTENDED (2026-11-06, additive scope): now also records each seed's posterior
# mean/std for alpha and beta directly (not just Corr(alpha,beta)), pooling across seeds
# -- this is what the final SBI-vs-NUTS comparison table needs on the SBI side, and it
# was free to add since corr_ab already draws the full samples, just discarded them
# before.
stage1_rows = []
for sc in SCENARIOS:
    x = obs_x_for_scenario(sc)
    if sc.mode == "open_loop":
        posts = openloop_posts
    else:
        posts = closed_loop_posts
    corrs, alpha_means, beta_means, alpha_stds, beta_stds = [], [], [], [], []
    for seed, post in posts.items():
        c, samp = corr_ab(post, x, seed=300_000 + sc.id * 100 + seed)
        corrs.append(c)
        alpha_means.append(samp[:, 0].mean()); beta_means.append(samp[:, 1].mean())
        alpha_stds.append(samp[:, 0].std()); beta_stds.append(samp[:, 1].std())
    corrs = np.array(corrs)
    stage1_rows.append({
        "scenario": sc.name, "id": sc.id, "alpha": sc.alpha, "beta": sc.beta,
        "mode": sc.mode, "n_seeds": len(corrs),
        "corr_mean": corrs.mean(), "corr_std": corrs.std(),
        "corr_min": corrs.min(), "corr_max": corrs.max(),
        "sbi_alpha_mean": np.mean(alpha_means), "sbi_alpha_std": np.mean(alpha_stds),
        "sbi_beta_mean": np.mean(beta_means), "sbi_beta_std": np.mean(beta_stds),
    })
    print(f"{sc.name:24s} mode={sc.mode:12s} Corr(a,b): mean={corrs.mean():+.3f} "
          f"std={corrs.std():.3f} range=[{corrs.min():+.3f},{corrs.max():+.3f}] (n={len(corrs)} seeds)  "
          f"a_hat={np.mean(alpha_means):.3f} b_hat={np.mean(beta_means):.3f}")

df_stage1 = pd.DataFrame(stage1_rows)
df_stage1.to_csv(RESULTS / "20_stage1_per_scenario_corr.csv", index=False)
print("\\nSaved results/20_stage1_per_scenario_corr.csv")
print("STAGE 1 DONE")
"""
    ),
    md(
        """## 3. Stage 2 -- six theories

Each theory gets its own falsification test and, in Stage 3, an explicit verdict cell."""
    ),
    md(
        """### Theory A -- energy-balance compensation ridge (physical)

Claim: $\\alpha$ (heat-generation proxy) and $\\beta$ (heat-removal capacity) enter a
nonlinear combination that integral control drives toward balance at steady state. Test:
scan the real closed-loop steady state (via `warm_start_ic`, the actual nonlinear ODE,
not a linearised approximation) over the $(\\alpha,\\beta)$ grid and look for a curve
along which $T_{ss}$, $C_{ss}$, $Q_{c,ss}$ stay (near-)invariant; compare its direction
to the dominant eigenvector of the trained posterior's covariance at Sc4."""
    ),
    code(
        """mark("theoryA_start")
def closed_loop_steady_state(alpha, beta):
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float64)
    y0 = warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
    return np.asarray(y0)  # [C, T, Tc, I]

A_GRID = np.linspace(0.4, 1.2, 17)
B_GRID = np.linspace(0.4, 1.2, 17)
ss_grid = np.zeros((len(A_GRID), len(B_GRID), 4))
for i, a in enumerate(A_GRID):
    for j, b in enumerate(B_GRID):
        ss_grid[i, j] = closed_loop_steady_state(a, b)

np.savez(RESULTS / "20_theoryA_steadystate_grid.npz", alpha=A_GRID, beta=B_GRID, ss=ss_grid)

# Qc_ss implied by the PI controller at steady state (energy balance proxy): for each
# alpha, find beta(alpha) such that Qc_ss stays closest to Qc_ss(alpha=1,beta=1) --
# i.e. the level set of Qc_ss through the healthy point, a direct nonlinear-energy-
# balance-ridge candidate (the controller's own steady-state coolant duty is exactly
# the quantity integral control uses to compensate -- NOT T_ss, which is pinned to the
# setpoint by construction whenever the controller is not saturated, per main.tex
# sec:identifiability's own dT_ss/dbeta=0 argument).
Qc_ss_grid = ss_grid[:, :, 3] * 0 + np.nan  # placeholder; Qc not directly in y0 state -- recompute via controller law
"""
    ),
    code(
        """# y0 = [C, T, Tc, I] does not carry Qc directly (Qc is a controller OUTPUT, not a
# state); recompute Qc_ss from the PI law using the converged I (integral term) and the
# (pinned) T_ss, matching NOMINAL_CTRL = [Kp, tau_I, Tsp, Qc0, Qc_min, Qc_max].
Kp, tau_I, Tsp, Qc0_c, Qc_min, Qc_max = [float(v) for v in NOMINAL_CTRL]

Qc_ss_grid = np.zeros((len(A_GRID), len(B_GRID)))
for i, a in enumerate(A_GRID):
    for j, b in enumerate(B_GRID):
        C_ss, T_ss, Tc_ss, I_ss = ss_grid[i, j]
        err = Tsp - T_ss
        Qc_raw = Qc0_c + Kp * err + Kp / tau_I * I_ss
        Qc_ss_grid[i, j] = float(np.clip(Qc_raw, Qc_min, Qc_max))

np.savez(RESULTS / "20_theoryA_qc_ss_grid.npz", alpha=A_GRID, beta=B_GRID, qc_ss=Qc_ss_grid)

# The healthy-point Qc_ss level set is the candidate "energy-balance ridge": for each
# alpha, find the beta whose Qc_ss is closest to the healthy Qc_ss.
healthy_i = np.argmin(np.abs(A_GRID - 1.0))
healthy_j = np.argmin(np.abs(B_GRID - 1.0))
Qc_ss_healthy = Qc_ss_grid[healthy_i, healthy_j]
ridge_beta = np.array([B_GRID[np.argmin(np.abs(Qc_ss_grid[i, :] - Qc_ss_healthy))] for i in range(len(A_GRID))])

fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(A_GRID, B_GRID, Qc_ss_grid.T, levels=20, cmap="viridis")
plt.colorbar(cf, ax=ax, label="Qc_ss (L/min)")
ax.plot(A_GRID, ridge_beta, color="red", lw=2, label="Qc_ss-constant ridge (candidate energy-balance ridge)")
ax.scatter([0.85], [0.85], color="white", edgecolor="black", s=60, zorder=5, label="Sc4")
ax.set_xlabel("alpha"); ax.set_ylabel("beta"); ax.legend(loc="lower left", fontsize=8)
ax.set_title("Theory A: steady-state Qc level sets vs. (alpha,beta)")
fig.tight_layout()
fig.savefig(FIGS / "20_theoryA_ridge.png", dpi=150)
plt.close(fig)

# Empirical ridge slope near Sc4 (local linear fit of ridge_beta(alpha) around alpha=0.85)
near = np.abs(A_GRID - 0.85) <= 0.15
ridge_slope = np.polyfit(A_GRID[near], ridge_beta[near], 1)[0]
print(f"Theory A: local slope of the Qc_ss-constant ridge near Sc4 (d_beta/d_alpha) = {ridge_slope:+.3f}")

# Dominant eigenvector direction of the trained posterior covariance at Sc4 (seed0, the
# largest-magnitude, SBC-clean seed per nb19).
_, samp_sc4_seed0 = corr_ab(closed_loop_posts[0], x_sc4, seed=424242)
cov_post = np.cov(samp_sc4_seed0.T)
eigvals, eigvecs = np.linalg.eigh(cov_post)
dom_vec = eigvecs[:, np.argmax(eigvals)]
posterior_slope = dom_vec[1] / dom_vec[0] if dom_vec[0] != 0 else np.inf
print(f"Theory A: dominant eigenvector slope of the trained posterior's covariance at Sc4 = {posterior_slope:+.3f}")
print(f"Theory A: do the two slopes agree in sign? {'YES' if ridge_slope * posterior_slope > 0 else 'NO'}")
"""
    ),
    md(
        """### Theory B -- open-loop masking confirmation (physical)

Direct test using Stage 1's per-scenario survey: compare Sc0 (open, healthy) vs Sc1
(closed, healthy), and Sc6 (open, fouling) vs Sc2 (closed, fouling). If integral
control is the ridge's cause, open-loop scenarios should show near-zero correlation
while their closed-loop counterparts show a real one."""
    ),
    code(
        """def get_row(name):
    return df_stage1.loc[df_stage1["scenario"] == name].iloc[0]

sc0, sc1 = get_row("Sc0_open_healthy"), get_row("Sc1_closed_healthy")
sc6, sc2 = get_row("Sc6_open_with_fault"), get_row("Sc2_closed_fouling")

print("Healthy pair:")
print(f"  Sc0 (open):   Corr={sc0['corr_mean']:+.3f} (std={sc0['corr_std']:.3f})")
print(f"  Sc1 (closed): Corr={sc1['corr_mean']:+.3f} (std={sc1['corr_std']:.3f})")
print("\\nFouling pair:")
print(f"  Sc6 (open):   Corr={sc6['corr_mean']:+.3f} (std={sc6['corr_std']:.3f})")
print(f"  Sc2 (closed): Corr={sc2['corr_mean']:+.3f} (std={sc2['corr_std']:.3f})")

theoryB_support = (abs(sc0["corr_mean"]) < abs(sc1["corr_mean"])) and (abs(sc6["corr_mean"]) < abs(sc2["corr_mean"]))
print(f"\\nBoth open-loop |Corr| < matched closed-loop |Corr|? {theoryB_support}")
"""
    ),
    md(
        """### Theory D -- protocol/warm-start artifact (methodological)

Retrain at the production budget with `scenario_specific_warm_start=False` (fixed
healthy warm start for every training draw, the OLD protocol this project already found
and fixed once for the beta-bias finding) instead of `True` (matched, per-draw warm
start), 3 seeds, and check whether Corr($\\alpha,\\beta$) at Sc4 (and across the grid)
moves meaningfully."""
    ),
    code(
        """mark("theoryD_start")
def train_fixedwarmstart(seed, n_simulations):
    cache_path = RESULTS / f"20_fixedwarmstart_posterior_seed{seed}_n{n_simulations}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        return d["posterior"]
    t0 = time.perf_counter()
    post, meta = train_sbi_posterior(
        prior, n_simulations=n_simulations, density_estimator="nsf",
        hidden_features=128, num_transforms=5, training_batch_size=256,
        max_num_epochs=200, seed=seed, scenario_specific_warm_start=False,
    )
    wall = time.perf_counter() - t0
    with open(cache_path, "wb") as f:
        pickle.dump({"posterior": post, "meta": meta, "wall_time_s": wall}, f)
    print(f"  fixed-warm-start seed={seed}: wall={wall:.0f}s")
    return post

N_D_SEEDS = 3
t0 = time.perf_counter()
fixedws_posts = {s: train_fixedwarmstart(s, PRODUCTION_BUDGET) for s in range(N_D_SEEDS)}
print(f"Theory D training: {time.perf_counter()-t0:.0f}s")

theoryD_rows = []
for sc in SCENARIOS:
    if sc.mode == "open_loop":
        continue  # fixed-warm-start variant only exists for the closed-loop simulator
    x = obs_x_for_scenario(sc)
    corrs_fixed = [corr_ab(p, x, seed=400_000 + sc.id * 10 + s)[0] for s, p in fixedws_posts.items()]
    corrs_matched_row = get_row(sc.name)
    theoryD_rows.append({
        "scenario": sc.name,
        "corr_matched_mean": corrs_matched_row["corr_mean"],
        "corr_fixed_mean": np.mean(corrs_fixed), "corr_fixed_std": np.std(corrs_fixed),
    })
    print(f"{sc.name:24s} matched-protocol Corr={corrs_matched_row['corr_mean']:+.3f}   "
          f"fixed-warm-start Corr mean={np.mean(corrs_fixed):+.3f} std={np.std(corrs_fixed):.3f}")

df_theoryD = pd.DataFrame(theoryD_rows)
df_theoryD.to_csv(RESULTS / "20_theoryD_warmstart_comparison.csv", index=False)
print("STAGE 2 THEORY D DONE")
"""
    ),
    md(
        """### Theory E -- sensor-noise cross-correlation (statistical)

Estimate the EMPIRICAL full covariance of the 29-D summary statistics from simulator
replicates (not the diagonal approximation nb19's local-FIM scan used) at each Sc0-7
point, recompute the local-FIM-implied Corr($\\alpha,\\beta$) using that full covariance,
and compare to the diagonal-covariance FIM and to the trained posterior's empirical
correlation."""
    ),
    code(
        """mark("theoryE_start")
def compute_fim_full_cov(feat_fn, theta_point, eps, n_reps_sigma=120, seed_offset=0):
    theta_point = np.asarray(theta_point, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    n_params = len(theta_point)
    reps = np.stack([feat_fn(theta_point, seed=seed_offset + 5000 + i) for i in range(n_reps_sigma)])
    Sigma_full = np.cov(reps.T) + 1e-10 * np.eye(reps.shape[1])
    Sigma_inv = np.linalg.inv(Sigma_full)
    n_feat = reps.shape[1]
    J = np.zeros((n_feat, n_params))
    for j, e in enumerate(eps):
        th_p = theta_point.copy(); th_p[j] += e
        th_m = theta_point.copy(); th_m[j] -= e
        s_p = feat_fn(th_p, seed=seed_offset + j * 10 + 1)
        s_m = feat_fn(th_m, seed=seed_offset + j * 10 + 2)
        J[:, j] = (s_p - s_m) / (2 * e)
    return J.T @ Sigma_inv @ J

EPS_PO = np.array([0.02, 0.02])
ALPHA_IDX_PO, BETA_IDX_PO = 0, 1

theoryE_rows = []
t0 = time.perf_counter()
for sc in SCENARIOS:
    theta_pt = np.array([sc.alpha, sc.beta])
    if sc.mode == "open_loop":
        ff = lambda th, seed: feat_summary_po_openloop(th, seed=seed)
    else:
        ff = lambda th, seed: feat_summary_po(th, seed=seed, drift_T=sc.drift_T)
    FIM_diag = compute_fim(ff, theta_pt, EPS_PO, n_reps_sigma=60)
    FIM_full = compute_fim_full_cov(ff, theta_pt, EPS_PO, n_reps_sigma=120)
    corr_diag = cov_offdiag_corr(FIM_diag, ALPHA_IDX_PO, BETA_IDX_PO)
    corr_full = cov_offdiag_corr(FIM_full, ALPHA_IDX_PO, BETA_IDX_PO)
    trained_mean = get_row(sc.name)["corr_mean"]
    theoryE_rows.append({
        "scenario": sc.name, "fim_corr_diag": corr_diag, "fim_corr_fullcov": corr_full,
        "trained_posterior_corr": trained_mean,
    })
    print(f"{sc.name:24s} FIM-diag Corr={corr_diag:+.3f}  FIM-fullcov Corr={corr_full:+.3f}  "
          f"trained-posterior Corr={trained_mean:+.3f}")
    mark(f"theoryE_progress_{sc.name}")

df_theoryE = pd.DataFrame(theoryE_rows)
df_theoryE.to_csv(RESULTS / "20_theoryE_fullcov_fim.csv", index=False)
print(f"\\nTheory E wall time: {time.perf_counter()-t0:.0f}s")
print("STAGE 2 THEORY E DONE")
"""
    ),
    md(
        """### Theory F -- summary-statistic compression artifact (representational)

Reuses `04b_embedding_net_study.ipynb`'s CNN-embedding SNPE architecture, evaluated on
the 6 CLOSED-LOOP scenarios (Sc1-5,Sc7) -- see this notebook's top-level docstring for
why Sc0/Sc6 are out of scope here.

**Budget decision, revised live during this run (2026-11-06) -- read before trusting
`cnn_posts`'s budget.** The original instruction was to train at `PRODUCTION_BUDGET`
("not 04b's original budget, per the explicit instruction to use the appropriate
training budget"). After Stage 0's convergence check escalated `PRODUCTION_BUDGET` to
40,000, attempting Theory F's CNN+raw-trajectory training at that budget stalled for
**7h41m with zero progress** (confirmed via direct process inspection: sustained CPU in
active `torch::autograd` linear-layer computation, but no new SBI training log
directory and no cached posterior appearing, against a direct empirical baseline of
~12 min/seed END-TO-END for the SAME training call at n=20,000 measured earlier in this
same session -- a 35x+ overrun with no sign of progress is treated as a real stall, not
a slow-but-legitimate stage). Root cause not fully pinned down (CNN+raw-trajectory
training cost evidently does NOT scale the way the hand-crafted-summary NSF training's
cost does with `n_simulations` -- plausibly a batching/threading pathology specific to
the embedding-net path at this data volume, not confirmed further for time reasons).
**Decision: Theory F's CNN training is capped at a fixed n=20,000 (reusing the already
-trained, already-cached, already-safe posteriors from earlier in this session),
explicitly NOT following `PRODUCTION_BUDGET`'s escalation to 40,000.** This is a
loud, explicit scope deviation from the original instruction, not a silent one -- Theory
F's comparison below is therefore at a different (smaller) budget than Stage 1's final
SBI deliverable and the Stage 0 convergence curve, and that mismatch must be stated
plainly in Stage 3's verdict, not glossed over."""
    ),
    code(
        """mark("theoryF_start")
from sbi.neural_nets.embedding_nets import CNNEmbedding

def build_embedding_net():
    return CNNEmbedding(
        input_shape=(120,), in_channels=4, out_channels_per_layer=[16, 32],
        num_conv_layers=2, num_linear_layers=2, num_linear_units=64,
        output_dim=20, kernel_size=5, pool_kernel_size=2,
    )

def train_cnn_embedding(seed, n_simulations):
    cache_path = RESULTS / f"20_theoryF_cnn_posterior_seed{seed}_n{n_simulations}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)["posterior"]
    mark(f"theoryF_training_seed{seed}_n{n_simulations}_start")
    emb = build_embedding_net()
    t0 = time.perf_counter()
    post, meta = train_sbi_posterior(
        prior, n_simulations=n_simulations, density_estimator="nsf",
        hidden_features=128, num_transforms=5, training_batch_size=256,
        max_num_epochs=300, seed=seed, embedding_net=emb, use_raw_obs=True,
        scenario_specific_warm_start=True,
    )
    wall = time.perf_counter() - t0
    with open(cache_path, "wb") as f:
        pickle.dump({"posterior": post, "meta": meta, "wall_time_s": wall}, f)
    print(f"  CNN-embedding seed={seed}: wall={wall:.0f}s")
    return post

N_F_SEEDS = 3
THEORYF_BUDGET = 20000  # capped, NOT PRODUCTION_BUDGET -- see markdown cell above for why
if THEORYF_BUDGET != PRODUCTION_BUDGET:
    print(f"NOTE: Theory F uses a capped budget ({THEORYF_BUDGET}), different from "
          f"PRODUCTION_BUDGET ({PRODUCTION_BUDGET}) used everywhere else in this notebook -- "
          f"see this section's markdown cell for why (a 7h41m stall at the production budget).")
t0 = time.perf_counter()
cnn_posts = {s: train_cnn_embedding(s, THEORYF_BUDGET) for s in range(N_F_SEEDS)}
print(f"Theory F training: {time.perf_counter()-t0:.0f}s")
print("STAGE 2 THEORY F TRAINING DONE")
"""
    ),
    code(
        """def obs_x_raw_for_scenario(sc, seed=999):
    obs, t_s = sim_raw_po(np.array([sc.alpha, sc.beta]), seed=seed, drift_T=sc.drift_T)
    return torch.tensor(np.asarray(obs).reshape(-1), dtype=torch.float32)

theoryF_rows = []
for sc in SCENARIOS:
    if sc.mode == "open_loop":
        continue
    x_raw = obs_x_raw_for_scenario(sc)
    corrs_cnn = [corr_ab(p, x_raw, seed=500_000 + sc.id * 10 + s)[0] for s, p in cnn_posts.items()]
    summary_row = get_row(sc.name)
    theoryF_rows.append({
        "scenario": sc.name,
        "corr_summary_stat_mean": summary_row["corr_mean"],
        "corr_cnn_raw_mean": np.mean(corrs_cnn), "corr_cnn_raw_std": np.std(corrs_cnn),
    })
    print(f"{sc.name:24s} summary-stat Corr={summary_row['corr_mean']:+.3f}   "
          f"CNN-raw Corr mean={np.mean(corrs_cnn):+.3f} std={np.std(corrs_cnn):.3f}")

df_theoryF = pd.DataFrame(theoryF_rows)
df_theoryF.to_csv(RESULTS / "20_theoryF_cnn_comparison.csv", index=False)
print("STAGE 2 THEORY F DONE")
"""
    ),
    code(
        """mark("theoryC_covariance_precompute_start")
# Precompute the full empirical summary-statistic covariance at each scenario's true
# theta, ONCE per scenario (not per window -- it only depends on theta, not window
# index), using the same replicate-based methodology as Theory E.
N_REPS_COV = 120
scenario_cov_obs = {}
t0 = time.perf_counter()
for sc in SCENARIOS:
    theta_pt = np.array([sc.alpha, sc.beta])
    if sc.mode == "open_loop":
        reps = np.stack([feat_summary_po_openloop(theta_pt, seed=7000 + i) for i in range(N_REPS_COV)])
    else:
        reps = np.stack([feat_summary_po(theta_pt, seed=7000 + i, drift_T=sc.drift_T) for i in range(N_REPS_COV)])
    Sigma_full = np.cov(reps.T) + 1e-6 * np.eye(reps.shape[1])
    scenario_cov_obs[sc.name] = Sigma_full
    print(f"{sc.name:24s}: cov_obs computed, shape={Sigma_full.shape}")
print(f"Covariance precompute: {time.perf_counter()-t0:.0f}s for {len(SCENARIOS)} scenarios")
"""
    ),
    md(
        r"""## 5. Theory-D-mechanism investigation (added 2026-11-07, coordinator-directed
follow-up to Theory D's "partially supported" verdict)

Theory D found that the matched-warm-start SBI posterior and the old fixed-warm-start
SBI posterior give very different Corr($\alpha,\beta$), but left open WHY, and whether
the matched protocol (already validated as fixing SBI's beta-bias) might itself be
injecting a spurious correlation rather than revealing a real one. This section isolates
the mechanism directly rather than leaving it as a correlation-of-correlations
observation.

**Step 1 -- does NUTS's own likelihood use the same per-theta matched warm start SBI's
training data generation uses?** Reading `cstr_generative_model`
(`src/cstr_sbi/inference.py`) directly: it takes `y0` as a FIXED keyword argument
(default `NOMINAL_Y0_CL`, the healthy nominal steady state) and uses it identically for
EVERY posterior draw --
```python
ts, ys, qc = simulate_closed_loop_trajectory(params, inlet, ctrl, y0, t_final=t_window, n_save=n_save)
```
-- `y0` is never recomputed from the sampled `alpha,beta`. `run_mcmc_baseline`'s own
default (`y0=None`) resolves to `warm_start_ic(NOMINAL_PARAMS_CL, inlet, ctrl)` -- the
HEALTHY (alpha=beta=1) steady state -- and this notebook's own `nuts_one_window` worker
never overrode it before this section. Meanwhile, the OBSERVED data NUTS is asked to fit
(`sim_raw_po` in this notebook, mirroring `simulation_wrapper_sbi`'s
`scenario_specific_warm_start=True` path) is generated using `warm_start_ic(params, ...)`
at the SCENARIO'S OWN true `(alpha,beta)`. **Confirmed mismatch**: NUTS's internal
generative model and the process that generated the data it is fitting were, until this
section, two different models -- structurally the same train/eval protocol mismatch
already found and fixed for SBI's own beta bias, apparently never ported to this NUTS
model."""
    ),
    code(
        r"""mark("section5_step2_decisive_test_start")
# Step 2 -- decisive test. Fix cstr_generative_model/run_mcmc_baseline with an additive,
# backward-compatible `matched_warm_start` flag (src/cstr_sbi/inference.py) that, when
# True, recomputes y0 PER DRAW as warm_start_ic(params, inlet, ctrl) -- the literal same
# function SBI's matched-protocol training calls -- instead of using a fixed y0. Compare
# matched_warm_start=False (the old/mismatched behaviour) against =True (the fix) on the
# IDENTICAL observed window at Sc4, holding the already-fixed full-covariance likelihood
# constant, to isolate this one variable.
_sc4 = SCENARIO_CONFIGS["Sc4_closed_combined"]
_sc4_theta = np.array([0.85, 0.85])
_s_sc4 = feat_summary_po(_sc4_theta, seed=777)
_cov_sc4 = scenario_cov_obs[_sc4.name]

step2_rows = []
for label, mws in [("matched_warm_start=False (old)", False), ("matched_warm_start=True (fix)", True)]:
    t0 = time.perf_counter()
    _mcmc_s2, _samp_s2, _wall_s2 = run_mcmc_baseline(
        _s_sc4, seed=0, n_chains=4, n_samples=500, n_warmup=300, progress_bar=False,
        cov_obs=_cov_sc4, matched_warm_start=mws,
    )
    wall = time.perf_counter() - t0
    by_chain_s2 = _mcmc_s2.get_samples(group_by_chain=True)
    import numpyro.diagnostics as _npd
    res_s2 = _npd.summary(by_chain_s2)
    corr_s2 = float(np.corrcoef(_samp_s2[:, 0], _samp_s2[:, 1])[0, 1])
    step2_rows.append({
        "setting": label, "corr": corr_s2,
        "r_hat_alpha": float(res_s2["alpha"]["r_hat"]), "r_hat_beta": float(res_s2["beta"]["r_hat"]),
        "wall_s": wall,
    })
    print(f"{label}: corr={corr_s2:+.3f}  r_hat a={res_s2['alpha']['r_hat']:.3f} "
          f"b={res_s2['beta']['r_hat']:.3f}  wall={wall:.0f}s")

df_step2 = pd.DataFrame(step2_rows)
df_step2.to_csv(RESULTS / "20_section5_step2_decisive_test.csv", index=False)
sbi_sc4_corr = get_row("Sc4_closed_combined")["corr_mean"]
print(f"\\nFor reference, SBI's matched-protocol posterior at Sc4: Corr={sbi_sc4_corr:+.3f}")
print("STEP 2 DONE")
"""
    ),
    md(
        r"""**Step 2 result (read directly from the printed/saved numbers above, not assumed):**
the correlation moved substantially toward SBI's own Sc4 value once NUTS's internal
model was switched to the matched warm start -- a large, same-signed shift, not a small
or null one. Both settings' R-hat stayed within the convergence bar at this single test
point (so this is a genuine shift in the posterior's correlation structure, not an
artifact of one setting failing to converge). The wall-clock cost of `matched_warm_start
=True` is substantially higher than `=False` (a second differentiable adaptive-step ODE
solve -- the warm-start itself -- is now nested inside every NUTS gradient evaluation),
consistent with the cost-check performed before committing to the full re-run below.

**Reading this result**: the protocol mismatch identified in Step 1 explains a large
part of why Theory C's original (mismatched) sweep disagreed with SBI. This is evidence
FOR the "NUTS-side protocol mismatch" explanation of the Theory C/E tension, not for
"SBI's correlation is a training artifact" -- the opposite direction from this
notebook's original Theory C verdict. Per the decisive-test design, this result commits
this notebook to re-running the full Theory C sweep under `matched_warm_start=True` and
rewriting Stage 3 accordingly (see Section 3/Theory C and Stage 3 below, both now
reflecting the corrected protocol)."""
    ),
    code(
        r"""mark("section5_step3_warmstart_coupling_start")
# Step 3 -- does the matched warm-start's OWN converged steady state have a built-in
# (alpha,beta) coupling, independent of the dynamic-trajectory question above? Compute
# the local sensitivity of warm_start_ic's own converged [C,T,Tc,I] to alpha vs beta at
# Sc4, and check whether an iso-contour direction (a combination of alpha,beta that
# leaves the warm-start steady state unchanged) aligns with the posterior's own dominant
# eigenvector direction (computed in Theory A, posterior_slope).
EPS_WS = 0.02
a0, b0 = 0.85, 0.85

def warm_start_state(alpha, beta):
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float64)
    return np.asarray(warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL))  # [C,T,Tc,I]

base_ws = warm_start_state(a0, b0)
d_da = (warm_start_state(a0 + EPS_WS, b0) - warm_start_state(a0 - EPS_WS, b0)) / (2 * EPS_WS)
d_db = (warm_start_state(a0, b0 + EPS_WS) - warm_start_state(a0, b0 - EPS_WS)) / (2 * EPS_WS)

state_names = ["C", "T", "Tc", "I"]
warmstart_rows = []
for i, name in enumerate(state_names):
    iso_slope = -d_da[i] / d_db[i] if d_db[i] != 0 else float("inf")
    warmstart_rows.append({"state": name, "d_dalpha": d_da[i], "d_dbeta": d_db[i], "iso_slope": iso_slope})
    print(f"{name}: d/dalpha={d_da[i]:+.4f}  d/dbeta={d_db[i]:+.4f}  "
          f"iso-{name} slope (dbeta/dalpha)={iso_slope:+.4f}")

df_warmstart = pd.DataFrame(warmstart_rows)
df_warmstart.to_csv(RESULTS / "20_section5_step3_warmstart_sensitivity.csv", index=False)

# Compare against Theory A's posterior dominant-eigenvector slope (already computed above).
print(f"\\nPosterior dominant-eigenvector slope at Sc4 (from Theory A): {posterior_slope:+.3f}")
print("Note: the iso-slope here is the direction that leaves the warm-start STATE "
      "unchanged, not the posterior's own spread direction -- comparing sign/magnitude, "
      "not expecting numerical equality.")
print("STEP 3 DONE")
"""
    ),
    md(
        r"""**Step 3 result.** The warm-start computation genuinely couples alpha and beta -- the
integral-controller state $I_{ss}$ in particular has large, comparable-magnitude
sensitivities to both ($\partial I_{ss}/\partial\alpha \approx +318$,
$\partial I_{ss}/\partial\beta \approx -2516$ at Sc4), consistent with the physical
picture that the steady-state coolant duty needed to hold $T_{ss}$ at setpoint depends
jointly on reaction heat generation (alpha) and heat-removal capacity (beta). However,
**the resulting iso-contour direction (the combination of alpha,beta that leaves
$T_{ss}/T_{c,ss}/I_{ss}$ unchanged, slope $\approx +0.10$ to $+0.13$) does NOT align with
the posterior's own dominant-eigenvector direction (slope $\approx -7.73$, Theory A) --
these are closer to orthogonal than aligned.** Reading: the matched warm-start's
converged state does have a real, physically-expected (alpha,beta) coupling of its own,
but this specific local-linear check does not show that coupling's own degenerate
direction is WHAT PRODUCES the large empirical posterior correlation -- that still looks
like it comes from the full nonlinear dynamic trajectory + summary-statistic map (as
Theory E's full-covariance finding already suggested), not from the warm-start solve in
isolation. INCONCLUSIVE for the specific "warm-start ridge alignment" hypothesis, though
it does confirm the warm-start is not theta-independent in any of its four states."""
    ),
    md(
        r"""**Step 4 -- is an alternative warm-start protocol needed?** Per the original design:
propose/test an alternative ONLY if Steps 1-3 show the matched protocol is injecting
something SPURIOUS (an artifact beyond physically-real coupling), not merely confirm a
NUTS-side mismatch. That is NOT what was found here: Step 2 shows that putting NUTS and
SBI on the SAME (matched) generative model brings them into agreement (large,
same-signed correlation), which argues the matched protocol is revealing something
real and consistent across two independent inference methods, not manufacturing an
artifact unique to one of them. Step 3 confirms the warm-start has real physical
(alpha,beta) coupling but does not show it specifically explains the empirical ridge
direction, so there is no identified "spurious coupling mechanism" to design around.
**Conclusion: no alternative warm-start protocol is proposed or tested here** -- the
evidence points toward the matched protocol being the physically appropriate one to use
consistently in BOTH methods (exactly what Theory D's original, pre-fix "fixed warm
start closer to NUTS" reading would have recommended AGAINST doing, and what this
section's Step 2 now argues was the wrong direction to read that result in). This
reframes Theory D's own verdict -- see the updated Theory D verdict in Stage 3 below."""
    ),
    md(
        """### Theory C -- SNPE vs NUTS ground truth (methodological)

**UPDATED (2026-11-06, additive scope from the coordinator mid-run).** The original
pass through this section cost-checked a REDUCED NUTS protocol (2 chains/300 samples/
200 warmup) against the manuscript's own Table VII protocol estimate (median ~50
min/window) and, taking that estimate at face value, concluded 10 windows/scenario x 8
scenarios was infeasible, settling for a cheap 4-windows/scenario qualitative probe.
**Directly timing both the reduced AND the Table-VII-style FULL protocol
(4 chains/500 samples/300 warmup, `run_mcmc_baseline`'s own defaults) on this machine,
in-process, gave ~13s and ~25s/window respectively** -- nowhere near the ~50 min/window
estimate (most likely because that estimate reflects different conditions, e.g. a cold,
non-JIT-warm process, or additional convergence diagnostics not exercised by this
quick timing; this notebook does not attempt to explain the manuscript's own number,
only reports what was actually measured here). Given this, the notebook now uses the
**full Table-VII-style protocol** (not the reduced one) and **20 windows/scenario across
all 8 scenarios** (160 windows total) -- chosen as enough for a stable *pooled
correlation* estimate per scenario (the goal here), not Table VII's own 50-window
high-precision bias/MAE target for a single scenario, which is a different, more
demanding goal this notebook is not attempting to replicate. Still parallelised across
all CPU cores via the standalone `scripts/nb20_nuts_worker.py` module (needed after an
earlier `multiprocessing`-pickling crash, see that file's own docstring).

**Likelihood-covariance fix, added 2026-11-06 (coordinator-flagged precondition, fixed
before any Theory C numbers below are trusted).** `cstr_generative_model` (the NUTS
target, `src/cstr_sbi/inference.py`) defaulted to a DIAGONAL Gaussian likelihood over
the 29-D summary statistics -- the SAME independence assumption Theory E (Section 3)
found to be wrong for the local-FIM diagnostic. Reproduced directly: a diagonal-
likelihood NUTS run at the healthy point gave R-hat(alpha)=3.4-5.2, R-hat(beta)=8.5-11.1
across 3 seeds, with chains visibly settling into genuinely DIFFERENT (alpha,beta)
basins -- re-running the IDENTICAL window with the full empirical covariance as the
likelihood's `cov_obs` instead collapsed this to R-hat $\\le$ 1.002 for both parameters
across all 3 seeds. **The original "multimodality" finding was therefore a
likelihood-misspecification artifact of the diagonal assumption, not genuine
ground-truth multimodality in the physical model.** `cstr_generative_model` and
`run_mcmc_baseline` now both accept an optional `cov_obs` full-covariance matrix
(additive, backward-compatible -- `sigma_obs`/diagonal behaviour is unchanged when
`cov_obs` is not given); this notebook always supplies it from here on, computed ONCE
per scenario (120 replicates at that scenario's true theta, the same methodology Theory
E already uses) and reused across all of that scenario's windows."""
    ),
    code(
        """mark("theoryC_costcheck_start")
# Cost-check: time ONE real window at both the reduced probe-grade protocol and the
# full Table-VII-style protocol, before committing to either. Uses the full-covariance
# likelihood (see markdown cell above) -- the cost-check must reflect the SAME
# likelihood the real sweep will use, not the now-abandoned diagonal one.
from nb20_nuts_worker import nuts_one_window

_sc4 = SCENARIO_CONFIGS["Sc4_closed_combined"]
PROTOCOLS_TO_PROBE = {
    "reduced (2ch/300samp/200warm)": (2, 300, 200),
    "full Table-VII-style (4ch/500samp/300warm)": (4, 500, 300),
}
probe_times = {}
for label, (nc, ns, nw) in PROTOCOLS_TO_PROBE.items():
    t0 = time.perf_counter()
    res = nuts_one_window((_sc4.name, _sc4.alpha, _sc4.beta, _sc4.mode, _sc4.drift_T,
                            9999, str(ROOT), nc, ns, nw, scenario_cov_obs[_sc4.name]))
    wall = time.perf_counter() - t0
    probe_times[label] = wall
    print(f"{label}: {wall:.1f}s (status={res['status']}, r_hat_alpha={res.get('r_hat_alpha')}, "
          f"r_hat_beta={res.get('r_hat_beta')})")

N_WINDOWS_PER_SCENARIO = 20
NUTS_PROTOCOL = (4, 500, 300)  # (n_chains, n_samples, n_warmup) -- the full protocol
probe_wall = probe_times["full Table-VII-style (4ch/500samp/300warm)"]
total_windows = N_WINDOWS_PER_SCENARIO * len(SCENARIOS)
eta_hours_serial = total_windows * probe_wall / 3600
eta_hours_parallel = eta_hours_serial / max(N_CPU - 1, 1)
print(f"\\nProjected ETA for {total_windows} windows at the full protocol: "
      f"{eta_hours_serial:.2f}h serial, ~{eta_hours_parallel:.2f}h parallelised across {N_CPU-1} workers")
if eta_hours_parallel > 3.0:
    print("WARNING: projected parallel ETA exceeds 3h even at the measured-cheap rate -- "
          "flagging loudly per the project's scope-reduction discipline. Falling back to "
          "10 windows/scenario (matching the original spec's depth exactly) instead of 20.")
    N_WINDOWS_PER_SCENARIO = 10
    total_windows = N_WINDOWS_PER_SCENARIO * len(SCENARIOS)
print(f"Final choice: {N_WINDOWS_PER_SCENARIO} windows/scenario x {len(SCENARIOS)} scenarios "
      f"= {total_windows} windows, protocol={NUTS_PROTOCOL}")
"""
    ),
    code(
        """mark("theoryC_sweep_start")
import multiprocessing as mp

tasks = []
for sc in SCENARIOS:
    for w in range(N_WINDOWS_PER_SCENARIO):
        tasks.append((sc.name, sc.alpha, sc.beta, sc.mode, sc.drift_T, w, str(ROOT),
                      *NUTS_PROTOCOL, scenario_cov_obs[sc.name]))

CACHE_C = RESULTS / "20_theoryC_nuts_results.pkl"
if CACHE_C.exists():
    with open(CACHE_C, "rb") as f:
        nuts_results = pickle.load(f)
    print(f"Reusing cached Theory C NUTS results from {CACHE_C.name} ({len(nuts_results)} windows)")
else:
    t0 = time.perf_counter()
    n_workers = max(min(N_CPU - 1, len(tasks)), 1)
    with mp.get_context("spawn").Pool(n_workers) as pool:
        nuts_results = pool.map(nuts_one_window, tasks)
    with open(CACHE_C, "wb") as f:
        pickle.dump(nuts_results, f)
    print(f"Theory C NUTS sweep: {len(tasks)} windows across {n_workers} workers, "
          f"{time.perf_counter()-t0:.0f}s wall time")

df_nuts = pd.DataFrame([{k: v for k, v in r.items() if k != "samples"} for r in nuts_results])
df_nuts.to_csv(RESULTS / "20_theoryC_nuts_results.csv", index=False)
print(df_nuts["status"].value_counts())
print("STAGE 2 THEORY C DONE")
"""
    ),
    md(
        """**Convergence diagnostics added 2026-11-06 (coordinator-flagged gap, fixed before
any pooling happens below).** The first pass through this cell pooled raw NUTS samples
across windows using only `status == "ok"` (did the NUTS call not crash), with NO check
on whether the chains behind each window's samples actually converged -- `_mcmc`, the
NumPyro `MCMC` object, was discarded immediately after extracting the flat sample array.
Fixed in `scripts/nb20_nuts_worker.py`: every window now also returns per-parameter
R-hat and effective sample size (`numpyro.diagnostics.summary`, grouped by chain) and a
divergence count (`mcmc.get_extra_fields()["diverging"]`), using the same R-hat
convention already established in `05_mcmc_baseline.ipynb` ("R-hat ~= 1.00" as the
passing bar; here made precise as R-hat $<$ 1.01 for BOTH alpha and beta). **A real,
unexpected finding surfaced while building this fix** (reproduced directly, not
inferred): a single test window at the healthy point (Sc1-like, alpha=beta=1.0) with
the full 4-chain/500-sample/300-warmup protocol gave R-hat(alpha)=4.9-5.2,
R-hat(beta)=8.5-11.1 across 3 different seeds -- NOT the "R-hat ~= 1.00" nb05 reports
for its own (differently-configured, 2-chain) NUTS run. Inspecting the per-chain means
directly explains why: 3 of 4 chains converge to (alpha~=0.978, beta~=0.744) while the
4th converges to a DIFFERENT basin (alpha~=1.021, beta~=1.118) -- this is genuine
posterior **multimodality** in this likelihood at this operating point, not a sampler
bug (switching `chain_method` between vectorized/sequential/parallel made no
difference). Every window below is now pooled ONLY if R-hat $<$ 1.01 for both
parameters; non-converged windows are excluded from the "final" pooled posterior (not
silently averaged in), and the exclusion rate itself is reported per scenario as a
first-class result, not swept under the rug."""
    ),
    code(
        """# Pool raw NUTS samples per scenario, CONVERGENCE-GATED: only windows with
# r_hat < 1.01 for BOTH alpha and beta go into the pooled posterior. Also report the
# UNFILTERED ("all ok windows regardless of convergence") pooled stats alongside, so the
# effect of gating is visible, not hidden.
diag_rows = [{k: v for k, v in r.items() if k != "samples"} for r in nuts_results]
df_nuts_diag = pd.DataFrame(diag_rows)
df_nuts_diag.to_csv(RESULTS / "20_theoryC_nuts_diagnostics.csv", index=False)

nuts_pooled_converged = {}
nuts_pooled_all = {}
convergence_summary_rows = []
for sc in SCENARIOS:
    ok = [r for r in nuts_results if r["scenario"] == sc.name and r["status"] == "ok"
          and r["samples"] is not None]
    converged = [r for r in ok if r.get("converged", False)]
    if ok:
        nuts_pooled_all[sc.name] = np.concatenate([r["samples"] for r in ok], axis=0)
    if converged:
        nuts_pooled_converged[sc.name] = np.concatenate([r["samples"] for r in converged], axis=0)
    r_hats_a = [r["r_hat_alpha"] for r in ok if np.isfinite(r.get("r_hat_alpha", np.nan))]
    r_hats_b = [r["r_hat_beta"] for r in ok if np.isfinite(r.get("r_hat_beta", np.nan))]
    n_div = sum(r.get("n_divergences", 0) for r in ok if r.get("n_divergences", -1) >= 0)
    convergence_summary_rows.append({
        "scenario": sc.name, "n_windows_ok": len(ok), "n_windows_converged": len(converged),
        "frac_converged": len(converged) / len(ok) if ok else np.nan,
        "mean_r_hat_alpha": np.mean(r_hats_a) if r_hats_a else np.nan,
        "mean_r_hat_beta": np.mean(r_hats_b) if r_hats_b else np.nan,
        "max_r_hat_alpha": np.max(r_hats_a) if r_hats_a else np.nan,
        "max_r_hat_beta": np.max(r_hats_b) if r_hats_b else np.nan,
        "total_divergences": n_div,
    })
    print(f"{sc.name:24s}: {len(converged)}/{len(ok)} windows converged (r_hat<1.01 both params), "
          f"mean r_hat a={np.mean(r_hats_a) if r_hats_a else np.nan:.2f} "
          f"b={np.mean(r_hats_b) if r_hats_b else np.nan:.2f}, {n_div} total divergences")

df_convergence_summary = pd.DataFrame(convergence_summary_rows)
df_convergence_summary.to_csv(RESULTS / "20_theoryC_nuts_convergence_summary.csv", index=False)
print("\\nSaved results/20_theoryC_nuts_diagnostics.csv and "
      "results/20_theoryC_nuts_convergence_summary.csv")

theoryC_rows = []
for sc in SCENARIOS:
    snpe_row = get_row(sc.name)
    pooled_c = nuts_pooled_converged.get(sc.name)
    pooled_a = nuts_pooled_all.get(sc.name)

    def _stats(pooled):
        if pooled is not None and len(pooled) > 10:
            return (float(np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1]),
                    pooled[:, 0].mean(), pooled[:, 0].std(),
                    pooled[:, 1].mean(), pooled[:, 1].std(), len(pooled))
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    nuts_corr, nuts_a_mean, nuts_a_std, nuts_b_mean, nuts_b_std, n_pooled = _stats(pooled_c)
    nuts_corr_unfiltered, *_rest_unfiltered, n_pooled_unfiltered = _stats(pooled_a)
    theoryC_rows.append({
        "scenario": sc.name,
        "snpe_corr_mean": snpe_row["corr_mean"],
        "nuts_corr": nuts_corr,  # convergence-gated: the trustworthy one
        "nuts_alpha_mean": nuts_a_mean, "nuts_alpha_std": nuts_a_std,
        "nuts_beta_mean": nuts_b_mean, "nuts_beta_std": nuts_b_std,
        "nuts_n_pooled_samples": n_pooled,
        "nuts_corr_unfiltered_all_windows": nuts_corr_unfiltered,  # includes non-converged -- for contrast only
        "nuts_n_pooled_samples_unfiltered": n_pooled_unfiltered,
    })
df_theoryC = pd.DataFrame(theoryC_rows)
df_theoryC.to_csv(RESULTS / "20_theoryC_snpe_vs_nuts.csv", index=False)
print(df_theoryC.to_string(index=False, float_format="%.3f"))
"""
    ),
    md(
        """## 3b. Final per-scenario SBI-vs-NUTS comparison (candidate SI content)

**Candidate SI content, numbers stable, formatting pending** -- same standard as the
Stage 0 convergence curve: clearly labelled, not yet promoted into `main.tex`/the SI/
`article_outline_CChE.md`, which stays a human decision. This table is the per-scenario
backbone evidence the coordinator asked for: SBI (production-budget, `FINAL_SEED_COUNT`
seeds, Stage 1) vs. NUTS (pooled, `N_WINDOWS_PER_SCENARIO` windows/scenario, the full
Table-VII-style protocol, Theory C) posterior summaries side by side for all 8
scenarios. Both are built on the SAME convergence/cost-check discipline used
throughout this notebook, not on arbitrarily-chosen settings.

**`nuts_corr` is convergence-gated** (R-hat $<$ 1.01 for both alpha and beta, pooling
only the windows that pass -- see Section 3/Theory C's diagnostics markdown cell for
why this gating was added and the multimodality finding that made it necessary).
`nuts_frac_converged`, `nuts_mean_r_hat_alpha/beta`, and the unfiltered
`nuts_corr_unfiltered_all_windows` column are carried into this table specifically so a
low convergence fraction is visible right next to the number it qualifies, not buried
in a separate diagnostics file."""
    ),
    code(
        """comparison_rows = []
for sc in SCENARIOS:
    s1 = get_row(sc.name)
    c = df_theoryC.loc[df_theoryC["scenario"] == sc.name].iloc[0]
    comparison_rows.append({
        "scenario": sc.name, "alpha_true": sc.alpha, "beta_true": sc.beta, "mode": sc.mode,
        "sbi_alpha_mean": s1["sbi_alpha_mean"], "sbi_alpha_std": s1["sbi_alpha_std"],
        "sbi_beta_mean": s1["sbi_beta_mean"], "sbi_beta_std": s1["sbi_beta_std"],
        "sbi_corr": s1["corr_mean"], "sbi_corr_std_across_seeds": s1["corr_std"],
        "nuts_alpha_mean": c["nuts_alpha_mean"], "nuts_alpha_std": c["nuts_alpha_std"],
        "nuts_beta_mean": c["nuts_beta_mean"], "nuts_beta_std": c["nuts_beta_std"],
        "nuts_corr": c["nuts_corr"], "nuts_n_pooled_samples": c["nuts_n_pooled_samples"],
        "nuts_corr_unfiltered_all_windows": c["nuts_corr_unfiltered_all_windows"],
        "nuts_frac_converged": df_convergence_summary.loc[
            df_convergence_summary["scenario"] == sc.name, "frac_converged"].iloc[0],
        "nuts_mean_r_hat_alpha": df_convergence_summary.loc[
            df_convergence_summary["scenario"] == sc.name, "mean_r_hat_alpha"].iloc[0],
        "nuts_mean_r_hat_beta": df_convergence_summary.loc[
            df_convergence_summary["scenario"] == sc.name, "mean_r_hat_beta"].iloc[0],
    })
df_comparison = pd.DataFrame(comparison_rows)
df_comparison.to_csv(RESULTS / "20_final_sbi_vs_nuts_comparison.csv", index=False)
print(df_comparison.to_string(index=False, float_format="%.3f"))
print("\\n*** nuts_corr above is CONVERGENCE-GATED (r_hat<1.01 windows only) -- "
      "compare nuts_frac_converged per scenario before trusting it as ground truth. ***")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x_pos = np.arange(len(SCENARIOS))
labels = [sc.name.replace("_", "\\n") for sc in SCENARIOS]

ax = axes[0]
width = 0.35
ax.bar(x_pos - width/2, df_comparison["sbi_corr"], width, yerr=df_comparison["sbi_corr_std_across_seeds"],
       label="SBI (production budget)", color="tab:blue", capsize=3)
ax.bar(x_pos + width/2, df_comparison["nuts_corr"], width, label="NUTS (pooled)", color="tab:orange", capsize=3)
ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel(r"Corr($\\alpha,\\beta$)")
ax.set_title("SBI vs. NUTS: Corr(alpha,beta) per scenario")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ax.errorbar(df_comparison["sbi_beta_mean"], df_comparison["sbi_alpha_mean"],
            xerr=df_comparison["sbi_beta_std"], yerr=df_comparison["sbi_alpha_std"],
            fmt="o", color="tab:blue", label="SBI", alpha=0.7, capsize=2)
ax.errorbar(df_comparison["nuts_beta_mean"], df_comparison["nuts_alpha_mean"],
            xerr=df_comparison["nuts_beta_std"], yerr=df_comparison["nuts_alpha_std"],
            fmt="s", color="tab:orange", label="NUTS", alpha=0.7, capsize=2)
for _, row in df_comparison.iterrows():
    ax.scatter([row["beta_true"]], [row["alpha_true"]], color="black", marker="x", s=40, zorder=5)
ax.set_xlabel("beta"); ax.set_ylabel("alpha")
ax.set_title("SBI vs. NUTS posterior means +/- std per scenario\\n(black x = true (alpha,beta))")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Candidate SI content, numbers stable, formatting pending -- NOT yet in main.tex/SI",
             fontsize=9, color="dimgrey")
fig.tight_layout()
fig.savefig(FIGS / "20_final_sbi_vs_nuts_comparison.png", dpi=150)
plt.close(fig)
print("\\nSaved results/20_final_sbi_vs_nuts_comparison.csv and figures/20_final_sbi_vs_nuts_comparison.png")
"""
    ),
    md("## 4. Stage 3 -- verdicts"),
    code(
        """print("="*70)
print("THEORY A -- energy-balance compensation ridge")
print("="*70)
print(f"Ridge slope (steady-state Qc level set near Sc4): {ridge_slope:+.3f}")
print(f"Posterior dominant-eigenvector slope at Sc4: {posterior_slope:+.3f}")
print(f"Same sign: {ridge_slope * posterior_slope > 0}")
"""
    ),
    md(
        r"""**Theory A verdict: NOT CONFIRMED / INCONCLUSIVE (as literally tested).** The
candidate "Qc_ss-constant" energy-balance ridge near Sc4 came out numerically flat
(slope $\approx 0$ -- `results/20_theoryA_qc_ss_grid.npz`), while the actual trained
posterior's dominant covariance eigenvector at Sc4 has slope $\approx -7.73$ (eigenvector
$(-0.128, 0.992)$, i.e. dominated by $\beta$ with a steep negative tilt). **These do not
match.** This does not cleanly falsify a physical energy-balance mechanism in general --
the $17\times17$ grid used here may simply be too coarse to resolve a weak local
sensitivity of $Q_{c,ss}$ to $\beta$ at fixed $\alpha$ near 0.85, which is what made the
ridge-slope construction degenerate rather than a confidently-near-zero physical result.
Stated plainly: as specifically implemented, Theory A's test gives no evidence FOR a
physical energy-balance ridge explaining the correlation direction, but the test itself
is weak enough that it should not be read as a confident refutation of the broader idea
either."""
    ),
    code(
        """print("="*70)
print("THEORY B -- open-loop masking confirmation")
print("="*70)
print(df_stage1[df_stage1["scenario"].isin(
    ["Sc0_open_healthy","Sc1_closed_healthy","Sc6_open_with_fault","Sc2_closed_fouling"]
)].to_string(index=False, float_format="%.3f"))
print(f"\\nBoth open-loop |Corr| < matched closed-loop |Corr|: {theoryB_support}")
"""
    ),
    md(
        r"""**Theory B verdict: FALSIFIED (as literally stated).** The naive open-loop-masking
prediction is "open-loop should show near-zero correlation, closed-loop should show the
real one." The actual numbers (`results/20_stage1_per_scenario_corr.csv`): Sc0 (open,
healthy) Corr $=+0.612$ vs. Sc1 (closed, healthy) Corr $=-0.650$; Sc6 (open, fouling)
Corr $=+0.564$ vs. Sc2 (closed, fouling) Corr $=-0.856$. Open-loop scenarios do NOT show
near-zero correlation -- they show SUBSTANTIAL correlation, comparable in magnitude to
their closed-loop counterparts, just consistently OPPOSITE in sign. Integral control is
therefore not creating the correlation from nothing; feedback appears to REORIENT an
already-present coupling rather than manufacture it. This is a more specific, more
interesting finding than the original hypothesis, but it is a falsification of Theory B
as stated, not a confirmation with caveats."""
    ),
    code(
        """print("="*70)
print("THEORY C -- SNPE vs NUTS (full protocol, full-covariance likelihood, convergence-gated)")
print("="*70)
print(df_theoryC.to_string(index=False, float_format="%.3f"))
"""
    ),
    md(
        r"""**Theory C verdict (REWRITTEN 2026-11-07 after the matched-warm-start fix and full
8-scenario corrected sweep -- see Section 5 for the full mechanism investigation):
CONFIRMED for the closed-loop scenarios with usable data -- evidence FOR the
correlation being a real feature of the posterior, reversing this notebook's own
earlier (mismatched-protocol) verdict.**

Once NUTS's likelihood uses the SAME per-theta matched warm start SBI's training uses
(`matched_warm_start=True`, Section 5), the real pooled NUTS posterior agrees in SIGN
with SBI's matched-protocol posterior at every closed-loop scenario that produced
usable data, with substantial (not token) magnitude agreement
(`results/20_final_sbi_vs_nuts_comparison.csv`, all R-hat $\le$1.003, clean
convergence):

| scenario | SBI corr | NUTS corr (pooled) | NUTS/SBI ratio |
|---|---|---|---|
| Sc1 (healthy) | $-0.650$ | $-0.423$ | 0.65 |
| Sc2 (fouling) | $-0.856$ | $-0.488$ | 0.57 |
| Sc3 (decay) | $-0.638$ | $-0.569$ | 0.89 |
| Sc4 (combined) | $-0.893$ | $-0.786$ | 0.88 |

Sc4 -- the scenario this entire investigation exists to resolve -- shows the closest
agreement (NUTS at 88% of SBI's magnitude, same sign), with 13/20 windows (65%)
converging cleanly. This directly contradicts this notebook's own earlier verdict
(built on the mismatched protocol, which found small/opposite-signed NUTS correlations)
and resolves the Theory C vs. Theory E tension this notebook originally raised: NUTS
(the actual posterior, once correctly specified) and the full-covariance local-FIM
(Theory E) now POINT THE SAME WAY -- toward the correlation being real.

**What remains genuinely unresolved, stated explicitly, not smoothed over:**
- **Sc7 (drift) shows a SIGN DISAGREEMENT**: SBI $+0.632$ vs. NUTS pooled $-0.666$
  (11/20 windows converged, clean R-hat). This is the one scenario where the two
  methods actively disagree even under the corrected protocol -- not explained here,
  flagged as a genuine open discrepancy, not rounded into the "they now agree" story.
- **Sc0/Sc6 (open-loop): 0/20 windows converged, both runs.** Root cause identified
  (Section 5): `cstr_generative_model` has no open-loop variant, so NUTS is fitting a
  structurally wrong model to this data -- a genuine, separate model mismatch. No NUTS
  ground truth exists for these two scenarios.
- **Sc5 (severe saturation): 0/20 windows converged**, despite being closed-loop and
  otherwise like Sc1-4/Sc7. A third, distinct, unresolved mechanism (Section 5),
  plausibly actuator-saturation-induced likelihood irregularity breaking NUTS's
  gradient-based sampling specifically.

Net: for the 4 scenarios with clean bilateral data (Sc1-4), the evidence now supports
the correlation being real. Sc7's sign disagreement, and the complete absence of NUTS
ground truth for Sc0/Sc5/Sc6, mean this is NOT a clean, unqualified confirmation across
the whole scenario grid -- three of eight scenarios still have no resolution at all,
and one of the four with data disagrees in sign."""
    ),
    code(
        """print("="*70)
print("THEORY D -- protocol/warm-start artifact")
print("="*70)
print(df_theoryD.to_string(index=False, float_format="%.3f"))
"""
    ),
    md(
        r"""**Theory D verdict (REWRITTEN 2026-11-07, reversing the immediately-prior interim
reading): CONFIRMED as the correct explanation, in the OPPOSITE direction from the
reading written earlier today.** Earlier in this notebook's own history, Theory D's
numbers were read as "the old fixed-warm-start protocol's near-zero correlations sit
closer to a NUTS-based ground truth, so the matched protocol might be the artifact."
Section 5's decisive test and the full corrected sweep directly falsify that reading:
once NUTS ALSO uses the matched protocol, NUTS and SBI agree (Theory C, above) --
meaning the matched-warm-start protocol was never the source of a spurious
correlation. It is the correct, physically-appropriate protocol; the OLD fixed-warm-start
protocol's near-zero numbers were themselves the artifact (the same conclusion this
project already reached once before, for SBI's beta bias, now confirmed to extend to
NUTS as well). Theory D, as a mechanism investigation, is fully resolved: the protocol
mismatch it investigates is real, its direction is now known, and the fix has been
applied and validated end-to-end."""
    ),
    code(
        """print("="*70)
print("THEORY E -- sensor-noise cross-correlation (full vs. diagonal summary covariance)")
print("="*70)
print(df_theoryE.to_string(index=False, float_format="%.3f"))
"""
    ),
    md(
        r"""**Theory E verdict: PARTIALLY SUPPORTED, but more narrowly than the initial reading --
this is the center of the Theory C/E tension addressed in the final synthesis below.**
Switching from the diagonal to the full empirical summary-statistic covariance in the
local-FIM calculation recovers most of the sign and much of the magnitude of the
MATCHED-PROTOCOL SBI posterior's correlation at the 5 non-drift closed-loop scenarios
(`results/20_theoryE_fullcov_fim.csv`: e.g. Sc4 diag $-0.048$ -> fullcov $-0.550$ vs.
SBI $-0.893$; Sc2 diag $-0.127$ -> fullcov $-0.748$ vs. SBI $-0.856$; Sc5 diag $+0.125$
-> fullcov $+0.753$ vs. SBI $+0.850$) -- confirming the diagonal-covariance assumption
IS a real source of error in the local-FIM diagnostic, closing most of nb19's "local-FIM
never predicted this" gap. **However, Theory C now shows the actual GLOBAL (NUTS)
posterior at these same points has small correlation, not large** -- so Theory E's
"recovery" was recovering agreement with SBI's (now independently suspect) large
correlation, not with the true posterior. The diagonal-vs-full-covariance distinction is
real and confirmed; whether the full-covariance LOCAL/linearised correction is itself
revealing a genuine local feature that the GLOBAL nonlinear posterior washes out, or
whether it simply shares the same upstream bias as SBI's training (e.g. inheriting the
matched-warm-start protocol's own distortion, since both are evaluated at/near the same
scenario points), is NOT distinguished by the evidence in this notebook."""
    ),
    code(
        """print("="*70)
print("THEORY F -- summary-statistic compression (CNN raw-trajectory vs. hand-crafted)")
print("="*70)
print(df_theoryF.to_string(index=False, float_format="%.3f"))
"""
    ),
    md(
        r"""**Theory F verdict: PARTIALLY FALSIFIED as literally stated, with a more informative
reframing once combined with Theory C/D.** The literal claim -- compression toward
summary statistics creates a spurious correlation that shrinks under raw-trajectory
access -- predicts the CNN-embedding posterior's correlation should shrink toward zero.
Instead (`results/20_theoryF_cnn_comparison.csv`) it stays LARGE and FLIPS SIGN at 5 of
6 scenarios (Sc1 $-0.650\to+0.777$; Sc2 $-0.856\to+0.920$; Sc3 $-0.638\to+0.740$; Sc4
$-0.893\to+0.868$; Sc7 $+0.632\to+0.886$), and only shrinks in magnitude without flipping
at the 6th (Sc5 $+0.850\to+0.314$). Neither representation's SBI posterior agrees with
Theory C's small-correlation NUTS ground truth at these same points. The more
informative reading: BOTH the hand-crafted-summary and the CNN-raw-trajectory SBI
posteriors disagree substantially with the converged NUTS ground truth, in different
(often opposite) directions -- pointing toward the common factor being upstream of
representation choice, most plausibly the amortised SNPE training protocol itself
(Theory D) rather than which features the training data is compressed into."""
    ),
    md(
        r"""**Final synthesis (REWRITTEN 2026-11-07, after Section 5's mechanism investigation and
the full corrected Theory C sweep -- supersedes this cell's earlier text, which was
written before that work existed and is now stale).**

**Per-theory calls, stated plainly, final:**
- **Theory A** (energy-balance ridge): NOT CONFIRMED / INCONCLUSIVE -- unchanged; the
  specific test implemented is too weak (coarse grid) to count as a confident
  refutation either way.
- **Theory B** (open-loop masking): FALSIFIED as literally stated -- unchanged;
  open-loop shows substantial, oppositely-signed correlation, not absent correlation.
- **Theory C** (SNPE vs. NUTS): CONFIRMED for Sc1-4 (clean bilateral agreement in sign,
  57-89% magnitude ratio) -- evidence FOR the correlation being real, reversing this
  notebook's earlier verdict once the warm-start protocol mismatch (Theory D) was fixed.
  Sc7 disagrees in sign (unresolved). Sc0/Sc5/Sc6 have no NUTS ground truth at all
  (two distinct, separately root-caused mechanisms: open-loop model mismatch for
  Sc0/Sc6; an unresolved saturation-linked NUTS failure for Sc5).
- **Theory D** (warm-start protocol): CONFIRMED, fully resolved -- the matched protocol
  is correct; the mismatch between NUTS's old fixed-warm-start likelihood and SBI's
  matched-protocol training data explained most of the original SNPE-vs-NUTS
  disagreement. Fixed at the source (`cstr_generative_model`/`run_mcmc_baseline`, now
  additive and backward-compatible).
- **Theory E** (diagonal vs. full summary covariance): CONFIRMED, and now corroborated
  rather than merely "narrowly supported" -- the full-covariance local-FIM correction
  and the corrected NUTS ground truth now agree (both point toward the correlation being
  real at Sc1-4), resolving the earlier Theory C/E tension this notebook raised. The
  diagonal-covariance local-FIM assumption remains confirmed wrong.
- **Theory F** (summary-statistic compression): PARTIALLY FALSIFIED -- unchanged;
  raw-trajectory access does not shrink the correlation, it mostly flips sign,
  implicating the training protocol (now understood, via Theory D) over the
  representation.

**The Theory C vs. Theory E tension, as originally raised, is RESOLVED for Sc1-4**:
both independent lines of evidence (the corrected NUTS ground truth, and the
full-covariance local-FIM diagnostic) now point the same way. The three candidate
reconciliations originally listed are superseded by a concrete answer: reading (2) was
correct -- the apparent disagreement was NUTS evaluating a different, timing-mismatched
generative model, not a flaw in SBI's training or in Theory E's local analysis.

**Is this publication-ready, or still open?** Materially more settled than before
Section 5, but NOT fully closed. For Sc1-4 specifically, the (alpha,beta) correlation
now has independent, bilateral, clean-R-hat support from both SBI and a correctly-
specified NUTS ground truth -- a substantially stronger evidentiary position than
nb19's original SBC-robustness argument alone. However: (a) Sc7's sign disagreement is
a real, unexplained exception within the very data this section collected, not a
pre-existing caveat being repeated; (b) Sc0/Sc5/Sc6 (3 of 8 scenarios) have NO NUTS
ground truth at all, for two distinct, separately-diagnosed-but-unresolved reasons; (c)
the SBI training-budget non-convergence flagged in `TODO.md` (n=10,000 and n=40,000 both
unconverged) is untouched by any of today's work and remains a precondition issue. A
human promotion decision should treat Sc1-4's finding as a genuinely strengthened,
much-more-defensible result, while treating Sc0/Sc5-7 and the budget question as
explicit, named, still-open items -- not as a uniformly resolved whole-notebook
conclusion."""
    ),
]


def build_and_execute() -> int:
    nb = new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }

    out_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "system_I"
        / "20_alpha_beta_correlation_mechanism_theories.ipynb"
    )

    client = NotebookClient(nb, timeout=None, kernel_name="python3")
    print(f"Executing notebook -> {out_path}")
    client.execute()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    print(f"Wrote {out_path}")
    return 0


def main() -> int:
    return build_and_execute()


if __name__ == "__main__":
    raise SystemExit(main())
