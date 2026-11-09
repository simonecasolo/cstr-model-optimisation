# Handoff — 2026-11-08 (session end): System I controller-form fix, warm-start study, nb04 retrain

Read this file first, then `docs/Manuscript/system_i_k0_fix_and_correlation_plan.md` (§0 REVISION is current),
then the top entries of `HANDOFF.md`. Everything below is stated as verified or NOT verified.

## 0. Running processes (checked at handoff time)

**Nothing from this session is running.** No training, no notebook execution, no monitors.
- nb04 50k training (was PID 41716/41718): **killed on purpose** (invalidated by the controller fix). No pickle was saved.
- Background tasks that already finished: `bbdjzrppc` (nb15 attempt, failed on path), `bs13sxb9a` (E1b LNA validation, done),
  `batgbvw6y` (nb04 monitor, ended when I killed the run), `b4j9xdox0` (E1b monitor, done).
- Leftover sentinel: `results/04_production_n50000_seed0_progress_start.txt` (harmless; delete it).
- Processes that ARE running but are NOT mine (VS Code / pre-existing Jupyter kernels — leave alone):
  ipykernel launchers PIDs **24156**, **87653**, **46532**; VS Code language servers 1548, 1844, 25971, 88232, 13233, 13248.
- Caution for whoever launches long jobs: `train_sbi_posterior` prints a tqdm bar to stdout; **never `tail` that log raw**
  (it floods context). Watch the sentinel files instead (see `scripts/nb04_production_50k.py`).

## 1. Decisions made by the user this session (do not relitigate)

1. **k0**: keep `K0_NOMINAL=16.96e12`, disclose in Table I as 60x Fogler (1.018e15 h^-1 equiv), a modification, ~98% conversion
   design; check the acid-catalysis rationale against Furusawa/Fogler first. Details: plan §0.1-0.4.
2. **Controller form: FIX IT** (decided on physical grounds, not by outcome). Coded law was
   `Qc = Qc0 + Kp*(T-Tsp) + I/tau_i` with `dI = T-Tsp` -> integral gain 1/tau_i = 0.1, not Kp/tau_i = 15 (150x too weak),
   dimensionally inconsistent, slow mode 1.6e3-5.5e3 min. Standard form now: `Qc = Qc0 + Kp*[(T-Tsp) + I/tau_i]`, I in K*min.
3. Keep Kp=150, tau_i=10 (counterfactual shows they are fine, see §3). Only the form of Eq. (3) changes in the manuscript.

## 2. Code state

### Done
- `src/cstr_sbi/physics.py` **edited** (uncommitted): `compute_qc`, `cstr_closed_loop_rhs` (both now
  `Qc0 + Kp*((T-Tsp)+I/tau_i)`), `closed_loop_steady_state_exact` (`I_ns = tau_i*(Qc_ns-Qc0)/Kp`,
  `I_star = tau_i*((Qc_fixed-Qc0+sat_sign*eps)/Kp - (T_star-Tsp))`), header comment + docstring.
  Pre-fix copy: `results/_invalidated_controller_bug/physics_pre_controller_fix.py`.
- `warm_start_ic` -> `closed_loop_steady_state_exact`, so SBI training, NUTS (`cstr_generative_model`,
  `run_mcmc_baseline`), `scenarios.py`, `simulator.py` all pick the fix up automatically through `physics.py`.

### **NOT verified — do this first**
- The fixed `physics.py` has **never been run**: not the steady-state residual (`|rhs|` at `closed_loop_steady_state_exact`,
  esp. the saturated branch / `_qc_eps` push), not `jax.grad` through it (nb23 NUTS path), not the slowest eigenvalue.
- `tests/test_controller_form.py` exists (2.7 KB, written, **never executed**). It tests dQc/dI = Kp/tau_i, dQc/dT = Kp,
  steady state is a fixed point incl. the integrator, slowest mode < 60 min (and ~tau_i at healthy). Run
  `python -m pytest tests/test_controller_form.py -q`, fix test bugs (first test has an awkward `0.0*1.0+1.0` argument; `grad` at
  I=1.0, T=Tsp is intended), then confirm it FAILS against the pre-fix copy (swap the file in temporarily) so the guard really guards.

### Still to change ("fix everywhere the controller appears" — user's list)
- **EKF/UKF** incl. the hand-derived 6x6 Jacobian: lives in notebooks (`notebooks/system_I/16_ekf_ukf_baseline.ipynb`, also
  referenced in 19, 20, 14). Not located/edited yet. The Jacobian's controller terms and the I-state scaling must change.
- **30-day tracking simulator** (nb10 / `generate_degradation_stream` in `scenarios.py`): check it only goes through
  physics.py; not verified.
- **NUTS workers**: `scripts/nb20_nuts_worker.py`, `scripts/nb21_nuts_worker.py`, `scripts/recompute_theoryC_section5.py`,
  `scripts/nb22_cnn_theoryF_recheck.py` reference the controller — grep them for hard-coded `I/tau_i`.
- `NOMINAL_Y0_CL = [0.0184, 312.15, 299.05, 0.0]` (physics.py ~l.120) is a hand-set approximate IC with I=0; used when
  `matched_warm_start=False`. Under the new law I=0 is a different Qc; flag or recompute.
- Notebooks that mention Kp/tau_i in text or code (grep result): 00,01,01a,02,03,04,04b,05,05a,06,07,08,11,12,14,15,16,19,20,23,25,26.
- Manuscript: Eq. (3) (`main.tex` ~l.320), Table I note, and any prose about "integral action masks the fault over days".
- `docs/Manuscript` plan items in §0.3 of the k0 plan (CNN <1% claim, beta-bias table, full-cov FIM ratio, ridge = physics + noise).

### Everything cached is invalid
All `results/*.pkl` posteriors, `data/observations.npz`, nb04-nb26 numbers, the manuscript's 0.927 etc. were produced under the
weak-integral law (and several also pre-date the 2026-11-07 warm-start fix). Treat nothing as current until regenerated.

## 3. Counterfactual results (`scripts/nb30_controller_counterfactual.py`, results `30_counterfactual_*.csv`; standard form)

- Time constants: slowest **10.0-26.4 min** (coded: 1,650-5,500 min); fastest 0.1-0.6 min.
- Damping: real poles at most points; worst = prior corner (0.4,1.2) zeta 0.45, period 7.8 min; (1.2,1.2) zeta 0.70. All stable.
- Saturation / anti-windup (conditional integration, `dI = gate*(T-Tsp)`): sensible, no NaNs. Sc5 at Qmax 95% of window
  (74% after an abrupt fault from healthy, recovers); Sc1/Sc4 <=1% at rails.
- Noise: process-noise-only (the controller reads true T; sensor noise never enters the loop in this model). sigma(T)=0.11-0.25 K,
  sigma(Qc)=17-57 L/min (set by Kp; unchanged by the fix).
- **Pre-existing finding: Sc2 (beta=0.7) also saturates at Qmax** (72% of window), not just Sc5. Manuscript may describe Sc2 as the unsaturated fault.
- Confirmation of the bug: `scripts/nb30_pi_form_eigcheck.py` (coded vs standard slowest eigenvalue, 5 points).

## 4. Warm-start study (nb29 — notebook NOT yet built; scripts exist)

User criterion to write in cell 1 BEFORE results: "optimal" = matches the situation the monitor faces and degrades gracefully
under mismatch, NOT best-looking accuracy (that is circular). Choose production protocol by the **P2 column** (realistic
incipient fault), robustness across other columns as tiebreaker, fixed in advance.

Protocols: P0 fixed healthy; P1 deterministic matched ss; P1b matched + stationary noise; P2 theta ramps at 30-day degradation
rates; P3 randomised theta' (sigma 0.03). Experiments E1 (time scales), E2 (full-cov raw-trajectory FIM, split level/transient/noise),
E3 (train x test protocol matrix, 5k sims, 3 seeds, equal budget; metrics bias, 90% coverage, SBC, Corr(a,b), macro-F1),
E4 (30/60/120-min windows), E5 optional (MMD misspecification test).

Redesign for 10-26 min settling (user): **P1b burn-in ~100-150 min**; **P2 history a few hundred min** at 30-day rates; P3 sigma 0.03.

Code: `scripts/ws_protocols.py` — vectorised jit+vmap simulator for all protocols (about 10x faster than the production loop;
60 sims < 1 s after compile). Validated vs production `simulation_wrapper_sbi` under P1: 27/29 summary features match
(the other two are saturation-fraction features ~0 at (0.85,0.85)). **Needs updating**: defaults are still the slow-mode ones
(`hist_min=8000`, P1b implemented as an LNA/Lyapunov stationary draw instead of burn-in). Change `hist_min` to ~300-400,
and either switch P1b to an explicit 100-150 min burn-in (user spec) or keep LNA after re-validating
(`scripts/nb29_e1b_lna_validation.py`; LNA matched a 16,000-min EM run within ~5% under the OLD law only). It imports physics
read-only, so it now runs under the FIXED law.
`scripts/nb29_e1_timescales.py` and `nb29_e1b_*.py` were run under the OLD law: results `29_e1_*.csv`, `29_e1b_*.csv` are
**superseded** (kept as evidence of the artefact); re-run them under the fix. Also: a rate-based P2 check of the plant lag
(old law: Qc lag up to 17 L/min) should now be ~150x smaller.
Naming note: the plan's "nb27/nb28" numbering clashes; scripts are `nb27_step0_*` (k0 operating-point scan), the warm-start study is `nb29`, controller is `nb30`.

## 5. Re-run order requested by the user (after the fix is in everywhere + tests pass)

1. warm-start notebook (nb29): E1, E2, E3 screen -> choose protocol by the pre-set criterion.
2. production training, budget chosen by checking that Corr(alpha,beta) has converged (use nb22's metric suite:
   `scripts/build_nb_system_I_22.py` functions `run_sbc_for_posterior`, `grid_diagnostics`, `corr_ab`, `obs_x_for_scenario`).
3. SBC on that exact posterior. 4. classification, threshold settled first. 5. NUTS (nb23 R-hat~500 regression unresolved).
6. EKF/UKF. 7. tracking. 8. alpha-beta decomposition and the **raw-trajectory Fisher check** (outline §8.1).

nb04 production script already exists: `scripts/nb04_production_50k.py` (trains NSF 128/5, matched start, evaluates with nb22's functions,
saves `results/04_production_n{N}_seed{S}.pkl` + `_metrics.json`, does NOT overwrite `sbi_posterior_final.pkl`). It needs the
protocol made a parameter once the study picks one. `notebooks/system_I/04_sbi_training.ipynb` has an appended **section 10** (cells 25-27)
that loads that JSON and compares to nb22's ladder — unexecuted, will break until the JSON exists. nb22 reference at the OLD law:
50k seeds had KS p 0.008/0.002 (seed 0), Corr@Sc4 -0.91; seed choice matters (SBC-best != classification-best).

## 6. Open question that is the user's, NOT mine — System II / Luyben have the same controller form

`src/cstr_sbi/recycle/physics.py` (`compute_qj`: `Qj0 + Kp1*(T_r-T_sp) + I_T/tau_i1`, `dI = e`; also loops 2/3 via
`compute_reflux_boilup`) and `src/cstr_sbi/luyben/physics.py` (`_pi_output`: `bias + Kp*(pv-sp) + I/tau_i`) use the identical
malformed form. System II is the paper's headline. **Not touched, not analysed.** The user must decide whether to fix those too;
if yes, the same counterfactual (eigenvalues/damping/saturation) should be run first, and it would invalidate the System II
calibration work recorded in memory (S-A/S-B/S-C SBC findings, banana results).

## 7. Other things learned (so they are not rediscovered)

- `notebooks/system_I/15_beta_bias_analysis.ipynb` builds `FIGS = Path.cwd().parent/'figures'` and loads
  `results/sbi_posterior_final.pkl` relative to cwd.parent -> it must be executed with cwd one level above `notebooks/` content
  (copy into a temp subdir of `notebooks/` + symlink `figures -> ../../figures`, `results` likewise). nbconvert always uses the
  notebook's own dir as cwd. Many nb01-18 share this layout; nb15's Step 5 depends on the stale production posterior.
- Step 0 result (k0): with `k0/60` no Tsp works across Sc0-Sc7 (`scripts/nb27_step0_*`, `results/27_step0_catalogue_scan.csv`). Moot given the keep-k0 decision.
- The OS `mach port` exhaustion crash in the long per-sample `simulate_for_sbi` loop (nb24 H3) is unresolved; the vectorised
  `ws_protocols` simulator avoids that loop entirely.
- Memory entries added: see `~/.claude/projects/.../memory/` (keep-k0-disclose, controller-form-bug).

## 8. Files created/changed this session

New: `scripts/{nb04_production_50k,ws_protocols,nb27_step0_catalogue_scan,nb27_step0_window_and_knobs,nb29_e1_timescales,nb29_e1b_lna_validation,nb30_controller_counterfactual,nb30_pi_form_eigcheck}.py`,
`tests/test_controller_form.py`, `results/{27_*,29_*,30_*}.csv`, `figures/29_e1_burnin.png`, `results/_invalidated_controller_bug/`.
Changed: `src/cstr_sbi/physics.py` (controller fix), `docs/Manuscript/system_i_k0_fix_and_correlation_plan.md` (§0 REVISION),
`HANDOFF.md` (STEP-0 entry), `notebooks/system_I/04_sbi_training.ipynb` (appended §10).
Not touched: `main.tex`, SI, `article_outline_CChE.md`, System II code, any cached result.
