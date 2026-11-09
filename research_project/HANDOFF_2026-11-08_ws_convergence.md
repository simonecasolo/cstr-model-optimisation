# Handoff — 2026-11-08: warm-start study (nb29) and budget convergence (nb31), ready to run

Written by Claude (cloud session); **the code has not been executed** — the cloud session has
no JAX/torch/sbi. The pure-numpy metric code and both decision rules were tested on
synthetic data (`tests/test_ws_metrics.py` passes; the `select` and `decide` rules give the
intended answers on constructed inputs). Expect small first-run fixes in the JAX/sbi glue;
fix them in place and note each fix at the bottom of this file.

Read `docs/Manuscript/system_i_rerun_plan.md` for the whole plan. Phase 0 fixes F1–F5 are
independent of this code, but **F4 (Sc2 saturation decision) must be settled before the
nb29 test sets are final**, because the scenario grid is part of the E3 test set. If F4 is
still open, run nb29 anyway (the decision rule uses the P2 column's SBC/TARP/RMSE, which do
not depend on the scenario grid) and re-run only `e3sim`'s test sets + `e3eval` after F4.

## New files

| File | What |
|---|---|
| `scripts/ws_metrics.py` | pure-numpy metrics: SBC ranks/KS, coverage, rank shape, TARP joint coverage, RMSE/bias, per-obs Corr(α,β), quadrant classification, macro-F1 |
| `scripts/ws_eval.py` | simulation (`ws_protocols`), NPE training from arrays (nb04 NSF recipe), prior-truncated batched sampling with leakage accounting, `EnsemblePosterior`, full evaluation |
| `scripts/nb29_ws_study.py` | E1–E4 + `select`, resumable stages, outputs `results/29fx_*` |
| `scripts/build_nb_system_I_29.py` | builds/executes `notebooks/system_I/29_warm_start_protocol_study.ipynb` (runs every stage) |
| `scripts/build_nb_system_I_31.py` | builds/executes `notebooks/system_I/31_budget_convergence.ipynb` (runs every stage) |
| `notebooks/system_I/29_*.ipynb`, `31_*.ipynb` | the two notebooks, built, not yet executed |
| `scripts/nb31_budget_convergence.py` | bank/test/train/eval/decide, outputs `results/31_*`; `load_production_posterior()` |
| `tests/test_ws_metrics.py` | unit tests for `ws_metrics` |
| `docs/Manuscript/system_i_rerun_plan.md` | full re-run plan + claims ledger |

## Run order — the notebooks are the experiments

**All work runs in notebooks stored in and executed from `notebooks/system_I/`.** The
notebooks call the stage functions in `scripts/nb29_ws_study.py` and
`scripts/nb31_budget_convergence.py` (one implementation, imported); the first code cell of
each notebook asserts that its working directory is `notebooks/system_I`. Each stage skips
work already on disk, so re-executing a notebook after an interruption resumes it.

```bash
cd /Users/simo/inso-po-RD/cstr-model-optimisation/research_project
python -m pytest tests/test_controller_form.py tests/test_ws_metrics.py -q

# 29 — warm-start protocol: open notebooks/system_I/29_warm_start_protocol_study.ipynb in
# Jupyter / VS Code (kernel cwd = notebooks/system_I) and run all, or headless:
python scripts/build_nb_system_I_29.py --execute     # rebuilds the .ipynb, executes it in notebooks/system_I
#   switches in its first code cell: PARALLEL_TRAINING (default False), RUN_E4 (default False)
#   STOP if the selection cell prints FLAG

# 31 — budget convergence (needs results/29fx_selection.json, unflagged):
python scripts/build_nb_system_I_31.py --execute
#   switches: BUDGETS (default 5k-40k), PARALLEL_TRAINING (default True, 3 seed groups), THREADS
```

Note: `--execute` rebuilds the notebook from its builder before executing, so edit the
builder (not the .ipynb) if a cell needs changing, as for the other `build_nb_system_I_*.py`.
If headless nbclient execution stalls (seen before in this project), open the same notebook
in Jupyter and run it there: the stages resume from disk.

Progress: watch `results/29fx_progress_*.txt` and `results/31_progress_*.txt` (never `tail`
a tqdm log). Every stage skips work already on disk, so a killed stage can simply be re-run.

## Checks after each stage (report these numbers back)

- **e1**: `slowest_tau_min` 10–26 at unsaturated points; `t95_stationary_min` well under the
  150-min P1b burn-in (if not, P1b's burn-in is too short: raise `burn_min` in `ws_protocols`
  before e3sim); `t_settle_abrupt_min` finite; `lag_Qc_x1` ≲ 3 L/min.
- **e2**: `corr_implied` by protocol × representation; whether raw-trajectory and summary
  correlations agree (the outline §8.1 check, now for System I).
- **e3eval**: no `MISSING` posteriors; `leak_sbc_mean` small (< 0.05); the train-P0 → test-P1
  cell should show the old β-bias mechanism.
- **select**: the printed table, the log lines and the choice.
- **nb31 decide**: the step table and the production budget; whether the ensemble is used.

## Known risks in the untested glue (look here first if something fails)

1. `ws_eval._draw_raw`: `DirectPosterior.sample_batched(..., reject_outside_prior=False)` is
   assumed to return shape `(n, B, 2)` (true for sbi 0.26.1 per its source). If it differs,
   fix the transpose there only.
2. `ws_eval.train_npe`: uses `sbi.inference.NPE` and `sbi.neural_nets.posterior_nn(model="nsf",
   z_score_x="independent", hidden_features=128, num_transforms=5)`; `inf._summary` keys
   `epochs_trained` / `best_validation_loss` are read defensively.
3. `nb29_ws_study.stage_e1`: calls `ws._em_ramp(..., sigma=jnp.zeros(4))` for noise-free runs
   and `ws._jac_batch` / `ws._ss_batch` with length-1 arrays.
4. Memory: e2 holds 5 × 1500 × 120 × 4 floats per (protocol, point) — small.

## nb29 outcome (2026-11-08, reviewed by Claude)

- Choice **P1b**, selection **flagged** (full-set α SBC KS fails for every protocol incl.
  matched pairs at 5k; not protocol-related). Conclusions and a **user sign-off cell** were
  added to notebook 29 (outputs of the executed cells preserved). nb31 refuses to start until
  that cell has written `results/29fx_selection_review.json`.
- E1's `t_settle_abrupt_min = inf` (Sc4, Sc7) and the `lag_Qc_*` offsets are float32 stalling
  artefacts of the noise-free diagnostic runs, not physics; do not cite them.
- Scenario issues for the user (re-run plan F4): Sc5 β on the prior edge; Sc2 saturated.

## Changes after nb29 (Claude, 2026-11-08)

- `ws_metrics`: interior-subset SBC KS (`ks_p_*_interior`, `n_interior`), per-scenario posterior sd.
- `ws_eval.evaluate`: flattens `sda_Sc*` / `sdb_Sc*`.
- `nb31_budget_convergence.py`: 160k bank (10k chunks), budgets 5k–160k, gate (d) on the
  interior subset (amendment documented in the docstring), review-file override for a
  flagged nb29 selection.
- `build_nb_system_I_31.py`: section 0 (scenario catalogue check), α-calibration and
  sd/CRB efficiency plot.
- nb29 E3 results are unaffected by these changes (no re-evaluation needed).

## Next: notebook 03 section 10 (summary statistics), BEFORE nb31

```bash
python -m pytest tests/test_summary_variants.py tests/test_ws_metrics.py -q
python scripts/append_nb03_section10.py --execute   # runs only section 10 of notebook 03,
                                                    # kernel cwd = notebooks/system_I (~30-45 min)
```
Or open notebook 03 in Jupyter and run section 10's cells (from its setup cell down).
Report back: the CRB table (10.2), the metrics/efficiency tables (10.4) and the decision (10.5).
New files: `scripts/summary_variants.py`, `scripts/nb03_summary_study.py`,
`scripts/append_nb03_section10.py`, `tests/test_summary_variants.py`.

## Scenario catalogue and data regeneration (2026-11-08)

- `src/cstr_sbi/scenarios.py`: Sc2 β 0.80, Sc5 β 0.50, Sc6 β 0.80 (user decisions; comments in the file).
- `scripts/ws_eval.py`: new `simulate_open_loop` (vectorised, 300-min noisy burn-in at θ), shared
  by notebook 02 (Sc0/Sc6 data) and notebook 07 (open-loop posterior training).
- **Notebook 02 is ready to run** (from `notebooks/system_I`): P1b closed loop, burn-in open loop,
  root-relative paths, metadata keys; §4b checks (Sc5 saturates; Sc2 saturation fraction reported,
  warning above 5%; bit-identical to the training simulator; open-loop burn-in converged).
- Notebook 07 updated (run after nb02 and nb31). Notebook 21 and the tracking stream (nb10): changes
  listed in the re-run plan, to be made with the NUTS and tracking re-runs.
- Order now: notebook 02 → notebook 03 §10 → nb31 → Phase 3.
- Update (18:55): notebook 03 §10 done, decision **keep the 29-D summaries**. Sc2/Sc6 moved to
  β = 0.85 (midpoint rule; 0.80 saturated 9.1% of samples). **Re-run notebook 02**, then nb31.
  nb31 now also shows per-scenario classification accuracy and β coverage.

## nb31 outcome and production posterior (2026-11-08, 20:55)

- `results/31_production.json`: **160k, 5-seed ensemble, P1b**, written by
  `stage_decide_amended(approve=True)` with the user's approval; the original rule (v1, 20k) is
  kept inside it as `rule_v1`. Notebook 31 has a new section 4b (re-running it rewrites the same record).
- **Next notebook to run: 07** (open vs closed loop). It is the only Phase-3 notebook already
  updated. The others (00, 01, 03 §1–9, 04, 05a, 06, 09, 10, 11, 16, ...) still use
  `Path.cwd().parent` paths, `sbi_posterior_final.pkl` and private simulators, and are being
  updated next, in the re-run-plan order.

## Fixes applied on first run

(append here)

## 2026-11-09 — production switched to the N3 sensor model; N3 re-run ready

* nb33 reviewed (interpretation written in its section 6). The user adopted N3 (σ = 0.01 mol/L,
  0.2 K, 0.2 K, 2 L/min) as production.
* New: `scripts/system_I_config.py` (sensor-model switch, default N3), `scripts/smoke_N3.py`,
  `scripts/patch_notebooks_N3.py` (applied: notebooks 02 and 07 now N3, executed N0 copies archived
  as `02_data_generation_N0.ipynb` and `07_failure_baseline_open_vs_closed_N0.ipynb`, the executed
  N0 notebooks 29, 31 and 03 §10 pinned to N0 with outputs kept).
* Parametrised by sensor model: `nb29_ws_study.py`, `nb31_budget_convergence.py`,
  `nb03_summary_study.py`, `ws_eval.simulate_open_loop` (`noise_abs`); builders take `--sensor N3`
  (`build_nb_system_I_29.py`, `build_nb_system_I_31.py`, `append_nb03_section10.py` → section 10b).
* **Run sequence and per-step checks: `docs/Manuscript/system_i_N3_runbook.md`** (02 → 29_N3 →
  03 §10b → 31_N3 → 07; ~5–6 h total, caffeinated, resumable).


---

## 2026-11-09 N3 production run — STOPPED before step 1 (agent run, started 08:10)

**What ran:** step 0 only. pytest (test_ws_metrics, test_summary_variants): 9 passed. `scripts/smoke_N3.py`: failed
at its closed-loop Qc noise check (7.98 vs 2.0 L/min, bound ratio < 3). **Steps 1-6 were not run** (stop rule).

**Why it failed, and why I did not touch it:** the check estimates sensor noise from first differences and assumes
process noise is negligible on Qc; in closed loop it is not (Kp = 150 makes Qc jump with every process-noise kick on
T: 7.73 L/min first-difference sd with the sensor noise switched off; the N0 run already showed 8.33x). A direct
same-seed measurement (N3 minus sensor-free run, `logs/N3_step0_sensor_noise_diag.py`) gives injected sensor noise
[0.0099, 0.2004, 0.2012, 2.0052] in BOTH the closed-loop P1b and the open-loop simulator vs expected
[0.01, 0.2, 0.2, 2.0]: the N3 sensor plumbing is correct; the test is mis-specified.

**Needs your decision:** (1) relax/replace the closed-loop Qc assertion in `smoke_N3.py` (my suggestion: assert the
same-seed difference as in the diagnostic) or tell me to proceed without the smoke gate; (2) confirm the nb29-N3
pre-approval wording ("CONFIRMED ... [or: NOT confirmed ...]" was left with both options). Nothing else was changed;
no N0 files touched; no notebooks executed; `logs/N3_step0_smoke.log` holds the failure.

Key numbers per step: none yet (no step 1-6 results exist).

*Update 2026-11-09 08:15:* re-ran step 0 after the user reported the smoke test was replaced; `scripts/smoke_N3.py` in this working tree
is still the old version (same failure, mtime 08:14:44, 55 lines). Waiting for the corrected file; steps 1-6 still not run. See runbook section 6.

---

## 2026-11-09 N3 production run — steps 0-3 done, STOPPED at nb29's review (agent run; 08:10-08:49)

**What ran:** step 0 (pytest 9 passed; smoke test passed on the corrected version after two failures on the old version, cause recorded in the runbook), step 1 (nb33 section 7, exit 0, 4 min), step 2 (nb02 N3 data, exit 0, 25 s, all §4b
asserts passed), step 3 (nb29 under N3, 24 min, **exit 1 = designed STOP**). Steps 4-6 not run (dependency + stop rule). No N0 files touched; no code, threshold or rule edited.

**Key numbers.** Step 1 (nb33 §7; 3 seeds, 40k): N3 macro-F1 0.929 +- 0.006 (prior draws) / 0.900 +- 0.020 (scenarios) [N0 0.967 / 0.987]; Sc4 0.88; **Sc7 (drift) 0.46** (N0 1.00, N2 0.00); Sc5 Corr(a,b) N0 +0.67 +- 0.04 -> N3 +0.10 +- 0.03;
alpha sd/prior sd N3 0.12-0.33, N4 0.67-0.91 (not ~1); beta sd/prior sd N3 0.065-0.11 (about 0.015-0.026 absolute). Step 2: Sc2 0.0% / Sc5 98.6% of samples at Qc_max; T_mean within 0.013 K of Tsp for Sc1-Sc4; Qc_mean 51.8 / 128.0 / 43.2 / 122.9 / 399.9.
Step 3 (P2 test column, mean of 3 seeds): P1b/P2 RMSE 0.0402 each; P1 beta coverage 0.72 and Sc4 correlation +0.38 +- 0.05 (the noise-free-start artefact, sign flipped vs N0); P0 beta coverage 0.41.

**Deviations from the runbook's expectations:** (1) nb33: without C alpha is weakly, not un-, identified; fouling class (not beta) is what is lost; Sc7 drift misdiagnosed under N3. (2) nb29: the rule selects P2 (only protocol passing a 0.01 gate; P1b misses at 0.0069) -> STOP; accuracy is a tie.
(3) The first two smoke-test attempts failed on the old (mis-specified) version of the test.

**Needs your decision:** which protocol is production for N3 (P2 vs P1b, see runbook §6, step 3 note); then steps 4-6 can run. The nb29 Interpretation cell, E4 and the ledger rows for nb29/nb03 10b/nb31/nb07 are pending that decision.

## 2026-11-09 — P2 adopted as the N3 production protocol
* nb29 under N3 selected P2 (pre-set rule, unflagged, small margin over P1b). User adopted P2; P1b reported side by side. Decision file `results/29fx_N3_selection_review.json`; rationale §10.
* Protocol now resolved by `scripts/system_I_config.protocol()` (N3 → P2, N0 → P1b) in notebooks 02, 03 §10b, 31, 07.
* Resume: runbook section 2b (R1 nb29 rebuild → R2 nb02 → R3 03 §10b → R4 nb31 → R5 nb07).


---

## 2026-11-09 N3 production run, resume — R1, R2 done; R3 STOPPED (agent run, 09:13-09:50)

**Done:** R1 (nb29 rebuild under the P2 decision; interpretation written incl. P1b-vs-P2 side by side, the beta = 0.70 "Sc2" label and the sign flip of the P1 artefact correlation; E4 not run because RUN_E4 is off), R2 (nb02 regenerated under N3 + P2: asserts passed, Sc2 0.0% / Sc5 99.2% at Qc_max;
section 10 and the P1b text updated). **R3 stopped by its rule:** the pre-set summary-set rule selects S1 (S0 + 3 plug-in estimators) over S0; R4 (nb31) and R5 (nb07) not started. Details and numbers: runbook section 6, R3 note.

**Needs your decision:** promote S1 or keep S0 (the margin is a TARP-KS gate, 0.0056 vs 0.048 around 0.01, with efficiency tied: 0.867 vs 0.880; S1's beta bound is 7-10% tighter, alpha unchanged). nb31's N3 bank is not built yet, so this is the cheapest moment to promote. The nb03 10b.7 interpretation and the ledger row for the summary-set decision wait for it.

---

## 2026-11-09 N3 production run — S1 promoted (user decision); R4 launched 10:18

**Decision:** S1_base_est (29 summaries + 3 plug-in estimators) is the N3 production summary set by the pre-set nb03 rule, clause (b); S0 reported side by side; N0 stays S0. Recorded in `results/03s_N3_decision_review.json`.
**Implementation (scoped exception, details in runbook section 6):** `system_I_config.summary_set()`, `ws_eval.features()` + keyword `features=` on both simulators, nb31 routed through the set (set in file names; S0 keeps the legacy names so N0 stays reproducible), notebook 07 (code edits, outputs cleared),
tests (30 pass; default path bit-identical to references stored before the change; S1 equals nb03's union exactly), smoke test extended. Backups in `logs/backup_before_S1/`.
**Open for notebook 34 (R2, inlet uncertainty):** it must test S1 (the estimators use the nominal inlet) with S0 as the comparison.
**State:** R3's 10b.7 interpretation written; R4 (nb31, ~4-5 h) running; R5 (nb07) follows. The nb31 interpretation will compare with nb33's N3/S0 numbers at 40k.

## 2026-11-09 — R2 (notebook 34) built, runs after R5
* Inlet-uncertainty study: Part A misspecification (Ti, Tci, Ci, one at a time) on the production posterior and the nb03 S0/S1 ensembles; alarm from the disagreement of the two plug-in β estimates; Part B robust training only if the pre-set rule fails.
* Files: scripts/ws_inlet.py, scripts/nb34_inlet_uncertainty.py, scripts/build_nb_system_I_34.py, tests/test_ws_inlet.py. Runbook §2c; rationale §12.


## 2026-11-09 — R4 (nb31, N3 + P2 + S1): all stages done, last notebook cell FAILED (stop)
nb31's bank, test set, 30 posteriors, evaluation and the production record (budget 40,000, single seed, rule v2, no flag) are on disk; the notebook's final sanity cell failed because `scripts/build_nb_system_I_31.py` l. 198 hard-codes `test_{PROTOCOL}.npz` while the test set is now `test_P2_S1_base_est.npz` (consequence of my file-naming change; the builder is not on the list of files I may edit).
**Needs your decision:** the one-line fix `B._tok(PROTOCOL)` (details and headline numbers in runbook section 6, R4 note). R5 (nb07) and R6 (notebook 34) not started; no interpretation written for nb31 yet.

*Update 2026-11-09 12:26:* the one-line builder fix was applied and R4 re-run: stages skipped, results identical, but the same final cell now fails on a second, unrelated bug (`.numpy()` on a tensor with grad: single-seed DirectPosterior; the cell was written for an ensemble). Proposed: `.detach()` in builder line 201 (needs your OK, outside the one-line scope). R5/R6 and the nb31 interpretation wait.

*Update 2026-11-09 12:40:* R4 completed (two display-cell fixes in the nb31 builder, recorded in the runbook); nb31 interpretation written. Production posterior N3: 40k single seed (seed 3), calibrated ensemble at every budget; flag for the user: the production member fails the interior beta marginal KS, the 40k ensemble passes everything (tie in the single-vs-ensemble rule). Next: R5 (nb07), R6a/R6 (notebook 34).

*Update 2026-11-09 12:51:* rule v2.1 (user-approved amendment, after results) applied for N3: the deployed production posterior is the 40k 5-seed ENSEMBLE (v2 had picked seed 3, which fails the rule's own calibration criterion). N0 unchanged (verified). R5 (nb07) failed on a latent `pd`-before-import bug in cell 10 (the cell writes a result CSV, so I need your approval for the fix). Sc4 accuracy explained (test-set sampling, not P2 or S1: 0.886 on 1,000 windows). Next after the fix: re-run R5 against the ensemble, then R6a/R6.

*Update 2026-11-09 12:57:* R5 done (nb07 commentary written; Claim 1 holds, the N0 form of the open-loop failure does not reproduce, details in the notebook and the ledger). **R6a failed one of three tests** (`test_feed_temperature_moves_the_plant`): a +1 K feed error pushes the healthy loop to valve saturation (27% of samples at Qc_max), so mean T shifts by +0.072 K, above the test's 0.05 K bound; at +0.5 K there is no saturation. Needs your decision (change the test's feed error to +0.5 K or relax the bound). R6 (notebook 34) not started.

## 2026-11-09 N3 production run — R6 (notebook 34, inlet uncertainty) done; the whole sequence R1-R6 is complete
Production posterior N3 (40k 5-seed ensemble, P2, S1, rule v2.1) is **not robust to realistic inlet errors**: 18 of 24 gated cells fail the pre-set rule; Tci ±0.5 K is catastrophic (β bias −0.14 / +0.22, 7–13 sd, accuracy ~0.5), Ti ±0.5 K and Ci ±1% give biases of 0.01–0.05; the β sd stays 0.015–0.020 (silent failure). Saturation amplifies the Ti effect but does not cause it (read-only analysis `logs/34_N3_saturation_fractions.*`). S1 is not more fragile than S0. The |z|>3 alarm catches Tci errors partially, not Ti or Ci. Part B (robust training) is calibrated under the realistic inlet distribution at ~5× the β sd; reported, not adopted.
**Needs your decision:** whether to deploy a robust posterior (or measure Tci to ~0.04 K); the paper wording for the inlet section (silent failure + checklist item). Everything is documented in runbook section 6, the ledger and rationale section 12.

## 2026-11-09 — R2 Part C (notebook 35) built
* Posterior conditioned on measured Ti, Tci (calibration error ε = 0, 0.05, 0.1, 0.2 K) vs marginal, inlet varying ±2 K; pre-set deployability rule; instrument requirement. Production unchanged until the user decides.
* Files: scripts/nb35_measured_inlet.py, scripts/build_nb_system_I_35.py, tests/test_nb35_features.py. Runbook §2d; rationale §13.


## 2026-11-09 N3 production run — R7 (notebook 35, measured inlet temperatures) done; stopped for your decision
Conditioning on measured inlet temperatures removes the nb34 silent failure (pooled β coverage 0.90–0.95 and mean |bias| < 0.02 within ±2 K, against 0.00–0.47 for the production posterior). Instrument requirement the data support: inlet temperature to ≤ 0.1 K; β sd at the nominal inlet: oracle 0.017, ε 0.05 K 0.028, **ε 0.1 K 0.039**, ε 0.2 K 0.055, marginal 0.111 (production 0.015, valid only at the nominal inlet). The literal rule finds no deployable variant and recommends "marginal", the worst variant; ε 0.05 and ε 0.1 each violate one cell of 52; the oracle fails TARP. **Needs your decision** (nothing changed in production): literal rule / ε = 0.1 K / ε = 0.05 K / amend the fallback. Details: runbook §6, rationale §13, ledger.

## 2026-11-09 — production switched to the measured-inlet design; notebook 36 built
* User: production = measured inlet temperatures, ε = 0.1 K; nb31 posterior = naive model; nb35 fallback amended v1.1.
* Files: scripts/nb36_production_measured_inlet.py, scripts/build_nb_system_I_36.py; scripts/nb35_measured_inlet.py (decision amended). Runbook §2e (R7b, R8); rationale §14.


## 2026-11-09 N3 production run — R8 (notebook 36, measured-inlet production posterior, eps = 0.1 K): trained and evaluated; stopped at the acceptance check
Production by nb31's rule v2.1 (unchanged, not flagged): **20,000 simulations, 5-seed ensemble**, calibrated (ensemble interior KS p alpha 0.063 / beta 0.030, TARP KS p 0.71; coverage 0.92 / 0.91), β sd 0.044, α sd 0.044–0.056. Over ±2 K it matches the nb35 ε = 0.1 K variant (coverage 0.84–0.94, |bias| 0.005) while the naive nominal-inlet posterior has coverage 0.00 and bias up to 0.31. **Acceptance fails by the pre-set count (4 violating cells, limit 1): all are class-accuracy shortfalls against the oracle at Sc2/Sc4 (β = 0.85, near the class threshold), none on coverage/calibration.** **Needs your decision:** accept as is, amend the acceptance criterion, or measure the shortfall on a larger window set first. Section 5 of notebook 36, the claims ledger and rationale §14 wait for it. Details: runbook §6.

## 2026-11-09 N3 production run — notebook 36 closed: production posterior accepted with a documented exception
User decision: do not amend the acceptance criterion; accept the measured-inlet production posterior (**20k, 5-seed ensemble, ε = 0.1 K**) with a documented exception: 4 of 52 cells miss the class-accuracy criterion (Sc2 ×3, Sc4 ×1; β = 0.85, 0.05 below the threshold; coverage 0.94–0.98). Read-only follow-up on ~1,000 windows per cell shows the shortfall is real (production 0.80–0.92 vs oracle 0.88–0.99; ε = 0.05 K recovers most), caused by the β width at ε = 0.1 K (0.045 vs 0.017). Record: `results/36_N3_acceptance_review.json`; follow-up: `results/36_N3_acceptance_followup.csv` (`logs/36_N3_acceptance_followup.py`). Notebook 36 section 5, the claims ledger and rationale §14 are written. (Re-running `build_nb_system_I_36.py` is safe since the reproducibility change below.) **Open:** Phase-3 notebooks need the measured inlet per window (nb02's data has none: to be checked); Claim 1 (nb07) is a nominal-inlet comparison and the paper must say so.

*Update 2026-11-09 18:06 — reproducibility of notebook 36:* `stage_accept` passes the recorded 4-cell exception through `results/36_N3_acceptance_review.json` (same count and cells; otherwise it raises as before) and the builder keeps the written section 5 when it rebuilds; verified without touching results (details: runbook §6). Not executed.

## 2026-11-09 — System R (redesign for diagnosability) planned, not started
* Plan: docs/Manuscript/system_R_redesign_plan.md. To be picked up after System I Phase 3. Paper direction: one paper on a plant designed for diagnosability (success), fragility shown on the same plant, System I as the marginal-design contrast.

