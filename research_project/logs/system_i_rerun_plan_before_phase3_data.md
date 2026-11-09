# System I — re-run plan after the controller-form fix (2026-11-08)

Rationale for every study parameter (values, evidence, pre-set vs amended): `docs/Manuscript/system_i_study_parameter_rationale.md`.

Scope: the paper now covers **System I only**. Everything cached before the 2026-11-08
controller fix is invalid (see `HANDOFF_2026-11-08_controller_fix.md` §2). This plan
takes the project from the fixed `physics.py` to a consistent set of results for the
manuscript. Phases run in order; within a phase, items marked ∥ can run in parallel.

Decisions already made (do not relitigate): keep k0 and disclose it as 60× Fogler; fix the
controller form (Kp = 150, τi = 10 min kept); System I only; the warm-start protocol is chosen
by the pre-set rule in `scripts/nb29_ws_study.py`; the production budget by the pre-set rule
in `scripts/nb31_budget_convergence.py`; classification threshold 0.90 (already the default
in `cstr_sbi.metrics.classify_fault`).

---

## Working storyline of the paper (agreed with the user 2026-11-09)

**Core message.** Simulation-based inference gives calibrated, near-optimal posteriors for fault
diagnosis in closed-loop processes, where the controller masks faults that open-loop methods
misread. But the posterior inherits every assumption of the simulator: the control law, the
operating history before the observation window (warm start), the sensor-noise model and the
inlet conditions can each produce overconfidence, bias or spurious α–β correlations, and these
failures are *silent*: the posterior looks confident and plausible. SBI is reliable if the
simulator tells the truth, and easy to get wrong otherwise.

**Structure.**
1. Motivating application, Claim 1: closed-loop SBI diagnoses faults the controller masks;
   open-loop-trained SBI fails, visibly, with no posterior mass inside the prior (nb07).
2. Core contribution, sources of silent failure, each with its experiment:
   control law (controller bug; nb30 counterfactual) · warm start (nb29: P0 reproduces the old
   β bias, P1 gives β overconfidence and a spurious correlation whose sign depends on the sensor
   model; P2/P1b side by side) · sensor model (nb33: N0–N4 ladder, artefact correlations, α
   without C) · inlet conditions (R2, notebook 34: done, not robust, silent; see the ledger) · **operating-mode mismatch** (nb07: a closed-loop posterior applied to open-loop data is biased and overconfident, β̂ +0.026 and 90% coverage 0.36, with only 0.08% of its samples outside the prior, so the prior-leakage diagnostic does **not** flag it: a silent failure; the converse mismatch, an open-loop posterior on closed-loop data, is flagged: 100% outside the prior).
3. Practical take-away, sensor specification: fouling (β) is diagnosable from T, Tc, Qc alone;
   catalyst activity (α) needs an online concentration measurement (nb33 §7).
4. Practitioner checklist (closing section), one item per experiment: real control law ·
   warm start matching the plant's history · sensor model of the real instruments, with the
   measurements each fault needs · inlet uncertainty as nuisance parameters · operating mode of the plant matches the simulator (prior leakage catches a gross mismatch, not a mild one) · joint calibration
   checks (SBC + TARP) · precision against the information limit (Cramér–Rao bounds).

**Production model (decided 2026-11-09 after nb35).** The paper's production posterior is conditioned on
measured inlet temperatures (ε = 0.1 K, notebook 36). The nominal-inlet posterior (nb31) is the
*naive* model: it shows the silent failure under realistic inlet errors (nb34) and is the model of
the nominal-inlet Claim 1 comparison (nb07). Headline precision and accuracy come from notebook 36.

**Evidence still needed for the "reliable" half:** Claim 1 under N3 (nb07), baselines
(NUTS, EKF/UKF), and R2 (β under a 0.5–1 K feed-temperature error, with and without Ti as a
nuisance parameter). Whatever R2 shows goes in: robustness, or another checklist item.

**For commentary cells.** Tie each result to one of the four parts above; say explicitly
whether it supports the "reliable" half (calibrated, near the bound, correct diagnosis) or
documents a silent failure mode (and how it would be detected). Report weaker numbers under
realistic sensors as findings explained by the sensor ladder, not as setbacks.

## Phase 0 — outstanding fixes (no long runs before these are done)

| # | Item | Where | Done when |
|---|---|---|---|
| F1 | EKF/UKF controller copy uses the old law in the dynamics, the 6×6 Jacobian (`dQc_dI = 1/TAUI`), the measurement Jacobian (`H[3,3] = 1/TAUI`) and the integrator initialisation (`I0 = (Qc−QC0−KP(T−TSP))·TAUI`). Replace by importing `compute_qc` / `cstr_closed_loop_rhs` from `physics.py`; analytic terms become `dQc/dI = KP/TAUI`, `I0 = TAUI·((Qc−QC0)/KP − (T−TSP))`. | `notebooks/system_I/16_ekf_ukf_baseline.ipynb` cells 4, 6, 9; `scripts/build_nb_19.py` l. 578–639 | a unit test compares the EKF's `f` and `h` with `physics.py` at 5 random states |
| F2 | `NOMINAL_Y0_CL = [0.0184, 312.15, 299.05, 0.0]` — **resolved 2026-11-08: documented, not changed.** Its own comment defines it as a start-up state near the open-loop point (not a closed-loop steady state); under the fixed law I = 0 simply starts the integral term at zero, a valid start-up condition (initial Qc = 27.5 L/min). No production path uses it: training, test sets and `observations.npz` all go through `ws_eval` with matched (P1b) warm starts. Notebook 01 uses it for start-up transients, which is its intended role. | `src/cstr_sbi/physics.py` l. 124 | no action |
| F3 | Check the 30-day tracking stream (`scenarios.py`, nb10) only simulates through `physics.py` (no private controller copy). | `src/cstr_sbi/scenarios.py`, nb10 | grep clean |
| F4 | **Scenario catalogue — DONE 2026-11-08 (user decisions).** Sc2 β 0.70 → **0.85**, placed by a rule that uses no inference result: the midpoint of the fouling band in which the controller stays unsaturated (fouling class β < 0.90; the valve saturates below β ≈ 0.795; the jacket cannot hold Tsp at all below β ≈ 0.775). An interim 0.80 was rejected after measurement (9.1% of samples at Qc_max, β 90% coverage 0.68–0.80). Sc7 is now exactly Sc2 + 2 K drift. Sc5 β 0.40 → **0.50** (0.40 was the prior edge). Sc6 β → **0.85** (open-loop twin of Sc2). Expect Sc2 classification accuracy below ~100% (truth ~1.7–2 posterior sd below the threshold): report it, it is the honest incipient-fault case. | `src/cstr_sbi/scenarios.py` | done; re-run notebook 02 |
| F5 | Run the tests: `python -m pytest tests/test_controller_form.py tests/test_ws_metrics.py -q`. | — | all pass |
| F6 | Manuscript-only edits (no compute): Eq. (3) standard PI form; Table I k0 disclosed as 60× Fogler (1.018×10¹⁵ h⁻¹ equivalent) after checking the acid-catalysis rationale against Furusawa/Fogler; remove "integral action masks the fault over days" prose. | `docs/Manuscript/main.tex` | — |

Already verified clean (2026-11-08): `scripts/nb20_nuts_worker.py`, `nb21_nuts_worker.py`,
`recompute_theoryC_section5.py`, `nb22_cnn_theoryF_recheck.py` have no private controller law.
`scripts/build_nb_system_I_23.py::old_closed_loop_steady_state_exact` is an intentional
old-law copy for comparison; nothing downstream may import it.

## Phase 1 — warm-start protocol (nb29) — DONE 2026-11-08

Notebook `notebooks/system_I/29_warm_start_protocol_study.ipynb` (stages from
`scripts/nb29_ws_study.py`). Result: **P1b** (deterministic steady state at θ followed by a
150-min noisy burn-in), chosen by the rule's fallback because the selection was **flagged**.
The flag is not protocol-related: the full-set α SBC KS fails for *every* protocol,
including matched train/test pairs, at 5k simulations, while α 90% coverage is ≈0.91. It is
handled by nb31 (budget) with an interior-SBC gate. P1b is also best on every other
calibration measure on the realistic (P2) column. **User sign-off required**: the review
cell in notebook 29 writes `results/29fx_selection_review.json`; nb31 refuses to start
without it. Findings that change the paper are listed in the notebook's Conclusions cell
and in the claims ledger below.

## Phase 1b — summary statistics (notebook 03, section 10) — before nb31

`notebooks/system_I/03_summary_statistics_design.ipynb` section 10 (cells appended by
`scripts/append_nb03_section10.py`, stages in `scripts/nb03_summary_study.py`, features in
`scripts/summary_variants.py`). Tests five summary sets under P1b at 5k/20k × 3 seeds: the
current 29-D (S0) and four candidates adding plug-in steady-state estimators (ln α̂, two
independent ln β̂), loop-dynamics features and detrended stds, or trimming min/max/slope.
Decision rule fixed in advance (`results/03s_decision.json`). If S0 is kept, nb31 runs as is.
If a candidate wins: add it to `cstr_sbi.summaries` as a new versioned set, make `ws_eval`
compute it, then run nb31 (its bank and test set use the production summaries). Sections 1–9
of notebook 03 are re-run later in Phase 3, after nb02 regenerates `observations.npz`.

## Phase 2 — training budget (nb31) — DONE 2026-11-08

Notebook `notebooks/system_I/31_budget_convergence.ipynb`. **Production posterior: 160k
simulations, 5-seed ensemble, P1b** (`results/31_production.json`; loaded downstream with
`from nb31_budget_convergence import load_production_posterior`).
- The pre-set rule (v1) selected 20k. It had three flaws (no persistence; gated on the
  misspecified Sc7; no precision criterion). The amended rule (v2, written after the results and
  approved by the user) selects 160k. Both are recorded; the paper must report both.
- At 160k: interior SBC KS p α 0.024, β 0.84; TARP KS p 0.86; α posterior sd 1.18–1.27× its
  29-summary Cramér–Rao bound (3.5× at 20k); β at its bound (0.97–1.01×) from 10k on.
  Ensemble α is mildly conservative (90% coverage 0.93); single seeds are unreliable (α KS p
  from 1e-5 to 0.97 across seeds at 160k), so the ensemble is required.
- Classification (ensemble, any budget): Sc1, Sc3, Sc5, Sc7 100%; Sc2 (β = 0.85) 98%,
  β̂ = 0.853, coverage 0.96; Sc4 94–96%. Sc5 α coverage 0.82–0.88 (saturation).
- **Sc7 is misspecified** (+2 K drift not in the model): β̂ 0.59–0.67 vs 0.85, coverage ≤ 0.56,
  budget-unstable. Its 100% classification is luck (drift pushes β̂ into the fouling region)
  and must not be presented as a success: the drift-aware model (nb09) goes in the main results.

## Phase 2b — robustness studies (R1 before Phase 3; R2 any time before submission)

### R1 — sensor-noise realism (notebook 33) — DONE 2026-11-09 → production switched to N3

**Outcome.** The pre-set rule flagged N3 (Sc4 accuracy 1.0 → 0.88, Sc2 0.94 → 1.0, Sc5 α–β
correlation +0.69 → +0.10), and the user adopted **N3 as the production sensor model on
2026-11-09**. Under N3 (40k, 3-seed ensembles): α posterior sd 0.03–0.06 (N0: 0.003–0.004),
β sd ≈ 0.015 (N0: 0.025–0.037), both at the 29-summary Cramér–Rao bound; α–β uncorrelated in
every scenario; β calibrated (interior KS 0.41, TARP 0.81), α interior KS fails at 40k (1e-4).
N4 (no concentration measurement): α unidentifiable (CRB 0.22–0.70), β unaffected, macro-F1 0.54.
Caveat: under N2/N3 the raw-trajectory Ledoit–Wolf bound for β is looser than the summary bound
(shrinkage artefact), so the summary bound is the β reference under N3.

**Protocol under N3 (2026-11-09): P2.** nb29 re-run under N3 selected P2 (pre-set rule, unflagged; P1b missed the α gate narrowly). The user adopted P2 and P1b is reported side by side; see rationale §10.

**Production chain re-run under N3** — autonomous sequence in
`docs/Manuscript/system_i_N3_runbook.md`: 02 → 29_N3 → 03 §10b → 31_N3 → 07. Configuration in
`scripts/system_I_config.py` (`CSTR_SENSOR_MODEL`, default N3); N3 outputs carry the suffix `_N3`
(`29fx_N3_*`, `03s_N3_*`, `31_N3_*`, `07_*_N3`), N0 outputs are kept unchanged for the SI. Every
Phase 3 notebook must set the sensor model explicitly when it is modernised.

*Original R1 design (kept for the record):*
**Why.** The inherited sensor model (σ = 0.5% of each channel's window maximum) gives
σ_T ≈ 1.56 K (pessimistic; industrial RTDs ±0.1–0.5 K) and σ_C ≈ 9×10⁻⁵ mol/L (optimistic;
online analysers are typically 1–2% of full scale). α is inferred mainly from C at ~98%
conversion, so α's precision is probably overstated. If realistic noise changes the
conclusions, the production noise model should change before the long Phase 3 re-runs.

**Code needed.** `ws_protocols._observe` / `make_batch_simulator`: an optional absolute
per-channel sensor σ (`noise_abs=(σ_C, σ_T, σ_Tc, σ_Qc)`) next to the existing
fraction-of-maximum convention (default unchanged, so nb29/nb31 stay reproducible).

**Variants** (all P1b, same θ draws and seeds; absolute σ):

| Variant | σ_C [mol/L] | σ_T, σ_Tc [K] | σ_Qc [L/min] | Purpose |
|---|---|---|---|---|
| N0 | 0.5% of max (≈9×10⁻⁵) | 0.5% of max (≈1.56) | 0.5% of max | current production (reference) |
| N1 | 0.01 (1% of a 1 mol/L analyser span) | as N0 | as N0 | realistic C only: isolates the α effect |
| N2 | as N0 | 0.2 | as N0 | realistic T only |
| N3 | 0.01 | 0.2 | 2 (≈0.5% of span) | realistic on all channels (plant with an online analyser) |
| N4 | 1.0 (no usable C) | 0.2 | 2 | realistic, no online analyser (added before training; reported, not gated) |

**Per variant:** Fisher-information Cramér–Rao bounds (as nb29 E2; minutes), then NPE at 40k
× 3 seeds (≈10 min simulation + 3 × ~4 min training), evaluated on a test set simulated
with the same variant. **Report:** per-scenario α/β posterior sd and coverage, classification
accuracy, Corr(α,β), SBC/TARP, against N0 at the same 40k budget.

**Decision rule (fixed before running):** if N3 changes any headline conclusion (a scenario's
class accuracy by > 5 points, β coverage outside 0.85–0.95, or the sign/regime of the α–β
correlation), the user decides whether to switch production to N3 before Phase 3. Otherwise
production stays N0 and N1–N3 go to the SI as the sensor-realism robustness check, with α's
precision reported under both.

### R2 — inlet-condition uncertainty (notebook 34) — before submission (built 2026-11-09; runs after R5, runbook §2c; rule in rationale §12)
**Why.** The jacket removes only ~19% of the reaction heat, so a 1 K feed-temperature error
shifts the jacket duty by ~27% (≈0.2 in β). The model treats Ti, Ci, Q, Tci as exactly known.
**Design.** (a) *Misspecification test*: apply the production posterior to test windows whose
plant has Ti ± 0.5 / 1 K, Ci ± 2%, Tci ± 1 K (one factor at a time), and report the β and α
bias and coverage loss. (b) *Robust training (if (a) shows large bias)*: train with Ti (and
Tci) drawn as nuisance variables, e.g. Ti ~ N(297, 0.5²), and compare. Needs a per-sample
inlet in `ws_protocols` (currently one global inlet). Does not change the production posterior
unless (b) is adopted.

## Phase 3 — notebook re-runs (in this order)

Old notebooks build paths as `Path.cwd().parent` / `"../data"`, which only works with the kernel
one level below the project root. Each must switch to the root resolution used in notebooks
02/07/29/31 (assert cwd = `notebooks/system_I`, `ROOT = cwd.parent.parent`) when it is re-run.

Rule for every notebook: stored in and executed from `notebooks/system_I/` (kernel cwd
asserted in the first cell; builders execute with nbclient `resources.metadata.path` set to
that folder); simulate through `ws_eval.simulate` / `ws_protocols` with the
chosen protocol (never a private simulator copy), load the production posterior via
`load_production_posterior()`, write outputs with a new suffix so old-law files cannot be
picked up, and record each manuscript number it produces in the claims ledger below.

| Order | Notebook | Action | Notes |
|---|---|---|---|
| 1 | 00 smoke test | **READY — run as is** (open-loop only, unaffected by the controller fix; no data paths) | quick |
| 2 | 01 model demonstration | **READY — run as is**, then text review | closed-loop time constants are now 10–26 min; start-up from `NOMINAL_Y0_CL` |
| 3 | 02 data generation | **DONE (N3 + P2) 2026-11-09 — executed, §10 written; §4b asserts passed (Sc2 0.0% / Sc5 99.2% at Qc_max, bit-identical to training simulator); N0 archived as `data/observations_N0.npz`; the earlier N3/P1b run is `logs/02_data_generation_N3_P1b_backup.ipynb`; `observations.npz` metadata `protocol` = P2, `burn_min` = 300.** Regenerates `data/observations.npz`: closed loop via `ws_eval.simulate` (P1b), open loop via `ws_eval.simulate_open_loop` (300-min burn-in), new catalogue, root-relative paths, metadata keys `protocol`/`burn_min`/`controller_law`. §4b checks Sc5 saturation, reports Sc2 saturation, verifies bit-identity with the training simulator and open-loop convergence | independent of nb31 and notebook 03 §10 |
| 4 | 03 summary statistics design | re-run sections 1–9 on the regenerated data (section 10 already done in Phase 1b) | check the saturation-fraction features now that Sc2 saturates |
| 5 | 04 SBI training | replace training by `load_production_posterior()`; keep the §10 comparison against nb31 | do not retrain here |
| 6 ∥ | 05a / 11 classification | re-run | threshold 0.90; report per-class F1 and the Sc7 drift confound |
| 6 ∥ | 06 multi-sample study | re-run | |
| 6 ∥ | 09 sensor drift | re-run | |
| 7 | 04b + 24 CNN embedding | merge into one; retrain the CNN under P1b at the production budget | lower priority after nb29 E2: raw-trajectory Cramér–Rao bounds are within ~10–20% of the 29-D summary bounds under P1b, so the summaries are close to sufficient; the CNN result becomes a short SI check. Replaces the retired "within 1%" claim |
| 8 | 05 + 23 NUTS baseline | merge; NUTS likelihood must use the fixed law, the full covariance (`cov_obs`) estimated from replicates **under the chosen protocol**, and the matched warm start | resolve nb23's R-hat regression first; if the chosen protocol has a random initial state (P1b/P2/P3), the Gaussian-summary likelihood absorbs it through `cov_obs` — state this |
| 9 | 16 EKF/UKF | re-run after F1 | |
| 10 | 10 sequential tracking | re-run with the production posterior; **change `scenarios.generate_degradation_stream`**: it warm-starts every window at the noise-free steady state (P1). Give each window the P1b burn-in (or chain windows so each starts from the previous window's end state, which is the physically realistic stream) | the posterior is trained on P1b; P1 test windows would reintroduce the nb29 mismatch |
| 11 | 08 identifiability & saturation | rewrite the FIM with full covariance and the raw-trajectory check (reuse `nb29_ws_study._fim`) at the final scenario points | nb29 E2 already gives, under P1b with full covariance, I_αα/I_ββ ≈ 380–1500 in control (Sc1 590, Sc3 1460, Sc4 380) and ≈36 at saturated Sc2; the diagonal FIM is wrong (4.4 at Sc2) |
| 12 | **new 32** α–β mechanism | consolidates 19, 20, 26 under the fixed law and P1b. nb29 already shows the story: α and β nearly uncorrelated while the loop is in control (SBI +0.02–0.03, FIM ±0.06), positive when the valve saturates (Sc2 +0.42, Sc5 +0.39; FIM +0.48) because saturation opens the loop and the heat-balance compensation appears. nb32 makes this a figure: Corr(α,β) and Qc saturation fraction across a β grid at α = 1 (and α grid at β = 1), SBI vs FIM, plus the P1-vs-P1b contrast showing the spurious negative correlation from a noise-free start | the earlier near-unity negative correlation = controller bug + noise-free start; no "noise-mediated" component is needed |
| 13 ∥ | 07 open vs closed | **RUN 2026-11-08; text rewritten.** Sc2: CL-SBI β̂ 0.861, cov 0.94, acc 0.96; OL-SBI **fails, here visibly** (94% of its samples outside the prior, β coverage 0, W1 15×; the N0 raw samples happened to fall at β̂ ≈ 1.0, which is not the robust finding — reworded 2026-11-09). Sc6 reverses (OL acc 0.88 vs CL 0.38). The August "β̂ ≈ 0.15" failure came from the controller bug. **Re-run once** (OL posterior cached, ~2 min) for the new section 3b (prior leakage + in-prior metrics). Optional: 5-seed OL ensemble for a like-for-like comparison | open-loop scenarios kept (user decision) |
| 13 ∥ | 21 open-loop NUTS | update together with the NUTS baseline (row 8): use Sc0/Sc6 windows from the regenerated `observations.npz` instead of self-generated ones; estimate the likelihood covariance from `ws_eval.simulate_open_loop` replicates; keep the generative model's open-loop warm start at θ's own steady state (the burn-in only randomises the start, which the covariance absorbs) | after nb31 and the R-hat fix |
| 13 ∥ | 12 resource analysis, 18 ablation | re-run | |
| 14 | 14 claims & conclusions, 17 SI figures | rebuild last, from the claims ledger | |

Retire (keep files, do not cite): 15 (β-bias analysis: the bias was a protocol artefact,
now reproduced by nb29 E3: train P0 → test P2 gives Sc2 β̂ = 0.555, bias −0.145, against the
old manuscript 0.551, bias −0.149), 19, 20, 26 (superseded by 32; nb20 Theory A also used an
inconsistent controller formula), 22 (superseded by nb31). Keep as SI evidence: 25 (k0),
`scripts/nb30_*` (controller counterfactual and eigenvalue check), `scripts/nb27_step0_*`.

## Claims ledger (fill in as notebooks are re-run)

| Manuscript claim | Section | Produced by | Status |
|---|---|---|---|
| Claim 1: closed-loop SBI diagnoses faults the controller masks; an open-loop-trained model fails (visibly: no posterior mass inside the prior) | §6.1 (motivating application) | nb07 (N3) | **done (N3 + P2 + S1, 40k ensemble), 2026-11-09:** Sc2 (β = 0.85): CL-SBI β̂ 0.848, coverage 0.94, accuracy 0.98, W1 0.0153 [N0 0.861, 0.94, 0.96, 0.0285]; OL-SBI has **100%** of its samples outside the prior (N0 94%), W1 0.304 [0.437] — **fails, here visibly** (the N0 wording "reads the fouled plant as healthy", accuracy 0.28, is retired: N3 raw β̂ 0.77, accuracy 0.64 on out-of-support samples, not quotable; what is robust is the out-of-support failure). Sc6: reversal holds for W1 (0.0108 vs 0.0275) and coverage (0.88 vs 0.36), not for class accuracy (0.98 both; N0 0.88 vs 0.38). Leakage flags the catastrophic mismatch, not the mild one (CL on open-loop data: 0.08% outside, coverage 0.36 = silent). Single OL seed, 50 windows |
| SBC KS p (α, β) of the production posterior | §6.1 | nb31 (N3) | **done (N3 + P2 + S1), 2026-11-09:** **deployed = 40k, 5-seed ENSEMBLE** (rule v2 found convergence at 20k→40k; v1 would give 10k; N0: 160k ensemble). Calibrated at every budget 5k–160k: interior KS α 0.22–0.88, β 0.04–0.81 (40k: 0.879 / 0.080), TARP KS 0.45–0.76 (40k 0.762), 90% coverage 0.90–0.91. **Amendment v2.1 (2026-11-09, after results, user-approved):** the deployed posterior must itself pass calibration (d); v2 would have deployed seed 3 (interior β KS p = 0.00005; 2 of 5 members fail on β, all pass TARP) on a tie (TARP dev 0.0200 vs 0.0200, below the ~0.043 sampling noise); v2 choice kept in the record; N0 unchanged. [N0 160k ensemble: interior α 0.024, β 0.84, TARP 0.86 → SI] |
| Sensor-model ladder N0–N4 (which measurement buys which diagnosis) | SI / §6.3 | nb33 | done: N3 α sd 0.03–0.06, β 0.015, no α–β confounding; N4 α weakly identified (sd/prior sd 0.67–0.91, not ≈ 1 — corrected 2026-11-09), β fine, macro-F1 0.69 ± 0.02 (prior draws) / 0.53 ± 0.05 (scenarios) |
| Classification F1 / confusion per sensor model; artefact α–β correlations (Sc5 under imprecise T, N4 without C, Sc7 drift); α without C | §6.3 / SI figure | nb33 §7 | **done (N3), 2026-11-09:** macro-F1 N3 0.929 ± 0.006 (prior draws) / 0.900 ± 0.020 (scenarios) [N0 0.967 ± 0.002 / 0.987 ± 0.007]; errors in the α-only class pairs; Sc4 0.88; **Sc7 (drift) 0.46 under N3 (N0 1.00, N2 0.00)** — drift robustness open → nb09. Sc5 ridge N0 +0.67 ± 0.04 → N3 +0.10 ± 0.03; N4 creates correlations (+0.15 to +0.51, Fisher +0.37 to +0.78); fouling class recall 0.96 → 0.49 without C while β is unaffected |
| Posterior precision vs Cramér–Rao bound | §6.1 / SI | nb31 (N3) | **done (N3, vs the production set's own bound S1, nb03 10b):** posterior sd / bound at 40k: α 0.98–1.02, β 0.85–1.04 (S0 bound: β 0.77–0.95); 160k similar; 20k 4–10% above for α. Raw-trajectory bound NOT a valid β reference under N3 (Ledoit–Wolf looser: 0.031 vs 0.014–0.016). Absolute: α sd 0.027–0.064, β 0.014–0.024 [N0: α 0.002–0.003, β 0.027–0.028 at Sc4: α 15× less precise, β 2× more precise under realistic sensors]. Diagnosis (40k ens): **Sc4 accuracy 0.886 ± 0.010 on 1,000 fresh windows = the production figure to quote**; the fixed 50-window test set gives 0.76–0.80 over budgets (quote only as the fixed-test-set figure: those windows are harder than average, binomial SE 0.057, shared by all budgets; not a P2 or S1 effect: nb29 E3 0.889 vs 0.900, S0 vs S1 on the same windows 0.846 vs 0.849); Sc7 (drift) 0.32 with β̂ 0.47: a silent failure from unmodelled sensor drift |
| Macro-F1 over closed-loop scenarios | abstract, §6.2 | nb05a / nb11 | pending (old 0.927 invalid) |
| Per-scenario α̂, β̂, 90% CI, coverage (Table 7) | §6.2 | nb05a | pending |
| I_αα / I_ββ ratio, full covariance | §6.3.1 | nb29 E2 → nb08 | provisional: 380–1500 in control, ≈36 at saturation (P1b) |
| ∂T_ss/∂β = 0 argument | §6.3.2 | analytic + nb01 | valid within a window now (10–26 min settling) |
| CNN vs hand-crafted comparison | §6.3.3 | nb04b/24 | pending; "within 1%" claim retired |
| Baselines table (SBI, NUTS, EKF, UKF) | §6.4 | nb05/23, nb16 | pending |
| 30-day tracking MAE | §6.5 | nb10 | pending |
| Initial-state protocol effect (old β bias explained) | new section | nb29 | **done (N3), 2026-11-09 (P2 test column, 3 seeds, 5k sims):** abrupt-fault (P0) and noise-free (P1) training fail under both sensor models (TARP dev 0.271 / 0.091 vs 0.041–0.059 for P1b/P2/P3; β coverage 0.41 / 0.72 vs 0.86–0.87); the noise-free start still overstates β precision (Fisher sd 1.8× / 3.3× too small at Sc4, summaries / raw) and invents an α–β correlation whose **sign flips with the sensor model** (Fisher at Sc4: N0 −0.69 / −0.50, N3 +0.36 / +0.52; SBI N0 −0.11, N3 +0.38 ± 0.05). Rule selects **P2** over P1b by a small margin (one P1b seed fails the 0.01 α-KS gate; RMSE 0.0402 for both); user adopted P2, P1b reported side by side. [N0: P0→P2 old Sc2 (β = 0.70) bias −0.145 vs −0.149 manuscript; N0 rule chose P1b, flagged.] E4 (window length) not run (RUN_E4 off). The E2 "Sc2" point is β = 0.70 (saturating), not today's Sc2 |
| Hand-crafted summaries close to sufficient | §6.3.3 | nb29 E2 | done (N0): raw-trajectory CRB within ~10–20% of the summary CRB. **N3 (2026-11-09):** α raw 0.045 vs summaries 0.051 (Sc1, P1b: raw ~12% tighter); for β the Ledoit–Wolf raw bound is *looser* (0.031 vs 0.016), a shrinkage artefact — do not cite β "summaries exceed raw"; use the summary bound as the β reference |
| Summary-statistic redesign (kept or replaced, and why) | §4.2 / SI | nb03 §10 | done: 29-D kept. Candidate features add no information (CRB equal within 1%); β already at its CRB; α 3–5× wider than its CRB (a learning gap, closed by budget: 5k→20k halves α sd); plug-in estimators narrow α ~30% but make it overconfident (interior KS fails, 90% cov 0.88). S1 kept as documented fallback. **N3 + P2 (2026-11-09):** the pre-set rule selects **S1_base_est** (29 + 3 plug-in estimators) by clause (b): S0 TARP KS p 0.0056 (gate 0.01), S1 0.048; RMSE 0.0370 vs 0.0385; efficiency tied (0.867 ± 0.010 vs 0.880 ± 0.014); S1's β bound 7–10% tighter, α identical. Marginal (3 seeds): user adopted S1 as the N3 production set, S0 shown side by side; N0 stays S0. Production features: `ws_eval.features`, `system_I_config.summary_set()`; R2 (notebook 34) must test S1 with S0 as the comparison (the estimators use the nominal inlet) |
| α–β correlation ≈ 0 in control, positive (heat-balance compensation) under valve saturation | discussion | nb29 → nb32 | provisional (nb29 at 5k); final numbers from nb32 with the production posterior |
| k0 = 60× Fogler, ~98% conversion | §3.1 | nb25 | done (disclosure text pending) |
| Controller-form correction | §3.1 / SI | nb30 scripts | done (text pending) |
| Warm-start protocol under realistic sensors (P2 chosen; P1b side by side; P0/P1 fail under both sensor models) | new section | nb29 (N3) | decided 2026-11-09: P2 production; interpretation pending |
| Robustness to inlet uncertainty (Ti, Tci, Ci); misspecification alarm from the two β estimates | §6.x / checklist | nb34 (R2) | **done (N3 + P2 + S1, deployed 40k ensemble), 2026-11-09: NOT robust; a silent failure.** Pre-set rule fails: 18 of 24 gated cells violate it (β coverage ≥ 0.80 in 6 of 24; only Ci −1% passes). Pooled Sc1–Sc4 β bias (own sd; coverage; accuracy): Ti ∓0.5 K +0.021 (1.2; 0.67; 0.95) / −0.048 (2.4; 0.30; 0.96); **Tci −0.5 K −0.144 (7.4; 0.00; 0.52), +0.5 K +0.216 (13.0; 0.00; 0.50)**; Ci ∓1% +0.008 (0.6; 0.84; 0.96) / −0.015 (0.9; 0.72; 0.97); β sd stays 0.015–0.020 (confident and wrong). The coolant inlet Tci dominates (jacket-balance estimate ±14–46% per 0.5 K; Tc − Tci is a few K); the β = 0.85 scenarios are pushed to the jacket limit by +0.5 K on the feed (Sc2 59%, Sc4 56% of samples at Qc ≥ 395) but saturation amplifies and does not cause it (Ti −0.5 K, 0% saturated, coverage 0.62–0.72; Tci biases 0–1% saturated). S1 is not more fragile than S0 (Tci −0.5 K −0.143 vs −0.140; Tci +0.5 K +0.205 vs +0.148; Ti, Ci less). Alarm (|z| > 3, false alarm 0.00): detects Tci ±0.5 K in 0.40–0.52 (0.80–0.90 for fouled Sc2/Sc4, ≤ 0.16 for healthy/decay), Ti ±0.5 K 0.02–0.03, Ci ≈ 0 — partial. Part B (reported, not adopted): robust training (Ti, Tci ~ N(297, 0.5²), Ci 1%) restores calibration under the realistic inlet distribution (β KS 0.76, TARP 0.81, coverage 0.888 vs 0.161 for the nominal posterior) at ~5× the β sd (0.015 → 0.079) and class accuracy 0.97 → 0.82. Checklist: measure or marginalise Tci (≈ 0.04 K), Ti (≈ 0.15 K), Ci (≈ 1%) |
| Measured inlet temperatures as inputs: instrument requirement (ε) and precision cost vs marginalising | §6.x / checklist | nb35 (R2 Part C) | **done (N3 + P2), 2026-11-09: the nb34 silent failure is removed** (pooled β coverage 0.90–0.95, mean \|bias\| < 0.02 at all 13 inlet conditions within ±2 K, against production 0.00–0.47 and bias 0.035–0.31). β sd at the nominal inlet: production 0.015 (nominal inlet assumed), oracle 0.017, **ε 0.05 K 0.028, ε 0.1 K 0.039, ε 0.2 K 0.055**, marginal 0.111; accuracy 0.94 / 0.93 / 0.905 / 0.82 / 0.655. **Instrument requirement the data support: coolant (and feed) inlet temperature to ≤ 0.1 K.** **Pre-set rule: no variant deployable → mechanically recommends "marginal", which is the worst variant (29 violations, accuracy 0.655); ε 0.05 and ε 0.1 each violate one of 52 cells (coverage 0.78 vs 0.80; accuracy 0.86 vs oracle 1.00 at the threshold-near Sc2); the oracle itself fails TARP (KS p 0.0004).** Decision for the user (not adopted): follow the rule, or propose ε = 0.1 K / 0.05 K, or amend the fallback |
| Production model with measured inlet (ε = 0.1 K): budget, calibration, precision, robustness over ±2 K | §6.1 (headline) | nb36 | **done (N3 + P2), 2026-11-09:** **20k, 5-seed ensemble** (rule v2.1, unchanged, not flagged; only 2 of 5 single members pass calibration); ensemble calibrated (interior KS α 0.063, β 0.030, TARP KS 0.71; coverage 0.919 / 0.905); β sd 0.044 / 0.043, α sd 0.056 / 0.044 (Sc1 / Sc4); over ±2 K: coverage 0.84–0.94 (mean 0.937), \|bias\| 0.005, accuracy 0.907, against the naive nominal-inlet posterior 0.00 / 0.166 / 0.610. **Acceptance missed by the pre-set count (4 of 52 cells, class accuracy only, Sc2/Sc4 at β = 0.85); accepted by the user with a documented exception, criterion not amended:** a real accuracy cost of ε = 0.1 K near the class threshold (1,000-window follow-up: Sc2 0.857–0.921 vs oracle 0.985–0.994, ε 0.05 K 0.935–0.976; Sc4 0.803 vs 0.877, 0.846; `results/36_N3_acceptance_review.json`). Trade-off: ε 0.1 K β sd 0.04; ε 0.05 K 0.03; oracle 0.017. **Open:** downstream notebooks need the measured Ti, Tci per window (nb02 data has no inlet measurement, to be checked); Claim 1 stays a nominal-inlet comparison |
