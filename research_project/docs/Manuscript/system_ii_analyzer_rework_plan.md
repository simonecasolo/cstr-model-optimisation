# System II Analyzer Rework Plan

**Status: plan of record, Phases 0-1 complete; Phase 2 REDIRECTED (2026-09-02).**
Written 2026-09-01 after `nb39`–`nb42` confirmed `(alpha, z_A0_eff)` and
`(eta_col, z_A0_eff)` are genuine, non-training-fixable identifiability limits under S-B
(and S-A) instrumentation. **Phase 0
(`notebooks/43_wu2003_analyzer_feasibility_check.ipynb`, executed) tested and rejected the
originally-proposed reactor-composition analyzer, then found a working alternative — see
"Phase 0 result" below before reading the rest of this plan.**

**Phase 2's original recipe (monolithic 5-D SNPE on S-C) is superseded**: both its
training levers failed (`z_score_x='none'` fixed `z_A0_eff` but left `alpha`/`xi_reb`
shrinkage; `n_train=15000` made everything worse), and a structural diagnosis now
explains why — see "Phase 2, Update 2026-09-02" below. The detailed replacement
(staged diagnostics, factorised posterior, pre-registered pass gates, and explicit
conditional sensor decisions) lives in
**`docs/Manuscript/system_ii_sbi_solidification_plan.md`** — read that before doing
any Phase 2 work; this file keeps only the summary and the unchanged Phases 3–7.

## Phase 0 result (supersedes the original channel choice)

- **Reactor-outlet composition analyzer (`z_A`): rejected.** Barely moves either confound even
  at full raw-trajectory resolution (e.g. `(alpha, z_A0_eff)` at scenario W2: `0.580 -> 0.588`,
  no improvement). Physical reason, visible directly in `physics.py`: `z_A`'s own dynamics
  are driven by *both* `alpha` (reaction term) and `z_A0_eff` (feed term) simultaneously, so
  observing it doesn't separate them — the same class of problem as inferring a CSTR's rate
  constant and feed concentration from outlet composition alone.
- **Feed-line composition analyzer (measuring `z_A0_eff` directly, upstream of the reactor and
  recycle mixing) works, but only above a precision threshold.** At the same `0.3%` relative
  noise assumed for every other sensor in this study it only partially helps, because the
  *existing* 66-D summaries already constrain both `alpha` and `z_A0_eff` to remarkably tight
  marginal precision (Cramer-Rao `sd~0.0003-0.0005`) along a correlated direction — a bolt-on
  sensor has to beat that *existing implicit precision*, not just add "a" measurement.
  **Settled threshold, after a 7-point sweep spanning both sides of the `(alpha, z_A0_eff)`
  sign flip (`nb40`/`41`): `0.03%` is *not* uniformly sufficient** — the `W2`-like operating
  point (`alpha=0.85`) still shows `|corr|=0.149` there. **The robust requirement is `0.01%`
  relative noise (`1e-4`)** — every tested point/pair clears `|corr|<0.05` at that level.
  `0.01%` precision is plausible for a well-calibrated online feedstock analyzer
  (NIR/Raman/GC), but must be stated and justified explicitly and precisely in the
  manuscript, with a citation to a representative real analyzer's published repeatability —
  not assumed at the `0.3%` used for the temperature/flow sensors, and not left at the
  looser `0.03%` figure from the first-pass check.

**Everything below is updated for a feed-line analyzer at `<=0.01%` relative noise, not the
reactor-composition analyzer or the `0.03%` figure originally proposed.**

## Target narrative (three-arm comparison)

| Arm | System | Instrumentation | Expected result |
|---|---|---|---|
| A | System I | (unchanged) | Classification + 30-day degradation tracking, no confounds — **already done** |
| B | System II | S-B **+ high-precision feed composition analyzer** ("S-C") | Classification + 30-day degradation tracking, no confounds — **new work** |
| C | System II | S-B, **no analyzer** (current) | Classification and tracking measurably worse on the *same* fault scenarios/trajectory, because of the confound — **mostly-existing evidence, needs one new controlled experiment** |

Arms B and C should run the *identical* new 30-day degradation trajectory (design in Phase 4)
so the comparison is a controlled ablation (same fault profile, same noise realizations,
differ only in whether the analyzer channel is present) — not two independently-designed
experiments that merely look similar.

## Why this is still cheaper than it sounds, revised for the feed analyzer

A feed-line composition reading is **not a state-dependent channel** — it's a direct,
near-constant-over-the-window reading of the parameter `z_A0_eff` itself (this model doesn't
vary `z_A0_eff` within a single 2h scenario), with sensor noise added. Unlike the rejected
`z_A` channel, this requires no new *dynamics* extraction either — `physics.py`'s
`extract_observations_explicit` doesn't need to change; the new channel is synthesized
directly from `theta[4]` plus assumed sensor noise, in `summaries.py` or the bank-generation
script. Still no `physics.py` ODE changes, no new control loop.

**Bank-generation efficiency, revised:** the feed-analyzer reading is synthesized from
`theta[4]` directly (not extracted from the raw trajectory), so a paired S-B/S-C bank is even
cheaper than the original plan assumed — generate the S-B raw-trajectory bank exactly as
today, then append one synthetic feed-reading feature column per row using the *same* theta
array already saved, no re-simulation needed at all. This makes Arm B vs. Arm C a true
paired/controlled design at essentially zero extra simulation cost.

---

## Phase 0 — Cheap feasibility check — **COMPLETE**, see "Phase 0 result" above

Ran `notebooks/43_wu2003_analyzer_feasibility_check.ipynb`: rejected the reactor-composition
channel; found the feed-line analyzer works, settled on `<=0.01%` relative noise after a
7-point sweep (the initial 3-point check's `~0.03%` estimate was not conservative enough —
see "Phase 0 result" above). Gate cleared for the *revised* channel choice and threshold —
proceed to Phase 1 below.

**Real-analyzer citation — settled (2026-09-01).** Checked two candidates:

- **IAS-PAT SE online NIR analyzer** (IAS Global): spec is absorbance-based
  (repeatability `<=0.0004 AU`, baseline noise `<=0.0004 AU`), not directly comparable to our
  `%`-composition requirement without an unstated calibration-curve assumption. Not used as
  the primary citation for this reason.
- **Kett BSS1700 NIR composition analyzer**: spec stated directly in composition-percent
  terms — accuracy `±0.01%`, precision/reproducibility `0.001%` (manufacturer notes
  "application-dependent" for all three). At the assumed feed composition
  (`z_A0_eff~0.70-0.90`, i.e. `~70-90%`), `0.01` percentage points of accuracy is `~0.0125%`
  *relative* and `0.001` percentage points of precision/reproducibility is `~0.00125%`
  *relative* — the accuracy figure sits right at our `0.01%` relative-noise requirement, and
  the tighter precision/reproducibility figures clear it with roughly an 8x margin. **Use the
  Kett BSS1700 as the manuscript's cited instrument**, with the "application-dependent"
  caveat stated honestly (a datasheet headline figure, not a guarantee for this specific
  feedstock chemistry without lab validation).

Draft manuscript sentence for Phase 7's methods section: *"We assume an online feed-line
composition analyzer (e.g. a near-infrared composition analyzer such as the Kett BSS1700,
manufacturer-quoted accuracy `±0.01%` and precision/reproducibility `~0.001%`,
application-dependent) measuring the fresh feed's reactant mole fraction upstream of the
reactor and recycle mixing point. At the operating compositions considered here
(`z_A0,eff~0.70-0.90`), this corresponds to a relative measurement noise of
`~0.0013-0.0125%`, at or below the `<=0.01%` threshold our Fisher-information analysis
(`notebooks/39-43`) found necessary to resolve the `(alpha, z_A0,eff)` and
`(eta_col, z_A0,eff)` confounds."*

---

## Phase 1 — Core library changes — **COMPLETE** (2026-09-01)

- `src/cstr_sbi/recycle/summaries.py`: added `feed_analyzer_reading()` (synthesizes
  `theta[Z_A0_EFF_PARAM_IDX] + noise` per scenario, not extracted from `RAW_CHANNELS`),
  `FEED_ANALYZER_NOISE_PCT = 1e-4` (the settled `0.01%` threshold), `N_SUMMARIES_SC = 72`.
  `compute_summaries`/`compute_summaries_batch`/`summary_names` all take an `"S-C"`
  structure now (`compute_summaries` gained optional `theta`/`rng`/`feed_analyzer_noise_pct`
  kwargs, required only for `"S-C"`; `"S-A"`/`"S-B"` callers are unaffected — verified by
  smoke test).
- `scripts/sbi_pipeline.py`: `_structure_config("S-C")` returns `NOMINAL_CTRL_SB` +
  `nominal_warm_start("S-B")` (no new control law); `_simulate_summary` now threads
  `theta_np`/`rng` through to `compute_summaries` (ignored for `"S-A"`/`"S-B"`).
- No changes needed: `physics.py`, `priors.py`, `scenarios.py`, `simulator.py`'s
  `nominal_warm_start` — confirmed.
- **Validated**: an end-to-end smoke test (`_structure_config("S-C")` →
  `_simulate_summary(..., "S-C", ...)`) produces a 72-D, NaN-free summary, and `"S-B"`'s
  output is byte-for-byte unaffected. More importantly, **re-ran the `(alpha, z_A0_eff)`
  FIM check through the real, integrated `compute_summaries("S-C", ...)`** (including the
  `_compress_heavy_tails` log1p step `nb43`'s standalone prototype didn't apply) at the
  degraded point and the `W2` worst case: results match `nb43`'s validated numbers almost
  exactly (degraded point at `1e-4` noise: `-0.011` vs. `nb43`'s `-0.010`; `W2` at `1e-4`:
  `+0.048` vs. `+0.047`) — the compression step does not change the conclusion.
- Flag for the user, out of this plan's scope: `PFD_Wu2003.png` needs the new analyzer drawn
  in by hand.

## Phase 1.5 — Model-verification companions (nb20a, nb21a) — **COMPLETE** (2026-09-01)

Per user request, redid the foundational `nb20`/`nb21` pair for S-C, numbered `nb20a`/`nb21a`
(both executed, no errors). Since S-C reuses S-B's control structure exactly, these are not
mechanical re-runs of `nb20`/`nb21` — they confirm the physics equivalence once and then
focus entirely on what's actually new: the feed-analyzer channel.

- **`notebooks/20a_wu2003_model_verification_sc.ipynb`**: confirmed S-C's raw trajectory is
  byte-identical to S-B's (max abs diff `0.0`, W12 scenario) — the analyzer really is
  observation-only, as designed. Verified `compute_summaries("S-C", ...)` dimensions/names
  (72-D, last 6 = `feed_analyzer_*`). 4-scenario disambiguation preview
  (`W1`/`W2`/`W10`/`W13`) shows the feed-analyzer reading depends only on `z_A0_eff`, with
  zero visible dependence on the simultaneous `alpha` fault in `W13` — the reactor-linked
  `F_R_norm` channel cannot make this distinction.
- **`notebooks/21a_wu2003_control_structures_sc.ipynb`**: extended the check to all 14
  closed-loop scenarios. **`corr(true z_A0_eff, feed_analyzer_reading) = +1.0000`** across
  the full catalogue (as expected for a direct measurement) vs. **`corr(true z_A0_eff,
  F_R_norm) = +0.3813`** (S-B's reactor-linked proxy) — a clean, quantitative,
  full-scenario-catalogue confirmation of exactly the mechanism `notebooks/39-42` predicted
  and Phase 0 validated deterministically.
- `nb21`'s open-loop/closed-loop masking section and its `F_R_norm`/`x_D` grids were **not**
  re-run — confirmed unaffected by S-C (pure S-B physics), cited directly instead of
  reproduced.

## Phase 1.6 — Dataset and summary-statistics companions (nb22a, nb23a) — **COMPLETE** (2026-09-01)

Extended the core library first (`src/cstr_sbi/recycle/summaries.py`: `SC_CHANNELS`;
`src/cstr_sbi/recycle/simulator.py`: `generate_sc_observations()`, and optional
`observations_sc`/`sc_channels` support in `save_dataset`/`load_dataset`), then built the
`nb22a`/`nb23a` pair (both executed, no errors).

- **`notebooks/22a_wu2003_data_generation_sc.ipynb`**: augments `nb22`'s existing
  `data/wu2003_observations.npz` with the feed-analyzer channel — **no ODE re-simulation**
  (the reading is synthesized directly from each window's already-known `z_A0_eff`), so this
  ran in under a second rather than nb22's ~5-10 minutes. Verified `corr(true z_A0_eff,
  feed_analyzer window-mean) = 1.000000` across all 420 windows, max abs error `2.6e-5`, and
  the per-scenario breakdown exactly reproduces each scenario's true `z_A0_eff`
  (`0.7800±0.0001` for W10, `0.8000±0.0001` for W13, `0.9000±0.0001` everywhere else). Saved
  to `data/wu2003_observations_sc.npz`.
- **`notebooks/23a_wu2003_summary_statistics_sc.ipynb`**: built the 72-D S-C summary matrix
  for all 420 windows and reran the FIM comparison (corrected `cov_offdiag_corr`
  methodology, not `nb23`'s original un-inverted convention — flagged explicitly, since
  `nb23`'s own Section 7 is the source of the article's Figure 8 and has the same
  un-audited-metric issue `nb42` already found in `nb33`). **Result: S-C collapses both
  confounds to near-zero at both tested points** (`(alpha, z_A0_eff)`: S-B `-0.478`/`-0.596`
  → S-C `-0.013`/`-0.011`; `(eta_col, z_A0_eff)`: S-B `+0.761`/`+0.182` → S-C
  `+0.026`/`+0.008`) — this is the same result Phase 0 (`nb43`) found deterministically, now
  reproduced through the identical PCA/MI/FIM pipeline structure `nb23` established.
  **One honest caveat surfaced and documented in the notebook**: `sklearn`'s
  `mutual_info_regression` ranks `feed_analyzer_mean` only 18th of 72 features for
  `z_A0_eff` (a near-tie among the top ~18), *despite* its `r=1.0000` correlation — a known
  KNN-MI-estimator artifact for near-discrete features (`z_A0_eff` only takes 3 distinct
  values across the 14-scenario catalogue). Documented rather than glossed over; the
  FIM/correlation evidence is the trustworthy signal here, not the MI ranking.
- Saved `data/wu2003_summary_features_sc.npz` (`X_sb`, `X_sc`) for `nb24a`.

---

## Phase 2 — System II + analyzer: data and training (the expensive phase) — **REDIRECTED 2026-09-02, see update at the end of this section**

`notebooks/24a_wu2003_sbi_training_sc.ipynb` (`n_train=5000`, matched protocol, lognormal
prior, plain identity-space training, 3 seeds): the contingency below actually triggered, but
not for a residual-coupling reason — a training/z-scoring artifact, now partially fixed.

- **Plain identity-space training failed catastrophically at N_SBC=120 and N_SBC=400**, and
  `z_A0_eff` — the analyzer's own target parameter — was the single worst-failing parameter
  in every seed. Diagnosed, not assumed: rejection sampling was ruled out
  (`reject_outside_prior` True/False gave near-identical `z_A0_eff` results; the real leakage
  driver was `eta_col`/`beta_r`); the real cause was a flow-training artifact (biased *and*
  wider-than-theoretical-floor posterior for `z_A0_eff` — the flow hadn't converged onto the
  sharp, near-deterministic relationship).
- **Fix found and confirmed at the N=400 confirmation budget**: `z_score_x='none'` alone
  (combining it with more training patience made it worse) took `z_A0_eff` to
  `p=0.359, 0.670, 0.896` across 3 seeds — genuinely fixed, holds at high statistical power.
- **But this exposed a second, previously-masked problem**: at N=400, `alpha` (2/3 seeds) and
  `xi_reb` (2/3 seeds, different seeds) now fail — a reduced-budget pass (2/3 seeds at N=120)
  had hidden this, exactly the `finding_sbc_power_eta_col` pattern recurring for a different
  parameter pair. **0/3 seeds pass overall at N=400** — S-C is not yet a calibrated posterior,
  though the specific analyzer-related failure is resolved.
- **Revised next step**: diagnose `alpha`/`xi_reb` with the same rigor already applied to
  `z_A0_eff` (leakage check, bias/width check) before assuming `n_train=15000` (the user's
  original ask, matching `nb24`'s own precedent, still untested) is the right lever — more
  data did not fix `z_A0_eff`'s problem and may not fix this one either if it's a similar
  training artifact rather than genuine sample starvation.
- **Contingency clause below did not apply** (no new residual coupling was needed to explain
  the failure) but is kept for reference in case `alpha`/`xi_reb`'s diagnosis points there
  instead: if a *different* residual coupling appears on the S-C representation, rerun the
  nb39-41 diagnostic suite before concluding a training fix is even the right category of fix.
- Still not done: retraining a **protocol-matched S-B posterior** from the same paired bank
  for a fair Arm B vs. Arm C comparison — deferred until S-C's own calibration is settled.

### Update 2026-09-02 — `n_train=15000` tested and REJECTED; root cause diagnosed; Phase 2 redirected

- **`n_train=15000` (matched recipe, `z_score_x='none'`) is unambiguously worse than
  `n=5000`**: 0/3 seeds pass at `N_SBC=400`
  (`results/24a_sysII_sc_zscorenone_confirm_sbc_n400_ntrain15000.csv`); `z_A0_eff`
  regressed (one seed at `p=2.4e-5` after being robustly fixed at n=5000), `xi_reb`
  went catastrophic (`6.9e-10`, `2.0e-32`), `eta_col` dropped from 2/3 to 1/3 passing.
  This rules out the shrinkage-as-sample-starvation hypothesis: 3x more data should
  have monotonically helped, and should not have broken an already-fixed parameter.
- **Structural root cause identified (2026-09-02)**: the S-C summary vector contains a
  near-noiseless copy of `theta[4]` — the analyzer window-mean has noise sd `~8.2e-6`
  vs a prior sd of `~0.05`, a ~6000x conditional sharpening in one dimension. In
  log-density terms that dimension offers **~8.7 nats** of training-loss reduction vs
  **~1.2 nats** for `alpha`, so the flow's objective is dominated ~7:1 by sharpening
  one delta-like coordinate. This one mechanism is consistent with *every* 24a
  observation: `z_A0_eff` worst under default z-scoring, `z_score_x='none'` fixing it,
  the exposed `alpha`/`xi_reb` tail shrinkage, more data making dominance (and
  everything else) worse at fixed 60/3 capacity, and `beta_r` — the parameter least
  coupled to `z_A0_eff` — passing everywhere at every budget. **No hyperparameter
  sweep fixes a structurally dominated objective; two sweeps have now failed in
  opposite directions.**
- **Phase 2 is therefore redirected to a factorised posterior**
  (full staged plan, pre-registered pass gates, budgets:
  `docs/Manuscript/system_ii_sbi_solidification_plan.md`):
  `p(theta|x) = p(z_A0_eff | analyzer) x p(alpha, beta_r, eta_col, xi_reb | x)` —
  the first factor is an **exact analytic truncated-Gaussian** (calibrated by
  construction, SBC becomes a unit test); the second is a **4-D SNPE in logit space
  with the exact induced prior** (nb39's code, killing the `eta_col=1.0` boundary
  pileup — nb38's best single lever, never yet combined with the analyzer), trained
  on the same 72-D summaries (analyzer features stay in `x` as conditioning),
  `zuko_nsf` 60/3, `n_train=5000`, `z_score_x='none'` first.
- **Second targeted improvement, no new sensor**: `xi_reb` enters the physics *only*
  through the observed `Q_reb` channel (verified in `physics.py` — it divides the
  duty proxy and appears nowhere in the dynamics), so all its information sits in ~8
  heavy-tailed features. Add the closed-form feature
  `mean(log(Q_reb / (QREB_NOM * throughput_rat * V_norm)))`
  `= log(1 + 2*x_B_excess + 0.5*(1-eta_col)) - log(xi_reb)` — everything in it is
  already observed under S-B. Version the summary vector (`S-C.2`); regenerate the
  bank (~18 min at n=5000).
- **Before retraining, two cheap deterministic diagnostics gate the work**
  (solidification plan Stage 1): (a) the residual-confound `cov_offdiag_corr` sweep
  under S-C for the six pairs never checked (`(alpha,beta_r)` and `(xi_reb,eta_col)`
  first — the latter is structurally coupled through `Q_reb`'s
  `(1 + 2*x_B_excess + 0.5*(1-eta_col))/xi_reb` form); (b) a warm-start convergence
  audit — the matched steady-state solve's loosened tolerances (`t_final=100`,
  `rtol=1e-4`) are unverified for tail draws and could be corrupting training data
  exactly where the shrinkage occurs.
- **Conditional sensor decisions, pre-registered** (only if the Stage-1 FIM sweep
  finds a genuine residual confound; nb43-style deterministic feasibility check
  first, never build pre-emptively):
  - `(xi_reb, eta_col)` or `(alpha, eta_col)` residual → **bottoms-product
    composition analyzer `x_B`** ("S-D"; raw channel 4, zero new physics, the most
    industrially defensible sensor — product quality).
  - `(alpha, beta_r)` residual → **reactor-outlet composition analyzer `z_A`**
    (raw channel 11). nb43's rejection of this channel is **obsolete in combination
    with the feed analyzer**: with `z_A0_eff` pinned to ~1e-5, `z_A` becomes a clean
    `alpha` probe.
  - No existing sensor is moved; the feed analyzer's placement (fresh-feed line,
    **upstream of the recycle mixing point**) is load-bearing and unchanged.
- Do not promote any existing `sbi-logs/wu2003_posterior_sc_*.pkl` (any `n_train`,
  any `z_score_x`) — all are superseded by the factorised design. The pass gate for
  any future promotion is pre-registered in the solidification plan (>=4/5 seeds,
  `N_SBC=400`, `n_post=400`, two independent SBC sets, coverage@90, and a
  posterior-mean-vs-truth anti-shrinkage slope in `[0.9, 1.1]` per parameter).

---

## Phase 3 — Classification (Arm B: with analyzer)

- Rerun the `nb31`-equivalent fault classification pipeline (same 14+ named scenarios, same
  `fault_unit()` ground truth, same posterior-mass threshold convention) against the S-C
  posterior.
- Expect: feed class F1 moves off 0.000 (W10 correctly classified), W13's multi-fault
  detection improves, reactor-class performance stays comparable or improves.
- Deliverable: new confusion matrix / macro-F1 table, directly comparable to the existing
  S-B table.

---

## Phase 4 — Design and run the shared confound-exercising 30-day trajectory (Arms B & C)

**Important: the existing `nb27` 30-day trajectory does not exercise this confound at all** —
its degradation profile only varies `alpha` (catalyst decay) and `beta_r` (fouling); `eta_col`,
`xi_reb`, and `z_A0_eff` are held healthy throughout. That trajectory is evidence for a
*different* issue (Artefact 2, EKF's raw-trajectory advantage) and should stay as-is for that
purpose. **A new trajectory is needed** for this narrative, with a degradation profile that
drifts `z_A0_eff` (feed purity) — alone or jointly with `alpha`/`eta_col`, analogous to W10's
or W13's fault composition — sustained over the 360-window/30-day run.

- Design the profile (e.g., a linear feed-purity decline reaching `z_A0_eff≈0.78` by day 30,
  optionally combined with a mild concurrent `alpha` or `eta_col` drift for the compound
  case), reusing `nb27`'s harness structure.
- Run it once through the **S-C** posterior (Arm B) — expect accurate, unbiased tracking of
  every parameter including `z_A0_eff`.
- Run the *identical* trajectory (same noise seeds) through the **protocol-matched S-B**
  posterior from Phase 2 (Arm C) — expect biased/degraded tracking specifically for the
  confounded parameters, mirroring nb40's already-quantified posterior behavior.
- Deliverable: new notebook (e.g. `notebooks/44_wu2003_feed_confound_30day_tracking.ipynb`)
  producing both arms' trajectories from one shared harness, plus a comparison figure
  (analogous to `fig:sysII_track`) with per-parameter MAE/coverage for both structures side
  by side.

---

## Phase 5 — Full cascade re-run (per `notebook_execution_plan.md` Item 13)

Item 13 was blocked on Item 11 (S-B matched-protocol calibration) never succeeding; the S-C
fix removes that blocker. Rerun against the settled S-C posterior:

- `nb26` (headline banana/EKF failure) — recheck whether the original banana framing needs
  updating now that the feed-analyzer channel exists.
- `nb27` (sequential tracking) — rerun its *existing* alpha/beta_r trajectory against S-C too,
  for completeness (Artefact 2 is a separate question from this plan's confound, but worth
  confirming S-C doesn't regress it).
- `nb29`/`nb29b` (eta_col investigation, identifiability scan) — confirm `eta_col` calibrates
  and the FIM couplings resolve under S-C, using the corrected `cov_offdiag_corr` throughout.
- `nb30` (claims and conclusions) — rewritten synthesis incorporating the three-arm result.
- `nb32` (CNN embedding) — likely retired: its whole purpose was working around the exact gap
  S-C now closes directly. Keep only if a reviewer-facing "amortised estimator vs. richer
  representation" comparison is still wanted.
- `nb33` (FIM cross-system validation) — redo with `cov_offdiag_corr`, not `offdiag_ratio`,
  finally producing manuscript-traceable numbers (nb42 found the current cited numbers
  untraceable to any script).

## Phase 6 — Housekeeping: fix the Artefact 2/3 provenance gap regardless of outcome

Independent of whether Phases 1–5 proceed: `nb42` found `main.tex`'s cited Artefact 2/3
raw-trajectory numbers (`-0.07`, `4.36e3→5.48`, `41.4→5.24`) are not reproducible from any
tracked script and were almost certainly computed with the buggy `offdiag_ratio` metric.
This needs a dedicated re-validation (`nb39`/`40`/`41`-style: multiple points, grid-scan,
raw trajectory, actual posterior) for Artefact 2 `(alpha, beta_r)` specifically, since it is
currently reported as fully resolved and operationally benign and has not had the same
scrutiny Artefact 3 just received. Do this whether or not the analyzer rework proceeds.

---

## Phase 7 — Manuscript rewrite

- Methods: new §describing the S-C instrumentation (justify physically — feed-line
  composition analyzers, e.g. NIR/Raman/GC on the fresh-feed stream, are real industrial
  equipment; state the assumed noise level explicitly and cite a representative real
  analyzer's precision spec, per Phase 0's finding that precision, not just presence, is
  what matters here; state the practical cost/latency tradeoff honestly).
- PFD diagram update (manual, out of this plan's automatable scope).
- New three-arm comparison table/figure: System I vs. System II+analyzer vs. System
  II-no-analyzer, classification F1 and 30-day tracking MAE/coverage side by side — this is
  the narrative's central table.
- Rewrite §Artefact 3 (`sec:wu_artifact3`): from "representation-induced, not a plant-level
  non-identifiability" to "confirmed genuine limit, predicted by FIM, resolved by adding the
  predicted sensor" — a stronger, prescriptive-and-validated framing.
- Rewrite §Artefact 2 per Phase 6's outcome once that re-validation completes.
- Abstract/Conclusion: state the three-arm result as the paper's System II headline.
- Supporting Information: mirror all of the above (scenario table, FIM tables, classification
  tables, tracking figure).

---

## Rough sequencing and cost

| Phase | Wall-clock character | Blocking? |
|---|---|---|
| 0 | Minutes, deterministic-only | Gates everything else — do not skip |
| 1 | Minutes, code-only | Blocks Phase 2 |
| 2 | Multi-hour retrain + SBC confirmation (redirected 2026-09-02: factorised posterior per `system_ii_sbi_solidification_plan.md`, gated by its Stage-1 deterministic diagnostics) | Blocks Phases 3–5 (the critical-path bottleneck) |
| 3 | Fast (inference only, no training) | Independent of Phase 4 |
| 4 | Fast (inference only) once the trajectory is designed | Independent of Phase 3 |
| 5 | Mostly fast reruns; nb26/29b's identifiability re-checks are deterministic-only | Depends on Phase 2 |
| 6 | Deterministic-only, independent of everything else — can run in parallel with Phase 0–5 | None |
| 7 | Writing, depends on 3/4/5/6 all landing | Last |

**Bottom line:** this is comparable in scope to redoing System II's core pipeline with a third
instrumentation variant — not a small patch. The critical path is Phase 2's retrain; almost
everything else is either fast reruns of existing notebook logic against new posteriors, or
independent deterministic work that can proceed in parallel. Phase 0 and Phase 6 have no
retraining dependency and can start immediately.
