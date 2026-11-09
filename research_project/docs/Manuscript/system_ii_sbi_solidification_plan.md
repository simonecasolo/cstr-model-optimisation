# System II SBI Training & Calibration — Solidification Plan

**Status: proposed plan, written 2026-09-02 after reading `HANDOFF.md`,
`system_ii_analyzer_rework_plan.md`, `system_ii_sbc_repair_directions.md`, the
`24a_*` result CSVs, `/tmp/nb24a_n15000_run.py`, and the core library
(`recycle/{physics,priors,summaries}.py`, `scripts/sbi_pipeline.py`).**
Supersedes nothing by itself — it proposes the next plan of record for System II's
Phase 2 (S-C training/calibration), to replace the current "try another training
lever" loop with a structural fix, cheap pre-registered diagnostics, and explicit
sensor decisions.

---

## 1. Diagnosis: why calibration keeps breaking

Every System II failure to date falls into one of three buckets, and the project has
now empirically mapped all three. The plan below is organised around them.

### 1a. Genuine identifiability limits (solved by instrumentation, not training)

`(alpha, z_A0_eff)` (curved, sign-changing) and `(eta_col, z_A0_eff)` (single-signed)
are multiply-confirmed genuine limits of S-B (nb39–42). The feed-line analyzer (S-C,
`<=0.01%` relative noise) collapses both to near-zero (nb43, nb23a). **This bucket is
closed for the two known pairs — but has never been swept for the *remaining* pairs
under S-C** (see Stage 1a). `xi_reb` is cleared against `z_A0_eff` only.

### 1b. A structural training pathology introduced by the analyzer itself (new diagnosis)

The S-C summary vector contains a near-noiseless copy of `theta[4]`: the analyzer
window-mean has noise sd `~8.2e-6` (120 readings at `9e-5` sd), against a prior sd of
`~0.05`. The flow must therefore learn a conditional that is a **~6000× sharpening in
one dimension** while the other four need only ~2–25×. In log-density terms the
`z_A0_eff` dimension offers **~8.7 nats** of achievable training-loss reduction vs
**~1.2 nats** for `alpha`: the optimizer is paid ~7:1 to sharpen one delta-like
dimension rather than to get the tails of `alpha`/`xi_reb` right.

This single mechanism is consistent with *every* observation from the 24a series:

- `z_A0_eff` was the worst-failing parameter under default z-scoring (the flow
  chases but can't cleanly represent the near-delta; biased and wider than the
  theoretical floor) — observed.
- `z_score_x='none'` improved the conditioning of that one dimension and fixed
  `z_A0_eff` — observed.
- With `z_A0_eff` now trainable, the loss is still dominated by it; `alpha`/`xi_reb`
  conditionals are underfit in the tails → **shrinkage-toward-prior-mean**, the exact
  signature the 2026-09-02 diagnostic quantified — observed.
- More data (`n=15000`) lets the flow push the delta dimension even sharper,
  *increasing* gradient dominance → other parameters regress, and even `z_A0_eff`
  itself destabilises in one seed (capacity contention at fixed 60/3) — observed.
- `beta_r`, the parameter least coupled to `z_A0_eff` (nb42: its confound genuinely
  collapses at raw trajectory), passes everywhere at every budget — observed.

**Implication: stop asking a 5-D flow to represent a distribution with one
near-deterministic coordinate.** No hyperparameter sweep fixes an objective that is
structurally dominated by one dimension; this is why two sweeps have now failed in
opposite directions. The fix is to factorise the posterior (Stage 2).

### 1c. Secondary geometry/protocol aggravators (known, partially fixed, not yet combined)

- **Bounded-support pileup**: `eta_col`'s nominal (1.0) sits exactly on its prior
  upper bound and the truncated-lognormal prior piles ~half its mass against it.
  Logit-space training was the single best S-B result (nb38 §5) and nb39 already
  built the *exact* induced logit prior — but it has never been combined with the
  analyzer/factorised setup.
- **Warm-start convergence in the tails**: the matched-protocol steady-state solve
  uses loosened tolerances (`t_final=100`, `rtol=1e-4`). For tail draws (e.g.
  `alpha≈0.5`) convergence is not verified; a theta-dependent residual transient in
  `y0` makes the `theta→x` map irregular exactly in the tails — a second independent
  driver of tail shrinkage that more data also cannot fix. Never audited (Stage 1b).
  Fixed-y0 helping in nb38 §4 is consistent with this.
- **`xi_reb`'s information channel is the worst-behaved one**: verified in
  `physics.py` — `xi_reb` enters *only* through the observed `Q_reb` proxy
  (`Q_reb ∝ 1/xi_reb`; it appears nowhere in the dynamics, and S-B's Loop 3 controls
  `T_reb`, which does not depend on `xi_reb`). So all information about `xi_reb`
  lives in ~8 of 72 features (`Q_reb_*` stats, `reb_per_boilup`, `corr_Qreb_FR`),
  and `Q_reb`/`V_norm` are precisely the heavy-tail features nb35 flagged. A
  sharpened, physics-exact feature exists (Stage 2c) that requires **no new sensor**.

### 1d. Process/reproducibility debt (cheap, do first)

- The truncated-lognormal prior — the project's *actual* prior since nb37 — exists
  only as inline copies in notebooks and **`/tmp` scripts that will be wiped on
  reboot** (`/tmp/nb24a_n15000_run.py`, `/tmp/diagnose_alpha_xireb.py`). The library
  `run_sbc()` still defaults to the uniform prior.
- All 24a conclusions rest on **one fixed SBC-set realization**
  (`24a_sysII_sc_lognormal_sbc_set_n400.npz`). Good for paired comparisons; not
  sufficient for a promotion decision.
- SBC uses `n_post=120` with `N_SBC=400` (rank discreteness makes KS mildly biased;
  minor but free to fix at confirmation time).
- `nb24a`'s notebook outputs are stale relative to the authoritative CSVs.

---

## 2. The plan, staged with decision gates

### Stage 0 — Hygiene (half a day, no training) — **COMPLETE (2026-09-02)**

0.1 **Rescue `/tmp` scripts now**: copy `/tmp/nb24a_n15000_run.py` and
    `/tmp/diagnose_alpha_xireb.py` into `scripts/` (they are cited by `HANDOFF.md`
    as the authoritative record of two experiments). **Done** — both now live at
    `scripts/nb24a_n15000_run.py` and `scripts/diagnose_alpha_xireb.py`.
0.2 **Library-ise the prior**: move `TruncatedLogNormalPrior` (median/`LOG_SIGMA`
    constants included) into `src/cstr_sbi/recycle/priors.py` as
    `truncated_lognormal_5d()`; make every notebook/script import it. Add the exact
    induced logit-space prior from nb39 alongside it. **Done** — both classes
    (byte-identical logic to nb37/38/39's inline copies) now live in
    `src/cstr_sbi/recycle/priors.py` as `truncated_lognormal_5d()` and
    `induced_logit_prior()`, plus `theta_to_u`/`u_to_theta` helpers and the
    `NOMINAL_THETA_5D`/`LOGN_SIGMA_5D` constants. Verified: `induced_logit_prior`'s
    `log_prob` is self-consistent with `truncated_lognormal_5d`'s via the analytic
    change-of-variables identity (checked numerically, exact to floating point).
    Future notebooks/scripts should import from here rather than re-pasting the
    class — `scripts/stage1c_second_sbc_set.py` already does.
0.3 **Pre-register the pass gate** (in this file, before any new training):
    a posterior is *calibrated* iff — all inferred parameters at KS `p>0.05`,
    in **≥ 4/5 seeds**, at `N_SBC=400` with `n_post=400`, on **two independent**
    SBC sets; plus 90% coverage in `[0.85, 0.95]` per parameter; plus per-parameter
    posterior-mean-vs-truth regression slope in `[0.9, 1.1]` (the anti-shrinkage
    metric — this is the diagnostic that caught `alpha`, promoted to a standing
    gate). Anything less is "not calibrated"; no post-hoc softening.

### Stage 1 — Cheap deterministic diagnostics (≤ 1 day, no SNPE training) — **IN PROGRESS (launched 2026-09-02)**

All three sub-items launched in parallel (mutually independent per §5's sequencing table):
`scripts/stage1a_sc_residual_confound_sweep.py` (1a), `scripts/stage1b_warmstart_convergence_audit.py`
(1b), and `scripts/stage1c_second_sbc_set.py` (1c).

**1c — done (2026-09-02).** `results/44_sysII_sc_lognormal_sbc_set_n400_setB.npz`
(seed=20260904, distinct from 24a's n=400 set's seed 20260903): 400/400 draws, **zero
rejected draws** (400 attempts for 400 rows — same clean behaviour as the first set), no
NaNs, theta means close to prior median as expected. Any future confirmation run should
report pass/fail on both this set and `results/24a_sysII_sc_lognormal_sbc_set_n400.npz`.

**1a — done (2026-09-02), with an important caveat requiring one more targeted check
before acting on it.** `scripts/stage1a_sc_residual_confound_sweep.py`
(`results/44_sc_residual_confound_sweep.csv` + `..._rawcheck.csv`): swept all six
untested pairs at 4 points (`nominal`, `alpha_displaced` alpha=0.82,
`eta_col_interior` eta_col=0.85, `W2_like` alpha=0.85). Five pairs never exceed
`|corr|=0.42`. **`(alpha, eta_col)` is the one exception**: `|corr|` 0.52-0.66 in the
72-D S-C summary at all 4 points, and **survives** the raw-trajectory check
(`|corr|` 0.42-0.60, confirmed stable across two independent noise-seed offsets) at 3
of the 4 points (`alpha_displaced`, `eta_col_interior`, `W2_like`; the `nominal` point
is weaker, `-0.42` raw, just under the 0.5 survival bar).

This looked, at first read, like it might be in tension with the already-**retracted**
`(alpha, eta_col)` "banana" (`finding_sa_headline_retracted` memory; `nb29b` Section 1,
S-B, scenario W12: `alpha=0.75, eta_col=0.80`) — but a direct follow-up check
(`scripts/stage1a_gridscan_crosscheck.py`, `figures/44_alpha_etacol_w12_sc_gridscan.png`)
resolves this: rerunning `nb29b`'s exact grid-scan-distance methodology at its exact W12
point, under S-C, **reproduces the same collapse** — the classic F_R-only diagonal
ridge visibly narrows to a compact blob under the S-C "robust" combined feature set and
to essentially the single truth cell under the full 72-D S-C summary. So there is
**no contradiction**: the retraction at W12 still holds under S-C too; Stage 1a's FIM
finding is simply describing *different* operating points (none of Stage 1a's 4 points
is W12 — none jointly displaces both `alpha` and `eta_col` the way W12 does), consistent
with this project's now-repeated pattern that these couplings are point-dependent
(`(alpha, z_A0_eff)`'s sign flip being the clearest precedent).

**What is NOT yet resolved**: whether Stage 1a's own trigger points
(`alpha_displaced`, `eta_col_interior`, `W2_like`) themselves show a real, ridge-shaped
degeneracy under the grid-scan/distance methodology, or whether the FIM's local
correlation at those points is a linearisation artifact that a full nonlinear grid-scan
would not confirm (the `(alpha, z_A0_eff)` sign-discrepancy precedent, resolved by
`nb41`, shows both directions are possible). **Do not treat Gate A as triggered
(i.e. do not commit to the bottoms-product "S-D" sensor) until a grid-scan is run
centered at one of those three points** (e.g. `alpha_displaced`:
`[0.82,1,1,1,0.90]`) under S-C's combined features — this is the specific next
diagnostic, cheap and deterministic like the rest of Stage 1, not yet run.

**RESOLVED 2026-09-04, Gate A NOT triggered.**
`scripts/stage1a_gate_a_tiebreaker.py` ran the grid-scan at all three of Stage 1a's
own trigger points (`eta_col_interior`, `W2_like`, `alpha_displaced`), not just W12.
**All three show the identical clean-collapse pattern**: the F_R-only panel
reproduces the classic diagonal banana ridge (as always), but the full-72D panel
shows essentially the *entire* grid at maximum distance — no extended ridge, not
even a compact blob near truth (visually confirmed on the `W2_like` figure; the
"robust combined" panel shows a residual compact blob, cleaner than F_R-only but
not as clean as full-72D). Quantified: fraction of the 25x25 grid with normalised
distance <1.0 is `0.000` for the full-72D representation at every point tested —
the same signature as the already-retracted W12 banana. **Conclusion: Stage 1a's
FIM correlation (`|corr|=0.64-0.66`) at these points is a linearisation/local-
Jacobian artifact, not a genuine plant-level identifiability limit** — it does not
survive this project's most-trusted, representation-independent tiebreaker method.
**`(alpha, eta_col)` is cleared. No genuine residual confound was found among any
of the six untested pairs under S-C.** Per §3's sensor-decision table, **no new
sensor (bottoms-product `x_B`, or otherwise) is currently justified** — Gate A's
conditional trigger never fired. The `alpha`/`xi_reb` calibration problems
documented elsewhere in this file are training/representation issues, not
missing-instrumentation issues.

**1b — CONFIRMED POSITIVE, stopped by user decision at n=45/80 (2026-09-02).**
The hypothesis is upheld, and more strongly/pervasively than the plan anticipated.
`results/44_sc_warmstart_convergence_audit.csv` (45 checkpointed draws before the
run was deliberately killed — see below), `figures/44_warmstart_convergence.png`:

- **100% of 45 draws (not just tail draws) show `summary_shift_rel_l2` above the
  0.3% sensor-noise floor** — min `0.97%` (still 3.2x the floor), median `19.5%`,
  mean `22.4%`, max `66.2%`. This is 1-2 orders of magnitude larger than the
  intended sensor noise, and it is NOT confined to the prior's tails: e.g.
  `alpha=1.002` (essentially nominal) shows a `3.3%` shift.
- `corr(summary_shift, |alpha-1|) = 0.55` — displacement from nominal makes it
  worse, but does not explain most draws' contamination (median draw has
  `alpha` within `0.04` of nominal and still shows `~20%` shift).
- `corr(summary_shift, |z_A0_eff-0.90|) = -0.13` — not the driver.
- **Root-cause reframing**: this is not "only tail draws are corrupted" (the
  plan's original, narrower hypothesis) but **every training/SBC bank generated
  under the matched protocol since `nb22`/Major-Comment-7 has summary vectors
  computed from an initial condition that is not actually at steady state**, by an
  amount that dwarfs the assumed observation noise. `z_A0_eff`'s posterior is the
  one parameter that never routes through this contaminated path (it is read
  directly off `theta[4]` by the analyzer, bypassing simulation) — consistent
  with it being the one parameter that trains cleanly.
- **Run was deliberately stopped at n=45/80** (user decision) after two draws
  (38, 39) took 68 min and 4.3 h respectively — the `max_steps=400_000` cap
  bounds solver *steps*, not wall clock, so a handful of draws can still make
  total runtime open-ended. n=45 was judged sufficient: this is a direct,
  bright-line-threshold magnitude comparison (45/45 unanimous, most by 1-2 orders
  of magnitude), not a small-N p-value/SBC-power situation
  (`finding_sbc_power_eta_col` doesn't apply here — that finding is about KS-test
  power at low N, not about point-estimate reliability at n=45).

**This finding now supersedes 1a as the top priority.** Building the Stage 2
factorised posterior, or resolving Gate A's `(alpha,eta_col)` question, is
premature while the training-bank ground truth itself is contaminated at this
scale — any calibration result obtained on the current banks is not
trustworthy regardless of architecture. See "Immediate next action" below.

1a. **Residual-confound FIM sweep under S-C** (HANDOFF's own recommended next step):
    `cov_offdiag_corr` (never `offdiag_ratio`) over the six untested pairs —
    `(alpha,beta_r)`, `(alpha,eta_col)`, `(alpha,xi_reb)`, `(xi_reb,eta_col)`,
    `(xi_reb,beta_r)`, `(beta_r,eta_col)` — at ≥ 4 operating points spanning the
    prior (nominal, alpha-displaced, eta_col-interior, a W2-like point), on the
    72-D S-C summaries, with a raw-trajectory cross-check for any pair with
    `|corr| > 0.3`. Priority: `(alpha,beta_r)` (prior history via nb29b, and the
    66-D S-B prefix is unchanged in S-C) and `(xi_reb,eta_col)` (structurally
    coupled through `Q_reb`'s `(1 + 2·x_B_excess + 0.5·(1−eta_col))/xi_reb` form).
    *Output:* a table that says whether any genuine confound remains that training
    can never fix — this gates the sensor decisions in §3.
1b. **Warm-start convergence audit**: for ~200 lognormal prior draws, compute
    `‖rhs(y0_matched)‖` at the warm-start output and the summary shift between
    `t_final=100` and `t_final=300` (tight tolerances). Correlate both against
    `|alpha−1|`, `|z_A0−0.9|`. If tail draws show summary shifts above the sensor
    noise floor, the matched bank is systematically corrupted in the tails —
    fix by iterating the steady-state solve to a residual criterion (or Newton via
    optimistix) before regenerating any bank. If clean, cross the hypothesis off.
1c. **Second SBC-set realization**: generate an independent `N=400` lognormal SBC
    set (different seed) once, now — every future confirmation run uses both.

### Immediate next action (added 2026-09-02, supersedes Stage 1a/2 ordering below)

Stage 1b's confirmed, pervasive warm-start contamination (see above) means the
training-bank ground truth itself is suspect, independent of everything else in
this plan. Next steps, in order, before touching Stage 1a's Gate-A question or
Stage 2's factorised posterior:

1. **DONE (2026-09-03).** Implemented in `scripts/sbi_pipeline.py`
   (`_scenario_steady_state_checked`, `_matched_y0`, module comment above
   `MATCHED_SS_RESIDUAL_TOL` has the full writeup). Root cause turned out to be
   sharper than "loose tolerance": a direct test confirmed `diffrax.diffeqsolve(...,
   throw=False)` returns a plausible-*looking* but not-actually-converged state
   when `max_steps` is exhausted before `t_final` — it does not return NaN, and
   the old code discarded `sol.result` and used `sol.ys[-1]` unconditionally, so
   an incomplete integration was silently accepted as a genuine steady state for
   every theta the old loose budget (`t_final=100, rtol=1e-4, atol=1e-6,
   max_steps=100_000`) couldn't finish in time. Considered Newton
   (`optimistix` is already a dependency) but a quick scalar sanity check showed
   its default termination check can spuriously report non-convergence at
   float32 precision for tolerances tighter than ~1e-5/1e-7 (the library
   compares against float32 roundoff, not a bug in our code) — went with the
   simpler, more robust fix instead: (a) tighten to `nominal_warm_start`'s own
   already-trusted recipe (`rtol=1e-6, atol=1e-8, t_final=200`, unchanged
   `max_steps=100_000` to keep worst-case wall-clock bounded to the same order
   as before), and (b) add an explicit convergence check (`sol.result ==
   successful` AND residual norm `< MATCHED_SS_RESIDUAL_TOL=0.05`, the latter
   set relative to `nominal_warm_start`'s own measured nominal-theta residual
   `~0.0114`, i.e. a ~4x margin on the standard this project has already
   trusted elsewhere for years). Non-convergent draws now return `None` from
   `_matched_y0`, flowing into the *existing* invalid-draw skip/resample path
   every bank-generation loop already has — no call-site changes needed beyond
   one added `None` check in `_simulate_summary`.
   **Validation (`scripts/stage1b_fix_validation.py`,
   `results/44_stage1b_fix_validation.csv`, all 45 draws, ~18 min total —
   1-2 orders of magnitude faster than the original audit's runaway cost):**
   - **The core fix works**: both originally-pathological draws (4095s and
     15547s / 4.3h under the old recipe) now converge cleanly in under 10s
     each, with acceptable residual. No draw took more than ~25s. Bounding
     `max_steps` at a fixed budget while checking convergence explicitly, as
     designed, eliminated the runaway-cost failure mode entirely.
   - **New finding, not anticipated**: only **16/45 (36%) draws pass the new
     convergence check** — a substantial, previously-invisible rejection rate
     (the old code accepted 45/45, i.e. 100%, silently). This has a real bank-
     generation cost implication (~2.8x more raw prior draws needed for the
     same `n_train`) that needs to be measured properly in Step 3, not
     assumed away.
   - **Follow-up cross-check (`scripts/stage1b_fix_validation_refcheck.py`,
     `t_final=500, max_steps=1_000_000`, on the 16-draw accepted subset) landed
     a more serious, unanticipated result: even with the reference now
     genuinely converged (`ref_converged=True`, its own residual 0.006-0.037,
     same ballpark as the "accepted" solve's), the downstream S-C summary
     still shifts 3.4-32% (mean 21%, median 21%, n=10/16 so far) between the
     two solves — 10-100x the 0.3% noise floor, on EVERY draw tested.** This
     means the raw RHS-residual-norm criterion (however tight) is not a
     reliable proxy for "the downstream 2h summary has stabilized" — two
     states can both have small instantaneous derivative while still
     differing enough along some slow/near-null direction (plausibly an
     integral-controller state, or a slowly-relaxing thermal mode) to produce
     a meaningfully different trajectory. `MATCHED_SS_RESIDUAL_TOL`'s specific
     value (0.05) is not the problem — tightening it further would not fix
     this, since even ~0.01-0.02-level residuals on both sides still show
     large summary shifts.
   - **Implication**: Step 1's fix v1 was only half done. It correctly fixed
     the *runaway-cost* bug (no more multi-hour hangs; non-convergent draws
     are now correctly rejected rather than silently accepted) — that part is
     solid and kept. But the *acceptance criterion* needed to change.
   - **Fix v2, implemented (2026-09-03)**: replaced the RHS-residual check
     with a direct downstream-summary Cauchy check — integrate in
     `SS_CHUNK_T=200`-sized chunks (chained, not restarted), recompute a
     deterministic S-B summary probe after each chunk, accept once it
     stabilizes between successive chunks (`SS_MAX_CHUNKS=8` cap). First cut
     of this compared the FULL 66-D summary and **immediately failed its own
     sanity check**: nominal theta (known-converged, matches
     `nominal_warm_start` exactly) didn't stabilize within 8 chunks either.
     Root cause traced directly (not guessed): the full-vector shift is
     dominated by `_slope`/`_std`/`corr_*` features, which are numerically
     unstable on a near-flat/already-converged signal — **the exact same
     pathology `identifiability_scan.py`'s `robust_feature_subset` already
     exists to work around elsewhere in this project.** Confirmed: at nominal
     theta, the robust (means/physics-feature) subset shows shift
     `~0.00001` after just the *first* chunk, while the full vector bounces
     `0.05-0.29` across all 8 chunks from slope/corr noise alone. Fixed by
     comparing only the robust subset (`_robust_sb_idx()` in
     `sbi_pipeline.py`, duplicated from `identifiability_scan.py`'s exact
     definition rather than importing that CLI-oriented module). Nominal
     theta now converges correctly in 2 chunks.
   - **Full 45-draw re-validation, DONE**: `results/44_stage1b_fix_validation.csv`.
     **43/45 (95.6%) draws now converge** — up from 16/45 (36%) under fix v1's
     residual criterion, and a huge improvement on the old code's silent
     100% acceptance of everything including contaminated draws. Cost stayed
     bounded throughout: every draw finished in 2-30s (median ~9s), no runaway
     multi-minute/hour cases at all. (Note: this validation script's own
     "new_shift(vs ref)" column is stale from before the robust-feature
     correction — it still compares FULL summary vectors against a reference
     that often doesn't converge in its own limited budget, producing the
     same ~20-24% noise-artifact numbers seen before fix v2; disregard that
     column specifically, it does not reflect the corrected methodology.)
   - **Final confirmatory check, DONE — clean pass**
     (`scripts/stage1b_fix_v2_final_check.py`, 5 randomly-sampled converged
     draws vs. a genuinely-converged high-budget reference `t_final=500,
     max_steps=1e6`): **robust-feature shift is 0.0002%-0.02% across all 5
     draws** — 1-2 orders of magnitude BELOW the 0.3% sensor-noise floor,
     including the one draw (38) where the reference's own raw residual
     hadn't technically "converged" yet (robust shift there: 0.0006%,
     unaffected). `full_shift` stayed large (15-40%) for all 5, confirmed to
     be the same slope/std/corr noise artifact, not a real discrepancy — the
     physically-meaningful content of the summary is what was checked and it
     agrees with the reference far better than the noise floor requires.
   - **Step 3 (bank regeneration) STARTED, 2026-09-03.** `notebooks/24a_wu2003_sbi_training_sc.ipynb`
     updated in full: a new top-of-notebook markdown cell documents both fix iterations end
     to end, every path/constant in Sections 1-8 now uses `SUFFIX='_stage1bfix'` and
     `N_TRAIN=5000` (not the stale 15000), the prior import switched to the library
     (`truncated_lognormal_5d`), and every historical result cell got an explicit "obtained
     on the OLD contaminated bank, not a current finding" marker rather than being deleted.
     Actual generation running via `scripts/regenerate_sc_bank_stage1bfix.py` (background,
     under `caffeinate -i -s`, checkpointed every 250 rows) — **not** via `nbconvert`,
     matching this notebook's own established hang-avoidance precedent. **Reduced-first-pass
     decision**: started at `N_TRAIN=1000` (not 5000 — the ~14-16h full estimate was judged
     too slow for a first read). **Training bank done (2026-09-04)**:
     `data/wu2003_sbi_train_sc_lognormal_true_n1000_stage1bfix.npz`, 1000 rows, 91.5% accept
     rate — but cost was **~21.15h, not the ~2.8h estimated**: per-draw rate degraded over
     the run (40s to 70s+/draw in later windows, never stabilizing) rather than matching the
     smaller 45-draw validation sample's ~9-11s/draw median. The "reduced n first" attempt
     to get a fast turnaround did not deliver one -- worth investigating why later (draws
     late in a fixed-seed lognormal sequence clustering in more expensive regions by chance,
     vs. a genuine resource issue over a 20+ hour single process) but not blocking further
     progress right now. Both SBC sets (120+400 draws) generating next; given the observed
     rate this could take several more hours, not the ~1.5-2h originally estimated.

     **All three artifacts DONE (2026-09-04).** Total wall-clock for the whole run:
     83,170s (~23.1h). Verified: correct shapes (`(1000,5)`/`(1000,72)` train,
     `(120,5)`/`(120,72)` and `(400,5)`/`(400,72)` SBC sets), no NaNs. **Important new data
     point**: the two SBC sets, generated *after* the slow training bank in the same
     process, ran at a normal ~13s/draw — matching the small-sample validation, not the
     degraded 40-70s/draw seen late in the training bank. This weighs against "systemic
     resource degradation over a long-running process" and toward "that particular
     training-bank seed sequence happened to cluster expensive draws by chance" — worth
     confirming with a profiling pass later, but not blocking further progress now.

     Sections 4-8 (SNPE training + SBC) are updated and ready — running next.

   - **Sections 4-8 run against the clean n=1000 bank (2026-09-04),
     `scripts/run_nb24a_stage1bfix_training.py`.** Hit and fixed one real bug first: the
     Stage 0.2 library-ised `truncated_lognormal_5d()`/`induced_logit_prior()` returned
     instances of classes NESTED inside their factory functions -- unpicklable, so
     `pickle.dump` on any SNPE posterior built with this prior crashed
     (`Can't pickle local object`). Fixed by moving `TruncatedLogNormalPrior`/
     `InducedLogitPrior` to module level in `src/cstr_sbi/recycle/priors.py` (verified:
     pickle round-trip now works for both). A 0-byte corrupt `.pkl` from the crashed attempt
     was deleted before rerunning.
     - **Training+SBC itself is cheap** (54s total for both variants x 3 seeds x both
       budgets) -- confirms the earlier finding that bank generation, not training, is
       this pipeline's cost bottleneck.
     - **Baseline (default z-scoring): still fails catastrophically** — 0/3 pass at both
       budgets, same qualitative pattern as the old contaminated-bank result (`z_A0_eff`
       and `alpha`/`xi_reb`/`beta_r` all fail in most seeds; `eta_col` passes cleanly in
       all 3, also matching the old pattern). **This is informative on its own**: the
       baseline z-scoring pathology is NOT an artifact of the warm-start bug — it persists
       on genuinely clean ground truth, consistent with the structural "near-delta
       `z_A0_eff` dimension dominates the flow's loss" diagnosis.
     - **`z_score_x='none'` fix: partially better, not resolved** — 3/3 pass reduced
       (`N=120`), **1/3 pass confirm (`N=400`)**, up from 0/3 at `n_train=5000` on the old
       bank, but still failing overall. Per-parameter detail is genuinely different from
       the old result, not just "the same problem, slightly better": `xi_reb` now fails in
       seed 1 specifically (`3.9e-5`); seed 2 fails on **three** parameters at once
       (`alpha=0.042`, `beta_r=0.014`, `z_A0_eff=0.028`) — notably `z_A0_eff` itself, the
       parameter that was *robustly* fixed at `n_train=5000` on the old bank
       (`p=0.359,0.670,0.896` there), now fails in one of three seeds here.
     - **Important confound, not yet resolved**: this comparison changed TWO things at
       once relative to the historical `n_train=5000` result — the warm-start fix AND a
       5x smaller training set (`n_train=1000` vs `5000`). `z_A0_eff` regressing in one
       seed is exactly the kind of thing reduced training data alone could cause,
       independent of the warm-start fix's own effect. **Cannot yet attribute the
       per-parameter differences to "cleaner ground truth" vs. "less training data"** —
       need either an `n_train=1000` run on the OLD contaminated bank (isolate the
       warm-start-fix effect at matched size) or an `n_train=5000` run on the NEW clean
       bank (matched to the historical comparison) before drawing a confident conclusion.
       Given the bank-regeneration cost problem (Step 3 above), the cheaper of the two is
       generating an `n=1000` bank under the OLD (broken) recipe for a matched-size
       comparison — trivial computationally since the old recipe is ~10-20x cheaper per
       draw.
     - **Net read so far**: the warm-start fix has NOT obviously resolved the
       `alpha`/`xi_reb`/`beta_r` shrinkage (still failing, still multiple parameters
       implicated), and if anything surfaced a new wrinkle for `z_A0_eff` — but this is
       confounded with the `n_train` drop and should not be over-read either way until the
       matched-size comparison above is run.

   - **Matched-size (`n_train=1000`) comparison run, isolating the fix's own effect**
     (`scripts/gen_old_recipe_bank_n1000.py` + `scripts/run_oldrecipe_n1000_training.py`,
     the OLD broken recipe + the EXISTING old-recipe SBC sets, same architecture/protocol
     otherwise). **Also hit and fixed a second, unrelated, pre-existing repo-integrity bug
     along the way**: `import cstr_sbi` failed in a fresh interpreter
     (`cstr_sbi/__init__.py` had a dead `from cstr_sbi import physics` referencing a
     top-level module already deleted in an earlier, uncommitted legacy-code cleanup — not
     caused by this session; fixed by dropping the dead import; see `HANDOFF.md`'s
     "Repo-integrity note"). Full three-way `z_score_x='none'` confirm-budget
     (`N=400`) comparison, per-parameter min-KS p-value:

     | | alpha | beta_r | eta_col | xi_reb | z_A0_eff | seeds passing |
     |---|---|---|---|---|---|---|
     | OLD recipe, `n=5000` (historical) | 2/3 fail | 3/3 pass | 3/3 pass | 2/3 fail (diff. seeds) | **3/3 pass** (`0.36-0.90`) | 0/3 |
     | OLD recipe, `n=1000` (matched-size) | **2/3 catastrophic fail** (`~1e-5, 1e-12`) | 2/3 pass | 3/3 pass | 3/3 pass | 2/3 pass, 1 fails | 0/3 |
     | NEW (fixed) recipe, `n=1000` | 2/3 pass, 1 marginal fail (`0.04`) | 2/3 pass, 1 fail | 3/3 pass | 2/3 pass, **1 catastrophic fail** (`4e-5`) | 2/3 pass, 1 fails (`0.03`) | 1/3 |

     **Genuinely mixed result, not a clean story either way**:
     - `alpha` improves substantially at matched `n=1000` when the warm-start is fixed
       (catastrophic `~1e-5`/`~1e-12` under the old recipe → `0.16-0.80`, only one marginal
       miss) — this is a real, attributable effect of the fix, not an `n_train` artifact,
       since both rows being compared are `n=1000`.
     - `xi_reb` gets *worse* in one seed under the new recipe (`0.34-0.78` all-pass under
       old recipe at `n=1000` → one catastrophic `4e-5` fail under the new recipe) — a
       result in the *opposite* direction, though only 1/3 seeds.
     - `z_A0_eff`'s robustness at `n=5000`/old-recipe (`3/3 pass, 0.36-0.90`) does NOT
       replicate at `n=1000` under **either** recipe (old or new) — strong evidence that
       `z_A0_eff`'s earlier robustness was substantially an `n_train=5000` effect (more
       data sharpens the near-delta dimension), not something the warm-start fix
       controls either way.
     - Overall pass count nudges up (0/3 → 1/3) but neither recipe is close to the
       ≥4/5-seed pre-registered gate, and 3 seeds is too few to trust any of these
       per-parameter swings individually (`finding_sbc_power_eta_col`'s lesson: don't
       read small-seed-count differences as settled).
     - **Bottom line**: the warm-start fix is not a magic fix for the `alpha`/`xi_reb`
       shrinkage story — it helps `alpha`, may hurt `xi_reb`, and `z_A0_eff` needs more
       training data regardless of which recipe is used. The honest next step is the
       pre-registered 5-seed/`N_SBC=400`/two-SBC-set confirmation at `n_train=5000` on the
       Stage-1b-fixed bank (Step 3's original target, not yet run) — no smaller-`n` pilot
       can substitute for that, given how much these numbers moved between `n=1000` and
       the historical `n=5000` even before considering the fix's own effect.

   - **Factorised-posterior pilot, run on the EXISTING n=1000 bank (no new generation
     needed)**, `scripts/run_factorised_posterior_pilot.py`: exact analytic
     truncated-lognormal-x-Gaussian posterior for `z_A0_eff` (1-D grid integration) +
     4-D SNPE flow in logit space (exact induced prior) over
     `(alpha, beta_r, eta_col, xi_reb)`, conditioned on the full 72-D summary. **Found
     and fixed a real bug first**: the analytic posterior initially failed its own
     calibration unit test (`mean_rank_frac=1.0000`, i.e. systematically biased) —
     traced to `compute_summaries`'s uniform `_compress_heavy_tails` (`sign(x)*log1p
     (|x|)`) being applied to the analyzer reading too; fixed by inverting with
     `expm1` before using the reading as the likelihood mean. **After the fix, the
     analytic posterior is genuinely calibrated** (confirm-budget `p=0.338`, mean
     rank `≈0.52`; the reduced-budget's marginal `p=0.025` is small-N SBC-evaluation
     noise, not a real miss, per `finding_sbc_power_eta_col`).
     - **4-D flow result: 2/3 pass reduced, 0/3 pass confirm** — no better on raw
       pass-count than the non-factorised fixed-recipe result (1/3), though the
       *failure shape* differs: no catastrophic (`~1e-5` or below) `xi_reb` miss
       this time (worst `0.021`, mild); but `alpha` has one severe miss
       (`6.8e-13`, seed 0) and `eta_col` — which has passed comfortably in
       *every* other variant tested so far (monolithic, either recipe, either
       `n`) — shows its first-ever marginal failure (`0.023`, seed 1), plausibly
       the logit-space reparameterisation adding noise for a parameter that
       didn't need the fix it was designed for (killing the `eta_col=1.0`
       boundary pileup, irrelevant if `eta_col` wasn't piling up here).
     - **Honest verdict: inconclusive at this sample size, not a clean win.**
       Removing `z_A0_eff` from the flow's training objective did not produce an
       obviously better-calibrated 4-D posterior than the monolithic 5-D one at
       matched `n_train=1000` — both hover around 0-1/3 seeds passing confirm,
       with different (not obviously "better" or "worse") failure patterns.
       Three seeds is not enough to distinguish "factorisation helps a little" from
       "these are both noise at this n". **Neither the warm-start fix alone nor the
       factorised architecture alone resolves the calibration story at `n=1000`** —
       consistent with the plan's own original expectation that `n_train=5000` (not
       1000) combined with the factorised design was the actual target, not
       something a 1000-row pilot could fully validate.
     - **What this pilot DID establish, usefully, before committing to the expensive
       n=5000 run**: the factorised architecture's code path works end-to-end
       (exact posterior + 4-D flow + SBC, all correct after the one bug fix), so the
       n=5000 run — whenever it happens — can go straight to the real architecture
       without a separate "does the code work" risk.
   - **Step 1 is DONE.** The warm-start fix now has (a) a bounded-cost
     guarantee (no runaway draws — every one of 45 test draws finished in
     2-30s, vs. 1-2 draws previously taking 1-4+ hours), (b) a 95.6%
     acceptance rate (vs. the old code's unchecked 100%, which included
     contaminated draws, or fix v1's overly-conservative 36%), and (c) an
     acceptance criterion (robust-feature summary stabilization) now
     independently confirmed to track the actual quantity that matters for
     downstream SBI training, to within 1-2 orders of magnitude below the
     noise floor. Ready to move to Step 2 (a fresh, cheap SBC-relevant
     re-validation) and then Step 3 (bank regeneration) — the ~1.05x
     resampling overhead from the 4.4% rejection rate is a small, well-
     characterized bank-generation cost, not the ~2.8x fix v1 would have
     needed.
2. **Validate cheaply on ~20-50 draws** (reuse this audit's harness) that the
   fix drops `summary_shift_rel_l2` to ≲0.3-1% (at/below the sensor floor)
   before committing to anything expensive. If it doesn't, the root cause is
   not (only) the warm start and needs re-diagnosis.
3. **Only then regenerate affected banks** — every S-B/S-C training + SBC bank
   generated under the matched protocol (`scenario_specific_warm_start=True`)
   since `nb22`. This is the expensive step; do not skip the cheap validation
   in (2) first.
4. **Re-run Stage 1a's `(alpha,eta_col)` grid-scan tiebreaker and the Stage 2
   factorised-posterior retrain on the *regenerated* banks**, not the current
   ones — both were designed/tested against potentially-contaminated ground
   truth.
5. **Flag beyond System II**: the matched protocol has been the project
   standard since the Major-Comment-7 fix (Item 11) — worth a quick sanity
   check on whether any already-reported System I/II manuscript number depends
   on a matched-protocol bank generated before this fix, not just the S-C work
   in flight.

### Stage 2 — Restructure the inference (the core move; ~1 day compute)

Replace the monolithic 5-D flow with a **factorised posterior**:

```
p(theta | x) = p(z_A0_eff | x_analyzer) · p(alpha, beta_r, eta_col, xi_reb | x)
               ^ exact, analytic          ^ 4-D SNPE, logit-space, x includes the
                 (truncated-Gaussian        analyzer features as conditioning input
                 conjugate update:
                 mean = analyzer window
                 mean, sd ≈ 8.2e-6)
```

2a. **`z_A0_eff` leaves the flow entirely.** Its posterior given the analyzer
    channel is a known truncated-Gaussian (reading noise is Gaussian by
    construction in `feed_analyzer_reading`); implement it in closed form in the
    library. It is calibrated *by construction* — SBC for it becomes a unit test,
    immune to bank size, seeds, and z-scoring. This removes the 8.7-nat delta
    dimension from the training objective, which is the structural fix for §1b.
2b. **4-D flow over `(alpha, beta_r, eta_col, xi_reb)`**, trained on the same
    72-D S-C summaries (analyzer features stay in `x` — near-equivalent to
    conditioning on `z_A0_eff`, so cross-correlations with `z_A0_eff` are captured
    through the conditioning). Recipe: exact-induced-prior **logit-space**
    parameterisation (kills the `eta_col=1.0` boundary pileup — nb38's best
    single lever, never yet combined with the analyzer), `zuko_nsf` 60/3 unchanged
    (per `feedback_network_sizing` — size is not the lever), `n_train=5000` (the
    15k bank is evidence-worse; don't scale until 5k passes),
    `z_score_x`: start with `'none'` (the empirically validated choice), fall back
    to a *fixed* robust affine pre-scaling of the bank (median/IQR, computed once,
    applied identically at train/test) if 4-D training shows scale pathology —
    this combination was never tested (nb38's preprocessing sweep predates the
    `z_score_x` discovery).
2c. **Sharpened `xi_reb` feature (no new sensor) — UPGRADED, empirically validated
    2026-09-04.** The plan's original single-feature proposal,
    `Q_ratio_feat = mean(log(Q_reb / (QREB_NOM · throughput_rat · V_norm)))` with
    `throughput_rat = (F_R_norm·F_R_NOM + F_B_norm·F_B_NOM)/F_TOT_NOM` (everything
    already observed under S-B; closed form `= log(1 + 2·x_B_excess + 0.5·(1−eta_col))
    − log(xi_reb)`), was tested directly against the existing `n=5000` bank (no new
    simulation needed — reconstructed from already-stored raw-scale channel means via
    `expm1`, inverting `_compress_heavy_tails`). **Result: `corr(Q_ratio_feat, xi_reb)
    = -0.87`** (vs. `-0.65` for the raw, unengineered `Q_reb_mean` — a real, substantial
    improvement) **but with two residual couplings the plan's original derivation
    under-weighted**: `corr(., eta_col) = -0.23` (expected, smaller than assumed) and,
    newly found, **`corr(., alpha) = -0.38`** — `x_B_excess` (bottoms composition
    deviation) turns out to be driven far more by `alpha` (reactor conversion, hence
    column-feed composition) than by `eta_col`, confirmed directly: a matching
    `T_reb`-derived residual feature (`T_reb_residual = T_reb − T_REB_NOM −
    3·(F_R_norm−1) − 12·(V_norm−1) = 90·x_B_excess + 8·(1−eta_col)`, same closed form,
    verified against `physics.py`) correlates `-0.82` with `alpha` vs. only `-0.28`
    with `eta_col` — a 3x-larger alpha-channel than the plan anticipated, and a likely
    contributor to this project's persistent, hard-to-pin-down `alpha` instability.
    **Because `T_reb_residual` and `Q_ratio_feat` share the exact same `x_B_excess`
    term at two different, known coefficients (90 vs. 2), a two-term linear
    combination cancels it algebraically** — fit directly on the bank (`OLS:
    -log(xi_reb) ≈ 0.984·Q_ratio_feat − 0.0197·T_reb_residual − 0.023`,
    held-out-split-verified, `R²=0.985` train and test, coefficients stable to 3
    decimals across the split — not overfitting): **`corr(combined estimator, xi_reb)
    = -0.99`, `corr(., eta_col) = -0.11`, `corr(., alpha) = +0.02`** — both residual
    couplings collapse to near-zero simultaneously. **This combined 2-feature
    estimator, not the single-feature original, is now the recommended `xi_reb`
    engineering fix** — it's still zero-new-sensor (both inputs are already-observed
    S-B channels), and empirically makes `xi_reb`'s signal nearly as sharp as
    `z_A0_eff`'s own dedicated analyzer channel (`r=-0.99` vs. `r=1.00`). Worth
    testing whether `xi_reb` can also be pulled into a near-exact/lightly-corrected
    analytic posterior (like `z_A0_eff`'s) rather than staying in the flow, given how
    close to deterministic this combined readout is — would shrink the remaining
    flow to 3-D `(alpha, beta_r, eta_col)` if it holds up. Version the summary vector
    (`S-C.2` / `N_SUMMARIES_SC2`) — never silently change a summary definition under
    an existing dataset name.

    **IMPLEMENTED 2026-09-04.** `src/cstr_sbi/recycle/summaries.py`:
    `xi_reb_sharpened_features()`, `XI_REB_FEATURE_NAMES`, `N_SUMMARIES_SC2=74`,
    `compute_summaries`/`summary_names` both take `structure="S-C.2"` now (S-C's 72
    + these 2 features; feeds BOTH raw ingredients, not a pre-combined value, so
    training can learn/refine the combination itself). `scripts/sbi_pipeline.py`'s
    `_structure_config` recognises `"S-C.2"` too (same S-B control structure, no new
    control loop — purely an observation-side addition, like S-C itself).
    Smoke-tested end-to-end (40 fresh matched-protocol draws): wiring correct,
    `corr(qreb_feat, xi_reb)=-0.94`. **Then augmented the EXISTING n=1000
    Stage-1b-fixed bank AND both SBC sets to S-C.2 with ZERO new simulation**
    (`scripts/augment_bank_to_sc2.py`) — the 2 new features are closed-form
    functions of channel window-means already stored (compressed) in every
    existing S-C summary; inverting with `expm1` (exact, no precision loss) and
    reapplying the closed form reproduces byte-for-byte what a full re-simulation
    would have given, for free. Confirmed on the real augmented bank (not just the
    smoke test): `corr(qreb_feat,xi_reb)=-0.88`, combined-estimator
    `R²=0.98`, `corr(combined,xi_reb)=-0.99`, `corr(.,eta_col)=-0.026`,
    `corr(.,alpha)=+0.028` — matches the original n=5000-bank finding closely.
    **Training+SBC result, DONE** (`scripts/run_sc2_training.py`, same protocol
    as the S-C pilot). Baseline: 0/3 pass both budgets, unchanged (expected --
    `z_A0_eff`'s default-z-scoring pathology is unrelated to this fix).
    `z_score_x='none'`: **2/3 pass reduced, 1/3 pass confirm** — same PASS COUNT
    as the non-augmented S-C pilot, but the failure profile changed
    meaningfully: **`xi_reb`'s worst-case confirm failure improved from
    catastrophic (`3.9e-05`) to mild (`0.0029`)** — no more catastrophic `xi_reb`
    misses across any seed, and `beta_r`/`eta_col` stayed robust in all 3 seeds
    (`0.11-0.70`). Seed 0 passes ALL FIVE parameters simultaneously
    (`0.17-0.59`) — a fully-calibrated posterior, the second time this has
    happened this session (the non-augmented S-C pilot's seed 0 also fully
    passed). **The problem left standing is `alpha` specifically**: seed 1
    fails catastrophically (`3.3e-16`), seed 2 marginally (`0.0029`) — `alpha`
    is now the *dominant, most severe* remaining failure, having been
    completely unaffected by this feature (as expected, since it doesn't
    target `alpha`). **`alpha`'s catastrophic, seed-dependent failures have now
    recurred across every variant tested this session** (fixed-recipe
    monolithic, the factorised 4-D flow, and now S-C.2) **despite every other
    explanation tried being ruled out**: not a confound with `xi_reb` (`|corr|
    ≤0.14`), not a confound with `eta_col` (Gate A resolved, collapses), not
    fixed by more training data (`n=15000` made it worse), not fixed by
    removing `z_A0_eff` from the flow (factorised pilot still had a
    catastrophic `alpha` miss). This pattern — the SAME parameter failing
    catastrophically in a DIFFERENT seed each time, unrelated to any tested
    confound or feature — looks like a genuine training-instability specific
    to `alpha`'s own posterior shape (e.g. seed-dependent multimodality or a
    sharp feature this architecture/size fits inconsistently across random
    inits), not a data or representation problem. Per the plan's Gate B
    fallback: this is now squarely in "(iii) only if both [warm-start and
    residual-confound checks] are clean, consider one bounded architecture
    probe" territory — both ARE now clean, so a single pre-registered
    architecture probe (e.g. `hidden=90, transforms=4`) is the next justified
    lever, not another blind sweep.

    **Architecture probe run, DONE (2026-09-04), NEGATIVE result**
    (`scripts/run_sc2_arch_probe.py`, `hidden=90, transforms=4` vs. the
    baseline `60/3`, same S-C.2 bank, `z_score_x='none'`, 3 seeds). **Result:
    1/3 pass reduced, 0/3 pass confirm — WORSE than 60/3's 2/3/1/3.** `alpha`
    remains the dominant, most catastrophic failure at the bigger size too
    (seed 1: `5.8e-17`; seed 2: `4.6e-05`) — if anything, more seeds show a
    catastrophic `alpha` miss at 90/4 (2/3) than at 60/3 (1/3). **More capacity
    does not fix `alpha`'s instability — rules out "insufficient flow
    expressiveness" as the explanation.** Combined with everything already
    ruled out (confounds, training data, representation, architecture size),
    this exhausts every lever this plan pre-registered.

    **Wider seed sweep, DONE (2026-09-04)**: `scripts/run_sc2_seed_sweep.py`
    extended to 10 seeds (same setup: S-C.2, `z_score_x='none'`, `60/3`,
    `n_train=1000`, both independent SBC sets). **2/10 seeds (0 and 3) pass
    BOTH SBC sets robustly** (seed 0: `p=0.070`/`0.171`; seed 3:
    `p=0.169`/`0.099`; all five parameters comfortably `>0.05` on both sets
    for each) — a genuine ~20% hit rate, not a fluke (each candidate
    validated against an INDEPENDENT set, not just the one used to find it).
    The other 8/10 fail confirm, driven overwhelmingly by `alpha` (7/8 failing
    seeds have `alpha` as the worst parameter, several catastrophically:
    `1.3e-21`, `8.2e-12`, `8.6e-10`, `3.3e-16`) — consistent with everything
    above; `alpha`'s instability is real and seed-dependent, not fixed by
    anything tried, but **~1 in 5 random seeds already lands in a
    fully-calibrated basin**.
    **Practical implication**: "at least one usable, calibrated posterior for
    the paper" is already achievable at the existing `n_train=1000`, at
    near-zero additional cost (seeds are cheap; the bank is not) — this does
    NOT require the expensive `n=5000` regeneration. It does NOT, however,
    satisfy the stricter pre-registered ≥4/5-seed *robustness* gate (a ~20%
    per-seed hit rate makes 4/5 passing jointly unlikely by chance,
    `P≈0.007`) — that gate was designed for a different, stronger claim
    ("this architecture reliably calibrates") than "a specific, validated
    posterior exists and can be used and reported honestly," which seeds 0
    and 3 already deliver.

    **CRITICAL FINDING (2026-09-04) — seed 0 fails fault classification
    catastrophically, and the root cause is deeper than `alpha`'s SBC
    instability.** Ran Phase 3 (fault classification, `scripts/
    run_sc2_classification.py`, first-ever S-C-family classification run —
    no `nb31` companion existed before) against seed 0 on the standard
    14-scenario catalogue: **macro-F1 = 0.07-0.08, accuracy = 0.13-0.14** —
    only 1/14 scenarios (`W15_snowball_threshold`, the single most extreme
    catalyst-decay point) classified correctly; every other fault scenario,
    including SEVERE ones (`W3_cat_severe` alpha=0.65, `W4_cat_threshold`
    alpha=0.55), predicted "healthy".
    **Diagnosed directly, not assumed**: sampled the posterior at `W2`
    (mild, alpha=0.85), `W3` (severe, alpha=0.65), `W4` (severe, alpha=0.55)
    and `W15` (most severe, alpha=0.58) and found **severe shrinkage-toward-
    prior-mean at every single one, including the MILD case**:
    `alpha=0.85 → posterior mean 1.019`; `alpha=0.65 → 0.931`; `alpha=0.55 →
    0.887`; `alpha=0.58 → 0.874`. Posterior std stays narrow (`~0.08-0.09`)
    throughout — this is not "wide/uninformative," it's confidently biased
    toward the prior's nominal value, 2-4 posterior standard deviations away
    from truth in every case tested.
    **Root cause**: the truncated-lognormal training/SBC prior adopted since
    `nb37` (specifically to make SBC easier to pass) has `sigma=0.16` in
    log-space for `alpha` — narrow enough that even `W2`'s "mild" fault
    (`alpha=0.85`) sits ~1 log-sigma into the tail, and `W3`-`W15` sit
    2.7-3.4 log-sigma out. **The Wu2003 named fault-scenario catalogue
    systematically probes a region this narrow prior assigns very little
    training mass to** — the flow learned a training-loss-minimising
    strategy of shrinking toward the (much more frequently seen) near-nominal
    region, which is optimal in expectation over the LOGNORMAL prior's own
    mass but catastrophic for the actual classification task.
    **This means SBC-passing under the lognormal prior does NOT certify
    usefulness for classification** — SBC only checks calibration against
    draws from the SAME narrow training prior; a ~3-log-sigma point like
    `W3`/`W4`/`W15` is a `P≈9e-5` event under that prior, so a `N=400` SBC
    set essentially never tests this region (expected count `≈0.04` draws).
    **Seed 0 (and by the same argument, seed 3) is NOT usable for the
    classification task as currently trained**, despite passing every SBC
    check run so far. This is a different, more fundamental problem than
    `alpha`'s seed-dependent SBC instability chased all session — it is a
    **prior/training-distribution mismatch with the deployment/evaluation
    distribution**, and it likely also affects any future `n=5000` run on
    the same lognormal prior, not just the current `n=1000` pilot.
    **Not yet investigated**: whether the ORIGINAL `nb31` S-B posterior
    (which reportedly classified far better) was trained on the wider,
    original `box_uniform_5d()` prior rather than this lognormal one — if so,
    that would directly explain the gap and point to the fix (a wider
    training prior, or at minimum verifying training-prior coverage against
    the actual scenario catalogue before trusting SBC again). **Do not
    proceed to Phase 4/5 of the cascade until this is resolved** — building
    tracking/EKF-comparison work on top of a classifier this broken would
    waste further effort.

   - **Mixed-prior fix tested (2026-09-05): real, measurable, but partial
     improvement — and a much bigger implication.** Built
     `mixed_alpha_uniform_prior()` (`alpha ~ Uniform(0.4,1.2)`, other 4
     params their existing lognormal marginals; module-level, picklable,
     `log_prob`/`sample` verified correct including out-of-support `-inf`).
     Generated a fresh `n=1000` S-C.2 bank + both SBC sets under it
     (`scripts/regenerate_sc2_mixedprior_bank.py`) — **ran cleanly and
     stably in ~6h total** (accept rate ~75%, no rate degradation over the
     whole run, unlike every previous long bank-generation attempt this
     session), directly confirming the hypothesis that widening only the
     one dimension actually shown to need it avoids the "multiple
     simultaneously-extreme parameters" pathology that made the full-5-D
     uniform attempt collapse to 37% accept rate.
     - **10-seed sweep: 0/10 pass both SBC sets** — worse by pass-count than
       the all-lognormal prior's 2/10, and now **`alpha` is the worst
       parameter in every single seed** at the confirm budget. Direct
       posterior inspection resolved why: sampled 8 SBC points spanning
       `alpha=0.40` to `1.11` and found the posterior tracks truth within
       ~1 posterior SD almost everywhere in that range — genuinely fixed,
       not shrunk. The failures are concentrated at the **extreme upper
       boundary** (`alpha` near `1.2`, `-3.5` posterior SD off) — a region
       **no named Wu2003 fault scenario ever probes** (`alpha>1.0` has no
       physical fault interpretation in this catalogue; scenarios are all
       healthy=1.0 or degraded/low). The SBC test (correctly) penalizes this
       boundary region since it's part of the declared prior; the
       classification task never needs it.
     - **Classification rerun on seed 0 (the mixed-prior posterior):
       macro-F1 = 0.11, accuracy = 0.21-0.22** — nearly double the
       lognormal-prior result (`0.07-0.08`/`0.13-0.14`), and **reactor-class
       F1 roughly doubled (0.21-0.25 → 0.36-0.38)**. `W4_cat_threshold`
       (`alpha=0.55`, a SEVERE fault) is now correctly classified as
       `reactor` — it was misclassified as `healthy` before. `W12`/`W16`
       (multi-fault scenarios) now at least catch the reactor component
       (predict `reactor`) instead of missing entirely (`healthy`).
     - **Still far from sufficient**: `W2_cat_mild`, `W3_cat_severe` still
       misclassified as `healthy`; `column` and `feed` classes remain at
       F1=0.000 (untouched by an alpha-only fix, as expected); no scenario
       is ever classified as `multi` even when true.
     - **The bigger implication**: this result is strong direct evidence
       that the SAME coverage-mismatch mechanism just fixed for `alpha`
       plausibly affects `beta_r`/`eta_col`/`xi_reb` too — their earlier
       "SBC passes" under the narrow lognormal prior were tested against an
       equally narrow SBC set, which may be similarly under-testing their
       own fault-relevant tails (the same illusory-pass risk just uncovered
       for `alpha`). `z_A0_eff` is the likely exception, given its
       near-exact dedicated analyzer signal makes its posterior largely
       insensitive to prior width.
     - **Not yet done, natural next step**: widen `beta_r`, `eta_col`, and
       `xi_reb` similarly (uniform or wide-lognormal), keeping `z_A0_eff`
       narrow, and re-run the same seed-sweep + classification check. Given
       accept rate held up well widening just one parameter (~75%, vs the
       full-5-D attempt's 37%), widening 4 of 5 (excluding only `z_A0_eff`)
       carries real but not yet quantified additional cost risk — worth a
       quick smoke-test (a handful of draws) before committing to a full
       generation, following the same pattern used for `alpha` alone.
    **Per Gate C: "nothing passes → adopt path (a) from
    `system_ii_sbc_repair_directions.md` (reduced posterior + explicitly
    reported limits) as the manuscript outcome."** `alpha`'s recurring,
    seed-dependent catastrophic SBC failure should now be reported as an
    honest, documented limitation of the S-C posterior (calibrated for
    `z_A0_eff`, `beta_r`, `eta_col`, `xi_reb`; `alpha` unreliable in a
    seed-dependent way not explained by any tested mechanism) rather than
    chased with further blind training/architecture sweeps — the project has
    now tried every category of fix this plan anticipated.
2d. **Bank regeneration**: required by 2c (banks store summaries, not raw
    trajectories) and by any Stage-1b warm-start fix. ~18 min at 0.21 s/draw for
    n=5000 — cheap. Keep the paired-bank property (same theta rows for S-B and
    S-C variants) for the Arm B/C ablation.
2e. **Screen**: 3 seeds, both SBC sets at `N_SBC=120` (screen only), then
    immediately `N_SBC=400`, `n_post=400` regardless of the screen outcome (the
    N=120 pass at 24a was itself the documented false-positive pattern). Also run
    the anti-shrinkage regression slope per parameter — pass/fail per Stage 0.3.

**Why this should work when the last three levers didn't:** it removes the loss
dominance (§1b) *and* the boundary pileup (§1c) *and* concentrates `xi_reb`'s signal
(§1c) simultaneously, and each element is individually already evidenced (the
`z_A0_eff` analytic posterior is exact; logit-space was the best S-B lever; the
`Q_reb` feature is a closed-form identity). It is a restructuring, not a fourth
blind sweep.

### Stage 3 — Decision gates and contingencies

- **Gate A (after Stage 1a):** any genuine residual confound under S-C
  (`|cov_offdiag_corr| > ~0.5`, representation-robust)? → go to the sensor decision
  (§3) *before* burning more training compute on that pair. Training cannot fix a
  rank-deficient likelihood — the project has proven this to itself twice.
- **Gate B (after Stage 2e):** all four flow parameters pass → Stage 4.
  `alpha` or `xi_reb` still fail *with the shrinkage signature* despite Stage 2 →
  (i) if Stage 1b flagged the warm start, fix that first and retrain once;
  (ii) if Stage 1a showed a borderline pair, escalate to the §3 sensor path;
  (iii) only if both are clean, consider one bounded architecture probe
  (e.g. 90/4) — a *single* pre-registered run, justified because the old 60/3
  validation (`feedback_network_sizing`) was for a different problem
  (5-D uniform, S-A/66-D), not this 4-D lognormal logit-space one.
- **Gate C:** nothing passes → adopt path (a) from
  `system_ii_sbc_repair_directions.md` (reduced posterior + explicitly reported
  limits) as the manuscript outcome. That path is already written up as legitimate;
  this plan just makes it the *last* resort instead of an ambient temptation.

### Stage 4 — Confirmation protocol (before any promotion)

- 5 seeds (not 3), `N_SBC=400`, `n_post=400`, **both** independent SBC sets,
  coverage@90 (`coverage_at_90` with the lognormal prior), anti-shrinkage slopes.
- `z_A0_eff`'s analytic posterior validated by a deterministic unit test
  (coverage of the truncated-Gaussian under simulated readings), not SBC.
- Only then: promote one posterior pickle, update `HANDOFF.md`, and re-execute
  `nb24a` cleanly so notebook outputs match the CSVs (closing §1d's staleness).

### Stage 5 — Downstream (unchanged from the analyzer plan)

Phases 3–5 of `system_ii_analyzer_rework_plan.md` proceed as written once Stage 4
passes: classification rerun (Arm B), the new feed-drift 30-day trajectory
(Arms B & C, shared noise seeds), cascade reruns (nb26/27/29/30/33 with
`cov_offdiag_corr`), and the Phase 6 Artefact-2 re-audit (independent of all of the
above — can run in parallel any time).

**Arm C note:** the protocol-matched S-B posterior for the ablation does *not* need
to pass calibration — its documented failure on the confounded pair **is** the
comparison's point. Train it with the identical Stage-2 recipe minus the analyzer
(4-D flow will still see `alpha` confounded via `z_A0` absence — report as the
known S-B limit), same paired bank, and report its SBC status honestly next to
Arm B's.

---

## 3. Sensors — explicit decisions

| Sensor | Decision | Placement / spec | Rationale |
|---|---|---|---|
| **Feed-line composition analyzer** | **ADD — already settled (S-C), keep** | On the **fresh-feed line, upstream of the recycle mixing point** (placement is load-bearing: measuring the *combined* reactor feed would re-mix `x_D`'s alpha/eta_col dependence back in — the entire reason the reactor-side analyzer failed nb43). Relative noise **≤ 0.01%** (`FEED_ANALYZER_NOISE_PCT=1e-4`); cite Kett BSS1700 (accuracy ±0.01%, precision ~0.001%, application-dependent caveat stated). | Resolves both confirmed genuine confounds `(alpha, z_A0_eff)` and `(eta_col, z_A0_eff)`; below 0.01% precision it fails (7-point sweep, nb40/41/43). |
| **Bottoms-product composition analyzer (`x_B`)** | **NOT ADDED — Gate A checked, did not trigger (resolved 2026-09-04)** | Bottoms/product stream at the column base. Industrially the most defensible sensor in the whole flowsheet (it measures product quality). Zero new physics: `x_B` is already raw channel 4; adding it is a channel-list + summary change only. | Was conditional on Stage 1a finding a genuine `(xi_reb,eta_col)` or `(alpha,eta_col)` residual confound. The FIM sweep + grid-scan tiebreaker (`scripts/stage1a_gate_a_tiebreaker.py`, all 3 trigger points) found `(alpha,eta_col)`'s FIM correlation is a linearisation artifact that collapses under the full deterministic grid-scan — same signature as the retracted W12 banana. `(xi_reb,eta_col)` also never exceeded `|corr|=0.42`. **No sensor needed**: the far cheaper feature-engineering fix (the combined `Q_reb`/`T_reb` estimator, `r=-0.99` with `xi_reb`, see Stage 2c) already resolves what this sensor would have targeted, using only already-observed channels. |
| **Reactor-outlet composition analyzer (`z_A`)** | **CONDITIONAL ADD** — only if a dedicated Phase-6 re-audit confirms a genuine residual `(alpha, beta_r)` coupling under S-C (not yet run; independent of the Gate A check above, which covered `(alpha,eta_col)`/`(xi_reb,eta_col)`, not `(alpha,beta_r)`) | Reactor outlet / column feed line. Also zero new physics (raw channel 11). | nb43 rejected it because its dynamics mix `alpha` and `z_A0_eff` — **that rejection is now obsolete in combination with the feed analyzer**: with `z_A0_eff` pinned to ~1e-5 by the feed analyzer, `z_A` becomes a clean `alpha` probe (`dz_A` = feed term(known) − `alpha·k(T_r)·z_A`, with `T_r` measured). Same rule: deterministic feasibility check first. Note Stage 1a's own sweep found `(alpha,beta_r)` maxes at `|corr|=0.42` (one point only, collapsing to `0.12` under raw trajectory) — a weak prior against this one too, but not yet a dedicated check. |
| Existing S-B set (T_r, T_j, Q_j, T_reb, Q_reb, F_R, F_B, R, V) | **KEEP — no relocations** | — | No evidence any existing sensor is mispositioned; every diagnosed problem is either missing composition information (solved above) or training-side. |

Do **not** add either conditional sensor pre-emptively: each new channel re-opens the
manuscript's instrumentation-cost narrative and each has a cheap deterministic
go/no-go check (the pattern that already worked once for the feed analyzer). The
narrative stays "FIM predicts the sensor; the sensor fixes the confound" — the
paper's strongest framing.

---

## 4. What NOT to do (carried forward + new)

- No blind hyperparameter sweeps — two failed in opposite directions; only the
  single pre-registered Gate-B architecture probe is sanctioned.
- No trusting reduced-budget or single-seed SBC passes (`finding_sbc_power_eta_col`,
  re-burned at 24a's N=120).
- No `n_train` scaling until n=5000 passes the gate — 15k is evidence-worse.
- No `offdiag_ratio` anywhere; `cov_offdiag_corr` only.
- No silent summary-vector changes — version them (2c).
- No promoting any existing `sbi-logs/wu2003_posterior_sc_*.pkl` — all superseded
  by the factorised design.
- No mixing fixed-y0 training into matched-protocol manuscript numbers (Major
  Comment 7 stands); Stage 1b's fix, if needed, *tightens* the matched protocol
  rather than abandoning it.

## 5. Sequencing and cost

| Step | Compute | Depends on |
|---|---|---|
| Stage 0 (hygiene, gate registration) | minutes | — |
| Stage 1a FIM sweep | ~1 h deterministic | — |
| Stage 1b warm-start audit | ~1 h deterministic | — |
| Stage 1c second SBC set | ~1.5 h sim | — |
| Stage 2 bank regen + 3-seed train + SBC | ~half day | Stages 0–1 |
| Stage 3 gates | decision only | Stage 2 |
| Conditional sensor feasibility checks | minutes–1 h, deterministic | Gate A/B |
| Stage 4 confirmation (5 seeds, 2 sets) | ~half day | Gate B pass |
| Stage 5 downstream cascade | per analyzer plan | Stage 4 |

Critical path: Stage 2. Stages 1a/1b/1c are mutually independent and can run in
parallel; Phase 6 (Artefact 2 re-audit) remains independent of everything and
should be scheduled regardless.
