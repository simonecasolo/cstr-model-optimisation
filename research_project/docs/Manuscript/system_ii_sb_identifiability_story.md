# System II (S-B): the identifiability story, in full

**Purpose.** This is the technical backbone for the claim "SBI's classification
and parameter-estimation accuracy under S-B is exactly as good as the plant's
*identifiability* allows, no better and no worse, and we can show this
quantitatively before and after the fact." It covers: which confounds are
expected and why, how the Fisher-information (FIM) methodology supports each
claim, how this shows up in the actual classification/tracking results, and
why EKF beats SBI in some regimes but not others. A final section gives the
plan for what still needs to happen before this story is publication-ready for
System I + System II (S-B).

---

## 0. Read this first: current status of the numbers below

Before using anything in this document as a settled manuscript claim, know
that **this session's own verification work surfaced three open problems**
that the current `main.tex` text does not yet reflect:

1. **The production S-B posterior (`sbi-logs/wu2003_posterior_sb.pkl`,
   2026-07-03) predates a confirmed physics bug fix (2026-08-12,
   `src/cstr_sbi/recycle/physics.py`, the `beta_r`/`Q_j` double-counting
   fix).** The bug made jacket fouling also erroneously discount the
   controller's *commanded* duty `Q_j`, not just the conductive heat-transfer
   term — a real, acknowledged modelling error, not a cosmetic one. Every
   cached S-B training bank and posterior from before this date is stale by
   the project's own standing rule, and `notebooks/legacy_systemII/31_wu2003_fault_classification.ipynb`
   (the source of `main.tex`'s classification Table, §7.3) **still loads this
   same stale, pre-fix posterior** as of its last edit.
2. **The corrected-physics retrain failed to calibrate.** The matched-protocol
   retrain under the fixed physics (`sbi-logs/wu2003_posterior_sb_matched_full.pkl`,
   2026-08-27, 8 seeds) **did not pass SBC for any seed** (best seed's
   confirmation-budget min-KS `p=9.0e-8`). `HANDOFF.md` is explicit: *"Do not
   promote `wu2003_posterior_sb_matched_full.pkl` or update System II
   manuscript results as calibrated."* Everything from `notebooks/35` through
   `notebooks/41` is a diagnosis-and-repair attempt for this failure; none
   fully succeeded under the original 5-D S-B scope.
3. **Root cause of (2) has since been pinned down, and it rewrites one of the
   manuscript's own identifiability conclusions.** `notebooks/39`–`42`
   (post-fix, using a corrected FIM metric — see §3.3 below) found that
   `main.tex`'s **Artefact 3 is currently mis-stated**: `(alpha, z_A0_eff)`
   and `(eta_col, z_A0_eff)` are **not** representation artefacts that
   resolve under the raw trajectory — they are genuine, representation-robust
   non-identifiabilities, confirmed independently five separate times. This
   is very likely *why* the corrected-physics S-B posterior cannot be
   calibrated at all: the flow is being asked to represent a correct
   posterior shape over a subspace that is structurally non-identifiable,
   which no amount of training fixes. **Artefact 2** (`alpha, beta_r`) has
   **not** yet received the same rigorous re-audit; a preliminary check in
   `notebooks/42` could not reproduce the magnitude of the claimed raw-trajectory
   collapse at W11 (and could not trace the cited numbers `-0.07`/`5.34`/`5.48`
   to any script or notebook at all), so treat its "resolved, benign" framing
   as provisional, not retracted.
4. **Update (this session, "Arm C"): point (2)'s SBC failure was at least
   partly a data-quality artefact, not purely the genuine confound in (3).**
   A clean paired S-B bank (`data/wu2003_sbi_train_sb_paired_lognormal_true_n1000_stage1bfix.npz`,
   generated *after* the warm-start fix as part of the S-C work but never
   actually used to retrain S-B — the plan explicitly deferred this
   "Arm C" comparison and it was never picked back up) was sitting on disk,
   unused. Training on it (`scripts/run_sb_arm_c_training.py`, zuko_nsf 60/3,
   `z_score_x='none'`, 3 seeds, `n_train=1000`) gives **seed 0 fully
   calibrated on all 5 parameters simultaneously at the confirm budget**
   (`N=400`: `alpha` p=0.698, `beta_r` p=0.171, `eta_col` p=0.137, `xi_reb`
   p=0.137, `z_A0_eff` p=0.171 — see `results/sb_armc_sbc_n1000_stage1bfix.csv`).
   **This directly contradicts the "0/8 seeds, S-B cannot be calibrated"
   reading of point (2)** — that 8-seed failure predates the warm-start fix
   and was run on contaminated data; nobody had re-tried full 5-D S-B on
   clean data until now. The catch: seeds 1 and 2 reproduce the *exact*
   seed-dependent `alpha`-instability pattern already documented for S-C
   (seed 1 marginal `alpha` fail, p=0.038; seed 2 catastrophic, p=1.4e-6) —
   so this generalizes: `alpha`'s training fragility is **not** an
   S-C/analyzer-specific side effect, it is a general SNPE/flow-training
   issue that also affects plain S-B. **Net effect: genuine confounds (3)
   and data-quality/training fragility (this point) were previously
   conflated in a single "S-B can't calibrate" verdict — they are separable,
   and at least one is a solved problem.**

   **Classification check on the fully-calibrated seed — RUN, and it confirms
   the narrow-lognormal-prior-coverage trap generalizes to S-B too.**
   (`scripts/eval_sb_armc_classification.py`, full 14-scenario catalogue.)
   The SBC-calibrated `zscorenone` seed 0 is **catastrophic at classification**:
   macro-F1 = 0.089, accuracy = 23.6%. Severe faults are shrunk toward
   healthy regardless of true value (`W4_cat_threshold`, true `alpha=0.55` →
   posterior mean `0.915`; `W15_snowball_threshold`, true `alpha=0.58` →
   `0.932`) — every reactor scenario except the mildest is misclassified.
   Counter-intuitively, the **SBC-failing** `baseline` (default z-scoring)
   seed 0 does noticeably *better* (macro-F1 = 0.346, accuracy = 50%) —
   the better-calibrated-by-SBC posterior is actively worse at the real
   task, because SBC only certifies calibration within the training prior's
   own (too-narrow) support, not at the deployment distribution (exactly
   S-C's "Gate C" lesson, `system_ii_sbi_solidification_plan.md`). The
   narrow truncated-lognormal prior was adopted (`nb37`) specifically to
   make SBC *easier* to pass — it achieves that narrow goal while breaking
   the thing that actually matters. The original stale posterior's decent
   classification (87% accuracy, §0.1) almost certainly came from training
   under the old wide uniform prior (`data/wu2003_sbi_train_sb.npz`,
   pre-`nb37`), which has never been combined with the physics fix or the
   warm-start fix.

   **Fix tested, result in: real improvement, not a full resolution, plus a
   new methodological finding about trusting SBC for model selection.** A
   full 4-parameter-widened bank (`mixed_wide_prior()`) was measured
   directly: ~42s/accepted draw, only ~40% of draws converge — a useful bank
   would cost 8-9h, too expensive for a quick test. The cheaper,
   already-validated-for-S-C partial fix — widen only `alpha` to
   `Uniform(0.4,1.2)` (`mixed_alpha_uniform_prior()`), keep the other 4
   parameters' existing narrow lognormal marginals — was measured at ~75%
   accept rate (matching S-C's own finding), ~20s/accepted row, and a pilot
   bank was generated and trained (`scripts/generate_sb_mixedalpha_bank.py`
   + `scripts/run_sb_mixedalpha_training.py`, `n_train=200` — intentionally
   small, a directional pilot not a final number).

   SBC: `zscorenone` seed 1 passes all 5 parameters at the confirm budget
   (`N=100`: `alpha` p=0.518, `beta_r` p=0.253, `eta_col` p=0.253, `xi_reb`
   p=0.685, `z_A0_eff` p=0.103) — the widened prior does let `alpha` train
   cleanly, at a fifth of the narrow-prior bank's size.

   Classification (`scripts/eval_sb_mixedalpha_classification.py`, same
   14-scenario catalogue): **real improvement, but counter-intuitively not
   from the SBC-best seed.**
   | seed | SBC (alpha widened) | macro-F1 | accuracy |
   |---|---|---|---|
   | `zscorenone` seed 1 | **all 5 pass** | 0.140 | 0.357 |
   | `zscorenone` seed 0 | does not fully pass | **0.211** | **0.479** |
   | `zscorenone` seed 2 | does not fully pass | 0.198 | 0.471 |
   | *(for reference)* narrow-prior `zscorenone` seed 0, all 5 pass SBC | — | 0.089 | 0.236 |

   Seed 0 and seed 2 (SBC-imperfect) show real `alpha` discrimination
   tracking true severity (seed 0: `W3` true `0.65`→est `0.818`, `W15` true
   `0.58`→est `0.825`, vs `W1`/`W6` true `1.0`→est `0.90`/`0.95`) — roughly
   2.4× the narrow-prior posterior's macro-F1, a real, usable improvement.
   Seed 1 (the one that fully passes SBC) instead collapsed to near-constant
   `alpha` estimates (~0.86–0.88 regardless of true value from 0.55 to 1.0)
   and is the *worst* classifier of the three — **the second time in this
   investigation that the SBC-best seed is not the task-best seed** (the
   first was baseline-vs-zscorenone on the narrow-prior bank, §0.4 above).
   This is now a standing methodological caution, not a one-off: **do not use
   small-`N`/small-`n_train` SBC pass/fail alone to pick a production seed
   for this system** — always cross-check against the actual named-scenario
   catalogue.

   `column` and `feed` classes remain at F1 = 0.000 in every variant — exactly
   as expected, since only `alpha`'s prior was widened; `eta_col`/`xi_reb`/
   `z_A0_eff` still never see realistic fault severities in training. This is
   a clean confirmation of the mechanism, not a new problem: each parameter's
   classification usefulness is gated by its *own* training-prior coverage.

**What this means practically:** the *mechanism* and *methodology* described
below (§§1–3, 6) are solid and reusable regardless of which exact posterior
backs the final numbers — they are the right lens either way. The specific
*numbers* in main.tex's current Artefact-2/3 sections and classification table
(§4 below) are the ones that need to be regenerated/re-verified before this
document's claims can be called final. §7's plan addresses this directly.

---

## 1. The shared mechanism: why we should expect *some* confound before we even train

Both CSTR systems in this study (System I, the single PO reactor; System II,
the Wu2003 reactor–column–recycle plant) share one structural fact that
predicts trouble before a single posterior is trained: **Loop 1's PI integral
action pins the controlled reactor temperature at setpoint in steady state.**
Formally, for both systems,

```
d T_ss / d(thermal fault parameter) ≡ 0
```

(System I: `beta`, the jacket-conductance factor; System II: `beta_r`, the
jacket-conductance factor, and more mildly `alpha`, the catalyst-activity
factor, since the reactor temperature is the actuated variable for both). This
is not a numerical approximation — it follows directly from the definition of
integral control: any sustained offset is driven to zero by construction, so a
parameter whose *only* steady-state effect is on that offset becomes invisible
in the controlled variable itself. The information does not vanish; it is
*redistributed* to the secondary channels that carry the controller's
corrective effort (`Q_c`/`Q_j`, jacket temperature `T_c`/`T_j`). This
redistribution is lossy and nonlinear, and it is the common root of every
confound discussed below.

**The corollary that should be stated ahead of any training run, not
discovered after:** any parameter pair that (a) shares a fault unit, and (b)
is compensated by the *same* actuator, is a strong a-priori candidate for a
posterior confound. For System II this immediately flags two pairs before any
data is simulated:

- `(alpha, beta_r)` — both reactor-side, both compensated via `Q_j`.
- `(alpha/eta_col, z_A0_eff)` — not sharing an actuator loop directly, but
  sharing a *stoichiometric* pathway: anything that reduces per-pass
  conversion (lower `alpha`, worse `eta_col`) increases the A-load returned to
  the reactor by the recycle, which raises `z_{A,in}` in exactly the direction
  a leaner fresh feed (`z_A0_eff` down) does not — i.e., multiple physically
  distinct causes drive the same downstream composition signature via the
  recycle loop (the "snowball effect", `main.tex` §5.2).

Both predictions turn out to be correct, but — this is the point of doing the
FIM check *before* training, not just after — **they are not equally severe**,
and the difference matters enormously for what a monitoring system can
promise.

---

## 2. Expected identifiability challenges, parameter by parameter

### 2.1 `(alpha, beta_r)` — reactor-internal, unit-benign (provisional — see §0.3)

Both act through `Q_j`. Section 1's masking argument applies to both equally.
**What should make us expect this to be recoverable at the *unit* level even
if not at the *parameter* level:** both parameters belong to the same fault
unit (reactor) under the manuscript's taxonomy (`sec:fault_class`), so even a
posterior that cannot tell "lower `alpha`" from "lower `beta_r`" apart can
still correctly say "the reactor is degraded." This is a specific, falsifiable
prediction: *expect near-perfect unit classification for reactor scenarios
despite poor individual-parameter disambiguation.* (Confirmed in the
classification results, §4 — pending the posterior re-verification in §0.)

### 2.2 `(alpha, z_A0_eff)` and `(eta_col, z_A0_eff)` — cross-unit, genuinely severe

These parameters do **not** share an actuator loop (feed composition is not
directly controlled against `z_A0_eff`), so the naive "shared Q-channel"
argument from §2.1 doesn't apply here — the mechanism is the recycle
stoichiometry argument above instead. **This is the pair that should worry a
monitoring system more, and here is the a-priori reason why, before even
looking at FIM numbers:** `alpha` and `eta_col` belong to *different* fault
units (reactor, column) than `z_A0_eff` (feed). Per §2.1's logic in reverse: a
confound that crosses a unit boundary cannot be rescued by unit-level
aggregation the way a same-unit confound can. **Expect this pair to degrade
not just parameter estimation but the unit-level fault classification
itself** — and specifically expect feed-related scenarios to be
misclassified as reactor or column faults, since the posterior mass genuinely
cannot distinguish "the reactor is converting less" from "the feed is leaner."

This prediction is the one independently confirmed five separate times
post-physics-fix (`notebooks/39`–`42`): the coupling is large under the
66-D summary representation (`compute_summaries`) **and stays large under the
raw, unaggregated trajectory of the same channels** — i.e., this is not a
compression artefact that a smarter feature set or raw-trajectory estimator
would fix; the *channels themselves*, as currently instrumented (`S-A`/`S-B`,
no reactor- or feed-side composition analyzer), do not contain the
distinguishing information. `(xi_reb, z_A0_eff)` was checked and ruled out —
reboiler fouling and feed purity are not confounded.

### 2.3 `xi_reb` (reboiler fouling) — expected to be well-identified

No a-priori reason to expect masking: Loop 3's boilup control does not drive a
steady-state error to zero in the same sense Loop 1 does, and `xi_reb`'s
effect on reboiler duty is not shared with any other parameter's primary
channel. Expect, and the results show (§4), near-perfect column-fault
classification.

---

## 3. How Fisher information supports these claims

### 3.1 The local-sensitivity/FIM methodology

Both systems use the same Gaussian sensitivity-information approximation
(`main.tex` Eq. `eq:local_sensitivity_information`, `scripts/fim_utils.py`):

```
I~(theta) = J^T Sigma^-1 J
```

where `J_ij = d E[s_i | theta] / d theta_j` (summary-statistic Jacobian, via
finite differences) and `Sigma` is the *empirical* noise covariance of the
summary vector at a fixed `theta` (estimated from real replicate/sensor
noise, not assumed analytically). This coincides with the Fisher information
for a Gaussian summary model with parameter-independent covariance; for the
engineered, nonlinear, non-Gaussian summaries actually used here it is a
**local sensitivity and conditioning diagnostic**, not a certificate of global
identifiability — this distinction matters because it is exactly how the
original `(alpha, eta_col)` "S-A headline" claim from earlier in the project
was over-interpreted and later retracted (see memory: `finding_sa_headline_retracted`).

Two scalar summaries of `I~` do the actual explanatory work:
- **Diagonal ratio** `I_aa / I_bb` (System I's headline `~236x` `alpha`/`beta`
  asymmetry at nominal) — how much more locally precise one parameter is than
  another, independent of any correlation between them.
- **Condition number** of the full FIM — how close the local information
  geometry is to a one-dimensional ridge (1 = isotropic; large = near-null
  direction, i.e. a near-degenerate combination of parameters the data barely
  constrains at all).

### 3.2 The decisive test: does the coupling survive the raw trajectory?

The question that actually separates "fixable with better features" from
"a real plant-level limit" is: **recompute the same FIM using the raw,
unaggregated trajectory of the same already-observed channels, instead of the
compressed summary vector.** If the normalized off-diagonal collapses toward
zero and the condition number drops by orders of magnitude, the compressed
representation was discarding genuinely distinguishing information (a
representation artefact — fixable by better features, a raw-trajectory
estimator, or a filter with full-trajectory access). If it does **not**
collapse, the confound is intrinsic to what the instrumented channels can see,
full stop — no feature engineering or raw-trajectory access changes that
(§2.2's case).

This RT-FIM check is cheap relative to training an amortised posterior
(`O(10^2)` simulator calls vs. `O(10^4)` for SBI training) — it is a
pre-training *predictive* diagnostic, not just a post-hoc explanation, which
is exactly what supports the "we can say ahead of time that classification
won't be perfect" framing the user wants: run this check on any new
parameter pair before training, and you already know whether to expect a
fixable or an intrinsic limit.

### 3.3 A real methodological trap already found and fixed here: raw information off-diagonal ≠ posterior correlation

`fim_utils.offdiag_ratio` returns a normalized entry of the **raw information
matrix**. For a multi-parameter FIM (more than 2 parameters), this is *not*
the same quantity as the actual posterior correlation coefficient implied by
the (Laplace-approximated) **inverted** FIM — and the two can have opposite
signs once the other parameters are properly marginalized out. This was found
the hard way in this project (`notebooks/39`/`40`: raw off-diagonal `+0.40`
vs. the correctly-inverted `cov_offdiag_corr` of `-0.60` at the same point)
and is now fixed by using `fim_utils.cov_offdiag_corr` (inverts the FIM first)
for any claim that should be compared against an actual trained posterior's
empirical `corrcoef`. **Any FIM-based correlation number computed before
2026-09-01 in this project's notebooks should be treated as using the wrong
(raw, not inverted) convention unless explicitly stated otherwise** — this
includes the original Artefact 2/3 numbers currently in `main.tex`.

---

## 4. How this is expected to show up in the classification results

Given §§2–3, the predicted — and (pending §0's posterior-regeneration caveat)
observed — shape of the classification results is:

| Fault unit | Scenarios | Expected | Why (from §2) |
|---|---|---|---|
| Reactor | W2–W6, W11, W15 | Near-perfect unit F1 despite poor `alpha`/`beta_r` disambiguation | Same-unit confound, §2.1 |
| Column | W7, W9 | Near-perfect | No predicted confound (§2.3) |
| Feed | W10 | **Expected to fail** | Cross-unit confound with reactor/column, §2.2 |
| Multi (incl. feed) | W12, W13, W16 | Degraded, worst on feed-containing combos | Same mechanism as above |

`main.tex`'s current (pre-verification) classification table reports almost
exactly this pattern: reactor F1 = 0.948, column F1 = 1.000, **feed F1 =
0.000** (misclassified in all 30 replicates), multi F1 = 0.854 with the
feed-containing W13 as the specific weak point (7/30). The headline aggregate
(macro-F1 = 0.694, 87.4% accuracy) looks mediocre read in isolation; read
against §2's a-priori predictions, it is **exactly the pattern a correctly
instrumented identifiability analysis would have predicted in advance** — a
materially stronger claim for a monitoring-SBI paper to make than "high
accuracy," because it demonstrates the method's accuracy tracks a
quantitatively predictable ceiling rather than being an opaque number.

---

## 5. Why EKF outperforms SBI in some regimes and not others

This is not a single answer — the manuscript's own data splits cleanly into
two regimes with two different explanations, and it is important not to
collapse them into "EKF is better" or "SBI is better" as a blanket claim.

### 5.1 Single-window / snapshot classification: SBI is generally more robust, EKF fails near nonlinearity

At the near-snowball scenarios (W12, `alpha=0.75`; W15, `alpha=0.58`), the EKF
achieves only **3% empirical coverage** on `alpha` — a failure of local
Gaussian linearisation under strong recycle nonlinearity near the snowball
tipping point, not a tuning artefact: the EKF's first-order Taylor
approximation of the recycle dynamics breaks down precisely where the
dynamics are most nonlinear. SBI, by contrast, learns the full non-Gaussian
posterior shape directly and achieves 100% coverage at W15. At W11, the EKF's
0% coverage is **tuning-sensitive, not structural** — an independently
re-tuned EKF with raw-trajectory access recovers both `alpha` and `beta_r` to
<2% error. SBI's unit-level classification remains correct at all three
scenarios regardless of EKF tuning. **Takeaway: near strong nonlinearity or
multimodal posterior geometry, amortised SBI's flexibility is a genuine
structural advantage over a linearised filter.**

### 5.2 30-day sequential tracking: EKF wins, because it isn't paying System II's representation tax

In 30-day tracking (360 independent 2-hour windows, slow monotonic
degradation, no actuator saturation, no near-snowball excursion), the EKF
**outperforms** the amortised SBI posterior. The manuscript's own diagnosis
(§8.3 `sec:disc_ekf_regimes`, Supporting Information §S11) rules out any
benefit from the EKF's state accumulation across windows and instead traces
the gap to the EKF's **privileged access to the raw, unaggregated trajectory
through the exact ODE model** — i.e., this is Artefact 2's mechanism in a
different guise: whole-window mean/std/slope/min/max/q_mean aggregation
discards the transient shape that distinguishes slow catalyst decay from slow
jacket fouling, and an amortised posterior trained only on that compressed
representation cannot recover information the compression already discarded.
A CNN-embedding posterior trained on raw trajectories closes part, not all, of
the gap (`beta_r` MAE improves from worse to `0.139`, but `alpha` MAE
*worsens* to `0.067`) — consistent with §3.2's general point that
raw-trajectory *access* helps but is not automatically *exploited* by an
arbitrary amortised architecture. **Takeaway: when the dominant information
loss is from summary-statistic compression rather than physical masking, a
model-based filter with exact raw-trajectory access has a real, mechanistic
edge that no amount of SBI training budget fixes — only a richer
representation (CNN embedding, raw-trajectory SBI, or a filter) does.**

### 5.3 The open asymmetry: System I doesn't show this split, and that is not yet explained

System I's own multi-method comparison (Sc2, Table `tab:method_comparison`)
and 30-day tracking **both** favor SBI over EKF/UKF throughout — there is no
regime in System I where the filter wins (EKF/UKF bias `-0.093` vs. SBI's
`-0.002` at Sc2; EKF/UKF MAE roughly an order of magnitude worse than SBI in
30-day tracking). This is an honest, currently-unexplained asymmetry between
the two systems: naively, §5.2's logic (raw-trajectory access helps when
representation loss dominates) should apply to System I's `beta` too, since
System I has its own genuine, Loop-1-driven information deficit for `beta`
(the `~236x` diagonal FIM ratio, §3.1). Candidate explanations not yet tested
in this project: System I's physics-informed summary features
(`UA_eff_proxy`, `k0_eff_proxy`) may already capture most of what a raw-access
filter would otherwise gain; or the System-I EKF/UKF implementations may
simply be less favorably tuned than System II's. **Do not state a causal
explanation for this asymmetry in the manuscript without first checking
which of these (or another candidate) actually holds** — it is currently an
open question, not a confirmed finding.

---

## 6. The synthesis: "SBI is as accurate as identifiability allows"

Putting §§1–5 together, the claim the user wants to make is specific and
falsifiable, not a vague hedge:

1. **Before training**, the shared-actuator argument (§1) and the
   fault-unit-crossing argument (§2) predict which parameter pairs will be
   confounded and whether that confound will or will not survive to the
   unit-classification level.
2. **The RT-FIM check (§3.2) independently predicts, before training an
   amortised posterior at all, whether a given confound is fixable (collapses
   under raw trajectory) or intrinsic (does not).**
3. **The actual classification results (§4) track this prediction exactly**:
   near-perfect where predicted, a complete, explained failure exactly where
   predicted (feed attribution), nothing else.
4. **Where a model-based alternative (EKF) outperforms SBI, it does so for a
   mechanistically identified reason (§5)** — raw-trajectory access defeating
   a representation-level information loss, or a flexible posterior defeating
   a filter's linearisation limit — not an unexplained method ranking.

This is a stronger, more defensible claim than "SBI achieves X% accuracy":
it says the method's failures are *predictable from first principles and an
inexpensive pre-training diagnostic*, which is exactly the property you want
for an operational monitoring tool (you can tell an operator in advance which
fault types the current instrumentation can and cannot reliably attribute,
rather than discovering it empirically after deployment).

---

## 7. What is still needed before this is a publication-ready story

Ranked by blocking severity, not by how it was originally planned:

### 7.0 (blocking, highest priority) Resolve the S-B calibration/physics-fix gap — UPDATED, now tractable

**Superseded by §0.4's "Arm C" result.** The entire §4 classification table
and the exact Artefact 2/3 numbers in `main.tex` are still from the pre-fix
posterior (§0.1), and the matched-protocol retrain that was supposed to
replace it genuinely failed SBC (§0.2) — but that failure is now known to be
*at least partly* a data-contamination artefact, not purely the genuine
`(alpha/eta_col, z_A0_eff)` confound (§0.3) as previously assumed. A single,
cheap retrain on the already-existing clean bank produced a fully-calibrated
5-parameter seed on the first attempt (§0.4). This reopens option (a) below,
which the previous version of this section rated "low expected value" —
that rating no longer holds.

Concrete next steps:
1. ~~Run a wider seed sweep on the narrow-lognormal clean bank~~ — **superseded**:
   the classification check (step 2) ran first and showed the narrow-prior
   posterior is unusable regardless of seed-sweep hit rate, so sweeping more
   seeds of the *same* narrow-prior recipe is not worth doing before the
   prior itself is fixed.
2. **Classification check on the fully-calibrated seed — DONE, confirms the
   S-C-style narrow-prior trap generalizes to S-B.** The SBC-calibrated
   `zscorenone` seed 0 scores macro-F1 = 0.089, accuracy = 23.6% on the
   14-scenario catalogue — severe faults shrink toward healthy regardless of
   true severity. Counter-intuitively, the SBC-*failing* baseline z-scoring
   seed does better (macro-F1 = 0.346) — SBC-calibration and classification-
   usefulness are not the same property when the training prior doesn't
   cover the deployment distribution (full detail in §0.4).
3. **Fix in progress**: widening the full prior (`mixed_wide_prior()`) was
   cost-checked at ~42s/draw, ~40% accept rate — 8-9h for a useful bank, too
   expensive to try casually. The cheaper, already-validated-for-S-C partial
   fix (`mixed_alpha_uniform_prior()`, only `alpha` widened) measured at
   ~75% accept rate, ~20s/accepted row — a pilot bank (`n_train=200`) was
   generated and trained. **Result: real improvement (macro-F1 0.089 →
   0.211, ~2.4×), not a full resolution** — see §0.4's table. `alpha` now
   shows genuine severity-tracking discrimination in the better seeds;
   `column`/`feed` remain unusable (expected — their own priors weren't
   widened). **New finding, not anticipated**: the SBC-*best* seed (all 5
   params pass) was the classification-*worst* of the three tested — do not
   use small-sample SBC pass/fail alone to pick a production seed here.
5. **Scale up, incrementally, one lever at a time** (the project's own
   established pattern, cost-checked before committing): (a) widen
   `n_train` from 200 toward 1000+ on the already-cheap `alpha`-only prior
   to see if more data stabilizes the good seeds' discrimination and raises
   the per-seed hit rate; (b) widen `eta_col`/`xi_reb` similarly (feed
   `z_A0_eff` likely does not need it, per the S-C precedent — its analyzer-
   grade signal, if added, or its own dedicated channel otherwise, makes it
   largely prior-width-insensitive) to recover `column` classification;
   (c) re-check `(alpha, z_A0_eff)`'s FIM-confirmed confound specifically
   under whichever wider `alpha` prior is adopted, since a structural
   confound's effect on classification can change with the region of
   parameter space actually being sampled.
6. In parallel, **do not treat "all parameters calibrated" as the bar for
   usability** — report classification accuracy as the primary metric
   (already the thesis's own headline per the 2026-09-28 pivot) with SBC
   status as a secondary, honestly-reported caveat per parameter, rather
   than gating the whole posterior's release on a joint SBC pass that this
   investigation twice found to be a poor proxy for real performance.
7. Options (b) (S-B3 scope reduction) and (c) (finish S-C) from the previous
   version of this section remain available **fallbacks** if 4–6 don't reach
   an acceptable macro-F1, but are no longer the first move — they were
   previously rated ahead of a full-5-D S-B retrain specifically because that
   retrain was believed to be hopeless; it no longer is, it is a tractable,
   partially-already-executed, incremental problem.
8. Whichever path is chosen, re-run the full classification table (§4) and
   the Artefact 2/3 FIM claims under that final, promoted posterior before
   citing any current number as final.

### 7.1 Re-audit Artefact 2 with the corrected methodology
`notebooks/39`-style multi-point, `cov_offdiag_corr`-based, raw-trajectory
re-validation has been done for Artefact 3 but explicitly **not yet** for
Artefact 2 (§0.3). Do this before the manuscript asserts `(alpha, beta_r)` is
"fully resolved and operationally benign" as a settled fact.

### 7.2 Rewrite `main.tex`'s Artefact 3 framing
Not a numbers update — the qualitative conclusion ("representation artefact,
resolved by raw trajectory") is wrong for `(alpha, z_A0_eff)` and
`(eta_col, z_A0_eff)`; both are genuine, confirmed limits. `notebooks/42`'s
verdict cell has a suggested framing to start from.

### 7.3 Confirm or retract Artefact 1 under the corrected physics
`(alpha, eta_col)` restricted-channel analysis predates the physics fix and
has not been explicitly re-run post-fix in what this session found — lowest
priority of the three Artefacts, but should not be assumed to transfer
without checking, for the same reason Artefact 3 didn't.

### 7.4 Decide and lock the EKF-asymmetry question (§5.3)
Either investigate why System I's filters underperform SBI everywhere while
System II's filter outperforms SBI in tracking (physics-informed feature
quality? EKF tuning quality? something else?), or explicitly scope the
manuscript's claim to avoid asserting a causal account that hasn't been
checked.

### 7.5 Once 7.0–7.3 are settled, re-verify this document's own §4 table and §6 synthesis against the final numbers
This document was written to be structurally correct independent of which
exact posterior ultimately backs it, but every *number* quoted in §4 needs a
final-posterior pass before publication.

### 7.6 System I: comparatively low-risk, mostly a documentation/integration task
System I's own identifiability chapter (`sec:identifiability`) does not have
an open calibration-failure problem the way System II does — its FIM/EKF/UKF
numbers are matched-protocol and settled (`HANDOFF.md`, "System I matched-protocol
identifiability cascade is now fully DONE"). What remains for System I is
narrower: integrate the cross-system comparison
(`notebooks/34_alpha_beta_r_banana_cross_system.ipynb`, this session) into the
Discussion if the `(alpha,beta)` vs. `(alpha,beta_r)` contrast is to be used
as a cross-system argument, and resolve the open EKF-asymmetry question
(§7.4) which necessarily involves both systems' filter implementations.
