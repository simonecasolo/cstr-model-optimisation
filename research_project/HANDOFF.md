# Session Handoff — 2026-08-18 (see the 2026-11-08 NB26 entry immediately below first — a cheap FIM-based probe finds the (alpha,beta) correlation collapses to ~0 at the K0-unit-bug-corrected operating point under the standard noisy controller, but the mechanism is genuinely mixed across controller-noise, physics-structural, and C/Qc-channel-cross-term explanations, not a single clean story — then the 2026-11-08 NB24-STOPPED entry

## 2026-11-08 SESSION-END POINTER

**Read `HANDOFF_2026-11-08_controller_fix.md` first**: the PI controller law was found malformed (Kp missing on the integral term, 150x weak integral action) and is FIXED in `physics.py` (unverified, unit test unrun); every cached System I result is invalid; System II/Luyben carry the same form (undecided).

## 2026-11-08 STEP-0 update (plan `docs/Manuscript/system_i_k0_fix_and_correlation_plan.md`, Step 0: NO Tsp-only corrected operating point exists across Sc0-Sc7 -- needs a user design decision before Step A/nb27)

Scripts: `scripts/nb27_step0_catalogue_scan.py`, `scripts/nb27_step0_window_and_knobs.py`
(read-only, no physics/simulator edits); results `results/27_step0_catalogue_scan.csv`.
At `k0=K0_NOMINAL/60` the steady-state T window reachable with Qc in [0,400] is only
~0.2-2.2 K wide per (alpha,beta) point (Sc1 [300.23,301.72], Sc2 [300.55,301.72], Sc3
[299.21,300.12], Sc4 [299.84,300.90], Sc5 [300.96,301.72], Sc7 [300.38,301.72]); prior-box
corners span 298.2-302.9 K. Intersection across Sc1-5,7 is EMPTY (even before corners, even
excluding Sc5). So nb26's Tsp=300.3 K is invalid as a general operating point: it saturates
Sc2, Sc5, Sc7 (and Sc3 is near its rail). Open-loop Sc0/Sc6 at Qc=80 settle at 300.3/300.6 K
(buggy: 312.2/313.1). Single-knob changes (Ci, Ti, UA, Tci) at nominal values do not fix it
either; only Ci>=2.5 mol/L or Ti>=~309 K widen the no-Sc5 window to ~4-11 K. Root cause:
at 47% conversion the heat release is small relative to the valve's authority (Kp=150 L/min/K
over a ~1 K window). Choice of knob (feed Ci/Ti, Q, controller gain/Qc range) is a user
decision; Steps A-D are gated on it. `main.tex`/SI/outline untouched.

## 2026-11-08 NB26 update (cheap, no-SBI/no-NUTS FIM probe of the (alpha,beta) correlation: COLLAPSES to ~0 at the K0-bug-corrected operating point under the standard noisy controller; mechanism is GENUINELY MIXED across all three candidate explanations, not reducible to one)

**Task**: nb25 (below) confirmed `K0_NOMINAL` is 60x too large (a hr-vs-min unit
mislabel), pinning the nominal operating point at ~98% conversion instead of
Fogler's intended ~47%. Separately, nb19/nb20 found a strong, seed-robust SBI
posterior Corr(alpha,beta)~-0.90 at Sc4 (alpha=beta=0.85) under the buggy operating
point. This session built `notebooks/system_I/26_cheap_correlation_prediction.ipynb`
(`scripts/build_nb_system_I_26.py`, self-contained, no edits to
physics.py/simulator.py/nb19-25/main.tex/SI/article_outline_CChE.md/K0_NOMINAL) to
cheaply check, via steady-state root-finds + finite differences + small
(120-replicate) EM-batch full-covariance FIMs (no SBI training, no NUTS -- total
wall-clock 271.5s), two questions: (1) is the correlation tied to the buggy ~98%
regime, and (2) is its mechanism controller-noise-driven, physics-structural, or a
specific-channel (C/Qc) noise cross-term, per the user's own hypothesis.

**Step 0 (minimal, loudly-flagged Tsp workaround -- NOT a production re-derivation)**:
the Tsp that maximises the worst-case valve margin at `k0_corrected=K0_NOMINAL/60`
across the healthy (1,1) and Sc4-equivalent (0.85,0.85) points is `Tsp_corrected =
300.30 K`, but the best achievable worst-case margin is only 9.54 L/min out of
[0,400] -- non-saturated but not generously centred (the two points' Qc_ss(Tsp)
curves don't overlap well at any single Tsp).

**Step 1 (steady-state sensitivities)**: confirms `dT_ss/dalpha=dT_ss/dbeta=0`
exactly at all 4 (op point x theta point) combinations (integral control pins
T=Tsp, none saturated); C/Tc/I move non-trivially under both parameters, and alpha's
`dC_ss/dalpha` grows ~9-10x from buggy to corrected (consistent with nb25's own
~14-16x alpha-sensitivity-gain finding via a different route).

**Step 2 (full-covariance FIM-implied Corr(alpha,beta), standard noisy controller,
120 reps, `cov_offdiag_corr` -- the number comparable to a real trained posterior)**:
buggy/Sc4-equiv = **-0.901** (closely reproduces nb19/nb20's independently-trained
SBI posteriors, ~-0.91 to -0.48 across seeds -- good cross-validation of this cheap
proxy); buggy/healthy = -0.195; buggy/asymmetric(0.85,1.0) = -0.303.
**corrected/Sc4-equiv = -0.029; corrected/healthy = +0.004; corrected/asymmetric =
-0.069 -- the correlation collapses to within noise of zero at every point tested
once the operating point is corrected.** Direct, clean answer to question 1: YES,
tied to the buggy regime, under the standard controller.

**Step 3 (new self-contained noise-free-controller EM-scan variant -- Qc decided by
a noise-free shadow trajectory, real C/T/Tc trajectory driven by that externally
-supplied Qc and carrying the actual process noise; sensor noise unchanged,
post-hoc)**: removing process noise from the controller's feedback does NOT merely
shrink the correlation toward zero or leave it unchanged -- it flips to a LARGE,
UNIFORMLY POSITIVE value (+0.82 to +0.97) at **every** point tested, INCLUDING the
corrected operating point where the noisy-controller number was ~0 (buggy/Sc4-equiv
noisefree=+0.918; corrected/Sc4-equiv noisefree=+0.955; full table in TODO.md/
`results/26_extra_points.csv`). This proves a real, strong, POSITIVE
physics/steady-state-structural coupling between alpha and beta exists at BOTH
operating points (consistent with Step 1) -- the correlation is not purely a noise
artifact. But the sign itself never matches between the noisy and noise-free
variants at any point tested.

**Step 4 (channel ablation on the FIM, Sc4-equivalent, standard controller, reusing
Step 2's replicates)**: at the BUGGY point, dropping C or Qc collapses the
correlation (-0.901 -> -0.174 / -0.254) while dropping T or Tc barely moves it
(-0.936 / -0.934) -- a clean, direct confirmation of the user's own C/Qc
noise-cross-term hypothesis AT THAT OPERATING POINT. At the CORRECTED point the
baseline is already ~0, so "collapse" isn't meaningful the same way; the most
notable change is dropping C, which flips the sign to +0.307 -- not a clean analog of
the buggy-point pattern.

**Honest verdict, both questions, not force-fit to a single clean story**:
1. Tied to the buggy regime? **YES**, under the controller representation actually
   comparable to a real trained posterior (standard/noisy): the correlation collapses
   from -0.90/-0.20/-0.30 (buggy) to -0.03/+0.00/-0.07 (corrected) at the three theta
   points tested.
2. Mechanism? **GENUINELY MIXED -- all three candidate explanations have real,
   simultaneous supporting evidence, not reducible to one.** A robust positive
   physics-structural ridge exists at both operating points (Step 3/Step 1) -- so it
   is not *purely* noise. A noise-mediated effect tied specifically to the C and Qc
   channels (Step 4) is large enough at the BUGGY point to not just cancel but REVERSE
   the sign of that structural ridge (net -0.90 vs. underlying noise-free +0.92),
   while at the CORRECTED point the two effects roughly cancel to near zero (net
   -0.03 vs. underlying noise-free +0.95) rather than either dominating. This mixed
   finding is reported as-is rather than resolved toward whichever sub-result matches
   the user's prior C-Qc hypothesis most cleanly (which IS supported, but only at the
   buggy point, and only as one of (at least) two interacting effects).

**Caveats, explicit**: this is a cheap Laplace/FIM proxy (`cov_offdiag_corr`), not a
real trained SBI posterior at the corrected operating point -- the near-zero
corrected-point numbers should be spot-checked against an actual SNPE retrain before
being treated as final (not done here, out of budget/scope). Diag-vs-full-cov FIM
numbers sometimes disagree substantially (e.g. corrected/asymmetric: full-cov -0.069
vs. diag +0.533), underscoring that off-diagonal feature-covariance structure (not
just per-feature variance) matters a lot -- consistent with and further supporting
the channel-cross-term finding. No code/manuscript changes made; nb19-25 untouched.
Results: `results/26_summary_table.csv`, `results/26_extra_points.csv`,
`results/26_step0_operating_points.csv`, `results/26_step1_sensitivities.csv`,
`figures/26_step0_tsp_scan.png`, `figures/26_summary_grid.png`.

**Suggested next step, not started**: if this mixed mechanistic finding is worth
pursuing further, the natural follow-up is an actual SNPE retrain at the corrected
operating point (not just the FIM proxy) to confirm the near-zero correlation holds
for a real trained posterior, and/or a targeted look at WHY the C/Qc channel pair
specifically reverses the structural ridge's sign at the buggy point (the FIM
machinery here shows THAT it happens but not a first-principles WHY).

---

## 2026-11-08 NB24-STOPPED update (CNN-vs-hand-crafted discrepancy investigation: H1/H2 COMPLETE with findings standing — CNN posterior severely uninformative about alpha, slope≈0.05, and a reproducible ~5-nat held-out log-density deficit vs. hand-crafted summary — H3 (5-seed CNN robustness) INCOMPLETE after 2 reproducible mach-port-exhaustion crashes, H4 never run; execution PAUSED by explicit user decision once nb25 (immediately below) found the K0 unit bug, not finished, not abandoned-as-failed)

**Task**: `scripts/build_nb_system_I_24.py` (built fresh this session; no notebook
rendered, per this project's own house rule of using a standalone script for any
long-running investigation — same convention as nb20-23) was built to investigate
WHY the hand-crafted-summary (HC) and raw-trajectory-CNN (Theory F) SBI posteriors
disagree so persistently on Corr(alpha,beta) at nb22's optimal n=50,000 budget, even
getting WORSE after the warm-start fix (4/6 closed-loop scenarios sign-disagree,
sharpening at Sc4 specifically — see `finding_system1_no_alpha_beta_banana.md`'s
2026-11-08 three-way-cross-check entry for full prior context). Four hypotheses
were planned: H1 CNN-side SBC/anti-shrinkage-slope calibration check, H2 held-out
predictive log-density (HC vs CNN), H3 CNN multi-seed robustness (3→5 seeds), H4
qualitative feature-space comparison.

**H1 (COMPLETE, reused the 3 already-cached CNN posteriors from nb22's own
`nb22_cnn_theoryF_recheck.py`, no retraining needed)**: the CNN-embedding posterior
has essentially collapsed on alpha — `slope_alpha` ≈ 0.046-0.053 across all 3 seeds
(vs. the hand-crafted summary's ≈0.99 at the same n=50,000 budget, per nb22's own
table), while `slope_beta` is much healthier (≈0.73-0.93). SBC: alpha nominally
"passes" (p=0.36-0.81 — consistent with calibrated-but-uninformative, not
miscalibrated), beta FAILS outright (p=0.000) at 2/3 seeds. Files:
`results/24_cnn_sbc_metrics.csv`, `results/24_cnn_grid_diagnostics.csv`,
`figures/24_cnn_sbc_rank_histograms.png`.

**H2 (COMPLETE, same theta/noise draws fed through both representations)**:
hand-crafted mean held-out log p(theta|x) ≈ +7.3 (consistent across all 5 HC seeds,
std≈0.01); CNN-raw ≈ +2.0-2.2 (consistent across all 3 CNN seeds, std≈0.13) — a
clear, reproducible ≈5-nat gap favoring the hand-crafted summary. Files:
`results/24_heldout_logprob_comparison.csv`, `figures/24_heldout_logprob_comparison.png`.

**H1+H2 together → leading hypothesis (NOT a final SBC-blessed conclusion, H3/H4
were meant to corroborate further and did not complete)**: the specific CNN
architecture reused unchanged from `04b_embedding_net_study.ipynb`/nb20's Theory F
extracts substantially less usable information about alpha specifically from the raw
trajectory than the hand-crafted 29-D summary extracts from the same signal — so the
CNN's reported Corr(alpha,beta) is more likely an artifact of this information
deficit than a competing, equally-trustworthy estimate of the true posterior
geometry. This points to an architecture/capacity explanation for the HC-vs-CNN sign
disagreement, not a "two representations reveal genuinely different physics"
explanation.

**IMPORTANT CAVEAT added after nb25's K0-unit-bug finding (read that entry below)**:
H1/H2's numbers were computed at the CURRENT, now-confirmed-buggy ~98%-conversion
operating point. nb25 independently found alpha-sensitivity is ~14-16x larger at the
corrected ~47%-conversion point. This does not contradict H1/H2 (the hand-crafted
summary already extracts real alpha signal at the current, low-sensitivity point;
the CNN's near-total failure on the exact same data is still a distinct, real
architectural deficiency) — but these specific numbers must be RE-VERIFIED once the
K0 fix lands, since a 14-16x change in available alpha signal could change the CNN's
calibration picture substantially, not just rescale it.

**H3 (INCOMPLETE — stopped by explicit user decision, not a silent failure)**:
attempted 3 times to train 2 more CNN seeds (seed3, seed4) at n=50,000 for 5-seed
parity with HC's own nb22 budget study. Hit a reproducible OS-level crash TWICE: the
kernel SIGKILLed the process both times (`caught allocating too many mach ports,
Num of ports allocated ~267,625`) — a genuine mach-port-exhaustion bug in the long
per-sample `simulate_for_sbi`/`simulation_wrapper_sbi_raw` loop
(`src/cstr_sbi/inference.py`, shared production code, NOT modified by this
investigation), once at 84% through seed3's training and once at 89% — both in
fresh single-process runs, ruling out cross-seed port accumulation. Puzzle left
open: the 3 already-cached CNN seeds completed this exact code path successfully in
an earlier session, so the crash is not 100%-reproducible every time; root cause not
diagnosed, no fix attempted (would require modifying shared `inference.py`, out of
scope for this investigation). **The user killed a 3rd attempt mid-launch and
explicitly instructed: stop retrying, leave nb21-24 as written but unexecuted
further, pending the K0 operating-point fix** — this is a deliberate pause, not an
abandoned/broken run. The pre-existing 3-seed CNN evidence
(`results/22_theoryF_cnn_recheck.csv`) remains the only seed-robustness evidence for
the CNN side.

**H4: never run for real** — only smoke-tested at reduced size (n_noise=3 instead of
20) during script development (discriminability hand-crafted=408 vs CNN=168,
suggestive but far too small a sample to report).

**Code status, per explicit user instruction**: `scripts/build_nb_system_I_24.py` is
COMPLETE and FINALIZED as written — syntax-valid, all 4 hypotheses (`run_h1`-`run_h4`)
implemented, resume logic in `main()` (skips H1/H2 recomputation if their CSVs
already exist, goes straight to H3). It does not need to have been executed to
completion to be considered done as CODE — it is deliberately left unexecuted further
pending the K0 fix (nb25), after which the whole script, and the cached posteriors it
depends on (`results/22_budget50000_seed{0-4}.pkl`,
`results/22_theoryF_cnn_posterior_seed{0,1,2}_n50000.pkl`), will need retraining at
the corrected operating point before re-running. `notebooks/system_I/TODO.md` updated
with the matching entry (prepended). No notebook file was rendered (script is the
deliverable, same house rule as nb20-23). `main.tex`, the SI, `article_outline_CChE.md`,
and nb19-23's own files were not touched.

## 2026-11-08 NB25-K0-UNITS-BUG update (CONFIRMED: `K0_NOMINAL` in `src/cstr_sbi/physics.py` is 60x too large due to a hr-vs-min unit mislabel — NOT YET FIXED, project-wide implications flagged, no code/manuscript changes made)

**Task**: an external literature-review document, pasted into chat (not in this
repo), claimed `K0_NOMINAL = 16.96e12` (commented `# 1/min`) is actually a
per-**hour** value from Fogler's propylene-oxide hydrolysis CSTR example
(Furusawa et al. 1969), making the coded constant 60x too large and pinning this
project's nominal operating point at ~98% conversion (tiny alpha-sensitivity)
instead of Fogler's intended ~47% (much larger alpha-sensitivity) — a candidate
independent, physics-level explanation for this project's long-running struggle to
extract clean alpha information (see `finding_system1_no_alpha_beta_banana.md`'s
nb24 entry: CNN posterior-mean slope on alpha ~0.05 vs hand-crafted's ~0.99 at the
same budget). Built a fast, verification-only notebook
(`notebooks/system_I/25_k0_units_verification.ipynb`,
`scripts/build_nb_system_I_25.py`) to fact-check this rigorously against primary/
independent sources rather than trust the literature document's citation chain.

**Verdict: CONFIRMED, with a real caveat about remediation.**
1. **Three independently-phrased web searches, three independent pages, zero
   disagreement**: every source that states explicit units for the
   pre-exponential factor numerically equal to `16.96e12` gives **hr^-1**, never
   min^-1 — including COMSOL's own documented reproduction of this exact Fogler
   3rd ed. Example 9-4 model (`A1=4.71e9 s^-1` = `1.696e13 hr^-1`, converts exactly
   to `16.96e12 hr^-1`), and a second independent citation of the Furusawa data
   whose activation energy (75362.4 J/mol) matches this codebase's own
   `E_A=75362.0 J/mol` to 6 significant figures — not a coincidence.
2. **Internal-consistency check, using only this codebase's own `E_A` and the
   Arrhenius equation** (no external citation trusted): the as-coded value gives
   a closed-form steady-state conversion X=98.2%, matching `main.tex`'s own stated
   "~98%" operating point (`sec:identifiability` context) almost exactly; the
   candidate-corrected value (`K0_NOMINAL/60`) gives X=47.2%, matching the
   external document's claim about Fogler's intended operating point almost
   exactly. Two independent external claims both independently reproduced from
   this project's own numbers via a from-scratch calculation.
3. **Direct evaluation of `physics.py`'s real closed-loop steady-state solver**
   (`closed_loop_steady_state_exact`, no retraining) confirms alpha-sensitivity
   `dX/dln(alpha)` is **~14-16x larger** at the corrected operating point —
   matching the external document's `~14x` claim.

**Real subtlety found only by running the actual closed-loop model (not in the
hand calc or the external document)**: naively dividing `K0_NOMINAL` by 60 in
isolation, with every other constant unchanged, drives the closed-loop healthy
operating point into **permanent coolant-valve saturation** (`Qc=Qc_min=0`, `T`
settles at ~301.7K instead of `Tsp=312.5K`) — this project's own PI-controller
tuning (`Tsp=312.5`, `Qc0=80`, ...) appears to have been chosen to work with the
as-coded, apparently-buggy `k0`, not Fogler's real value. **This means the fix is
not a one-line constant edit**: a correct remediation requires re-deriving a
consistent nominal operating point (setpoint/feed conditions/controller tuning)
alongside the kinetic constant, which was explicitly out of scope for this
verification-only notebook and was NOT done.

**What this means for the rest of the investigation, stated as a hypothesis, not
a resolution**: if the nominal operating point has genuinely been pinned at ~98%
conversion this whole time, every representation this project has tested
(hand-crafted summary, CNN embedding, NUTS) has had ~14-16x less real alpha
signal available in the observed trajectory than it would at Fogler's intended
~47% point — a purely physics-level headwind that nb19-24's entire
representation/training-level investigation never checked. This does NOT
contradict nb24's CNN finding (the hand-crafted summary still extracts real
alpha signal at the current point, slope~0.99; the CNN's near-total failure at
the identical point is still a separate, real representation-level deficiency),
but it predicts that correcting the operating point should help EVERY
representation, not just explain away the CNN-specific gap — not tested here.
Also flagged (not a correction, a caution): `main.tex`'s existing ~236x FIM ratio
(`eq:fim_ratio`) is a *relative* alpha-vs-beta comparison at this same,
currently-buggy operating point, not an absolute measure of how recoverable alpha
actually is for a finite, noisy amortised estimator — this finding is a reason for
caution before citing that ratio as evidence alpha is "easy" in an absolute sense.

**Explicitly NOT done, by design/scope**: `K0_NOMINAL` itself was NOT changed in
`src/cstr_sbi/physics.py` — changing a live constant used by every other System I
notebook (several cited in the manuscript) needs an explicit user decision, not a
unilateral edit from a verification notebook. `main.tex`, the SI, and
`article_outline_CChE.md` were NOT touched. No SBI retraining was run. nb19-24's
own files were not touched (nb24's CNN investigation was running concurrently in
the background in this same session and was left alone).

**Suggested next step, not started**: decide with the user whether/how to
re-derive a consistent corrected operating point (this will require touching
`Tsp`/`Qc0`/feed conditions, not just `K0_NOMINAL`), since any such change
invalidates every cached System I posterior and SBC/FIM number produced by any
prior notebook in this project, including numbers already cited in `main.tex`.

## 2026-11-08 NB23-STOPPED update (JAX-differentiable reimplementation of `closed_loop_steady_state_exact` implemented and verified for correctness/differentiability, FIXES the original `ConcretizationTypeError` crash -- but the actual decisive NUTS test it was built to re-enable comes back with R-hat~500, a NEW and WORSE problem; investigation STOPPED by explicit user decision before diagnosing it, not resolved)

**Task**: the entry below ("NUTS-REGRESSION update") found that nb22's warm-start
root-find fix (`closed_loop_steady_state_exact`, `src/cstr_sbi/physics.py`) broke
NUTS's `matched_warm_start=True` path outright (`float()`/`scipy.optimize.brentq`
cannot be traced by JAX; instant `ConcretizationTypeError` crash at Sc4/Sc5). This
session's job was to reimplement that function JAX-natively (fixed-iteration
`jax.lax.fori_loop` bisection + `jnp.where` branch selection in place of
`brentq`/Python `if`) so NUTS works again, verify correctness/differentiability are
preserved, and re-run the original decisive single-window test (Sc4,
`matched_warm_start=True`, full-covariance likelihood, 4 chains/500 samples/300
warmup -- `scripts/build_nb_system_I_20.py` Section 5 Step 2's protocol).

**Part 1 DONE and verified: the differentiability crash is fixed.** No `float()`
calls remain in `closed_loop_steady_state_exact`. Directly confirmed (not assumed):
new implementation matches the old scipy implementation to ~1e-11 or better at all 6
closed-loop scenarios, with `|rhs|` at the fixed point (4e-16 to 3.5e-13) actually
BETTER than the old implementation's own (1e-4 to 1e-7); `jax.grad` succeeds (finite)
at a non-saturated point (Sc4) and two saturated points (Sc2, Sc5); a cheap
`run_mcmc_baseline(..., matched_warm_start=True)` smoke test (1 chain, 20/20)
completed without crashing at Sc4 (12.7s) and Sc5 (4.5s) -- the exact call that
previously crashed instantly.

**Part 2: the actual decisive test surfaced a NEW, worse, unresolved problem.**
Running the real protocol (Sc4, 4 chains, 500 samples, 300 warmup, full-covariance
`cov_obs`, `matched_warm_start=True`) gave:

```
wall=3404.9s  corr=-0.114
r_hat alpha: 505.7   r_hat beta: 346.9
```

R-hat in the HUNDREDS -- far worse than any other R-hat seen anywhere in this
project's whole NUTS investigation (even the known diagonal-likelihood
misspecification bug only reached R-hat~3-11; the pre-regression differentiable
version, i.e. before nb22's root-find fix existed, got clean R-hat~1.0-1.01 under
this identical full-covariance protocol). Also ~57 minutes wall time for one window
-- at the high end of this project's documented heavy-tailed NUTS cost tail, but not
itself the anomaly; the R-hat is.

**Leading, UNTESTED hypothesis (coordinator's, recorded for whoever resumes this)**:
a gradient-QUALITY problem, not a gradient-CORRECTNESS problem -- differentiating
straight through a fixed-iteration bisection's discrete internal branching can give a
gradient that is a poor local approximation of the true implicit derivative, even
though it is finite and "looks sane" when probed at a single point (as was done in
Part 1's differentiability check). The mathematically correct fix would be implicit
differentiation (`jax.custom_vjp` via the implicit function theorem:
d(root)/d(theta) = -(dF/dtheta)/(dF/dT) at the root -- cheap, exact, no need to
differentiate through the iteration).

**Second, equally plausible, also UNTESTED hypothesis**: the new implementation
always evaluates BOTH the saturated and non-saturated candidate branches (required
for `jnp.where` to select between them), unlike the old Python `if`, which only ever
evaluated one. This is exactly the setup for the classic JAX `0 * inf = nan`
backward-pass footgun: if the unselected branch's algebra hits a hidden singularity
(a division that's fine at the inputs the selected branch is evaluated at, but blows
up for inputs the OTHER branch happens to see during some HMC leapfrog excursion),
`jnp.where`'s backward pass can silently poison the whole gradient with NaN even
though the forward value and a single-point gradient probe both look completely
fine. A diagnostic script to test both hypotheses (autodiff-vs-finite-difference
gradient comparison over a grid; a denominator-singularity scan of both branches over
a wide theta range; checking NumPyro's actual divergence count) was WRITTEN but
**NEVER RUN** -- the user explicitly stopped the investigation first. The script
lived only in the session scratchpad, not the repo; it is not preserved anywhere and
would need to be re-derived.

**Explicit status: STOPPED BY EXPLICIT USER DECISION, not resolved, not abandoned as
a dead end.** The fix in `src/cstr_sbi/physics.py` is being LEFT IN PLACE (real,
independently-verified improvement for the original crash, and every other call site
in the project only ever evaluates it eagerly/non-differentiated and is unaffected by
this NUTS-sampling-quality question). **But "NUTS is back online" is NOT the correct
read of this session** -- it no longer crashes, but produces badly wrong output
(R-hat~500) under the one full protocol tested, arguably a worse trap than the crash
it replaced (a crash is obviously broken; a silently-completing R-hat~500 run that
still prints a plausible-looking correlation number is much easier to cite by
accident). **Do not use any NUTS correlation number produced under this fix for any
purpose until this is diagnosed and fixed.** No background processes were left
running (verified clean at wind-down). `notebooks/system_I/23_nuts_differentiable_warmstart_fix.ipynb`
does **not** exist -- `scripts/build_nb_system_I_23.py` was written (nbformat/nbclient,
following nb20/21/22's conventions) but never executed, per the explicit stop
instruction. No `results/23_*`/`figures/23_*` files exist. `notebooks/system_I/TODO.md`
updated with the matching entry (prepended).

## 2026-11-08 THEORY-F-RECHECK update (CNN-raw-trajectory recheck at corrected warm-start + optimal n=50,000 COMPLETE: the CNN-vs-hand-crafted (alpha,beta) sign flip does NOT resolve -- at Sc4 it sharpens; also found the warm-start fix flips the HAND-CRAFTED summary's own correlation sign at Sc2/Sc7, independent of CNN)

Coordinator asked whether Theory F's CNN-embedding posterior agrees with the
hand-crafted-summary posterior once both are trained under the corrected warm-start at
the nb22-determined optimal budget (n=50,000). Cost-checked first (benchmarked n=1000:
29s, n=5000: 142s under the fix -- linear scaling, no sign of the old pre-fix 7h41m
stall at n=40,000), then ran 3 seeds at n=50,000 with the identical `04b`/nb20 CNN
architecture (`scripts/nb22_cnn_theoryF_recheck.py`, self-caffeinated background,
completed cleanly in ~95 min, no stall).

**Direct answer: no, they do not agree, and at Sc4 (the priority scenario) the
disagreement is UNCHANGED in direction and SHARPER in magnitude than before the fix.**
Full three-way table in `results/22_theoryF_three_way_comparison.csv`:

| scenario | HC pre-fix | CNN pre-fix (n20k) | HC post-fix (n50k) | CNN post-fix (n50k) | sign agree? |
|---|---|---|---|---|---|
| Sc1 healthy | -0.650 | +0.777 | -0.719 | +0.785 | NO |
| Sc2 fouling | -0.856 | +0.920 | **+0.959** | +0.286 | yes |
| Sc3 decay | -0.638 | +0.740 | -0.799 | +0.748 | NO |
| **Sc4 combined** | **-0.893** | **+0.868** | **-0.910** | **+0.935** | **NO** |
| Sc5 saturated | +0.850 | +0.314 | +0.919 | +0.150 | yes (6x magnitude gap) |
| Sc7 drift | +0.632 | +0.886 | **-0.907** | +0.948 | NO |

Both methods became MORE confident in their own, opposite, directions at Sc4 (HC:
-0.893->-0.910; CNN: +0.868->+0.935). 4/6 scenarios disagree in sign both before and
after the fix. This is NOT the three-independent-methods-converge result that would
have been the strongest confirmation of the correlation being real -- the honest
reading is that hand-crafted-summary vs. raw-trajectory-CNN representation drives a
large, persistent, warm-start-independent disagreement that Theory F's original
"compression artifact" hypothesis still cannot explain.

**Unexpected side-finding, unrelated to CNN, found while completing the comparison
table**: computed the hand-crafted posterior's own Corr(alpha,beta) at all 6 scenarios
under the corrected warm-start (cheap -- reused nb22's already-trained n=50,000
posteriors, sampling only, no retraining) and found the warm-start fix flips the
HAND-CRAFTED summary's own correlation SIGN at **Sc2** (fouling: -0.856->+0.959) and
**Sc7** (drift: +0.632->-0.907). This is a hand-crafted-pre-fix-vs-post-fix comparison,
nothing to do with the CNN. It is consistent with the original warm-start audit's
finding that the old bug's injected artifact was strongly scenario-dependent (the
60-min-window `Qc` discrepancy was ~35x larger at Sc2 than at healthy/Sc4) -- Sc2/Sc7
were evidently among the scenarios where that artifact was large enough to flip the
inferred correlation's sign outright. **Any pre-fix manuscript-facing claim about
Sc2's or Sc7's (alpha,beta) correlation sign should now be treated as unreliable**,
independent of the Theory F/CNN question.

Files: `results/22_theoryF_cnn_recheck.csv` (new CNN numbers),
`results/22_handcrafted_n50k_all_scenarios.csv` (new hand-crafted numbers across all 6
scenarios), `results/22_theoryF_three_way_comparison.csv` (the merged table above).
`notebooks/system_I/TODO.md` updated with the matching entry (prepended).
`main.tex`/SI/`article_outline_CChE.md`/nb20's/nb21's own files NOT touched.

## 2026-11-08 NUTS-REGRESSION update (coordinator-requested differentiability check FAILED: the warm-start root-find fix is not JAX-differentiable, so NUTS's `matched_warm_start=True` now crashes instantly instead of running slowly; CNN-raw-trajectory Theory F recheck at n=50,000 launched separately, see its own entry once complete)

Coordinator asked (alongside the Theory F CNN recheck below) to re-test NUTS under
the corrected warm-start, since Theory D previously found `matched_warm_start=True`
made NUTS ~11x slower per window (28s->310s) by differentiating through the old
warm-start's long forward ODE integration, and the new root-find is ~640x cheaper to
*evaluate* -- the open question was whether it's also cheap/well-behaved to
*differentiate through*.

**Checked explicitly, per the coordinator's instruction, before trusting any timing
result: it is not differentiable at all, and this is a real regression.**
`jax.grad` through `warm_start_ic` raises `ConcretizationTypeError` (`float()` called
on a traced value) -- `closed_loop_steady_state_exact` uses plain `float()`/
`scipy.optimize.brentq`, neither of which JAX can trace. Confirmed end-to-end (not
just the isolated grad test): `run_mcmc_baseline(obs, matched_warm_start=True)` at Sc4
and separately at Sc5 both crash with the identical error in ~1.4s -- it fails
immediately, before any real NUTS sampling happens. **Net effect: `matched_warm_start=True`
NUTS, which used to work (slowly) before this fix, is now completely unusable** --
a regression introduced by fixing the SBI-side bug, specific to the one use case
(differentiated gradients) that the fix's scipy-based implementation cannot support.
Every other use of `warm_start_ic` in the project (all SBI training/eval, which only
ever calls it eagerly, never under `jax.grad`) is unaffected and already verified
working (nb22's full retrain sweep completed cleanly on it).

**Confirmed, not assumed, that this does NOT affect the open-loop path**: Sc0/Sc6's
fixed, theta-independent warm start (`scripts/nb21_nuts_worker.py`, and
`cstr_generative_model`'s default `y0`) computes `warm_start_ic` eagerly outside any
gradient computation -- ran a minimal `mode="open_loop"` NUTS test end-to-end and it
succeeded normally.

**No single-window or multi-window NUTS re-test was run for the closed-loop
matched-protocol scenarios (Sc1-5,7) the coordinator asked about** -- there is nothing
to time or measure Corr(alpha,beta) from; every one of those runs would hit the
identical instant crash regardless of budget, window count, or timeout. This is a
hard correctness blocker, not a cost/scheduling problem, so no amount of the
compute-budget discipline this project has otherwise relied on today would help here.

**Not fixed this session, flagged for a decision rather than silently attempted**: a
real fix needs a JAX-native differentiable reimplementation of the root-find (e.g. a
fixed-iteration `jax.lax.while_loop` Newton iteration in place of
`scipy.optimize.brentq`, and a `jnp.where`-based differentiable formulation of the
saturated-vs-non-saturated branch selection in place of the current Python `if`) --
a nontrivial additional task with its own correctness risk (the saturated branch in
particular would need care to stay differentiable through the bound-selection logic).
Not attempted without the coordinator's sign-off first. `notebooks/system_I/TODO.md`
updated with the matching entry (prepended).

## 2026-11-07 FIX update (`warm_start_ic` root-find fix implemented, verified, and propagation-checked — coordinator-approved; INVALIDATES every previously-cached posterior trained with `scenario_specific_warm_start=True`/`matched_warm_start=True`; nb22 Part 2/3 full retrain now launched on the corrected version, see separate entry once it completes)

Following the audit below (which found `warm_start_ic`'s default `t_warm=1500`
forward-integration left the PI integrator state un-converged), implemented and
verified a fix, with the coordinator's explicit sign-off before launching the
downstream retrain.

**Fix**: `warm_start_ic` (`src/cstr_sbi/simulator.py`) now calls a new function,
`closed_loop_steady_state_exact` (`src/cstr_sbi/physics.py`), that solves the
steady-state algebraic equations directly via root-finding instead of forward-
integrating. Non-saturated branch: `dI/dt=0` (gate open) pins `T=Tsp` exactly
(integral control removes steady-state error); `C`, `Tc`, `Qc`, `I` follow in closed
form, no iteration. Saturated branch (Sc2/Sc5's heavy fouling, where even max coolant
flow can't reach `Tsp`): `Qc` pinned at the active bound, 1-D `brentq` root-find in `T`
(Arrhenius-nonlinear), `C`/`Tc` follow in closed form.

**Two sub-bugs caught and fixed while building this, not shipped broken**: (1) the
first attempt picked the saturated bound from the sign of the non-saturated
candidate's implied `Qc`, which is backwards for Sc2/Sc5 (`Tc` is physically bounded
in `[Tci,T]` for any `Qc>=0`; a candidate implying `Tc<Tci` means "insufficient cooling
even at infinite flow", requiring `Qc_max`, not `Qc_min` -- the first attempt picked
`Qc_min` and got `T=316.4` instead of the correct `T=312.87`). Fixed by evaluating
`T_ss` at both bounds directly and using `T_ss(Qc)`'s verified monotonicity to decide
feasibility. (2) landing `I` exactly on the saturation boundary let float32 rounding of
a large `I` occasionally reopen the anti-windup gate, giving a spurious nonzero `dI` at
an otherwise-exact point -- fixed with a small one-sided epsilon push past the
boundary.

**Verification, same methodology as the audit that found the bug**: `|rhs|` at the new
warm start is `1e-6`-`1e-4` at every Sc0-7 closed-loop point (vs. the original
`0.02`-`0.42`), 50-min forward-redrift is `1e-5`-`1e-3` (vs. `1`-`21`). Cross-checked
against independent brute-force long-horizon integration (old `t_warm=20000`): Sc2
gives `I=2646.8,T=312.869,Tc=297.340` (new exact solve) vs. `I=2645.3,T=312.870,
Tc=297.340` (slow brute force) -- matches, confirming the root-find found the same
physical fixed point.

**Propagation, traced not assumed**: `warm_start_ic` is the single source of truth for
System I -- `scenarios.py`, `inference.py`'s `cstr_generative_model`
(`matched_warm_start=True`) and `run_mcmc_baseline`, and `simulation_wrapper_sbi`/
`_raw`'s `scenario_specific_warm_start=True` (SBI's actual training-data generator) all
call this exact function, so the fix reaches everything automatically, including
nb22's own retrain path. `scripts/nb21_nuts_worker.py`'s open-loop path also calls it
(for its fixed, theta-independent warm start) -- auto-fixed too, without changing
nb21's own investigation logic.

**Two explicit non-fixes, flagged rather than left silently inconsistent**: (1)
`cstr_generative_model`'s default `y0` (`matched_warm_start=False` path) is
`NOMINAL_Y0_CL`, a hand-set module constant, not computed via `warm_start_ic` --
pre-existing, documented as deliberately approximate, untouched. (2)
`src/cstr_sbi/luyben/simulator.py` has its own, completely separate `warm_start_ic` for
the Luyben/System II model -- a different codebase, NOT audited or touched.

**Side benefit**: the new root-find is ~640x faster than the old forward-integration
(~0.55ms vs ~350ms per call) -- the matched-warm-start training overhead that used to
be a real budget concern is now negligible.

**Consequence stated plainly**: this fix changes the data-generating process for every
SBI/NUTS run that used the per-theta matched warm start. Every previously-cached
posterior trained with `scenario_specific_warm_start=True`/`matched_warm_start=True`
(including nb19's n=10,000x10-seed posteriors and nb20's budget-sweep cache) is now
built on an invalidated warm start and must not be reused/cited as current without
retraining. nb22 Part 2/3's full 6-budget x5-seed retrain sweep is now launched on the
corrected version (coordinator-approved, ~3.5h projected serial wall time, see cost
estimate in the audit entry below) -- see the separate entry written once it completes
for results. Files touched: `src/cstr_sbi/physics.py` (new function, additive),
`src/cstr_sbi/simulator.py` (`warm_start_ic` body, signature unchanged).
`notebooks/system_I/TODO.md` updated with the matching entry (prepended, same content
structure). `main.tex`/SI/`article_outline_CChE.md` NOT touched.

## 2026-11-07 audit-only update (nb22 warm-start audit: `warm_start_ic` found NOT converged at its default `t_warm=1500`, PI integrator state is a slow mode that is still actively drifting — blocking nb22's Part 2/3 budget sweep, flagged as a confound for every prior SBI/NUTS result built on this warm start — SUPERSEDED by the FIX entry above, kept for history of how the bug was found)

Tasked with building `notebooks/system_I/22_sbi_full_convergence_study.ipynb` to
resolve the still-open SBI-training-budget-convergence question flagged repeatedly
below (10k->20k->40k Corr(alpha,beta) drift at Sc4 never converged). The brief's Part 1
explicitly required auditing `warm_start_ic`'s physical correctness FIRST, before
trusting any training run — this audit found a real, previously-unchecked problem that
blocks proceeding to Part 2/3 as specified.

**Finding, numerically verified, not inferred**: `warm_start_ic`
(`src/cstr_sbi/simulator.py` -> `simulate_closed_loop_to_steady_state`,
`src/cstr_sbi/physics.py`) forward-simulates the closed-loop ODE for a fixed
`t_warm=1500` minutes and returns the final state as "the" steady-state IC. The
physical states (`C`, `T`, `Tc`) do converge within that horizon
(`|rhs| <~1e-4` at every Sc0-7 point checked), but the PI integrator state `I` does
NOT — at `t_warm=1500`, `dI/dt = T - Tsp` (its un-gated residual) is still `-0.086`
(healthy) up to `+0.425` (Sc2 fouling), actively drifting, not at a fixed point.
Escalating to `t_warm=20000` improves but does not obviously finish this (`I` itself
is still 4x different between the two horizons at Sc2: `660` vs `2645`), and naively
escalating further (`t_warm=100000`) hits a second, compounding bug: `diffeqsolve`'s
`max_steps=2_000_000` is silently exhausted (`throw=False` swallows it) and the
function returns a state truncated at `t~64056`, not the requested horizon, with no
warning. Only Sc5 (severe saturation) "converges" at `t_warm=1500`, and trivially —
its anti-windup gate freezes `dI=0` exactly because the actuator is saturated, not
because the unsaturated dynamics settled.

**Checked whether this actually matters for real training/eval data, not just as an
abstract tolerance issue**: resimulated the real 60-minute observation window
(`t_window=60`, SBI's actual default) from the `t_warm=1500` vs `t_warm=20000` warm
starts. Realized `Qc` (coolant flow, `[0,400]` range) differs by up to **190 (Sc2
fouling)**, vs only ~5.5-5.8 at healthy/Sc4 — the artifact size is itself strongly
**(alpha,beta)-dependent**, which is exactly the signature needed to inject a spurious
posterior correlation. `T`/`C` over the same window are barely affected
(`<~0.09` K / `<~1.4e-4` mol/L) because they are fast, self-correcting states; `Qc` is
not. This is a genuinely new candidate mechanism for the (alpha,beta) correlation
question this whole multi-notebook investigation (nb19/nb20, see entries below) has
been chasing, and it was never isolated before now because no prior check compared
`warm_start_ic` outputs across different `t_warm` horizons or against the ODE's own
RHS.

**Explicit, plain statement, not glossed over**: `warm_start_ic` is NOT a correct
closed-loop fixed point at its current default for every Sc0-7 point checked except
the trivial Sc5 case. This is a confound for (a) nb22's own commissioned budget sweep
(Part 2/3, NOT executed this session — see below), (b) every prior SBI training run
using `scenario_specific_warm_start=True` (nb04 onward), and (c) today's earlier
`matched_warm_start` NUTS fix (Theory D, entry immediately below), which reuses the
identical `warm_start_ic` call and therefore inherits the same non-convergence — so
the "real, bilateral SBI+NUTS correlation at Sc4" verdict in the entry below should now
be read as resting on an unaudited warm-start foundation, not as an independent
re-confirmation immune from this issue.

**Not fixed, not patched around, per the brief's explicit instruction to stop and flag
on finding a discrepancy rather than proceed.** nb22's Part 2 (budget sweep,
1000-50000 sims/seed) and Part 3 (optimal-budget recommendation) were NOT run this
session — training/SBC-validating new posteriors on top of a known-incorrect,
scenario-dependent warm-start artifact would produce numbers that need to be redone
once this is fixed, which would waste the compute the brief was explicitly careful
about. Concrete next step for a future session: fix `warm_start_ic` (raise `t_warm`
substantially AND `max_steps`, with an explicit `|rhs| < tol` convergence assertion
before returning, rather than trusting a fixed horizon blindly — or solve for the true
fixed point directly via root-finding instead of forward-simulation), re-run this same
numerical audit to confirm convergence, THEN proceed to nb22 Part 2/3. Audit numbers:
`results/22_warmstart_audit.csv`, `results/22_warmstart_window_impact.csv`.
`notebooks/system_I/TODO.md` updated (prepended) with the same finding. No notebook
file was committed for nb22 this session (the audit was done via standalone scripts,
per the same "don't commit an unvetted build" discipline as nb19/20/21) —
`scripts/build_nb_system_I_22.py` was NOT created, since the brief's Part 1 gate was
not cleared. `main.tex`/SI/`article_outline_CChE.md` NOT touched.

## 2026-11-07 LATE update (Theory D's warm-start-protocol mismatch root-caused, fixed, validated at full scale — REVERSES the same-day Theory C verdict below: the (alpha,beta) correlation now has real bilateral SBI+NUTS support for Sc1-4)

Coordinator-directed follow-up to Theory D's "partially supported" status (see the
entry immediately below for the context this builds on). Found a second real bug,
independent of the diagonal-covariance one already fixed that same day:
`cstr_generative_model` (NUTS's likelihood target, `src/cstr_sbi/inference.py`) used a
FIXED warm start (`y0`, default healthy-nominal) for EVERY posterior draw, never
recomputed from the sampled `(alpha,beta)` — while the data it was asked to fit was
generated with a PER-THETA matched warm start (`warm_start_ic(params,...)`, the same
convention SBI's own training already uses via `scenario_specific_warm_start=True`).
NUTS's internal model and the data-generating process were two different models — the
exact same train/eval protocol mismatch already found and fixed for SBI's beta bias,
apparently never ported to NUTS.

**Fix**: added an additive, backward-compatible `matched_warm_start` flag to
`cstr_generative_model`/`run_mcmc_baseline`. A single-window decisive test at Sc4 moved
Corr(alpha,beta) from $+0.03$ (mismatched, the old behaviour) to $-0.75$ (matched) —
reproduced identically 4 separate times (different processes/seeds), always to three
decimal places. Also checked (Step 3) whether the matched warm-start's own converged
steady state has a built-in (alpha,beta) coupling that might explain the posterior's
ridge direction: real coupling exists (the integral-controller state $I_{ss}$ has large
sensitivity to both parameters) but its local iso-contour slope does NOT align with the
posterior's own dominant-eigenvector slope — inconclusive for that specific hypothesis,
though it confirms the coupling is real. Per the original conditional design, no
alternative warm-start protocol was proposed (Steps 1-3 showed a timing mismatch, not
that the matched protocol itself injects anything spurious).

**Confirmed at full scale, not just the single-window spot check**: re-ran Theory C's
entire 8-scenario, 20-windows/scenario sweep under `matched_warm_start=True`. This
required building a parallel-execution pipeline robust to a NUTS-specific
discovery along the way: per-window cost under the fix has a wide, heavy-tailed
distribution (confirmed via direct `JAX_LOG_COMPILES=1` and `progress_bar=True`
instrumentation to NOT be a JIT cold-start or CPU-oversubscription issue — a real,
data-dependent NUTS tree-depth effect, since every leapfrog step now re-solves the
nested matched-warm-start ODE) — handled with a per-window kill-timeout (20 min,
`multiprocessing.Process.terminate()`, not `Pool.apply_async(...).get(timeout=...)`,
which was tried first and rejected because it cannot actually free a stuck worker's
slot). The sweep itself needed to be run via a standalone script
(`scripts/recompute_theoryC_section5.py`), not by re-executing the notebook through
`nbclient`, after 3 separate multi-hour/multi-ten-minute nbclient stalls made that
impractical; it also needed a resumable relaunch (`--resume-from`/`--priority`/
`--deadline-hours` flags) when the original run's 3h hard deadline fired just as Sc4
was starting — a live deadline extension on the already-running process was not
possible (plain in-memory comparison, no IPC hook), so the run was allowed to exit
naturally (which wrote a real, reusable partial-results pickle) and relaunched
prioritizing Sc4 per the coordinator's explicit instruction. (One bug caught and fixed
in the resume logic itself before it caused silent data loss: the original run's
hard-deadline exit writes placeholder "never actually dispatched" entries for every
task it didn't get to, and the first resume attempt was wrongly treating those as
genuine attempts, which would have skipped Sc5-7 entirely — fixed by only treating
`"ok"`/confirmed-`"timeout"` as genuinely attempted.)

**Final, real result** (`results/20_final_sbi_vs_nuts_comparison.csv`): NUTS's pooled
posterior now AGREES IN SIGN with SBI's matched-protocol posterior at every closed-loop
scenario that produced usable data, with substantial magnitude agreement, all R-hat
$\le$1.003:

| scenario | SBI corr | NUTS corr (pooled) | windows converged |
|---|---|---|---|
| Sc1 (healthy) | $-0.650$ | $-0.423$ | 12/20 |
| Sc2 (fouling) | $-0.856$ | $-0.488$ | 8/20 |
| Sc3 (decay) | $-0.638$ | $-0.569$ | 14/20 |
| **Sc4 (combined)** | $-0.893$ | $-0.786$ | **13/20** |

**This REVERSES the Theory C verdict documented in the entry immediately below** (which
was built on the mismatched protocol) and resolves the Theory C vs. Theory E tension
this notebook itself raised earlier the same day: the full-covariance local-FIM
diagnostic (Theory E) and the now-correctly-specified NUTS ground truth (Theory C) both
point the same way, toward the correlation being a real feature of the posterior for
Sc1-4, strengthening the evidentiary position beyond nb19's original SBC-robustness
argument alone.

**Explicitly NOT a clean sweep — three genuine open items, stated plainly, not rounded
away:**
1. **Sc7 shows a SIGN DISAGREEMENT**: SBI $+0.632$ vs. NUTS pooled $-0.666$ (11/20
   windows, clean R-hat 1.001-1.002 — a real, well-converged disagreement, not a
   convergence artifact). Unexplained.
2. **Sc0/Sc6 (open-loop): 0/20 windows converged, in BOTH the original and resumed
   runs.** Root cause identified: `cstr_generative_model` has no open-loop variant, so
   NUTS is fitting a structurally wrong (always-closed-loop) model to open-loop data —
   a genuine, separate model mismatch, distinct from the warm-start timing issue this
   entry fixes. No NUTS ground truth exists for these two scenarios anywhere in this
   notebook.
3. **Sc5 (severe actuator saturation): 0/20 windows converged**, despite being
   closed-loop and otherwise structurally like Sc1-4/Sc7. A THIRD, distinct, UNRESOLVED
   mechanism — plausibly actuator-saturation-induced likelihood irregularity (consistent
   with this project's own existing caution in `main.tex` about interpreting local
   sensitivity "cautiously" near saturation) breaking NUTS's gradient-based sampling
   specifically. Flagged as a concrete follow-up, not resolved here.

Three of eight scenarios have no NUTS ground truth at all; a fourth disagrees in sign.
The SBI training-budget non-convergence flagged in the project-wide-risk entry further
below is completely untouched by any of today's work and remains a separate,
unresolved precondition issue.

Stage 3's Theory C verdict, Theory D verdict (now CONFIRMED and fully resolved, reversed
from its earlier same-day "partially supported, old protocol looks closer to truth"
reading), and final synthesis were all rewritten in the executed notebook
(`notebooks/system_I/20_alpha_beta_correlation_mechanism_theories.ipynb`) and synced
into `scripts/build_nb_system_I_20.py`'s `CELLS` list, per this project's established
convention for patching text directly rather than re-running an unreliable rebuild.
`notebooks/system_I/TODO.md` updated (prepended). `main.tex`/the SI/
`article_outline_CChE.md` still NOT touched — this is a human promotion decision,
unchanged by any of today's work.

## 2026-11-07 update (notebook 20 COMPLETE, all 6 theories + Stage 3 verdicts written — nb19's headline (alpha,beta) correlation finding is now REVERSED, not confirmed)

Built and executed `notebooks/system_I/20_alpha_beta_correlation_mechanism_theories.ipynb`
(`scripts/build_nb_system_I_20.py`), testing 6 theories for WHY nb19's
Corr(alpha,beta)@Sc4 $\approx -0.48$ finding exists. This was a long, multi-stage run
(spanning roughly 2026-11-05 evening through 2026-11-07) that hit and fixed two real
bugs along the way (see below) before Stage 3 could be written honestly. Full numbers
in `notebooks/system_I/TODO.md`'s newest status entry; summary here.

**Stage 0 (training-budget convergence)**: `n=10,000` (nb19's basis) was NOT converged
-- the 10k->20k step moved the Sc4 correlation by 0.12 (above a 0.05 threshold);
escalating to `n=40,000` per this project's own "one escalation level, flag don't
silently stop" rule made it WORSE, not better (20k->40k moved by 0.17) -- the estimate
is actively drifting with training budget, not merely slow to settle. `PRODUCTION_BUDGET`
ended at 40,000 as "the best available, NOT confirmed converged" choice, stated as such.

**Stage 1 (per-scenario survey)**: a real, non-monotonic, partially sign-flipping
pattern across Sc0-7 -- see TODO.md for the full table. Required building a NEW
open-loop SNPE training path (none existed in this codebase) to get Sc0/Sc6 coverage.

**Theory C (SNPE vs. NUTS) surfaced TWO separate real bugs, both found and fixed
live during this session, not assumed away:**
1. A `multiprocessing` "spawn"-vs-interactive-closure pickling crash (fixed by moving
   the worker to a real module, `scripts/nb20_nuts_worker.py`).
2. **`cstr_generative_model`'s NUTS likelihood defaulted to a DIAGONAL Gaussian over the
   29-D summaries** -- the SAME independence assumption Theory E found wrong for the
   local-FIM diagnostic. This produced an apparent "posterior multimodality" at the
   healthy point (R-hat 3.4-11.1, chains settling into different (alpha,beta) basins).
   Directly re-running the identical window with the FULL empirical covariance as the
   likelihood (same 120-replicate methodology as Theory E) instead collapsed this to
   R-hat<=1.002 across 3 seeds -- **the multimodality was a likelihood-misspecification
   artifact, not real.** Fixed at the source: `cstr_generative_model`/`run_mcmc_baseline`
   (`src/cstr_sbi/inference.py`) both gained an optional `cov_obs` full-covariance
   parameter (additive, backward-compatible), computed once per scenario and used
   throughout Theory C from then on. Also added full R-hat/ESS/divergence capture
   (`numpyro.diagnostics.summary`, `mcmc.get_extra_fields()`) and convergence-gated
   pooling (only R-hat<1.01 windows go into the "real" pooled NUTS posterior) --
   `_mcmc` was previously discarded immediately, meaning NO convergence checking
   existed at all before this fix.

**With both fixes in place, Theory C's real result is the headline reversal**: on
every CLOSED-LOOP scenario (Sc1,2,3,4,5,7), NUTS converges cleanly (R-hat<=1.004, 90-100%
of windows) and finds SMALL Corr(alpha,beta) (-0.06 to +0.16) -- while SBI's
matched-protocol posterior shows LARGE correlation (-0.89 to +0.85) at the SAME points,
including a sign flip at Sc4 (SBI -0.89, NUTS +0.08). This directly contradicts nb19's
SBC-robustness-based reading that the correlation is "probably real." Sc0/Sc6
(open-loop) remain unresolved -- NUTS still fails to converge there (R-hat in the
hundreds-to-thousands) even after the covariance fix, a SEPARATE problem, explicitly
flagged as needing its own follow-up, not resolved here.

**Also found two more things worth carrying forward**: (1) Theory D (warm-start
protocol) shows the OLD, previously-"fixed" protocol's near-zero correlations sit much
closer to NUTS's ground truth than the current matched-protocol's large ones -- hinting
the matched-warm-start protocol itself may be the source of the spurious correlation,
not a fix for one. (2) Theory F (CNN raw-trajectory) does not shrink the correlation
toward the (now-suspected-small) truth -- it flips sign at 5/6 scenarios -- arguing the
issue is upstream of representation choice, most likely the SNPE training protocol.

**Stage 3's own final synthesis states an explicit, unresolved tension** between Theory
C (small correlation in the real posterior) and Theory E (full-covariance local-FIM
recovers most of SBI's large correlation) -- three candidate reconciliations are listed,
none confirmed by the evidence gathered. **Net: still OPEN, arguably LESS settled than
before this notebook existed.** Nothing from nb19 or nb20 about this correlation should
be promoted toward the manuscript. `main.tex`/SI/`article_outline_CChE.md` NOT touched.

**Project-wide risk flagged (not acted on)**: if `n=10,000` wasn't converged for this
question, other System I production results built on similar budgets (Table VII, EKF/UKF
baselines, 30-day tracking, SBC claims) may also not be at a converged budget -- flagged
in `notebooks/system_I/TODO.md` as a future dedicated-audit item, not started.

## 2026-11-05 update (Section 10: investigating WHY seed4/cached fail alpha-SBC — inconclusive, two different failure signatures, not one shared mechanism) — notebook 19, new Section 10 (10a-10d)

Extended `scripts/build_nb_19.py` in place again (same notebook,
`notebooks/system_I/19_alpha_beta_banana_and_ekf_asymmetry.ipynb`, new Section 10), to run
the explicit next step the previous round (10-seed extension entry below) left open:
investigate *why* seed4 and the single cached posterior (`sbi_posterior_matched_protocol_n10000.pkl`)
fail alpha-SBC at extreme significance ($p\sim10^{-5}$) while 8 of the other 9 posteriors in
the 11-posterior SBC batch pass cleanly or land ambiguous. Three diagnostics were run,
reusing the already-trained/cached posteriors and the already-generated SBC held-out set
(`results/19_sbc_heldout_n300.npz`, $N=300$) — no retraining.

**10a. Rank-histogram inspection** (keeping the raw per-draw SBC ranks this time, which
Section 6b's `evaluate_sbc_nb19` helper discarded after computing the KS p-value/C2ST
summary; `results/19_sbc_rank_diagnostics.csv`, `figures/19_sbc_rank_histograms.png`).
Checked seed4 and the cached posterior (both FAIL alpha) against seed0, seed1, seed6 (all
PASS). **Finding: the two failing posteriors do not share one clean failure-mode shape.**
For a calibrated posterior with `L_RANK=500`, `mean_rank` should be $\approx$250 and the
fraction of ranks in the outer 20% (`frac_edge_20pct`) / central 20% (`frac_mid_20pct`)
should both be $\approx$0.20. seed4's alpha ranks: `mean_rank=208.9` (left-shifted $\approx$16%),
`frac_edge_20pct=0.16`, `frac_mid_20pct=0.21` — edge/middle fractions close to the uniform
expectation, so this is a **shifted-mean signature, not U-shaped overdispersion or
peaked/inverse-U underdispersion**. The cached posterior's alpha ranks: `mean_rank=216.9`
(also left-shifted, same direction as seed4) but `frac_edge_20pct=0.33` (well above 0.20)
and `frac_mid_20pct=0.17` (below 0.20) — a **genuine overdispersion signature layered on
top of the same directional mean shift**. The 3 passing posteriors have `mean_rank` in
[236.6, 259.6] and both fraction columns within $\approx$0.01–0.04 of 0.20 on both alpha
and beta. Beta stays well-calibrated-looking in all 5 posteriors checked (`mean_rank`
243–260), consistent with Section 6b/8.

**10b. Prior-edge saturation check** (`results/19_sbc_edge_saturation.csv`, `figures/
19_sbc_edge_saturation.png`): split the $N=300$ held-out draws into near-prior-edge
($n=64$, true alpha within 20% of the half-range of `ALPHA_LOW=0.4`/`ALPHA_HIGH=1.2`) vs.
away-from-edge ($n=236$) subsets, ran a separate KS test on each, for seed4 and cached.
**Finding: actively CONTRADICTED for seed4, only weakly/partially consistent for cached —
not a shared explanation.** For seed4, the near-edge subset passes far more comfortably
(`KS p=0.098`) than the away-from-edge subset (`KS p=1.999e-6`, the same extreme level as
the full-sample failure) — seed4's miscalibration is concentrated **away from** the prior
boundary, the *opposite* of what edge saturation predicts. For the cached posterior, both
subsets already fail (near-edge `p=2.27e-4`; away-from-edge `p=4.10e-3`), with near-edge
failing somewhat more severely — a difference of degree between two already-failing
subsets, not "near-edge fails while away-from-edge is calibrated."

**10c. Training-level correlate check**: only hyperparameters, `n_simulations` and
`wall_time_s` are persisted in each cached posterior's `.pkl` (`train_sbi_posterior`'s
`metadata` dict, `src/cstr_sbi/inference.py`) — epochs-trained and best-validation-loss are
printed to stdout during training but never saved, so this check is limited to wall time
and could not go further without a fresh, instrumented retrain (explicitly out of scope
for this check). seed4's wall time (322.6s) does **not** stand out — seed3 (319.9s), seed6
(330.2s) and seed7 (310.3s) all took as long or longer and all pass SBC cleanly. Reported
as "checked, found nothing, limited by what's logged," not a clean negative finding.

**Verdict (Section 10d, written directly from the above, no placeholder text left in the
notebook): genuinely unresolved at the mechanism level, and importantly NOT one shared
cause.** seed4's failure looks like a systematic mean-shift in the rank statistic with no
strong dispersion distortion, concentrated away from the prior boundary (directly
inconsistent with edge saturation). The cached posterior's failure looks like a mix of the
same directional mean shift plus a real overdispersion signature, with only a weak
within-both-already-failing-subsets edge correlate. Neither failure is explained by the
one training-level factor available (wall time), though that check cannot rule out an
epochs/validation-loss-level cause this project's artefacts do not record. **No single
mechanism among prior-edge saturation / overdispersion / underdispersion / shared
systematic bias / a training-level factor cleanly and fully explains BOTH outright
failures together** — a mean-bias-style account partially fits seed4, a mix of
overdispersion+mean-bias partially fits cached, and edge saturation/training-time are not
supported as general explanations. Stated plainly per this project's own discipline
against overclaiming resolved mechanisms that later need walking back: **two specific
posteriors out of 11 trained under an identical recipe fail alpha-SBC at extreme
significance, for related-but-not-identical rank-distribution signatures, and the root
cause is not yet pinned down.**

Artefacts added: `results/19_sbc_rank_diagnostics.csv`, `results/19_sbc_edge_saturation.csv`,
`figures/19_sbc_rank_histograms.png`, `figures/19_sbc_edge_saturation.png`. Notebook's new
Section 10 (10a-10d) added after the existing Section 9; the Section 10d verdict cell in
the executed `.ipynb` contains the final numbers above (not placeholder text — patched
directly after the full notebook rebuild finished, rather than re-running the ~minutes-to-
tens-of-minutes-long full rebuild a second time just to update one markdown cell's text;
`scripts/build_nb_19.py`'s own Section 10d source was updated identically, so a future
rebuild from the script reproduces the same verdict text). `notebooks/system_I/TODO.md`'s
status note updated (prepended, prior content kept). `main.tex`/the SI/
`article_outline_CChE.md` not touched, per this project's promotion discipline — if
anything, this result argues for continued caution, not promotion: the SBC-failure
mechanism investigated here remains open, not resolved.

## 2026-11-05 update (10-seed extension of the multi-seed sweep + new 2-D posterior figure) — notebook 19, Sections 3/6b/7/8/9

Extended `scripts/build_nb_19.py` in place again (same notebook,
`notebooks/system_I/19_alpha_beta_banana_and_ekf_asymmetry.ipynb`), two parts.

**Part 1 — 5 more seeds (seed5-seed9), same recipe/budget as seed0-4 (n=10,000
sims/seed, matched-protocol warm start).** Per-seed training time for the original 5
seeds (read from cached `.pkl` metadata before this run) was 218-323s/seed, confirming
the ~220-320s/seed prior estimate; the 5 new seeds took a comparable amount of time
in-process. Newly cached: `results/19_multiseed_posterior_seed{5,6,7,8,9}.pkl`.

**Note on a methodological wrinkle found and fixed during this extension**: the
`posterior.sample()` calls used to compute Corr(alpha,beta) from a trained posterior were
not seeded, so re-running the exact same cell against the exact same 10 trained
posteriors gave noticeably different per-seed correlation values from one notebook
execution to the next (observed drift of several tenths in individual seeds, though the
qualitative picture -- all negative, all moderate-to-large -- never changed). This was
fixed by adding `torch.manual_seed()` per-seed immediately before each `.sample()` call
in Section 3's two correlation cells, so the numbers below are now reproducible on
re-execution. The single cached posterior's own correlation query (a separate, older
code path, Section 2) was *not* similarly fixed and remains Monte-Carlo-noisy between
runs -- flagged in the notebook, not resolved, since it's out of this task's scope.

**Full 10-seed Corr(alpha,beta) at Sc4** (`results/19_multiseed_corr.npz`, now
reproducible): mean=-0.482, std=0.134, range=[-0.762,-0.269] (per-seed: seed0=-0.762,
seed1=-0.554, seed2=-0.498, seed3=-0.549, seed4=-0.269, seed5=-0.370, seed6=-0.394,
seed7=-0.520, seed8=-0.345, seed9=-0.556). Compare the original 5-seed sample:
mean=-0.472, std=0.102, range=[-0.645,-0.383]. **The mean is essentially unchanged
(-0.472 -> -0.482, a difference of 0.01)**; seed0 -- the same seed that had the largest
magnitude in the original 5-seed sample -- remains the single largest-magnitude seed at
-0.762, and seed4 remains the smallest at -0.269 (also the one posterior that fails
alpha-SBC, see below). Every one of the 10 seeds remains negative and none is near zero.
The grid cross-check (`results/19_crosscheck_singleposterior_vs_multiseed.csv`) tells the
same story at every other operating point: 10-seed means stay within ~0.15 of the 5-seed
means at every point, no sign flips, no collapse toward zero.

**SBC on all 10 new seeds too, giving an 11-posterior total** (10 seeds + the 1
pre-existing cached posterior, same N_SBC=300 held-out set reused from the SBC-validation
entry below, `results/19_sbc_fullbudget_posteriors.csv`):

| posterior | KS p (alpha) | KS p (beta) | verdict |
|---|---|---|---|
| seed0 | 0.243 | 0.572 | PASS |
| seed1 | 0.783 | 0.818 | PASS |
| seed2 | 0.211 | 0.320 | PASS |
| seed3 | 0.444 | 0.059 | PASS |
| seed4 | 9.7e-6 | 0.306 | **FAIL (alpha)** |
| seed5 | 0.101 | 0.011 | **AMBIGUOUS (beta)** |
| seed6 | 0.411 | 0.156 | PASS |
| seed7 | 0.201 | 0.591 | PASS |
| seed8 | 0.071 | 0.921 | PASS |
| seed9 | 0.031 | 0.649 | **AMBIGUOUS (alpha)** |
| cached (`sbi_posterior_matched_protocol_n10000.pkl`) | 6.8e-6 | 0.727 | **FAIL (alpha)** |

**Breakdown: 7/11 PASS cleanly, 2/11 FAIL (alpha only, both unchanged from the 5-seed
pass: seed4 and the cached posterior), 2/11 newly-observed AMBIGUOUS (seed5 on beta,
seed9 on alpha, both just inside the 0.01-0.05 ambiguous band). This SBC table is
deterministic (reused cached held-out set + `run_sbc`/`check_sbc`, no sampling-seed issue
here) and reproduced identically across all re-executions of this notebook during this
extension.**

**Does the 5-seed sample's "1-in-3 alpha-SBC-failure rate" hold up, strengthen, or weaken
at n=10 seeds? It weakens, moderately, stated explicitly as asked.** Outright alpha-FAIL
rate drops from 2/6 (33%) to 2/11 (18%) — no NEW outright failures appeared among the 5
new seeds; they landed either cleanly (seed6,7,8) or ambiguous (seed5, seed9), never
outright failing. The two original failures (seed4, the cached posterior) are unchanged
— both still fail alpha-SBC at the same extreme significance ($p\sim10^{-5}$) as before,
not a borderline shift. Combining fail+ambiguous on either parameter gives 4/11 (36%),
higher than the original 2/6 (33%) if ambiguous cases are counted as a concern, so the
honest statement is: **outright failure got rarer in relative terms, but genuinely
borderline cases appeared that weren't visible in the smaller sample — call this "mostly
calibrated, same core 2-posterior problem, slightly murkier edges," not a clean
improvement or a clean deterioration.**

**Magnitude-vs-SBC-failure relationship, re-checked on the reproducible 10-seed numbers:
genuinely mixed, no clean story in either direction.** Ranked smallest to largest
|Corr(alpha,beta)|: seed4 (-0.269, FAIL-alpha), seed8 (-0.345), seed5 (-0.370,
AMBIGUOUS-beta), seed6 (-0.394), seed2 (-0.498), seed7 (-0.520), seed3 (-0.549), seed1
(-0.554), seed9 (-0.556, AMBIGUOUS-alpha), seed0 (-0.762, PASS cleanly). The outright
FAIL (seed4) and one AMBIGUOUS case (seed5) are among the smallest-magnitude
correlations, and the single largest (seed0) passes cleanly — both points against "bigger
correlation, worse calibration." But the other AMBIGUOUS case (seed9) has the
*second-largest* magnitude in the batch — against "smaller correlation, worse
calibration" too. **Net reading: correlation magnitude and SBC outcome look like two
largely independent properties of a trained posterior under this recipe, which argues
against a single systematic-bias mechanism driving both** (such a mechanism should leave
a consistent trace in this ranking, not a scattered one), but this is weaker, more honest
evidence than the cleaner "opposite-direction" pattern an earlier, not-yet-reproducible
draft of this analysis (based on an unseeded sampling run) appeared to show — corrected
here before being written down as a conclusion.

**Net verdict: the "real posterior-geometry feature, not a systematic-bias artefact"
reading from the 5-seed/6-posterior pass is unchanged in direction at 10 seeds/11
posteriors, and the specific "1-in-3 outright failure" worry is measured as somewhat
rarer (2/11 vs. 2/6), but this is still not a clean, unqualified validation** — seed4 and
the single cached posterior continue to fail alpha-SBC at extreme significance,
unexplained, and 2 new ambiguous cases widen (not narrow) the set of posteriors that
would need their own SBC check before being trusted. **main.tex/the SI still not
touched** — same blockers as the SBC-validation entry below, now with more precise (not
resolved) numbers.

**Part 2 — new Section 9: 2-D (alpha,beta) posterior scatter figure.** Added a direct
visual complement to the scalar Corr(alpha,beta) numbers: a 2-panel scatter plot
(`figures/19_posterior_2d_seed0_sc4.png`) of seed0's posterior samples at the healthy
point (1.00,1.00) and at Sc4 (0.85,0.85), using this project's existing 2-D
alpha/beta-posterior scatter convention from `notebooks/system_I/04_sbi_training.ipynb`
(raw scatter, true value marked with dashed crosshairs, axes clipped to prior support).
seed0 was used per the task's original specification (SBC-clean, originally
largest-magnitude in the 5-seed sample) -- and, per the corrected/reproducible 10-seed
numbers above, seed0 remains the single largest-magnitude seed at 10 seeds too (-0.762),
so no change of pick was needed. This section's own (separately-sampled) query of
seed0's posterior reads -0.224 at healthy and -0.439 at Sc4 -- a smaller magnitude than
Section 3's -0.762 for the same trained posterior, since it's a different
`posterior.sample()` draw (3000 vs. 5000 samples); this is flagged explicitly in the
notebook as a reminder that any single quoted Corr(alpha,beta) number here should be
read as "real, moderate-to-large, negative," not to three-decimal precision. The figure
shows a compact, near-isotropic cloud at the healthy point and a visibly elongated
diagonal ridge at Sc4 for the same posterior — the correlation is not just a scalar
statistic but a real, visible banana-shaped feature of the trained posterior's geometry
at the fault point, clearly distinct in shape from the healthy-point posterior.

Artefacts added/updated: `results/19_multiseed_posterior_seed{5,6,7,8,9}.pkl` (new
cache), `results/19_multiseed_corr.npz` (overwritten in place, now 10-seed and
reproducible), `results/19_crosscheck_singleposterior_vs_multiseed.csv` (overwritten in
place, now 10-seed), `results/19_sbc_fullbudget_posteriors.csv` (overwritten in place,
now 11 rows), `figures/19_posterior_2d_seed0_sc4.png` (new figure). Notebook's Sections
3/6b/7/8 text updated with 10-seed/11-posterior numbers (prior 5-seed/6-posterior numbers
kept as explicit historical comparison, not deleted); new Section 9 added.
`notebooks/system_I/TODO.md`'s status note updated (prepended, prior content kept).
`main.tex`/the SI/`article_outline_CChE.md` not touched, per this project's promotion
discipline.


## 2026-11-05 update (SBC validation of the full-budget run's open item) — notebook 19, Section 6b: the -0.47 Corr(alpha,beta) is mostly, but not cleanly, SBC-validated

Closes the explicit open item left by the full-budget corroboration run described in the
entry directly below: are the 6 full-budget (n=10,000/seed) posteriors behind the
seed-robust Corr(alpha,beta)=-0.47 finding well-calibrated, or is the correlation a
systematic-miscalibration artefact that happens to reproduce across seeds? Extended
`scripts/build_nb_19.py` in place (same notebook,
`notebooks/system_I/19_alpha_beta_banana_and_ekf_asymmetry.ipynb`, new Section 6b) to run
simulation-based calibration (SBC, KS test on rank statistics, N_SBC=300 held-out
matched-protocol draws, `sbi.diagnostics.run_sbc`/`check_sbc` — this project's standard
System-I SBC recipe from `notebooks/system_I/18_ablation_study.ipynb`) on all 6 posteriors:
the 5 independently-trained full-budget seeds from Section 3 (reused via a new on-disk
cache, `results/19_multiseed_posterior_seed{0..4}.pkl` — not retrained) plus the single
cached posterior (`sbi_posterior_matched_protocol_n10000.pkl`). Runtime was a few minutes
(SBC sampling from already-trained amortised posteriors is cheap; only the 300-draw
held-out simulation set, cached to `results/19_sbc_heldout_n300.npz`, cost real time).

**Result (`results/19_sbc_fullbudget_posteriors.csv`, threshold p>=0.05 pass / p<0.01
fail, this project's standard convention):**

| posterior | KS p (alpha) | KS p (beta) | verdict |
|---|---|---|---|
| seed0 | 0.243 | 0.572 | PASS |
| seed1 | 0.783 | 0.818 | PASS |
| seed2 | 0.211 | 0.320 | PASS |
| seed3 | 0.444 | 0.059 | PASS |
| seed4 | 9.7e-6 | 0.306 | **FAIL (alpha)** |
| cached (`sbi_posterior_matched_protocol_n10000.pkl`) | 6.8e-6 | 0.708 | **FAIL (alpha)** |

**4 of 6 pass SBC cleanly on both parameters; 2 of 6 (seed4 and the original cached
posterior) fail on alpha specifically, at extreme significance, while beta passes in all 6
posteriors without exception.** This is a genuinely mixed result, reported as such.

**Verdict, stated carefully, not rounded to the convenient answer either way.** The
SBC-failing posteriors are NOT distinguished from the passing ones by Corr(alpha,beta)
magnitude: seed0 has the largest-magnitude correlation of the five (-0.645) and passes SBC
cleanly; seed4 has one of the smaller-magnitude correlations (-0.401) and fails. If a
systematic training-recipe bias were manufacturing the correlation, failure and magnitude
should track together; they do not. Combined with Section 3's seed-robustness (rules out a
single-run fluke) and beta's universal pass, this is reasonable evidence that **the
Corr(alpha,beta)=-0.47-ish finding at Sc4 is a real feature of System I's trained posterior
geometry, not a systematic-miscalibration artefact that coincidentally reproduces across
seeds.** However, **this is explicitly not a clean, unqualified validation**: a 1-in-3
alpha-SBC-failure rate across the 6 posteriors tested — including the very posterior this
whole investigation started from — is a real, unexplained problem, not a rounding error.
**Do not promote the (alpha,beta) correlation finding into `main.tex`/the SI on the
strength of this result alone.** Before any manuscript claim: (a) investigate why
seed4/the cached posterior fail alpha-SBC specifically (rank-histogram inspection,
possible prior-edge saturation — not done here), and (b) treat any future System-I SNPE
retrain under this same recipe as needing its own SBC check, since a 1-in-3 failure rate in
this sample is too high to assume calibration by default.

Artefacts added (new files, none of Section 3/7's existing outputs were touched):
`results/19_sbc_fullbudget_posteriors.csv`, `results/19_sbc_heldout_n300.npz`,
`results/19_multiseed_posterior_seed{0,1,2,3,4}.pkl` (posterior cache, reusable by future
notebook reruns to avoid retraining), `figures/19_sbc_fullbudget_posteriors.png`. Notebook's
new Section 6b and Section 8 verdict addendum. `notebooks/system_I/TODO.md`'s status note
updated (prepended, prior content kept). `main.tex`/the SI/`article_outline_CChE.md` not
touched, per this project's promotion discipline.

## 2026-11-05 update (full-budget corroboration, same day) — notebook 19 re-run at full budget: the (alpha,beta) "no banana" conclusion REVERSES; EKF-asymmetry findings hold

Re-executed `notebooks/system_I/19_alpha_beta_banana_and_ekf_asymmetry.ipynb` in place
(same `scripts/build_nb_19.py`, same notebook path — not a new notebook) at **full
production budget**: n=10,000 simulations/seed for every SNPE retrain (previously n=800)
and all 720 windows for the 30-day tracking comparison (previously every 10th window, 72
of 720). Both scope reductions explicitly flagged in the reduced-budget pass described in
the entry directly below are now lifted. Runtime was far shorter than the conservative
~6.5h cold-process-rate estimate used to plan the launch: the 5-seed retrain + no-proxy
ablation retrain + full 720-window tracking loop together took **~20-35 minutes** total
(simulator already JIT-warm within the process), not hours — useful calibration for
future full-budget System-I retrains in this notebook family.

**Headline result: going to full budget REVERSES the Item 1+3/Item 2 conclusion from the
reduced-budget pass.** At n=800, all 5 retrained seeds gave Corr(alpha,beta) at Sc4 within
about ±0.1 of zero (mean ≈-0.05) and the single cached production posterior's large
correlation (|Corr| up to ~0.8 across the grid) was concluded to be a single-training-run
artefact. **At the full n=10,000 budget, the opposite holds**: all 5 independently-trained
seeds give Corr(alpha,beta) at Sc4 of mean=-0.472 (std=0.102, range=[-0.645,-0.383]) —
consistently large, consistently negative, matching the single cached posterior's sign and
order of magnitude. The same pattern holds at every other cross-checked grid point
(healthy, Sc4, moderate, severe, severe-asym): all 5 full-budget seeds agree in sign with
the single cached posterior, landing in the same moderate-to-large |Corr| regime (roughly
0.2-0.8), while the local-FIM/Laplace diagnostic stays small everywhere
([-0.052,+0.209]) and is the one diagnostic that fails to predict this. **Conclusion: the
large (alpha,beta) correlation is a real, seed-robust feature of System I's trained
posterior geometry, not a training artefact** — the n=800 "no correlation" result was
itself a false negative from undertraining the 2-D NSF posterior, the SBI-training
analogue of this project's own `finding_sbc_power_eta_col.md` lesson (N=200 SBC gives
false passes; need N≥400) now showing up for posterior correlation structure under
insufficient training-simulation budget rather than insufficient SBC-rank-statistic count.
This reopens `notebooks/34_alpha_beta_r_banana_cross_system.ipynb`'s "System I has no
(alpha,beta) banana" conclusion — not to a System-II-style near-unity banana (nothing here
approaches +0.998), but to a real, moderate, seed-robust correlation the earlier "no
banana" framing understated.

**Important caveat, explicitly not resolved**: none of the 6 posteriors involved (5 new
full-budget retrains + the pre-existing cached one) have been SBC-checked in this
notebook. Seed-robustness at full budget rules out "single-training-run artefact," but not
"systematically miscalibrated under this training recipe" — per this project's own
SBC-best-seed != classification-best-seed lesson for System II, calibration is not implied
by cross-seed agreement alone. **This is the explicit next step before this finding goes
anywhere near `main.tex`/the SI.**

**Items 4/5 (EKF-asymmetry hypotheses, noise robustness): conclusions UNCHANGED at full
budget/resolution**, tightened in places:
- (a) physics-informed proxies: still NOT the driver — SBI's beta MAE with vs. without the
  proxies is now even closer (0.0205 vs. 0.0204, full 720-window resolution) than the
  reduced-budget pass's 0.0226 vs. 0.0264, while EKF/UKF stay at 0.068-0.090 throughout
  (alpha MAE also sharpened: SBI ≈0.003 vs. EKF/UKF's 0.024-0.025 at full budget).
- (b) EKF/UKF tuning quality: still NOT the driver (this check doesn't depend on SBI
  training budget) — retuning to the honest protocol still makes beta MAE slightly worse
  (0.097→0.106 at Sc2, 0.065→0.071 at Sc4), identical to the reduced-budget numbers.
- (c) linearisation quality: still the best-supported explanation (also budget-independent)
  — System I stays at machine precision (4.7e-12 to 3.5e-11) across its whole range vs.
  System II's growth to 4.3e-4 near-snowball, a ratio of ~1.23e7.
- Noise-robustness (Item 5): FIM-based conclusion unchanged (small across 0.5x/1x/2x noise
  at 3 points), but now flagged in the notebook as resting on the same local-FIM diagnostic
  just shown to be unreliable for the correlation question above — re-running against
  trained posteriors is a flagged follow-up, not done here.
- The residual EKF-vs-SBI gap (EKF linearises almost perfectly yet still loses to SBI by
  3-4x on beta MAE) remains open, as in the reduced-budget pass.

Artefacts overwritten in place (same filenames as the reduced-budget pass, now with
full-budget numbers): `results/19_grid_corr_scan.csv`,
`results/19_crosscheck_singleposterior_vs_multiseed.csv`, `results/19_multiseed_corr.npz`,
`results/19_noise_robustness.csv`, `results/19_proxy_ablation_tracking.csv`,
`results/19_ekf_retune_test.csv`, `results/19_linearisation_error_sys{I,II}.csv`,
`figures/19_grid_corr_scan.png`, `figures/19_linearisation_error_comparison.png`. Status
note at the top of `notebooks/system_I/TODO.md` updated to reflect the full-budget
completion and the reversal. `main.tex`/the SI/`article_outline_CChE.md` untouched, per
this project's promotion discipline — the SBC-validation caveat above is the explicit
blocker.

## 2026-11-05 update (reduced-budget pass, same day, superseded by the full-budget update above for Items 1+3/2) — System I notebook 19 built: (alpha,beta) banana generalised; EKF asymmetry mostly explained

Built and executed `notebooks/system_I/19_alpha_beta_banana_and_ekf_asymmetry.ipynb`
(`scripts/build_nb_19.py`), per `notebooks/system_I/TODO.md`'s 6-item spec (now marked
built at the top of that file). This extends `notebooks/34`'s single-spot-check (System I
has no `(alpha,beta)` banana, unlike System II's `+0.998` at W11) to a grid/multi-seed/
noise-robust check, and directly tests the three candidate explanations for the open
EKF/UKF-vs-SBI asymmetry documented in `docs/Manuscript/system_ii_sb_identifiability_story.md`
S5.3 (System II's EKF beats SBI near nonlinear regimes; System I's EKF/UKF lose to SBI
everywhere, unexplained until now).

**Item 1+3 (grid, local-FIM + trained posterior) — an important sub-finding on its own.**
The local-FIM Corr(alpha,beta) stays small across a 10-point grid spanning the whole prior
([-0.052, +0.209]). The **single cached production posterior**
(`sbi_posterior_matched_protocol_n10000.pkl`), queried at the SAME grid points, showed
alarmingly large correlations (up to -0.87, including -0.59 to -0.87 AT Sc4, the exact
point nb34 called "modest" using a different cached artefact) — which would have reopened
the "no banana" conclusion if taken at face value. **Resolved by Item 2's cross-check**:
5 independently retrained seeds at the same grid points give means of -0.05 to +0.04,
consistent with the local-FIM, with the single cached posterior sitting far outside every
5-seed range. **Verdict: the single posterior's large correlations are a training
artefact** (amortised-posterior unreliability away from its best-validated scenarios), not
a real feature of System I's geometry — a reusable lesson: never trust one cached
posterior's correlation structure off of its originally-validated scenario set, exactly
the single-seed trap this project has been burned by twice before for System II.

**Item 4 — EKF-asymmetry hypotheses, tested directly:**
- **(a) physics-informed proxies: NOT the driver.** Ablating `UA_eff_proxy`/`k0_eff_proxy`
  barely changes SBI's beta MAE (0.0226 -> 0.0264) while EKF/UKF stay at 0.07-0.10
  throughout (reduced-resolution 30-day tracking, 72 subsampled windows).
- **(b) EKF/UKF tuning quality: NOT the driver.** System I's as-published
  `Q_param/P0_param` ratio is ~2 orders of magnitude tighter/more overconfident than
  System II's honest-protocol tuning, but retuning System I's EKF to the honest protocol
  made beta MAE slightly *worse* (0.097->0.106 at Sc2, 0.065->0.071 at Sc4), not better.
- **(c) linearisation quality: best-supported explanation.** System I's one-step
  linearisation error stays at machine precision (1e-11 to 1e-12) across its entire
  healthy-to-severe range; System II's grows to ~4e-4 near its near-snowball regime --
  about 7 orders of magnitude (1.2e7x) larger at the top of System II's range. System I's
  smaller, PI-dominated state space really is close to linear everywhere its EKF operates.
- **Open residual**: even though System I's EKF linearises almost perfectly, it still
  loses to SBI by 3-4x on beta MAE -- that gap is NOT explained by any of (a)/(b)/(c) and
  is left explicitly open in the notebook's verdict cell, not papered over.

**Scope reductions (disclosed, not promoted to the manuscript):** multi-seed (Item 2) and
proxy-ablation (Item 4a) retrains used n=800 simulations/seed, not the production 10,000 --
measured first (0.49s/simulation-call with matched-protocol warm start; full 10k budget
measured at 4731s, `results/sbi_training_metadata.json`) before choosing the reduced
budget, per this project's cost-discipline convention. Item 4a's 30-day tracking used every
10th window of the full 720-window stream (72 windows, same per-window cost as
`notebooks/system_I/16_...ipynb`), a window-count reduction, not a per-window resolution
cut. **Do not cite this notebook's numbers in `main.tex`/the SI** until corroborated at
full budget/resolution -- stated explicitly in the notebook's own verdict cell.

New artefacts: `results/19_grid_corr_scan.csv`, `results/19_crosscheck_singleposterior_vs_multiseed.csv`,
`results/19_multiseed_corr.npz`, `results/19_noise_robustness.csv`,
`results/19_proxy_ablation_tracking.csv`, `results/19_ekf_retune_test.csv`,
`results/19_linearisation_error_sys{I,II}.csv`, `figures/19_grid_corr_scan.png`,
`figures/19_linearisation_error_comparison.png`.

## 2026-10-14 update — S-B's wide-prior campaign finished (see block below); now redoing S-B3 honestly for a fair comparison

**Read this block first**, then the 2026-10-06 block below it (same overall thread,
this is the continuation). The S-B 5-D wide-prior campaign described below **is
complete** — see that block for the full result: `baseline` z-scoring recipe gives
macro-F1 0.345-0.449 / accuracy 0.429-0.536 on the full 14-scenario catalogue (best
S-B result this session), and the Artefact 2 (`(alpha,beta_r)` W11 "banana",
manuscript's headline +0.998 claim) **does not reproduce** under this posterior
(correlations -0.235 to +0.674 across 5 seeds). Full writeup, confusion matrix, and
the Artefact 2/3 re-audit are now in
`notebooks/36_wu2003_sb_wideprior_classification_and_artefact_reaudit.ipynb` (built
this session specifically because the first pass at these results only existed as
console output — don't let that happen again, always save results to a notebook or
CSV, not just stdout).

**User asked: S-B3 (`nb31e`, cherry-picked seed) reports macro-F1=1.000 — which
should go in the article?** Answer given: **S-B**, not that S-B3 number — it's a
single seed selected post-hoc by classification outcome despite failing SBC for 3/4
parameters, and the `(alpha,beta_r)` correlation across S-B3 seeds was already shown
(earlier this session) to be wildly seed-unstable (-0.43 to +0.33). Not a fair
comparison. **But also agreed to redo S-B3 honestly** (multi-seed, `baseline`
z-scoring per the lesson just learned from S-B, proper SBC, no seed cherry-picking)
before concluding S-B is definitively better — removing `z_A0_eff` from scope might
legitimately help the other 4 parameters train better even under an honest standard,
since it removes Artefact 3's disruptive effect from the joint posterior.

**S-B3 honest-redo campaign: LAUNCHED 2026-10-14.**
`scripts/generate_sb3_wideprior_bank.py` (`box_uniform_4d_sb3()` — full uniform
prior over `alpha, beta_r, eta_col, xi_reb`; `z_A0_eff` fixed exogenous at nominal
0.90, reusing the *same* clean matched-protocol simulation pipeline already
validated for S-B, just via `theta4_to_theta5` padding). Cost-checked before
launching (same discipline as S-B): a 40-attempt sample gave accept rate 0.50,
56s/accepted row → **~86h projected** for `n_train=5000` + SBC sets (120+400),
comparable to S-B's campaign. Checkpointed every 50 rows, logs to
`data/wu2003_sbi_train_sb3_n5000_sb3_wideprior_stage1bfix.progress.log` (tail that
to watch progress). **Not yet trained or evaluated — bank generation only, just
started.** Same resume discipline as S-B: if picking this up cold, check
`ps aux | grep generate_sb3_wideprior_bank`; if not running, just re-run the same
command, it resumes from `.partial.npz`.

**Once the bank is done**: train multi-seed with `baseline` z-scoring (mirror
`scripts/run_sb_wideprior_training.py`, swap in `cstr_sbi.recycle.sb3`'s 4-D prior/
theta handling and the `FAULT_UNITS_SB3`/`fault_unit_sb3_truth`/`list_closed_loop_sb3`
4-class/12-scenario taxonomy), run SBC with the anti-shrinkage-slope check, then
classification — build this as a proper notebook from the start this time (not a
bare script), same lesson as above. Compare its macro-F1/accuracy directly against
S-B's 0.345-0.449/0.429-0.536 before writing anything into `main.tex` about which
scope "wins."

## 2026-10-06 update — S-B is not hopeless; a 5-D wide-prior re-training campaign is in flight

**Read this block first**, then the 2026-09-28 pivot block below it. This session
revisited System II's S-B scope after the 2026-09-28 pivot's S-B3 direction ran into its
own problems (cherry-picked seed, unstable `(alpha,beta_r)` correlation — see
[[finding_sb3_banana_unconfirmed]] memory). Full technical detail for everything below
lives in `docs/Manuscript/system_ii_sb_identifiability_story.md` (§0 for status, §7 for
the plan) — this entry is the short pointer version.

**What was found, in order:**

1. `sbi-logs/wu2003_posterior_sb.pkl` (2026-07-03, `main.tex`'s current source) and
   `sbi-logs/wu2003_posterior_sb_matched_full.pkl` (2026-08-27, 0/8 seeds SBC) both
   predate or were undermined by two separate, now-understood problems: the
   2026-08-12 `beta_r`/`Q_j` physics fix, and a silently-non-converged-warm-start data
   contamination bug (fixed as part of the S-C work, `_matched_y0`/`sbi_pipeline.py`).
2. A clean S-B retrain on the already-existing, never-before-used post-fix bank
   (`data/wu2003_sbi_train_sb_paired_lognormal_true_n1000_stage1bfix.npz`, "Arm C") gets
   **full 5-parameter SBC pass** (`zscorenone` seed 0) — contradicting the belief that
   S-B's `(alpha/eta_col, z_A0_eff)` confound alone makes calibration impossible.
3. But that SBC-perfect posterior is **catastrophic at classification**
   (macro-F1=0.089, `scripts/eval_sb_armc_classification.py`) — the narrow
   truncated-lognormal training/SBC prior (adopted `nb37` specifically to make SBC
   *easier*) doesn't cover real fault severities. Same mechanism S-C already hit
   ("Gate C", `system_ii_sbi_solidification_plan.md`), now confirmed project-wide.
4. Widening only `alpha`'s prior (`mixed_alpha_uniform_prior()`, cheap, ~75% accept
   rate) gave a real ~2.4x classification improvement at a tiny `n_train=200` pilot
   (macro-F1 0.089→0.211) — not a full fix (`column`/`feed` still F1=0.000, since their
   own priors weren't touched). **New standing methodological finding, seen twice now**:
   the SBC-*best* seed was the classification-*worst* seed in both the narrow-prior and
   alpha-widened experiments — do not use small-sample SBC pass/fail alone to pick a
   production seed for this system.

**Current plan: full 5-D wide-prior S-B bank, `n_train=5000`.**

- `scripts/generate_sb_wideprior_bank.py`: `mixed_wide_prior(uniform_idx=(0,1,2,3,4))`
  — **all five** of alpha/beta_r/eta_col/xi_reb/z_A0_eff widened to `Uniform` over
  their prior box (equivalent to `box_uniform_5d()`). Checkpointed every 50 accepted
  rows, resumable (loads `.partial.npz` and continues — safe to interrupt at any time).
- **Why all five, including z_A0_eff** (this needed a correction mid-session — an
  earlier justification claiming z_A0_eff's fault deviations are "proportionally small"
  does not survive its own quantitative check: z_A0_eff's max deviation from the narrow
  prior's median across the 14-scenario catalogue is 2.60 log-sigma, *larger* than
  eta_col's/xi_reb's (2.23 each). The real reason z_A0_eff's *classification* behaviour
  is still expected to improve less than the other four is mechanistic, not
  magnitude-based: it has a confirmed, representation-robust confound with
  `alpha`/`eta_col` (Artefact 3, 5 independent confirmations, survives raw-trajectory
  FIM) — an information-theoretic limit of S-B's instrumentation, not a
  prior-coverage problem. Widening still buys honest, non-shrunk z_A0_eff point
  estimates and correct SBC for it, just not fixed `feed`-class classification.
- **Cost, measured not assumed**: a 10-row smoke test of the (now-superseded) 4-param
  version measured 68.4s/accepted-row at 43% accept rate → **~105h projected** for the
  full `n_train=5000 + N_SBC=120 + N_SBC=400 = 5520` accepted rows. This is a multi-day,
  multi-session background campaign, not a single run — expect to resume it repeatedly.
  `notebooks/35_wu2003_sb_wideprior_bank_generation.ipynb` (+ `scripts/build_nb_35.py`)
  documents the recipe, the prior-coverage check, and a fresh smoke test under the
  now-corrected 5-parameter prior (re-running as of this entry — check its output for
  the up-to-date accept rate/ETA before trusting the 105h figure above).
- **Beta_r isolation test: DONE, result is clean and positive.**
  `scripts/generate_sb_betaronly_bank.py` + `scripts/run_sb_betaronly_training.py`
  (`n_train=200`, `beta_r` widened alone via `mixed_wide_prior(uniform_idx=(1,))`,
  other 4 params on their original narrow lognormal). Bank generation accept rate
  ~92% (vs. alpha-only's ~75%) — beta_r's linear role is easier for the warm-start
  solver too, not just for training, consistent with the mechanistic hypothesis.
  **`beta_r`'s own SBC p-value is remarkably stable and reproducible across seeds**:
  `zscorenone`, confirm budget (N=100), **exactly p=0.518 for all 3 seeds** — nothing
  like `alpha`'s wild, seed-dependent swings (p≈0.5 down to p=1e-16 across seeds in
  every widened-prior variant tested). The actual failing parameter in every run is
  `xi_reb` (p=0.0006-0.036, consistently) — expected, since it was deliberately kept
  on its original narrow prior in this isolation test and is already known to be the
  worst-behaved parameter (information lives in one narrow `Q_reb` channel only).
  **Conclusion: `beta_r` is cleared for the full 5-parameter campaign** — a flat prior
  for it is not a stability risk, it is if anything the most well-behaved of the
  widened parameters tested so far. (`results/sb_betaronly_sbc_n200.csv` has the full
  per-parameter breakdown.)

**Next steps, in order:** (1) ~~let the beta_r-only isolation pilot finish~~ **DONE,
cleared beta_r** (above). (2) ~~let notebook 35's fresh 5-parameter smoke test
finish~~ **DONE** — initial 10-row smoke test gave 56.5s/accepted row, 56% accept
rate, ~86.6h projected; **superseded by the real run's own first checkpoint** (50
rows/136 attempts, a much less noisy sample): accept rate settling around **37%**,
**~102h revised ETA** — the smoke test's 56% was small-sample noise, not a real
number; trust the running campaign's own logged accept rate over either estimate as
more data accumulates. (3) **LAUNCHED 2026-10-06** —
`caffeinate -i -s python3 scripts/generate_sb_wideprior_bank.py` is running in the
background now, logging to
`data/wu2003_sbi_train_sb_n5000_wideprior_stage1bfix.progress.log` (tail that file
directly to watch progress — a checkpoint line every 50 accepted rows, plus a
timestamped `=== (re)started ===` line on every resume). **This WILL get interrupted
repeatedly** (background process timeout caps, machine sleep/restart, session
boundaries) over its ~100h+ projected total — that is expected and fine, the script
is checkpointed every 50 accepted rows and resumes automatically from `.partial.npz`
on re-run. **If you are picking this up in a new session: check whether the process
is still running (`ps aux | grep generate_sb_wideprior_bank`); if not, just re-run
the exact same command to resume — do not regenerate from scratch.** Progress so far
can be read from the progress log above, from
`data/wu2003_sbi_train_sb_n5000_wideprior_stage1bfix.partial.npz`, or the
non-partial `.npz` once done, and the equivalent SBC-set partial files in `results/`.
(4) ~~once the bank + SBC sets are done, train... run SBC~~ **DONE 2026-10-14.** Bank
generation campaign finished (full 8-day wall-clock span, most of it lost to two sleep
interruptions — actual compute ~169h; final accept rate 0.38). All three artifacts
confirmed correct shape: train `(5000,5)`/`(5000,66)`, SBC-reduced `(120,5)`/`(120,66)`,
SBC-confirm `(400,5)`/`(400,66)`. Trained 5 seeds × 2 z-scoring variants
(`scripts/run_sb_wideprior_training.py`, `results/sb_wideprior_sbc_n5000.csv`).

**Initial KS-SBC read (superseded below, keep for the methodological lesson):**
under `zscorenone`, all 5 seeds pass `alpha`/`beta_r`/`eta_col`/`z_A0_eff` comfortably
at the confirm budget (KS p ~0.05-0.96) and only `xi_reb` fails catastrophically in
every one of the 10 (seed x z-scoring) combinations (p from 8e-3 to 5.9e-16).
**Classification on these `zscorenone` posteriors was catastrophic** (macro-F1
0.067-0.142, accuracy 0.164-0.286, `results/` — worse than even the narrow-prior
baseline) **despite the "good" SBC read.**

**Root cause found, and it is the real headline finding of this run, not `xi_reb`.**
Checked the pre-registered anti-shrinkage regression slope (posterior-mean-vs-truth)
and 90%-coverage diagnostics `scripts/run_sb_wideprior_training.py` computes
alongside KS (exactly the gate `system_ii_sbi_solidification_plan.md` §Stage 0.3
added after it first caught `alpha`'s shrinkage). Under `zscorenone`: `alpha` slope
~0.7-0.82 (fine), `xi_reb` slope ~0.81-0.90 (fine, so `xi_reb`'s KS failure is a
shape/calibration problem, not an informativeness problem — the
"information-poverty" explanation below is still directionally right but is not
the dominant effect here) — but **`beta_r` slope ~0.26-0.42, `eta_col` slope
~0.09-0.18, `z_A0_eff` slope ~0.01-0.09**. Those three parameters' posterior means
barely move as a function of the observation at all — a near-complete collapse to
the prior, three of five dimensions. **This still "passes" SBC and even shows
correct ~90% coverage**, because a posterior that just resamples from the prior
regardless of `x` has uniform ranks and correct marginal coverage *by construction*
(the true value is itself drawn from that same prior) — KS/coverage alone cannot
distinguish "well-calibrated and informative" from "uninformative but
accidentally well-calibrated." This is the same lesson as every prior seed-selection
finding this session, now shown to also apply at the representation/hyperparameter
level, not just the seed level.

**Fix found immediately, same session: switch to default (`baseline`) z-scoring.**
Checked `baseline`'s own slopes: **0.945-1.004 for all 5 parameters, all 5 seeds** —
essentially perfect, unshrunk truth-tracking. `baseline` instead fails KS-SBC in
most seeds (likely overconfident/misshapen intervals, not checked in detail yet),
but a biased-interval-width problem is far less damaging for classification than an
uninformative-posterior-mean problem. **`z_score_x='none'` was adopted earlier in
this project specifically to fix `z_A0_eff`'s near-delta-dimension domination under
the S-C *analyzer* feature** (`finding_sc_delta_dimension_hypothesis` memory) — that
fix does not transfer to this context (plain S-B, no analyzer feature, full 5-D
uniform prior spanning each summary statistic's full raw-scale range) and is
actively harmful here. **Re-ran the classification check on the `baseline`
(not `zscorenone`) posteriors — DONE, confirms the hypothesis cleanly.**
`baseline` classification (same 14-scenario catalogue, 5 seeds):
macro-F1 **0.345–0.449**, accuracy **0.429–0.536**
(`scripts/eval_sb_wideprior_classification.py`, edited to the `baseline` variant) —
roughly 3–4× `zscorenone`'s 0.067–0.142, and clearly better than every earlier S-B
variant this session (narrow-prior clean retrain: 0.089; alpha-only-widened pilot:
0.211). Best seed (seed 1): macro-F1=0.449, with **`column` finally working**
(F1=0.600) — the first time `column` has scored above 0.000 in *any* variant tried
this entire investigation (narrow prior, alpha-only, mixed-alpha, beta_r-only, and
`zscorenone`-wideprior all gave `column` F1=0.000). `feed` stays at F1=0.000 in
every seed, as expected — that is the genuine Artefact-3 confound
(`confound_and_artefact_catalogue.md` §B3), not a training-recipe problem, and no
training fix was expected to touch it.
**Promote `baseline` (default z-scoring) as the production recipe for this bank.**
Treat `zscorenone` as a cautionary, context-specific lesson (it was adopted for the
S-C *analyzer*-feature near-delta-dimension problem and does not transfer to plain
S-B's full-range-uniform-prior context), not a default to carry forward blindly.
**One more diagnosed detail worth tracking**: per-scenario inspection of seed 1's
`baseline` posterior shows `xi_reb`'s point estimate swinging wildly low (0.03-0.19)
specifically on catalyst-fault scenarios (`W2`-`W4`, `W11`, true `xi_reb=1.0`) —
i.e. `xi_reb`'s KS-SBC failure may be (at least partly) a genuine residual confound
with `alpha`/reactor-unit faults under this representation, not purely the
previously-assumed information-poverty story. Worth a dedicated
`cov_offdiag_corr`-based FIM check on `(alpha, xi_reb)` under S-B before committing
to the `Q_ratio_feat` feature-engineering fix below.

**`xi_reb` info-poverty note, still likely worth pursuing regardless of the above**:
its own prior WAS widened in this run and it still fails KS/shape calibration (slope
is fine, so this is specifically a shape/overconfidence issue) — the already-designed
`Q_ratio_feat`/`T_reb_residual` engineered feature combo (validated for S-C.2 at
`corr=-0.99` with `xi_reb`, built from already-observed S-B channels, no new sensor)
remains the natural next lever if `xi_reb`'s calibration still isn't good enough
once the z-scoring fix is in place.

(5) ~~re-check the `(alpha,z_A0_eff)`/Artefact-3 confound and the
`(alpha,beta_r)`/Artefact-2 banana~~ **DONE 2026-10-14**, on the actual trained
`baseline`-recipe posterior (all 5 seeds), not just a methodology audit of the old
numbers.

**Artefact 2 (`(alpha,beta_r)` at W11) does NOT reproduce.** `Corr(alpha,beta_r)`
across the 5 seeds: **−0.115, +0.674, +0.296, −0.235, −0.025** — nowhere near the
manuscript's +0.998, and not even consistent in sign. This is now direct evidence,
not just the earlier provenance/magnitude doubt (`notebooks/42`) — **treat the
original +0.998 finding as not reproducible with current tooling.** Full detail
and two open interpretations (stale-posterior artefact vs. genuinely seed-unstable
quantity) in `confound_and_artefact_catalogue.md` §B2's 2026-10-14 update.

**Artefact 3 is partially corroborated, partially inconclusive from this specific
check.** `(alpha,z_A0_eff)` at W13: positive in all 5 seeds (+0.245 to +0.717) —
consistent sign is mild support, still seed-variable in magnitude. `(eta_col,
z_A0_eff)` at W10: weak and sign-flips across seeds (−0.286 to +0.318) — this
specific point may just not be where that pair's coupling is strong (consistent
with this project's established point-dependence pattern); the W2/interior-point
FIM evidence remains the primary basis for this pair, not this check. Detail in
`confound_and_artefact_catalogue.md` §B3's 2026-10-14 update.

**Side investigation, inconclusive, flagged not resolved**: chased a lead that
`xi_reb`'s own KS-SBC failure might be a genuine `(alpha,xi_reb)` confound (its
point estimate swings wildly low, 0.03-0.19, specifically on catalyst-fault
scenarios with true `xi_reb=1.0`). Local FIM found only weak correlation
(−0.05 to +0.21 across 3 points) — no confound. The deterministic grid-scan
(`scripts/identifiability_scan.py --param1 alpha --param2 xi_reb`) gave an
**unexplained alternating-row pattern** (grid points at `alpha=0.607`/`0.664` show
distance ~1-2, while equidistant-from-truth neighbors `0.636`/`0.693` show
distance ~4-5 — not a smooth ridge/minimum, in both the robust and full feature
subsets) — likely a numerical artefact in the deterministic warm-start/steady-state
solve for this specific parameter pair, not independently debugged. **Do not treat
either the FIM or the grid-scan result as a settled answer on `(alpha,xi_reb)`** —
this needs its own follow-up before concluding anything.

Related memory: [[finding_sb_calibration_gap_and_artefact3_wrong]],
[[finding_sb_mixedalpha_prior_fix]], [[finding_sb3_banana_unconfirmed]].

## 2026-09-28 update — thesis pivot: anomaly detection is now the headline, not identifiability

**Read this block first.** Everything else in this file predates a narrative pivot decided
this session via a structured design review ("GRILL" session).

**The pivot:**
- **Old framing:** identifiability analysis (the (α,η_col)/(α,β_r) "banana" diagnostics)
  was the paper's headline contribution for System II.
- **New framing:** SBI for robust fault/anomaly detection in plant-wide, closed-loop
  systems is the headline. Identifiability analysis is retained but demoted to an
  *explanatory* mechanism (why certain fault pairs are hard to individually attribute),
  not a standalone result. This matches the framing that got a closely related paper —
  Casolo et al., "Fast Bayesian equipment condition monitoring via simulation based
  inference: applications to heat exchanger health," accepted at *Engineering
  Applications of Artificial Intelligence* (EAAI-26-10566) — through 5 reviewers. Source
  PDF (with the full review letter) is at
  `/Users/simo/Downloads/EAAI-26-10566_R1 (1).pdf` if the reasoning below ever needs to be
  re-derived.

**`article_outline_CChE.md` has already been fully rewritten to this new framing** (done
this session, 2026-09-28) — read it, not the pre-2026-09-28 framing described later in
this file, for the paper's current intended structure and claims. Key decisions recorded
there in more detail than here:
1. Identifiability demoted to an explanatory device; classification/detection promoted to
   headline (System II's results section is now classification-first).
2. No real Wu 2003 plant data exists or is coming — the paper explicitly argues
   synthetic-only is the *correct* evidence standard for calibration/SBC claims (real
   plant data can't provide known ground truth), rather than leaving this as an implicit
   gap for a reviewer to flag.
3. System I (propylene oxide CSTR) is kept close to as-written — it already reads as the
   "controlled reference point" the new thesis needs (calibrated, every weak spot
   explained mechanistically, not hidden). No major changes needed there.
4. **System II's reported scope is now "S-B3" only**: 4 endogenous parameters
   (`alpha, beta_r, eta_col, xi_reb`), `z_A0_eff` held exogenous, feed-fault scenarios
   (W10, W13) excluded. The older 5-D `z_A0_eff` identifiability findings
   ((α,z_A0_eff)/(eta_col,z_A0_eff) couplings, nb39-42) survive only as brief motivating
   context for why the scope was narrowed — not as parallel results.
5. **No EKF baseline is reported for System II in the current scope** — turned into an
   explicit positioning argument instead (NUTS is infeasible at this plant's scale; EKF
   needs white-box raw-model access, in tension with the black-box/legacy-simulator
   framing SBI is being sold on). Live contingency: re-run EKF specifically under S-B3 if
   the classification numbers land weak enough to need a comparison point.
6. The outline (and, when you get to it, `main.tex`) were deliberately written to be
   **agnostic to whether the pending S-B3 SBC campaign (below) fully succeeds** — the
   headline is classification/detection robustness either way; calibration gets reported
   honestly with whatever the best current evidence shows. Do not block finishing the
   paper on the campaign fully clearing SBC.

**`docs/Manuscript/main.tex` has NOT been updated to this framing yet.** The user asked
for the outline plus this HANDOFF entry first, specifically so the next steps are clear
once the simulation bank lands — `main.tex` editing is explicitly still pending. Do both
the outline's `[TBD]` fill-ins and `main.tex`'s System II section rewrite in the same pass
once the campaign numbers are in, following the same decisions recorded in the outline
(don't let the two documents drift out of sync once `main.tex` editing starts).

### What to do when the S-B3 simulation bank is ready

The 20,000-50,000-simulation widened-lognormal-prior campaign was launched across ~4-6
machines this session (see
`docs/Manuscript/sb3_widelognormal_campaign_machine_setup.md` for the machine/dependency
manifest, and `notebooks/22e_wu2003_data_generation_sb3_widelognormal.ipynb` for the
per-machine generator — the sigma widening itself lives in
`src/cstr_sbi/recycle/sb3.py`'s `LOGN_SIGMA_SB3_4D_WIDE`). **Once every machine's chunk is
copied back into one shared `data/` folder:**

1. Run `notebooks/22f_wu2003_sbc_confirm_sb3_widelognormal.ipynb` **once, centrally**
   (reserved seed 999999), if not already done, to produce the held-out N=800 SBC
   confirmation set. Do not skip this even if it feels redundant with the training banks —
   it is the only honest calibration check, since it is never pooled into training.
2. Run `notebooks/22g_wu2003_bank_aggregation_sb3_widelognormal.ipynb` to pool every
   machine's `data/wu2003_sbi_train_sb3_widelognormal_seed*_n*.npz` into one bank. Check
   its round-trip assertions (row counts, no duplicate seeds) actually pass before
   proceeding — don't eyeball it.
3. Run `notebooks/24d_wu2003_sbi_training_sb3_widelognormal.ipynb`: trains a seed sweep,
   evaluates each seed against the N=800 held-out confirm set, reports per-parameter KS
   p-values explicitly (not just an aggregate pass/fail), and — if a seed clears SBC —
   runs classification against the 12-scenario S-B3 catalogue.
4. **Read the per-parameter SBC result and the classification macro-F1/per-class-F1
   honestly, whatever they turn out to be.** Per the framing decision above, a
   fully-calibrated posterior is not required for this paper's thesis to hold — but the
   real numbers are needed either way, not a hoped-for outcome.
5. Fill in every `[TBD]` marker in `article_outline_CChE.md` (Abstract, §7.1, §7.2, §9
   Conclusion, Figures/Tables summaries) with the actual results from steps 3-4.
6. Edit `docs/Manuscript/main.tex`'s System II sections (`Results: System~II`, the
   `Structural Identifiability Analysis` subsection — reframe per the outline's §8.1
   explanatory framing — the Discussion's "One Genuine Limit, Three Representation
   Artefacts" section, and the Conclusion) to match the outline's new content. The
   outline is the plan; `main.tex` is the thing that actually needs to be correct for
   submission.
7. **Decide the EKF contingency** (outline §7.4/§8.4): if classification numbers are weak
   enough to want a comparison point, re-run EKF specifically under the S-B3 4-D/12-
   scenario scope before writing the Discussion's EKF section; if classification numbers
   are strong, the positioning argument already written in outline §7.4 stands as-is and
   no EKF re-run is needed.

---

**Read this first, then follow the pointers below into the planning docs.**
This file is deliberately short: it says *what's outstanding and where the detail
lives*, not the detail itself. The source-of-truth documents are:

- `docs/Manuscript/pending_manuscript_fixes.md` — the running log. Every concrete
  finding, number, and status update lives here, in execution order. **This is the
  one to read for "what actually happened."**
- `docs/Manuscript/notebook_execution_plan.md` — the master numbered task list
  (items 1–21, grouped into Stages 0–7) that `pending_manuscript_fixes.md`'s entries
  are keyed against. **This is the one to read for "what's left, in what order."**
- `docs/Manuscript/SI_completion_plan.md` — the Supporting Information's own defect
  list and step plan (Steps −1 through 6), currently blocked on System II's Stage 3
  numbers settling (System I's are now settled — see below). **This is the one to
  read before touching `supporting_information.tex`.**
- `docs/Manuscript/reviewer_response_plan.md` — the reviewer-comment-by-comment
  response tracker (Part A/B/C), useful for *why* a fix matters to a referee, less
  useful for *what to do next operationally*.
- `docs/Manuscript/system_ii_analyzer_rework_plan.md` — **plan of record, Phase 0
  complete, channel choice revised** (2026-09-01): adds an analyzer to resolve the
  confirmed `(alpha, z_A0_eff)`/`(eta_col, z_A0_eff)` identifiability limits
  (`notebooks/39-42`), targeting a three-arm manuscript narrative (System I / System
  II+analyzer / System II-no-analyzer, all classification + 30-day tracking).
  **Phase 0 (`notebooks/43_wu2003_analyzer_feasibility_check.ipynb`, executed)
  rejected the originally-proposed reactor-composition analyzer** (`z_A`; barely
  moves either confound even at full raw-trajectory resolution — its own dynamics
  are driven by both `alpha` and `z_A0_eff` at once, so observing it doesn't
  separate them) **and found a feed-line composition analyzer works instead, but
  only above a 0.01% relative-noise precision threshold** (at the standard `0.3%`
  sensor noise used elsewhere in this study it only partially helps, because the
  existing 66-D summaries already constrain both parameters to `sd~0.0003-0.0005`
  along a correlated direction — a bolt-on sensor must beat that *existing implicit
  precision*, not just add "a" measurement; a first-pass 3-point check suggested
  `0.03%` would suffice, but a fuller 7-point sweep spanning both sides of the
  `(alpha, z_A0_eff)` sign flip found `0.03%` still leaves `|corr|=0.149` at the
  `W2`-like operating point — **`0.01%` is the settled, robust figure**, not
  `0.03%`). **Real-analyzer citation now settled too**: the Kett BSS1700 NIR
  composition analyzer (`references.bib:KettBSS1700`) publishes accuracy `±0.01%`
  and precision/reproducibility `~0.001%` directly in composition-percent terms —
  at the assumed `z_A0_eff~0.70-0.90` operating range this is `~0.0013-0.0125%`
  relative, at or below the `0.01%` threshold. (A second candidate, the IAS-PAT SE
  NIR analyzer, was checked and rejected as the primary citation — its spec is
  absorbance-based (AU), not directly comparable without an unstated calibration
  curve.) Read the plan's "Phase 0 result" section before starting Phase 1 — the
  channel choice, noise-level assumption, and citation all changed from the plan's
  original draft.
  **Phase 1 (core library changes) is now also done**: `src/cstr_sbi/recycle/summaries.py`
  and `scripts/sbi_pipeline.py` both support a new `"S-C"` structure (S-B + the
  feed-line analyzer, `FEED_ANALYZER_NOISE_PCT=1e-4`). Verified end-to-end (smoke
  test + a fresh FIM check through the real, integrated `compute_summaries` —
  matches `nb43`'s numbers almost exactly, so the `_compress_heavy_tails` log1p
  step doesn't change the conclusion) — `"S-A"`/`"S-B"` behavior is unaffected.
  **`nb20a`/`nb21a` (S-C companions to `nb20`/`nb21`, both executed) are also done**:
  confirmed S-C's raw physics is byte-identical to S-B's (no new control loop, as
  designed), and confirmed across all 14 closed-loop scenarios that the
  feed-analyzer reading correlates `+1.0000` with true `z_A0_eff` vs. only `+0.38`
  for S-B's reactor-linked `F_R_norm` — a full-catalogue, quantitative confirmation
  of the mechanism `notebooks/39-42` predicted.
  **`nb22a`/`nb23a` (S-C companions to `nb22`/`nb23`, both executed) are also done.**
  Extended `src/cstr_sbi/recycle/{summaries,simulator}.py` first
  (`SC_CHANNELS`, `generate_sc_observations()`, optional `observations_sc` support in
  `save_dataset`/`load_dataset`). `nb22a` augments `nb22`'s existing dataset with the
  feed-analyzer channel with **no ODE re-simulation** (synthesized directly from each
  window's already-known `z_A0_eff`) — `r=1.000000` across all 420 windows, saved to
  `data/wu2003_observations_sc.npz`. `nb23a` built the 72-D S-C summary matrix and
  reran the FIM comparison with the **corrected** `cov_offdiag_corr` (not `nb23`'s
  original un-inverted convention, which is the same issue `nb42` already flagged for
  `nb33` — `nb23`'s own Section 7/Figure 8 still needs that re-audit, not done here):
  **S-C collapses both confounds to near-zero** (`(alpha,z_A0_eff)`: `-0.48`/`-0.60`→
  `-0.01`; `(eta_col,z_A0_eff)`: `+0.76`/`+0.18`→`+0.03`/`+0.01`), matching Phase 0's
  deterministic finding. One honest caveat documented in-notebook: `sklearn`'s
  `mutual_info_regression` ranks the feed-analyzer feature only 18th/72 for
  `z_A0_eff` despite `r=1.0` — a known KNN-MI-estimator artifact for near-discrete
  features (only 3 distinct `z_A0_eff` values in the scenario catalogue), not a real
  finding; trust the correlation/FIM evidence instead. Saved
  `data/wu2003_summary_features_sc.npz` for `nb24a`.
  **Phase 2 is IN PROGRESS as of this writing** (`notebooks/24a_wu2003_sbi_training_sc.ipynb`,
  launched caffeinated/backgrounded, not yet complete — check for a completed
  `.ipynb` with populated outputs and `results/24a_sysII_sc_*.csv` before assuming
  it finished). Before launching, clarified where SBC has actually lived in this
  project (the user's question): there is a canonical `run_sbc()` in
  `scripts/sbi_pipeline.py`, but it draws from the *original uniform* prior — since
  `nb37` switched to a lognormal training prior, every SBC run since (`nb37`,
  `38`, `39`) hand-rolls its own inline SBC loop instead. Fixed in the library
  (not just noted): `run_sbc()` now takes an optional `prior=` argument. `nb24a`
  itself still uses the established inline/shared-SBC-set pattern (needed for a
  fair cross-seed comparison `run_sbc`'s per-call-fresh-draws design doesn't give).
  `nb24a` trains plain identity-space SNPE on S-C (matched protocol, lognormal
  prior, `n=5000`, 3 seeds) and runs SBC at a reduced budget (`N_SBC=120`) then
  automatically escalates to a confirmation budget (`N_SBC=400`) regardless of the
  reduced-budget outcome, per `finding_sbc_power_eta_col`'s lesson not to trust
  small-N SBC alone. **Do not cite or promote any `sbi-logs/wu2003_posterior_sc_*.pkl`
  until this notebook's confirmation-budget results are in and reported.**
  **Plain identity-space training failed catastrophically at both budgets**
  (0/3 seeds at N=120 and N=400; results in `results/24a_sysII_sc_reduced_sbc_n120_ntrain5000.csv`
  / `..._confirm_sbc_n400_ntrain5000.csv`) — and **`z_A0_eff` itself, the parameter the
  analyzer directly targets, was the single worst-failing parameter in every seed**
  (matched `min_ks_pvalue` exactly each time). Diagnosed rather than assumed: a
  `reject_outside_prior=True` vs. `False` check ruled out rejection sampling as the
  cause for `z_A0_eff` (near-identical rank behaviour either way; `z_A0_eff` never
  left its own prior box in any test draw) — the real leakage driver turned out to be
  `eta_col`/`beta_r`, not `z_A0_eff`. `z_A0_eff`'s actual failure: a genuine flow
  training artifact — the posterior was both biased (~`0.001-0.003` offset) and
  wider than the theoretical noise floor (`std~0.0003-0.0014` vs. a theoretical
  `~0.0001`), i.e. the flow hadn't converged onto the sharp, near-deterministic
  relationship the analyzer provides. **Fix found and confirmed**:
  `posterior_nn(..., z_score_x='none')` alone (not combined with more training
  patience, which made it worse) took `z_A0_eff` from `p=1.26e-10` to **`p=0.359,
  0.670, 0.896` at the N=400 confirmation budget** — genuinely fixed, holds at high
  power, not a small-N false positive. **However, the reduced-budget pass this fix
  gave (2/3 seeds at N=120) was itself misleading** — at N=400, 0/3 seeds pass
  overall, because `alpha` (2/3 seeds) and `xi_reb` (2/3 seeds, different seeds)
  now fail instead, a separate problem unrelated to the feed analyzer that the N=120
  budget didn't have power to detect. Results in
  `results/24a_sysII_sc_zscorenone_confirm_sbc_n400_ntrain5000.csv`. **Net: S-C is
  confirmed better than S-B in one specific, real respect (the analyzer's own
  parameter is now robustly resolvable) but is not yet a fully calibrated
  posterior** — `alpha`/`xi_reb` need the same diagnostic treatment (leakage check,
  bias/width check) before assuming `n_train=15000` (the user's original ask,
  still untested) is the right lever. Do not promote any
  `sbi-logs/wu2003_posterior_sc_*_zscorenone_*.pkl` for downstream use yet.
  (Execution note: `notebooks/24a_wu2003_sbi_training_sc.ipynb`'s own `nbconvert`
  re-execution hung for 2+ hours with almost no CPU use after this fix was applied,
  cause not fully diagnosed; the results above were obtained by running the same
  code directly as a script instead, saved to the notebook's own expected CSV
  paths. The notebook's cells are not fully re-executed/populated with outputs as
  of this writing — treat the CSVs and this HANDOFF entry as authoritative over the
  notebook's own (stale) cell outputs until it is cleanly re-run.)
  **2026-09-02 follow-up — `alpha`/`xi_reb` diagnosed, then `n_train=15000` tested and
  REJECTED.** Diagnostic script (`/tmp/diagnose_alpha_xireb.py`, `sbi-logs/wu2003_posterior_sc_lognormal_true_n5000_zscorenone_seed{0,1}.pkl`,
  `results/24a_sysII_sc_lognormal_sbc_set_n400.npz`) ruled out leakage as the driver
  (per-parameter box violations now small and diffuse, `0-1.4%` in the spot check, no
  single dominant culprit like the old `eta_col` leakage) and instead found a clean,
  quantitatively confirmed **shrinkage-toward-prior-mean** pattern for `alpha`: plotting
  `(true - 1.0)` against `(posterior_mean - true)` across 10 SBC draws on the failing
  seed shows a strong, monotonic, sign-flipped relationship — e.g. `true=0.60 →
  posterior pulled UP by +0.21`, `true=1.18 → posterior pulled DOWN by -0.15` — the flow
  regressing toward the prior's central tendency instead of tracking the true value in
  the tails. `xi_reb` showed a noisier version of the same thing, with 2/10 draws
  showing large (`~0.10-0.20`) downward misses. This is a *different* failure signature
  than `z_A0_eff`'s (which was a small consistent offset with an otherwise well-scaled
  width, fixed by `z_score_x='none'`) — shrinkage is the textbook symptom of
  insufficient effective training density away from the prior mode, which more training
  data is supposed to fix. **On that basis, `n_train=15000` (matched protocol,
  `z_score_x='none'` retained, same bank-generation/training/SBC recipe, standalone
  script `/tmp/nb24a_n15000_run.py` — full listing preserved there, run caffeinated in
  background, bank generation took ~54 min at `0.21-0.22s/draw`, zero rejected draws)
  was tested. Result: still 0/3 seeds pass at `N_SBC=400`
  (`results/24a_sysII_sc_zscorenone_confirm_sbc_n400_ntrain15000.csv`), and it is
  unambiguously WORSE than `n_train=5000` in several respects, not better:**
  - `z_A0_eff` **regressed** — it was robustly fixed at `n=5000` (`p=0.359, 0.670,
    0.896` all 3 seeds) but now fails outright in one seed at `n=15000` (`p=2.4e-5`).
  - `xi_reb` got **dramatically worse** — from borderline at `n=5000` (1 pass, 2 fails
    around `1e-5`-`1e-6`) to catastrophic at `n=15000` (two seeds at `6.9e-10` and
    `2.0e-32`).
  - `eta_col` went from 2/3 passing at `n=5000` to 1/3 at `n=15000`.
  - `alpha` still fails all 3 seeds at `n=15000`, no clear improvement over `n=5000`.
  - `beta_r` stayed solid at both `n_train` values — never the problem.
  Leakage stayed modest (`5-8%` outside prior support) at `n=15000` too, so this isn't a
  rejection-sampling story either. **This rules out the shrinkage/insufficient-data
  hypothesis as the fix**: 3x more data should have monotonically helped if that were
  the true cause, and should not have broken something (`z_A0_eff`) that was already
  fixed. The pattern instead looks like bank/training instability that does not scale
  cleanly with `n_train` — e.g. the flow's fixed capacity (`hidden=60`,
  `num_transforms=3`, unchanged since `nb37`'s smaller-scale problem — see
  `feedback_network_sizing` memory, which validated this size for a *different*,
  already-solved problem) not scaling with more data, or genuine sensitivity to the
  particular lognormal-prior bank realization at higher `N`. **Do not try another
  training-hyperparameter sweep blind — two have now failed (more patience combined
  with `z_score_x='none'` made things worse; more data made things worse).** The
  natural next diagnostic, not yet run: repeat the `nb39-42` FIM/`cov_offdiag_corr`
  methodology on S-C specifically for `(alpha, ?)` and `(xi_reb, ?)` pairs (not just the
  `z_A0_eff`-involving pairs already checked in `nb23a`) to check whether there is a
  *residual, genuine* identifiability confound the feed analyzer never targeted (e.g.
  `alpha`'s well-established coupling with `beta_r`, previously investigated as a
  representation artifact in `nb29b` under S-B/S-A — worth re-checking under S-C's
  66-D-inherited summary block, which is unchanged from S-B's). Do not promote any
  `sbi-logs/wu2003_posterior_sc_*_zscorenone_*_n15000_*.pkl` for downstream use; they
  are worse than the `n=5000` versions, which are themselves not yet usable.

---

## Headline: the System I matched-protocol identifiability cascade is now fully DONE (2026-08-18)

**User decision (2026-08-15, unchanged):** the System I β-bias narrative was shown to
be substantially a training-protocol artifact (Item 8), not a genuine irreducible
identifiability limit, and the user decided to **remove it entirely from the
manuscript**. Across the 2026-08-15 → 2026-08-17 → 2026-08-18 sessions, the full
cascade this required has now been executed end to end:

`04 (second rerun, 0.90 threshold) → 06 → 07 → 09 → 10 → 11 → 12 → 16 → 15/33 (FIM
re-derivation) → 14 (claims/conclusions synthesis)` — **every notebook in this list
is DONE**, matched-protocol, and its numbers are reflected in the current
`main.tex`. Full evidence trail, in execution order, in
`pending_manuscript_fixes.md` — search `### Item 8` (many sub-entries, one per
notebook/finding; the last three are `nb16`, `nb15 + nb33`, and `nb14`).

**What this means concretely, for `main.tex`:**

- Table VI (`tab:scenario_results`) and Table VII (`tab:method_comparison`) are both
  final, matched-protocol, 0.90-threshold numbers.
- § Structural Identifiability Analysis (`sec:identifiability`) has been rewritten:
  `eq:fim_ratio` now gives the healthy-point ratio (236×) with prose covering the
  Sc2 value (12×) and the full operating-point sweep (5×–170×); the previously
  untraceable "~600×" analytical figure is replaced with the true, live result
  (~0.6×, opposite ordering, explained rather than hidden); a new paragraph
  incorporates nb33's diagonal-vs-full-covariance methodology-dependence finding
  (~0.9× vs ~11–12× for the same Sc2 point, same summary vector). `pdflatex
  -draftmode` verified clean (twice; no undefined references).
- The `%TODO` for nb16's EKF/UKF tracking comparison is filled in
  (`sec:sequential_tracking`).
- `notebooks/14_claims_and_conclusions.ipynb` — the internal claims/conclusions
  synthesis notebook, previously the single most stale artifact in the repo (it
  predated the entire Item 8 investigation and still stated the fully-retracted
  "250–500×, six-method-confirmed, irreducible" narrative as settled fact) — has
  been rewritten wholesale and re-executed cleanly. Its live-computed dashboard
  numbers now cross-check against Table VI/VII exactly.

**Nothing is currently blocking further System I manuscript work.** The next
things to reach for, roughly in priority order, are listed below.

### Known, deliberately-not-yet-fixed item (tracked, not forgotten)

1. **`supporting_information.tex`'s `fig:analytical_bias` caption (§S8)** repeats
   the same "~600×" error and an incorrect channel-attribution claim that `main.tex`
   just had corrected. Left alone on purpose — `SI_completion_plan.md` already flags
   all of §S3/S6/S8 as needing a substantial, coordinated rewrite once Stage 3's
   numbers settled (which, for System I, they now have); patching this one caption
   in isolation would pre-empt that coordinated pass. Flagged in
   `SI_completion_plan.md`'s §S8 bullet so it isn't lost — **do this as part of the
   SI rewrite, not before it.**
**Resolved 2026-08-26:** the small `nb14` dashboard ambiguity about macro-F1 aggregation
has been fixed. The dashboard now prints the simple scenario-mean F1 and the manuscript's
Table VI class-macro F1 (0.927) with separate labels, so the two conventions are no
longer easy to confuse.

---

## What to work on next (in rough priority order)

0. **[CURRENT TOP PRIORITY, 2026-09-05] S-C.2 classification is genuinely
  improving (mixed prior on `alpha` alone: macro-F1 0.07→0.11) but not yet
  usable — next step is widening `beta_r`/`eta_col`/`xi_reb`'s training-prior
  coverage the same way, smoke-test first.** Full blow-by-blow detail lives in
  `docs/Manuscript/system_ii_sbi_solidification_plan.md` (read that file
  first for anything below this summary) — this entry is the short version.

  **Where this came from.** The matched-protocol warm-start bug (previously
  top priority) is fixed and closed out (see the "Warm-start fix, closed
  out" note right below this block for the historical record — do not
  re-diagnose it). Every S-C/S-C.2 bank in this section was generated
  **after** that fix (suffix `_stage1bfix` on all filenames). With the warm
  start no longer a confound, the remaining S-C.2 `alpha`/`xi_reb` SBC
  failures turned out to be a genuinely different, deeper problem, run down
  as follows this session:
  - Ruled out, in order, with a dedicated check each: residual confounds
    (Gate A grid-scan tiebreaker — clean), more training data (`n=15000` —
    made things *worse*), bigger architecture (`90/4` probe — made `alpha`
    *worse*), low fit quality (`inspect_training_loss.py` — "bad" seeds have
    equal-or-better loss than "good" ones, so it isn't underfitting).
  - **Root cause found**: a 10-seed sweep (`scripts/run_sc2_seed_sweep.py`,
    all-lognormal training prior, `z_score_x='none'`, `60/3`, `n_train=1000`)
    gets 2/10 seeds (0 and 3) passing BOTH independent SBC sets — `alpha`'s
    failure is seed-dependent, not systematic, ~20% hit rate. But even a
    passing seed (seed 0) then **catastrophically fails fault classification**
    (`scripts/run_sc2_classification.py`: macro-F1 0.07, only 1/14 Wu2003
    scenarios correct) — direct posterior inspection at `W2`/`W3`/`W4`/`W15`
    showed severe shrinkage-toward-prior-mean (e.g. true `alpha=0.55` →
    posterior mean `0.887`, std only `~0.08`) at every fault scenario tested,
    including mild ones. **The mechanism**: the narrow lognormal training
    prior (`sigma=0.16` in log-space for `alpha`, adopted post-`nb37`
    specifically to make SBC *easier*) assigns almost no mass to the region
    the actual Wu2003 fault catalogue lives in (`W3`-`W15` sit 2.7-3.4
    log-sigma out — a `P≈9e-5` event under the training prior, so a
    `N=400` SBC set essentially never probes it). **SBC-passing under this
    prior does not certify usefulness for classification** — it only checks
    calibration against draws from the same narrow prior, which never tests
    the deployment region.
  - **Fix tried: `mixed_alpha_uniform_prior()`** (`src/cstr_sbi/recycle/
    priors.py` — module-level `MixedAlphaUniformPrior`, `alpha ~
    Uniform(0.4,1.2)`, other 4 params keep their existing lognormal
    marginals; a full 5-D uniform prior was tried first and **aborted** —
    accept rate collapsed to ~37% because multiple simultaneously-extreme
    parameters make the matched-protocol warm-start solve hard to converge).
    Bank regenerated cleanly in ~6h, stable accept rate ~75%
    (`scripts/regenerate_sc2_mixedprior_bank.py` →
    `data/wu2003_sbi_train_sc2_true_n1000_mixedprior_stage1bfix.npz` + both
    SBC sets). **Result** (`scripts/run_sc2_mixedprior_seed_sweep.py`,
    `results/sc2_mixedprior_zscorenone_seedsweep_n10_mixedprior_stage1bfix.csv`):
    **0/10 seeds pass both SBC sets** — worse by raw pass-count than the
    all-lognormal prior's 2/10, and `alpha` is now the worst parameter in
    every seed at the confirm budget. This looked like a regression until
    direct posterior inspection (8 SBC points spanning `alpha=0.40`-`1.11`)
    showed the posterior actually tracks truth within ~1 SD almost
    everywhere — **genuinely fixed, not shrunk** — and the SBC failures are
    concentrated at the extreme upper boundary (`alpha` near `1.2`), a
    region **no named Wu2003 fault scenario ever probes** (faults are all
    `alpha≤1.0`). SBC correctly penalizes that boundary region since it's
    part of the declared prior; the classification task never needs it.
    **Classification rerun on seed 0 under the mixed prior**
    (`results/sc2_seed0_mixedprior_classification_metrics.npz`): **macro-F1
    0.112, accuracy 0.221** (vs `0.071`/`0.133` under all-lognormal) —
    reactor-class F1 roughly doubled; `W4_cat_threshold` (severe,
    `alpha=0.55`) now correctly classified as `reactor` (was `healthy`);
    `W12`/`W16` multi-fault scenarios now at least catch the reactor
    component. **Still far from sufficient**: `W2`/`W3` still misclassified
    as `healthy`, `column`/`feed` classes remain F1=0.000 (untouched by an
    alpha-only fix, as expected), nothing is ever classified `multi`.

  **Next step (not yet done — this is the concrete, actionable item for the
  next session)**: the same coverage-mismatch mechanism just fixed for
  `alpha` plausibly affects `beta_r`/`eta_col`/`xi_reb` too — their earlier
  "SBC passes" were tested against an equally narrow SBC set that may
  similarly under-test their own fault-relevant tails. `z_A0_eff` is the
  likely exception (near-exact dedicated analyzer signal makes it largely
  prior-width-insensitive) — **keep it narrow/lognormal**. Concretely:
  1. Extend `mixed_alpha_uniform_prior`-style construction (or add a new
     `MixedWidePrior` class in `src/cstr_sbi/recycle/priors.py`) to widen
     `beta_r`, `eta_col`, `xi_reb` (uniform or wide-lognormal, informed by
     each parameter's own Wu2003 fault-scenario range) while keeping `alpha`
     as already fixed and `z_A0_eff` narrow.
  2. **Smoke-test first** — a handful of draws — before committing to a full
     bank generation. Widening one parameter (`alpha`) held accept rate at
     ~75%; widening 4 of 5 carries real but not-yet-quantified additional
     cost risk (the full 5-D attempt collapsed to 37%), so check this isn't
     repeating that failure mode before spending ~6h+ on a full bank.
  3. Re-run the same seed-sweep (`run_sc2_mixedprior_seed_sweep.py` pattern)
     and classification check (`run_sc2_mixedprior_classification.py`
     pattern) against the new bank.
  Per the plan's Gate C fallback, if this doesn't resolve `column`/`feed`/
  `multi` classes, the honest manuscript path is a reduced posterior with
  explicitly reported limits (calibrated for `z_A0_eff`/`beta_r`/`eta_col`/
  `xi_reb`, `alpha` reliable only away from its prior's upper boundary) —
  not another blind training/architecture sweep. **Do not proceed to Phase
  4/5 of the analyzer-rework cascade until classification is usable.**

**2026-09-05 evening follow-up (in progress, stopped early per user request — "stop
as soon as you have some preliminary results"):** implemented the "next step" above.
Added `MixedWidePrior`/`mixed_wide_prior()` to `src/cstr_sbi/recycle/priors.py`
(generalizes `MixedAlphaUniformPrior` to an arbitrary uniform-dims subset; default
widens `alpha, beta_r, eta_col, xi_reb` to `Uniform` over their prior box, keeps
`z_A0_eff` narrow lognormal — sanity-checked directly, bounds/log_prob correct).
Wrote the full follow-on pipeline mirroring the alpha-only scripts:
`scripts/regenerate_sc2_wideprior_bank.py`, `run_sc2_wideprior_seed_sweep.py`,
`run_sc2_wideprior_classification.py` (not yet run at full scale — ready to launch).
**Smoke test** (`scripts/smoketest_sc2_wideprior.py`, 10 draws): accept rate
6/10=60%, comfortably above the full-5-D-uniform collapse (37%) though below the
alpha-only case (75%) — looked viable, so a **preliminary small-scale check** was
started (`scripts/prelim_sc2_wideprior_check.py`: n_train=150, one N=40 SBC screen,
single seed, reduced classification budget) but **stopped ~1h in, still only 60/150
training draws generated** (accept rate stabilized 46-48%, consistent with the
smoke test, but real per-accepted-draw wall time was much higher than the smoke
test's early attempts suggested — ~54s/accepted-draw average over 129 attempts,
not the ~18s/attempt the smoke test's first 10 attempts implied). **No SBC or
classification signal was obtained** — bank generation itself was the bottleneck.
**Revised cost estimate for the full confirmation pipeline (n_train=1000 + N_SBC
120+400, 10 seeds)**: at this observed rate, bank generation alone would take
several hours longer than the alpha-only run's ~6h (that run's accept rate was
~75% vs this one's ~47%, so total draws-needed-per-second is roughly
`0.75/0.47 ≈ 1.6x` worse just from the accept-rate difference, before accounting
for the also-higher per-attempt wall time observed here). **Not yet run**: the
full bank (`regenerate_sc2_wideprior_bank.py`), the 10-seed sweep, or
classification. **Next session**: either (a) relaunch
`regenerate_sc2_wideprior_bank.py` in the background for a genuinely long
(~10h+) unattended run and check back later, or (b) reduce scope first — e.g.
widen only 2 of the 3 remaining parameters (the plan's own fallback framing,
Gate C) if the full 4-parameter run's cost looks disproportionate to the
expected benefit. No decision made yet on which.

**2026-09-05 night follow-up — multi-round SNPE pilot (user request: "assess whether
multiround SBI inference can improve the classification issues"):** built
`scripts/pilot_multiround_sc2.py` (+ `_continue.py`), a single-scenario (`W2_cat_mild`,
alpha=0.85, true_class=reactor, still misclassified as `healthy` under the alpha-only
mixed-prior posterior) 2-round SNPE-C test: round 1 = the existing alpha-only
mixed-prior bank/posterior (cheap retrain from disk, ~5s); round 2 = new simulations
drawn from `round1_posterior.set_default_x(x_o)` (sbi's native multi-round API,
atomic-loss proposal correction). **Positive, confirmed finding**: round-2 proposal-
focused simulation is dramatically more efficient than uniform-prior widening —
**99.5% accept rate** (200/201 draws) vs. the wide-prior approach's 37-60%, because
draws cluster near a real, physically-plausible operating point instead of scattered
prior corners. **Two real `sbi` 0.26.1 library gotchas hit and fixed along the way**
(worth remembering for any future multi-round work in this project): (1) round-2
training crashes (`ValueError: NaN/Inf present in prior eval`) if the round-2 proposal
is sampled with `reject_outside_prior=False` — a few draws land just outside the
declared prior box and the atomic loss needs `prior.log_prob` finite for every
training theta; fix is `reject_outside_prior=True` (or post-hoc-filter, done here since
only 3/200 draws were affected) when generating round-2 training draws. (2) **More
seriously**: the classic `SNPE_C`/`trainers/base.py` trainer does **not** reset its
early-stopping bookkeeping (`_val_loss`, `_best_val_loss`,
`_epochs_since_last_improvement`) between `.train()` calls across rounds — a second
round's `.train()` call will silently compare its own (differently-scaled, since the
atomic loss with `discard_prior_samples=True` uses a different contrast-set size) loss
against a **stale round-1 loss value on an incomparable scale**, judge every round-2
epoch as "no improvement" by construction, and revert to round-1's untouched weights —
**with zero error, a normal-looking "converged after N epochs" message, and a posterior
that LOOKS trained but has literally 0.0 max weight diff from round 1** (confirmed
directly by diffing state dicts). Only resetting `_best_val_loss` (not also `_val_loss`)
still reproduces the bug. Correct fix: reset **all three** attributes
(`inference._val_loss = inference._best_val_loss = float('inf');
inference._epochs_since_last_improvement = 0`) immediately before every round's
`.train()` call beyond the first. This is completely undocumented in the library and
would silently invalidate any multi-round result in this project if not caught —
**flag this prominently if multi-round SNPE is used again.**
**Negative/central finding, once the training bug was actually fixed and a real
round-2 update was confirmed (103 epochs, nonzero weight diff)**: round 2 did **not**
fix the classification bias for this pilot scenario — posterior mean for alpha moved
from 1.013 (round 1) to 1.047 (round 2), i.e. slightly **further** from the true 0.85,
and the predicted class stayed `healthy` (true `reactor`) in both rounds. **Diagnosed
why**: round-1's posterior at `x_o` is not merely imprecise, it's *biased* — true
alpha=0.85 sits ~2-2.2 posterior-std below round-1's estimated mean (~1.01-1.03,
std~0.08-0.09). A round-2 proposal sampled directly from that biased round-1 posterior
inherits the same bias (only a handful of the 197 training draws land near the true
region), so round-2 training mostly reinforces the existing wrong neighborhood rather
than correcting it — a known failure mode of sequential NPE when round 1 itself is
poorly calibrated (not simply wide), since later rounds only refine wherever round 1
already believes probable mass lives. **Conclusion**: naive multi-round SNPE (round-2
proposal = raw round-1 posterior at x_o) is not a substitute for fixing round-1's
prior-coverage bias — but the confirmed simulation-efficiency win (99.5% vs 37-60%
accept rate) suggests combining the two is promising: fix round-1's coverage bias
first (the already-planned wide-prior widening, paused per user request earlier this
session), THEN layer a multi-round refinement on top of that less-biased round 1 (or
alternatively, deliberately broaden/hedge the round-2 proposal, e.g. a round1-
posterior/original-prior mixture, instead of using round-1's raw posterior — not yet
tried). **Not yet decided which path to pursue next** — needs the user's input.

**2026-09-06 — full 5-D `box_uniform_5d()` smoke-tested properly post-Stage-1b-fix
(user request, following up the multi-round pilot above):** research into project
history (via agent) found the one clean test that had never actually been run: the
original nb31 uniform-prior posterior gave the best classification result ever seen
in this project (macro-F1=0.694), but that predates both a physics bug fix (Item 6)
and the Stage-1b warm-start fix; its one later "matched protocol" SBC failure
(2026-08-27, catastrophic, 0/8 seeds) **predates the Stage-1b fix** and is therefore
confounded with that since-fixed bug; and this session's own post-fix re-test
(2026-09-04, `scripts/regenerate_sc2_uniform_bank.py`) was aborted at the bank-
generation stage purely on accept-rate grounds (~37%, never even reaching SBC).
**Ran a fresh, careful smoke test** (`scripts/smoketest_sc2_fulluniform.py`, same
Stage-1b-fixed matched-protocol simulate call as every other prior variant this
session) to get a clean read. **Result: 0/5 accepted draws, and — unlike every other
prior tested this session — every single attempt (success or fail) cost a uniform
~45s**, not the fast-success/slow-fail split seen for lognormal/alpha-mixed/wide-
mixed. Stopped early per user instruction ("stop after the smoke test results") once
the pattern was clear. **This confirms and sharpens the ~37% figure rather than
overturning it**: even setting the accept-rate question aside, a full bank generation
at ~45s/attempt regardless of outcome would cost roughly 45s × (1520 draws needed /
accept_rate) — at even a generous 35% accept rate that is ~5,500s × ... **order
~50+ hours**, far beyond every other prior variant tried (lognormal ~18min/5000,
alpha-mixed ~6h/1520, wide-mixed est. 8-12h/1520). **Conclusion: the full 5-D uniform
prior is now confirmed (not just historically suspected) to be prohibitively
expensive under the current matched-protocol warm-start solve, independent of
whether it would eventually calibrate** — the accept-rate collapse is real and,
worse, the failures themselves are expensive (not cheap-and-fast like other variants'
rejected draws), so there is no cheap way to "just let it run longer." **Not
recommended to pursue further without a different mitigation** (e.g. `sbi`'s
`RestrictionEstimator` — a classifier that learns to predict invalid draws *before*
simulating, found in this session's tool search but not yet tried — could in
principle cut the wasted 45s-per-reject cost if invalid regions are learnable/
predictable ahead of the expensive ODE solve; not attempted this session). Small
number of samples (n=5) means the true accept rate/cost could differ, but the
directional conclusion (dramatically worse than every already-tested alternative) is
robust enough not to justify further spend without the user opting back in. **No
bank, SBC, or classification result exists for the full uniform prior post-fix** —
this question remains genuinely open only if a cost-mitigation strategy is found.

**2026-09-06 — three notebooks built for the user to run themselves (token-saving
handoff), covering the remaining options to salvage System II classification after the
full-uniform-prior re-test above was found prohibitively expensive:**

- **`notebooks/44_wu2003_wideprior_multiround_classification.ipynb` (Option 3)** —
  trains round 1 on `mixed_wide_prior` (the paused wide-prior fix, finished here) and
  then, per scenario, runs a focused round-2 multi-round refinement using the exact
  bug-fixed recipe from this session's pilot (`reject_outside_prior=True` for round-2
  training draws; the `_val_loss`/`_best_val_loss`/`_epochs_since_last_improvement`
  triple-reset before round-2 `.train()`). **Fixed one additional bug found while
  reviewing this notebook before handoff** (not yet tested in the live pilot, caught by
  re-reading the SNPE-C source): a fresh, independent round-1 `(inference, posterior)`
  pair must be retrained per scenario in the Step-4 loop — reusing one shared
  `inference` object across scenarios would silently let each scenario's round-2 data
  accumulate on top of every previous scenario's (SNPE-C's `discard_prior_samples=True`
  only discards round-0 data, not intermediate rounds), contaminating what's supposed to
  be an independent per-scenario refinement. `train_round1_fresh()` retrains cheaply
  per scenario instead. Configurable scenario subset (default: 8 of 14, spanning every
  fault class) vs. the full 14, since round-2 cost scales per scenario. Hypothesis: a
  less-biased round 1 (wide prior, vs. the pilot's more-biased alpha-only-mixed round 1)
  might let round-2 refinement actually help, unlike the pilot's negative result.
- **`notebooks/45_wu2003_direct_classifier.ipynb` (Option 4)** — bypasses SBI posterior
  estimation/SBC entirely for classification: generates labeled `(summary,
  true_fault_class)` pairs via `mixed_wide_prior`, trains both a RandomForest and a
  small torch MLP classifier directly, evaluates both on the real 14-scenario catalogue.
  Tests whether the summary vector's information content is fine (arguing the
  bottleneck has been posterior-estimation machinery, not data) or whether classifier
  performance is *also* poor (suggesting the same coverage-mismatch problem recurring in
  a new guise — check the printed training-label class balance first if so).
- **`notebooks/46_wu2003_gate_c_honest_report.ipynb` (Option 5)** — operationalizes the
  project's own pre-registered Gate-C fallback as an actual artifact rather than prose:
  loads the best-already-validated posterior (`mixed_alpha_uniform_prior` seed 0),
  reports a per-parameter calibration status table sourced directly from the existing
  seed-sweep CSV, and adds a confidence/abstention layer (flags each classification call
  `confident` vs `boundary_adjacent` based on how much posterior `alpha`-mass falls above
  the validated region, `>1.11`) — reports three numbers: naive macro-F1, confident-
  subset macro-F1, and the abstention rate, plus a per-scenario breakdown.

All three were built, JSON/syntax-validated, and had their core logic (imports, the
`round2_refine`/`classify_with_confidence` functions, the existing posterior/CSV file
paths) smoke-tested against the real repo before handoff — **not fully executed
end-to-end** (that's the user's task, explicitly to save tokens). **User will run these
themselves and report results back for evaluation** — do not assume any numeric result
from them until reported.

**Follow-up (same session): `notebooks/47_wu2003_restriction_estimator_fulluniform.ipynb`
(Option 2)** — the fourth and final option. Uses `sbi.utils.RestrictionEstimator`
(Deistler et al. 2022) to make the historically-best full-uniform-prior training
(nb31, pre-bugfix, macro-F1=0.694) computationally tractable: trains a small classifier
to predict `valid`/`invalid` draws from `theta` alone *before* paying for the expensive
~45s/attempt ODE solve found in this session's `smoketest_sc2_fulluniform.py`, then
samples training data from the resulting `restricted_prior` instead of the raw
`box_uniform_5d()`. **Structure**: Step 1 bootstraps a labeled dataset (real, expensive
simulate() calls — unavoidable upfront cost, ~250 draws by default); Step 2 trains the
classifier and builds `restricted_prior`; **Step 3 is an explicit go/no-go smoke test**
(20 fresh draws against the *real* simulator, required to clear a 60% accept-rate
threshold before any cell past it runs) — follows this session's own established
"smoke test before committing" practice, since the whole point of this notebook only
pays off if the classifier's restriction genuinely improves the real accept rate over
the historical ~0-37%. Steps 4-6 (full bank, SBC screen, classification) only run if
Step 3 passes (or the user manually overrides). **Design choice, flagged in-notebook**:
bootstrap data is not reused in the final training bank (a subtle correctness/efficiency
trade-off — see the notebook's own markdown). **Pre-handoff validation caught one real
issue**: the classifier's default `stop_after_epochs=20` can under-train on a modestly-
sized bootstrap set (confirmed via a synthetic test — converged to "predict almost
everything valid," 0% rejection, not a genuine boundary); fixed by passing
`stop_after_epochs=40` explicitly and adding an in-notebook warning if the classifier's
own predicted acceptance rate comes back suspiciously close to 1.0. Full
`RestrictionEstimator`/`RestrictedPrior` API workflow (`append_simulations`, `train`,
`restrict_prior(allowed_false_negatives=0.0)`, `.sample(...)`) verified directly against
the installed `sbi==0.26.1` before writing the notebook, not assumed from memory.
**Also not yet executed** — same token-saving handoff as 44-46, user will run and
report back.

**Follow-up (same session, user's own idea): `notebooks/48_wu2003_pooled_multiproposal_classification.ipynb`
(Option 6, new)** — "use all the simulation that has been successfully run so far" —
pools every currently-valid (post-physics-bug-fix, post-Stage-1b-warm-start-fix,
S-C.2-compatible) real simulation already on disk across every prior tried this
project, into ONE training run, rather than committing to any single prior. Verified
directly against the installed `sbi==0.26.1` source (not assumed) that this is
mathematically valid, not a hack: `append_simulations(theta, x, proposal=P)` tags each
batch with its own originating distribution; calling it once per historical bank
(each with *that bank's own exact prior* as `proposal=`), then a **single** final
`.train()` call, trains one flow on the union via SNPE-C's atomic-loss proposal
correction — and a single `.train()` call also sidesteps the early-stopping bug found
in the multi-round pilot (that bug only bit *repeated* `.train()` calls).
**Audited what's actually poolable before writing the notebook** (important, not just
assumed): found `wu2003_sbi_train_sc2_lognormal_true_n1000_stage1bfix.npz` and
`wu2003_sbi_train_sc_lognormal_true_n1000_stage1bfix.npz` are **the same 1000 theta
rows** (confirmed via direct array comparison — one is the S-C.2-augmented version of
the other) — only the S-C.2 version is pooled, avoiding double-counting. Excluded
every other bank in `data/` for lacking the `_stage1bfix` suffix (stale per this
project's standing rule) and all S-B-structure banks (wrong observation dimensionality).
**Gives ~2240 pooled training rows today** (2×1000 training banks + 2×120 SBC "screen"
sets folded in as free extra data; the 2×400 "confirm" sets are reserved, NOT pooled,
so an honest held-out SBC check survives per source prior) — modest, but the registry
is designed to be extended once notebooks 44/47 produce more banks (commented-out
example included). Deliberately excludes notebook 47's `RestrictedPrior`-drawn data for
now (pooling it correctly needs the saved classifier object too — flagged as a natural
follow-up, not attempted here). **Pre-handoff validation**: ran the exact pooling
mechanism for real (not synthetic) against the actual on-disk banks — 4 sources
appended with distinct proposals, `SNPE-C` correctly triggered atomic loss (confirmed
via the expected "proposal is neither the prior nor a NeuralPosterior" warning),
trained cleanly in 28s (94 epochs), produced sensible posterior samples. **Also not yet
run for real classification/SBC results** — user's task, report back when done.

## Warm-start fix, closed out (2026-09-03/04) — do not re-diagnose

`docs/Manuscript/system_ii_sbi_solidification_plan.md`'s Stage 1b audit
(`scripts/stage1b_warmstart_convergence_audit.py`,
`results/44_sc_warmstart_convergence_audit.csv`, n=45) found
`_matched_y0`'s old fixed-horizon steady-state solve
(`t_final=100, rtol=1e-4, atol=1e-6` in `scripts/sbi_pipeline.py`) did
**not** actually converge: 100% of 45 tested draws showed a downstream
summary-vector shift of 1-66% relative to a tightly-resolved re-solve, 1-2
orders of magnitude above the 0.3% sensor-noise floor, not confined to
prior tails. Root cause confirmed directly: `diffrax.diffeqsolve(...,
throw=False)` returns a plausible-looking but non-converged state on
`max_steps` exhaustion (does not return NaN), and the old code discarded
that failure signal.

**Fix (in `scripts/sbi_pipeline.py`, `_matched_y0`/
`_scenario_steady_state_checked`) went through two iterations**:
- v1 (tighter tolerance + RHS-residual-norm acceptance check) fixed the
  runaway-cost bug (no more 1-4h hangs) but was insufficient on its own —
  two properly-converged states can still show 3-32% downstream summary
  shift, since raw residual norm doesn't detect a slowly-relaxing
  near-null direction (e.g. an integral-controller state).
- v2 (current) replaced the acceptance criterion with a direct check that
  the downstream **summary** stabilizes across successive chunked
  integration extensions, using the **robust feature subset** (means/
  physics features — excludes numerically-unstable slope/std/corr features
  on near-flat signals, same fix `identifiability_scan.py` already needed
  elsewhere). **Fully validated**: 43/45 (95.6%) draws converge (up from
  36% under v1), every draw bounded to 2-30s wall-clock (no runaway cases),
  and a final check against an independent high-budget reference shows
  robust-feature shift of only 0.0002-0.02% — 1-2 orders of magnitude
  *below* the noise floor.

Every S-C/S-C.2 bank referenced in the priority-0 item above (suffix
`_stage1bfix`) was generated under this fixed protocol — the bug is not a
live confound for any current finding. Still open, low-priority: whether any
already-*reported manuscript* number depends on a pre-fix matched-protocol
bank (not checked this session — System I numbers predate this bug area and
are believed unaffected, but this hasn't been explicitly audited).

1. **System II matched ongoing-degradation calibration failure diagnosis** — the
  full-budget confirmation has now been run and **failed** (`n=15{,}000`, 8 seeds,
  S-B, matched per-draw steady-state warm starts; see `pending_manuscript_fixes.md`,
  Item 11 update dated 2026-08-27). No seed passed the min-KS SBC criterion; the
  least-bad seed (3) failed independent `N_SBC=800` confirmation with min KS
  $p=9.03\times10^{-8}$. Do **not** promote
  `sbi-logs/wu2003_posterior_sb_matched_full.pkl` or update System II manuscript
  results as calibrated. A first diagnostic notebook has now been created and run:
  `notebooks/35_wu2003_sbc_failure_diagnostic.ipynb` (executed copy beside it). It
  identifies `V_norm*` and `Q_j*` summaries as the dominant heavy-tail/outlier features,
  finds nonzero outside-prior posterior mass for every saved seed, and shows seed 3's
  shared rank probe remains high-biased across several parameters. Next step: test
  reduced-budget preprocessing/support fixes (robust/log transforms or clipping,
  alternative summary scaling such as `z_score_x='none'`, bounded/support-aware
  parameterisation) before any new full ensemble or downstream cascade.
  Follow-up notebook `notebooks/36_wu2003_strict_rejection_sbc_size_sweep.ipynb` has
  also been created and run under `caffeinate`: it trains seed 3 on 5k/10k/15k matched
  bank prefixes and evaluates canary SBC with `reject_outside_prior=True`. Strict
  rejection is computationally viable in the canary but not a standalone fix: min-KS
  values were 0.298, 0.019, and 0.070 respectively at `N_SBC=20`, `N_POST=50`.
  Follow-up notebook `notebooks/37_wu2003_lognormal_prior_sbc.ipynb` tested the user's
  prior hypothesis using a truncated lognormal prior centered at nominal operation and a
  5k importance-resampled matched bank. It also failed seed stability: 0/4 seeds passed
  shared `N_SBC=120`, `N_POST=120` SBC, with min-KS values between $3.38\times10^{-11}$
  and $1.50\times10^{-7}$. Strict rejection was fast, so the next serious experiment is
  not more rejection logic alone; it should combine a true simulator-generated prior bank
  with summary preprocessing/support-aware parameterisation.
  Section 8 of the same notebook now adds the true simulator-generated lognormal-bank
  follow-up (`n=1,000`). It completed, but still gave 0/4 calibrated seeds at shared
  `N_SBC=120`, `N_POST=120`: min-KS values were $1.10\times10^{-4}$, $5.30\times10^{-4}$,
  $2.21\times10^{-3}$, and $3.15\times10^{-6}$. This is better than the resampled-bank
  catastrophe but still not stable calibration; remaining failures concentrate around
  `xi_reb`/`z_A0_eff`, so prioritize summary preprocessing and support-aware
  parameterisation next.
  New standalone notebook `notebooks/38_wu2003_sbc_repair_experiments.ipynb` now holds
  the next planned experiments rather than adding more exploratory code to nb37:
  preprocessing sweep, fixed-y0 warm-start ablation, and a logit-parameter-space training
  scaffold. The user plans to run it manually.
  **2026-08-27 update:** wrote and executed `docs/Manuscript/system_ii_sbc_repair_directions.md`
  (a menu of ranked next steps off nb38) and `notebooks/39_wu2003_identifiability_and_sbc_repair.ipynb`
  (executed copy alongside it). nb39 Part I re-checked identifiability for `xi_reb`/`z_A0_eff`
  using the same FIM + grid-scan + raw-trajectory methodology that previously retracted the
  `(alpha, eta_col)` banana and partially retracted `(alpha, beta_r)` (`nb29b`) — deliberately
  cautious given that history. Result: `(xi_reb, z_A0_eff)` is **not** confounded (FIM
  off-diagonal ~0.0-0.11 across representations); `(alpha, z_A0_eff)` away from nominal **is**
  a real, representation-robust coupling (off-diagonal 0.39-0.68, survives raw trajectory) —
  the first identifiability finding in this investigation to clear the project's own bar,
  though still provisional (one operating point, 3 noise seeds); `(eta_col, z_A0_eff)` at
  nominal shows a large off-diagonal too but nominal `eta_col=1.0` sits exactly on that
  parameter's prior upper bound, so it may be a boundary artifact — needs re-testing at an
  interior point before it's trustworthy. nb39 Part II recombined nb38's two partial SBC
  fixes (fixed-y0 warm start + an analytically exact induced logit-space prior, replacing
  nb38's Gaussian surrogate) — **no improvement**: all 4 arms × 3 seeds stayed below
  `p=0.05`, and the predicted-best "combined" arm was in fact the worst. Do not use any
  `sbi-logs/wu2003_posterior_39_*.pkl` for manuscript numbers. Next step is in
  `system_ii_sbc_repair_directions.md`'s revised recommendation: target the
  `(alpha, z_A0_eff)` coupling specifically rather than another global training/geometry fix.
  **2026-08-27/09-01 follow-up (`notebooks/40_wu2003_alpha_zA0_confirmation.ipynb`, executed):**
  built to (1) visually confirm `(alpha, z_A0_eff)` in an actual trained posterior and (2)
  replicate the FIM finding at more operating points. This run **found a methodological bug**
  in nb39/40's FIM reporting: both used `fim_utils.offdiag_ratio`'s raw, normalised
  *information*-matrix entry as if it were a posterior *correlation* — for a multi-parameter
  FIM these can have the opposite sign (confirmed: raw off-diag `-0.90` vs. actual posterior
  correlation `+0.93` at one point). Fixed at the root by adding
  `fim_utils.cov_offdiag_corr` (the correlation from the *inverted* FIM/covariance, the
  quantity actually comparable to a posterior) and updating both notebooks to use it; both
  re-executed cleanly. **Corrected picture:** `(xi_reb, z_A0_eff)` still ruled out (unaffected
  by the bug). `(alpha, z_A0_eff)` is now confirmed by two independent methods everywhere
  `alpha` is meaningfully displaced from nominal (corrected FIM and real trained-posterior
  correlation agree in sign and ordering across `nb40`'s P2/P3/P4 test points) — the strongest
  cross-validated identifiability finding so far. `(eta_col, z_A0_eff)` at nominal is now
  *also* confirmed to survive the representation check under the corrected metric (`+0.76 →
  +0.52` compute_summaries → raw trajectory) but the pre-existing boundary caveat
  (`eta_col=1.0` sits exactly on `ETA_COL_HIGH`) is still unresolved. **One new, narrower
  discrepancy remains**: at `alpha` exactly at nominal (nb39's degraded point, nb40's `P1`),
  the corrected FIM says strongly negative (`-0.6` to `-0.7`) while the real posterior says
  weakly positive (`+0.26`–`+0.28`) — unresolved, recommended tiebreaker is a deterministic
  `identifiability_scan.run_scan` grid-scan at that exact point (distance-based, not subject
  to this FIM-inversion issue). Do not treat `(alpha, z_A0_eff)` as fully characterised near
  `alpha=nominal` until that's run.
  **2026-09-01 tiebreakers run (`notebooks/41_wu2003_confound_tiebreakers.ipynb`, executed)
  — both open questions above are now resolved:**
  (1) **`alpha=nominal` sign discrepancy resolved 2-against-1, against the posterior.** The
  deterministic grid-scan ridge slope is negative at both test points (`-2.06`, `-2.30`),
  matching the corrected FIM (`-0.6` to `-0.7`), not the uncalibrated posterior's weak
  `+0.26`/`+0.28`. The ridge visibly bends into a different, flatter segment once `alpha`
  drops below ~0.65-0.7 — independently reproducing nb40's sign-flip finding with a method
  that depends on neither a linearisation nor a trained network. **`(alpha, z_A0_eff)` is a
  genuinely curved, sign-changing degenerate manifold** — negative near `alpha=1.0`, positive
  once `alpha` is displaced, confirmed on both sides by two independent methods. A
  straight-line/rotation reparameterisation fix will not work for this pair; its ridge
  direction itself changes across the space it needs to cover.
  (2) **`eta_col` boundary re-test: not an artifact.** At an interior point (`eta_col=0.85`),
  the coupling with `z_A0_eff` is if anything *stronger* than at the boundary (`+0.62` to
  `+0.86` compute_summaries, `+0.66`–`+0.76` raw trajectory) and the grid-scan shows a real,
  compact low-distance region at truth, not a boundary-hugging band. **`(eta_col, z_A0_eff)`
  is confirmed as a second, genuine, representation-robust limit**, and — unlike
  `(alpha, z_A0_eff)` — appears single-signed (consistently positive) everywhere tested so
  far.
  **Both open items are now closed. The identifiability picture for System II is: `xi_reb`
  fully cleared; `z_A0_eff` carries two independent, confirmed, non-training-fixable
  couplings — a curved/sign-changing one with `alpha` and a simpler positive one with
  `eta_col`.** Combined with nb38/39 Part II's exhaustive (and unsuccessful) sweep of
  training/geometry fixes, this closes out the "is it fixable by better training" question:
  it is not, for these two pairs. The way-forward decision is no longer blocked on missing
  evidence — see `system_ii_sbc_repair_directions.md`'s final recommendation.
  **2026-09-01, `notebooks/42_wu2003_manuscript_fim_reaudit.ipynb` (executed) — the
  manuscript's own Artefact 2/3 claims use the same buggy metric, and Artefact 3's
  conclusion is wrong, not just its numbers.** `main.tex`'s §Artefact 3 (`sec:wu_artifact3`)
  currently claims the `(alpha, z_A0_eff)` coupling at scenario W2 "collapses" under raw
  trajectory (off-diag `-0.89→-0.07`, condition number `4.36e3→5.48`) and concludes this is
  "representation-induced rather than a plant-level non-identifiability" — but that raw-
  trajectory number was necessarily computed with the same uninverted-FIM metric found
  buggy above (the exact cited figures could not be traced to any tracked notebook/script,
  but the qualitative "collapses to near-zero" claim does not reproduce). Recomputed at W2
  with the corrected `cov_offdiag_corr` and the standard 9-channel raw-trajectory
  methodology: correlation stays large (`+0.57` to `+0.79`) at raw trajectory, barely
  smaller than at 66-D — a 5th independent confirmation this is genuine, not fixable.
  **`main.tex`'s Artefact 3 section needs to be rewritten**, not just re-numbered — do not
  cite the current "-0.07"/"representation-induced" language for `(alpha, z_A0_eff)` in any
  future manuscript pass. Good news found in the same run: `(beta_r, z_A0_eff)` at W5 *does*
  substantially collapse (`-0.32→-0.05`), consistent with the manuscript's claim — `beta_r`
  does not appear to share `alpha`'s composition-confound, physically sensible since it has
  no direct stoichiometric link to feed composition. **Open, higher-stakes question raised
  by the same check**: Artefact 2 (`(alpha, beta_r)` at W11, currently reported as fully
  resolved/benign, reactor F1=0.948) also did not reproduce its own headline collapse using
  this notebook's methodology at that exact point (`+0.05` to `+0.28`, far below the
  `+0.901`/`+0.998` figures cited elsewhere for this pair) — not yet evidence Artefact 2 is
  wrong (different points legitimately differ), but it has **not** received the same
  dedicated multi-point/grid-scan/posterior re-validation Artefact 3 just got, and should
  before being treated as settled. This is now the single highest-priority open item.
2. **System II, Items 12–13** (cross-regime transfer, full downstream cascade) —
  still not started and now blocked on fixing/understanding the matched-posterior
  calibration failure above.
3. **Supporting Information completion** (`SI_completion_plan.md`) — System I's
   numbers are now settled (see above), but System II's are not (Item 11 above).
   The plan's own header says not to proceed past its Step 6 (final recompile)
   until *both* systems' Stage 3 numbers settle — re-read and update its "⚠️
   Update" block before starting, since System I's portion of that blocker is now
   lifted but System II's is not.
4. **Stage 7 (manuscript wording pass)** — not done. Major Comment 10's
   four-category artifact taxonomy (`reviewer_response_plan.md`, search "Major
   Comment 10") has not been formally inserted into `main.tex` itself yet.

## Repo-integrity note: `cstr_sbi/__init__.py` had a dead import, now fixed (2026-09-04)

Discovered while running post-Stage-1b-fix training scripts: `import cstr_sbi` (or any
`from cstr_sbi.recycle... import ...`, which always initializes the parent package first)
failed in a **fresh interpreter** with `ImportError: cannot import name 'physics' from
partially initialized module 'cstr_sbi'`. Root cause: `src/cstr_sbi/__init__.py` line 17
did `from cstr_sbi import physics` (the old top-level "propylene oxide system" module), but
`src/cstr_sbi/physics.py` (along with `plotting.py`, `simulator.py`) is **already deleted**
in the working tree — part of a large, pre-existing, uncommitted legacy-code cleanup (`git
status` shows most of `legacy_code/`, old notebooks `00-03`/`13`, and their data/figures
also deleted, clearly intentional and from a prior session, not something this session did)
that migrated the project onto the `luyben`/`recycle` subpackages but never updated this one
`__init__.py` line to match. **Fixed** by dropping the dead `physics` import (and its
`__all__` entry) — `luyben`/`recycle` imports are untouched, both still exist and work.
Verified: `import cstr_sbi` and `from cstr_sbi.recycle.physics import ...` both succeed now
in a fresh interpreter. Working theory for why this went unnoticed all session despite dozens
of successful script runs: a single very-long-running background process (the ~23h S-C bank
regeneration) started *before* whatever deleted these files and kept `sys.modules['cstr_sbi']`
cached from its own process start, so it never needed to re-resolve the import; only a
*freshly started* Python process after that point would hit the break — plausible but not
independently confirmed, since the exact deletion time isn't recoverable from `git status`
alone. Nothing else in this session touched the legacy-cleanup deletions themselves; only the
one dead `__init__.py` line was changed.

## One thing to know about `main.tex` specifically

The user has been directly editing `main.tex` in parallel with sessions' work
(most recently § Structural Identifiability Analysis, the abstract, and the
Conclusions) — tightening language beyond what was auto-generated here. When
these two edit streams collided once already, it left a duplicated/orphaned
`\begin{table}...\end{table}` block that failed to compile; this was fixed.
**Always check `pdflatex -draftmode` compiles cleanly before assuming the file is in
a good state**, since it's being edited from two directions.
