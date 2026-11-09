**STATUS (2026-11-08, NEWEST, nb26 (`notebooks/system_I/26_cheap_correlation_prediction.ipynb`,
`scripts/build_nb_system_I_26.py`) -- cheap (4.5 min wall-clock, no SBI/no NUTS)
FIM-based probe finds the (alpha,beta) correlation COLLAPSES to ~0 at the corrected
(~47%-conversion) operating point under the standard noisy controller, but the
mechanism is GENUINELY MIXED, not a single clean story -- read before nb24/nb25
below.)**

**Task**: given nb25's confirmed K0_NOMINAL 60x unit bug (below), check cheaply
(steady-state root-finds + finite differences + small EM-replicate FIM batches only)
whether the long-running Corr(alpha,beta)~-0.90 finding (nb19/nb20, Sc4
alpha=beta=0.85) is tied to the buggy ~98%-conversion regime, and whether its
mechanism is controller-noise-driven, physics-structural, or a specific-channel
(C/Qc) noise cross-term, per the user's own hypothesis. Did NOT touch nb19-25,
main.tex, the SI, article_outline_CChE.md, K0_NOMINAL, physics.py, or simulator.py --
all new code (a noise-free-controller EM-scan variant, generic k0/ctrl-parameterised
raw-trajectory/summary wrappers, a full-covariance-FIM helper copied from nb20
Theory E) is self-contained in `scripts/build_nb_system_I_26.py`.

**Step 0 (minimal Tsp workaround, explicitly NOT a production re-derivation)**:
scanning Tsp at `k0_corrected = K0_NOMINAL/60` for the Tsp that maximises the
worst-case valve margin across the healthy (1,1) and Sc4-equivalent (0.85,0.85)
points gives `Tsp_corrected = 300.30 K` (vs. the buggy `Tsp=312.5 K`) -- but the best
achievable worst-case margin is only **9.54 L/min** out of the [0,400] L/min valve
range (at the Sc4-equivalent point; the healthy point has a much larger 113 L/min
margin). This is non-saturated but NOT "comfortably centred" -- the healthy and
Sc4-equivalent Qc_ss(Tsp) curves at this k0 do not overlap well, so no single Tsp
gives both points generous headroom simultaneously. Flagged honestly, not glossed
over; a real operating-point re-derivation would need more than a Tsp tweak (e.g.
feed conditions or controller re-tuning) to do better.

**Step 1 (steady-state sensitivities, central FD on `closed_loop_steady_state_exact`)**:
confirms `dT_ss/dalpha = dT_ss/dbeta = 0` exactly at all 4 (op point x theta point)
combinations (non-saturated branch, integral control pinning T=Tsp, as expected --
none of the 4 combinations fell into the saturated branch). `C_ss`, `Tc_ss`, `I_ss`
all move non-trivially under both alpha and beta at every combination; alpha's
`dC_ss/dalpha` sensitivity grows from -0.017 (buggy, healthy) to -0.164 (corrected,
healthy) -- roughly a 9-10x increase, consistent with nb25's own ~14-16x
alpha-sensitivity-gain finding via a different route.

**Step 2 (full-covariance FIM-implied Corr(alpha,beta), standard noisy controller,
120 replicates, `cov_offdiag_corr` on the inverted FIM -- this is the number
comparable to a real trained posterior's correlation, per `fim_utils`'s own
docstring)**: 
| op point | Sc4-equiv (0.85,0.85) | healthy (1,1) | asymmetric (0.85,1.0) |
|---|---|---|---|
| buggy | **-0.901** | -0.195 | -0.303 |
| corrected | **-0.029** | +0.004 | -0.069 |

The buggy-point Sc4-equivalent number (-0.901) closely reproduces nb19/nb20's
independently-trained-SBI-posterior finding (~-0.91 to -0.48 depending on seed) --
good cross-validation of this cheap FIM proxy. **At the corrected operating point,
the correlation collapses to within noise of zero at all three theta points tested.**
This directly answers the "is it tied to the buggy regime" question: under the
standard (noisy) controller, YES -- the large-magnitude correlation does not survive
the operating-point correction.

**Step 3 (noise-free-controller variant -- a new shadow/real dual-trajectory EM scan
where Qc is decided by a noise-free shadow trajectory, never by the noisy real one;
sensor noise unchanged, applied post-hoc as before)**: this is where the story gets
genuinely mixed, not clean. Removing process noise from the controller's feedback
does NOT just "shrink toward zero" or "leave unchanged" -- it flips the correlation to
a LARGE, UNIFORMLY POSITIVE value (+0.82 to +0.97) at **every** point tested,
including the corrected operating point where the noisy-controller number was ~0:
| op point | Sc4-equiv | healthy | asymmetric |
|---|---|---|---|
| buggy (noisefree) | +0.918 | +0.815 | +0.973 |
| corrected (noisefree) | +0.955 | +0.885 | +0.932 |

This proves a real, strong, POSITIVE physics/steady-state-structural coupling between
alpha and beta exists at BOTH operating points (consistent with Step 1's sensitivity
table: C_ss/Tc_ss/I_ss moving together under alpha and beta) -- i.e. the correlation
is NOT purely a noise artifact; a genuine structural ridge is there throughout. But it
also means the noise-free test does NOT simply "confirm the standard-controller sign
and magnitude survive" -- the sign ITSELF only matches between noisy/noisefree at no
point tested; the noisy-controller numbers are either near-zero or negative while the
noisefree numbers are uniformly strongly positive.

**Step 4 (channel ablation on the FIM, Sc4-equivalent only, standard noisy
controller, reusing Step 2's replicates -- no resimulation)**:
| op point | drop C | drop T | drop Tc | drop Qc | (full-channel baseline) |
|---|---|---|---|---|---|
| buggy | **-0.174** | -0.936 | -0.934 | **-0.254** | -0.901 |
| corrected | +0.307 | +0.003 | -0.076 | -0.166 | -0.029 |

At the **buggy** operating point this is a clean, direct confirmation of the user's
own C/Qc-noise-cross-term hypothesis: dropping C or Qc collapses the correlation from
-0.90 toward -0.17/-0.25, while dropping T or Tc barely moves it (if anything,
slightly larger magnitude, -0.93/-0.93). At the **corrected** operating point, the
baseline is already near zero, so "collapse" doesn't mean much; the most notable
change is dropping C, which flips the sign to a moderate +0.307 -- not a clean analog
of the buggy-point pattern.

**Honest, non-cherry-picked verdict (both of the user's two questions)**:
1. **Is the correlation tied to the buggy regime?** Under the standard (noisy)
   controller -- the representation actually comparable to a real trained SBI
   posterior -- YES: -0.901/-0.195/-0.303 at the buggy point collapse to
   -0.029/+0.004/-0.069 at the corrected point. The K0_NOMINAL unit bug is doing real
   work in producing the large-magnitude correlation nb19/nb20 reported.
2. **Mechanism -- controller-noise, physics-structural, or channel cross-term?**
   GENUINELY MIXED, not reducible to one cause: a robust, strong, POSITIVE
   physics-structural ridge between alpha and beta exists at BOTH operating points
   (Step 3, Step 1) -- so the correlation is not "purely" a noise artifact. But a
   noise-mediated effect specifically tied to the C and Qc channels (Step 4) is large
   enough, specifically at the buggy operating point, to not just cancel but REVERSE
   the sign of that structural ridge (net -0.90 vs. the underlying noise-free +0.92),
   while at the corrected operating point the two effects roughly cancel to near zero
   (net -0.03 vs. underlying noise-free +0.95) instead of either dominating cleanly.
   **All three of the candidate mechanisms the user asked about (controller-noise,
   physics-structural, channel cross-term) have real supporting evidence here
   simultaneously -- this is the honest answer, not a cleaner single-cause story.**

**Caveats/limits of this notebook, stated explicitly**: this is a cheap
Laplace/FIM-approximation proxy for the correlation (`cov_offdiag_corr`), not a real
trained SBI posterior at the corrected operating point -- the near-zero corrected-point
numbers should be spot-checked against an actual SNPE retrain before being treated as
final (not done here, out of scope/budget for this notebook). The diag-vs-full-cov
FIM numbers sometimes disagree substantially (e.g. corrected/asymmetric: full-cov
-0.069 vs. diag +0.533, see `results/26_extra_points.csv`), underscoring that the
feature-covariance's off-diagonal structure (not just per-feature variance) matters a
lot here -- consistent with, and further supporting, the channel-cross-term finding.
Results: `results/26_summary_table.csv`, `results/26_extra_points.csv`,
`results/26_step0_operating_points.csv`, `results/26_step1_sensitivities.csv`,
`figures/26_step0_tsp_scan.png`, `figures/26_summary_grid.png`. Total notebook
wall-clock: 271.5s (~4.5 min). No code/manuscript changes made.

---

**STATUS (2026-11-08, nb24 (`notebooks/system_I/24_cnn_vs_handcrafted_discrepancy.ipynb` --
not yet rendered as a notebook; `scripts/build_nb_system_I_24.py` is the authoritative,
finalized code -- STOPPED MID-EXECUTION BY EXPLICIT USER DECISION once nb25 (immediately
below) found the K0 unit bug that will invalidate every number below anyway. H1/H2
findings stand as reported; H3/H4 never completed. Read the nb25 entry below this one
too -- it supersedes the *operating point* this whole investigation ran at, though not
nb24's specific CNN-architecture finding.):**

**Task**: investigate WHY the hand-crafted-summary (HC) and raw-trajectory-CNN
(Theory F) SBI posteriors disagree so persistently on Corr(alpha,beta) at n=50,000
(nb22's optimal budget) -- see `finding_system1_no_alpha_beta_banana.md`'s 2026-11-08
three-way-cross-check entry for the full prior context (HC and CNN disagree in sign
at 4/6 closed-loop scenarios, sharpening rather than resolving after the warm-start
fix).

**H1 (CNN-side SBC + anti-shrinkage slope at n=50,000, COMPLETE, reused the 3 already-
cached CNN posteriors, no retraining)**: the CNN-embedding posterior has essentially
collapsed on alpha -- `slope_alpha` ~= 0.046-0.053 across all 3 seeds (vs. the
hand-crafted summary's ~0.99 at the same budget per nb22's own table), while
`slope_beta` is much healthier (~0.73-0.93). SBC: alpha nominally "passes" (p=0.36-0.81,
consistent with calibrated-but-uninformative rather than miscalibrated), beta FAILS
outright (p=0.000) at 2/3 seeds. Full numbers: `results/24_cnn_sbc_metrics.csv`,
`results/24_cnn_grid_diagnostics.csv`, `figures/24_cnn_sbc_rank_histograms.png`.

**H2 (held-out predictive log-density, HC vs CNN, same theta/noise draws, COMPLETE)**:
hand-crafted mean held-out log p(theta|x) ~= +7.3 (consistent across all 5 HC seeds,
std~0.01); CNN-raw ~= +2.0-2.2 (consistent across all 3 CNN seeds, std~0.13) -- a
clear, reproducible ~5-nat gap in favor of the hand-crafted summary. Full numbers:
`results/24_heldout_logprob_comparison.csv`, `figures/24_heldout_logprob_comparison.png`.

**H1+H2 together support an architecture/capacity explanation, not a pure-
representation-reveals-different-truth explanation, for the HC-vs-CNN sign
disagreement**: the specific CNN architecture reused unchanged from
`04b_embedding_net_study.ipynb`/nb20's Theory F (2 conv layers [16,32], 2 linear
layers x64, output_dim=20) appears to extract substantially less usable information
about alpha specifically from the raw trajectory than the hand-crafted 29-D summary
extracts from the same underlying signal -- so whatever Corr(alpha,beta) the CNN
reports is a more likely artifact of that information deficit than a competing,
equally-trustworthy estimate of the true posterior geometry. **This was the leading
hypothesis at the point execution was stopped, not a final, SBC-blessed conclusion**
(H3/H4 below were meant to corroborate it further and did not complete).

**IMPORTANT CAVEAT, added after nb25's finding (read nb25's entry below) -- these
H1/H2 numbers were computed at the CURRENT, now-confirmed-buggy ~98%-conversion
operating point** (`K0_NOMINAL` 60x too large per nb25). nb25 independently found
alpha-sensitivity (`dX/dln(alpha)`) is ~14-16x larger at the corrected ~47%-conversion
point. This does NOT contradict H1/H2's finding that the CNN is *specifically* worse
than the hand-crafted summary at extracting alpha (the hand-crafted summary already
gets real alpha signal at the current, low-sensitivity operating point, slope~0.99;
the CNN's near-total failure at the very same point, same data, is still a distinct,
real architectural deficiency) -- but it means these exact numbers (slope values,
the ~5-nat gap) must be RE-VERIFIED once the K0 fix lands, since a 14-16x change in
available alpha signal could plausibly change the CNN's own slope/calibration
picture substantially, not just rescale it.

**H3 (5-seed CNN robustness check, INCOMPLETE -- stopped by user decision, not a
silent failure)**: attempted 3 times to train 2 additional CNN seeds (seed3, seed4)
at n=50,000 to bring the CNN side to 5-seed parity with HC's own nb22 budget study.
**Hit a reproducible OS-level crash twice**: the kernel SIGKILLed the training
process both times (`caught allocating too many mach ports, Num of ports allocated
~267,625` -- a genuine mach-port-exhaustion bug in the long per-sample
`simulate_for_sbi`/`simulation_wrapper_sbi_raw` loop in `src/cstr_sbi/inference.py`,
shared production code, NOT modified by this investigation), once at 84% through
seed3's training and once at 89% -- both fresh single-process runs, ruling out
cross-seed port accumulation as the cause. **Puzzle left open**: the 3 already-cached
CNN seeds (0,1,2, from nb22's own `nb22_cnn_theoryF_recheck.py`) completed this exact
code path successfully in an earlier session, so the crash is not 100%-reproducible
every time -- root cause (why the leak rate/GC timing varies run to run) was not
diagnosed; a fix (e.g. periodic `jax.clear_caches()`, or isolating each sample sim in
a subprocess) was not attempted, since it would require modifying shared
`inference.py` code used by nb19-23, out of this investigation's scope. **User
stopped a 3rd attempt mid-launch (before any crash/success signal) once nb25's K0 bug
finding made continuing pointless** -- this is a deliberate pause, not abandonment.
The 3-seed CNN evidence already on record (`results/22_theoryF_cnn_recheck.csv`)
remains the only seed-robustness evidence for the CNN side.

**H4 (qualitative feature-space comparison): NEVER RUN for real** -- only smoke-tested
at reduced size (n_noise=3 instead of 20) during script development, which gave
discriminability (between/within variance ratio across 3 theta points) hand-crafted=408
vs CNN=168, consistent with H1/H2's direction but far too small a sample to report as
a finding.

**Code status**: `scripts/build_nb_system_I_24.py` is COMPLETE and FINALIZED as
written -- syntax-valid, all 4 hypotheses (`run_h1`-`run_h4`) implemented, resume
logic in `main()` (skips H1/H2 recomputation if their CSVs already exist on disk,
goes straight to H3) so a future re-run won't repeat the ~10 min of already-valid
work. **Deliberately left unexecuted to completion** -- do not treat this as a stalled
or broken build; it is paused by explicit user decision pending the K0 operating-point
fix (nb25), after which the whole script should be re-run from scratch (all cached
results it depends on, `results/22_budget50000_seed{0-4}.pkl` and
`results/22_theoryF_cnn_posterior_seed{0,1,2}_n50000.pkl`, will themselves need
retraining at the corrected operating point first). The notebook file
`24_cnn_vs_handcrafted_discrepancy.ipynb` itself was never rendered (per this
project's own house rule of using a standalone script for anything long-running,
same as nb20/21/22/23) -- the script IS the deliverable.

---

**STATUS (2026-11-08, NEWEST OF ALL -- nb25 (`notebooks/system_I/25_k0_units_verification.ipynb`,
`scripts/build_nb_system_I_25.py`) CONFIRMS a genuine 60x unit bug in `K0_NOMINAL`
(`src/cstr_sbi/physics.py`), NOT YET FIXED -- read this before anything else,
including the nb23 entry below.**

An external literature-review document claimed `K0_NOMINAL = 16.96e12` (commented
`# 1/min`) is actually a per-**hour** literature value (Fogler/Furusawa 1969
propylene-oxide hydrolysis kinetics), making the coded constant 60x too large.
**Verified, independently, three ways:**
1. Three separately-phrased web searches each cite the pre-exponential factor
   16.96e12 (equivalently `4.71e9 s^-1`) explicitly in **hr^-1**, never min^-1 --
   including COMSOL's own documented reproduction of this exact Fogler 3rd ed.
   Example 9-4 model, and a second independent citation whose activation energy
   (75362.4 J/mol) matches this codebase's `E_A=75362.0 J/mol` to 6 significant
   figures (effectively exact).
2. Internal consistency, using only this codebase's own `E_A`/Arrhenius equation
   (no external citation trusted): the as-coded value gives a steady-state
   conversion X~98.2%, matching `main.tex`'s own stated "~98%" operating point
   almost exactly; the candidate-corrected value (`K0_NOMINAL/60`) gives X~47.2%,
   matching the external document's claim about Fogler's *intended* operating
   point almost exactly. Two independent external claims both reproduced from this
   project's own numbers.
3. Direct evaluation of `physics.py`'s actual closed-loop steady-state solver
   (`closed_loop_steady_state_exact`, no training/retraining) confirms the
   alpha-sensitivity `dX/dln(alpha)` is **~14-16x larger** at the corrected
   operating point than at the current one -- matching the external document's
   `~14x` claim.

**Caveat found only by running the real closed-loop model (not in the external
document or the simple hand calc)**: naively dividing `K0_NOMINAL` by 60 in
isolation drives the closed-loop healthy operating point into permanent coolant
actuator saturation (`Qc=Qc_min=0`, `T` settling at ~301.7K instead of
`Tsp=312.5K`) -- the PI-controller tuning (`Tsp`, `Qc0`, ...) was evidently chosen
to work with the as-coded (buggy) `k0`, not Fogler's real value. **This is not a
one-line fix**: correcting the unit bug properly requires re-deriving a consistent
nominal operating point (setpoint/feed conditions/controller tuning) alongside the
kinetic constant, not just patching the constant. Not done in nb25 by design
(verification-only scope); no edits were made to `K0_NOMINAL`, `main.tex`, the SI,
or `article_outline_CChE.md`.

**Why this matters for the whole (alpha,beta) investigation**: if the nominal
operating point has genuinely been pinned at ~98% conversion (the flat/saturated
part of the Da/(1+Da) curve) this whole time, every representation tested in this
project (hand-crafted summary, CNN embedding, NUTS) has had ~14-16x less real
alpha signal available in the trajectory than it would at the intended ~47% point
-- a physics-level headwind that was never on this investigation's radar (nb19-24
all diagnosed the correlation/CNN-blindness question purely at the
representation/training level). Does NOT contradict nb24's CNN finding (the
hand-crafted summary still gets *some* alpha signal at the current point, slope
~0.99; the CNN's near-total failure at the same point is still a separate,
real deficiency) -- but predicts operating-point correction should help every
representation, not just explain away the CNN gap. **Next step, not started**:
decide with the user whether/how to re-derive the operating point (not a
unilateral change -- it invalidates every cached System I posterior, including
anything already cited in `main.tex`) before touching `K0_NOMINAL` for real.

---

**STATUS (2026-11-08, nb23 (`notebooks/system_I/23_nuts_differentiable_warmstart_fix.ipynb`,
`scripts/build_nb_system_I_23.py`) STOPPED MID-INVESTIGATION BY EXPLICIT USER
DECISION, not completed, not abandoned-as-failed -- the differentiability crash IS
fixed, but a NEW, unresolved, worse problem (catastrophic NUTS non-convergence) was
found on top of the fix and the investigation was halted before diagnosing it. Do
NOT build on this fix for any NUTS sampling claim without first reading this entry
in full.):**

**What this session was asked to do**: nb22's `warm_start_ic` root-find fix
(`closed_loop_steady_state_exact`, `src/cstr_sbi/physics.py`) made System I's
closed-loop warm start physically correct but used plain `float()`/
`scipy.optimize.brentq`, which broke NUTS's `matched_warm_start=True` path outright
(instant `ConcretizationTypeError` crash at Sc4/Sc5, confirmed in the entry below this
one). This session's job was to reimplement it JAX-natively so NUTS works again.

**Part 1, DONE and verified -- the differentiability crash itself is fixed.**
`closed_loop_steady_state_exact`'s saturated-branch root-find now uses a
fixed-iteration (`n_iter=60`) `jax.lax.fori_loop` bisection on `T in [250,450]` in
place of `scipy.optimize.brentq`, and the saturated-vs-non-saturated branch selection
uses `jnp.where` in place of the Python `if`/`float()`. No `float()` calls remain in
the function. Directly verified, not assumed:
- **Correctness preserved**: new implementation agrees with the old scipy
  implementation to ~1e-11 or better at all 6 closed-loop scenarios (Sc1-5,7); `|rhs|`
  at the new fixed point (4e-16 to 3.5e-13) is actually BETTER than the old
  implementation's own (1e-4 to 1e-7).
- **Differentiability fixed**: `jax.grad` succeeds (finite values) at a non-saturated
  point (Sc4) and two saturated points (Sc2, Sc5) -- no more `ConcretizationTypeError`.
- **End-to-end smoke test passed**: a cheap `run_mcmc_baseline(...,
  matched_warm_start=True)` call (1 chain, 20 samples/20 warmup) completed without
  crashing at both Sc4 (12.7s) and Sc5 (4.5s) -- the exact call that previously
  crashed instantly.

**Part 2, the NEW problem found, NOT fixed, NOT diagnosed -- catastrophic NUTS
non-convergence at full protocol.** Running the actual decisive-test protocol (Sc4,
4 chains, 500 samples, 300 warmup, full-covariance likelihood via `cov_obs`,
`matched_warm_start=True` -- the same settings `scripts/build_nb_system_I_20.py`
Section 5 Step 2 used) gave:

```
FULL PROTOCOL (full-cov likelihood): wall=3404.9s, corr=-0.114
r_hat alpha: 505.7   r_hat beta: 346.9
```

R-hat in the HUNDREDS -- far worse than anything else seen anywhere in this whole
investigation (even the old diagonal-likelihood misspecification bug, Theory E's
original finding, only hit R-hat~3-11; the pre-regression differentiable
scipy-free... no, pre-regression DIFFERENTIABLE version, i.e. before nb22's
root-find fix existed at all, got clean R-hat~1.0-1.01 under this identical
full-covariance protocol per nb20's Section 5 Step 2). Also ~57 minutes wall time
for one window, at the high end of this project's documented heavy-tailed NUTS cost
distribution.

**Leading hypothesis, NOT YET TESTED -- recorded for whoever picks this up next.**
The coordinator's working theory: this is a gradient-QUALITY problem, not a
gradient-CORRECTNESS problem. A fixed-iteration `jax.lax.fori_loop` bisection gives a
correct function VALUE, and a well-defined (finite) gradient when probed at a single
point -- both confirmed above -- but naively differentiating straight through the
discrete bisection's internal branching can give a gradient that is a poor local
approximation of the true implicit derivative, especially under HMC's demand for
smooth, accurate gradients across many leapfrog steps. The mathematically correct
fix would be **implicit differentiation** (e.g. `jax.custom_vjp` using the implicit
function theorem: d(root)/d(theta) = -(dF/dtheta)/(dF/dT) evaluated at the root --
cheap, exact, and avoids differentiating through the iteration at all).

**A second, untested, equally-plausible hypothesis this session did not get to rule
out**: the new implementation ALWAYS evaluates BOTH the saturated and non-saturated
candidate branches (needed so `jnp.where` can select between them), whereas the OLD
Python `if` only ever evaluated ONE branch. This is exactly the shape of the classic
JAX/autodiff footgun where an unselected branch's hidden singularity (e.g. a division
that is fine at the selected branch's inputs but blows up to `inf`/`nan` for inputs
the OTHER, unselected branch happens to be evaluated at during some leapfrog
excursion) poisons the backward pass of `jnp.where` via `0 * inf = nan`, even though
the forward VALUE is correct and even though a single-point `jax.grad` probe at the
nominal scenario point looks totally fine. A diagnostic script to test both
hypotheses (compare autodiff grad vs. finite-difference grad of the true value over a
grid; scan both branches' intermediate denominators for near-zero values over a wide
theta range; check NumPyro's divergence count on the actual run) was WRITTEN but
NEVER RUN -- `/private/tmp/.../scratchpad/diagnose_nb23.py` (session-local scratchpad,
not copied into the repo) -- the user explicitly stopped this investigation before it
could be executed. Whoever resumes this should re-derive/re-run that check (or an
equivalent) before assuming either hypothesis is the answer.

**Explicit status: STOPPED BY USER DECISION, not resolved, not abandoned as a dead
end.** The differentiability-crash fix in `src/cstr_sbi/physics.py` is being LEFT IN
PLACE (it is a real, independently-verified improvement for the original
`ConcretizationTypeError` crash and for every other call site, which only ever
evaluates it eagerly/non-differentiated and is unaffected by this NUTS-specific
sampling-quality question). **But it must NOT be treated as "NUTS is back online and
usable"** -- it is differentiable now, but produces NUTS output that is badly wrong
(R-hat~500) under the one protocol tested, which is worse than the crash it replaced
in terms of silently-trustable output (a crash is obviously broken; an R-hat~500 run
that still "completes" and prints a correlation number is a much easier trap to fall
into citing by accident). **Do not use any NUTS correlation number produced under
this fix until this is diagnosed and fixed.** No notebook/script in this entry's
scope was built or launched beyond the diagnosis described above --
`notebooks/system_I/23_nuts_differentiable_warmstart_fix.ipynb` does NOT exist (the
build script `scripts/build_nb_system_I_23.py` was written but never executed, per
this explicit stop instruction). Files: none in `results/23_*`/`figures/23_*` exist.

---

**STATUS (2026-11-08, NEWEST OF ALL — Theory F (CNN-raw-trajectory) recheck at the
corrected warm-start + optimal n=50,000 COMPLETE: the CNN-vs-hand-crafted sign flip
does NOT resolve under the fix -- at Sc4 (the priority scenario) the disagreement
SHARPENS, not narrows. A SEPARATE, unexpected finding surfaced along the way: the
warm-start fix itself flips the HAND-CRAFTED summary's own correlation sign at 2 of 6
scenarios (Sc2, Sc7) -- read this entry in full before citing any Theory F number as
settled. The NUTS-regression entry immediately below this one is still current and
unresolved; read it second.):**

**Three-way comparison, full table in `results/22_theoryF_three_way_comparison.csv`**
(hand-crafted=HC, pre-fix HC was whatever budget nb20's Theory F used per scenario,
pre-fix CNN capped at n=20,000 per nb20's own 7h41m-stall decision, post-fix HC/CNN
both at the corrected warm-start and the nb22-determined optimal n=50,000):

| scenario | HC pre-fix | CNN pre-fix (n20k) | HC post-fix (n50k) | CNN post-fix (n50k) | CNN vs HC post-fix agree? |
|---|---|---|---|---|---|
| Sc1 healthy | -0.650 | +0.777 | -0.719 | +0.785 | NO |
| Sc2 fouling | -0.856 | +0.920 | **+0.959** | +0.286 | yes (both +) |
| Sc3 decay | -0.638 | +0.740 | -0.799 | +0.748 | NO |
| **Sc4 combined (priority)** | **-0.893** | **+0.868** | **-0.910** | **+0.935** | **NO** |
| Sc5 saturated | +0.850 | +0.314 | +0.919 | +0.150 | yes (both +, very different magnitude) |
| Sc7 drift | +0.632 | +0.886 | **-0.907** | +0.948 | NO |

**Direct answer to the coordinator's question: no, the CNN-embedding posterior does
NOT come into agreement with the hand-crafted-summary posterior once both are trained
under the corrected warm-start at the optimal budget.** At Sc4 specifically, the sign
disagreement is UNCHANGED in direction and, if anything, SHARPER in magnitude on both
sides (HC: -0.893->-0.910; CNN: +0.868->+0.935 -- both methods became MORE confident in
their own, opposite, directions, not less). 4/6 scenarios disagree in sign both before
and after the fix (Sc1, Sc3, Sc4, Sc7); 2/6 agree in sign (Sc2, Sc5), but Sc2's
agreement is itself a NEW artifact of the fix (see below), not a pre-existing one, and
Sc5's magnitudes differ by ~6x even where signs agree. **This is NOT the "three
independent methods converge" result that would have been the strongest possible
confirmation of the (alpha,beta) correlation being real** -- the honest reading is that
the representation (hand-crafted summary statistics vs. raw-trajectory CNN embedding)
drives a large, scenario-dependent, now-fix-independent disagreement that Theory F's
original "compression artifact" hypothesis still cannot explain (CNN access to more
information does not converge the two methods, exactly as nb20 already found pre-fix).

**Unexpected, separate finding, not glossed over**: computing the HAND-CRAFTED
posterior's own Corr(alpha,beta) at all 6 scenarios under the corrected warm-start (using
nb22's already-trained n=50,000 cached posteriors, `results/22_budget50000_seed{0-4}.pkl`
-- cheap, sampling only, no retraining) found that the warm-start fix flips the
HAND-CRAFTED summary's own correlation SIGN at Sc2 (fouling: -0.856->+0.959) and Sc7
(drift: +0.632->-0.907) -- this is the hand-crafted-vs-hand-crafted, pre-fix-vs-post-fix
comparison, nothing to do with CNN. This is consistent with (and a direct, concrete
confirmation of) the original warm-start audit's finding that the OLD warm-start bug's
injected artifact size was strongly scenario-dependent (the 60-minute-window `Qc`
discrepancy was ~35x larger at Sc2 than at healthy/Sc4) -- Sc2 and Sc7 were evidently
among the scenarios where the old bug's artifact was large enough to flip the sign of
the inferred correlation outright, not just its magnitude. **Any manuscript-facing
claim about Sc2's or Sc7's (alpha,beta) correlation sign that predates this session's
warm-start fix should be treated as unreliable, independent of anything about Theory F
or CNN** -- this is a correction to the hand-crafted-summary result itself, found as a
side effect of building the three-way table, not a CNN-specific finding.

**Cost/process note**: 3 seeds at n=50,000 for the CNN-embedding training, self-
caffeinated background run, completed cleanly in ~95 min total (seed0: 1289s, seed1:
1877s, seed2: not individually logged but total run was well-behaved) with ZERO sign
of the old pre-fix pathological stall (7h41m with zero progress at n=40,000, documented
in `build_nb_system_I_20.py`) -- the warm-start fix's ~640x evaluation speedup appears
to have also resolved (or at least not reproduced) whatever caused that stall, though
the stall's root cause was never fully pinned down, so this is an empirical
observation, not a confirmed causal fix. Full results: `results/22_theoryF_cnn_recheck.csv`,
`results/22_handcrafted_n50k_all_scenarios.csv`, `results/22_theoryF_three_way_comparison.csv`.

---

**STATUS (2026-11-08, the `warm_start_ic` root-find fix BREAKS NUTS's
`matched_warm_start=True` path OUTRIGHT (crash, not slow) because the fix is
scipy-based and not JAX-differentiable; also, a CNN-raw-trajectory Theory F recheck at
the new optimal budget n=50,000 is running in the background. Read this first for the
NUTS regression; a separate entry below covers the CNN recheck result once it
completes.):**

**NUTS differentiability check requested by the coordinator -- FAILED, confirmed
directly, not assumed.** `jax.grad` through `warm_start_ic` raises
`ConcretizationTypeError: Abstract tracer value encountered where concrete value is
expected ... The problem arose with the float function` -- the new
`closed_loop_steady_state_exact` (`src/cstr_sbi/physics.py`) uses plain
`float(params[i])`/`scipy.optimize.brentq`, which cannot be traced by JAX's autodiff at
all. This is fine for every SBI/SNPE use (data generation is always a plain Python
eager loop, never differentiated -- confirmed in the original fix's propagation check)
but it is fatal for NUTS's `cstr_generative_model(..., matched_warm_start=True)`, which
needs HMC gradients through the whole model including the warm-start call. **Confirmed
end-to-end, not just the isolated grad check**: `run_mcmc_baseline(..., matched_warm_start=True)`
at Sc4 and separately at Sc5 both raise the identical `ConcretizationTypeError` within
~1.4s (i.e. it crashes before doing any real sampling work, not a slow run). **This is
a regression introduced by the warm-start fix**: before the fix, `matched_warm_start=True`
NUTS was differentiable (through `diffrax`'s autodiff-compatible solver) but very slow
(Theory D found ~11x slower per window, 28s->310s, from differentiating through the
long forward integration) -- now it is fast to EVALUATE (0.55ms, confirmed earlier) but
cannot be differentiated through AT ALL, so `matched_warm_start=True` is currently
**completely unusable for NUTS**, a strictly worse outcome for that one use case even
though it's a strict improvement for every SBI use case. **NOT fixed this session** --
a real fix would need a JAX-native differentiable reimplementation (e.g. a fixed-
iteration-count `jax.lax.while_loop`/Newton iteration replacing `scipy.optimize.brentq`,
plus a differentiable formulation of the saturated/non-saturated branch selection,
likely via `jnp.where` rather than a Python `if`) -- a nontrivial additional task with
its own correctness risk, not attempted without checking in first. **Confirmed
unaffected**: the open-loop warm start (`scripts/nb21_nuts_worker.py`'s Sc0/Sc6 path,
and `cstr_generative_model`'s fixed, theta-independent default `y0`) computes
`warm_start_ic` EAGERLY, outside any gradient computation, so it is not broken by this
-- verified directly (`mode="open_loop"` NUTS run succeeds end-to-end). **No single-
window or multi-window NUTS re-test was attempted for the closed-loop matched-protocol
scenarios (Sc1-5,7)** -- doing so would just reproduce the same instant crash; this is
a hard blocker for that specific comparison, not a cost/timeout issue, so no amount of
budget would have helped. Reported to the coordinator as the primary NUTS finding;
decision on whether to invest in the JAX-native fix is theirs.

---

**STATUS (2026-11-07, `warm_start_ic` FIX implemented and verified
(coordinator-approved), replacing the audit-only entry immediately below. Read THIS
entry first if you want to know what `warm_start_ic` actually does now; read the entry
below it for how the bug was originally found. THIS FIX INVALIDATES EVERY PREVIOUSLY
CACHED POSTERIOR IN THE PROJECT that used `scenario_specific_warm_start=True`/
`matched_warm_start=True` — any such cached `.pkl` was trained on the old, wrong warm
start and must be retrained, not reused, for any future work.):**

**What changed, concretely**: `warm_start_ic` (`src/cstr_sbi/simulator.py`) no longer
forward-integrates the closed-loop ODE for a fixed horizon. It now calls a new function,
`closed_loop_steady_state_exact` (`src/cstr_sbi/physics.py`), that solves the steady-state
algebraic equations directly via root-finding:
- **Non-saturated branch** (closed form, no iteration): `dI/dt=0` with the anti-windup
  gate open pins `T=Tsp` exactly (integral control removes steady-state temperature
  error). Given `T=Tsp`, `C`, `Tc`, `Qc` follow in sequence from `dC/dt=dT/dt=dTc/dt=0`,
  and `I` follows from the controller law.
- **Saturated branch** (used for Sc2/Sc5's heavy fouling, where even maximum coolant
  flow can't remove enough heat to reach `Tsp`): `Qc` is pinned at the active bound and
  a 1-D root-find (`scipy.optimize.brentq`, Arrhenius-nonlinear in `T`) gives `T`; `C`
  and `Tc` follow in closed form.

**A second sub-bug caught and fixed DURING this fix, not before shipping it**: the
first implementation picked which bound (`Qc_min` vs `Qc_max`) to saturate at from the
*sign* of the non-saturated candidate's implied `Qc` -- this is WRONG and gave a badly
incorrect answer for Sc2/Sc5 (picked `Qc_min=0`, giving `T=316.4`, `|rhs|=3.9`, when the
correct saturated state is at `Qc_max`, `T=312.87`, `|rhs|~1e-4`). Root cause: `Tc` is
physically bounded within `[Tci, T]` for any `Qc>=0` (as `Qc->inf`, `Tc->Tci`; as
`Qc->0`, `Tc->T`), so a non-saturated candidate implying `Tc < Tci` (insufficient
cooling capacity even at infinite flow -- true for Sc2/Sc5) means saturate at MAX
cooling, not min. Fixed by evaluating `T_ss` at both `Qc_min` and `Qc_max` directly
(two cheap 1-D root-finds) and using the verified-numerically monotonicity of
`T_ss(Qc)` (more flow -> lower `T`) to decide feasibility, instead of trusting the
non-saturated candidate's sign. **A third, smaller issue** also caught: landing `I`
exactly on the saturation boundary let a very large `I` value's float32 rounding
occasionally reopen the anti-windup gate (`qc_unclamped` rounding to just inside the
bound instead of just outside it), giving a spurious nonzero `dI` at an otherwise-exact
fixed point -- fixed with a small (`2e-2` in `Qc` units) one-sided epsilon push past the
boundary.

**Verification** (same RHS-evaluation + forward-redrift methodology as the original
audit, see the entry below for the pre-fix numbers for comparison): at EVERY Sc0-7
closed-loop scenario, `|rhs|` at the new warm start is `1e-6` to `1e-4` (vs. the
original `0.02`-`0.42`) and 50-minute forward-redrift is `1e-5` to `1e-3` (vs. the
original `1`-`21`). Cross-checked independently against brute-force long-horizon
forward integration at the OLD `t_warm=20000`: Sc2 gives `I=2646.8, T=312.869,
Tc=297.340` from the new exact solve vs. `I=2645.3, T=312.870, Tc=297.340` from the
slow brute-force approach -- matches closely, confirming the root-find converges to the
same physical fixed point the expensive method was slowly approaching.

**Propagation check (traced every call site via `grep`, not assumed)**: `warm_start_ic`
is the single source of truth for System I -- `src/cstr_sbi/scenarios.py` (evaluation-
data generation), `src/cstr_sbi/inference.py`'s `cstr_generative_model`
(`matched_warm_start=True`, the NUTS path) and `run_mcmc_baseline`'s default, and
`simulation_wrapper_sbi`/`_raw`'s `scenario_specific_warm_start=True` (SBI's own
training-data generator) all call this exact function with no separate
implementations, so the fix reaches all of them automatically, INCLUDING the exact path
nb22's budget sweep depends on. `scripts/nb21_nuts_worker.py`'s open-loop path also
calls the same function (for its fixed, theta-independent warm start) -- also
auto-fixed, and this does NOT change nb21's own open-loop/closed-loop model-mismatch
investigation's logic (that warm start was never theta-dependent to begin with).

**Two explicit NON-fixes, flagged rather than left silently inconsistent:**
1. `cstr_generative_model`'s default `y0` (used only when `matched_warm_start=False`)
   is `NOMINAL_Y0_CL`, a hand-set module constant in `physics.py`
   (`[0.0184, 312.15, 299.05, 0.0]`) -- NOT computed via `warm_start_ic`, explicitly
   documented in its own existing comment as a deliberately-approximate
   "avoid-a-stiff-cold-start" IC, predating and unrelated to this bug. Not touched.
2. `src/cstr_sbi/luyben/simulator.py` has its OWN, completely separate `warm_start_ic`
   implementation for the Luyben/System II model -- a different codebase, NOT audited
   or touched this session. Flagged explicitly since it shares a function name with
   nothing else in common.

**Side benefit found while fixing this**: the new root-find costs ~0.55ms/call vs. the
old forward-integration's ~350ms/call (~640x faster) -- the per-theta matched-warm-start
overhead that used to be a real training-budget cost concern (`simulation_wrapper_sbi`'s
own docstring previously warned "+1hr for n=10,000") is now negligible.

**Performed by**: direct numerical audit + fix in this session, 2026-11-07, coordinator-
approved before launch of nb22 Part 2/3 (the full budget-convergence+SBC retrain sweep,
now running in the background on this corrected warm start -- see the Part 2/3 entry,
written separately once that run completes, per the coordinator's explicit instruction
to keep this fix's own entry findable without digging through Part 2/3's results).
Files touched: `src/cstr_sbi/physics.py` (new `closed_loop_steady_state_exact`),
`src/cstr_sbi/simulator.py` (`warm_start_ic` body only, signature unchanged).

---

**STATUS (2026-11-07, nb22 Part 1 AUDIT-ONLY entry, kept for history of how the bug was
found — the fix itself is documented in the entry above, read that one first — nb22
(`notebooks/system_I/22_sbi_full_convergence_study.ipynb` build attempted, blocked at
Part 1 by a NEW, SEVERE warm-start bug in `warm_start_ic`/
`simulate_closed_loop_to_steady_state` itself — this is upstream of and independent from
the matched-vs-fixed warm-start TIMING mismatch (Theory D) fixed in the entries below;
read this first, it is a blocking confound for the entire budget-convergence study this
notebook was commissioned to run, and arguably for every SBI/NUTS result built on
`warm_start_ic` anywhere in this project to date):**

Tasked with auditing `warm_start_ic`'s correctness (Part 1 of the nb22 brief) BEFORE
trusting any training-budget convergence sweep. Numerically verified (script contents
preserved in `results/22_warmstart_audit.csv` and `results/22_warmstart_window_impact.csv`;
ad-hoc scripts not committed, logic reproduced below) that `warm_start_ic`'s default
`t_warm=1500` minutes is **NOT long enough for the PI integrator state `I` to reach
steady state**, even though the physical states `C`, `T`, `Tc` do converge quickly
(`|dC/dt|,|dT/dt|,|dTc/dt| <~1e-4` by `t_warm=1500` at every Sc0-7 point checked).

Concretely, at `t_warm=1500` (the value used everywhere: SBI's own training-data
generation via `scenario_specific_warm_start=True`, and NUTS's `matched_warm_start`
fix from earlier today), `dI/dt = T - Tsp` (the un-gated integrator residual) is still
`-0.086` (healthy), `+0.425` (Sc2 fouling), `-0.104` (Sc3 decay), `+0.021` (Sc4
combined), `+0.045` (Sc7 drift) — NOT near zero, i.e. `I` is still actively integrating,
not at a fixed point. Escalating `t_warm` to 20000 min brings `dI/dt` down to
`~1e-4"-1e-3` (much closer to converged) but the absolute value of `I` itself is wildly
different depending on how long it was allowed to integrate: e.g. Sc2 fouling:
`I=660` at `t_warm=1500` vs `I=2645` at `t_warm=20000` — a 4x difference, still not
obviously converged even at `t_warm=20000` (the physical states are fast modes; `I` is
a genuinely slow mode, consistent with `tau_i=10` min and a near-zero steady-state
temperature error that `I/tau_i` must balance almost exactly in the `Qc` formula).
**A second, independent, compounding bug makes this worse**: naively raising `t_warm`
further (e.g. to 100000) does NOT help — `simulate_closed_loop_to_steady_state`'s
`diffrax.diffeqsolve` call uses `throw=False` with a FIXED `max_steps=2_000_000`; at
`t_warm=100000` the adaptive-step integrator exhausts the step budget partway through
(confirmed via `sol.result`/`sol.stats`: it silently stops at `t=64055.8`, not
`t=100000`, with `sol.result` reporting `RESULTS<max steps reached>`) and
`warm_start_ic` returns that truncated, non-requested-horizon state with NO warning
anywhere in the call chain — so "just increase `t_warm`" is not a safe fix without also
raising `max_steps` and checking `sol.result`.

**Why this matters for everything downstream, checked directly, not assumed**:
re-simulated the actual 60-minute observation window (`t_window=60`, the real default
used by SBI's training-data generator) starting from the `t_warm=1500` warm start vs.
the more-converged `t_warm=20000` warm start, same (alpha,beta), same inlet/ctrl.
Realized coolant flow `Qc` over that 60-minute window differs by up to **190 (mean
diff 189) out of a `[0,400]` actuator range at Sc2 fouling** (`alpha=1,beta=0.7`),
**5.5-5.8 at the healthy and Sc4 points** — i.e. the size of the artifact this
non-convergence injects into the actual training/likelihood data is itself strongly
**(alpha,beta)-dependent** (it is ~35x larger at Sc2 than at healthy/Sc4). `T` and `C`
over the same window are barely affected (`maxDiff <~0.09` K, `<~1.4e-4` mol/L) because
the controller/physical states are fast and self-correcting, but `Qc` — which shapes
`T`'s subsequent trajectory and whatever downstream summary statistics depend on
control effort/saturation fraction — is not. **This is a previously unchecked
candidate mechanism for exactly the kind of spurious, scenario-dependent artifact
Theory D/nb19/nb20's whole (alpha,beta)-correlation investigation has been chasing**,
and it was never isolated before because no prior check compared warm-start outputs
at different `t_warm` horizons against each other or against the ODE's own RHS.

**Explicit, plain statement per this project's own "do not silently patch around it"
discipline: `warm_start_ic` as currently implemented and defaulted is NOT a correct
closed-loop steady state for every Sc0-7 point checked except Sc5 (which only
"converges" trivially because actuator saturation freezes `dI` to exactly 0 via the
anti-windup gate, not because the unsaturated dynamics actually settled).** This is
flagged as a confound for: (a) nb22's own Part 2/3 budget-convergence study (training
new posteriors on data generated with this warm start would just re-measure budget
sensitivity on top of an uncontrolled, scenario-dependent artifact, not isolate it);
(b) every SBI training run in this project that used `scenario_specific_warm_start=True`
(nb04 onward); (c) NUTS's `matched_warm_start` fix from earlier today 2026-11-07 (it
reuses the exact same `warm_start_ic` call, so it inherits the same non-convergence);
(d) by extension, the Theory D "resolved, real correlation" verdict in the entries
below, which was built on this same warm start and should be treated as resting on an
unaudited foundation, not as independently re-confirmed by this finding.

**NOT fixed here, NOT silently patched, per the brief's explicit instruction.** nb22's
Part 2 (training budget sweep across 1000-50000 sims/seed) and Part 3 (optimal-budget
recommendation) were NOT executed this session: running an expensive multi-budget,
multi-seed training+SBC sweep on top of a known-incorrect warm start would produce
numbers that need to be redone once the warm start is fixed, and the brief's own Part 1
instruction is to stop and flag, not proceed, on finding a discrepancy. **Concrete next
step, not yet started**: decide and implement a real fix (candidates not yet evaluated:
(i) raise `t_warm` substantially AND raise `max_steps`/add an explicit convergence
check — e.g. assert `|rhs| < tol` before returning rather than trusting a fixed
horizon; (ii) solve for the true fixed point directly via a nonlinear root-find on the
4-state RHS instead of forward-simulating; (iii) if the real production observation
windows start from some other, already-converged operating-point convention elsewhere
in the codebase, check whether `warm_start_ic` is even on the critical path for
production training/eval — not verified either way this session), then re-run this
same numerical audit to confirm, THEN proceed to nb22 Part 2/3 on the corrected warm
start. Raw audit numbers: `results/22_warmstart_audit.csv`,
`results/22_warmstart_window_impact.csv`.

---

**STATUS (2026-11-07, LATE UPDATE — the warm-start-mismatch mechanism (Theory D) was
root-caused, fixed, and validated at full scale, REVERSING the Theory C verdict below
this entry — read this first, it supersedes everything below including the entry
immediately beneath it, which was itself already a reversal of nb19):**

Following up on Theory D's "partially supported" status, found and fixed a second real
bug: `cstr_generative_model` (NUTS's likelihood, `src/cstr_sbi/inference.py`) used a
FIXED warm start for every posterior draw, while the data it was fitting was generated
with a PER-THETA matched warm start (the same convention SBI's own training already
uses) — a genuine model mismatch, not just the diagonal-covariance bug fixed earlier
the same day. Added an additive `matched_warm_start` flag; fixing it moved a
single-window Sc4 test from Corr$=+0.03$ to $-0.75$ (reproduced identically 4 times).
**Confirmed at full scale**: re-ran Theory C's entire 8-scenario sweep under the fix
(via a standalone script after 3 separate nbclient stalls made re-running the full
notebook impractical — see `scripts/recompute_theoryC_section5.py` and the notebook's
own new Section 5 for the full story). Result: NUTS's real pooled posterior now AGREES
IN SIGN with SBI at every closed-loop scenario with usable data —
Sc1: SBI$-0.650$/NUTS$-0.423$; Sc2: $-0.856$/$-0.488$; Sc3: $-0.638$/$-0.569$; **Sc4
(the key scenario): $-0.893$/$-0.786$ (88% magnitude agreement, 13/20 windows
converged, clean R-hat)**. This REVERSES the Theory C verdict below (which was built on
the mismatched protocol) and resolves the Theory C/Theory E tension in favor of the
correlation being REAL for Sc1-4.

**Explicitly NOT a clean sweep, stated plainly**: Sc7 shows a SIGN DISAGREEMENT (SBI
$+0.632$ vs. NUTS $-0.666$, both well-converged) — unexplained, flagged not glossed
over. Sc0/Sc6 (open-loop): 0/20 windows converged either run — root cause identified
(NUTS's model is always closed-loop, a structural mismatch for open-loop data, separate
from the warm-start issue). Sc5 (severe saturation): 0/20 windows converged despite
being closed-loop — a third, distinct, UNRESOLVED mechanism, plausibly
saturation-induced likelihood irregularity. Three of eight scenarios have no NUTS
ground truth at all; a fourth disagrees in sign. The SBI training-budget
non-convergence flagged further below is untouched by any of this. Stage 3's Theory C
verdict, Theory D verdict, and final synthesis were all rewritten in the executed
notebook and `scripts/build_nb_system_I_20.py` to reflect this. `main.tex`/SI/
`article_outline_CChE.md` still NOT touched.

**STATUS (2026-11-07, notebook 20 COMPLETE, Stage 3 verdicts written — superseded by
the entry above; kept for history, read it second):**
`notebooks/system_I/20_alpha_beta_correlation_mechanism_theories.ipynb`
(`scripts/build_nb_system_I_20.py`) finished all 6 theories + Stage 3. **Headline
reversal: nb19's "real, seed-robust Corr(alpha,beta) at Sc4" finding is now actively
CONTRADICTED by this notebook's own converged, correctly-specified NUTS ground truth at
that exact point** (SBI: -0.893, NUTS: +0.082 — opposite sign, both from this session's
own computations, not a re-read of old numbers). Per-theory verdicts: **A** NOT
CONFIRMED/INCONCLUSIVE (ridge-slope test too weak to resolve); **B** FALSIFIED as
literally stated (open-loop shows SUBSTANTIAL, oppositely-signed correlation, not
near-zero — Sc0 +0.612 vs Sc1 -0.650, Sc6 +0.564 vs Sc2 -0.856); **C** evidence AGAINST
the correlation being real for all 6 closed-loop scenarios (NUTS converges cleanly,
R-hat<=1.004, 90-100% windows, and finds SMALL correlation -0.06 to +0.16 everywhere,
vs. SBI's LARGE -0.89 to +0.85) — but Sc0/Sc6 (open-loop) leave NUTS itself
non-convergent (R-hat in the hundreds-to-thousands) even after fixing a real
diagonal-covariance likelihood bug that had been misdiagnosed as genuine posterior
multimodality (see below) — INCONCLUSIVE there, unresolved, flagged as needing its own
follow-up; **D** PARTIALLY SUPPORTED, and now reads as the more likely culprit: the
OLD (pre-fix) fixed-warm-start protocol's near-zero correlations sit much closer to
NUTS's converged ground truth than the current matched-warm-start protocol's large
ones do; **E** PARTIALLY SUPPORTED narrowly — the diagonal-vs-full-covariance distinction
in the local-FIM diagnostic is confirmed real, but it was closing a gap toward SBI's
now-independently-suspect large correlation, not toward the true (NUTS) posterior; **F**
PARTIALLY FALSIFIED — raw-trajectory CNN access does not shrink the correlation, it
flips sign at 5/6 scenarios, implicating the training protocol over the representation.
**A second, separate, important bug was found and fixed while building Theory C**:
`cstr_generative_model` (the shared NUTS likelihood, `src/cstr_sbi/inference.py`)
defaulted to a diagonal Gaussian over the 29-D summaries — the SAME flaw Theory E
diagnoses. Fixing it (added an optional `cov_obs` full-covariance parameter, additive
and backward-compatible) collapsed an apparent "posterior multimodality" finding
(R-hat 3-11) down to R-hat~1.00 at the healthy point — the multimodality was a
likelihood-misspecification artifact, not real. **Net verdict (Stage 3's own final
synthesis, read it in full in the notebook): still OPEN, arguably LESS settled than
before this notebook existed** — the Theory C (small, NUTS-ground-truth correlation) vs.
Theory E (full-covariance local-FIM recovers SBI's large correlation) tension is stated
explicitly as unresolved, not picked a side on, with three candidate reconciliations
listed and none confirmed. **Nothing from nb19 or nb20 about System I's (alpha,beta)
correlation should be treated as settled or promoted toward the manuscript** until (a)
the Theory C/E tension is disambiguated and (b) the SBI training-budget non-convergence
below is resolved. `main.tex`/the SI/`article_outline_CChE.md` NOT touched, per this
project's promotion discipline — if anything, this result argues harder for continued
caution than any prior round did.

**STATUS (2026-11-06, PROJECT-WIDE OPEN RISK FLAGGED, not acted on yet — read this
first, it is a flag for a FUTURE dedicated audit, not a task started now):** Building
`notebooks/system_I/20_alpha_beta_correlation_mechanism_theories.ipynb`
(`scripts/build_nb_system_I_20.py`) found that nb19's own working assumption --
`n=10,000` simulations/seed is "the production budget" for System I's 2-D SNPE
posteriors -- was NOT actually converged: a dedicated budget sweep found
Corr(alpha,beta) at Sc4 still moving by ~0.13 between `n=10,000` and `n=20,000` (well
above a 0.05 convergence threshold), and escalating once more to `n=40,000` (per this
project's own "go one level higher, don't silently stop" discipline) still did not
cleanly confirm convergence. **This raises an open, project-wide question that nb20
does NOT attempt to answer and is explicitly out of scope for it**: if `n=10,000` was
not a converged training budget for THIS (alpha,beta)-correlation question, are any of
System I's OTHER production posteriors/results that were also trained at `n=10,000`
(or similar historically-assumed-sufficient budgets) -- e.g. Table VII's Sc2
multi-method comparison, the EKF/UKF baseline comparisons (`16_ekf_ukf_baseline.ipynb`),
the 30-day sequential-degradation-tracking numbers (`10_sequential_degradation_tracking.ipynb`),
and existing SBC calibration claims across System I's notebooks -- also at an
unconverged training budget, not just this one correlation finding? **Flagged here as a
real, project-wide risk worth a dedicated convergence audit across System I's other
production posteriors before any of them is treated as final/publication-ready -- NOT
started, NOT assumed answered either way, purely a flag for a future session.**

**STATUS (2026-11-05, updated again — Section 10, investigating WHY seed4/cached fail
alpha-SBC: no clean mechanism found, read this first, then the 10-seed-extension entry
right below for the context it builds on):** `scripts/build_nb_19.py`/the same notebook
was extended once more today with a new Section 10 (10a-10d), the explicit next step
flagged by the 10-seed-extension entry below: why do seed4 and the cached posterior fail
alpha-SBC at extreme significance ($p\sim10^{-5}$) while 8 of the other 9 posteriors don't?
Three diagnostics were run, reusing already-trained/cached posteriors and the existing
SBC held-out set (no retraining): (1) **rank-histogram inspection** (`results/
19_sbc_rank_diagnostics.csv`, `figures/19_sbc_rank_histograms.png`) for seed4/cached (FAIL)
vs. seed0/seed1/seed6 (PASS); (2) **prior-edge saturation check** (`results/
19_sbc_edge_saturation.csv`, `figures/19_sbc_edge_saturation.png`) splitting the N=300
held-out draws into near-prior-edge ($\alpha$ within 20% of the half-range of
ALPHA_LOW=0.4/ALPHA_HIGH=1.2) vs. away-from-edge subsets; (3) **training-level correlate
check** using the only diagnostic actually persisted in the cached `.pkl` metadata
(`wall_time_s` — epochs-trained/validation-loss are printed during training but not saved,
so this check could not go further without a retrain, which was explicitly out of scope).
**Result: genuinely inconclusive, and importantly NOT one shared mechanism for both
failures.** (1) seed4's alpha ranks show a left-shifted mean (`mean_rank=208.9` vs. the
uniform expectation of 250) with near-uniform edge/middle fractions (`frac_edge_20pct=0.16`,
`frac_mid_20pct=0.21`) — a mean-bias signature, not U-shaped overdispersion or
peaked/inverse-U underdispersion. The cached posterior's alpha ranks show a *different*
signature: also a left-shifted mean (`mean_rank=216.9`) but ALSO a genuine overdispersion
signature (`frac_edge_20pct=0.33` vs. expected 0.20, `frac_mid_20pct=0.17`). (2) Prior-edge
saturation is actively CONTRADICTED for seed4 (the near-edge subset passes comfortably,
`KS p=0.098`, while the away-from-edge subset fails at the same extreme level as the full
sample, `KS p=1.999e-6` — the failure is concentrated AWAY from the boundary, the opposite
of what edge saturation predicts) and only weakly/partially consistent with the cached
posterior's failure (both near-edge and away-from-edge subsets already fail; near-edge
fails somewhat more severely, `p=2.27e-4` vs. `p=4.10e-3`, a difference in degree between
two already-failing subsets, not "near-edge fails, away-from-edge is calibrated"). (3) No
training-time correlate: seed4's wall time (322.6s) does not stand out from several
cleanly-passing seeds that took as long or longer (seed3=319.9s, seed6=330.2s,
seed7=310.3s) — reported as "nothing found, but the check is limited to wall time," not a
clean negative, since richer training diagnostics were never persisted for any seed.
**Net verdict: this remains unresolved at the mechanism level.** seed4 and the cached
posterior fail alpha-SBC for failure-mode signatures that are related (both show a
leftward mean shift) but not identical (cached additionally shows overdispersion that
seed4 does not), prior-edge saturation does not explain either failure cleanly (actively
wrong for seed4, only a weak partial correlate for cached), and the training-level check
found nothing distinguishing among what this project's artefacts actually log. No tidy
single-mechanism story is being claimed here — explicitly, per this project's own
discipline against overclaiming resolved mechanisms that later need walking back. **Still
NOT promoted into `main.tex`/the SI** — unchanged from before, now with this specific
open item investigated (not resolved) rather than simply flagged as untouched.

**STATUS (2026-11-05, updated again — 10-seed extension + 2D posterior figure, read this
next, then the SBC-validation entry right below for the 6-posterior history it extends):**
`scripts/build_nb_19.py`/the same notebook was extended twice more today. (1) The 5-seed
multi-seed sweep (Section 3) was extended to 10 seeds (seed0-seed9, same full-budget
recipe, newly-cached at `results/19_multiseed_posterior_seed{5..9}.pkl`); the Corr(alpha,beta)
distribution at Sc4 across all 10 is mean=-0.482, std=0.134, range=[-0.762,-0.269]
(`results/19_multiseed_corr.npz`, now reproducible across re-execution -- an unseeded
`posterior.sample()` call was found and fixed mid-extension, see `HANDOFF.md`) —
essentially the same mean as the original 5-seed sample (-0.472), confirming the
correlation is not a lucky draw of 5 unusually-correlated runs. (2) SBC (Section 6b) was
re-run across all 10 new seeds, giving an 11-posterior total (10 seeds + the 1
pre-existing cached posterior, `results/19_sbc_fullbudget_posteriors.csv`): 7 PASS
cleanly, 2 FAIL on alpha (seed4 and the cached posterior — unchanged from the 5-seed
pass), 2 newly-observed AMBIGUOUS cases (seed5 on beta p=0.011, seed9 on alpha p=0.031).
**The 5-seed sample's "1-in-3 alpha-SBC-failure rate" weakens at n=10/11**: outright
alpha-FAIL drops to 2/11 (18%); no new outright failures appeared among the 5 new seeds
(only clean passes or ambiguous cases). The magnitude-vs-failure relationship checked at
10 seeds is genuinely mixed (not a clean "opposite direction" story): the outright FAIL
(seed4, magnitude -0.269) and one AMBIGUOUS case (seed5, -0.370) are among the
smallest-magnitude correlations, and the single largest (seed0, -0.762) passes cleanly —
but the other AMBIGUOUS case (seed9, -0.556) is the *second-largest* magnitude in the
batch, so magnitude and calibration outcome look like largely independent properties, not
a monotonic trend in either direction. **Net verdict: the "real, not a
systematic-miscalibration artefact" reading from the 5-seed/6-posterior pass is unchanged
in direction at 10 seeds/11 posteriors — but is still not a clean, unqualified
validation**: seed4 and the single cached posterior still fail alpha-SBC at extreme
significance in both the 5-seed and 10-seed passes, unexplained, plus 2 new ambiguous
cases that widen (not narrow) the set of posteriors needing scrutiny before a future
retrain under this recipe should be trusted without its own SBC check. (3) A new Section
9 was added: a 2-D (alpha,beta) posterior scatter figure for seed0 at the healthy point
and at Sc4 (`figures/19_posterior_2d_seed0_sc4.png`) — the direct visual complement to
the scalar correlation numbers, showing a near-isotropic blob at healthy vs. a visibly
elongated diagonal ridge at Sc4 for the same (SBC-clean) posterior. seed0 remains the
single largest-magnitude seed at 10 seeds too (unlike an earlier, not-yet-reproducible
intermediate draft of this analysis that transiently showed seed2 as largest before the
sampling-seed fix), so no change of seed pick was needed. See `HANDOFF.md`'s newest
2026-11-05 entry for full numbers. **Still NOT promoted into `main.tex`/the SI** — same
blockers as before (mixed SBC result, unexplained alpha-calibration failure), now with
slightly more precise failure-rate numbers but the same qualitative caveat.

**STATUS (2026-11-05, updated again — SBC validation now closed, see below first, then
the full-budget-run paragraph right under it for context): the open item left by the
full-budget run — "is the seed-robust Corr(alpha,beta)=-0.47 real, or systematic
miscalibration that happens to reproduce across seeds?" — has been tested with SBC
(Section 6b of the notebook; `scripts/build_nb_19.py`, same notebook, same path). Result
is a genuine mixed finding, not a clean pass or fail: 4 of the 6 posteriors (seeds 0-3)
pass SBC cleanly on both alpha and beta; seed4 and the single cached posterior
(`sbi_posterior_matched_protocol_n10000.pkl`) FAIL on alpha specifically (KS p ~1e-5), while
beta passes in ALL 6 posteriors without exception. Critically, the two SBC-failing
posteriors are not distinguished from the four passing ones by the magnitude of their
Corr(alpha,beta) — seed0 has the LARGEST-magnitude correlation (-0.645) and passes SBC
cleanly, seed4 has one of the SMALLER-magnitude correlations (-0.401) and fails — so a
"systematic bias manufactures the correlation" mechanism is not well supported by this
pattern (it predicts failure/magnitude should track together; they don't). **Overall
verdict: evidence leans toward the correlation being a real feature of System I's trained
posterior geometry, not a miscalibration artefact, but this is NOT a clean, unqualified
validation** — a 1-in-3 alpha-SBC-failure rate across the sampled posteriors (including the
original cached one) is high enough that it must be called out explicitly, not rounded to
"passes." See `results/19_sbc_fullbudget_posteriors.csv`, the notebook's new Section 6b and
Section 8 verdict addendum, and `HANDOFF.md`'s newest entry for full numbers.
**Still NOT promoted into `main.tex`/SI** — the mixed SBC result, and the unexplained
alpha-calibration failure in 2/6 posteriors, are new explicit blockers before any
manuscript claim (replacing "not yet SBC-validated" as the blocker, which is now resolved
into "SBC-validated, with a caveat").

**STATUS (2026-11-05, full-budget run, context for the above): `19_alpha_beta_banana_and_ekf_asymmetry.ipynb` has been
BUILT and EXECUTED AT FULL BUDGET** (`scripts/build_nb_19.py`; full production budget,
n=10,000 simulations/seed for every retrain, all 720 windows for the tracking comparison
— this corroborates and supersedes an earlier same-day reduced-budget pass, n=800/
72-of-720 windows). **The full-budget run REVERSES the reduced-budget pass's headline
Item 1+3/Item 2 conclusion**: "no banana generalises" does NOT hold — at full training
budget, 5 independently-trained seeds give a real, seed-robust, moderate-to-large
Corr(alpha,beta) (roughly -0.2 to -0.6 depending on operating point, one point +0.34),
consistently signed with the single cached production posterior, NOT near zero as the
n=800 pass had found. The n=800 result was a false negative from undertraining the 2-D
NSF posterior, analogous to this project's own SBC-power lesson (N=200 gives false passes;
need N≥400) — this time for posterior correlation structure rather than SBC rank
statistics. The local-FIM/Laplace diagnostic stays small everywhere and is the one shown
to be unreliable for this question. Of the three EKF-asymmetry hypotheses (unaffected by
the budget reversal, numbers held or tightened at full budget/resolution): (a) proxies and
(b) tuning were tested and found NOT to be the primary driver, while (c) linearisation
quality is the best-supported explanation (System I's linearisation error stays at machine
precision across its whole range; System II's grows ~7 orders of magnitude larger near its
near-snowball regime) — but a residual EKF-vs-SBI gap remains even after accounting for
(c), left explicitly open in the notebook's verdict cell. See `HANDOFF.md`'s latest
2026-11-05 entry (read on top of, not instead of, the earlier same-day entry) and the
notebook itself for full numbers. **Still not promoted into `main.tex`/SI** — not because
of budget this time (both scope reductions are now lifted), but because the reversed
(alpha,beta) correlation finding has not yet been SBC-validated (seed-robustness rules out
a single-training-run artefact, not systematic miscalibration of this training recipe) —
explicitly flagged as the next step before any manuscript claim.

# System I notebooks — outstanding work

Full prioritized plan: `docs/Manuscript/system_i_publication_plan.md`. This file is
just the notebook-folder-local pointer.

## TODO: build `19_alpha_beta_banana_and_ekf_asymmetry.ipynb`

**Goal.** Extend `notebooks/34_alpha_beta_r_banana_cross_system.ipynb`'s finding
("System I has no `(alpha,beta)` banana — its composition channel `C` structurally
excludes `beta` from its own governing equation, unlike System II's unexposed `z_A`
equivalent") rigorously enough to also explain the still-open EKF/UKF-vs-SBI asymmetry
between the two systems (`docs/Manuscript/system_ii_sb_identifiability_story.md` §5.3):
System II's EKF beats SBI in 30-day tracking near nonlinear regimes (raw-trajectory
access recovers representation-compression loss); System I's EKF/UKF lose to SBI
*everywhere*, despite System I having its own genuine ~236x FIM-ratio information
deficit for beta. Nobody has yet explained why.

**Be rigorous — this is a single-spot-check notebook (nb34) being promoted to a
publication-grade robustness result, not a quick follow-up.** At minimum:

1. **Extend nb34's `Corr(alpha,beta)` check beyond Sc4/Sc5** to every System I fault
   scenario where both parameters are simultaneously perturbed, not just the two already
   checked — confirm the "no banana" result is not an artifact of which two scenarios
   were picked.
2. **Multi-seed, not single-posterior.** Retrain (or resample from) multiple SNPE seeds
   and report the Corr(alpha,beta) distribution across seeds, the same lesson this
   project already learned the hard way for System II (SBC-best seed != classification-
   best seed; don't trust a single training realisation for a headline claim).
3. **RT-FIM check, matched-protocol.** Run the raw-trajectory FIM diagnostic
   (`scripts/fim_utils.py`, same methodology as nb34 §3/§4a and System II's nb39) at
   multiple operating points spanning the healthy-to-faulted range, not just two points,
   and check whether it agrees with the trained-posterior correlation everywhere, not
   just at the two spot-checked points.
4. **Test the three EKF-asymmetry candidate explanations directly, don't just list them:**
   - (a) Ablate System I's physics-informed proxies (`UA_eff_proxy`, `k0_eff_proxy`) the
     same way nb34 §4a ablates the `C` channel, and re-run the EKF/SBI tracking
     comparison without them — if the filter's disadvantage shrinks once those proxies
     are removed, that's direct evidence the proxies are closing the compression gap
     that favours EKF in System II.
   - (b) Audit System I's and System II's EKF/UKF gain-matrix tuning side by side
     (process/measurement noise covariances, linearisation step) — check whether System
     I's filter is simply worse-tuned, not structurally disadvantaged.
   - (c) Check System I's linearisation error (actual vs. linearised dynamics) across
     its full operating range vs. System II's, to test the state-space-size/
     linearisation-quality hypothesis directly rather than asserting it.
5. **Noise/perturbation robustness.** Repeat the headline `Corr(alpha,beta)` and RT-FIM
   checks under at least 2-3 sensor-noise levels (not just the nominal one) — confound
   magnitude can be noise-level-dependent, and this hasn't been checked for System I
   the way it has for some System II artefacts.
6. **Write a verdict cell, like nb34's, that states plainly**: is the EKF asymmetry now
   explained, and by which of (a)/(b)/(c) (or some combination)? If not fully resolved,
   say exactly what's still open rather than overclaiming.

**Conventions to follow** (see `scripts/build_nb_34.py` for the reference pattern):
build via a `scripts/build_nb_19.py` script using `nbformat`/`nbclient` (markdown +
code cells, executed, not hand-assembled in the Jupyter UI), save any new result arrays
to `results/`, save new figures to `figures/`, and reuse existing simulator wrappers
(`notebooks/33_fim_cross_system_validation.ipynb`'s matched-protocol wrappers,
`scripts/fim_utils.py`) rather than reimplementing them.

**Do not promote any finding from this notebook into `main.tex`/the SI until it's
been run multi-seed and the verdict cell is written** — this project has been burned
twice already (System II's alpha/beta_r and eta_col findings) by promoting a single-
seed or single-scenario result to a manuscript claim.
