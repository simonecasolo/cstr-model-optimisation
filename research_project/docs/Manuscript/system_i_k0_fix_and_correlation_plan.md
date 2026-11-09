# System I — K0 units bug, corrected-operating-point correlation decomposition, and next-step plan

**Status as of 2026-11-08.** Written so a fresh session (no prior conversation context)
can pick this up directly. Read this file, then
`/Users/simo/.claude/projects/-Users-simo-inso-po-RD-cstr-model-optimisation/memory/finding_system1_no_alpha_beta_banana.md`
(the full narrative history, very long — read at least its final ~5 dated entries,
most recent first) before touching anything. Also check
`notebooks/system_I/TODO.md` and `HANDOFF.md` (top entries) for anything that moved
after this file was written.


---

## 0. REVISION (2026-11-08, user decision) -- supersedes §3 Steps 0 and D and the "fix k0" framing

**Decision: keep the current `K0_NOMINAL = 16.96e12` (and `Tsp=312.5`, PI tuning, feed). Do NOT
re-derive a corrected operating point.** Step-0 evidence (HANDOFF "STEP-0 update",
`results/27_step0_catalogue_scan.csv`) is the reason a correction is not cheap: at k0/60 no
single Tsp avoids saturation across Sc0-Sc7 (reachable-T window ~1 K per point, empty
intersection), so a correction would need a feed/gain redesign and would invalidate every
System I result. Nothing in notebooks 01-24 is invalidated by this decision; the caution in
§1.3 ("re-verify once k0 fix lands") is withdrawn. nb26's buggy-point numbers become the
System I numbers of record rather than one arm of a comparison.

### 0.1 How k0 is presented (manuscript, Table I and text)
- Present k0 as **60x Fogler's value (1.018e15 h^-1 equivalent; = 16.96e12 min^-1 as coded)**,
  explicitly a *modification*, not attributed to Fogler. Disclose it as a choice made after the
  60x discrepancy was found; do not imply it was planned from the start.
- State the implication: a high-conversion design (~98%), arguably closer to industrial practice
  than the ~47% teaching example.
- Candidate physical rationale (UNVERIFIED -- check before use): Furusawa's reaction is
  acid-catalysed, so the apparent first-order constant scales with catalyst loading; a
  higher-acid/more active catalyst would justify a larger k0 and fits alpha as a
  catalyst-activity factor. **Check against Furusawa 1969 / Fogler**: catalyst (H2SO4) loading
  used in the data, whether rate is linear in acid, and whether a 60x factor is plausible. If it
  does not hold up, state the scaling plainly with no rationale.
- Fix the stale comment `# 1/min` in `physics.py` (comment-only edit, no value change) -- needs
  user OK since the file is shared; flag, don't do silently.

### 0.2 alpha identifiability at high conversion -- verify, then cite
Claim to verify numerically (cheap, `closed_loop_steady_state_exact`): at Da ~ 54, conversion is
insensitive to alpha but the measured C is not: dlnC/dlnalpha ~ -0.98, so alpha is well
identified provided C sensor noise << ~0.018 mol/L. Confirm Da, the -0.98 slope at Sc1/Sc4, and
the C-sensor-noise-to-0.018 ratio from `DEFAULT_SENSOR_NOISE_PCT`. This is consistent with the
hand-crafted alpha slope ~0.99 (nb22). The CNN alpha collapse (nb24, slope ~0.05) is a separate
representation problem, to be reported as a CNN limitation, not as a plant property. This also
tempers nb25's "14-16x signal" framing: that is a conversion-sensitivity statement, not a C one.
The ~236x FIM-ratio caution in the HANDOFF NB25 entry should be re-read in this light.

### 0.3 Manuscript changes required regardless of k0 (main.tex; user to approve each edit)
1. **§6.3.3**: cut "CNN reproduces the bias within <1%". nb04b (matched protocol) shows the CNN
   cuts beta bias by 86%; nb24 shows CNN alpha collapse. Rewrite as a CNN-representation limitation.
2. **§6.4 beta-bias table** (0.551, -0.149, ...): from the old fixed-warm-start protocol; the matched
   protocol removed nearly all of it. Re-run under the matched protocol or reframe.
3. **§4.3 and §6.3.1 Fisher information**: diagonal noise covariance is wrong for off-diagonals.
   Recompute the 250-500x (and ~236x, `eq:fim_ratio`) ratio with the full covariance
   (`compute_fim_full_cov`-style, as in nb20 Theory E / nb26).
4. **alpha-beta correlation**: report as a positive physical heat-balance ridge (noise-free
   +0.92 buggy-point) plus a negative component from the controller passing noise between
   channels (net -0.90 at Sc4; C/Qc channels drive it). Relative size depends on operating point
   and noise level -- do not quote a single net Corr as the headline.
5. Do NOT touch `article_outline_CChE.md` / SI without a separate instruction (house rule stands).

### 0.4 Added check before writing up point 4: raw-trajectory Fisher test of the ridge
Outline §8.1 says a joint degeneracy found with compressed features must be checked with a
raw-trajectory Fisher analysis before being called physical. Apply to System I: compute the
full-covariance FIM on **raw (C,T,Tc,Qc trajectories)** rather than the 29-D summary, noise-free
controller and noisy controller, at healthy and Sc4, at the current operating point.
- Ridge (+0.9 noise-free) survives on raw -> System I has a genuine physical ridge; System II
  does not -- report the contrast.
- Collapses -> consistent with the System II story (feature-compression artefact).
Either outcome is reportable; skipping it would contradict our own §8.1.

### 0.5 Revised sequence (replaces §3)
- **Step A' (nb27, FIM-only, minutes)**: at the current operating point: (i) protocol audit of
  nb26's noise-free-controller variant (same warm start, same obs noise); (ii) process-noise-only vs
  sensor-noise-only ablation; (iii) eigenvalues/eigenvectors + marginal widths instead of scalar Corr;
  (iv) net Corr vs sigma_T and vs process-noise level, noise-free value as reference line, with the
  predicted zero-crossings; (v) **§0.4 raw-trajectory Fisher check**; (vi) §0.2 alpha-sensitivity check;
  (vii) full-covariance recompute of the FIM ratio (§0.3 item 3). Reuse nb26 machinery; the
  corrected-k0 arm is kept only as a sensitivity comparison, flagged non-production (Step-0 saturation).
- **Step B (NUTS)**: unchanged decision point (nb23 R-hat~500 unresolved; or grid/rejection posterior
  at one point as a substitute) -- needed only for the local-vs-global agreement spot-check.
- **Step C**: SBI retrain at the corrected point is DROPPED. Instead, compare existing/nb22
  n=50,000 SBI posteriors against the FIM prediction (eigen-structure, TARP/C2ST joint check).
- **Step D (re-run notebooks 01-18)**: DROPPED -- only the manuscript edits in §0.3 remain, plus
  re-running the specific items that are stale (beta-bias table; FIM ratio).
- Nb21-24: resume per their own status at the CURRENT operating point (no retrain needed); nb24 H3
  still blocked by the mach-port crash.

Original text below is retained for history; where it conflicts with §0, §0 wins.

---

## 1. What's already confirmed (do not re-derive, just reuse)

### 1.1 The K0_NOMINAL 60x unit bug — CONFIRMED
`K0_NOMINAL = 16.96e12` in `src/cstr_sbi/physics.py` line ~37 is commented `# 1/min`
but the literature value (Fogler/Furusawa 1969) is actually in **1/hr**. Verified
three independent ways in `notebooks/system_I/25_k0_units_verification.ipynb`
(`scripts/build_nb_system_I_25.py`):
- 3 independent primary sources all state the value in hr⁻¹, two matching this
  codebase's own `E_A=75362.0 J/mol` to 6 significant figures.
- Internal consistency: as-coded → 98.2% conversion (matches `main.tex`'s own
  stated "~98%"); corrected (÷60) → 47.2% conversion.
- Direct simulation: alpha-sensitivity dX/d(ln alpha) is 14-16x larger at the
  corrected point.
- **The fix is NOT a one-line edit** — naively dividing `K0_NOMINAL` by 60 drives
  the current setpoint (`Tsp=312.5K`) into permanent actuator saturation, because
  the PI tuning/setpoint/feed were apparently chosen around the buggy value. A
  consistent operating-point re-derivation is needed (see §3 below).

### 1.2 Cheap FIM/sensitivity decomposition at both operating points — DONE (notebook 26)
`notebooks/system_I/26_cheap_correlation_prediction.ipynb`
(`scripts/build_nb_system_I_26.py`) compared the buggy (98% conv., Tsp=312.5K) vs. a
**minimal, non-production-ready** corrected operating point (K0/60, Tsp lowered to
300.3K just enough to avoid saturation) via full-covariance FIM (`cov_offdiag_corr`,
same trusted methodology as the earlier "Theory E" work in notebook 20).

| op point | full channels, noisy controller | full channels, noise-free controller | drop C | drop T | drop Tc | drop Qc |
|---|---|---|---|---|---|---|
| buggy (98% conv.) | **-0.901** | **+0.918** | -0.174 | -0.936 | -0.934 | -0.254 |
| corrected (47% conv.) | **-0.029** | **+0.955** | +0.307 | +0.003 | -0.076 | -0.166 |

The buggy/noisy number (-0.901) closely reproduces the independently-trained SBI
posteriors from notebooks 19/20/22 (-0.91 to -0.48 across seeds/budgets) — good
cross-validation that this cheap FIM proxy is trustworthy.

**Key finding, mixed mechanism, not reducible to one story:**
- A robust, strong, **positive** physics-structural ridge exists at BOTH operating
  points (noise-free-controller gives +0.82 to +0.97 everywhere tested) — matches
  the simple heat-balance-compensation sign prediction (lower alpha + lower beta
  both reduce heat duty, so they trade off positively at the physics level).
- A noise-mediated effect tied to the C/Qc channels (dropping either collapses the
  correlation) exists but **only at the buggy operating point** — at the corrected
  point the ablation pattern doesn't replicate cleanly (dropping C flips sign).
- **Net picture**: at the buggy point, the noise effect is large enough to REVERSE
  the sign of the underlying positive structural ridge (net -0.90 vs underlying
  +0.92). At the corrected point, the two effects roughly CANCEL to near zero (net
  -0.03 vs underlying +0.95) rather than either dominating.

**Caveats on notebook 26's corrected operating point, important for everything
downstream:**
- It's a cheap FIM/Laplace proxy, not a retrained SBI posterior — the near-zero
  corrected number should be spot-checked against a real retrain/NUTS before being
  treated as final.
- `Tsp=300.3K` was chosen to maximize worst-case valve margin across only 2 theta
  points (healthy, Sc4-equiv: alpha=beta=0.85). Even its best achievable margin is
  thin: only **9.54 L/min out of [0,400]** at the Sc4-equivalent point (vs 113 L/min
  at healthy) — the two points' Qc_ss(Tsp) curves don't overlap well at any single
  Tsp. **This has only been checked at 2-3 of the 8 catalogue scenarios (Sc0-Sc7)**
  — a scenario like Sc5 (severe fault) could plausibly saturate at this thin margin.
  **Do not treat Tsp=300.3K as the final corrected operating point for re-running
  notebooks 01-18 without validating it across the full Sc0-Sc7 catalogue and
  getting explicit user sign-off.**

### 1.3 Notebooks 21-24 — written, unexecuted, paused
All four exist as valid, unexecuted `.ipynb` files (zero outputs,
`execution_count: null` everywhere) at `notebooks/system_I/21_open_loop_nuts_ground_truth.ipynb`,
`22_sbi_full_convergence_study.ipynb`, `23_nuts_differentiable_warmstart_fix.ipynb`,
`24_cnn_vs_handcrafted_discrepancy.ipynb`. Their underlying `scripts/build_nb_system_I_2{1,2,3,4}.py`
are finalized, correct code, previously executed via standalone-script runs (not
through the saved `.ipynb`) to produce the results already summarized in the memory
file — but ALL of those results were computed at the BUGGY (98%-conversion)
operating point and need re-verification once a validated corrected operating point
exists. Do not re-run them yet (see §3).

Per-notebook status, from the memory file (check there for full detail):
- **nb21** (open-loop NUTS fix): partial. Found that stacking an open-loop `mode`
  parameter with the full-covariance likelihood substantially improves R-hat
  (11.1→2.3 alpha, 14.8→7.6 beta on one test window) but hadn't cleared the <1.01
  bar. Status: incomplete, not resumed since.
- **nb22** (SBI budget/SBC study): COMPLETE at the buggy operating point.
  Recommended n_sims=50,000 (SBC + anti-shrinkage slope + correlation-convergence
  all agree at that budget). Will need to be entirely redone at the corrected
  operating point once one is validated — the budget that's optimal at 98%
  conversion may not be optimal at 47%.
- **nb23** (NUTS differentiability fix): the `ConcretizationTypeError` crash WAS
  fixed (JAX-native bisection + `jnp.where` replacing `scipy.optimize.brentq`/Python
  `if`), verified correct (~1e-11 agreement with old implementation) and
  differentiable (`jax.grad` succeeds). **But the real decisive NUTS test came back
  with R-hat~500/347 — catastrophically non-converged, a NEW unresolved problem.**
  STOPPED by user decision before diagnosis. Two untested hypotheses recorded in
  the memory file: (a) differentiating through a fixed-iteration unrolled bisection
  gives a technically-valid but poor-quality gradient (fix: implicit differentiation
  via the implicit function theorem, `jax.custom_vjp`); (b) `jnp.where` still
  backpropagates through BOTH branches, so a NaN in the unselected branch could
  poison the gradient even though never selected in the forward pass — flagged as
  at least as plausible as (a), especially since at Sc4 the selected non-saturated
  output doesn't even depend on the bisection values, only the branch condition, so
  (a) alone doesn't obviously explain Sc4's blowup.
- **nb24** (CNN vs hand-crafted investigation): H1 (CNN SBC/slope) and H2 (held-out
  log-density) COMPLETE — found the CNN-embedding posterior is severely
  uninformative about alpha (slope≈0.05 vs hand-crafted's ≈0.99) and fits held-out
  data ~5 nats worse than the hand-crafted summary, consistently across seeds. H3
  (5-seed CNN robustness) INCOMPLETE — hit 2 reproducible OS-level mach-port-
  exhaustion crashes (SIGKILL, "allocating too many mach ports", confirmed a real
  kernel-level resource exhaustion in the shared `simulate_for_sbi`/
  `simulation_wrapper_sbi_raw` code path used throughout the project, not something
  specific to this notebook — worth a dedicated fix later, e.g. periodic
  `jax.clear_caches()` or subprocess isolation per sample). H4 never run. **All
  findings here were computed at the buggy operating point and flagged as needing
  re-verification once the k0 fix lands** — nb25 found alpha-sensitivity is 14-16x
  different at the corrected point, which could change whether the CNN's apparent
  alpha-collapse is really an architecture problem or partly a signal-starvation
  problem shared by any representation at the wrong operating point.

---

## 2. User-supplied analysis to plan against (verbatim reasoning, already agreed)

The user supplied this reasoning (from an external review) and asked to plan against
it before acting further. It explains the puzzles found so far:
- Fixed-IC runs lose the steady-state information that carries both alpha and beta
  components, so both vanish — consistent with why removing the matched warm start
  collapsed the correlation to ~0 in earlier rounds.
- A diagonal FIM can't see a cross-channel noise term, so it predicted ~0 — matches
  "Theory E"'s original finding exactly.
- Operating-point dependence: at 98% conversion alpha barely moves C, so the physics
  ridge is only weakly pinned along alpha and the noise cross-term can flip the
  sign; at 47% conversion alpha's signal is ~14x stronger, so the two effects
  roughly cancel. This matches notebook 26's result exactly.

**What the user wants done with this, in order:**

1. **Don't report the net correlation as the headline number.** A net Corr≈-0.03 at
   the corrected point sounds like "alpha and beta are independent," but it's really
   two strong effects cancelling. Instead report: posterior eigenvalues/eigenvectors
   (or marginal widths), plus the explicit physics/noise decomposition already in
   notebook 26. Then check the ACTUAL posterior shape with NUTS at that point —
   when two opposite effects cancel, the posterior is often not elliptical (a cross
   or a tilted banana), and a scalar Corr hides that.
2. **Show how much the sign depends on the noise model.** The noise component scales
   with the assumed sensor noise (especially on T, which feeds the controller) and
   with process noise. So the net correlation's sign is not a property of the plant
   alone. Produce a figure: net Corr vs. sigma_T (and separately vs. process-noise
   level), at BOTH operating points, with the noise-free structural value (+0.92/
   +0.95) drawn as a reference line. Expect zero-crossings at different noise levels
   per operating point — this figure should make the whole story legible in one
   plot.
3. **Make sure the decomposition is clean** (three checks):
   - Confirm the noise-free-controller runs used the SAME matched warm start and the
     SAME observation noise as the noisy-controller runs — only the controller's
     input should have changed. Audit the code, don't just assume notebook 26 got
     this right.
   - Check whether PROCESS noise (not just sensor noise) also contributes to the
     negative component, since process noise also reaches C and Qc, not just
     through the sensor-noise-on-T path.
   - **Agreement check**: since the +0.9/-0.9/~0 figures so far are all FIM-based
     (local/Laplace), spot-check at least one point with NUTS so the local and
     global answers are known to agree — THIS IS BLOCKED on notebook 23's unresolved
     R-hat~500 regression (see §1.3). Either fix NUTS first, or use an alternative
     ground-truth check (e.g. a brute-force grid/rejection-sampling posterior at a
     single cheap point) as a substitute.
4. **Retrain SBI only at the corrected operating point.** The prediction is now
   explicit (net close to zero, strong structural ridge underneath) — SBI should
   either reproduce this or reveal a learning problem. Compare against NUTS (if
   fixed) using TARP or C2ST for joint-dependence (not just scalar Corr). This makes
   the earlier "does correlation converge with budget" question testable rather than
   an open worry, but should be done AT THE CORRECTED OPERATING POINT, which itself
   needs validating across the full Sc0-Sc7 catalogue first (per §1.2's caveat).

---

## 3. Sequenced plan to execute (in a fresh agent session)

**Step 0 — validate the corrected operating point across the full catalogue (if not
already done when this is picked up).** Extend notebook 26's Tsp-correction logic
to check non-saturation across all 8 Sc0-Sc7 scenarios, not just 2-3. If the current
Tsp=300.3K choice saturates anywhere in the catalogue, find a better choice (or
accept that a single Tsp can't avoid saturation everywhere and report that
explicitly — may need feed/controller-gain changes too, not just Tsp). Get explicit
user sign-off on the final choice before treating it as "the" corrected operating
point for anything downstream.

**Step A — notebook 27: decomposition diagnostics + noise sweep** (cheap, FIM-only,
should run in minutes like notebook 26). Implements items 1-2 and the first two
sub-items of item 3 from §2:
- Protocol-consistency audit of notebook 26's noise-free-controller implementation
  (confirm warm-start and observation-noise model were held fixed, only the
  controller's input changed) — read the code, fix if a real discrepancy is found.
- Add a process-noise-only ablation (remove process noise but keep sensor noise,
  and vice versa, in addition to the existing fully-noise-free variant) to isolate
  which noise source(s) drive the negative component.
- Replace/augment scalar Corr reporting with posterior eigenvalues/eigenvectors
  (ridge direction + curvature) and marginal widths, at both operating points.
- Produce the headline figure: net Corr vs. sigma_T and vs. process-noise level, at
  both operating points, with the structural (noise-free) value drawn as a
  reference line — looking for the predicted zero-crossings.

**Step B — decision/dependency: NUTS.** The "agreement" check (item 3's third
sub-item) and the retrain comparison (item 4) both need a working NUTS. Notebook
23 stopped with a real, undiagnosed R-hat~500 regression. Before Step C, EITHER:
(a) resume notebook 23's diagnosis (test the two recorded hypotheses: implicit
differentiation vs. jnp.where NaN-leakage from the unselected branch — see §1.3),
and fix it properly, OR (b) substitute a cheaper one-off ground-truth check (e.g. a
brute-force grid-based or rejection-sampling posterior at a single point, avoiding
NUTS/gradient issues entirely) if the user wants to avoid re-investing in the NUTS
fix right now. **This is a decision point for whoever picks this up — ask the user
if it's not already been decided when this file is read.**

**Step C — notebook 28: retrain SBI at the (validated) corrected operating point**,
comparing against whichever ground truth Step B provides, using TARP or C2ST for
joint-dependence assessment (not just scalar Corr). This directly tests item 4 and
makes the earlier budget-dependence question (nb22's finding that correlation
magnitude was still drifting at n=40,000-50,000 even after the warm-start fix, at
the BUGGY operating point) testable again at the corrected point.

**Step D (later, not yet scoped in detail) — once Steps A-C give a validated,
trustworthy picture at the corrected operating point**: plan the actual re-run scope
for notebooks 01-18 (the foundational System I notebooks — data generation, original
production SBI training, the manuscript's Table VI/VII/identifiability numbers),
since ALL of them currently use the buggy `K0_NOMINAL`. This is a large undertaking
and should only be scoped once Steps A-C have produced a stable, validated
corrected-operating-point story.

---

## 4. House rules to carry forward (already established this investigation, don't relitigate)

- Never round to a convenient conclusion; always report what's NOT resolved
  explicitly (see the memory file's own repeated emphasis on this).
- Report distributions across seeds, not single numbers, wherever training is
  involved.
- Cost-check before any expensive step; self-caffeinate anything long-running;
  use disk-flushed progress-sentinel files for real-time visibility (nbclient
  buffers print output until a run finishes, which has caused multiple
  multi-hour-stall diagnosis problems this investigation — see the memory file's
  entries on mach-port exhaustion, oversubscription, and `Pool.apply_async(...).get(timeout=...)`
  not actually freeing stuck workers).
- Build via standalone Python scripts for anything with real compute (not
  nbclient directly), then assemble/write the `.ipynb` as a separate, fast,
  non-executing step if an unexecuted artifact is wanted, or execute via nbclient
  only for genuinely fast notebooks.
- Do not touch `main.tex`, the SI, or `article_outline_CChE.md` from any of this
  work — all of it stays notebook/results-only pending explicit user decisions
  about what (if anything) gets promoted to the manuscript.
