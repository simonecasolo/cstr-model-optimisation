# System II SBC Repair — Directions for Next Steps

**Status: diagnostic, not a plan of record.** Written after reading `nb35`–`nb38` and
the executed outputs currently in `results/38_sysII_*`. Nothing here should be treated
as decided; it's a menu, ranked by evidence strength, for the next session to pick from.
Cross-reference `HANDOFF.md` ("System II matched ongoing-degradation calibration
failure diagnosis") for the authoritative status log — update that file, not this one,
once a direction is actually tried.

**Update 2026-08-27 (`nb39_wu2003_identifiability_and_sbc_repair.ipynb`, executed):**
Section 3c (identifiability re-check) and a recombination of Sections 3a/3b have now both
been run. Headline results, full detail in the notebook's Part I/II verdict cells:

- **The `xi_reb`/`z_A0_eff` pairing this doc's Section 3c proposed checking is NOT
  identifiability-confounded** — noise-calibrated FIM off-diagonal stays in `-0.003` to
  `+0.11` across three representations (compute_summaries, subwindow, raw trajectory),
  an order of magnitude below the article's own vetted `(alpha, beta_r)` benchmark. Do not
  pursue this pairing further.
- **`(alpha, z_A0_eff)` away from nominal is a real, representation-robust coupling**
  (off-diagonal `+0.39` to `+0.68`, does not collapse going to raw trajectory, both points
  comfortably interior to their priors) — the first identifiability finding in this
  investigation that clears the project's own "survives combination and richer
  representation" bar. Treat as provisional pending replication at more operating points.
- **`(eta_col, z_A0_eff)` at nominal also shows a large off-diagonal, but nominal `eta_col=1.0`
  sits exactly on `eta_col`'s prior upper bound** — likely at least partly a boundary
  artifact, not yet trustworthy. Re-run at an interior `eta_col` point before citing this one.
- **Section 3a/3b's recombination (fixed-y0 warm start × exact induced logit prior) did NOT
  produce a stable improvement** — none of 4 arms × 3 seeds cleared `p>0.05`, and the
  "combined" arm this doc predicted would be the best bet was in fact the worst. A
  same-posterior, different-eval-seed comparison also showed a `0.070→0.043` swing from SBC
  evaluation noise alone at this budget — reinforcing Section 3e's point that single-seed,
  reduced-budget numbers in this whole investigation (including nb38's) are not far from
  noise and shouldn't be over-read.

**Revised next step:** neither the identifiability hypothesis (as originally framed around
`xi_reb`) nor the two training/geometry fixes tested so far explain the failure. The most
promising untried direction is narrower than either: a fix targeted specifically at the
`(alpha, z_A0_eff)` coupling (e.g. correlated-block reparameterisation, or features built
from the raw-trajectory directions Section 3's FIM check found retain signal for this pair)
rather than a global change applied uniformly to all 5 parameters.

**Update 2026-09-01 (`nb40_wu2003_alpha_zA0_confirmation.ipynb`, executed; both nb39 and
nb40 subsequently patched and re-executed):** the follow-up built to visually confirm
`(alpha, z_A0_eff)` and replicate it across more operating points instead found a
methodological bug: nb39/40's FIM reporting used the raw, normalised *information*-matrix
off-diagonal (`fim_utils.offdiag_ratio`) as if it were a posterior *correlation* — these can
have opposite signs for a multi-parameter FIM (confirmed: raw `-0.90` vs. actual trained-
posterior correlation `+0.93` at one point). Fixed by adding `fim_utils.cov_offdiag_corr`
(correlation from the *inverted* FIM, the quantity actually comparable to a posterior); both
notebooks patched to use it and re-executed. **The numbers above (`+0.39` to `+0.68`,
`-0.003` to `+0.11`, etc.) are the pre-fix, wrong-metric values — superseded, do not cite
them.** Corrected picture:

- **`(xi_reb, z_A0_eff)`: still ruled out** — this pair's near-zero result was small enough
  in magnitude that the sign bug didn't change the conclusion.
- **`(alpha, z_A0_eff)`: now confirmed by two independent methods** (corrected FIM and the
  real trained posterior's empirical correlation agree in sign and relative ordering) at
  every point where `alpha` is meaningfully away from nominal. The strongest
  cross-validated finding in this whole investigation so far.
- **`(eta_col, z_A0_eff)` at nominal also survives the corrected representation check**
  (`+0.76 → +0.52`, compute_summaries → raw trajectory) — but the boundary caveat
  (`eta_col=1.0` sits exactly on its prior's upper bound) is unchanged and still unresolved.
- **A new, narrower discrepancy**: exactly at `alpha=nominal` (nb39's degraded point,
  nb40's `P1`), the corrected FIM says strongly negative (`-0.6` to `-0.7`) while the real
  posterior says weakly positive (`+0.26`–`+0.28`) — sign still disagrees there. Recommended
  tiebreaker: a deterministic `identifiability_scan.run_scan` grid-scan at that exact point
  (distance-based, not subject to the FIM-inversion issue) — not yet run.

**Doubly-revised next step:** the `(alpha, z_A0_eff)` fix-targeting recommendation above
stands and is now on firmer footing. Before implementing a fix, resolve the two remaining
open items: (1) the `alpha=nominal` FIM-vs-posterior disagreement via the grid-scan
tiebreaker, and (2) the `eta_col=1.0` boundary re-test at an interior point (e.g. `0.80`–
`0.85`) — `eta_col`'s coupling with `z_A0_eff` may turn out to be a second, independent
target requiring its own fix, not just `(alpha, z_A0_eff)`.

## Update 2026-09-01 (`nb41_wu2003_confound_tiebreakers.ipynb`, executed) — final identifiability picture, decision point reached

Both open items above are now closed, deterministically (no training involved):

1. **`alpha=nominal` tiebreak resolved against the posterior.** A model-free grid-scan ridge
   is negative at both test points (`-2.06`, `-2.30`), matching the corrected FIM, not the
   uncalibrated posterior's weak positive value. The ridge visibly bends into a different,
   flatter segment once `alpha` drops below ~0.65-0.7 — independently reproducing the sign
   flip `nb40` found by a different method. **`(alpha, z_A0_eff)` is a genuinely curved,
   sign-changing degenerate manifold**, confirmed on both sides of the flip by two
   independent methods. A rotation/correlated-block reparameterisation fix will not work
   here — the ridge direction itself changes across the region it needs to cover.
2. **`eta_col` boundary re-test: not an artifact.** At an interior point (`eta_col=0.85`) the
   coupling with `z_A0_eff` is as strong or stronger than at the boundary and the grid-scan
   shows a real, compact valley at truth. **Confirmed as a second, independent, genuine
   limit** — and unlike `(alpha, z_A0_eff)`, single-signed (positive) everywhere tested.

### Where this leaves the "way forward" decision

Combined with nb38/39 Part II's exhaustive sweep of training/geometry fixes (preprocessing,
warm-start, logit-space reparameterisation, exact induced priors, and their recombination —
all tried, none worked), the evidence base is now as complete as cheap deterministic checks
can make it, without spending further SNPE training budget chasing a training explanation
that has been ruled out twice over. **Recommendation: stop looking for a calibration fix for
`alpha` and `z_A0_eff`, and choose between the two legitimate paths already identified**:

- **(a) Reduced-dimension posterior + explicit reported limitation.** Fit/report a posterior
  over the subset that isn't implicated (`beta_r`, `xi_reb`, confirmed clear) plus report
  `(alpha, z_A0_eff)` and `(eta_col, z_A0_eff)` as genuine, physically-explained,
  multiply-confirmed identifiability limits of the S-B instrumentation — analogous to how
  `(alpha, beta_r)` was framed for System I, but on considerably firmer methodological
  footing here (checked against grid-scan, linearised FIM, raw-trajectory representation,
  *and* an actual trained posterior, at multiple operating points, with an identified and
  fixed metric bug along the way). This is a legitimate, defensible scientific contribution,
  not a fallback.
- **(b) Name the missing instrumentation.** Both confounds trace to the same root cause: S-B
  (and S-A) have no reactor-side composition sensor, so nothing distinguishes "the reaction
  ran faster" (`alpha`) or "the column recovered less product" (`eta_col`) from "less
  reactant came in" (`z_A0_eff`) via reactor-composition-driven channels alone. Worth stating
  explicitly as the concrete sensor addition that would resolve it, even if adding and
  re-validating a new instrumentation structure is out of scope for the current manuscript
  cycle.

Both are compatible — (a) is the manuscript-ready outcome, (b) is the explanatory mechanism
that makes (a) a genuine finding rather than a shrug. Neither requires further SNPE training
experiments; the remaining work is manuscript-side (writing up the finding) and, if pursued,
verifying the reduced-dimension posterior's own SBC at `N_SBC>=400` before reporting it.

## 1. Where things stand after nb38

Three repair experiments ran against the 5k true-simulator lognormal-prior bank
(`wu2003_sbi_train_sb_lognormal_true_n5000.npz`), each compared to the nb37 baseline
(matched warm start, identity preprocessing, min-KS `p = 2.21e-3` at best seed):

| Experiment | Best seed min-KS p | Verdict |
|---|---|---|
| Preprocessing sweep (`identity`/`robust_clip_scale`/`signed_log_robust`/`drop_heavy_tail`) | `1.10e-3` (`drop_heavy_tail`, seed 0) | **No improvement.** All 4 variants × 3 seeds fail by orders of magnitude. |
| Fixed-nominal warm start (vs. per-draw matched steady state) | `2.54e-2` | **Partial improvement**, still fails 0.05. |
| Logit parameter-space training (`u = logit((θ−low)/(high−low))`) | `7.03e-2` — **only formal pass among all 38-series runs** | **Best single result, but not stable**: seed 1 → `2.0e-5`, seed 2 → `3.0e-6`. 1/3 seeds passing is not evidence of calibration (see [[finding_sbc_power_eta_col]] — small-N, single-seed passes are exactly the failure mode already burned once on this project). |

Two things stand out that should anchor whatever is tried next:

1. **Preprocessing is now ruled out as the primary lever.** Outlier scaling/clipping of
   `V_norm*`/`Q_j*` was the cheapest, most mechanical hypothesis, and it produced zero
   movement relative to `identity`. Whatever the remaining problem is, it is not a
   simple feature-scaling artifact — don't spend more budget on preprocessing variants
   without a new idea about *why* one would help.
2. **The two experiments that did move the needle both touch the training/parameter
   geometry, not the observation vector**: removing the per-draw steady-state warm
   start (simpler geometry between θ and y0) and reparameterizing to unconstrained
   `u`-space (removing bounded-support pileup near the prior edges) each gave a
   ~10–30× improvement in best-seed p-value over the matched/identity baseline. That's
   a meaningful signal even though neither is sufficient alone.

## 2. Most-likely root cause, given the evidence so far

The pattern (bounded reparameterization helps a lot, warm-start simplification helps a
lot, preprocessing does nothing, and `xi_reb`/`z_A0_eff` are the recurring worst
offenders per nb35/nb37/nb38) is most consistent with:

> **A composition of (a) genuinely weak identifiability for `xi_reb` and `z_A0_eff`
> under the S-B channel set, and (b) training/geometry instability that both the warm
> start and the bounded support make worse.** Neither cause alone explains why fixing
> either one only partially helps — this looks like two problems stacked, not one.

This is consistent with the earlier [[project_wu2003_sbi_calibration]] finding that
S-A's calibration failure was a genuine identifiability problem, not a network-capacity
problem, and [[feedback_network_sizing]]'s finding that architecture size is not the
lever here either — worth keeping in mind before reaching for "bigger network" as a fix.

## 3. Directions, ranked by evidence-to-cost ratio

### 3a. Combine the two partial wins before trying anything new (cheapest, do first)

nb38 tested warm-start-fix and logit-reparameterization **separately**. Neither has
been tried **together**, and both are compatible (logit-space training with a fixed
nominal warm start). Given each independently produced the two best results in the
notebook, this is the highest-expected-value next experiment and requires no new
infrastructure — it's a recombination of code already written in nb38 §4 and §5.

- Train the logit-`u` scaffold (nb38 §5) using the fixed-y0 bank (nb38 §4) instead of
  the matched bank.
- If this clears 2+/3 seeds at the current reduced budget (`N_SBC=120`, `N_POST=120`),
  it's the first result in this notebook worth a `N_SBC≥400` confirmation run per
  [[finding_sbc_power_eta_col]] before calling it real.

### 3b. Fix the logit-space prior properly (cheap, addresses a real gap in nb38 §5)

The current scaffold approximates the induced prior in `u`-space with an **empirical
Gaussian fit** to the training sample (`prior_u = Independent(Normal(u_mean, u_std))`,
nb38 cell 16) rather than the true induced density of a uniform-in-θ prior pushed
through the logit transform. That mismatch means SNPE's `prior` argument doesn't match
the density actually used to generate training data, which biases the learned
posterior in exactly the way that could produce the seed-to-seed instability observed
(seed 0 passes, seeds 1–2 fail badly). Two fixes, in order of effort:

1. Derive the closed-form induced density for `u = logit((θ-low)/(high-low))` when θ
   is uniform (it's a logistic-shaped density with known Jacobian — `duplomax
   |dθ/du| = (high-low) · sigmoid(u) · (1-sigmoid(u))`), and pass a proper
   `torch.distributions.TransformedDistribution` as `prior` instead of the Gaussian
   surrogate.
2. If (1) is fiddly with `sbi`'s prior-checking utilities, at minimum widen/refit the
   Gaussian surrogate more carefully (e.g. moment-match the true logit-uniform density
   analytically rather than from a finite training sample) and check sensitivity to
   the surrogate's tail behavior — `z_A0_eff` and `eta_col`'s `u`-space std in the nb38
   output (1.05–1.23) suggests non-trivial spread that a poorly-tailed Gaussian could
   clip.

This is a plausible explanation for **why** logit-space only worked for 1/3 seeds
rather than being a clean win, and is worth fixing before concluding "logit space
doesn't work."

### 3c. Diagnose `xi_reb` / `z_A0_eff` identifiability directly, not just via SBC pass/fail

nb35/37/38 consistently flag these two parameters as the worst offenders but no
notebook in the 35–38 series has run the deterministic identifiability-scan
methodology (`scripts/identifiability_scan.py`) for this pair, the way it was done for
`(alpha, eta_col)` and `(alpha, beta_r)` in the earlier S-A/S-B investigation (see
[[finding_alpha_beta_r_banana]] and [[finding_sa_headline_retracted]]). That scan
directly answers "is this a real degenerate ridge in the S-B channel set, or a
training/geometry artifact?" — which SBC alone cannot distinguish (that's the exact
gap that caused the `(alpha, eta_col)` banana to be reported and later retracted).
Concretely:

- Run the identifiability scan over `(xi_reb, z_A0_eff)` and each against `alpha` at
  the matched-protocol scenario(s) used for the SNPE bank.
- If a genuine flat/degenerate direction shows up in the deterministic simulator (no
  neural network involved), that's strong evidence the SBC failure is a real
  identifiability limit of the S-B instrumentation for these two parameters, not a
  training artifact — which would justify reducing the inference target (see 3d) as a
  legitimate finding rather than a workaround.
- If no degenerate direction shows up deterministically, that redirects effort back to
  training/geometry (3a/3b) with more confidence, rather than continuing to guess.

This is the notebook's own §6 "Decision Guide" 4th bullet ("All fail → reduce to 3-D or
4-D subsets") but done with the deterministic tool that actually diagnoses *why*,
before spending more SNPE training budget on ablations.

### 3d. Reduced-dimension posterior as a fallback, not a first move

nb38's own decision guide already proposes this: fit a 3-D or 4-D posterior over
`{alpha, beta_r, eta_col}` (dropping `xi_reb`, `z_A0_eff`, or both) to check whether the
remaining parameters calibrate cleanly once the two problem dimensions are removed.
Useful as a bisection tool to confirm 3c's hypothesis cheaply (no identifiability-scan
script needed), but **do not treat a reduced posterior as a manuscript result** without
first understanding whether the dropped parameters are truly unidentifiable (3c) or the
training instability (3a/3b) was the real cause and would have also broken the reduced
posterior eventually.

### 3e. Longer shot: more SBC-seed robustness before scaling the bank

Every experiment in nb37/38 uses only 3 seeds at reduced budget
(`N_SBC=120, N_POST=120`). Given [[finding_sbc_power_eta_col]]'s finding that N=200 SBC
already produces false passes that don't replicate at N=400, a single seed passing at
`N_SBC=120` (the logit-`u` seed 0 result) is weaker evidence than it looks — closer to
noise than confirmation. Before investing in a 15k-sample full bank for any winning
recipe, budget for **5+ seeds at N_SBC≥400** on the *reduced* bank first. This is cheap
insurance against repeating the exact false-positive pattern already documented for
`eta_col` under S-A.

## 4. What NOT to do (already ruled out or contraindicated)

- **Don't try bigger/smaller network architectures.** [[feedback_network_sizing]]
  already showed both directions hurt calibration on this problem; there's no reason
  to expect a different result for System II's 5-D posterior over 66-D summaries.
- **Don't trust a single passing seed at reduced SBC budget as "solved."** This is the
  exact mistake documented in [[finding_sbc_power_eta_col]] — always confirm at
  `N_SBC≥400` with multiple seeds before promoting a posterior or updating manuscript
  numbers.
- **Don't spend more budget on summary preprocessing variants** (clipping, log
  transforms, feature dropping) — nb38 §3 tested the four most obvious variants and
  none moved the needle. If a future idea targets a *specific* summary-vector defect
  (e.g., a physics-informed feature that's actually wrong, not just heavy-tailed),
  that's different from another generic scaling sweep and would be worth trying, but
  "scale the outliers differently" as a category is exhausted.
- **Don't mix fixed-y0 training with matched-protocol validation/manuscript numbers**
  — nb38 §4's own interpretation note already flags this; fixed-y0 is a geometry
  diagnostic only, never a substitute for the matched protocol the manuscript commits
  to (Major Comment 7).

## 5. Suggested next-session order of operations

1. Run 3a (recombine fixed-y0 + logit-u) — cheapest, uses existing code, highest
   expected value given both partial wins are compatible.
2. If 3a's result is promising but still seed-unstable, do 3b (fix the induced prior)
   before concluding logit-space doesn't work — the current surrogate prior is a known
   approximation, not the real thing.
3. In parallel (cheap, deterministic, no GPU/training time), run 3c on `xi_reb`/
   `z_A0_eff` to get an independent, non-SBC-based answer to "is this identifiable at
   all under S-B" — this de-risks every other direction by telling you whether you're
   chasing a training bug or a physical limit.
4. Only after 1–3, if nothing clears calibration, fall back to 3d (reduced-dimension
   posterior) as a documented, deliberate scope reduction rather than a silent one.
5. Whatever wins, confirm with 3e's larger-seed/larger-N protocol before touching
   `HANDOFF.md`'s "known failure" status or any manuscript number.
