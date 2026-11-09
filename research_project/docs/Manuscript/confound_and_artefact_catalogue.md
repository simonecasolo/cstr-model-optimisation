# Confound and artefact catalogue: every parameter-pair identifiability issue found in this project, explained

**Purpose.** A single reference for every confound, artefact, and parameter-pair
identifiability issue discovered across both systems in this study — what it is,
why it exists physically/structurally, how it was tested, and what it means for
practical fault identifiability. Written so a reader can look up any pair
(`alpha`/`beta`, `alpha`/`z_A0_eff`, etc.) and get the full story without
re-deriving it from the manuscript or the notebooks.

**Vocabulary used throughout**, because "confound" gets used loosely elsewhere
in this project's notebooks and needs a precise meaning here:

- **Structural non-identifiability**: two parameters appear in the model
  equations *only* as a fixed combination (e.g. a product). No measurement,
  however precise or extensive, can separate them — this is a property of the
  equations, not of any estimator or dataset.
- **Genuine (representation-robust) confound**: two parameters are not tied by
  a structural identity, but the *physically available measurement channels*
  cannot distinguish their effects at the operating points tested. Confirmed by
  checking that the confound **survives** a move from compressed summary
  statistics to the raw, unaggregated sensor trajectory of the same channels
  (the raw-trajectory Fisher information, "RT-FIM", check used throughout this
  project). This is a property of the *instrumentation*, fixable only by adding
  a measurement, not by better feature engineering or more training data.
- **Representation artefact**: two parameters look confounded under a specific
  compressed feature set (e.g. whole-window means, or a restricted channel
  subset) but the confound **collapses** once the full available channel set or
  raw trajectory is used. This is a property of the chosen *summary statistics*,
  fixable by better features or raw-trajectory access — the underlying physical
  signal already contains the distinguishing information.
- **Local-sensitivity asymmetry (not a confound)**: one parameter is harder to
  estimate precisely than another, without the two being correlated or
  confused with each other. A real practical-identifiability issue, but a
  different phenomenon from a confound (it concerns `diag(FIM)`, not an
  off-diagonal term).

---

## Part A — System I (propylene oxide CSTR, closed-loop PI temperature control)

### A1. `beta` × `UA`: exact structural non-identifiability (the only *true* structural confound in this project)

**What:** the jacket heat-transfer degradation factor `beta` and the clean-service
heat-transfer coefficient `UA` appear in the energy-balance equations
(`main.tex` Eqs. `dT_faulted`/`dTc_faulted`) **exclusively as the product**
`beta · UA`. No other term in the model equations separates them.

**Cause:** this is a modelling choice, not a numerical coincidence — jacket
fouling and a lower nominal heat-transfer coefficient have an identical
physical effect on the energy balance (both reduce the achievable heat flux
for a given temperature difference), so the governing equations cannot
distinguish "a dirtier jacket" from "a jacket that was never very good to
begin with."

**Resolution used:** `UA` is treated as a known design constant
(`UA = 12{,}500` cal·(min·K)⁻¹, clean-service value), and the inference target
is redefined as the *relative* degradation factor `beta`. This sidesteps the
non-identifiability rather than solving it — it is unsolvable as stated, by
construction (Ljung's sense of structural non-identifiability).

**Expected effect on identifiability:** none, once resolved this way — `beta`
alone is a well-posed inference target. This entry exists in the catalogue
because it is the conceptual baseline every other entry should be compared
against: a confound that is provably permanent (equation structure), as
opposed to every other confound below, which is contingent on instrumentation
or representation choices and is therefore at least in principle fixable.

---

### A2. `(alpha, beta)`: **not** a confound, despite superficial resemblance to System II's banana

**What was checked:** whether System I's two fault parameters — catalyst
activity `alpha` and jacket conductance `beta` — show the kind of near-perfect
posterior correlation ("banana") System II's `(alpha, beta_r)` pair shows at
scenario W11.

**Result: no.** At the directly analogous scenario (Sc4, both parameters
moderately degraded, no actuator saturation), the pooled trained-posterior
correlation is small (not the near-unity ridge System II shows). Confirmed two
independent ways: the trained SNPE posterior itself, and a fresh local Fisher
information matrix (Laplace-approximation correlation) at both the healthy and
Sc4-equivalent operating points (`notebooks/34_alpha_beta_r_banana_cross_system.ipynb`).

**Why not, mechanistically — the cause is the inverse of System II's Artefact 2
(B2 below), which is the point of comparing them:** both systems share the
identical closed-loop masking mechanism (PI integral control zeroes the
steady-state temperature sensitivity to the thermal-fault parameter, so
`beta`/`beta_r` is only observable indirectly through the actuator). What
differs is that **System I retains a reactor-composition sensor** —
concentration `C` is measured directly, and `C`'s governing equation
(`dC_faulted`) contains `alpha` but **not** `beta` at all. `C` is therefore a
mechanism-pure channel for `alpha`, entirely independent of the
actuator-shared pathway (`Q_c`) that would otherwise create the same ambiguity
System II has for `(alpha, beta_r)`. Confirmed causally, not just by
elimination: stripping `C`'s features (and the `k0_eff_proxy` physics feature
built from it) out of System I's own 29-D summary set makes its
`(alpha,beta)` correlation jump roughly 9× at the same operating points — i.e.
removing System I's composition sensor reproduces System II's degeneracy
mechanism on demand.

**Expected effect on identifiability:** none under non-saturating operation —
`alpha` and `beta` are separately, if asymmetrically, identifiable (see A3).

---

### A3. `(alpha, beta)` local-sensitivity asymmetry — a real practical-identifiability issue, but not a confound

**What:** `alpha` is substantially easier to estimate precisely than `beta`.
The local Fisher-information diagonal ratio `I_alpha,alpha / I_beta,beta`
(full 29-D summary representation, healthy operating point) is of order
10²–10³ depending on exact FIM methodology used (see the caveat below), and
remains strongly operating-point-dependent — e.g. roughly an order of
magnitude smaller at the Sc2 jacket-fouling condition than at the healthy
point.

**Cause:** the same PI-masking mechanism as A2 — `beta`'s steady-state
signature is suppressed by integral control, leaving only indirect,
noisier/less-sensitive secondary channels (`T_c`, `Q_c`, saturation behaviour)
to carry its information, whereas `alpha` has the direct, high-SNR
concentration channel `C`.

**Methodological caveat, found this project (`notebooks/33_fim_cross_system_validation.ipynb`):**
the diagonal-Sigma FIM methodology used elsewhere in this project for
collapse/no-collapse artefact tests (System II's Artefacts below) gives a
*much smaller* `I_aa/I_bb` ratio (~2–7×) for this exact pair than the
full-covariance methodology that originally established the ~250–500×
headline figure. Both are reproducible, correct computations — the
discrepancy is a genuine property of the two FIM conventions (diagonal vs.
full noise covariance), not a bug. **Do not compare System I's diagonal-magnitude
deficit against System II's using different FIM conventions as if they were
on the same scale** — the qualitative conclusion (System I's masking is more
severe) is still very likely correct given the very different channel-sharing
mechanisms in the two plants, but the specific multiplier is methodology-dependent.

**Expected effect on identifiability:** `beta`'s credible intervals are wider
and more sensitive to operating point, noise, and representation than
`alpha`'s; under non-saturating conditions this does not prevent correct
*fault-class* recovery (Sc2's `beta` is accurately estimated, Table
`tab:scenario_results`), but it does mean `beta`'s *precision* is the weaker
link in System I's joint posterior, and degrades further under actuator
saturation (A4).

---

### A4. Sc5 actuator saturation: a nonlinear `(alpha, beta)` correlation, mechanistically unrelated to System II's banana

**What:** at severe fouling (Sc5, `beta*=0.40`), the coolant valve saturates
(`Q_c = Q_{c,max}`) for roughly a third of each observation window. The pooled
posterior shows a large-magnitude `(alpha,beta)` correlation at this scenario
(confirmed negative and substantial in the trained posterior).

**Cause:** this is **not** the shared-information-pathway mechanism behind A2
or System II's B2 — it is actuator clipping. Once the valve saturates, the
actuator output carries no further marginal information about how much worse
`beta` has become, and the loss of that one degree of freedom manifests as an
apparent correlation with whatever else the remaining, non-saturated
information can still constrain. A fundamentally different, nonlinear-regime
phenomenon from a representation- or instrumentation-driven confound.

**Expected effect on identifiability:** fault *detection* remains robust (all
replicates correctly classified as fouling-dominant, since posterior mass sits
well inside the fouling region), but *severity quantification* degrades — 90%
credible-interval coverage for `beta` falls to 0.62 at Sc5, a real
under-coverage, not merely wider intervals.

---

### A5. Sc7 temperature-sensor-bias / fault confound — a measurement-model confound, not a parameter-pair one

**What:** Sc7 introduces a fixed +2 K bias in the reactor-temperature
measurement alongside mild jacket fouling (`beta*=0.85`). The posterior shifts
toward the healthy side of the classification boundary (`beta_hat=0.89`
against truth `0.85`), and the Sc7 fault-class F1 drops to 0.67 (90% coverage
for `beta` falls to 0.52).

**Cause:** the PI controller responds to the *biased* temperature
measurement, not the true one — part of the closed-loop actuator response
that would otherwise be attributed to heat-transfer loss can equally well be
explained by the sensor drift. This is a confound between an **unmodelled
measurement artefact** and a **real physical fault**, not between two
inference-target parameters — a structurally different category from every
other entry in this catalogue, included here because it produces the same
practical symptom (biased point estimate, reduced coverage) as a genuine
parameter-pair confound would.

**Expected effect on identifiability:** not a strict non-identifiability of
`beta` under a clean measurement model — it is a practical confound that
exists only because the sensor-bias failure mode was not itself included in
the inference model. Resolvable in principle by explicitly modelling sensor
drift as an additional nuisance parameter, which this study did not attempt.

---

## Part B — System II (Wu 2003 reactor–column–recycle plant, S-B instrumentation)

**Shared starting mechanism for every System-II entry below:** Loop 1's PI
integral control pins the reactor temperature `T_r` at setpoint in steady
state, exactly as in System I, so `d T_r,ss / d(alpha) ≈ d T_r,ss / d(beta_r) ≡ 0`.
Every reactor-side parameter's steady-state signature is therefore
redistributed into the jacket/actuator channels (`T_j`, `Q_j`), creating the
same *a-priori* expectation of a reactor-side confound that System I has — the
entries below are about which of those expected confounds turn out to be real
(survive raw-trajectory access) versus artefactual (collapse under it), and
why.

### B1. Artefact 1 — `(alpha, eta_col)` restricted-channel coupling: a resolved representation artefact

**What:** when the identifiability check is restricted to recycle-flow-derived
features only (`F_R,norm` and features built from it), `alpha` (reactor
conversion) and `eta_col` (column tray efficiency) appear strongly confounded
— both reduce per-pass conversion, driving more unconverted A back through the
recycle loop in the same direction (the "snowball effect"), so a
recycle-flow-only view cannot tell which cause is responsible.

**Cause:** both parameters act on the *same single observable aggregate*
(recycle flow) through physically distinct but directionally identical
pathways — a textbook restricted-channel degeneracy.

**Resolution:** once the already-measured reboiler channels (`T_reb`,
`Q_reb`) are included in the full feature set, the ambiguity disappears —
posterior correlations drop below 0.15 and `eta_col`'s 90% credible interval
narrows to ~0.01–0.02. Consistent with this, the cross-unit scenario W12 is
classified correctly as a compound reactor–column fault in every tested
replicate. **Re-confirmed at additional operating points** in the later S-C
solidification work (`system_ii_sbi_solidification_plan.md`, "Stage 1a" / "Gate
A tiebreaker"): a residual FIM correlation (`|corr|` 0.52–0.66) reappeared at
three *other* operating points under the richer 72-D S-C summary set, but a
full deterministic grid-scan at all three points showed the same
clean-collapse signature as the original W12 retraction — confirming this is
a **local-linearisation ("FIM") artefact, not a genuine residual confound**,
at every point tested so far.

**Expected effect on identifiability:** none, once the full available channel
set is used — this is purely a lesson about not under-featurising, not a real
plant limitation. No sensor addition was or is needed for this pair.

---

### B2. Artefact 2 — `(alpha, beta_r)` temporal-aggregation confound: the W11 "banana" — UPDATE 2026-10-14: does not reproduce under the corrected-physics, properly-trained posterior

**What:** the strongest summary-level confound found in System II. At W11
(`alpha=beta_r=0.80`), the trained posterior correlation is **+0.998** under
the standard 66-D S-B summary representation, and the FIM-normalised
off-diagonal is approximately +0.85 to +0.90 — an almost one-dimensional
ridge.

**Cause, confirmed mechanistically (not just observed):** both lower catalyst
activity and lower jacket conductance are compensated by Loop 1 through the
*same* actuator, `Q_j` — mechanistically identical to System I's A2/A3
situation. What makes System II different is instrumentation: System I avoids
this because it retains a reactor-composition sensor (`C`) whose governing
equation structurally excludes `beta`; System II's S-B/S-A channel sets have
**no analogous reactor-composition sensor**, even though the simulator
internally tracks `z_A` (reactor A mole fraction) — it is computed but never
exposed as an observed channel. Feature-level diagnosis additionally
identified **whole-window aggregation of `Q_j`** as the dominant contributor
to the *summary-level* coupling: both faults produce the same sustained level
shift in `Q_j`, and whole-window mean/std/slope/min/max/q_mean statistics
preserve that common level shift while discarding the transient *shape* that
would distinguish the two underlying causes.

**Status: the headline "resolved, benign" framing is not yet properly
re-verified, and a preliminary check raises real doubt about part of it.**
Originally reported as a **partially**-resolved representation artefact: the
raw-trajectory FIM was reported to collapse the normalised off-diagonal to
~0 (condition number falling from 3.45×10³ to 5.34), suggesting the summary
compression — not the plant — destroys the distinguishing information, and a
retuned EKF with raw-trajectory access does successfully recover both
parameters to <2% error at W11, supporting that reading operationally.
**However:**
- This conclusion predates both the 2026-08-12 `beta_r`/`Q_j` physics fix and
  the later-discovered warm-start/data-contamination bug — it has not been
  recomputed on corrected physics or clean data.
- A dedicated re-audit (`notebooks/42_wu2003_manuscript_fim_reaudit.ipynb`)
  using the corrected, inverted-FIM `cov_offdiag_corr` metric (see B3's note
  on the sign-bug below) **could not reproduce the cited collapse magnitude**
  at W11 (raw-trajectory correlation measured at only +0.05 to +0.28, far
  below the claimed near-total collapse) and **could not trace the originally
  cited numbers to any script or notebook at all** — their provenance is
  currently unknown.
- Separately, a tested raw-trajectory CNN-embedding SBI posterior (as opposed
  to a hand-crafted-feature posterior) **failed** to resolve this pair even
  with full trajectory access (`alpha`-`beta_r` posterior correlation stayed
  above +0.99) — showing that "the raw signal contains the information" (an
  RT-FIM/EKF finding) and "an amortised estimator actually recovers it" are
  two different claims, and only the EKF side of that gap has been confirmed
  for this specific pair.

**UPDATE 2026-10-14 — direct evidence from the first properly-trained,
corrected-physics, classification-validated S-B posterior (not just a
methodology audit of the old numbers).** The full wide-prior retrain
(`scripts/run_sb_wideprior_training.py`, `baseline` z-scoring — the recipe
that also gives working classification, see `system_ii_sb_identifiability_story.md`
§0.4/§7.0) was sampled at W11 across all 5 trained seeds:
`Corr(alpha, beta_r)` = **−0.115, +0.674, +0.296, −0.235, −0.025**. This is not
a near-unity ridge in any seed, and it is not even consistent in *sign* across
seeds. **This is the strongest evidence yet that the manuscript's headline
+0.998 figure does not reproduce under the current best-understood, correctly
trained posterior** — on top of the provenance and magnitude doubts already
raised by the methodology re-audit below. Treat the original +0.998 finding
as **not reproducible with current tooling**, not merely "unverified." Two
readings remain open and should be distinguished before rewriting `main.tex`:
(a) the +0.998 figure really was an artefact of the stale, pre-physics-fix,
single-seed posterior and the confound was never as severe as reported, or
(b) the corrected training recipe (needed to fix classification, per §0.4) has
itself introduced enough seed-to-seed posterior-shape variability that this
specific pairwise correlation is simply not a stable quantity to report
regardless of which seed is "the" production one — in which case the
*reporting practice* (point to one seed's correlation as if it were a plant
property) needs to change, not just the number.

**Expected effect on identifiability, as things currently stand:** if the
original resolved-artefact reading holds up under re-audit, this confound is
**operationally benign** regardless — both `alpha` and `beta_r` are
reactor-unit parameters under the hierarchical fault taxonomy, so even a
posterior that cannot disambiguate them still correctly flags "reactor fault"
(W11 classifies correctly in every tested replicate historically). If the
re-audit instead confirms it as representation-robust, it would need
re-classifying as a genuine confound alongside B3 below — **this is the single
most important open item in this catalogue** (see
`docs/Manuscript/system_ii_sb_identifiability_story.md` §7.1).

---

### B3. Artefact 3 — `(alpha, z_A0_eff)` and `(eta_col, z_A0_eff)`: genuine, representation-robust confounds — the one confirmed real plant-level limit

**What:** `alpha` (catalyst decay) or `eta_col` (column efficiency loss)
cannot be reliably distinguished from `z_A0_eff` (lean fresh feed) under S-B's
instrumentation. At the W2 operating point, the `(alpha, z_A0_eff)` FIM
off-diagonal is large under the 66-D summary (originally reported ≈−0.89,
re-measured under the corrected metric at +0.57 to +0.79 — sign and exact
magnitude differ across methodology versions, but the **qualitative
conclusion is consistent**: large and robust).

**UPDATE 2026-10-14 — partially corroborated, partially complicated by the first
properly-trained posterior.** Sampled across all 5 `baseline`-recipe seeds
(same posterior as B2's update above): `(alpha, z_A0_eff)` at W13
(`alpha=0.80, z_A0_eff=0.80`) is **positive in every seed** (+0.245 to +0.717)
— unlike B2, this one at least has a *consistent sign* across seeds, which is
mild additional support for it being a real effect rather than training noise,
though the magnitude still varies substantially seed-to-seed. `(eta_col,
z_A0_eff)` at W10 (`z_A0_eff=0.78`, `eta_col` nominal) is much weaker and
**flips sign across seeds** (−0.286 to +0.318) — this specific operating point
(W10, a pure feed fault with `eta_col` at nominal) may just not be where this
particular coupling is strong (consistent with this project's repeated finding
that these couplings are point-dependent), so this is not yet a reason to
doubt the pair's status, but it is weaker direct support than hoped from this
check and the W2/interior-point FIM evidence remains the stronger basis for
this pair's "genuine" classification, not this W10 posterior check.

**Cause:** unlike B2, this is **not** a shared-actuator effect — `z_A0_eff`
is not directly regulated by any PI loop. The shared pathway is
**stoichiometric/recycle-structural**: anything that reduces per-pass
conversion (lower `alpha`, worse `eta_col`) increases the unconverted-A load
returned to the reactor via the recycle loop, raising `z_{A,in}` in
qualitatively the same direction a leaner fresh feed does — the same
"snowball" mechanism behind B1, but this time the confound does **not**
collapse when richer channels are added, because the plant genuinely has no
sensor that observes feed composition independently of the mixed, recycled
reactor inlet.

**Confirmed representation-robust 5 independent ways** (multiple operating
points, multiple FIM methodology corrections, multiple notebooks across
2026-09-01 through 2026-09-04 — `notebooks/39`, `40`, `41`, `42`, and the
trained-posterior correlation itself): the raw-trajectory FIM does **not**
collapse this coupling the way it collapses B1 and (per the original report)
B2. `(eta_col, z_A0_eff)` was independently confirmed not to be a prior-boundary
artefact either — re-tested at an interior `eta_col` value away from its
prior's upper bound, the coupling if anything strengthened. **`(xi_reb,
z_A0_eff)` was checked and explicitly ruled out** — stays small and
sign-inconsistent across noise seeds at every tested point, unlike the other
two pairs.

**Important correction this confound required to the manuscript's own
framing:** `main.tex`'s current Artefact 3 section describes this pair using
the same "representation artefact, resolved by raw trajectory" language as
B1 — **that is incorrect for this pair specifically** and needs rewriting,
not just updating with new numbers (`docs/Manuscript/system_ii_sb_identifiability_story.md`
§7.2 / `notebooks/42`'s verdict cell has the suggested replacement framing).

**Resolution path (confirmed at the FIM/deterministic level, not yet at the
trained-posterior level):** a feed-line composition analyzer (the "S-C"
instrumentation extension), placed upstream of the recycle mixing point
specifically (placement is load-bearing — measuring the *combined* reactor
feed re-mixes the ambiguity back in) with relative noise ≤0.01%, collapses
both couplings to near-zero in the deterministic FIM check. **This resolution
has not yet been confirmed at the trained-SNPE-posterior level** — S-C's own
SBI training has independently struggled (a *different*, estimator-level
problem; see the "what remains" discussion for System II, not covered in this
catalogue).

**Expected effect on identifiability — the most operationally severe entry
in this catalogue:** unlike B1/B2, this confound crosses a fault-unit
boundary (reactor/column vs. feed), so unit-level classification is directly
corrupted, not just within-unit parameter attribution. This is the confirmed,
root-cause explanation for why the feed-fault scenario W10 is misclassified
in all 30 tested replicates (F1 = 0.000) and why the compound scenario W13
(catalyst decay + lean feed) is weak (7/30 correct) — and is very likely also
the root cause of why the full 5-D S-B posterior could not be calibrated at
all under the corrected physics (a flow cannot represent a correct posterior
shape over a subspace that is structurally non-identifiable, no matter how it
is trained).

---

### B4. Null control — `(beta_r, xi_reb)`: confirms the diagnostic itself is not generating false positives

**What:** `beta_r` (reactor jacket) and `xi_reb` (reboiler duty) belong to
different units, different PI loops, and share no plausible physical
mechanism. Checked specifically as a negative control for the RT-FIM
methodology itself.

**Result:** small coupling under both the 66-D summary (off-diagonal ≈ −0.17)
and the raw trajectory (≈ −0.11), at both nominal and W9-truth operating
points — essentially unchanged between representations, as expected for a
genuinely unrelated pair.

**Why this entry matters even though it is not itself a confound:** it is
the evidence that the RT-FIM collapse/no-collapse test has a low false-positive
rate — i.e. that B1's and (provisionally) B2's "collapses to near-zero" readings
are a real signal the method can detect, not an artefact of the method itself
being insensitive or noisy. Without this check, "representation artefact,
resolved" readings elsewhere in this catalogue would be less trustworthy.

---

## Part C — Summary table

| Pair | System | Type | Cause | Status | Operational severity |
|---|---|---|---|---|---|
| `beta`×`UA` | I | Structural | Only appear as a product in the ODE | Permanent (resolved by fixing `UA`) | N/A — not a real inference target once resolved |
| `(alpha,beta)` | I | — (checked, not a confound) | N/A — System I retains a composition sensor `C` that excludes `beta` | Settled negative | None |
| `(alpha,beta)` FIM asymmetry | I | Local-sensitivity asymmetry, not a confound | PI masking leaves `beta` only indirectly observable | Settled, operating-point- and FIM-convention-dependent magnitude | Wider `beta` intervals; benign under non-saturating conditions |
| `(alpha,beta)` at Sc5 | I | Nonlinear/saturation, not the PI-masking mechanism | Coolant valve clipping | Settled | Reduced coverage, not detection |
| `beta` × sensor bias (Sc7) | I | Measurement-model confound | Controller reacts to biased reading | Settled, unresolved in the inference model | Reduced F1 (0.67) and coverage (0.52) |
| `(alpha,eta_col)` (Artefact 1) | II | Representation artefact | Restricted recycle-flow-only channel view | Resolved, re-confirmed at 3 more points | None once full feature set used |
| `(alpha,beta_r)` (Artefact 2, W11) | II | Claimed representation artefact | Shared `Q_j` actuator + whole-window aggregation | **Provisional — re-audit pending, magnitude disputed** | Benign for unit classification if confirmed; otherwise reclassify as genuine |
| `(alpha,z_A0_eff)` / `(eta_col,z_A0_eff)` (Artefact 3) | II | **Genuine, representation-robust confound** | Recycle stoichiometry — "snowball effect," no feed-composition sensor | Confirmed 5 ways | **Severe — crosses unit boundary, drives W10/W13 failures** |
| `(xi_reb,z_A0_eff)` | II | — (checked, not a confound) | No mechanistic link | Ruled out | None |
| `(beta_r,xi_reb)` | II | — (null control) | Different units, no mechanism | Confirms RT-FIM method validity | None |

---

## Part D — The one general lesson this catalogue supports

Across every entry: whether a given parameter pair is a problem for fault
*classification* depends on two independent factors, not one —

1. **Representation-robustness** (does the confound survive raw-trajectory
   access? — distinguishes a fixable feature-engineering problem from a real
   instrumentation limit), and
2. **Fault-taxonomy placement** (do the confused parameters share a fault
   unit? — determines whether unit-level classification is corrupted or just
   benign within-unit parameter attribution).

Artefacts 1 and 3 sit at opposite corners of this 2×2 (resolved/benign vs.
genuine/severe); Artefact 2's placement is currently unconfirmed on the first
axis pending re-audit, which is exactly why it is the top open item for
finishing this part of the identifiability story.
