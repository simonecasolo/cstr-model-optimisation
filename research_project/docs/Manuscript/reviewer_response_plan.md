# Plan: Responding to `docs/simulated_reviewer_report.md` (v3 — tightened for resubmission)

**Status:** Not started (written 2026-08-12; revised same day after a meta-review of this
plan itself — see "Revision history" below)
**Scope:** `docs/Manuscript/main.tex` (revtex4-2, `aps,pre,twocolumn`, confirmed target
*Computers & Chemical Engineering*, no layout changes in scope).

## Revision history

**v2 → v3 changes**, made in response to a critique that v2 was too permissive about
treating central validation work as optional. The critique's ten points are folded
directly into the affected sections below (each marked with which point drove the
change) rather than kept as a separate list, so this document stays a single source of
truth. Net effect: several items move from "optional Tier 3" to "mandatory," one item
(Table VI) loses its cheap-alternative option entirely, and the execution order changes
substantially — protocol-locking and the System II physics audit now come *before* any
data regeneration, and the manuscript-wide wording pass is now explicitly *last*, not
interleaved.

## How this plan is organized

Most of this reviewer's major comments are not resolved by rewording alone — several
point at a real, confirmed bug, a real train/test and cross-system inconsistency in the
data pipeline, or a genuine gap in statistical rigor with no cheap fix. Tiers are now
**strict and mutually exclusive** — each action below appears in exactly one tier (v3
point 10):
- **Tier 0 — Invalid results or claims that block submission.** Confirmed bugs or
  confirmed-false claims currently in the text; fixed by regenerating numbers from
  existing code/data or correcting a specific claim, no new experiment design needed.
- **Tier 1 — Text-only corrections, applied *after* Tier 0/2/3 numbers are final.** No
  computation.
- **Tier 2 — Reruns using existing trained models/code.** No retraining, no new
  experiment design — but per v3, several items that look like Tier 2 on the surface
  actually require a paired retraining leg to be conclusive; those are marked explicitly
  and live in Tier 3 instead of being miscategorized as "cheap."
- **Tier 3 — Retraining or material model/experiment-design changes.** The expensive,
  highest-risk tier. v3 promotes several items here from "optional" to "mandatory for a
  defensible resubmission" — this tier is no longer synonymous with "skip if short on
  time."
- **Tier 4 — Optional extensions.** Genuinely not required for a defensible resubmission.

---

## Part A — Major Comments (15), verdicts

### Major Comment 1: Table VI/VII numerical bugs

**Verdict: FIXED AND RE-EXECUTED (2026-08-12).** Final numbers in
`pending_manuscript_fixes.md` Stage 2, Items 1–2, ready to transcribe into `main.tex`.

Traced with two independent agent investigations plus my own file inspection to a
**cross-experiment mix-up inside `notebooks/16_ekf_ukf_baseline.ipynb`, cell 20**. That
cell builds two *separate* tables in sequence:
1. A **30-day sequential-tracking** table (`df_track_metrics`, from the unrelated
   720-window degradation-tracking run) — SBI row: `β MAE ≈ 0.0329`.
2. A **Sc2 50-replicate snapshot** table (`df_sc2`) — SBI row: `β̂ mean ≈ 0.5512`,
   `β bias ≈ -0.1488`, `β̂ std ≈ 0.0315`.

Table VII's SBI row (`β̂=0.551, bias=-0.149, MAE=0.033`) matches `β̂ mean`/`β bias` from
table (2) but its `MAE=0.033` matches `β MAE` from the **unrelated table (1)**. Given
`bias=-0.149` and a tight per-replicate spread (`std≈0.032`), a correctly computed MAE
for the *same* 50 Sc2 replicates would be **on the order of 0.15, not 0.033**. Checking
the *other* rows of Table VII against this same inequality during this revision also
surfaced that **EKF's row (bias=-0.093, MAE=0.065) fails the same `MAE ≥ |bias|` check**
(0.065 < 0.093) — this is not just an SBI-specific bug; the whole table's MAE column
needs regenerating from `df_sc2`, not only the SBI row. `results/metrics_summary.json`
and `results/07_cl_vs_ol_metrics.csv` independently corroborate that `β̂=0.551`/`bias=
-0.149` are correctly computed from the true 50-replicate population.

**Table VI (`β̂=0.62`) and the prose CNN-comparison paragraph (`β̂=0.616`) are the same
underlying number**, both traced to `pick_obs(sc_id, rep=0)` in `notebooks/04_sbi_training.ipynb`
and `04b_embedding_net_study.ipynb` — **a single representative replicate**, not a
50-replicate mean, despite Table VI's caption stating "50 replicates per scenario."

**v3 change (point 1): "re-caption as a representative replicate" is removed as an
acceptable outcome.** A hand-selected single replicate is not an appropriate basis for
reporting posterior means, bias, coverage, or classification performance in a principal
results table, and it invites an unanswerable question (why that replicate, chosen
before or after seeing the result?). **The only acceptable fix is to regenerate every
row of Table VI from all 50 replicates**, from one named, archived result file,
reporting: aggregate posterior mean, between-replicate standard deviation, MAE, and
empirical 90% coverage, per parameter per scenario — plus classification accuracy/recall
where relevant. If a single representative replicate is still useful for a
visualization (e.g. a posterior histogram figure), it may appear in a *figure*, with an
explicit, stated selection rule (e.g. "the replicate closest to the population median"),
never in the results table.

A third, still-unresolved discrepancy: Table VII's `ms/window = 16` matches neither
`23.86` (the per-window solve time actually computed in the same nb16 cell) nor `12.5`
(a different SBI-timing number in `notebooks/12_resource_analysis.ipynb`).

**Required action — all done:**
1. [x] Regenerated Table VII's MAE column for all methods from `df_sc2` — new numbers:
   SBI 0.5512/$-$0.1488/0.1488, EKF 0.6067/$-$0.0933/0.0974, UKF 0.6073/$-$0.0927/0.0973,
   NUTS 0.6298/$-$0.0702/0.0702 ($\hat\beta$/bias/MAE).
2. [x] Regenerated Table VI in full as a genuine 50-replicate aggregate — also caught
   and fixed a second, independent boundary-condition bug in the process (Sc4/Sc7's
   "true class" re-derivation, see `pending_manuscript_fixes.md` for detail). New
   per-scenario macro-F1 (per-replicate classification, closed-loop only): **0.978** —
   close to but not identical to the existing "0.990" headline figure, which uses a
   different (pooled-posterior) aggregation computed elsewhere (`nb11`); report both
   with their distinct methodologies stated, don't force them to match.
3. [x] `ms/window=16` corrected to 22.95.
4. [x] Automated `MAE ≥ |bias|` check added to `nb16`, passing for all four methods.

### Major Comment 7: Initialization protocol / artificial fault-onset transient

**⚠️⚠️ UPDATE (2026-08-13): CONFIRMED at full production budget with SBC — this is now a
settled, high-priority finding, not a provisional one.** See `pending_manuscript_fixes.md`
Stage 3, Item 8, for complete evidence. Headline results, full budget ($n=10{,}000$),
matched-protocol retrain vs. current production posterior, both evaluated identically on
Sc2:

| | $\hat\alpha$ | $\alpha$ bias | $\alpha$ MAE | $\hat\beta$ | $\beta$ bias | $\beta$ MAE |
|---|---|---|---|---|---|---|
| Production (old protocol) | 0.950 | $-0.051$ | 0.051 | 0.551 | $-0.149$ | 0.149 |
| **Matched protocol (new)** | **1.000** | **$+0.0001$** | **0.004** | **0.698** | **$-0.002$** | **0.012** |

**SBC ($N=500$) moves the same direction**: $\alpha$'s KS $p$-value goes from a
rejected $0.016$ to a passing $0.0665$; $\beta$'s goes from $0.014$ to $0.041$ (still
formally rejects at 5%, but its C2ST score, $0.509$, is now essentially at the
uninformative $0.5$ baseline, down from $0.53$). This is not a reduced-budget artifact
— the $n=2{,}000$ diagnostic and this $n=10{,}000$ confirmation agree closely.

**Two additional findings from cheap code inspection (no retraining needed), both
directly weakening the paper's convergent-evidence argument**: NUTS's own generative
model (`cstr_generative_model`, and the notebook's inline duplicate) and the
CNN-embedding "irreducibility" experiment (§6.3.3) **both share the identical
fixed-healthy-warm-start assumption** as SBI's training simulator — their agreement
with SBI's original bias is very likely the same bug, not independent confirmation.
EKF and UKF do **not** share this bug (their filter state initializes from the real,
correctly-generated observation data) — their bias likely has a different, possibly
still-genuine origin and should be treated as the more credible of the "independent
confirmations."

**UPDATE (2026-08-13): NUTS and CNN-embedding retrained under the matched protocol,
Table VII updated.** NUTS's inline generative model now computes
`warm_start_ic(alpha, beta)` at every leapfrog step instead of a fixed `NOMINAL_Y0_CL`
(dramatically more expensive — ~25× slower per iteration — since the steady-state
solve must be differentiated through). Single-replicate Sc2 result: $\hat\beta=0.751$
(bias $+0.051$), down in magnitude from the old protocol's bias $-0.070$ and
sign-flipped, in 43 minutes. **User caught a methodological inconsistency this
introduced**: SBI/EKF/UKF's Table VII rows are 50-replicate population statistics,
while NUTS's was (necessarily, for cost reasons) a single representative replicate —
an internal inconsistency in the same table, not acceptable for the headline
comparison. **UPDATE (2026-08-15): NUTS's 50-replicate rerun is done.** Took far
longer than the ~4–5h estimate (multiple infrastructure incidents — system sleep,
then CPU thread oversubscription across parallel workers — see
`pending_manuscript_fixes.md` Stage 3, Item 8 update, for the full incident log).
Final result: **32/50 (64%) converged within a 3-hour per-window compute budget;
18/50 (36%) did not and are excluded.** Population statistic over the 32:
$\hat\beta=0.6998$, bias $=-0.0002$, MAE $=0.0111$ — essentially unbiased, consistent
with the single-replicate result. Table VII updated accordingly, with the 64%/36%
convergence split reported explicitly rather than silently dropped, and framed as a
finding in its own right: NUTS's computational *reliability*, not just its
already-documented raw speed disadvantage, is compromised under the matched protocol.
CNN-embedding (§6.3.3) was also retrained under the
matched protocol: $\hat\beta=0.696$ (bias $+0.004$), down from $\approx-0.08$ under
the old protocol — its "confirms irreducibility to within 1%" claim does not survive.
Still open: a second seed to bound variance; re-deriving whether the FIM/analytical
steady-state argument (mechanism: integral control zeroing
$\partial T_\mathrm{ss}/\partial\beta$) is itself protocol-independent (plausible — it
would mean the *mechanism* survives even though the *empirical four-method
confirmation of its magnitude* does not).

**Manuscript impact**: the abstract, §6.3 ("Structural Identifiability Analysis"),
§8.1, and the conclusions currently present System I's β-deficit as irreducible and
confirmed by four independent methods. This can no longer be the paper's framing as
written.

**Terminology correction (2026-08-13): this is *not* a third "representation
artefact."** System II's Artefacts 1–2 concern information lost or distorted by the
measurement/compression mapping $y\to s(y)$ at *fixed, matched* data-generating
conditions — a **representation-induced coupling**. System I's finding is different in
kind: the posterior was trained and evaluated under two different conditional data
distributions, $p_\mathrm{train}(y\mid\theta)\neq p_\mathrm{test}(y\mid\theta)$ (fixed
healthy warm-start vs. scenario-specific warm-start) — a **protocol-induced inference
artefact**. Conflating the two would misstate the mechanism: representation artefacts
are about *what survives the map from raw trajectory to summary*; this one is about
*whether training and deployment ever simulated the same distribution in the first
place*. See Major Comment 10 below for the full three-level taxonomy this paper should
now adopt, and `pending_manuscript_fixes.md` Stage 3, Item 8, for the reframed
"what remains valid" discussion.

The corrected framing is arguably a **stronger** methodological contribution, not a
weaker one: it shows the paper's raw-trajectory/matched-protocol checking discipline
(already applied to System II's representation artefacts) also catches a second,
independent failure mode — a simulation-design/protocol mismatch — that four
supposedly-independent validation methods failed to catch because two of them
(NUTS, CNN-embedding) silently shared the same buggy simulator. This is a substantive
rewrite, not a numbers swap — do not attempt it piecemeal without the follow-ups above.
**Bias should no longer be presented as the principal evidence of closed-loop
information loss**; see the reframed diagnostic question in
`pending_manuscript_fixes.md` Stage 3, Item 8.

**Original verdict: CONFIRMED for System II; a *different*, more subtle bug found for System I. Tier 0 disclosure is necessary but not sufficient — v3 promotes matched-protocol validation to mandatory (points 2 and 3).**

- **System I evaluation data** (`notebooks/02_data_generation.ipynb`): confirmed
  **scenario-specific** warm start — no artificial onset transient.
- **System I *training* simulator** (`notebooks/04_sbi_training.ipynb`): a **fixed
  healthy warm-start** (`Y0_TRAIN`) is used for *every* prior draw during training and
  SBC. The trained posterior therefore approximates
  $q_\phi(\theta\mid s)$ fit under $p_\mathrm{train}(s\mid\theta)$ (healthy-onset
  trajectories) but is deployed against $p_\mathrm{test}(s\mid\theta)$
  (scenario-specific-steady-state trajectories) — a genuine train/test distribution
  mismatch, not merely an undisclosed detail.
- **System II, both training and evaluation** (`src/cstr_sbi/recycle/simulator.py`'s
  `generate_dataset()` and `scripts/sbi_pipeline.py`): confirmed **uniform healthy
  warm-start for every scenario**, training and evaluation alike — self-acknowledged in
  `notebooks/22_wu2003_data_generation.ipynb`'s own markdown.

**v3 change (point 2): disclosure alone is not an acceptable resolution for System I.**
A mismatch between the training and test conditional distributions can bias posterior
means, distort SBC, inflate or deflate coverage, and confound the CNN-vs-hand-crafted
comparison — precisely the quantities the paper's headline claims rest on. **At least a
one-seed matched-initialization retraining experiment is promoted to mandatory** (Tier
3, since it requires retraining): retrain one seed of the System I posterior with
scenario-specific *training* warm-starts, matching the evaluation protocol, and compare
against the current production posterior on bias, MAE, and SBC. If results change
materially, the production ensemble must be retrained under the matched protocol. If
they do not change materially, that robustness result itself becomes citable evidence
and the current production posterior may be retained — but only *after* this check, not
in place of it.

**v3 change (point 3): the System II fix cannot be "evaluate the existing posterior on
new data" alone.** That diagnostic evaluates a posterior trained under one regime
(healthy onset) against data generated under a *different* regime (steady degraded
state) — it reintroduces the identical train/test mismatch identified above and cannot
distinguish a genuine deployment-regime effect from an artifact of that new mismatch. It
answers "is this posterior robust to out-of-distribution shift?", not the paper's actual
question ("can SBI identify faults during ongoing degraded operation?"). **Required
design: three matched experiments**, promoted to Tier 3 (mandatory, requires
retraining for at least two of the three legs):
1. **Matched onset regime** — train and evaluate with healthy-state initialization
   followed by abrupt degradation (i.e., today's setup, but explicitly labelled and
   validated as such).
2. **Matched ongoing-degradation regime** — train and evaluate from
   parameter-specific degraded steady states (`scenario_specific_warm_start=True` on
   both legs, not just evaluation).
3. **Cross-regime transfer** — train in one regime, evaluate in the other, to
   characterize the actual sensitivity to deployment-regime mismatch.

This turns the discovered bug into a genuine scientific result about deployment-regime
sensitivity rather than a defect to quietly patch, and directly answers the operational
question a plant-monitoring reviewer will actually ask: does this work during ongoing
degradation, not just at fault onset?

**Required action (Tier 0, do regardless of the above):** Reword the SI's ambiguous
"nominal closed-loop steady state... around the prescribed operating point" sentence
(§S1, §S7.3) to state the actual, verified protocol for each system and each of the
three regimes above precisely, once they are run.

### Major Comment 2: Cramér-Rao bound misused to explain bias

**Verdict: Partially fixed already; the proposed replacement wording is still too strong (v3 point 6). Tier 0/1.**

The CNN-embedding paragraph (`main.tex` line ~1040) still states an invalid Cramér-Rao
inference. The v2 fix language ("the offset is data-driven rather than algorithm-specific")
is also too strong: the methods being compared share the same simulated process, the
same generated data, likely the same observation assumptions, and (per Major Comment 7)
possibly the same initial-condition regime. Agreement across methods rules out that the
bias is an artifact of one specific implementation, but it cannot by itself distinguish
a genuine data-information effect from a shared model, initialization, or
likelihood-construction artifact.

**Required action (Tier 0/1, wording only), corrected per v3:** *"The common bias
direction indicates that the offset is not unique to the SBI architecture. However,
because all estimators share the same simulated process and observation assumptions,
this comparison does not by itself distinguish limited data information from shared
model, initialization, or likelihood effects."* Drop "empirical confirmation of the
Cramér-Rao bound" entirely, and drop the earlier v2 "data-driven, not algorithm-specific"
framing along with it.

**⚠️ UPDATE (2026-08-13): v3's "possibly the same initial-condition regime" hedge is now
confirmed literally true.** Major Comment 7's investigation found the CNN-embedding
posterior (the specific comparison this paragraph makes) was trained under the
identical fixed-healthy-warm-start protocol as the hand-crafted-feature posterior — its
"agreement to within 1%" is not informative about feature-representation-independence
in the way the manuscript claims, since both share the actual root cause of the bias
this investigation found. The wording fix above is no longer just a defensible
softening; it is now the only defensible framing. Do not finalize this paragraph until
Major Comment 7's CNN-embedding rerun (its follow-up item 3) is done or explicitly
deferred with this caveat stated.

### Major Comment 3: FIM validity (diagonal Σ, correlated features, autocorrelation, non-Gaussian summaries, saturation nonsmoothness, no FD convergence study)

**Verdict: CONFIRMED gap. v3 promotes a scoped covariance-robustness check from optional Tier 3 to mandatory (point 4).**

The FIM is computed as `I = J^T Σ^{-1} J` with `Σ` an explicitly diagonal covariance
estimated from healthy replicates and used at faulted operating points, with no
discussion of cross-feature correlation, autocorrelation, or non-Gaussianity.

**v3 change (point 4):** merely renaming the quantity ("local Gaussian
sensitivity-information approximation") does not establish that the observed *reduction*
in coupling under RT-FIM is a real information gain rather than an artifact of the
diagonal-covariance approximation — and the representation-artifact claim is one of the
paper's principal contributions, so this cannot be waved off as exploratory-only without
at least a bounded check. **Promoted to mandatory, scoped to a pragmatic minimum**: for
the three key parameter pairs — $(\alpha,\beta_r)$, $(\alpha,z_{A0,\mathrm{eff}})$,
$(\beta_r,z_{A0,\mathrm{eff}})$ — repeat the local sensitivity calculation under: (a) the
current diagonal covariance, (b) a shrinkage-estimated full covariance, (c) a
block/lag-aware covariance for the raw-trajectory representations, at (d) 2–3 different
finite-difference perturbation sizes and (e) 2–3 different Monte Carlo replicate counts
for estimating Σ. If the representation-artifact conclusion survives this grid, report
it as robustness-confirmed; if it does not survive, the corresponding claim must be
downgraded to "exploratory, not established" rather than reported as a resolved finding.

**Required action:**
- **Tier 1 (do regardless):** Add an explicit scope paragraph at first use of the FIM,
  naming the assumptions plainly and calling it a "local Gaussian sensitivity-information
  approximation" at least once.
- **Tier 3 (mandatory per v3, not optional):** The covariance-robustness grid above, for
  the three key pairs. This is the pragmatic minimum, not the reviewer's strongest
  alternative (a full likelihood derivation) — but it is no longer skippable.

### Major Comment 4: Raw-trajectory FIM does not prove plant-level identifiability

**Verdict: Already substantially hedged; one more precise qualifier still worth adding. Tier 1.**

Unchanged from v2: add a defining sentence for "representation artefact" at first use,
scoping it to "a local coupling substantially amplified by the summary representation
under the assumed Gaussian sensitivity metric, established at the tested operating
points." This scoping is now also directly load-bearing for Major Comment 3's robustness
grid above — the definition should explicitly note the claim is conditional on that grid
holding.

### Major Comment 5: System I physics-informed summary derivations

**Verdict: Confirmed by independent re-derivation. v3 escalates the $s_{UA}$ item from "quantify approximation error" to a full ablation, mandatory (point 8). Tier 0/1 (wording) + Tier 2/3 (ablation).**

**5.1 — `s_UA` is missing a factor**, confirmed by re-derivation: the correct relation is
`(β·UA)^{-1} = (T-Tc) / [ρ_c·C_pc·Q_c·(Tc-Tci)]`; the paper's `s_UA = (T̄-T̄c)/Q̄c` omits
`(Tc-Tci)`.

**v3 change (point 8): the blast radius is not limited to the illustrative 2-feature LDA
argument.** $s_{UA}$ is one of the 29 features fed to the *production* NSF posterior — a
flexible density estimator might learn to compensate for the misspecification, or it
might not; this must be tested, not assumed away. **Promoted to a mandatory three-way
ablation** (Tier 2/3, reuses existing simulation data and training code, no new
simulator development): retrain/re-evaluate with (a) the existing proxy, (b) the
corrected balance proxy $\frac{\bar T-\bar T_c}{\bar Q_c(\bar T_c - T_{ci})}$, and (c) no
jacket proxy at all (relying on the other 28 features), comparing β bias, β MAE, SBC
rank uniformity, credible-interval coverage, classification accuracy, and sensitivity
near actuator saturation. If the corrected feature reduces bias, report this as a
genuine methodological improvement, not merely an error correction.

**5.2 — `s_k0` (kinetic proxy)** — lower severity, confirmed as a terminology
imprecision only (affine in $-\ln(\alpha k_0)$, not proportional to its reciprocal); the
qualitative monotonic argument is unaffected. Tier 1 wording fix only, unchanged from v2.

### Major Comment 6: System II model under-specified / potentially inconsistent

**Verdict: AUDIT COMPLETE (2026-08-12) — see `pending_manuscript_fixes.md` Stage 1 for full evidence. Outcome is substantially better than feared: mostly Tier 1/2 fixes, not the large Tier 3 undertaking originally scoped.**

The full audit against the actual primary source (Wu, Yu, Luyben & Skogestad, 2003, text
extracted directly from the paper) plus the actual production code
(`src/cstr_sbi/recycle/physics.py`) resolved every sub-item:

1. **The reactor energy/jacket dynamics are not in the cited benchmark at all** — Wu et
   al. (2003)'s own equations are composition/holdup balances only; the same author
   group's companion paper (Larsson, Skogestad & Yu, 1999) states explicitly they "only
   use simple models, which does not include any energy balances... since normally the
   temperature in the reactor is given from kinetic considerations." **The manuscript's
   entire reactor-jacket thermal model (and the $\beta_r$ parameterization) is this
   paper's own addition**, using the benchmark's Table 1 steady-state numbers as
   parameter values but not its dynamic form. This is a legitimate modeling extension,
   but it is currently undisclosed — main text line 500 reads as though only the fault
   *parameters* are new. **Tier 1 fix**: one sentence disclosing the extension where
   System II is introduced. Does **not** require recasting System II as a "surrogate" —
   the topology and steady-state operating point remain faithfully Wu et al. (2003).
2. **$F_R = D\,x_D/z_{A,\mathrm{in}}$ (`eq:wu_recycle`) is confirmed wrong against the
   primary source** (which gives simply $F_R=D$, verified via $RR=D/B\approx1.09$ =
   Table 1's own numbers) — **but the actual simulator code already computes $F_R=D$
   correctly** (`recycle/physics.py:331`). Downgraded from feared Tier 3 to **Tier 1,
   text-only**: correct the written equation; no rerun needed.
3. **$\beta_r$ scaling the entire jacket bracket including the commanded duty $Q_j$ is
   confirmed as a genuine code-level choice** (`recycle/physics.py:465-474`), not a
   documentation slip, with no justification anywhere. **Tier 2, unchanged**: fix the
   code to scale only the conductive term, or add an explicit rationale.
4. **Constants are correctly sourced where checked** ($\Delta H_r$, $UA_r$ both exact
   matches to Wu et al.'s Table 1) — no fabricated values found. Two values
   (`MJ_CPJ`'s jacket holdup/heat-capacity factors) could not be traced to the 2003
   paper's Table 1 (consistent with Finding 1 — that table has no jacket entries) and
   may originate from the earlier, paywalled Wu & Yu (1996) paper the 2003 paper itself
   cites for its parameters, or may be this manuscript's own assumption — **flagged for
   author follow-up**, not resolved by this investigation. Table III's missing feed
   temperature is now directly fillable: $T_0=70°F=294.3$ K, from Wu et al.'s Table 1.
5. **$Q_\mathrm{reb}$'s documented equation is an incomplete transcription of the actual
   code**, which includes undocumented bottoms-purity-excess and column-severity
   correction terms. **Tier 1/2 fix**: extend the written equation to match, or justify
   the simplification.
6. **$\eta_\mathrm{col}$ "tray efficiency" naming confirmed to be this manuscript's own
   invention** — neither primary source contains a tray-efficiency or Murphree-efficiency
   concept. Reinforces the already-planned Tier 1 rename.

**Required action, revised down from "the largest item in this plan" to a short, mostly
Tier 1/2 punch list:**
1. **Tier 1:** Disclose the reactor-jacket thermal model as an added extension (Finding
   1); correct `eq:wu_recycle` to $F_R=D$ (Finding 2); rename $\eta_\mathrm{col}$
   (Finding 6); add $T_0$ and other now-known constants to Table III (Finding 4);
   extend the documented $Q_\mathrm{reb}$ equation (Finding 5).
2. **Tier 2:** Resolve or justify the $\beta_r$-scales-$Q_j$ modeling choice (Finding 3);
   confirm the provenance of `MJ_CPJ` with the author team (Finding 4).
3. **No Tier 3 action remains for Major Comment 6** — the feared "full specification
   audit, potentially requiring recast as a surrogate benchmark" is complete and the
   model's topology/steady-state operating point are confirmed faithful to Wu et al.
   (2003); only the added thermal-fault layer needed disclosure, not correction or
   re-scoping.

### Major Comment 8: SBC failure interpretation ("structural, not training deficiency")

**⚠️ UPDATE (2026-08-13): the reviewer's skepticism was correct, and now empirically
vindicated.** Major Comment 7's matched-protocol retrain (full budget, $n=10{,}000$) reran
SBC under the corrected protocol: $\alpha$'s KS $p$-value goes from a rejected $0.016$
to a passing $0.0665$; $\beta$'s goes from $0.014$ to $0.041$ (still formally rejects at
5%, but its C2ST is now $0.509$, essentially at the uninformative baseline, down from
$0.53$). **The original SBC failure was substantially — very likely mostly — a training
deficiency (a training/evaluation protocol mismatch) exactly as the reviewer argued,
not the structural information deficit the manuscript claimed.** Full numbers in
`pending_manuscript_fixes.md` Stage 3, Item 8.

**Original verdict (confirmed correct in direction, now superseded in detail): the
paper's C2ST≈0.5-alongside-KS-rejection argument was suggestive, not dispositive —
weak identifiability alone does not cause SBC rank non-uniformity if the posterior
approximation is otherwise faithful.**

**Required action — status updated:**
- **Tier 1 (mandatory, wording now more specific than originally planned):** Replace
  "the deviation reflects structural information deficit rather than a training
  deficiency" with language reflecting what was actually found: the original SBC
  failure was substantially a training-protocol artifact, now resolved for $\alpha$ and
  much reduced for $\beta$; do not claim the residual $\beta$ deviation is
  "structural" either without the FIM re-derivation (Major Comment 7's follow-up item 2)
  to back that up.
- **Tier 3 — DONE.** SBC re-run under the matched protocol at full budget
  ($N=500$, same as production standard). Posterior predictive checks in trajectory
  space are still a good addition but no longer blocking — the rank-based result alone
  is already decisive enough to require the wording change above.

### Major Comment 9: "90% CI correct at every time step" from one trajectory

**Verdict: CONFIRMED. Tier 1 (mandatory wording) + Tier 3 (promoted from "recommended Tier 2" — v3 point 10 fixes a tier inconsistency here: this item appeared as both Tier 2 and Tier 3 in v2).**

**Required action:**
- **Tier 1 (mandatory, minimum fix):** Reword to purely descriptive, non-coverage
  language for the single representative run.
- **Tier 3 (the only place this item now appears — resolves the v2 inconsistency):**
  Repeat the sequential-tracking experiment across 10–20 independently sampled
  degradation paths and report genuine empirical pointwise coverage. Classified as Tier 3
  (not Tier 2) because it is part of this plan's "Strongly Recommended, best cost/benefit"
  bucket (Part C) and should be scheduled alongside the other Tier 3 validation work, not
  treated as a quick Tier 2 aside.

### Major Comment 10: Structural/fundamental/irreducible language overstates a practical-identifiability finding

**Verdict: CONFIRMED, pervasive. Tier 1, a dedicated terminology pass. UPDATED
(2026-08-13) — promoted to house the paper's formal artifact/limitation taxonomy,
following the System I matched-protocol finding (Major Comment 7).**

Distinguish, precisely and exhaustively, four *sources* of apparent identifiability
loss the paper's own results actually instantiate — do not use "irreducible,"
"fundamental," or "structurally removed" as a synonym for any of the latter three:

**(a) Genuine structural non-identifiability** — the one case where no amount of data
or protocol correction resolves it: β·UA in the recycle-loop energy balance (correctly
labelled already; unaffected by anything below).

**(b) Closed-loop practical information reduction** — feedback control (integral
action) suppresses the steady-state sensitivity of a measured channel (e.g.
$\partial T_\mathrm{ss}/\partial\beta\to 0$) to a subset of parameters, potentially
reducing the *achievable* precision from that channel alone, at fixed, correctly-matched
training/deployment conditions. This is a real, data-generating-process property, not
an artefact of how the data is processed or how the experiment was run — but its
practical severity is a question of precision (posterior SD, CI width, local
sensitivity, mutual information), not of the point estimate being biased.

**(c) Representation-induced coupling** — summary-statistic (or other lossy)
compression $y\to s(y)$ discards distinctions present in the raw measured trajectory,
manufacturing apparent parameter coupling/non-identifiability that a raw-trajectory or
higher-fidelity representation resolves. This is what Major Comment 4's System II
Artefacts 1–2 (and the RT-FIM diagnostic) actually document. Scoped per Major Comment 4.

**(d) Protocol-induced inference artefact** — training and evaluation simulators draw
from *different* conditional data distributions, $p_\mathrm{train}(y\mid\theta)\neq
p_\mathrm{test}(y\mid\theta)$ (e.g., System I's fixed-healthy vs. scenario-specific
warm-start mismatch, Major Comment 7), producing bias and calibration failure that is
an artefact of the *experimental/simulation design*, not an inherent plant or
measurement limitation. This is mathematically and conceptually distinct from (c): (c)
is about what survives a fixed map $y\to s(y)$; (d) is about whether the training and
test data were ever generated by the same process at all. A raw-trajectory check does
not, by itself, diagnose or fix (d) — only matching the protocol does.

**Required action:** every place the manuscript currently attributes a finding to
generic "identifiability limits," classify it into exactly one of (a)-(d) above and use
that category's precise language. In particular: the System I β-deficit (§6.3, abstract,
§8.1, conclusions) must be re-classified from an implied (a)/(c)-flavored "irreducible"
claim to (d) for the *bias/miscalibration* that has now been shown to (mostly) resolve
under the matched protocol — while claims about (b) (integral action suppressing
$\partial T_\mathrm{ss}/\partial\beta$, information transferring to jacket/control-effort
channels, β remaining less identifiable than α) **may still be valid** but require
re-analysis under the matched protocol before being restated as findings, per
`pending_manuscript_fixes.md` Stage 3, Item 8.

### Major Comment 11: Likelihood-intractability motivation / NUTS likelihood unclear

**Verdict: Partially already improved. v3 adds a structural clarification with real conceptual payoff (point 7). Tier 1.**

The current text already narrows the claim to "the likelihood of the resulting
window-level summaries is not analytically tractable" — more defensible than a blanket
claim. What's still missing, and what v3 adds:

**The paper should explicitly distinguish three different posterior targets**, since
$p(\theta\mid y) \neq p(\theta\mid s(y))$ unless $s(y)$ is sufficient for $\theta$:
1. The **raw-data Bayesian posterior** $p(\theta\mid y)$ — what NUTS actually targets,
   via the tractable state-space likelihood implied by the Euler–Maruyama simulator.
2. The **summary-conditioned posterior** $p(\theta\mid s(y))$ — the true target of the
   SBI approach, whose likelihood is not analytically tractable.
3. The **neural approximation** $q_\phi(\theta\mid s(y))$ — what SBI actually delivers,
   an approximation to (2).

NUTS and summary-based SBI are therefore not estimating the same posterior target, and
their means/bias/coverage are not directly comparable estimator properties without this
caveat made explicit. **This is not merely a caveat to add — it is a genuinely useful
organizing principle for the paper**, since it cleanly separates three distinct sources
of "loss" the paper currently conflates under generic language: the step from $y$ to
$s(y)$ is *representation loss*; the step from $p(\theta\mid s)$ to $q_\phi(\theta\mid s)$
is *approximation error*; and low information already present in $y$ itself is a
*process/measurement limitation*, not a representation or estimator problem at all.

**Required action (Tier 1, one paragraph in Methodology, but written to double as this
three-way decomposition, not just a NUTS clarification):** State the three targets
explicitly, identify which quantity each of the paper's own findings actually belongs to
(the Artefact 1–2 "representation artefact" findings are representation loss; the
CNN-vs-SBI gap on Artefact 2 is approximation error), and use this decomposition to
reframe the Discussion section's synthesis, not just the Methodology's motivation.

**UPDATE (2026-08-13):** the worked example originally given here — "the β
Fisher-information deficit is a process/measurement limitation" — is no longer safe to
state as-is. Per Major Comment 7/10, a large part of that deficit's headline
manifestation (the point-estimate bias and SBC failure) has since been shown to be a
**protocol-induced inference artefact** (category (d) in Major Comment 10's taxonomy),
not a process/measurement limitation. This three-way decomposition (raw-data posterior
/ summary-conditioned posterior / neural approximation) is a genuinely distinct,
complementary framework — it classifies *which posterior target* a quantity belongs to,
whereas Major Comment 10's four-category list classifies *why* identifiability appears
reduced — and should **not** be merged with it. But any residual, genuine closed-loop
β information deficit (category (b): integral action suppressing $\partial
T_\mathrm{ss}/\partial\beta$) still needs its own re-derivation under the matched
protocol before it can again be cited as this section's example of a "process/
measurement limitation" — do not restate the old example until that re-derivation is
done (`pending_manuscript_fixes.md` Stage 3, Item 8).

### Major Comment 12: Baseline comparison fairness (EKF/SBI information asymmetry, MCMC details)

**Verdict: Mostly already adequate. Tier 1, one small addition. Unchanged from v2.**

### Major Comment 13: Near-perfect classification not persuasive (train/test leakage, no held-out severities)

**Verdict: CONFIRMED valid methodological gap. Tier 1 (mandatory caveat) + Tier 4 (optional validation, per the reviewer's own "Strongly Recommended" not "Essential" framing — unchanged from v2, no promotion warranted here since the reviewer themselves did not rank this as essential).**

### Major Comment 14: Feed-fault failure undersells the "plant-wide" framing

**Verdict: Already substantially resolved. Tier 4, optional polish only. Unchanged from v2.**

### Major Comment 15: Sensor-drift experiment (Sc7) ambiguous

**Verdict: RESOLVED WITH CODE EVIDENCE. Tier 1 (mandatory disclosure) + Tier 4 (optional companion experiment). Unchanged from v2 — the companion closed-loop-drift experiment remains genuinely optional since it tests a different scenario than any of the paper's existing claims depend on, unlike the initialization-protocol issue in Major Comment 7 which the paper's existing claims already depend on.**

---

## Part B — Additional Technical & Presentation Comments (20), verdicts

Unchanged from v2 except item 14, revised per v3 point 9 below. See v2 history in git
for the full table; item 14 in full:

**Item 14 — $z_{A0,\mathrm{eff}}$ threshold: RESOLVED (2026-08-12), no bug found —
manuscript wording needs clarifying, not the computation.** Traced the actual code
(`nb31_wu2003_fault_classification.ipynb`'s `sample_fault_unit()`) rather than assuming
the manuscript's ambiguous prose describes it correctly: the threshold is **already**
computed as `0.85 * 0.90 = 0.765` (Policy A, exactly), and the archived
`results/31_classification_summary.json` (accuracy 0.8738, macro-F1 0.6937) matches
`main.tex`'s reported 87.4%/0.694 exactly — confirming these numbers were already
produced under the correct computation. No rerun was needed or performed.

**What *was* found**, while checking this: (1) ground-truth scenario labels use a
**different** threshold (0.90/absolute-tolerance, not 0.85/relative) than posterior
predictions — an undisclosed but not incorrect methodological detail, needs one
clarifying sentence; (2) a bigger, independent finding — **Table V's scenario parameter
values are stale relative to the current code** for 7 of 14 scenarios (W2, W4, W7, W9,
W10, W15, W16) — the archived classification results are correct and already reflect
the *current* code's values, so only Table V's descriptive text needs correcting. Full
tables of both in `pending_manuscript_fixes.md` Stage 2, Item 4/4b.

**Required action — done (text-only, ready for Stage 7):**
1. [x] Decided: Policy A, already implemented — no rerun needed.
2. [x] Confirmed the downstream chain (labels → confusion matrix → F1 → accuracy) is
   already correct and self-consistent; W13 does not change class (verified against
   the archived, already-executed result).
3. [x] New: prepared Table V's corrected scenario values for transcription.

---

## Part C — "Experiments Required for a Publishable Revision" — mapped to tiers

Unchanged from v2's mapping, with two exceptions per the promotions above: item #7
(fault-specific steady states) is now Tier 3 mandatory in its full three-experiment form
(Major Comment 7), not "partially already satisfied," and item #9 (full covariance/
temporal-dependence sensitivity) is now Tier 3 mandatory in its scoped form (Major
Comment 3), not optional.

**Strongly Recommended (11–19)** — the reviewer's own non-essential bucket. Two remain
the best cost/benefit if further effort is invested beyond the mandatory package:
repeated sequential-tracking across multiple degradation paths (Major Comment 9's Tier 3
item), and reliability diagrams/Brier scores for unit-level class probabilities. The
rest remain flagged as future work, not blocking.

---

## Recommended minimum resubmission package (v3 point 10's consolidation)

Everything below is mandatory; nothing in this section is a "nice to have."

**A. Numerical integrity — DONE (2026-08-12)**
- [x] Regenerate Tables VI and VII from named archived result files (Major Comment 1).
- [x] Add an automated `MAE ≥ |bias|` check, applied to every reported table.
- [x] Resolve all latency/`ms/window` values.
- [x] Recompute threshold-dependent labels and metrics after the Policy A/B decision
  (Additional Comment 14) — turned out Policy A was already correctly implemented; no
  recomputation needed, only Table V's stale scenario values need correcting.
- [x] Include complete parameter-wise aggregate results (mean, SD, MAE, coverage) for
  System I (Table VI). System II's equivalent is unaffected by this session's fixes
  (already correct) — no change needed there.

**B. Matched simulation protocol — System I DONE and CONFIRMED (2026-08-13); System II full-budget S-B attempt DONE, but failed calibration (2026-08-27)**
- [x] Define the intended monitoring task explicitly: abrupt onset, ongoing degradation, or
  both (Major Comment 7) — **decided**: ongoing-degradation is primary, see Stage 0 in
  `pending_manuscript_fixes.md`.
- [x] Match training and SBC initialization distributions for System I — done, both at
  reduced budget ($n=2{,}000$) and full production budget ($n=10{,}000$); confirmed by
  full SBC re-run, not just point-estimate comparison.
- [x] Retrain the System I production posterior under the corrected protocol — done at
  full budget; result saved separately
  (`results/sbi_posterior_matched_protocol_n10000.pkl`) pending a decision on whether to
  promote it to the new production posterior (recommend yes, given the strength of the
  evidence — see `pending_manuscript_fixes.md` Stage 3, Item 8).
- [x] Run the full-budget matched System II S-B ongoing-degradation confirmation —
  done with corrected $\beta_r$/$Q_j$ physics, `n=15,000`, 8 seeds, `N_SBC=400` per
  seed plus independent `N_SBC=800` selected-seed confirmation. **Negative result:**
  no seed passed min-KS > 0.05; selected seed 3 failed confirmation with min KS
  $p=9.03\times10^{-8}$. Do not promote the matched full posterior or update System II
  manuscript results as calibrated.
- [ ] Diagnose the System II matched-posterior calibration failure before cross-regime
  transfer or downstream cascade: out-of-prior posterior mass, heavy-tailed summary
  dimensions/z-scoring, and reduced-budget preprocessing/architecture fixes.
- [ ] Cross-regime transfer and full downstream System II cascade remain **blocked** on
  resolving or explicitly accepting the failed matched-posterior calibration result.

**C. Statistical validity**
- Re-run SBC under the corrected pipeline (Major Comment 8).
- Stop calling a posterior "calibrated" where rank uniformity is rejected, without the
  caveat language specified above.
- Add posterior predictive checks in trajectory space.
- Provide the covariance-robustness grid for the three key parameter pairs (Major
  Comment 3).
- State the three-level posterior-target decomposition (raw-data / summary-conditioned /
  neural-approximation) and use it to attribute each of the paper's findings to the
  correct source of "loss" (Major Comment 11).

**D. Physical validity**
- ~~Complete the System II audit against Wu et al. (2003)~~ **Done** (Major Comment 6):
  outcome is disclosure of an added thermal layer, not fix-and-rerun or recast-as-surrogate
  — apply the six Tier 1/2 fixes it produced (disclose the reactor-jacket extension,
  correct the recycle relation, resolve/justify $\beta_r$-on-$Q_j$, confirm `MJ_CPJ`'s
  provenance, extend the $Q_\mathrm{reb}$ documentation, rename $\eta_\mathrm{col}$).
- Run the three-way $s_{UA}$ ablation (Major Comment 5.1).

**E. Claim correction (Tier 1, do last, once A–D are settled)**
- Replace structural-non-identifiability language with practical-identifiability
  language where the finding is actually a precision/SNR effect, not literal
  non-identifiability (Major Comment 10).
- Define "representation artefact" locally and conditionally, tied to the covariance
  robustness grid in C (Major Comment 4).
- Describe cross-method bias agreement as "not algorithm-specific," never as proof of a
  data-intrinsic effect (Major Comment 2).
- State clearly that feed attribution remains unsuccessful (already mostly done, Major
  Comment 14).
- Clarify that Sc7 is historian-channel observation bias, not closed-loop sensor drift
  (Major Comment 15).

---

## Suggested execution order (replaces v2's "Recommended order of execution")

The order changed substantially from v2: the System II physics audit and
protocol-locking now come *before* any data/table regeneration (regenerating numbers
from a model that may itself be wrong, or from a training/eval protocol that hasn't been
settled, wastes the regeneration work), and the manuscript-wide wording pass is now
explicitly *last*.

1. ~~Lock the scientific task definition and initialization protocols~~ **Done** (Major
   Comment 7): primary/headline regime = ongoing degraded steady-state operation, onset
   is secondary — see `pending_manuscript_fixes.md` Stage 0 for the evidence and
   consequences for later stages.
2. ~~Audit and correct System II equations~~ **Done** (Major Comment 6): outcome was
   disclosure of an added, undisclosed thermal-model extension plus a handful of Tier
   1/2 text/code fixes — no blocking Tier 3 finding, so this no longer gates data
   regeneration the way it was expected to. Apply its punch list alongside step 3 below.
3. **Fix the $z_{A0,\mathrm{eff}}$ threshold policy and the metric-generation code**
   (Additional Comment 14, plus the Table VI/VII bug fixes in Major Comment 1's code).
4. **Regenerate Tables VI and VII** from the fixed code (Major Comment 1).
5. **Run matched-initialization retraining and evaluation** for both systems (Major
   Comment 7's mandatory retraining legs).
6. **Repeat SBC and add posterior predictive checks** under the now-corrected protocol
   (Major Comment 8).
7. **Run the covariance/finite-difference robustness grid** for the three key parameter
   pairs (Major Comment 3).
8. **Run the corrected $s_{UA}$ physics-proxy three-way ablation** (Major Comment 5.1).
9. **Repeat sequential tracking over independent degradation paths** (Major Comment 9),
   if pursued alongside the mandatory package.
10. **Only then, perform the manuscript-wide wording revision** (Part A's Tier 1 items,
    Major Comment 11's three-level decomposition reframe, and Part B's remaining text
    fixes) — so the wording is written against final, locked numbers and conclusions,
    not revised twice.
