# Session Handoff — nb31 follow-up: fault_unit() labelling bug fixed; a THIRD representation artifact found and confirmed by FIM

**Date:** 2026-07-05, same-day continuation of the nb30/nb31 build below. After nb30/nb31
were built and reported, the user asked "can the W15 misclassification be solved? What about
re-labelling the scenarios?", then corrected an initial framing mistake ("actually it is both
W10 and W15 that fail completely"). Investigating both led to two real fixes/findings, both
executed and verified — **not just a labelling patch**:

1. **`RecycleScenarioConfig.fault_unit()` had a real bug, now fixed.** The old version used
   scenario-*name* substring matching (`"cat_"`, `"snowball"`, etc.) before falling back to a
   parameter-threshold check. This silently mislabelled two of the 14 closed-loop scenarios:
   **W15** (`multi` via a "snowball" keyword, even though only α — not η_col=0.90, a 10%
   deviation — crosses the 15%-deviation threshold used everywhere else) and **W13**
   (`reactor` via a "cat_" keyword, even though it has two genuinely degraded units: α=0.80
   **and** z_A0_eff=0.80). Rewritten to a pure, name-independent counting rule (same logic as
   `nb31`'s own `sample_fault_unit` classifier, and the pre-existing pattern already used in
   `cstr_sbi.luyben.scenarios` — this was not a novel design, just applying an existing
   in-repo precedent consistently). Verified by dry run before editing: **only W13 and W15
   change label**; all other 12 scenarios are unaffected, and no other code path depends on
   the changed behaviour for these two scenarios (`nb20`'s reference table and
   `simulator.py`'s `generate_dataset()` also call `fault_unit()`, but only for
   visualisation/labels in a dataset-generation path with no known current SBI-training
   consumer — not touched, low priority, cosmetic only).
2. **After the fix, `nb31` was re-run end-to-end**: W15 now classifies correctly (30/30,
   `reactor`, matching what the classifier already predicted before the fix — the
   *classifier* was never wrong, only the ground-truth label was). W13's corrected `multi`
   label exposes a **real** weakness (7/30 correct) that the old, wrong `reactor` label had
   been masking. New overall numbers: **87.4% accuracy, macro-F1 = 0.694** (up from 84.0%/
   0.669 in the original nb31 build).
3. **The user's correction about W10 led to a real, corroborated finding — not solved by
   relabelling.** W10 (feed fault) still fails completely (F1 = 0.00) after the relabelling
   fix, because it was never a labelling problem. The *original* explanation written into
   `nb31`/this handoff ("posterior width vs. fault size at a single window — pool more
   replicates to fix it") was **checked and found wrong**: per-replicate z_A0_eff posterior
   means are precise (std ≈ 0.01 across replicates at fixed truth), so pooling would not move
   the estimate at all. The actual mechanism, found by comparing `z_A0_eff_mean` across all 14
   scenarios' already-computed results (`results/31_fault_classification_metrics.csv`): **W2
   — a *pure* catalyst-decay fault with z_A0_eff untouched at its true nominal 0.90 — shows a
   pooled z_A0_eff estimate of 0.812, almost as low as W10's actual feed fault (0.801, true
   0.78)**. Checked across 3 independently-trained seed posteriors (seed 4/5/6 from the
   8-seed ensemble, `sbi-logs/wu2003_posterior_variant_A_sb_v2_seed{4,5,6}.pkl`) — the pattern
   reproduces in all three, ruling out seed-specific noise (L9). **This is a third,
   genuine-looking joint near-degeneracy: (α, z_A0_eff) and (β_r, z_A0_eff) under
   `compute_summaries`.** Applying this paper's own decisive check (the exact
   `FIM = J^T Σ^{-1} J` noise-calibrated methodology from §7.2.2/`nb29b` §4, generalised to
   this new parameter pair): off-diagonal is weak at nominal (≈ −0.12) but jumps to **≈ −0.89**
   at the W2 truth (α degraded) and **≈ −0.48** at the W5 truth (β_r degraded) under
   `compute_summaries` — matching the magnitude of the paper's own (α, β_r) headline
   +0.901 number — and **collapses to ≈ −0.07 under the raw trajectory in both cases**,
   identical to the (α, β_r)/(α, η_col) signature. **Conclusion: this is a third instance of
   the same representation-artifact mechanism already documented twice in this paper (L4,
   L4′), not a new kind of finding, and not fixable by pooling.** Added to the article as
   limitation **L4″** (§8.4), and as a new `nb31` §6b addendum (executed, real FIM numbers).
   The paper's §8.1 "zero-for-two" framing is now "zero-for-three."

**What changed on disk this sub-session:**
- `src/cstr_sbi/recycle/scenarios.py`: `fault_unit()` rewritten (threshold-based, no name
  matching); old version fully replaced, not kept as a fallback.
- `notebooks/31_wu2003_fault_classification.ipynb`: re-executed twice (once after the
  `fault_unit()` fix, once more after adding new §6b — the FIM investigation — and rewriting
  the §6/§7 commentary to remove the retracted "pooling would help" claim). All real outputs.
- `notebooks/30_wu2003_claims_and_conclusions.ipynb`: re-executed once (reads live from
  `results/31_*.json/csv`, so numbers updated automatically; §7/§9 commentary and the
  limitations table cell manually updated to describe the corrected mechanism and add L4″).
- `results/31_fault_classification_metrics.csv`, `results/31_classification_summary.json`:
  regenerated (final: 87.4% accuracy, macro-F1 0.694).
- `figures/nb31_confusion_matrix.png`: regenerated.
- `article_outline_CChE.md`: §7.3 rewritten with final numbers and the corrected mechanism;
  §8.1 updated from "zero-for-two" to "zero-for-three"; §8.4 limitations table: removed the
  now-superseded "L11" entry (wrong diagnosis) and replaced with **L4″** (correct mechanism,
  positioned next to L4/L4′ since it's the same family of finding); pre-submission checklist
  nb30/nb31 entries updated with final numbers.

**Still true, unaffected by this sub-session:** everything in "Finding 9" / §7.4 / §8.4 L4′
below (the (α, β_r) headline retraction) — this sub-session's finding is a *third, separate*
instance of the same phenomenon, not a revision of it. The decision not to attempt the
embedding-net SBI retrain (§7.4) still stands and was not revisited.

---

# Session Handoff — nb30/nb31 built and executed; article checklist updated

**Date:** 2026-07-05, continuation. Picked up exactly where the prior handoff (below) left
off: "Still outstanding" item 4 — build nb30 (claims-and-conclusions synthesis) and nb31
(fault classification), paused at the start of this session per explicit prior-session
instruction, against the *current* artifact-diagnostic article framing (not the retracted
"genuine banana" framing). **Both notebooks are now built and executed end-to-end with real
outputs — no fabricated numbers anywhere in either.**

## What was built this session

- **`notebooks/31_wu2003_fault_classification.ipynb`** (new, executed). Posterior-mass fault
  classification (same methodology as the PO system's `nb11`/§4.5/§6.2): a new
  `classify_fault_recycle`/`compute_classification_metrics_recycle` pair (written into the
  notebook, generalising `cstr_sbi.metrics`'s PO-specific 2-D/4-class version to this plant's
  5-D/5-class `FAULT_UNITS` taxonomy), using the **same 0.85 relative threshold** already
  established for the PO system, applied uniformly to all 5 parameters (one-sided for
  `z_A0_eff`, since every feed fault here is lean, never rich). Run: 14 closed-loop scenarios
  (`list_closed_loop()`) x 30 replicates x 200 posterior draws, calibrated seed-4 S-B
  posterior only (S-A intentionally **not loaded** — 0/40 seeds passed SBC, §8.4 L10, not
  used for any quantitative claim). Uses `sample_posterior_direct` (the nb27 fix for sbi
  0.24's unconditional prior-support-rejection hang) throughout, as a precaution — no hangs
  observed at these 14 in-distribution scenarios, but cheap insurance.
  **Result (SUPERSEDED — see the sub-session above): 84.0% accuracy, macro-F1 = 0.669**
  (per-class F1: healthy 0.667, reactor 0.916, column 1.000, feed 0.000, multi 0.764). These
  numbers were from the labelling-bug-affected `fault_unit()`; the final, corrected numbers
  are **87.4% accuracy, macro-F1 = 0.694** (see top of this file). Saved to
  `results/31_fault_classification_metrics.csv` and `results/31_classification_summary.json`;
  confusion-matrix figure at `figures/nb31_confusion_matrix.png` (both regenerated since).
  - **W11 (the (α, β_r) "banana" scenario) classifies correctly as `reactor` in 30/30
    replicates**, despite the underlying posterior correlation(α, β_r) = 0.998 — because both
    parameters map to the same fault unit, the representation artifact (§7.4/Finding 9)
    corrupts *parameter-level attribution* but not *unit-level detection*. This is a clean,
    non-obvious, real result and is the notebook's central "worked example of the
    artifact-diagnostic" payoff requested by the article outline's §7.3 note.
  - **W12 (the retracted (α, η_col) candidate) also classifies perfectly (30/30) as
    compound**, with zero reactor/column leakage — confirms §7.2.3's retraction at the
    classification level too, exactly as the article outline predicted it should.
  - **New finding, not previously documented anywhere in this project: feed-fault (W10,
    z_A0_eff=0.78) detection fails completely at single-replicate granularity (F1 = 0.00,
    0/30 correct, all called `healthy`).** **The mechanism proposed here — "a detection-power
    problem (posterior width vs. fault size at one window), fixable by pooling evidence
    across consecutive windows" — is WRONG and is RETRACTED by the sub-session at the top of
    this file.** The real mechanism is a third representation artifact, (α/β_r, z_A0_eff),
    confirmed by FIM (see top of file, and article §8.4 L4″). Left here for provenance only —
    do not cite this paragraph's explanation.
  - **W15 turned out to be a labelling-convention edge case, not a classifier error — FIXED,
    see top of this file.** Its ground-truth `fault_unit()` label was `multi` only because
    the scenario *name* contains "snowball" (a keyword-matching rule), while its actual
    η_col=0.90 deviation (10% below nominal) falls *below* the same 0.85 threshold used
    consistently everywhere else in this taxonomy. This paragraph originally recommended
    "but did not implement" fixing `fault_unit()` — **it has since been implemented** (same
    session, later sub-session): rewritten to a pure threshold-based rule, `nb31` re-run,
    W15 now classifies correctly (30/30).
- **`notebooks/30_wu2003_claims_and_conclusions.ipynb`** (new, executed). Styled after `nb14`
  (the PO system's equivalent): synthesises nb20–nb29b and nb31 into a figure-illustrated
  narrative, a live quantitative dashboard (reads `wu2003_posterior_sb.pkl`'s real ensemble
  metadata and `nb31`'s real JSON output — nothing hardcoded that could instead be loaded from
  disk), a limitations table (L3/L4/L4′/L7/L9/L10 plus the new L11), and a pre-submission
  checklist. This is the Wu-2003-plant counterpart to `nb14`, covering article §7–§9.
- **`article_outline_CChE.md`** updated: §7.3's placeholder bullets (which said "nb31 ...
  to be created" and described *expected* results) replaced with the real nb31 numbers above;
  the pre-submission checklist's nb20–nb31 block changed from mostly `[ ]` to `[x]` with
  citations to the executed notebooks; a new limitation **L11** added to the §8.4 table for
  the feed-detection finding; fixed a pre-existing Table-numbering inconsistency (§7.3's
  inline text said "Table 9" for the classification table while the "Tables summary" master
  registry called the same table "Table 10" — now consistently "Table 10" in both places).

## What is still outstanding (carried forward, unchanged in substance)

- The decisive-but-not-attempted fix from Finding 9 (raw-trajectory-aware embedding-net SBI
  retrain for (α, β_r)) remains explicitly out of scope this session, per the user's prior
  decision — see "Finding 9" below.
- Checklist items not yet done: full 300 dpi/double-column figure regeneration pass,
  transcribing Table 10 into actual manuscript prose, nomenclature table, citations audit,
  data-availability/CRediT/COI boilerplate. None of these are further scientific
  investigation — they are manuscript-production tasks.
- `scripts/build_nb_*.py` — still stale/disconnected, not touched this session, safe to
  delete once no longer needed for provenance.

---



**Date:** 2026-07-05, continuation. This session investigated Finding 10, then found strong
counter-evidence for Finding 9 (the (α, β_r) "genuine banana"), then — **at explicit user
decision** — chose **not** to attempt the decisive fix (retraining SBI with a raw-trajectory
embedding net), and instead to report this as a validated methodological caution. **The
article outline (`article_outline_CChE.md`) has been substantially rewritten this session
to reflect that decision as the paper's final framing, not a placeholder.** The new
headline for the recycle-plant section is not "here is a banana" but **"both candidate
joint degeneracies investigated in this plant were artifacts of the observation
representation, not physics; the one identifiability limit that is genuine is a scalar
masking effect that transfers exactly from the simpler system."** This required edits to
the Highlights, Abstract, Contributions (§1.2), §3.2, §7.2, §7.4 (rewritten in full), §7.5,
§7.6, §8.1 (rewritten in full), §8.2 (rewritten in full), §8.4 (L4, L4′), §9 Conclusion, and
the pre-submission checklist's nb30/nb31 notes. Read the article outline directly for the
final wording — it is now internally consistent (no more "PROVISIONAL"/"pending" language
left over from mid-investigation). Prior investigation detail (Finding 10 correction,
Finding 9 counter-evidence, the subwindow negative control) is preserved below under
"Finding 10" / "Finding 9" and remains accurate.

---

## Finding 10 — CORRECTED (was: unresolved/suspicious; now: mechanism identified)

Investigated in `notebooks/27_wu2003_sequential_tracking.ipynb` §9 (new section, executed
in-notebook — all numbers below are real outputs, not hand-transcribed). Four control
experiments, in order:

1. **Same-shape trajectories** (both α and β_r linear decay instead of different shapes):
   EKF tracking accuracy unchanged from baseline. Rules out "differing temporal shapes let
   a recursive filter disambiguate" (hypothesis (a) from the prior handoff).
2. **Static truth held at the exact W11 point (0.80, 0.80) for all 360 windows:** EKF
   converges to near-perfect accuracy (err ≈ +0.003/+0.017) already in **window 0** — not
   gradually over the run. Rules out "many windows of accumulated information" as well.
3. **P/Q tuning swap on the identical single noisy W11 window:** nb26's tight parameter
   covariance (`P[6:,6:]≈1e-4`) reproduces nb26's own reported failure (err +0.107/+0.186,
   matching its "+0.10/+0.19" almost exactly); nb27's diffuse tuning (`P[6,6]=0.05,
   P[7,7]=0.02`) on the *same data* converges to err +0.003/+0.003. **The EKF tuning
   difference between nb26 and nb27, not sequential tracking, is what flips the outcome.**
4. **Robustness check** (15 noise seeds at W11, 4 more truth points spanning the
   identifiability-scan grid): nb27's diffuse-tuning EKF consistently recovers (α, β_r) to
   within ~1-2% everywhere tested — not a lucky seed or a coincidental attractor at one
   point.

**Conclusion:** Finding 10's original "EKF is recursive, SBI is memoryless" explanation is
wrong. **Do not cite it.** The real explanation raises a bigger issue — see next section.

## Finding 9 — decisive counter-evidence: (α, β_r) is very likely a feature artifact

Follow-up investigation in `notebooks/29b_identifiability_scan.ipynb` §§3-4 (new sections,
executed in-notebook, run explicitly "aligned with nb26" per user request — same grid,
same truth `truth_w11 = [0.80, 0.80, 1.0, 1.0, 0.90]`, same `run_scan`-family
infrastructure where possible). Two follow-up tests, in order:

**§3 — prior-normalised distance scan, extended to richer features (ambiguous on its own).**
Extended `scripts/identifiability_scan.py` (new, additive `run_scan_custom` /
`sim_subwindow_summary` / `sim_raw_trajectory` / `compute_norm_std` — the existing `run_scan`
nb26/nb29b already use is untouched) to run the *same* (α, β_r)-at-W11 grid scan with a
108-D per-sub-window summary and a 360-D raw-trajectory representation (the same 3 channels
the EKF observes: T_r, T_j, F_R_norm), instead of `compute_summaries`'s 66-D whole-window
aggregation. Result: the shallow diagonal valley's width barely changes (0.138 / 0.138 /
0.115 for robust-26D / subwindow-108D / raw-360D) — **on its own, this looks like it
contradicts the nb27 EKF finding.**

**§4 — noise-calibrated FIM, exactly the article's own §7.2.2/nb23 methodology (decisive).**
The scan in §3 normalises distance by *prior-driven* spread (how much a feature moves
across the whole parameter range) — the wrong yardstick for "is this resolvable at the
actual ~0.3% sensor noise level." Redid the comparison with `FIM = J^T Σ^{-1} J`, Σ from
*real replicate noise variance* (exactly nb23's Figure-8 methodology, which is where the
article's headline **+0.901** off-diagonal number comes from). Sanity check first
reproduces that number in magnitude (`compute_summaries` gives off-diagonal ≈ −0.6 to
−0.85 across seeds and at both nominal and W11 — same strong near-degenerate coupling, sign
is a re-implementation/parametrisation-convention detail, not a discrepancy). **Then, on the
exact same 3 raw physical channels (T_r, T_j, F_R_norm), unaggregated: off-diagonal
collapses to ≈ 0.00 (range −0.033 to +0.021 across 3 independent noise-seed offsets at
nominal, and −0.028 at W11) — every single time tested.** This is a clean, reproducible,
~30x collapse in the coupling strength, using the article's own established, noise-realistic
FIM methodology, not an ad hoc metric.

**Why §3 and §4 disagree, and why §4 is the one to trust:** §3 asks "do these grid points
look far apart relative to typical parameter-driven variation" (a question §4 doesn't ask).
§4 asks "does perturbing α vs. β_r locally move the observation in collinear directions at
the real noise scale" — which is what actually determines a trained posterior's achievable
precision, and exactly what the article's own +0.901 number claims to measure. §4's answer:
under `compute_summaries`, yes, nearly perfectly collinear; under the raw signal, no.

**Follow-up (2026-07-05): the fix is harder than "add more summary features" — modest finer
time-resolution does NOT help.** Ran the same §4 FIM methodology on the `subwindow`
representation (108-D: same 9 S-B channels, split into 6 sub-windows with mean+std each —
6x finer time resolution than `compute_summaries`, but still an aggregate/hand-crafted
statistic, not raw access). Result: off-diagonal stays at ≈ −0.62 to −0.86 (nominal: −0.835,
−0.851, −0.855 across 3 seeds; W11: −0.619) — **no better than the original 66-D summary,
despite 6x finer time bins.** Only the fully unaggregated 360-D raw trajectory (previous
result) collapses the coupling. **The informative signal that separates α from β_r lives in
fine-grained transient shape/timing that mean+std aggregation destroys even at 20-step
resolution — this cannot be fixed by a modest hand-crafted feature addition (unlike nb29's
`reb_per_boilup` fix for the unrelated, now-retracted (α, η_col) confound). It would require
a genuinely raw-trajectory-aware representation** (a CNN/RNN embedding net trained
end-to-end, as in article §6.3.3/`nb04b`) **or a hand-engineered fine-grained-timing feature
(e.g. per-channel rise-time/lag), not more summary-statistic bins.**

**Combined with nb27's independent EKF result** (an EKF given raw-trajectory access
resolves α, β_r to ~1-2% at W11 and 4 other grid points — see "Finding 10" above), there are
now **two independent methods, one of them using the article's own headline methodology,**
pointing the same direction: **the (α, β_r) "genuine banana" is very likely an artifact of
the 66-D hand-crafted summary-statistic feature set (`compute_summaries`), not a physical,
sensor-independent non-identifiability of the plant — but the practical fix is a real
architecture change (embedding net), not a quick feature-engineering patch.**

**What is still NOT shown (do not overclaim further than this):** neither test is a trained
SBI posterior. The actual decisive test — matching the article's own gold-standard
methodology for exactly this question (§6.3.3/`nb04b`'s raw-CNN-embedding irreducibility
test for the propylene-oxide system, where it *confirmed* a hand-crafted-feature limit as
physical) — is to retrain SBI with a raw-trajectory-aware embedding net for Wu 2003 and check
whether the *trained, calibrated* posterior's (α, β_r) correlation and CI width actually
shrink at W11. Given the `subwindow` null result above, this is now known to require a
bigger architecture change than originally hoped, and this project's own training-
instability history (L9/L10 — SNPE is seed-unstable for this 5-param/66-72D problem *even
with hand-crafted features*, S-A calibration never succeeded at all across 40 seeds) makes
an embedding-net retrain a *higher*-risk undertaking than a modest feature change would have
been — plausibly a multi-session effort, not a quick follow-up.

**Action item, top priority for next session:** decide whether to attempt the embedding-net
SBI retrain (high effort/risk, would let the article claim an actionable fix if it works) or
to report this as a validated methodological caution without attempting the fix (lower risk,
still a real contribution — "always FIM-check a hand-crafted-feature non-identifiability
against raw-trajectory access before reporting it as physical"). **In the meantime,
`article_outline_CChE.md` has already been updated this session** (see below) to stop
presenting §7.2.2/§7.4/§8.1/§8.4(L4′) as a settled finding and to reflect this evidence —
see the article outline itself for the specific edits, and revert/firm them up once a
decision is made. Full detail and real printed numbers are in `nb27` §9 and `nb29b` §§3-4's
markdown/code cells.

---

## Prior session context (2026-07-04, first continuation — nb26/29b rebuild)

**What follows was written before Finding 10 was investigated; it is retained for
provenance. The "Still outstanding" list below is superseded by the two sections above.**

## READ FIRST — what's new this session

1. **`notebooks/26_wu2003_headline_banana_ekf_failure.ipynb` — fully rebuilt from scratch.**
   All old W12/(α, η_col) cells, prose, and figure references removed. New headline: at W11
   (α=0.80, β_r=0.80), SBI S-B shows the genuine banana (corr≈0.997), SBI S-A does **not**
   resolve it (corr≈0.999 — a newly *tested*, not just inferred, confirmation), and EKF
   fails completely (0% coverage both params). Executes live end-to-end (no OOM, no
   standalone-script workaround needed) — see "EKF speedup" below for why.
2. **`notebooks/29b_identifiability_scan.ipynb` — new.** Formalises the deterministic
   identifiability-scan methodology: side-by-side heatmaps showing (α, η_col) at W12
   collapses to a localized region once F_R is combined with T_reb (retraction evidence),
   while (α, β_r) at W11 keeps an extended diagonal valley even combined (confirmation
   evidence). This is the evidence backing Figure 8/10b referenced in the article outline.
3. **`notebooks/24` and `notebooks/25` — addenda added, one figure title corrected.**
   nb24's own W12 corner plot (`nb24_w12_posterior_sb.png`) already showed the real
   (α, β_r) banana (corr≈1.00) alongside a non-banana (α, η_col) (corr≈0.13) — it was just
   mislabeled with the old title. Title fixed, figure regenerated, addendum added
   explaining the correction. nb25 gets an addendum documenting the settled-negative S-A
   architecture sweep (Finding 7 below) and the tested (not just inferred) result that S-A
   does not help with (α, β_r) either.
4. **`notebooks/29_etacol_sbc_investigation.ipynb` — addendum added.** Its confound
   diagnosis and fix (`reb_per_boilup`) both still stand and are corroborated by nb24's
   passing SBC. Its interpretive claim about what a narrow η_col posterior *means* is
   corrected: narrow-and-honest is the *correct* answer (η_col is genuinely identifiable via
   T_reb), not a residual approximation defect, per nb29b.
5. **`notebooks/27_wu2003_sequential_tracking.ipynb` — executed successfully for the first
   time.** Found and fixed a real hang (not the one the prior handoff described — see API
   correction below) and a >12h-runtime EKF bottleneck. Produced a result that **should be
   treated as suspicious, not accepted at face value**: EKF tracks (α, β_r) almost
   perfectly while SBI fails badly — see Finding 10 and the explicit warning there. **Do
   not cite Finding 10's numbers or explanation in the article until this is investigated.**
6. **API correction (IMPORTANT, supersedes prior handoff's note):** sbi 0.24 did **not**
   make direct/non-rejection sampling the default when it removed `reject_outside_prior`.
   `DirectPosterior.sample()` unconditionally rejects any proposal outside the prior box
   (`accept_reject_fn=lambda theta: within_support(self.prior, theta)`), with no opt-out.
   This **hangs** (0% acceptance, infinite loop) whenever the flow's mass for a given
   observation lies outside the prior box — which happened in nb27 around day 28-30 of the
   360-window run (ξ_reb drifted to ~1.27-1.28, prior upper bound 1.2). Fix: sample
   `posterior.posterior_estimator` directly — this is the actual equivalent of the old
   `reject_outside_prior=False`. **If any other script/notebook calls `.sample()` on an
   out-of-training-distribution or sequentially-drifted observation, check for this hang
   risk before assuming the removed kwarg is harmless.**

---

## Finding 10 (ORIGINAL WRITE-UP, SUPERSEDED — see "Finding 10 — CORRECTED" at the top)

nb27 tracks a 30-day linear α decay (1.0→0.65) and Kern-Seaton β_r fouling (1.0→0.90) over
360 sequential 2h windows, comparing SBI (posterior re-sampled fresh each window) against
an augmented EKF (recursive, covariance carried window-to-window).

| Param | Method | MAE | Bias | 90% Coverage |
|---|---|---|---|---|
| α | SBI | 0.048 | −0.048 | 0.15 |
| α | EKF | 0.002 | +0.001 | 1.00 |
| β_r | SBI | 0.198 | −0.198 | 0.12 |
| β_r | EKF | 0.004 | +0.004 | 0.96 |

**This is the opposite of every other SBI-vs-EKF comparison in this project, and it is
specifically suspicious, not just "an interesting reversal."** This session's write-up
(now revised) offered a "recursive vs. memoryless" explanation for why SBI loses here — but
that explanation does not obviously square with the rest of this project's own findings.
(α, β_r) is the plant's confirmed genuine banana (Finding 9): the identifiability scan
(nb29b) shows the degeneracy along an extended diagonal valley, and separately, the EKF has
been shown *repeatedly* in this project (W11 in nb26, W12/W15 previously) to fail badly at
representing exactly this kind of curved/non-Gaussian joint uncertainty via linearisation.
**Yet in nb27 the same EKF design tracks both α and β_r individually with near-perfect MAE
and 90-100% coverage.** If (α, β_r) were genuinely non-identifiable from S-B observations in
the way nb29b claims, no filter — recursive or not — should be able to pin both down
accurately from this data. Either:
(a) the sequential setup provides real disambiguating information a static snapshot
    doesn't (α and β_r evolve on different functional shapes — linear vs. Kern-Seaton — so
    their time-derivatives differ, which a recursive filter can exploit even though a
    single 2h window cannot; SBI's per-window resampling never sees this because it treats
    each window as an independent random prior draw), or
(b) something about the EKF setup, the SBI sampling fix (`sample_posterior_direct`,
    bypassing prior-support rejection), the training-bank/theta_true realism, or the
    "recursive vs memoryless" framing itself is masking a bug or a mismatched comparison,
    and the apparently-clean EKF result is not trustworthy either.
**This has not been distinguished. The next session should investigate (a) vs (b) before
citing Finding 10's numbers, explanation, or qualitative conclusion anywhere in the
article.** A concrete starting point: check whether EKF's near-perfect tracking survives if
`theta_true`'s two trajectories are swapped/perturbed to have similar shapes (if (a) is
right, disambiguation should degrade), and whether SBI's failure is really about
memorylessness or is dominated by the raw-flow-sample-mean sensitivity issue noted below
(try a per-window sequential Bayesian update chaining SBI posteriors, or at minimum a
median/trimmed-mean instead of the sample mean, before concluding SBI structurally cannot
compete here).

Other partial explanations already identified, still worth checking but not sufficient on
their own: part of the gap is the already-known systematic downward bias (structural, Loop 1
masking); part (specifically β_r's larger-than-usual −0.198 bias) traces to the first ~8
days where the un-rejected raw flow samples are more scattered — a median/trimmed-mean
summary instead of the sample mean would likely be more robust.

---

## What changed on disk this session

- **Rewrote:** `notebooks/26_wu2003_headline_banana_ekf_failure.ipynb` (from scratch).
- **New:** `notebooks/29b_identifiability_scan.ipynb`.
- **Edited (addenda + one figure-title fix, all executed, real outputs):**
  `notebooks/24_wu2003_sbi_training_sb.ipynb`, `notebooks/25_wu2003_sbi_training_sa.ipynb`,
  `notebooks/29_etacol_sbc_investigation.ipynb`.
- **Fixed and executed:** `notebooks/27_wu2003_sequential_tracking.ipynb` — removed
  now-invalid `reject_outside_prior` kwarg (2 call sites), replaced manual
  finite-difference EKF with `jax.jacfwd`+`jit` (same pattern as nb26), added a
  `sample_posterior_direct()` helper to avoid the rejection-sampling hang, fixed a
  pre-existing `N_WINDOWS`/"720" labelling bug (30 days × 12 windows/day = 360, not 720)
  including a real bug in the timing table (hardcoded `720` instead of `N_WINDOWS`).
- **New figures:** `nb26_w11_headline.png`, `nb26_w11_physics_origin.png`,
  `nb29b_alpha_etacol_retraction.png`, `nb29b_alpha_betar_confirmation.png`.
- **Regenerated (title/content fix):** `nb24_w12_posterior_sb.png` (now correctly shows
  which pair is/isn't a banana).
- **Regenerated (first successful run):** `nb27_degradation_profile.png`,
  `nb27_sequential_tracking.png`.
- **No posterior `.pkl` files changed** — `wu2003_posterior_sb.pkl` (seed 4) and
  `wu2003_posterior_sa.pkl` (seed 13, `calibrated: False`) are untouched.

---

## Still outstanding (in priority order)

0. ~~run the nb04b-style raw-trajectory/richer-feature irreducibility test for (α, β_r)~~ —
   **done.** Two independent tests (`nb29b` §3 geometric scan — ambiguous — and §4
   noise-calibrated FIM — decisive, off-diagonal collapses ~0.6-0.85 → ~0.00, plus a
   negative `subwindow` control ruling out an easy fix) corroborate nb27's EKF finding.
1. ~~investigate Finding 10~~ — **done.**
2. ~~Decide: attempt the SBI-retrain fix, or report as a validated caution~~ — **decided by
   user: option (b), report as a caution, do not attempt the retrain.**
3. ~~Propagate into `article_outline_CChE.md`~~ — **done this session, finalized (not
   provisional)**: Highlights, Abstract, §1.2 Contributions, §3.2, §7.2, §7.4 (rewritten),
   §7.5, §7.6, §8.1 (rewritten), §8.2 (rewritten), §8.4 (L4, L4′), §9 Conclusion, and the
   nb30/nb31 checklist notes. New framing: "both candidate joint degeneracies in the
   recycle plant were representation artifacts; the one genuine limit is a scalar masking
   effect transferring from the PO system." **This is now the article's stated position —
   no further verification of Finding 9 is planned.**
4. **nb30/nb31 (claims-synthesis / fault-classification)** — still not built. Per explicit
   user instruction, **paused before starting these** — this is the next task when work
   resumes. Build them against the *current* article framing (§7.4/§8.1's artifact-diagnostic
   story), not the retracted "genuine banana" framing — see the updated checklist notes in
   `article_outline_CChE.md`'s Pre-submission checklist section for what nb31 specifically
   needs to verify/frame.
5. **`scripts/build_nb_*.py`** — still confirmed stale/disconnected; safe to delete once no
   longer needed for provenance. Not touched this session.

## Push reminder (carried over, still applies)
```bash
rm /Users/simo/inso-po-RD/cstr-model-optimisation/.git/hooks/pre-push
git -C /Users/simo/inso-po-RD/cstr-model-optimisation push origin main
```
