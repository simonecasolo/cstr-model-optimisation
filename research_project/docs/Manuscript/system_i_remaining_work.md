# What remains to complete the full System I study

**Scope.** System I = the single propylene-oxide CSTR with PI temperature control
(`notebooks/system_I/`, `main.tex` §§"System I"/"Results: System I"). This document
is deliberately narrow — System II's (Wu2003 recycle plant) open items are tracked
separately (`docs/Manuscript/system_ii_sb_identifiability_story.md` §7, `HANDOFF.md`'s
2026-10-06 entry) and are **not** repeated here, even where they're more numerous and
more urgent right now. The headline finding: **System I's core science is settled**
— its matched-protocol identifiability cascade is complete, unlike System II's
still-open calibration gap — but several integration, consistency, and
manuscript-completeness items remain, one of which is a concrete, verifiable gap
found while writing this document, not a vague "needs review."

---

## 1. Status: what is actually done (verified directly, not assumed)

- **The matched-protocol retrain is complete and is the source of every number
  currently in `main.tex`.** The original production posterior had a
  fixed-healthy-warm-start bug shared with NUTS and the CNN-embedding experiment
  (but *not* EKF/UKF, whose filter state initializes from real data) — this was the
  paper's own Major Comment 7 finding. Full-budget ($n=10{,}000$) matched-protocol
  retrain: `beta` bias collapsed from $-0.149$ to $-0.002$ (not just improved —
  essentially eliminated), MAE improved ~13×. SBC moved from rejected (`alpha`
  $p=0.016$, `beta` $p=0.014$) to passing: `main.tex`/SI §S6 report $p=0.21$
  (`alpha`) / $0.13$ (`beta`) as the headline figure from the promoted production
  posterior, and §S6 separately documents (and explicitly reconciles) that SNPE
  training is seed/realisation-sensitive enough that three independent re-runs of
  SBC against the same posterior give KS $p$ in $0.30$–$0.98$ (`alpha`) /
  $0.18$–$0.78$ (`beta`) — every run passes, so the *substantive* conclusion
  (well-calibrated) is robust even though the exact decimal is run-specific. This
  is already correctly explained in the SI text itself, not a loose end.
- **All 19 System I notebooks (`00`–`18`) have executed outputs** — verified directly,
  not assumed from filenames. Includes the later additions (`15_beta_bias_analysis`,
  `16_ekf_ukf_baseline`, `17_generate_supplementary_figures`, `18_ablation_study`) that
  postdate the original `14_claims_and_conclusions` synthesis notebook.
- **`main.tex`'s Table VII (multi-method comparison) is the final, matched-protocol
  version** for all four methods (SBI/NUTS/EKF/UKF) — confirmed by direct inspection,
  not just citation: SBI $\hat\beta=0.698$, bias $-0.002$; NUTS $\hat\beta=0.700$,
  bias $-0.0002$; EKF/UKF both $\hat\beta=0.607$, bias $-0.093$. This closes out
  Reviewer Major Comment 7's System I thread and Major Comment 2's Cramér-Rao-bound
  framing fix.
- **`supporting_information.tex` §S3 (Training Budget Sensitivity) and §S6 (Posterior
  Calibration)** already describe the matched-protocol results, not the stale
  pre-fix numbers — directly confirmed by grepping for the matched-protocol language
  and the specific KS p-values inside both sections. §S3a (Extended Hyperparameter
  Ablation) and §S6a (System I constants) also exist and correspond to the later
  notebooks (`18_ablation_study` and friends).

**Bottom line: the core identifiability/calibration science is settled and
internally consistent.** But "settled" does not mean "nothing left" — §2–4 below
list a genuine open scientific question, several concrete, verified manuscript gaps
(including one — §3.0a — that the SI's own text explicitly flags as not yet done,
not something this document is speculating about), and housekeeping items. All were
found by direct inspection of the current files, not inferred from planning-doc
dates (several of this project's own planning docs are themselves 5–7 weeks stale
relative to the repo's actual state — e.g. `HANDOFF.md`'s "fully DONE" declaration
for System I predates `17`/`18`'s work and the items below by about 7 weeks).

---

## 2. Open scientific question: the EKF/UKF-vs-SBI asymmetry between the two systems

**What's unexplained.** System I's own multi-method comparison and 30-day tracking
**both** favour SBI over EKF/UKF throughout — there is no regime in System I where the
filter wins (EKF/UKF bias $-0.093$ vs. SBI's $-0.002$ at Sc2; EKF/UKF MAE roughly an
order of magnitude worse than SBI in 30-day tracking). System II shows the opposite
split: SBI wins on single-window classification near nonlinear (snowball) regimes, but
a well-tuned EKF with raw-trajectory access **outperforms** SBI in 30-day tracking
(`docs/Manuscript/system_ii_sb_identifiability_story.md` §5 has the detailed mechanism
for System II's own split). Naively, the same "raw-trajectory access defeats
summary-compression information loss" argument that explains System II's EKF
advantage in tracking should also apply to System I's own genuine information deficit
for `beta` (§3.1 of the identifiability-story doc, the ~250-500× FIM asymmetry) — but
it doesn't show up there.

**Candidate explanations, not yet tested:**
- System I's physics-informed summary features (`UA_eff_proxy`, `k0_eff_proxy`) may
  already capture most of what a raw-access filter would otherwise gain, closing the
  compression gap that otherwise favours EKF in System II.
- The System I EKF/UKF implementations may simply be less favourably tuned than
  System II's (a mundane but real possibility — nothing in the two systems' filter
  setups has been directly compared for tuning quality).
- System I's state space (3 states) vs. System II's (up to 8, plus recycle delay) may
  make System I's linearisation a much better local approximation throughout, not just
  away from a tipping point — reducing the EKF's structural disadvantage there.

**What to do:** either investigate this directly (compare filter tuning
methodology/gain matrices between the two systems' EKF implementations; check whether
System I's engineered features really do match raw-trajectory information content via
an RT-FIM check analogous to the one already used for System II), or explicitly scope
the manuscript's Discussion to avoid asserting an unestablished causal account for the
asymmetry. Currently `main.tex`'s Discussion (`sec:disc_ekf_regimes`) describes each
system's EKF behaviour separately and correctly, but does not claim to explain the
cross-system asymmetry — that is honest as far as it goes, but means a reviewer could
reasonably ask "why do these differ?" and the paper currently has no answer prepared.

---

## 3. Genuinely open, not-yet-started items (verified, not inferred from stale planning docs)

### 3.0a Multi-seed calibration corroboration — explicitly recorded in the SI itself as the next step, not yet run

`supporting_information.tex` §S3a (Extended Hyperparameter Ablation) states
directly: none of its nine tested architecture/budget configurations passes SBC on
both `alpha` and `beta` simultaneously — including the configuration matching the
production posterior's own architecture and budget, trained as an independent
retrain. The section's own conclusion: a single SNPE training run (including the
currently-promoted production posterior) is "one draw from a calibration-sensitive
process, not a guaranteed-calibrated artefact," and recommends a multi-seed sweep
at the production architecture/budget to confirm calibration holds across an
ensemble — mirroring the multi-seed-ensemble-and-select protocol already used for
System II. **The SI's own text states: "This corroboration has not yet been run
for System~I and is recorded here as the concrete next step."** This is not a
speculative gap — it is the paper's own documented unfinished item, found this
session by reading §S3a directly. Running it is cheap relative to anything in
System II's current campaign (System I trains in seconds per seed, not hours).

### 3.0b Stage 5 item 17 — the three-way `s_UA` physics-proxy ablation (Major Comment 5.1)

`docs/Manuscript/notebook_execution_plan.md` Stage 5 lists a three-way comparison
of the `s_UA` physics-informed summary proxy (existing proxy vs. a corrected-balance
proxy vs. no jacket proxy at all) as a planned response to Reviewer Major Comment
5.1. No trace of this ever being executed was found anywhere in the repo (checked
`main.tex`, the SI, `pending_manuscript_fixes.md`, `HANDOFF.md`). This is the
clearest fully-unstarted System I item — confirm whether Major Comment 5.1 is still
being tracked as needing this specific ablation, or whether it was addressed a
different way and the plan entry is simply stale.

### 3.0c Minor, already-essentially-resolved items worth a final check, not full re-investigation

- **Major Comment 10's four-category taxonomy** (structural / practical /
  protocol-induced / representation-artefact) has not been formally inserted into
  `main.tex` as a named taxonomy, though the informal language ("protocol-induced")
  already appears in the relevant places (`main.tex` near the identifiability and
  conclusions sections). Cosmetic/framing gap, not a substantive one.
- **`article_outline_CChE.md`** still cites the retracted "250–500×" FIM ratio figure
  in six places. `main.tex` itself already uses the corrected figures (236× at the
  healthy point, ~12× at Sc2, 5–170× across the tested range, per `main.tex`
  Eq./discussion near the identifiability section). `article_outline_CChE.md` is
  explicitly documented elsewhere (`SI_completion_plan.md`) as "a stale planning doc,
  useful only for the science narrative, not structure/numbering" — so this is a
  low-priority stale cross-reference in a non-authoritative document, not a
  manuscript defect, but worth a find-and-replace if that outline is ever reused.
- **`nb09_sensor_drift_substudy`**'s result (the Sc7 sensor-bias/fault confound,
  catalogue entry A5 in `confound_and_artefact_catalogue.md`) is integrated into
  `main.tex` via a footnote, but its full quantitative write-up remains explicitly
  deferred to "SI-only future work" and has not been added. Low priority — the
  manuscript-level claim doesn't depend on the extended write-up existing.

---

## 4. Manuscript-integration items

### 4.1 Fold in (or explicitly decide not to) the cross-system `(alpha,beta)` vs `(alpha,beta_r)` comparison

`notebooks/34_alpha_beta_r_banana_cross_system.ipynb` (built this session) is a
complete, executed analysis showing System I does *not* replicate System II's
`(alpha,beta_r)` banana, and mechanistically why (System I's composition sensor `C`
structurally excludes `beta` from its own governing equation; System II's `z_A`
equivalent exists in the simulator but is never exposed as an observed channel).
This is a genuinely new, confirmed-by-ablation finding that currently exists only as a
notebook — it is not referenced anywhere in `main.tex` or the SI. Two options, not
yet decided:
- Promote it into the Discussion as the explanatory mechanism for *why* the two
  systems' reactor-side identifiability differs this much (currently `main.tex`'s
  Discussion §"One Genuine Limit, Three Representation Artefacts" gestures at this —
  "This effect appears strongly for beta in System I and more mildly for alpha and
  beta_r in System II" — without the composition-sensor mechanism nb34 established).
- Treat it as supporting material only (cite informally, don't add new manuscript
  text) if the identifiability narrative is being demoted further per the
  2026-09-28 thesis pivot (identifiability as explanatory device, not headline).

**This decision is blocked on nothing** — it can be made independently of System II's
open calibration work.

### 4.2 `supporting_information.tex`: confirm whether §S8 (System I's own Multi-Method
Baseline per-replicate detail) is missing by design or by accident

**Concrete finding, verified directly, not inferred:** the version of
`supporting_information.tex` at the last git commit (`HEAD`) has eleven numbered
sections, **S1 through S11**, including `§S8. Multi-Method Baseline: Implementation
Details and Per-Replicate Distributions` (this is System I's own EKF/UKF/NUTS/SBI
per-replicate detail — not System II's) and `§S7`/`§S9`/`§S10`/`§S11` (all System II
content). **The current, uncommitted working copy of `supporting_information.tex`
only goes up to §S6a** — sections S7 through S11 are not present in the file on disk
right now. This is consistent with the large, pre-existing, uncommitted reorganisation
already flagged in this session's git-commit work (deletions/renames spanning far
beyond System I or II specifically) — most likely an in-progress edit (plausibly
stripping stale System II SI content ahead of a rewrite) that has not yet restored or
reconciled System I's own §S8.

**Action needed:** before treating the SI as complete, confirm with whoever is driving
that reorganisation whether §S8's removal was intentional. If not, it can be recovered
directly from git history (`git show HEAD:research_project/docs/Manuscript/supporting_information.tex`
has the full S7–S11 text) and System I's §S8 content re-inserted without needing to
regenerate anything — the content itself was already correct/matched-protocol as of
the last commit, assuming the matched-protocol numbers haven't moved since (worth one
quick diff check against current `main.tex` Table VII before re-inserting verbatim,
since §S3/§S6 did receive later refinements S8 might also need).

### 4.3 Reviewer-response closeout check

`docs/Manuscript/reviewer_response_plan.md`'s Major Comments 2, 3, 7, and 10 all have
System-I-specific threads, and most show `[x]`-checked sub-items or "RESOLVED"
verdicts as of 2026-08-13. One item is explicitly conditional and should be
double-checked against the current manuscript text rather than assumed closed: Major
Comment 2's framing fix was gated on "Major Comment 7's CNN-embedding rerun... is done
or explicitly deferred with this caveat stated" — `pending_manuscript_fixes.md` records
this rerun as completed (`results/sbi_posterior_embedding_matched_protocol.pkl`,
single-replicate result $\hat\beta=0.728$ hand-crafted vs. $0.696$ CNN-embedding,
correctly described as single-replicate not population-level) and a specific broken
cross-reference to "Section S8" for this result was fixed by stating the number
directly in `main.tex` instead. Worth a two-minute check that this still reads
correctly in the current `main.tex`, given §4.2 above found the SI itself is
currently missing the S8 section this fix explicitly chose to route around.

### 4.4 Retire or refresh `notebooks/system_I/14_claims_and_conclusions.ipynb`

This notebook predates `15`–`18` (beta-bias analysis, EKF/UKF baseline, supplementary
figures, ablation study) and describes itself as synthesising "notebooks 02–12" — it
has not been updated to reflect the later additions. It has executed outputs, so it is
not broken, just stale as a "current state" reference. Low priority (it is a working
aid, not manuscript source), but worth either a refresh pass or an explicit note at its
top marking it superseded by `main.tex` itself as the source of truth, so a future
session doesn't accidentally treat its numbers as current.

### 4.5 `main.tex`: restore the commented-out "Multi-Method Baseline Comparison" subsection heading

**Concrete, verified, trivial-to-fix defect:** `main.tex`'s `\subsection{Multi-Method
Baseline Comparison}` and its `\label{sec:baseline_comparison}` are commented out
(`%\subsection{...}`, `%\label{...}`), while the subsection's own body text and
Table VII (`tab:method_comparison`) immediately below remain live and uncommented.
The result: this System I results subsection currently has **no visible heading in
the compiled PDF**, even though its content and table render normally. A one-line
fix (uncomment both lines) — flagged here because it is easy to miss precisely
because the content still compiles without error.

---

## 5. Summary checklist

| Item | Type | Blocked on System II? | Priority |
|---|---|---|---|
| **Multi-seed calibration corroboration (§3.0a)** | Open item, explicitly recorded in the SI itself | No | **High — the SI's own stated "concrete next step," cheap to run** |
| Stage 5 `s_UA` three-way ablation (§3.0b) | Never-started reviewer-response item | No | Medium — confirm still needed before running |
| EKF/UKF-vs-SBI cross-system asymmetry (§2) | Open scientific question | No | Medium — affects a Discussion claim's completeness, not a result's correctness |
| Integrate or scope nb34's cross-system finding (§4.1) | Manuscript integration | No | Medium |
| Resolve §S8 missing-from-working-copy (§4.2) | Manuscript completeness / possible accidental content loss | No (System I content specifically) | **High — verify before any SI finalization pass** |
| Restore commented-out Table VII subsection heading (§4.5) | Compiled-PDF defect | No | Low effort, should still be fixed |
| Reviewer Major Comment 2 cross-reference check (§4.3) | Verification | No | Low (likely already fine, quick check) |
| Major Comment 10 taxonomy formal insertion (§3.0c) | Framing/cosmetic | No | Low |
| Stale "250-500x" figure in `article_outline_CChE.md` (§3.0c) | Stale cross-reference in a non-authoritative doc | No | Low |
| nb09 sensor-drift full write-up (§3.0c) | Deferred SI content | No | Low |
| Refresh/retire nb14 (§4.4) | Housekeeping | No | Low |

**Two items (multi-seed corroboration, §S8 recovery) are genuinely high-priority and
cheap** — both are small, well-scoped, don't require new science, and were found by
direct inspection rather than inherited from a stale planning document. The rest are
real but lower-stakes. None require new simulation or training runs at meaningful
compute cost — System I's compute-heavy work is done. This is almost entirely
writing, verification, and git-hygiene work, and all of it can proceed in parallel
with System II's ongoing wide-prior campaign.
