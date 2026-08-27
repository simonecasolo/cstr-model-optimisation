# Session Handoff — 2026-08-18

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

## One thing to know about `main.tex` specifically

The user has been directly editing `main.tex` in parallel with sessions' work
(most recently § Structural Identifiability Analysis, the abstract, and the
Conclusions) — tightening language beyond what was auto-generated here. When
these two edit streams collided once already, it left a duplicated/orphaned
`\begin{table}...\end{table}` block that failed to compile; this was fixed.
**Always check `pdflatex -draftmode` compiles cleanly before assuming the file is in
a good state**, since it's being edited from two directions.
