# Plan: Make `supporting_information.tex` complete and publication-ready

**Status:** Not started (planning only, written 2026-08-10; re-verified 2026-08-11 against
the author's major restructuring/shortening pass on `main.tex`)
**Scope:** `docs/Manuscript/supporting_information.tex`, cross-checked against
`docs/Manuscript/main.tex` (revtex4-2, `aps,pre,twocolumn` — the authoritative current
draft, confirmed targeting *Computers & Chemical Engineering*, no layout changes in scope;
`article_outline_CChE.md` is a stale planning doc, useful only for the science narrative,
not structure/numbering).

## Update 2026-08-11: re-verified against the restructured `main.tex`

`main.tex` was substantially shortened (~11,500 → ~9,900 words, all headline numbers
pulled out of the Introduction and left only in Results). **Every `\label{}` anchor in
`main.tex` is unchanged** (diffed programmatically: identical 90-label set before and
after) — so all defects below that depend on `main.tex` labels (1, 5, 6) are re-verified
against the current text and still apply exactly as described, with one softening (see
defect 6). One **new defect was introduced** during this pass: a paste error in the SI
itself (defect 12 below) — a System-I paragraph was pasted into a System-II section.

## ⚠️ Update 2026-08-13: CONFIRMED at full budget with SBC — S3, S6, S8 need substantial revision, not just cosmetic fixes

**Update 2026-08-27:** System I's matched-protocol cascade is settled in `main.tex`,
but System II is still not ready for SI finalization. The full-budget S-B matched
ongoing-degradation run (`n=15,000`, 8 seeds) completed and failed SBC calibration
(0/8 seeds pass min-KS > 0.05; selected seed 3 fails independent `N_SBC=800`
confirmation with min KS $p=9.03\times10^{-8}$). Treat System II matched-protocol
results as a negative/diagnostic finding until the calibration failure is understood
or deliberately adopted as part of the study narrative. Do not finalize System II SI
sections from optimistic reduced-budget diagnostics.

**Status update (later same day): the diagnostic below is no longer pending — it is
now CONFIRMED at full production budget ($n=10{,}000$) with full SBC ($N=500$).** See
`pending_manuscript_fixes.md` Stage 3, Item 8 ("full validation — DONE") for complete
results. **Terminology note:** this finding is a **protocol-induced inference
artefact** — training and evaluation used different conditional data distributions
($p_\mathrm{train}(y\mid\theta)\neq p_\mathrm{test}(y\mid\theta)$) — and must *not* be
described as a "representation artefact" like System II's Artefacts 1–2 (which concern
the fixed-condition map $y\to s(y)$, not a train/test distribution mismatch). See
`reviewer_response_plan.md` Major Comment 10 for the full four-category taxonomy this
SI's narrative should now follow, and the "what remains valid" list at
`pending_manuscript_fixes.md` Stage 3, Item 8 for which closed-loop-identifiability
claims survive re-analysis and which do not.

While executing `reviewer_response_plan.md`'s Major Comment 7 / `notebook_execution_plan.md`
Stage 3, a matched-initialization diagnostic (training System I's SBI posterior with a
per-scenario, not fixed-healthy, warm-start — matching the evaluation-data protocol)
collapsed System I's headline $\beta$ bias by ~100× (from $-0.149$ to $-0.001$) and MAE
by ~13× at a *smaller* training budget than the current production posterior; the full
validation reproduced this at full budget ($-0.149\to-0.002$) with SBC also improving
substantially ($\alpha$ now formally passes, $\beta$ improves but still narrowly
fails). Code inspection (cheap, already done) additionally found that **NUTS and the
CNN-embedding experiment share the identical training/generative-model bug**, while
**EKF/UKF do not** — see the Item 8 write-up.

**Now that this is confirmed**, the following SI sections describe results that need
substantial revision, not just the structural/cosmetic fixes below — flagged here so
whoever finalizes the SI doesn't lock in numbers that are about to change:
- **§S3 "Training Budget Sensitivity"** — currently documents the *old-protocol*
  budget-sensitivity study ($\hat\beta$ shifting 0.680→0.618 as $n_\mathrm{sim}$:
  1000→10,000, all under the fixed-healthy warm-start). If the matched protocol is
  adopted, this study should be regenerated under it — the whole point of the
  section (motivating $n_\mathrm{sim}=10{,}000$) may look different once the systematic
  bias this budget was implicitly compensating for is removed.
- **§S6 "Posterior Calibration: SBC Protocol and Results"** — currently reports and
  interprets the KS-rejected $p=0.016/0.014$ failure as a "structural information
  deficit, not a training deficiency." If the matched-protocol SBC passes (or
  substantially improves), this section's entire interpretive argument inverts: the
  original failure would instead have been a training-protocol artifact, exactly the
  kind of thing SBC is supposed to catch and did catch — which is a *cleaner*, more
  favorable story for the paper's statistical rigor (SBC did its job) but requires
  rewriting the section's conclusion, not just its numbers.
- **§S8 "Multi-Method Baseline: Implementation Details and Per-Replicate Distributions"**
  — underpins Table VII (already regenerated once this session for the MAE/bias bug,
  see Stage 2). If the System I production posterior is retrained under the matched
  protocol, Table VII's SBI number changes again, and the section's framing ("all four
  methods show the same bias direction and comparable magnitude, confirming the offset
  is not unique to the SBI architecture") needs the same reframing as the main text
  (Major Comment 2's v3 wording) plus an explicit note that NUTS is not an independent
  check on this specific finding (shares the bug), while EKF/UKF are.
  **New concrete defect confirmed 2026-08-18** (`pending_manuscript_fixes.md`, Item 8,
  nb15/nb33 rerun): `fig:analytical_bias`'s caption (§S8, "Top right" panel) states the
  analytical 4-observable Fisher-information ratio is "$\approx 600\times$" and that
  $I_{\beta\beta}$ "relies entirely on the noisier $T_c$ and $Q_c$ channels" — both now
  directly contradicted by the matched-protocol rerun: the live ratio is
  $\approx 0.6\times$ (opposite ordering from the 29-D numerical result, not consistent
  with/amplifying it as the "Bottom right" caption text also claims), and $I_{\beta\beta}$
  is 100\% $T_c$, 0\% $Q_c$. `main.tex`'s equivalent passage
  (`sec:identifiability`) has been corrected accordingly this session; this SI figure
  caption has **not** — deliberately left alone pending the full S8 rewrite above rather
  than patched in isolation, since S8's framing needs the same "all four methods share
  one bug, EKF/UKF don't" reframing regardless. Whoever does the S8 rewrite should treat
  this caption as already-known-wrong, not a fresh finding to re-derive.

**Do not proceed past Step 6 below (final recompile) until this resolves** — recompiling
against soon-to-change numbers wastes the compile-and-check effort.

## Context

`main.tex` cites the SI ~15 times as `Section~S1`...`S11`, `Table~S1`/`S2`/`S3`,
`Fig.~S1`/`S3`. The SI currently has 12 physical section blocks (S1–S11, plus one
unnumbered stray section) covering: data generation, prior-predictive check,
training-budget sensitivity, PCA, feature/ablation tables, SBC protocol, System-II
scenario/training details, multi-method baseline detail, System-II features+calibration,
CNN embedding full results, sequential-tracking full results, and a Limitations table.

S1–S6, S8, S10, S11 are well-written, internally consistent with the paper's final
"zero-for-three" retraction narrative (Artefact 1/2/3, RT-FIM check terminology matches
main text exactly), properly captioned, and every referenced figure file exists on disk.
The defects below are localized, not systemic.

## Confirmed defects (verified via grep/diff of labels, refs, cites, and figure files — not just reading)

### Critical — will render as `??` or fail to compile cleanly

1. **13 broken cross-references.** The SI uses `\ref{...}` for labels that are only
   defined in `main.tex` (a separate compilation unit), so they will print as `??`:
   `sec:wu_artifact1`, `sec:wu_artifact2`, `sec:wu_artifact3`, `sec:wu_identifiability`,
   `sec:identifiability`, `sec:summary_stats`, `sec:wu_summary_stats`, `sec:fault_class`,
   `sec:disc_artefacts`, `sec:disc_practical`, `sec:sequential_tracking`,
   `sec:wu_sequential`, `tab:wu_cnn_classification`.
2. **Missing bib key**: `\citep{LopezPaz2017}` (C2ST reference, §S6) has no entry in
   `references.bib`. (Adjacent bug, same class, main-text side: `main.tex` cites
   `KernSeaton`, also missing from `references.bib` — fix in the same pass.)
3. **Empty section**: `S7.2 Fault scenario catalogue` is a bare `\subsection*` header
   with zero content before `S7.3` starts.
4. **Non-conforming section**: `\section{summary statistics}` (line 228, between S3 and
   S4) is the only section using `\section` instead of `\section*{Sx. ...}` — it will
   get an automatic number and break the visual S1→S11 sequence.

### Structural — SI numbering doesn't match what `main.tex` promises

5. No `\renewcommand{\thefigure}{S\arabic{figure}}` / `\thetable` (/ `\theequation`) in
   the SI preamble — figures/tables will compile as `FIG. 1`, `TABLE I`, etc., not
   `S1`, `S2` as `main.tex` explicitly promises.
6. Several inline `Section/Table/Figure Sx` citations in `main.tex` point at the wrong
   SI section given the SI's current physical order:
   - `Fig.~S1` (main.tex ~line 859) is cited for the prior-predictive figure, which
     actually lives in **S2**.
   - `Figures~S3` (main.tex ~line 865) is cited for PCA, which lives in **S4** (S3 is
     training-budget sensitivity — unrelated).
   - `Table~S1` (main.tex ~line 779) is cited for the 29-feature table, which lives in
     **S5**.
   - *(Softened by the restructuring, still a content gap.)* `main.tex` (~line 1138) no
     longer cites a specific "Table~S3" number for the promised per-parameter
     **mutual-information ranking for System II** — it now says only "detailed in the
     Supporting Information," a generic pointer. The numbering-mismatch bug is gone, but
     the underlying content still does not exist anywhere in the current SI (see defect
     10) and the sentence over-promises until it's added.

### Content — stale duplication, not genuine supplementary material

7. SI's `S9` **"Fault Classification"** subsection (~200 lines, prose form) is a
   near-verbatim duplicate of `main.tex`'s own §7.3 "Fault Classification" — same tables
   (`tab:wu_classification`, `tab:wu_ekf_classification`), same figure
   (`fig:wu_confusion`) — except it's the **older, unbulleted draft**; main text has
   since been rewritten as itemized bullets. Dead weight, not supplementary content.
8. The unnumbered `\section{summary statistics}` (physics-proxy features, Group 4)
   duplicates `main.tex`'s identical prose almost word-for-word, but drops the actual
   figure — it references `\ref{fig:physics_scatter}`, which is defined and included
   only in `main.tex`.
9. `S7.1` is titled **"Nominal parameter values"** but its body is actually the CL-vs-OL
   masking figure (`nb21_cl_vs_ol_masking`) — title and content are unrelated. The real
   nominal-parameter table already lives in `main.tex` Table 3.
10. Unused figure assets in `figures/` with no consumer in either document:
    `nb23_mi_analysis.png`, `nb23_sa_vs_sb_xd.png`, `nb23_tsne_sb.png`,
    `nb23_fim_heatmap.png` (non-`_single` variant) — likely the material needed for
    item 6's promised MI-ranking table/figure.
11. Minor: `\usepackage{tcolorbox}` is loaded in the SI and never used.
12. **New (introduced 2026-08-11): a System-I paragraph was pasted into a System-II
    section.** `git diff` on the SI shows an 8-line insertion at the end of §S9.1 (the
    66-D System-II feature-vector description), immediately before §S9.2: it is the
    System-I "macro-F1 = 0.990... Sc\,1--5, Sc\,7... Sc\,4 combined-moderate" snapshot-
    classification paragraph (identical text to `main.tex`'s System-I §"Training
    Validation and Fault Classification"), sitting where a System-II-specific sentence
    was clearly intended. It's out of place both topically (System I content inside the
    System II features section) and numerically (its own scenario IDs/values belong to
    the 2-parameter CSTR, not the 5-parameter recycle plant discussed around it).

---

## Implementation steps

### Step -1 — Remove the newly introduced paste error (defect 12, do this first — it's a one-paragraph delete)
- [ ] Delete the misplaced System-I snapshot-classification paragraph
      ("The trained NSF posterior applied 400 evaluation windows...") from the end of
      §S9.1, restoring the direct transition from the feature-vector description into
      §S9.2 "Posterior calibration."

### Step 0 — Freeze the section skeleton before editing prose
- [ ] Fold or delete the stray `\section{summary statistics}` physics-proxy block
      (defect 8). Recommendation: **delete** — fully covered in `main.tex` with its own
      figure; adds nothing S5 doesn't already cover.
- [ ] Retitle `S7.1` to match its actual content, e.g. "Control-masking mechanism,
      extended figure" (defect 9). Do not fabricate a parameter table — it already
      exists in main-text Table 3.
- [ ] Resolve empty `S7.2` (defect 3): either fill with genuine supplementary detail not
      in main-text Table 4 (per-scenario replicate seeds, noise-realization counts, or a
      merged note on the W8/W14 exclusion) or delete the empty header.
- [ ] Add counter-renaming to the SI preamble (defect 5):
      `\renewcommand{\thefigure}{S\arabic{figure}}`,
      `\renewcommand{\thetable}{S\arabic{table}}`,
      `\renewcommand{\theequation}{S\arabic{equation}}`.
- [ ] Re-walk every `Section/Table/Figure Sx` citation in `main.tex` against the now-frozen
      SI order and correct all mismatches from defect 6 (`Fig.~S1`, `Figures~S3`,
      `Table~S1`, `Table~S3`).

### Step 1 — Fix all 13 broken cross-references (defect 1)
- [ ] For each of `sec:wu_artifact1`, `sec:wu_artifact2`, `sec:wu_artifact3`,
      `sec:wu_identifiability`, `sec:identifiability`, `sec:summary_stats`,
      `sec:wu_summary_stats`, `sec:fault_class`, `sec:disc_artefacts`,
      `sec:disc_practical`, `sec:sequential_tracking`, `sec:wu_sequential`,
      `tab:wu_cnn_classification`: replace the dangling `\ref{}` with plain prose
      pointing at the main text — the pattern already used correctly elsewhere in the SI
      (e.g. "the structural masking mechanism discussed in Section 6.3 of the main text").

### Step 2 — Remove the duplicated §9 "Fault Classification" content (defect 7)
- [ ] Cut the stale prose-form duplicate of main-text §7.3.
- [ ] Replace with genuinely supplementary depth not in the main text, e.g. the full
      per-scenario (not per-unit) classification table across all 14×30 replicates, or
      per-replicate posterior scatter plots.

### Step 3 — Add the promised MI-ranking material (closes the content gap in defect 6)
- [ ] Build the per-parameter mutual-information ranking table for System II's 66-D
      feature set promised in `main.tex`, using the already-generated
      `nb23_mi_analysis.png` (defect 10).
- [ ] Optionally include `nb23_tsne_sb.png` / `nb23_sa_vs_sb_xd.png` as companion figures
      if they add diagnostic value beyond the existing PCA figure.

### Step 4 — Bibliography and package cleanup (defects 2, 11)
- [ ] Add missing `LopezPaz2017` bib entry to `references.bib`.
- [ ] Add missing `KernSeaton` bib entry (main-text side, same bug class).
- [ ] Remove unused `\usepackage{tcolorbox}` from the SI preamble.

### Step 5 — Full recompile + label/ref audit
- [ ] Recompile SI standalone; confirm zero `??` in the log and no
      `Citation ... undefined` warnings.
- [ ] Recompile `main.tex`; re-verify every `Section/Table/Figure Sx` pointer resolves to
      the correct SI location by eye.

### Step 6 — Narrative consistency pass
- [ ] Grep the SI for stray earlier-draft language ("banana", "genuine
      non-identifiability", etc.) that might contradict the final "zero-for-three"
      retraction framing (Artefact 1/2/3 all shown to be representation artifacts, not
      physical joint confounds). S9/S10 already look consistent from initial reading —
      this is a final confirmation pass, not expected to surface much.

---

## Verification commands (for whoever picks this up)

```bash
cd docs/Manuscript

# Labels defined in each doc
grep -oE '\\label\{[^}]+\}' main.tex | sed -E 's/\\label\{(.+)\}/\1/' | sort -u > /tmp/main_labels.txt
grep -oE '\\label\{[^}]+\}' supporting_information.tex | sed -E 's/\\label\{(.+)\}/\1/' | sort -u > /tmp/si_labels.txt

# Refs used in SI that aren't defined in SI itself (should be empty after Step 1)
grep -oE '\\(ref|Cref|cref)\{[^}]+\}' supporting_information.tex \
  | sed -E 's/\\(ref|Cref|cref)\{(.+)\}/\2/' | sort -u > /tmp/si_refs.txt
comm -23 /tmp/si_refs.txt /tmp/si_labels.txt

# Missing bib keys (should be empty after Step 4)
for f in main.tex supporting_information.tex; do
  grep -oE '\\cite[a-zA-Z]*\{[^}]+\}' "$f" | grep -oE '\{[^}]+\}' | tr -d '{}' | tr ',' '\n' | sort -u \
    | while read -r k; do grep -q "^@.*{$k," references.bib || echo "MISSING in $f: $k"; done
done
```
