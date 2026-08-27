# Manuscript impact plan — `docs/Manuscript/main.tex`

**Reviewed:** 2026-08-02 (previous reviews: 2026-08-01, 2026-07-29), against the current
`main.tex` (~24 pages, ~13,000 words, compiles clean with zero undefined references) and the
newly-executed notebooks (`29b`, `31`, `33`, `34`, `27`).

---

## Session summary (2026-08-02) — the rest of P1 and P3 implemented

Nearly everything remaining in P1 and P3 was implemented and verified this session. Figures
(P2) were explicitly deferred by the user to a later pass and are untouched.

1. **P1.3 write-up (Cramér–Rao bounds):** the ellipse/angle-vs-size/condition-number
   explanation (worked out in conversation) was formalised into §Structural Identifiability
   Analysis (System I), right where the FIM is first introduced, and the condition-number
   evidence (3.45×10³→5.34 at W11 for Artifact 2; 4.36×10³→5.48 and 41.4→5.24 for Artifact
   3's two pairs) was added to both Artefact 2 and Artefact 3 subsections. **Done, in
   `main.tex`.**
2. **P1.4 (aggregation mechanism):** computed and verified (robust across 3 noise seeds,
   added as a reproducible cell in `nb29b`) which specific features drive the (α, β_r)
   off-diagonal. **Real finding, different from the outline's speculative guess:** jacket
   cooling duty $Q_j$'s whole-window mean and quantile — not cross-channel correlation
   features — account for ~65–90% of the coupling, because both α and β_r shift Loop 1's
   required steady-state $Q_j$ level similarly; only $Q_j$'s fine-grained transient shape
   distinguishes them, which whole-window statistics discard. Written into §Artefact 2 with
   the formal linear-map argument ($\mathbf{J}_\mathrm{summary}=\mathbf{A}\mathbf{J}_\mathrm{raw}$)
   generalising the mechanism. **Done, in `main.tex`.**
3. **P1.1 write-up (positive/null control):** the null control ((β_r, ξ_reb), clean under
   both representations) is now stated in §Structural Identifiability Analysis, right after
   the new RT-FIM check box. The messier positive-control finding (the diagonal-covariance
   FIM methodology doesn't reproduce System I's own 250–500× ratio, traced to the covariance
   approximation, not a bug or representation effect) is written into the new Limitations
   table as its own entry, with the correct scope caveat (does not retract any existing
   claim; the 250–500× vs. 1.1–1.4× cross-system comparison should be read qualitatively).
   **Done, in `main.tex`.**
4. **P1.5 + P3.6 (contributions/taxonomy):** the Introduction's contributions paragraph was
   rewritten from four to five results: the FIM ratio is uncommented, "method-independent"
   (four paradigms agree) is separated from "irreducible" (CNN embedding), macro-F1=0.694 is
   given context (87.4% accuracy, reactor 0.948, column 1.000) before being stated, and a
   fifth contribution states the taxonomy-dependence finding and the SBI-fit-for-purpose
   case. **Done, in `main.tex`.**
5. **P1.6 (sequential-SBI pooling attempt):** built and executed as new cells in `nb27` — a
   recursive grid-based Bayesian filter over (α, β_r) using the already-trained S-B
   posterior's `.log_prob` as a pseudo-likelihood, no retraining, with a random-walk
   transition kernel (pre-registered step size, not tuned to the outcome). **Result: a
   genuine, measured negative result** — pooling makes tracking *worse*, not better
   (MAE$_\alpha$ 0.048→0.119; MAE$_{\beta_r}$ 0.198→0.273), because the (α, β_r) bias is
   systematic rather than random noise across windows, so a recursive filter compounds
   rather than averages it. Written into the Discussion's EKF-regimes subsection and echoed
   in the Conclusions, replacing the old passive "future work would have to..." sentence.
   **Done, in `main.tex`.**
6. **P3.3 (Related Work section):** built from the article outline's four subsections (each
   ending with an explicit **Gap:** statement), adapted to current claims (no retracted
   material, correct "artefact" spelling), with `Collett2026` given a proper delta statement.
   All citations verified to already exist in `references.bib`. **Done, in `main.tex`.**
7. **P3.4 (name the diagnostic):** the **raw-trajectory Fisher-information (RT-FIM) check**
   is now formally named and given a boxed (`tcolorbox`) four-step protocol with a cost
   estimate (~70 simulator calls per representation per operating point), placed right before
   the three Artefact subsections it governs. **Done, in `main.tex`, tcolorbox compiles clean.**
8. **P3.5 + P3.9 (Limitations + Discussion restructure):** the Discussion was restructured
   into six labelled subsections — *One Genuine Limit, Three Representation Artefacts* →
   *Why Amortised SBI Is the Appropriate Instrument for This Problem Class* → *Artefact
   Severity Depends on the Fault Taxonomy, Not Confound Magnitude* (promoted/moved from
   before the SBI subsection, per the original P3.9 plan) → *EKF Failure and Success Regimes*
   → *Persistent Excitation and Open-Loop Recalibration* → **Limitations** (new). The
   Limitations table has 6 entries, including the new diagonal-covariance-FIM caveat from
   item 3 above. **Done, in `main.tex`.**
9. **P3.8 (language/consistency pass):** "artefact"/"artifact" unified to British "artefact"
   throughout prose and subsection headings (internal LaTeX labels like `sec:wu_artifact1`
   deliberately left unchanged — renaming those serves no reader-visible purpose and risks
   silently breaking a cross-reference); `sbi-toolkit`→`sbi` (correct package name);
   `[NSF; 11]` raw numeric citation → `\citep{Durkan2019}`; the "three successive stages"
   promise now lists all three stages together instead of introducing stage (iii) twenty
   lines later; 11 instances of `\\`-as-paragraph-break replaced with real blank-line
   breaks (verified each was prose, not a table/array row, before touching it). **Done, in
   `main.tex`.**
10. **P3.1 + P3.2 (title + abstract):** title changed to *"Amortised Simulation-Based
    Inference for Plant-Wide Fault Diagnosis Under Closed-Loop Control: A Raw-Trajectory
    Fisher-Information Check for Artefactual Non-Identifiability"*. Abstract fully rewritten
    (the inline `% TODO(P3.2)` marker is now resolved and removed) to cover: the identifiability
    headline for both systems, the named RT-FIM check with condition-number evidence and the
    null control, the taxonomy-dependence finding, and the SBI-fit-for-purpose case — all
    without referencing the EKF re-tuning (per the standing decision below). **Done, in
    `main.tex`.**

Every item above was verified by recompiling after each edit (`latexmk -pdf`, checked for
zero `undefined`/`multiply defined` warnings) rather than batched at the end, so any LaTeX
error would have been caught immediately next to its cause.

---

## What is left

Essentially only **P2 (figures)**, which the user has explicitly deferred to a separate pass,
plus a small number of genuinely optional items below. There is no other outstanding P1 or P3
work from the prior review.

### P2 — Figures (deferred by user, untouched this session)
All of P2.1–P2.9 from the prior review still apply exactly as previously written: the
diagnostic has no figure despite the assets existing on disk
(`figures/nb29b_alpha_betar_richer_features.png` and friends), System I's headline FIM ratio
has no figure, the four-method-agreement result is table-only, the placeholder schematic
(`main.tex` still says "PLACEHOLDER:" in the Figure 1 caption) needs replacing, and the new
content added this session (RT-FIM check, condition numbers, P1.4's feature ranking, the
null control, the P1.6 pooling result) has generated fresh candidate figures of its own
worth considering (e.g. a small panel showing the (α, β_r) off-diagonal collapsing alongside
its condition number, or the P1.6 pooled-vs-independent-vs-EKF tracking comparison already
plotted and saved to `figures/27_pooled_sbi_tracking.png`).

### Remaining optional items (low priority, not blocking)

- **P1B.1 deployment-case table:** still not built as a standalone table; the Discussion's
  "why SBI" subsection covers the same ground in prose. Optional — a table would be a
  complementary, not a redundant, addition if there's appetite for one later.
- **P1B.5's classification-reframing ask:** the Introduction's contributions paragraph
  (item 4 above) already leads with the strong per-unit numbers before macro-F1, resolving
  this in the one place it matters most (the paragraph most reviewers read closely); the
  Results section's own Table caption (`tab:wu_classification`) still states macro-F1 first,
  which is a much lower-stakes residual instance of the same issue.
- **P1B.6 model-mismatch stretch goal:** untouched, as originally scoped — a separate,
  larger piece of work (a new simulator-perturbation experiment), not attempted this session.
- **Minor residual overlap:** the Introduction's original "three properties" paragraph
  (amortisation / non-Gaussian posteriors / no likelihood derivation, `main.tex` shortly
  after the Introduction's opening paragraphs) now sits somewhat redundantly alongside the
  Discussion's more fully developed five-argument "why SBI" subsection. Not incorrect, just
  a mild stylistic overlap; consider trimming the Introduction's version to a single
  forward-pointing sentence if doing another editing pass.
- **`supporting_information.tex` has its own, separate undefined-reference problem**
  (discovered, not fixed, this session): it references `\ref{sec:disc_artefacts}` and
  `\ref{sec:disc_practical}`, neither of which exists anywhere in the SI itself. This is a
  pre-existing issue in a different file, out of this session's explicit scope (main.tex
  only) — flagged here so it isn't lost, not yet acted on.
- **P4 (journal compliance)** and **P5 (impact amplification beyond the manuscript)**:
  untouched, exactly as in the prior review (still APS/revtex4-2 format; no CRediT,
  competing-interest, or data-availability statements; `scripts/fim_utils.py` remains a
  ready-to-package asset for P5.1).

---

## Decisions recorded (carried forward, still binding)

- **Do not reflect the EKF re-tuning finding (nb34) in the manuscript** (decided
  2026-08-01): the existing EKF-regimes subsection stays exactly as it was. Nothing in this
  session's work touches or references nb34's mixed W11/W12/W15 result. Still closed unless
  explicitly reopened.
- **The Discussion's "why SBI" subsection must not read as an EKF comparison** (decided
  2026-08-01): still five independent, non-comparative arguments; this session added to and
  restructured around it but did not alter its core argument or reintroduce the original
  EKF-contrastive framing.
- **New this session:** the P1.6 sequential-pooling result is a genuine negative result and
  is reported as such (tracking got worse, not better) — this does not touch or contradict
  the standing EKF-retuning decision above, since the pooling experiment and the retuning
  experiment are unrelated (pooling changes how per-window SBI posteriors are combined;
  retuning changes the EKF's own covariance). Both are real, both are now in the manuscript,
  and they answer different questions.
