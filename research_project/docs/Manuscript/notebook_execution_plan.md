# Scoped notebook dependency map & execution plan (v2 — re-sequenced)

**Purpose:** turn `reviewer_response_plan.md`'s mandatory items into a concrete, ordered,
risk-labelled list of which of the 36 notebooks to change and rerun. Built by tracing
actual load/save paths in the notebook source and `src/cstr_sbi/` code — every
dependency claim is grounded in a specific file/cell, not guessed.

## v1 → v2 changes

`reviewer_response_plan.md` was revised after a meta-review found it too permissive
about treating central validation as optional. Two structural changes carry over here:
1. **The old "Batch 2" cheap diagnostic for System II is retired.** Evaluating the
   *existing* (healthy-onset-trained) posterior against scenario-specific-warm-start data
   only tests out-of-distribution robustness — it reintroduces the same train/test
   mismatch it was meant to investigate and cannot answer the paper's actual question
   ("does this work during ongoing degraded operation?"). It's replaced by the three
   matched experiments below, which require retraining.
2. **Execution order changed.** The System II physics audit and protocol-locking now
   come *before* any data/table regeneration — regenerating numbers from a model that
   audit later finds wrong, or from an unsettled protocol, wastes the work. The old
   "batch by cost" ordering is replaced by "batch by the plan's actual dependency order."

**Key structural finding, unchanged and still load-bearing:** each system has two
*separate* initial-condition pathways — an evaluation-data pathway and a training-data
pathway — that the manuscript conflates into one sentence. This is still what keeps the
Table VI/VII fixes cheap even though the protocol work around them is now mandatory and
expensive.

---

## Do-not-touch list (unaffected by any mandatory item)

`00, 01, 01a, 03, 13, 20, 21, 28, 33` — no finding touches these. `08, 09, 15, 23, 34`
only need touching downstream, after upstream numbers/protocol are final (see Stage 6).

---

## Dependency graph (confirmed from source — unchanged from v1, reproduced for reference)

```
System I:
nb02 (data_generation) → data/observations.npz  [scenario-specific warm start, per-scenario steady state]
  │                                                                              │
  ├──────────────┐                                                              │
  ▼              ▼                                                              ▼
nb04 (sbi_training)                                                      nb16 (ekf_ukf_baseline)
  Y0_TRAIN = fixed healthy warm-start for ALL prior draws  ⚠️              loads posterior + observations.npz
  writes: sbi_posterior_final.pkl, sbi_m4_scenario_recovery.csv            directly; computes Table VII
  (Table VI source, via pick_obs(rep=0) — single replicate) ⚠️             in-notebook (no save — bug
  RUN_FINAL toggle: False = load cached, no retraining                     went undetected because
  │                                                                         nothing was archived)
  ├─────────────┬─────────────┬─────────────┬─────────────┐
  ▼             ▼             ▼             ▼             ▼
nb04b (CNN)  nb05 (MCMC)  nb06 (metrics_  nb07 (OL/CL)  nb09,10,11,12
             saves samps_  summary.json)

System II:
nb22 (wu2003_data_generation) → data/wu2003_observations.npz
  scenario_specific_warm_start defaults False, never overridden  ⚠️
  (notebook's own markdown admits every window has an onset transient)
        │                                    │
        ▼                                    ▼
   nb23 (summary stats)          scripts/sbi_pipeline.py → training bank
                                  ALSO uses fixed nominal_warm_start  ⚠️ (same issue, training side)
                                        │
                                        ▼
                                  nb24/nb25 (SBI training, 8-seed ensemble)
                                  loads CACHED ensemble SBC results + selected seed-4 posterior
                                  (wu2003_posterior_sb.pkl) — rerunning nb24 itself is cheap;
                                  regenerating the 8-seed ensemble from scratch is NOT
                                  (multi-hour, seed-unstable per HANDOFF.md)
                                        │
                                        ▼
                                  nb26,27,29,29b,31,32 (classification/tracking/FIM/CNN)
```

---

## Stage 0 — Lock the protocol decision (DONE)

**Decision: primary/headline regime = ongoing (matched) degraded steady-state operation;
onset-transient detection is a secondary, explicitly labelled scenario.** Justified
directly from the paper's own stated scope (condition monitoring, predictive
maintenance, progressive/gradual degradation, 30-day continuous tracking) — see
`pending_manuscript_fixes.md` Stage 0 for full evidence, including a self-contradiction
already in the current text (line 814 credits "transient dynamics during fault onset"
as informative, despite the paper's stated scope being ongoing-degradation monitoring).
This sets Stage 3's "matched ongoing-degradation" experiment as the new primary
configuration for both systems, not the current onset-mismatched setup.

## Stage 1 — System II physics audit (DONE)

Audited directly against the primary source (Wu, Yu, Luyben & Skogestad, 2003, full
text extracted from the paper) and the actual production code
(`src/cstr_sbi/recycle/physics.py` — note `src/cstr_sbi/luyben/physics.py` is a
separate, unused-by-the-paper implementation, not the one audited). Full findings in
`pending_manuscript_fixes.md` Stage 1; headline result: **the outcome is disclosure, not
correction or recast-as-surrogate** — the plant topology and steady-state operating
point are faithfully Wu et al. (2003), but the reactor-jacket thermal dynamics and
$\beta_r$ are this manuscript's own undisclosed addition (that benchmark family has no
reactor energy balance at all — temperature is treated as externally given). The
written recycle equation ($F_R=D\,x_D/z_{A,\mathrm{in}}$) is wrong but the *code* already
computes $F_R=D$ correctly, so no rerun is needed there. $\beta_r$ scaling the commanded
duty $Q_j$ (not just conduction) is confirmed as a real, unjustified code-level choice
requiring a Tier 2 fix-or-justify decision. No Tier 3 work remains for this stage — see
Stage 2 below, which now absorbs Stage 1's punch list since none of it blocks or
requires new data generation.

## Stage 2 — Cheap, self-contained code fixes (DONE, 2026-08-12)

Full evidence and final numbers in `pending_manuscript_fixes.md` Stage 2. Summary:

1. **Done.** `notebooks/16_ekf_ukf_baseline.ipynb` cell 20 fixed and re-executed —
   `MAE` for all three snapshot methods (EKF, UKF, SBI) now computed from `df_sc2`
   directly; NUTS row now sourced from archived posterior samples instead of a
   hand-typed placeholder; an automated `MAE ≥ |bias|` check added and passing. Final
   Table VII numbers recorded.
2. **Done.** `notebooks/04_sbi_training.ipynb` cell 19 fixed and re-executed (twice —
   the first run surfaced a second, genuine boundary-condition bug in the "true class"
   re-derivation for Sc4/Sc7, fixed and reconfirmed) — Table VI now a genuine
   50-replicate aggregate (mean/SD/MAE/coverage/classification), not a single
   mislabelled replicate. `RUN_SENSITIVITY`/`RUN_FINAL` kept `False` throughout — no
   retraining occurred. A stray `darkhorse` kernelspec (pointing at an unrelated
   project's virtualenv) was also fixed.
3. **Done**, resolved as part of item 1: correct value is 22.95 ms, not 16.
4. **Done — turned out to need no rerun.** Traced the full computation chain and found
   the $z_{A0,\mathrm{eff}}$ threshold is *already* correctly implemented as a genuine
   0.765 (15%-relative) rule in `nb31`'s prediction code — the "bug" was a manuscript
   wording ambiguity, not a computation error. `nb31` itself was **not rerun** (not
   needed — its archived results already match `main.tex`'s headline numbers exactly).
   Surfaced a bigger, previously-unknown finding instead: Table V's scenario parameter
   values are stale relative to the current code (7 scenarios affected) — see
   `pending_manuscript_fixes.md` Stage 2, Item 4b, for the full table of corrections.
5. **Ready to apply at Stage 7** (unchanged from Stage 1 — no notebook execution
   needed): rename $\eta_\mathrm{col}$, Table III constants, `eq:wu_recycle`,
   `eq:wu_qreb_degraded`, jacket-model disclosure.
6. **Decided and fixed (code only, not rerun).** No physical justification found for
   $\beta_r$ scaling the commanded duty $Q_j$; the sibling (unused) `luyben` module
   never did this, corroborating it was an unintentional bug. Fixed in
   `src/cstr_sbi/recycle/physics.py` (both `recycle_rhs` and `recycle_rhs_explicit`).
   **This invalidates all cached System II artifacts** (already true anyway per Major
   Comment 7's mandatory Stage 3 retraining) — Stage 3 must retrain under this
   corrected physics.
7. **Blocked, flagged for the author team** — `MJ_CPJ`'s provenance could not be
   traced to Wu et al. (2003)'s own Table 1 (which has no jacket entries at all,
   consistent with Stage 1's Finding 1); may be in the paywalled Wu & Yu (1996), or may
   be this manuscript's own assumption. Not resolvable by further investigation here.

## Stage 3 — Matched-protocol retraining (mandatory, expensive — Tier 3)

**System I** (one-seed matched-initialization retrain, per Major Comment 7):
8. Modify `notebooks/04_sbi_training.ipynb` so `Y0_TRAIN` is computed per-scenario
   (matching `nb02`'s protocol) instead of a single fixed healthy warm-start, retrain one
   seed, and compare bias/MAE/SBC against the current production posterior.
9. If results change materially, retrain the full production posterior under the
   corrected protocol and cascade to `04b, 05, 06, 07, 08, 09, 10, 11, 12, 14, 15, 16`.

**System II** (three matched experiments, per Major Comment 7 — supersedes v1's Batch 2):
10. **Matched onset regime**: today's setup, explicitly validated and labelled as such —
    no new run needed beyond confirming the label is accurate.
11. **Matched ongoing-degradation regime**: regenerate `wu2003_observations.npz` with
   `scenario_specific_warm_start=True`, regenerate the training bank via
   `scripts/sbi_pipeline.py` with the same setting, and retrain the 8-seed ensemble
   (`nb24`, S-B only — `nb25`/S-A remains a settled negative result, low priority to
   redo) under this regime. **Status update 2026-08-27:** the full-budget S-B attempt
   has now been run with corrected $\beta_r$/$Q_j$ physics (`n=15,000`, 8 seeds,
   `N_SBC=400` per seed plus independent `N_SBC=800` selected-seed confirmation) and
   it failed SBC: 0/8 seeds passed min-KS > 0.05; selected seed 3 confirmed at min KS
   $p=9.03\times10^{-8}$. Do not promote the matched posterior. Before repeating any
   full run, diagnose summary outliers/z-scoring, out-of-prior posterior mass, and
   reduced-budget preprocessing/architecture fixes.
12. **Cross-regime transfer**: evaluate the onset-trained posterior against
    ongoing-degradation data and vice versa, to characterize deployment-regime
   sensitivity directly. **Blocked until Item 11's calibration failure is understood
   or explicitly accepted as a negative result.**
13. **Full cascade once Stage 3 System II retraining is done**: rerun
    `26, 27, 29, 29b, 30, 31, 32` against whichever posterior(s) the matched-protocol
   decision settles on. **Blocked by Item 11.**

## Stage 4 — Statistical validity work (after Stage 3's protocol is settled)

14. Re-run SBC under the corrected/matched protocol for both systems (Major Comment 8).
15. Add posterior predictive checks in trajectory space.
16. Covariance-robustness grid for $(\alpha,\beta_r)$, $(\alpha,z_{A0,\mathrm{eff}})$,
    $(\beta_r,z_{A0,\mathrm{eff}})$: diagonal vs. shrinkage-estimated vs. block/lag-aware
    covariance, 2–3 finite-difference step sizes, 2–3 Monte Carlo replicate counts
    (Major Comment 3) — reuses the existing FIM/RT-FIM code (likely `nb23`/`nb29b`-style
    machinery), new analysis cells, no new simulator development.

## Stage 5 — Physics-proxy ablation and optional strengthening

17. Three-way $s_{UA}$ ablation (existing proxy / corrected balance proxy / no jacket
    proxy) — compare β bias, MAE, SBC, coverage, classification, saturation sensitivity
    (Major Comment 5.1). Reuses `nb02`'s data and `nb04`'s training code with a modified
    feature set; requires retraining 2 additional posterior variants for comparison.
18. (Optional, Tier 3 "Strongly Recommended") Repeated sequential-tracking across 10–20
    independent degradation paths (Major Comment 9).
19. (Optional, Tier 3 "Strongly Recommended") Reliability diagrams/Brier scores for
    unit-level posterior class probabilities.

## Stage 6 — Downstream synthesis notebooks (last)

20. `nb14_claims_and_conclusions.ipynb` and `nb30_wu2003_claims_and_conclusions.ipynb`
    read live from the results files touched above — rerun these *last*.

## Stage 7 — Manuscript wording pass (last of all, per the reviewer-response plan's new execution order)

21. Only after Stages 0–6 are settled: apply Part A/B's Tier 1 text-only fixes to
    `main.tex`, including the three-level posterior-target decomposition (Major Comment
    11) as an organizing reframe for the Discussion section, not just a Methodology
    caveat.

---

## Summary judgment

v1's cheap-first ordering optimized for quick wins; v2 optimizes for not wasting work —
physics gets audited and protocol gets locked before anything downstream is regenerated,
matched-protocol retraining is now accepted as mandatory rather than deferred, and
wording is the very last step so it's written against final numbers exactly once.
