# System I — remaining work before the manuscript is publication-ready

**Status as of 2026-11-05.** Consolidates three sources so there's one place to
work from: a direct-inspection audit of `main.tex`/the SI/the outline/`notebooks/system_I`
run this session, the pre-existing `docs/Manuscript/system_i_remaining_work.md`
(written earlier, still accurate), and the System-II identifiability writeup's own
System-I-relevant open question (`system_ii_sb_identifiability_story.md` §5.3/§7.4/§7.6).
Cross-reference those two docs for the full detail behind each item below — this file
is the prioritized action list, not a replacement for either.

**Headline: System I's core science (SBC calibration, FIM/identifiability, classification,
multi-method baseline, 30-day tracking) is settled and internally consistent in `main.tex`.**
Unlike System II, there is no open calibration-failure crisis. What remains is one
genuinely open scientific question, two concrete/cheap manuscript-completeness gaps,
and a handful of low-priority housekeeping items. None require new training compute
at System-II scale — System I trains in seconds per seed, not hours.

---

## P0 — High priority, cheap, do first

### 1. Multi-seed SBC calibration corroboration
The headline KS p = 0.21 (α) / 0.13 (β) is from a single training seed.
`supporting_information.tex` §S3a (extended ablation, `notebooks/system_I/18_ablation_study.ipynb`)
already found that **no** tested config passes SBC on both parameters simultaneously on
a re-run, with β's p dropping as low as 0.001 — and its own text states explicitly:
*"This corroboration has not yet been run for System I and is recorded here as the
concrete next step."*
- **Action:** run a multi-seed sweep (ensemble-and-select, mirroring System II's own
  protocol) at the production architecture/budget; report the ensemble's KS p-value
  spread, not just one run's point estimate.
- **Blocked on:** nothing. Cheap (seconds/seed).

### 2. Supporting Information §S8 is missing from the working copy
`main.tex` cites "Section S8" twice (Table VII per-replicate methodology and compute
environment) but the current `supporting_information.tex` stops at §S6a. The full
§S7-S11 text exists in git history (`git show HEAD:...supporting_information.tex`) and
was apparently dropped during a System-II-motivated SI reorganization.
- **Action:** recover §S8 from git history; diff its matched-protocol numbers against
  current Table VII before re-inserting verbatim (§S3/§S6 received later refinements
  §S8 might also need). Confirm with whoever drove the SI reorg whether the removal was
  intentional before reinstating.
- **Blocked on:** nothing (System I-specific; independent of System II's SI rewrite).

---

## P1 — Open scientific question / new analysis (this session's new notebook, see below)

### 3. The EKF/UKF-vs-SBI cross-system asymmetry
System II's filter (EKF, raw-trajectory access) **beats** SBI in 30-day tracking near
nonlinear regimes — a mechanistically explained result (representation-compression loss
that raw access recovers). System I shows the **opposite pattern everywhere**: SBI beats
EKF/UKF throughout (Table VII: EKF/UKF bias −0.093 vs SBI's −0.002 at Sc2; EKF/UKF MAE
roughly an order of magnitude worse than SBI in 30-day tracking), despite System I having
its own genuine β information deficit (~236× FIM ratio, Loop-1 integral masking). This is
currently an **unexplained asymmetry** between the two systems — `main.tex`'s Discussion
describes each system's EKF behaviour correctly but does not attempt to explain why they
differ, which a reviewer could reasonably probe.

Candidate explanations, none yet tested:
- System I's physics-informed summary proxies (`UA_eff_proxy`, `k0_eff_proxy`) may
  already capture most of what a raw-access filter would otherwise gain, closing the
  compression gap that favours EKF in System II.
- System I's EKF/UKF implementations may simply be less favourably tuned than System II's
  (tuning quality was never directly compared between the two systems' filters).
- System I's smaller state space (3 states vs. System II's up to 8 plus recycle delay)
  may make linearisation a good local approximation everywhere, not just away from a
  tipping point, reducing EKF's structural disadvantage.

- **Action:** either resolve this with evidence (filter-tuning audit across both
  systems' gain matrices; an RT-FIM check for System I analogous to the one already run
  for System II in `notebooks/34`/`39`, confirming whether System I's engineered features
  already match raw-trajectory information content) or explicitly scope `main.tex`'s
  Discussion to avoid asserting a causal account that hasn't been checked.
- **This is the task for the new notebook** (`notebooks/system_I/19_...ipynb`, see the
  TODO in `notebooks/system_I/TODO.md`) — build directly on `notebooks/34`'s already-
  validated finding that System I has **no** `(alpha,beta)` banana (composition channel
  `C` structurally excludes β from its own governing equation), and extend that
  mechanism rigorously enough to also explain the EKF asymmetry, with extensive testing
  across scenarios/seeds/noise levels rather than a single spot-check.

### 4. Integrate (or explicitly scope out) notebook 34's cross-system finding
`notebooks/34_alpha_beta_r_banana_cross_system.ipynb` is a complete, executed, ablation-
confirmed result (System I's composition sensor `C` structurally excludes β from its
governing equation, unlike System II's unexposed `z_A` channel) that currently exists
only as a notebook — not cited in `main.tex` or the SI.
- **Action (decision, not research):** either promote it into the Discussion as the
  explanatory mechanism for why the two systems' reactor-side identifiability differs
  this much, or note explicitly that it's being treated as supporting material only per
  the 2026-09-28 thesis pivot (identifiability demoted to explanatory device). Should be
  decided together with item 3, since the new notebook (item 3) extends nb34 directly.
- **Blocked on:** nothing.

---

## P2 — Verification items (likely fine, confirm before closing out)

### 5. Two unverified equation issues from the external reviewer-style audit
Found live in `main.tex`, not yet checked against the actual simulator code (this
project has repeatedly found "bugs" that turn out to be LaTeX-only, with the code
itself correct):
- (a) Energy balance: ΔH_r stated in J·mol⁻¹, C_p in cal·(g·K)⁻¹, no stated conversion
  factor (`main.tex` Table, "params_system1", around line 244-245).
- (b) PI control law: integral term in K added directly to a flow-rate term without the
  K_p multiplier a dimensional reading would require (`eq:pi_law`, around line 303-306).
- **Action:** check both against `src/cstr_sbi/physics.py`/the simulator implementation
  directly; either fix the equations or add a one-line unit-conversion clarification.

### 6. Reviewer Major Comment 5.1's three-way `s_UA` proxy ablation
Planned in `notebook_execution_plan.md` Stage 5 (existing proxy vs. corrected-balance
proxy vs. no jacket proxy); no trace found anywhere (`main.tex`, SI, `pending_manuscript_fixes.md`,
`HANDOFF.md`) that it was ever run.
- **Action:** confirm whether Major Comment 5.1 is still being tracked as needing this
  specific ablation, or was addressed a different way and the plan entry is stale.

### 7. Reviewer-response closeout double-check
Major Comment 2's framing fix was gated on the CNN-embedding rerun being "done or
explicitly deferred with this caveat stated" — appears closed (single-replicate number
stated directly in `main.tex` instead of citing the now-missing §S8), but worth a
two-minute re-check once §S8 (item 2) is restored, since the original fix explicitly
routed around the missing section.

---

## P3 — Housekeeping (low effort, no science, do opportunistically)

- [ ] Uncomment `main.tex`'s orphaned `\subsection{Multi-Method Baseline Comparison}`
  heading/label — currently commented out while the body text and Table VII render
  live, so the subsection has no visible heading in the compiled PDF.
- [ ] Refresh `article_outline_CChE.md`'s stale System I numbers: old 250-500× FIM ratio
  (6 places), the deleted CNN-embedding-irreducibility narrative (§6.3.3), pre-fix bias/
  tracking numbers (§6.4/§6.5), stale limitations table (§8.4 L2/L6). `main.tex` itself
  is correct throughout — only the outline needs to catch up.
- [ ] Commit or explicitly discard the currently-uncommitted Sep 29/30 work:
  `notebooks/system_I/17_generate_supplementary_figures.ipynb`,
  `notebooks/system_I/18_ablation_study.ipynb`, `results/sysI_ablation_study.csv`,
  `figures/18_ablation_study.png`, and the `supporting_information.tex` edits that go
  with them.
- [ ] Hedge the Abstract's "Fisher information analysis" phrase to match the body text's
  already-added caveat ("local sensitivity and conditioning diagnostic, rather than an
  exact Fisher information matrix").
- [ ] Formally insert Major Comment 10's four-category confound taxonomy (structural /
  practical / protocol-induced / representation-artefact) as a named taxonomy in
  `main.tex` — informal language already exists, just not as a named structure.
- [ ] Add nb09's full sensor-drift/Sc7 write-up to the SI (currently deferred, manuscript
  claim doesn't depend on it, but it's listed as outstanding).
- [ ] Refresh or mark-superseded `notebooks/system_I/14_claims_and_conclusions.ipynb`
  (predates notebooks 15-18; still has valid executed outputs, just stale as a "current
  state" reference).

---

## Summary table

| # | Item | Priority | Blocked on System II? | New compute needed? |
|---|---|---|---|---|
| 1 | Multi-seed SBC corroboration | P0 | No | Minimal (seconds/seed) |
| 2 | Recover SI §S8 | P0 | No | None |
| 3 | EKF-asymmetry investigation + new notebook | P1 | No | Moderate (new notebook, this session) |
| 4 | Integrate/scope nb34's finding | P1 | No | None (decision only) |
| 5 | Unit-consistency checks (energy balance, PI law) | P2 | No | None |
| 6 | Major Comment 5.1 `s_UA` ablation status check | P2 | No | None (unless re-triggered) |
| 7 | Reviewer-response closeout re-check | P2 | No | None |
| 8-13 | Housekeeping (outline refresh, uncommitted work, abstract hedge, taxonomy, nb09 writeup, nb14 refresh) | P3 | No | None |
