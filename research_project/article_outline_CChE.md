# Article outline — Computers & Chemical Engineering (C&ChE)

> Target: Computers & Chemical Engineering (Elsevier, IF ~4.0)
> Format: double-column, ~25-35 pages, ~10,000-12,000 words
> Key C&ChE expectations: rigorous process model with published parameters,
> industrial relevance, comparison with standard industrial baselines (EKF/UKF),
> scalability demonstration, honest treatment of limitations

---

## Reframing note (2026-09-28)

**The paper's thesis changed this session, after a structured design review ("GRILL"
session, see `HANDOFF.md`).** Everything below reflects the new framing. The old framing —
identifiability analysis as the headline result, built around two "banana" degeneracies in
the Wu 2003 recycle plant — is retired. Do not resurrect the retired framing when drafting;
see the "what NOT to claim" list below for the specific claims that no longer belong in this
paper.

**New thesis:** amortised simulation-based inference (SBI) is a practical method for
**robust fault/anomaly detection in complex, plant-wide, closed-loop systems** — scaling
from a single PI-controlled unit (System I) to a multi-unit reactor-column-recycle plant
with decentralized control (System II). Structural identifiability analysis is retained,
but demoted from *the* contribution to the *explanatory mechanism* for why certain fault
classes are harder to separate than others — mirroring the framing that got a closely
related sister paper (see below) accepted.

**Why this changed:** the Wu 2003 System II work reached a natural resting point once the
(α, β_r) "banana" was diagnosed as a representation artifact (§8.1 history, retained
below for the record) — there was no more open identifiability mystery to chase, and
continuing to sell the paper on identifiability alone would mean selling a null result.
Separately, the System II problem itself was rescoped this session to a reduced 4-D
"S-B3" formulation (endogenous parameters only, `z_A0_eff` held exogenous, feed-fault
scenarios excluded) specifically to get a *usable, classifiable* posterior, and a
large-scale (20,000-50,000 simulation) recalibration campaign is running now to try to
fully clear SBC under a widened prior. Anomaly/fault detection performance, not parameter-
level identifiability, is what that campaign is actually trying to deliver — the paper's
framing should match what the work is now optimizing for.

**Evidence this framing is viable, not just a narrative patch:** a closely related paper
from the same group — *"Fast Bayesian equipment condition monitoring via simulation based
inference: applications to heat exchanger health"* (Casolo et al., accepted at
*Engineering Applications of Artificial Intelligence*, manuscript EAAI-26-10566) — used
almost exactly this framing (SBI for condition monitoring; identifiability/calibration
limits reported honestly as characterized regimes, not hidden) and passed review from 5
reviewers. Concrete lessons pulled from that review process and folded into this outline:
reviewers want a hard quantitative number in the abstract, not qualitative language; they
aggressively police overclaiming relative to what was actually validated; they reward
explicit quantification of *where and why* posteriors get overconfident rather than a bare
"reliable UQ" claim; and they treat "compared against nothing" as a serious weakness unless
the absence of a baseline is itself explicitly argued for.

**What NOT to claim (updated):**
- Do not present the (α, β_r)/(α, η_col) "banana" diagnostics as the paper's headline —
  they are retained as *explanatory* content in the Discussion (why certain reactor-side
  faults are harder to attribute to a single parameter), not as a result on equal footing
  with classification performance.
- Do not claim "plant-wide" or "digital twin" capability beyond what was actually tested —
  System II is a specific 2-unit recycle benchmark with 3 PI loops, not an arbitrary
  plant-wide claim. Scope language precisely (this is exactly the kind of claim the sister
  paper's reviewers forced a walk-back on).
- Do not treat "no real Wu 2003 plant data" as an implicit gap to hope nobody notices —
  argue explicitly (Discussion/Limitations) that a synthetic benchmark with known ground
  truth is the *correct* evidence standard for the calibration/SBC claims this paper makes,
  which real plant data could never support.
- Do not claim an EKF baseline for System II unless it is actually (re-)run under the
  current S-B3 scope — see §7.5/§8.2 for the current positioning (NUTS-infeasibility +
  EKF's white-box requirement, argued explicitly, instead of a baseline comparison).
- Do not call fault classification "unsupervised" — it is "label-free" (unchanged from
  before).
- Still: do not claim the closed-loop identifiability limitation itself as a discovery —
  cite Ljung (1977) (unchanged from before).

---

## Highlights (C&ChE requires 3-5 bullet points, ≤85 characters each)

- Amortised SBI enables real-time, calibrated fault detection in plant-wide recycle systems
- SBI scales from a single PI-controlled CSTR to a multi-unit recycle plant with 3 loops
- Fault/anomaly classification remains robust even where individual parameters are hard
  to identify
- Structural identifiability analysis explains *why* specific fault pairs are harder to
  attribute, not just *that* they are
- Honest, quantified reporting of where posterior calibration breaks down, and why

**Retired (2026-07 through 2026-09) — do not reuse as headlines:**
- "Recycle coupling creates (α, η_col) banana posterior invisible to EKF" (retracted
  2026-07-03 — see old §7.2.3 history, preserved in §8.1 for the record).
- "Recycle plant reveals a genuine (α, β_r) banana" (retracted 2026-07-05 — the pair is a
  summary-statistic aggregation artifact, not physics).
- Both diagnostics remain true and are kept as *explanatory* material (§8.1), not
  headlines — the paper's headline is now classification/detection robustness (§7).

---

## Title options

1. **Amortised simulation-based inference for robust fault detection in plant-wide,
   closed-loop chemical processes** (recommended — leads with the capability
   (detection), names the challenge (plant-wide, closed-loop), matches the new thesis)

2. **Scaling Bayesian fault diagnosis from a single unit to a plant-wide recycle process:
   an amortised simulation-based inference approach** (alternative, emphasises the
   System I → System II escalation structure explicitly)

3. **Structural identifiability limits of Bayesian fault diagnosis under feedback
   control** (retired — this was the old identifiability-first framing; keep only as a
   fallback if a reviewer specifically wants more theory foregrounded)

Recommendation: option 1. It puts the capability claim (robust fault detection) first,
which is what C&ChE and EAAI reviewers alike responded to positively in the sister paper,
and "plant-wide, closed-loop" signals the scaling story without overclaiming a specific
plant-wide generality that wasn't tested.

---

## Abstract (~250 words, structured)

**Status: skeleton below, headline numbers marked `[TBD: pending S-B3 campaign]` where
they depend on the widened-prior recalibration run in progress. Fill in once
`notebooks/24d`'s held-out SBC and classification results land; do not block drafting the
rest of the paper on this.**

**Background.** Fault diagnosis in feedback-controlled chemical processes is complicated
by the controller's tendency to compensate for degradation, suppressing the parametric
signatures that estimation methods rely on, and by the scale of plant-wide systems where
classical exact Bayesian inference becomes computationally infeasible.

**Methods.** We apply simulation-based inference (SBI) — training a neural spline flow to
approximate the Bayesian posterior over degradation parameters from physics-informed
summary statistics — across two systems of increasing complexity: a PI-controlled
propylene oxide CSTR (2 parameters, 1 unit) and a reactor-column-recycle benchmark with
decentralized control (4 parameters, 2 units plus recycle, 3 PI loops; Wu, Yu, Luyben &
Skogestad 2003). We validate calibration via simulation-based calibration (SBC) and report
fault classification performance across each system's full fault-scenario catalogue.

**Results.** For the propylene oxide system, SBI achieves calibrated posteriors (SBC KS
p = 0.21/0.13) and macro-F1 = 0.927 fault classification, with the one genuine
identifiability limitation (jacket-fouling signal masked by integral temperature control)
quantified and explained mechanistically (Fisher-information ratio 250-500×), not hidden.
For the recycle plant, restricted to its four endogenous degradation parameters, fault
classification achieves macro-F1 = `[TBD]` across `[N]` scenarios and 4 fault classes,
`[TBD: SBC pass/fail summary once the widened-prior campaign completes]`. We show that
reliable fault/anomaly detection is achievable even in regimes where individual-parameter
calibration is imperfect, and that a specific pair of degradation parameters (catalyst
activity, jacket fouling) is difficult to attribute individually because of how their
combined effect is compressed into whole-window summary statistics, not because of a
physical non-identifiability in the plant — a distinction that matters for how such
confounds should be diagnosed and reported. Classical exact-Bayesian inference (NUTS/MCMC)
is infeasible at this plant's scale; SBI provides real-time (<20 ms/window) calibrated
inference where no exact alternative exists.

**Conclusions.** Amortised SBI scales from single-unit to plant-wide, multi-loop closed-
loop fault diagnosis while remaining computationally tractable and honestly quantifying
where and why calibration is imperfect. We argue, and support with a companion validation
in a separate publication, that this class of method is ready for industrial condition-
monitoring deployment; the identifiability limitations we characterize are inputs to that
deployment (they tell a practitioner which parameters to trust quantitatively and which to
treat as directional), not obstacles to it.

---

## Keywords (6-8)

simulation-based inference; amortised Bayesian inference; fault diagnosis; anomaly
detection; closed-loop identifiability; recycle process; plant-wide monitoring;
predictive maintenance

---

## 1. Introduction (~2.5 pages)

### 1.1 Industrial motivation (3 paragraphs)

**Para 1 — The plant-wide fault diagnosis problem.**
Catalyst deactivation, heat exchanger fouling, pump wear, and separator degradation are
the dominant failure modes in continuous chemical plants. Current industrial practice
relies on periodic laboratory measurements, model-based observers (EKF, UKF), or
data-driven anomaly detectors. These approaches produce point estimates or binary alarms,
not calibrated probability distributions over fault severity — a critical gap for
risk-informed maintenance scheduling and root-cause attribution. Unlike a single unit,
a plant-wide system compounds this problem: faults in one unit propagate through
recycle and control loops to appear, sometimes ambiguously, in another unit's
measurements.

**Para 2 — The feedback masking problem.**
In feedback-controlled plants, decentralized PI controllers actively compensate for
faults: a fouled heat exchanger is countered by increased coolant flow; a deactivated
catalyst is countered by higher reactor temperature setpoints. This compensation
suppresses the very parametric signatures that estimation methods rely on — a
classical closed-loop identifiability limitation established since Gustavsson, Ljung &
Söderström (1977). Crucially, this limitation is not a deficiency of any particular
estimation method; it affects all methods — Bayesian or frequentist, online or offline —
because it is rooted in the information content of the closed-loop data. **The practical
question this paper asks is not "can this limitation be removed" (it generally cannot,
without new instrumentation or excitation) but "does fault/anomaly detection remain
reliable despite it, and can the limitation itself be characterized precisely enough to
know when to trust a quantitative estimate versus a qualitative alarm."**

**Para 3 — Why SBI, and why now.**
Simulation-based inference (SBI; Cranmer et al. 2020) replaces an explicit likelihood
with a neural density estimator trained on simulated (parameter, observation) pairs.
Three properties make it attractive for plant-wide fault diagnosis: (a) amortisation —
a single trained network processes any new observation in milliseconds, enabling
real-time deployment across multiple units or assets simultaneously; (b) full posterior
distributions — capturing non-Gaussian posterior shapes that classical methods cannot
represent; (c) no likelihood derivation required — the process simulator is used
directly, without linearisation or Gaussian noise assumptions, so the same workflow
applies to proprietary or legacy plant models. These advantages are critical near
recycle tipping points, where nonlinear dynamics produce posterior shapes that invalidate
the Gaussian approximations underlying EKF/UKF, and at plant scale, where exact Bayesian
sampling (MCMC/NUTS) becomes computationally infeasible for a multi-unit, multi-loop
system evaluated repeatedly over a monitoring horizon.

### 1.2 Contributions (numbered list, exactly 4)

> 1. **Closed-loop identifiability characterisation for a PI-controlled CSTR, and
>    calibrated real-time fault classification built on it.**
>    We numerically compute the 2×2 Fisher information matrix for catalyst activity (α)
>    and jacket fouling (β), showing I_αα/I_ββ = 250-500× across all operating points, and
>    confirm via a raw-trajectory CNN-embedding experiment that this bias is irreducible by
>    any choice of summary statistics (a Cramér-Rao-bound confirmation). A trained neural
>    spline flow achieves macro-F1 = 0.927 fault classification with calibrated uncertainty
>    (SBC KS p = 0.21/0.13), processing each window in milliseconds — establishing that
>    reliable detection survives a known, quantified, and explained parameter-level bias.
>
> 2. **Scaling to a plant-wide, multi-loop reactor-column-recycle system (Wu et al. 2003),
>    and demonstrating that fault/anomaly detection remains robust at this larger scale.**
>    We extend to a 2-unit recycle plant with 3 decentralized PI loops (14-scenario fault
>    catalogue), restricted to its four endogenous degradation parameters for a tractable,
>    classifiable posterior. We report fault-unit classification performance across the
>    full scenario catalogue and show it holds up even where individual parameters are only
>    partially identifiable — the central practical claim of this paper.
>
> 3. **A worked diagnostic distinguishing genuine parameter non-identifiability from an
>    artifact of the summary-statistic representation, applied to explain, not just
>    report, a specific classification limitation.** Two candidate joint degeneracies
>    (catalyst activity with column efficiency; catalyst activity with jacket fouling)
>    initially appeared as genuine joint non-identifiabilities under the standard
>    whole-window summary-statistic representation. Applying this paper's own
>    Fisher-information methodology to the raw, unaggregated sensor trajectory of the same
>    already-observed channels collapses both couplings to near-zero, and an independently
>    re-tuned EKF given the same raw-trajectory access corroborates this. We use this
>    finding to explain *why* the two implicated parameters (catalyst activity, jacket
>    fouling) are harder to individually attribute within the "reactor fault" class, rather
>    than reporting it as a standalone contribution.
>
> 4. **An explicit argument for amortised SBI as the appropriate instrument for this
>    problem class, in place of both classical exact inference and classical filtering.**
>    Exact Bayesian sampling (NUTS/MCMC) is infeasible at this plant's scale within a
>    practical monitoring cadence. A classical filter (EKF) requires white-box access to
>    the raw process model and its Jacobian at every step, in tension with the black-box,
>    legacy-simulator applicability that motivates SBI in the first place; where tested
>    (System I; System II near recycle tipping points), EKF's Gaussian linearisation
>    produces overconfident or divergent estimates that amortised SBI avoids. SBI's
>    amortisation cost is paid once at training time; subsequent inference takes
>    milliseconds, enabling monitoring of a full multi-week operating horizon in seconds.

### 1.3 Paper organisation

One sentence per section. Section 2 reviews related work. Section 3 formulates both
process systems. Section 4 describes the methodology. Section 5 details the experimental
setup. Sections 6 and 7 present results for the propylene oxide and Wu 2003 systems
respectively. Section 8 discusses implications, limitations, and future work.

---

## 2. Related work (~2 pages)

Four subsections, each ending with a sentence stating what the current paper adds.

### 2.1 Model-based fault diagnosis in chemical processes

- Observer-based FDI: residual generation via parity equations (Frank 1990), structured
  residuals, unknown input observers (Isermann 2006)
- Moving-horizon estimation (Rawlings & Mayne 2009): online optimisation, systematic
  but computationally expensive
- EKF/UKF for process monitoring: standard industrial approach (Jazwinski 1970;
  Julier & Uhlmann 2004); provides point estimates + Gaussian uncertainty; requires
  hand-tuned noise covariances and white-box model/Jacobian access; Gaussian
  approximation fails when posterior is multimodal or strongly non-Gaussian
- Data-driven methods: PCA/PLS monitoring (Qin 2012); deep learning anomaly detection
  (LSTM, autoencoders); these produce alarms, not posteriors
- **Gap:** no method produces calibrated full posteriors over multi-dimensional fault
  parameters in a plant-wide setting, and no prior work characterizes fault/anomaly
  detection robustness under decentralized control at multi-unit scale.

### 2.2 Closed-loop identifiability (demoted — supporting mechanism, not the paper's focus)

This subsection is now brief and purely supportive: it grounds the explanatory diagnostic
in §7/§8, it is not itself the literature this paper is positioned against (see §2.4 for
that).

- Classical results: Gustavsson, Ljung & Söderström (1977); Forssell & Ljung (1999)
  — three families of closed-loop identification (direct, indirect, joint input-output)
- Fisher information for closed-loop experiment design: Gevers, Bombois, Hildebrand &
  Solari (2011); Bombois et al. (2006)
- Persistent excitation under feedback: the reference signal must have non-zero spectral
  density for consistent parameter identification (Forssell & Ljung 1999 §3); a fixed
  setpoint implies zero excitation in the classical sense
- Snowball effect and recycle instability: Luyben (1994); plant-wide control design
  (Luyben, Tyreus & Luyben 1997)
- **What this paper adds:** rather than a new identifiability result, a worked
  methodology (raw-trajectory Fisher-information check) for telling genuine closed-loop
  identifiability limits apart from artifacts of a summary-statistic representation —
  used here to *explain* classification behaviour, not as a standalone finding.

### 2.3 Simulation-based inference

- Foundations and review: Cranmer, Brehmer & Louppe (2020, PNAS)
- Neural posterior estimation: NPE-A (Papamakarios & Murray 2016), NPE-B (Lueckmann
  et al. 2017), SNPE-C (Greenberg et al. 2019)
- Density estimators: masked autoregressive flows (Papamakarios et al. 2017), neural
  spline flows (Durkan et al. 2019)
- sbi software library: Tejero-Cantero et al. (2020)
- Applications: gravitational waves (Dax et al. 2021 NeurIPS), epidemiology,
  neuroscience; condition monitoring of individual equipment (Casolo et al., EAAI,
  in press — heat exchanger fouling/leakage, the closest sibling application to this
  paper)
- Misspecification detection: Schmitt et al. (2024) — MMD-based diagnostic for
  out-of-distribution observations; does not correct structural bias
- **Gap:** SBI has been applied to feedback-controlled single-unit systems (including the
  authors' own prior work) but not yet demonstrated at plant-wide, multi-loop scale with a
  full fault-detection evaluation; this paper is that demonstration, not a new SBI method.

### 2.4 Plant-wide process monitoring

- Classical plantwide control: Luyben et al. (1997), Seborg et al. (2011)
- Multivariable fault isolation: directional residuals, causal analysis (Bauer et al.
  2007)
- Recycle dynamics: snowball effect (Luyben 1994); interaction between unit-level
  controllers under plant-wide disturbances
- Hybrid-fusion and multi-model plant-wide monitoring reviews (see e.g. the panoramic
  review in Ali et al. 2025, *Ind. Eng. Chem. Res.*) — a substantial and active
  literature on combining models/modalities for plant-wide monitoring, positioned
  against which this paper's contribution is a *single, likelihood-agnostic, amortised
  Bayesian* route to the same goal, returning a full posterior rather than a fused score
  or point estimate
- **Gap, and what this paper adds:** no prior work applies amortised probabilistic
  Bayesian inference for simultaneous fault/anomaly detection across multiple coupled
  units in a recycle process under decentralized PI control, with a validated
  characterization of where and why individual-parameter estimates should or should not
  be trusted.

---

## 3. Process models (~3 pages)

### 3.1 System I: PI-controlled propylene oxide CSTR

**Reaction:** Acid-catalysed hydrolysis, C₃H₆O + H₂O → C₃H₈O₂ (pseudo-first-order,
exothermic, ΔH_r = −20,220 cal/mol). Parameters from Fogler (2016) Module 13 /
Furusawa et al. (1969). The Fogler CSTR is the standard nonlinear CSTR benchmark in
process control (Seborg et al. 2011 Ch.4).

**ODE (closed-loop, 4 states):**
```
dC/dt  = (Q/V)*(Ci - C) - α*k0*exp(-Ea/RT)*C
dT/dt  = (Q/V)*(Ti - T) - ΔH_r*α*k0*exp(-Ea/RT)*C/(ρCp) - β*UA*(T - Tc)/(ρCp*V)
dTc/dt = (Qc/Vc)*(Tci - Tc) + β*UA*(T - Tc)/(ρc*Cpc*Vc)
dI/dt  = (T - Tsp)   [anti-windup integrator]
```

**PI controller:** Qc = clip(Qc0 + Kp*(T - Tsp) + I/τi, 0, Qc_max)

**Table 1:** All parameter values (V, Q, Ci, Ti, k0, Ea, ΔH_r, UA, ρ, Cp, Vc, ρc,
Cpc, Kp, τi, Tsp, Qc0, Qc_max)

**Fault parameterisation (2-D):**
- α ∈ [0.4, 1.2]: catalyst activity factor (k_eff = α·k0·exp(−Ea/RT))
- β ∈ [0.4, 1.2]: jacket conductance factor (UA_eff = β·UA)
- Both uniform priors; upper bound 1.2 provides numerical buffer for the density
  estimator at the physically meaningful boundary α, β = 1.0

**Observations:** 60-min windows at 0.5 min resolution → 120×4 time series [C, T, Tc, Qc];
29-D summary statistics (see §4.2); 8 fault scenarios (Table 2); 50 replicates each.

### 3.2 System II: Wu 2003 reactor-column-recycle benchmark

**Reaction:** A → B (first-order, irreversible, exothermic, liquid phase). The canonical
reactor-separator-recycle benchmark from the "Dynamics and control of recycle systems"
series (Luyben 1993 I&EC Research). **Complete parameters from Wu, Yu, Luyben &
Skogestad (2003, Comput. Chem. Eng. 27(3):401-421) Table 1** — the same paper that
introduces the plantwide control structures compared in this work.

**Plant topology:**
```
Fresh A feed (460 lbmol/h, z₀=0.90 mol/mol)
        ↓
    ┌──────────────────┐
    │   CSTR           │  T_r = 342 K, M_r = 2400 lbmol
    │   A → B          │  Jacket cooling (PI Loop 1: T_r → Q_c)
    └────────┬─────────┘
             │ Reactor effluent (CSTR overflow)
        ┌────▼──────────────┐
        │  Distillation     │  20 trays, α_rel = 2, feed tray 12
        │  column           │  Reflux ratio R = 2.2 (L/D)
        └──────┬─────┬──────┘
               │     │
        B-rich │     │ A-rich distillate = RECYCLE
        bottoms│     │  (500 lbmol/h, x_D = 0.95)
               │     └────────────────────────────────→ CSTR feed
               ↓
          Product B (460 lbmol/h, x_B = 0.011)
```

**Snowball mechanism:** α↓ → less conversion → more A in column feed → distillate flow
and/or composition must increase to maintain bottoms purity → **recycle load increases**,
further diluting the reactor. Under fixed-ratio control (S-B), this creates a positive
feedback loop: more recycle → more dilution → less conversion → even more recycle.

**ODE state vector (6 states + QSS column):**

| State | Symbol | Units | Dynamics |
|-------|--------|-------|----------|
| Reactor composition | z_A | mol/mol | Differential |
| Reactor temperature | T_r | K | Differential |
| Jacket temperature | T_j | K | Differential |
| Reactor temp. integrator | I_T | K·h | Differential |
| Column QC integrator (S-A) | I_QC | — | Differential (S-A only) |
| Recycle ratio integrator (S-B) | I_R | — | Differential (S-B only) |
| Distillate composition | x_D | mol/mol | **Algebraic** (QSS column) |

**Justification for QSS column:** Liquid hydraulic time constant τ_hyd = 4 s;
reactor residence time τ_r = M_r / F_total ≈ 5.2 h. Ratio ~4700×. Column reaches
steady state 4700 times faster than the reactor-recycle loop — QSS is exact for all
dynamics of interest.

**Key ODEs:**
```
dz_A/dt = (F_in/M_r)*(z_A_in - z_A) - α*k(T_r)*z_A

dT_r/dt = (F_in/M_r)*(T_in - T_r) + (-ΔH_r)*α*k(T_r)*z_A/(ρCp)
          - β_r*UA_r*(T_r - T_j)/(ρCp*V_r)

dT_j/dt = [UA_r*(T_r - T_j) - Q_c] / (ρ_c*Cp_c*V_j)

QSS column: x_D = f(z_F, α_eff, N_T)  [Kremser shortcut; α_eff = 1 + η_col*(α_rel-1)]

Recycle: F_R = D*(x_D)/z_A  [from overall material balance]
```

**3 decentralized PI loops:**
```
Loop 1 (Reactor temp):     Q_c → T_r     [fast; same mechanism as PO System I]
Loop 2 (Column quality):   R   → x_D     [S-A: QC on distillate; S-B: fixed ratio RC]
Loop 3 (Column recovery):  V   → x_B     [S-A: QC on bottoms purity; S-B: TC on T_reb]
```

**Instrumentation used for the reported results (S-B):** T_r, T_j, Q_c, T_reb, Q_reb,
F_R, F_B — the conventional/simpler control structure. The richer S-A structure (adds a
distillate composition analyser, x_D) was explored earlier in this project as a candidate
fix for a joint-parameter confound that was later found to be a representation artifact,
not a genuine limitation requiring new instrumentation (§8.1) — S-A is not used for any
result reported in this paper.

**Full physical parameter set (5-D) and the reduced reporting scope used in this paper
("S-B3"):**

| # | Symbol | Prior | Physical meaning | Primary observable | In this paper's results (§7)? |
|---|--------|-------|-----------------|-------------------|---|
| 1 | α | see §7.1 | Catalyst activity | z_A, F_R (via snowball) | Yes |
| 2 | β_r | see §7.1 | Reactor jacket fouling | T_j, Q_c (masked by Loop 1) | Yes |
| 3 | η_col | see §7.1 | Column tray efficiency | x_D, T_reb, Q_reb | Yes |
| 4 | ξ_reb | see §7.1 | Reboiler heat transfer fouling | Q_reb (Loop 3 compensation) | Yes |
| 5 | z_A0 | U[0.70, 0.95] | Feed purity (A fraction) | All channels via reactor SS | **No — held exogenous** |

The simulator itself is the full 5-D physical model above. **The results reported in this
paper (§7) use a deliberately reduced 4-D scope** ("S-B3"): `z_A0` is treated as exogenous
(fixed at its nominal value) and the feed-fault scenarios (W10, W13) are excluded from the
fault-taxonomy evaluated. This is a scope decision made to obtain a tractable, classifiable
posterior over the plant's own endogenous degradation modes (reactor and column faults),
not a claim that feed-composition faults are unimportant — see §8.4 for the motivating
identifiability findings that led to this scope reduction, reported there as context, not
as a parallel headline result.

**Identifiability structure (retained as explanatory context for §7/§8, not a standalone
result):**
- T_r is masked by Loop 1 for BOTH α and β_r (∂T_r_ss/∂α ≡ ∂T_r_ss/∂β_r ≡ 0), and Wu 2003
  has no observable concentration channel (unlike the PO CSTR) — this is a genuine,
  representation-independent **scalar** reduction affecting both parameters roughly
  symmetrically: I_αα/I_β_r ≈ 1.1-1.4× (nearly equal, vs. PO's 250-500×). Under the
  standard 66-D hand-crafted summary-statistic feature set, (α, β_r) *also* appears
  jointly confounded (off-diagonal +0.90 at nominal) — but this joint/manifold component is
  a representation artifact, not physical: the same analysis on the raw, unaggregated
  observation trajectory collapses the off-diagonal to ≈0.00 (§8.1). This is used in §7 to
  explain why the two parameters are harder to individually attribute within the reactor
  fault class, not reported as its own finding.
- (α, η_col): initially suspected as jointly confounded via the snowball's shared effect on
  recycle flow (F_R), but this does **not** survive once the column's own temperature and
  flow channels are combined with F_R — S-B's existing instrumentation already resolves
  most of this pair without a composition analyser (§8.1).
- ξ_reb: identifiable via Q_reb (controller output signal), analogous to Qc for β_r.

**Observations (7 channels under S-B, the structure used for all reported results):**
T_r, T_j, Q_c, T_reb, Q_reb, F_R, F_B. 4-hour windows at 1-min resolution → 240×7 time
series → 66-D summary statistics (extended horizon vs. the original 2-hour window;
required for the slowest scenario, W11, to approach steady response — see §5).

**Table 3:** All nominal plant parameters from Wu et al. (2003) Table 1 (converted to SI):
reactor holdup M_r=2400 lbmol, k_ss=0.33/h, Ea=71.74 kJ/mol, UA=254,000 kJ/(h·K),
ΔH_r=69,780 kJ/kmol, ρCp=2.82 MJ/(m³·K), α_rel=2.0, N_T=20, R=2.2, τ_hyd=4 s.

**Fault scenario catalogue used for the reported results (12 scenarios, "S-B3" scope):**
the full 14-scenario Wu 2003 catalogue minus the two feed-fault scenarios (W10, W13; out
of scope per the reduced parameterisation above), each evaluated with `[N: TBD]`
replicates. Fault taxonomy for classification (§4.5): healthy, reactor (α or β_r
degraded), column (η_col or ξ_reb degraded), multi (both). **Table 4** (updated from the
original 5-class taxonomy): scenario list, true parameter values, and taxonomy label.

**Note on removed scenarios (unchanged from earlier project history):** W8 (η_col=0.65)
and W14 (η_col=0.75+ξ_reb=0.75) are excluded because the Kremser shortcut column model
becomes numerically unstable for η_col < 0.80 from the nominal warm start (ODE blow-up).
The stability boundary η_col ≥ 0.80 covers all scientifically relevant operating points in
the reported catalogue.

### 3.3 Stochastic simulation

Both systems use Euler-Maruyama integration (JAX scan, JIT-compiled) with additive
process noise and Gaussian sensor noise (0.3% of channel range for System II; see §5).
Warm-start initial conditions are computed via diffrax adaptive integration from the
nominal steady state, verified to converge before use (a fixed-horizon warm-start
convergence bug was found and fixed this project cycle — see §8.4).

---

## 4. Methodology (~3 pages)

### 4.1 Simulation-based inference (SNPE-C)

NPE minimises the forward KL divergence over simulated (θ, x) pairs to train a
conditional density estimator q_ϕ(θ|s(x)):

```
L = E_{p(θ,x)} [-log q_ϕ(θ | s(x))]
```

where s(x) are summary statistics. SNPE-C (Greenberg et al. 2019) uses importance
weighting to correct for the proposal-prior mismatch in sequential rounds; we use a
single round (amortised) since the prior covers the full parameter space.

**Connection to classical closed-loop identification.** Following Forssell & Ljung
(1999), NPE operates directly on the measured histories without an explicit controller
model, making it an analogue of the *direct method* of closed-loop identification. Like
the classical direct approach, it is universally applicable to nonlinear, saturating
feedback and exploits in-loop process noise to partially reduce posterior variance.

**Density estimator:** Neural Spline Flow (NSF; Durkan et al. 2019). System I: 128 hidden
units, 5 transforms. System II: 60 hidden units, 3 transforms — deliberately smaller than
an initial larger architecture, which was tested and found to perform *worse* (see §8.4),
consistent with this being a data/identifiability-limited rather than capacity-limited
problem. Software: `sbi` library v0.26+.

### 4.2 Summary statistics

**Propylene oxide (29-D):**
- Per-channel base (5×4 = 20): mean, std, slope, min, max over [C, T, Tc, Qc]
- Final-window means (4): last-25% average per channel
- Control aggregates (3): ∫|T − Tsp|dt, Qc_sat_low_frac, Qc_sat_high_frac
- Physics-informed (2): UA_eff_proxy = (T_mean − Tc_mean)/Qc_mean; k0_eff_proxy = log(C_mean/(Ci − C_mean))

**Wu 2003 plant (66-D, S-B structure):**
- Per-channel base statistics across the 7 observed S-B channels (means, spreads,
  slopes, final-window values)
- Control aggregates: integral error terms, actuator saturation fractions
- Physics-informed features, including a corrected `reb_per_boilup` feature (replacing an
  earlier `reb_intensity` feature found to introduce a spurious α correlation — a
  structural improvement independent of the narrative changes in this outline)

**Table 5:** Full 29-D and 66-D feature definitions with physical interpretation.

### 4.3 Fisher information analysis

The local Fisher information matrix at θ is:
```
I(θ) = J^T Σ_obs^{-1} J,   J_{ij} = ∂s_i/∂θ_j
```
where s is the summary statistics vector and Σ_obs is the diagonal observation noise
covariance estimated from healthy replicates. J is computed via finite differences
on the simulator. The diagonal elements I_αα, I_ββ quantify how much information the
data carry about each parameter; off-diagonals reveal correlations. **In this paper, FIM
analysis is used for two purposes: (a) the System I scalar-masking result, reported as a
finding in its own right (§6.3), and (b) as a diagnostic tool applied to the raw,
unaggregated observation trajectory to distinguish genuine System II identifiability
limits from representation artifacts (§8.1) — the latter use is methodological support
for the classification results, not an independent headline.**

The Cramér-Rao bound gives Var(θ̂_i) ≥ [I(θ)^{-1}]_{ii} for any unbiased estimator.
When I_ββ ≪ I_αα, β is fundamentally harder to identify regardless of the inference method.

**Analytical derivation (propylene oxide):** Under steady-state PI control (T = Tsp),
the temperature row of J is identically zero: ∂T_ss/∂β = 0. This eliminates the
highest-SNR channel from β's information budget. The remaining channels (Tc, Qc) are
noisier and smaller in magnitude, giving I_ββ/I_αα ≈ 1/250-1/500.

**5×5 FIM for Wu 2003:** Numerical computation at the nominal operating point and at
several fault-scenario operating points; used in §8.1 as the diagnostic that resolves
the two candidate joint confounds discussed there.

### 4.4 EKF baseline

**Propylene oxide (6-state augmented EKF):**
Augmented state [C, T, Tc, I, α, β]. Analytical 6×6 Jacobian (hand-derived). Parameters
treated as random-walk states: d(param)/dt = 0 with small process noise. Measurement
equation: [C, T, Tc, Qc] directly; Qc computed from PI controller equations. Used as a
reported baseline for System I (§6.4).

**Wu 2003 plant (augmented EKF, Jacobian via `jax.jacobian`):** implemented and used
earlier in this project cycle for the pre-rescoping identifiability investigation (§8.1
history) — those numbers are retained there as explanatory context. **Not used as a
System II headline baseline in this paper's current scope**: exact/near-exact classical
baselines are argued to be inapplicable at this plant's scale (NUTS infeasibility, §7.5)
or to require white-box raw-model access in tension with SBI's black-box applicability
(EKF, §8.2) rather than re-run and reported as a matched comparison. A matched EKF
re-run under the current S-B3 scope is a live contingency (§8.4), not committed content.

**UKF (propylene oxide only):** Sigma-point propagation with 13 points; included to
confirm that the β bias is not a linearisation artifact of the EKF.

### 4.5 Fault classification

**Propylene oxide (2-D, quadrant taxonomy):**
- Healthy: α ≥ 0.85, β ≥ 0.85
- Fouling-dominant: α ≥ 0.85, β < 0.85
- Decay-dominant: α < 0.85, β ≥ 0.85
- Combined: α < 0.85, β < 0.85
Threshold 0.85 is domain-informed to account for the expected structural β bias.

**Wu 2003 plant (4-D, reduced "S-B3" taxonomy):**
- Healthy: α, β_r, η_col, ξ_reb all near nominal
- Reactor: α or β_r degraded (column parameters normal)
- Column: η_col or ξ_reb degraded (reactor parameters normal)
- Multi: two or more of the above degraded simultaneously
(A feed-fault class existed in the original 5-D taxonomy but is out of scope under the
current S-B3 reduction — see §3.2.)

**Classification rule:** assign posterior probability mass to each region; report the
mode class and the probability of the most likely class as calibrated confidence.

---

## 5. Experimental setup (~1.5 pages)

### 5.1 Training configuration

**Table 6:** Hyperparameters for both systems

| Hyperparameter | Propylene oxide | Wu 2003 (S-B3) |
|---|---|---|
| Training simulations | 10,000 | `[TBD: 20,000-50,000, campaign in progress]` |
| NSF hidden units | 128 | 60 |
| NSF transforms | 5 | 3 |
| Prior (endogenous parameters) | Uniform | `alpha, beta_r` uniform; `eta_col, xi_reb`
  widened truncated-lognormal (worst named fault at 1 log-sigma, up from ~2.2) |
| SBC test cases | 500 | 800 (held-out, generated once from a reserved seed, never
  pooled into training) |
| Eval. replicates/scenario | 50 | `[TBD]` |

### 5.2 Baseline methods

| Method | System | Purpose | Status |
|--------|--------|---------|--------|
| NUTS MCMC (NumPyro) | Propylene oxide | Gold-standard Bayesian | Done |
| EKF (analytical Jacobian) | Propylene oxide | Industrial baseline | Done |
| UKF (sigma-point) | Propylene oxide | Nonlinear filter baseline | Done |
| LDA on physics features | Propylene oxide | Simple ML baseline | Done |
| NUTS/MCMC feasibility argument | Wu 2003 | Explicit infeasibility-at-scale argument | Done (argument, not a run) |
| EKF (jax.jacobian) | Wu 2003 | Industrial baseline | **Deferred** — contingent on classification results (§8.4); prior EKF numbers exist for the pre-rescoping 5-D problem and are retained only as explanatory context (§8.1), not a matched comparison |

### 5.3 Evaluation metrics

- **Posterior accuracy:** CRPS, Wasserstein-1 distance, credible interval coverage (50%, 90%, 95%)
- **Point accuracy:** MAE (posterior mean vs. true θ), bias (signed MAE)
- **Fault classification:** per-class F1, macro-F1 (all classes equally weighted)
- **Calibration:** SBC rank histograms, KS p-value (per-parameter, reported explicitly
  rather than as an aggregate pass/fail — see §7.1)
- **Timing:** ms per observation window (inference only, post-training)

### 5.4 Sequential degradation tracking protocol

30-day profile: 720 consecutive windows (propylene oxide: 60-min windows; Wu 2003: 4-hour
windows). Each window processed independently — no temporal prior propagation. This is a
deliberate design choice: the amortised posterior already processes each window in
milliseconds, and temporal correlations are secondary to the structural identifiability
gap. Window-by-window independence simplifies deployment in existing SCADA historians.

---

## 6. Results: Propylene oxide CSTR (~4.5 pages)

*(Unchanged from prior framing — this system already reads as the "controlled reference
point" the new thesis needs: calibrated, quantitative, every weak spot explained
mechanistically. Only the subsection label in §6.3 changes, from "core contribution" to
reflect its role in this system's own section, not the whole paper.)*

### 6.1 Training validation

**Prior predictive check:** 500 prior draws span the observed data range for all 4 channels.
**SBC:** KS p = 0.21 for α and 0.13 for β — passes at the 5% level. Two-sample classifier
scores 0.53, 0.52 (near 0.5 baseline).

**Figure 1:** Prior predictive coverage (top) + SBC rank histograms (bottom).

### 6.2 Snapshot fault classification

All 8 scenarios, 50 replicates each. Key results:
- Macro-F1 (6 closed-loop scenarios): **0.927**
- Coverage close to nominal for non-saturating scenarios; drops to 0.62 for the
  severe-fouling scenario due to coolant actuator saturation (explained mechanistically,
  not just reported)
- A genuine confound between a temperature-measurement bias and β degradation explains
  the lower score for the sensor-drift scenario

**Figure 2:** 2-D joint posteriors for all 8 scenarios with fault quadrant overlays.

**Table 7:** Per-scenario results (true θ, posterior mean, 90% CI, F1, coverage).

### 6.3 Local sensitivity and practical identifiability (explains the classification results above)

**This analysis explains why jacket fouling is harder to individually estimate than
catalyst activity — it is not itself the paper's headline.**

**6.3.1 Fisher information asymmetry**

Numerical FIM at nominal operating point: I_αα = 850k-975k, I_ββ = 2,000-3,500.
Ratio I_αα/I_ββ = 250-500× across all tested operating points.

**Figure 3:** Fisher information bar chart — I_αα and I_ββ with per-channel contributions.

**6.3.2 Analytical derivation**

Under perfect PI integral action (T ≈ Tsp), the steady-state Jacobian satisfies
∂T_ss/∂β ≡ 0. β is then identified only through the noisier Tc and Qc channels. This is a
specific quantification of the general principle (Ljung 1977; Gevers et al. 2011) for a
CSTR with competing fault parameters.

**6.3.3 Embedding-net irreducibility proof**

A CNN trained on raw (120×4) time series, bypassing the 29-D summaries entirely,
reproduces the same β bias to within <1% — confirming the bias is in the physics
(controller masks β), not in the choice of features. This is the empirical Cramér-Rao-
bound confirmation this paper's System II diagnostic (§8.1) later reuses the same logic
for, applied to a joint rather than a scalar case.

**Figure 4:** Side-by-side marginal posteriors — CNN embedding vs. 29-D hand-crafted.

### 6.4 Multi-method baseline comparison

| Method | β̂ (true=0.70) | β bias | β MAE (tracking) | ms/window | Output |
|--------|-------------------|--------|-----------------|-----------|--------|
| SBI (NSF) | 0.551 | −0.149 | 0.033 | 16 | Full posterior |
| NUTS (MCMC) | 0.598 | −0.102 | 0.102 | 150,000 | Full posterior |
| EKF | 0.607 | −0.093 | 0.065 | 30 | Gaussian (μ, Σ) |
| UKF | 0.607 | −0.093 | 0.090 | 358 | Gaussian (μ, Σ) |

All four methods show the same bias direction and comparable magnitude — this is the
single most important empirical result of Section 6: the bias is structural, not a
deficiency of any particular method.

**Figure 5:** Baseline dashboard.

### 6.5 30-day sequential degradation tracking

- SBI: MAE_α = 0.004, MAE_β = 0.033, wall time 10.4 s for 720 windows
- EKF: MAE_α = 0.012, MAE_β = 0.065
- Correct fault classification in > 95% of windows (SBI); > 88% (EKF)

**Figure 6:** 30-day tracking. **Table 8:** Phase-by-phase tracking metrics.

---

## 7. Results: Wu 2003 reactor-column-recycle benchmark (~5 pages)

**This section is rewritten around the S-B3 scope and the new thesis (classification/
detection as the headline). Numbers marked `[TBD]` are pending the widened-prior
recalibration campaign; the section is structured so it does not need restructuring once
those numbers land, only filling in.**

### 7.1 Training validation

**Prior.** `alpha, beta_r` use a flat/Uniform prior — already validated (§7.1 below shows
SBC passes for both). `eta_col, xi_reb` previously used a narrow truncated-lognormal prior
that put the worst named fault severity only ~2.2 log-sigma into the tail; a 20-seed sweep
under that prior found 0/20 seeds pass SBC for all 4 parameters simultaneously. **The
current, widened prior** puts the worst named fault value at exactly 1 log-sigma for both
`eta_col` and `xi_reb`, and a 20,000-50,000-simulation bank pooled across several machines
is being trained and evaluated against a dedicated, held-out N=800 SBC confirmation set
(never pooled into training).

**SBC (per-parameter KS p-values): `[TBD, pending campaign]`.** Report honestly whichever
of the following turns out to be true, without restructuring the section:
- If all 4 parameters clear SBC: report this as full calibration and proceed to §7.2 with
  no caveat.
- If some parameters do not clear SBC (the realistic contingency to plan for): report the
  per-parameter pass/fail explicitly (not just an aggregate), and lean on §7.2's
  classification results plus the explanatory diagnostic in §8.1 to make the case that
  detection remains reliable regardless — mirroring the sister EAAI paper's own framing
  ("failure-mode identification is highly robust even in regimes where individual
  parameters are weakly identifiable").

**Figure 7:** SBC rank histograms and KS p-values for all 4 S-B3 parameters at the N=800
confirmation budget.

### 7.2 Fault classification (12 scenarios, posterior-mass approach) — headline result

**This is now the paper's headline result for System II.** 12 scenarios (the S-B3
subset — feed-fault scenarios W10/W13 excluded, §3.2), `[N: TBD]` replicates each.
Classification uses posterior mass in fault-unit regions (same approach as §6.2), against
the 4-class S-B3 taxonomy (healthy, reactor, column, multi; §4.5).

**Results: `[TBD]` overall accuracy, `[TBD]` macro-F1, per-class F1 `[TBD]`.** Report
per-class performance explicitly, including any known weak point (e.g. if a specific
scenario is misclassified, explain *why* using §8.1's diagnostic — e.g. a specific
parameter's posterior mean shrinking toward the healthy value past the classification
threshold — rather than leaving it unexplained). This mirrors this paper's own System I
practice (§6.2) of explaining every below-ceiling result mechanistically.

**Figure 8:** Marginal posterior summaries — posterior mean ± 90% CI for all 4 parameters
across the 12 scenarios. **Table 9:** Per-scenario classification results (posterior-mass
F1, 90% CI coverage).

### 7.3 Explanatory diagnostic: why catalyst activity and jacket fouling are hard to
    individually attribute (not a standalone finding)

**Reframed from the old "headline banana" section.** Two candidate joint degeneracies
were investigated under the 66-D summary-statistic representation:

1. **(α, η_col):** initially appeared confounded via the snowball effect's shared push on
   recycle flow (F_R), but a scan combining F_R with the column's own T_reb-derived
   features (both already part of standard S-B instrumentation) collapses the apparent
   ridge to a localized region around the true value. The calibrated posterior shows
   η_col essentially pinned (narrow 90% CI) with near-zero correlation to α. **Not a
   genuine joint non-identifiability** — resolved by combining existing channels, no new
   instrumentation required.

2. **(α, β_r):** appeared far more robust — surviving combination of the entire 66-D
   summary-statistic feature set (FIM off-diagonal +0.90 at nominal, trained-posterior
   correlation ≈0.998 at a combined-fault operating point). Re-deriving the same
   Fisher-information methodology on the *raw, unaggregated* trajectory of the same three
   already-observed channels collapses the off-diagonal to ≈0.00, corroborated
   independently by a re-tuned EKF (raw-trajectory access) recovering both parameters to
   ~1-2% accuracy at points the summary-statistic representation calls non-identifiable.
   **A representation (aggregation) artifact, not a physical joint non-identifiability.**

**Why this matters for §7.2's classification result, and why it is reported here rather
than as its own headline:** both α and β_r fall under the same `reactor` fault class in
the S-B3 taxonomy, so this confound corrupts *parameter-level attribution* (which of the
two is actually degraded) without necessarily corrupting *unit-level detection* (that
*something* in the reactor is degraded) — the practically relevant quantity for
fault/anomaly detection. This is the mechanistic explanation for any residual
parameter-level uncertainty in §7.2's reactor-class results, not a separate contribution.

**Figure 9:** Raw-trajectory vs. summary-statistic FIM off-diagonal comparison for both
candidate pairs, showing the collapse for (α, β_r) and the channel-combination fix for
(α, η_col).

### 7.4 Why no matched classical baseline is reported for this system

**Positioning argument, not an apology for a missing experiment.** Two independent
reasons, both argued explicitly:

1. **NUTS/MCMC is infeasible at this scale.** Even for the 2-D propylene oxide system,
   NUTS mixing is sensitive to posterior geometry; a 4-D recycle-coupled posterior
   evaluated repeatedly across a monitoring horizon multiplies this cost well beyond a
   practical real-time budget. This is a scaling argument, not a claim that MCMC would
   give a wrong answer if run to convergence — it is the reason amortised inference is
   needed at all for this problem class.
2. **EKF requires white-box, raw-model access** (the process model and its Jacobian at
   every step) — in direct tension with the black-box, legacy-simulator applicability
   that motivates SBI (§1.1, §2.3). Where an EKF *was* tested on an earlier, broader
   version of this problem (pre-S-B3 rescoping), it showed the same qualitative pattern
   as System I: reliable near nominal operation, overconfident or divergent near strong
   recycle nonlinearity — retained as supporting context (§8.1 history), not a matched
   S-B3 comparison.

**Contingency:** if §7.2's classification numbers land weak enough to need a comparison
point, re-running EKF specifically under the S-B3 scope is the identified next step
(§8.4) — not attempted preemptively, to avoid spending the same kind of experimental
budget currently committed to the SBC recalibration campaign on a baseline that may not
be needed.

---

## 8. Discussion (~2 pages)

### 8.1 Structural identifiability as an explanatory tool, not the contribution

**One paragraph restating the thesis change, for a reader who skips to the Discussion:**
this paper's contribution is that amortised SBI delivers robust fault/anomaly detection
at plant-wide, multi-loop scale (§6, §7.2); the identifiability analysis below explains
*why* certain parameters are harder to individually attribute, it does not stand as an
independent result.

**System I** illustrates the fundamental *single-loop* mechanism: the reactor temperature
PI controller zeros ∂T_ss/∂β, removing the highest-SNR channel from β's information
budget (I_β/I_α = 1/250-500). This is a **scalar reduction** — one parameter is harder to
identify, but all parameters are identifiable in principle — and it was checked against a
raw-signal representation (the CNN-embedding irreducibility test, §6.3.3) before being
reported as physical.

**System II** reveals the same scalar mechanism, transferring exactly: Loop 1 zeros
∂T_r_ss/∂α and ∂T_r_ss/∂β_r simultaneously, an exact analytical statement about the
plant's steady state. Because Wu 2003 gives neither α nor β_r an independent rescue
channel (unlike System I, where α keeps its concentration channel), masking reduces both
parameters' information roughly *symmetrically* (I_αα/I_β_r ≈ 1.1-1.4×) rather than
singling one out.

**What System II does *not* reveal, on rigorous re-examination, is a genuine joint
(manifold) degeneracy between two different parameters.** Two candidates were
investigated (§7.3) and both failed to survive a raw-trajectory check — one a
restricted-channel artifact, one a lossy-aggregation artifact. **The general lesson,
retained from this project's identifiability work but now framed as a methodological
contribution supporting the classification results, not a headline in its own right:** a
*scalar* identifiability reduction is a robust, representation-independent effect that
survives arbitrarily rich observation access; a *manifold* degeneracy surfaced by a
compressed-feature SBI or FIM analysis should be treated as provisional and checked
against a raw-trajectory Fisher-information test before being reported as physical. We
recommend this check as standard practice for any closed-loop SBI application reporting a
joint non-identifiability.

*(Historical detail retained for completeness/traceability, condensed from the prior
outline's much longer treatment: the (α, η_col) candidate was originally reported as a
confirmed joint degeneracy resolvable by adding a composition analyser (S-A structure);
this was retracted once combining existing S-B channels was shown to resolve it without
new instrumentation. The (α, β_r) candidate was then reported as the paper's headline
finding; it was also retracted once the raw-trajectory FIM check and a re-tuned EKF both
showed it to be a summary-statistic aggregation artifact. A third candidate, (α/β_r,
`z_A0_eff`), was found via an unexplained feed-fault classification failure and showed the
identical signature — this is part of why `z_A0_eff` was rescoped out of the paper's
current reporting scope (§3.2) rather than pursued as a fourth "banana" chapter. Full
detail of this history is preserved in `HANDOFF.md` and project memory, not reproduced
here.)*

### 8.2 EKF's role: a genuine failure mode in System I, an argued-around comparison in System II

EKF is an excellent industrial baseline when the posterior is approximately Gaussian —
which holds for System I near the nominal operating point (all four tested methods show
similar bias and comparable uncertainty). Where System I's EKF *does* fail — near strong
recycle/snowball nonlinearity, tested in the pre-rescoping System II work — it is a
genuine, representation-independent Gaussian-linearisation failure (EKF coverage drops to
single digits while SBI correctly widens its uncertainty), distinct from the artifact
cases in §8.1. For the current System II scope, no matched EKF baseline is reported
(§7.4); this is an explicit scope decision (infeasibility of exact MCMC at this scale,
EKF's white-box requirement), not a gap, with a re-run identified as a contingency should
the classification results warrant a comparison point.

### 8.3 Persistent excitation and open-loop recalibration

Classical closed-loop identification requires the reference signal to be persistently
exciting (Forssell & Ljung 1999). At a fixed setpoint, Φ_r(ω) = 0 — the data are
theoretically uninformative for the parameters masked by integral control. The fact
that SBI still achieves practical fault classification reflects its exploitation of
transient dynamics (overshoots, settling oscillations) that carry implicit parametric
excitation beyond the classical frequency-domain criterion. Practical implication:
scheduled open-loop excitation windows would provide unmasked estimates for the
scalar-masked parameters, for practitioners who need quantitative (not just directional)
estimates.

### 8.4 Limitations

| # | Limitation | Scope | Mitigation |
|---|---|---|---|
| L1 | Synthetic data only, argued explicitly as the correct evidence standard here | Both systems | Calibration/SBC claims require known ground truth, which real plant data cannot provide; a companion paper (Casolo et al., EAAI, condition monitoring of heat exchangers) demonstrates real-world simulator-to-reality transfer for this same SBI methodology on a different asset class, where calibration is not the claim being validated |
| L2 | β bias −0.08 to −0.15 | Propylene oxide | Quantified; predictable; recalibration via OL excitation |
| L3 | Scalar α/β_r information reduction under S-B | Wu 2003 | Same structural mechanism as L2; explanatory, §8.1 |
| L4 | Two candidate joint degeneracies investigated and found to be representation artifacts, not physical limits | Wu 2003 | Retained as explanatory context for §7.2/§7.3, not a standalone finding; full history in `HANDOFF.md` |
| L5 | S-B3 recalibration campaign (widened prior, 20k-50k simulations) in progress at time of writing | Wu 2003 | Section 7.1/7.2 written to be correct whether or not the campaign fully clears SBC; update with final numbers when available |
| L6 | SBC mild miscalibration (KS p=0.016 in an earlier iteration) | Propylene oxide | Structural, not a training deficiency |
| L7 | QSS column shortcut unstable for η_col < 0.80 | Wu 2003 | W8, W14 removed; η_col=0.80 covers headline scenario |
| L8 | No real-time deployment tested | Both | SCADA integration is future work |
| L9 | SNPE training is seed-unstable for the Wu 2003 posteriors | Wu 2003 | Mitigated by a multi-seed sweep with held-out confirmation-budget SBC before promoting any posterior |
| L10 | No matched EKF baseline for the current S-B3 scope | Wu 2003 | Explicit positioning argument (§7.4/§8.2) in place of a re-run; re-running is a live contingency if classification results need a comparison point |
| L11 | `z_A0_eff` (feed composition) and its identifiability couplings are out of scope for this paper's reported results | Wu 2003 | Motivated the S-B3 rescoping (§3.2); a natural extension for future work, not attempted here |

---

## 9. Conclusion (~0.5 pages)

This work demonstrates amortised SBI as a framework for robust Bayesian fault and anomaly
detection in closed-loop chemical processes, scaling from a single PI-controlled unit to
a plant-wide, multi-loop reactor-column-recycle system.

1. For the propylene oxide CSTR, SBI achieves calibrated posteriors (SBC KS p=0.21/0.13)
   and macro-F1=0.927 fault classification despite a known, quantified, and
   mechanistically explained identifiability limit (I_αα/I_ββ=250-500×, confirmed
   irreducible by a raw-trajectory CNN-embedding experiment).

2. For the Wu 2003 reactor-column-recycle plant, restricted to its four endogenous
   degradation parameters, fault/anomaly classification achieves `[TBD]` macro-F1 across
   12 scenarios and 4 fault classes — the paper's central scaling claim: **reliable
   detection survives the transition from a single unit to a multi-loop, plant-wide
   system**, `[TBD: with/without full individual-parameter calibration — state whichever
   is true once the recalibration campaign resolves]`.

3. A worked raw-trajectory Fisher-information diagnostic distinguishes genuine
   identifiability limits from artifacts of a summary-statistic representation, and
   explains why catalyst activity and jacket fouling are hard to individually attribute
   within the reactor fault class — supporting evidence for the classification results,
   not an independent finding.

4. Amortised SBI is argued, not just asserted, as the appropriate instrument for this
   problem class: exact Bayesian sampling is infeasible at plant scale, and classical
   filtering requires white-box model access in tension with the black-box applicability
   that motivates SBI. Where classical filtering was tested, it showed the same
   Gaussian-linearisation failure mode as in the single-unit system, under strong
   nonlinearity.

**Future work:** completion and reporting of the S-B3 recalibration campaign currently in
progress; a matched EKF re-run for System II if warranted by the classification results;
extension to feed-composition faults (`z_A0_eff`, currently out of scope); real-time SCADA
integration; and — following the same audit this paper itself performed — checking other
published closed-loop SBI/FIM "banana" claims against the raw-trajectory diagnostic
proposed here.

---

## Figures summary

| # | Content | Section |
|---|---------|---------|
| 1 | Prior predictive + SBC ranks (System I) | §6.1 |
| 2 | Joint posteriors all 8 scenarios (System I) | §6.2 |
| 3 | Fisher information channel decomposition (System I) | §6.3.1 |
| 4 | CNN embedding vs. hand-crafted (System I) | §6.3.3 |
| 5 | 4-method baseline dashboard (System I) | §6.4 |
| 6 | 30-day tracking with CI bands (System I) | §6.5 |
| 7 | SBC rank histograms and KS p-values, all 4 S-B3 parameters, N=800 confirm | §7.1 |
| 8 | Marginal posterior summaries, 12 scenarios, S-B3 | §7.2 |
| 9 | Raw-trajectory vs. summary-statistic FIM off-diagonal comparison (explanatory) | §7.3 |

All figures: 300 dpi minimum, double-column compatible, colour-blind palette (viridis/Okabe-Ito).

---

## Tables summary

| # | Content | Section |
|---|---------|---------|
| 1 | Propylene oxide model parameters | §3.1 |
| 2 | PO fault scenarios (Sc1-Sc8) | §3.1 |
| 3 | Wu 2003 plant parameters (from Table 1, SI units) | §3.2 |
| 4 | Wu 2003 fault scenarios (12-scenario S-B3 subset) and taxonomy | §3.2 |
| 5 | Full summary statistics definition (29-D PO; 66-D Wu 2003) | §4.2 |
| 6 | SBI training hyperparameters for both systems | §5.1 |
| 7 | PO per-scenario classification results | §6.2 |
| 8 | 4-method baseline comparison (SBI, NUTS, EKF, UKF) | §6.4 |
| 9 | Wu 2003 per-scenario classification results (S-B3) | §7.2 |

---

## Appendices

- **A. Nomenclature:** All symbols, units, and definitions (C&ChE requirement)
- **B. Density estimator comparison:** MAF vs. NSF on propylene oxide; justification for NSF choice
- **C. MCMC convergence diagnostics:** R̂, ESS, trace plots for NUTS on propylene oxide
- **D. Analytical steady-state Jacobian:** Full derivation of ∂T_ss/∂β = 0 under PI control
- **E. Additional posterior plots:** Remaining scenario pairplots for both systems
- **F. Identifiability diagnostic history:** condensed record of the (α, η_col) and
  (α, β_r) investigation, retained for a reader who wants the full methodological trail
  behind §8.1 without it occupying the main text

---

## Pre-submission checklist

**Completed (propylene oxide work):**
- [x] EKF/UKF baseline
- [x] 4-method β bias confirmation
- [x] SBC with honest reporting
- [x] CNN embedding irreducibility proof
- [x] Fisher information analysis
- [x] 30-day tracking with MAE/CRPS

**Completed (Wu 2003, pre-rescoping identifiability work — retained as §8.1 explanatory
context):**
- [x] S-B calibration (5-D, original scope)
- [x] (α, η_col) and (α, β_r) joint-degeneracy investigation and retraction
- [x] Raw-trajectory FIM diagnostic methodology
- [x] EKF runs at the pre-rescoping scenario set

**Required before submission (current, S-B3-scoped):**
- [ ] S-B3 widened-prior recalibration campaign complete (20k-50k simulations, pooled,
      held-out N=800 SBC confirm) — **in progress**
- [ ] Final per-parameter SBC KS p-values for S-B3 reported in §7.1, whatever they are
- [ ] Final classification metrics (accuracy, macro-F1, per-class F1) reported in §7.2
- [ ] Decision made and recorded on whether a matched EKF re-run for S-B3 is needed
      (§7.4/§8.4 contingency)
- [ ] Abstract, Highlights, Contributions, Conclusion numbers filled in from the above
- [ ] All figures regenerated at publication quality (300 dpi, double-column)
- [ ] Nomenclature table with every symbol (C&ChE requirement)
- [ ] No undefined acronyms in abstract
- [ ] Ljung (1977), Gevers et al. (2011), Forssell & Ljung (1999), Luyben (1994) cited
- [ ] Fault classification framed as "label-free" not "unsupervised"
- [ ] Data availability statement (C&ChE requirement)
- [ ] CRediT author statement
- [ ] Conflict of interest statement
- [ ] Simulator/training data available (GitHub or Zenodo)

**Removed from checklist (not required):**
- ~~Prior sensitivity study for propylene oxide~~ — bias proved structural via 4-method
  confirmation + analytical derivation
- ~~Model mismatch robustness~~ — deferred to future work; L1 covers the synthetic-data
  positioning
- ~~A second "banana" chapter for `z_A0_eff`~~ — rescoped out of the paper (§3.2, L11)

---

## Effort estimate

| Task | Effort | Status |
|------|--------|--------|
| Propylene oxide results | — | **DONE** |
| Wu 2003 plant implementation and pre-rescoping identifiability work | — | **DONE** (retained as §8.1 context) |
| S-B3 widened-prior recalibration campaign (20k-50k sims) | days (running now, multi-machine) | **In progress** |
| S-B3 classification + final SBC report | ~1 day once campaign completes | Blocked on campaign |
| Contingent: EKF re-run for S-B3 | 2-3 days | Live contingency, not committed |
| Outline/`main.tex` narrative realignment (this pass) | — | **DONE this session** |
| Publication-quality figures | 2-3 days | Open |
| Paper writing (remaining polish) | 1-2 weeks | After S-B3 results land |
