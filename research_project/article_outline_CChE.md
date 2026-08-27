# Article outline — Computers & Chemical Engineering (C&ChE)

> Target: Computers & Chemical Engineering (Elsevier, IF ~4.0)
> Format: double-column, ~25-35 pages, ~10,000-12,000 words
> Key C&ChE expectations: rigorous process model with published parameters,
> industrial relevance, comparison with standard industrial baselines (EKF/UKF),
> scalability demonstration, honest treatment of limitations

---

## Critical framing advice

**What C&ChE reviewers will look for:**
1. A process systems engineering contribution — not just "ML applied to a reactor"
2. A model grounded in published parameter sources (Fogler, Luyben) — no made-up systems
3. Comparison with standard industrial state estimators (EKF is mandatory; UKF welcome)
4. Scalability: single-unit → plant-wide, few parameters → many parameters
5. Honest treatment of identifiability limitations (don't hide the β bias)

**What this paper delivers that is genuinely new:**
- Systematic quantification of how decentralized plant-wide control degrades parameter
  identifiability — from a single PI loop (2-D, I_bb 250-500× smaller than I_aa) to
  five coupled PI loops in a recycle plant (8-D, non-Gaussian posteriors)
- Empirical proof that the identifiability loss is irreducible (CNN embedding, 4-method
  confirmation on propylene oxide; EKF overconfidence on Luyben snowball)
- First demonstration that amortised SBI is the only practical full-Bayesian method
  at plant scale (MCMC infeasible at 8-D; SBI amortises the cost)

**What NOT to claim:**
- Do not claim the closed-loop identifiability limitation as a discovery — cite Ljung (1977)
- Do not call fault classification "unsupervised" — it is "label-free"
- Do not present speedup over MCMC as the primary contribution; the inferential
  correctness story (non-Gaussian posteriors vs. EKF Gaussian assumption) is stronger

**The strongest framing for C&ChE:**
> We characterise the structural identifiability limitations imposed by decentralized
> feedback control on Bayesian fault diagnosis — scaling from a single PI-controlled
> CSTR to a plant-wide recycle process — and demonstrate that amortised simulation-based
> inference correctly recovers non-Gaussian posteriors where the extended Kalman filter
> gives overconfident Gaussian approximations and MCMC becomes computationally infeasible.

---

## Highlights (C&ChE requires 3-5 bullet points, ≤85 characters each)

- Feedback control structurally reduces fault parameter identifiability in closed-loop systems
- Fisher information I_β/I_α ratio = 1/250–1/500 for PI-controlled CSTR; irreducible
- Amortised SBI recovers non-Gaussian posteriors; EKF fails completely (3% coverage) at snowball
- **Two apparent "banana" degeneracies in a recycle plant were both artifacts, not physics —
  a general SBI pitfall under closed-loop control**
- **A raw-trajectory Fisher-information check (not more summary features) is the fix: it
  collapses a 0.90-correlation confound to ~0.00 using sensors already in place**

**Retracted this session (2026-07-03) — do not reuse:** "Recycle coupling creates (α, η_col)
banana posterior invisible to EKF; SBI captures it (100% coverage)." See §7.2.3/§7.4/§8.4 L4
for why: this pair is not a genuine joint degeneracy once the full S-B feature set (not just
F_R) is considered — see HANDOFF.md session 2026-07-03c.

**SUPERSEDED (2026-07-05) — do not reuse:** "Recycle plant reveals a genuine (α, β_r)
banana; SBI captures it (corr=0.998), EKF cannot." Two independent checks (a re-tuned EKF,
and this paper's own FIM methodology applied to the raw sensor trajectory instead of
hand-crafted summary statistics — see §7.4, §8.4 L4′) show this second "banana" is *also*
an artifact — this time of temporal aggregation in the 66-D summary-statistic feature set,
not of a restricted channel set as (α, η_col) was. **This is now the article's actual
headline finding for the recycle system** (replacing the retracted claim above), reframed
as the two new highlights added at the top of this list. A trained, richer-feature (CNN
embedding) SBI posterior confirming this at the calibration level was later attempted
(`nb32`, §7.4.1) and did **not** narrow the correlation (0.993 vs. hand-crafted 0.994,
robust across 19 tested conditions) — see §8.4 L11. This does not overturn the finding: the
FIM and re-tuned-EKF evidence for "artifact, not physics" stands independently of whether an
amortized CNN posterior can extract that information at a modest training budget. The claim
here remains scoped to "very likely an artifact, demonstrated by two independent methods,"
not "fixed by a trained SBI posterior" — that specific attempt did not succeed.

---

## Title options

1. **Plant-wide Bayesian fault diagnosis in recycle processes: amortised simulation-based
   inference under multi-loop feedback control** (recommended — descriptive, C&ChE idiom)

2. **Structural identifiability limits of Bayesian fault diagnosis under feedback control:
   from a PI-controlled CSTR to a Luyben reactor-column-recycle plant** (leads with the theory)

3. **Amortised simulation-based inference for real-time probabilistic fault diagnosis in
   feedback-controlled chemical processes** (broader, mirrors EAAI option but more suitable
   if journal scope is uncertain)

Recommendation: option 1. It names the process (recycle), the method (SBI), and the
challenge (multi-loop control). C&ChE readers will immediately understand the scope.

**Revisit after 2026-07-05 reframing:** the paper's actual headline result is no longer
"a banana exists" but "closed-loop SBI systematically over-reports joint degeneracies
under compressed summary statistics, and a raw-trajectory Fisher-information check
distinguishes real from artifactual ones" (see §7.4, §8.1). Option 2 ("Structural
identifiability limits... from a PI-controlled CSTR to a Luyben reactor-column-recycle
plant") now fits this better than option 1, since it foregrounds the diagnostic/theoretical
contribution rather than a specific (now-retracted-twice) confound. Consider a fourth option
explicitly naming the diagnostic, e.g. **"Genuine vs. artifactual non-identifiability in
closed-loop simulation-based inference: a Fisher-information diagnostic for plant-wide
fault diagnosis."**

---

## Abstract (~250 words, structured)

**Background.** Fault diagnosis in feedback-controlled chemical processes is fundamentally
limited by the controller's tendency to compensate for degradation, suppressing the
parametric signatures that estimation methods rely on. This closed-loop identifiability
limitation is classical (Ljung 1977) but has not been systematically characterised for
multi-unit plant-wide processes or for amortised Bayesian inference methods.

**Methods.** We apply simulation-based inference (SBI) — training a neural spline flow
to approximate the Bayesian posterior over degradation parameters from physics-informed
summary statistics of fixed-length observation windows — across two systems of increasing
complexity: a PI-controlled propylene oxide CSTR (2 parameters, 1 unit) and the Luyben
reactor-column-recycle benchmark (5 parameters, 2 units plus recycle, 3 PI loops; Wu,
Yu, Luyben & Skogestad 2003). We characterise identifiability using numerical Fisher
information analysis, compare against extended Kalman filter baselines using automatic
differentiation of the ODE Jacobian, and validate with simulation-based calibration.

**Results.** For the propylene oxide system, the Fisher information for the fouling
parameter is 250–500× smaller than for catalyst activity — confirmed empirically by four
independent methods (SBI, MCMC, EKF, UKF) all showing identical structural bias, and shown
to be irreducible by a raw-time-series CNN embedding that bypasses hand-crafted features
entirely (a Cramér-Rao-bound confirmation). For the recycle plant we investigated two
candidate joint (2-parameter) degeneracies — catalyst decay with column efficiency
(α, η_col), and catalyst decay with jacket fouling (α, β_r) — and found **both to be
artifacts of the observation representation, not of the plant physics**. The first
(α, η_col) resolves once a second, already-available S-B channel (T_reb) is combined with
the recycle flow; restricting attention to the recycle flow alone reproduces the
appearance of a banana. The second, (α, β_r) — which initially appeared as a strong,
robust confound (Fisher-information off-diagonal +0.90, trained-posterior correlation
0.998, and a degeneracy that survived combination of the *entire* 66-D hand-crafted
summary-statistic feature set) — collapses to an off-diagonal of ≈0.00 when the same
Fisher-information methodology is applied to the raw, unaggregated sensor trajectory of the
same three already-observed channels instead of their whole-window summary statistics; an
independently re-tuned extended Kalman filter given the same raw-trajectory access
corroborates this, recovering both parameters to ~1-2% accuracy at points the
summary-statistic representation calls non-identifiable. Neither fix required new
instrumentation — only less-aggregated use of existing sensors. Modestly finer time
resolution (sub-window statistics) was not sufficient; a genuinely raw-trajectory-aware
representation was needed. We trained a CNN-embedding SBI posterior directly on raw
trajectories to test this at the level of a calibrated, amortized estimator: it did not
narrow the (α, β_r) correlation (0.993 vs. 0.994, robust across 19 tested conditions), while
independently confirming the resolution of a related three-way confound — distinguishing
information-theoretic recoverability (established, via Fisher information and the EKF) from
recoverability by this specific estimator at a modest training budget (not established).
Separately, at a strongly degraded catalyst-decay operating
point the EKF achieves only 3% empirical coverage under strong recycle ("snowball")
nonlinearity — a real, representation-independent failure mode distinct from either banana
investigation. SBI inference takes under 20 ms per window after a one-time training cost;
30-day sequential monitoring completes in seconds.

**Conclusions.** Amortised SBI is the only currently practical method that is both
computationally feasible for real-time monitoring and qualitatively correct in representing
non-Gaussian posterior geometry, and it degrades gracefully under strong recycle
nonlinearity where EKF fails outright (3% coverage). But this work's central methodological
finding is a caution about SBI itself in closed-loop, multi-unit settings: **every apparent
joint non-identifiability we found in the recycle plant turned out to be an artifact of the
hand-crafted summary-statistic representation, not a property of the plant** — first a
restricted-channel artifact, then a lossy-temporal-aggregation artifact — while the one
identifiability limitation that proved genuine and representation-independent (reactor
thermal faults masked by integral temperature control) is a *scalar* reduction in
information, not a manifold constraint between two different parameters, and reproduces
identically across both systems studied. We propose that any joint degeneracy surfaced by
a compressed-feature SBI posterior in a closed-loop system should be checked against a
raw-trajectory Fisher-information test before being reported as physical, and provide a
worked methodology for doing so. This shifts the paper's design guidance from sensor
placement (a new analyser) to feature engineering (less lossy use of existing sensors) as
the higher-value, lower-cost intervention for this class of system.

---

## Keywords (6-8)

simulation-based inference; amortised Bayesian inference; fault diagnosis;
closed-loop identifiability; recycle process; Fisher information; extended Kalman filter;
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
risk-informed maintenance scheduling and root-cause attribution.

**Para 2 — The feedback masking problem.**
In feedback-controlled plants, decentralized PI controllers actively compensate for
faults: a fouled heat exchanger is countered by increased coolant flow; a deactivated
catalyst is countered by higher reactor temperature setpoints. This compensation
suppresses the very parametric signatures that estimation methods rely on — a
classical closed-loop identifiability limitation established since Gustavsson, Ljung &
Söderström (1977). Crucially, this limitation is not a deficiency of any particular
estimation method; it affects all methods — Bayesian or frequentist, online or offline —
because it is rooted in the information content of the closed-loop data.

**Para 3 — Why SBI, and why now.**
Simulation-based inference (SBI; Cranmer et al. 2020) replaces an explicit likelihood
with a neural density estimator trained on simulated (parameter, observation) pairs.
Three properties make it attractive for plant-wide fault diagnosis: (a) amortisation —
a single trained network processes any new observation in milliseconds, enabling
real-time deployment; (b) full posterior distributions — capturing non-Gaussian posterior
shapes that classical methods cannot represent; (c) no likelihood derivation required —
the process simulator is used directly, without linearisation or Gaussian noise assumptions.
These advantages are critical near recycle tipping points, where the nonlinear dynamics
produce posterior shapes that invalidate the Gaussian approximations underlying EKF/UKF,
and where the monitoring cadence (2-hour windows across a 30-day operating period) makes
MCMC's per-window cost prohibitive.

### 1.2 Contributions (numbered list, exactly 4)

> 1. **Closed-loop identifiability characterisation for a PI-controlled CSTR.**
>    We numerically compute the 2×2 Fisher information matrix for catalyst activity (α)
>    and jacket fouling (β), showing I_αα/I_ββ = 250–500× across all operating points.
>    We derive the mechanism analytically and confirm via a CNN embedding experiment
>    that the bias is irreducible by any choice of summary statistics (Cramér-Rao bound).
>    Four independent methods (SBI, NUTS, EKF, UKF) show identical structural bias,
>    transforming the finding from "SBI has a bias" to "the data are informationally
>    limited for β under closed-loop control."
>
> 2. **Amortised SBI for a PI-controlled CSTR: real-time fault classification.**
>    A trained neural spline flow achieves macro-F1 = 0.990 on six fault scenarios with
>    calibrated uncertainty, processing each 60-min window in 15 ms — 53,000× faster than
>    NUTS. A 30-day sequential tracking study demonstrates MAE_α = 0.004 and MAE_β = 0.033
>    with correct fault classification in >95% of windows.
>
> 3. **Plant-wide fault localization in the Wu 2003 reactor-column-recycle benchmark, and
>    two candidate joint degeneracies that both turned out to be representation artifacts.**
>    We extend to the Wu et al. (2003) CSTR-column-recycle process (Comput. Chem. Eng.
>    27(3):401–421; 5 fault parameters, 3 PI loops, liquid recycle, 14 scenarios). Unlike the
>    PO CSTR, where α is cleanly identified via the concentration channel, Wu 2003 has **no
>    observable concentration channel under either instrumentation structure**, which
>    initially appeared to produce two severe joint (2-parameter) confounds: (a) catalyst
>    decay with column tray efficiency (α, η_col), which resolves once the column's own
>    temperature/flow channels (already part of standard S-B instrumentation) are combined
>    with the recycle flow — an artifact of restricting attention to one channel; and (b)
>    catalyst decay with reactor jacket fouling (α, β_r), which appeared far more robust —
>    Fisher-information off-diagonal +0.90, trained-posterior correlation 0.998, and a
>    degeneracy that survived combination of the *entire* 66-D hand-crafted summary-
>    statistic feature set — but which we show (Contribution 4) is *also* an artifact, this
>    time of lossy temporal aggregation rather than channel restriction. The one
>    identifiability limitation in this plant that IS genuine and representation-independent
>    is a *scalar* one: both α and β_r lose their shared highest-SNR channel (T_r) to Loop-1
>    integral masking — the same mechanism as the PO CSTR's β masking, transferring exactly,
>    but here reducing both parameters' information roughly symmetrically (I_αα/I_β_r ≈
>    1.1–1.4×, far milder than the PO system's 250–500×) rather than singling one out.
>
> 4. **A raw-trajectory Fisher-information diagnostic distinguishes genuine from artifactual
>    non-identifiability in closed-loop SBI — demonstrated, not just proposed.** Applying
>    this paper's own FIM methodology (§4.3) to the raw, unaggregated sensor trajectory of
>    the *same three already-observed channels* (T_r, T_j, F_R_norm) — instead of their
>    66-D whole-window summary statistics — collapses the (α, β_r) off-diagonal from ≈0.90
>    to ≈0.00, reproducibly across noise seeds and at multiple operating points. An
>    independently re-tuned EKF given the same raw-trajectory access corroborates this,
>    recovering both parameters to ~1-2% accuracy at points the summary-statistic
>    representation calls non-identifiable. Modestly finer time resolution (sub-window
>    statistics) is *not* sufficient — the information lost by whole-window aggregation is
>    in fine-grained transient shape, not coarser-grained level and spread. **This reframes
>    the paper's design guidance from sensor placement to feature engineering**: the fix for
>    this class of apparent confound is a less lossy use of existing sensors, not a new
>    analyser — a cheaper and more general intervention than the composition-analyser
>    guidance our earlier, retracted framing would have produced. We additionally trained a
>    calibrated CNN-embedding SBI posterior on raw trajectories to test this diagnosis
>    directly (§7.4.1): it did not narrow the (α, β_r) correlation (0.993 vs. 0.994, robust
>    across 19 tested conditions), distinguishing information-theoretic recoverability
>    (established by the FIM and EKF) from recoverability by this specific amortized
>    estimator at a modest training budget — a training-practicality gap, not evidence
>    against the representation-artifact diagnosis. The same trained network did
>    independently confirm the resolution of a related three-way confound
>    ((α/β_r, z_A0_eff), §7.3), showing the two candidates differ in how hard they are for a
>    modest CNN embedding to fix in practice. Separately, EKF achieves only 3% empirical
>    coverage under strong catalyst-decay/snowball
>    conditions (mean estimate 1.185 vs. true α = 0.75) — a distinct, representation-
>    independent failure mode. SBI's amortisation cost is paid once at training time;
>    subsequent inference takes <20 ms, enabling 720-window 30-day monitoring in seconds.

### 1.3 Paper organisation

One sentence per section. Section 2 reviews related work. Section 3 formulates both
process systems. Section 4 describes the methodology. Section 5 details the experimental
setup. Sections 6 and 7 present results for the propylene oxide and Luyben systems
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
  hand-tuned noise covariances; Gaussian approximation fails when posterior is
  multimodal or banana-shaped
- Data-driven methods: PCA/PLS monitoring (Qin 2012); deep learning anomaly detection
  (LSTM, autoencoders); these produce alarms, not posteriors
- **Gap:** no method produces calibrated full posteriors over multi-dimensional fault
  parameters in a plant-wide setting, and no prior work quantifies how the control
  structure systematically limits what can be inferred.

### 2.2 Closed-loop identifiability

- Classical results: Gustavsson, Ljung & Söderström (1977); Forssell & Ljung (1999)
  — three families of closed-loop identification (direct, indirect, joint input-output)
- Fisher information for closed-loop experiment design: Gevers, Bombois, Hildebrand &
  Solari (2011); Bombois et al. (2006)
- Persistent excitation under feedback: the reference signal must have non-zero spectral
  density for consistent parameter identification (Forssell & Ljung 1999 §3); a fixed
  setpoint implies zero excitation in the classical sense
- Fouling masking under temperature control: acknowledged qualitatively in the
  heat-exchanger literature (Chen & Victorino, Chem. Eng. Res. Des. 2022)
- Snowball effect and recycle instability: Luyben (1994); plant-wide control design
  (Luyben, Tyreus & Luyben 1997)
- **Gap:** the Fisher information ratio for competing fault parameters (kinetic vs.
  thermal) has not been computed for CSTRs, and the multi-unit generalisation (how
  additional control loops progressively degrade identifiability) is unexplored.

### 2.3 Simulation-based inference

- Foundations and review: Cranmer, Brehmer & Louppe (2020, PNAS)
- Neural posterior estimation: NPE-A (Papamakarios & Murray 2016), NPE-B (Lueckmann
  et al. 2017), SNPE-C (Greenberg et al. 2019)
- Density estimators: masked autoregressive flows (Papamakarios et al. 2017), neural
  spline flows (Durkan et al. 2019)
- sbi software library: Tejero-Cantero et al. (2020)
- Applications: gravitational waves (Dax et al. 2021 NeurIPS), epidemiology, neuroscience
- Misspecification detection: Schmitt et al. (2024) — MMD-based diagnostic for
  out-of-distribution observations; does not correct structural bias
- **Gap:** SBI has not been applied to feedback-controlled process systems. Existing
  applications assume the simulator faithfully generates i.i.d. data; the structured
  information loss from closed-loop control creates a qualitatively different challenge.

### 2.4 Plant-wide process monitoring

- Classical plantwide control: Luyben et al. (1997), Seborg et al. (2011)
- Multivariable fault isolation: directional residuals, causal analysis (Bauer et al. 2007)
- Recycle dynamics: snowball effect (Luyben 1994); interaction between unit-level
  controllers under plant-wide disturbances
- **Gap:** no prior work applies probabilistic Bayesian inference for simultaneous
  estimation of degradation parameters across multiple coupled units in a recycle process
  under decentralized PI control.

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

### 3.2 System II: Luyben reactor-column-recycle benchmark

**Reaction:** A → B (first-order, irreversible, exothermic, liquid phase). The canonical
reactor-separator-recycle benchmark from the "Dynamics and control of recycle systems"
series (Luyben 1993 I&EC Research). **Complete parameters from Wu, Yu, Luyben &
Skogestad (2003, Comput. Chem. Eng. 27(3):401–421) Table 1** — the same paper that
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

**Control structure comparison (from Wu et al. 2003):**

| Feature | Structure S-A (rich) | Structure S-B (conventional) |
|---------|---------------------|------------------------------|
| Measurements | T_r, T_j, Q_c, x_D, T_reb, Q_reb, F_R, F_B | T_r, T_j, Q_c, T_reb, Q_reb, F_R, F_B |
| Column composition control | Cascade QC: x_D, x_B | Ratio RC: F_R/F_fresh fixed |
| Fault masking | Partial: x_D directly observed | Strong: composition drift invisible |
| Wu 2003 analogue | **B-2** (two comp. loops: x_D, x_B) | Simpler than B-1 (zero comp. loops) |

**Fault parameterisation (5-D):**

| # | Symbol | Prior | Physical meaning | Primary observable |
|---|--------|-------|-----------------|-------------------|
| 1 | α | U[0.4, 1.2] | Catalyst activity | z_A, F_R (via snowball) |
| 2 | β_r | U[0.4, 1.2] | Reactor jacket fouling | T_j, Q_c (masked by Loop 1) |
| 3 | η_col | U[0.5, 1.0] | Column tray efficiency | x_D, T_reb, Q_reb |
| 4 | ξ_reb | U[0.4, 1.2] | Reboiler heat transfer fouling | Q_reb (Loop 3 compensation) |
| 5 | z_A0 | U[0.70, 0.95] | Feed purity (A fraction) | All channels via reactor SS |

**Identifiability structure (key finding, revised 2026-07-05 — see §7.2/§7.4/§8.1/§8.4
L4, L4′):**
- T_r is masked by Loop 1 for BOTH α and β_r (∂T_r_ss/∂α ≡ ∂T_r_ss/∂β_r ≡ 0), and Wu 2003
  has no observable concentration channel (unlike the PO CSTR) — this is a genuine,
  representation-independent **scalar** reduction affecting both parameters roughly
  symmetrically: I_αα/I_β_r ≈ 1.1–1.4× (nearly equal, vs. PO's 250–500×). Under the
  standard 66-D hand-crafted summary-statistic feature set, (α, β_r) *also* appears
  jointly confounded (off-diagonal +0.90 at nominal) — but this joint/manifold component is
  very likely a representation artifact, not physical: the same analysis on the raw,
  unaggregated observation trajectory collapses the off-diagonal to ≈0.00 (§7.4, §8.1).
- (α, η_col): initially suspected as banana-shaped via the snowball's shared effect on
  recycle flow (F_R), but this does **not** survive once the column's own temperature and
  flow channels are combined with F_R — S-B's existing instrumentation already resolves
  most of this pair without a composition analyser (see §7.2.3, §8.4 L4).
- ξ_reb: identifiable via Q_reb (controller output signal), analogous to Qc for β_r
- z_A0: identifiable via z_A and F_R (different steady-state trajectory)

**Observations (8 channels under S-A; 7 channels under S-B):**
T_r, T_j, Q_c, T_reb, Q_reb, F_R, F_B [both]; x_D [S-A only].
2-hour windows at 1-min resolution → 120×7/8 time series → 55-D summary statistics.

**Table 3:** All nominal plant parameters from Wu et al. (2003) Table 1 (converted to SI):
reactor holdup M_r=2400 lbmol, k_ss=0.33/h, Ea=71.74 kJ/mol, UA=254,000 kJ/(h·K),
ΔH_r=69,780 kJ/kmol, ρCp=2.82 MJ/(m³·K), α_rel=2.0, N_T=20, R=2.2, τ_hyd=4 s.

**14 fault scenarios, 30 replicates each (Table 4):**
Individual faults W2–W7, W9 (one parameter degraded; W8 η_col=0.65 removed — see §8.4),
combined faults W10–W13, W15 (cross-unit; W14 η_col+ξ_reb removed — see §8.4),
snowball threshold W15 (α near critical tipping point), full multi-fault W16.
Both control modes S-A and S-B run for each scenario (420 windows per structure, 840 total).

**Note on removed scenarios:** W8 (η_col=0.65) and W14 (η_col=0.75+ξ_reb=0.75) are excluded
because the Kremser shortcut column model becomes numerically unstable for η_col < 0.80 from
the nominal warm start under S-B (ODE blow-up). The stability boundary η_col ≥ 0.80 covers
the headline scenario W12 (η_col=0.80) and all other scientifically relevant operating points.

### 3.3 Stochastic simulation

Both systems use Euler-Maruyama integration (JAX scan, JIT-compiled) with additive
process noise and Gaussian sensor noise (0.5% of channel range). Warm-start initial
conditions are computed via diffrax Tsit5 adaptive integrator from the nominal steady
state.

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
(1999), NPE operates directly on the measured histories [C or Ca/Cb, T, Tc, Qc, ...]
without an explicit controller model, making it an analogue of the *direct method* of
closed-loop identification. Like the classical direct approach, it is universally
applicable to nonlinear, saturating feedback and exploits in-loop process noise to
partially reduce posterior variance.

**Density estimator:** Neural Spline Flow (NSF; Durkan et al. 2019) with 128 hidden
units and 5 transforms (propylene oxide, 2-D) or 192 hidden units and 7 transforms
(Luyben, 8-D). Justified over MAF by tighter posteriors at the same training budget
(see Appendix B). Training: 10,000 prior draws per posterior (propylene oxide); 10,000–
20,000 (Luyben, tuned based on SBC calibration). Software: sbi library v0.23+.

### 4.2 Summary statistics

**Propylene oxide (29-D):**
- Per-channel base (5×4 = 20): mean, std, slope, min, max over [C, T, Tc, Qc]
- Final-window means (4): last-25% average per channel
- Control aggregates (3): ∫|T − Tsp|dt, Qc_sat_low_frac, Qc_sat_high_frac
- Physics-informed (2): UA_eff_proxy = (T_mean − Tc_mean)/Qc_mean; k0_eff_proxy = log(C_mean/(Ci − C_mean))

**Luyben plant (65-D):**
- Per-channel base (5×8 = 40): same base stats for all 8 channels
- Final-window means (8)
- Control aggregates (6): ∫|T_r err|dt, ∫|T_s err|dt, Qc_sat_frac, Q_s_sat_frac, F_R_std, F_P_std
- Physics-informed (11): UA_r_eff_proxy, UA_s_eff_proxy, recycle_load, purge_deviation,
  pump_head_proxy, conversion_proxy, recycle_richness, feed_preheat_proxy,
  corr(Qc, F_R), corr(Q_s, T_s), corr(F_R, F_P)

**Table 5:** Full 29-D and 65-D feature definitions with physical interpretation.

### 4.3 Fisher information analysis

The local Fisher information matrix at θ is:
```
I(θ) = J^T Σ_obs^{-1} J,   J_{ij} = ∂s_i/∂θ_j
```
where s is the summary statistics vector and Σ_obs is the diagonal observation noise
covariance estimated from healthy replicates. J is computed via finite differences
on the simulator. The diagonal elements I_αα, I_ββ quantify how much information the
data carry about each parameter; off-diagonals reveal correlations.

The Cramér-Rao bound gives Var(θ̂_i) ≥ [I(θ)^{-1}]_{ii} for any unbiased estimator.
When I_ββ ≪ I_αα, β is fundamentally harder to identify regardless of the inference method.

**Analytical derivation (propylene oxide):** Under steady-state PI control (T = Tsp),
the temperature row of J is identically zero: ∂T_ss/∂β = 0. This eliminates the
highest-SNR channel from β's information budget. The remaining channels (Tc, Qc) are
noisier and smaller in magnitude, giving I_ββ/I_αα ≈ 1/250–1/500.

**8×8 FIM for Luyben:** Numerical computation at the nominal operating point provides
the identifiability hierarchy across all 8 parameters and reveals the (α, η_sep) 
off-diagonal coupling under partial observability (no concentration measurements).

### 4.4 EKF baseline

**Propylene oxide (6-state augmented EKF):**
Augmented state [C, T, Tc, I, α, β]. Analytical 6×6 Jacobian (hand-derived, as per nb16).
Parameters treated as random-walk states: d(param)/dt = 0 with small process noise.
Measurement equation: [C, T, Tc, Qc] directly; Qc computed from PI controller equations.

**Luyben plant (21-state augmented EKF):**
Augmented state: 13 plant/controller states + 8 degradation parameters.
**Jacobian computed via jax.jacobian(rhs) at each EKF step** — no hand derivation of
the 21×21 matrix. This eliminates the dominant implementation risk while providing
the exact local linearisation.
Discretisation: F = expm(A·dt) via scipy.linalg.expm at each observation step.
Noise covariances Q, R tuned on Sc_L1 (healthy) data.

**UKF (propylene oxide only):** Sigma-point propagation with 13 points; included to
confirm that the β bias is not a linearisation artifact of the EKF.

### 4.5 Fault classification

**Propylene oxide (2-D, quadrant taxonomy):**
- Healthy: α ≥ 0.85, β ≥ 0.85
- Fouling-dominant: α ≥ 0.85, β < 0.85
- Decay-dominant: α < 0.85, β ≥ 0.85
- Combined: α < 0.85, β < 0.85
Threshold 0.85 is domain-informed to account for the expected structural β bias.

**Luyben plant (8-D, hierarchical taxonomy):**
- Healthy: all parameters near nominal
- Reactor-fault: α or β_r degraded, separator/recycle normal
- Separator-fault: η_sep or β_s degraded, reactor normal
- Recycle-fault: η_p or ξ degraded
- Feed-fault: κ or δ outside tolerance
- Combined: multiple units degraded simultaneously

**Classification rule:** assign posterior probability mass to each region; report the
mode class and the probability of the most likely class as calibrated confidence.

---

## 5. Experimental setup (~1.5 pages)

### 5.1 Training configuration

**Table 6:** Hyperparameters for both systems

| Hyperparameter | Propylene oxide | Luyben plant |
|---|---|---|
| Training simulations | 10,000 | 10,000–20,000 |
| NSF hidden units | 128 | 192 |
| NSF transforms | 5 | 7 |
| Batch size | 256 | 256 |
| Max epochs | 200 | 300 |
| SBC test cases | 500 | 500 |
| Eval. replicates/scenario | 50 | 30 |

### 5.2 Baseline methods

| Method | System | Purpose | Status |
|--------|--------|---------|--------|
| NUTS MCMC (NumPyro) | Propylene oxide | Gold-standard Bayesian | Done (nb05) |
| EKF (analytical Jacobian) | Propylene oxide | Industrial baseline | Done (nb16) |
| UKF (sigma-point) | Propylene oxide | Nonlinear filter baseline | Done (nb16) |
| LDA on physics features | Propylene oxide | Simple ML baseline | Done (nb05a) |
| EKF (jax.jacobian) | Luyben | Industrial baseline | Planned (nb36) |
| MCMC infeasibility argument | Luyben | Dimensionality argument | Explicit (nb33) |

### 5.3 Evaluation metrics

- **Posterior accuracy:** CRPS, Wasserstein-1 distance, credible interval coverage (50%, 90%, 95%)
- **Point accuracy:** MAE (posterior mean vs. true θ), bias (signed MAE)
- **Fault classification:** per-class F1, macro-F1 (all classes equally weighted)
- **Calibration:** SBC rank histograms, KS p-value, C2ST score (Kolmogorov-Smirnov test on ranks)
- **Timing:** ms per observation window (inference only, post-training)

### 5.4 Sequential degradation tracking protocol

30-day profile: 720 consecutive 2-hour windows (propylene oxide: 60-min windows, 720 total).
Each window processed independently — no temporal prior propagation. This is a deliberate
design choice: the amortised posterior already processes each window in < 20 ms, and
temporal correlations are secondary to the structural identifiability gap. Window-by-window
independence simplifies deployment in existing SCADA historians.

---

## 6. Results: Propylene oxide CSTR (~4.5 pages)

### 6.1 Training validation

**Prior predictive check:** 500 prior draws span the observed data range for all 4 channels.
**SBC:** KS p = 0.016 for both α and β — formal miscalibration. C2ST scores 0.52, 0.53
(near 0.5 baseline). Interpretation: consistent with structural β bias, not a training
deficiency (C2ST near 0.5 indicates the neural network approximates the posterior well
given the available information).

**Figure 1:** Prior predictive coverage (top) + SBC rank histograms (bottom).
Source: nb04 outputs.

### 6.2 Snapshot fault classification

All 8 scenarios, 50 replicates each. Key results:
- Macro-F1 (6 closed-loop scenarios): **0.990**
- Perfect classification: Sc1 (healthy), Sc2 (fouling), Sc3 (decay) — F1 = 1.0 each
- Combined fault (Sc4): F1 = 0.94 (boundary cases)
- Open-loop with fault (Sc6): F1 = 0.08 when CL-trained posterior applied; correct
  when OL-trained posterior used — this is the mode-mismatch demonstration

**Figure 2:** 2-D joint posteriors for all 8 scenarios with fault quadrant overlays.
Source: nb06.

**Table 7:** Per-scenario results (true θ, posterior mean, 90% CI, F1, coverage).

### 6.3 Structural identifiability analysis (core contribution)

This is the central scientific contribution of the propylene oxide section.

**6.3.1 Fisher information asymmetry**

Numerical FIM at nominal operating point (Sc1): I_αα = 850k–975k, I_ββ = 2,000–3,500.
Ratio I_αα/I_ββ = 250–500× across all tested operating points.

Channel decomposition (propylene oxide):
- I_αα: 60% from C channel, 40% from Qc; T and Tc contribute < 5%
- I_ββ: 0% from C (β does not appear in the component balance), 0% from T (zeroed
  by controller), ~100% from Tc — the single noisiest channel

**Figure 3:** Fisher information bar chart — I_αα and I_ββ with per-channel contributions.
Source: nb15.

**6.3.2 Analytical derivation**

Under perfect PI integral action (T ≈ Tsp), the steady-state Jacobian satisfies
∂T_ss/∂β ≡ 0. The temperature channel — which carries the primary heat-exchanger
signal — is structurally zeroed by the controller. β is then identified only through
the noisier Tc and Qc channels, giving I_ββ ∝ (∂Tc/∂β)² / σ²_Tc + (∂Qc/∂β)² / σ²_Qc,
which is orders of magnitude smaller than I_αα.

This is a specific quantification of the general principle (Ljung 1977; Gevers et al.
2011) for a CSTR with competing fault parameters. The derivation shows the ratio grows
with controller gain Kp — a tighter controller makes β identification harder.

**6.3.3 Embedding-net irreducibility proof**

A CNN trained on raw (120×4) time series (61,636 parameters) — bypassing the 29-D
summaries entirely — produces β̂_Sc2 = 0.621 vs 0.616 for hand-crafted features
(< 1% difference). The bias is in the physics (controller masks β), not in the features.
This is the empirical confirmation of the Cramér-Rao bound: no choice of summary
statistics or inference algorithm can recover information the data do not contain.

**Figure 4:** Side-by-side marginal posteriors — CNN embedding vs. 29-D hand-crafted
for 5 scenarios. Source: nb04b.

### 6.4 Multi-method baseline comparison

The structural β bias confirmed across all four independent methods:

| Method | β̂ (Sc2, true=0.70) | β bias | β MAE (tracking) | ms/window | Output |
|--------|-------------------|--------|-----------------|-----------|--------|
| SBI (NSF) | 0.551 | −0.149 | 0.033 | 16 | Full posterior |
| NUTS (MCMC) | 0.598 | −0.102 | 0.102 | 150,000 | Full posterior |
| EKF | 0.607 | −0.093 | 0.065 | 30 | Gaussian (μ, Σ) |
| UKF | 0.607 | −0.093 | 0.090 | 358 | Gaussian (μ, Σ) |

All four methods show the same bias direction and comparable magnitude — this is the
single most important empirical result of Section 6. The bias is structural, not a
deficiency of any particular method.

**Figure 5:** Baseline dashboard — β estimates for Sc2 across 50 replicates for all
4 methods; β bias bar chart; inference timing comparison. Source: nb16.

### 6.5 30-day sequential degradation tracking

720 consecutive 60-min windows. Linear α decay + Kern-Seaton β fouling curve.

- SBI: MAE_α = 0.004, MAE_β = 0.033, wall time 10.4 s for 720 windows
- EKF: MAE_α = 0.012, MAE_β = 0.065
- Correct fault classification in > 95% of windows (SBI); > 88% (EKF)
- Systematic β offset of −0.08 throughout — predictable and consistent

**Figure 6:** 30-day tracking — α(t) and β(t) estimates with 90% CI bands for SBI
and EKF overlay; rolling classification accuracy. Source: nb10, nb16.

**Table 8:** Phase-by-phase tracking metrics (healthy, onset, severe fouling).

---

## 7. Results: Luyben reactor-column-recycle benchmark (~5 pages)

*Section 7 will be populated after completing notebooks nb20–nb28 per
project_wu2003_sbi.md. The structure and expected findings are specified below.*

### 7.1 Training validation

**S-B is calibrated.** An 8-seed ensemble was trained (`zuko_nsf` 60/3, 0.3% noise,
`reb_per_boilup` compressed features) and evaluated with a corroborated multi-N SBC
protocol (N=200 → 400 → 800, independent RNG draws at each stage — a single SBC pass at
any one N is not trusted; see §8.4 L9). Seed 4 passed all 5 parameters at all three N,
including β_r and η_col, and was promoted as the production S-B posterior. **This
resolves the earlier η_col overconfidence concern (previously reported at SBC p=0.0001)
for the marginal calibration** — but see §7.4/§8.4 L5 for an important caveat: marginal
SBC calibration does not by itself guarantee the *joint* posterior geometry is correct at
any one specific scenario, only that rank statistics are uniform in aggregate.

**S-A is NOT calibrated — a settled negative result, not an open question.** 16 first-round
seeds, plus 24 further seeds testing PCA feature-dimension reduction (15/25 components) and
architectures both larger (80/5, 128/5 hidden/transforms) and smaller (30/2, 40/2) than the
production 60/3, all failed SBC at N=400, predominantly on η_col and ξ_reb. No architecture
or feature-space intervention recovered calibration; larger networks performed *worse* than
the original 60/3 (see §8.4 L10). Do not use S-A results for any quantitative claim — S-A is
reported as a limitation (§8.4 L10), not a resolved result.

**Figure 7:** SBC results for 5 Wu 2003 parameters, S-B only (calibrated) — rank histograms
and KS p-values at N=200/400/800 for seed 4. S-A SBC results shown separately as evidence of
the unresolved calibration failure (not as a usable posterior). Source: nb23, nb24, nb25.

### 7.2 Identifiability structure — FIM analysis (core contribution)

5×5 FIM computed via finite differences on the 66-D S-B summary statistics at the nominal
operating point. Source: nb23 §7. Figure 8 = normalised FIM heatmaps (S-B and S-A).

**7.2.1 T_r masking — shared mechanism confirmed**

∂T_r_ss/∂β_r ≡ 0 AND ∂T_r_ss/∂α ≡ 0 under Loop 1 PI control — confirmed numerically:
T_r features contribute **0.00%** to both I_αα and I_β_r. This is the same analytical
result as the PO system (nb15), transferred exactly to Wu 2003.

**7.2.2 Wu 2003 FIM structure is fundamentally different from PO**

| | PO CSTR | Wu 2003 |
|--|---------|---------|
| I_αα/I_β_r | 250–500× | **1.1×** (nearly equal) |
| Primary α channel | C (concentration, 60%) | corr_Qreb_FR (77%) |
| Primary β channel | T_c, Q_c (decoupled from C) | corr_Qreb_FR (49%) — same channel |
| (α, β_r) normalised off-diagonal | small | **+0.901** (near-degenerate) |

The PO 250× asymmetry arose because α was identified through the concentration channel C,
which β_r cannot access. **Wu 2003 has no observable concentration channel** — z_A is an
internal state not in S-B. Both α and β_r primarily excite the same physics correlations
(corr_Qreb_FR, corr_Qj_FR, corr_Rn_Vn), making them nearly indistinguishable at the nominal
operating point.

At the level of the 66-D hand-crafted summary-statistic feature set, this looks like a
**more severe** identifiability challenge than PO: not just β_r is hard to identify, but the
entire (α, β_r) subspace appears jointly confounded through shared features (confirmed, at
that representation, by a dedicated identifiability scan and by the trained, calibrated SBI
posterior itself, §7.4). **§7.4 shows this does not survive a representation-level check,
and should not be reported as a physical limit of the plant** — see below.

> **Finding (2026-07-05, final framing for this paper — see §7.4 and §8.4 L4′ for full
> detail):** re-computing this section's own FIM methodology (`FIM = J^T Σ^{-1} J`, Σ from
> real replicate noise, exactly as above) on the *raw, unaggregated* trajectory of the same
> 3 physical channels (T_r, T_j, F_R_norm) collapses the (α, β_r) normalised off-diagonal
> from ≈0.6–0.85 (matching the +0.901 reported here) to ≈0.00, reproducibly across multiple
> noise seeds and at both the nominal point and W11. A negative control rules out a cheap
> fix: a modestly finer time-resolution summary (6 sub-windows, mean+std per channel, 108-D)
> shows *no* improvement (off-diagonal ≈0.6–0.86, indistinguishable from the 66-D baseline)
> — the lost information is in fine-grained transient shape, not coarser level/spread
> statistics. An EKF given full raw-trajectory access independently corroborates the FIM
> result, resolving (α, β_r) to ~1–2% at points this section's 66-D-feature analysis calls
> non-identifiable. **We report this as strong, convergent, methodologically rigorous
> evidence that the (α, β_r) "banana" is very likely an artifact of the summary-statistic
> representation, not of the plant physics — the paper's actual headline finding for this
> system (§7.4, §8.1).** A trained, calibrated CNN-embedding SBI posterior on a
> raw-trajectory-aware representation was subsequently built and tested (§7.4.1); it did
> **not** narrow the correlation (0.993 vs. 0.994, robust across 19 tested conditions) —
> confirming that recoverability in principle (FIM, EKF) does not guarantee recoverability
> by this specific amortized estimator at a modest training budget (§8.4 L9, L11).

**7.2.3 (α, η_col) does NOT show a persistent joint degeneracy once the full feature set is used**

Local FIM at nominal: normalised (α, η_col) = **−0.142** (slightly negative — no coupling
at nominal). This pair was initially investigated as a headline candidate because both
faults drive the recycle flow (F_R) upward via the snowball effect at degraded values
(α≈0.75, η_col≈0.80), and a *restricted* identifiability scan using only F_R-derived
summary features does reproduce an extended degenerate ridge in (α, η_col) space —
reproducing the classic "banana" shape. **However**, a scan combining F_R with the
column's own T_reb-derived features (both already part of the standard S-B measurement
set) collapses this ridge to a much more localized region around the true value: the
two individual channels' ambiguities run at different angles in (α, η_col) space, so
combining them substantially resolves the pair using existing S-B instrumentation alone.
This is corroborated by the calibrated S-B posterior itself, which shows a narrow η_col
credible interval (90% CI width ≈ 0.01–0.02, essentially fixing η_col) with near-zero
correlation to α (|corr| < 0.15) across several tested operating points — the opposite of
banana-shaped. **Conclusion: (α, η_col) is not reported as a genuine joint
non-identifiability in this paper** (see §8.4 L4 for the retraction and methodological
note on why marginal SBC alone could not have caught this).

**7.2.4 z_A0 is the most locally identifiable parameter**

Largest FIM diagonal: I_z_A0 = 2.15×10¹⁴ (vs. I_αα = 2.22×10¹³). Feed purity affects
the entire reactor steady state through the inlet composition — a decoupled signal not shared
with other parameters.

**7.2.5 S-A's local information gain for η_col is not the relevant comparison**

S-A/S-B I_η_col ratio = 1.32× at nominal (x_D measurement provides some additional locally
decoupled η_col signal) — this numeric ratio is retained for completeness, but it is not
evidence for or against a S-A/S-B *banana resolution* claim, since (α, η_col) is not treated
as a genuine banana in this paper (§7.2.3), and (α, β_r) is now also not treated as one at
the 66-D summary-statistic level (§7.4, §8.1) — it is very likely a representation
artifact. Both parameters are reactor-side only regardless; S-A's column composition
analyser does not measure either reactor channel and would not be expected to help with
either the (now-retracted) apparent confound or a genuine scalar masking effect, were one
to be found (untested; flagged as future work, §8.4).

**Figure 8:** 5×5 normalised FIM heatmaps (S-B left, S-A right). Source: nb23 §7.
Key features: β_r and α both near-zero T_r contribution; high (α, β_r) off-diagonal (+0.901)
under the 66-D summary-statistic representation, but collapsing to ≈0.00 under the raw
trajectory (§7.4) — very likely a representation artifact, not a genuine joint confound;
η_col off-diagonal near-zero at nominal in both
structures (−0.142 S-B, +0.195 S-A) — consistent with no persistent (α, η_col) degeneracy.

### 7.3 Snapshot fault classification (14 scenarios, posterior-mass approach)

14 scenarios (W8, W14 removed — see §8.4), 30 replicates each. Classification uses
posterior mass in fault-unit regions (same approach as PO §6.2 / nb11), not thresholded
posterior mode. Fault units: healthy, reactor (α↓ or β_r↓), column (η_col↓ or ξ_reb↓),
feed (z_A0↓), compound (multiple degraded).

Key results (source: `nb31_wu2003_fault_classification.ipynb`, executed 2026-07-05, against
the calibrated seed-4 S-B posterior; S-A intentionally excluded per §8.4 L10). 14 closed-loop
scenarios x 30 replicates x 200 posterior draws, same 0.85-relative-threshold posterior-mass
convention as the PO system (nb11/§4.5), applied uniformly across all 5 parameters (one-sided
for z_A0_eff since every feed fault in this taxonomy is lean, never rich). Ground-truth
fault-unit labels come from `RecycleScenarioConfig.fault_unit()`, which was **corrected this
session** (2026-07-05) from a scenario-*name* substring-matching rule (which silently
mislabelled two scenarios, below) to a pure parameter-threshold rule matching this
classifier's own counting logic (and the pre-existing pattern in `cstr_sbi.luyben`).

**Overall: 87.4% accuracy, macro-F1 = 0.694** (per-class F1: healthy 0.667, reactor 0.948,
column 1.000, feed 0.000, multi 0.854).

- **Reactor faults (W2–W6, W11, W15): near-perfect** (F1 = 0.95, every replicate classifies
  correctly). **Compound reactor fault (W11, α=0.80 & β_r=0.80): classifies correctly as
  `reactor` in 30/30 replicates**, despite the underlying (α, β_r) posterior correlation of
  0.998 (§7.4) — because α and β_r map to the *same* fault unit, the representation artifact
  corrupts *parameter-level attribution* within the reactor unit but not *unit-level
  detection*. This is the paper's clearest demonstration that §7.4's diagnostic finding,
  while real, does not automatically propagate to every downstream task.
- **Column fault (W7, W9): perfect** (F1 = 1.00) — confirms η_col/ξ_reb are well-identified
  under the calibrated S-B posterior at the classification level too.
- **Compound fault (W12, α=0.75 & η_col=0.80): classifies correctly as compound in 30/30
  replicates, with zero reactor/column leakage** — confirms §7.2.3's retraction exactly as
  predicted: since (α, η_col) is not a genuine joint degeneracy, S-B alone suffices.
- **A `fault_unit()` labelling bug was found and fixed.** The original name-matching rule
  mislabelled W15 (`multi`, via a "snowball" keyword, even though only α — not η_col=0.90 —
  crosses the 15%-deviation threshold) and W13 (`reactor`, via a "cat_" keyword, even though
  it has two genuinely degraded units: α=0.80 **and** z_A0_eff=0.80). After the fix: **W15
  now classifies correctly in 30/30 replicates** (its corrected label matches what the
  classifier already predicted); **W13's corrected `multi` label is classified correctly in
  only 7/30 replicates (23%)** — a real weakness, not a labelling artifact, explained below.
- **Feed fault (W10) and compound fault W13 are the weak points (feed F1 = 0.00; `multi`
  F1 = 0.85, dragged down by W13), and the mechanism is a third representation artifact, not
  a detection-power problem.** An initial diagnosis (posterior scatter at a single 2h window
  comparable to the fault size, fixable by pooling evidence across windows) **is retracted**:
  per-replicate z_A0_eff posterior estimates are actually precise (std ≈ 0.01), so pooling
  would not move the mean. The real mechanism, confirmed via this paper's own noise-calibrated
  FIM methodology (§7.2.2/nb29b §4): a genuine (α, z_A0_eff) near-degeneracy under
  `compute_summaries` that is weak at the nominal point (off-diagonal ≈ -0.12) but jumps to
  ≈ -0.89 — the same magnitude as the paper's own (α, β_r) headline number — the moment α is
  degraded (checked at the W2 truth), and an analogous (β_r, z_A0_eff) coupling at the W5
  truth (≈ -0.48). **Both collapse to ≈ -0.07 under the raw trajectory** — the identical
  signature as the two already-documented (α, β_r) and (α, η_col) artifacts. z_A0_eff has no
  analytical dependence on α or β_r in the plant's governing equations; the summary-statistic
  representation conflates "a reactor fault is present" with "z_A0_eff has drifted" because
  both perturbations move overlapping recycle/conversion-related features. Added to the
  limitations table as **L4″** (§8.4) — a third instance of this paper's central
  methodological finding (§8.1), not a new mechanism.

**Figure 9:** Marginal posterior summaries — posterior mean ± 90% CI for all 5 parameters
across 14 scenarios, side-by-side for S-A (left) and S-B (right). Source: nb24, nb25.

**Table 10:** Per-scenario classification results (posterior-mass F1, 90% CI coverage),
S-B only — S-A shown as a calibration-failure limitation, not a comparison (§8.4 L10). η_col
results include the SBC calibration history (§8.4 L5). Source:
`results/31_fault_classification_metrics.csv`, `results/31_classification_summary.json`.

### 7.4 Headline: two apparent "banana" degeneracies, both artifacts of the observation representation

**This section was substantially revised on 2026-07-03, then again on 2026-07-05 — this is
now the paper's final framing for this system, not a placeholder pending further work.**
This project investigated two candidate joint (2-parameter) degeneracies in the recycle
plant, in sequence, using progressively more rigorous checks. **Both failed to survive
scrutiny — every joint degeneracy this paper investigated turned out to be an artifact of
how the observation was represented, not a property of the plant.** This is the headline
finding for the recycle system: not "here is a banana," but "here is why apparent bananas
in closed-loop SBI need to be checked before they are reported, and how to check them."
The one identifiability limitation in this plant that *is* genuine and representation-
independent — reactor thermal faults masked by Loop-1 integral control — is a *scalar*
reduction in information, structurally identical to the PO CSTR's β masking (§6.3, §8.1),
not a manifold constraint between two different parameters.

**Candidate 1 — (α, η_col) at W12 (α=0.75, η_col=0.80): retracted, a restricted-channel
artifact.** The original headline claimed a banana here, resolved by S-A (α CI width
0.240→0.059, −75%). Investigation found:
1. The apparent banana was reproduced only when the identifiability scan or the SBI
   training features were restricted to F_R-derived information alone — a genuine but
   narrow artifact of that restriction, not of the full S-B measurement set. Combining F_R
   with the column's own T_reb-derived features (both already standard S-B instrumentation)
   collapses the ridge to a localized region.
2. The calibrated seed-4 S-B posterior shows η_col essentially pinned (90% CI width
   0.01–0.02) with near-zero correlation to α (|corr| < 0.15) at every tested operating
   point — the opposite of a banana.
3. S-A calibration was never achieved (0 of 40 total seeds tested across two sessions
   passed SBC — see §8.4 L10), so the previously-reported "75% CI width reduction" was an
   artifact of an overconfident, uncalibrated posterior and cannot be re-derived from a
   working S-A posterior at all.
**The −75%/−80% numbers must not be cited anywhere in this paper.** The fix that worked:
*combine a second already-available channel* — no new instrumentation, no representation
change.

**Candidate 2 — (α, β_r) at W11 (α=0.80, β_r=0.80): appeared far more robust, retracted
anyway, for a deeper reason.** Unlike Candidate 1, this pair survived every check performed
*at the level of the 66-D hand-crafted summary-statistic representation*:
- Loop 1 holds T_r at setpoint → masks both α and β_r via the temperature channel
  (∂T_r_ss/∂α ≡ ∂T_r_ss/∂β_r ≡ 0, confirmed numerically in §7.2.1), removing the channel
  that rescued α (via concentration) in the PO system.
- FIM off-diagonal +0.901 at nominal (§7.2.2) — the strongest coupling in the whole 5×5
  matrix.
- Trained, calibrated seed-4 S-B posterior at the W11 truth: posterior mean α = 0.724 (true
  0.80, 90% CI width 0.208), posterior mean β_r = 0.694 (true 0.80, 90% CI width 0.295),
  **correlation(α, β_r) = +0.998**. For comparison, η_col at this scenario has 90% CI width
  ≈ 0.006 and correlation with either α or β_r below 0.24 — confirming η_col is not
  entangled and the confound is specific to (α, β_r).
- A profile-distance identifiability scan shows the achievable best fit stays flat and low
  across a wide β_r range at fixed α (and vice versa) — unlike Candidate 1, this survives
  *combining the entire 66-D summary-feature set*, not just one restricted channel. At the
  time, this was read as ruling out a Candidate-1-style artifact.
- An executed EKF at W11 (`nb26`) achieves **0% coverage on both α and β_r** — taken alone,
  this looked like the confirming result the section needed.

**Why Candidate 2 still failed, one level deeper: it survives channel combination but not
representation richness.** All of the checks above operate on the same 66-D whole-window
summary statistics (`compute_summaries`) — none test whether the *degree of temporal
aggregation itself* is discarding information, as opposed to the *choice of channels*. Two
independent follow-up checks did test exactly that:
1. **EKF tuning.** `nb26`'s EKF uses a very tight initial parameter covariance
   (`P[6:,6:]≈1e-4`) that prevents it from moving far from its nominal initial guess within
   one 2h window, regardless of what the data says — this alone, not the plant physics,
   produces the 0% coverage above. Substituting a more diffuse covariance (`P[6,6]=0.05,
   P[7,7]=0.02` — already used elsewhere in this project) on the *identical* noisy W11
   window converges to within ~0.3% of the true (α, β_r), reproducibly across 15+ noise
   seeds and 4 additional (α, β_r) points spanning the scan grid.
2. **Raw-trajectory FIM.** Re-deriving this section's own §7.2.2 methodology
   (`FIM = J^T Σ^{-1} J`, real noise-driven Σ) on the *raw, unaggregated* trajectory of the
   same 3 observed channels (T_r, T_j, F_R_norm) — instead of the 66-D `compute_summaries`
   feature set — collapses the (α, β_r) normalised off-diagonal from ≈0.6–0.85 (matching
   this section's +0.901 number) to ≈0.00, reproducibly across noise seeds and at both the
   nominal point and W11 (I_αα/I_β_r stays a mild ≈1.1–1.4×, confirming this is a
   *collinearity* problem, not one parameter being individually weak). A negative control
   rules out a cheap fix: a modestly finer time-resolution summary (6 sub-windows,
   mean+std/channel, 108-D) shows **no** improvement (off-diagonal ≈0.6–0.86, indistinguishable
   from the 66-D baseline) — the lost information is fine-grained transient *shape*, not
   coarser-grained level/spread, so no incremental hand-crafted feature addition (of the
   kind that fixed the unrelated η_col SBC issue in §7.1) is expected to fix this.

**Both independent methods point the same direction, and neither required a single new
sensor** — only less-aggregated use of the three channels already being measured. **We
report this as strong, convergent evidence that (α, β_r), like (α, η_col), is an artifact
of the observation representation rather than a physical property of the plant — but at
one level deeper: a lossy-aggregation artifact rather than a restricted-channel artifact.**

#### 7.4.1 Testing the decisive fix: a CNN-embedding SBI posterior (nb32)

The two checks above (raw-trajectory FIM, re-tuned EKF) establish that the information
needed to disentangle (α, β_r) exists in the raw signal and that a privileged, exact-model
estimator can exploit it. Neither is a trained, amortized SBI posterior. This paper's own
irreducibility-test precedent (§6.3.3/`nb04b`, propylene oxide) is to close that gap
directly: train an embedding net on raw trajectories and check whether the confound
survives. We ran this experiment for the recycle plant (`notebooks/32_wu2003_cnn_embedding.ipynb`):
an 8-seed CNN-embedding SNPE ensemble (62,686-parameter CNN, 2 conv layers, 30-D learned
embedding, 15,000 training simulations) trained directly on raw `(120, 9)` S-B trajectories,
with 1/8 seeds passing marginal SBC (seed 6, min KS p = 0.137) — the same order of
seed-lottery difficulty as the hand-crafted pipeline's own 8-seed search (§8.4 L9).

**Result: the ridge does not narrow.** At W11 the trained CNN posterior gives
correlation(α, β_r) = 0.993, statistically indistinguishable from the hand-crafted
posterior's 0.994. A grid-check (15 noise seeds at W11 plus 4 additional (α, β_r) truth
points spanning the identifiability-scan grid) confirms this is not a one-point artifact:
correlation stays in 0.987–0.996 (std ≤ 0.004) across all 19 tested conditions — a robust
property of the trained network, not seed noise.

**This adds precision to Finding 9/L4′, it does not overturn it.** Three different
recoverability questions now have three different, independently-established answers: (i)
*information-theoretic* recoverability (raw-trajectory FIM): yes — the off-diagonal
collapses from ≈0.85 to ≈0.00; (ii) *privileged-model-based* recoverability (a re-tuned EKF
with exact ODE access to the raw trajectory): yes — recovers both parameters to ~1–2%
accuracy; (iii) *amortized, learned-representation* recoverability (this CNN-embedding SBI
posterior): no, at this architecture and training budget. Marginal SBC does not test joint
posterior shape (the same caveat already documented for L4/L5) — passing it is not evidence
the (α, β_r) joint structure was learned correctly, and this result confirms it was not.

**The same trained network does independently confirm a *different* candidate's
resolution.** At W10/W13, correlation(α, z_A0_eff) collapses from ≈0.99 (hand-crafted) to
≈−0.004/−0.003 (CNN) — a second, non-linearized method now agreeing with the FIM diagnosis
for the (α/β_r, z_A0_eff) pair (§7.3, L4″). The same architecture and training run resolves
one candidate and not the other — itself informative about which class of confound is easy
versus hard for a modest amortized estimator to fix in practice. Overall, CNN-based fault
classification is worse than hand-crafted (macro-F1 0.270 vs. 0.679; accuracy 0.619 vs.
0.869) and 30-day tracking is mixed (α MAE worsens, 0.048→0.067; β_r MAE improves,
0.198→0.139 — still far short of EKF's 0.002/0.004) — see §7.5.

**Practical reading: fixing a diagnosed representation artifact does not automatically fix
the downstream task, in either direction.** `nb31` (§7.3) already showed an *unresolved*
artifact, (α, β_r), does not hurt unit-level fault classification at W11 (both parameters
map to the same `reactor` label). This experiment shows the mirror image: even where the
CNN *does* resolve the (α, z_A0_eff) correlation, downstream feed-fault classification is
still wrong (F1 = 0.00, matching hand-crafted) — for a different reason now (the CNN's own
systematic α bias, not the correlation). The two — correlation and downstream task
performance — must be checked separately.

**EKF result at W12/W15 (unrelated to either banana candidate, numbers stand):** at W12
(α=0.75, η_col=0.80), EKF α mean = 1.185 ± 0.084 (true = 0.75), **3% empirical coverage**
— the EKF never moves away from nominal under strong catalyst-decay snowball conditions.
Near the snowball tipping point (W15, α=0.58), EKF achieves 3% coverage (mean = 0.990, true
= 0.58). These are genuine, representation-independent demonstrations of EKF failure under
strong nonlinear/snowball conditions — a different failure mode from either banana
investigation above, and not affected by this section's reframing.

**Figure 10 (redefined as the paper's diagnostic figure, replacing the retired banana
scatter):** (a) FIM off-diagonal comparison bar chart — `compute_summaries` (66-D),
`subwindow` (108-D), `raw_trajectory` (360-D), at nominal and W11, showing the collapse
from ≈0.85 to ≈0.00 and the negative `subwindow` control; (b) the same EKF tuning
(tight vs. diffuse covariance) on the identical W11 window, showing the 0%-coverage result
flip to ~0.3% error; (c) for contrast, the Candidate-1 (α, η_col) restricted-vs-combined
channel scan from §7.2.3, establishing that this paper found and correctly diagnosed two
*different* mechanisms producing the same symptom. Source: `nb26`, `nb29b` §§3-4, `nb27` §9.

### 7.5 EKF baseline comparison

Augmented EKF: 9-state vector [z_A, T_r, T_j, I_T, R_state, V_state, α, β_r, η_col].
Pure-numpy implementation with precomputed QSS column lookup table (avoids JAX OOM in
the 720-step sequential loop). Observations used: T_r, T_j, F_R_norm.

Actual findings (W12/W15, executed under the old framing — see §7.4 retraction for why
these are no longer described as "banana" results, though the EKF numbers themselves stand):
- Both methods show structural β_r and α bias (~0.10 downward for α; same mechanism as PO β)
- **EKF completely fails** under strong catalyst-decay/snowball conditions: 3% α coverage
  at W12, mean estimate far from truth
- EKF near tipping point (W15): 3% coverage, mean 0.990 vs true 0.58
- SBI W15: 100% coverage, CI [0.436, 0.608] correctly contains truth
- EKF at W11: 0% coverage on both α and β_r (executed, `nb26`) — **explained in §7.4 as a
  tuning artifact of that specific EKF configuration (tight initial parameter covariance),
  not evidence for a physical banana** — the identical architecture with a more diffuse
  covariance recovers ~0.3% error on the same data.

**30-day sequential tracking (executed, `nb27`):** α/β_r MAE, bias, 90% coverage —

| Param | Method | MAE | Bias | 90% Coverage |
|---|---|---|---|---|
| α | SBI | 0.048 | −0.048 | 0.15 |
| α | EKF | 0.002 | +0.001 | 1.00 |
| β_r | SBI | 0.198 | −0.198 | 0.12 |
| β_r | EKF | 0.004 | +0.004 | 0.96 |

**Correct framing for this table (final, per §7.4): "a differently-tuned model-based filter
with raw-trajectory access outperforms an amortised posterior trained on compressed summary
statistics"** — not "EKF beats SBI at the banana," and not "SBI structurally cannot compete
with recursive filtering." It is the opposite of every other SBI-vs-EKF comparison in this
paper, and `nb27` §9 traced the reversal to the EKF's raw-trajectory access plus tuning, not
to sequential tracking resolving a genuine degeneracy. Read together with §7.4, this table
is itself evidence *for* the representation-artifact finding, not a separate result: it
shows a raw-trajectory-based estimator succeeding exactly where the summary-statistic-based
one fails, at the same scenario the FIM diagnostic flags.

**The CNN-embedding SBI posterior (§7.4.1) does not close this gap.** Applied to the
identical 30-day trajectory, it *worsens* α tracking (MAE 0.048→0.067) and only partially
improves β_r tracking (MAE 0.198→0.139) — both remain over an order of magnitude worse than
the EKF's 0.002/0.004. This reinforces rather than contradicts the raw-trajectory-access
explanation: the EKF's advantage comes from an exact, privileged ODE model consuming the
unaggregated signal, not merely from "using raw data" in any form — an amortized network
trained on the same raw signal, at a modest architecture and budget, does not reproduce it.

**Figure 11:** SBI vs. EKF 30-day tracking for α, β_r with CI bands. Source: nb27 (executed;
caption must include the caveat above, not just the raw table).

**Table 10:** Wu 2003 SBI vs. EKF — bias, MAE, coverage; snapshot comparison for W11, W12,
W15 (W11 executed — see above; caption needs the same caveat).

### 7.6 NUTS infeasibility for Wu 2003

NUTS was not run on the Wu 2003 system. Even for the 2-D propylene oxide system, NUTS
convergence was unreliable in the presence of structural identifiability limits — the
posterior geometry (banana manifold, wide β bias) creates mixing difficulties that NUTS
does not resolve. For 5-D parameter spaces with recycle-coupled non-Gaussian posteriors,
NUTS would face the same qualitative failure as EKF, compounded by exponentially slower
mixing. MCMC is therefore neither fast enough nor reliable enough for this problem.
SBI is the only method that is simultaneously: (a) fast (< 20 ms/window), (b) correct in
posterior geometry given whatever the observation representation actually supports — it
reproduces a banana at (α, β_r) under the 66-D summary-statistic feature set precisely
*because* that representation is genuinely collinear for those two parameters (§7.4), which
is a form of correctness, not a failure, even though the representation itself turned out
to be avoidably lossy — and (c) calibrated for the well-identified parameters (η_col, ξ_reb,
z_A0).

**Table 11:** Feasibility comparison — SBI vs. EKF vs. NUTS (infeasible + unreliable).

*Note: §7.7 (model mismatch robustness) is deferred to future work — see §8.4.*

---

## 8. Discussion (~2 pages)

### 8.1 Structural identifiability across scales: one genuine mechanism, two artifactual ones

**Two systems, one confirmed mechanism, and a cautionary tale about how easily a second,
spurious mechanism can look confirmed too.**

The propylene oxide CSTR illustrates the fundamental *single-loop* mechanism: the reactor
temperature PI controller zeros ∂T_ss/∂β_r, removing the highest-SNR channel from β_r's
information budget and giving I_β_r/I_α = 1/250–500. This is a **scalar reduction**: one
parameter is harder to identify, but all parameters are identifiable in principle. Critically,
this is the one identifiability claim in the whole paper that was checked against a
raw-signal representation (the §6.3.3 CNN-embedding irreducibility test) *before* being
reported as physical, and it survived that check — confirming it is a genuine property of
the plant and controller, not an artifact of the 29-D hand-crafted feature set.

The Wu 2003 CSTR-column-recycle plant reveals the **same scalar mechanism, transferring
exactly**: Loop 1 zeros ∂T_r_ss/∂α and ∂T_r_ss/∂β_r simultaneously (§7.2.1), and this holds
regardless of observation representation — it is an exact analytical statement about the
plant's steady state, not a summary-statistic effect. One quantitative difference from PO
is worth foregrounding: because Wu 2003 gives *neither* α nor β_r an independent rescue
channel (unlike PO, where α keeps its concentration channel), masking reduces both
parameters' information roughly *symmetrically* (I_αα/I_β_r ≈ 1.1–1.4×, confirmed under
both the summary-statistic and the raw representation) — nowhere near PO's 250–500× — rather
than singling one parameter out.

**What Wu 2003 does *not* reveal, on rigorous re-examination, is a genuine joint (manifold)
degeneracy between two different parameters.** Two candidates were investigated, in
sequence, with progressively stricter checks — and **both failed, for two different
reasons**:
1. **(α, η_col) at W12** — a restricted-*channel* artifact. A banana is genuinely visible if
   one restricts attention to the recycle flow (F_R) alone, but combining it with the
   column's own T_reb signal (already standard S-B instrumentation) collapses the ridge.
   Retracted (§7.2.3, §8.4 L4).
2. **(α, β_r) at W11** — a lossy-*aggregation* artifact, one level deeper. This pair
   survived combination of the *entire* 66-D summary-statistic feature set (unlike
   Candidate 1), which at the time looked like confirmation it was physical. It is not: the
   same Fisher-information methodology applied to the raw, unaggregated trajectory of the
   same three already-observed channels collapses the coupling from ≈0.85 to ≈0.00, and a
   negative control (modestly finer time-resolution summary statistics) rules out an easy
   fix — the lost information is in fine-grained transient shape. Retracted (§7.4, §8.4
   L4′). **A trained, calibrated CNN-embedding SBI posterior confirming this at the level
   of an amortized estimator was subsequently attempted (§7.4.1, §8.4 L11) and did *not*
   narrow the correlation** (0.993 vs. 0.994, robust across 19 tested conditions) —
   distinguishing information-theoretic recoverability (established here, by FIM and EKF)
   from recoverability by a modest-budget trained estimator (not established). This is a
   training-practicality gap, not a retraction of the artifact diagnosis; see §8.4 L9/L10
   for the seed-instability context that makes this a real, not merely formal, limitation.

**A further nuance, discovered via a third candidate found through fault classification
rather than a direct scan: not every representation artifact has the same downstream
cost.** (α/β_r, z_A0_eff) (§7.3, §8.4 L4″) is mechanistically identical to Candidate 2 —
same FIM-collapse signature, raw trajectory to ≈0.00 — yet its classification consequence
is severe (feed-fault F1 = 0.00) where Candidate 2's is benign (W11 still classifies
correctly, 29/30). The difference is not the confound's strength but the fault taxonomy:
α and β_r share a fault unit (reactor), so a confound between them corrupts only
parameter-level attribution; z_A0_eff has its own unit (feed), so the identical *kind* of
confound corrupts unit-level detection directly. **Whether a joint representation artifact
is classification-benign or classification-severe therefore depends on where the confused
parameters sit in the fault taxonomy, not on the raw Fisher-information magnitude of the
confound** — a distinction a practitioner designing a classification scheme, not just
diagnosing an identifiability limit, needs to make.

**The general lesson, stated as this paper's central methodological finding:** in a
closed-loop, multi-unit system, a *scalar* identifiability reduction (one parameter loses
information because its highest-SNR channel is masked) is a robust, physical,
representation-independent effect that survives arbitrarily rich observation access — we
confirmed this twice, in two different systems, using an irreducibility test against a raw
signal. A *manifold* degeneracy (two different parameters' effects appear collinear) is, in
this study, **zero-for-three** at surviving the same kind of check. The third instance
((α/β_r, z_A0_eff), §7.3/§8.4 L4″) was found by a different route than the first two — not
by a direct identifiability scan, but as the explanation for an unexpected fault-
classification failure (§7.3) — and re-derived the same FIM off-diagonal collapse
(≈−0.89/−0.48 under `compute_summaries`, both degraded points, to ≈−0.07 under the raw
trajectory) on the first check. **The three ways this study's candidates failed (channel
restriction, aggregation, and now a downstream-task symptom) are different enough that a
practitioner checking only for the first kind (e.g. "did I combine all my sensors?") could
easily miss the others.** We recommend that any joint non-identifiability surfaced by a
compressed-feature SBI or FIM analysis **— or any unexplained downstream-task failure that
correlates with a specific parameter pair —** in a closed-loop system be checked against a
raw-trajectory (or otherwise minimally-aggregated) Fisher-information test before being
reported as physical or attributed to an unrelated cause (e.g. "detection power") — exactly
the check that overturned Candidate 2 here and explained L4″, and the natural
generalisation of the irreducibility-test methodology this paper already uses for scalar
claims (§6.3.3) to joint ones.

**Revised design guidance.** The original sensor-placement conclusion this section would
have drawn — "install a reactor-side concentration or heat-duty analyser to resolve
(α, β_r)" — does not survive this reframing: if the confound is a representation artifact,
new instrumentation cannot fix it, because the missing information was never absent from
the sensors, only from how their signal was compressed. **The higher-value, lower-cost
intervention this plant's diagnosis calls for is feature engineering (raw-trajectory-aware
summary statistics or an embedding net for the reactor-side channels), not new sensors** —
a materially different, and for a plant operator considerably cheaper, recommendation than
the one the original headline would have produced. Measurements of controller output
signals (Q_c, Q_reb) remain essential for the *scalar* masking problem, which does require
those specific channels — that guidance is unaffected by this reframing.

### 8.2 When EKF fails, when SBI wins, and when the difference is really about the feature set

EKF is an excellent industrial baseline when the posterior is approximately Gaussian —
which holds for the propylene oxide system near the nominal operating point (all four
methods show similar bias and comparable uncertainty). This paper demonstrates three EKF
failure/success modes; only one of them is best explained by "EKF cannot represent a
non-Gaussian posterior":

1. **A tuned-EKF pitfall at (α, β_r), W11 — not a Gaussian-collapse failure.** The
   as-deployed augmented EKF achieves 0% coverage on both α and β_r at W11 (§7.4). The
   originally-intended reading — "the Gaussian assumption collapses the genuine (α, β_r)
   manifold to an ellipse" — does not hold up: §7.4 shows (α, β_r) is very likely not a
   genuine manifold at all, and the identical EKF architecture with a more diffuse initial
   parameter covariance converges to within ~0.3% of truth on the *same* data. **The
   practical lesson is still real and still worth reporting, just different**: an EKF's
   default/naive tuning can produce a confidently-wrong 0%-coverage result that looks
   exactly like "the method has hit a fundamental non-Gaussian-posterior wall," when the
   actual cause is a tuning choice interacting with a lossy observation representation. A
   practitioner who only checks "does my EKF converge" (not "would a differently-tuned or
   differently-fed EKF also converge") risks the same misdiagnosis this project made in an
   earlier draft of this section.

2. **Genuine non-Gaussian tracking under strong recycle nonlinearity** (W12 α=0.75; W15
   α=0.58) — the paper's actual demonstrated EKF-fails/SBI-wins case. The Jacobian of the
   recycle dynamics changes rapidly as α departs from nominal; the EKF linearises around
   its current estimate (~nominal, α≈1.0) — in entirely the wrong dynamical regime.
   **Empirical result: EKF achieves 3% coverage at both W12 (mean = 1.185, true = 0.75) and
   W15 (mean = 0.990, true = 0.58). SBI W15: 100% coverage, CI [0.436, 0.608].** SBI was
   trained across the full prior including near-tipping samples and correctly widens its
   uncertainty in this regime. This failure mode is driven by snowball nonlinearity, not by
   any joint parameter confound (that attribution is retracted, §7.4), and is unaffected by
   this paper's reframing — it stands as the clean, representation-independent
   demonstration of EKF's Gaussian-linearisation limits.

3. **A raw-trajectory-fed, correctly-tuned EKF outperforming SBI** (30-day tracking, §7.5)
   — the mirror image of item 1, and further evidence for §7.4's diagnosis rather than a
   counter-example to it: given the same three raw channels SBI's summary statistics
   compress, and a covariance that does not artificially over-constrain it, the EKF
   recovers (α, β_r) more precisely than the amortised posterior. This is not "EKF beats
   SBI" in general — it is "a less-compressed representation beats a more-compressed one,"
   and it happens to be the EKF, not SBI, that had raw-trajectory access in this
   comparison. **Giving SBI the same raw-trajectory access (the CNN-embedding posterior,
   §7.4.1) only partially closes the gap** — β_r tracking improves (MAE 0.198→0.139) but α
   tracking worsens (0.048→0.067), both remaining far short of the EKF — showing the EKF's
   advantage is not raw-trajectory access alone but also its privileged, exact-model (not
   merely raw-signal) treatment of that trajectory, which no amortized estimator receives.

**The recommendation, revised:** SBI remains the right default for real-time, calibrated,
non-Gaussian-aware fault diagnosis in this plant class, and is the only method demonstrated
to degrade gracefully near recycle tipping points (item 2). But **this paper's stronger,
more general recommendation is upstream of the EKF-vs-SBI choice**: before concluding that
*either* method has hit a genuine non-Gaussian/joint-confound wall, check whether a
richer, less-aggregated observation representation changes the answer (§8.1) — an EKF
tuning artifact and an SBI summary-statistic artifact can both produce the same "0%
coverage" or "posterior looks like a banana" symptom, for reasons that have nothing to do
with the plant's physics.

### 8.3 Persistent excitation and open-loop recalibration

Classical closed-loop identification requires the reference signal to be persistently
exciting (Forssell & Ljung 1999). At a fixed setpoint, Φ_r(ω) = 0 — the data are
theoretically uninformative for the parameters masked by integral control. The fact
that SBI still achieves practical fault classification reflects its exploitation of
transient dynamics (overshoots, settling oscillations) that carry implicit parametric
excitation beyond the classical frequency-domain criterion.

Practical implication: scheduled open-loop excitation windows (5–10 min at shift
changes) would provide unmasked estimates for β, β_r, and β_s for calibration.
The bias-correction strategy: report the expected structural offset as a systematic
calibration constant, derived from the Fisher information analysis.

### 8.4 Limitations

| # | Limitation | Scope | Mitigation |
|---|---|---|---|
| L1 | Synthetic data only | Both systems | Real data as future work; model mismatch study deferred |
| L2 | β bias −0.08 to −0.15 | Propylene oxide | Quantified; predictable; recalibration via OL excitation |
| L3 | α bias ~0.10 downward under S-B | Wu 2003 | Same structural mechanism as L2; reported in §7.3 |
| L4 | **RETRACTED (2026-07-03): (α, η_col) is not a genuine partial non-identifiability** | Wu 2003 S-B | Originally reported as a banana confirmed by F_R iso-contours. Retracted: an identifiability scan combining F_R with T_reb (both standard S-B channels) shows the degeneracy collapses to a localized region, not an extended ridge; the calibrated seed-4 posterior shows η_col essentially pinned (90% CI ≈0.01–0.02) with near-zero correlation to α at every tested point. The F_R-only restriction that originally suggested a banana is not representative of the full S-B measurement set. The next candidate investigated, (α, β_r), is *also* now retracted for a related but distinct reason — see L4′ below; this plant's only confirmed genuine identifiability limitation is the scalar β_r/α masking discussed in §7.2/§8.1, not a joint confound. **Methodological note:** this was not caught by the marginal SBC pass for seed 4, because marginal SBC (rank uniformity averaged over the whole prior) does not test whether the *joint* posterior at any one specific scenario has the correct shape — a necessary-but-not-sufficient distinction worth flagging for the SBI community generally. |
| L4′ | **RETRACTED (2026-07-05, final): (α, β_r) banana is very likely a summary-statistic aggregation artifact, like L4 above but one level deeper** | Wu 2003, both S-A and S-B | Originally reported as confirmed by (a) FIM off-diagonal +0.901 at nominal (§7.2.2), (b) an identifiability scan showing the degeneracy survives combination of the full S-B *summary-statistic* feature set, (c) the calibrated seed-4 posterior showing correlation(α, β_r) = 0.998 at W11, (d) an executed EKF at W11 (0% coverage). **Two independent counter-results, both using this paper's own established methodology**: (i) the W11 EKF failure reproduces exactly if its tuning is copied, but converges to ~0.3% error if a more diffuse (already-used-elsewhere-in-this-project) covariance is substituted on the identical data, robust across 15+ seeds and 4 additional grid points; (ii) re-deriving §7.2.2's own FIM methodology on the *raw, unaggregated* trajectory of the same 3 observed channels (not the 66-D summary statistics) collapses the normalised off-diagonal from ≈0.6–0.85 to ≈0.00, reproducibly across seeds and at both nominal and W11 (I_αα/I_β_r stays ≈1.1–1.4× throughout, confirming this is a collinearity problem, not an individual-parameter weakness). A negative control (a 108-D sub-window summary — 6× finer time resolution, same channels) shows **no** improvement, ruling out an easy hand-crafted-feature fix. **This mirrors L4's retraction pattern** (a restricted/compressed view mistaken for a physical limit) but one step further removed — a lossy-*aggregation* artifact rather than a restricted-*channel* artifact, requiring raw-trajectory or embedding-net access to resolve rather than combining an additional already-available channel. **Confirming this at the level of a trained, calibrated SBI posterior was attempted (not deferred): see L11.** The decisive test (matching §6.3.3/`nb04b`'s methodology) was run — a CNN-embedding SBI posterior trained on raw S-B trajectories — and did not resolve the correlation. **Report as: strong, convergent, two-independent-method (FIM + EKF) evidence that this pair is an artifact of the observation representation, with the practical fix (feature engineering, not new sensors) identified but not yet achieved by a trained amortized estimator at the budget explored.** See `HANDOFF.md` "Finding 9" for full detail. |
| L4″ | **A third representation artifact: (α/β_r, z_A0_eff) near-degeneracy under `compute_summaries`, discovered via fault classification** | Wu 2003 S-B | `nb31` (2026-07-05): posterior-mass fault classification across 14 scenarios x 30 replicates found feed-fault detection unreliable (F1 = 0.00 for W10; `multi` F1 dragged to 0.85 by W13) — initially misdiagnosed in-notebook as a single-window detection-*power* problem (posterior scatter comparable to fault size, fixable by pooling across windows). **That diagnosis is retracted**: per-replicate z_A0_eff posterior means are precise (std ≈0.01), so pooling would not move the estimate. Re-deriving this paper's own noise-calibrated FIM methodology (§7.2.2/L4′) for (α, z_A0_eff) and (β_r, z_A0_eff) instead: off-diagonal is weak at nominal (≈−0.12) but jumps to ≈−0.89 (α degraded, W2 truth) and ≈−0.48 (β_r degraded, W5 truth) under `compute_summaries` — matching the magnitude of the L4′ (α, β_r) headline number — and **collapses to ≈−0.07 under the raw trajectory in both cases**, the identical signature as L4/L4′. z_A0_eff has no analytical dependence on α or β_r in the plant's governing equations; the summary-statistic representation conflates "a reactor fault is present" with "z_A0_eff has drifted" via overlapping recycle/conversion-related features. **Not a new mechanism — a third confirming instance of L4/L4′'s pattern**, found this time via its effect on a downstream task (classification) rather than via a direct identifiability scan. Fix, if pursued: same raw-trajectory-aware representation change identified for L4′, not implemented (same L9/L10 training-instability risk). |
| L5 | η_col posterior overconfidence — **status changed from "unresolved" to "resolved for S-B, marginally"** | Wu 2003 S-B | The 8-seed-ensemble/multi-N-SBC-corroboration procedure (seed 4, confirmed at N=200/400/800) resolves the marginal SBC failure previously reported at SBC p=0.0001. **However**, per L4 above, this marginal pass does not by itself certify the joint (α, η_col) posterior shape at any specific scenario — that required the separate identifiability-scan verification in §7.2.3. Report both: the marginal calibration fix, and the joint-shape verification methodology, as this session's actual resolution path (not the originally-attempted `reb_per_boilup` feature fix alone, which was necessary but shown insufficient on its own — see L9). |
| L6 | SBC mild miscalibration (KS p=0.016) | Propylene oxide | Structural, not a training deficiency |
| L7 | QSS column shortcut unstable for η_col < 0.80 | Wu 2003 | W8, W14 removed; η_col=0.80 covers headline scenario |
| L8 | No real-time deployment tested | Both | SCADA integration is future work |
| L9 | SNPE training is seed-unstable for the Wu 2003 5-param / 66-72D-summary posteriors | Wu 2003 S-A and S-B | Identical training data + architecture, only the random seed differs, produces SBC pass/fail flips (e.g. η_col p=0.96 vs p=0.0000). 90% CI coverage stays near-nominal regardless, so it cannot detect this — only rank-uniformity SBC does. The working mitigation adopted for S-B: an 8-seed ensemble with multi-N (200/400/800) SBC corroboration before promoting any posterior — see L5. This mitigation was necessary but, per the extensive S-A search (L10), not always sufficient. |
| L10 | **S-A calibration is a settled, unresolved negative result** | Wu 2003 S-A | 0 of 40 total trained posteriors passed SBC at N=400 across two sessions: 16 seeds at the production architecture (zuko_nsf 60/3), plus 24 further seeds testing PCA feature-dimension reduction (15/25 components) and both larger (80/5, 128/5) and smaller (30/2, 40/2) architectures. Larger networks performed *worse* than the production 60/3 (some showed uniform miscalibration across all 5 parameters, consistent with overfitting on the 15k-sample bank), ruling out "network too small" as the explanation. Root cause not identified; most consistently failing parameters are η_col and ξ_reb, plausibly because S-A's composition-control loops (R→x_D, V→x_B) actively suppress exactly the signal SNPE needs to learn for these two parameters. **Do not use any S-A posterior for a quantitative claim in this paper.** The qualitative x_D-breaks-a-degeneracy argument may still be usable via the physical iso-contour figure (simulator-only, no trained posterior involved) if a suitable degeneracy is later found where S-A is relevant — not the case for the current (α, β_r) headline, which is reactor-side. |
| L11 | **CNN-embedding SBI for Wu 2003 was trained and evaluated (`nb32`); confirms L4″, does not resolve L4′** | Wu 2003 S-B | An 8-seed CNN-embedding SNPE ensemble (62,686-parameter CNN, 30-D learned embedding, 15,000 training simulations, raw `(120,9)` S-B trajectories) was trained as the decisive test flagged in L4′/§7.4.1. 1/8 seeds passed marginal SBC (seed 6, min KS p=0.137) — the same order of seed-lottery difficulty as the hand-crafted pipeline's own 8-seed search (L9). Result: correlation(α, β_r) at W11 = 0.993 (hand-crafted: 0.994) — unchanged, and robust across a 19-condition grid-check (0.987–0.996, std ≤0.004; `nb27` Addendum), not a one-point fluke. The same trained network *does* independently confirm L4″'s resolution: correlation(α, z_A0_eff) collapses from ≈0.99 (hand-crafted) to ≈−0.004 (CNN) at W10/W13. Overall CNN-based fault classification is worse than hand-crafted (macro-F1 0.270 vs. 0.679) and 30-day tracking is mixed (α MAE worsens 0.048→0.067; β_r MAE improves 0.198→0.139, still ≫ EKF's 0.002/0.004). **Interpretation: distinguishes information-theoretic recoverability (FIM, EKF — established) from recoverability by this specific amortized estimator at a modest training budget (not established) — a training-practicality gap, not evidence against the representation-artifact diagnosis.** Marginal SBC does not test joint posterior shape (L4/L5); passing it here was not evidence the (α, β_r) joint structure was learned correctly. A larger-capacity or differently-structured embedding (more conv layers, an RNN/GRU embedding, or more training simulations) might still succeed, but this is not a safe assumption to skip validating — L10 already documents that larger networks have previously made SNPE calibration *worse*, not better, for this system. |
| Note | ξ_reb peaked SBC histogram (S-A) was a rejection sampling artifact | Wu 2003 S-A | With `reject_outside_prior=False` (superseded terminology in sbi 0.24 — direct sampling is now the default), ξ_reb SBC p=0.146 in one run — but per L10, S-A training instability means this single result should not be treated as a settled calibration claim either. |

---

## 9. Conclusion (~0.5 pages)

Restate the contributions with their quantitative outcomes:

1. For the propylene oxide CSTR: I_αα/I_ββ = 250–500×; confirmed irreducible by
   4-method agreement and a raw-time-series CNN-embedding experiment that bypasses
   hand-crafted features entirely; macro-F1 = 0.990 despite structural bias; 30-day
   tracking with SBI processing 720 windows in seconds.

2. For the Wu 2003 CSTR-column-recycle plant, the same masking mechanism transfers exactly
   (∂T_r_ss/∂α ≡ ∂T_r_ss/∂β_r ≡ 0, an analytical, representation-independent fact), but with
   a materially different quantitative signature: it reduces both α's and β_r's information
   roughly *symmetrically* (I_αα/I_β_r ≈ 1.1–1.4×, confirmed under both compressed and raw
   representations) rather than singling one out as in PO (250–500×), because Wu 2003 gives
   neither parameter an independent rescue channel.

3. **Two candidate joint (2-parameter) degeneracies were investigated in the recycle plant,
   and both were found to be artifacts of the observation representation rather than
   physical properties of the plant — the paper's central methodological finding.**
   (α, η_col) at W12 was a restricted-*channel* artifact: a banana visible only when the
   recycle flow (F_R) is considered in isolation, resolved by combining an already-available
   second channel (T_reb), with no new instrumentation. (α, β_r) at W11 appeared far more
   robust — surviving combination of the *entire* 66-D hand-crafted summary-statistic
   feature set (FIM off-diagonal +0.90, trained-posterior correlation 0.998) — but is a
   lossy-*aggregation* artifact one level deeper: the identical Fisher-information
   methodology applied to the raw, unaggregated trajectory of the same three
   already-observed channels collapses the coupling to ≈0.00, corroborated independently by
   a re-tuned EKF recovering both parameters to ~1-2% accuracy on data the summary-statistic
   representation calls non-identifiable. A negative control rules out an easy fix: modestly
   finer time-resolution summary statistics do not help. We additionally trained a
   calibrated CNN-embedding SBI posterior directly on raw trajectories to test this at the
   level of a trained amortized estimator: it did **not** narrow the correlation (0.993 vs.
   0.994, robust across a 19-condition grid-check), distinguishing information-theoretic
   recoverability (established, via FIM and EKF) from recoverability by this specific
   estimator at a modest training budget (not established) — a training-practicality gap,
   not a retraction of the representation-artifact diagnosis. The same trained network did,
   however, independently confirm the resolution of a related third candidate degeneracy,
   (α/β_r, z_A0_eff) — discovered via a fault-classification failure rather than a direct
   identifiability scan (§7.3, L4″) — collapsing its correlation from ≈0.99 to ≈0.00, and
   showing that a modest CNN embedding can resolve some but not all of the artifacts this
   representation produces.

4. **The general lesson we draw is a diagnostic recommendation for the field, not just a
   correction to our own earlier claims**: in closed-loop, multi-unit SBI applications, a
   *scalar* identifiability reduction (masking of one parameter's primary channel) is a
   robust effect that survives arbitrarily rich observation access — confirmed twice here,
   via an irreducibility test in each system. A *joint/manifold* degeneracy surfaced by a
   compressed-feature SBI or FIM analysis should be treated as provisional and checked
   against a raw-trajectory (or otherwise minimally-aggregated) Fisher-information test
   before being reported as physical; in this study, every such candidate failed that check.
   This reframes the plant's design guidance from sensor placement (a new analyser, which
   would not have fixed a representation artifact) to feature engineering (less lossy use of
   the sensors already in place) — a cheaper, more broadly applicable intervention.
   Separately, EKF fails under strong catalyst-decay/snowball nonlinearity (3% coverage at
   two tested scenarios) — a genuine, representation-independent failure mode distinct from
   either banana investigation, and the clearest demonstration in this paper of where SBI's
   correct non-Gaussian handling earns its computational cost.

Forward-looking statement: a raw-trajectory-aware (CNN-embedding) SBI posterior for Wu 2003
was trained and evaluated at the level of calibrated posterior coverage (§7.4.1, L11) — it
neither trains substantially more nor less stably than the hand-crafted-feature networks
(1/8 seeds passed SBC, matching the existing pipeline's own seed-lottery rate), and it does
not, at this architecture and training budget, resolve the (α, β_r) correlation, although it
does resolve a related three-way confound. The open question this reframes, rather than
closes: whether a larger-capacity or differently-structured embedding (more conv layers, an
RNN/GRU-based embedding, or substantially more training simulations) can close this specific
gap — not a safe assumption, given this project's own evidence (L10) that larger networks
have previously made SNPE calibration worse, not better, for this system. Beyond that: a
general audit of other published closed-loop SBI/FIM "banana" claims against the
raw-trajectory check proposed here; extension to MPC-controlled plants (stronger masking);
integration with digital twins and plant historians; active experiment design for scheduled
open-loop excitation; multi-plant transfer learning with shared priors.

---

## Figures summary

| # | Content | Source | Section |
|---|---------|--------|---------|
| 1 | Prior predictive + SBC ranks (PO) | nb04 | §6.1 |
| 2 | Joint posteriors all 8 scenarios (PO) | nb06 | §6.2 |
| 3 | Fisher information channel decomposition (PO) | nb15 | §6.3.1 |
| 4 | CNN embedding vs. hand-crafted (PO) | nb04b | §6.3.3 |
| 5 | 4-method baseline dashboard (PO) | nb16 | §6.4 |
| 6 | 30-day tracking with CI bands (PO) | nb10, nb16 | §6.5 |
| 7 | SBC for 5 parameters, S-B calibrated (seed 4) + S-A failure evidence | nb23, nb24, nb25 | §7.1 |
| 8 | 5×5 Fisher information heatmap showing (α, β_r) coupling (+0.90) and T_r near-zero | nb24 | §7.2.1-2 |
| 9 | Marginal posteriors 14 scenarios, S-B calibrated (S-A shown as uncalibrated/limitation only) | nb24, nb25 | §7.3 |
| 10 | Headline: (α, β_r) banana at W11 (S-B posterior scatter) vs. EKF (pending) | nb26 (full rewrite) | §7.4 |
| 10b | (retired figure) F_R-only vs. combined-feature identifiability scan for (α, η_col) — supports the retraction | new notebook (TBD) | §7.2.3 |
| 11 | SBI vs. EKF 30-day tracking for α, β_r, η_col with CI bands | nb27 | §7.5 |
| 12 | NUTS timing comparison and monitoring cadence feasibility | nb26 | §7.6 |
| 13 | CNN-embedding vs. hand-crafted joint posteriors at W10/W11/W12 (Wu 2003) — the decisive-test result | nb32 | §7.4.1 |
| 14 | CNN-embedding vs. hand-crafted 30-day tracking for α, β_r (Wu 2003) | nb32 | §7.4.1/§7.5 |

All figures: 300 dpi minimum, double-column compatible, colour-blind palette (viridis/Okabe-Ito).

---

## Tables summary

| # | Content | Section |
|---|---------|---------|
| 1 | Propylene oxide model parameters | §3.1 |
| 2 | PO fault scenarios (Sc1–Sc8) | §3.1 |
| 3 | Wu 2003 plant parameters (from Table 1, SI units) | §3.2 |
| 4 | Wu 2003 fault scenarios (W1–W16) and control structures S-A/S-B | §3.2 |
| 5 | Full summary statistics definition (29-D PO; 55-D Wu 2003) | §4.2 |
| 6 | SBI/EKF training hyperparameters for both systems | §5.1 |
| 7 | PO per-scenario classification results | §6.2 |
| 8 | 4-method baseline comparison (SBI, NUTS, EKF, UKF) | §6.4 |
| 9 | PO 30-day tracking phase metrics | §6.5 |
| 10 | Wu 2003 per-scenario classification results (S-B calibrated; S-A shown only as a calibration-failure limitation, not a comparison) | §7.3 |
| 11 | Wu 2003 SBI vs. EKF comparison — bias, MAE, coverage, F1 (W11, W12, W15) | §7.5 |
| 12 | Monitoring cadence feasibility: SBI vs. EKF vs. NUTS | §7.6 |
| 13 | Wu 2003 model mismatch robustness | §7.7 |

---

## Appendices

- **A. Nomenclature:** All symbols, units, and definitions (C&ChE requirement)
- **B. Density estimator comparison:** MAF vs. NSF on propylene oxide; justification for NSF choice
- **C. MCMC convergence diagnostics:** R̂, ESS, trace plots for NUTS on propylene oxide
- **D. Analytical steady-state Jacobian:** Full derivation of ∂T_ss/∂β = 0 under PI control
- **E. Additional posterior plots:** Remaining scenario pairplots for both systems

---

## Pre-submission checklist

**Completed (propylene oxide work):**
- [x] EKF/UKF baseline (nb16)
- [x] 4-method β bias confirmation
- [x] SBC with honest reporting (KS p = 0.016)
- [x] CNN embedding irreducibility proof (nb04b)
- [x] Fisher information analysis (nb15)
- [x] 30-day tracking with MAE/CRPS (nb10)
- [x] Analytical bias derivation (bias_explanation_2.md)

**Required before submission (revised 2026-07-03 — see HANDOFF.md for the full session log):**
- [x] Wu 2003 nb20-nb28 implementation and execution (done)
- [x] S-B calibration resolved: 8-seed ensemble + multi-N (200/400/800) SBC corroboration,
      seed 4 promoted, all 5 parameters pass
- [x] S-A calibration exhaustively attempted and confirmed as a settled negative result
      (40 seeds across two sessions); reported as limitation L10, not pursued further
- [x] (α, β_r)/(α, η_col) identifiability scans formalised into a notebook (`nb29b`) —
      produced Figure 10's supporting evidence and the Figure 8/10b F_R-only vs.
      combined-feature comparison
- [x] EKF run at W11, W12, W15 (`nb26`) — including the tight-vs-diffuse tuning follow-up
      that overturned the original 0%-coverage reading (`nb27` §9)
- [x] FIM analysis block in nb23 §7 (5×5 FIM heatmap — Figure 8), re-derived on the raw
      trajectory in `nb29b` §4 — relabelled to emphasise (α, β_r), then retracted (§7.4)
- [x] nb26 (headline banana/EKF figure) — full rewrite around W11/(α,β_r); old
      W12/(α,η_col) framing retired
- [x] nb27 sequential tracking re-executed with the calibrated seed-4 S-B posterior
- [x] nb24/nb25 Assessment cells reflect the (α, β_r) finding
- [x] nb29 (η_col SBC investigation) follow-up note added: the original confound
      diagnosis/fix stand, but the interpretive claim about what a narrow η_col posterior
      means was corrected (narrow-and-honest is the correct answer, per `nb29b`)
- [x] nb30: claims-and-conclusions synthesis notebook (styled like nb14), covering
      nb20-nb29b, nb31, and the finding that all three candidate joint degeneracies found in
      this system ((α,η_col), (α,β_r), (α/β_r,z_A0_eff)) are representation artifacts —
      `notebooks/30_wu2003_claims_and_conclusions.ipynb`, executed 2026-07-05
- [x] nb31: Wu 2003 fault classification notebook (posterior-mass approach, §7.3) —
      `notebooks/31_wu2003_fault_classification.ipynb`, executed 2026-07-05, then re-executed
      same day after fixing a `fault_unit()` labelling bug (§8.4 L4″ note) and adding a FIM
      investigation (§6b). Final: 87.4% accuracy, macro-F1 0.694; W11 confirmed to classify
      correctly (30/30, `reactor`) despite the (α, β_r) representation artifact — framed
      explicitly as a worked example of the artifact diagnostic (§8.1), not as confirmation
      of a physical banana. Surfaced a third representation artifact, (α/β_r, z_A0_eff),
      confirmed by FIM and added to §8.4 as **L4″** (feed F1=0.00; not a detection-power
      issue as first suspected — see §7.3)
- [x] nb32: CNN-embedding SBI posterior for Wu 2003 built, trained (8-seed ensemble), and
      evaluated — `notebooks/32_wu2003_cnn_embedding.ipynb`, executed 2026-07-14. This was
      the decisive test flagged and deliberately deferred at L4′/`nb27` §9 ("train an SBI
      posterior on a richer summary representation and check whether the (α, β_r) ridge
      narrows"). Result: does **not** narrow (0.993 vs. 0.994, robust across a 19-condition
      grid-check) — added as **L11**, §7.4.1. Independently confirms L4″'s resolution
      (correlation collapses to ≈0.00 for (α, z_A0_eff)). `nb27` given a matching 
- [ ] All figures regenerated at publication quality (300 dpi, double-column)
- [ ] Nomenclature table with every symbol (C&ChE requirement)
- [ ] Highlights updated (3-5 bullets, ≤85 chars — done this session, verify final wording)
- [ ] Abstract rewritten with corrected EKF/MCMC framing (done this session, verify against
      final W11 EKF numbers once that run completes)
- [ ] No undefined acronyms in abstract
- [ ] Ljung (1977), Gevers et al. (2011), Forssell & Ljung (1999), Luyben (1994) cited
- [ ] Fault classification framed as "label-free" not "unsupervised"
- [ ] Data availability statement (C&ChE requirement)
- [ ] CRediT author statement
- [ ] Conflict of interest statement
- [ ] Simulator/training data available (GitHub or Zenodo)

**Removed from checklist (not required):**
- ~~Prior sensitivity study for propylene oxide~~ — bias proved structural via 4-method confirmation + analytical derivation
- ~~Model mismatch robustness (§7.7)~~ — deferred to future work; §8.4 L1 acknowledges synthetic-data limitation

---

## Effort estimate

| Task | Effort | Status |
|------|--------|--------|
| Propylene oxide results | — | **DONE (nb01–nb16)** |
| Luyben plant implementation | 6-8 weeks | Planned (project_luyben_extension.md) |
| Prior sensitivity study (PO) | 2-3 days | Open |
| Publication-quality figures | 2-3 days | Open |
| Paper writing | 3-4 weeks | After Luyben results |
| **Total to submission** | **~10-12 weeks** | — |
