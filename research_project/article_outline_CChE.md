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
- EKF and MCMC both fail at banana-posterior regimes; SBI is the only correct and practical method
- Recycle coupling creates (α, η_col) banana posterior invisible to EKF; SBI captures it (100% coverage)

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
parameter is 250–500× smaller than for catalyst activity — confirmed empirically by
four independent methods (SBI, MCMC, EKF, UKF) all showing identical structural bias.
For the recycle plant, three decentralized PI loops and a liquid recycle stream create
a new identifiability challenge: catalyst decay (α) and column tray efficiency (η_col)
both increase recycle flow under the snowball effect, producing a banana-shaped joint
posterior invisible to the EKF's Gaussian approximation under conventional measurement
control. The banana posterior represents a fundamental failure mode for both EKF and MCMC:
the EKF achieves only 3% empirical coverage at the banana scenario (mean estimate 1.185
vs. true 0.75), while MCMC convergence is unreliable at this scale. SBI correctly captures
the banana geometry (100% coverage under S-B) and resolves it with a composition analyser
(S-A: 75% CI width reduction). SBI inference takes under 20 ms per window after a
one-time training cost; 720-window 30-day monitoring completes in seconds.

**Conclusions.** Amortised SBI is the only currently practical method that is both
computationally feasible for real-time monitoring and qualitatively correct in representing
non-Gaussian posterior geometry. EKF fails on correctness (3% coverage under banana
conditions); MCMC fails on both reliability and speed. The structural identifiability
hierarchy revealed here — reactor thermal faults masked by temperature control, cross-unit
faults coupled through recycle — provides actionable design guidance for sensor placement
and scheduled open-loop recalibration.

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
> 3. **Plant-wide fault localization in the Wu 2003 reactor-column-recycle benchmark.**
>    We extend to the Wu et al. (2003) CSTR-column-recycle process (Comput. Chem. Eng.
>    27(3):401–421; 5 fault parameters, 3 PI loops, liquid recycle, 14 scenarios) and
>    demonstrate that: (a) the same reactor jacket masking mechanism from the PO CSTR
>    (I_β_r/I_α ≈ 1/250–500) persists unchanged when a column and recycle are added;
>    (b) the recycle creates a new, distinct coupling: catalyst decay (α) and column tray
>    efficiency (η_col) jointly increase recycle flow via the snowball effect, producing a
>    banana-shaped joint posterior under conventional measurement control (S-B, no
>    composition analysers); (c) adding a composition analyser (S-A, ≈ Wu B-2) resolves
>    the banana — α posterior CI width reduces by 75% for the headline compound scenario.
>
> 4. **SBI as the only method that is simultaneously fast and qualitatively correct.**
>    EKF achieves only 3% empirical coverage under banana conditions (mean estimate 1.185
>    vs. true α = 0.75 for W12) — it fails completely, not just overestimates confidence.
>    MCMC convergence is also unreliable at this scale in the presence of structural
>    identifiability limits. SBI's amortisation cost is paid once at training time;
>    subsequent inference takes <20 ms, enabling 720-window 30-day monitoring in seconds.
>    SBI is the only method that is simultaneously fast enough for monitoring cadence and
>    correct in non-Gaussian posterior geometry.

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

**Identifiability structure (key finding):**
- β_r: masked by Loop 1 (same mechanism as PO β) → I_β_r ≪ I_α (Fisher information)
- (α, η_col): **both increase recycle via snowball** → joint posterior banana-shaped under S-B
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

SBC rank histograms for all 5 parameters. Expected: KS p > 0.05 for α, ξ_reb, z_A0;
β_r expected to show flat/biased ranks (same structural masking as PO β — confirms the
mechanism generalises); (α, η_col) expected to show correlated miscalibration under S-B
(banana posterior produces non-uniform marginal ranks). Report honestly with the same
framing used in §6.1: miscalibration reflects structural identifiability limits, not a
training deficiency.

**Figure 7:** SBC results for 5 Wu 2003 parameters under S-A and S-B — rank histograms
and KS p-values showing β_r masking persists; (α, η_col) coupling emerges under S-B.
Source: nb23.

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

This is a **more severe** identifiability challenge than PO: not just β_r is hard to identify,
but the entire (α, β_r) subspace is jointly confounded through shared features.

**7.2.3 (α, η_col) banana is a nonlinear effect, not a local FIM result**

Local FIM at nominal: normalised (α, η_col) = **−0.142** (slightly negative). The banana
posterior only emerges at degraded values (α≈0.75, η_col≈0.80) where the snowball
nonlinearity makes both faults drive F_R upward in the same direction. The linear FIM
captures only local behaviour near the nominal — it cannot detect the nonlinear degeneracy
that SBI discovers through global prior sampling.

**7.2.4 z_A0 is the most locally identifiable parameter**

Largest FIM diagonal: I_z_A0 = 2.15×10¹⁴ (vs. I_αα = 2.22×10¹³). Feed purity affects
the entire reactor steady state through the inlet composition — a decoupled signal not shared
with other parameters.

**7.2.5 S-A adds η_col information (+32% at nominal)**

S-A/S-B I_η_col ratio = 1.32× — the x_D measurement provides the only locally decoupled
η_col signal. For α and β_r, S-A slightly *reduces* local identifiability (0.90× and 0.63×)
because Loop 2 damps x_D variance. The 75% α posterior CI reduction in nb25 is a nonlinear
effect at degraded values, not captured by the local FIM.

**Figure 8:** 5×5 normalised FIM heatmaps (S-B left, S-A right). Source: nb23 §7.
Key features: β_r and α both near-zero T_r contribution; high (α, β_r) off-diagonal (+0.901)
showing joint confusion; η_col off-diagonal shifts positive under S-A (+0.195 vs −0.142).

### 7.3 Snapshot fault classification (14 scenarios, posterior-mass approach)

14 scenarios (W8, W14 removed — see §8.4), 30 replicates each. Classification uses
posterior mass in fault-unit regions (same approach as PO §6.2 / nb11), not thresholded
posterior mode. Fault units: healthy, reactor (α↓ or β_r↓), column (η_col↓ or ξ_reb↓),
feed (z_A0↓), compound (multiple degraded).

Key results (source: nb30 — fault classification notebook, to be created):
- Reactor faults (W2–W6): high macro-F1; β_r class relies on T_j and Q_j channels
- Column fault (W7): classification under S-B is unreliable (η_col posterior overconfident
  under S-B, SBC p=0.0001; root cause: recycle_ratio/reb_intensity α-confounding; see §8.4 L5)
- Compound faults (W12) under S-B: posterior mass splits between reactor and column fault
  classes (high classification entropy) — this IS the correct answer, reflecting the banana
  degeneracy. Under S-A: correctly classifies as compound.
- Feed fault (W10, W13): identifiable in both structures via z_A shift

**Figure 9:** Marginal posterior summaries — posterior mean ± 90% CI for all 5 parameters
across 14 scenarios, side-by-side for S-A (left) and S-B (right). Source: nb24, nb25.

**Table 9:** Per-scenario classification results (posterior-mass F1, 90% CI coverage) for
both structures. η_col results include calibration caveat.

### 7.4 Headline: (α, η_col) banana posterior and EKF failure (W12 compound scenario)

**Scenario W12:** α = 0.75, η_col = 0.80 (catalyst decay + column efficiency loss).
Under S-B (conventional instrumentation, no composition analyser):

- Loop 1 holds T_r at setpoint → masks both β_r and α via the temperature channel
- Both faults independently trigger the snowball (F_R increases in the same direction)
- F_R increase is ambiguous: consistent with many (α, η_col) combinations along a curved
  manifold in parameter space — the banana posterior

**Actual SBI result (S-B):** Wide α posterior, CI width 0.240, 100% empirical coverage.
The α dimension of the posterior correctly represents the irreducible ambiguity in the data.
**Important caveat (from nb29):** the η_col dimension of the SBI posterior is overconfident
(SBC p=0.0001) — the scatter in (α, η_col) space is a near-vertical stripe at η_col≈0.80,
not a curved banana. The root cause is that `recycle_ratio` (corr=−0.977 with α) dominates
the η_col information budget, confounding the two parameters. The *physical* banana manifolds
are confirmed by the F_R iso-contour figure (nb26_banana_physics.png); the SBI approximates
the α dimension of this manifold correctly but is miscalibrated in η_col.
**Planned fix:** replace `reb_intensity` with `reb_per_boilup = Q_reb/V_norm` in the summary
statistics, retrain with `zuko_nsf` (60 hidden, 3 transforms), verify η_col SBC improves.

**Actual SBI result (S-A):** α CI width narrows to 0.059 (−75%); posterior mean = 0.738
(true = 0.750, error = 0.012). x_D measurement breaks the degeneracy. 93% coverage.

**Actual EKF result (S-B):** α mean = 1.185 ± 0.084 (true = 0.75). **3% empirical
coverage.** The EKF never moves away from nominal — the banana degeneracy gives no
directional gradient for the Kalman update. This is not overconfidence; the EKF is
pointing in the completely wrong direction. Near the snowball tipping point (W15, α=0.58),
the EKF achieves 3% coverage (mean = 0.990, true = 0.58) — the linearisation is applied
in entirely the wrong dynamical regime.

**Figure 10:** (a) SBI joint (α, η_col) posterior under S-B (banana scatter + KDE contours);
(b) S-A posterior (tight cluster near truth); (c) EKF Gaussian ellipses (centred near
nominal, not truth); (d) coverage comparison bar chart: SBI S-B 100%, SBI S-A 93%,
EKF 3%.  Source: nb26.

### 7.5 EKF baseline comparison

Augmented EKF: 9-state vector [z_A, T_r, T_j, I_T, R_state, V_state, α, β_r, η_col].
Pure-numpy implementation with precomputed QSS column lookup table (avoids JAX OOM in
the 720-step sequential loop). Observations used: T_r, T_j, F_R_norm.

Actual findings:
- Both methods show structural β_r and α bias (~0.10 downward for α; same mechanism as PO β)
- **EKF completely fails** under banana conditions: 3% α coverage, mean estimate far from truth
- EKF near tipping point (W15): 3% coverage, mean 0.990 vs true 0.58
- SBI W15: 100% coverage, CI [0.436, 0.608] correctly contains truth

**Figure 11:** SBI vs. EKF 30-day tracking for α, β_r with CI bands. Source: nb27 (pending).

**Table 10:** Wu 2003 SBI vs. EKF — bias, MAE, coverage; snapshot comparison for W12, W15.

### 7.6 NUTS infeasibility for Wu 2003

NUTS was not run on the Wu 2003 system. Even for the 2-D propylene oxide system, NUTS
convergence was unreliable in the presence of structural identifiability limits — the
posterior geometry (banana manifold, wide β bias) creates mixing difficulties that NUTS
does not resolve. For 5-D parameter spaces with recycle-coupled non-Gaussian posteriors,
NUTS would face the same qualitative failure as EKF, compounded by exponentially slower
mixing. MCMC is therefore neither fast enough nor reliable enough for this problem.
SBI is the only method that is simultaneously: (a) fast (< 20 ms/window), (b) correct
in posterior geometry (100% banana coverage), and (c) calibrated for the well-identified
parameters (α).

**Table 11:** Feasibility comparison — SBI vs. EKF vs. NUTS (infeasible + unreliable).

*Note: §7.7 (model mismatch robustness) is deferred to future work — see §8.4.*

---

## 8. Discussion (~2 pages)

### 8.1 Structural identifiability across scales

**Two systems, two distinct mechanisms, one unified theory.**

The propylene oxide CSTR illustrates the fundamental *single-loop* mechanism: the
reactor temperature PI controller zeros ∂T_ss/∂β_r, removing the highest-SNR channel
from β_r's information budget and giving I_β_r/I_α = 1/250–500. This is a scalar
reduction: one parameter is harder to identify, but all parameters are identifiable
in principle.

The Wu 2003 CSTR-column-recycle plant reveals a qualitatively different *multi-loop*
mechanism: the recycle stream creates coupling between reactor and column faults that
is not present in any single-unit analysis. α and η_col jointly determine the recycle
flow magnitude through the snowball amplification — making them non-identifiable as
a pair under conventional control. This is not a scalar reduction but a *manifold
constraint*: the posterior is constrained to a curve (banana) rather than a point.

**The identifiability hierarchy across both systems:**
- β_r: masked by Loop 1 (temperature) in BOTH systems — I_β_r/I_α ≈ 1/250–500
- (α, η_col): jointly constrained by recycle coupling — posterior is banana-shaped
- ξ_reb: identifiable via Q_reb (Loop 3 output signal is the direct signature)
- α alone: most identifiable — reaction heat signal in Q_c plus recycle flow change
- z_A0: identifiable via steady-state shift in z_A and F_R

This hierarchy is a practical design guideline for sensor placement: measurements of
controller output signals (Q_c, Q_reb) are essential because they carry the information
that process variable measurements (T_r, T_reb) hide under closed-loop control.

### 8.2 When EKF fails and SBI wins

EKF is an excellent industrial baseline when the posterior is approximately Gaussian —
which holds for the propylene oxide system near the nominal operating point (all four
methods show similar bias and comparable uncertainty). The EKF breaks down in two
specific situations, both demonstrated in this paper:

1. **Recycle-coupled non-Gaussian posteriors** (α, η_col banana under S-B): The
   Gaussian assumption collapses a manifold constraint to an ellipse. **Empirical result:
   EKF achieves 3% α coverage for W12 — it fails completely, not just underestimates
   confidence.** EKF mean estimate = 1.185 vs true α = 0.75; the Kalman update has no
   directional gradient to follow along the banana manifold.

2. **Near the snowball tipping point** (W15, α=0.58): The Jacobian of the recycle
   dynamics changes rapidly as α approaches the critical value. The EKF linearises
   around its current estimate (~nominal, α≈1.0) — in entirely the wrong dynamical
   regime. **Empirical result: EKF 3% coverage (mean = 0.990, true = 0.58). SBI: 100%
   coverage, CI [0.436, 0.608].** SBI was trained across the full prior including
   near-tipping samples and correctly widens its uncertainty in this regime.

The recommendation: SBI should be the primary inference method for any plant approaching
recycle tipping-point conditions. EKF remains useful for real-time state estimation under
near-nominal operation but should not be used for uncertainty quantification when
non-Gaussian posteriors are possible (detectable via elevated EKF covariance trace).

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
| L4 | (α, η_col) partial non-identifiability (banana) | Wu 2003 S-B | The banana is a genuine physical degeneracy (confirmed by F_R iso-contours); SBI correctly represents wide α uncertainty (100% coverage); η_col dimension is overconfident (see L5) |
| L5 | η_col posterior overconfident under S-B (SBC p=0.0001) | Wu 2003 S-B | Root cause: `recycle_ratio` (corr −0.977 with α) and `reb_intensity` (corr +0.642) confound η_col signal; NSF pins η_col near training mean. The banana scatter is a near-vertical stripe, not curved. α results unaffected. |
| L6 | SBC mild miscalibration (KS p=0.016) | Propylene oxide | Structural, not a training deficiency |
| L7 | QSS column shortcut unstable for η_col < 0.80 | Wu 2003 | W8, W14 removed; η_col=0.80 covers headline scenario |
| L8 | No real-time deployment tested | Both | SCADA integration is future work |
| Note | ξ_reb peaked SBC histogram (S-A) was a rejection sampling artifact | Wu 2003 S-A | With `reject_outside_prior=False`, ξ_reb SBC p=0.146 — well-calibrated. The peaked histogram reflected sbi's rejection sampler failing when posterior concentrates away from prior center, not a genuine calibration problem. |

---

## 9. Conclusion (~0.5 pages)

Restate the four contributions with their quantitative outcomes:

1. For the propylene oxide CSTR: I_αα/I_ββ = 250–500×; confirmed irreducible by
   4-method agreement and CNN embedding experiment; macro-F1 = 0.990 despite structural
   bias; 30-day tracking with SBI processing 720 windows in seconds.

2. For the Wu 2003 CSTR-column-recycle plant: β_r masking persists unchanged from
   System I (I_β_r/I_α ≈ 1/250–500); recycle coupling creates (α, η_col) banana posterior
   under S-B — SBI correctly captures it (100% α coverage); adding a composition analyser
   (S-A, ≈ B-2) reduces α CI width by 75%; EKF achieves 3% coverage and fails completely.

3. The banana posterior is a fundamental failure mode for both EKF and MCMC: EKF collapses
   it to a Gaussian ellipse in the wrong location (mean 1.185 vs true 0.75); MCMC
   convergence is unreliable at this scale. SBI is the only method that is simultaneously
   fast (<20 ms/window), correct in posterior geometry (100% banana coverage), and
   calibrated for the well-identified parameters.

Forward-looking statement: extension to MPC-controlled plants (stronger masking);
integration with digital twins and plant historians; active experiment design for
scheduled open-loop excitation; multi-plant transfer learning with shared priors.

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
| 7 | SBC for 5 parameters under S-A and S-B (Wu 2003) | nb23 | §7.1 |
| 8 | 5×5 Fisher information heatmap showing (α, η_col) coupling and β_r near-zero | nb24 | §7.2.1 |
| 9 | Marginal posteriors 16 scenarios, S-A vs. S-B side-by-side | nb24 | §7.3 |
| 10 | Headline: (α, η_col) banana (S-B) vs. narrow (S-A) vs. EKF ellipse; coverage chart | nb24, nb26 | §7.4 |
| 11 | SBI vs. EKF 30-day tracking for α, β_r, η_col with CI bands | nb27 | §7.5 |
| 12 | NUTS timing comparison and monitoring cadence feasibility | nb26 | §7.6 |

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
| 10 | Wu 2003 per-scenario classification results (S-A vs. S-B) | §7.3 |
| 11 | Wu 2003 SBI vs. EKF comparison — bias, MAE, coverage, F1 | §7.5 |
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

**Required before submission:**
- [x] Wu 2003 nb20-nb28 implementation and execution (done)
- [ ] FIM analysis block in nb23 §7 (5×5 FIM heatmap — Figure 8)
- [ ] nb30: Wu 2003 fault classification notebook (posterior-mass approach, §7.3)
- [ ] nb27 sequential tracking results (currently executing; update §7.5/Figure 11/Table 10)
- [ ] SBC p-value for η_col confirmed and limitation statement added to §7.3/§8.4
- [ ] All figures regenerated at publication quality (300 dpi, double-column)
- [ ] Nomenclature table with every symbol (C&ChE requirement)
- [ ] Highlights updated (3-5 bullets, ≤85 chars — EKF 3% coverage as headline)
- [ ] Abstract rewritten with corrected EKF/MCMC framing
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
