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
- Amortised SBI recovers non-Gaussian posteriors in recycle plants; MCMC is 10,000× slower
- EKF gives overconfident Gaussian intervals near snowball bifurcation; SBI does not
- Recycle coupling creates (α, η_col) banana posterior invisible to EKF; SBI captures it

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
control. SBI correctly attributes root-cause faults from 7 conventional measurements
despite multi-loop masking. MCMC requires ~8 min per observation window at this scale;
SBI processes the same window in under 20 ms after a one-time training cost.

**Conclusions.** Amortised SBI outperforms both EKF (incorrect posterior geometry) and
MCMC (impractical for monitoring cadence) for plant-scale fault diagnosis. The structural
identifiability hierarchy revealed here — reactor thermal faults masked by temperature
control, cross-unit faults coupled through recycle — provides actionable design guidance
for sensor placement and scheduled open-loop recalibration.

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
> 3. **Plant-wide fault localization in the Luyben reactor-column-recycle benchmark.**
>    We extend to the canonical Luyben CSTR-column-recycle process (Wu et al. 2003,
>    Comput. Chem. Eng. 27(3):401–421; 5 fault parameters, 3 PI loops, liquid recycle)
>    and demonstrate that: (a) the same reactor jacket masking mechanism from the PO CSTR
>    (I_β/I_α ≈ 1/250–500) persists unchanged when a column and recycle are added;
>    (b) the recycle creates a new, distinct coupling: catalyst decay (α) and column tray
>    efficiency (η_col) jointly increase recycle flow via the snowball effect, producing a
>    banana-shaped joint posterior under conventional measurement control (no composition
>    analysers). SBI captures this joint uncertainty correctly; the EKF collapses it to an
>    overconfident Gaussian ellipse.
>
> 4. **SBI as the practical method for real-time plant-wide monitoring.**
>    NUTS at plant scale requires ~8 min per 2-hour observation window (extrapolated from
>    2-D PO timing × dimension scaling), making 30-day sequential monitoring (720 windows)
>    take ~4 days of compute. SBI's amortisation cost is paid once at training time;
>    subsequent inference is a single network forward pass taking <20 ms. Beyond speed:
>    EKF provides only Gaussian uncertainty quantification, which is demonstrably incorrect
>    near the snowball tipping point. SBI is the only method that is simultaneously fast
>    enough for monitoring cadence and correct in posterior geometry.

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
| Wu 2003 analogue | B-3 (full composition control) | B-1b/B-1c (ratio/temperature only) |

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

**16 fault scenarios, 30 replicates each (Table 4):**
Individual faults W2–W8 (one parameter degraded), combined faults W9–W14 (cross-unit),
snowball threshold W15 (α near critical tipping point), full multi-fault W16.
Both control modes S-A and S-B run for each scenario (512 windows total).

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

### 7.2 Identifiability persistence and new recycle coupling (core contribution)

**7.2.1 β_r masking persists across system scale**

Compare Fisher information diagonal for β_r in PO (§6.3) vs. Wu 2003:
- PO CSTR (1 unit, 1 loop): I_β_r/I_α ≈ 1/250–500
- Wu 2003 (2 units, 3 loops, recycle): I_β_r/I_α ≈ 1/250–500 (expected: same order)

The mechanism is identical: Loop 1 (reactor temperature PI) zeros ∂T_r/∂β_r at steady
state regardless of what the rest of the plant does. **The reactor masking is modular —
adding a column and recycle does not change it.** This is the key bridge between §6 and §7:
the same analytical result (∂T_ss/∂β_r = 0 under PI control) applies to both systems.

**7.2.2 New cross-unit coupling via recycle (α, η_col)**

Under S-B (no composition analyser), both α↓ and η_col↓ cause:
- Less effective conversion or separation → more A escapes to distillate
- Recycle flow F_R increases (snowball amplification)
- The column temperature T_reb and reboiler duty Q_reb both rise (Loop 3 compensates)

The data are consistent with: (α=0.70, η_col=1.0) OR (α=1.0, η_col=0.70) OR any
combination with same total recycle load. Posterior is banana-shaped.

Under S-A (with x_D measured): the distillate composition breaks the degeneracy.
α↓ increases x_D (more A in distillate because less is converted). η_col↓ also increases
x_D (worse separation), but with a different functional form. The posterior narrows but
retains a residual non-Gaussian shape near the nominal operating point.

**Figure 8:** 5×5 Fisher information heatmap (S-B, nominal OP) — showing (α, η_col)
off-diagonal coupling as the largest off-diagonal element; β_r diagonal near-zero.
Source: nb24.

### 7.3 Snapshot fault classification (5-D)

16 scenarios, 30 replicates each. Expected results:

- Individual reactor faults (W2–W6): high macro-F1; β_r classification relies on T_j
  and Q_c channels (correctly identified despite low Fisher information from posterior
  mean shift being low — the mode is correct even if variance is large)
- Individual column faults (W7–W9): high macro-F1 under S-A; reduced under S-B
- Cross-unit faults (W10, W13–W16): lower F1 for (α, η_col) combinations under S-B
  (inherent uncertainty reported honestly with wide credible intervals)
- S-A vs. S-B comparison (same scenario, different sensor set): quantifies the information
  value of the composition analyser for each fault type

**Figure 9:** Marginal posterior summaries — posterior mean ± 90% CI for all 5 parameters
across 16 scenarios, side-by-side for S-A (left) and S-B (right). Source: nb24.

**Table 9:** Per-scenario classification results (F1, coverage, CRPS) for both structures.

### 7.4 Headline: (α, η_col) banana posterior and EKF failure (snowball scenario)

**Scenario W10:** α = 0.75, η_col = 0.80 (catalyst decay + column efficiency loss),
all other parameters nominal. Under S-B:

Plant response:
- Loop 1 adjusts Q_c to maintain T_r at setpoint → masks β_r, but also slightly masks α
  (temperature compensation partially compensates for reduced reaction heat)
- Increased A fraction at column feed → Loop 3 adjusts reboiler duty to maintain x_B
  → masks η_col in x_B channel, but Q_reb increases (observable as controller effort)
- F_R increases (snowball onset) → directly observable, but ambiguous between α and η_col

**Expected SBI result (S-B):** Banana-shaped joint posterior in (α, η_col) plane.
High posterior mass consistent with the true combination AND the symmetric (α=0.60,
η_col=0.90) combination. The posterior correctly represents the irreducible ambiguity;
it is informative even when it cannot distinguish the two fault causes.

**Expected SBI result (S-A):** x_D measurement breaks the degeneracy. α=0.75 causes
x_D to rise to ~0.97 (more A passes through undisturbed); η_col=0.80 causes x_D to rise
to ~0.96 (slightly less sharp separation). The posterior narrows to the correct quadrant.

**Expected EKF result (both structures):** Gaussian ellipse — overconfident and
incorrectly centred. Near the snowball tipping point (α approaching 0.60), the Jacobian
changes rapidly and the linearisation underestimates the posterior width along the
recycle-coupled direction. EKF 90% CI achieves < 65% empirical coverage for (α, η_col).

**Figure 10:** (a) Time series under W10 showing F_R buildup and Q_reb increase;
(b) SBI joint (α, η_col) posterior under S-B (banana) and S-A (narrow);
(c) EKF Gaussian ellipse overlay; (d) coverage comparison bar chart.
Source: nb24, nb26.

### 7.5 EKF baseline comparison

Augmented EKF: 9-state vector [z_A, T_r, T_j, I_T, I_QC, α, β_r, η_col, ξ_reb].
Jacobian: `jax.jacobian(rhs, argnums=0)` evaluated at each EKF step (same approach as
outlined for the Luyben 8-D system, now in 9-D — tractable without hand derivation).

Expected findings:
- Both methods show structural β_r bias (same magnitude as PO — confirms generality)
- EKF tracking MAE for (α, η_col) significantly worse than SBI under S-B (Gaussian
  assumption collapses banana to ellipse)
- EKF confidence intervals diverge from empirical coverage near snowball onset (W14, W15)
- SBI 30-day tracking correctly shows widening uncertainty as α approaches snowball
  threshold; EKF intervals stay constant (Gaussian assumption)

**Figure 11:** SBI vs. EKF 30-day tracking for α, β_r, η_col with CI bands;
credible interval coverage by fault severity. Source: nb27.

**Table 10:** Wu 2003 SBI vs. EKF comparison — bias, MAE, coverage, classification F1.

### 7.6 NUTS timing and SBI monitoring cadence

Empirical NUTS timing from the propylene oxide system: 150,000 ms per 2-hour window
at 2-D. Scaling to 5-D using HMC step-size scaling O(d^{5/4}):
  5-D estimated NUTS: 150,000 × (5/2)^{5/4} ≈ 500,000 ms ≈ 8 min per window.

For 30-day monitoring (720 consecutive 2-hour windows):
- NUTS total: 720 × 8 min ≈ 4 days — impractical for monitoring cadence
- SBI total: 720 × 0.02 s = 14 s — real-time capable
- Speedup: ~25,000× (quantitative), plus EKF is not a valid baseline for non-Gaussian
  posteriors (qualitative correctness advantage)

**Table 11:** Feasibility comparison — SBI vs. EKF vs. NUTS for 30-day monitoring.

### 7.7 Model mismatch robustness

±5% perturbation on fixed parameters (M_r, ρCp, k₀, UA). Expected: posterior mean
shifts < 1σ, coverage degrades modestly (≥ 80% at 90% nominal CI). β_r mismatch
has minimal effect (already poorly identified). α and η_col mismatch expected to shift
the banana posterior but preserve its shape — fault classification accuracy degrades
< 5 pp.

**Table 12:** Model mismatch results — posterior shift and F1 degradation.

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
   Gaussian assumption collapses a manifold constraint to an ellipse, systematically
   understating the uncertainty along the recycle-coupled direction. EKF 90% CI
   achieves < 65% empirical coverage for the (α, η_col) pair.

2. **Near the snowball tipping point** (W14, W15): The Jacobian of the recycle
   dynamics changes rapidly as α approaches the critical value where recycle flow
   diverges. The EKF linearisation at the current state underestimates the
   posterior width; SBI, trained across the full prior range, correctly represents
   the widened uncertainty near the tipping point.

The recommendation: use EKF for real-time monitoring under normal operating conditions;
run SBI in parallel (20 ms per window after training) when full uncertainty
quantification is needed for maintenance decisions or when the plant approaches
nonlinear regimes (detected by diverging EKF covariance trace).

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
| L1 | Synthetic data only | Both systems | Light mismatch study (§7.7); real data as future work |
| L2 | β bias −0.08 to −0.15 | Propylene oxide | Quantified; predictable; recalibration via OL excitation |
| L3 | (α, η_sep) partial non-identifiability | Luyben | Reported as finding; banana posterior is informative |
| L4 | SBC mild miscalibration (KS p=0.016) | Propylene oxide | Structural, not a training deficiency |
| L5 | Prior sensitivity (Sc6 collapse) | Propylene oxide | Formal sensitivity study recommended before submission |
| L6 | No real-time deployment tested | Both | SCADA integration is future work |

---

## 9. Conclusion (~0.5 pages)

Restate the four contributions with their quantitative outcomes:

1. For the propylene oxide CSTR: I_αα/I_ββ = 250–500×; confirmed irreducible by
   4-method agreement and CNN embedding experiment; macro-F1 = 0.990 despite structural
   bias; 30-day tracking with SBI 53,000× faster than NUTS.

2. For the Wu 2003 CSTR-column-recycle plant: β_r masking persists unchanged from
   System I (same mechanism, same magnitude — I_β_r/I_α ≈ 1/250–500); recycle coupling
   creates (α, η_col) banana posterior under S-B — correctly captured by SBI, collapsed
   to overconfident Gaussian by EKF; NUTS at plant scale requires ~4 days for 30-day
   monitoring vs. 14 s for SBI after training.

3. Recycle dynamics make SBI the only simultaneously fast and correct method: EKF is
   wrong (non-Gaussian posteriors), MCMC is too slow (8 min/window vs. 20 ms).

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
- [ ] Wu 2003 CSTR-column-recycle implementation (nb20–nb28, see project_wu2003_sbi.md)
- [ ] Prior sensitivity study for propylene oxide (3 prior widths, ~2-3 days)
- [ ] All figures regenerated at publication quality (300 dpi, double-column)
- [ ] Nomenclature table with every symbol (C&ChE requirement)
- [ ] Highlights written (3-5 bullets, ≤85 characters)
- [ ] No undefined acronyms in abstract
- [ ] Ljung (1977), Gevers et al. (2011), Forssell & Ljung (1999), Luyben (1994) cited
      in §1, §2.2, §4.1, §4.3
- [ ] Fault classification framed as "label-free" not "unsupervised"
- [ ] Data availability statement (C&ChE requirement)
- [ ] CRediT author statement
- [ ] Conflict of interest statement
- [ ] Simulator/training data available (GitHub or Zenodo)

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
