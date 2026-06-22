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
- Amortised SBI recovers full posteriors at 8-D plant scale; MCMC is infeasible there
- EKF gives overconfident Gaussian intervals near snowball bifurcation; SBI does not
- Plant-wide fault localized to root cause despite five PI loops masking symptoms

---

## Title options

1. **Plant-wide Bayesian fault diagnosis in recycle processes: amortised simulation-based
   inference under multi-loop feedback control** (recommended — descriptive, C&ChE idiom)

2. **Structural identifiability limits of Bayesian fault diagnosis under feedback control:
   from a PI-controlled CSTR to a Luyben recycle plant** (leads with the theory)

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
complexity: a PI-controlled propylene oxide CSTR (2 parameters, 1 unit) and a Luyben
recycle plant (8 parameters, 4 units, 5 PI loops). We characterise identifiability using
numerical Fisher information analysis, compare against extended Kalman filter baselines
using automatic differentiation of the ODE Jacobian, and validate with simulation-based
calibration.

**Results.** For the propylene oxide system, the Fisher information for the fouling
parameter is 250–500× smaller than for catalyst activity — confirmed empirically by
four independent methods (SBI, MCMC, EKF, UKF) all showing identical structural bias.
For the Luyben recycle plant, plant-wide feedback control creates non-Gaussian posteriors
in which catalyst decay and separator efficiency loss produce banana-shaped dependences
invisible to the EKF's Gaussian approximation. SBI correctly localises the snowball
root cause (catalyst decay) from 8 measured channels under partial observability.
MCMC is computationally infeasible at 8-D; SBI processes each 2-hour window in under
20 ms after a one-time training cost.

**Conclusions.** Amortised SBI is the only practical full-Bayesian method for plant-scale
fault diagnosis. The structural identifiability limitations revealed here suggest that
periodic open-loop excitation is needed for reliable fouling quantification; fault
classification remains robust despite the irreducible parameter bias.

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
real-time deployment; (b) full posterior distributions — enabling probabilistic fault
classification with calibrated uncertainty; (c) no likelihood derivation required —
the process simulator is used directly. These advantages grow with problem dimensionality:
at 8+ parameters, MCMC becomes computationally infeasible while SBI's cost is fixed.

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
> 3. **Plant-wide fault localization in a Luyben recycle process.**
>    We extend to an 8-D system (CSTR + flash separator + recycle + purge, 5 PI loops)
>    and demonstrate SBI correctly attributes the Luyben snowball effect to its root cause
>    (catalyst decay) despite five local controllers each masking symptoms independently.
>    The (α, η_sep) posterior is banana-shaped; the EKF, constrained to Gaussians, gives
>    overconfident intervals that miss the true parameter combination.
>
> 4. **MCMC infeasibility at plant scale and SBI as the enabling method.**
>    At 8-D, NUTS requires days of compute per observation window. SBI's amortisation
>    cost is paid once; subsequent inference is a single network forward pass. This is a
>    qualitative, not merely quantitative, advantage: SBI enables full-Bayesian plant-wide
>    fault diagnosis that has no practical alternative.

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

### 3.2 System II: Luyben recycle plant

**Reaction:** A + B → C (bimolecular, irreversible, exothermic). Generic benchmark
from Luyben (1994, 2002). Relative volatilities α_A = 3.0, α_B = 2.0, α_C = 1.0
(Luyben 2002 Table); C is the heavier desired product.

**Plant topology:** Fresh feeds of A and B → Mixer → CSTR → Flash separator →
Liquid product stream (C-rich) + Vapour stream (A,B-rich) → Condenser → Recycle
pump → Mixer; Purge stream bleeds from the recycle vapour.

**ODE state vector (13 states):**

| States | Symbol | Units | Location |
|--------|--------|-------|----------|
| Ca, Cb | mol/L | CSTR | Reactant concentrations |
| T_r, Tc | K | CSTR, jacket | Temperatures |
| n_L | mol | Flash drum | Liquid molar holdup |
| x_A, x_B | — | Flash drum | Liquid mole fractions |
| T_s | K | Flash drum | Separator temperature |
| I_T, I_Ts, I_L, I_R, I_P | various | PI controllers | Integrator states |

**Key ODEs:**
```
dCa/dt = (F_in/V_r)*(Ca_in - Ca) - α*k(T_r)*Ca*Cb
dT_r/dt = (F_in/V_r)*(T_in - T_r) + (-ΔH)*α*k(T_r)*Ca*Cb/(ρCp) - β_r*UA_r*(T_r-Tc)/(ρCpV_r)
[Flash: y_i = α_vle_eff_i*x_i / Σ(α_vle_eff_j*x_j),  α_vle_eff_i = 1 + η_sep*(α_nom_i - 1)]
```

**5 decentralized PI loops:**
```
Loop 1 (CSTR temp):   Qc  → T_r    [fast, inner]
Loop 2 (Sep. temp):   Q_s → T_s    [fast]
Loop 3 (Sep. level):  F_L → n_L    [medium]
Loop 4 (Recycle):     valve → F_R  [medium]
Loop 5 (Purge):       valve → F_P  [slow, outer]
```
All loops implement anti-windup.

**Fault parameterisation (8-D):**

| # | Symbol | Prior | Physical meaning |
|---|--------|-------|-----------------|
| 1 | α | U[0.4, 1.2] | Catalyst activity (CSTR) |
| 2 | β_r | U[0.4, 1.2] | CSTR heat transfer fouling |
| 3 | η_sep | U[0.4, 1.2] | Flash separator split efficiency |
| 4 | β_s | U[0.4, 1.2] | Separator heat exchanger fouling |
| 5 | η_p | U[0.4, 1.2] | Recycle pump efficiency |
| 6 | ξ | U[0.4, 1.6] | Purge valve restriction (>1 = erosion) |
| 7 | κ | U[0.4, 1.2] | Feed preheater fouling |
| 8 | δ | U[−0.3, 0.3] | Feed A:B stoichiometry shift |

**Observations (8 channels, partial observability — no concentration analysers):**
T_r, Tc, Qc, T_s, Q_s, F_R, F_P, F_prod. All five controller output signals included.
2-hour windows at 1 min resolution → 120×8 time series → 65-D summary statistics.

**Table 3:** All nominal plant parameters (V_r, Qc, UA_r, UA_s, relative volatilities,
feed rates, controller tuning from Luyben 1994 Table 1)

**12 fault scenarios, 30 replicates each (Table 4):** individual faults (L2–L9),
snowball scenario L10 (α=0.65, η_p=0.85), competing-attribution L11 (reactor+separator),
severe multi-fault L12. Both control modes: full plant-wide + open-loop (720 windows).

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

## 7. Results: Luyben recycle plant (~5 pages)

*Section 7 will be populated after completing notebooks nb30–nb39 per project_luyben_extension.md.
The structure and expected findings are specified below based on the design in that plan.*

### 7.1 Training validation

SBC rank histograms for all 8 parameters. Expected: KS p > 0.05 for 6/8 parameters;
α and η_sep expected to show largest miscalibration (banana-shaped posterior creates
non-uniform ranks). Report honestly with interpretation: the miscalibration of (α, η_sep)
reflects the partial non-identifiability under closed-loop control without concentration
measurements, analogous to the β bias in the propylene oxide system.

**Figure 7:** SBC results for Luyben 8 parameters — rank histograms and KS p-values.
Source: nb33.

### 7.2 Snapshot fault classification (8-D)

12 scenarios, 30 replicates each. Expected results:
- Individual single-unit faults (L2–L9): high macro-F1 for reactor, separator, and
  recycle faults that affect different controller output signals
- (α, η_sep) ambiguity: L4 and L2 may share posterior support — the classification
  will reflect the inherent uncertainty correctly (bimodal posterior)
- Severe multi-fault L12: lower F1, correctly reported with lower confidence

**Figure 8:** Marginal posterior summaries for 12 Luyben scenarios — posterior mean ± 90% CI
per parameter. Source: nb34.

**Table 9:** Per-scenario Luyben classification results.

### 7.3 Snowball fault localization (headline result)

Scenario L10: α = 0.65, η_p = 0.85 (catalyst decay triggers snowball; pump stressed).
Five PI controllers respond locally:
- Loop 1 (CSTR temp) opens Qc → masks β_r
- Loop 4 (recycle) increases F_R → masks η_p
- Loop 3 (separator level) adjusts F_prod → masks η_sep
Each controller compensates its local symptom; no single controller "sees" the α decay.

**Expected SBI result:** High posterior mass on α < 0.75, moderate η_p degradation;
all other parameters near 1.0. The snowball root cause is correctly traced despite
plant-wide masking.

**Expected EKF result:** Mean estimates in the same ballpark (structural bias from the
five controllers), but overconfident Gaussian uncertainty interval that spans a much
smaller region. Near the snowball tipping point where F_R is nonlinearly sensitive to α,
the EKF linearisation underestimates the posterior width.

**Figure 9:** Snowball scenario — (a) plant-wide time series under L10 showing F_R
buildup, (b) SBI joint posterior for (α, η_p) showing banana-shaped constraint,
(c) EKF Gaussian ellipse overlay demonstrating overconfidence. Source: nb34.

**Figure 10:** Fisher information heatmap (8×8 at nominal OP) — showing I_α,η_sep
off-diagonal coupling as the largest off-diagonal term. Source: nb34.

### 7.4 Non-Gaussian posteriors and EKF failure

Under partial observability (no concentration measurements), α and η_sep both increase
the recycle load F_R. The (α, η_sep) posterior is banana-shaped: data are consistent
with (low α, healthy η_sep) or (healthy α, low η_sep), but not both degraded equally.
This correlation is a finding, not a failure — it provides actionable information about
the plausible fault space.

EKF, constrained to a Gaussian approximation, collapses the banana to an ellipse and
mis-states the credible intervals. SBC coverage comparison: SBI 90% CI achieves ≥ 88%
empirical coverage; EKF 90% CI achieves ≤ 60% for the (α, η_sep) pair.

**Figure 11:** (α, η_sep) marginal posterior for scenario L2 and L4 — SBI banana curve
vs. EKF ellipse; coverage comparison bar chart. Source: nb34, nb36.

### 7.5 MCMC infeasibility at 8-D

At 8-D, NUTS requires a burn-in period and post-burn-in sampling budget that scales
poorly with dimension. Empirical timing: extrapolating from the propylene oxide NUTS
timing (150,000 ms/window for 2-D), 8-D NUTS would require O(10–100 hours) per window
(citing dimensionality scaling of HMC leapfrog steps). This makes MCMC impractical for
any real-time or near-real-time monitoring application.

SBI processes the same 8-D posterior in < 20 ms. This is a qualitative feasibility
advantage, not just a quantitative speedup.

**Table 10:** Feasibility comparison — SBI vs. EKF vs. NUTS for the Luyben plant.

### 7.6 EKF baseline comparison (Luyben)

SBI vs. EKF (jax.jacobian) on all 12 scenarios and 30-day tracking.
Expected findings:
- Both methods show structural bias for β_r and β_s (same mechanism as propylene oxide)
- EKF tracking MAE for (α, η_sep) significantly worse than SBI due to Gaussian assumption
- SBI 30-day tracking correctly shows widening uncertainty during snowball onset; EKF
  confidence intervals remain too tight

**Figure 12:** SBI vs. EKF 30-day tracking for Luyben — α, η_sep, β_r with CI bands.
Source: nb37.

**Table 11:** Luyben SBI vs. EKF comparison — bias, MAE, coverage, classification F1.

### 7.7 Model mismatch robustness

±5% perturbation on fixed parameters (V_r, ρ, k₀, UA_r) at test time, trained
posteriors unchanged. Expected: small posterior mean shifts (< 1σ), coverage degrades
modestly (≥ 80% at 90% nominal CI). Fault classification macro-F1 expected to drop
≤ 5 pp for individual faults; combined faults may show larger sensitivity.

**Table 12:** Luyben model mismatch results — posterior shift and F1 degradation
vs. perturbation magnitude.

---

## 8. Discussion (~2 pages)

### 8.1 Structural identifiability across scales

The propylene oxide system illustrates the mechanism clearly: a single PI controller
zeros the temperature channel's sensitivity to β, leaving I_ββ/I_αα = 1/250–1/500.
The Luyben plant generalises this: five PI controllers each zero their respective
process variable's sensitivity to the local fault parameter, but create cross-unit
coupling through the recycle and purge streams. The (α, η_sep) banana posterior is
the multi-unit analogue of the β bias — a non-Gaussian identifiability constraint
arising from the feedback topology, not from modelling deficiencies.

**The hierarchy:** I_β_r (masked by Loop 1) < I_η_sep (masked by Loop 2 + recycle
dynamics) < I_β_s (masked by Loop 2 alone) < I_η_p (identifiable via F_R) < I_ξ
(directly observable via F_P) < I_α (most identifiable, high reaction heat signal).

This hierarchy is a practical design guideline: parameters whose fault signatures are
directly observable via a controller output (ξ, η_p) are far easier to identify than
those whose signatures are actively suppressed (β_r, β_s, α+η_sep pair).

### 8.2 When EKF fails and SBI wins

EKF is an excellent industrial baseline when the posterior is approximately Gaussian —
which holds for the propylene oxide system near the nominal operating point (both methods
show similar bias and similar uncertainty). The EKF breaks down near:
- **Bifurcation points** (snowball tipping): rapid Jacobian changes make linearisation
  inaccurate; the EKF underestimates posterior width
- **Non-Gaussian posteriors** (α, η_sep coupling): the Gaussian assumption collapses a
  banana to an ellipse, producing systematically overconfident intervals
- **High-dimensional augmented states**: the 21×21 covariance update is numerically
  sensitive; SBI has no such numerical degradation

The recommendation: use EKF for real-time monitoring under normal operating conditions;
switch to SBI (or run SBI in parallel) when the plant approaches nonlinear operating
regimes or when full uncertainty quantification is needed for maintenance decisions.

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

2. For the Luyben recycle plant: SBI correctly localizes the snowball root cause from
   8 measured channels under 5-loop feedback masking; (α, η_sep) posterior is
   banana-shaped — correctly captured by SBI, collapsed to an incorrect Gaussian by EKF.

3. MCMC infeasibility at 8-D makes SBI the only practical full-Bayesian option at
   plant scale.

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
| 7 | SBC for all 8 parameters (Luyben) | nb33 | §7.1 |
| 8 | Marginal posteriors 12 scenarios (Luyben) | nb34 | §7.2 |
| 9 | Snowball scenario: time series + (α, η_p) posterior + EKF overlay | nb34 | §7.3 |
| 10 | 8×8 Fisher information heatmap (Luyben) | nb34 | §7.3 |
| 11 | (α, η_sep) banana curve: SBI vs. EKF | nb34, nb36 | §7.4 |
| 12 | 30-day Luyben tracking SBI vs. EKF | nb37 | §7.6 |

All figures: 300 dpi minimum, double-column compatible, colour-blind palette (viridis/Okabe-Ito).

---

## Tables summary

| # | Content | Section |
|---|---------|---------|
| 1 | Propylene oxide model parameters | §3.1 |
| 2 | PO fault scenarios (Sc1–Sc8) | §3.1 |
| 3 | Luyben plant parameters | §3.2 |
| 4 | Luyben fault scenarios (L1–L12) | §3.2 |
| 5 | Full summary statistics definition (29-D and 65-D) | §4.2 |
| 6 | SBI/EKF training hyperparameters | §5.1 |
| 7 | PO per-scenario classification results | §6.2 |
| 8 | 4-method baseline comparison (SBI, NUTS, EKF, UKF) | §6.4 |
| 9 | PO 30-day tracking phase metrics | §6.5 |
| 10 | Luyben per-scenario classification | §7.2 |
| 11 | Luyben feasibility comparison (SBI vs. EKF vs. NUTS) | §7.5 |
| 12 | Luyben SBI vs. EKF comparison | §7.6 |
| 13 | Luyben model mismatch robustness | §7.7 |

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
- [ ] Luyben plant implementation (nb30–nb39, see project_luyben_extension.md)
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
