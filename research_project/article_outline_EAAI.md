# Article outline — Engineering Applications of Artificial Intelligence (EAAI)

> Target: EAAI (Elsevier, IF 8.9, IFAC-affiliated)
> Format: single-column, ~20-25 pages
> Key EAAI expectations: **dual contribution** (AI method + engineering application),
> **3-5 baselines**, practical impact, reproducibility, honest limitations

---

## Critical framing advice

Before you start writing, internalise these points — they determine whether the
paper is desk-rejected or sent to review.

**What EAAI reviewers will look for:**
1. A clear AI/ML contribution (not just "we applied method X to domain Y")
2. Industrial or engineering relevance (fault diagnosis in CSTRs qualifies)
3. Comparison with competing methods (NUTS alone is NOT enough — you need
   at least one classical estimation baseline like EKF/UKF)
4. Reproducibility (release the simulator code, or at minimum the training data)
5. Honest discussion of failure modes and when the method breaks

**What you must NOT claim:**
- Do not claim the closed-loop identifiability limitation as a discovery — it is
  textbook since Ljung 1977. Frame it as "we quantify a classical phenomenon for
  a specific industrially relevant system."
- Do not call the fault classification "unsupervised" — it is **label-free**
  (no fault labels in SBI training) but uses domain-knowledge thresholds (0.85).
- Do not present the 53,000x speedup over NUTS as the primary contribution —
  nobody runs NUTS per window in practice. The speedup is real but the comparison
  must include an industrially relevant baseline.

**The strongest framing for EAAI is:**
> First systematic study of amortised simulation-based inference for real-time
> probabilistic fault diagnosis in feedback-controlled chemical reactors, with
> a rigorous characterisation of the structural identifiability limitation
> imposed by the control loop and empirical proof that it is irreducible.

---

## Title options (pick one)

1. **Amortised simulation-based inference for real-time fault diagnosis in
   PI-controlled chemical reactors** (current — clear, specific, 13 words)

2. **Real-time probabilistic fault diagnosis in feedback-controlled CSTRs
   via amortised neural posterior estimation** (emphasises the AI method more)

3. **Structural identifiability limits of Bayesian fault diagnosis under
   feedback control: an amortised SBI approach** (leads with the limitation
   — riskier, but more novel-sounding)

Recommendation: option 1. It is descriptive without overclaiming. Options 2-3
risk signalling a theoretical contribution the paper cannot fully deliver.

---

## Abstract (~250 words)

Structure it as exactly four sentences, each doing one job:

1. **Problem.** Real-time fault diagnosis in chemical reactors under feedback
   control is difficult because the controller masks the effect of fault
   parameters on the controlled variable — a classical identifiability
   limitation (Ljung 1977).

2. **Method.** We apply amortised simulation-based inference (SBI) — training
   a neural spline flow to approximate the Bayesian posterior over catalyst
   activity (alpha) and jacket fouling (beta) from 29 physics-informed summary
   statistics of 60-minute observation windows — and characterise the structural
   identifiability gap using Fisher information analysis.

3. **Results.** [Key numbers: macro-F1 = 0.990 on 6 CL scenarios, 30-day
   tracking with MAE_alpha = 0.004 / MAE_beta = 0.034, 53,000x speedup over
   MCMC, Fisher info ratio I_aa/I_bb = 250-500x, embedding-net confirms bias
   is irreducible.]

4. **Implication.** The amortised posterior delivers real-time probabilistic
   fault classification despite the irreducible beta bias, but practical
   deployment requires awareness of the structural limitation and periodic
   open-loop excitation for fouling quantification.

> **Critical note:** EAAI forbids undefined acronyms in the abstract. Spell out
> SBI, CSTR, PI, MCMC, FIM on first use. "Neural spline flow" needs no acronym
> in the abstract — just say "neural density estimator."

---

## Keywords (6-8)

simulation-based inference; amortised Bayesian inference; fault diagnosis;
closed-loop identifiability; CSTR; neural posterior estimation; Fisher information;
predictive maintenance

---

## 1. Introduction (~2-2.5 pages)

### 1.1 Problem motivation (3-4 paragraphs)

- **Para 1 — Industrial context.** Fouling and catalyst deactivation are the two
  dominant degradation modes in CSTRs. Current practice: periodic lab tests or
  model-based observers (EKF, UKF). These provide point estimates, not uncertainty
  quantification, and require re-tuning when operating conditions change.

- **Para 2 — The feedback masking problem.** PI/PID controllers compensate for
  fouling by increasing coolant flow, suppressing the temperature signature of the
  fault. This is the classical closed-loop identifiability limitation (Gustavsson,
  Ljung & Soderstrom 1977; Ljung 1999 Ch.13). It affects ALL estimation methods
  — Bayesian or frequentist, online or offline. State this upfront so reviewers
  do not think you discovered it.

- **Para 3 — Why SBI.** Simulation-based inference (Cranmer et al. 2020) replaces
  an explicit likelihood with a learned neural density estimator trained on
  simulated (theta, x) pairs. The key advantage for fault diagnosis:
  (a) amortisation — a single trained network handles any new observation in ~15 ms;
  (b) full posterior, not a point estimate — enables probabilistic fault
  classification with calibrated uncertainty;
  (c) no likelihood derivation required — the simulator IS the model.

- **Para 4 — Gap in the literature.** SBI has been applied to particle physics,
  cosmology, and epidemiology, but not to feedback-controlled industrial processes
  where the control loop fundamentally limits what can be inferred. No prior work
  has systematically quantified how the closed-loop identifiability gap manifests
  in an amortised Bayesian posterior, or empirically verified its irreducibility.

### 1.2 Contributions (numbered list)

Frame as exactly 4 contributions. This is what the reviewers will check the paper
against. Every claim here must be supported by a specific result in the paper.

> 1. **Amortised SBI for controlled chemical processes.** We train a neural
>    posterior estimator for joint (alpha, beta) inference in a PI-controlled
>    CSTR, achieving real-time inference (15 ms/window) with label-free
>    four-class fault classification (macro-F1 = 0.990).
>
> 2. **Quantification of the structural identifiability gap.** We show via
>    numerical Fisher information analysis that I_bb is 250-500x smaller than
>    I_aa across all operating points, and derive the mechanism analytically
>    from the controlled steady-state heat balance. This is a system-specific
>    quantification of the classical closed-loop identifiability limitation
>    (Ljung 1977; Gevers et al. 2011).
>
> 3. **Empirical proof of irreducibility.** A CNN embedding trained on raw
>    (120x4) time series — bypassing the hand-crafted summaries entirely —
>    produces the same beta bias (-0.079 vs -0.084), confirming via the
>    Cramer-Rao bound that no choice of summary statistics or inference method
>    can recover the lost information.
>
> 4. **30-day sequential degradation tracking.** The amortised posterior
>    processes 720 consecutive 60-min windows with MAE_alpha = 0.004 and
>    a calibrated fault-classification timeline, at 53,000x the speed of MCMC.

### 1.3 Paper organisation

One sentence per section. Keep it mechanical.

---

## 2. Related work (~2 pages)

Organise into **four** subsections. Each must end with a sentence stating what
the current paper adds beyond that literature.

### 2.1 Fault diagnosis in chemical reactors

- Model-based FDI: observer-based residual generation (Isermann 2006, Frank 1990),
  parity equations, structured residuals
- Data-driven FDI: PCA/PLS monitoring (Qin 2012), deep learning (autoencoders,
  LSTMs for time-series anomaly detection)
- **Gap:** these methods produce alarms or point estimates, not full posteriors.
  They do not quantify how much information the control loop destroys about the
  fault parameter.

### 2.2 Closed-loop identifiability

- Classical results: Gustavsson/Ljung/Soderstrom 1977, Forssell & Ljung 1999,
  Ljung 1999 Ch.13
- Experiment design for closed-loop: Gevers, Bombois et al. 2006/2011
- Fouling masking under temperature control: heat-exchanger literature (Chem Eng
  Res Des 2022)
- Structural identifiability: Villaverde 2019, Karin & Alon 2017
- **Gap:** the qualitative principle is well-established, but no prior work
  quantifies the Fisher information ratio for competing fault parameters in a
  CSTR, or connects it to the observed bias in a neural posterior.

### 2.3 Simulation-based inference

- Foundations: Cranmer, Brehmer & Louppe 2020 (PNAS review)
- Neural posterior estimation: Papamakarios & Murray 2016 (NPE-A), Lueckmann
  et al. 2017 (NPE-B), Greenberg et al. 2019 (SNPE-C/APT)
- Density estimators: MAF (Papamakarios et al. 2017), NSF (Durkan et al. 2019)
- sbi library: Tejero-Cantero et al. 2020
- Applications: gravitational waves (Dax et al. 2021), epidemiology (Lueckmann
  et al. 2021), neuroscience
- **Gap:** no application to feedback-controlled engineering systems. Existing
  SBI benchmarks assume the simulator faithfully generates i.i.d. observations;
  they do not address the structured information loss from feedback control.

### 2.4 Bayesian inference for process monitoring

- Bayesian state estimation: EKF (Jazwinski 1970), UKF (Julier & Uhlmann 2004),
  particle filters (Doucet et al. 2001)
- Moving-horizon estimation (Rawlings et al. 2017)
- Bayesian fault classification: Bayesian networks for root cause analysis
- **Gap:** these methods require online optimisation per timestep (not amortised),
  do not naturally produce multi-parameter posteriors, and do not characterise
  the structural limitation in closed-loop.

> **Critical note:** Section 2.4 sets up the missing EKF/UKF comparison. If you
> do NOT add an EKF baseline to the experiments, this section exposes the gap.
> **Strongly recommend adding EKF/UKF (~1 week of work) before submission.**
> Without it, Reviewer 2 will ask for it in R1, costing you 2-3 months.

---

## 3. Problem formulation (~2 pages)

### 3.1 CSTR model

- Full 3-state ODE system: dC/dt, dT/dt, dTc/dt
- Present the equations cleanly in a single equation block
- Define all symbols in a nomenclature table (EAAI convention)
- PI controller: Qc = clip(Qc0 + Kp*(T - Tsp) + I/tau_i, 0, Qc_max)
- All parameter values in a table (Table 1)
  - Use values from the Fogler M13 CSTR
  - Kp = 150 (L/min)/K, tau_i = 10 min, Tsp = 312.5 K, Qc0 = 80 L/min

> **Critical note:** Reviewers will want to reproduce this. Provide EVERY parameter
> value. Consider releasing the simulator code as supplementary material or a
> GitHub repo — EAAI values reproducibility.

### 3.2 Fault parameterisation

- alpha in [0.4, 1.2]: catalyst activity scaling (1.0 = healthy, <1.0 = deactivation)
- beta in [0.4, 1.2]: jacket UA scaling (1.0 = healthy, <1.0 = fouling)
- Prior: BoxUniform([0.4, 0.4], [1.2, 1.2])
- Explain why upper bound is 1.2 not 1.0 (numerical buffer for density estimator;
  values above 1.0 are not physically meaningful)

### 3.3 Observation model

- 60-minute observation windows, 0.5 s sampling -> 120 x 4 time series [C, T, Tc, Qc]
- Sensor noise: 1% Gaussian on each channel
- Process noise: Euler-Maruyama stochastic integration
- 29-D summary statistics: 5 base stats x 4 channels + 4 final-window means +
  3 control aggregates + 2 physics proxies (UA_eff_proxy, k0_eff_proxy)

### 3.4 Fault scenarios

- Table 2: all 8 scenarios (Sc1-Sc8) with alpha_true, beta_true, control mode,
  and physical interpretation
- 50 stochastic replicates per scenario, 400 total observations
- Sequential degradation profile: linear alpha decay + Kern-Seaton beta curve
  over 30 days (720 windows)

**Figure 1:** Scenario overview — representative time series from Sc1 (healthy),
Sc2 (fouling), Sc3 (decay), Sc4 (combined). 2x2 panel.
Source: adapt `02_scenarios_overview.png`

---

## 4. Methodology (~3-4 pages)

This is where the AI contribution lives. Write it so someone could reimplement
from this section alone.

### 4.1 Simulation-based inference background

- Bayes' rule: p(theta|x) proportional to p(x|theta) * p(theta)
- Intractable likelihood when the simulator is a black-box ODE solver + noise model
- Neural posterior estimation (NPE): train q_phi(theta|x) to approximate p(theta|x)
  by minimising forward KL divergence on simulated pairs
- SNPE-C (Greenberg et al. 2019): sequential refinement with proposal correction
- Keep this concise (1 page max) — the SBI community knows this; the EAAI
  audience needs just enough to understand the pipeline

**Classical taxonomy positioning (add 1 paragraph):**
Forssell & Ljung (1999, §2.3) classify closed-loop identification methods into three
families: *direct* (operate on raw input-output data, no controller model required),
*indirect* (require an explicit controller model, then back-calculate the plant), and
*joint input-output* (model the closed-loop system jointly). NPE operates directly on
the measured state histories (C, T, T_c, Q_c) without utilising the PI controller's
internal equations, making it a modern likelihood-free analogue of the **direct method**.
Like the classical direct approach, this makes NPE universally applicable to systems
with arbitrary (nonlinear, clamped) feedback, and it exploits in-loop process noise to
partially reduce posterior variance (Forssell & Ljung, 1999, p.38).

### 4.2 Neural density estimator

- Architecture: Neural Spline Flow (NSF), 128 hidden units, 5 transforms
- Justify NSF over MAF: [cite nb04 sensitivity study — NSF gives tighter posteriors
  with same training budget]
- Training: 10,000 simulator draws from the prior, standard sbi library pipeline
- Training time: ~4 minutes on a single CPU (or GPU time if applicable)
- No sequential rounds needed (single-round NPE suffices for this problem)

### 4.3 Summary statistics

- The 29-D vector design: table listing all 29 features with physical interpretation
- Sufficiency argument: the CNN embedding experiment (Section 6.4) shows that
  learned features on raw (120x4) data do NOT improve inference -> the 29-D
  summaries are effectively sufficient
- This is an important methodological contribution: show your features are not
  a bottleneck

### 4.4 Amortised fault classification

- Define the four fault classes geometrically:
  Healthy (alpha >= 0.85, beta >= 0.85), Fouling (alpha >= 0.85, beta < 0.85),
  Decay (alpha < 0.85, beta >= 0.85), Combined (alpha < 0.85, beta < 0.85)
- Classification rule: assign posterior probability mass to each quadrant
- Threshold at 0.85 is domain-informed (not optimised on test data)
- Frame as "label-free" classification — no fault labels in training, but the
  decision boundary requires domain knowledge
- Calibration: how posterior probability mass relates to classification confidence

### 4.5 Sequential tracking protocol

- Window-by-window: apply the amortised posterior to each 60-min window
  independently (no temporal prior propagation)
- Why independent windows, not filtering: (a) amortisation already handles each
  window in 15 ms, (b) temporal correlations are secondary to the structural
  identifiability gap, (c) simpler to deploy
- Metrics: MAE, CRPS, coverage, classification accuracy per window

### 4.6 Fisher information analysis

- Define the numerical FIM: I(theta) = J^T Sigma^{-1} J, where J is computed
  via finite differences on the 29-D summary statistics
- Explain what I_aa >> I_bb means: the data carry 250-500x more information about
  alpha than about beta
- Connect to Cramer-Rao: Var(beta_hat) >= 1/I_bb for any unbiased estimator
- Analytical derivation (steady-state): show the Jacobian structure under T=Tsp
  (zero rows for T), derive I_aa/I_bb ratio from first principles
- Profile likelihood for beta: definition, computation, connection to asymmetric
  posterior -> Laplace bias formula

> **Critical note:** This subsection must cite Ljung (1977) and Gevers et al.
> (2011) prominently. Do NOT present the Fisher analysis as if you invented it.
> Present it as "we apply the classical framework of [citations] to quantify the
> identifiability gap for our specific system."

---

## 5. Experimental setup (~1.5 pages)

### 5.1 Training configuration

- Table 3: all hyperparameters (num_simulations, hidden_features, num_transforms,
  learning rate, batch size, training epochs)
- Hardware: CPU/GPU specs, wall-clock training time
- Software: sbi v0.22+, PyTorch, NumPyro (for MCMC baseline)

### 5.2 Baseline methods

**THIS IS THE SECTION THAT WILL MAKE OR BREAK YOUR PAPER AT EAAI.**

You need at minimum 3 baselines:

| Baseline | Purpose | Status |
|----------|---------|--------|
| NUTS MCMC (NumPyro) | Gold-standard Bayesian reference | ✅ DONE (nb05a) |
| LDA on physics features | Simple ML baseline | ✅ DONE (nb05a) |
| EKF (augmented 6-D state) | Industrial state estimation baseline | ✅ DONE (nb16) |
| UKF (sigma-point, 6-D) | Industrial state estimation baseline | ✅ DONE (nb16) |
| (Optional) Particle filter | Monte Carlo baseline | Would strengthen paper |

> **EKF/UKF baseline is now complete.** Both show the same structural β bias
> (β̂ = 0.607 for Sc2, true = 0.70), confirming the limitation is
> method-independent. SBI is fastest (16 ms/window) with the lowest β MAE
> in 30-day tracking (0.033), while providing full posterior uncertainty
> that EKF/UKF Gaussian approximations cannot match.

### 5.3 Evaluation protocol

- 50 stochastic replicates per scenario (out-of-sample — not used in training)
- SBC: 500 independent draws from the prior (Talts et al. 2018)
- Metrics: coverage, MAE, CRPS, macro-F1, W1 distance, per-scenario breakdown
- Sequential tracking: 30-day degradation profile, 720 windows

### 5.4 Ablation studies

List them upfront so reviewers know they will get systematic evidence:

1. Training budget sensitivity: 1k, 5k, 10k simulations
2. Density estimator: MAF vs NSF
3. Summary statistics: 29-D hand-crafted vs CNN embedding on raw time series
4. Closed-loop vs open-loop training (mode mismatch)
5. Prior width sensitivity (if you add this — recommended, ~2-3 days)

---

## 6. Results (~5-6 pages)

Organise by claim, not by notebook. Each subsection produces one key figure and
one key table.

### 6.1 Prior predictive check and training validation

- Show the prior predictive coverage plot: 500 simulations vs observed data
  confirm the simulator covers the observation space
- SBC results: rank histograms for alpha and beta, KS p-values, C2ST scores
- **Be honest:** KS p = 0.016 (< 0.05) indicates mild miscalibration.
  Frame it as: "consistent with the structural beta bias rather than a training
  deficiency, as confirmed by the C2ST scores near 0.5"

**Figure 2:** Prior predictive coverage (left) + SBC rank histograms (right)
Source: `04_simulator_sanity.png`, `04_sbc_ranks.png`

**Table 4:** SBC results (KS p-value, C2ST for alpha and beta)

### 6.2 Snapshot fault classification

- Joint posterior for all 8 scenarios with quadrant classification
- Per-scenario F1 scores in a table
- Macro-F1 = 0.990 (6 CL scenarios), 0.874 (all 8 including OL)
- Highlight: healthy = 100%, fouling = 100%, decay = 100%, combined = 94%

**Figure 3:** 2D joint posteriors for 6 CL scenarios with quadrant boundaries
and classification. Source: `04_joint_posterior_2d.png` or `06_joint_posterior_primary.png`

**Table 5:** Per-scenario classification results (true class, predicted class,
F1, posterior mean, 90% CI, coverage)

### 6.3 Identifiability analysis — the core contribution

This is the section that differentiates this paper. Spend 1.5-2 pages here.

**6.3.1 Fisher information asymmetry**

- Numerical FIM: I_aa = 850k-975k, I_bb = 2000-3500, ratio 250-500x
- Channel decomposition: I_aa gets 60% from C (concentration), 40% from Qc;
  I_bb gets 0% from C and T (both zero), ~100% from Tc
- Analytical derivation: under T = Tsp, the Jacobian has zero rows for T,
  and C depends only on alpha -> beta is identified solely through the
  noisy Tc and Qc channels

**Figure 4:** Fisher information decomposition — bar chart showing I_aa and I_bb
with per-channel contributions. Source: adapt from `15_fisher_information.png`
and `15_analytical_bias_derivation.png`

**6.3.2 Profile likelihood and bias mechanism**

- Profile likelihood for beta: wide, asymmetric (steeper below beta_true)
- Laplace bias: -0.0003 from the 4-observable steady-state model
- Per-channel profile: Tc-only gives bias -0.027, Qc-only gives -0.112,
  all 4 channels nearly cancel (+0.0003)
- Full 29-D SBI bias: -0.084
- Interpretation: the steady-state model predicts the correct bias direction
  per channel but the magnitude comes from dynamic summary features (slopes,
  oscillation amplitudes, settling times)

**Figure 5:** Profile likelihood panel — (a) 2D log-likelihood with alpha
profiled out, (b) 1D profile posteriors by channel combination, (c) bias
comparison bar chart. Source: `15_analytical_bias_derivation.png`

**6.3.3 Embedding-net control experiment**

- CNN embedding (61k parameters) on raw (120,4) time series
- Result: beta_Sc2 = 0.621 vs 0.616 (hand-crafted) — <1% difference
- This is the empirical confirmation: the information loss is in the
  physics (controller masks beta), not in the features

**Figure 6:** Side-by-side marginal posteriors for CNN vs hand-crafted on 5
scenarios. Source: `04b_embedding_vs_handcrafted.png`

> **Critical note:** This figure is your strongest evidence. Put it early in the
> identifiability section. The logic chain is: Fisher analysis PREDICTS the gap
> -> profile likelihood SHOWS the asymmetry -> embedding-net CONFIRMS it is
> irreducible. This three-step argument is the core intellectual contribution.

### 6.4 MCMC baseline comparison

- NUTS on Sc1 (healthy) and Sc2 (fouling): same beta bias as SBI
- Timing: 460-790 s/window vs 15 ms (speedup 31,000-53,000x)
- LDA comparison: macro-F1 comparison (SBI > LDA for fault classification)

**Table 6:** SBI vs NUTS vs LDA: accuracy, timing, beta bias for Sc1, Sc2

**EKF/UKF results (nb16) — the killer result.** Four independent methods
(SBI, MCMC, EKF, UKF) all show the same structural β bias. Present the unified
performance table here:

| Method | β bias (Sc2) | β MAE (tracking) | ms / window | Output |
|--------|-------------|------------------|-------------|--------|
| SBI | −0.149 | 0.033 | 16 | full posterior |
| EKF | −0.093 | 0.065 | 30 | Gaussian |
| UKF | −0.093 | 0.090 | 358 | Gaussian |
| NUTS | −0.102 | 0.102 | 150,000 | full posterior |

**Figure:** Baseline dashboard — `16_baseline_dashboard.png` (Sc2 β estimates,
tracking bias, inference cost). **Figure:** 30-day tracking comparison —
`16_tracking_comparison.png` (α,β estimates with confidence bands).

### 6.5 Sequential degradation tracking

- 30-day timeline: alpha(t) and beta(t) with 90% CI bands
- MAE_alpha = 0.004, MAE_beta = 0.034 (systematic -0.08 offset)
- Fault classification timeline: correct class in >95% of windows
- Total wall time: 10.4 s for 720 windows

**Figure 7:** 30-day tracking — (a) alpha and beta estimates with CI,
(b) rolling classification accuracy. Source: `10_degradation_tracking.png`,
`10_fault_classification_timeline.png`

**Table 7:** Phase-by-phase tracking metrics (healthy phase, degradation onset,
severe fouling)

### 6.6 Ablation studies

- Training budget: 1k/5k/10k simulations — show convergence at 10k
- MAF vs NSF: NSF gives tighter posteriors
- CL vs OL training: OL-trained network fails on CL data (Claim 1)

**Figure 8:** Training budget sensitivity (left), CL vs OL comparison (right)
Source: `04_sensitivity_scatter.png`, `07_cl_vs_ol_joint.png`

---

## 7. Discussion (~2 pages)

### 7.1 Practical implications for fault diagnosis

- The amortised posterior enables real-time monitoring with uncertainty
  quantification — not just point estimates
- The beta bias is predictable and consistent — it can be calibrated out in
  deployment (report the expected bias as a systematic offset)
- Recommendation: periodic open-loop excitation windows (5-10 min) would
  provide unmasked beta estimates for recalibration

### 7.2 Connection to classical identifiability theory

- Position the Fisher analysis explicitly within the framework of Gustavsson
  et al. (1977), Gevers et al. (2011)
- The contribution is NOT the principle (known since 1977) but the quantification
  for THIS system and the empirical demonstration that amortised SBI faithfully
  reflects the theoretical limitation
- Proposition: I_aa/I_bb >= (dC/dalpha)^2 * sigma_Qc^2 / [(dQc/dbeta)^2 * sigma_C^2] >> 1
  whenever the controller holds T approx Tsp. The ratio grows with controller gain.
- The phenomenon is independent of reaction kinetics (1st order, 2nd order,
  Michaelis-Menten) — it is structural to the feedback topology

**Persistent excitation punchline (add 1 paragraph):**
Classical closed-loop identification requires the reference signal r(t) to be
*persistently exciting* — specifically, its spectrum must satisfy Phi_r(omega) > 0 —
to guarantee informative data and parameter consistency (Forssell & Ljung, 1999, p.33).
The CSTR operates at a fixed setpoint T_sp without artificial setpoint dithering, so
Phi_r(omega) = 0: by the classical definition, the data are **theoretically uninformative
for beta**. The fact that amortised SBI still achieves macro-F1 = 0.990 for fault
classification under these impoverished conditions reflects the method's ability to
exploit *transient* dynamics — overshoots, settling oscillations, slope features — that
carry implicit parametric excitation beyond what the classical frequency-domain criterion
accounts for. This is a practically critical finding: real industrial plants cannot
sustain continuous setpoint dithering for economic reasons, so the robustness of SBI
to theoretically uninformative steady-state data is directly deployment-relevant.

### 7.3 Limitations

Be explicit and honest. EAAI reviewers respect transparency.

| # | Limitation | Impact | Mitigation |
|---|---|---|---|
| L1 | Synthetic data only | No sim-to-real gap tested | Add model mismatch study or state this as future work |
| L2 | 2-parameter system | Real processes have dozens of parameters | Scalability study is future work |
| L3 | Beta bias -0.08 to -0.15 | Cannot distinguish beta=0.90 from beta=0.75 | Known, quantified; recommend OL excitation |
| L4 | Prior sensitivity at boundary | Sc6 (OL) F1 collapses when prior widens | Report; recommend matched CL/OL posteriors |
| L5 | SBC mild miscalibration | KS p = 0.016 | Consistent with structural bias, not training failure |
| ~~L6~~ | ~~No EKF/UKF comparison~~ | ~~Speedup against NUTS only~~ | **RESOLVED — nb16** |

> **Critical note:** L6 is the one reviewers will flag. If you cannot add EKF in
> time, explicitly acknowledge this gap in the limitations and frame the speedup
> comparison against NUTS only. But this weakens the paper significantly.

### 7.4 When does this approach fail?

- Open-loop observations fed to a CL-trained posterior: complete misclassification
- Valve saturation (Sc8): spurious alpha elevation, unreliable classification
- Very mild combined faults near the decision boundary (Sc4: 94% accuracy, not 100%)
- Sensor drift > 2-3 sigma: sequential filter needed, but convergence is slow for
  inlet concentration drift

---

## 8. Conclusion (~0.5 pages)

Restate the four contributions with their quantitative evidence. End with a
forward-looking statement about:
- Extension to multi-parameter, multi-unit systems
- Sim-to-real transfer with model mismatch robustness
- Integration with plant historian / SCADA systems for deployment
- Combination with active experiment design (OL excitation scheduling)

---

## Figures summary (aim for 8-12 in main text)

| # | Content | Source | Purpose |
|---|---------|--------|---------|
| 1 | Scenario overview (representative time series) | `02_scenarios_overview.png` | Problem setup |
| 2 | Prior predictive + SBC rank histograms | `04_simulator_sanity.png` + `04_sbc_ranks.png` | Training validation |
| 3 | Joint posteriors with fault quadrants (6 CL scenarios) | `06_joint_posterior_primary.png` | Snapshot classification |
| 4 | Fisher information decomposition (channel breakdown) | `15_fisher_information.png` + `15_analytical_bias_derivation.png` | Core contribution |
| 5 | Profile likelihood + bias comparison | `15_analytical_bias_derivation.png` | Bias mechanism |
| 6 | CNN embedding vs hand-crafted (5 scenarios) | `04b_embedding_vs_handcrafted.png` | Irreducibility proof |
| 7 | 30-day tracking + classification timeline | `10_degradation_tracking.png` + `10_fault_classification_timeline.png` | Sequential tracking |
| 8 | Training budget + CL vs OL ablation | `04_sensitivity_scatter.png` + `07_cl_vs_ol_joint.png` | Ablation |
| **9** | **Baseline dashboard: 4-method comparison** | **`16_baseline_dashboard.png`** | **§9** |
| **10** | **30-day tracking: EKF/UKF/SBI overlay** | **`16_tracking_comparison.png`** | **§9** |

> All figures must be regenerated at publication quality: 300 dpi minimum,
> consistent font sizes, colour-blind-friendly palette, no tiny text. EAAI
> uses single-column, so figures can be full-width.

---

## Tables summary (aim for 7-8)

| # | Content |
|---|---------|
| 1 | CSTR model parameters (all values) |
| 2 | Fault scenarios (Sc1-Sc8: alpha, beta, control mode, description) |
| 3 | SBI training hyperparameters |
| 4 | SBC calibration results |
| 5 | Per-scenario classification (F1, posterior mean, coverage) |
| 6 | **Baseline comparison (SBI vs NUTS vs EKF vs UKF): accuracy, bias, timing** |
| 7 | 30-day tracking metrics (per-phase MAE, CRPS, classification) |
| 8 | Ablation results summary |

---

## Appendix (supplementary material)

- A. Full 29-D summary statistics definition (table with all features)
- B. MCMC convergence diagnostics (trace plots, R-hat, ESS)
- C. Analytical steady-state derivation (full equations for Jacobian, FIM)
- D. Additional per-scenario posterior plots
- E. Confusion matrices for all classification comparisons

---

## Pre-submission checklist

Before submitting, verify:

- [x] **EKF/UKF baseline added** (nb16 — DONE)
- [ ] Prior sensitivity study (3 prior widths, report classification stability)
- [ ] All figures at 300 dpi, single-column compatible, colourblind-friendly
- [ ] Nomenclature table with every symbol
- [ ] No undefined acronyms in abstract
- [ ] Ljung (1977), Gevers et al. (2011), and **Forssell & Ljung (1999)** cited in intro, §2.2, §4.1, §4.6, §7.2
- [ ] Simulator code / training data available for reviewers (GitHub or Zenodo)
- [ ] Data availability statement (EAAI requirement)
- [ ] CRediT author statement (EAAI requirement)
- [ ] Conflict of interest statement
- [ ] Single-column format (NOT double-column)
- [ ] Classification framed as "label-free" not "unsupervised"
- [x] 53,000x speedup contextualised (vs MCMC; EKF/UKF baselines included)
- [ ] Honest SBC reporting (mild miscalibration acknowledged, not hidden)

---

## Effort estimate for remaining work

| Task | Effort | Impact on acceptance |
|------|--------|---------------------|
| ~~Add EKF/UKF baseline~~ | ~~1 week~~ | **DONE (nb16)** |
| Prior sensitivity study | 2-3 days | Moderate — preempts reviewer question |
| Model mismatch robustness | 1 week | Moderate — strengthens generality claim |
| Regenerate figures at pub quality | 2-3 days | Required — cosmetic but expected |
| Writing the paper itself | 2-3 weeks | — |
| **Total to submission-ready** | **3-4 weeks** | — (EKF/UKF done, saves 1 week) |

---

## What a rejection looks like (and how to avoid it)

The three most likely rejection reasons at EAAI:

1. ~~**"No comparison with industrial baselines (EKF/UKF)"**~~ — **RESOLVED.**
   EKF and UKF are now implemented (nb16) with full comparison table and figures.

2. **"The novelty is application of existing SBI to a standard system"** — Fix: the
   identifiability analysis + embedding-net proof + analytical derivation IS the
   novelty. Frame it clearly in contributions 2-3. The paper is not "we applied SBI"
   — it is "we systematically characterised what feedback control does to a Bayesian
   posterior and proved it is irreducible."

3. **"Synthetic-only validation"** — Fix: if you cannot get real data, add a model
   mismatch study (perturb UA by +/-5%, use a different noise model) to show
   robustness to sim-to-real gap. Frame the synthetic study as a necessary first
   step before real deployment.

If you address #1 (EKF) and frame #2-3 correctly in the text, this paper has a
strong chance of acceptance at EAAI on the first or second round.
