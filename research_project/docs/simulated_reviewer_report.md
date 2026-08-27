# Peer Review Report

## Manuscript

**Simulation-Based Inference for Plant-Wide Fault Diagnosis and Artefactual Non-Identifiability Under Closed-Loop Control**

This is an ambitious and potentially valuable study at the intersection of chemical-process monitoring, closed-loop identification, and simulation-based inference. I deliberately reviewed it from a skeptical technical perspective, looking for reasons the central claims might not survive peer review. My conclusion is that the paper contains a publishable idea, but the present manuscript has several serious mathematical, statistical, and process-modeling problems. I would **not recommend acceptance in its current form**.

**Recommendation: Reject and encourage resubmission after substantial reconstruction.**

The strongest reason for rejection is not that SBI fails to outperform the EKF. The paper is commendably candid about that. The problem is that several of the manuscript's most important conclusions, particularly the claims of method-independent bias, "genuine" information loss, and representation-level non-identifiability, are not established by the analyses presented.

---

# 1. Summary of the Claimed Contribution

The paper applies neural posterior estimation to two simulated closed-loop chemical processes:

1. A PI-controlled propylene-oxide CSTR with catalyst activity and jacket-fouling parameters.
2. A reactor-column-recycle system with five degradation parameters and three PI loops.

The authors argue that:

- Integral temperature control causes genuine loss of information about heat-transfer degradation.
- Some parameter couplings observed with engineered summaries disappear under a raw-trajectory Fisher-information calculation and are therefore representation artefacts.
- Amortised SBI can provide rapid joint posterior inference without labelled historical fault data.
- The operational significance of a parameter confound depends on whether the coupled parameters belong to the same maintenance unit.
- SBI is useful despite not universally outperforming model-based filters.

These are worthwhile questions for *Computers & Chemical Engineering*. The distinction between process-level identifiability and representation-induced ambiguity could be an important contribution if formalized and demonstrated convincingly.

---

# 2. Overall Assessment

## Strengths

- The paper tackles a practically important but under-discussed issue: feedback control can conceal process degradation from fault-diagnosis methods.
- The authors do not claim universal superiority of SBI and explicitly report cases where the EKF performs better.
- The distinction between parameter-level ambiguity and unit-level maintenance decisions is insightful.
- The manuscript is generally well organized and visually clear.
- The comparison of engineered summaries, a learned embedding, filters, MCMC, and Fisher-type diagnostics has the potential to be informative.
- The use of continuous parameter priors rather than only discrete fault labels is conceptually appropriate.

## Fundamental Weaknesses

The current manuscript suffers from four overarching problems:

1. **Internal numerical inconsistencies undermine confidence in the reported results.**
2. **The quantity called a Fisher information matrix is not demonstrated to be the Fisher information of either the raw data or the summary statistics.**
3. **Cramér-Rao theory is repeatedly used to explain estimator bias, which it does not do.**
4. **The System II model is too incomplete and ad hoc to support plant-level conclusions about identifiability or industrial fault diagnosis.**

In addition, the posterior fails the paper's own simulation-based calibration test, feed-fault diagnosis fails completely, the raw-data CNN does not recover the supposedly available information, and the evaluation protocol appears to generate fault-revealing initialization transients.

---

# 3. Major Comments

## Major Comment 1: Table VII Contains an Apparent Mathematical Impossibility

This is the most immediate credibility problem.

For Sc 2, Table VII reports for SBI:

- true $\beta^\ast = 0.70$,
- $\hat{\beta} = 0.551$,
- bias $=-0.149$,
- MAE $=0.033$.

For the same set of observations and the same definition of the estimate, the mean absolute error must satisfy

$$
\mathrm{MAE} \geq \left|\mathrm{mean\ error}\right|
= |\mathrm{bias}|.
$$

Thus, an MAE of 0.033 cannot coexist with an absolute bias of 0.149. This violates Jensen's inequality unless these quantities are calculated from different objects, populations, or estimands, which the table does not disclose.

The inconsistency extends beyond this inequality:

- Table VI reports $\hat{\beta}=0.62$ for Sc 2.
- The text reports approximately 0.616 for the handcrafted-summary posterior.
- Table VII reports 0.551.
- The discussion alternates between an offset near $-0.08$ and a bias as large as $-0.15$.

These are not harmless rounding differences. They alter the paper's central quantitative claim. The authors must regenerate all tables directly from a single archived result set and include a unit-tested script verifying bias, MAE, RMSE, coverage, and posterior means. Until this is resolved, the multi-method comparison should not be trusted.

**Required action:** Provide replicate-level results, exact metric definitions, seeds, aggregation rules, and a consistency check proving that all reported metrics refer to the same 50 replicates.

---

## Major Comment 2: The Interpretation of Cramér-Rao Theory Is Incorrect

The manuscript states that agreement between CNN-SBI and summary-based SBI is an "empirical confirmation of the Cramér-Rao bound," and later argues that the common bias across SBI, MCMC, EKF, and UKF establishes a method-independent property of the data.

This is not valid.

The classical Cramér-Rao inequality concerns the covariance of an **unbiased estimator**, subject to regularity conditions. It does not predict the sign or magnitude of estimator bias. For biased estimators, a modified bound involving the derivative of the bias function is needed. Neural posterior means, constrained-prior Bayesian estimators, EKF estimates, and regularized state-parameter filters are not generally unbiased.

A low Fisher information can explain high variance or broad likelihood geometry. It does not, by itself, explain why four estimators should all be shifted downward. A common bias could instead arise from:

- common simulator mismatch,
- incorrect or discretized likelihood,
- prior truncation and boundary effects,
- incorrect noise covariance,
- a shared initialization transient,
- nonlinear transformation bias,
- errors in the process model,
- finite-window estimation,
- filter tuning,
- implementation errors,
- or reporting inconsistencies such as those already visible in Table VII.

This is particularly problematic for the NUTS result. If data are generated from the same probabilistic model used by NUTS, and the likelihood and prior are implemented correctly, the Bayesian posterior should show appropriate repeated-sampling calibration when averaged over the prior predictive distribution. A persistent scenario-specific posterior-mean bias is possible, but it is not proof of an information-theoretic bias and must be separated from posterior uncertainty and prior shrinkage.

**Required action:** Replace all claims that the Cramér-Rao bound explains or confirms bias. Analyze bias separately using likelihood profiles, repeated simulation, prior sensitivity, and a biased-estimator bound if the authors want a formal result.

---

## Major Comment 3: The Reported "FIM" Is Closer to a Local Sensitivity Information Matrix Than a Demonstrated Fisher Information Matrix

The manuscript computes

$$
I = J^\top \Sigma^{-1}J,
$$

where $J$ is the Jacobian of expected summaries and $\Sigma$ is described as a diagonal noise covariance estimated from healthy replicates.

This expression is Fisher information only under restrictive assumptions, for example, that the selected statistic is Gaussian with parameter-independent covariance and the specified covariance is correct. None of these conditions is established.

Important missing terms and assumptions include:

1. **Parameter-dependent covariance.**  
   If $\Sigma=\Sigma(\theta)$, Gaussian Fisher information contains an additional covariance-derivative term.

2. **Cross-feature correlations.**  
   The 29 and 66 engineered summaries are almost certainly strongly correlated. Replacing their covariance by a diagonal matrix can arbitrarily inflate information and alter normalized off-diagonal couplings.

3. **Temporal autocorrelation.**  
   Raw trajectories from a controlled stochastic dynamical system are highly autocorrelated. Treating sampled time points as independent measurements can dramatically overstate information.

4. **Non-Gaussian summary distributions.**  
   Ratios, quantiles, extrema, trends, and saturation counts are not generally jointly Gaussian.

5. **Nonregular operation near actuator saturation or snowball limits.**  
   Derivatives can be discontinuous or unstable around clipping, regime changes, and algebraic-solver boundaries.

6. **Finite-difference sensitivity.**  
   No convergence study is shown for perturbation size, replicate count, common random numbers, or numerical solver tolerance.

Consequently, the matrix may be a useful empirical sensitivity diagnostic, but the manuscript overstates what it proves. In particular, a reduced normalized off-diagonal entry does not demonstrate that a confound is merely representational in a global or posterior sense.

Figure 7 is presented as a normalized Fisher matrix, but the normalization hides absolute information scale and does not resolve the assumptions above.

**Required action:** Either derive the likelihood and compute the actual Fisher information, or rename the quantity a "local Gaussian sensitivity-information approximation." Use the full covariance with regularization, report covariance condition numbers, and demonstrate robustness to perturbation size and Monte Carlo sample size.

---

## Major Comment 4: The Raw-Trajectory FIM Does Not Prove Plant-Level Identifiability

The paper's central System II argument is:

1. A summary-space coupling is large.
2. The coupling becomes small using raw trajectories.
3. Therefore, the coupling is a representation artefact rather than a plant-level non-identifiability.

This conclusion is too strong.

A local, finite-difference, Gaussian-approximation information calculation at one operating point establishes, at most, local distinguishability under the assumed measurement and noise model. It does not establish:

- global identifiability,
- structural identifiability,
- practical recoverability over the prior,
- recoverability near other operating regimes,
- robustness to model discrepancy,
- or recoverability by the proposed estimator.

The manuscript itself supplies evidence against the strong interpretation: the CNN-SBI model applied to the raw trajectories does not resolve the $\alpha-\beta_r$ ambiguity and does not improve feed-fault classification. Thus, the paper shows that a particular local matrix becomes better conditioned, not that the ambiguity has been operationally removed.

The phrase "representation artefact" should be qualified as something like:

> A local coupling that is substantially amplified by the selected summary representation under the assumed Gaussian sensitivity metric.

That is still useful, but it is narrower than the present claim.

---

## Major Comment 5: The System I Physics-Informed Summaries Are Not Derived Correctly as Written

### 5.1 Jacket-Conductance Proxy

The paper defines

$$
s_{UA} = \frac{\bar{T}-\bar{T}_c}{\bar{Q}_c}
$$

and claims $s_{UA}\propto(\beta UA)^{-1}$.

At steady state, the jacket balance gives

$$
\beta UA(T-T_c)
=
\rho_c C_{pc} Q_c(T_c-T_{ci}).
$$

Therefore,

$$
(\beta UA)^{-1}
=
\frac{T-T_c}
{\rho_c C_{pc}Q_c(T_c-T_{ci})}.
$$

The proposed proxy omits $T_c-T_{ci}$. Unless this temperature difference is effectively constant over the entire parameter range, the claimed proportionality does not hold. Since $T_c$ is specifically said to shift with fouling, the omitted factor is not obviously constant.

This omission may be a direct source of the reported $\beta$ bias.

### 5.2 Kinetic Proxy

The paper defines

$$
s_{k_0} = \ln\left(\frac{\bar C}{C_i-\bar C}\right)
$$

and states that it is proportional to $(\alpha k_0)^{-1}$.

From the steady-state material balance,

$$
\alpha k(T)
=
\frac{Q}{V}\left(\frac{C_i}{C}-1\right),
$$

so

$$
\frac{C}{C_i-C}
=
\frac{Q}{V\alpha k(T)}.
$$

Taking the logarithm gives

$$
s_{k_0}
=
\ln(Q/V)-\ln\alpha-\ln k(T).
$$

Thus, $s_{k_0}$ is affine in the **negative log** of the effective kinetic constant, not proportional to its reciprocal.

These distinctions matter because the paper uses the two features to argue near-sufficient identification of the two parameters.

**Required action:** Correct the derivations and repeat the analysis using physically exact balance-residual or parameter-inversion features.

---

## Major Comment 6: The System II Model Is Insufficiently Specified and May Not Be Physically Self-Consistent

The System II section does not provide enough equations or parameter values to reproduce the process model. Several elements are especially concerning.

### 6.1 The Recycle Relation in Eq. 7 Is Unclear

The expression

$$
F_R = \frac{D x_D}{z_{A,in}}
$$

requires a careful derivation. If all distillate is recycled, one would normally expect $F_R=D$. If only component-A flow is being equated, the definition of $F_R$ must be clarified. As written, the equation mixes total stream flow and component flow in a way that is not self-evident.

### 6.2 The Distillation Model Is Missing

The manuscript states that Fenske-Underwood-Gilliland shortcut equations determine $x_D$, $x_B$, and recycle flow, but the actual equations, specifications, recoveries, and solution procedure are absent. FUG relations are principally shortcut design relations, not a dynamic column model. It is not clear how the reflux and reboiler PI loops interact consistently with this algebraic block.

### 6.3 Important Constants Are Absent

Quantities such as $UA_r$, $V_r$, $M_j$, $C_{pj}$, nominal jacket duty, reboiler conductance, and relevant feed temperatures do not appear in the nominal-parameter table shown in the manuscript.

### 6.4 Reboiler Degradation Is Imposed Through an Ad Hoc Algebraic Relation

Equation 11 effectively defines a duty scaling rather than deriving the behavior from a reboiler energy balance. This may make $\xi_{\mathrm{reb}}$ nearly directly observable by construction.

### 6.5 Tray-Efficiency Degradation Is Represented by Scaling Relative Volatility

The relation

$$
\alpha_{\mathrm{eff}}
=
1+\eta_{\mathrm{col}}(\alpha_{\mathrm{rel}}-1)
$$

is an ad hoc mapping. Tray efficiency normally affects the number of equilibrium stages or stage approach to equilibrium, not molecular relative volatility. This parameter may be useful as an empirical degradation index, but it should not be called tray efficiency without stronger justification.

### 6.6 Dynamic Claims Are Questionable

The paper attributes identifiability to transient shapes, while replacing the 20-stage column by a quasi-steady algebraic shortcut. The resulting model cannot reproduce the inventory, hydraulic, and compositional dynamics that would generate many of the real transient signatures used for fault separation.

Figure 2 visually illustrates feedback masking, but that qualitative result does not validate the complete recycle-column model or the claimed parameter observability.

Because the System II model is central to the plant-wide novelty, lack of a complete reproducible specification is grounds for rejection.

---

## Major Comment 7: The Initialization Protocol Likely Creates an Artificial Fault-Identification Signal

The manuscript says stochastic trajectories start from the nominal healthy closed-loop steady state, while the fault parameters are then assigned degraded values.

If this is done for every fault scenario, each observation window contains an implicit fault-onset experiment at $t=0$. The initial-state mismatch produces a parameter-dependent transient that would not generally be present in gradual degradation monitoring. An estimator can then identify the fault from the artificial relaxation trajectory rather than from naturally occurring process variation.

This is particularly important because the paper repeatedly attributes improved distinguishability to transient shape information.

Three distinct experiments are needed:

1. Each fault initialized at its own degraded steady state.
2. Gradual degradation beginning before the observation window.
3. An explicitly labelled abrupt fault-onset experiment.

Performance should be reported separately. If classification falls substantially when initialized at the faulted steady state, the current results do not demonstrate passive condition monitoring.

---

## Major Comment 8: The Simulation-Based Calibration Result Is a Failure, Not a Minor Imperfection

For System I, the manuscript reports KS p-values of 0.016 for $\alpha$ and 0.014 for $\beta$. At the stated 5% level, both parameters fail the rank-uniformity test.

The manuscript then invokes two-sample classifier scores of 0.52 and 0.53 and claims this indicates a structural information deficit rather than a training deficiency. That interpretation is unsupported.

A posterior can be broad because the data are weak and still be perfectly calibrated. Weak identifiability does not cause SBC ranks to become nonuniform if the posterior approximation is correct. SBC detects algorithmic or implementation-related miscalibration under the joint prior-predictive model.

Similarly, a classifier score near 0.5 does not automatically diagnose the source of SBC failure. The paper must specify exactly which distributions the classifier compares, its uncertainty, sample size, classifier capacity, and test power.

The production posterior therefore fails the manuscript's declared calibration criterion, yet is used for credible intervals, posterior class probabilities, and maintenance confidence. Those quantities should not be presented as calibrated until the SBC issue is resolved.

---

## Major Comment 9: "Correct 90% Intervals at Every Time Step" Is Not Credible Evidence of Calibration

The paper states that the 90% credible interval for $\beta(t)$ contains the true curve at every evaluated time step over the 30-day run, despite a persistent offset. That is 100% empirical coverage for a nominal 90% interval.

This could happen because:

- intervals are overly wide,
- windows are highly dependent,
- the same model error persists across all windows,
- or the reported statement is based on a single degradation trajectory rather than repeated experiments.

Coverage cannot be assessed from one correlated time series. The sequential monitoring experiment must be repeated across many independently simulated degradation paths, and pointwise and simultaneous coverage must be distinguished.

---

## Major Comment 10: The Paper Claims Structural or Fundamental Non-Identifiability Where It Appears to Show Practical Information Reduction

For System I, $\beta$ remains observable through $T_c$ and $Q_c$. The paper itself provides a steady-state expression involving $\beta$ and successfully estimates it, albeit less accurately than $\alpha$.

Therefore, this is not structural non-identifiability of $\beta$ under the stated measurement set. It is weaker practical identifiability, lower sensitivity, or poorer signal-to-noise ratio. The text sometimes acknowledges this, but elsewhere uses stronger language such as "fundamentally limited," "irreducible," and "information removed."

Similarly, integral control makes

$$
\frac{\partial T_{ss}}{\partial \beta}=0
$$

for the controlled variable at steady state, but does not imply zero information in:

- the controller output,
- jacket temperature,
- transient response,
- saturation behavior,
- or disturbance response.

The paper should distinguish carefully among:

- structural identifiability,
- local identifiability,
- practical identifiability,
- channel-specific masking,
- steady-state sensitivity suppression,
- and finite-window estimator bias.

At present these concepts are used too interchangeably.

---

## Major Comment 11: The Claim That SBI Is Needed Because the Likelihood Is Intractable Is Not Established

The simulations use Euler-Maruyama integration with additive process and sensor noise. Conditional on the previous state and parameters, Euler-Maruyama with additive Gaussian noise commonly yields an explicit Gaussian transition density. Sensor noise is likewise often Gaussian and tractable.

If that is the actual simulator, a discretized state-space likelihood may be available. It may be expensive, high-dimensional, or require latent-state integration, but that is not the same as being unavailable.

Moreover, the authors run NUTS, which ordinarily requires a differentiable log posterior. The paper must explain:

- What likelihood was used by NUTS?
- Was it the full trajectory likelihood?
- Were latent states sampled?
- Was the same stochastic model used as in data generation?
- If an exact or discretized likelihood exists, why is the approach called likelihood-free?
- Is the real benefit amortization rather than likelihood intractability?

The motivation should be reframed unless a genuinely implicit simulator component is demonstrated.

---

## Major Comment 12: The Baseline Comparison Is Not Fair Enough to Support Method-Level Conclusions

The estimators do not have equivalent information or targets:

- The System II EKF estimates only three of the five parameters.
- SBI receives 66 summaries, while the EKF receives raw trajectories and the exact state equations.
- The CNN-SBI architecture and training budget may be inadequate.
- EKF results are acknowledged to be strongly tuning-sensitive.
- MCMC latency is reported without hardware, chain count, warmup, effective sample size, convergence diagnostics, or accuracy-matched stopping criteria.

The paper should not imply a clean comparison of SBI, MCMC, EKF, and UKF. These are different experimental configurations.

A fair comparison would hold constant:

- observed channels,
- observation window,
- simulator discretization,
- parameter set,
- prior,
- noise model,
- initialization,
- and performance metrics.

At minimum, the paper must present the comparison as a collection of illustrative baselines, not evidence of general algorithmic behavior.

---

## Major Comment 13: Perfect or Near-Perfect Classification Is Not Persuasive Under the Current Evaluation Design

System I reaches nearly perfect classification, and two handcrafted features reportedly achieve 98.3% LDA accuracy across eight scenarios. However:

- The scenarios are widely separated fixed parameter combinations.
- Multiple stochastic replicates share the same scenario values.
- All simulations appear to use the same nominal initial state.
- The engineered features algebraically invert the same model used to generate the data.
- No model discrepancy is present.
- No independent plant, kinetic, or controller uncertainty is introduced.

Cross-validation by randomly splitting replicates can leak scenario identity because training and test folds contain replicates from the same exact parameter values.

More informative tests would include:

- leave-one-parameter-value-out validation,
- interpolation and extrapolation to unseen fault severities,
- controller-tuning variation,
- uncertain physical parameters,
- uncertain sensor bias,
- degraded-state initialization,
- colored and non-Gaussian noise,
- and simulator mismatch.

The current classification numbers primarily show that the simulator can be inverted under matched assumptions.

---

## Major Comment 14: Feed-Fault Identification Fails Completely, Weakening the "Plant-Wide" Claim

The feed class has $F_1=0$, and W10 is misclassified in all 30 replicates. The raw CNN representation also fails to improve feed classification.

This is not a secondary limitation. Feed composition is one of only three unit-level fault categories in the plant-wide experiment. Failure on the entire category means the proposed framework does not yet perform plant-wide fault attribution across the stated fault taxonomy.

The abstract and conclusion should not claim successful plant-wide fault identification without strongly qualifying that one complete unit class is undetectable. A more accurate statement would be that the method demonstrates reactor and column fault classification in a simplified simulated benchmark while exposing unresolved feed-fault attribution.

---

## Major Comment 15: The Sensor-Drift Experiment Is Not Adequately Defined

For Sc 7, the manuscript says a $+2\,\mathrm{K}$ drift is superimposed on the temperature measurement. It is unclear whether:

1. the biased measurement is seen by both the PI controller and the diagnostic system, or
2. the controller sees the true temperature while only the inference pipeline sees the bias.

The two cases have completely different closed-loop physics. A real sensor drift in the feedback sensor changes the actual reactor temperature because the controller regulates the biased measurement to the setpoint. If only the historian channel is shifted, this is an observation corruption, not closed-loop sensor drift.

The experiment must be reformulated and both cases should be distinguished.

---

# 4. Additional Technical and Presentation Comments

1. The abstract states that "two retracted confounds" are shown. "Retracted" appears to be the wrong term. "Representation-induced," "resolved," or "reduced" may be intended.
2. The abstract says "when no tractable likelihood exist." This should be "exists."
3. The manuscript alternates between "System 1" and "System I."
4. Figure 2's caption appears to say "jacket temperature $T_r$ in orange," where $T_j$ is intended.
5. Section VI B contains missing symbols in "ambiguity between and r," apparently $\alpha$ and $\beta_r$.
6. Section VI C2 contains "TThe."
7. Equation 5c needs unambiguous parentheses and units:

   $$
   \frac{dT_j}{dt}
   =
   \frac{\beta_r UA_r(T_r-T_j)-Q_j}{M_j C_{pj}}
   $$

   if that is the intended form.

8. The use of both SI and US customary units in System II is acceptable for reproducing the benchmark, but every equation should be dimensionally verified and conversion constants reported.
9. The "unique physically meaningful steady state" claim for the nonlinear exothermic CSTR should be supported over the relevant operating region, not only asserted at one nominal point.
10. The statement that increasing proportional gain consistently makes $\beta$ harder to identify requires a documented gain sweep. Higher gain can redistribute information into manipulated-variable activity and transient behavior, so monotonic deterioration is not automatic.
11. Noise intensities are central to all information ratios but are relegated to unavailable Supporting Information.
12. The use of a diagonal covariance estimated from healthy replicates at faulted operating points is particularly questionable.
13. Near saturation, conditional anti-windup introduces nonsmooth dynamics. Finite-difference information calculations around these points require special treatment.
14. The classification threshold of 0.85 is not applied transparently to $z_{A0,\mathrm{eff}}$. A 15% relative reduction from 0.90 is 0.765, not 0.85. The exact thresholding formula should be shown.
15. For $\eta_{\mathrm{col}}\in[0.8,1.0]$, a threshold of 0.85 places most of the allowed degradation interval on the "healthy" side. The effect of this asymmetric prior and narrow degraded region must be analyzed.
16. The prior probability of each unit-level class is highly uneven because class probabilities are induced by threshold volumes in a five-dimensional prior. Posterior class confidence should be interpreted with this prior imbalance in mind.
17. Scenario W8 and W14 are missing from the numbering. If removed experiments existed, explain their removal to avoid concern about selective reporting.
18. "Typical latency below 20 ms" should include processor, batch size, posterior sample count, and whether sampling or only network evaluation is timed.
19. Credible-interval coverage is reported only for selected parameters. Coverage should be given for all parameters and all scenarios.
20. The code repository is still a placeholder, `https://github.com/XXXXX`. This alone prevents reproducibility and should be corrected before submission.

---

# 5. Experiments Required for a Publishable Revision

I would consider a fundamentally revised submission if it included the following.

## Essential

1. **Reconcile all numerical inconsistencies**, especially Table VI versus Table VII.
2. **Release complete code and configuration files**, including all seeds and model equations.
3. **Correct the physics-informed feature derivations.**
4. **Replace or qualify the Fisher-information claims.**
5. **Separate estimator bias from information content.**
6. **Provide full SBC for every parameter**, with adequate test counts and multiple calibration diagnostics.
7. **Repeat experiments from fault-specific steady states**, not only the nominal healthy initial state.
8. **Fully specify and dimensionally validate System II.**
9. **Run sensitivity tests with full covariance**, temporal dependence, perturbation size, and Monte Carlo replication.
10. **Reframe feed-fault failure as a central negative result.**

## Strongly Recommended

11. Include uncertainty in nuisance physical parameters and controller tuning.
12. Evaluate model mismatch by generating data with a higher-fidelity model than the inference simulator.
13. Use leave-fault-severity-out validation.
14. Compare against a likelihood-based Bayesian state-space model if the Euler-Maruyama likelihood is tractable.
15. Show posterior predictive checks in trajectory space, not only summary-space coverage.
16. Repeat sequential experiments across many independent degradation paths.
17. Report calibration of unit-level posterior probabilities using reliability diagrams or Brier scores.
18. Investigate whether a temporal encoder such as a TCN, state-space encoder, or transformer can exploit the raw-trajectory information identified by the sensitivity analysis.
19. Include at least one real or semi-synthetic industrial dataset, if available.

---

# 6. Publication Judgment

## Novelty

**Moderate.** Applying amortised SBI to chemical-process fault diagnosis is interesting, but the current principal novelty is the proposed distinction between closed-loop information loss and representation-induced confounding. That distinction is not yet demonstrated rigorously.

## Technical Correctness

**Insufficient in the present version.** The MAE-bias contradiction, misuse of Cramér-Rao arguments, non-calibrated posterior, approximate FIM interpretation, and incomplete System II model are substantial.

## Reproducibility

**Insufficient.** The Supporting Information is essential to the claims but was not included in the provided manuscript, and the code link is a placeholder.

## Practical Relevance

**Potentially high, currently limited.** The work is entirely simulator-matched, feed-fault classification fails, and the System II plant is a simplified algebraic approximation.

## Fit for *Computers & Chemical Engineering*

**Yes, in principle.** The topic and methodological scope fit the journal well. The current execution does not yet meet the standard needed for publication.

---

# 7. Final Recommendation to the Editor

> **Reject, with encouragement to resubmit as a substantially revised manuscript.**

The paper has the seed of a strong contribution, particularly in connecting representation design, closed-loop parameter sensitivity, and maintenance-oriented fault taxonomies. However, the numerical contradictions and theoretical overclaims affect the central conclusions rather than peripheral details. A conventional minor or major revision may not be sufficient because the authors need to rerun substantial portions of the analysis, correct the interpretation of Fisher and Cramér-Rao theory, and reconstruct the System II validation.

My honest assessment is that the article is **not publishable as submitted**, but it could become publishable if the authors narrow the claims, repair the statistical foundations, fully expose the model, and demonstrate that the reported performance is not driven by artificial initialization transients or simulator-matched feature engineering.
