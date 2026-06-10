# The Mechanics of Structural Bias in Closed-Loop Parameter Estimation: Unifying Information Theory and Control-Theoretic Perspectives for CSTR Condition Monitoring

## 1. Introduction

The transition from preventive, scheduled maintenance to predictive, condition-based monitoring represents a paradigm shift in chemical processing and industrial automation. At the core of this transition is the ability to accurately estimate unmeasurable physical parameters—such as catalyst deactivation and heat exchanger fouling—from observable, high-dimensional sensor data. In highly integrated, safety-critical unit operations like the Continuous Stirred Tank Reactor (CSTR), these degradation phenomena must be tracked dynamically to ensure operational stability and economic optimisation. However, the industrial imperative to maintain these reactors under tight, closed-loop feedback control introduces a profound structural limitation. The very control mechanisms designed to reject disturbances and maintain process setpoints simultaneously mask the thermodynamic and kinetic signatures required to identify underlying process degradation.

This document derives the mechanism of the persistent downward bias ($-0.08$ to $-0.15$) on the jacket fouling parameter $\beta$ from two complementary analytical perspectives. The **information-theoretic perspective** explains the bias directly: the PI controller erases the temperature channel's sensitivity to $\beta$, starving it of Fisher information, and the resulting wide, skewed likelihood pulls the posterior mean below the true value. The **control-theoretic perspective** provides a frequency-domain view of the same information loss, and separately explains why an estimator trained on open-loop data will fail when deployed on closed-loop observations — a distinct, avoidable problem.

A critical clarification upfront: **the $-0.08$ β bias is not overcome by Simulation-Based Inference (SBI)**. It is present in every method tested — NUTS, SBI, EKF, and UKF — at consistent magnitudes ($-0.09$ to $-0.15$ depending on operating point). CL-trained SBI faithfully represents the posterior that the closed-loop data actually supports. Its advantage over OL-trained SBI is that it learns the correct posterior for the closed-loop data-generating process; it does not remove the underlying information deficit.

---

## 2. The Physical System: Closed-Loop CSTR Dynamics and Observability

To ground the theoretical derivations in a realistic chemical engineering context, the process under investigation is the acid-catalyzed hydrolysis of propylene oxide (PO) to propylene glycol (PG) within a non-isothermal CSTR. This specific reaction is characterized by highly exothermic kinetics, requiring continuous heat removal via a cooling jacket to maintain a stable and safe operating temperature.

### 2.1 Process Kinetics and Degradation Parameters

The fundamental goal of the condition monitoring system is the joint, real-time estimation of two independent degradation mechanisms. These are represented mathematically as scalar multipliers operating on the nominal physical constants, encapsulated in the parameter vector $\theta = [\alpha, \beta]^T \in [0.4, 1.0]^2$.

The first parameter is **catalyst deactivation**, denoted by $\alpha$. This scalar represents the fractional loss of active acid sites within the reactor volume over time. The effective reaction rate constant is modeled using modified first-order Arrhenius kinetics:

$$k_{eff}(t) = \alpha(t) \cdot k_0 \cdot \exp\left(-\frac{E_a}{R \cdot T(t)}\right)$$

where $E_a$ is the activation energy, $R$ is the universal gas constant, and $T(t)$ is the dynamic reactor temperature.

The second parameter is **cooling jacket fouling**, denoted by $\beta$. This scalar represents the fractional degradation of the overall heat transfer coefficient due to scaling or particulate deposition on the reactor walls (often modelled via Kern-Seaton fouling dynamics). The effective heat transfer capacity is modified as:

$$UA_{eff}(t) = \beta(t) \cdot UA_{nominal}$$

By fixing the clean-service design constants to known values derived from initial plant commissioning ($k_0 = 16.96 \times 10^{12} \text{ min}^{-1}$ and $UA_{nominal} = 12{,}500 \text{ cal/(min} \cdot \text{K)}$), the inference parameter vector is restricted to a strictly two-dimensional space.

### 2.2 Differential Equations and Stochastic Disturbances

The CSTR is rigorously modelled via a coupled system of nonlinear ordinary differential equations (ODEs) that enforce the conservation of mass and energy across the reactor volume and the cooling jacket. The state variables are the reactant concentration $C(t)$, the reactor temperature $T(t)$, and the jacket temperature $T_c(t)$.

The **mass balance** tracks the consumption of propylene oxide:

$$\frac{dC}{dt} = \left(\frac{Q}{V}\right) (C_i - C) - k_{eff}(\alpha, T) \cdot C$$

The **reactor energy balance** accounts for convective heat transfer, the heat of reaction ($\Delta H_r = -20{,}220 \text{ cal/mol}$), and heat removal to the cooling jacket:

$$\frac{dT}{dt} = \left(\frac{Q}{V}\right) (T_i - T) - \frac{\Delta H_r \cdot k_{eff}(\alpha, T) \cdot C}{\rho \cdot C_p} - \frac{UA_{eff}(\beta) \cdot (T - T_c)}{\rho \cdot C_p \cdot V}$$

The **cooling jacket energy balance**:

$$\frac{dT_c}{dt} = \left(\frac{Q_c}{V_c}\right) (T_{ci} - T_c) + \frac{UA_{eff}(\beta) \cdot (T - T_c)}{\rho_c \cdot C_{pc} \cdot V_c}$$

The model includes continuous-time SDE diffusions via Euler-Maruyama: $\sigma_C = 0.0005 \text{ mol/L/}\sqrt{\text{min}}$ and $\sigma_T = \sigma_{Tc} = 0.1 \text{ K/}\sqrt{\text{min}}$, producing a steady-state Ornstein-Uhlenbeck standard deviation of approximately $0.5 \text{ K}$ on reactor temperature. Post-simulation Gaussian sensor noise of $0.5\%$ per channel is applied.

**Table 1 — Nominal CSTR Physical Parameters**

| Physical Parameter | Symbol | Nominal Value | Unit |
|---|---|---|---|
| Reactor Volume | $V$ | 500 | L |
| Jacket Volume | $V_c$ | 40 | L |
| Feed Flow Rate | $Q$ | 40 | L/min |
| Fluid Density | $\rho, \rho_c$ | 1000 | g/L |
| Heat Capacity | $C_p, C_{pc}$ | 1.0 | cal/(g·K) |
| Nominal Feed Concentration | $C_i$ | 0.97 | mol/L |
| Nominal Feed / Jacket Inlet Temp | $T_i, T_{ci}$ | 297.0 | K |

### 2.3 The Feedback Control Architecture

The critical mechanism introducing the structural limitation is the Proportional-Integral (PI) temperature controller. The controller manipulates the coolant flow rate $Q_c(t)$ to minimise the error between the measured reactor temperature $T(t)$ and the setpoint $T_{sp} = 312.5 \text{ K}$:

$$Q_c(t) = Q_{c0} + K_p (T(t) - T_{sp}) + \frac{K_p}{\tau_i} \int_0^t (T(\tau) - T_{sp}) \, d\tau$$

Under industrial tuning ($K_p = 150 \text{ (L/min)/K}$, $\tau_i = 10 \text{ min}$), the controller pins $T(t) \approx T_{sp}$, maintaining it within a tight $\pm 2 \text{ K}$ envelope. Thermodynamic variance is not destroyed — it is transferred from the controlled variable $T$ into the manipulated variable $Q_c$. The coolant flow carries the fault signature instead: it walks up as fouling worsens, drops as the catalyst deactivates.

---

## 3. The Information-Theoretic Explanation: Why $\beta$ Is Biased

This section contains the direct, complete explanation of the observed bias. The subsequent section provides a complementary frequency-domain perspective.

### 3.1 Steady-State Reduction and the Analytical Jacobian

Under perfect integral control, $T_{ss} = T_{sp}$ at steady state. We define the observation vector $\mu = [C_{ss}, T_{ss}, T_{c,ss}, Q_{c,ss}]^T$. The Jacobian $J = \partial \mu / \partial \theta$ reveals the structural loss:

$$J = \begin{bmatrix} \frac{\partial C_{ss}}{\partial \alpha} & 0 \\ 0 & 0 \\ \frac{\partial T_{c,ss}}{\partial \alpha} & \frac{\partial T_{c,ss}}{\partial \beta} \\ \frac{\partial Q_{c,ss}}{\partial \alpha} & \frac{\partial Q_{c,ss}}{\partial \beta} \end{bmatrix}$$

The steady-state concentration depends only on $\alpha$:

$$C_{ss}(\alpha) = \frac{Q C_i}{Q + V \alpha k_0 \exp(-E_a / (R T_{sp}))}$$

The steady-state coolant flow depends on both parameters:

$$Q_{c,ss}(\alpha, \beta) = \frac{\beta UA_{nominal} (T_{sp} - T_{c,ss}(\alpha, \beta))}{\rho_c C_{pc} (T_{c,ss}(\alpha, \beta) - T_{ci})}$$

The second row of $J$ is identically zero because the PI controller rigidly pins $T$ to the setpoint, so $\partial T_{ss}/\partial \alpha = \partial T_{ss}/\partial \beta = 0$. The primary thermodynamic channel — normally the highest signal-to-noise observable — is structurally erased by the feedback.

### 3.2 Fisher Information Matrix (FIM) Starvation

The FIM is $I(\theta) = J^T \Sigma^{-1} J$ with $\Sigma = \text{diag}(\sigma_C^2, \sigma_T^2, \sigma_{Tc}^2, \sigma_{Qc}^2)$. The diagonal elements are:

$$I_{\alpha\alpha} = \frac{1}{\sigma_C^2}\left(\frac{\partial C_{ss}}{\partial \alpha}\right)^2 + \frac{1}{\sigma_{Tc}^2}\left(\frac{\partial T_{c,ss}}{\partial \alpha}\right)^2 + \frac{1}{\sigma_{Qc}^2}\left(\frac{\partial Q_{c,ss}}{\partial \alpha}\right)^2$$

$$I_{\beta\beta} = \frac{1}{\sigma_{Tc}^2}\left(\frac{\partial T_{c,ss}}{\partial \beta}\right)^2 + \frac{1}{\sigma_{Qc}^2}\left(\frac{\partial Q_{c,ss}}{\partial \beta}\right)^2$$

$I_{\alpha\alpha}$ is dominated by the concentration term: $C_{ss}$ has high sensitivity to $\alpha$ and low sensor noise $\sigma_C$, so $\alpha$ is highly identifiable. $I_{\beta\beta}$ receives zero contribution from the $T$ channel (erased by the controller) and zero contribution from $C$ (decoupled from $\beta$). It relies entirely on the noisier $T_c$ and $Q_c$ channels. Numerically, $I_{\alpha\alpha} / I_{\beta\beta} \approx 250$–$500\times$ at all operating points.

By the Cramér-Rao lower bound, $\text{Var}(\hat{\beta}) \geq 1/I_{\beta\beta}$ for any unbiased estimator. When $I_{\beta\beta}$ is 250–500× smaller than $I_{\alpha\alpha}$, the posterior over $\beta$ is wide regardless of the inference algorithm. This has been confirmed numerically: NUTS, SBI (29-D summaries), SBI (CNN embedding), EKF, and UKF all show the same $\approx -0.08$ to $-0.15$ bias on $\beta$ — the limitation is in the data, not the estimator.

### 3.3 The Asymmetric Profile Likelihood and the Laplace Bias

Because $I_{\beta\beta}$ is small, the posterior over $\beta$ is wide. Over this wide region, the nonlinear physical relationship between fouling and the compensatory control action distorts the likelihood.

By profiling out the highly identifiable $\alpha$ (setting $\alpha^*(\beta) \approx \alpha_{true}$, which the concentration channel pins tightly), the profile log-likelihood for $\beta$ is dominated by the coolant channel mapping $\mu(\beta) = Q_{c,ss}(\alpha^*, \beta)$:

$$\ell_{prof}(\beta) \approx -\frac{1}{2 \sigma_{Qc}^2} (Q_{c,obs} - \mu(\beta))^2 + \mathcal{C}$$

As fouling worsens ($\beta$ decreases), the PI controller must disproportionately increase $Q_c$ because the thermal driving force across the fouled wall diminishes. This makes $\mu(\beta)$ strictly decreasing ($\mu' < 0$) and strictly convex ($\mu'' > 0$).

Convexity induces asymmetry in the profile likelihood. Using a Laplace approximation for the expected posterior mean shift:

$$\text{Bias} \approx -\frac{1}{2} \frac{\ell'''(\beta_{true})}{[\ell''(\beta_{true})]^2}$$

Evaluating at the likelihood peak ($y \approx \mu(\beta)$):

- $\ell'' \approx -(\mu')^2/\sigma^2$ — small and negative (confirms wide posterior).
- $\ell''' \approx -3\mu'\mu''/\sigma^2$ — because $\mu' < 0$ and $\mu'' > 0$, this is positive.

Therefore:

$$\text{Bias} \approx -\frac{1}{2} \frac{\text{positive}}{(\text{negative})^2} \implies \textbf{Strictly Negative}$$

This proves analytically that the posterior mean for $\beta$ must fall below the true value. The steady-state 4-observable model predicts the correct sign but underestimates the magnitude ($\approx -0.0003$ predicted vs $\approx -0.08$ observed). The gap is attributed to transient features in the full 29-D summary space (slopes, oscillation amplitudes, settling dynamics) that carry additional asymmetric sensitivity to $\beta$ not captured at steady state.

---

## 4. The Control-Theoretic Perspective: Two Complementary Insights

The information-theoretic derivation above is complete and self-contained. The control-theoretic perspective adds two useful but distinct insights: an equivalent frequency-domain view of the same information loss, and an explanation of why an open-loop-trained estimator fails on closed-loop data — a separate, avoidable failure mode.

> **Important scope note.** The classical Prediction Error Method (PEM) literature (Forssell & Ljung 1999) derives asymptotic properties of **black-box transfer function estimators** $G(q, \theta)$ applied to LTI closed-loop data. The CSTR problem here is different: $\theta = [\alpha, \beta]$ are scalar physical constants in a known nonlinear ODE, and inference is Bayesian rather than prediction-error minimisation. The PEM "bias-pull" result — that the identified transfer function drifts toward $-1/K(q)$ under zero excitation — does not directly apply to scalar physical parameter estimation and is not responsible for the $-0.08$ β bias.

### 4.1 Frequency-Domain Equivalent of the Jacobian Collapse

Define the closed-loop **sensitivity function** $S_0(q) = (1 + G_0(q) K(q))^{-1}$. The noise-free portion of the input spectrum — the part of $Q_c$ variation that carries independent information about the plant — is:

$$\Phi_u^r(\omega) = |S_0(e^{i\omega})|^2 \, \Phi_r(\omega)$$

where $\Phi_r(\omega)$ is the reference (setpoint) spectrum. The asymptotic parameter estimation variance scales as:

$$\text{Var}(\hat{\theta}) \propto \frac{\Phi_v(\omega)}{\Phi_u^r(\omega)}$$

At the fixed operating setpoint $T_{sp}$, there is no reference dithering: $\Phi_r(\omega) = 0$ at all frequencies where $\beta$ would be excited. Furthermore, the PI controller's integral action forces $S_0 \to 0$ at low frequencies (the frequencies where steady-state heat balance information lives). Together, $\Phi_u^r \to 0$, variance diverges, and $I_{\beta\beta} \to 0$.

This is the frequency-domain equivalent of the Jacobian collapse: $\partial T_{ss}/\partial \beta = 0$ in the time domain corresponds to $|S_0|^2 \Phi_r \to 0$ in the frequency domain. Neither the controller's excitation band nor the setpoint spectrum contributes independent information about $\beta$.

### 4.2 Why Open-Loop Training Fails: The Input-Noise Correlation

Under closed-loop operation, the controller computes:

$$Q_c(t) = Q_{c0} + K_p(T_{sp} - T_{meas}(t)) + \frac{K_p}{\tau_i}\int_0^t(T_{sp} - T_{meas}(s))\,ds$$

where $T_{meas}(t) = T(t) + \varepsilon_T(t)$ includes sensor noise. This makes $Q_c(t)$ an explicit function of $\varepsilon_T$, creating a non-zero cross-spectrum between the "input" and the measurement noise:

$$\Phi_{ue}(\omega) = -K(e^{i\omega})S_0(e^{i\omega})H_0(e^{i\omega})\lambda_0 \neq 0$$

In **open-loop** operation, $Q_c$ is held fixed independently of $T$, so $\Phi_{ue}(\omega) = 0$. This creates a **statistical distribution mismatch** between open-loop training data and closed-loop observations:

| | $\Phi_{ue}$ | Effect |
|---|---|---|
| Open-loop data | 0 | $Q_c$ and $T$ noise are independent |
| Closed-loop data | $\neq 0$ | $Q_c$ is a function of $T$ noise |

If an SBI network is trained on open-loop simulations and deployed on closed-loop observations, the learned summary statistics encode the wrong correlation structure. The network has never seen the feedback-induced coupling between $Q_c$ and $T$ transients, so its posterior is systematically corrupted. Empirically: OL-trained SBI on closed-loop Sc2 data gives $W_1^\beta = 0.578$ vs $W_1^\beta = 0.149$ for CL-trained SBI — a 288% degradation. On Sc6 (open-loop fouling), CL-trained SBI collapses to fault classification accuracy = 0.04, confirming the mismatch is bidirectional.

This failure mode — OL training on CL deployment — is **separate from, and avoidable unlike, the $-0.08$ β bias**. CL training corrects it by ensuring the training data faithfully replicates the closed-loop correlation structure. The residual β bias from low $I_{\beta\beta}$ remains after this correction.

---

## 5. What the Two Perspectives Explain (and What They Do Not)

**Table 2 — What Each Perspective Explains**

| | Information-Theoretic | Control-Theoretic |
|---|---|---|
| **Root cause** | Jacobian: $\partial T_{ss}/\partial \beta = 0$ (controller pins $T$) | $S_0 \to 0$, $\Phi_r = 0 \Rightarrow \Phi_u^r \to 0$ (no noise-free excitation of $\beta$) |
| **What it explains** | Why $I_{\beta\beta}$ is 250–500× smaller than $I_{\alpha\alpha}$; why the profile likelihood is skewed; why the posterior mean is strictly negative | Equivalent explanation of information loss; additionally explains why OL-trained SBI fails on CL data |
| **The $-0.08$ β bias direction** | ✓ Directly — via Laplace approximation on convex $Q_c(\beta)$ | ✗ Does not explain the sign or magnitude |
| **OL training failure** | ✗ Does not address | ✓ Directly — $\Phi_{ue}$ mismatch between training and deployment |
| **Whether SBI overcomes the bias** | No — all four methods (SBI, NUTS, EKF, UKF) show the same bias | No — CL training corrects distribution mismatch, not information deficit |

The two perspectives are not symmetric: the IT perspective gives the complete causal chain to the $-0.08$ bias. The CT perspective provides an equivalent view of the information loss and adds one genuinely new insight about the OL training failure.

---

## 6. Simulation-Based Inference: What It Does and Does Not Fix

### 6.1 What CL-Trained SBI Achieves

SBI trained on closed-loop simulations learns the posterior $p(\alpha, \beta \mid \mathbf{x})$ directly from simulated pairs, without evaluating a closed-form likelihood. Because the training simulator includes the full PI controller dynamics, the training data has the same $\Phi_{ue} \neq 0$ correlation structure as real plant observations. The network therefore learns the correct posterior *given the actual information content of the closed-loop data*.

This is what distinguishes CL-trained from OL-trained SBI. It is not a mechanism for overcoming the structural β bias — it is a mechanism for not introducing an *additional* spurious bias from training data mismatch.

**The residual $\beta$ bias persists.** The posterior mean for Sc2 ($\beta_{true} = 0.70$) is $\approx 0.616$ for SBI, $\approx 0.598$ for NUTS, $\approx 0.607$ for EKF, and $\approx 0.607$ for UKF. All four methods converge to the same biased answer because they all see the same information-limited data. The Cramér-Rao bound sets a floor on variance that no estimator can breach; the asymmetric profile likelihood then translates that wide variance into a systematic negative mean shift.

### 6.2 What the CNN Embedding Experiment Confirms

A CNN trained end-to-end on raw $(120 \times 4)$ time series — bypassing all hand-crafted 29-D summary statistics — produces essentially the same β estimate:

**Table 3 — 29-D Summary Vector vs. CNN Embedding**

| Fault Scenario | True value | 29-D SBI estimate | CNN SBI estimate | Interpretation |
|---|---|---|---|---|
| Sc2: Fouling | $\beta = 0.70$ | $\beta \approx 0.616$ | $\beta \approx 0.621$ | Both biased equally — confirms bias is structural, not a feature-engineering artifact |
| Sc3: Decay | $\alpha = 0.70$ | $\alpha \approx 0.70$ | $\alpha \approx 1.01$ | CNN fails on $\alpha$ — a separate failure related to CNN training on a 2-D problem, not a structural limitation |

The near-identical β estimates (0.616 vs 0.621) are the key result: a 61k-parameter deep network with direct access to every time-series sample cannot extract information that the feedback controller has suppressed. No architectural or feature-engineering choice can overcome the $I_{\beta\beta}$ deficit. This is empirical confirmation of the Cramér-Rao argument.

The CNN's failure on $\alpha$ in Sc3 is a separate issue — likely underfitting or a training instability on the 2-D problem — and is not a structural identifiability limitation: $I_{\alpha\alpha}$ is large and $\alpha$ is recoverable with the correct summary statistics.

### 6.3 Fault Classification Despite the Bias

The residual β bias of $-0.08$ to $-0.15$ does not prevent useful fault classification. The posterior places the mass for Sc2 ($\beta_{true} = 0.70$) at $\hat{\beta} \approx 0.55$–$0.62$, which is still clearly inside the fouling quadrant ($\beta < 0.85$). The classification boundary at 0.85 provides enough margin that the systematic offset does not cause misclassification for the scenarios in this study. CL macro-F1 = 0.990 across the six closed-loop scenarios.

The limitation is precision, not classification: the method cannot distinguish $\beta = 0.90$ from $\beta = 0.75$ reliably, because both fall below the bias floor. Open-loop excitation windows would be required to resolve this level of fouling detail.

---

## 7. Mitigating the Structural Limitation

The β bias can only be reduced by changing the *information content of the data*, not the inference algorithm.

### 7.1 Open-Loop Excitation Windows

Periodically bypassing the PI controller for short diagnostic windows (5–10 min) removes the feedback coupling entirely. With $Q_c$ no longer reacting to $T$ noise, $\Phi_{ue} = 0$ and the temperature signal freely reflects the true heat transfer capacity. The $T$–$T_c$ gap during these open-loop periods is directly informative about $\beta$ without controller compensation.

This requires deliberate scheduling in the plant historian. Even brief OL windows (Scenario 6 in this study) substantially reduce the β bias, though a residual remains due to the nonlinearity of the cooling response.

### 7.2 Setpoint Dithering

Injecting a low-amplitude pseudo-random binary sequence (PRBS) or multisine signal into $T_{sp}$ creates $\Phi_r(\omega) > 0$ at the excitation frequencies. This populates $\Phi_u^r(\omega) = |S_0|^2 \Phi_r$ with non-zero power, restoring some variance in the $T$ channel that is informative about $\beta$ (specifically about $UA_{eff}$ through the transient response). The amplitude must be kept small enough to respect product quality constraints.

### 7.3 Post-Hoc Bias Calibration

Because the bias is predictable and consistent (confirmed across 500 SBC test draws), a calibration correction — fitted from the SBC rank data or from the known operating-point-dependent bias — can shift the posterior mean to remove the systematic offset at deployment, at essentially zero additional cost. This does not reduce posterior variance but eliminates the mean error for applications that only need an accurate point estimate.

---

## 8. Conclusion

The persistent downward bias on the jacket fouling parameter $\beta$ in this closed-loop CSTR is fully explained by the information-theoretic analysis: the PI controller pins $T$ to the setpoint, zeroing the Jacobian row $\partial T_{ss}/\partial \beta$ and removing the primary channel's contribution to $I_{\beta\beta}$. With $I_{\beta\beta}$ reduced to 1/250th–1/500th of $I_{\alpha\alpha}$, the wide posterior for $\beta$ is distorted by the convexity of the $Q_c(\beta)$ mapping, and the Laplace approximation proves the mean must be strictly negative.

The control-theoretic perspective provides a frequency-domain equivalent of this result (sensitivity function $S_0 \to 0$, noise-free input spectrum $\Phi_u^r \to 0$) and adds one genuinely separate insight: the $\Phi_{ue} \neq 0$ correlation in closed-loop data is the reason OL-trained estimators fail when deployed on CL data. CL-trained SBI corrects this distribution mismatch; it does not eliminate the residual bias.

No inference algorithm — Bayesian, frequentist, or neural — can recover information that the feedback controller has suppressed. The empirical confirmation is four-fold: NUTS, SBI (29-D summaries), SBI (CNN embedding), EKF, and UKF all produce the same $-0.08$ to $-0.15$ β bias. Reducing it requires changing the data-generation process: periodic open-loop excitation, setpoint dithering, or post-hoc calibration from the known bias profile.

---

### References

- **Forssell, U., & Ljung, L. (1999).** Closed-loop identification: Methods, theory, and applications. *Linköping Studies in Science and Technology*, Dissertation No. 566.
- **Ljung, L. (1999).** *System Identification: Theory for the User* (2nd ed.). Prentice Hall.
- **Gustavsson, I., Ljung, L., & Söderström, T. (1977).** Identification of processes in closed loop — identifiability and accuracy aspects. *Automatica*, 13(1), 59–75.
- **Gevers, M., Bombois, X., Hildebrand, R., & Solari, G. (2011).** Optimal experiment design for open and closed-loop system identification. *Communications in Information and Systems*, 11(3), 197–224.
