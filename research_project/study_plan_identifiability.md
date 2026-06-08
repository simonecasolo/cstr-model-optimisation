# Study plan: closed-loop identifiability theory and analytical bias derivation

## Part A — Background reading

The goal is to understand the theoretical foundations behind three results
in this project:

1. The Fisher information ratio I_ββ/I_αα = 250-500x (nb15)
2. The Cramér-Rao argument for irreducibility (nb14 §6)
3. The embedding-net confirmation (nb04b)

### A.1 System identification under feedback (1-2 days)

**Core question**: why does closing a feedback loop reduce the information
content of output data for estimating certain plant parameters?

**Read in this order:**

1. **Ljung (1999), Chapter 13** — *System Identification: Theory for the User*,
   2nd ed., Prentice Hall. Sections 13.3-13.5 ("Closed-loop identification").
   This is the canonical textbook treatment. Focus on:
   - Why does the feedback path create a correlation between the input u(t) and
     the noise e(t)? (Section 13.3, the "direct method" problem)
   - Under what conditions is a closed-loop system still identifiable?
     (Theorem 13.1: requires external excitation or specific model structure)
   - How does the information matrix degrade as the controller gets tighter?

2. **Gustavsson, Ljung & Söderström (1977)**, "Identification of processes in
   closed loop — identifiability and accuracy aspects", *Automatica* 13(1), 59-75.
   The foundational paper. Focus on:
   - Theorem 1 (identifiability conditions for closed-loop)
   - Section 4 (accuracy analysis — this is where the Fisher information
     connection appears for the first time in this context)
   - The key insight: the controller makes the input a *function of past outputs*,
     so the input no longer provides independent excitation for all parameters

3. **Forssell & Ljung (1999)**, "Closed-loop identification revisited",
   *Automatica* 35(7), 1215-1241. A modern summary and unification. Skim for:
   - Table 1 (comparison of methods: direct, indirect, joint I/O)
   - The distinction between "informativity" and "identifiability" in closed-loop

**Self-check**: after reading, you should be able to answer:
- In our CSTR, what plays the role of "external excitation"? (Answer: process noise)
- Why is α more identifiable than β? (Answer: α affects C, which is not controlled;
  β affects T, which the PI controller regulates — so β's effect is compensated in
  the T channel, and only leaks into Qc indirectly)

### A.2 Fisher information for experiment design (1 day)

**Core question**: how is the Fisher information matrix used to quantify
identifiability, and what does the Cramér-Rao bound actually guarantee?

**Read:**

4. **Gevers, Bombois, Hildebrand & Solari (2011)**, "Optimal experiment design
   for open and closed-loop system identification", *Communications in
   Information and Systems* 11(3), 197-224. Focus on:
   - Section 2: definition of the Fisher information matrix for a parametric
     model, and its connection to the Cramér-Rao lower bound
   - Section 3: how the FIM depends on the experiment (input spectrum, noise
     spectrum, controller) — equation (7) is the key expression
   - Section 4.2: the closed-loop case — how the controller enters the FIM

5. **Bombois, Scorletti, Gevers, Van den Hof & Hildebrand (2006)**, "Least
   costly identification experiment for control", *Automatica* 42(10), 1651-1662.
   Skim Section 2 for the FIM decomposition and how controller tightness trades
   off against parameter accuracy.

**Self-check**: after reading, you should be able to:
- Write down the general formula I(θ) = J^T Σ^{-1} J and explain each term
  (J = Jacobian of expected observations w.r.t. parameters, Σ = observation
  noise covariance)
- Explain why Var(θ̂) >= I(θ)^{-1} for any unbiased estimator (Cramér-Rao)
- Explain why the FIM in our nb15 uses 29-D summaries rather than raw time
  series (the summaries are sufficient statistics for the neural posterior;
  the FIM computed on them lower-bounds the FIM on the raw data, so our
  250-500x ratio is conservative)

### A.3 The Cramér-Rao bound and Bayesian posteriors (half day)

**Core question**: our estimator is Bayesian (posterior mean), not MLE.
Does the Cramér-Rao bound still apply?

**Read:**

6. **Van Trees (1968)**, Chapter 2 — or any Bayesian estimation textbook.
   The **Bayesian Cramér-Rao bound** (Van Trees inequality) states:

   E[(θ̂ - θ)^2] >= 1 / (I_F(θ) + I_P(θ))

   where I_F is the Fisher information from the data and I_P is the
   "prior information" (= -E[d²log p(θ)/dθ²]). For a uniform prior,
   I_P = 0, so the bound reduces to 1/I_F — identical to the frequentist
   Cramér-Rao bound. Since our prior is uniform, the standard Cramér-Rao
   bound applies directly.

7. For the connection to SBI specifically: **Cranmer, Brehmer & Louppe (2020)**,
   "The frontier of simulation-based inference", *PNAS* 117(48), 30055-30062.
   Section 3.3 discusses how neural posterior estimation learns an approximation
   to the true posterior — the Cramér-Rao bound constrains the *true* posterior,
   and any approximate posterior can only be wider. So if the true posterior
   already has Var(β) >= 1/I_ββ, no neural network can do better.

**Self-check**: after reading, you should be able to:
- Explain to a reviewer why "no choice of inference method or summary statistics"
  can fix the β bias (answer: Cramér-Rao bounds the true posterior variance;
  summaries can only lose information, not create it; the embedding-net
  experiment confirms this empirically)
- Articulate the subtle point: the Cramér-Rao bound constrains *variance*, not
  *bias*. Our β estimator is biased. The connection is: when I_ββ is very small,
  the posterior is wide, and a wide posterior on a nonlinear problem is susceptible
  to bias because the posterior mean is pulled by asymmetry in the likelihood.
  This is the link between Fisher information and the observed bias.

### A.4 Fouling detection under feedback control (half day)

**Core question**: is the specific fouling-masking phenomenon known for other
systems?

**Read:**

8. **Heat exchanger fouling monitoring under closed-loop temperature control**
   (Chemical Engineering Research and Design, 2022). Key quote: "conventional
   fouling monitoring strategies based on heat transfer rate are not effective
   for heat exchangers with closed-loop temperature control." This is exactly
   our β problem in a different vessel.

9. **Isermann (2006)**, *Fault-Diagnosis Systems*, Chapter 10 — process fault
   diagnosis. Skim for how classical FDI (fault detection and isolation) handles
   the closed-loop masking problem. The standard approach is "residual
   generation" with observer-based methods — compare to our SBI approach.

---

## Part B — Analytical bias derivation for a simplified 2-state CSTR

**Goal**: derive, from first principles, why the posterior mean for β is
biased downward in closed-loop, and produce a closed-form expression (or
semi-analytical bound) that matches the numerically observed ~0.08 bias.

### B.0 The full CSTR system (reference)

The closed-loop CSTR has state [C, T, Tc, I] with:

```
k_eff  = α · k₀ · exp(-Ea/(R·T))
UA_eff = β · UA

dC/dt  = (Q/V)(Ci - C) - k_eff · C
dT/dt  = (Q/V)(Ti - T) - (Hᵣ · k_eff · C)/(ρ·Cₚ) - UA_eff·(T - Tc)/(ρ·Cₚ·V)
dTc/dt = (Qc/Vc)(Tci - Tc) + UA_eff·(T - Tc)/(ρc·Cpc·Vc)
dI/dt  = (T - Tsp) · gate

Qc = clip(Qc₀ + Kp·(T - Tsp) + I/τᵢ,  0,  Qc_max)
```

Parameters: Kp = 150 (L/min)/K, τᵢ = 10 min, Tsp = 312.5 K, Qc₀ = 80 L/min.

### B.1 Reduce to a 2-state system (T, Tc)

**Simplification**: at steady state with high controller gain, the concentration
equation decouples (C is determined by α and T, which is pinned near Tsp). Drop
the C and I states. Assume perfect integral control: T = Tsp exactly at steady
state (I adjusts to make the error zero). This gives:

```
0 = (Q/V)(Ti - Tsp) - (Hᵣ · α · k₀ · exp(-Ea/(R·Tsp)) · C_ss)/(ρ·Cₚ)
    - β·UA·(Tsp - Tc_ss)/(ρ·Cₚ·V)
```

where C_ss is determined by α:

```
C_ss(α) = Q·Ci / (Q + V·α·k₀·exp(-Ea/(R·Tsp)))
```

This is one equation in two unknowns (β, Tc_ss), with Qc as the free variable
the controller adjusts.

**Step 1**: Derive C_ss(α) explicitly by setting dC/dt = 0 at T = Tsp.

**Step 2**: Derive the steady-state heat balance to get Qc_ss(α, β) — the
coolant flow the controller must provide to maintain T = Tsp.

**Step 3**: Derive Tc_ss(Qc, β) from dTc/dt = 0.

This gives you the **observation model**: given (α, β), the steady-state
observables are (C_ss(α), Tsp, Tc_ss(α,β), Qc_ss(α,β)). The key feature is
that C_ss depends on α only, while Tc_ss and Qc_ss depend on both.

### B.2 Compute the analytical Jacobian

Define the 4-observable vector μ = [C_ss, T_ss, Tc_ss, Qc_ss] as a function
of θ = [α, β]. Compute the Jacobian J = ∂μ/∂θ analytically:

```
J = | ∂C_ss/∂α     ∂C_ss/∂β    |     | non-zero   0        |
    | ∂T_ss/∂α     ∂T_ss/∂β    |  =  | 0          0        |  (T = Tsp)
    | ∂Tc_ss/∂α    ∂Tc_ss/∂β   |     | non-zero   non-zero |
    | ∂Qc_ss/∂α    ∂Qc_ss/∂β   |     | non-zero   non-zero |
```

The critical structure:
- Row 1 (C): only depends on α. Full sensitivity to α, zero to β.
- Row 2 (T): zero sensitivity to both (controller pins it). **This is the
  information loss.**
- Rows 3-4 (Tc, Qc): depend on both, but through a coupled mapping.

### B.3 Compute the analytical Fisher information matrix

With Gaussian observation noise Σ = diag(σ²_C, σ²_T, σ²_Tc, σ²_Qc):

```
I(α, β) = J^T · Σ^{-1} · J
```

Expand:

```
I_αα = (∂C/∂α)²/σ²_C + (∂Tc/∂α)²/σ²_Tc + (∂Qc/∂α)²/σ²_Qc
I_ββ = (∂Tc/∂β)²/σ²_Tc + (∂Qc/∂β)²/σ²_Qc
I_αβ = (∂Tc/∂α)(∂Tc/∂β)/σ²_Tc + (∂Qc/∂α)(∂Qc/∂β)/σ²_Qc
```

**The key result**: I_αα has a large contribution from (∂C/∂α)²/σ²_C — the
concentration channel, which has high sensitivity to α and low noise. I_ββ
has NO contribution from the C or T channels. It relies entirely on Tc and
Qc, which are noisier and have lower sensitivity because the controller
absorbs β changes.

**Step 4**: Plug in the CSTR parameter values and compute I_αα/I_ββ
analytically. Compare to the numerically computed 250-500x from nb15.

### B.4 Profile likelihood for β

The profile likelihood removes α by maximizing over it:

```
L_prof(β) = max_α  p(x_obs | α, β)
```

For Gaussian noise on the 4 observables:

```
log L_prof(β) = max_α  -½ Σᵢ [(xᵢ_obs - μᵢ(α,β))² / σᵢ²]
```

Since C_ss depends only on α, the optimal α*(β) is determined by the
concentration channel alone (for small noise). Substituting α*(β) back:

```
log L_prof(β) ≈ -½ [(Tc_obs - Tc_ss(α*(β), β))² / σ²_Tc
                   + (Qc_obs - Qc_ss(α*(β), β))² / σ²_Qc]
                + const
```

**Step 5**: Compute L_prof(β) on a grid of β values using the analytical
steady-state expressions. Plot it and show:
- It is wide (low curvature) — confirming small I_ββ
- It is asymmetric — steeper on one side, which pulls the posterior mean
  away from the true β

**Step 6**: The curvature of log L_prof at the true β is exactly the
"profile Fisher information" for β:

```
I_prof(β) = -d²/dβ² log L_prof(β)|_{β=β_true}
```

This should be close to the (2,2) entry of I^{-1} from the full FIM.

### B.5 Analytical bias from asymmetry

For a 1D posterior with a weakly informative likelihood, the bias of the
posterior mean relative to the MAP is approximately:

```
bias ≈ -(1/2) · L'''(β*) / [L''(β*)]²
```

where L is the log-profile-likelihood and β* is the mode. This is the
**skewness correction** from a Laplace approximation.

**Step 7**: Compute L''(β*) and L'''(β*) from the analytical profile
likelihood. Evaluate the bias formula and compare to the observed ~0.08.

If the formula gives the right sign (negative = downward bias) and
approximately the right magnitude, you have a closed-form explanation.
If the magnitude is off, the discrepancy comes from the dynamic
(non-steady-state) information in the 29-D summaries that the 4-observable
steady-state model does not capture.

### B.6 Verify numerically

**Step 8**: Implement all of the above in a new notebook (or extend nb15)
and compare:

| Quantity | Analytical (2-state) | Numerical (nb15, 29-D) |
|----------|---------------------|----------------------|
| I_αα     | from B.3            | ~900k (nb15)         |
| I_ββ     | from B.3            | ~2000 (nb15)         |
| I_αα/I_ββ | from B.3          | 250-500x (nb15)      |
| Profile L width | from B.4    | (new)                |
| Bias prediction | from B.5    | ~0.08 observed       |

If the analytical 2-state model captures the qualitative structure
(I_αα >> I_ββ, asymmetric profile likelihood, correct bias sign) but
underestimates the magnitude, that's still a strong result — it shows the
mechanism analytically and the residual is attributable to dynamic
information not captured by the steady-state approximation.

### B.7 Write-up for the paper

The analytical derivation gives you a **Proposition** for the paper:

> **Proposition.** For a PI-controlled CSTR with reaction rate parameter α
> and heat-transfer coefficient scaling β, the Fisher information satisfies
> I_αα/I_ββ >= (∂C_ss/∂α)² σ²_Qc / [(∂Qc_ss/∂β)² σ²_C] >> 1
> whenever the controller holds T ≈ Tsp. The ratio grows with controller
> gain Kp (tighter control erases more temperature information).

This connects directly to Ljung (1977) — you're deriving the specific
form of the classical identifiability loss for your system — and to the
Cramér-Rao bound for the irreducibility claim.

---

## Timeline estimate

| Block | Estimated time | Dependencies |
|-------|---------------|--------------|
| A.1 Ljung + Gustavsson | 1-2 days reading | None |
| A.2 Gevers + Bombois | 1 day reading | A.1 |
| A.3 Bayesian Cramér-Rao | Half day | A.2 |
| A.4 Fouling literature | Half day | None (parallel with A.1-A.3) |
| B.1-B.3 Steady-state + Jacobian + FIM | 1 day derivation | A.1-A.3 |
| B.4-B.5 Profile likelihood + bias | 1 day derivation | B.1-B.3 |
| B.6-B.7 Numerical verification + write-up | 1-2 days coding | B.4-B.5 |

**Total: ~1 week** (3-4 days reading, 3-4 days derivation and coding).
