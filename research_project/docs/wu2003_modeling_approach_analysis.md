# Wu 2003 Modeling Approach Analysis: Current Shortcut, Minimal Explicit Recycle, and Paper-Aligned Control

## Purpose

This note compares three levels of modeling fidelity for the Wu 2003 CSTR-column-recycle extension:

1. **Current nb22 shortcut**: a 4-state reactor/jacket ODE, quasi-steady/empirical column closure, and S-A/S-B represented as different measurement views of the same trajectory.
2. **Minimal explicit recycle/control implementation**: keep a simplified quasi-steady column, but make reflux ratio/reflux and vapor boilup/reboiler actions explicit controller outputs, and simulate S-A and S-B as genuinely different control structures.
3. **Paper-aligned rigorous dynamic control approach**: the approach used in the two PDFs in `docs/`, which uses plantwide control structures, explicit manipulated variables such as reflux flow, reflux ratio, vapor boilup, and boilup ratio, and rigorous dynamic simulation under simplified distillation assumptions.

The question is not simply which model is most detailed. The practical question is which implementation is adequate for the objectives in [project_wu2003_sbi.md](../project_wu2003_sbi.md), [article_outline_CChE.md](../article_outline_CChE.md), and [publication_assessment.md](../publication_assessment.md): plant-wide closed-loop identifiability, recycle snowball coupling, partial observability, non-Gaussian posteriors, EKF limitations, and amortised SBI at a monitoring cadence.

## Executive Recommendation

The recommended path is the **minimal explicit recycle/control implementation**.

It should replace the current nb22 shortcut before the Wu 2003 case is used as a central publication result. It is more faithful to the papers because S-A and S-B become true control structures, not only different channel selections. At the same time, it avoids the risk of a full dynamic tray/MESH model, which is not necessary for the paper's inferential objectives and would add numerical and calibration burden disproportionate to the scientific gain.

In short:

- The **current nb22 shortcut is adequate as a baseline/prototype**, but too weak as the final C&ChE plant-wide case because it does not instantiate the published S-A/S-B control-structure difference.
- The **minimal explicit recycle/control implementation is adequate and recommended** because it captures the key mechanisms needed by the paper: multi-loop feedback masking, control compensation signals, recycle snowball coupling, and S-A/S-B partial observability.
- The **risky full plan is not recommended for the current paper** unless validation shows the minimal model cannot reproduce the expected closed-loop signatures. A full tray/MESH model would be an optional future extension, not a prerequisite for the C&ChE objectives.

## Evidence from the Two PDFs

Text was extracted from the two PDFs in `docs/` using Ghostscript text output. The extracted text shows that the papers are not equivalent to the current nb22 shortcut, but also do not require a maximal first-principles tray-by-tray MESH implementation.

### PDF 1: [docs/0098-1354%2895%2900248-0.pdf](../docs/0098-1354%2895%2900248-0.pdf)

This paper studies the reactor/separator recycle process and the snowball effect. The extracted text includes the following evidence:

- The process is a CSTR feeding a **20-tray distillation column**.
- The column has a **partial reboiler** and a **total condenser**.
- The model assumes **constant relative volatility**.
- The paper discusses reactor/separator control structures for avoiding or mitigating the **snowball effect**.
- It explicitly mentions manipulation of **reflux flow rate** and **vapor boil-up**.
- In the dynamics and control section, it states that the process is analyzed using **rigorous dynamic simulations**.
- It states that assumptions of **theoretical tray**, **equimolar overflow**, and **constant relative volatility** are made for modeling the distillation column.
- It uses PI controllers for quality loops after inventory loops are under control.

That is not the same as current nb22. Current nb22 has no explicit reflux or boilup loop. However, the PDF also does not demand the highest-risk full plan in which every tray hydraulic and energy balance is built as a new high-order JAX model. It uses rigorous dynamic simulation with standard simplifying distillation assumptions.

### PDF 2: [docs/1-s2.0-S0098135402002181-main.pdf](../docs/1-s2.0-S0098135402002181-main.pdf)

This is the Wu, Yu, Luyben, and Skogestad reactor/separator recycle paper used in the project plan. The extracted text shows:

- The paper is explicitly about **plantwide control structure design**.
- It compares **full composition control** and **partial composition control** structures.
- It defines variables including:
  - `R`: reflux flow rate,
  - `RR`: reflux ratio,
  - `V`: vapor boilup rate,
  - `BR`: boilup ratio.
- The plant is again a CSTR coupled to a **20-tray distillation column** with **partial reboiler** and **total condenser**.
- Table information includes nominal reflux flow, vapor boilup, number of trays, feed tray, relative volatility, and liquid hydraulic time constant.
- Control structures are explicitly named and compared:
  - `B-3`: full composition control,
  - `B-2`: two composition loops,
  - `B-1`, `B-1a`, `B-1b`, `B-1c`: partial composition or ratio-based alternatives.
- The paper says the dynamics are analyzed using **rigorous dynamic simulations**.
- It includes analyzer dead time and temperature measurement lag in the dynamic control study.
- It describes structures where the **reflux ratio is fixed**, the reflux drum level is maintained by changing reflux flow, and bottoms composition is controlled by changing **boilup ratio**.

This supports a paper-aligned implementation with explicit control policies and manipulated variables. It does not support keeping S-A/S-B as only measurement projections of the same trajectory.

## The Three Candidate Approaches

### 1. Current nb22 Shortcut

The current nb22 implementation uses:

- Reactor/jacket state vector:

```text
[z_A, T_r, T_j, I_T]
```

- One explicit controller:

```text
Loop 1: reactor temperature -> jacket cooling duty Q_j
```

- A column closure approximately of the form:

```text
column_qss(z_F, eta_col) -> x_D, x_B, D_frac
```

- Recycle flow derived algebraically from the column split.
- `T_reb` and `Q_reb` computed as proxy outputs.
- S-A and S-B generated by projecting the same raw trajectory into different channel sets:
  - S-A includes `x_D`.
  - S-B excludes `x_D`.

The current nb22 dataset is valuable. It verifies the basic data contract:

- 23 scenarios,
- 30 replicates,
- 2 structures,
- 120 time points,
- S-A shape `(690, 120, 8)`,
- S-B shape `(690, 120, 7)`,
- no NaNs or infinities,
- visible snowball onset, masking, compensation, column degradation, and open-loop contrast.

But the shortcut has a structural limitation: S-A and S-B are not separate closed-loop control policies. They are two views of the same dynamics.

#### Strengths

- Fast and stable.
- JAX/diffrax-friendly.
- Adequate for testing data-generation plumbing.
- Captures the first-order paper mechanisms: recycle snowball, reactor temperature masking, compensation through controller outputs, and S-A/S-B measurement asymmetry.
- Useful as a regression baseline before more explicit control loops are added.

#### Weaknesses

- It does not implement the published plantwide control-structure distinction.
- It cannot test whether S-A and S-B produce different plant trajectories.
- It cannot show reflux ratio, reflux flow, vapor boilup, or boilup ratio as compensation channels.
- It makes column/reboiler masking largely implicit rather than observable in controller outputs.
- It risks a reviewer objection: the paper claims a Wu 2003 plantwide control comparison, but the implementation only changes measurement channels.

### 2. Minimal Explicit Recycle/Control Implementation

This approach keeps the model compact but makes the relevant control loops explicit.

A minimal state vector would be:

```text
[z_A, T_r, T_j, I_T, I_L, I_V]
```

or, if actuator lags/rate limits are needed:

```text
[z_A, T_r, T_j, I_T, R_state, V_state, I_L, I_V]
```

The column can remain quasi-steady, but its closure should accept explicit control variables:

```text
column_qss(z_F, eta_col, R_or_L, V_or_BR) -> x_D, x_B, D_frac
```

The control structures would become real simulation modes:

```text
S-A:
  Loop 1: T_r -> Q_j
  Loop 2: x_D -> reflux ratio R or reflux flow L
  Loop 3: x_B -> vapor boilup V or boilup ratio BR

S-B:
  Loop 1: T_r -> Q_j
  Loop 2: recycle/fresh ratio or reflux ratio policy -> L or R
  Loop 3: T_reb -> V, BR, or Q_reb
```

Dataset generation would change from:

```text
one trajectory per scenario -> project to S-A and S-B
```

to:

```text
one S-A trajectory per scenario + one S-B trajectory per scenario
```

This doubles deterministic integrations from 23 to 46, which is still very manageable compared with SBI training.

#### Strengths

- Captures the key control-structure distinction in Wu 2003.
- Keeps the column fast enough for repeated simulation and SBI data generation.
- Makes `R`, `L`, `V`, `BR`, and/or `Q_reb` visible as compensation signals.
- Preserves the planned 5-D degradation vector unless controller uncertainty becomes a scientific question.
- Supports the paper's comparison of conventional measurement control versus composition-analyzer control.
- Avoids introducing many poorly identified tray holdup/energy parameters.

#### Weaknesses

- Still approximate relative to the rigorous dynamic simulations in the papers.
- Requires careful sign conventions for the reflux and boilup controllers.
- Requires validation that the QSS column with explicit manipulated variables remains causal and smooth.
- May need small actuator/column response lags to avoid instantaneous algebraic artifacts.

### 3. Paper-Aligned Rigorous Dynamic Control Approach

The papers use explicit control structures and rigorous dynamic simulation. The extracted text indicates:

- dynamic simulation rather than only steady-state sensitivity,
- theoretical-tray assumptions,
- equimolar overflow,
- constant relative volatility,
- column and reactor holdups,
- reflux flow and vapor boilup as central variables,
- composition, ratio, level, and temperature control loops.

A full implementation in this style would likely require:

- dynamic column states for tray liquid compositions and possibly holdups,
- condenser and reboiler level dynamics,
- explicit reflux drum and bottoms inventory loops,
- vapor/liquid traffic relationships,
- controller dead time and measurement lag,
- several additional controller states,
- substantially more validation against the paper figures.

This is the most faithful route, but it is also the most expensive and riskiest.

#### Strengths

- Closest to the simulation studies in the PDFs.
- Strongest defense against a fidelity-focused process control review.
- Can reproduce individual control-structure response plots more directly.
- Makes reflux, boilup, level, and composition dynamics physically explicit.

#### Weaknesses

- Much higher implementation complexity.
- More states and more stiffness/solver risk.
- More tuning and validation burden.
- More numerical failure modes in SBI data generation.
- Additional poorly known parameters may distract from the paper's inferential contribution.
- Full fidelity may not improve the central identifiability story enough to justify the cost.

## Comparison Matrix

| Dimension | Current nb22 | Minimal explicit recycle/control | Paper-aligned rigorous dynamic control |
|---|---:|---:|---:|
| Reactor dynamics | Explicit | Explicit | Explicit |
| Loop 1 reactor temperature | Explicit | Explicit | Explicit |
| Reflux/reflux-ratio loop | Hidden/implicit | Explicit | Explicit |
| Reboiler/boilup loop | Proxy only | Explicit | Explicit |
| S-A vs S-B | Measurement projection | Different control policies | Different control policies |
| Column model | QSS/empirical closure | QSS with explicit manipulated variables | Dynamic theoretical-tray simulation |
| Reflux/boilup observations | Not explicit | Available | Available |
| Computational cost | Lowest | Low/moderate | High |
| Risk of solver/data-generation failures | Low | Moderate | High |
| Fidelity to paper control structures | Partial | Good | Best |
| Adequacy for SBI objectives | Prototype only | Recommended | More than needed initially |

## Fit to project_wu2003_sbi.md Objectives

The project plan identifies six reasons for the Wu 2003 extension:

1. published parameters,
2. correct CSTR-column-recycle topology,
3. snowball dynamics,
4. JAX implementability with a small QSS model,
5. published control structures,
6. continuity with the propylene oxide CSTR masking mechanism.

The minimal explicit recycle/control implementation satisfies these objectives better than both alternatives.

### Published Parameters

The current nb22 already uses the key published parameters: reactor holdup, nominal temperatures, kinetic constants, column tray count, reflux ratio, recycle flow, bottoms flow, vapor boilup, and compositions.

The minimal explicit implementation keeps these parameters as anchors and adds explicit nominal values for reflux/reboiler manipulated variables. It should not expand the SBI degradation vector initially. That preserves the clean 5-D inference problem:

```text
[alpha, beta_r, eta_col, xi_reb, z_A0_eff]
```

The full dynamic plan would require more poorly constrained internal parameters: tray holdups, condenser/reboiler hydraulic constants, valve gains, actuator lags, measurement lags, and perhaps vapor/liquid traffic correlations. That may be faithful to a control simulator, but it weakens the claim that the SBI benchmark is grounded in complete published parameters.

### Correct Topology

All approaches include the CSTR, distillation column, and recycle loop. The difference is whether the control topology is real.

The current nb22 has the plant topology but not the real S-A/S-B control topology. The minimal explicit implementation adds the missing control topology without overbuilding the column.

That matters because [project_wu2003_sbi.md](../project_wu2003_sbi.md) explicitly positions the Wu case as a plantwide feedback-control extension, not just a larger observation vector.

### Snowball Dynamics

The project needs snowball dynamics because catalyst decay and column degradation should become coupled through recycle flow. The minimal implementation is adequate if it preserves the following causal chain:

```text
alpha decreases
  -> reactor conversion decreases
  -> reactor outlet z_A increases
  -> column/recycle load changes
  -> F_R increases
  -> reactor dilution increases
  -> alpha and eta_col become coupled in posterior geometry
```

Current nb22 already shows snowball onset numerically, but the compensation is embedded in an empirical `D_frac` response. The minimal explicit model can keep the same snowball backbone while making reflux and boilup compensation observable.

The full dynamic column may change the precise transient shape, but the inferential objective only requires the coupled nonlinear mapping from degradation parameters to closed-loop observations.

### JAX Implementability

The project plan explicitly values a small JAX-implementable model. This strongly favors the minimal explicit implementation.

A 6- to 8-state ODE with a QSS column is compatible with:

- `diffrax` integration,
- `jax.jacobian` for EKF,
- batched data generation,
- prior predictive simulation,
- SBI training at thousands of simulations,
- notebook-based verification.

A full dynamic theoretical-tray model would be feasible but much more fragile. It would consume effort on numerical stability and column-model validation instead of the paper's main contribution.

### Published Control Structures

This is where current nb22 is weakest.

The project plan says the paper needs PWC-A/PWC-B or S-A/S-B as published control structures. The PDFs support this: they compare full and partial composition control, ratio-based alternatives, reflux ratio, boilup ratio, level control, and composition loops.

The minimal explicit implementation is enough to meet this requirement if:

- S-A and S-B are separate control modes,
- S-A has composition-analyzer control paths,
- S-B has conventional ratio/temperature paths,
- the dataset is generated from separate trajectories,
- the channel sets preserve the planned partial-observability asymmetry.

A full dynamic tray model is not necessary for that publication objective.

### Continuity with Propylene Oxide Masking

The article needs the reactor jacket masking mechanism from the PO case to generalize into the plantwide case. This mechanism is already in the current model through Loop 1:

```text
beta_r affects heat transfer
  -> Loop 1 compensates with Q_j
  -> T_r remains near setpoint
  -> beta_r information moves into Q_j and T_j
```

The minimal implementation preserves this exactly and adds analogous column-loop compensation:

```text
eta_col or xi_reb changes separation/reboiler behavior
  -> Loop 2/3 compensate via R/L/V/BR/Q_reb
  -> product-quality variables may be masked
  -> controller outputs carry fault information
```

That is the plantwide analogue required by the paper.

## Fit to article_outline_CChE.md Objectives

The article outline says C&ChE reviewers will look for:

1. a process systems engineering contribution,
2. a model grounded in published parameters,
3. comparison with industrial baselines such as EKF/UKF,
4. scalability from single-unit to plantwide,
5. honest treatment of identifiability limitations.

### Process Systems Engineering Contribution

The contribution is not merely applying SBI to time series. It is about how decentralized feedback control changes fault identifiability.

The current nb22 shortcut risks undercutting this because the S-A/S-B difference is only a sensor-selection difference. A reviewer could reasonably ask: where are the different plantwide control structures?

The minimal explicit implementation answers that objection while staying focused. It gives the paper a defensible process-control mechanism:

- Loop 1 masks reactor fouling.
- Loop 2/3 mask or expose column faults depending on S-A/S-B.
- Recycle couples reactor and column degradation through snowball behavior.
- Controller outputs become diagnostic channels.

### Published-Parameter Grounding

The full dynamic plan could paradoxically weaken this claim because it would require filling in missing dynamic and controller details. The minimal explicit implementation keeps the model anchored to the published nominal table and uses a controlled number of additional tuning constants.

That is a better fit for C&ChE if the paper is honest that the model is a control-oriented simulator rather than an exact reproduction of every tray dynamic.

### EKF/UKF Baselines

The article outline emphasizes that EKF is mandatory and UKF is welcome. The minimal explicit model is much easier to support with EKF because:

- the state dimension remains modest,
- the RHS remains JAX-differentiable,
- the augmented parameter-state vector remains manageable,
- Jacobians remain interpretable,
- failure modes can be attributed to posterior geometry rather than model numerical noise.

The full dynamic model would make EKF implementation and tuning much more complex. That may be publishable in a different paper, but it is a distraction here.

### Scalability

The two-system story is:

```text
PO CSTR: one unit, one PI loop, 2 degradation parameters
Wu recycle plant: two units, recycle, multiple control loops, 5 degradation parameters
```

The minimal explicit implementation supports this scaling story. It demonstrates a meaningful jump in plant topology and control complexity while staying computationally feasible for SBI.

The current nb22 shortcut scales the topology but not the control structure. The full plan may overscale the model and compromise the ability to complete the inferential experiments.

### Honest Identifiability Limitations

The article outline explicitly says not to hide beta bias or closed-loop limitations. The minimal explicit implementation makes it easier to show the limitations cleanly:

- beta_r masking remains visible through Loop 1,
- eta_col and alpha coupling appears through recycle flow,
- S-B lacks direct composition information,
- S-A improves identifiability by measuring and controlling composition,
- EKF collapses non-Gaussian posterior geometry.

This is exactly the story the paper wants.

## Fit to publication_assessment.md Objectives

The publication assessment says the strongest eventual paper is a two-system progression:

1. PO system: structural beta bias verified by SBI, NUTS, EKF, UKF, Fisher information, and CNN embedding.
2. Recycle plant: plantwide masking, snowball fault attribution, non-Gaussian posteriors, EKF overconfidence, and SBI as the practical full-Bayesian method.

The minimal explicit implementation is adequate for this because it supports the essential claims.

### Claim 1: Structural Identifiability Limitation Generalizes

The PO case established that reactor temperature control masks beta-like thermal faults. In the Wu plant, the model must show that this mechanism persists with recycle and column dynamics.

The minimal implementation preserves the same reactor thermal structure and adds plantwide loops. It is enough to show generalization.

### Claim 2: Recycle Creates a New Non-Gaussian Coupling

The publication assessment highlights banana-shaped posteriors for coupled reactor/column degradation. This does not require a full tray model. It requires a nonlinear mapping in which two parameters can produce similar recycle-flow and product-quality signatures.

A QSS column with explicit reflux/boilup loops can create exactly that mapping:

```text
alpha decrease -> higher z_A and recycle demand
eta_col decrease -> poorer separation and altered recycle/product split
both -> similar F_R/T_reb/Q_reb signatures under S-B
```

This is sufficient for the SBI-vs-EKF argument.

### Claim 3: SBI is Practical Where MCMC/EKF Are Not

The full dynamic plan could make MCMC and EKF infeasibility obvious, but for the wrong reason: the simulator would be too heavy. The paper's stronger claim is not merely that a complex simulator is slow. The stronger claim is:

```text
At realistic plantwide dimension and monitoring cadence, amortised SBI is practical,
while EKF gives the wrong posterior geometry and MCMC is too slow.
```

The minimal explicit implementation provides enough complexity to test that claim without making simulation cost dominate the story.

### Claim 4: Snowball Fault Attribution

The publication assessment identifies snowball fault attribution as a key novelty. A full tray model is not needed to demonstrate root-cause ambiguity and posterior coupling. What is needed is a physically grounded recycle feedback loop and controller-mediated masking.

The minimal explicit model supplies those ingredients.

## What the Recommended Model Should Contain

The recommended implementation should include the following concrete elements.

### State Vector

Start with:

```text
y = [z_A, T_r, T_j, I_T, I_L, I_V]
```

If needed for smoothness:

```text
y = [z_A, T_r, T_j, I_T, R_state, V_state, I_L, I_V]
```

The second version is preferred if direct algebraic changes in `R` and `V` create discontinuities or unrealistic instantaneous response.

### Control Modes

Define a structure mode independent of open-loop/closed-loop:

```text
structure = "S-A" or "S-B"
```

Define an open-loop policy separately:

```text
loop_policy = "closed" | "loop1_open" | "all_open" | "column_open"
```

This avoids the current ambiguity where open-loop primarily means bypassing the reactor temperature controller.

### S-A Controls

```text
Loop 1: T_r -> Q_j
Loop 2: x_D -> R or L
Loop 3: x_B -> V or BR
```

S-A should include online composition analyzers and therefore should be able to correct `x_D` and `x_B` more directly.

### S-B Controls

```text
Loop 1: T_r -> Q_j
Loop 2: recycle/fresh ratio or reflux-ratio policy -> L/R
Loop 3: T_reb -> V/BR/Q_reb
```

S-B should not use `x_D` or `x_B` as feedback measurements. That is the partial-observability mechanism.

### Column Closure

Use a QSS column closure with explicit manipulated variables:

```text
x_D, x_B, D_frac = column_qss(z_F, eta_col, R, V_norm)
```

or:

```text
x_D, x_B, D_frac = column_qss(z_F, eta_col, R, BR)
```

The closure should be monotone and anchored at the nominal Wu values:

```text
R_nom = 2.198
L_nom = 1100 lbmol/h
V_nom = 1600.4 lbmol/h
x_D_nom = 0.95
x_B_nom = 0.0105
D_nom = 500.4 lbmol/h
B_nom = 460.0 lbmol/h
```

### Observation Channels

Raw observations should expand from 10 channels to about 12-14 channels:

```text
z_A
T_r
T_j
Q_j
x_D
x_B
F_R_norm
T_reb
Q_reb
F_B_norm
R_norm or L_norm
V_norm or BR
optional saturation flags
```

The final training channel sets should remain aligned with the publication story:

```text
S-A: includes x_D, plus conventional measurements and controller outputs
S-B: excludes composition analyzer channels, but includes conventional measurements and controller outputs
```

Whether `R_norm` and `V_norm` should be included in the SBI observation vector should be a deliberate decision. They are legitimate controller output signals if available in a plant historian, but internal integrator states should remain diagnostic only.

## Validation Requirements

The recommended model should not be accepted just because it is more explicit. It should pass targeted checks.

### nb20: Model Verification

- Nominal steady state near Wu table values:
  - `T_r`, `T_j`, `x_D`, `x_B`, `F_R`, `F_B`, `Q_j`, `Q_reb`, `R`, `V`.
- Loop 1 beta_r masking check:
  - `T_r` remains near setpoint,
  - `Q_j` and `T_j` move.
- Loop 2 sign check:
  - when `x_D` moves below setpoint in S-A, reflux action changes in the corrective direction.
- Loop 3 sign check:
  - when `x_B` or `T_reb` deviates, vapor/reboiler action changes in the corrective direction.
- Saturation/anti-windup check:
  - integrators do not run away under severe faults.

### nb21: Control Structure Verification

- S-A and S-B trajectories must differ before measurement projection.
- S-A should show composition correction via analyzer-based loops.
- S-B should show conventional control masking and residual composition ambiguity.
- W1-W5 should be enough for initial verification:
  - healthy,
  - catalyst decay,
  - reactor fouling,
  - column tray loss,
  - reboiler degradation.

### nb22: Data Generation

- Dataset still has 23 scenarios and 30 replicates.
- S-A and S-B should be generated from separate deterministic base trajectories.
- NaN/inf rate should be zero.
- Snowball, masking, compensation, column degradation, and open-loop contrast should pass.
- Metadata should record the structure mode and control policy.

### nb23 and Later

Summary features should include physics-informed terms for explicit loop signals:

- `R_norm` or `L_norm` mean/slope/final value,
- `V_norm` or `BR` mean/slope/final value,
- correlations between `F_R`, `Q_j`, and `Q_reb`,
- recycle excess,
- reboiler intensity,
- composition-analyzer contribution under S-A.

## Why the Full Dynamic Plan Should Wait

A full dynamic distillation model is attractive, but it should be deferred because it solves a different problem.

The current paper needs to establish:

- feedback control reduces identifiability,
- controller outputs carry hidden degradation information,
- recycle creates coupled nonlinear posteriors,
- conventional measurements are less informative than composition analyzers,
- EKF is overconfident when posterior geometry is non-Gaussian,
- SBI is practical at plant monitoring cadence.

These claims require the right closed-loop causal structure, not maximum column fidelity.

A full model would introduce many additional questions:

- Are tray holdups known?
- Are valve gains known?
- Are measurement lags and analyzer dead times calibrated?
- Are vapor-liquid dynamics stable over the full prior?
- Does the EKF fail because of posterior geometry or because the simulator is stiff and poorly tuned?
- Does SBI performance reflect fault identifiability or numerical artifacts?

Those questions are valid, but they are not the core paper.

## Recommended Development Sequence

1. Preserve current nb22 as `shortcut baseline`.
2. Implement explicit S-A/S-B control modes with the QSS column.
3. Regenerate nb20 and nb21 until the control loops are physically credible.
4. Regenerate nb22 from separate S-A/S-B trajectories.
5. Only then proceed to nb23 summaries and SBI training.
6. Keep the full dynamic tray model as a future validation study or appendix-level sensitivity test if needed.

## Suggested Wording for the Paper

A defensible methods statement would be:

> We use a control-oriented dynamic model of the Wu reactor-column-recycle benchmark. The reactor and jacket are modeled dynamically, while the distillation column is represented by a quasi-steady theoretical-stage closure justified by the 4 s liquid hydraulic time constant relative to the 5.2 h reactor residence time. The published plantwide control structures are retained explicitly: the analyzer-rich structure manipulates reflux and boilup to regulate distillate and bottoms compositions, whereas the conventional structure uses ratio and reboiler-temperature control. This keeps the simulator differentiable and fast enough for SBI while preserving the closed-loop compensation and recycle snowball mechanisms that determine parameter identifiability.

This wording is honest: it does not claim full rigorous tray dynamics, but it does claim the control mechanisms needed for the inference study.

## Bottom Line

The current nb22 shortcut is a good prototype but not the final recommended scientific model. The papers use explicit plantwide control structures and dynamic simulations, so the Wu extension should not leave S-A/S-B as only measurement projections.

However, the objectives in [project_wu2003_sbi.md](../project_wu2003_sbi.md), [article_outline_CChE.md](../article_outline_CChE.md), and [publication_assessment.md](../publication_assessment.md) do not require a full dynamic tray/MESH implementation. They require a model that preserves:

- reactor thermal masking,
- explicit controller compensation,
- recycle snowball coupling,
- S-A/S-B partial observability,
- non-Gaussian posterior geometry,
- EKF limitations,
- and SBI scalability.

The **minimal explicit recycle/control implementation** is the best match to those objectives. It is faithful enough to the papers to support the plantwide control claim, compact enough for SBI and EKF studies, and focused enough to keep the publication contribution centered on closed-loop Bayesian fault diagnosis rather than distillation-model construction.
