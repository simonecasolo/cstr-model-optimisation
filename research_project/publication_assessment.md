# Critical assessment for publication

## What's genuinely strong

**The embedding-net control experiment (nb04b) is strong methodology.** Most SBI papers
hand-wave about summary sufficiency. This project actually tested it with sbi's built-in
CNNEmbedding on raw (120x4) time series, bypassing the 29-D summaries entirely. The
learned embedding produced the same beta bias (0.621 vs 0.616) and worse alpha recovery.
This proves the information loss is in the physics, not the features.

**The Fisher information quantification is clean and specific.** The numerically computed
Fisher information matrix (nb15) shows I_bb is 250-500x smaller than I_aa across all
operating points. While the *general principle* that feedback control reduces parameter
identifiability is classical (Gustavsson, Ljung & Söderström 1977; Ljung 1999), and
the Fisher information framework for closed-loop experiment design is mature (Gevers,
Bombois et al. 2006-2011), the **specific quantification** of the I_bb/I_aa ratio for
a CSTR with competing fault parameters (catalyst activity vs fouling) under PI control
does not appear in the prior literature. The heat-exchanger fouling literature
acknowledges the qualitative problem but does not provide this kind of parameter-pair
comparison.

**The amortised sequential tracking is a clean demonstration.** 720 windows at 15 ms
each, with quantified MAE and calibrated uncertainty — this makes the predictive
maintenance case concrete. The 53,000x speedup over NUTS is real and robust across
the MCMC budget sensitivity sweep.

**SBC calibration is included.** Formal simulation-based calibration (500 test cases,
Talts et al. 2018) was performed. The result is honest: mild miscalibration
(KS p = 0.016) but C2ST near 0.5, attributed to the structural beta bias.

## Novelty assessment — what is and isn't new

### Known (not novel)

- **The general principle** that feedback control reduces parameter identifiability
  has been established since Gustavsson, Ljung & Söderström (1977). This is textbook
  material in system identification (Ljung, *System Identification: Theory for the
  User*, 1999).
- **The Fisher information framework** for quantifying identifiability under
  closed-loop is mature (Gevers, Bombois, Hildebrand & Solari 2011; Bombois et al.
  2006).
- **The Cramér-Rao irreducibility** — no estimator (Bayesian or frequentist) can
  beat 1/I(θ) — is a standard statistical result. The claim "irreducible by any
  choice of inference method" is true but not original.
- **Fouling masking by temperature control** is acknowledged in the heat-exchanger
  literature: "conventional fouling monitoring strategies based on heat transfer rate
  are not effective for heat exchangers with closed-loop temperature control"
  (Chemical Engineering Research and Design, 2022).
- **The phenomenon does not depend on reaction kinetics** — it is structural to the
  feedback topology. Parameters whose effects the controller compensates lose
  identifiability regardless of reaction order (1st, 2nd, Michaelis-Menten).
  A tighter controller makes it worse; MPC would be even worse than PI.

### Appears new (modest novelty — propylene oxide system)

- **The specific quantification** of the I_bb/I_aa ratio (250-500x) for a CSTR with
  competing α (catalyst activity) and β (fouling) parameters under PI temperature
  control. No prior paper reports this parameter-pair asymmetry for CSTRs.
- **The empirical proof that summary-statistic choice cannot help**: the CNN embedding
  experiment (nb04b) confirms the Cramér-Rao argument empirically — learned features
  on raw time series produce the same bias as hand-crafted summaries.
- **The connection to amortised SBI**: using neural posterior estimation to reveal and
  characterise the identifiability gap, then showing the method still achieves
  practical fault classification despite it.

### New novelty unlocked by the Luyben extension

- **Plant-wide fault localization under multi-loop feedback masking.** Eight degradation
  parameters across four unit operations (CSTR, flash separator, recycle pump, purge
  valve, feed preheater), each locally compensated by its own PI controller. No prior
  SBI paper addresses multi-unit fault attribution under decentralized plant-wide control.
- **Non-Gaussian posteriors as an inferential finding.** Under partial observability
  (no concentration analyzers), α and η_sep are partially non-identifiable — the
  posterior is banana-shaped in the (α, η_sep) plane. EKF, constrained to Gaussian
  approximations, gives overconfident wrong intervals. SBI captures the full shape.
  This shifts the contribution from "SBI is fast" to "SBI is *correct* where EKF is not."
- **Snowball fault attribution.** A subtle catalyst decay (α ↓) triggers a cascade of
  plant-wide symptoms (F_R ↑, pump stress, separator overload) that each local controller
  compensates independently, masking the root cause. SBI traces the root cause correctly
  from the joint posterior; classical symptom-by-unit diagnosis fails. This is the first
  demonstration of SBI for Luyben snowball fault attribution.
- **MCMC infeasibility at scale.** At 8-D, NUTS requires days of compute per observation
  window. SBI is the only practical full-Bayesian method at this parameter dimensionality.
  This is a qualitative, not just quantitative, advantage over MCMC.

## What's weak or missing

### ~~1. No comparison with standard industrial baselines~~ → RESOLVED (nb16)

**Now done.** Notebook 16 implements both EKF (analytical Jacobian + matrix
exponential discretisation) and UKF (sigma-point propagation, 13 points) for the
augmented 6-D state [C, T, Tc, I, α, β]. Both are evaluated on the identical
observations (50 replicates × 4 scenarios) and 30-day degradation stream (720
windows) used for SBI. Key results:

| Method | β bias (Sc2) | β MAE (tracking) | ms / window | Output |
|--------|-------------|------------------|-------------|--------|
| SBI | −0.149 | 0.033 | 16 | full posterior |
| EKF | −0.093 | 0.065 | 30 | Gaussian (μ, Σ) |
| UKF | −0.093 | 0.090 | 358 | Gaussian (μ, Σ) |
| NUTS | −0.102 | 0.102 | 150,000 | full posterior |

All four methods show β bias → structural confirmation. SBI has the lowest
tracking MAE and fastest inference while providing full posterior uncertainty.

For the Luyben system: EKF with jax.jacobian (automatic differentiation) will be
implemented on the 21-state augmented system [13 plant states + 8 parameters]. NUTS
is not attempted (8-D inference, days per window). The comparison becomes SBI vs EKF,
with MCMC infeasibility as an explicit finding.

### 2. Synthetic data only

All observations come from the simulator. There's no sim-to-real gap, no model
mismatch, no measurement delays, no unmeasured disturbances. For a chemical
engineering journal, this makes the "real-time fault diagnosis" framing aspirational
rather than demonstrated.

**Plan:** Light model mismatch study included in the Luyben extension (nb38): ±5%
perturbation on fixed parameters (V_r, ρ, k₀, UA_r) at test time. Reports posterior
mean shift, coverage degradation, and classification F1 drop. This directly addresses
the weakness without requiring experimental data.

### ~~3. The system is trivially small~~ → RESOLVED by Luyben extension

~~Two parameters, one reactor, four measured channels.~~

**Now addressed.** The Luyben recycle plant (project_luyben_extension.md) introduces:
- 8 degradation parameters across 4 unit operations (4× the propylene oxide system)
- 13 plant states + 5 controller states
- 8 measured channels under partial observability (no concentration analyzers)
- 5 decentralized PI control loops creating plant-wide fault masking
- 12 fault scenarios including the snowball effect

The two-system progression (propylene oxide → Luyben recycle plant) directly
demonstrates scalability from a simple benchmark to an industrially realistic plant.

### 4. The fault classification is not really "unsupervised"

Applying a hand-tuned threshold (0.85) to posterior quadrants is not what the ML
community means by unsupervised classification. It's posterior thresholding with
expert-chosen boundaries. The claim should be reframed as "label-free" — the SBI
training uses no fault labels, but the classification rule is designed with domain
knowledge.

**Effort to fix: text only.** Reframe in the paper. For the Luyben system, the
hierarchical fault taxonomy (healthy / reactor-fault / separator-fault / recycle-fault /
combined) will be described as "label-free domain-informed classification" from the
outset.

### 5. SBC shows the posterior is miscalibrated

KS p < 0.05 for both parameters. The paper frames this as "mild" and the C2ST
scores (0.52, 0.53) are near the 0.5 baseline, but it is a formal calibration
failure. A Bayesian statistics reviewer would note this.

**Effort to fix: none needed for propylene oxide.** Report honestly (already done).
The miscalibration is consistent with the structural beta bias rather than a training
deficiency. For the Luyben system: SBC will be run for each of the 8 parameters
separately; the (α, η_sep) banana-shaped posterior is expected to show the largest
miscalibration, and will be reported alongside the identifiability analysis.

### 6. Prior sensitivity not analyzed

The prior was changed from [0.4, 1.0] to [0.4, 1.2] and Sc6 classification
collapsed from F1=0.91 to F1=0.08. This is extreme prior sensitivity for a result
claimed to be robust. No formal prior sensitivity study is included.

**Effort to fix: ~2-3 days.** Run the pipeline with 2-3 different prior widths and
report the sensitivity. This remains an open item for the propylene oxide section.
For the Luyben system, prior bounds will be set conservatively from the outset
([0.5, 1.2] for most parameters) with a brief sensitivity check before the full
evaluation.

### 7. The beta bias mechanism is characterized but not fully explained

The Fisher information analysis (I_bb << I_aa) quantifies the identifiability gap.
The notebook tested and disproved two specific mechanistic hypotheses:
- 1D Qc convexity (Jensen's inequality): gives negligible bias (+0.004)
- alpha-beta anti-correlation (marginalisation): rho = -0.02, essentially zero

The bias arises from the nonlinear mapping from theta to the full 29-D summary
space, but the exact mechanism in 29 dimensions resists a simple closed-form
argument. A formal analytical derivation (e.g. computing the expected posterior
bias from a linearized model) would strengthen the theoretical contribution.

**Effort to fix: ~1 week.** Derive the bias analytically for a simplified 2-state
model, or compute the profile likelihood for beta to show the asymmetry directly.
For the Luyben system, the 8×8 Fisher information matrix will be computed numerically
(nb34) and the (α, η_sep) off-diagonal terms will explain the partial non-identifiability.

## Where this could be published

| Venue tier | Examples | Fit | What's missing |
|---|---|---|---|
| Top ML | NeurIPS, ICML, AISTATS | Poor | No methodological novelty in SBI itself |
| Top general | Nature Comms, PNAS | Poor | Too narrow, no real data |
| **Good process control** | **C&ChE, Journal of Process Control** | **Best fit** | Model mismatch (partially addressed by nb38) |
| Good Bayesian/stats | Bayesian Analysis, Stat & Computing | Moderate | Needs formal identifiability theory |
| Applied ML | Eng. Applications of AI | Good | Could go as-is after Luyben extension |

With the Luyben extension complete, C&ChE becomes the primary target. The two-system
progression (simple single-unit → complex plant-wide), snowball fault attribution, and
EKF comparison on a 21-state augmented system are exactly what C&ChE reviewers expect.

## What would make it a strong paper

In priority order:

1. ~~**Add an EKF/UKF baseline**~~ → **DONE** (nb16). The β bias appears in all
   four methods (SBI, NUTS, EKF, UKF), transforming the evidence from "SBI has a
   bias" to "the bias affects all methods." This was the #1 priority and is now
   the strongest empirical result in the paper.

2. ~~**The system is trivially small**~~ → **ADDRESSED** (project_luyben_extension.md).
   The Luyben recycle plant (8-D, plant-wide, snowball effect) replaces the planned
   Van de Vusse extension as the complex case study. Timeline: 6-8 weeks.

3. **Connect to the classical identifiability literature explicitly** (~days).
   The Fisher information result (I_bb 250-500x smaller than I_aa) should be
   presented with proper attribution to the closed-loop identification
   literature (Ljung 1977, Gevers et al. 2011) and positioned as a
   system-specific quantification, not a discovery.

4. **Add model mismatch** → **INCLUDED IN LUYBEN PLAN** (nb38, ~2-3 days). Perturb
   simulator parameters (±5% on V_r, ρ, k₀, UA_r) and report posterior robustness.

5. **Reframe the contribution.** The paper is not "we discover a fundamental
   identifiability limitation" (that's known since 1977). The paper is:

   > We quantify classical closed-loop identifiability limitations for Bayesian fault
   > diagnosis in feedback-controlled chemical processes — from a simple PI-controlled
   > CSTR (2-D, I_bb 250-500× smaller than I_aa) to a plant-wide recycle process
   > (8-D, 5 PI loops, Luyben snowball effect) — and demonstrate that amortised SBI
   > is the only practical full-Bayesian method at plant scale, correctly recovering
   > non-Gaussian posteriors where EKF gives overconfident Gaussian approximations.

## Bottom line

The propylene oxide work is technically solid and well-documented. **With the EKF/UKF
baseline complete** (nb16), it is publishable in EAAI or JPC as a standalone paper.
However, it remains vulnerable to the "trivially small system" critique at C&ChE.

**The Luyben extension resolves this decisively.** The two-system paper — propylene
oxide as a validated simple benchmark (Section 6) + Luyben recycle plant as the complex
case study (Section 7) — makes three claims that together are novel and strong:

1. The structural identifiability limitation (I_bb 250-500× I_aa) is empirically
   irreducible (PO system, 4-method confirmation including EKF/UKF).
2. At plant scale (8-D, 5 control loops), this limitation generalises and non-Gaussian
   posteriors emerge (Luyben system, α-η_sep banana-shaped posterior).
3. SBI is the only practical full-Bayesian method at 8-D: MCMC is infeasible, EKF
   gives wrong coverage in non-Gaussian regimes, SBI correctly localizes the snowball
   root cause.

Positioned this way, the paper is a strong submission to **Computers & Chemical
Engineering** with a realistic path to acceptance.
