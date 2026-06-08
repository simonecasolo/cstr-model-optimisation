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

### Appears new (modest novelty)

- **The specific quantification** of the I_bb/I_aa ratio (250-500x) for a CSTR with
  competing α (catalyst activity) and β (fouling) parameters under PI temperature
  control. No prior paper reports this parameter-pair asymmetry for CSTRs.
- **The empirical proof that summary-statistic choice cannot help**: the CNN embedding
  experiment (nb04b) confirms the Cramér-Rao argument empirically — learned features
  on raw time series produce the same bias as hand-crafted summaries.
- **The connection to amortised SBI**: using neural posterior estimation to reveal and
  characterise the identifiability gap, then showing the method still achieves
  practical fault classification despite it.

## What's weak or missing

### 1. No comparison with standard industrial baselines

This is the biggest gap. The comparison is SBI vs NUTS (both Bayesian), but a
reviewer in a process control journal will immediately ask: "How does this compare
to an Extended Kalman Filter, an Unscented Kalman Filter, or a moving-horizon
estimator?" These are the methods plants actually use. Without this comparison,
the "53,000x speedup" claim is against the wrong baseline — nobody runs NUTS per
window in practice.

**Effort to fix: ~1 week.** Implement EKF/UKF for the same CSTR system and show
the beta bias also appears (strengthening the structural argument).

### 2. Synthetic data only

All observations come from the simulator. There's no sim-to-real gap, no model
mismatch, no measurement delays, no unmeasured disturbances. For a chemical
engineering journal, this makes the "real-time fault diagnosis" framing aspirational
rather than demonstrated.

**Effort to fix: ~2 weeks.** Add model mismatch study (perturbed UA_nominal,
different noise model) or partner with a lab for experimental data.

### 3. The system is trivially small

Two parameters, one reactor, four measured channels. Real processes have dozens of
correlated parameters, multiple units, and partial observability. The paper
acknowledges this (L6) but doesn't test scalability at all.

**Effort to fix: ~2-3 weeks.** Extend to a 4-6 parameter system (e.g. add inlet
flow perturbation, heat loss coefficient) or a two-reactor cascade.

### 4. The fault classification is not really "unsupervised"

Applying a hand-tuned threshold (0.85) to posterior quadrants is not what the ML
community means by unsupervised classification. It's posterior thresholding with
expert-chosen boundaries. The claim should be reframed as "label-free" — the SBI
training uses no fault labels, but the classification rule is designed with domain
knowledge.

**Effort to fix: text only.** Reframe in the paper.

### 5. SBC shows the posterior is miscalibrated

KS p < 0.05 for both parameters. The paper frames this as "mild" and the C2ST
scores (0.52, 0.53) are near the 0.5 baseline, but it is a formal calibration
failure. A Bayesian statistics reviewer would note this.

**Effort to fix: none needed.** Report honestly (already done). The miscalibration
is consistent with the structural beta bias rather than a training deficiency.

### 6. Prior sensitivity not analyzed

The prior was changed from [0.4, 1.0] to [0.4, 1.2] and Sc6 classification
collapsed from F1=0.91 to F1=0.08. This is extreme prior sensitivity for a result
claimed to be robust. No formal prior sensitivity study is included.

**Effort to fix: ~2-3 days.** Run the pipeline with 2-3 different prior widths and
report the sensitivity.

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

## Where this could be published

| Venue tier | Examples | Fit | What's missing |
|---|---|---|---|
| Top ML | NeurIPS, ICML, AISTATS | Poor | No methodological novelty in SBI itself |
| Top general | Nature Comms, PNAS | Poor | Too narrow, no real data |
| **Good process control** | **Journal of Process Control, C&ChE** | **Best fit** | Needs EKF/UKF baseline, model mismatch |
| Good Bayesian/stats | Bayesian Analysis, Stat & Computing | Moderate | Needs formal identifiability theory |
| Applied ML | Eng. Applications of AI | Good | Could go as-is with minor additions |

## What would make it a strong paper

In priority order:

1. **Add an EKF/UKF baseline** (~1 week). Show the beta bias also appears with
   classical state estimation. This transforms the evidence from "SBI has a
   bias" to "the bias affects all methods," empirically confirming the
   classical theory for this specific system.

2. **Connect to the classical identifiability literature explicitly** (~days).
   The Fisher information result (I_bb 250-500x smaller than I_aa) should be
   presented with proper attribution to the closed-loop identification
   literature (Ljung 1977, Gevers et al. 2011) and positioned as a
   system-specific quantification, not a discovery. Derive the physical
   mechanism from the controlled heat balance to connect to structural
   identifiability theory.

3. **Add model mismatch** (~1 week). Perturb simulator parameters (e.g. +/-5% on
   UA_nominal, different noise model) and show the posterior is robust — or
   honestly characterize when it breaks.

4. **Reframe the contribution.** The paper is not "we discover a fundamental
   identifiability limitation" (that's known since 1977). The paper is:

   > We quantify a classical closed-loop identifiability limitation (Ljung 1977;
   > Gevers et al. 2011) for Bayesian fault diagnosis in a PI-controlled CSTR —
   > showing that the Fisher information for the fouling parameter is 250-500×
   > smaller than for catalyst activity — and demonstrate empirically that no
   > choice of summary statistics (hand-crafted or learned) recovers the lost
   > information, consistent with the Cramér-Rao bound. Despite this irreducible
   > limitation, amortised SBI delivers real-time probabilistic fault
   > classification with CL macro-F1 = 0.990 at 53,000× the speed of MCMC.

## Bottom line

The work is technically solid and well-documented. The strongest aspects are the
**empirical methodology** (embedding-net ablation, SBC, multi-method confirmation
of the bias) and the **practical demonstration** (real-time tracking with calibrated
uncertainty). The identifiability finding is a careful quantification of a classical
phenomenon — valuable as applied contribution but not a theoretical advance. Positioned
honestly (applied SBI paper that quantifies a known identifiability limitation for a
specific industrially relevant system), with an EKF baseline and proper literature
context, this is publishable in a good process control journal. Without an EKF baseline,
it reads as a thorough SBI application study — publishable but in a lower-impact venue.
