# CSTR Fault Diagnosis via Simulation-Based Inference: Research Specification

## Document purpose

This document defines the research objectives, required codebase, and a complete gap analysis between the existing code and what needs to be built.

**Last revised: 2026-05-18** — major update following M5 MCMC findings:

1. **Parameter reduction:** UA and k0 are fixed design constants; the inference parameter vector is reduced from 4-D `[UA, k0, α, β]` to **2-D `[α, β]`**. This eliminates the structural non-identifiability discovered in M5 (UA and β only appeared as the product β·UA, making them individually unrecoverable) and aligns with the physical reality that clean-service reactor constants are known.

2. **Headline contribution upgraded:** The paper now makes two simultaneous claims — (i) closed-loop SBI outperforms open-loop SBI for fault diagnosis; (ii) **amortised sequential inference tracks α(t) and β(t) over a 30-day degradation trajectory and outputs probabilistic fault classifications at each observation window**. This frames the paper as *predictive maintenance* (broader audience) rather than just parameter estimation.

---

## 1. Research context and scientific objective

### Background

A prior publication demonstrated that Simulation-Based Inference (SBI) — specifically amortised neural posterior estimation via SNPE — can perform fast Bayesian condition monitoring of heat exchangers. The method trains a neural density estimator offline on simulator-generated (parameter, observation) pairs, then performs near-instantaneous posterior inference at deployment time, achieving accuracy comparable to MCMC at a fraction of the computational cost.

The goal of this project is to extend that methodology to **Continuous Stirred Tank Reactors (CSTRs)** under closed-loop PI temperature control, targeting a publication in *Computers & Chemical Engineering*.

### Core scientific claim (revised 2026-05-18)

**Claim 1 — Closed-loop awareness:**
> SBI trained on a **closed-loop** simulator (PI controller included) can diagnose process faults — catalyst decay (α) and cooling jacket fouling (β) — that are **masked or distorted by the compensating controller**, in cases where an open-loop-trained SBI fails or produces misleading posteriors.

**Claim 2 — Amortised predictive maintenance:**
> A single SBI posterior, trained once offline, can be deployed to **sequentially track the degradation state (α(t), β(t)) over a 30-day operating window** by applying the amortised posterior independently to each 60-minute observation window, producing a time-series of probabilistic fault classifications with near-zero deployment latency.

**Claim 3 — Fault classification:**
> The 2-D posterior over `[α, β]` automatically yields a **probabilistic fault isolation** result — the posterior mass in each quadrant of the (α, β) space identifies the active fault type: healthy (α≈1, β≈1), catalyst decay (α<1, β≈1), jacket fouling (α≈1, β<1), or combined degradation (α<1, β<1) — without any supervised training on fault labels.

These three claims together reframe the paper from "parameter estimation for process monitoring" to **"amortised Bayesian predictive maintenance for closed-loop chemical processes"**.

### Secondary scientific claim (sensor drift substudy)

> The extended SBI parameter space — augmented with additive sensor drift coefficients `[α, β, δT, δCi]` — can partially disentangle sensor artifacts from genuine process degradation, with identifiability limits characterised explicitly (Scenario 7).

---

## 2. Industrial reaction basis

### 2.1 Chosen reaction: hydrolysis of propylene oxide to propylene glycol

The simulation study is grounded in the **acid-catalysed hydrolysis of propylene oxide (PO) to propylene glycol (PG)**:

```
C₃H₆O  +  H₂O  →  C₃H₈O₂
(propylene oxide)     (propylene glycol)
```

This reaction is the standard CSTR benchmark in chemical reaction engineering education and research, most prominently used by Fogler in *Elements of Chemical Reaction Engineering* (5th ed., 2016, pp. 157–172). Propylene glycol is produced at approximately 2.2 million tonnes per year globally.

**Why this reaction fits the model structure:**

- It is **irreversible** under typical industrial conditions and therefore well-described by a simple first-order rate law in propylene oxide concentration.
- It is **exothermic** (ΔHᵣ ≈ −84,666 J/mol), requiring active cooling — the cooling jacket and PI temperature controller are physically motivated.
- It proceeds in the **liquid phase** at temperatures of 330–370 K and near-atmospheric pressure.
- It has been extensively studied with well-characterised kinetic parameters available in the open literature.

### 2.2 Kinetic parameters

The reaction rate follows first-order Arrhenius kinetics:

```
r  = k * C_PO
k  = k0 * exp(-Ea / (R * T))
```

**Parameter values (Fogler 2016 / Furusawa 1969, adopted in physics.py):**

| Parameter | Value | Units |
|---|---|---|
| k0 (pre-exponential factor) | 16.96 × 10¹² | min⁻¹ |
| Ea (activation energy) | 75,362 | J/mol |
| ΔHᵣ (heat of reaction) | −20,220 | cal/mol |
| UA (clean-service heat transfer coefficient) | 12,500 | cal/(min·K) |

**UA and k0 are fixed design constants** — they are known from clean-service measurements and do not appear in the inference parameter vector. Only the degradation scalars α and β are inferred.

### 2.3 Catalyst deactivation (the α factor)

Catalyst decay occurs via progressive neutralisation of the acid catalyst by alkaline trace impurities in the feed. The effective rate constant becomes:

```
k_eff(t) = α(t) · k0 · exp(−Ea / (R·T))
α(t) = 1 − 0.1 · t / Tcrit      (linearised, 10% loss at t = Tcrit = 43200 min)
```

For inference, α is the **instantaneous snapshot** at the time of observation — a scalar in [0.4, 1.0].

### 2.4 Cooling jacket fouling (the β factor)

Jacket fouling follows the Kern–Seaton model (Kern & Seaton, 1959). The effective heat transfer becomes:

```
UA_eff(t) = β(t) · UA_nominal
β(t) = 1 − 0.1 · t / Tcrit      (linearised, 10% loss at t = Tcrit)
```

For inference, β is the **instantaneous snapshot** — a scalar in [0.4, 1.0].

### 2.5 Modified CSTR ODEs

The full degraded-process model is:

```
dC/dt  = (Q/V)·(Ci − C) − α·k0·exp(−Ea/(R·T))·C
dT/dt  = (Q/V)·(Ti − T) − Hr·α·k0·exp(−Ea/(R·T))·C/(ρ·Cp)
         − β·UA·(T − Tc)/(ρ·Cp·V)
dTc/dt = (Qc/Vc)·(Tci − Tc) + β·UA·(T − Tc)/(ρc·Cpc·Vc)
```

When α = β = 1 the equations reduce to the nominal healthy model. **The inference problem is to recover (α, β)** — the degradation state — from closed-loop sensor observations [C(t), T(t), Tc(t), Qc(t)].

---

## 3. Physical model

### 3.1 Physical constants (locked in physics.py)

```python
# Reactor geometry
V     = 500      # L,          reactor volume (Fogler M13)
V_c   = 40       # L,          jacket volume
Q     = 40       # L/min,      feed flow rate (τ = 12.5 min)

# Reaction kinetics (Fogler 2016 / Furusawa 1969)
k0    = 16.96e12 # min⁻¹,      pre-exponential factor   ← FIXED
Ea    = 75362    # J/mol,      activation energy
Hr    = -20220   # cal/mol,    heat of reaction

# Thermophysical
Cp    = 1.0      # cal/(g·K)
Cpc   = 1.0      # cal/(g·K)
rho   = 1000     # g/L
rho_c = 1000     # g/L

# Heat transfer
UA    = 12500    # cal/(min·K), clean-service value       ← FIXED

# Nominal feed
Ci    = 0.97     # mol/L
Ti    = 297.0    # K
Tci   = 297.0    # K
```

### 3.2 PI controller

```
Qc(t) = Qc0 + Kp·(T − Tsp) + (1/τi)·∫(T − Tsp) dt
```

| Parameter | Value |
|---|---|
| Tsp | 312.5 K |
| Kp  | 150 (L/min)/K |
| τi  | 10 min |
| Qc0 | 80 L/min |
| Qc_min | 0 L/min |
| Qc_max | 400 L/min |

Anti-windup: freeze integral accumulation when Qc is clamped.

### 3.3 Inference parameter vector (2-D, revised 2026-05-18)

```
θ = [α, β]
```

| Parameter | Meaning | Prior | Physical interpretation |
|---|---|---|---|
| α | Catalyst activity | Uniform[0.4, 1.0] | Fraction of original acid concentration remaining |
| β | Jacket conductance | Uniform[0.4, 1.0] | Fraction of clean UA remaining |

**Why 2-D, not 4-D:** UA and k0 are known clean-service design constants. They do not degrade. The original 4-D formulation `[UA, k0, α, β]` caused structural non-identifiability because UA and β always appeared as the product β·UA in the ODE, making them individually unrecoverable from data. See `notebooks/05_mcmc_baseline.ipynb §5` for the formal analysis.

**Extended parameter vector (sensor drift substudy, Scenario 7):**

```
θ_ext = [α, β, δT, δCi]
```

where δT [K] ∈ [−3, 3] and δCi [mol/L] ∈ [−0.1, 0.1] are additive sensor offsets.

### 3.4 Observation vector and summary statistics

The observation `x` passed to the SBI neural network is a **29-D summary statistics vector** computed from a 60-minute window of `[C(t), T(t), Tc(t), Qc(t)]`:

- 5 base stats per channel × 4 channels = 20 features (mean, std, slope, min, max)
- Final-window (last 25%) mean × 4 channels = 4 features
- Control aggregates = 3 features (∫|T−Tsp|dt, Qc lower saturation fraction, Qc upper saturation fraction)
- **Physics-informed features = 2 features** (added 2026-05-18):
  - `UA_eff_proxy = (T_mean − Tc_mean) / Qc_mean` ∝ 1/(β·UA) — encodes β directly given fixed UA
  - `k0_eff_proxy = log(C_mean / (Ci − C_mean))` ∝ 1/(α·k0) — encodes α directly given fixed k0

With fixed UA and k0, `UA_eff_proxy ∝ 1/β` and `k0_eff_proxy ∝ 1/α` without any product ambiguity. These two features alone achieve **98.3% scenario classification accuracy** (notebook 03 §7b).

### 3.5 Sensor noise and drift (data generation)

- Gaussian process noise: 0.0005 mol/L/√min on C, 0.1 K/√min on T and Tc
- Sensor noise: 0.5% of channel maximum (Gaussian, post-simulation)
- Sensor drift (Scenario 7 only): +2 K on T, +0.1 mol/L on Ci
- Inlet perturbation every 60 min: Ti, Tci ± 2 K; Ci ∈ [0.9, 1.0] mol/L

---

## 4. Experimental scenarios

### Scenario 0: Healthy baseline (open-loop)
- α=1, β=1, Qc fixed at Qc0=80 L/min
- Purpose: validate simulator steady state [C≈0.018 mol/L, T≈312 K, Tc≈299 K]

### Scenario 1: Healthy baseline (closed-loop)
- α=1, β=1, PI controller active
- Purpose: verify closed-loop steady state; calibrate sigma_obs for MCMC likelihood

### Scenario 2: Jacket fouling (primary scenario)
- α=1, β=0.7; PI controller active
- Expected: T stays near Tsp, Qc rises to compensate
- Purpose: primary SBI + MCMC comparison; confirm β recovery

### Scenario 3: Catalyst decay
- α=0.7, β=1; PI controller active
- Expected: less heat generated, Qc drops, C rises
- Purpose: confirm α recovery; contrast fault signature with Sc2

### Scenario 4: Combined degradation
- α=0.85, β=0.85; PI controller active
- Purpose: test joint (α, β) recovery; posterior correlation analysis

### Scenario 5: Controller saturation (severe fouling)
- α=1, β=0.4; PI controller active, Qc hits Qc_max
- Purpose: characterise identifiability breakdown at saturation; SBI posterior widens

### Scenario 6: Failure baseline (open-loop SBI on closed-loop data)
- Train SBI on open-loop data (Qc fixed); apply to closed-loop observation with β=0.7
- Purpose: demonstrate core failure mode — open-loop SBI gives wrong posteriors when controller is active

### Scenario 7: Sensor drift confounded with fouling
- β=0.85, δT=+2 K; 4-D parameter vector [α, β, δT, δCi]
- Purpose: identifiability of sensor drift vs. process degradation

### Scenario 8 (new — sequential degradation tracking)
- 30-day continuous simulation with α(t) = 1 − 0.1·t/Tcrit and β(t) = 1 − 0.1·t/Tcrit
- Sliced into 720 non-overlapping 60-minute windows
- SBI applied independently to each window → time-series of posteriors (α̂(t), β̂(t))
- Purpose: demonstrate **amortised sequential tracking** — the key new contribution relative to the HX paper and the legacy CSTR work

### Scenario 9 (new — fault classification)
- Uses the posterior from each window of Sc8 to classify the active fault type
- Fault classes: {healthy (α>0.95, β>0.95), fouling-dominant (β<0.95, α>0.95), decay-dominant (α<0.95, β>0.95), combined (both <0.95)}
- Classification rule: posterior mass in each quadrant of the (α, β) unit square
- Purpose: demonstrate **probabilistic fault isolation** without any supervised fault labels

---

## 5. Inference design

### 5.1 SBI method

- **SNPE_C** with **NSF** (Neural Spline Flow) density estimator: `posterior_nn("nsf", hidden_features=128, num_transforms=5)`
- **2-D prior** `BoxUniform([0.4, 0.4], [1.0, 1.0])` over `[α, β]`
- Training budget: sensitivity study over `n_simulations ∈ {1k, 5k, 10k, 20k}`; target: 10k
- The trained posterior is amortised — a **single trained network** is used for all scenarios and all windows of the sequential tracking experiment

### 5.2 MCMC baseline

- **NumPyro NUTS**, `dense_mass=True`, 200 warmup + 300 draws × 2 chains (sequential — see notebook 05 §4)
- **Deterministic diffrax Tsit5** integrator inside the likelihood (not Euler-Maruyama — see notebook 05 §3)
- 2-D prior matching the SBI prior; `physics (2)` features used for Sc1/Sc2 main runs; feature-subset comparison in §6
- Per-observation wall time recorded for break-even analysis

### 5.3 Sequential inference (Scenario 8)

The amortised posterior is applied **independently to each 60-minute window** of the 30-day stream:

```python
for t, window in enumerate(degradation_stream):
    x_summary = compute_summary_statistics(window)
    samples = posterior.sample(n_samples=5000, x=x_summary)
    alpha_hat[t] = samples[:, 0].mean()
    beta_hat[t]  = samples[:, 1].mean()
    fault_class[t] = classify_fault(samples)
```

This exploits the amortisation property of SBI: **inference is instantaneous at deployment** (no MCMC required per window). MCMC on the same 720-window stream would take ~720 × 460 s ≈ 92 hours; SBI takes seconds.

### 5.4 Fault classification

The posterior over `[α, β]` naturally partitions into four fault quadrants:

| Class | Condition | Physical meaning |
|---|---|---|
| Healthy | α ≥ 0.85 AND β ≥ 0.85 | Both factors near nominal |
| Fouling-dominant | β < 0.85 AND α ≥ 0.85 | Jacket fouling with active catalyst |
| Decay-dominant | α < 0.85 AND β ≥ 0.85 | Catalyst decay with clean jacket |
| Combined | α < 0.85 AND β < 0.85 | Both degradation mechanisms active |

> **Threshold rationale:** 0.85 rather than 0.95 — calibrated against the M5/M6 finding
> that the closed-loop SBI and NUTS posteriors for β sit ~0.10–0.15 below the true value
> (UA–β compensation effect). A threshold of 0.95 forces the fault boundary above the
> posterior body for moderate faults (β_true ≥ 0.70), causing systematic misclassification.

Classification uses the **posterior probability** in each quadrant — giving calibrated uncertainty on the fault type, not just a point estimate. The confusion matrix and per-class F1 are reported against the ground-truth labels from the degradation model.

---

## 6. Code architecture

### 6.1 `physics.py` — ODE right-hand sides (complete)

`cstr_open_loop_rhs`, `cstr_closed_loop_rhs`, `compute_qc`, integrators.
UA, k0 are module-level constants; the 4-state ODE takes `params = [UA, k0, α, β]` internally
but UA and k0 are always set to their nominal values when called from the inference layer.

### 6.2 `simulator.py` — stochastic EM integrator (complete)

`simulate_em_window`, `simulate_em_window_open_loop`, `generate_replicates`, `warm_start_ic`.

### 6.3 `summaries.py` — 29-D feature vector (complete)

`compute_summary_statistics`, `compute_summary_statistics_batch`, `FEATURE_NAMES`, `FEATURE_GROUPS`.
Includes `UA_eff_proxy` and `k0_eff_proxy`.

### 6.4 `priors.py` — parameter distributions

**Primary (2-D):** `box_uniform_2d()` → `BoxUniform([0.4, 0.4], [1.0, 1.0])` over `[α, β]`
**Extended (4-D):** `box_uniform_4d()` — retained for backward compatibility and the sensor-drift substudy extension
**NumPyro priors:** `sample_numpyro_prior_2d()` for the MCMC generative model

### 6.5 `inference.py` — SBI + MCMC

`cstr_generative_model` — **2-D version** (α, β as sampled parameters; UA, k0 fixed at nominals)
`run_mcmc_baseline` — updated for 2-D
`simulation_wrapper_sbi` — produces 2-D theta
`train_sbi_posterior` — 2-D NSF

### 6.6 `scenarios.py` — data generators

Scenarios 0–7 (existing) + new:
- `generate_degradation_stream(Tcrit, dt_window, seed)` → 30-day trajectory sliced into 720 windows with α(t), β(t) ground truth (Scenario 8)

### 6.7 `metrics.py` — posterior comparison (complete)

CRPS, Wasserstein-1, coverage, R̂, ESS. Plus new:
- `classify_fault(samples_2d, alpha_threshold=0.95, beta_threshold=0.95)` → fault class + posterior probabilities per class
- `compute_classification_metrics(predicted_classes, true_classes)` → confusion matrix, F1

### 6.8 `plotting.py` — figures

- `plot_degradation_track(t, alpha_hat, beta_hat, alpha_true, beta_true)` — time-series tracking
- `plot_fault_classification_timeline(t, fault_class_probs)` — stacked probability over time
- `plot_posterior_pairplot_2d(samples, true_alpha, true_beta)` — 2-D corner plot
- All existing HX-template ports

---

## 7. Notebook plan (revised 2026-05-18)

| Notebook | Status | Content |
|---|---|---|
| `01_model_demonstration` | ✅ Done | Open/closed-loop ODE sanity checks |
| `02_data_generation` | ✅ Done | 8-scenario dataset, 400 windows |
| `03_summary_statistics_design` | ✅ Done | 29-D features, manifold, MI ranking |
| `04_sbi_training` | 🔄 Revised | SNPE_C training (2-D prior); simulator fixes (sensor layer, fixed IC, k0_eff_proxy clip) |
| `05_mcmc_baseline` | ✅ Done | 2-D NUTS on Sc1/Sc2; identifiability justification (§5); feature-subset comparison |
| `05a_sbi_classification` | ✅ Done | LDA + NUTS fault classification; OL vs CL identifiability; `simulate_open_loop_trajectory_fixed` |
| `06_multi_sample_study` | ✅ Done | SBI 50-replicate study (all 8 scenarios); W1/CRPS metrics; classification threshold lowered to 0.85 |
| `07_failure_baseline_open_vs_closed` | ⬜ Next | Headline Sc6 experiment: CL-trained vs OL-trained SBI |
| `08_identifiability_and_saturation` | ⬜ | Sc4, Sc5; posterior width vs severity |
| `09_sensor_drift_substudy` | ⬜ | Sc7, 4-D θ=[α,β,δT,δCi] |
| `10_sequential_degradation_tracking` | ⬜ **NEW** | Sc8: 30-day amortised tracking; α̂(t), β̂(t) time series |
| `11_fault_classification` | ⬜ **NEW** | Sc9: probabilistic fault isolation from posterior quadrants; F1, confusion matrix |
| `12_resource_analysis` | ⬜ | SBI/MCMC break-even; sequential speedup (Sc8 vs MCMC) |
| `13_figures_for_publication` | ⬜ | All paper-ready figures |

---

## 8. Gap analysis (updated 2026-05-18)

### Remaining gaps (M4 onwards)

| # | Item | Status |
|---|---|---|
| 3 | 2-D SBI training (notebook 04) | 🔄 In progress — simulator fixes applied; re-run needed |
| 4 | Per-scenario sigma_obs calibration | ⬜ Current sigma from Sc1 causes residual β bias (partially irreducible via UA–β compensation) |
| 5 | Sequential degradation stream (Scenario 8) | ⬜ `generate_degradation_stream()` in `scenarios.py` |
| 7 | Notebooks 07–13 | ⬜ Failure baseline (Claim 1), saturation, sensor drift, sequential tracking (Claim 2), classification (Claim 3), resource analysis |
| 8 | `plotting.py` — degradation track + classification timeline | ⬜ New plot types |

### Completed (M0–M5)

| Item | Milestone |
|---|---|
| JAX/diffrax physics, closed-loop ODE, PI controller, α/β factors | M0–M1 |
| Euler-Maruyama simulator, sensor layer, 8 scenarios, `observations.npz` | M2 |
| 29-D summary statistics, manifold analysis, MI ranking | M3 |
| `priors.py` (2-D + 4-D compat), `metrics.py` (`classify_fault`), `inference.py` (2-D NUTS) | M4/M5 |
| MCMC baseline (Sc1, Sc2); 2-D NUTS; feature-subset comparison | M5 |
| LDA + NUTS fault classification; OL vs CL identifiability; `simulate_open_loop_trajectory_fixed` | M5a |
| `simulation_wrapper_sbi` fixed (sensor layer + fixed IC); `k0_eff_proxy` clipped | M4 fix |

---

## 9. Key references

- Fogler, H.S. (2016). *Elements of Chemical Reaction Engineering*, 5th ed. Prentice Hall.
- Kern, D.Q. & Seaton, R.E. (1959). A theoretical analysis of thermal surface fouling. *British Chemical Engineering*, 4(5), 258–262.
- Pilario, K.E.S. and Cao, Y. (2018). Canonical variate dissimilarity analysis for process incipient fault detection. *IEEE Trans. Industrial Informatics*, 14(12), 5308–5315.
- Cranmer, K., Brehmer, J. & Louppe, G. (2020). The frontier of simulation-based inference. *PNAS*, 117(48), 30055–30062.
- Tejero-Cantero, A. et al. (2020). sbi: A toolkit for simulation-based inference. *JOSS*, 5(52), 2505.

---

*End of specification. The execution plan (`cstr_sbi_execution_plan.md`) is the companion implementation roadmap.*
