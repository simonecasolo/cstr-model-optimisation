# cstr-sbi — Closed-Loop CSTR Fault Diagnosis via Simulation-Based Inference

> **Target venue:** *Computers & Chemical Engineering*
> **Status:** M0–M3 complete; M4 (SBI training) in progress.

---

## What this project does

A prior publication showed that **Simulation-Based Inference (SBI)** — specifically amortised neural posterior estimation via SNPE — can perform fast Bayesian condition monitoring of heat exchangers, matching MCMC accuracy at a fraction of the cost.

This project extends that methodology to **Continuous Stirred Tank Reactors (CSTRs) under closed-loop PI temperature control**.

### Scientific claims (revised 2026-05-18)

**Claim 1 — Closed-loop awareness:**
> SBI trained on a **closed-loop** simulator can diagnose faults masked by the PI controller, while open-loop SBI fails on the same observations.

**Claim 2 — Amortised predictive maintenance:**
> A single trained SBI posterior tracks **α(t) and β(t) over a 30-day degradation trajectory** by applying amortised inference to each 60-minute window — near-zero latency vs. hours for MCMC.

**Claim 3 — Probabilistic fault classification:**
> The 2-D posterior over `[α, β]` naturally yields **probabilistic fault isolation** (healthy / fouling / decay / combined) by posterior mass in each quadrant — no supervised fault labels needed.

When jacket fouling occurs, the PI controller compensates by opening the coolant valve, keeping reactor temperature *T* near setpoint. A naive fault detector trained on open-loop data sees stable *T* and anomalous *Qc* and cannot reconcile them — because it was never trained on scenarios where a controller actively moves *Qc* in response to a fault. The closed-loop-aware SBI learns the joint dynamics of fault severity, controller response, and sensor observations.

A secondary claim addresses **sensor drift identifiability**: an extended 6-D parameter space augmented with additive drift coefficients can partially disentangle sensor artefacts from genuine process degradation.

---

## Reaction basis

The simulation is grounded in the **acid-catalysed hydrolysis of propylene oxide (PO) to propylene glycol (PG)** — the standard non-isothermal CSTR benchmark from Fogler (2016, pp. 157–172):

```
C₃H₆O  +  H₂O  →  C₃H₈O₂
```

Parameters follow the Fogler / Furusawa (1969) values, reproduced in *Scientific Reports* (2022):

| Parameter | Value | Units |
|---|---|---|
| Pre-exponential factor k₀ | 16.96 × 10¹² | min⁻¹ |
| Activation energy Eₐ | 75,362 | J/mol |
| Heat of reaction ΔHᵣ | −20,220 | cal/mol |
| Nominal heat transfer UA | 12,500 | cal/(min·K) |
| Feed flow Q | 40 | L/min |
| Reactor volume V | 500 | L |

---

## Model

### State vector and ODE

The **closed-loop** model has four states `y = [C, T, Tc, I]`:

```
dC/dt  = (Q/V)(Cᵢ − C) − α·k₀·exp(−Eₐ/RT)·C
dT/dt  = (Q/V)(Tᵢ − T) − ΔHᵣ·k_eff·C/(ρ·Cₚ) − β·UA·(T−Tc)/(ρ·Cₚ·V)
dTc/dt = (Qc/Vc)·(Tcᵢ−Tc) + β·UA·(T−Tc)/(ρc·Cpc·Vc)
dI/dt  = (T − Tsp) · [1 if Qc unclamped, else 0]   ← anti-windup
```

where `Qc = clip(Qc0 + Kp·(T−Tsp) + I/τᵢ, Qc_min, Qc_max)` is the PI controller output.

### Degradation factors

| Parameter | Meaning | Healthy | Degraded |
|---|---|---|---|
| **α** | Catalyst activity (scales k₀) | 1.0 | 0.7–1.0 |
| **β** | Jacket conductance (scales UA) | 1.0 | 0.4–1.0 |

**Inference parameter vector: 2-D `θ = [α, β]`** (revised 2026-05-18).

UA and k0 are **fixed design constants** (known from clean-service measurements) and are not inferred. α and β are the degradation factors: α ∈ [0.4, 1.0] is catalyst activity, β ∈ [0.4, 1.0] is jacket conductance fraction. Fixing UA and k0 eliminates the structural non-identifiability found in M5 MCMC analysis (UA and β only appeared as their product β·UA in the ODE, making them individually unrecoverable). See `notebooks/05_mcmc_baseline.ipynb §5a` and `cstr_sbi_research_spec.md §3.3`.

Extended vector for the sensor-drift substudy: `θ_ext = [α, β, δT, δCᵢ]` (4-D).

### Stochastic simulator

Integration uses **Euler-Maruyama** (JAX scan, JIT-compiled) with process noise injected during integration and Gaussian sensor noise applied post-simulation. A single 60-minute, 120-sample window is the unit of observation.

---

## Scenarios

Eight scenarios cover the experimental design space:

| ID | Name | α | β | Mode | Role |
|---|---|---|---|---|---|
| Sc0 | open-loop healthy | 1.0 | 1.0 | open-loop | Steady-state validation |
| Sc1 | closed-loop healthy | 1.0 | 1.0 | closed-loop | SBI training/validation baseline |
| Sc2 | jacket fouling | 1.0 | 0.7 | closed-loop | Primary inference target |
| Sc3 | catalyst decay | 0.7 | 1.0 | closed-loop | α identifiability test |
| Sc4 | combined moderate | 0.85 | 0.85 | closed-loop | Joint identifiability |
| Sc5 | severe fouling | 1.0 | 0.4 | closed-loop | Controller-saturation regime |
| Sc6 | open-loop with fault | 1.0 | 0.7 | open-loop | **Headline experiment** — failure baseline |
| Sc7 | mild fouling + drift | 1.0 | 0.85 | closed-loop | Sensor drift substudy (6-D θ) |

**Dataset (`data/observations.npz`):** 400 windows (50 replicates × 8 scenarios), each 60 min long at 0.5 min resolution, 4 channels `[C, T, Tc, Qc]`. These windows are **evaluation observations** — SBI training simulations are drawn on-the-fly from the prior in `04_sbi_training.ipynb`.

---

## Repository layout

```
research_project/
├── pyproject.toml                    # poetry; pinned dependencies
├── src/cstr_sbi/
│   ├── physics.py                    # ODE right-hand sides (open/closed-loop), PI controller,
│   │                                 # degradation factors, diffrax integrators
│   ├── simulator.py                  # Euler-Maruyama integrator, sensor layer, replicate generator
│   ├── scenarios.py                  # SCENARIO_CONFIGS table, perturb_inlet
│   ├── summaries.py                  # 27-D summary statistics, batch variant, feature groups
│   ├── priors.py                     # BoxUniform priors (4-D and 6-D) — M4 stub
│   ├── inference.py                  # SNPE_C training, NUTS baseline — M4/M5 stub
│   ├── metrics.py                    # CRPS, Wasserstein-1, coverage — M7 stub
│   ├── plotting.py                   # Scenario and posterior plots — M7 stub
│   └── style.py                      # Paper style, save_fig — M7 stub
├── notebooks/
│   ├── 00_m0_smoke_test_results.ipynb        # JAX open-loop smoke test + vmap benchmark
│   ├── 01_model_demonstration.ipynb          # Open/closed-loop sanity checks vs. steady state
│   ├── 01a_model_demonstration_pilario.ipynb # Comparison with Pilario & Cao (2018) parameters
│   ├── 02_data_generation.ipynb              # M2: generates data/observations.npz
│   ├── 03_summary_statistics_design.ipynb    # M3: PCA/t-SNE manifold, MI ranking, LDA ablation
│   ├── 04_sbi_training.ipynb                 # M4: SNPE_C training over 4-D prior [stub]
│   ├── 05_mcmc_baseline.ipynb                # M5: NumPyro NUTS baseline [stub]
│   ├── 06_multi_sample_study.ipynb           # M6: W1/CRPS/coverage MCMC vs. SBI [stub]
│   ├── 07_failure_baseline_open_vs_closed.ipynb  # M6: headline Sc6 experiment [stub]
│   ├── 08_identifiability_and_saturation.ipynb   # M6: Sc4 and Sc5 [stub]
│   ├── 09_sensor_drift_substudy.ipynb            # M6: Sc7, 6-D θ [stub]
│   ├── 10_resource_analysis.ipynb                # M7: SBI/MCMC break-even [stub]
│   └── 11_figures_for_publication.ipynb          # M7: paper-ready figures [stub]
├── data/
│   ├── observations.npz              # 400 evaluation windows (M2)
│   ├── scenario_configs.csv          # scenario truth table
│   ├── 03_feature_ablation.csv       # LDA CV accuracy per feature subset (M3)
│   └── 03_feature_shortlist.json     # top-10 MI feature names for M4
├── results/                          # sbi_posteriors.npz, mcmc_posteriors.npz (Git LFS, M6)
├── figures/                          # 02_scenarios_overview.png, 03_pca.png, ... (M2–M7)
├── docs/
│   ├── m0_baseline_benchmarks.md     # jit/vmap timing on 1k/10k/50k samples
│   └── simulator.md                  # Euler-Maruyama design notes
└── scripts/
    ├── m0_smoke_test.py
    ├── build_nb_01.py
    ├── build_nb_01a.py
    └── build_nb_02.py
```

---

## Technology stack

| Layer | Library | Purpose |
|---|---|---|
| ODE integration | `diffrax` (JAX) | Tsit5 adaptive solver for deterministic warm-starts; Euler-Maruyama scan for stochastic SBI simulations |
| MCMC | `numpyro` + NUTS | Generative model baseline; same ODE, differentiable via JAX |
| SBI | `sbi` (SNPE_C + NSF) | Amortised neural posterior estimation; NSF density estimator (128 hidden, 5 transforms) |
| Manifold / ablation | `scikit-learn` | PCA, t-SNE, LDA, mutual information in notebook 03 |
| Numerics | `jax`, `numpy`, `scipy` | JAX for vmap/jit; numpy/scipy for post-processing |
| Plots | `matplotlib` | All figures |
| Package management | `poetry` | `pyproject.toml` with pinned ranges |

---

## Installation

```bash
# Python 3.10 or 3.11 required
python -m venv research_project/.venv
source research_project/.venv/bin/activate

# Install the package (editable) with all dependencies
pip install -e "research_project[dev]"

# Verify the physics layer works
python -c "import cstr_sbi; print('OK')"
python research_project/scripts/m0_smoke_test.py
```

> **Note on Git LFS:** `results/*.npz` are tracked by Git LFS. The `git-lfs` binary must be installed before pushing large artefacts produced at M6. The `.gitattributes` and `data/*.npz` are committed as regular blobs until M6 produces the inference results.

---

## Progress

| Milestone | Deliverable | Status |
|---|---|---|
| **M0** | Package skeleton, JAX open-loop smoke test, vmap benchmark | ✅ Done |
| **M1** | 4-state closed-loop ODE, PI controller, anti-windup, α/β factors | ✅ Done |
| **M2** | Euler-Maruyama simulator, sensor layer, 8-scenario dataset (400 windows) | ✅ Done |
| **M3** | 29-D summary stats (+ `UA_eff_proxy`, `k0_eff_proxy`), PCA/t-SNE, MI ablation | ✅ Done |
| **M4** | SNPE_C training over **2-D prior `[α, β]`**; 2-D NUTS re-run | ⬜ Next |
| **M5** | NUTS baseline (4-D, historical); collinearity → 2-D model decision | ✅ Done |
| **M6** | Snapshot scenarios Sc1–Sc7; headline Sc6 open-vs-closed comparison | ⬜ Pending |
| **M6b** | 30-day sequential degradation tracking (720 windows, amortised SBI) | ⬜ Pending |
| **M6c** | Probabilistic fault classification from 2-D posterior quadrants | ⬜ Pending |
| **M7** | Resource analysis, break-even + sequential speedup; publication figures | ⬜ Pending |

### M3 headline results (2026-05-17, revised 2026-05-18)

The summary statistics (`src/cstr_sbi/summaries.py`) achieve **100 % cross-validated LDA accuracy** on the 8-scenario evaluation set. PCA and t-SNE produce visually clean cluster separation. Following the M5 identifiability finding, two physics-informed features were added:

- **`UA_eff_proxy`** = `(T_mean − Tc_mean) / Qc_mean` — directly proportional to `1/(β·UA)`, the primary jacket-fouling signal
- **`k0_eff_proxy`** = `log(C_mean / (Ci − C_mean))` — encodes `α·k0_eff` from the component balance

The feature vector is now **29-D** (was 27-D). The top-10 MI shortlist (`data/03_feature_shortlist.json`) will be re-run with the 29-D vector in M6.

### M5 headline results + model revision (2026-05-18)

NUTS (4-D `[UA, k0, α, β]`, `dense_mass=True`, 800 warmup + 500 draws × 4 chains) revealed that UA and β only appear as `β·UA` in the ODE — they are structurally non-identifiable. This led to the **2-D model revision**: UA and k0 are now fixed at their nominal design constants; only `[α, β]` are inferred. In the 2-D model NUTS will have no ridge problem and clean R̂/ESS for both parameters. Wall time for 4-D NUTS: ~460 s per observation (historical; 2-D will be faster).

The **new paper narrative** (revised 2026-05-18):
- **Claim 1:** Closed-loop SBI vs. open-loop SBI for fault diagnosis (Sc6 experiment)
- **Claim 2:** Amortised sequential α(t), β(t) tracking over a 30-day degradation trajectory (Sc8, notebook 10)
- **Claim 3:** Probabilistic fault classification from the 2-D posterior (Sc9, notebook 11)

---

## Design documents

| Document | Purpose |
|---|---|
| [`../cstr_sbi_research_spec.md`](../cstr_sbi_research_spec.md) | Scientific specification: chemistry, ODEs, PI controller, degradation models, 8 scenarios, 39-item gap analysis |
| [`../cstr_sbi_execution_plan.md`](../cstr_sbi_execution_plan.md) | Execution roadmap: milestone definitions, acceptance criteria, HX-template mapping table |
| [`../cstr_parameters_recommended.md`](../cstr_parameters_recommended.md) | Parameter rationale: Fogler vs. Pilario & Cao values, PI tuning, operating point |
| [`docs/m0_baseline_benchmarks.md`](docs/m0_baseline_benchmarks.md) | JAX vmap/jit timing on 1 k / 10 k / 50 k parameter samples |
| [`docs/simulator.md`](docs/simulator.md) | Euler-Maruyama design decisions and noise-level rationale |

---

## Reference

This project is the CSTR analogue of the heat-exchanger SBI study at
[`../sbi_mcmc_heat_exchanger/`](../sbi_mcmc_heat_exchanger). The HX repo's
`src/hx_models/` package and numbered notebooks (`01`–`08`) are the structural
template; the execution plan's §6 mapping table documents the correspondence
artefact by artefact.
