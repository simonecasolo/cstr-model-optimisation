# Plan: Van de Vusse Reaction System Extension

## Context

The current CSTR-SBI project demonstrates amortised Bayesian fault diagnosis for a propylene oxide hydrolysis reactor (2 parameters, PI control, 4 channels). The publication assessment identifies the system as "trivially small" — a key weakness for Computers & Chemical Engineering. This plan extends the work with a van de Vusse reaction system (4 parameters, cascade control, 6 channels), preserving all existing work as the "simple case" in a two-system progression paper.

**Key design decisions (from interview):**
- Reaction: Van de Vusse (A->B->C, 2A->D)
- Parameters: 4-D inference (alpha1, alpha2, gamma, beta)
- Control: Cascade (inner PI on T, outer PI on Cb)
- Channels: 6 (Ca, Cb, T, Tc, Qc, Tsp)
- Headline experiment: Control-layer ablation (open-loop -> PI-only -> cascade)
- Baselines: SBI + EKF (skip UKF/NUTS for van de Vusse)
- Paper: Both systems, targeting C&ChE
- Timeline: 3-4 weeks
- Light model mismatch study included

---

## 1. Van de Vusse System Specification

### Reaction scheme
- A -> B (desired), r1 = alpha1 * k10 * exp(-E1/RT) * Ca
- B -> C (undesired), r2 = alpha2 * k20 * exp(-E2/RT) * Cb
- 2A -> D (side reaction), r3 = gamma * k30 * exp(-E3/RT) * Ca^2

### State vector (cascade mode, 6 states)
`[Ca, Cb, T, Tc, I_inner, I_outer]`

### ODEs
```
dCa/dt = (F/V)*(Caf - Ca) - alpha1*k1(T)*Ca - 2*gamma*k3(T)*Ca^2
dCb/dt = -(F/V)*Cb + alpha1*k1(T)*Ca - alpha2*k2(T)*Cb
dT/dt  = (F/V)*(Tf - T) + [-DH1*alpha1*k1*Ca - DH2*alpha2*k2*Cb - DH3*gamma*k3*Ca^2]/(rho*Cp) - beta*UA*(T-Tc)/(rho*Cp*V)
dTc/dt = (Fc/Vc)*(Tcf - Tc) + beta*UA*(T - Tc)/(rhoc*Cpc*Vc)
dI_inner/dt = (T - Tsp(t))         [anti-windup]
dI_outer/dt = (Cb_sp - Cb)         [anti-windup]
```

### Cascade controller
- **Inner PI**: Qc = Qc0 + Kp_inner*(T - Tsp(t)) + I_inner/tau_i_inner, clamped [Qc_min, Qc_max]
- **Outer PI**: Tsp(t) = Tsp0 + Kp_outer*(Cb_sp - Cb) + I_outer/tau_i_outer, clamped [Tsp_min, Tsp_max]
- Outer loop must be 5-10x slower than inner loop

### 4 inferred parameters
| Parameter | Prior | Physical meaning |
|-----------|-------|-----------------|
| alpha1 | U[0.4, 1.2] | k1 degradation (A->B, desired reaction) |
| alpha2 | U[0.4, 1.2] | k2 degradation (B->C, undesired) |
| gamma | U[0.4, 1.6] | k3 degradation (2A->D, side reaction); >1.0 = selectivity loss |
| beta | U[0.4, 1.2] | Fouling factor on UA |

Note: gamma's upper bound is 1.6 (not 1.2) because gamma > 1.0 represents increased side reaction rate (catalyst selectivity loss), which is a physically meaningful degradation mode.

### Parameters from literature
Use Engell (2007) / Klatt & Engell (1998) standard benchmark values. Operate at the upper steady state (high Cb yield, moderate T).

---

## 2. Code Organisation

### New subpackage: `src/cstr_sbi/vdv/`

| File | Contents |
|------|----------|
| `__init__.py` | Exports |
| `physics.py` | ODEs (3 modes), cascade controller, constants, steady-state solver |
| `simulator.py` | EM scan for 6/5/4-state systems, sensor layer, replicate generator |
| `scenarios.py` | 8 faults x 3 control modes = 24 scenario configs |
| `summaries.py` | 45-D summary statistics for 6 channels |
| `priors.py` | 4-D BoxUniform |
| `inference.py` | SBI wrapper, training (3 posteriors) |
| `ekf.py` | Augmented 10-state EKF |

### Existing files to modify (minimally)
- `metrics.py`: Generalise `classify_fault` to 4-D (6-class taxonomy: healthy, fouling, k1_decay, k2_decay, selectivity_loss, combined)
- `__init__.py`: Add vdv subpackage re-export

### Existing propylene oxide code: untouched
All existing notebooks, data, and results remain byte-identical.

### Reuse patterns
- `vdv/physics.py` mirrors `physics.py` (ODE structure, controller dataclasses, integration wrappers)
- `vdv/simulator.py` mirrors `simulator.py` (EM scan pattern, sensor layer, replicate generator)
- `vdv/summaries.py` mirrors `summaries.py` (per-channel/final-window/aggregate/physics-informed decomposition)
- `vdv/inference.py` mirrors `inference.py` (simulation_wrapper_sbi, train_sbi_posterior)
- `vdv/ekf.py` mirrors notebook 16 EKF (augmented state, analytical Jacobian, expm discretisation)

---

## 3. Control-Layer Ablation (Headline Experiment)

Three configurations trained and evaluated independently:

| Mode | Controller | Measured Qc | Measured Tsp | States |
|------|-----------|-------------|-------------|--------|
| Open-loop | None (fixed Qc, fixed Tsp) | Constant | Constant | 4 |
| PI-only | Inner PI on T | Varies | Constant | 5 |
| Cascade | Inner PI on T + Outer PI on Cb | Varies | Varies | 6 |

### Expected identifiability gradient
- **Open-loop**: Best for all 4 parameters (no masking)
- **PI-only**: beta harder (T stabilised), kinetic params still from Ca/Cb
- **Cascade**: beta AND alpha1/alpha2 harder (both T and Cb stabilised); gamma still from Ca

### Cross-configuration transfer matrix (9 cells)
Train on each mode, test on all modes. Key predictions:
- Cascade-trained on cascade: good (baseline)
- Open-loop-trained on cascade: complete failure (Qc/Tsp vary, model expects fixed)
- Cascade-trained on open-loop: moderate degradation

### Fisher information
Compute numerical 4x4 FIM at nominal operating point for each mode. Compare diagonal elements across modes to quantify the identifiability gradient.

---

## 4. Scenario Matrix (per control mode)

| ID | Name | alpha1 | alpha2 | gamma | beta | Description |
|----|------|--------|--------|-------|------|-------------|
| V1 | healthy | 1.0 | 1.0 | 1.0 | 1.0 | Nominal |
| V2 | fouling | 1.0 | 1.0 | 1.0 | 0.7 | Heat transfer degradation |
| V3 | k1_decay | 0.7 | 1.0 | 1.0 | 1.0 | Primary reaction slowing |
| V4 | k2_decay | 1.0 | 0.7 | 1.0 | 1.0 | Secondary reaction affected |
| V5 | side_rxn | 1.0 | 1.0 | 1.5 | 1.0 | Catalyst selectivity loss |
| V6 | combined_12 | 0.8 | 0.8 | 1.0 | 0.8 | Multi-fault |
| V7 | combined_3b | 1.0 | 1.0 | 1.3 | 0.7 | Side reaction + fouling |
| V8 | severe | 0.6 | 0.6 | 1.0 | 0.5 | Heavy degradation |

8 scenarios x 3 control modes x 30 replicates = 720 evaluation windows.

---

## 5. Summary Statistics (45-D)

| Group | Count | Features |
|-------|-------|----------|
| Per-channel (6 channels x 5) | 30 | mean, std, slope, min, max for Ca, Cb, T, Tc, Qc, Tsp |
| Final-window means | 6 | Last-25% mean for each channel |
| Control aggregates | 5 | int_abs_T_err, int_abs_Cb_err, Qc_sat_low/high_frac, Tsp_range |
| Physics-informed | 4 | UA_eff_proxy, k1_eff_proxy, selectivity_proxy, Cb_Tc_cross_corr |

Key physics features:
- `UA_eff_proxy = (T_mean - Tc_mean) / max(Qc_mean, eps)` -- encodes beta
- `k1_eff_proxy = log(Ca_mean / max(Caf - Ca_mean, eps))` -- encodes alpha1
- `selectivity_proxy = Cb_mean / max(Ca_mean, eps)` -- sensitive to alpha1/alpha2/gamma balance
- `Cb_Tc_cross_corr = corr(Cb, Tc)` -- encodes cascade coupling

---

## 6. EKF Baseline

Augmented 10-state: `[Ca, Cb, T, Tc, I_inner, I_outer, alpha1, alpha2, gamma, beta]`
- 10x10 analytical Jacobian (~30 nonzero entries)
- 6-D measurement: [Ca, Cb, T, Tc, Qc, Tsp] with Qc/Tsp computed from controller equations
- Discretisation: `F = expm(A * dt)`
- Implement Jacobian symbolically first (verify with SymPy), then hardcode

---

## 7. Model Mismatch Study

Light study on the van de Vusse system only:
- Train posteriors on nominal parameters
- Test with +/-5% perturbation on fixed params (rho, Cp, V, F)
- Evaluate: posterior mean shift, coverage degradation, classification F1 drop
- ~2-3 days of work

---

## 8. Implementation Schedule

### Week 1: Physics and simulation
- Days 1-2: `vdv/physics.py` (constants, 3 ODE modes, controllers, steady-state solver), notebook 20 (model demo)
- Days 3-4: `vdv/simulator.py` (EM scan for 3 modes, sensor layer), controller tuning verification
- Day 5: `vdv/scenarios.py`, `vdv/summaries.py`, notebook 21 (data generation)

### Week 2: Inference pipeline
- Days 6-7: `vdv/priors.py`, `vdv/inference.py`, notebook 22 (summary statistics analysis)
- Days 8-10: Notebook 23 (train 3 posteriors, SBC validation)

### Week 3: Headline experiment + baselines
- Days 11-13: Notebook 24 (control-layer ablation, Fisher information, cross-mode evaluation)
- Days 14-15: `vdv/ekf.py`, notebook 25 (EKF baseline)

### Week 3-4: Robustness and paper
- Days 16-17: Notebook 26 (model mismatch), notebook 27 (publication figures)
- Days 18-21: Paper writing and integration

---

## 9. Risk Areas

1. **Numerical stiffness near fold bifurcation**: Use implicit solver (Kvaerno5) for warm-starts, state clipping in EM, NaN guards. Narrow prior to [0.5, 1.2] if >5% of training sims diverge.

2. **Cascade controller tuning**: Start from Klatt & Engell (1998), verify with step response. Outer loop must be 5-10x slower than inner.

3. **alpha2 identifiability under cascade**: The outer Cb loop may mask alpha2 — this is a feature (the expected finding), not a bug.

4. **4-D posterior estimation**: May need 20-30k training sims (vs 10k for 2-D). Monitor SBC calibration. Consider larger NSF (192 hidden, 7 transforms).

5. **10-state EKF Jacobian complexity**: ~30 nonzero entries with Arrhenius derivatives. Verify symbolically before hardcoding.

---

## 10. Paper Structure (C&ChE, ~25-30 pages)

1. Introduction (closed-loop identifiability + SBI for fault diagnosis)
2. Related work (fault diagnosis, closed-loop ID theory, SBI, EKF/UKF)
3. Problem formulation (both systems, fault parameterisation, observation model)
4. Methodology (SNPE-C, NSF, summary statistics, fault classification, Fisher info, EKF)
5. Experimental setup (training, baselines, evaluation, control-layer ablation protocol)
6. Results: Propylene oxide CSTR (existing work — 2-D, PI, 4-method comparison)
7. Results: Van de Vusse CSTR (new — 4-D, control ablation, EKF comparison, mismatch)
8. Discussion (information flow under feedback, scalability, limitations)
9. Conclusion

---

## 11. Verification

- **Physics**: Steady-state matches Engell (2007), mass/energy balances close, controller step response correct
- **Simulator**: Deterministic limit matches diffrax, noise scaling correct, 3 modes produce different trajectories
- **Summaries**: 45-D output, NaN-safe, PCA shows fault/mode separation
- **Inference**: Prior predictive coverage, SBC ranks, known-parameter recovery
- **EKF**: Convergence on synthetic data, agreement with SBI posterior means
- **Mismatch**: Systematic assessment of +/-5% perturbation on fixed parameters

---

## New notebooks

| # | Purpose |
|---|---------|
| 20 | VdV model demonstration (steady states, 3-mode trajectories, parameter sensitivity) |
| 21 | VdV data generation (720 windows -> data/vdv_observations.npz) |
| 22 | VdV summary statistics analysis (PCA, MI, ablation) |
| 23 | VdV SBI training (3 posteriors + SBC) |
| 24 | Control-layer ablation (headline experiment) |
| 25 | VdV EKF baseline |
| 26 | Model mismatch study |
| 27 | VdV publication figures |
