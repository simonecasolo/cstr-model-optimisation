# Plan: Luyben Recycle Plant Extension (Replaces Van de Vusse)

## Context

The current CSTR-SBI project (propylene oxide, 2-D inference, PI control) is "trivially small" for publication in Computers & Chemical Engineering. Van de Vusse (4-D, single CSTR, cascade control) was initially planned as the extension but was judged insufficient to showcase SBI's unique power over classical methods.

**Why the Luyben recycle plant instead:** The recycle plant (A+B→C, dynamic flash separator, 5 PI control loops, 8 degradation parameters, partial observability) addresses ALL major weaknesses simultaneously:
- Replaces "trivially small" (2-D → 8-D, single unit → plant-wide)
- Demonstrates SBI's unique inferential advantage (EKF struggles with 21-state augmented filter near nonlinear operating points; MCMC infeasible at 8-D)
- Creates a compelling headline (snowball fault attribution under plant-wide masking)
- Matches Computers & Chemical Engineering's expectation for industrially relevant, multi-unit studies

**Key design decisions (interview-confirmed):**
- Reaction: A+B→C, irreversible, exothermic (Luyben 1994 benchmark)
- Topology: CSTR + dynamic flash separator + recycle + purge stream
- Control: 5 decentralized PI loops
- Parameters: 8-D inference (see §3)
- Measurements: 8 channels, NO online concentration analyzers (partial observability)
- Headline: Fault localization under the snowball effect
- Baselines: SBI + EKF with jax.jacobian (automatic differentiation of Jacobian)
- Paper: PO full case study (Section 6) + Luyben full case study (Section 7), C&ChE target
- Timeline: 6-8 weeks
- Van de Vusse: dropped entirely

---

## 1. Plant Specification

### Reaction
- A + B → C (bimolecular, irreversible, exothermic)
- Rate: r = α · k₀ · exp(−Eₐ/(RT)) · Cₐ · C_B
- Reference: Luyben (1994) "Snowball Effects in Reactor/Separator Processes with Recycle," I&EC Research

### Plant units
1. **Mixer/feed point**: Fresh A + fresh B + recycle stream → combined reactor feed
2. **CSTR (Reactor)**: Exothermic A+B→C, PI temperature control via jacket cooling
3. **Feed preheater** (heat exchanger between feeds and reactor): degradable UA
4. **Flash separator** (dynamic): Splits reactor outlet into vapor (A,B-rich, lighter) and liquid (C-rich, heavier); vapor is condensed and recycled
5. **Purge stream**: Small bleed from recycle vapor to prevent inert/impurity buildup
6. **Recycle compressor/pump**: Returns condensed overhead to reactor feed

### ODE state vector (13 states)

| State | Symbol | Units | Location |
|-------|--------|-------|----------|
| Reactant A conc. | Cₐ | mol/L | CSTR |
| Reactant B conc. | C_B | mol/L | CSTR |
| Reactor temperature | T_r | K | CSTR |
| Coolant temperature | Tc | K | CSTR jacket |
| Sep. liquid holdup | n_L | mol | Flash drum |
| Sep. liquid comp. A | x_A | — | Flash drum |
| Sep. liquid comp. B | x_B | — | Flash drum |
| Sep. temperature | T_s | K | Flash drum |
| CSTR temp. integrator | I_T | K·h | PI loop 1 |
| Sep. temp. integrator | I_Ts | K·h | PI loop 2 |
| Sep. level integrator | I_L | mol·h | PI loop 3 |
| Recycle flow integrator | I_R | (m³/h)·h | PI loop 4 |
| Purge flow integrator | I_P | (m³/h)·h | PI loop 5 |

### Key ODEs

**CSTR component balances:**
```
dCa/dt = (F_in/V_r)*(Ca_in - Ca) - α*k(T_r)*Ca*Cb
dCb/dt = (F_in/V_r)*(Cb_in - Cb) - α*k(T_r)*Ca*Cb
```
where F_in = F_fresh_A*(1+δ) + F_fresh_B*(1-δ) + F_R (recycle flow); κ modifies T_in (feed preheat).

**CSTR energy balance:**
```
dT_r/dt = (F_in/V_r)*(T_in - T_r) + (-ΔH)*α*k(T_r)*Ca*Cb/(ρ*Cp) - β_r*UA_r*(T_r - Tc)/(ρ*Cp*V_r)
dTc/dt  = (Qc/Vc)*(Tc_in - Tc) + β_r*UA_r*(T_r - Tc)/(ρc*Cpc*Vc)
```

**Flash separator (simplified VLE, Raoult-like):**
```
y_i = α_vle_eff_i * x_i / Σ(α_vle_eff_j * x_j)   [phase equilibrium]
α_vle_eff_i = 1 + η_sep*(α_vle_nom_i - 1)           [degraded separation]

d(n_L)/dt = F_in_sep*(1 - V_frac) - F_L_out
d(x_A)/dt = (F_in_sep*z_A - F_V*y_A - F_L*x_A) / n_L
d(x_B)/dt = (F_in_sep*z_B - F_V*y_B - F_L*x_B) / n_L
dT_s/dt   = [energy balance with β_s*UA_s heat exchanger term]
```

**5 PI controllers:**
```
Loop 1 (CSTR temp):   Qc  = Qc0  + Kp1*(T_r - T_r_sp)  + I_T/τi1   [clamped, anti-windup]
Loop 2 (Sep. temp):   Q_s = Qs0  + Kp2*(T_s - T_s_sp)  + I_Ts/τi2  [clamped, anti-windup]
Loop 3 (Sep. level):  F_L = FL0  + Kp3*(n_L - n_L_sp)  + I_L/τi3   [clamped, anti-windup]
Loop 4 (Recycle):     F_R = FR0  + Kp4*(F_R - F_R_sp)  + I_R/τi4   [clamped, anti-windup]
Loop 5 (Purge):       F_P = FP0  + Kp5*(x_purge - x_P_sp) + I_P/τi5 [clamped, anti-windup]
```

### Published parameters
Use Luyben (1994) Table 1 values for all nominal operating conditions. Relative volatilities from Luyben (2002) generic A+B→C benchmark (α_A=3.0, α_B=2.0, α_C=1.0).

---

## 2. The 8 Degradation Parameters

| # | Symbol | Prior | Physical meaning | Primary observable effect |
|---|--------|-------|-----------------|--------------------------|
| 1 | α | U[0.4, 1.2] | Catalyst activity (CSTR) | Outlet conversion → recycle load |
| 2 | β_r | U[0.4, 1.2] | CSTR heat transfer fouling | Qc signal (Loop 1 compensates) |
| 3 | η_sep | U[0.4, 1.2] | Separator split efficiency | Recycle composition, F_R, F_product |
| 4 | β_s | U[0.4, 1.2] | Separator heat exchanger fouling | Q_s signal (Loop 2 compensates) |
| 5 | η_p | U[0.4, 1.2] | Recycle pump efficiency | F_R at given pump speed |
| 6 | ξ | U[0.4, 1.6] | Purge valve flow restriction (>1.0 = erosion) | F_P (Loop 5 compensates) |
| 7 | κ | U[0.4, 1.2] | Feed preheater fouling | T_in → T_r transients |
| 8 | δ | U[-0.3, 0.3] | Feed A:B stoichiometry shift | Ca/Cb imbalance → conversion |

All treated as constant within a 2-hour observation window (slow degradation assumption).

### Identifiability design rationale
- α and η_sep: both affect unconverted A,B reaching the separator; partially non-identifiable under closed-loop without concentration measurements → expected banana-shaped posterior. This is a finding, not a flaw — analogous to the UA-β non-identifiability discovered in the PO system.
- β_r and β_s: independently identified via their respective control loop outputs (Qc vs Q_s).
- η_p: identified via F_R deviation from its setpoint.
- ξ: identified via F_P deviation.
- κ and δ: affect T_in and Ca/Cb ratio respectively; both create CSTR transient signatures.

---

## 3. Observable Channels (8-D, no concentration analyzers)

| # | Channel | Sensor type | Primary sensitivity |
|---|---------|------------|-------------------|
| 1 | T_r | Thermocouple | β_r, α (via heat generation) |
| 2 | Tc | Thermocouple | β_r |
| 3 | Qc | Flow meter | β_r, α (controller compensation effort) |
| 4 | T_s | Thermocouple | β_s, η_sep |
| 5 | Q_s | Flow/power meter | β_s (controller compensation effort) |
| 6 | F_R | Flow meter | η_p, α, η_sep (recycle load) |
| 7 | F_P | Flow meter | ξ (purge restriction) |
| 8 | F_prod | Flow meter | α, η_sep (product outflow) |

All controller output channels (Qc, Q_s, F_R, F_P) are included — these carry the compensation signals that reveal parameter degradation despite the controllers masking it in the process variables.

---

## 4. Headline Experiment: Fault Localization Under Snowball Effect

### Physical mechanism
1. Catalyst decay (α ↓) → less conversion → more A,B in separator feed
2. Separator processes A,B-rich stream → F_R increases (snowball begins)
3. Loop 4 (recycle controller) opens pump wider → η_p stressed
4. Loop 1 (CSTR temp) compensates for changed heat generation → Qc changes
5. Five local controllers each respond to their local symptom, masking the root cause (α)

### Experiment design
- **Scenario**: α = 0.65, η_p = 0.85, all others = 1.0 (catalyst decay + mild pump stress)
- **Test**: Given 2-hour observation window during snowball, SBI posterior over all 8 parameters
- **Expected SBI result**: High posterior mass on low α, moderate η_p degradation; all other params near 1.0
- **Expected EKF result**: Correct mean estimates (both methods show same bias from masking), but EKF gives overconfident Gaussian uncertainty while SBI shows wider, more honest uncertainty — the snowball dynamics create non-Gaussian tails

### The non-Gaussian insight
Near the snowball tipping point, the mapping from (α, η_p) → observables becomes highly nonlinear. Small α changes produce large F_R responses (snowball amplification). The posterior becomes banana-shaped or asymmetric. EKF, assuming Gaussian, gives wrong credible intervals. SBI captures the full shape correctly. This is shown by:
- Posterior pairplot: SBI shows banana curve in (α, η_p) plane; EKF shows upright ellipse
- Coverage check: SBI 90% CI contains true parameter 90% of the time; EKF CI is overconfident

---

## 5. Code Organisation

### New subpackage: `src/cstr_sbi/luyben/`

| File | Contents | Reused pattern |
|------|----------|---------------|
| `__init__.py` | Exports | — |
| `physics.py` | 13-state ODE (reactor + separator + 5 controllers), constants, steady-state solver | Mirrors `cstr_sbi/physics.py` |
| `simulator.py` | EM scan for 13-state system, 8-channel sensor layer, replicate generator | Mirrors `cstr_sbi/simulator.py` |
| `scenarios.py` | 12 fault scenarios × 2 control modes (full plant-wide vs. open-loop) | Mirrors `cstr_sbi/scenarios.py` |
| `summaries.py` | ~65-D summary statistics for 8 channels | Extends `cstr_sbi/summaries.py` pattern |
| `priors.py` | 8-D BoxUniform for 8 parameters | Mirrors `cstr_sbi/priors.py` |
| `inference.py` | SBI wrapper, training (2 posteriors: open-loop + full plant-wide) | Mirrors `cstr_sbi/inference.py` |
| `ekf.py` | Augmented 21-state EKF with jax.jacobian | Mirrors nb16 EKF pattern, uses JAX autodiff |

### Existing files modified (minimally)
- `src/cstr_sbi/__init__.py`: add `from cstr_sbi import luyben`
- `src/cstr_sbi/metrics.py`: generalise `classify_fault` to 8-D (hierarchical: healthy / reactor-fault / separator-fault / recycle-fault / combined)

### Propylene oxide code: entirely untouched

---

## 6. Summary Statistics (~65-D)

| Group | Count | Content |
|-------|-------|---------|
| Per-channel base (8 × 5) | 40 | mean, std, slope, min, max for all 8 channels |
| Final-window means (8) | 8 | Last-25% mean per channel |
| Control aggregates (6) | 6 | int\|T_r err\|, int\|T_s err\|, Qc_sat_frac, Q_s_sat_frac, F_R_std, F_P_std |
| Physics-informed (11) | 11 | See below |
| **Total** | **65** | |

Key physics-informed features:
- `UA_r_eff_proxy = (T_r - Tc) / max(Qc, eps)` → encodes β_r
- `UA_s_eff_proxy = (T_s - T_s_ref) / max(Q_s, eps)` → encodes β_s
- `recycle_load = F_R / F_R_nominal` → encodes α (conversion) + η_sep
- `purge_deviation = F_P / F_P_nominal` → encodes ξ
- `pump_head_proxy = F_R / max(pump_speed_signal, eps)` → encodes η_p
- `conversion_proxy = F_prod / (F_fresh_A + F_fresh_B)` → encodes α
- `recycle_richness = T_s / T_r` → encodes η_sep (lighter component fraction changes T_s)
- `feed_preheat_proxy = T_r_initial_deviation` → encodes κ (feed preheat change creates initial transient)
- `Qc_FR_correlation = corr(Qc, F_R)` → encodes snowball coupling
- `Qs_Ts_correlation = corr(Q_s, T_s)` → encodes separator loop coupling
- `FR_FP_correlation = corr(F_R, F_P)` → encodes recycle-purge coupling

---

## 7. EKF Baseline (21-state augmented)

**Augmented state vector:**
`[Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s, I_T, I_Ts, I_L, I_R, I_P, α, β_r, η_sep, β_s, η_p, ξ, κ, δ]`

- Parameters (last 8 states) are random-walk: `d(param)/dt = 0` with small process noise Q_param
- **Jacobian computed via `jax.jacobian(rhs, argnums=0)` at each EKF step** — no hand-derived 21×21 matrix
- Discretisation: `F = expm(A * dt)` via `scipy.linalg.expm`
- Measurement vector (8-D): `[T_r, Tc, Qc, T_s, Q_s, F_R, F_P, F_prod]`
- Measurement Jacobian H (8×21): analytic for process states (identity-like), computed via jax.jacobian for controller outputs
- Noise tuning: Q and R initialised from Luyben benchmark; tuned on Sc_L1 (healthy) data

**Key advantage of jax.jacobian:** The ODE RHS is already a JAX function; `jax.jacobian(rhs)` computes the exact numerical Jacobian at each evaluation point in ~1ms (JIT-compiled). Eliminates risk of hand-derivation error in a 21×21 matrix.

---

## 8. Fault Scenario Matrix

12 scenarios × 30 replicates × 2-hour windows = 360 evaluation windows per control mode.
2 control modes (full plant-wide + open-loop) = 720 total evaluation windows.

| ID | Name | α | β_r | η_sep | β_s | η_p | ξ | κ | δ | Description |
|----|------|---|-----|-------|-----|-----|---|---|---|-------------|
| L1 | healthy | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | Nominal |
| L2 | cat_decay | 0.65 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | Catalyst decay only |
| L3 | rxr_fouling | 1.0 | 0.65 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | CSTR fouling only |
| L4 | sep_eff | 1.0 | 1.0 | 0.65 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | Separator efficiency loss |
| L5 | sep_fouling | 1.0 | 1.0 | 1.0 | 0.65 | 1.0 | 1.0 | 1.0 | 0.0 | Separator HEX fouling |
| L6 | pump_deg | 1.0 | 1.0 | 1.0 | 1.0 | 0.65 | 1.0 | 1.0 | 0.0 | Recycle pump degradation |
| L7 | purge_block | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.5 | 1.0 | 0.0 | Purge valve erosion |
| L8 | feed_preheat | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.65 | 0.0 | Feed preheater fouling |
| L9 | stoich_shift | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.25 | Feed A:B imbalance |
| L10 | snowball | 0.65 | 1.0 | 1.0 | 1.0 | 0.85 | 1.0 | 1.0 | 0.0 | **Headline scenario** |
| L11 | reactor_sep | 0.80 | 0.80 | 0.75 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | Competing reactor+sep |
| L12 | severe_multi | 0.60 | 0.70 | 0.70 | 0.70 | 0.80 | 1.0 | 1.0 | 0.0 | Heavy multi-unit |

---

## 9. Implementation Schedule (6-8 weeks)

### Week 1-2: Physics and simulation
- **Days 1-3**: `luyben/physics.py` — CSTR ODE, flash separator VLE+dynamics, 5 PI controllers with anti-windup, constants from Luyben (1994). Verify steady-state against published values.
- **Days 4-5**: `luyben/simulator.py` — EM scan for 13-state system, 8-channel sensor layer. Test deterministic limit vs diffrax.
- **Days 6-8**: Controller tuning (step-response tests, loop interaction verification). Snowball demonstration (nb30).
- **Days 9-10**: `luyben/scenarios.py` — 12 fault configs. Generate dataset (nb31). Verify each scenario reaches degraded steady state without NaN.

### Week 3-4: Summary statistics and inference
- **Days 11-13**: `luyben/summaries.py` — 65-D features. Notebook (nb32): PCA/t-SNE over 12 scenarios, mutual information ranking, discriminability check.
- **Days 14-17**: `luyben/priors.py`, `luyben/inference.py`. Train 2 SBI posteriors (open-loop + full plant-wide). 10k simulations per mode. SBC validation for each mode.
- **Days 18-20**: Hyperparameter tuning if SBC shows poor calibration (increase to 20k; adjust NSF to 192 hidden, 7 transforms for 8-D parameter space).

### Week 5-6: Headline experiment and EKF
- **Days 21-25**: Snowball fault localization experiment (nb34). Fisher information 8×8 FIM at nominal OP. Open-loop vs. plant-wide comparison (nb35).
- **Days 26-30**: `luyben/ekf.py` — 21-state augmented EKF with `jax.jacobian`. Test on L1 (healthy), verify convergence. Evaluate on all 12 scenarios (nb36).

### Week 7-8: Robustness, tracking, and paper
- **Days 31-33**: 30-day sequential degradation tracking, SBI vs EKF comparison (nb37).
- **Days 34-36**: Light model mismatch study ±5% on V_r, ρ, k₀, UA_r (nb38).
- **Days 37-42**: Publication figures (nb39), paper Section 7 writing, integration with PO results.

---

## 10. Risk Areas and Mitigation

1. **Snowball instability during SBI training**: Low-α + low-η_p prior draws may push plant past snowball tipping point → simulation diverges. Mitigation: state clipping (Ca, Cb ≥ 0; T ∈ [200, 600] K; n_L > ε; x_A, x_B ∈ [ε, 1-ε]); NaN guards in EM scan; narrow α prior to [0.5, 1.2] if >5% of training sims diverge.

2. **Flash VLE numerical issues**: Raoult-like VLE degenerate when x_A + x_B → 0 or T_s too low. Mitigation: minimum vapor fraction floor; clip mole fractions; test full prior range before SBI training.

3. **Controller tuning complexity**: 5 PI loops interact; recycle (Loop 4) and purge (Loop 5) can interact via composition dynamics. Mitigation: start from Luyben (1994) Table 2 tuning; inner loops (1, 2) must be 5-10× faster than outer loops (3, 4, 5); tune with step tests on each loop independently.

4. **8-D SBI training quality**: 10k simulations may undersample 8-D prior. Mitigation: check SBC before evaluation; if KS p < 0.05 for >2 parameters, increase to 20k; NSF with 192 hidden, 7 transforms.

5. **EKF divergence near snowball**: Discrete-time linearization from jax.jacobian may be poor when Jacobian changes rapidly near snowball bifurcation. Mitigation: smaller EKF integration dt (0.01 h); increase Q for α and η_p near snowball operating point.

6. **Parameter non-identifiability (α, η_sep)**: Banana-shaped posterior expected. Report as a finding (analogous to UA-β in PO system). The (α, η_sep) non-identifiability is in fact the key non-Gaussian result that most clearly demonstrates SBI's advantage over EKF.

---

## 11. Paper Structure (C&ChE, ~30-35 pages)

**Title:** "Plant-wide Bayesian fault diagnosis in recycle processes: amortised simulation-based inference under multi-loop feedback control"

1. Introduction (~2.5 p): feedback masking, multi-unit challenge, SBI motivation, contribution list
2. Related work (~2 p): closed-loop ID theory, fault detection literature, SBI, EKF/UKF filters
3. Problem formulation (~3 p): both systems (PO and Luyben), fault parameterisation, observation model
4. Methodology (~3 p): SNPE-C, NSF, summary statistics, fault classification, Fisher info, EKF with jax.jacobian
5. Experimental setup (~2 p): training configs, scenario matrix, evaluation protocol
6. **Results: Propylene oxide CSTR** (~4 p): existing work — 2-D inference, 4-method bias confirmation, SBC, 30-day tracking
7. **Results: Luyben recycle plant** (~5 p): 8-D inference, snowball localization, non-Gaussian posteriors, EKF comparison, mismatch robustness
8. Discussion (~2 p): information flow under multi-loop feedback, (α, η_sep) non-identifiability, scalability, limitations
9. Conclusion (~0.5 p)

---

## 12. Verification Checklist

- [ ] **Physics**: Steady-state matches Luyben (1994) Table 1; mass/energy balances close; snowball instability demonstrable in open-loop with low α
- [ ] **Simulator**: EM deterministic limit matches diffrax; 8-channel noise at expected 0.5% range; all 12 scenarios reach degraded steady state without NaN
- [ ] **Summaries**: 65-D output, NaN-safe; PCA shows at minimum Sc L2-L9 separable from L1
- [ ] **Inference**: Prior predictive coverage; SBC KS p > 0.05 for ≥6/8 parameters; known-parameter recovery for each fault
- [ ] **EKF**: Convergence on L1 (all 8 params → nominal within 30 min); posterior mean agreement with SBI within 2σ for L2-L9
- [ ] **Snowball experiment**: SBI posterior for L10 assigns >80% probability mass to α < 0.75; EKF Gaussian ellipse spans wider region with incorrect coverage

---

## New notebooks

| # | Purpose |
|---|---------|
| nb30 | Luyben model demo: steady states, snowball demonstration, 3-mode trajectories |
| nb31 | Data generation: 720 windows → data/luyben_observations.npz |
| nb32 | Summary statistics: PCA, MI ranking, discriminability over 12 scenarios |
| nb33 | SBI training: 2 posteriors (open-loop + full plant-wide) + SBC validation |
| nb34 | Snowball fault localization experiment (headline) |
| nb35 | Open-loop vs. plant-wide control comparison |
| nb36 | EKF baseline: 21-state augmented with jax.jacobian |
| nb37 | Sequential 30-day degradation tracking (SBI vs EKF) |
| nb38 | Light model mismatch study (±5% on fixed parameters) |
| nb39 | Publication figures |

---

## Key References

- Luyben, W.L. (1994). "Snowball effects in reactor/separator processes with recycle." *I&EC Research*, 33(2), 299–305.
- Luyben, M.L., Tyreus, B.D., Luyben, W.L. (1997). "Plantwide control design procedure." *AIChE J.*
- Luyben, W.L. (2002). *Plantwide Dynamic Simulations in Chemical Processing and Control.* Marcel Dekker.
- Schmitt, M. et al. (2024). "Detecting Model Misspecification in Amortised Bayesian Inference." arXiv:2406.03154.
- Gustavsson, Ljung & Söderström (1977). Closed-loop identifiability. *Automatica.*
- Gevers, Bombois et al. (2011). Fisher information for closed-loop experiment design. *Automatica.*
