# Wu 2003 CSTR-Column-Recycle — SBI Extension Plan

**Process source:** Wu, Yu, Luyben & Skogestad — *Reactor/separator processes with
recycles-2. Design for composition control* — Computers & Chemical Engineering 2003,
27(3), 401–421. This extends the Luyben (1993) "Dynamics and control of recycle systems"
series (I&EC Research 32:466–503).  
**Source code target:** `src/cstr_sbi/recycle/`  
**Notebooks target:** `notebooks/nb20` through `notebooks/nb28`

---

## 1. Why this model

The propylene oxide case study (System I) demonstrates the fundamental single-loop
masking mechanism in a 2-D system. The Wu 2003 CSTR-column-recycle process extends
this to the plant-wide setting with:

- **Complete published parameters** — every number needed is in Wu et al. (2003) Table 1
- **Correct topology** — CSTR + distillation column + liquid recycle is the most common
  continuous plant-wide topology in the process industry
- **Snowball dynamics** — the recycle amplifies catalyst decay signatures, creating
  the nonlinear posterior geometry the paper requires
- **JAX implementable today** — ~6 ODE states with a QSS column approximation (justified
  by 4700× separation of time scales: column τ_hyd = 4 s vs. reactor τ = 5.2 h)
- **Published control structures** — the same paper introduces PWC-B (conventional
  measurements) and PWC-A (composition analysers) that the SBI paper needs for the
  partial observability story
- **Same unit as PO** — the CSTR with jacket cooling is structurally identical to
  System I; β_r masking must generalise here, which is the paper's continuity argument

---

## 2. Plant topology

```
Fresh A feed (F₀ = 460 lbmol/h, z₀ = 0.90 mol/mol)
        │
        ▼
┌──────────────────────────────┐
│         CSTR                 │  M_r = 2400 lbmol
│  A → B  (first-order, exo.)  │  T_r = 342.2 K  (T_sp)
│                              │  Jacket cooling (Loop 1)
└──────────────┬───────────────┘
               │  z_A_out (reactor effluent)
               │
        ┌──────▼──────────────────┐
        │   Distillation column   │  20 trays, feed tray 12
        │   A (light) / B (heavy) │  α_rel = 2.0
        │                         │  R = L/D = 2.2
        └──────┬──────────┬───────┘
               │          │
        B-rich │    A-rich distillate (D = 500 lbmol/h, x_D = 0.95)
        bottoms│    ←─────────────────────────────────────┐
       (product│                     RECYCLE              │
        B = 460│                                          │
        x_B=0.011)                               ─────────┘
               ▼
          Product B
```

**Snowball mechanism:** When catalyst activity α drops, conversion falls, z_A at reactor
outlet rises. The column receives more A-rich feed; the distillate must carry more A
overhead to maintain bottoms purity x_B. Under fixed-ratio control (S-B), this means
recycle flow F_R increases — which dilutes the reactor, reduces conversion further,
creating a positive feedback. **This is the Luyben snowball in its simplest form.**

---

## 3. Published parameters (Wu 2003 Table 1, converted to SI)

### Reactor

| Parameter | Wu 2003 value | SI value | Symbol |
|---|---|---|---|
| Reactor holdup | 2400 lbmol | 2400 lbmol | M_r |
| Steady-state temperature | 156.4 °F | **342.2 K** | T_r_ss |
| Jacket temperature | 136.1 °F | **331.0 K** | T_j_ss |
| Activation energy | 30,841 Btu/lbmol | **71.74 kJ/mol** | Ea |
| Pre-exponential factor | 2.297 (gives k=0.33/h) | **2.297×10^x /h** | k₀ |
| Rate constant at SS | 0.33 /h | 0.33 /h | k_ss |
| Overall heat transfer coeff. | 150.5 Btu/(h·ft²·°F) | — | U |
| Heat transfer area | 3206.8 ft² | 298 m² | A_HX |
| **UA (product)** | — | **254,000 kJ/(h·K)** | UA_r |
| Heat capacity | 0.7 Btu/(lb·°F) | **2.93 kJ/(kg·K)** | Cp |
| Heat of reaction | 30,000 Btu/lbmol | **69,780 kJ/kmol** | ΔH_r |
| Density | 60.05 lb/ft³ | **961 kg/m³** | ρ |
| **ρCp product** | — | **2,815 kJ/(m³·K)** | ρCp |

**Derived:** k₀ from Arrhenius: k₀ = k_ss × exp(Ea/(R×T_ss)) = 0.33 × exp(71740/(8.314×342.2)) = 0.33 × exp(25.22) = 0.33 × 8.94×10^10 ≈ **2.95×10^10 /h**

### Distillation column

| Parameter | Value |
|---|---|
| Number of trays N_T | 20 |
| Feed tray position | 12 |
| Relative volatility α_rel | 2.0 |
| Reflux ratio R = L/D | 2.2 (derived: L=1100, D=500.4) |
| Liquid hydraulic time constant | 4 s |
| Distillate composition x_D | 0.95 mol/mol A |
| Bottoms composition x_B | 0.0105 mol/mol A |
| Vapor boilup V | 1600.4 lbmol/h |

### Feed and recycle steady state

| Stream | Flow (lbmol/h) | Composition (mol/mol A) |
|---|---|---|
| Fresh feed F₀ | 460.0 | 0.90 |
| Recycle (distillate) D | 500.4 | 0.95 |
| Total reactor feed F_in | 960.4 | — |
| Product (bottoms) B | 460.0 | 0.0105 |
| Reflux L | 1100.0 | — |
| Vapor boilup V | 1600.4 | — |
| **Recycle ratio R_rec = D/B** | **1.09** | — |

---

## 4. ODE formulation

### 4.1 Reactor (full dynamics)

```
dz_A/dt = (F_in/M_r) * (z_A_in - z_A) - α * k(T_r) * z_A

dT_r/dt = (F_in/M_r) * (T_in - T_r)
          + (-ΔH_r) * α * k(T_r) * z_A / Cp_molar
          - β_r * UA_r * (T_r - T_j) / (M_r * Cp_molar)

dT_j/dt = [β_r * UA_r * (T_r - T_j) - Q_c] / (M_j * Cp_c)
```

where:
- `k(T_r) = k₀ * exp(-Ea / (R_gas * T_r))`
- `Cp_molar` = molar heat capacity [kJ/(lbmol·K)] (from ρCp and density)
- `M_j` = jacket liquid holdup [lbmol]
- `Q_c` = jacket cooling duty [kJ/h] = output of Loop 1 PI controller

**Loop 1 (reactor temperature PI):**
```
Q_c = clip(Q_c0 + Kp1*(T_r - T_sp) + I_T/τi1, Q_c_min, Q_c_max)
dI_T/dt = (T_r - T_sp)  [anti-windup: zero when Q_c saturated]
```

### 4.2 QSS distillation column (algebraic)

Justified: τ_hyd = 4 s ≪ τ_reactor ≈ 5.2 h.

**Shortcut model (Kremser equation for binary system):**

Given feed composition z_F = z_A_out, feed flow F_col, relative volatility α_eff:
```
α_eff = 1.0 + η_col * (α_rel - 1.0)   [degraded relative volatility]
```

For specified N_T effective trays and reflux ratio R:
```
N_min = log[(x_D/(1-x_D)) * ((1-x_B)/x_B)] / log(α_eff)   [Fenske]
```

Solve for x_D and x_B given N_T, R, z_F using iterative Kremser-Gilliland-Underwood or
Newton solve of the MESH steady-state equations (2 equations, 2 unknowns x_D, x_B).

**Implementation note:** For SBI training efficiency, pre-compute a lookup table or
train a fast neural network surrogate for the column split (x_D, x_B) = f(z_F, η_col)
at the fixed R = 2.2. This avoids an inner Newton solve at every ODE step.

**Column output to ODE:**
```
F_col = F_in         [CSTR effluent flows directly to column]
z_F   = z_A          [reactor composition = column feed composition]
D     = F_col * (z_F - x_B) / (x_D - x_B)   [material balance]
B     = F_col - D
F_R   = D            [distillate = recycle]
z_A_in = (F₀ * z₀ + F_R * x_D) / (F₀ + F_R)   [mixed reactor feed]
```

**Reboiler heat duty:**
```
Q_reb = ξ_reb * Q_reb_nom * (V / V_nom)   [proportional to vapor flow]
```
where Q_reb_nom is derived from the nominal energy balance. Loop 3 controller adjusts V
(vapor boilup) to maintain bottoms composition x_B (S-A) or reboiler temperature T_reb (S-B).

### 4.3 State vector

**Structure S-A (composition analysers available, 7 states):**
```
y = [z_A, T_r, T_j, I_T, x_D, I_QC, I_R]
     reactor  jacket  loop1 col(alg) loop2 loop3
```

**Structure S-B (conventional measurements only, 6 states):**
```
y = [z_A, T_r, T_j, I_T, I_R_ratio, I_TC_reb]
     reactor  jacket  loop1 ratio_ctrl  TC_loop3
```

*In practice the column variables (x_D) are algebraic in the QSS approximation; the
actual ODE has 4 differential states (z_A, T_r, T_j, I_T) + algebraic column outputs.*

---

## 5. Degradation parameters

| # | Symbol | Physical meaning | Mechanism | Prior | Modelica analogue |
|---|--------|-----------------|-----------|-------|-------------------|
| 1 | α | Catalyst activity | k_eff = α·k₀·exp(-Ea/RT) | **U[0.4, 1.2]** | `k_drop` in CSTR |
| 2 | β_r | Reactor jacket fouling | UA_eff = β_r·UA_r | **U[0.4, 1.2]** | `U_drop` in CSTR |
| 3 | η_col | Column tray efficiency | α_eff = 1 + η_col·(α_rel−1) | **U[0.5, 1.0]** | `t_col_eff_on` |
| 4 | ξ_reb | Reboiler HX fouling | Q_reb_required = Q_reb_nom / ξ_reb | **U[0.5, 1.2]** | — |
| 5 | z_A0 | Feed purity (A fraction) | z₀ degraded by impurities | **U[0.70, 0.95]** | `x_feed` step |

**Nominal:** α=1.0, β_r=1.0, η_col=1.0, ξ_reb=1.0, z_A0=0.90

### Identifiability structure

| Parameter | Primary observable | Masking mechanism | Identifiability |
|---|---|---|---|
| α | z_A, F_R (snowball), Q_c transient | Loop 1 partially masks via T compensation | **High** (multiple channels) |
| β_r | T_j, Q_c (controller output) | Loop 1 zeros ∂T_r_ss/∂β_r → same as PO β | **Low** (~250–500× less than α) |
| η_col | x_D (S-A only), T_reb, Q_reb | Loop 3 masks in x_B; Q_reb carries compensation signal | **Medium** under S-A; **Low** under S-B |
| ξ_reb | Q_reb directly (controller compensation) | Partially masked by Loop 3 | **Medium** |
| z_A0 | z_A steady-state shift, F_R magnitude | Slow drift; distinguishable from α by T_r response shape | **Medium** |

**Key coupling:** Under S-B, α and η_col are **jointly non-identifiable** — both increase
F_R via the snowball. The posterior over (α, η_col) is banana-shaped. This is the
plant-wide analogue of the PO β bias.

---

## 6. Control structures

### Structure S-A — Information-rich (Wu 2003 B-3)

| Loop | CV | MV | Setpoint |
|---|---|---|---|
| 1 | T_r (reactor temp) | Q_c (cooling duty) | T_sp = 342.2 K |
| 2 | x_D (distillate composition) | L/D (reflux ratio) | x_D_sp = 0.95 |
| 3 | x_B (bottoms purity) | V (vapor boilup) | x_B_sp = 0.011 |
| Level 1 | M_r (reactor holdup) | F_out (overflow) | M_sp = 2400 lbmol |
| Level 2 | Condenser drum level | D (distillate takeoff) | — |
| Level 3 | Reboiler level | B (bottoms takeoff) | — |

Requires online composition analysers for x_D and x_B.

### Structure S-B — Conventional measurements only (Wu 2003 B-1b/B-1c)

| Loop | CV | MV | Type |
|---|---|---|---|
| 1 | T_r (reactor temp) | Q_c (cooling duty) | PI (same as S-A) |
| 2 | F_R / F_fresh ratio | L (reflux) | Ratio controller RC |
| 3 | T_reb (reboiler temp) | V (vapor boilup) | PI temperature |
| Level | Same as S-A | Same as S-A | — |

No composition measurements. The Wu 2003 paper shows that S-B fails to maintain
product quality under SD1 (k_drop) and SD6 (feed composition) — this is the masking
narrative: conventional control cannot compensate for faults it cannot measure.

---

## 7. Fault scenarios (16 scenarios)

**Individual faults — reactor:**

| ID | Name | α | β_r | η_col | ξ_reb | z_A0 | Description |
|----|------|---|-----|-------|-------|------|-------------|
| W1 | healthy | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | Nominal SS |
| W2 | cat_mild | 0.85 | 1.00 | 1.00 | 1.00 | 0.90 | Mild catalyst decay — modest snowball onset |
| W3 | cat_severe | 0.65 | 1.00 | 1.00 | 1.00 | 0.90 | Severe decay — pronounced snowball |
| W4 | cat_threshold | 0.55 | 1.00 | 1.00 | 1.00 | 0.90 | Near snowball critical point — strong nonlinearity |
| W5 | jacket_mild | 1.00 | 0.80 | 1.00 | 1.00 | 0.90 | Jacket fouling — analogous to PO Sc2 |
| W6 | jacket_severe | 1.00 | 0.60 | 1.00 | 1.00 | 0.90 | Severe jacket fouling |

**Individual faults — column:**

| ID | Name | α | β_r | η_col | ξ_reb | z_A0 | Description |
|----|------|---|-----|-------|-------|------|-------------|
| W7 | col_eff_mild | 1.00 | 1.00 | 0.80 | 1.00 | 0.90 | Tray fouling — degraded separation |
| W8 | col_eff_severe | 1.00 | 1.00 | 0.65 | 1.00 | 0.90 | Severe tray fouling |
| W9 | reb_fouling | 1.00 | 1.00 | 1.00 | 0.70 | 0.90 | Reboiler HX fouling — higher steam needed |
| W10 | feed_impurity | 1.00 | 1.00 | 1.00 | 1.00 | 0.78 | Feed contains 12% impurity (less A) |

**Combined faults — reactor:**

| ID | Name | α | β_r | η_col | ξ_reb | z_A0 | Description |
|----|------|---|-----|-------|-------|------|-------------|
| W11 | reactor_combined | 0.80 | 0.80 | 1.00 | 1.00 | 0.90 | Both reactor faults — competing signals |

**Combined faults — cross-unit (headline scenarios):**

| ID | Name | α | β_r | η_col | ξ_reb | z_A0 | Description |
|----|------|---|-----|-------|-------|------|-------------|
| W12 | snowball_compound | **0.75** | 1.00 | **0.80** | 1.00 | 0.90 | **Headline**: (α, η_col) banana posterior under S-B |
| W13 | cat_feed | 0.80 | 1.00 | 1.00 | 1.00 | 0.80 | Catalyst decay + poor oil feed |
| W14 | col_reb | 1.00 | 1.00 | 0.75 | 0.75 | 0.90 | Both column faults — separation section |
| W15 | snowball_threshold | **0.58** | 1.00 | 0.90 | 1.00 | 0.90 | Near snowball tipping point — strong EKF failure |
| W16 | full_multi | 0.75 | 0.80 | 0.80 | 0.85 | 0.90 | All four degradation parameters |

**Open-loop variants (no controllers, for comparison):**
W2-OL through W8-OL: same fault values, all PI loops bypassed. Shows full observability
without masking — used in §6 PO paper analogue to demonstrate masking is a closed-loop
phenomenon.

**Total:** 16 closed-loop + 7 open-loop = **23 scenarios × 30 replicates × 2 structures
= 1380 observation windows.**

---

## 8. Observable channels

**Structure S-A (8 channels):**
| # | Channel | Variable | Sensitive to |
|---|---------|----------|-------------|
| 1 | T_r | Reactor temperature [K] | α (indirectly via heat), β_r |
| 2 | T_j | Jacket temperature [K] | β_r (direct: jacket cools differently) |
| 3 | Q_c | Jacket cooling duty [kJ/h] | β_r (controller compensation output) |
| 4 | **x_D** | Distillate composition [mol/mol A] | η_col, α (via z_A_out) — **S-A only** |
| 5 | T_reb | Reboiler temperature [K] | η_col, ξ_reb, α |
| 6 | Q_reb | Reboiler duty [kJ/h] | ξ_reb (compensation), η_col |
| 7 | F_R | Recycle (distillate) flow [lbmol/h] | α (snowball), η_col, z_A0 |
| 8 | F_B | Product (bottoms) flow [lbmol/h] | α (production rate) |

**Structure S-B (7 channels):** Same minus channel 4 (no x_D composition analyser).

**Key asymmetry:** Under S-B, Q_c and Q_reb carry the critical information (controller
compensation signals). Without x_D, (α, η_col) are jointly constrained but not
individually resolved. This is the partial observability story: conventional instruments
are not blind, but they cannot resolve all fault combinations.

---

## 9. Summary statistics (~55-D)

| Group | Count | Description |
|-------|-------|-------------|
| Per-channel base (7 × 5) | 35 | mean, std, slope, min, max for all 7 S-B channels |
| Final-quarter means (7) | 7 | Last-25% mean per channel — captures new quasi-SS |
| Control effort proxies (3) | 3 | Q_c/Q_c_nom, Q_reb/Q_reb_nom, F_R/F_R_nom |
| Physics-informed (10) | 10 | See below |
| **Total (S-B)** | **55** | |
| Extra for S-A (1 channel × 6) | +6 | Same stats for x_D channel |
| **Total (S-A)** | **61** | |

**Physics-informed features (10):**
1. `UA_proxy = β_r * UA_r proxy = Q_c / max(T_r - T_j, 1e-3)` → encodes β_r
2. `recycle_ratio = F_R / F_R_nom` → encodes α (snowball) + η_col
3. `col_recovery_proxy = F_B / (F₀ + F_R)` → encodes α × η_col combined
4. `reb_intensity = Q_reb / F_R` → encodes ξ_reb (reboiler effort per unit throughput)
5. `reactor_conversion_proxy = (F_B * x_B_nom) / F₀` → encodes α (roughly)
6. `recycle_excess = F_R - F_R_nom` → encodes snowball severity
7. `T_r_T_j_ratio = T_r / T_j` → encodes β_r
8. `Q_c_slope` = slope of Q_c over observation window → encodes transient α response
9. `corr(Q_c, F_R)` → encodes snowball coupling (α)
10. `corr(Q_reb, F_R)` → encodes column-recycle coupling (η_col)

---

## 10. SBI training configuration

| Hyperparameter | Value |
|---|---|
| Training simulations | 15,000 (per control structure) |
| NSF hidden units | 128 |
| NSF transforms | 5 |
| Batch size | 256 |
| Max epochs | 250 |
| SBC test cases | 500 |
| Eval. replicates per scenario | 30 |
| Observation window | 2 h at 1-min resolution (120 timesteps × 7/8 channels) |
| Prior | 5-D BoxUniform as specified in §5 |

**Two posteriors trained:** one for S-A (61-D summaries, x_D available) and one for
S-B (55-D summaries, no x_D). The S-A vs. S-B comparison is a direct quantification
of the information value of the composition analyser.

---

## 11. EKF baseline

**Augmented state vector (9-D):**
```
[z_A, T_r, T_j, I_T, α, β_r, η_col, ξ_reb, z_A0]
```

Parameters treated as random-walk states: dθ/dt = 0 + process noise.

**Jacobian:** `jax.jacobian(rhs, argnums=0)` evaluated at each EKF step. The ODE RHS
is a pure JAX function — no hand derivation of the 9×9 Jacobian needed.

**Discretisation:** Euler-Maruyama (same as Luyben EKF plan):
```
F = I + A * dt   (first-order)
P = F @ P @ F.T + Q
```

**Measurement vector (7-D under S-B):** [T_r, T_j, Q_c, T_reb, Q_reb, F_R, F_B]

**Noise covariances:** Q (process) and R (measurement) tuned on W1 (healthy) data.
Q_α and Q_ηcol larger than Q_βr (α and η_col change faster during snowball onset).

**Key EKF failure mode to document:** Under W12 and W15, the (α, η_col) banana creates
a non-Gaussian posterior. EKF, constrained to Gaussians, will be overconfident. The
coverage test (empirical 90% CI coverage < 65% for the pair) is the quantitative proof.

---

## 12. Code organisation

New subpackage: `src/cstr_sbi/recycle/`

| File | Contents |
|------|----------|
| `__init__.py` | Exports |
| `physics.py` | CSTR ODE + QSS column model, all constants, steady-state solver |
| `simulator.py` | Euler-Maruyama scan (JAX), sensor noise, replicate generator |
| `summaries.py` | 55-D (S-B) and 61-D (S-A) summary statistics |
| `priors.py` | 5-D BoxUniform prior |
| `scenarios.py` | 23 scenario configs (W1–W16 + open-loop variants) |
| `inference.py` | SNPE-C wrapper, trained model loader |
| `ekf.py` | 9-state augmented EKF with `jax.jacobian` |

**Reuse from existing code:**
- `src/cstr_sbi/physics.py`: CSTR ODE pattern (same structure as PO — adapt directly)
- `src/cstr_sbi/inference.py`: SNPE-C wrapper (copy, adjust embedding_net input dim)
- `notebooks/nb04.ipynb` → `nb20.ipynb`: SBC/prior predictive pattern
- `notebooks/nb16.ipynb` → `nb26.ipynb`: EKF implementation pattern

---

## 13. Notebooks

| # | Title | Purpose | Key output |
|---|-------|---------|-----------|
| nb20 | Wu 2003 model verification | Steady state, step tests, snowball demo | Verified SS, Figure showing snowball onset |
| nb21 | Control structure implementation | S-A and S-B closed-loop trajectories for W1–W4 | Closed-loop vs. open-loop comparison |
| nb22 | Data generation | 1380 windows → `data/wu2003_observations.npz` | Training + test datasets |
| nb23 | Summary statistics + discriminability | PCA/t-SNE over 23 scenarios, MI ranking | Feature selection (55-D trimmed) |
| nb24 | SBI training (S-B) | Train SNPE-C on conventional measurement set; SBC | Trained posterior, rank histograms |
| nb25 | SBI training (S-A) | Train SNPE-C on rich measurement set; SBC | Trained posterior, S-A vs. S-B comparison |
| nb26 | Headline experiment | W12 banana posterior; W15 snowball EKF failure | Figures 10, 12 for paper |
| nb27 | 30-day sequential tracking | SBI vs. EKF on degradation trajectory | Figure 11 for paper |
| nb28 | Publication figures | All figures 300 dpi | Paper-ready figures |

---

## 14. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| QSS column approximation inaccurate near snowball | Medium | Validate against diffrax full MESH tray-by-tray at 5 test points; if error > 2%, use 3-tray lumped column model instead |
| k₀ value ambiguous (PDF extracted as "2.297", units unclear) | Medium | Back-calculate from k_ss=0.33/h and T_ss=342.2 K: k₀ = 0.33 × exp(71740/(8.314×342.2)) ≈ 2.95×10^10 /h; verify that this gives k=0.33/h at steady state |
| Snowball tipping point too close to prior boundary | Medium | Run 200 pilot simulations over full prior; if >10% diverge, narrow α prior lower bound to 0.50 |
| (α, η_col) posterior too flat under S-B (non-informative) | Low | Add corr(Q_c, F_R) physics feature which breaks the degeneracy; test discriminability in nb23 |
| EKF divergence near W15 (near-snowball) | Medium | Reduce EKF dt; increase Q_α and Q_ηcol; use Joseph form covariance update |

---

## 15. Execution checklist

- [ ] Verify k₀ back-calculation: k₀ = k_ss × exp(Ea/(R×T_ss)) → check k(342.2 K) = 0.33/h
- [ ] Implement `physics.py`: CSTR ODE + Kremser QSS column; run steady-state solver → verify z_A=?, T_r=342.2 K, F_R=500.4 lbmol/h, x_D=0.95, x_B=0.011
- [ ] **nb20**: Step test α from 1.0 → 0.65 under S-B, verify F_R increases (snowball); plot Q_c, F_R, T_reb trajectories
- [ ] **nb20**: Step test β_r from 1.0 → 0.60, verify T_r stays at setpoint (Loop 1 compensates), T_j and Q_c change
- [ ] **nb21**: Implement S-A and S-B control structures; verify W3 (α=0.65) shows x_D drift under S-B, recovered under S-A
- [ ] **nb22**: Generate 23 × 30 × 2 = 1380 windows; check NaN rate < 1%; verify W15 (near-snowball) does not diverge
- [ ] **`summaries.py`**: Implement 55-D features; verify no NaN; run MI ranking in nb23
- [ ] **nb24**: Train S-B posterior; SBC KS p > 0.05 for α, ξ_reb, z_A0; β_r and (α,η_col) expected miscalibration is a finding not a failure
- [ ] **nb25**: Train S-A posterior; compare SBC vs. S-B to quantify x_D information value
- [ ] **nb26**: Run W12 (α=0.75, η_col=0.80) under S-B → confirm banana posterior; run EKF → confirm coverage < 65%
- [ ] **nb26**: Run W15 (α=0.58) → confirm EKF divergence / overconfidence near snowball threshold
- [ ] **nb27**: 30-day tracking with linear α decay + polynomial η_col fouling; SBI vs. EKF comparison
- [ ] **nb28**: All 12 paper figures at 300 dpi, double-column, Okabe-Ito palette
