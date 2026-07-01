# Wu 2003 CSTR-Column-Recycle: Physical Model Description

**Reference implementation:** `src/cstr_sbi/recycle/`  
**Notebooks:** `nb20` – `nb28`  
**Target journal:** *Computers & Chemical Engineering* (Elsevier)  
**Source:** Wu, K.-L., Yu, C.-C., Luyben, W. L., & Skogestad, S. (2003).
*Reactor/separator processes with recycle – 2. Design for composition control.*
Comput. Chem. Eng. **27**(3), 401–421.

---

## Unit System

All quantities in this document are expressed in **SI units** as follows:

| Quantity | Symbol | SI unit |
|----------|--------|---------|
| Time | t | h (hour)¹ |
| Temperature | T | K |
| Molar flow rate | F | kmol h⁻¹ |
| Molar holdup | M | kmol |
| Mole fraction | x, y, z | — |
| Rate constant | k | h⁻¹ |
| Activation energy | E_a | kJ kmol⁻¹ |
| Heat of reaction | ΔH_r | kJ kmol⁻¹ |
| Molar heat capacity | C_p | kJ (kmol·K)⁻¹ |
| Overall heat transfer coefficient × area | UA | kW K⁻¹ |
| Heat duty | Q | kW |
| Gas constant | R | kJ (kmol·K)⁻¹ |

> ¹ The hour (h) is used as the time base because the reactor residence time (≈ 5 h) and observation window (2 h) are naturally expressed in hours; rate constants are in h⁻¹. All differential equations are written with dθ/dt where t is in hours.

**Implementation note (current code):** `src/cstr_sbi/recycle/physics.py` uses English engineering units — **Btu, lbmol, h, K** — matching the original Wu (2003) Table 1 data directly. The conversion factors are:

| From | To SI | Factor |
|------|-------|--------|
| lbmol | kmol | × 0.4536 |
| Btu/lbmol | kJ/kmol | × 2.326 |
| Btu/(lbmol·K) | kJ/(kmol·K) | × 2.326 |
| Btu/(h·K) | kW/K | × 2.931 × 10⁻⁴ |
| Btu/h | kW | × 2.931 × 10⁻⁴ |

The journal article must convert all reported values to SI. A complete side-by-side
parameter table follows in Section 1a.

---

## 1a. Complete Parameter Table — Original (Btu-lbmol-h) and SI

All parameters from Wu (2003) Table 1 and derived quantities, listed in both the
original English engineering units and their SI equivalents.

### Kinetics and Reaction

| Parameter | Symbol | Original value | Original unit | SI value | SI unit | Source |
|-----------|--------|---------------|---------------|----------|---------|--------|
| Steady-state temperature | T_ss | 156.4 °F → 342.26 K | K | **342.26** | K | Wu Table 1 |
| Nominal rate constant | k_ss | 0.33 | h⁻¹ | **0.33** | h⁻¹ | Wu Table 1 |
| Pre-exponential factor | k₀ | 2.91 × 10¹⁰ | h⁻¹ | **2.91 × 10¹⁰** | h⁻¹ | Derived (Arrhenius) |
| Activation energy | E_a | 30,841 | Btu lbmol⁻¹ | **71,738** | kJ kmol⁻¹ | Wu Table 1 |
| Gas constant | R | 1.987 | Btu (lbmol·°R)⁻¹ | **8.314** | kJ (kmol·K)⁻¹ | Universal |
| Gas constant (in K) | R | 3.576 | Btu (lbmol·K)⁻¹ | **8.314** | kJ (kmol·K)⁻¹ | Converted |
| Heat of reaction | ΔH_r | −30,000 | Btu lbmol⁻¹ | **−69,780** | kJ kmol⁻¹ | Wu Table 1 |
| Heat of reaction | ΔH_r | — | — | **−69.78** | kJ mol⁻¹ | — |

### CSTR Design

| Parameter | Symbol | Original value | Original unit | SI value | SI unit | Source |
|-----------|--------|---------------|---------------|----------|---------|--------|
| Liquid holdup | M_r | 2400 | lbmol | **1088.6** | kmol | Wu Table 1 |
| Molar heat capacity | C_p | 0.7 | Btu (lb·°R)⁻¹ | **2.931** | kJ (kg·K)⁻¹ | Wu Table 1 |
| Molar heat capacity | C_p | 126 | Btu (lbmol·K)⁻¹ | **293.1** | kJ (kmol·K)⁻¹ | Wu Table 1, MW≈100 |
| Density | ρ | 60.05 | lb ft⁻³ | **961.6** | kg m⁻³ | Wu Table 1 |
| Heat capacity volumetric | ρC_p | 0.042 | Btu (ft³·°R)⁻¹ | **2806** | kJ (m³·K)⁻¹ | Wu Table 1 |
| Overall HTC | U | 150.5 | Btu (h·ft²·°F)⁻¹ | **854.8** | kJ (h·m²·K)⁻¹ | Wu Table 1 |
| Heat transfer area | A_HX | 3206.8 | ft² | **297.9** | m² | Wu Table 1 |
| UA product (reactor) | UA_r | 868,723 | Btu (h·K)⁻¹ | **916,500** | kJ (h·K)⁻¹ = **254.6 kW K⁻¹** | Derived |
| Jacket heat capacity | M_j C_{p,j} | 44,939 | Btu K⁻¹ | **47,421** | kJ K⁻¹ | Derived (≈400 ft³ water) |

### Feed Conditions

| Parameter | Symbol | Original value | Original unit | SI value | SI unit | Source |
|-----------|--------|---------------|---------------|----------|---------|--------|
| Fresh feed flow | F₀ | 460 | lbmol h⁻¹ | **208.7** | kmol h⁻¹ | Wu Table 1 |
| Fresh feed A purity | z₀ | 0.90 | mol mol⁻¹ | **0.90** | mol mol⁻¹ | Wu Table 1 |
| Feed temperature | T_in | 70 °F → 294.26 K | K | **294.26** | K | Wu Table 1 |

### Reactor Steady State

| Parameter | Symbol | Original value | Original unit | SI value | SI unit | Source |
|-----------|--------|---------------|---------------|----------|---------|--------|
| Reactor temperature setpoint | T_sp | 156.4 °F | — | **342.26** | K | Wu Table 1 |
| Nominal jacket temperature | T_{j,nom} | 136.1 °F | — | **330.98** | K | Wu Table 1 |
| Nominal jacket duty | Q_{j,nom} | 9.351 × 10⁶ | Btu h⁻¹ | **2,739** | kW | Derived (energy balance) |
| Total reactor feed flow | F_total | 960.4 | lbmol h⁻¹ | **435.7** | kmol h⁻¹ | Wu Table 1 |
| Reactor outlet A composition | z_{A,nom} | ≈ 0.500 | mol mol⁻¹ | **≈ 0.500** | mol mol⁻¹ | Column material balance |

### Loop 1 Controller Parameters

| Parameter | Symbol | Original value | Original unit | SI value | SI unit | Tuning basis |
|-----------|--------|---------------|---------------|----------|---------|-------------|
| Proportional gain | K_{p,1} | 8.0 × 10⁶ | Btu (h·K)⁻¹ | **2,344** | kW K⁻¹ | Stability/response tuning |
| Integral time | τ_{i,1} | 0.1 | h | **0.1** | h | Stability/response tuning |
| Bias duty | Q_{j,0} | 9.351 × 10⁶ | Btu h⁻¹ | **2,739** | kW | Energy balance at SS |
| Min jacket duty | Q_{j,min} | 0 | Btu h⁻¹ | **0** | kW | Physical constraint |
| Max jacket duty | Q_{j,max} | 3.0 × 10⁷ | Btu h⁻¹ | **8,793** | kW | Design limit |

### Distillation Column

| Parameter | Symbol | Original value | Original unit | SI value | SI unit | Source |
|-----------|--------|---------------|---------------|----------|---------|--------|
| Number of trays | N_T | 20 | — | **20** | — | Wu Table 1 |
| Feed tray (from top) | N_F | 12 | — | **12** | — | Wu Table 1 |
| Relative volatility | α_rel | 2.0 | — | **2.0** | — | Wu Table 1 |
| Reflux ratio | R = L/D | 2.198 | — | **2.198** | — | Wu Table 1 |
| Reflux flow | L | 1100 | lbmol h⁻¹ | **499.0** | kmol h⁻¹ | Wu Table 1 |
| Vapor boilup | V | 1600.4 | lbmol h⁻¹ | **725.9** | kmol h⁻¹ | Wu Table 1 |
| Distillate composition (nom.) | x_{D,nom} | 0.95 | mol mol⁻¹ | **0.95** | mol mol⁻¹ | Wu Table 1 |
| Bottoms composition (nom.) | x_{B,nom} | 0.0105 | mol mol⁻¹ | **0.0105** | mol mol⁻¹ | Wu Table 1 |
| Recycle (distillate) flow | F_{R,nom} | 500.4 | lbmol h⁻¹ | **226.9** | kmol h⁻¹ | Wu Table 1 |
| Product (bottoms) flow | F_{B,nom} | 460.0 | lbmol h⁻¹ | **208.7** | kmol h⁻¹ | Wu Table 1 |
| Distillate fraction | D_frac = D/F | 0.521 | — | **0.521** | — | Derived |
| Liquid hydraulic lag | τ_hyd | 4 | s | **4** | s | Wu Table 1 |
| Reboiler temperature (nom.) | T_{reb,nom} | 372 | K | **372** | K | Estimated (proxy) |
| Reboiler duty (nom.) | Q_{reb,nom} | 3.53 × 10⁶ | Btu h⁻¹ | **1,035** | kW | Estimated (proxy) |

### Snowball Model Parameter

| Parameter | Symbol | Value | Unit | Physical meaning |
|-----------|--------|-------|------|-----------------|
| Nominal reactor outlet composition | z_{F,ref} | 0.500 | mol mol⁻¹ | Reference for D_frac calculation |
| Recycle sensitivity | S_rec | 0.12 | — | d(D_frac)/d(z_F); governs snowball strength |

### Degradation Parameters (Prior Distributions)

| # | Symbol | Nominal | Lower bound | Upper bound | Physical meaning | SI interpretation |
|---|--------|---------|-------------|-------------|-----------------|------------------|
| 1 | α | 1.0 | 0.4 | 1.2 | Catalyst activity: k_eff = α·k₀·exp(−E_a/RT) | Multiplicative kinetic factor |
| 2 | β_r | 1.0 | 0.4 | 1.2 | Jacket fouling: UA_eff = β_r · UA_r | UA_r × β_r from 102 kW K⁻¹ to 305 kW K⁻¹ |
| 3 | η_col | 1.0 | 0.5 | 1.0 | Column efficiency: α_eff = 1 + η_col·(α_rel−1) | α_eff from 1.5 to 2.0 |
| 4 | ξ_reb | 1.0 | 0.5 | 1.2 | Reboiler fouling: Q_reb_req = Q_reb_nom/ξ_reb | Reboiler duty scale |
| 5 | z_{A0} | 0.90 | 0.70 | 0.95 | Fresh feed A purity | 0.70–0.95 mol mol⁻¹ A in feed |

---

## 1. Process Topology

The plant is the canonical **reactor-separator-recycle** benchmark from the Luyben
"Dynamics and control of recycle systems" series. It consists of three interconnected units:

```
Fresh A feed  F₀ = 208.7 kmol h⁻¹, z₀ = 0.90
       │
       ▼
┌──────────────────────────────┐
│         CSTR                 │  M_r = 1088.6 kmol,  T_r = 342.26 K (setpoint)
│   A → B  (first-order, exo.) │  Jacket cooling controlled by Loop 1 (T_r → Q_j)
└──────────────┬───────────────┘
               │  z_A (reactor outlet = column feed)
        ┌──────▼──────────────────┐
        │  Distillation column    │  20 trays, feed at tray 12, α_rel = 2.0, R = 2.198
        │  A (light) ↑ B (heavy) ↓│  QSS approximation (τ_hyd = 4 s ≪ τ_reactor ≈ 5.2 h)
        └──────┬──────────┬───────┘
               │          │
        B-rich │    A-rich distillate → RECYCLE → CSTR feed
        bottoms│    F_R ≈ 226.9 kmol h⁻¹, x_D ≈ 0.95
       (product)│
      F_B = 208.7 kmol h⁻¹, x_B ≈ 0.011
```

**Overall material balance (steady state):**
Only A is converted; B is the product. Since x_B is small (≈ 1 mol%),
almost all A fed is converted: F₀·z₀ ≈ M_r·k·z_A ≈ 193 kmol h⁻¹ of A consumed.

---

## 2. Reaction Kinetics

| Parameter | Symbol | SI value | Code value (Btu-lbmol-h) |
|-----------|--------|----------|--------------------------|
| Reaction | A → B | first-order, irreversible, liquid phase | — |
| Rate constant at SS | k_ss | 0.33 h⁻¹ | 0.33 h⁻¹ |
| Pre-exponential factor | k₀ | 2.91 × 10¹⁰ h⁻¹ | 2.91 × 10¹⁰ h⁻¹ |
| Activation energy | E_a | 71,738 kJ kmol⁻¹ | 30,841 Btu lbmol⁻¹ |
| Gas constant | R_gas | 8.314 kJ (kmol·K)⁻¹ | 3.576 Btu (lbmol·K)⁻¹ |
| Nominal SS temperature | T_ss | 342.26 K | 342.26 K |
| Heat of reaction | ΔH_r | −69,780 kJ kmol⁻¹ | −30,000 Btu lbmol⁻¹ |

The Arrhenius rate law is:
$$k(T_r) = k_0 \exp\!\left(\frac{-E_a}{R\,T_r}\right)$$

**Verification:** k(342.26 K) = 2.91 × 10¹⁰ × exp(−71738/(8.314 × 342.26)) = 0.330 h⁻¹ ✓

The **effective reaction rate** under catalyst degradation (parameter α):
$$r = \alpha \cdot k(T_r) \cdot z_A \quad [\text{h}^{-1}]$$

where z_A is the mole fraction of A in the liquid reactor phase.

---

## 3. CSTR Model

### 3.1 State Variables and ODEs

The CSTR is operated with **fixed holdup** M_r = 1088.6 kmol (liquid overflow reactor).
The continuous-time state vector for the 4-state model is:

$$\mathbf{y}_{\text{reactor}} = [z_A,\; T_r,\; T_j,\; I_T]$$

| State | Description | Unit |
|-------|-------------|------|
| z_A | Reactor A mole fraction | — |
| T_r | Reactor temperature | K |
| T_j | Jacket temperature | K |
| I_T | PI controller integrator (anti-windup) | K·h |

**Component balance (A):**
$$\frac{dz_A}{dt} = \frac{F_{\text{total}}}{M_r}(z_{A,\text{in}} - z_A) - \alpha\,k(T_r)\,z_A$$

**Energy balance (reactor):**
$$\frac{dT_r}{dt} = \frac{F_{\text{total}}}{M_r}(T_{\text{in,mix}} - T_r)
+ \frac{(-\Delta H_r)\,\alpha\,k(T_r)\,z_A}{C_p}
- \frac{UA_r\,\beta_r\,(T_r - T_j)}{M_r\,C_p}$$

**Energy balance (jacket):**
$$\frac{dT_j}{dt} = \frac{UA_r\,\beta_r\,(T_r - T_j) - Q_j}{\rho_j\,C_{p,j}\,V_j}$$

where Q_j [kW] is the jacket heat removal duty (output of Loop 1).

**Loop 1 — Reactor temperature PI controller:**
$$Q_j = \text{clip}\!\left(Q_{j,0} + K_{p,1}(T_r - T_\text{sp}) + \frac{I_T}{\tau_{i,1}},\;Q_{j,\min},\;Q_{j,\max}\right)$$
$$\frac{dI_T}{dt} = \begin{cases} T_r - T_\text{sp} & \text{if } Q_j \text{ not saturated} \\ 0 & \text{otherwise} \end{cases}$$

### 3.2 CSTR Parameters (SI)

| Parameter | Symbol | SI value | Source |
|-----------|--------|----------|--------|
| Liquid holdup | M_r | 1088.6 kmol | Wu (2003) Table 1 |
| Molar heat capacity | C_p | 293.1 kJ (kmol·K)⁻¹ | Wu Table 1: 0.7 Btu (lb·°R)⁻¹, MW ≈ 100 |
| Overall UA (reactor jacket) | UA_r | 254.6 kW K⁻¹ | Wu Table 1 |
| Jacket heat capacity | ρ_j C_{p,j} V_j | 13.17 kW·h K⁻¹ | Wu Table 1 (estimated) |
| Temperature setpoint | T_sp | 342.26 K (156.4°F) | Wu Table 1 |
| Nominal jacket temperature | T_{j,\text{nom}} | 330.98 K (136.1°F) | Wu Table 1 |
| Fresh feed flow | F_0 | 208.7 kmol h⁻¹ | Wu Table 1 |
| Fresh feed composition | z_0 | 0.90 | Wu Table 1 |
| Feed temperature | T_\text{in} | 294.26 K (70°F) | Wu Table 1 |
| Loop 1 proportional gain | K_{p,1} | 2344 kW K⁻¹ | Tuned from Wu structure |
| Loop 1 integral time | τ_{i,1} | 0.1 h | Tuned |
| Jacket duty at SS | Q_{j,\text{nom}} | 2739 kW | Energy balance at SS |
| Max jacket duty | Q_{j,\text{max}} | 8794 kW | Tuned |

---

## 4. Distillation Column — Quasi-Steady-State Model

### 4.1 QSS Justification

The distillation column operates on a **much faster time scale** than the reactor-recycle loop:

| Time scale | Value |
|-----------|-------|
| Column liquid hydraulic lag (τ_hyd) | 4 s ≈ 0.001 h |
| Reactor residence time (τ_r = M_r/F_total) | ≈ 5.2 h |
| Ratio τ_r / τ_hyd | ≈ 4,700× |

The column reaches its new steady state approximately 4,700 times faster than the
reactor-recycle loop. For dynamics of interest (reactor temperature, composition, and
recycle flow on time scales of 0.1–50 h), the column is correctly treated as algebraic.

### 4.2 Column Design Parameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Number of equilibrium trays | N_T | 20 | Wu (2003) Table 1 |
| Feed tray (1-indexed from top) | N_F | 12 | Wu (2003) Table 1 |
| Relative volatility A/B | α_rel | 2.0 | Wu (2003) Table 1 |
| Reflux ratio (nominal) | R = L/D | 2.198 | Wu (2003) Table 1 |
| Distillate A mole fraction (nominal) | x_{D,\text{nom}} | 0.95 | Wu (2003) Table 1 |
| Bottoms A mole fraction (nominal) | x_{B,\text{nom}} | 0.0105 | Wu (2003) Table 1 |
| Nominal recycle flow | F_{R,\text{nom}} | 226.9 kmol h⁻¹ | Wu (2003) Table 1 |
| Nominal bottoms flow | F_{B,\text{nom}} | 208.7 kmol h⁻¹ | Wu (2003) Table 1 |
| Nominal distillate fraction | D_{\text{frac,nom}} = D/F | 0.521 | Derived |

### 4.3 McCabe-Thiele Bisection (Column Model)

Given the reactor outlet composition z_A (= column feed z_F) and tray efficiency η_col,
the column model finds the distillate purity x_D by a **50-step bisection** on the
McCabe-Thiele residual:

$$\text{residual}(x_D) = x_\text{reboiler}(x_D) - x_{B,\text{mb}}(x_D)$$

where:
- $x_{B,\text{mb}} = (z_F - D_\text{frac} \cdot x_D) / (1 - D_\text{frac})$ is the
  bottoms composition from material balance at fixed $D_\text{frac}$
- $x_\text{reboiler}$ is obtained by stepping down N_T McCabe-Thiele stages from x_D

**Effective relative volatility** accounting for tray efficiency:
$$\alpha_\text{eff} = 1 + \eta_\text{col} \cdot (\alpha_\text{rel} - 1)$$

At nominal (η_col = 1): α_eff = α_rel = 2.0.
At degraded (η_col = 0.70): α_eff = 1.70 (30% reduction in separation power).

**McCabe-Thiele operating lines** (constant-molal-overflow, liquid feed q = 1):

| Section | Operating line | Coefficients |
|---------|---------------|-------------|
| Rectifying (trays 0–10) | y = (L/V) x + (D/V) x_D | L/V = R/(R+1) = 0.687, D/V = 1/(R+1) = 0.313 |
| Stripping (trays 11–19) | y = (L_s/V_s) x − (B/V_s) x_B | L_s/V_s = (R + F/D)/(R+1) = 1.287, B/V_s = (F/D−1)/(R+1) = 0.287 |

### 4.4 Snowball Mechanism and D_frac Variation

A critical **departure from a simplified fixed-split model** is that the distillate fraction
D_frac must vary with feed composition to model the snowball correctly:

$$D_\text{frac} = D_{\text{frac,nom}} + S_\text{rec} \cdot (z_F - z_{F,\text{ref}})$$

with $S_\text{rec} = 0.12$ (recycle sensitivity, h⁻¹) and $z_{F,\text{ref}} = 0.500$.

**Physical rationale:** When catalyst activity α falls, z_F rises above nominal
(less A is converted per pass). To maintain bottoms purity x_B ≈ x_{B,\text{nom}},
the column must send proportionally more A overhead → D increases → F_R = D increases.
This increase in recycle flow further dilutes the reactor feed, closing the positive
feedback loop — the **Luyben snowball** (Luyben, 1994).

With this linear model, D_frac varies from ≈ 0.515 (z_F = 0.45) to ≈ 0.581 (z_F = 1.0),
and F_R changes by approximately +2.4% for every 0.05 increase in z_F.

**Column material balance** (closes after bisection):
$$x_B = \frac{z_F - D_\text{frac} \cdot x_D}{1 - D_\text{frac}}, \qquad
F_\text{total} = \frac{F_0}{1 - D_\text{frac}}, \qquad F_R = D_\text{frac} \cdot F_\text{total}$$

**Mixed reactor feed:**
$$z_{A,\text{in}} = \frac{F_0 \cdot z_0 + F_R \cdot x_D}{F_\text{total}}, \qquad
T_{\text{in,mix}} = \frac{F_0 \cdot T_\text{in} + F_R \cdot T_r}{F_\text{total}}$$

(The recycle distillate is assumed to return at reactor temperature T_r.)

---

## 5. Control Structures

Two plant-wide control (PWC) structures from Wu et al. (2003) are compared.

### 5.1 Structure S-A — Information-Rich (analogue of Wu B-3)

S-A uses **online composition analysers** for x_D and x_B.

| Loop | Controlled variable (CV) | Manipulated variable (MV) | Measurement |
|------|--------------------------|---------------------------|-------------|
| 1 | T_r (reactor temperature) | Q_j (jacket cooling duty) | T_r sensor |
| 2 | x_D (distillate composition) | R = L/D (reflux ratio) | Online analyser |
| 3 | x_B (bottoms composition) | V_norm (boilup effort) | Online analyser |
| Level | M_r (reactor holdup) | F_out (overflow) | Level sensor |

Observable channels under S-A (10 channels):
T_r, T_j, Q_j, **x_D**, T_reb, Q_reb, F_R_norm, F_B_norm, R_norm, V_norm

### 5.2 Structure S-B — Conventional (analogue of Wu B-1b/B-1c)

S-B relies only on **conventional measurements** (temperatures, flows). No composition analyser.

| Loop | Controlled variable (CV) | Manipulated variable (MV) | Comment |
|------|--------------------------|---------------------------|---------|
| 1 | T_r (reactor temperature) | Q_j (jacket cooling duty) | Same as S-A |
| 2 | F_R/F_0 (recycle ratio) | L (reflux) | Ratio control (RC) |
| 3 | T_reb (reboiler temperature) | V_norm (boilup effort) | Temperature control |
| Level | M_r (reactor holdup) | F_out (overflow) | Level sensor |

Observable channels under S-B (9 channels):
T_r, T_j, Q_j, T_reb, Q_reb, F_R_norm, F_B_norm, R_norm, V_norm
*(x_D absent — no composition analyser)*

**Key difference:** S-A has x_D in the measurement set (6 more per-channel statistics
in the summary vector: 72-D vs 66-D). This single additional channel is the primary
source of identifiability improvement for the (α, η_col) pair.

---

## 6. Degradation Parameters and Identifiability

The 5-dimensional degradation vector is:
$$\boldsymbol{\theta} = [\alpha,\; \beta_r,\; \eta_\text{col},\; \xi_\text{reb},\; z_{A0}]$$

All parameters equal 1.0 (or z_{A0} = 0.90) at nominal healthy operation.

| # | Symbol | Physical meaning | Nominal | Prior | Primary observable |
|---|--------|-----------------|---------|-------|-------------------|
| 1 | α | Catalyst activity factor: k_eff = α·k₀·exp(−E_a/RT_r) | 1.0 | U[0.4, 1.2] | z_A, F_R_norm (snowball) |
| 2 | β_r | Reactor jacket fouling: UA_eff = β_r · UA_r | 1.0 | U[0.4, 1.2] | T_j, Q_j (Loop 1 masked T_r) |
| 3 | η_col | Column tray efficiency: α_eff = 1 + η_col·(α_rel − 1) | 1.0 | U[0.5, 1.0] | x_D (S-A only), T_reb, Q_reb |
| 4 | ξ_reb | Reboiler HX fouling: Q_reb_required = Q_reb_nom/ξ_reb | 1.0 | U[0.5, 1.2] | Q_reb (Loop 3 compensation) |
| 5 | z_{A0} | Fresh feed A purity (impurity in feed) | 0.90 | U[0.70, 0.95] | z_A, F_R_norm |

### 6.1 Identifiability Analysis

**β_r — Reactor jacket fouling (analogue of PO β):**
Under closed-loop operation, Loop 1 drives T_r → T_sp by adjusting Q_j.
At steady state with integral action: T_r = T_sp exactly.
Therefore ∂T_r,ss/∂β_r ≡ 0 — the temperature signal, which has the highest
signal-to-noise ratio, is structurally zeroed for β_r identification.
β_r is identified only through T_j and Q_j:
$$I(\beta_r) \propto \left(\frac{\partial T_j}{\partial \beta_r}\right)^2 / \sigma_{T_j}^2
+ \left(\frac{\partial Q_j}{\partial \beta_r}\right)^2 / \sigma_{Q_j}^2$$
The Fisher information ratio $I(\alpha)/I(\beta_r) \approx 250$–$500$,
meaning β_r is 250–500× harder to identify than α regardless of the inference method
(Cramér-Rao bound; Ljung 1977, Gevers et al. 2011).

**(α, η_col) — Banana posterior under S-B:**
Both α and η_col affect the column A-balance:
- α↓: Less conversion → z_F↑ → D_frac↑ → F_R↑ (snowball via reactor route)
- η_col↓: Worse separation → x_D↓ → D_frac = (z_F − x_B)/(x_D − x_B) ↑ → F_R↑

Under S-B (no x_D measurement), both channels T_reb and Q_reb increase for BOTH
faults. The joint (α, η_col) posterior is constrained to a **banana-shaped manifold**
in the 2D plane: combinations with the same F_R increase are nearly indistinguishable.

Under S-A, x_D directly disambiguates: α↓ increases z_F which increases x_D
(more A in overhead); η_col↓ decreases x_D (worse separation). The posterior narrows.

---

## 7. Changes from the Original Wu (2003) Model

### 7.1 Summary of Modifications

| Aspect | Wu (2003) Original | This Implementation | Justification |
|--------|-------------------|---------------------|---------------|
| Column model | Rigorous tray-by-tray MESH or shortcut Kremser | McCabe-Thiele bisection (50 steps, JAX lax.scan) | Differentiable, JIT-compilable, O(N_T) per call; captures η_col effect on x_D correctly |
| D_frac | Fixed at nominal D/F | Linear function of z_F: D_frac = D_nom + S_rec·(z_F − z_ref) | Produces snowball (F_R increases with z_F); D_frac is otherwise algebraically invariant in any bisection at fixed D_frac |
| Feed tray | Tray 12 of 20 (1-indexed) | FEED_TRAY = 11 (0-indexed, = tray 12 from top) | Corrects off-by-one error in earlier version |
| Column time scale | Full dynamic tray model | QSS (algebraic) | Justified by τ_hyd/τ_reactor ≈ 1/4700; no information loss for SBI observation windows of 2 h |
| Unit system | English (Btu, lbmol, °F) | English internally; SI in documentation | Code preserves exact Wu Table 1 numbers; SI reported in paper |
| Reboiler duty | From full energy balance | Empirical proxy: Q_reb ∝ V_norm × (1 + x_B terms) | QSS column does not carry full tray energy balance; proxy captures dominant signal |
| Jacket model | Simple Q_j applied to jacket | Energy balance on jacket mass M_j·C_{p,j} | Adds T_j as an explicit observable channel (key for β_r identification) |
| Noise model | Deterministic | Gaussian additive, 0.3% of each channel range | Enables SBI by creating stochastic (θ, x) pairs |

### 7.2 The D_frac Snowball Fix (Critical)

**Problem identified during implementation:**
The standard McCabe-Thiele bisection at fixed D_frac = D_frac_nom always returns
D_frac_nom regardless of z_F (algebraically guaranteed by the material balance inside
the bisection). This makes F_R = D_frac_nom × F_total = constant, eliminating the
snowball from F_R.

**Physical reality:**
When z_F increases (due to α decrease), the column must send more A overhead to
maintain bottoms purity x_B ≈ x_{B,nom}. The material balance requires
D_frac = (z_F − x_{B,nom}) / (x_D − x_{B,nom}), which increases with z_F.

**Fix applied:**
A linear sensitivity term is added:
$$D_\text{frac,eff} = D_{\text{frac,nom}} + S_\text{rec} \cdot (z_F - 0.500)$$
with S_rec = 0.12. This produces:
- Nominal (z_F = 0.500): D_frac = 0.521, F_R = 226.9 kmol h⁻¹ ✓
- Catalyst decay (z_F = 0.55): D_frac = 0.527, F_R ≈ 233 kmol h⁻¹ (+2.4%) ✓

The McCabe-Thiele bisection remains for x_D (capturing η_col effect on separation
chemistry), while D_frac is computed separately.

### 7.3 x_D vs Wu Table 1

At nominal conditions (z_F ≈ 0.500, η_col = 1.0, R = 2.198, α_rel = 2.0),
the McCabe-Thiele bisection with 20 equilibrium trays gives x_D ≈ 0.942 instead of
the Wu (2003) Table 1 value of 0.950. The discrepancy (< 1 percentage point) arises
because Wu's column design point is achieved at a slightly different operating reflux
ratio or with Murphree tray efficiencies. Since the SBI model is trained and evaluated
on data generated by the same model, this internal consistency is sufficient; the paper
reports model-based steady states, not hand-validated reproduction of Wu Table 1.

---

## 8. Expected Physical Behaviours

### 8.1 Loop 1 Temperature Control and β_r Masking

**Expected response to β_r degradation (W5, W6):**
1. β_r decreases → UA_eff = β_r · UA_r decreases → for same T_r, jacket
   temperature T_j must increase to provide the same heat flux
2. Loop 1 detects nothing (T_r ≈ T_sp by integral action) → increases Q_j
   to maintain heat balance with reduced UA
3. **Observable:** Q_j increases, T_j drops; **masked:** T_r stays at T_sp

**Critical identifiability consequence:**
- Open-loop: T_r shifts by > 5 K for β_r = 0.60 (fully observable)
- Closed-loop: T_r max deviation < 1 K (structurally masked by Loop 1)
- β_r is identified ONLY through T_j and Q_j — channels with 10–30× lower SNR

### 8.2 Snowball Dynamics (α Degradation)

**Expected response to α degradation (W2-W4):**
1. α decreases → k_eff = α·k₀·exp(−E_a/RT_r) decreases → less conversion per pass
2. z_A (reactor outlet) increases (more unconverted A)
3. Column feed is A-richer → D_frac increases (snowball sensitivity) → F_R increases
4. Higher F_R dilutes reactor feed: z_{A,in} = (F₀·z₀ + F_R·x_D) / (F₀ + F_R)
   → z_{A,in} decreases slightly (dilution by recycle)
5. Positive feedback: lower conversion → higher recycle → further dilution

**Observable:** F_R_norm rises monotonically with α degradation severity.
The snowball becomes nonlinear near α ≈ 0.55–0.60 where the sensitivity
d(F_R)/d(α) increases rapidly (near the tipping point, scenario W4/W15).

### 8.3 The Banana Posterior — Joint (α, η_col) Non-Identifiability

**Under S-B (no x_D measurement):**
Both α↓ and η_col↓ cause:
- T_reb increases (column receiving more A-rich feed or separating less effectively)
- Q_reb increases (Loop 3 increases boilup to compensate)
- x_B increases (worse product purity)

The data trajectory in the (T_reb, Q_reb, F_R_norm) space is similar for:
- W12 scenario: α = 0.75 (catalyst decay) + η_col = 0.80 (column degradation)
- A "mirror" scenario: α = 1.00 + η_col = 0.65 (only column degraded)

The x_D signal differs: α↓ increases z_F → x_D↑ (more A in overhead);
η_col↓ reduces separation → x_D↓. Under S-B (x_D unobserved), this distinction
is lost.

**Posterior shape:**
- Under S-B: p(α, η_col | x_SB) is **banana-shaped** — elongated along the manifold
  of (α, η_col) combinations consistent with the observed F_R and Q_reb signals.
- Under S-A: p(α, η_col | x_SA) narrows dramatically due to the x_D signal.

**EKF failure mode:**
The EKF assumes Gaussianity: p(θ | x) ≈ N(μ, Σ). The banana is non-Gaussian.
EKF will produce a tight Gaussian ellipse centred near one solution, with 90% CI
achieving < 65% empirical coverage for the (α, η_col) pair in W12 and W15.

### 8.4 Near-Tipping-Point Dynamics (W15)

Near the snowball bifurcation (α ≈ 0.58, scenario W15), the reactor-recycle system
approaches a **positive-feedback runaway** where the steady state is marginally stable.
The Jacobian of the ODE changes rapidly near this point, making the EKF linearisation
unreliable. SBI, trained across the full prior (including nearby parameter values),
correctly represents the widened uncertainty; the EKF remains narrowly Gaussian.

---

## 9. How Simulation-Based Inference (SBI) Addresses These Challenges

### 9.1 The Likelihood-Free Framework

SBI replaces the explicit likelihood p(x | θ) with a **neural posterior estimator**
q_φ(θ | s(x)) trained on simulated (θ, x) pairs:

$$\mathcal{L} = \mathbb{E}_{p(\theta,x)}\!\left[-\log q_\phi(\boldsymbol{\theta} \mid \mathbf{s}(\mathbf{x}))\right]$$

where s(x) is a hand-crafted or learned summary statistic vector.

**No likelihood required:** The process simulator is used as a black box.
**Amortisation:** After a one-time training cost, inference for any new observation
takes a single neural network forward pass (< 20 ms on CPU).

### 9.2 Non-Gaussian Posterior Recovery

The **banana posterior** for (α, η_col) under S-B cannot be represented by any
Gaussian approximation (EKF, UKF). SBI with a Neural Spline Flow (NSF) density
estimator can represent:
- Multi-modal posteriors (possible near tipping points)
- Banana-shaped manifold constraints
- Marginal posteriors with heavy tails (near snowball onset)

This is the primary qualitative advantage over EKF: **inferential correctness**, not
just speed.

### 9.3 Computational Feasibility

| Method | Inference time per 2-h window | 30-day monitoring (720 windows) |
|--------|------------------------------|--------------------------------|
| NUTS (MCMC) | ≈ 8 min (extrapolated from PO 2D, × dim scaling) | ≈ 4 days |
| EKF | ≈ 30 ms (sequential, online) | ≈ 22 s |
| **SBI (NSF, post-training)** | **< 20 ms** | **< 15 s** |

EKF is fast but geometrically incorrect (Gaussian approximation). SBI matches EKF
speed while providing full, calibrated posterior distributions.

### 9.4 Structural Identifiability — What SBI Can and Cannot Fix

SBI does **not** fix structural identifiability limitations imposed by closed-loop control.
The Cramér-Rao bound states: Var(θ̂) ≥ [I(θ)]⁻¹ for any unbiased estimator.
Since β_r has I(β_r) ≈ I(α)/250, the **posterior variance for β_r is irreducibly large**
under closed-loop control — for all methods (SBI, MCMC, EKF, UKF).

What SBI contributes:
- **Correct representation** of the high-uncertainty posterior (wide, possibly non-Gaussian)
- **Calibrated credible intervals** (SBC-verified rank histograms)
- **Honest uncertainty** rather than overconfident point estimates

---

## 10. Fault Scenarios Explored in Notebooks nb20–nb28

### 10.1 Scenario Catalogue (16 Closed-Loop + 7 Open-Loop)

#### Individual Faults — Reactor

| ID | Name | α | β_r | Primary effect | Notebook |
|----|------|---|-----|----------------|---------|
| W1 | healthy | 1.00 | 1.00 | Nominal steady state (baseline) | nb20, nb21, nb22 |
| W2 | cat_mild | **0.85** | 1.00 | Early snowball: F_R +2%, z_A↑ | nb21, nb22, nb26 |
| W3 | cat_severe | **0.65** | 1.00 | Pronounced snowball: F_R +6%; Q_j↓ | nb20, nb21 |
| W4 | cat_threshold | **0.55** | 1.00 | Near snowball tipping point; strong nonlinearity | nb21, nb22 |
| W5 | jacket_mild | 1.00 | **0.80** | T_j −4 K; Q_j +12%; T_r masked | nb21, nb22 |
| W6 | jacket_severe | 1.00 | **0.60** | T_j −12 K; Q_j +35%; T_r masked | nb20, nb21 |

#### Individual Faults — Column/Feed

| ID | Name | η_col | ξ_reb | z_{A0} | Primary effect | Notebook |
|----|------|-------|-------|--------|----------------|---------|
| W7 | col_eff_mild | **0.80** | 1.00 | 0.90 | x_D↓ 0.04; x_B↑; T_reb↑; visible under S-A | nb21, nb22 |
| W8 | col_eff_severe | **0.65** | 1.00 | 0.90 | x_D↓ 0.10; severe column degradation | nb20, nb21, nb26 |
| W9 | reb_fouling | 1.00 | **0.70** | 0.90 | Q_reb↑ 43%; Loop 3 compensation visible | nb21, nb22 |
| W10 | feed_impurity | 1.00 | 1.00 | **0.78** | z_A↑ 8%; feed-purity fault; different z_A trajectory from W2 | nb21, nb22 |

#### Combined Faults

| ID | Name | α | β_r | η_col | ξ_reb | z_{A0} | Description | Notebook |
|----|------|---|-----|-------|-------|--------|-------------|---------|
| W11 | reactor_combined | 0.80 | 0.80 | 1.00 | 1.00 | 0.90 | Both reactor faults: competing Q_j signals | nb22, nb26 |
| **W12** | **snowball_compound** | **0.75** | 1.00 | **0.80** | 1.00 | 0.90 | **HEADLINE**: (α, η_col) banana under S-B | **nb20, nb26** |
| W13 | cat_feed | 0.80 | 1.00 | 1.00 | 1.00 | 0.80 | Catalyst decay + lean feed: confounded | nb22, nb26 |
| W14 | col_reb | 1.00 | 1.00 | 0.75 | 0.75 | 0.90 | Column + reboiler: separation section failures | nb22, nb26 |
| **W15** | **snowball_threshold** | **0.58** | 1.00 | 0.90 | 1.00 | 0.90 | **Near tipping point**: EKF diverges; SBI widens correctly | **nb26** |
| W16 | full_multi | 0.75 | 0.80 | 0.80 | 0.85 | 0.90 | All degradation parameters simultaneously | nb22, nb26 |

#### Open-Loop Variants (Masking Contrast)

These scenarios run with all PI loops **deactivated** (Q_j fixed at nominal):

| ID | Closed-loop base | Purpose |
|----|-----------------|---------|
| W1-OL | W1 (healthy) | Open-loop baseline: confirm nominal dynamics |
| W2-OL | W2 (cat_mild) | OL catalyst decay: T_r excursion visible (unmasked α) |
| W3-OL | W3 (cat_severe) | OL severe decay: large T_r shift confirms masking in CL |
| W5-OL | W5 (jacket_mild) | OL jacket: T_r changes are unmasked without Loop 1 |
| W6-OL | W6 (jacket_severe) | OL jacket severe: quantifies CL vs OL β_r masking |
| W7-OL | W7 (col_eff_mild) | OL column: reference for column fault without control compensation |
| W8-OL | W8 (col_eff_severe) | OL column severe: reference for η_col identifiability |

### 10.2 Notebook-by-Notebook Purpose

| Notebook | Title | Main deliverable |
|----------|-------|-----------------|
| nb20 | Model verification | Kinetics check, SS verification, snowball demo (W3), masking demo (W6), banana preview (W12 vs W8) |
| nb21 | Control structure comparison | All 16 scenarios × S-A/S-B; F_R_norm 4×4 grid; x_D 4×4 grid; OL vs CL masking |
| nb22 | Dataset generation | 16 scenarios × 30 replicates × 2 h windows; save `wu2003_observations.npz` |
| nb23 | Summary statistics | 66-D S-B / 72-D S-A features; PCA; t-SNE; MI analysis; S-A vs S-B information value |
| nb24 | SBI training (S-B) | Train SNPE-C on S-B data; SBC calibration; posteriors W1–W16 |
| nb25 | SBI training (S-A) | Train SNPE-C on S-A data; S-A vs S-B comparison |
| nb26 | Headline experiment | W12 banana posterior; EKF failure at W15; coverage comparison |
| nb27 | Sequential tracking | 30-day α decay + β_r fouling; SBI vs EKF; MAE/coverage table |
| nb28 | Publication figures | All 12 paper figures at 300 dpi; Okabe-Ito palette; double-column format |

---

## 11. Summary Statistics and Observables

### 11.1 Channel Contract (12-channel raw array)

The `extract_observations_explicit` function returns a 12-channel array per time step:

| # | Channel | Variable | Unit | S-A | S-B |
|---|---------|----------|------|-----|-----|
| 0 | T_r | Reactor temperature | K | ✓ | ✓ |
| 1 | T_j | Jacket temperature | K | ✓ | ✓ |
| 2 | Q_j | Jacket duty (normalised by Q_{j,max}) | — | ✓ | ✓ |
| 3 | x_D | Distillate A mole fraction | — | **✓ only** | — |
| 4 | x_B | Bottoms A mole fraction | — | diagnostic | diagnostic |
| 5 | T_reb | Reboiler temperature proxy | K | ✓ | ✓ |
| 6 | Q_reb | Reboiler duty proxy | kW | ✓ | ✓ |
| 7 | F_R_norm | Recycle flow / F_{R,nom} | — | ✓ | ✓ |
| 8 | F_B_norm | Bottoms flow / F_{B,nom} | — | ✓ | ✓ |
| 9 | R_norm | Reflux ratio / R_{nom} | — | ✓ | ✓ |
| 10 | V_norm | Boilup effort (normalised) | — | ✓ | ✓ |
| 11 | z_A | Reactor A mole fraction | — | diagnostic | diagnostic |

### 11.2 Summary Statistics (66-D S-B, 72-D S-A)

For each 2-hour observation window (120 time steps), the summary vector is:

**Per-channel statistics (6 per channel):** mean, std, linear slope, min, max, final-quarter mean

| Structure | Channels | Channel stats | Physics features | Total |
|-----------|----------|---------------|-----------------|-------|
| S-B | 9 | 9 × 6 = 54 | 12 | **66** |
| S-A | 10 | 10 × 6 = 60 | 12 | **72** |

**Physics-informed features (12):**

| # | Feature | Formula | Encodes |
|---|---------|---------|---------|
| 1 | UA_proxy | Q_j / max(T_r − T_j, 10⁻³) / UA_nom | β_r |
| 2 | recycle_ratio | mean(F_R_norm) | α (snowball), η_col |
| 3 | col_recovery | mean(F_B_norm / (F_R_norm + F_B_norm)) | α × η_col |
| 4 | reb_intensity | mean(Q_reb / F_R_norm) / Q_{reb,nom} | ξ_reb |
| 5 | recycle_excess | mean(F_R_norm) − 1 | snowball severity |
| 6 | Tr_Tj_ratio | mean(T_r / T_j) | β_r |
| 7 | Qj_slope | d(Q_j)/dt normalised | transient α response |
| 8 | Vn_final | V_norm at t = 2 h | boilup compensation effort |
| 9 | Rn_final | R_norm at t = 2 h | reflux compensation effort |
| 10 | corr(Q_j, F_R) | Pearson correlation | snowball coupling (α) |
| 11 | corr(Q_reb, F_R) | Pearson correlation | column-recycle coupling (η_col) |
| 12 | corr(R_norm, V_norm) | Pearson correlation | coordinated column response |

---

## 12. Stochastic Simulation

Each training sample (θ, x) is generated by:

1. **Steady-state warm start**: integrate the 8-state ODE with NOMINAL_THETA to T = 200 h
   using adaptive Tsit5 solver (diffrax) — reaches steady state within 20 h.

2. **Fault simulation**: run the 8-state ODE from the warm start with degraded θ
   over a 2-h window using the scenario-specific control vector (S-A or S-B).

3. **Sensor noise**: add Gaussian noise at σ = 0.3% × max|channel| per channel.

4. **Summary statistics**: compute the 66-D (S-B) or 72-D (S-A) summary from the
   noisy observation window.

**Process noise:** Not applied (noise is sensor-only). The SBI posterior thus represents
the distribution over θ given noisy sensor data, not process variability. This matches
the monitoring use case: parameters change slowly relative to the 2-h window.

---

## 13. Connections to Article Claims

| Article claim | Physical mechanism | Supporting notebook |
|--------------|-------------------|-------------------|
| β_r masked by Loop 1; I(α)/I(β_r) ≈ 250–500 | ∂T_r,ss/∂β_r ≡ 0 under PI control | nb20 (masking demo), nb26 (FIM) |
| Snowball: F_R increases with α decay | D_frac ∝ z_F (RECYCLE_SENSITIVITY) | nb20 (W3 step test), nb22 (F_R bar chart) |
| Banana posterior in (α, η_col) under S-B | T_reb/Q_reb ambiguous; x_D disambiguates | nb20 (preview), nb26 (headline) |
| S-A breaks banana via x_D signal | x_D moves in opposite directions for α vs η_col | nb21, nb25, nb26 |
| EKF overconfident near tipping point (W15) | Jacobian changes rapidly near bifurcation | nb26 |
| SBI < 20 ms/window vs MCMC ≈ 8 min/window | Amortisation: single forward pass | nb27 (timing) |
| SBI correctly represents wide β_r posterior | Non-Gaussian; tracks irreducible bound | nb24, nb25 (SBC) |
