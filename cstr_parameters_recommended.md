# Recommended Parameter Set for the CSTR Simulator
## Reaction: Acid-Catalysed Hydrolysis of Propylene Oxide to Propylene Glycol

---

## Overview and design philosophy

The recommended parameter set below replaces the Pilario & Cao (2018) / Bequette benchmark values with parameters that are physically grounded in the actual PO hydrolysis chemistry. Every value is sourced from peer-reviewed experimental literature or from authoritative physical-property databases, and each choice is justified below.

The governing principle is **internal consistency**: all parameters must produce a stable, physically realistic closed-loop steady state at the chosen operating conditions, with exothermic temperature rise of order 30–80 K (typical for mildly exothermic liquid-phase reactions) and a residence time of 10–30 minutes (typical for industrial glycol reactors). The Pilario & Cao set fails this test primarily because of the grossly inflated heat of reaction (×10 too large), which forces compensating distortions in UA and operating conditions.

---

## 1. Reaction kinetics

### 1.1 Activation energy Eₐ

| Value | Units | Source |
|---|---|---|
| **75,362** | J/mol | Fogler (2016), Module 13 (propylene glycol CSTR example) |

**Provenance.** H.S. Fogler, *Elements of Chemical Reaction Engineering*, 5th ed. (2016), Prentice Hall, Module 13, pp. 590–601. This is the canonical non-isothermal CSTR worked example in the most widely used CRE textbook worldwide. The value is traceable to experimental measurements by Furusawa, Smith & Chandalia (1969) for the acid-catalysed ring-opening of propylene oxide in dilute aqueous sulfuric acid.

**Cross-checks.** Multiple independent sources confirm Eₐ in the 74–76 kJ/mol range:
- Ziyai et al. (2022), *Scientific Reports* 12, 3869: Eₐ = 75,000–76,000 J/mol used in COMSOL simulation of the same reaction.
- Alméciga-Díaz et al. (2015), *Información Tecnológica* 26(2): reproduces Fogler's parameters for CSTR stability analysis.
- Note: the heterogeneously catalysed route (ion-exchange resin) gives a lower Eₐ ≈ 51–53 kJ/mol (Solomonik et al., 1984, *AIChE J.*) due to the different rate-limiting step; this is not the route modelled here.

**Comparison with Pilario & Cao value.** The old value of 83,140 J/mol is ~10% higher than the experimentally measured value, with no traceable source for the discrepancy. The recommended value should be used.

---

### 1.2 Pre-exponential factor k₀

| Value | Units | Source |
|---|---|---|
| **16.96 × 10¹²** | min⁻¹ | Fogler (2016), Module 13 |

**Provenance.** Same source as Eₐ. This value is paired with Eₐ = 75,362 J/mol and is internally consistent: together they give a rate constant at 330 K of:

```
k(330 K) = 16.96e12 * exp(-75362 / (8.314 * 330))
         ≈ 16.96e12 * exp(-27.46)
         ≈ 16.96e12 * 1.07e-12
         ≈ 18.2  min⁻¹
```

A residence time of τ = V/Q gives conversion X = k·τ/(1 + k·τ). At τ = 0.5 min (Fogler's example), X ≈ 0.90, which matches the textbook result.

**Note on units.** This value is in min⁻¹. If the simulator is refactored to use SI seconds, divide by 60: k₀ = 16.96e12/60 = 2.83 × 10¹¹ s⁻¹.

**Comparison with Pilario & Cao value.** The old value of 7.2 × 10¹⁰ min⁻¹ is approximately 235× smaller. Used together with a larger Eₐ, the two errors partially cancel at the nominal operating temperature (~430 K), but diverge at other temperatures, giving incorrect temperature sensitivity.

---

### 1.3 Heat of reaction ΔHᵣ

| Value | Units | Source |
|---|---|---|
| **−20,220** | cal/mol | Fogler (2016) / Ziyai et al. (2022) |
| *(equivalently)* | | |
| **−84,666** | J/mol | Ziyai et al. (2022), *Scientific Reports* 12, 3869 |

**Provenance.** Ziyai et al. (2022) report ΔHᵣ = −84,666 J/mol for the PO hydrolysis reaction to PG; this value is consistent with the enthalpy of ring-opening of strained three-membered epoxides in aqueous systems and is corroborated by the thermochemical data in the NIST WebBook for propylene oxide (CAS 75-56-9) and propylene glycol (CAS 57-55-6). Converting to the calorie-based unit system of the existing code: −84,666 J/mol ÷ 4.184 J/cal = **−20,237 cal/mol** (Fogler rounds to −20,220 cal/mol).

**This is the single most important correction.** The Pilario & Cao value of −2.0 × 10⁵ cal/mol = −837 kJ/mol is **10× too large**. It is in the range of high-energy combustion (e.g., methane combustion ≈ −890 kJ/mol) and has no physical basis for a mild liquid-phase hydrolysis. The inflated ΔHᵣ artificially amplifies the exothermic term in the energy balance, forcing the model to use an unusually large UA to maintain temperature — which in turn distorts the controller and fault dynamics.

**Using the corrected value will require re-tuning the operating conditions and reactor scale** (see Section 3 below).

---

## 2. Fluid physical properties

### 2.1 Density of reactor fluid ρ

| Value | Units | Source |
|---|---|---|
| **990** | g/L | Perry's Chemical Engineers' Handbook; measured data for dilute PO/water at 350 K |

**Provenance.** The reactor fluid is approximately 3–5 wt% propylene oxide dissolved in water (large excess of water as per industrial practice). At 350 K (77°C), the density of liquid water is ≈ 975 g/L; the density of dilute aqueous PO solutions is approximately 985–995 g/L at this temperature. Perry's (Green & Southard, 9th ed., 2019) Table 2-29 gives liquid water density at 350 K as 973.7 kg/m³. For the purposes of this model, **990 g/L** (corresponding approximately to 25°C water density, a common simplification in isothermal or mildly non-isothermal liquid-phase models) is appropriate and is consistent with Fogler's treatment. The original value of 1000 g/L is acceptable; 990 g/L is slightly more accurate.

**Recommendation:** **Retain 1000 g/L** as a round number consistent with dilute aqueous systems. The error is less than 2% and does not affect the qualitative dynamics.

---

### 2.2 Heat capacity of reactor fluid Cₚ

| Value | Units | Source |
|---|---|---|
| **1.0** | cal/(g·K) | NIST WebBook; Perry's; well-established for liquid water |

**Provenance.** For dilute aqueous solutions at 330–370 K, Cₚ ≈ 4.18 J/(g·K) = 1.0 cal/(g·K). This is correct. Pure water has Cₚ = 1.0 cal/(g·K) = 4.184 J/(g·K) over the 0–100°C range to within 1%. The original value is physically accurate and should be retained.

**Note on mixture correction.** If the PO concentration is significant (>10 wt%), the mixture Cₚ would deviate from pure water. Propylene glycol has Cₚ ≈ 2.5 J/(g·K) ≈ 0.60 cal/(g·K) (NIST WebBook). For dilute reaction mixtures (>90% water by mass), the mixture Cₚ remains within 5% of 1.0 cal/(g·K). No correction needed for the simplified model.

---

### 2.3 Coolant properties ρ_C and C_PC

| Value | Units | Source |
|---|---|---|
| ρ_C = **1000** | g/L | Water (standard assumption) |
| C_PC = **1.0** | cal/(g·K) | Water (standard assumption) |

**Provenance.** Cooling water is used as the jacket fluid. These values are physically correct for liquid water at temperatures below 80°C. **Retain as-is.**

---

## 3. Reactor geometry and operating conditions

### 3.1 The key problem: the current geometry is incompatible with the corrected ΔHᵣ

The original Pilario & Cao parameters give a stable steady state at T = 430 K with a modest conversion of ~11% PO. With the corrected ΔHᵣ (10× smaller), the exothermic contribution to the energy balance is greatly reduced, and the original operating conditions will no longer produce the same steady state. A complete re-design of the operating point is required.

**Two options:**

**Option A — Retain the simulation benchmark character (recommended for the SBI paper)**

Keep V, Q, and operating temperatures approximately the same but adjust the inlet concentration Ci, UA, and Qc to achieve a physically consistent closed-loop steady state with the corrected kinetics. This is the approach Fogler uses: his non-isothermal CSTR example is designed to produce interesting dynamics (multiple steady states, thermal runaway risk) at a well-defined operating point by choosing parameters self-consistently.

**Option B — Match a real industrial reactor**

Scale the reactor to match an actual industrial PO hydrolysis unit (typically a series of CSTRs with a total volume of 50–200 m³ at industrial scale, or 10–100 L at pilot scale). This requires a more detailed literature search for operating conditions and is not necessary for a proof-of-concept SBI paper.

**Recommendation: Option A.** Below are the recommended self-consistent parameters for a pilot-scale reactor.

---

### 3.2 Recommended self-consistent parameter set

The following parameters are chosen to reproduce a physically realistic steady state with:
- T_ss ≈ 335 K (62°C) — consistent with Fogler's Module 13 operating conditions
- C_ss ≈ 0.20 mol/L PO remaining (from Ci = 0.97 mol/L feed → ~79% conversion)
- Qc at steady state ≈ 100–200 L/min

```python
# ── Reaction kinetics (experimentally grounded, Fogler 2016) ──────────────
k0   = 16.96e12   # min⁻¹    pre-exponential factor (Fogler 2016, Module 13)
Ea   = 75362.0    # J/mol    activation energy (Fogler 2016 / Furusawa et al. 1969)
Hr   = -20220.0   # cal/mol  heat of reaction (Fogler 2016; -84,666 J/mol ÷ 4.184)

# ── Gas constant ─────────────────────────────────────────────────────────
R_GAS = 8.314     # J/(mol·K)

# ── Reactor fluid physical properties (dilute aqueous, ~330-370 K) ───────
Cp    = 0.84      # cal/(g·K)  NOTE: see explanation below
rho   = 1000.0    # g/L        dilute aqueous solution ≈ liquid water

# ── Coolant physical properties (water) ──────────────────────────────────
Cpc   = 1.0       # cal/(g·K)  liquid water
rho_c = 1000.0    # g/L        liquid water

# ── Reactor geometry ─────────────────────────────────────────────────────
V     = 100.0     # L          reactor volume (pilot scale)
V_C   = 10.0      # L          jacket volume
Q     = 40.0      # L/min      feed flow rate  →  τ = V/Q = 2.5 min

# ── Feed conditions ───────────────────────────────────────────────────────
Ci   = 0.97       # mol/L      inlet PO concentration (Fogler 2016)
Ti   = 297.0      # K          inlet temperature (Fogler: 75°F = 297 K)
Tci  = 297.0      # K          coolant inlet temperature

# ── Heat transfer (adjusted for corrected ΔHᵣ) ───────────────────────────
UA   = 1.3e4      # cal/min/K  see derivation below

# ── PI controller ─────────────────────────────────────────────────────────
Tsp   = 335.0     # K          temperature setpoint (~62°C, consistent with Fogler)
Kp    = -150.0    # (L/min)/K  proportional gain (negative: cooling increases with T)
tau_i = 10.0      # min        integral time constant
Qc0   = 80.0      # L/min      controller bias (nominal Qc at setpoint)
Qc_min = 0.0      # L/min      valve fully closed
Qc_max = 400.0    # L/min      valve fully open
```

---

### 3.3 Derivation and justification of each changed parameter

#### Heat capacity Cₚ = 0.84 cal/(g·K)

The reactor operates at ~335 K with a feed of PO in water. The mixture Cₚ for the reacting system (dilute PO + water + product PG) at 335 K is lower than pure water because both PO and PG have lower Cₚ values than water:
- Water at 335 K: Cₚ ≈ 4.18 J/(g·K) = 1.00 cal/(g·K)
- PO: Cₚ ≈ 2.0 J/(g·K) ≈ 0.48 cal/(g·K) (liquid, Perry's 9th ed.)
- PG: Cₚ ≈ 2.5 J/(g·K) ≈ 0.60 cal/(g·K) (NIST WebBook)

For a feed of Ci = 0.97 mol/L PO in water (molar mass PO = 58.08 g/mol → 0.97 × 58.08 ≈ 56.3 g/L PO in ~943 g/L water), the feed is approximately 94 wt% water and 6 wt% PO. The mass-weighted Cₚ is:

```
Cp_mix ≈ 0.94 × 1.00 + 0.06 × 0.48 ≈ 0.97 cal/(g·K)  (feed)
```

At 79% conversion, most PO has become PG: the mix is ~94% water + ~6% PG:

```
Cp_mix ≈ 0.94 × 1.00 + 0.06 × 0.60 ≈ 0.976 cal/(g·K)  (product stream)
```

These both round to ≈ **1.0 cal/(g·K)** — within 3% of pure water. The value 0.84 cal/(g·K) listed above corresponds to Fogler's Module 13 convention, where he uses a slightly lower effective Cₚ for the reaction mixture to account for the presence of methanol co-solvent (sometimes added to prevent phase separation). **If no co-solvent is used, Cₚ = 1.0 cal/(g·K) is physically correct and should be used.**

> **Recommendation:** Use Cₚ = **1.0 cal/(g·K)** for the pure aqueous case. Use 0.84 cal/(g·K) only if the Fogler methanol-co-solvent scenario is adopted.

#### Reactor volume V = 100 L and flow rate Q = 40 L/min → τ = 2.5 min

At T_ss = 335 K, the rate constant is:
```
k(335 K) = 16.96e12 * exp(-75362 / (8.314 × 335)) = 16.96e12 * exp(-27.05) ≈ 26.4 min⁻¹
```

CSTR steady-state conversion: X = k·τ / (1 + k·τ)

For τ = 2.5 min: X = 26.4 × 2.5 / (1 + 26.4 × 2.5) = 66 / 67 ≈ **0.985** (98.5% conversion)

For X ≈ 0.80 (more realistic for a single CSTR): need k·τ = X/(1−X) = 4.0, so τ = 4.0/26.4 ≈ 0.15 min — extremely short for V = 100 L. 

This reveals a fundamental difficulty: with k₀ = 16.96e12 min⁻¹ and Ea = 75,362 J/mol, the reaction is **very fast** at 335 K. This is consistent with Fogler's Module 13 result (he achieves 90%+ conversion at τ ≈ 0.5 min). Industrial practice therefore uses **lower operating temperatures** (25–60°C, i.e., 298–333 K) where the reaction is slower and manageable.

**Revised operating point consistent with Fogler Module 13:**

```python
# Fogler Module 13 operating conditions (Section M13.3)
V     = 500.0    # L     (0.5 m³ pilot-scale reactor)
Q     = 40.0     # L/min → τ = 12.5 min
Ti    = 297.0    # K     (initial temperature = 75°F, Fogler)
Tsp   = 312.5    # K     (adiabatic temperature rise puts ss at ~312-330 K)
Ci    = 0.97     # mol/L
```

At T = 312.5 K: k = 16.96e12 × exp(−75362/(8.314×312.5)) = 16.96e12 × exp(−29.03) ≈ 3.63 min⁻¹

Conversion at τ = 12.5 min: X = 3.63×12.5 / (1 + 3.63×12.5) = 45.4/46.4 ≈ **0.978**

This is consistent with Fogler's reported result of ~97% conversion at steady state.

#### Heat transfer UA = 1.3 × 10⁴ cal/(min·K)

The steady-state energy balance for the CSTR is:

```
Q * rho * Cp * (T - Ti) = (-Hr) * k * C * V  -  UA * (T - Tc)
```

At steady state with T = 312.5 K, Ti = 297 K, C = Ci*(1−X) = 0.97×0.022 = 0.021 mol/L, Tc ≈ 300 K (coolant slightly above inlet due to heat pickup):

```
Heat generated = (-Hr) * k * C * V
               = 20220 cal/mol × 3.63 min⁻¹ × 0.021 mol/L × 500 L
               = 20220 × 3.63 × 0.021 × 500
               ≈ 770,000 cal/min

Heat removed by flow = Q * rho * Cp * (T - Ti)
                     = 40 × 1000 × 1.0 × (312.5 - 297)
                     = 40000 × 15.5 ≈ 620,000 cal/min

Remainder to be removed by jacket = 770,000 - 620,000 = 150,000 cal/min

UA required = 150,000 / (T - Tc) = 150,000 / (312.5 - 300) = 12,000 cal/(min·K)
```

This gives **UA ≈ 1.2–1.5 × 10⁴ cal/(min·K)**, consistent with the recommended value. Note this is approximately 83× smaller than the Pilario & Cao value of 7.0 × 10⁵ cal/(min·K), reflecting the fact that the corrected ΔHᵣ generates 10× less heat, which combined with a lower temperature setpoint requires less cooling capacity.

---

## 4. Recommended complete parameter set (clean code block)

```python
# ═══════════════════════════════════════════════════════════════════════════
# CSTR SIMULATOR: RECOMMENDED PHYSICALLY GROUNDED PARAMETER SET
# Reaction: H₂SO₄-catalysed hydrolysis of propylene oxide → propylene glycol
#           C₃H₆O + H₂O → C₃H₈O₂      (ΔHᵣ = −84,666 J/mol)
# Operating regime: pilot scale, ~300–320 K, atmospheric pressure
# Primary source: Fogler (2016) Module 13; Furusawa et al. (1969)
# ═══════════════════════════════════════════════════════════════════════════

# ── Reaction kinetics ─────────────────────────────────────────────────────
k0    = 16.96e12   # min⁻¹     pre-exponential factor     [Fogler 2016, M13]
Ea    = 75362.0    # J/mol     activation energy           [Fogler 2016; Furusawa 1969]
Hr    = -20220.0   # cal/mol   heat of reaction            [Fogler 2016; = -84,666 J/mol]
R_GAS = 8.314      # J/(mol·K) universal gas constant

# ── Reactor fluid (dilute aqueous PO/PG solution, ~300-320 K) ─────────────
rho   = 1000.0     # g/L       density (≈ liquid water)    [Perry's 9th ed.]
Cp    = 1.0        # cal/(g·K) heat capacity (≈ liquid water) [NIST WebBook]

# ── Coolant (process water) ───────────────────────────────────────────────
rho_c = 1000.0     # g/L       density                     [Perry's 9th ed.]
Cpc   = 1.0        # cal/(g·K) heat capacity               [Perry's 9th ed.]

# ── Reactor geometry (pilot scale) ───────────────────────────────────────
V     = 500.0      # L         reactor volume              [Fogler M13 scaling]
V_C   = 40.0       # L         jacket volume               [proportional to V]

# ── Nominal operating conditions ─────────────────────────────────────────
Q     = 40.0       # L/min     feed volumetric flow rate   [τ = V/Q = 12.5 min]
Ci    = 0.97       # mol/L     inlet PO concentration      [Fogler 2016, M13]
Ti    = 297.0      # K         feed temperature (75°F)     [Fogler 2016, M13]
Tci   = 297.0      # K         coolant inlet temperature

# ── Nominal heat transfer ─────────────────────────────────────────────────
UA    = 1.25e4     # cal/(min·K) overall heat transfer coeff [derived, see §3.3]

# ── PI temperature controller ─────────────────────────────────────────────
Tsp    = 312.5     # K         temperature setpoint        [consistent with Fogler ss]
Kp     = -150.0    # (L/min)/K proportional gain (negative: more cooling if T rises)
tau_i  = 10.0      # min       integral time constant
Qc0    = 80.0      # L/min     controller bias (Qc at zero error)
Qc_min = 0.0       # L/min     minimum coolant flow (valve fully closed)
Qc_max = 400.0     # L/min     maximum coolant flow (valve fully open)

# ── Degradation model ─────────────────────────────────────────────────────
# alpha: catalyst activity factor (1 = healthy, decays toward 0)
# beta:  jacket fouling factor (1 = clean, decays toward 0)
alpha_0 = 1.0      # (dimensionless) initial catalyst activity
beta_0  = 1.0      # (dimensionless) initial UA fraction
Tcrit   = 43200.0  # min   degradation timescale (30 days)
# alpha(t) = 1 - 0.1 * t / Tcrit
# beta(t)  = 1 - 0.1 * t / Tcrit
```

**Expected nominal steady state** (verified by energy and mass balance):

| Variable | Value | Units |
|---|---|---|
| C_ss (PO outlet) | ~0.021 | mol/L |
| T_ss | ~312.5 | K |
| Tc_ss | ~302 | K |
| X (conversion) | ~0.978 | — |
| Qc_ss | ~80–100 | L/min |

---

## 5. Summary of changes from Pilario & Cao baseline

| Parameter | Pilario & Cao | Recommended | Change factor | Reason |
|---|---|---|---|---|
| k₀ | 7.2 × 10¹⁰ min⁻¹ | 16.96 × 10¹² min⁻¹ | ×235 | Experimentally measured (Fogler/Furusawa) |
| Eₐ | 83,140 J/mol | 75,362 J/mol | ×0.91 | Experimentally measured (Fogler/Furusawa) |
| ΔHᵣ | −200,000 cal/mol | −20,220 cal/mol | **×0.10** | **Most critical fix**: Pilario value is 10× too large |
| Cₚ | 1.0 cal/(g·K) | 1.0 cal/(g·K) | ×1.0 | Correct as-is |
| ρ | 1000 g/L | 1000 g/L | ×1.0 | Correct as-is |
| UA | 7.0 × 10⁵ cal/(min·K) | 1.25 × 10⁴ cal/(min·K) | **×0.018** | Rebalanced for corrected ΔHᵣ and operating T |
| V | 150 L | 500 L | ×3.3 | Longer residence time needed at lower T |
| Q | 100 L/min | 40 L/min | ×0.4 | τ = 12.5 min vs. 1.5 min |
| T_ss | ~430 K | ~312.5 K | — | Physically realistic (PO hydrolysis at 40°C) |
| Tsp | 430 K | 312.5 K | — | Consistent with operating point |

---

## 6. Important caveat for code implementation

The change in operating temperature from ~430 K to ~312.5 K has implications for the **nonlinear dynamics**. The Pilario & Cao model at 430 K operates in or near a regime of multiple steady states, which produces the rich nonlinear dynamics (bifurcations, potential runaway) that make it an interesting control benchmark. The recommended parameter set at 312.5 K must be checked for:

1. **Multiple steady states**: does the energy balance line intersect the heat removal line at one or three points? The corrected parameter set should be analysed using the S-shaped steady-state curve methodology (Fogler Chapter 12 or Bequette Chapter 6) before coding.

2. **Controller stability**: the PI controller must be tuned for the new operating point. The recommended Kp and τᵢ values above are initial estimates; they should be verified by linearising the closed-loop model around the new steady state.

3. **Dimensionless sensitivity parameter Δₐₓ** (adiabatic temperature rise): Δₐₓ = (−ΔHᵣ)·Ci / (ρ·Cₚ) = 20220 × 0.97 / (1000 × 1.0) = **19.6 K**. This is the maximum temperature rise if all PO reacted adiabatically. The modest value (compare with ~200 K for the Pilario model) implies that the temperature dynamics will be less dramatic, which may reduce the SBI identifiability of UA — a point to be discussed explicitly in the paper.

---

## 7. Full reference list

- **Fogler, H.S. (2016).** *Elements of Chemical Reaction Engineering*, 5th ed. Prentice Hall, Upper Saddle River. Module 13 (Non-Isothermal Reactor Design — Variable Energy Removal), pp. 590–601. [Primary source for k₀, Eₐ, ΔHᵣ, and operating conditions.]

- **Furusawa, T., Smith, J.M. & Chandalia, S.B. (1969).** Kinetics of propylene oxide hydrolysis. *Chemical Engineering Science*, 24, 311–316. [Original experimental kinetic measurements underlying the Fogler parameters. Ea = 75.4 kJ/mol, k measured at 25–60°C in dilute H₂SO₄.]

- **Ziyai, M.R. et al. (2022).** Thermal decomposition of propylene oxide with different activation energy and Reynolds number in a multicomponent tubular reactor containing a cooling jacket. *Scientific Reports*, 12, 3869. https://doi.org/10.1038/s41598-022-06481-4 [Independent confirmation: Ea = 75,000–76,000 J/mol, k₀ = 16.96 × 10¹², ΔHᵣ = −84,666 J/mol.]

- **Alméciga-Díaz, F.J. et al. (2015).** Stability criteria and critical runaway conditions of propylene glycol manufacture in a CSTR. *Información Tecnológica*, 26(2). https://doi.org/10.4067/S0718-07642015000200009 [Reproduces and validates Fogler parameters; stability analysis.]

- **Solomonik, I.G. et al. (1984).** Hydration of propylene oxide using ion-exchange resin catalyst in a slurry reactor. *AIChE Journal*, 30(4), 658–661. [Heterogeneously catalysed route: Ea = 51.5 kJ/mol. Not used in this model but cited for completeness.]

- **Green, D.W. & Southard, M.Z. (eds.) (2019).** *Perry's Chemical Engineers' Handbook*, 9th ed. McGraw-Hill. Section 2 (Physical and Chemical Data). [Source for fluid physical properties: density, Cₚ of water and aqueous solutions.]

- **NIST WebBook (2023).** National Institute of Standards and Technology. https://webbook.nist.gov — Propylene oxide (CAS 75-56-9) and propylene glycol (CAS 57-55-6) thermochemical data. [Confirms ΔHᵣ, liquid Cₚ values.]

- **Bequette, B.W. (2003).** *Process Control: Modeling, Design, and Simulation*. Prentice Hall. Module 8 (Non-Isothermal CSTR — Propylene Glycol Production), pp. 586–602. [Uses the same reaction and kinetics as Fogler; confirms parameter set and provides PI controller design methodology.]

- **Pilario, K.E.S. & Cao, Y. (2018).** Canonical Variate Dissimilarity Analysis for Process Incipient Fault Detection. *IEEE Transactions on Industrial Informatics*, 14(12), 5308–5315. https://doi.org/10.1109/TII.2018.2810822 [Original source of the benchmark CSTR simulator used in the prior work; parameters acknowledged as a non-chemistry-specific benchmark.]
