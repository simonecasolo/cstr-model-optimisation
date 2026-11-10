# System III — design (2026-11-10)

*Formerly "System R". System III is the main system of the paper; System I is the marginal-design
contrast. Status of each decision: `docs/decision_log.md`.*

* Criteria were fixed before any candidate was computed
(`SCORECARD.md`, reproduced in §2). Toolkit: `design_lib.py`, `candidates.py`, `check_ss.py`,
`check_ctrl.py`, `tune_search.py`, `stoch.py`, `run_c6*.py`, `run_c8.py` (numpy/scipy; in `system_III/design/`). The toolkit reproduces System I's known numbers (98% conversion, 19%
jacket share, 3.5 K coolant rise, β saturation at 0.79; β Cramér–Rao sd 0.014 with the inlet
known and 0.032 with it measured, matching nb35), which validates it.*

> **Update 2026-11-10:** realism upgrade (analyser and thermowell dynamics, cascade control) in
> `docs/realism_upgrade_2026-11-10.md`. It changes the controller (decision 12), the α information (C6 still
> passes) and the scenario placement, and discloses a C3 failure of the as-built PI. The numbers below are the S3/PI design.

## 1. Recommendation

**R3: Fogler's propylene-glycol CSTR (Example 9-4, 3rd ed.; COMSOL "Startup of a CSTR"),
with the heat-transfer capacity sized with a fouling allowance (UA = 2 × Fogler's value), a coolant
valve sized for 35% opening at nominal, and a re-tuned PI controller.**

| Quantity | R3 | System I |
|---|---|---|
| Source | Fogler Ex. 9-4 / COMSOL (industrial scale) | Fogler-inspired, k0 units bug (60×) |
| Reactor volume, feed | 1890 L, 208 L/min (τ = 9.1 min) | 500 L, 40 L/min (τ = 12.5 min) |
| PO feed concentration | 2.90 mol/L (with water and methanol) | 0.97 mol/L |
| Kinetics | k0 = 4.71e9 s⁻¹ (= 2.826e11 min⁻¹), E = 75.36 kJ/mol | k0 60× too large |
| Heat of reaction; adiabatic rise | −84.7 kJ/mol; 71 K | −84.6 kJ/mol; 20 K |
| Feed / coolant inlet | 297 K / 289 K (chilled water) | 297 K / 297 K |
| Setpoint; conversion | 337 K (Fogler's steady state); **X = 84%** | 312.5 K; X = 98% |
| UA | 1.013e6 J/(min K) (16.9 kW/K = 2 × Fogler) | 5.23e4 J/(min K) |
| Jacket share of the reaction heat | 33% | 19% |
| **Coolant temperature rise** | **34 K** | 3.5 K |
| Nominal coolant flow / valve max | 101 / 288 L/min (35%) | 43 / 400 L/min |
| **β at valve saturation** (α = 1) | **0.39** (worst inlet corner 0.51) | 0.79 (worst inlet > 1.2) |
| PI controller | Kp = 50 (L/min)/K, τI = 8 min | Kp = 150, τI = 10 |
| Sensors | C 0.03 mol/L (1% of a 3 mol/L span), T, Tc 0.2 K, Qc 1.4 L/min (0.5% of span); Ti, Tci measured (offset 0.1 K) | C 0.01, T 0.2, Tc 0.2, Qc 2 |

Physically: catalyst activity α is the acid-catalyst (H₂SO₄) strength, which drops through
neutralisation by feed impurities. Fouling β is coolant-side scaling. Both are realistic
degradation mechanisms for this process.

## 2. Scorecard (criteria fixed before computing)

| Criterion | Target | R3 | R5 (UA 2.5×) | Fogler as published (R1) | System I |
|---|---|---|---|---|---|
| C1 unique closed-loop SS (17×17 grid × 17 inlet corners) | all | **pass** (0 multiple) | pass | fail (155 multiple, saturated) | pass |
| C2 every SS T ≤ 355 K | yes | **pass** (max 343 K) | pass (340 K) | pass (353 K) | pass |
| C3 stable, slowest τ ≤ 15 min (grid + corners) | yes | **pass** (11.9 min; min damping 0.14) | pass (14.5 min, Kp 40/τI 10; damping 0.23) | — | — |
| C4 Qc ≤ 40% of max; β_sat ≤ 0.55; no low-side saturation | yes | **pass** (35%; 0.39; none) | pass (35%; 0.32) | fail (β_sat 0.69) | fail (β_sat 0.79) |
| C5 conversion 70–90% | yes | **pass** (84%) | pass | pass | fail (98%) |
| C6 Bayesian CRB at nominal: sd_α ≤ 0.03, sd_β ≤ 0.02 | yes | **pass**: 0.007 / 0.004 | pass: 0.007 / 0.004 | — | fail: 0.052 / 0.032 |
| C7 inlet-error effect ≤ 1 × C6 sd_β | yes | **fail** (see §4) | fail | fail | fail |
| C8 proxy rule accuracy ≥ 0.95 at ≥ 3 sd; cross-sensitivity ≤ 0.10 | yes | **pass** (1.00; ≤ 0.013) | pass | — | — |
| C9 summary robustness (descriptive) | — | P1b vs P2 shift ≤ 0.08 sd; no clipping | same | — | — |

Two variants were dropped at the steady-state stage. A 1.5× jacket had multiple steady states when
saturated. Preheating the feed to 310 K raised the jacket share to 55% but saturated the valve at
β = 0.59.

**R3 vs R5:** both pass everything except C7, with the same information. R3 has the more
plausible heat-transfer area (2× Fogler's, achievable with jacket plus coil) and larger controller
margin. R5 has more saturation margin and stays unique down to β = 0.3. The pre-set tie-break
(largest minimum margin over C3–C8) slightly favours R3 at the chosen tunings, and realism favours
it too.

## 3. Information and the proxies (why this plant succeeds)

**Information limits** at nominal (34 production-type features: 29 summaries, 3 plug-in
estimators with the measured inlet, 2 measured inlet temperatures):

| Nuisances marginalised | sd_α | sd_β | Corr(α,β) |
|---|---|---|---|
| Inlet known exactly (oracle) | 0.006 | 0.003 | — |
| Inlet offsets 0.1 K, Ci 1%, Q 0.5% (C6) | 0.007 | 0.004 | 0.17 |
| **Plus every sensor calibration offset**: T (in-loop) 0.1 K, Tc 0.1 K, coolant flowmeter 1%, analyser 1% | **0.014** | **0.016** | 0.42 |
| System I, same full nuisance set | 0.053 | 0.047 | 0.17 |

With every realistic calibration offset treated as unknown, R3 is 3–4× more precise than
System I. The remaining uncertainty is set by instrument calibration (mainly the coolant
flowmeter and the jacket-outlet sensor), not by noise. That is the honest, realistic picture and
the right baseline for the paper.

**The two physics proxies separate each fault category** (400 windows per point, full realistic
variation: inlet ±2 K measured with 0.1 K offset, Ci ±1%, Q ±0.5%):

| Point (α, β) | Class | Plain 0.90-threshold rule on (α̂, β̂_jac) | α̂ mean ± sd | β̂_jac mean ± sd |
|---|---|---|---|---|
| (1.05, 1.05) | healthy | 1.00 | 1.050 ± 0.015 | 1.051 ± 0.004 |
| (1.05, 0.70) | fouling | 1.00 | 1.052 ± 0.014 | 0.701 ± 0.003 |
| (0.70, 1.05) | decay | 1.00 | 0.700 ± 0.011 | 1.051 ± 0.005 |
| (0.75, 0.75) | combined | 1.00 | 0.750 ± 0.011 | 0.751 ± 0.003 |
| (1.00, 0.88) | near-threshold fouling | 1.00 | 1.000 ± 0.015 | 0.881 ± 0.004 |
| (0.88, 1.00) | near-threshold decay | 0.955 | 0.879 ± 0.013 | 1.001 ± 0.004 |
| (1.00, 0.45) | deep fouling, 5% saturated | 1.00 | 1.001 ± 0.015 | 0.451 ± 0.003 |

- **Unbiased and orthogonal:** each proxy is unbiased everywhere. Their cross-sensitivities are
  |dα̂/dβ| ≤ 0.013 and |dβ̂/dα| ≤ 0.004 over the unsaturated box.
- **Proxy choice: replace the legacy `UA_eff_proxy`.** The legacy (T−Tc)/Qc is confounded with
  catalyst activity: its sensitivity to α is 0.17–1.2× its sensitivity to β. It depends on the
  jacket duty, which α changes. The jacket-balance estimate β̂_jac = ρcCpc·Qc·(Tc−Tci) / (UA·(T−Tc))
  removes the duty and is α-free. The legacy `k0_eff_proxy` is fine (cross ≤ 0.012) and is
  equivalent to α̂ up to the measured temperature. So the two physics proxies should be ln α̂ and
  ln β̂_jac.
- **Summary robustness (C9):**
  - the summaries hardly depend on the warm start (largest shift P1b vs P2: 0.08 sd);
  - no feature clips;
  - only the two saturation fractions are constant at nominal (non-zero only near saturation);
  - the covariance is well conditioned enough for the information computation.

## 4. What did not pass, and why (C7)

C7 asked that an unmeasured inlet error of 0.5 K move the β estimates by at most one posterior
sd. It fails for every candidate, System I included. For R3:
- Tci + 0.5 K moves the jacket estimate by 1.5%; the C6 sd is 0.4% (3.7 sd), the full-nuisance
  sd 1.6% (0.9 sd);
- Ti + 0.5 K moves the reactor-balance estimate by 2.5%.

These are 10× (Tci) and 5× (Ti) smaller effects than in System I. They still exceed the C6 sd,
because the plant is now so informative that small systematic errors are many standard deviations.

Two consequences:
- **In production** the inlet is measured, at 0.1 K, so the effect is 0.3% (< 1 sd): the
  production model is not affected.
- **For the paper this is the fragility mechanism in its cleanest form:** the better the plant and
  instruments, the smaller the systematic error that breaks calibration. That is a stronger
  version of the silent-failure story than System I gave.

I am reporting C7 as failed, not amending it.

## 5. Things to know before building

- **Open-loop operation is multistable,** which is why Fogler teaches multiplicity with this
  reactor. At the nominal coolant flow, 13% of the grid has 3 open-loop steady states, and
  decay (α = 0.7) at fixed coolant flow extinguishes to 302 K. Closed loop is unique everywhere.
  For Claim 1, the open-loop scenarios must be restricted to points with a unique open-loop state,
  e.g. healthy and fouling (α = 1, β = 0.7: unique, 340 K). Better still, the fact that this
  reactor *cannot* be operated open loop is the strongest motivation for Claim 1.
- **Prior box:** α, β ∈ [0.4, 1.2]. Below β ≈ 0.3 the saturated reactor becomes multistable;
  above 0.4 it is unique.
- **The warm start barely matters here.** The slowest closed-loop time constant is ~12 min against
  a 300-min history, so the P0/P1 artefacts of System I will be much weaker on R3. That is itself
  a design lesson (fast dynamics relative to the history). The fragility section should lean on
  sensor calibration, inlet assumptions and open-loop training, with the warm start shown as small.
- **Controller realism (implemented in `src/system_iii/simulator.py`):** the loop acts on the *measured*
  temperature, as a DCS does: digital PI executed every 1.2 s with a 0.2-min PV filter, sensor noise
  and calibration offset in the loop. A control-sensor offset therefore moves the plant (Sc10). A slower
  execution (0.5 min) with a 1-min filter made the loop noise-driven at the prior corners, so the DCS
  rate matters. Verified over the corners: true-T sd 0.12–0.24 K, no mean bias; Qc sd 6–13 L/min.
- **The package reproduces the design numbers** with this controller (`design/verify_package.json`):
  Bayesian CRB with all calibration offsets 0.014 (α), 0.016 (β); proxy-rule accuracy 1.00 on
  Sc1–Sc4 and the saturated case, 0.985 and 0.955 near the thresholds (S3; old numbering), under the full realistic variation.
- **First signs for the fragility section** (notebook 02 dry run): a +0.5 K control-sensor offset
  biases both proxies by about 4% (α̂ 0.96, β̂ 0.97 at a healthy plant), roughly 3 posterior sd, while
  the logged temperature stays at setpoint, so the residual detector cannot see it. A +2 K monitoring
  drift biases them much more (α̂ 0.85, β̂ 0.88) but is visible in the residual.
- **Masking confirmed on the information side** (notebook 04 dry run): with temperature-only features
  the β bound is ~3 (no information); with C and T but no coolant side it is 0.40; with the full set
  0.004 (inlet offsets only).
- **Scenario proposal** (all unsaturated and unique unless stated; 3 sd ≈ 0.05):

**Superseded 2026-11-10 (decision 14)** by the catalogue aligned with System I's numbering:

| Scenario | (α, β) | Notes |
|---|---|---|
| Sc0 open-loop healthy | (1.0, 1.0) | Qc fixed at nominal; unique SS 337 K |
| Sc1 healthy | (1.0, 1.0) | Qc 101 L/min |
| Sc2 fouling | (1.0, 0.70) | Qc 123 |
| Sc3 decay | (0.70, 1.0) | Qc 76, X 79% |
| Sc4 combined | (0.75, 0.75) | Qc 91 |
| Sc5 severe fouling | (1.0, 0.50) | Qc 177, unsaturated (clamp 0.8% of the time under inlet variation); System I's Sc5 β, saturated there |
| Sc6 open-loop fouling | (1.0, 0.70) | open-loop twin of Sc2; unique SS 340 K |
| Sc7 fouling + drift | (1.0, 0.70) | Sc2 + 2 K on the logged T, outside the loop (as System I's Sc7) |
| Sc8 near-threshold fouling | (1.0, 0.87) | ≈ 1.9 sd (S5) |
| Sc9 near-threshold decay | (0.87, 1.0) | ≈ 1.5 sd (S5) |
| Sc10 control-sensor offset | (1.0, 1.0) | +0.5 K on the control T sensor, in the loop |

Valve saturation is no longer a catalogue scenario: inside the prior box it exists only at β ≈ 0.40–0.44 for
α > 1, i.e. at the prior edge. It is tested in notebook 12 (α = 1.2, β = 0.41).

- **System II:** a better reactor fixes the reactor-internal pair (α, β_r), which is well
  separated here by orthogonal proxies. It cannot fix System II's cross-unit confounds (α vs feed
  composition, column efficiency). Those need a composition measurement on the fresh feed or
  recycle, i.e. the analyser rework already planned. If System II is rebuilt around this reactor,
  require that analyser.

## 6. Claim 1, revised

The System I form of Claim 1 (an open-loop-trained posterior misreads closed-loop data) is a foregone
conclusion here, because this reactor cannot be operated open loop (multistable at a fixed coolant
flow; decay extinguishes it). The revised claim, tested in notebook 08:

1. **Masking (motivation).** With T held at setpoint, the fault information is in Qc and Tc; a
   temperature-only monitor carries none of the β information.
2. **What must be modelled.** A controller-aware simulator (production) against a controller-agnostic
   simulator driven by an exogenous coolant-flow trajectory, the posterior conditioning on the observed Qc.
   If the agnostic version matches, the statement is "model the loop or condition on its manipulated
   variable; never assume the open-loop operating point". If it does not, that is the result.
3. **Controller mis-specification** (retuned loop, control-sensor offset) moves to the fragility section
   (notebook 12).
4. The old open-loop-trained comparison is kept as one arm, briefly.

## 7. Decisions for the user

1. R3 (recommended) or R5.
2. Accept C7 as failed and use it as the fragility mechanism (recommended), rather than amend it.
3. The two physics proxies become ln α̂ (component balance) and ln β̂_jac (jacket balance); the
   legacy `UA_eff_proxy` is dropped (α-confounded).
4. Controller acting on the measured temperature (recommended) or the true temperature (as
   System I).
5. Scenario set as proposed in §5.
6. Then: new folder and branch, and the pipeline (protocol study, summary study, budget study with
   rule v2.1, production, stress tests).
