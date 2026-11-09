# System R — a CSTR designed for diagnosability (redesign plan)

*Written 2026-11-09, at the end of the System I N3 re-run (notebooks 29–36). To be picked up after
System I Phase 3. Nothing here is built yet.*

## 1. Why a redesign

**Intended paper message (user):** show a system on which SBI-based condition monitoring
*succeeds* — calibrated, near the information limit, high diagnostic accuracy, robust to realistic
operating variation — and then show how easily it *fails silently* when conditions are slightly
different (warm start, sensor model, inlet conditions, training regime).

**Why System I (Fogler PO hydrolysis, current operating point) cannot carry the "succeeds" half.**
It is marginal by design, independently of the inference method:

| Property | System I | Consequence (measured) |
|---|---|---|
| Share of the reaction heat removed by the jacket | ~19% (the feed absorbs ~81%) | β is a small term in the energy balance; every inlet error is amplified |
| Coolant temperature rise in the jacket | ~3 K | 0.5 K error on the coolant inlet Tci → jacket duty off by 15–45% → β biased by 7–13 posterior sd (nb34) |
| Feed-temperature sensitivity | 1 K error in Ti ≈ 27% of the jacket duty | idem for Ti |
| Cooling margin | jacket capacity limit at β ≈ 0.775; +1 K feed error saturates the valve for 27% of healthy windows | small disturbances push the plant into saturation, which the posterior reads as fouling |
| Conversion | ~98% (Da ≈ 54) | α is barely visible in C; without C α is only weakly constrained (nb33 N4) |
| Scenario placement | Sc2 and Sc4 at β = 0.85, 0.05 below the 0.90 class threshold | any honest widening of the posterior costs fouling-class accuracy (nb36: 0.86–0.92 vs 0.99 for the oracle) |

Result: the honest production model (measured inlet, ε = 0.1 K, nb36) is calibrated and robust,
but its β sd is ~0.044 and its macro-F1 0.81–0.83. A good *cautionary* system; not a success story.

## 2. The reframe: diagnosability by design

1. **A plant designed for diagnosability succeeds** (System R).
2. **The same method fails silently when the simulator's assumptions are slightly wrong** — shown
   on System R itself (warm start, sensor model, inlet errors, open-loop training), and, briefly, on
   the marginal design (System I) as a contrast: same chemistry, different design, different
   diagnosability.
3. **What separates good from marginal designs is computable before any training**: a
   diagnosability scorecard from steady-state physics and the Fisher information. This is a
   transferable design rule and a contribution in itself (the information limit used as a plant and
   instrument design criterion for monitoring).

Paper structure (user preference, 2026-11-09): **one paper on the robust plant, with the
fragility as a section**; System I appears as the marginal-design contrast (one section or the SI).
Active excitation (setpoint dither) is out of scope for now.

## 3. Design principles

1. **The jacket carries most of the heat**: ≥ 60–70% of the reaction heat removed by the jacket
   (feed pre-heated closer to the reactor temperature, or more heat release per volume).
2. **Large coolant temperature rise**: Tc − Tci ≥ 8–10 K at the nominal point (lower nominal
   coolant flow, chilled coolant, UA sized accordingly).
3. **Cooling margin**: nominal coolant flow at 30–40% of the valve maximum; saturation only at
   severe fouling (onset β ≈ 0.5–0.6), kept as one deliberate scenario.
4. **Moderate conversion** (70–90%): both C and the heat release carry α.
5. **Unique steady state** over the whole prior box and all scenario inlet conditions (no
   multiplicity, no extinction boundary in reach of the fault range).
6. **Moderate adiabatic temperature rise** (no runaway-prone regime).
7. **Scenarios well separated from the class thresholds** (≥ ~3 posterior sd), plus one deliberate
   near-threshold scenario to show the limit honestly.
8. **Honest classification**: three-way decision (healthy / faulty / undetermined, from posterior
   probabilities) instead of forced four-class labels.
9. **Measured inputs from day one**: Ti, Tci (ε ≤ 0.1 K, from nb35), feed flow; C online (needed for α).

## 4. Candidate designs

| Candidate | Description | Known issues |
|---|---|---|
| **R1: redesigned Fogler** (preferred starting point) | same PO-hydrolysis kinetics; new operating point and jacket: Ti, Tci, coolant sizing, UA, τ (residence time), Tsp chosen by the scorecard | needs the design search (section 5) |
| R2: Pilario & Cao (2018) as-is | the published benchmark in `notebooks/system_I/01a_model_demonstration_pilario.ipynb` | jacket share ~55% and coolant rise ~64 K are good, but: multiple steady states (extinction near Qc ≈ 165 L/min with Qc_max 200), nominal Qc at ~74% of max, adiabatic rise 194 K, chemistry not physical, controller constants reconstructed |
| R3: Pilario modified | R2 with a larger valve and an operating point away from the multiplicity boundary | loses the "recognised benchmark" advantage partly |
| Reference | System I as is | the marginal design, scored for contrast |

Rough scorecard of what exists (from the constants in the code; to be recomputed properly in D0):

| Property | System I | Pilario (R2) | Target |
|---|---|---|---|
| Jacket share of reaction heat | ~19% | ~55% | ≥ 60% |
| Coolant temperature rise | ~3 K | ~64 K | ≥ 8–10 K |
| Jacket-duty error per 1 K Ti error | ~27% | ~1% | ≤ ~5% |
| Jacket-duty error per 0.5 K Tci error | 15–45% | ~1% | ≤ ~5% |
| Conversion | ~98% | ~89% | 70–90% |
| Nominal Qc / Qc_max | ~13% (capacity-limited at β ≈ 0.78) | ~74% | ≤ 40% |
| Steady states | unique | multiple | unique |
| Adiabatic rise | 20 K | 194 K | moderate |

## 5. Procedure

### D0 — design notebook (cheap: steady states and Fisher information, no neural networks)
1. **Fix the scorecard criteria before computing anything** (the targets of sections 3–4, written
   into the notebook header and the decision log, as for every System I study).
2. **Grid over the design variables** (R1): Ti, Tci, nominal coolant flow and valve maximum, UA,
   residence time (or V, Q), Tsp; plus R2/R3 and System I as fixed points.
3. **Per design compute:** heat split; coolant rise; valve margin and saturation-onset β; conversion;
   adiabatic rise; steady-state uniqueness over the prior box and the inlet range (explicit
   multiplicity check); Cramér–Rao sd of α and β under N3 sensors with the measured inlet (ε = 0.1 K)
   for the 29 summaries + plug-in estimators; the bias of the plug-in β estimates per 0.5 K Tci and
   1 K Ti error, in units of that sd; controller tuning (stable PI, reasonable settling vs the
   60-min window).
4. **Pick the design** by the pre-set criteria (tie-break: largest margin to the worst criterion);
   then place the scenarios (fault magnitudes ≥ 3 CRB sd from the class thresholds, one near-threshold
   case, one saturation case, one sensor-drift case).

### Then the pipeline (System I infrastructure, with the lessons as defaults)
* Sensors **N3-type** absolute noise from the start; warm start **P2** (fault ramp history);
  controller law verified against the standard PI form (the System I bug).
* **Measured inlet temperatures** as model inputs (ε = 0.1 K; nb35/nb36 feature construction);
  inlet varying over the training prior.
* Features: 29 summaries + physics-based plug-in estimators (S1-type, recomputed with the measured
  inlet); re-test the summary set with the nb03 rule.
* Budget and deployed object: nb31 rule **v2.1** (persistent convergence, precision criterion, the
  deployed posterior must itself pass calibration); 5-seed ensembles.
* Validation: SBC (interior) + TARP + posterior sd vs the CRB of the production features.
* **Success results**: calibration, precision vs bound, three-way classification, Claim 1 (closed
  vs open-loop training), robustness over the realistic inlet range.
* **Fragility section** on System R: warm start (P0/P1), sensor model (N0–N4 ladder), inlet errors
  with the inlet *assumed* nominal (nb34-type), open-loop training — each with how it would be
  detected (leakage, estimator disagreement, coverage).
* **Contrast**: the same scorecard and two or three headline numbers for System I.
* Baselines: EKF/UKF with the measured inlet as a known input; NUTS on a few windows.

## 6. What to reuse from System I

* Scripts: `ws_protocols.py` (protocol simulator; generalise the physics import), `ws_inlet.py`
  (per-sample inlet), `ws_eval.py` (simulate/features/evaluate), `ws_metrics.py`, `summary_variants.py`,
  `nb29_ws_study.py` (protocol study), `nb31_budget_convergence.py` (rule v2.1), `nb33_sensor_noise.py`,
  `nb34`/`nb35`/`nb36` (inlet studies), `system_I_config.py` (sensor model / protocol / summary-set
  switches), the notebook builders and tests.
* Process: rules fixed before results, dated amendments only for logical flaws (v2, v2.1, v1.1),
  decision log, runbook + autonomous agent with stop conditions and standing permissions,
  claims ledger, storyline.
* Lessons that become defaults: realistic absolute sensor noise; noisy warm start with fault history;
  measured inlet; physics-based features; ensemble deployment; check new tests against physics
  (e.g. saturation) before trusting their premise; fresh large test sets for headline accuracy.

## 7. Open decisions (when this is picked up)

1. **Folder**: inside the current project (`research_project/system_R/`, reuses `src/` and
   `scripts/` directly; Claude can work there immediately) or a sibling repository
   (`cstr-model-optimisation/robust_cstr`, cleaner; needs folder access granted).
2. R1 only, or R1 and R3 scored side by side in D0 (recommended: score both; the data decides).
3. Whether System I's role is a full contrast section or SI only.
4. Scope of baselines and of the tracking study for System R.
