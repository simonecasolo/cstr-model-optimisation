# System III — paper plan, claims ledger and notebook map (2026-11-10)

System III is the **main system** of the paper. System I is the **marginal-design contrast**: the same
chemistry and method on a plant that was not designed for diagnosability, used to show what the
methodology can and cannot do when the plant (or the simulator's assumptions) is poor.

## 1. Storyline

**Message.** Simulation-based inference gives calibrated, near-optimal fault diagnosis for a
closed-loop process *designed for diagnosability*. The posterior inherits every simulator assumption,
though, and the better the plant and its instruments, the smaller the systematic error that silently
breaks calibration. What separates a diagnosable plant from a marginal one can be computed before any
training, from steady-state physics and the Fisher information.

**Structure**
1. **Motivation, controller masking (C-M).** With T held at setpoint, the fault information moves into
   the coolant flow and jacket temperature. T-only monitoring sees almost nothing.
2. **Design for diagnosability (D).** The scorecard (fixed before computing), System III against
   System I. Information limits, physics proxies, unique steady states, cooling margin.
3. **SBI succeeds on System III.** Calibration (SBC + TARP), precision at the information limit,
   three-way classification, robustness to the realistic inlet range (measured inlet), baselines
   (EKF/UKF), tracking.
4. **Claim 1, revised: what must be modelled about the loop.** Controller-aware simulator versus a
   controller-agnostic simulator driven by an exogenous coolant-flow trajectory. The old form
   (open-loop-trained model) is shown briefly: this reactor cannot even run open loop.
5. **Fragility: silent failures on a good plant.**
   - unmeasured inlet assumed nominal;
   - calibration offsets beyond those assumed;
   - sensor-model mismatch (S0–S4 ladder);
   - controller retuned after training;
   - control-sensor offset (in the loop) versus monitoring-sensor drift (outside it);
   - warm start (expected weak here; why).

   For each: the size of the error that breaks coverage, whether it is detectable (leakage,
   proxy disagreement, residual on the controlled variable), and the remedy.
6. **Contrast with System I.** The marginal design fails at much larger errors *and* at nominal
   conditions. The same diagnostics flagged it.
7. **Practitioner checklist and instrument requirement.**

## 2. Claims ledger (to be filled as notebooks run)

| ID | Claim | Notebook | Pre-registered test | Status |
|---|---|---|---|---|
| D1 | System III passes the diagnosability scorecard; System I fails it | 00 | SCORECARD.md criteria | design done (C7 failed, reported) |
| C-M | Controller masking: T-only features carry ≲10% of the β information of the full set | 01, 04 | CRB ratio | pending |
| C2 | Production posterior calibrated (interior SBC KS α, β ≥ 0.01; TARP ≥ 0.01) | 06, 07 | nb31 rule v2.1 | pending |
| C3 | Posterior sd within 1.2× the Bayesian CRB of the production features | 04, 07 | ratio per scenario | pending |
| C4 | Three-way classification: ≥ 0.95 correct on Sc1–Sc4; near-threshold cases mostly "undetermined", not wrong | 07 | per-scenario counts | pending |
| C5 | Robust over ±2 K inlet (measured, 0.1 K offsets): β coverage ≥ 0.85 everywhere | 07, 11 | nb35 criteria | pending |
| C1r | Revised Claim 1: controller-aware vs controller-agnostic simulator | 08 | coverage / sd / bias comparison, rule fixed in nb08 | pending |
| F1 | Unmeasured inlet assumed nominal → silent failure at errors of ~0.2–0.5 K | 11 | coverage vs error size | pending |
| F2 | Calibration offsets beyond the assumed sd → coverage loss scaling with offset / CRB | 12 | sweep | pending |
| F3 | Sensor-model mismatch (S0–S4): artefact correlations; α needs C | 10 | ladder | pending |
| F4 | Controller retuned after training: effect on posterior | 12 | Kp × {0.5, 2}, τI × {0.5, 2} | pending |
| F5 | Control-sensor offset (in loop) vs monitoring drift (outside): which is detectable how | 12 | residual detector + proxies | pending |
| F6 | Warm start matters little on a fast plant (contrast with System I) | 05 | protocol matrix | pending |
| I1 | Instrument requirement for the inlet (ε) and calibration | 04, 11 | ε sweep | pending |
| B1 | SBI vs EKF/UKF (+ offset states): accuracy similar, SBI calibrated | 09 | coverage, RMSE, tails | pending |
| T1 | 30-day tracking with drifting coolant supply and persistent offsets | 14 | coverage, delay | pending |
| V1 | Individual-posterior validity (L-C2ST, standardised variant) | 13 | p-values, power check | pending |
| X1 | System I contrast: same scorecard, same stress tests | 16 | side by side | pending |
| P1 | Prior choice: headline results (Sc1–Sc4) robust to the prior when it contains the truth; sensitive near thresholds, at box edges, under masked features; an excluding prior fails silently unless leakage flags it | 19 | mean shift ≤ 0.5 sd, sd ratio 0.8–1.25, decisions unchanged ≥ 95% | pending |

## 3. Notebook map

The structure mirrors System I. Every notebook header states its rule; **rules are fixed before the
notebook is run**, and amendments are dated, logged, and only for logical flaws.

| # | Notebook | Content | Needs | Status of the code |
|---|---|---|---|---|
| 00 | design_scorecard | re-run the design scorecard and the package verification | numpy | complete |
| 01 | model_demonstration | trajectories per scenario, masking figure, open-loop multistability, offset vs drift | numpy | complete |
| 02 | data_generation | observation sets: D-prod (production conditions), D-nom (nominal inlet, no offsets), L-sets (1000 per scenario) | numpy | complete |
| 03 | physics_proxies_and_summaries | proxy separation, legacy proxy confound, cross-sensitivity, robustness; summary-set selection (S0 / S1m / S1m_no_legacy, 20k × 3 seeds) | torch/sbi | complete (training cells) |
| 04 | information_limits | Bayesian CRB by nuisance set, instrument breakdown, T-only vs full (masking) | numpy | complete |
| 05 | warm_start_protocols | P0/P1/P1b/P2 train × test matrix, 20k × 3 seeds, System I rule | torch/sbi | complete |
| 06 | budget_and_production | budgets 5k–160k × 5 seeds, rule v2.1, production record | torch/sbi | complete |
| 07 | calibration_and_classification | production on L-sets: SBC, TARP, coverage, three-way classification, CRB ratio | torch/sbi | complete |
| 08 | claim1_controller_modelling | controller-aware vs controller-agnostic (exogenous Qc) vs open-loop-trained | torch/sbi | complete (rule to fix) |
| 09 | baselines_ekf_ukf | EKF/UKF with measured inlet + offset states, tuned on a separate set | numpy | **to build** (spec in notebook) |
| 10 | fragility_sensor_model | S0–S4 ladder, 40k × 3 seeds each | torch/sbi | complete |
| 11 | fragility_inlet | naive (nominal-inlet) vs production under inlet errors; proxy-disagreement alarm; ε sweep | torch/sbi | complete |
| 12 | fragility_controller_and_sensors | retuned controller, control-sensor offset, monitoring drift + residual detector, offsets beyond assumed | torch/sbi | complete |
| 13 | individual_posterior_checks | L-C2ST (standardised), power check | torch/sbi | **to build** (spec in notebook) |
| 14 | tracking_30day | 30-day stream, drifting coolant supply, persistent offsets | numpy + posterior | **to build** (spec in notebook) |
| 15 | cost_and_timing | simulation, training, inference cost; break-even vs EKF/UKF | all | complete (after 06, 09) |
| 16 | system_I_contrast | both scorecards, System I results side by side | System I results | complete (reads files) |
| 17 | claims_and_conclusions | dashboard from results/ | all | complete |
| 18 | si_figures | SI figures from stored results only (never writes data/) | all | complete |
| 19 | prior_sensitivity | U-box vs U-narrow / U-wide (retrained) and H, D, LogU (reweighted); excluding prior + leakage; T-only prior dominance; population ≠ prior | torch/sbi | complete (rule to fix) |

## 4. Run order (when the user starts)

1. tests: `python -m pytest system_III/tests -q`
2. 00 → 01 → 02 → 04 (numpy only, ~30 min in total)
3. 03 (summary-set decision) → 05 (protocol decision) → 06 (production, ~2–4 h) → 07
4. 08, 10, 11, 12, 19 (each 1–3 h) in any order, one at a time (19 reuses 08's T-only members if present)
5. 09, 13, 14 once built; then 15, 16, 17, 18

Simulation is numpy: about 5 s per 10,000 windows on one core, so a 160k bank takes about a minute.
Training dominates (≈ 1–6 min per posterior).

## 5. Pre-registration checklist (fill in each notebook header before running)

- **Summary set (03):** System I nb03 rule (efficiency vs CRB, calibration gate), sets S0, S1m,
  S1m_no_legacy.
- **Protocol (05):** System I nb29 rule; production default P2 unless the rule says otherwise.
- **Budget (06):** nb31 rule v2.1, unchanged.
- **Claim 1 revised (08):** controller-agnostic simulator design (exogenous Qc level distribution and
  fluctuation) and the comparison metric. To be fixed by the user before running.
- **Fragility (10–12):** error grids and "silent failure" definition (coverage < 0.5 with
  leakage < 1% and sd within 20% of nominal).
- **Prior sensitivity (19):** prior set, P1 thresholds, ESS floor (200/2000), leakage flag (0.2).
- **Detectors:** thresholds fixed on separate calibration sets, evaluated on independent windows.
