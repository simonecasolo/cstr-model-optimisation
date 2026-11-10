# Realism upgrade: sensor dynamics and control structure (2026-11-10)

**Question (user):** is System III industrially realistic? The most optimistic assumption was the concentration
measurement (a reading every 30 s, no delay); the temperature sensors had no dynamics.

## 1. What was added (code)

- **Sensor set S5** (`sensors.py`, `SENSOR_DYN`): S3 noise plus
  - online PO analyser (GC/NIR class): a new value every **5 min**, from a sample taken **2 min** earlier,
    held between updates; random sampling phase per window (≈ 12 distinct C readings per 60-min window);
  - first-order **thermowells**, τ = **0.5 min**, on the reactor T and jacket-outlet Tc; the controller sees
    the lagged reactor reading.
- **Cascade control** (`simulator.py`, `control="cascade"`): outer PI (measured reactor T, PV filter 0.1 min)
  sets the jacket-outlet temperature setpoint; inner PI (measured jacket-outlet Tc, filter 0.05 min) moves
  the coolant valve. Tuning (Ko, τo, Ki, τi) = (4 K/K, 8 min, 20 (L/min)/K, 1 min). Anti-windup on both loops.
- `information.bayes_crb(..., **sim_kw)` passes simulator options (control, cascade, filters).
- Static sets S0–S4 and `control="closed"` are unchanged; all 9 tests pass.

## 2. What the checks found

Scripts and results: `design/verify_s5.py|json`, `design/check_ctrl_s5.py|json`, `design/tune_digital.py`
(now takes the sensor set), `design/check_cascade.py`, `design/sim_ctrl_s5.py|json`, `design/verify_s5_cascade.py|json`.

### 2.1 The single PI loop does not survive the thermowell (and C3 was already marginal)

Linearised loop over the 17 × 17 prior grid × 5 inlet corners (C3: stable, slowest time constant ≤ 15 min):

| Configuration | slowest τ (min) | min damping | stable everywhere |
|---|---|---|---|
| as built: PI 50/8, PV filter 0.2, no thermowell | **26.5** (15.9 at nominal inlet) | 0.05 | yes |
| PI 50/8, filter 0.2, thermowell 0.5 | — | < 0 | **no** (limit cycle at low α, low β) |
| best single PI found, thermowell 0.5 (20/12, filter 0.1) | 88 | 0.03 | yes |
| best single PI, faster thermowell 0.25 (30/12, filter 0.05) | 22.5 | 0.09 | yes |
| **cascade 4/8/20/1, thermowell 0.5, filter 0.1** | **14.2** | **0.17** | **yes** |

**Disclosure.** The design-stage C3 check (design/check_ctrl.py) used a continuous PI on the true T. The PV filter
added afterwards (decision 6) already breaks C3 at the low-α, low-β corner and the inlet corners (8 of 1415 grid
points, τ up to 26.5 min). I had not re-run C3 after that change. The thermowell makes it worse.

**Why no single PI works.** The process gain changes by about 6× across the box: at high coolant flow
(β ≈ 0.4–0.75) the loop needs a high gain; at low flow (α, β ≈ 0.4, Qc ≈ 40 L/min, slow jacket) the lag makes a
high gain oscillate. A cascade on the jacket temperature is the standard industrial remedy (the inner loop
linearises the valve and jacket), and it fixes both ends.

Simulation (S5, P2, inlet nominal and ±2 K corners, unsaturated points): true-T sd ≤ 0.22 K (0.13 at
nominal), bias ≤ 0.01 K, valve movement sd 9.5 L/min at nominal (as built: 7.4).

### 2.2 Information (C6) with S5 + cascade

| | sd_α | sd_β | corr |
|---|---|---|---|
| S3, PI (as built), inlet offsets only | 0.0074 | 0.0040 | 0.16 |
| S3, PI (as built), all offsets | 0.0142 | 0.0159 | 0.41 |
| **S5, cascade, inlet offsets only** | **0.0183** | **0.0054** | 0.12 |
| **S5, cascade, all offsets** | **0.0218** | **0.0164** | 0.31 |
| S5, cascade, T-only features (masking) | 15.1 | 0.93 | 0.79 |

- **C6 still passes** (α ≤ 0.03, β ≤ 0.02). The analyser costs α about 2.5× (inlet-only) or 1.5× (all offsets);
  β is almost unchanged (it comes from the temperatures and the flow).
- **Masking persists:** a temperature-only monitor gets β sd 0.93 (vs 0.016 with the full set).
- S5 with the as-built PI gives similar numbers (α 0.019/0.022, β 0.005/0.016), but that loop is not acceptable (§2.1).

### 2.3 Proxy separation (C8) with S5 + cascade

*(Old scenario numbering; decision 14 renamed them: old Sc5/Sc6 → Sc8/Sc9, old Sc7 saturated → notebook 12.)*

Plain 0.90 rule on α̂ and β̂_jac, 400 production windows each: Sc1–Sc4 and Sc7 accuracy **1.00**; Sc5
(β = 0.87) 0.99; **Sc6 (α = 0.87) 0.875**. With the new α sd (0.018–0.022) Sc6 is only ≈ 1.4–1.7 sd from the
threshold (Sc5: ≈ 1.9 sd). **Correction (same day):** Sc5 and Sc6 were never "≥ 3 sd" scenarios; they are the
deliberate near-threshold pair (`scenarios.py`, not in `GATED`), so C8 passes and nothing has to move. With S5
they sit at ≈ 1.5–2 sd, close to the scorecard's "1 sd near-threshold case"; their role is to show calibrated
uncertainty near the boundary, and their lower accuracy is the expected result.

### 2.4 Unaffected

C1, C2, C4, C5 are steady-state properties with integral action (T = Tsp) and do not depend on the sensor
dynamics or on the loop structure. C7 depends on the proxies' algebra, not the sensors (re-check optional).

## 3. Recommendation (to confirm)

1. Production sensors **S5** (analyser 5 min / 2 min dead time, thermowells 0.5 min).
2. Production control **cascade 4/8/20/1**, outer PV filter 0.1 min.
3. Keep Sc5/Sc6 at 0.87 (fixed before the sensor change; moving them after seeing their accuracy would be
   the post-hoc choice). Report their distance from the threshold in sd under S5.
4. Keep S3 + single PI as a fragility/sensitivity arm ("an idealised analyser overstates α precision by 2.5×").
5. Further realism (valve stiction, coloured noise, analyser drift) as fragility tests in notebooks 10–12, not baseline.

If confirmed: change `config.py` defaults (sensors S5, control cascade), update decision 6 and 9, the scorecard
results table, and `ss`/notebook text that quotes the S3 numbers.
