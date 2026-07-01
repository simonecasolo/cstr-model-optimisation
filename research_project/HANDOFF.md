# Session Handoff — Wu 2003 nb20 Execution and Model Diagnostics

**Date:** 2026-07-01  
**Session status at handoff:** nb20 has executed cleanly. nb21–nb28 not yet run.

---

## What was accomplished in this session

### 1. nb20 bugs fixed and executed

Two bugs prevented nb20 from running. Both were fixed in the notebook JSON.

#### Bug 1 — Missing `NOMINAL_INLET` import (Cell 2)

Cell 11 (pre-compilation step) calls:
```python
simulate_trajectory_explicit(NOMINAL_THETA, NOMINAL_INLET, NOMINAL_CTRL_SB, ...)
```
but `NOMINAL_INLET` was absent from the `from cstr_sbi.recycle.physics import (...)` block in Cell 2. Fixed by adding it to the import list.

#### Bug 2 — Syntax error in Cell 19 (banana preview)

`fig.suptitle(...)` contained a literal newline inside a regular string literal — Python SyntaxError. Fixed by merging to a single string with `\n` escape.

**nb20 now runs to completion in the project `.venv` kernel.** Use:
```bash
caffeinate -i .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
  notebooks/20_wu2003_model_verification.ipynb
```

---

## nb20 output summary

All 10 code cells passed, no exceptions.

| Cell | Result |
|------|--------|
| Kinetics check | k(342.26 K) = 0.3300 /h ✓ |
| QSS column (z_F=0.50, η=1.0) | x_D=0.942, x_B=0.020, D_frac=0.521 ✓ (assertions passed — see §3 below) |
| Scenario catalogue | 16 CL scenarios correct |
| S-A warm start | z_A=0.508, T_r=342.25 K |
| S-B warm start | z_A=0.510, T_r=342.26 K |
| W3 step test (α=0.65, S-B, 50 h) | F_R_norm: 1.005→1.065 (+6.0%) — snowball PASSED (threshold >3%) |
| W6 step test (β_r=0.60, S-B, 50 h) | Max T_r dev = 1.32 K, T_j drops 5.9 K — masking visible |
| S-A vs S-B W3 (x_D) | S-A: 0.9500 (controlled), S-B: 0.9922 (drifts up) |
| Banana preview (W12 vs W8, 2 h) | **See §2 below — unexpected result** |

---

## 2. Banana preview — unexpected result (OPEN ISSUE)

### What the notebook shows

Cell 19 compares two scenarios over a 2 h window under S-B:

| Scenario | Parameters | Final F_R_norm | Final x_D |
|----------|-----------|---------------|-----------|
| W1 healthy | α=1.0, η=1.0 | 1.005 | — |
| W12 snowball_compound | α=0.75, η=0.80 | **1.029** | 0.9525 ↑ |
| W8 col_eff_severe | α=1.0, η=0.65 | **1.275** | 0.9998 ↑ |

### What was expected

The paper narrative requires W12 and W8 to have **similar** F_R and Q_reb trajectories under S-B (the degeneracy that creates the banana posterior). The x_D signal should then **differ** between them (α decay → x_D ↑; η_col decay → x_D ↓), disambiguating them under S-A.

### What was found

1. **F_R signatures are very different**: W8 shows +27% F_R vs W12's +3%. Not confusable.  
2. **x_D for W8 = 0.9998** (nearly pure A in distillate). Under η_col degradation, x_D should *decrease* (worse separation). Getting 0.9998 contradicts the expected direction.

### Root cause hypothesis

The W8 result appears to reflect a model dynamics issue during the 2 h transient window, not a steady-state value. Starting from the S-B warm start (z_A ≈ 0.510), the column model with η_col=0.65 initially gives x_D=0.857. But something in the closed-loop dynamics drives z_A (or the actuator states) to a regime where the column output changes dramatically over the short 2 h window.

The column bisection behaviour is also relevant — see §3.

### Implications for the paper

The banana posterior narrative depends on (α, η_col) combinations being hard to distinguish under S-B. The specific comparison of W12 and W8 does **not** support this in its current form. Two possibilities:

a) The banana emerges across the **full prior** in a way that two specific scenarios don't capture. The SBI training (nb24/25) will test this directly.  
b) There is a genuine dynamics bug causing W8's F_R to spike over the 2 h window.

**Before running nb22 (which generates the full training dataset), investigate the W8 transient.** See §4 for recommended diagnostic steps.

---

## 3. Column bisection behaviour — z_F dependence

### Observed

At fixed η_col=1.0 and R=2.198 (nominal), `column_qss` returns:

| z_F | x_D | x_B (output) | x_B (bisect MB) |
|-----|-----|--------------|-----------------|
| 0.48 | 0.908 | 0.019 | 0.014 |
| 0.50 | 0.942 | 0.020 | 0.020 |
| 0.51 | 0.956 | 0.022 | 0.025 |
| 0.52 | 0.968 | 0.028 | 0.033 |
| 0.55 | 0.984 | 0.067 | 0.078 |
| 0.60 | 0.990 | 0.155 | 0.176 |
| 0.65 | 0.993 | 0.249 | 0.277 |

As z_F increases above ~0.51, **x_D shoots toward 1.0 and x_B becomes very large** (10–25% A in bottoms for severe α decay). The column model is returning extreme values for the snowball regime that are not physically realistic for a well-operated distillation column.

### Why this happens

The bisection solves for x_D using the material balance:
```
x_B_mb = (z_F − D_FRAC_NOM × x_D) / (1 − D_FRAC_NOM)
```
with `D_FRAC_NOM = 0.521` **fixed** regardless of z_F.

For z_F > 0.50, x_B_mb at physically reasonable x_D values (0.90–0.95) is large (0.03–0.09). The McCabe-Thiele stepping with good separation (α_eff=2.0) reaches x_reb ≈ 0.01–0.02 — always below x_B_mb. The bisection interprets this as "column over-separates for this D_frac split" and drives x_D toward the hi bound (0.9998).

At z_F=0.55 (α=0.65 snowball regime), there is no x_D in [0.51, 0.9998] where x_reb = x_B_mb for D_FRAC_NOM = 0.521, because the fixed D_FRAC_NOM is too LOW — physically, D_frac must increase to 0.57+ to maintain reasonable x_B when z_F rises.

### System self-consistency vs physical accuracy

Despite this, the ODE reaches a **self-consistent** steady state for W3 (α=0.65): F_R_norm = 1.065 at 50 h. The high x_D (≈0.983) at z_F=0.55 causes z_A_in to increase, which forces a specific steady-state z_A, and the system locks into a fixed point. The Δ(F_R) = +6% is plausible for severe catalyst decay, and the W3 assertions pass.

However, the x_D values in the snowball regime (z_F > 0.52) are not accurate representations of Wu's column physics. In Wu's system, proper Loop 2/3 action would adjust R and V to maintain x_D ≈ 0.95 and x_B ≈ 0.011 despite α decay.

### What this means for the SBI study

- **Training data is internally consistent** — the model generates (θ, x) pairs from the same simulator used for inference. SBI correctness does not require physical accuracy.  
- **Physical claims need care** — the paper should not claim the model reproduces Wu's exact column steady states for high-z_F scenarios.  
- **The W3 F_R snowball (+6%) is the key observable** and it does emerge correctly from the dynamics.  
- **x_D is diagnostic only** — it is not used in S-B SBI summaries. Its incorrect values in the snowball regime do not directly corrupt the S-B training data.  
- **S-A training data uses x_D** — the anomalous x_D values (0.983 instead of 0.95 in the snowball regime) will appear in S-A channel 3 and in the S-A SBI summaries. This may distort the S-A posterior.

### η_col sweep at nominal z_F (working correctly)

At z_F = 0.500 and 0.510, the bisection correctly captures η_col degradation:

| η_col | x_D (z_F=0.50) | x_B (z_F=0.50) |
|-------|----------------|----------------|
| 1.00  | 0.942 | 0.020 |
| 0.80  | 0.904 | 0.060 |
| 0.65  | 0.857 | 0.112 |
| 0.50  | 0.796 | 0.178 |

Column degradation (η_col↓) correctly gives x_D↓ and x_B↑ when z_F stays near nominal.

---

## 4. Known discrepancies (carried from previous session)

These were documented in `docs/wu2003_model_description.md` §7.3 and are unchanged:

| Quantity | Model (nom. R=2.198) | Wu Table 1 | Acceptable? |
|----------|---------------------|------------|-------------|
| x_D | 0.942 | 0.950 | Yes — S-A Loop 2 drives to 0.95 in closed-loop |
| x_B | 0.020 (S-B) | 0.0105 | Yes — S-B has no x_B composition control |

Under S-A with Loop 2 active: R_state is driven up above 2.198 so x_D → 0.95, which makes x_B → 0.010 ≈ x_B_nom. The discrepancy is a **S-B/open-loop phenomenon** only.

---

## 5. Recommended diagnostic steps before proceeding to nb21/nb22

### 5a. Diagnose W8 transient (URGENT before nb22)

Run this diagnostic in a notebook or script:

```python
import sys; sys.path.insert(0, '../src')
import numpy as np
from cstr_sbi.recycle.simulator import nominal_warm_start, deterministic_window
from cstr_sbi.recycle.scenarios import get_scenario
from cstr_sbi.recycle.physics import column_qss

y0_sb = nominal_warm_start("S-B")
sc_w8 = get_scenario("W8_col_eff_severe")

# Run 10h to see if F_R settles
t, raw = deterministic_window(sc_w8, structure="S-B", y0=y0_sb, t_final_h=10.0, n_save=100)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
axes[0].plot(t, raw[:, 0]);  axes[0].set_ylabel("T_r")
axes[1].plot(t, raw[:, 3]);  axes[1].set_ylabel("x_D")
axes[2].plot(t, raw[:, 7]);  axes[2].set_ylabel("F_R_norm")
axes[3].plot(t, raw[:, 10]); axes[3].set_ylabel("V_norm")
plt.tight_layout(); plt.savefig("W8_diagnostic.png")

print(f"R_state init:  {float(y0_sb[4]):.4f}")
print(f"V_norm init:   {float(y0_sb[5]):.4f}")
print(f"F_R_norm final: {raw[-1, 7]:.4f}")
print(f"x_D final:     {raw[-1, 3]:.4f}")
```

**Interpretation:** if F_R plateaus well below 1.3, the issue is a short-window transient. If it keeps rising, there is a positive-feedback runaway in the model for η_col=0.65.

### 5b. Verify W12 vs W1 distinguishability

Check whether W12 and **W2** (α=0.85, η=1.0) produce similar F_R and Q_reb trajectories. W2 is a better banana comparison partner than W8 since both W12 and W2 have moderate combined effects.

### 5c. Check if banana posterior emerges from the full prior

After nb24/25 (SBI training), generate posterior samples for W12 and plot the marginal over (α, η_col). If banana-shaped, the paper claim stands even if the W12 vs W8 comparison doesn't show it.

---

## 6. Execution plan (unchanged from previous session)

Run sequentially after nb20 is reviewed:

```bash
# nb21 — control structure comparison (S-A vs S-B for all 16 scenarios)
caffeinate -i .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3 \
  notebooks/21_wu2003_control_structures.ipynb && echo "nb21 done"

# nb22 — data generation (wait for W8 diagnostic first)
caffeinate -i .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
  notebooks/22_wu2003_data_generation.ipynb && echo "nb22 done"

# nb23 — summary statistics, PCA, t-SNE, MI
caffeinate -i .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3 \
  notebooks/23_wu2003_summary_statistics.ipynb && echo "nb23 done"
```

**USER REVIEW after nb23** before proceeding to SBI training (nb24/25).

---

## 7. File changes this session

| File | Change |
|------|--------|
| `notebooks/20_wu2003_model_verification.ipynb` | Fixed 2 bugs (NOMINAL_INLET import; suptitle syntax); executed successfully |
| `HANDOFF.md` | This file — updated |

---

## 8. Open design questions (from GRILL session)

1. **nb27 trajectory**: agreed α(t) = 1.0 − 0.35×(t/30d) and η_col(t) = 1.0 − 0.20×(t/30d)^1.5 — not yet implemented.
2. **EKF nb26**: subagent's implementation is "simplified" — needs independent verification before use.
3. **Banana posterior evidence**: must confirm from full SBI training (nb24/25), not from the W12 vs W8 preview.
