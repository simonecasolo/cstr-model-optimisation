# Session Handoff — Wu 2003 nb29 Complete, Next: Retrain with zuko_nsf + Summary Stat Fix

**Date:** 2026-07-03  
**Session status:** nb29 executed. nb27 killed (rejection sampling). Next session starts implementation of three-variant retraining plan.

---

## What was completed in this session

### 1. nb29 SBC Investigation — key findings

| Finding | Result |
|---------|--------|
| η_col SBC (S-B) | **p = 0.0001**, U-shaped — **genuine overconfidence** |
| Root cause | `recycle_ratio` (corr=−0.977 with α), `reb_intensity` (corr=+0.642 with α) dominate η_col signal |
| η_col signal strength | Summaries move 1.45× MORE with η_col than α — NOT a weak-signal problem |
| ξ_reb SBC (S-A) | **p = 0.1464 with reject_outside_prior=False** — PASSES. Peaked histogram was a **rejection sampling artifact**, not a real calibration problem |
| Banana posterior | SBI scatter is a near-vertical stripe at η_col≈0.80, NOT a curved banana. Physical banana visible in iso-contour figure (nb26_banana_physics.png) |

### 2. Article outline updated
- §7.4 banana claim qualified: SBI posterior is near-vertical stripe; physical banana confirmed by iso-contours
- §8.4 L5: η_col root cause specified (recycle_ratio confounding)
- §8.4 Note: ξ_reb peaked histogram = rejection sampling artifact, posterior actually calibrated
- nb26 updated with banana physics figure and commentary

### 3. nb27 killed (was running ~7h, stuck in rejection sampling)
The simulation phase (360 windows) completed in 30s. The SBI inference phase hit rejection sampling failures at α≈0.65 windows. Process killed. nb27 needs to be rewritten with `reject_outside_prior=False` before next attempt.

---

## What needs to be done next

### Priority 1: Retrain with three variants (plan file: `/Users/simo/.claude/plans/partitioned-swinging-aho.md`)

**Motivation:** η_col overconfidence is caused by summary statistic confounding + architecture overfitting. Fix both simultaneously.

**Three variants:**

| Variant | Architecture | noise_pct | Summary stats | Goal |
|---------|-------------|-----------|---------------|------|
| **A** (primary) | `zuko_nsf` 60/3 | 0.3% | `reb_per_boilup` (new) | Fix η_col; primary paper results |
| **B** | `zuko_nsf` 60/3 | 1.0% | `reb_per_boilup` (new) | Noise sensitivity test |
| **C** | `nsf` 128/5 | 1.0% | `reb_per_boilup` (new) | Isolate: architecture vs noise |

**Implementation order:**

1. **`src/cstr_sbi/recycle/summaries.py`** — replace `reb_intensity` with `reb_per_boilup`:
   ```python
   # OLD:
   reb_intensity = np.mean(Q_reb / np.maximum(F_R_n, 0.1)) / QREB_NOM
   # NEW (position 3 in return array, same index):
   reb_per_boilup = np.mean(Q_reb / np.maximum(V_norm * QREB_NOM, 1e3))
   ```
   Update `PHYSICS_FEATURE_NAMES[3]` from `"reb_intensity"` to `"reb_per_boilup"`.
   N_SUMMARIES unchanged (66/72). Dimension stays the same.

2. **Re-run nb22** (data generation) with `noise_pct=0.003` → primary dataset
   Also save `wu2003_observations_1pct.npz` with `noise_pct=0.010`

3. **Re-run nb23** (summary statistics) — updates summary features file

4. **Retrain nb24/25** (use standalone scripts to avoid OOM):
   ```python
   # Change in training cell:
   density_estimator = posterior_nn(model='zuko_nsf', hidden_features=60, num_transforms=3)
   ```
   Use `reject_outside_prior=False` in ALL posterior.sample() calls throughout.

5. **Verify:** η_col SBC KS p > 0.05 for Variant A (target)

### Priority 2: Rewrite nb27 (sequential tracking)
- Use `reject_outside_prior=False` in the SBI inference loop
- This was the cause of the ~7h stall

### Priority 3: Create nb30 (Wu 2003 fault classification)
- Same approach as nb11 (posterior mass in fault-unit regions)
- Already outlined in article plan §7.3

---

## Current state of all notebooks

| Notebook | Status | Notes |
|----------|--------|-------|
| nb20 | ✅ Executed, documented | Model verification |
| nb21 | ✅ Executed, documented | Control structure comparison |
| nb22 | ✅ Executed | Dataset — needs re-run after summaries.py fix |
| nb23 | ✅ Executed + FIM section | Needs re-run after summaries.py fix |
| nb24 | ✅ Executed (nsf 128/5) | Needs retrain with zuko_nsf 60/3 |
| nb25 | ✅ Executed (nsf 128/5) | Needs retrain with zuko_nsf 60/3 |
| nb26 | ✅ Figures via script | OOM in notebook; banana physics figure added |
| nb27 | ❌ Killed (rejection sampling at α≈0.65) | Needs rewrite with reject_outside_prior=False |
| nb28 | ⏭️ Skipped | Publication figures — after all results stable |
| nb29 | ✅ Executed | η_col root cause confirmed; ξ_reb artifact resolved |
| nb30 | 📋 Not started | Fault classification (nb11 approach) |

---

## Key numbers for the paper (current best, pre-fix)

### From nb24/25/26 (nsf 128/5, 0.3% noise):
- W12 α 90% CI (S-B): **0.240** — banana width, 100% coverage ✅
- W12 α 90% CI (S-A): **0.059** — 75% reduction, 93% coverage ✅  
- EKF α coverage (W12): **3%** — complete failure ✅
- W15 SBI coverage: **100%** vs EKF 3% ✅
- η_col posterior: **overconfident** (SBC p=0.0001) — not yet fixed

### From nb23 FIM:
- T_r → I_αα: **0.00%**, T_r → I_β_r: **0.00%** — Loop 1 masking confirmed ✅
- I_αα/I_β_r = **1.1×** (not 250-500× like PO — different because no observable concentration channel)
- (α, β_r) off-diagonal = **+0.901** — both confounded through shared features at nominal

---

## Critical open questions before paper submission

1. Does `zuko_nsf` (60/3) + `reb_per_boilup` fix η_col SBC? (Variant A test)
2. Does 1% noise change the banana/α results significantly? (Variant B test)  
3. Can nb27 produce clean 30-day tracking results?
4. Is nb30 fault classification feasible given η_col overconfidence?

---

## Push reminder
Several commits are pending push. The pre-push lfs hook must be removed first:
```bash
rm /Users/simo/inso-po-RD/cstr-model-optimisation/.git/hooks/pre-push
git -C /Users/simo/inso-po-RD/cstr-model-optimisation push origin main
```
