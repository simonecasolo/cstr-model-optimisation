"""Build and execute notebook 23 (System I): JAX-differentiable fix for the
``closed_loop_steady_state_exact`` warm-start regression that broke NUTS's
``matched_warm_start=True`` path.

**Context (read `notebooks/system_I/TODO.md` and `HANDOFF.md`'s newest entries, and
`/Users/simo/.claude/projects/.../memory/finding_system1_no_alpha_beta_banana.md`'s
2026-11-08 entries, before trusting anything here).** nb22
(`scripts/build_nb_system_I_22.py`) fixed a real bug in `warm_start_ic`
(`src/cstr_sbi/simulator.py` -> `closed_loop_steady_state_exact`,
`src/cstr_sbi/physics.py`): it now solves for the EXACT closed-loop steady state via
root-finding instead of an under-converged fixed-horizon forward integration, and
determined n_sims=50,000 is the right SBI training budget on the corrected warm-start.
That fix's saturated-branch implementation used plain Python `float()`/
`scipy.optimize.brentq`, which JAX cannot trace -- this broke NUTS's
`matched_warm_start=True` path outright (`ConcretizationTypeError`, confirmed at Sc4
and Sc5, ~1.4s instant crash, not a timeout). Pre-fix, this path was slow
(~310s/window, heavy-tailed up to 60+ min) but differentiable; post-fix, the warm-start
is ~640x cheaper to EVALUATE but was completely unusable for NUTS.

**This notebook's job**: reimplement the saturated branch (and the branch-selection
logic) in a JAX-native, differentiable way -- fixed-iteration-count bisection via
`jax.lax.fori_loop` in place of `scipy.optimize.brentq`, `jnp.where`-based branch
selection in place of the Python `if`/`float()` -- verify this preserves the exact
same physics (same numbers as the old scipy implementation, same |rhs|~0 and
no-redrift verification nb22 used), verify `jax.grad` and `run_mcmc_baseline(...,
matched_warm_start=True)` no longer crash, and re-run the original single-window
decisive test (Sc4, matched_warm_start=True, full-covariance likelihood, 4 chains/500
samples/300 warmup -- the exact protocol `scripts/build_nb_system_I_20.py` Section 5
Step 2 used) to see whether NUTS now agrees with the hand-crafted SBI number at
n=50,000, Corr(alpha,beta)@Sc4 ~= -0.910 (`results/22_final_recommendation.txt`).

**The fix itself was implemented directly in `src/cstr_sbi/physics.py`**
(`closed_loop_steady_state_exact`), not duplicated here -- every call site
(SBI training data generation, NUTS's matched warm start, nb21's open-loop path) picks
it up automatically, exactly like nb22's original fix did. This notebook's job is to
verify it, not re-derive it. A standalone pre-build sanity check (isolated correctness
+ differentiability probe, not inside this notebook) already confirmed: new-vs-old
numerical agreement to ~1e-10 or better at every closed-loop Sc0-7 point, |rhs| actually
IMPROVED (1e-13 to 1e-16 vs the old implementation's 1e-4 to 1e-7, likely because the
new bisection runs 60 fixed iterations to float64 precision while the old `brentq` call
used a slightly looser `xtol=1e-12`), and `jax.grad` succeeds at a non-saturated point
(Sc4) and at two saturated points (Sc2, Sc5) with finite, sane gradients.

**Scope, decided up front**: does NOT touch nb19/20/21/22's own notebooks or scripts,
does NOT redo the full three-method (hand-crafted/CNN/NUTS) cross-check (that is a
separate, later task per the coordinator's explicit instruction) -- this notebook only
fixes and verifies the NUTS differentiability regression and re-runs the ONE decisive
single-window test plus, cost permitting, a small multi-window extension at Sc4 for a
more reliable pooled estimate. Does NOT touch `main.tex`, the SI, or
`article_outline_CChE.md`.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    return new_markdown_cell(source, metadata={"language": "markdown"})


def code(source: str):
    return new_code_cell(source, metadata={"language": "python"})


CELLS = [
    md(
        r"""# Notebook 23 (System I) -- JAX-differentiable fix for the warm-start / NUTS regression

**Read `notebooks/system_I/TODO.md` and `HANDOFF.md`'s newest entries first.** nb22's
`warm_start_ic` fix (`closed_loop_steady_state_exact`, `src/cstr_sbi/physics.py`) made
System I's closed-loop warm start physically correct (exact steady state via
root-finding, not an under-converged fixed-horizon integration) but used plain
`float()`/`scipy.optimize.brentq`, which broke NUTS's `matched_warm_start=True` path
outright (`ConcretizationTypeError`, instant crash, confirmed at Sc4 and Sc5).

**This notebook's job**: verify a JAX-native reimplementation of that function
(fixed-iteration `jax.lax.fori_loop` bisection in place of `brentq`, `jnp.where`-based
branch selection in place of the Python `if`) preserves the exact same physics, fixes
differentiability, and re-run the original decisive single-window NUTS test at Sc4
(4 chains, 500 samples, 300 warmup, full-covariance likelihood, matched_warm_start=True
-- the same protocol `scripts/build_nb_system_I_20.py` Section 5 Step 2 used) to see
whether NUTS now agrees with the hand-crafted SBI posterior's Corr(alpha,beta)@Sc4 ~=
-0.910 at the corrected-warm-start, n=50,000 budget (`results/22_final_recommendation.txt`).

**Does NOT** redo the full three-method cross-check (CNN vs hand-crafted vs NUTS) --
that is a separate, later task. **Does NOT** touch nb19/20/21/22's own files, or
`main.tex`/SI/`article_outline_CChE.md`."""
    ),
    md("## 0. Setup"),
    code(
        r"""import sys, time, pickle, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from cstr_sbi.physics import (
    K0_NOMINAL, UA_NOMINAL, NOMINAL_INLET_CL, NOMINAL_CTRL,
    closed_loop_steady_state_exact, cstr_closed_loop_rhs,
    E_A, R_GAS, Q, V, H_R, RHO, C_P, RHO_C, C_PC,
)
from cstr_sbi.simulator import (
    warm_start_ic, simulate_em_window, apply_sensor_layer, DEFAULT_SENSOR_NOISE_PCT,
)
from cstr_sbi.summaries import compute_summary_statistics
from cstr_sbi.inference import run_mcmc_baseline
from cstr_sbi.scenarios import SCENARIO_CONFIGS, list_configs

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


def mark(stage: str) -> None:
    \"\"\"Disk-flushed progress sentinel -- nb20/21/22's convention, real-time
    visibility around nbclient's stdout buffering.\"\"\"
    (RESULTS / f"23_progress_{stage}.txt").write_text(
        f"{time.time()}\\n{pd.Timestamp.now().isoformat()}\\n"
    )


mark("notebook_start")
print("Setup OK")
"""
    ),
    md(
        r"""## 1. The fix (implemented directly in `src/cstr_sbi/physics.py`, verified here)

`closed_loop_steady_state_exact` now computes the saturated-branch 1-D root-find via a
fixed-iteration-count bisection (`jax.lax.fori_loop`, 60 iterations on `T in [250,450]`)
instead of `scipy.optimize.brentq`, and selects between the non-saturated and saturated
candidate results via `jnp.where` instead of a Python `if`. No `float()` calls remain
anywhere in the function. The physics (non-saturated closed form, saturation-bound
feasibility check via `T_ss(Qc)` monotonicity, the anti-windup epsilon push) is
unchanged -- only the implementation mechanics changed, to make the whole function
traceable and differentiable by JAX."""
    ),
    md("## 2. Correctness check: new JAX-native implementation vs. the old scipy implementation"),
    code(
        r"""# Self-contained copy of the OLD (pre-fix) scipy-based implementation, for a direct
# numerical cross-check -- per this project's own convention (nb19/20 copied
# nb34's helpers verbatim rather than importing a since-changed function).
import numpy as _np
from scipy.optimize import brentq as _brentq

def old_closed_loop_steady_state_exact(params, inlet, ctrl):
    UA = float(params[0]); k0 = float(params[1])
    alpha = float(params[2]); beta = float(params[3])
    Ci, Ti, Tci = float(inlet[0]), float(inlet[1]), float(inlet[2])
    Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max = (float(ctrl[i]) for i in range(6))
    UA_eff = beta * UA

    def k_eff(T): return alpha * k0 * _np.exp(-E_A / (R_GAS * T))
    def C_of_T(T): return (Q / V) * Ci / (Q / V + k_eff(T))
    def Tc_of_T(T, Qc): return (Qc * RHO_C * C_PC * Tci + UA_eff * T) / (Qc * RHO_C * C_PC + UA_eff)
    def dT_residual(T, Qc):
        ke = k_eff(T); C = C_of_T(T); Tc = Tc_of_T(T, Qc)
        return (Q / V) * (Ti - T) - H_R * ke * C / (RHO * C_P) - UA_eff * (T - Tc) / (RHO * C_P * V)
    def solve_T_at(Qc):
        return _brentq(lambda T: dT_residual(T, Qc), 250.0, 450.0, xtol=1e-12, rtol=1e-14)

    T_at_Qmax = solve_T_at(Qc_max)
    T_at_Qmin = solve_T_at(Qc_min)
    if T_at_Qmax <= Tsp <= T_at_Qmin:
        ke_sp = k_eff(Tsp); C_ns = C_of_T(Tsp)
        rhs_no_cool = (Q / V) * (Ti - Tsp) - H_R * ke_sp * C_ns / (RHO * C_P)
        Tc_ns = Tsp - rhs_no_cool * (RHO * C_P * V) / UA_eff
        Qc_ns = -UA_eff * (Tsp - Tc_ns) / (RHO_C * C_PC * (Tci - Tc_ns))
        I_ns = tau_i * (Qc_ns - Qc0)
        return _np.array([C_ns, Tsp, Tc_ns, I_ns])
    if Tsp < T_at_Qmax:
        Qc_fixed, T_star = Qc_max, T_at_Qmax
    else:
        Qc_fixed, T_star = Qc_min, T_at_Qmin
    C_star = C_of_T(T_star); Tc_star = Tc_of_T(T_star, Qc_fixed)
    _qc_eps = 2e-2
    _sat_sign = 1.0 if Qc_fixed == Qc_max else -1.0
    I_star = tau_i * (Qc_fixed - Qc0 - Kp * (T_star - Tsp) + _sat_sign * _qc_eps)
    return _np.array([C_star, T_star, Tc_star, I_star])


rows = []
for sc in list_configs():
    if sc.mode == "open_loop":
        continue
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, sc.alpha, sc.beta], dtype=jnp.float64)
    y_new = closed_loop_steady_state_exact(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
    y_old = old_closed_loop_steady_state_exact(np.array(params), np.array(NOMINAL_INLET_CL), np.array(NOMINAL_CTRL))
    diff = np.array(y_new) - y_old
    rhs_new = cstr_closed_loop_rhs(0.0, y_new, (params, NOMINAL_INLET_CL, NOMINAL_CTRL))
    rhs_old = cstr_closed_loop_rhs(0.0, jnp.array(y_old), (params, NOMINAL_INLET_CL, NOMINAL_CTRL))
    rows.append({
        "scenario": sc.name, "alpha": sc.alpha, "beta": sc.beta,
        "C_new": float(y_new[0]), "T_new": float(y_new[1]), "Tc_new": float(y_new[2]), "I_new": float(y_new[3]),
        "max_abs_diff_vs_old": float(np.max(np.abs(diff))),
        "rhs_max_new": float(jnp.max(jnp.abs(rhs_new))),
        "rhs_max_old": float(jnp.max(jnp.abs(rhs_old))),
    })
    print(f"{sc.name:24s} max|diff|={rows[-1]['max_abs_diff_vs_old']:.3e}  "
          f"|rhs|_new={rows[-1]['rhs_max_new']:.3e}  |rhs|_old={rows[-1]['rhs_max_old']:.3e}")

df_correctness = pd.DataFrame(rows)
df_correctness.to_csv(RESULTS / "23_correctness_check.csv", index=False)
mark("correctness_check_done")
print(f"\\nMax |diff| across all scenarios: {df_correctness['max_abs_diff_vs_old'].max():.3e}")
print(f"Max |rhs| (new impl) across all scenarios: {df_correctness['rhs_max_new'].max():.3e}")
print(f"Max |rhs| (old impl) across all scenarios: {df_correctness['rhs_max_old'].max():.3e}")
assert df_correctness["max_abs_diff_vs_old"].max() < 1e-6, "New JAX-native implementation disagrees with old scipy implementation beyond tolerance!"
assert df_correctness["rhs_max_new"].max() < 1e-6, "New implementation's own residual is not near zero -- not actually at the fixed point!"
print("\\nCORRECTNESS CHECK PASSED: new JAX-native implementation matches old scipy implementation, |rhs|~0 at every closed-loop Sc0-7 point.")
"""
    ),
    md(
        r"""## 3. Forward-redrift check (same nb22 methodology)

Confirm the warm start is a genuine fixed point under the real EM simulator, not just
`|rhs|~0` at t=0 -- integrate forward from the warm start for the real 60-minute
observation window and check the state hasn't drifted."""
    ),
    code(
        r"""from cstr_sbi.physics import simulate_closed_loop_trajectory

redrift_rows = []
for sc in list_configs():
    if sc.mode == "open_loop":
        continue
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, sc.alpha, sc.beta], dtype=jnp.float64)
    y0 = closed_loop_steady_state_exact(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
    _, ys, qc = simulate_closed_loop_trajectory(
        params, NOMINAL_INLET_CL, NOMINAL_CTRL, y0, t_final=60.0, n_save=121,
    )
    drift = np.array(ys[-1, :3]) - np.array(y0[:3])  # C, T, Tc drift over 60 min
    redrift_rows.append({
        "scenario": sc.name,
        "max_abs_drift_C_T_Tc": float(np.max(np.abs(drift))),
    })
    print(f"{sc.name:24s} max|drift over 60min|={redrift_rows[-1]['max_abs_drift_C_T_Tc']:.3e}")

df_redrift = pd.DataFrame(redrift_rows)
df_redrift.to_csv(RESULTS / "23_redrift_check.csv", index=False)
mark("redrift_check_done")
print(f"\\nMax drift across all scenarios: {df_redrift['max_abs_drift_C_T_Tc'].max():.3e}")
assert df_redrift["max_abs_drift_C_T_Tc"].max() < 1e-2, "Warm start is not a stable fixed point under forward simulation!"
print("REDRIFT CHECK PASSED: warm start is a genuine fixed point, confirmed by forward simulation.")
"""
    ),
    md("## 4. Differentiability check"),
    code(
        r"""def warmstart_sumsq(theta):
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, theta[0], theta[1]], dtype=jnp.float64)
    y0 = closed_loop_steady_state_exact(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
    return jnp.sum(y0 ** 2)


diff_rows = []
for sc_name in ["Sc1_closed_healthy", "Sc2_closed_fouling", "Sc3_closed_decay",
                "Sc4_closed_combined", "Sc5_closed_saturated", "Sc7_closed_drift"]:
    sc = SCENARIO_CONFIGS[sc_name]
    theta = jnp.array([sc.alpha, sc.beta], dtype=jnp.float64)
    try:
        g = jax.grad(warmstart_sumsq)(theta)
        status = "OK"
        g_alpha, g_beta = float(g[0]), float(g[1])
    except Exception as exc:
        status = f"FAILED: {exc!r}"
        g_alpha, g_beta = float("nan"), float("nan")
    diff_rows.append({"scenario": sc_name, "status": status, "grad_alpha": g_alpha, "grad_beta": g_beta})
    print(f"{sc_name:24s} jax.grad: {status}  grad=({g_alpha:.3g},{g_beta:.3g})")

df_diff = pd.DataFrame(diff_rows)
df_diff.to_csv(RESULTS / "23_differentiability_check.csv", index=False)
mark("differentiability_check_done")
assert (df_diff["status"] == "OK").all(), "jax.grad still fails through closed_loop_steady_state_exact at some scenario!"
print("\\nDIFFERENTIABILITY CHECK PASSED: jax.grad succeeds through closed_loop_steady_state_exact at every closed-loop scenario (non-saturated AND saturated).")
"""
    ),
    md(
        r"""## 5. End-to-end check: `run_mcmc_baseline(..., matched_warm_start=True)` at Sc4 and Sc5

The actual regression that was reported: this call used to crash instantly with
`ConcretizationTypeError` at both of these scenarios. A small/cheap NUTS run (not the
full decisive protocol yet) confirms it no longer crashes before committing to the
full-cost run below."""
    ),
    code(
        r"""def feat_summary_po(theta, seed, drift_T=0.0):
    alpha, beta = float(theta[0]), float(theta[1])
    params = jnp.array([UA_NOMINAL, K0_NOMINAL, alpha, beta], dtype=jnp.float64)
    y0 = warm_start_ic(params, NOMINAL_INLET_CL, NOMINAL_CTRL)
    proc_key, sens_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    t, ys, qc = simulate_em_window(params, NOMINAL_INLET_CL, NOMINAL_CTRL, y0, key=proc_key)
    obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], qc], axis=1)
    obs = apply_sensor_layer(obs, key=sens_key, noise_pct=DEFAULT_SENSOR_NOISE_PCT)
    if drift_T != 0.0:
        obs = obs.at[:, 1].add(drift_T)
    t_s = jnp.arange(1, ys.shape[0] + 1) * 0.5
    return np.asarray(compute_summary_statistics(jnp.array(obs), jnp.array(t_s)))


smoke_rows = []
for sc_name in ["Sc4_closed_combined", "Sc5_closed_saturated"]:
    sc = SCENARIO_CONFIGS[sc_name]
    theta = np.array([sc.alpha, sc.beta])
    s = feat_summary_po(theta, seed=777, drift_T=sc.drift_T)
    t0 = time.perf_counter()
    try:
        _mcmc, samp, wall = run_mcmc_baseline(
            s, seed=0, n_chains=1, n_samples=20, n_warmup=20,
            progress_bar=False, matched_warm_start=True,
        )
        dt = time.perf_counter() - t0
        status = "SUCCEEDED"
    except Exception as exc:
        dt = time.perf_counter() - t0
        status = f"FAILED: {exc!r}"
    smoke_rows.append({"scenario": sc_name, "status": status, "wall_s": dt})
    print(f"{sc_name:24s} {status}  wall={dt:.1f}s")
    mark(f"smoke_mcmc_{sc_name}_done")

df_smoke = pd.DataFrame(smoke_rows)
df_smoke.to_csv(RESULTS / "23_mcmc_smoke_check.csv", index=False)
assert (df_smoke["status"] == "SUCCEEDED").all(), "run_mcmc_baseline(matched_warm_start=True) still crashes somewhere!"
print("\\nEND-TO-END SMOKE CHECK PASSED: run_mcmc_baseline(matched_warm_start=True) no longer crashes at Sc4 or Sc5.")
"""
    ),
    md(
        r"""## 6. The decisive test: Sc4, full protocol, matched_warm_start=True, full-covariance likelihood

Same protocol as `scripts/build_nb_system_I_20.py` Section 5 Step 2 (the original
Theory D decisive test that first found NUTS agreeing with SBI's sign at Sc4): 4
chains, 500 samples, 300 warmup, full empirical covariance as the likelihood (not the
diagonal default), `matched_warm_start=True`. Measures wall time, R-hat, and
Corr(alpha,beta), compared against the hand-crafted SBI number at the corrected
warm-start + n=50,000 budget: Corr(alpha,beta)@Sc4 ~= -0.910
(`results/22_final_recommendation.txt`)."""
    ),
    code(
        r"""N_REPS_COV = 120

def compute_full_cov_sc4(seed_offset=7000):
    sc4 = SCENARIO_CONFIGS["Sc4_closed_combined"]
    theta_pt = np.array([sc4.alpha, sc4.beta])
    reps = np.stack([
        feat_summary_po(theta_pt, seed=seed_offset + i, drift_T=sc4.drift_T)
        for i in range(N_REPS_COV)
    ])
    Sigma_full = np.cov(reps.T) + 1e-6 * np.eye(reps.shape[1])
    return Sigma_full


mark("cov_precompute_start")
t0 = time.perf_counter()
cov_sc4 = compute_full_cov_sc4()
print(f"cov_obs precompute: {time.perf_counter()-t0:.1f}s, shape={cov_sc4.shape}")
mark("cov_precompute_done")

sc4 = SCENARIO_CONFIGS["Sc4_closed_combined"]
theta_sc4 = np.array([sc4.alpha, sc4.beta])
s_sc4 = feat_summary_po(theta_sc4, seed=777, drift_T=sc4.drift_T)

mark("decisive_test_start")
t0 = time.perf_counter()
mcmc_dec, samp_dec, wall_dec = run_mcmc_baseline(
    s_sc4, seed=0, n_chains=4, n_samples=500, n_warmup=300,
    progress_bar=False, cov_obs=cov_sc4, matched_warm_start=True,
)
dt_dec = time.perf_counter() - t0
mark("decisive_test_done")

import numpyro.diagnostics as _npd
by_chain_dec = mcmc_dec.get_samples(group_by_chain=True)
res_dec = _npd.summary(by_chain_dec)
r_hat_alpha = float(res_dec["alpha"]["r_hat"])
r_hat_beta = float(res_dec["beta"]["r_hat"])
corr_dec = float(np.corrcoef(samp_dec[:, 0], samp_dec[:, 1])[0, 1])

print(f"\\nDECISIVE TEST (Sc4, matched_warm_start=True, full-cov likelihood, 4x500/300):")
print(f"  wall_time_s = {dt_dec:.1f}  (wall_time_s from run_mcmc_baseline: {wall_dec:.1f})")
print(f"  r_hat alpha = {r_hat_alpha:.4f}   r_hat beta = {r_hat_beta:.4f}")
print(f"  Corr(alpha,beta) = {corr_dec:+.3f}")

HC_SBI_SC4 = -0.910  # results/22_final_recommendation.txt, n=50,000, corrected warm-start
print(f"\\nFor reference, hand-crafted SBI at the corrected warm-start, n=50,000: Corr(alpha,beta)@Sc4 = {HC_SBI_SC4:+.3f}")
print(f"NUTS vs SBI sign agreement: {'YES' if (corr_dec * HC_SBI_SC4 > 0) else 'NO'}")
print(f"NUTS vs SBI magnitude ratio: {abs(corr_dec/HC_SBI_SC4)*100:.0f}%")

df_decisive = pd.DataFrame([{
    "scenario": "Sc4_closed_combined", "wall_time_s": dt_dec,
    "r_hat_alpha": r_hat_alpha, "r_hat_beta": r_hat_beta,
    "corr_nuts": corr_dec, "corr_sbi_handcrafted_n50000": HC_SBI_SC4,
    "sign_agree": bool(corr_dec * HC_SBI_SC4 > 0),
}])
df_decisive.to_csv(RESULTS / "23_decisive_test_sc4.csv", index=False)
np.save(RESULTS / "23_decisive_test_sc4_samples.npy", samp_dec)
mark("decisive_test_saved")
"""
    ),
    md(
        r"""## 7. Figure: Sc4 posterior scatter (NUTS, single window) vs. the SBI number"""
    ),
    code(
        r"""fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(samp_dec[:, 0], samp_dec[:, 1], s=4, alpha=0.3, label="NUTS (this notebook)")
ax.axvline(sc4.alpha, color="k", ls=":", lw=0.8)
ax.axhline(sc4.beta, color="k", ls=":", lw=0.8)
ax.set_xlabel("alpha"); ax.set_ylabel("beta")
ax.set_title(f"Sc4 NUTS posterior (fixed warm-start regression), single window\\n"
             f"Corr={corr_dec:+.3f}  (R-hat a={r_hat_alpha:.3f}, b={r_hat_beta:.3f})\\n"
             f"hand-crafted SBI @ n=50000: Corr={HC_SBI_SC4:+.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "23_decisive_test_sc4_posterior.png", dpi=130)
plt.close()
print("Saved figures/23_decisive_test_sc4_posterior.png")
mark("figure_done")
"""
    ),
    md(
        r"""## 8. Cost-check for a small multi-window extension at Sc4

Single-window wall time (Section 6, printed above) gives a per-window cost estimate.
Only proceed to a multi-window pooled estimate (5-10 windows) if the extrapolated cost
is modest (a few minutes, not hours) -- this notebook's job is to fix and verify the
differentiability regression, not to redo the full cross-check, so the multi-window
extension is a nice-to-have bonus, gated explicitly on cost, not launched blindly."""
    ),
    code(
        r"""PER_WINDOW_COST_S = dt_dec
N_EXTRA_WINDOWS = 9  # bringing the total to 10, if cost allows
projected_s = PER_WINDOW_COST_S * N_EXTRA_WINDOWS
print(f"Single-window cost: {PER_WINDOW_COST_S:.1f}s")
print(f"Projected cost for {N_EXTRA_WINDOWS} more windows (serial): {projected_s:.1f}s ({projected_s/60:.1f} min)")

RUN_MULTIWINDOW = projected_s < 20 * 60  # cost gate: proceed only if under 20 minutes serial
print(f"RUN_MULTIWINDOW = {RUN_MULTIWINDOW}")
mark("cost_gate_decided")
"""
    ),
    code(
        r"""multiwindow_rows = [{
    "window": 0, "wall_time_s": dt_dec, "r_hat_alpha": r_hat_alpha, "r_hat_beta": r_hat_beta,
    "corr": corr_dec, "converged": bool(r_hat_alpha < 1.01 and r_hat_beta < 1.01),
    "samples_pkl_idx": 0,
}]
pooled_samples = [samp_dec] if (r_hat_alpha < 1.01 and r_hat_beta < 1.01) else []

if RUN_MULTIWINDOW:
    for w in range(1, 1 + N_EXTRA_WINDOWS):
        theta_w = theta_sc4
        s_w = feat_summary_po(theta_w, seed=777 + w, drift_T=sc4.drift_T)
        t0 = time.perf_counter()
        mcmc_w, samp_w, wall_w = run_mcmc_baseline(
            s_w, seed=w, n_chains=4, n_samples=500, n_warmup=300,
            progress_bar=False, cov_obs=cov_sc4, matched_warm_start=True,
        )
        dt_w = time.perf_counter() - t0
        by_chain_w = mcmc_w.get_samples(group_by_chain=True)
        res_w = _npd.summary(by_chain_w)
        r_hat_a_w = float(res_w["alpha"]["r_hat"])
        r_hat_b_w = float(res_w["beta"]["r_hat"])
        corr_w = float(np.corrcoef(samp_w[:, 0], samp_w[:, 1])[0, 1])
        converged_w = bool(r_hat_a_w < 1.01 and r_hat_b_w < 1.01)
        multiwindow_rows.append({
            "window": w, "wall_time_s": dt_w, "r_hat_alpha": r_hat_a_w, "r_hat_beta": r_hat_b_w,
            "corr": corr_w, "converged": converged_w, "samples_pkl_idx": w,
        })
        if converged_w:
            pooled_samples.append(samp_w)
        print(f"window {w}: wall={dt_w:.1f}s corr={corr_w:+.3f} "
              f"r_hat=({r_hat_a_w:.3f},{r_hat_b_w:.3f}) converged={converged_w}")
        mark(f"multiwindow_{w}_done")

df_multiwindow = pd.DataFrame(multiwindow_rows)
df_multiwindow.to_csv(RESULTS / "23_multiwindow_sc4.csv", index=False)

if len(pooled_samples) > 0:
    pooled = np.concatenate(pooled_samples, axis=0)
    pooled_corr = float(np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1])
    np.save(RESULTS / "23_multiwindow_sc4_pooled_samples.npy", pooled)
    print(f"\\nPOOLED (convergence-gated, {len(pooled_samples)}/{len(multiwindow_rows)} windows): "
          f"Corr(alpha,beta)@Sc4 = {pooled_corr:+.3f}")
else:
    pooled_corr = float("nan")
    print("\\nNo converged windows to pool.")

mark("multiwindow_done")
"""
    ),
    md("## 9. Final verdict -- written after seeing the real numbers above, not before"),
    code(
        r"""print("=" * 70)
print("VERDICT")
print("=" * 70)
print(f"Correctness preserved: max|diff vs old scipy impl| = {df_correctness['max_abs_diff_vs_old'].max():.3e}")
print(f"|rhs| at fixed point (new impl): {df_correctness['rhs_max_new'].max():.3e}")
print(f"Redrift over 60min forward sim: {df_redrift['max_abs_drift_C_T_Tc'].max():.3e}")
print(f"Differentiability: jax.grad succeeds at all 6 closed-loop scenarios: {(df_diff['status']=='OK').all()}")
print(f"run_mcmc_baseline(matched_warm_start=True) no longer crashes at Sc4/Sc5: {(df_smoke['status']=='SUCCEEDED').all()}")
print()
print(f"Decisive single-window test @ Sc4:")
print(f"  wall_time_s = {dt_dec:.1f}")
print(f"  r_hat alpha = {r_hat_alpha:.4f}, r_hat beta = {r_hat_beta:.4f}")
print(f"  Corr(alpha,beta) NUTS = {corr_dec:+.3f}")
print(f"  Corr(alpha,beta) hand-crafted SBI (n=50000) = {HC_SBI_SC4:+.3f}")
print(f"  Sign agreement: {corr_dec * HC_SBI_SC4 > 0}")
if RUN_MULTIWINDOW and len(pooled_samples) > 0:
    print(f"  Multi-window pooled ({len(pooled_samples)}/{len(multiwindow_rows)} converged): Corr={pooled_corr:+.3f}")
mark("notebook_complete")
"""
    ),
]


def build_and_execute() -> int:
    nb = new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }

    out_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "system_I"
        / "23_nuts_differentiable_warmstart_fix.ipynb"
    )

    client = NotebookClient(nb, timeout=None, kernel_name="python3")
    print(f"Executing notebook -> {out_path}")
    client.execute()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    print(f"Wrote {out_path}")
    return 0


def main() -> int:
    return build_and_execute()


if __name__ == "__main__":
    raise SystemExit(main())
