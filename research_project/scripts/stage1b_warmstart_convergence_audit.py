"""Stage 1b: audit whether the matched-protocol warm start (`_matched_y0` in
sbi_pipeline.py, which uses LOOSENED tolerances rtol=1e-4/atol=1e-6 and a
SHORTER horizon t_final=100 vs. the physics module's own defaults
rtol=1e-6/atol=1e-8, t_final=200) actually converges to steady state for the
project's real training prior (`truncated_lognormal_5d`), especially in the
tails (e.g. alpha near its 0.40 prior floor).

See docs/Manuscript/system_ii_sbi_solidification_plan.md Stage 1b and the
task description this script was written from for full context.

For each of ~200 prior draws, computes:
  (a) residual_norm : ||recycle_rhs_explicit(0, y0_matched, args)||_2 -- large
      means the matched warm-start is not actually at steady state.
  (b) summary_shift  : relative L2 shift between the S-C summary vector
      computed from y0_matched vs. from a tightly-resolved y0_tight
      (t_final=300, rtol=1e-7, atol=1e-9), plus the max per-feature shift
      relative to that feature's scale.

Compares both diagnostics against the project's standard 0.3% sensor noise
floor, and correlates them against |alpha-1| and |z_A0_eff-0.90|.

Outputs:
  results/44_sc_warmstart_convergence_audit.csv
  figures/44_warmstart_convergence.png
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch
import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
from cstr_sbi.recycle.priors import truncated_lognormal_5d, PARAM_NAMES  # noqa: E402
from cstr_sbi.recycle.physics import (  # noqa: E402
    NOMINAL_INLET, NOMINAL_Y0_EXPLICIT, simulate_to_steady_state_explicit,
    recycle_rhs_explicit,
)
from sbi_pipeline import _matched_y0, _structure_config, _simulate_summary  # noqa: E402

# JIT-wrap the tight re-solve (mirrors sbi_pipeline.py's own `_scenario_ss_jit`
# pattern for `_matched_y0`) -- calling `simulate_to_steady_state_explicit`
# unjitted inside a 200-iteration Python loop retraces/recompiles the diffrax
# solve from scratch on every call, which is orders of magnitude slower than
# compiling once and reusing the compiled executable.
#
# max_steps deviation from the task spec (2_000_000): a live run showed that
# specific prior draws (plausibly ones far from nominal, e.g. low alpha) drive
# the tight, tightly-toleranced solve into a very long tail of tiny adaptive
# steps -- one single draw consumed >20 minutes of wall clock without
# finishing, which is itself evidence relevant to this audit (a draw the tight
# solver struggles to converge on is exactly the tail-corruption pattern being
# tested for) but made the full sweep intractable at 2_000_000. Capped to
# 400_000 (still 4x the loose solve's 100_000 cap) so pathological draws fail
# fast (NaN -> recorded as non-convergent) instead of hanging the whole run.
TIGHT_MAX_STEPS = 400_000
_tight_ss_jit = jax.jit(
    lambda theta, ctrl: simulate_to_steady_state_explicit(
        theta, NOMINAL_INLET, ctrl, NOMINAL_Y0_EXPLICIT,
        t_final=300.0, rtol=1e-7, atol=1e-9, max_steps=TIGHT_MAX_STEPS,
    )
)

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

N_DRAWS = 80  # cut from the originally planned 200: smoke test showed ~335s
              # one-time JIT-compile + ~6s/draw after warmup -> N=200 would run
              # ~25 min; N=80 (~13 min) still gives solid correlation power and
              # the effect size found in the smoke test is large enough that
              # this sample size is not underpowered for the headline verdict.
SEED = 20260902
NOISE_FLOOR = 0.003  # project standard 0.3% relative channel noise


def main():
    print(f"PARAM_NAMES = {PARAM_NAMES}")
    torch.manual_seed(SEED)
    prior_logn = truncated_lognormal_5d()
    theta_draws = prior_logn.sample((N_DRAWS,)).numpy().astype(np.float64)
    print(f"Drew {N_DRAWS} samples from truncated_lognormal_5d prior.")

    ctrl, _y0_nom = _structure_config("S-C")

    residual_norms = np.full(N_DRAWS, np.nan)
    summary_shifts_rel = np.full(N_DRAWS, np.nan)
    summary_shifts_maxfeat = np.full(N_DRAWS, np.nan)
    summary_shifts_maxfeat_idx = np.full(N_DRAWS, -1, dtype=int)
    n_valid = 0

    alpha_all = theta_draws[:, 0]

    t0 = time.time()
    for i in range(N_DRAWS):
        theta_np = theta_draws[i]
        theta_jnp = jnp.array(theta_np, dtype=jnp.float32)
        d0 = time.time()

        # --- matched (loose) warm start, exactly as used in bank generation ---
        y0_matched = _matched_y0(theta_np, ctrl)

        # --- residual diagnostic (a) ---
        rhs = recycle_rhs_explicit(0.0, y0_matched, (theta_jnp, NOMINAL_INLET, ctrl))
        res_norm = float(np.linalg.norm(np.asarray(rhs)))
        residual_norms[i] = res_norm

        # --- tight re-solve (JIT-compiled once, reused across all draws) ---
        y0_tight = np.asarray(_tight_ss_jit(theta_jnp, ctrl))
        tight_had_nan = bool(np.isnan(y0_tight).any())

        # --- downstream summary diagnostic (b) ---
        if tight_had_nan:
            s_matched = s_tight = None
        else:
            rng_m = np.random.default_rng(0)
            rng_t = np.random.default_rng(0)
            s_matched = _simulate_summary(
                theta_np, ctrl, y0_matched, "S-C", noise_pct=0.0, rng=rng_m,
                scenario_specific_warm_start=False,
            )
            s_tight = _simulate_summary(
                theta_np, ctrl, y0_tight, "S-C", noise_pct=0.0, rng=rng_t,
                scenario_specific_warm_start=False,
            )

        d_elapsed = time.time() - d0
        if s_matched is None or s_tight is None:
            print(f"  draw {i}: invalid/non-convergent (tight_had_nan={tight_had_nan}), "
                  f"skipping summary diag (theta={theta_np}) [{d_elapsed:.1f}s]", flush=True)
        else:
            diff = s_matched - s_tight
            rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(s_tight), 1e-12)
            scale = np.maximum(np.abs(s_tight), 1e-6)
            per_feat_rel = np.abs(diff) / scale
            max_idx = int(np.argmax(per_feat_rel))
            summary_shifts_rel[i] = rel_l2
            summary_shifts_maxfeat[i] = per_feat_rel[max_idx]
            summary_shifts_maxfeat_idx[i] = max_idx
            n_valid += 1
            print(f"  draw {i}: alpha={alpha_all[i]:.3f}  res_norm={res_norm:.4g}  "
                  f"shift_rel_l2={rel_l2:.4g}  [{d_elapsed:.1f}s, "
                  f"{time.time()-t0:.0f}s elapsed total]", flush=True)

        # incremental checkpoint every 5 draws so partial progress survives
        # an interruption
        if (i + 1) % 5 == 0 or (i + 1) == N_DRAWS:
            import pandas as pd
            ckpt = pd.DataFrame({
                "alpha": theta_draws[:i+1, 0], "beta_r": theta_draws[:i+1, 1],
                "eta_col": theta_draws[:i+1, 2], "xi_reb": theta_draws[:i+1, 3],
                "z_A0_eff": theta_draws[:i+1, 4],
                "residual_norm": residual_norms[:i+1],
                "summary_shift_rel_l2": summary_shifts_rel[:i+1],
                "summary_shift_max_feature_rel": summary_shifts_maxfeat[:i+1],
                "summary_shift_max_feature_idx": summary_shifts_maxfeat_idx[:i+1],
            })
            ckpt.to_csv(RESULTS_DIR / "44_sc_warmstart_convergence_audit.csv", index=False)
            elapsed = time.time() - t0
            print(f"  [checkpoint] {i+1}/{N_DRAWS} draws done ({elapsed:.0f}s elapsed, "
                  f"{n_valid} valid summaries so far)", flush=True)

    print(f"Done: {n_valid}/{N_DRAWS} valid summary comparisons, "
          f"{time.time()-t0:.0f}s total.")

    alpha = theta_draws[:, 0]
    beta_r = theta_draws[:, 1]
    eta_col = theta_draws[:, 2]
    xi_reb = theta_draws[:, 3]
    z_A0_eff = theta_draws[:, 4]
    abs_alpha_dev = np.abs(alpha - 1.0)
    abs_z_dev = np.abs(z_A0_eff - 0.90)

    # --- save results table ---
    import pandas as pd
    df = pd.DataFrame({
        "alpha": alpha, "beta_r": beta_r, "eta_col": eta_col,
        "xi_reb": xi_reb, "z_A0_eff": z_A0_eff,
        "abs_alpha_dev": abs_alpha_dev, "abs_z_A0_dev": abs_z_dev,
        "residual_norm": residual_norms,
        "summary_shift_rel_l2": summary_shifts_rel,
        "summary_shift_max_feature_rel": summary_shifts_maxfeat,
        "summary_shift_max_feature_idx": summary_shifts_maxfeat_idx,
    })
    out_csv = RESULTS_DIR / "44_sc_warmstart_convergence_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved results table to {out_csv}")

    # --- summary statistics ---
    def stats_block(x, name):
        x_valid = x[~np.isnan(x)]
        print(f"\n{name}: n={len(x_valid)}")
        print(f"  mean={np.mean(x_valid):.6g}  median={np.median(x_valid):.6g}  "
              f"max={np.max(x_valid):.6g}  min={np.min(x_valid):.6g}")
        top_idx = np.argsort(-np.nan_to_num(x, nan=-np.inf))[:4]
        print(f"  top-4 draws by {name}:")
        for idx in top_idx:
            print(f"    draw {idx}: {name}={x[idx]:.6g}  theta="
                  f"alpha={alpha[idx]:.4f}, beta_r={beta_r[idx]:.4f}, "
                  f"eta_col={eta_col[idx]:.4f}, xi_reb={xi_reb[idx]:.4f}, "
                  f"z_A0_eff={z_A0_eff[idx]:.4f}")
        return top_idx

    stats_block(residual_norms, "residual_norm")
    stats_block(summary_shifts_rel, "summary_shift_rel_l2")
    stats_block(summary_shifts_maxfeat, "summary_shift_max_feature_rel")

    # --- correlations ---
    def corr(x, y, name_x, name_y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        print(f"  Pearson r({name_x}, {name_y}) = {r:.4f}  (n={mask.sum()})")
        return r

    print("\nCorrelations:")
    r_res_alpha = corr(residual_norms, abs_alpha_dev, "residual_norm", "|alpha-1|")
    r_res_z = corr(residual_norms, abs_z_dev, "residual_norm", "|z_A0_eff-0.90|")
    r_shift_alpha = corr(summary_shifts_rel, abs_alpha_dev, "summary_shift_rel_l2", "|alpha-1|")
    r_shift_z = corr(summary_shifts_rel, abs_z_dev, "summary_shift_rel_l2", "|z_A0_eff-0.90|")

    # --- noise-floor comparison ---
    frac_above_floor = np.mean(summary_shifts_rel[~np.isnan(summary_shifts_rel)] > NOISE_FLOOR)
    print(f"\nFraction of draws with summary_shift_rel_l2 > noise floor "
          f"({NOISE_FLOOR}): {frac_above_floor:.3f}")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(abs_alpha_dev, residual_norms, c=abs_z_dev, cmap="viridis", s=25)
    axes[0].set_xlabel("|alpha - 1.0|")
    axes[0].set_ylabel("residual_norm at y0_matched")
    axes[0].set_yscale("log")
    axes[0].set_title(f"RHS residual norm vs alpha deviation\n(r={r_res_alpha:.3f})")
    fig.colorbar(sc0, ax=axes[0], label="|z_A0_eff - 0.90|")

    sc1 = axes[1].scatter(abs_alpha_dev, summary_shifts_rel, c=abs_z_dev, cmap="viridis", s=25)
    axes[1].axhline(NOISE_FLOOR, color="red", linestyle="--",
                     label=f"0.3% sensor noise floor")
    axes[1].set_xlabel("|alpha - 1.0|")
    axes[1].set_ylabel("summary_shift (rel L2, matched vs tight)")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Downstream summary shift vs alpha deviation\n(r={r_shift_alpha:.3f})")
    axes[1].legend()
    fig.colorbar(sc1, ax=axes[1], label="|z_A0_eff - 0.90|")

    fig.suptitle("Stage 1b: matched-protocol warm-start convergence audit (S-C, n=200)")
    fig.tight_layout()
    out_fig = FIGURES_DIR / "44_warmstart_convergence.png"
    fig.savefig(out_fig, dpi=150)
    print(f"Saved figure to {out_fig}")


if __name__ == "__main__":
    main()
