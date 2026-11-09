"""PILOT: does classic multi-round (sequential) SNPE_C fix the S-C.2
classification failures, as an alternative/complement to the wide-prior
widening approach (`system_ii_sbi_solidification_plan.md` priority-0)?

Motivation: the diagnosed failure mode -- narrow-lognormal training prior
puts almost no mass where the real Wu2003 fault scenarios live, causing
shrinkage-toward-prior-mean and catastrophic classification (macro-F1~0.07-
0.11) -- is the textbook use case for sequential/focused NPE (Papamakarios &
Murray 2016; Greenberg/Nonnenmacher/Macke 2019 "APT"/SNPE-C): round 1 trains
broadly (as already done), round 2 draws NEW training simulations from
`round1_posterior.set_default_x(x_o)` (i.e. concentrated near the ACTUAL
observation of interest) instead of from the raw prior, and SNPE-C's atomic
proposal correction keeps the estimate a valid p(theta|x) under the ORIGINAL
prior throughout (not a truncation/reweighting hack) -- see sbi's own
multi-round tutorial. This should fix the coverage mismatch far more
cheaply than uniform-widening (round-2 draws are concentrated near a real,
physically-plausible operating point, not scattered over prior corners the
ODE solver struggles with) and can better follow a curved manifold (e.g. the
nb39-41 alpha/z_A0_eff ridge) that a box-uniform prior cannot target.

Scope (deliberately small, per this session's earlier over-run lesson):
ONE scenario (W2_cat_mild, alpha=0.85 -- a subtle case still misclassified
as `healthy` even under the alpha-only mixed-prior fix, per HANDOFF.md), one
seed, round 2 capped at N_ROUND2_TARGET accepted draws / N_ROUND2_CAP
attempts. This is a directional pilot, NOT a confirmation-grade result.

Caveat (documented, not swept under the rug): a round-2 posterior is
"focused" -- accurate near its target x_o, not globally amortized like
round 1. For CLASSIFICATION this is fine (at diagnosis time you have the
real x_o and can afford 1-2 extra focused rounds before calling it); it does
change the manuscript narrative from "one amortized posterior classifies
everything" to a two-stage (broad round 1 -> focused round 2) architecture,
which needs the user's buy-in before this becomes the adopted design.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from cstr_sbi.recycle.priors import mixed_alpha_uniform_prior
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.scenarios import get_scenario, FAULT_UNITS
from cstr_sbi.recycle.simulator import nominal_warm_start, deterministic_window, noisy_replicates
from cstr_sbi.recycle.summaries import compute_summaries
from sbi_pipeline import _structure_config, _simulate_summary

DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-C.2"
NOISE_PCT = 0.003
ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
SEED = 0
BANK_SUFFIX = "_mixedprior_stage1bfix"  # reuse the existing round-1 (alpha-only-wide) bank
N_TRAIN = 1000

TARGET_SCENARIO = "W2_cat_mild"          # alpha=0.85, still misclassified as healthy
N_ROUND2_TARGET = 200                     # accepted round-2 draws to aim for
N_ROUND2_CAP_ATTEMPTS = 400               # hard stop even if accept rate is poor
THRESH, Z0_NOM = 0.85, 0.90


def sample_fault_unit(theta_row, thresh=THRESH, z0_nom=Z0_NOM):
    alpha, beta_r, eta_col, xi_reb, z_A0 = theta_row
    reactor_bad = (alpha < thresh) or (beta_r < thresh)
    column_bad = (eta_col < thresh) or (xi_reb < thresh)
    feed_bad = z_A0 < thresh * z0_nom
    n_bad = int(reactor_bad) + int(column_bad) + int(feed_bad)
    if n_bad == 0:
        return "healthy"
    if n_bad >= 2:
        return "multi"
    if reactor_bad:
        return "reactor"
    if column_bad:
        return "column"
    return "feed"


def classify(samples):
    labels = [sample_fault_unit(row) for row in samples]
    probs = {u: float(np.mean([l == u for l in labels])) for u in FAULT_UNITS}
    return max(probs, key=probs.__getitem__), probs


def main():
    t_all = time.time()
    prior = mixed_alpha_uniform_prior()

    print("=== ROUND 1: retrain from the existing bank (cheap -- data already on disk) ===", flush=True)
    bank = np.load(DATA / f"wu2003_sbi_train_sc2_true_n{N_TRAIN}{BANK_SUFFIX}.npz")
    theta_train = np.asarray(bank["thetas"], dtype=np.float32)
    summary_train = np.asarray(bank["summaries"], dtype=np.float32)

    torch.manual_seed(SEED)
    inference = SNPE(prior=prior, density_estimator=posterior_nn(
        model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS, z_score_x="none"))
    inference.append_simulations(torch.tensor(theta_train), torch.tensor(summary_train), proposal=prior)
    t0 = time.time()
    est1 = inference.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1,
                            stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    posterior1 = inference.build_posterior(est1)
    print(f"round-1 trained in {time.time()-t0:.0f}s", flush=True)

    print(f"\n=== Building x_o for target scenario {TARGET_SCENARIO} ===", flush=True)
    sc = get_scenario(TARGET_SCENARIO)
    theta_true = np.asarray(sc.theta(), dtype=np.float32)
    true_class = sc.fault_unit()
    y0_sb = nominal_warm_start("S-B")
    t_h, raw_obs = deterministic_window(sc, structure="S-B", y0=y0_sb)
    t_h_arr = np.asarray(t_h)
    rng = np.random.default_rng(20260705)
    reps = noisy_replicates(raw_obs, n_replicates=1, rng=rng, noise_pct=NOISE_PCT)
    x_o_np = compute_summaries(reps[0], STRUCTURE, t_h_arr, theta=theta_true, rng=rng)
    x_o = torch.tensor(x_o_np, dtype=torch.float32)
    print(f"true theta = {dict(zip(PARAM_NAMES, np.round(theta_true, 3).tolist()))}, true_class={true_class}", flush=True)

    print("\n=== ROUND 1 posterior at x_o (baseline, expected to shrink/misclassify) ===", flush=True)
    torch.manual_seed(999)
    with torch.no_grad():
        samp1 = posterior1.sample((500,), x=x_o, show_progress_bars=False, reject_outside_prior=False).numpy()
    mean1 = {p: float(np.mean(samp1[:, i])) for i, p in enumerate(PARAM_NAMES)}
    std1 = {p: float(np.std(samp1[:, i])) for i, p in enumerate(PARAM_NAMES)}
    pred1, probs1 = classify(samp1)
    print(f"round-1 posterior mean = {({k: round(v,3) for k,v in mean1.items()})}", flush=True)
    print(f"round-1 posterior std  = {({k: round(v,3) for k,v in std1.items()})}", flush=True)
    print(f"round-1 predicted class = {pred1}  (true={true_class})  probs={probs1}", flush=True)

    print(f"\n=== ROUND 2: simulate ~{N_ROUND2_TARGET} draws from posterior1.set_default_x(x_o) ===", flush=True)
    proposal2 = posterior1.set_default_x(x_o)
    ctrl, y0 = _structure_config(STRUCTURE)
    sim_rng = np.random.default_rng(20260906)
    theta_r2, summary_r2 = [], []
    attempts = 0
    t0 = time.time()
    torch.manual_seed(20260906)
    while len(theta_r2) < N_ROUND2_TARGET and attempts < N_ROUND2_CAP_ATTEMPTS:
        attempts += 1
        with torch.no_grad():
            th = proposal2.sample((1,), show_progress_bars=False, reject_outside_prior=False).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, sim_rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_r2.append(th)
        summary_r2.append(s)
        if len(theta_r2) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  round2 sim: {len(theta_r2)}/{N_ROUND2_TARGET}, attempts={attempts} "
                  f"(accept_rate={len(theta_r2)/attempts:.3f}), elapsed={elapsed:.0f}s", flush=True)
    accept_rate2 = len(theta_r2) / attempts
    print(f"round-2 simulation done: {len(theta_r2)} accepted / {attempts} attempts "
          f"(accept_rate={accept_rate2:.3f}), wall_s={time.time()-t0:.0f}", flush=True)
    if len(theta_r2) < 20:
        print("*** too few round-2 draws accepted -- aborting pilot, this proposal region is not simulable. ***")
        return

    theta_r2 = np.asarray(theta_r2, dtype=np.float32)
    summary_r2 = np.asarray(summary_r2, dtype=np.float32)
    np.savez(RESULTS / f"pilot_multiround_sc2_{TARGET_SCENARIO}_round2_draws.npz",
             thetas=theta_r2, summaries=summary_r2, attempts=attempts, target_scenario=TARGET_SCENARIO)

    print("\n=== ROUND 2: train (round1 data + round2 data, proposal-corrected) ===", flush=True)
    inference.append_simulations(torch.tensor(theta_r2), torch.tensor(summary_r2), proposal=proposal2)
    t0 = time.time()
    est2 = inference.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1,
                            stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    posterior2 = inference.build_posterior(est2)
    print(f"round-2 trained in {time.time()-t0:.0f}s", flush=True)

    print("\n=== ROUND 2 posterior at the SAME x_o ===", flush=True)
    torch.manual_seed(999)
    with torch.no_grad():
        samp2 = posterior2.sample((500,), x=x_o, show_progress_bars=False, reject_outside_prior=False).numpy()
    mean2 = {p: float(np.mean(samp2[:, i])) for i, p in enumerate(PARAM_NAMES)}
    std2 = {p: float(np.std(samp2[:, i])) for i, p in enumerate(PARAM_NAMES)}
    pred2, probs2 = classify(samp2)
    print(f"round-2 posterior mean = {({k: round(v,3) for k,v in mean2.items()})}", flush=True)
    print(f"round-2 posterior std  = {({k: round(v,3) for k,v in std2.items()})}", flush=True)
    print(f"round-2 predicted class = {pred2}  (true={true_class})  probs={probs2}", flush=True)

    print(f"\n=== SUMMARY ({TARGET_SCENARIO}, true_class={true_class}, true theta={np.round(theta_true,3).tolist()}) ===")
    print(f"round 1: mean={ {k: round(v,3) for k,v in mean1.items()} }  pred={pred1}")
    print(f"round 2: mean={ {k: round(v,3) for k,v in mean2.items()} }  pred={pred2}")
    print(f"round-2 sim accept_rate={accept_rate2:.3f} (compare: alpha-only-wide-prior bank ~0.75-0.9, "
          f"full-5D-uniform bank ~0.37)")
    print(f"\nAll done in {time.time()-t_all:.0f}s total.")


if __name__ == "__main__":
    main()
