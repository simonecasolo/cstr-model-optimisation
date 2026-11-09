"""Continuation of `pilot_multiround_sc2.py` after the round-2 training crash
(`ValueError: NaN/Inf present in prior eval`). Root cause: round-2 proposal
draws were sampled with `reject_outside_prior=False`, so 3/200 landed just
outside the declared prior box (e.g. alpha=1.203 vs bound 1.20) -- SNPE-C's
atomic loss needs `prior.log_prob(theta)` finite for every training theta, and
the truncated-lognormal/mixed prior returns -inf outside its box by design.
Fix: filter those 3 rows out (197 remain, still plenty) and redo just the
training + evaluation steps, reusing the already-simulated round-2 draws
(the expensive ~45 min simulation phase does NOT need to be repeated).
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
from cstr_sbi.recycle.priors import mixed_alpha_uniform_prior, PRIOR_LOW_5D, PRIOR_HIGH_5D
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.scenarios import get_scenario, FAULT_UNITS
from cstr_sbi.recycle.simulator import nominal_warm_start, deterministic_window, noisy_replicates
from cstr_sbi.recycle.summaries import compute_summaries

DATA = ROOT / "data"
RESULTS = ROOT / "results"

STRUCTURE = "S-C.2"
NOISE_PCT = 0.003
ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
SEED = 0
BANK_SUFFIX = "_mixedprior_stage1bfix"
N_TRAIN = 1000
TARGET_SCENARIO = "W2_cat_mild"
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

    print("=== ROUND 1: retrain from the existing bank ===", flush=True)
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

    torch.manual_seed(999)
    with torch.no_grad():
        samp1 = posterior1.sample((500,), x=x_o, show_progress_bars=False, reject_outside_prior=False).numpy()
    mean1 = {p: float(np.mean(samp1[:, i])) for i, p in enumerate(PARAM_NAMES)}
    std1 = {p: float(np.std(samp1[:, i])) for i, p in enumerate(PARAM_NAMES)}
    pred1, probs1 = classify(samp1)
    print(f"round-1 posterior mean = {({k: round(v,3) for k,v in mean1.items()})}", flush=True)
    print(f"round-1 posterior std  = {({k: round(v,3) for k,v in std1.items()})}", flush=True)
    print(f"round-1 predicted class = {pred1}  (true={true_class})  probs={probs1}", flush=True)

    print("\n=== Loading saved round-2 draws, filtering to in-prior-support rows ===", flush=True)
    r2 = np.load(RESULTS / f"pilot_multiround_sc2_{TARGET_SCENARIO}_round2_draws.npz")
    theta_r2_all = np.asarray(r2["thetas"], dtype=np.float32)
    summary_r2_all = np.asarray(r2["summaries"], dtype=np.float32)
    inside = ((theta_r2_all >= PRIOR_LOW_5D) & (theta_r2_all <= PRIOR_HIGH_5D)).all(axis=1)
    theta_r2 = theta_r2_all[inside]
    summary_r2 = summary_r2_all[inside]
    print(f"kept {inside.sum()}/{len(inside)} rows (dropped {len(inside)-inside.sum()} outside-prior-box)", flush=True)

    proposal2 = posterior1.set_default_x(x_o)
    print("\n=== ROUND 2: train (round1 data + filtered round2 data, proposal-corrected) ===", flush=True)
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
    print(f"\nAll done in {time.time()-t_all:.0f}s total.")


if __name__ == "__main__":
    main()
