"""Arm C: train + SBC-evaluate S-B (5-D, no feed analyzer) on the clean,
post-warm-start-fix paired bank (`wu2003_sbi_train_sb_paired_lognormal_true_n1000_stage1bfix.npz`).

This was planned in `docs/Manuscript/system_ii_analyzer_rework_plan.md` ("Arm C")
and `docs/Manuscript/system_ii_sbi_solidification_plan.md` as the fair S-B-vs-S-C
comparison point, but deliberately deferred ("deferred until S-C's own calibration
is settled") and never actually run. Since S-C's calibration was never settled
(Gate C invoked instead), this notebook/script answers a now more urgent question
directly: does S-B's SBC failure (the one behind `wu2003_posterior_sb_matched_full.pkl`,
0/8 seeds, pre-dating the warm-start fix) survive on genuinely clean data, or was
part of it a data-quality artifact?

No new simulation needed for the SBC sets: S-C's summary vector is exactly
S-B's 66-D summary (first 66 columns) plus 6 appended feed-analyzer-channel
stats, computed from the same theta/raw-trajectory as the paired bank design
guarantees -- so slicing columns [:66] of the existing S-C stage1bfix SBC sets
gives a valid, zero-extra-cost S-B SBC set at the same theta rows.

Same protocol as `scripts/run_sc2_training.py` (baseline + z_score_x='none',
3 seeds, reduced N=120 then confirm N=400 SBC, zuko_nsf 60/3, n_train=1000) for
direct comparability.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.priors import truncated_lognormal_5d

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SBI_LOGS = ROOT / "sbi-logs"
for p in [DATA, RESULTS, SBI_LOGS]:
    p.mkdir(exist_ok=True)

ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
N_TRAIN = 1000
SUFFIX = "_stage1bfix"
N_POST = 120
SEEDS = [0, 1, 2]

prior_logn = truncated_lognormal_5d()

bank = np.load(DATA / f"wu2003_sbi_train_sb_paired_lognormal_true_n{N_TRAIN}{SUFFIX}.npz")
theta_train = np.asarray(bank["thetas"], dtype=np.float32)
summary_train = np.asarray(bank["summaries"], dtype=np.float32)
assert summary_train.shape[1] == 66, f"expected 66-D S-B summary, got {summary_train.shape}"

r120_sc = np.load(RESULTS / f"24a_sysII_sc_lognormal_sbc_set_n120{SUFFIX}.npz")
r400_sc = np.load(RESULTS / f"24a_sysII_sc_lognormal_sbc_set_n400{SUFFIX}.npz")
theta_sbc, summary_sbc = np.asarray(r120_sc["thetas"], dtype=np.float32), np.asarray(r120_sc["summaries"], dtype=np.float32)[:, :66]
theta_sbc_c, summary_sbc_c = np.asarray(r400_sc["thetas"], dtype=np.float32), np.asarray(r400_sc["summaries"], dtype=np.float32)[:, :66]
print(f"theta_train={theta_train.shape} summary_train={summary_train.shape} "
      f"theta_sbc={theta_sbc.shape} summary_sbc={summary_sbc.shape} "
      f"theta_sbc_c={theta_sbc_c.shape} summary_sbc_c={summary_sbc_c.shape}")


def train_posterior(torch_seed, z_score_x=None):
    torch.manual_seed(torch_seed)
    kwargs = dict(model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS)
    if z_score_x is not None:
        kwargs["z_score_x"] = z_score_x
    inference = SNPE(prior=prior_logn, density_estimator=posterior_nn(**kwargs))
    inference.append_simulations(torch.tensor(theta_train, dtype=torch.float32), torch.tensor(summary_train, dtype=torch.float32))
    t0 = time.time()
    trained = inference.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1, stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    return inference.build_posterior(trained), time.time() - t0


def run_sbc(posterior, theta_sbc_set, summary_sbc_set, seed, n_post=N_POST):
    torch.manual_seed(seed)
    ranks = {name: [] for name in PARAM_NAMES}
    means = {name: [] for name in PARAM_NAMES}
    for th, x_obs in zip(theta_sbc_set, summary_sbc_set):
        with torch.no_grad():
            samp = posterior.sample((n_post,), x=torch.tensor(x_obs, dtype=torch.float32),
                                     show_progress_bars=False, reject_outside_prior=False).detach().cpu().numpy()
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
            means[name].append(float(samp[:, k].mean()))
    results = {"n_sbc": int(len(theta_sbc_set)), "n_post": int(n_post)}
    for name in PARAM_NAMES:
        r = np.asarray(ranks[name], dtype=float)
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        true_vals = theta_sbc_set[:, PARAM_NAMES.index(name)]
        est_vals = np.asarray(means[name])
        slope = np.polyfit(true_vals, est_vals, 1)[0] if np.std(true_vals) > 0 else float("nan")
        results[f"{name}_ks_p"] = float(ks.pvalue)
        results[f"{name}_mean_rank_frac"] = float(r.mean() / n_post)
        results[f"{name}_slope"] = float(slope)
    return results


all_rows = []
for z_score_label, z_score_x in [("baseline", None), ("zscorenone", "none")]:
    for seed in SEEDS:
        posterior, train_s = train_posterior(seed, z_score_x=z_score_x)
        reduced = run_sbc(posterior, theta_sbc, summary_sbc, seed, n_post=N_POST)
        confirm = run_sbc(posterior, theta_sbc_c, summary_sbc_c, seed, n_post=400)
        print(f"[{z_score_label} seed={seed}] trained in {train_s:.1f}s")
        print("  reduced (N=120):", {k: round(v, 4) for k, v in reduced.items() if k.endswith("_ks_p")})
        print("  confirm (N=400):", {k: round(v, 4) for k, v in confirm.items() if k.endswith("_ks_p")})
        min_ks_reduced = min(reduced[f"{n}_ks_p"] for n in PARAM_NAMES)
        min_ks_confirm = min(confirm[f"{n}_ks_p"] for n in PARAM_NAMES)
        all_rows.append(dict(z_score=z_score_label, seed=seed, budget="reduced_n120",
                              min_ks_pvalue=min_ks_reduced, **{f"{k}": v for k, v in reduced.items()}))
        all_rows.append(dict(z_score=z_score_label, seed=seed, budget="confirm_n400",
                              min_ks_pvalue=min_ks_confirm, **{f"{k}": v for k, v in confirm.items()}))
        with open(SBI_LOGS / f"wu2003_posterior_sb_armc_{z_score_label}_seed{seed}{SUFFIX}.pkl", "wb") as f:
            import pickle
            pickle.dump({"posterior": posterior, "seed": seed, "z_score_x": z_score_x,
                         "structure": "S-B", "bank": f"wu2003_sbi_train_sb_paired_lognormal_true_n{N_TRAIN}{SUFFIX}.npz"}, f)

df = pd.DataFrame(all_rows)
df.to_csv(RESULTS / f"sb_armc_sbc_n{N_TRAIN}{SUFFIX}.csv", index=False)
print("\nSaved", RESULTS / f"sb_armc_sbc_n{N_TRAIN}{SUFFIX}.csv")
print(df[["z_score", "seed", "budget", "min_ks_pvalue"]].to_string(index=False))
