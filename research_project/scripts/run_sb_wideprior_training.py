"""Train + SBC-evaluate S-B on the full, real wide-prior bank
(`scripts/generate_sb_wideprior_bank.py`: n_train=5000, all 5 parameters
widened to uniform via `mixed_wide_prior(uniform_idx=(0,1,2,3,4))`, clean
post-warm-start-fix data). This is the real/final run, not a pilot -- 5
seeds (not 3), matching the project's pre-registered "Stage 4" confirmation
protocol (`system_ii_sbi_solidification_plan.md`): both SBC sets (reduced
N=120, confirm N=400), per-parameter KS p-value, anti-shrinkage regression
slope (promoted to a standing gate after it caught `alpha`'s shrinkage
earlier in this project), and 90% credible-interval coverage.
"""
from __future__ import annotations

import pathlib
import pickle
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
from cstr_sbi.recycle.priors import mixed_wide_prior

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SBI_LOGS = ROOT / "sbi-logs"

ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
N_TRAIN = 5000
SUFFIX = "_wideprior_stage1bfix"
SEEDS = [0, 1, 2, 3, 4]

prior = mixed_wide_prior(uniform_idx=(0, 1, 2, 3, 4))

bank = np.load(DATA / f"wu2003_sbi_train_sb_n{N_TRAIN}{SUFFIX}.npz")
theta_train = np.asarray(bank["thetas"], dtype=np.float32)
summary_train = np.asarray(bank["summaries"], dtype=np.float32)

r120 = np.load(RESULTS / f"sb_sbc_set_n120{SUFFIX}.npz")
r400 = np.load(RESULTS / f"sb_sbc_set_n400{SUFFIX}.npz")
theta_sbc, summary_sbc = np.asarray(r120["thetas"], dtype=np.float32), np.asarray(r120["summaries"], dtype=np.float32)
theta_sbc_c, summary_sbc_c = np.asarray(r400["thetas"], dtype=np.float32), np.asarray(r400["summaries"], dtype=np.float32)
print(f"theta_train={theta_train.shape} summary_train={summary_train.shape} "
      f"theta_sbc={theta_sbc.shape} theta_sbc_c={theta_sbc_c.shape}")


def train_posterior(torch_seed, z_score_x=None):
    torch.manual_seed(torch_seed)
    kwargs = dict(model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS)
    if z_score_x is not None:
        kwargs["z_score_x"] = z_score_x
    inference = SNPE(prior=prior, density_estimator=posterior_nn(**kwargs))
    inference.append_simulations(torch.tensor(theta_train, dtype=torch.float32), torch.tensor(summary_train, dtype=torch.float32))
    t0 = time.time()
    trained = inference.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1, stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    return inference.build_posterior(trained), time.time() - t0


def run_sbc(posterior, theta_sbc_set, summary_sbc_set, seed, n_post):
    torch.manual_seed(seed)
    ranks = {name: [] for name in PARAM_NAMES}
    means = {name: [] for name in PARAM_NAMES}
    covered90 = {name: [] for name in PARAM_NAMES}
    for th, x_obs in zip(theta_sbc_set, summary_sbc_set):
        with torch.no_grad():
            samp = posterior.sample((n_post,), x=torch.tensor(x_obs, dtype=torch.float32),
                                     show_progress_bars=False, reject_outside_prior=False).detach().cpu().numpy()
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
            means[name].append(float(samp[:, k].mean()))
            lo, hi = np.quantile(samp[:, k], [0.05, 0.95])
            covered90[name].append(bool(lo <= th[k] <= hi))
    results = {"n_sbc": int(len(theta_sbc_set)), "n_post": int(n_post)}
    for name in PARAM_NAMES:
        r = np.asarray(ranks[name], dtype=float)
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        true_vals = theta_sbc_set[:, PARAM_NAMES.index(name)]
        est_vals = np.asarray(means[name])
        slope = np.polyfit(true_vals, est_vals, 1)[0] if np.std(true_vals) > 0 else float("nan")
        results[f"{name}_ks_p"] = float(ks.pvalue)
        results[f"{name}_slope"] = float(slope)
        results[f"{name}_coverage90"] = float(np.mean(covered90[name]))
    return results


all_rows = []
for z_score_label, z_score_x in [("baseline", None), ("zscorenone", "none")]:
    for seed in SEEDS:
        posterior, train_s = train_posterior(seed, z_score_x=z_score_x)
        reduced = run_sbc(posterior, theta_sbc, summary_sbc, seed, n_post=120)
        confirm = run_sbc(posterior, theta_sbc_c, summary_sbc_c, seed, n_post=400)
        print(f"[{z_score_label} seed={seed}] trained in {train_s:.1f}s")
        print("  reduced (N=120):", {k: round(v, 4) for k, v in reduced.items() if k.endswith("_ks_p")})
        print("  confirm (N=400):", {k: round(v, 4) for k, v in confirm.items() if k.endswith("_ks_p")})
        print("  confirm slopes:", {k: round(v, 3) for k, v in confirm.items() if k.endswith("_slope")})
        print("  confirm coverage90:", {k: round(v, 3) for k, v in confirm.items() if k.endswith("_coverage90")})
        min_ks_reduced = min(reduced[f"{n}_ks_p"] for n in PARAM_NAMES)
        min_ks_confirm = min(confirm[f"{n}_ks_p"] for n in PARAM_NAMES)
        all_pass_confirm = all(confirm[f"{n}_ks_p"] > 0.05 for n in PARAM_NAMES)
        print(f"  ALL 5 PASS (confirm, p>0.05): {all_pass_confirm}")
        all_rows.append(dict(z_score=z_score_label, seed=seed, budget="reduced_n120",
                              min_ks_pvalue=min_ks_reduced, all_pass=None, **reduced))
        all_rows.append(dict(z_score=z_score_label, seed=seed, budget="confirm_n400",
                              min_ks_pvalue=min_ks_confirm, all_pass=all_pass_confirm, **confirm))
        with open(SBI_LOGS / f"wu2003_posterior_sb_wideprior_{z_score_label}_seed{seed}.pkl", "wb") as f:
            pickle.dump({"posterior": posterior, "seed": seed, "z_score_x": z_score_x,
                         "structure": "S-B", "prior": "mixed_wide_uniform_idx_01234",
                         "bank": f"wu2003_sbi_train_sb_n{N_TRAIN}{SUFFIX}.npz"}, f)

df = pd.DataFrame(all_rows)
df.to_csv(RESULTS / f"sb_wideprior_sbc_n{N_TRAIN}.csv", index=False)
print("\nSaved", RESULTS / f"sb_wideprior_sbc_n{N_TRAIN}.csv")
print(df[["z_score", "seed", "budget", "min_ks_pvalue", "all_pass"]].to_string(index=False))

n_pass = df[(df["budget"] == "confirm_n400") & (df["all_pass"] == True)].shape[0]
print(f"\n{n_pass}/{len(SEEDS)*2} (z_score x seed) combinations pass all 5 parameters at confirm budget.")
