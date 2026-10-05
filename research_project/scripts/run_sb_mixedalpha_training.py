"""Train + SBC-evaluate S-B on the alpha-widened-uniform-prior bank
(`scripts/generate_sb_mixedalpha_bank.py`). Same protocol (zuko_nsf 60/3,
baseline + z_score_x='none', 3 seeds, reduced N=40 then confirm N=100 SBC)
as `scripts/run_sb_arm_c_training.py`, just a different training prior.
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
from cstr_sbi.recycle.priors import mixed_alpha_uniform_prior

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SBI_LOGS = ROOT / "sbi-logs"

ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
N_TRAIN = 200
SUFFIX = "_mixedalpha_stage1bfix"
SEEDS = [0, 1, 2]

prior = mixed_alpha_uniform_prior()

bank = np.load(DATA / f"wu2003_sbi_train_sb_n{N_TRAIN}{SUFFIX}.npz")
theta_train = np.asarray(bank["thetas"], dtype=np.float32)
summary_train = np.asarray(bank["summaries"], dtype=np.float32)

r40 = np.load(RESULTS / f"sb_sbc_set_n40{SUFFIX}.npz")
r100 = np.load(RESULTS / f"sb_sbc_set_n100{SUFFIX}.npz")
theta_sbc, summary_sbc = np.asarray(r40["thetas"], dtype=np.float32), np.asarray(r40["summaries"], dtype=np.float32)
theta_sbc_c, summary_sbc_c = np.asarray(r100["thetas"], dtype=np.float32), np.asarray(r100["summaries"], dtype=np.float32)
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
        results[f"{name}_slope"] = float(slope)
    return results


all_rows = []
for z_score_label, z_score_x in [("baseline", None), ("zscorenone", "none")]:
    for seed in SEEDS:
        posterior, train_s = train_posterior(seed, z_score_x=z_score_x)
        reduced = run_sbc(posterior, theta_sbc, summary_sbc, seed, n_post=40)
        confirm = run_sbc(posterior, theta_sbc_c, summary_sbc_c, seed, n_post=100)
        print(f"[{z_score_label} seed={seed}] trained in {train_s:.1f}s")
        print("  reduced (N=40):", {k: round(v, 4) for k, v in reduced.items() if k.endswith("_ks_p")})
        print("  confirm (N=100):", {k: round(v, 4) for k, v in confirm.items() if k.endswith("_ks_p")})
        min_ks_reduced = min(reduced[f"{n}_ks_p"] for n in PARAM_NAMES)
        min_ks_confirm = min(confirm[f"{n}_ks_p"] for n in PARAM_NAMES)
        all_rows.append(dict(z_score=z_score_label, seed=seed, budget="reduced_n40",
                              min_ks_pvalue=min_ks_reduced, **reduced))
        all_rows.append(dict(z_score=z_score_label, seed=seed, budget="confirm_n100",
                              min_ks_pvalue=min_ks_confirm, **confirm))
        with open(SBI_LOGS / f"wu2003_posterior_sb_mixedalpha_{z_score_label}_seed{seed}.pkl", "wb") as f:
            pickle.dump({"posterior": posterior, "seed": seed, "z_score_x": z_score_x,
                         "structure": "S-B", "prior": "mixed_alpha_uniform",
                         "bank": f"wu2003_sbi_train_sb_n{N_TRAIN}{SUFFIX}.npz"}, f)

df = pd.DataFrame(all_rows)
df.to_csv(RESULTS / f"sb_mixedalpha_sbc_n{N_TRAIN}.csv", index=False)
print("\nSaved", RESULTS / f"sb_mixedalpha_sbc_n{N_TRAIN}.csv")
print(df[["z_score", "seed", "budget", "min_ks_pvalue"]].to_string(index=False))
