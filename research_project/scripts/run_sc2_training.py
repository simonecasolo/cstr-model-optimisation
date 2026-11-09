"""Train + SBC-evaluate on the S-C.2 bank (S-C + 2 xi_reb-sharpening features,
`scripts/augment_bank_to_sc2.py`, zero new simulation). Same protocol as
`scripts/run_nb24a_stage1bfix_training.py` (baseline + z_score_x='none', 3
seeds, reduced+confirm SBC) -- the only change is the bank/SBC-set files (74-D
S-C.2 instead of 72-D S-C). Answers: does sharpening xi_reb's own signal (no
new sensor, no architecture change) improve its SBC calibration, and does it
have any knock-on effect on alpha/eta_col given the residual couplings the
new features also reduce?
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

bank = np.load(DATA / f"wu2003_sbi_train_sc2_lognormal_true_n{N_TRAIN}{SUFFIX}.npz")
theta_train, summary_train = np.asarray(bank["thetas"], dtype=np.float32), np.asarray(bank["summaries"], dtype=np.float32)
r120 = np.load(RESULTS / f"24a_sysII_sc2_lognormal_sbc_set_n120{SUFFIX}.npz")
theta_sbc, summary_sbc = np.asarray(r120["thetas"], dtype=np.float32), np.asarray(r120["summaries"], dtype=np.float32)
r400 = np.load(RESULTS / f"24a_sysII_sc2_lognormal_sbc_set_n400{SUFFIX}.npz")
theta_sbc_c, summary_sbc_c = np.asarray(r400["thetas"], dtype=np.float32), np.asarray(r400["summaries"], dtype=np.float32)
print(f"theta_train={theta_train.shape} summary_train={summary_train.shape} "
      f"theta_sbc={theta_sbc.shape} theta_sbc_c={theta_sbc_c.shape}")


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
    for th, x_obs in zip(theta_sbc_set, summary_sbc_set):
        with torch.no_grad():
            samp = posterior.sample((n_post,), x=torch.tensor(x_obs, dtype=torch.float32),
                                     show_progress_bars=False, reject_outside_prior=False).detach().cpu().numpy()
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
    results = {"n_sbc": int(len(theta_sbc_set)), "n_post": int(n_post)}
    for name in PARAM_NAMES:
        r = np.asarray(ranks[name], dtype=float)
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        results[name] = {"ks_pvalue": float(ks.pvalue), "mean_rank_frac": float(r.mean() / n_post)}
    results["min_ks_pvalue"] = min(results[name]["ks_pvalue"] for name in PARAM_NAMES)
    return results


def flatten(row):
    rec = {k: row[k] for k in ["seed", "train_wall_s"] if k in row}
    rec["min_ks_pvalue"] = row["sbc"]["min_ks_pvalue"]
    for name in PARAM_NAMES:
        rec[f"{name}_ks_pvalue"] = row["sbc"][name]["ks_pvalue"]
        rec[f"{name}_mean_rank_frac"] = row["sbc"][name]["mean_rank_frac"]
    return rec


def train_and_sbc(variant_tag, z_score_x):
    posteriors = {}
    reduced_rows, confirm_rows = [], []
    for seed in SEEDS:
        posterior_path = SBI_LOGS / f"wu2003_posterior_sc2_lognormal_true_n{N_TRAIN}_{variant_tag}_seed{seed}{SUFFIX}.pkl"
        if posterior_path.exists():
            with open(posterior_path, "rb") as f:
                payload = pickle.load(f)
            posterior, train_wall_s = payload["posterior"], float(payload.get("train_wall_s", 0.0))
        else:
            posterior, train_wall_s = train_posterior(seed, z_score_x=z_score_x)
            with open(posterior_path, "wb") as f:
                pickle.dump({"posterior": posterior, "structure": "S-C.2", "n_train": N_TRAIN,
                             "torch_seed": seed, "z_score_x": z_score_x, "train_wall_s": train_wall_s,
                             "note": "S-C.2 pilot, xi_reb-sharpening features"}, f)
        posteriors[seed] = posterior

        sbc_r = run_sbc(posterior, theta_sbc, summary_sbc, seed=430000 + seed)
        sbc_c = run_sbc(posterior, theta_sbc_c, summary_sbc_c, seed=440000 + seed)
        print(f"  [{variant_tag}] seed {seed}: train_wall_s={train_wall_s:.1f}  "
              f"reduced min_ks={sbc_r['min_ks_pvalue']:.4g}  confirm min_ks={sbc_c['min_ks_pvalue']:.4g}")
        print(f"    reduced: " + "  ".join(f"{n}={sbc_r[n]['ks_pvalue']:.3g}" for n in PARAM_NAMES))
        print(f"    confirm: " + "  ".join(f"{n}={sbc_c[n]['ks_pvalue']:.3g}" for n in PARAM_NAMES))
        reduced_rows.append({"seed": seed, "train_wall_s": train_wall_s, "sbc": sbc_r})
        confirm_rows.append({"seed": seed, "sbc": sbc_c})

    reduced_df = pd.DataFrame([flatten(r) for r in reduced_rows])
    confirm_df = pd.DataFrame([flatten(r) for r in confirm_rows])
    reduced_df.to_csv(RESULTS / f"sc2_{variant_tag}_reduced_sbc_n120_ntrain{N_TRAIN}{SUFFIX}.csv", index=False)
    confirm_df.to_csv(RESULTS / f"sc2_{variant_tag}_confirm_sbc_n400_ntrain{N_TRAIN}{SUFFIX}.csv", index=False)
    n_pass_r = int((reduced_df["min_ks_pvalue"] > 0.05).sum())
    n_pass_c = int((confirm_df["min_ks_pvalue"] > 0.05).sum())
    print(f"[{variant_tag}] {n_pass_r}/{len(SEEDS)} pass reduced, {n_pass_c}/{len(SEEDS)} pass confirm")
    return reduced_df, confirm_df


def main():
    t0 = time.time()
    print("=== Variant: baseline (default z-scoring) ===")
    train_and_sbc("baseline", None)
    print("\n=== Variant: z_score_x='none' ===")
    train_and_sbc("zscorenone", "none")
    print(f"\nAll done in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
