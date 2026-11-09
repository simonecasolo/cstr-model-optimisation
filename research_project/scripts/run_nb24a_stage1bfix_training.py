"""Run notebook 24a's Sections 4-8 (SNPE training + SBC, baseline and the
z_score_x='none' fix) against the Stage-1b-fixed S-C bank (`N_TRAIN=1000`,
`SUFFIX='_stage1bfix'`). Mirrors the notebook's cells exactly (same functions,
same seeds, same SBC procedure) so results are directly comparable to the old,
contaminated-bank results already in the notebook.

Run standalone (not via nbconvert) per this notebook's own established
hang-avoidance precedent; the notebook's own cells will just load the
resulting posteriors/CSVs afterward (same idempotent pattern used for the
bank-generation cells).

Question this answers: does the alpha/xi_reb shrinkage nb24a diagnosed as a
structural "flow objective dominated by the near-delta z_A0_eff dimension"
problem persist on Stage-1b-fixed (clean) ground truth, or was it partly/
wholly an artifact of the warm-start contamination?
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
FIGURES = ROOT / "figures"
for p in [DATA, RESULTS, SBI_LOGS, FIGURES]:
    p.mkdir(exist_ok=True)

STRUCTURE = "S-C"
ARCH = "zuko_nsf"
HIDDEN = 60
TRANSFORMS = 3
MAX_EPOCHS = 120
STOP_AFTER_EPOCHS = 12

N_TRAIN = 1000
SUFFIX = "_stage1bfix"
N_SBC_REDUCED = 120
N_SBC_CONFIRM = 400
N_POST = 120
SEEDS = [0, 1, 2]

prior_logn = truncated_lognormal_5d()

SC_BANK_PATH = DATA / f"wu2003_sbi_train_sc_lognormal_true_n{N_TRAIN}{SUFFIX}.npz"
SC_SBC_SET_PATH = RESULTS / f"24a_sysII_sc_lognormal_sbc_set_n{N_SBC_REDUCED}{SUFFIX}.npz"
SC_SBC_CONFIRM_PATH = RESULTS / f"24a_sysII_sc_lognormal_sbc_set_n{N_SBC_CONFIRM}{SUFFIX}.npz"

bank = np.load(SC_BANK_PATH)
theta_train, summary_train_sc = np.asarray(bank["thetas"], dtype=np.float32), np.asarray(bank["summaries"], dtype=np.float32)
sbc_r = np.load(SC_SBC_SET_PATH)
theta_sbc, summary_sbc_sc = np.asarray(sbc_r["thetas"], dtype=np.float32), np.asarray(sbc_r["summaries"], dtype=np.float32)
sbc_c = np.load(SC_SBC_CONFIRM_PATH)
theta_sbc_c, summary_sbc_c = np.asarray(sbc_c["thetas"], dtype=np.float32), np.asarray(sbc_c["summaries"], dtype=np.float32)
print(f"theta_train={theta_train.shape} theta_sbc={theta_sbc.shape} theta_sbc_c={theta_sbc_c.shape}")


def train_sc_posterior(torch_seed, z_score_x=None):
    torch.manual_seed(torch_seed)
    kwargs = dict(model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS)
    if z_score_x is not None:
        kwargs["z_score_x"] = z_score_x
    inference = SNPE(prior=prior_logn, density_estimator=posterior_nn(**kwargs))
    inference.append_simulations(torch.tensor(theta_train, dtype=torch.float32), torch.tensor(summary_train_sc, dtype=torch.float32))
    t0 = time.time()
    trained = inference.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1, stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    return inference.build_posterior(trained), time.time() - t0


def run_lognormal_sbc(posterior, theta_sbc_set, summary_sbc_set, seed, n_post=N_POST):
    torch.manual_seed(seed)
    ranks = {name: [] for name in PARAM_NAMES}
    sample_seconds = []
    for th, x_obs in zip(theta_sbc_set, summary_sbc_set):
        t_sample = time.time()
        with torch.no_grad():
            samp = posterior.sample((n_post,), x=torch.tensor(x_obs, dtype=torch.float32),
                                     show_progress_bars=False, reject_outside_prior=False).detach().cpu().numpy()
        sample_seconds.append(time.time() - t_sample)
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
    results = {"n_sbc": int(len(theta_sbc_set)), "n_post": int(n_post),
               "sample_seconds_mean": float(np.mean(sample_seconds))}
    for name in PARAM_NAMES:
        r = np.asarray(ranks[name], dtype=float)
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        results[name] = {"ks_pvalue": float(ks.pvalue), "mean_rank_frac": float(r.mean() / n_post)}
    results["min_ks_pvalue"] = min(results[name]["ks_pvalue"] for name in PARAM_NAMES)
    return results


def flatten_result(row):
    rec = {k: row[k] for k in ["seed", "n_train", "train_wall_s"] if k in row}
    rec["min_ks_pvalue"] = row["sbc"]["min_ks_pvalue"]
    for name in PARAM_NAMES:
        rec[f"{name}_ks_pvalue"] = row["sbc"][name]["ks_pvalue"]
        rec[f"{name}_mean_rank_frac"] = row["sbc"][name]["mean_rank_frac"]
    return rec


def train_and_sbc(variant_tag, z_score_x, sbc_set_pair, confirm_set_pair):
    """variant_tag: 'baseline' or 'zscorenone'. Returns (reduced_df, confirm_df, posteriors)."""
    posteriors = {}
    reduced_results = []
    for seed in SEEDS:
        posterior_path = SBI_LOGS / f"wu2003_posterior_sc_lognormal_true_n{N_TRAIN}_{variant_tag}_seed{seed}{SUFFIX}.pkl"
        if posterior_path.exists():
            with open(posterior_path, "rb") as f:
                payload = pickle.load(f)
            posterior = payload["posterior"]
            train_wall_s = float(payload.get("train_wall_s", 0.0))
            print(f"[{variant_tag}] loaded existing posterior: {posterior_path.name}")
        else:
            posterior, train_wall_s = train_sc_posterior(seed, z_score_x=z_score_x)
            with open(posterior_path, "wb") as f:
                pickle.dump({"posterior": posterior, "structure": "S-C", "n_train": N_TRAIN,
                             "torch_seed": seed, "z_score_x": z_score_x,
                             "prior": "truncated_lognormal_centered_nominal",
                             "bank_path": str(SC_BANK_PATH), "train_wall_s": train_wall_s,
                             "stage1b_fix": "v2"}, f)
            print(f"[{variant_tag}] saved posterior: {posterior_path.name} (train_wall_s={train_wall_s:.1f})")
        posteriors[seed] = posterior

        theta_s, summary_s = sbc_set_pair
        sbc = run_lognormal_sbc(posterior, theta_s, summary_s, seed=430000 + seed)
        reduced_results.append({"seed": seed, "n_train": N_TRAIN, "train_wall_s": train_wall_s, "sbc": sbc})
        print(f"  [{variant_tag}] seed {seed} reduced: min_ks_pvalue={sbc['min_ks_pvalue']:.4e}")

    reduced_df = pd.DataFrame([flatten_result(r) for r in reduced_results])
    reduced_csv = RESULTS / f"24a_sysII_sc_{variant_tag}_reduced_sbc_n{N_SBC_REDUCED}_ntrain{N_TRAIN}{SUFFIX}.csv"
    reduced_df.to_csv(reduced_csv, index=False)
    print(f"[{variant_tag}] saved {reduced_csv}")

    confirm_results = []
    theta_c, summary_c = confirm_set_pair
    for seed in SEEDS:
        posterior = posteriors[seed]
        sbc = run_lognormal_sbc(posterior, theta_c, summary_c, seed=440000 + seed)
        confirm_results.append({"seed": seed, "n_train": N_TRAIN, "sbc": sbc})
        print(f"  [{variant_tag}] seed {seed} confirm: min_ks_pvalue={sbc['min_ks_pvalue']:.4e}")

    confirm_df = pd.DataFrame([flatten_result(r) for r in confirm_results])
    confirm_csv = RESULTS / f"24a_sysII_sc_{variant_tag}_confirm_sbc_n{N_SBC_CONFIRM}_ntrain{N_TRAIN}{SUFFIX}.csv"
    confirm_df.to_csv(confirm_csv, index=False)
    print(f"[{variant_tag}] saved {confirm_csv}")

    n_pass_r = int((reduced_df["min_ks_pvalue"] > 0.05).sum())
    n_pass_c = int((confirm_df["min_ks_pvalue"] > 0.05).sum())
    print(f"[{variant_tag}] {n_pass_r}/{len(SEEDS)} pass reduced (N={N_SBC_REDUCED}), "
          f"{n_pass_c}/{len(SEEDS)} pass confirm (N={N_SBC_CONFIRM})")
    return reduced_df, confirm_df, posteriors


def main():
    t0 = time.time()
    print("=== Variant: baseline (identity theta-space, default z-scoring) ===")
    base_reduced, base_confirm, _ = train_and_sbc(
        "baseline", None, (theta_sbc, summary_sbc_sc), (theta_sbc_c, summary_sbc_c))

    print("\n=== Variant: z_score_x='none' fix ===")
    fixed_reduced, fixed_confirm, _ = train_and_sbc(
        "zscorenone", "none", (theta_sbc, summary_sbc_sc), (theta_sbc_c, summary_sbc_c))

    print(f"\nAll training+SBC done in {time.time()-t0:.0f}s.")

    print("\n=== Per-parameter confirm-budget breakdown (baseline) ===")
    for _, row in base_confirm.iterrows():
        print(f"  seed {int(row.seed)}: " + "  ".join(
            f"{name}={row[f'{name}_ks_pvalue']:.3g}" for name in PARAM_NAMES))

    print("\n=== Per-parameter confirm-budget breakdown (z_score_x='none') ===")
    for _, row in fixed_confirm.iterrows():
        print(f"  seed {int(row.seed)}: " + "  ".join(
            f"{name}={row[f'{name}_ks_pvalue']:.3g}" for name in PARAM_NAMES))


if __name__ == "__main__":
    main()
