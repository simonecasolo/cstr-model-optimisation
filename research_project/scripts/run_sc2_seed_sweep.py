"""Wider seed sweep on the EXISTING n=1000 S-C.2 bank (S-C.2, z_score_x='none',
60/3 -- the best setup found so far) to test whether "at least one usable,
calibrated posterior" is reachable cheaply via more seeds rather than paying
for a n=5000 bank regeneration. The n=1000 pilot found a 1/3 hit rate (seed 0
fully passed); this sweep extends to more seeds and validates any candidate
against BOTH independent SBC sets (n=120 and n=400), not just whichever one
it was best on -- avoids reporting a cherry-picked fluke
(`finding_sbc_power_eta_col`'s lesson).
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

ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
N_TRAIN = 1000
SUFFIX = "_stage1bfix"
N_POST = 120
SEEDS = list(range(10))   # widen from [0,1,2] to 10 seeds

prior_logn = truncated_lognormal_5d()

bank = np.load(DATA / f"wu2003_sbi_train_sc2_lognormal_true_n{N_TRAIN}{SUFFIX}.npz")
theta_train, summary_train = np.asarray(bank["thetas"], dtype=np.float32), np.asarray(bank["summaries"], dtype=np.float32)
r120 = np.load(RESULTS / f"24a_sysII_sc2_lognormal_sbc_set_n120{SUFFIX}.npz")
theta_sbc, summary_sbc = np.asarray(r120["thetas"], dtype=np.float32), np.asarray(r120["summaries"], dtype=np.float32)
r400 = np.load(RESULTS / f"24a_sysII_sc2_lognormal_sbc_set_n400{SUFFIX}.npz")
theta_sbc_c, summary_sbc_c = np.asarray(r400["thetas"], dtype=np.float32), np.asarray(r400["summaries"], dtype=np.float32)
print(f"Seed sweep: {SEEDS}, S-C.2, z_score_x='none', {HIDDEN}/{TRANSFORMS}")


def train_posterior(torch_seed):
    torch.manual_seed(torch_seed)
    inference = SNPE(prior=prior_logn, density_estimator=posterior_nn(
        model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS, z_score_x="none"))
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
        results[name] = {"ks_pvalue": float(ks.pvalue)}
    results["min_ks_pvalue"] = min(results[name]["ks_pvalue"] for name in PARAM_NAMES)
    return results


def main():
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        posterior_path = SBI_LOGS / f"wu2003_posterior_sc2_lognormal_true_n{N_TRAIN}_zscorenone_seed{seed}{SUFFIX}.pkl"
        if posterior_path.exists():
            with open(posterior_path, "rb") as f:
                payload = pickle.load(f)
            posterior, train_wall_s = payload["posterior"], float(payload.get("train_wall_s", 0.0))
        else:
            posterior, train_wall_s = train_posterior(seed)
            with open(posterior_path, "wb") as f:
                pickle.dump({"posterior": posterior, "structure": "S-C.2", "n_train": N_TRAIN,
                             "torch_seed": seed, "z_score_x": "none", "train_wall_s": train_wall_s,
                             "note": "seed sweep for usable-model search"}, f)

        sbc_r = run_sbc(posterior, theta_sbc, summary_sbc, seed=430000 + seed)
        sbc_c = run_sbc(posterior, theta_sbc_c, summary_sbc_c, seed=440000 + seed)
        pass_r = sbc_r["min_ks_pvalue"] > 0.05
        pass_c = sbc_c["min_ks_pvalue"] > 0.05
        both = pass_r and pass_c
        print(f"seed {seed}: train_wall_s={train_wall_s:.1f}  "
              f"reduced_min={sbc_r['min_ks_pvalue']:.3g} ({'PASS' if pass_r else 'fail'})  "
              f"confirm_min={sbc_c['min_ks_pvalue']:.3g} ({'PASS' if pass_c else 'fail'})  "
              f"{'*** BOTH PASS ***' if both else ''}", flush=True)
        rows.append({"seed": seed, "train_wall_s": train_wall_s,
                      "reduced_min_ks": sbc_r["min_ks_pvalue"], "confirm_min_ks": sbc_c["min_ks_pvalue"],
                      "pass_reduced": pass_r, "pass_confirm": pass_c, "pass_both": both,
                      **{f"{n}_reduced_ks": sbc_r[n]["ks_pvalue"] for n in PARAM_NAMES},
                      **{f"{n}_confirm_ks": sbc_c[n]["ks_pvalue"] for n in PARAM_NAMES}})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"sc2_zscorenone_seedsweep_n{len(SEEDS)}{SUFFIX}.csv", index=False)
    n_both = int(df["pass_both"].sum())
    print(f"\n{n_both}/{len(SEEDS)} seeds pass BOTH independent SBC sets.")
    print(f"All done in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
