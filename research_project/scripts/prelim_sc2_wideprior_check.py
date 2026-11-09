"""FAST preliminary check for the S-C.2 WIDE MIXED PRIOR (alpha/beta_r/eta_col/
xi_reb ~ Uniform over their prior box, z_A0_eff lognormal) -- deliberately
small-scale (n_train=150, one SBC screen set n=40, single seed, reduced
classification budget) so it finishes in well under an hour, NOT a
confirmation-grade run. Purpose: get a fast directional read on whether
widening beta_r/eta_col/xi_reb (in addition to alpha, already fixed) helps
the same way it did for alpha alone, before committing to the full
n_train=1000 + N_SBC=120/400 + 10-seed pipeline
(regenerate_sc2_wideprior_bank.py / run_sc2_wideprior_seed_sweep.py /
run_sc2_wideprior_classification.py -- already written, ready to launch if
this preliminary signal looks good).

CAVEAT (per finding_sbc_power_eta_col memory): N=40 SBC is well below the
N>=400 threshold known to reliably catch eta_col miscalibration -- treat any
SBC "pass" here as a screen, not a certification. Classification budget is
also reduced (N_REPS=10 vs the standard 30, N_POST=100 vs 200).
"""
from __future__ import annotations

import pathlib
import pickle
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from cstr_sbi.recycle.priors import mixed_wide_prior
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.scenarios import list_closed_loop, FAULT_UNITS
from cstr_sbi.recycle.simulator import nominal_warm_start, deterministic_window, noisy_replicates
from cstr_sbi.recycle.summaries import compute_summaries, N_SUMMARIES_SC2
from sbi_pipeline import _structure_config, _simulate_summary

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SBI_LOGS = ROOT / "sbi-logs"
for d in (DATA, RESULTS, SBI_LOGS):
    d.mkdir(exist_ok=True)

STRUCTURE = "S-C.2"
NOISE_PCT = 0.003
N_TRAIN = 150
N_SBC_SCREEN = 40
ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
SEED = 0
SUFFIX = "_wideprior_prelim"

N_REPS, N_POST = 10, 100
THRESH, Z0_NOM = 0.85, 0.90

torch.manual_seed(20260905)
prior_wide = mixed_wide_prior()


def generate_bank(out_path, n_rows, seed, label):
    if out_path.exists():
        bank = np.load(out_path)
        print(f"[{label}] loading existing: {out_path}", flush=True)
        return np.asarray(bank["thetas"], dtype=np.float32), np.asarray(bank["summaries"], dtype=np.float32)
    ctrl, y0 = _structure_config(STRUCTURE)
    rng = np.random.default_rng(seed)
    theta_rows, summary_rows = [], []
    t0 = time.time()
    attempts = 0
    while len(theta_rows) < n_rows:
        attempts += 1
        th = prior_wide.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th)
        summary_rows.append(s)
        if len(theta_rows) % 10 == 0:
            elapsed = time.time() - t0
            accept_rate = len(theta_rows) / attempts
            print(f"[{label}] {len(theta_rows)}/{n_rows}, attempts={attempts} "
                  f"(accept_rate={accept_rate:.3f}), elapsed={elapsed:.0f}s", flush=True)
    theta_arr = np.asarray(theta_rows, dtype=np.float32)
    summary_arr = np.asarray(summary_rows, dtype=np.float32)
    assert summary_arr.shape[1] == N_SUMMARIES_SC2
    np.savez(out_path, thetas=theta_arr, summaries=summary_arr, attempts=attempts, seed=seed,
             structure=STRUCTURE, prior="mixed_wide_prior", scenario_specific_warm_start=True,
             note="PRELIMINARY small-scale wide-prior bank, not confirmation-grade")
    print(f"[{label}] saved: {out_path}, wall_s={time.time()-t0:.0f}, accept_rate={len(theta_arr)/attempts:.3f}", flush=True)
    return theta_arr, summary_arr


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


def classify_fault_recycle(samples):
    labels = [sample_fault_unit(row) for row in samples]
    probs = {u: float(np.mean([l == u for l in labels])) for u in FAULT_UNITS}
    predicted = max(probs, key=probs.__getitem__)
    return {"class": predicted, "probs": probs}


def compute_classification_metrics(preds, trues, classes=FAULT_UNITS):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    c2i = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(trues, preds):
        cm[c2i[true], c2i[pred]] += 1
    f1_per_class = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_per_class[cls] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    macro_f1 = float(np.mean(list(f1_per_class.values())))
    accuracy = float(np.diag(cm).sum() / cm.sum()) if cm.sum() > 0 else 0.0
    return {"confusion_matrix": cm, "class_names": classes, "per_class_f1": f1_per_class,
            "macro_f1": macro_f1, "accuracy": accuracy}


def main():
    t_all = time.time()
    print("=== STEP 1: generate small preliminary bank ===", flush=True)
    theta_train, summary_train = generate_bank(
        DATA / f"wu2003_sbi_train_sc2_true_n{N_TRAIN}{SUFFIX}.npz", N_TRAIN, seed=20260905001, label="train")
    theta_sbc, summary_sbc = generate_bank(
        RESULTS / f"sc2_sbc_set_n{N_SBC_SCREEN}{SUFFIX}.npz", N_SBC_SCREEN, seed=20260905002, label="sbc_screen")

    print("\n=== STEP 2: train single seed ===", flush=True)
    torch.manual_seed(SEED)
    inference = SNPE(prior=prior_wide, density_estimator=posterior_nn(
        model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS, z_score_x="none"))
    inference.append_simulations(torch.tensor(theta_train, dtype=torch.float32),
                                  torch.tensor(summary_train, dtype=torch.float32))
    t0 = time.time()
    trained = inference.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1,
                               stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    posterior = inference.build_posterior(trained)
    train_wall_s = time.time() - t0
    print(f"trained in {train_wall_s:.0f}s", flush=True)
    posterior_path = SBI_LOGS / f"wu2003_posterior_sc2_true_n{N_TRAIN}_zscorenone_seed{SEED}{SUFFIX}.pkl"
    with open(posterior_path, "wb") as f:
        pickle.dump({"posterior": posterior, "structure": "S-C.2", "n_train": N_TRAIN, "torch_seed": SEED,
                     "z_score_x": "none", "train_wall_s": train_wall_s, "prior": "mixed_wide_prior",
                     "note": "PRELIMINARY, small-scale, single-seed -- not for promotion"}, f)

    print("\n=== STEP 3: SBC screen (N=40, NOT confirmation-grade) ===", flush=True)
    torch.manual_seed(500001)
    ranks = {name: [] for name in PARAM_NAMES}
    for th, x_obs in zip(theta_sbc, summary_sbc):
        with torch.no_grad():
            samp = posterior.sample((N_SBC_SCREEN,), x=torch.tensor(x_obs, dtype=torch.float32),
                                     show_progress_bars=False, reject_outside_prior=False).detach().cpu().numpy()
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
    sbc_rows = []
    for name in PARAM_NAMES:
        r = np.asarray(ranks[name], dtype=float)
        ks = scipy_stats.ks_1samp(r / N_SBC_SCREEN, scipy_stats.uniform.cdf)
        sbc_rows.append({"param": name, "ks_pvalue": float(ks.pvalue)})
        print(f"  {name:10s} ks_pvalue={ks.pvalue:.4g}", flush=True)
    min_ks = min(row["ks_pvalue"] for row in sbc_rows)
    print(f"  min_ks_pvalue = {min_ks:.4g} ({'PASS (screen only)' if min_ks > 0.05 else 'fail'})", flush=True)
    pd.DataFrame(sbc_rows).to_csv(RESULTS / f"sc2_wideprior_prelim_sbc_screen_n{N_SBC_SCREEN}.csv", index=False)

    print("\n=== STEP 4: reduced-budget classification (N_REPS=10, N_POST=100) ===", flush=True)
    torch.manual_seed(20260705)
    y0_sb = nominal_warm_start("S-B")
    scenarios = list_closed_loop()
    rep_rows, per_scenario_rows = [], []
    for sc in scenarios:
        theta_np = np.asarray(sc.theta(), dtype=np.float32)
        true_class = sc.fault_unit()
        t_h, raw_obs = deterministic_window(sc, structure="S-B", y0=y0_sb)
        t_h_arr = np.asarray(t_h)
        rng = np.random.default_rng(20260705)
        reps = noisy_replicates(raw_obs, n_replicates=N_REPS, rng=rng, noise_pct=NOISE_PCT)
        pooled_samples, rep_predicted = [], []
        for i in range(N_REPS):
            s = compute_summaries(reps[i], STRUCTURE, t_h_arr, theta=theta_np, rng=rng)
            with torch.no_grad():
                samp = posterior.sample((N_POST,), x=torch.tensor(s, dtype=torch.float32),
                                         show_progress_bars=False, reject_outside_prior=False).detach().cpu().numpy()
            pooled_samples.append(samp)
            rep_result = classify_fault_recycle(samp)
            rep_predicted.append(rep_result["class"])
            rep_rows.append({"scenario": sc.name, "replicate": i, "true_class": true_class,
                              "pred_class": rep_result["class"], **rep_result["probs"]})
        pooled = np.concatenate(pooled_samples, axis=0)
        pooled_result = classify_fault_recycle(pooled)
        majority_class = Counter(rep_predicted).most_common(1)[0][0]
        print(f"  {sc.name:22s} true={true_class:8s}  pooled_pred={pooled_result['class']:8s}  "
              f"majority_pred={majority_class:8s}", flush=True)
        per_scenario_rows.append({"scenario": sc.name, "true_class": true_class,
                                   "pooled_pred": pooled_result["class"], "majority_pred": majority_class,
                                   **pooled_result["probs"]})

    rep_df = pd.DataFrame(rep_rows)
    scenario_df = pd.DataFrame(per_scenario_rows)
    rep_df.to_csv(RESULTS / "sc2_wideprior_prelim_classification_per_replicate.csv", index=False)
    scenario_df.to_csv(RESULTS / "sc2_wideprior_prelim_classification_per_scenario.csv", index=False)

    metrics_rep = compute_classification_metrics(rep_df["pred_class"].tolist(), rep_df["true_class"].tolist())
    metrics_scenario = compute_classification_metrics(scenario_df["pooled_pred"].tolist(), scenario_df["true_class"].tolist())
    print(f"\n=== PRELIMINARY RESULT (small-scale, single-seed, NOT confirmation-grade) ===", flush=True)
    print(f"per-replicate: macro-F1={metrics_rep['macro_f1']:.3f}  accuracy={metrics_rep['accuracy']:.3f}", flush=True)
    print(f"per-class F1: {({k: round(v,3) for k,v in metrics_rep['per_class_f1'].items()})}", flush=True)
    print(f"per-scenario pooled: macro-F1={metrics_scenario['macro_f1']:.3f}  accuracy={metrics_scenario['accuracy']:.3f}", flush=True)
    print(f"\nAll done in {time.time()-t_all:.0f}s total.", flush=True)


if __name__ == "__main__":
    main()
