"""Classification check for the alpha-widened-prior S-B posteriors
(`scripts/run_sb_mixedalpha_training.py`) against the full 14-scenario
catalogue -- the decisive test of whether widening just `alpha`'s training
prior fixes the catastrophic classification failure found for the narrow
lognormal-prior posterior (`scripts/eval_sb_armc_classification.py`:
macro-F1=0.089).
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cstr_sbi.recycle.scenarios import list_closed_loop
from cstr_sbi.recycle.simulator import nominal_warm_start, deterministic_window, noisy_replicates
from cstr_sbi.recycle.summaries import compute_summaries
from cstr_sbi.recycle.physics import PARAM_NAMES

SBI_LOGS = ROOT / "sbi-logs"

THRESH, Z0_NOM = 0.85, 0.90
FAULT_UNITS = ["healthy", "reactor", "column", "feed", "multi"]


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


def metrics(preds, trues, classes=FAULT_UNITS):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    c2i = {c: i for i, c in enumerate(classes)}
    for t, p in zip(trues, preds):
        cm[c2i[t], c2i[p]] += 1
    f1 = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1[cls] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return cm, f1, float(np.mean(list(f1.values()))), float(np.trace(cm) / cm.sum())


for z_score_label, seed in [("zscorenone", 1), ("zscorenone", 2), ("zscorenone", 0), ("baseline", 0)]:
    pkl_path = SBI_LOGS / f"wu2003_posterior_sb_mixedalpha_{z_score_label}_seed{seed}.pkl"
    with open(pkl_path, "rb") as f:
        posterior = pickle.load(f)["posterior"]
    y0_sb = nominal_warm_start("S-B")
    all_preds, all_trues, rows = [], [], []
    for sc in list_closed_loop():
        t_h, raw_obs = deterministic_window(sc, structure="S-B", y0=y0_sb)
        t_h_arr = np.asarray(t_h)
        rng = np.random.default_rng(900001 + sc.id)
        reps = noisy_replicates(np.asarray(raw_obs), n_replicates=10, rng=rng, noise_pct=0.003)
        true_theta = np.asarray(sc.theta())
        true_unit = sample_fault_unit(true_theta)
        rep_preds = []
        pooled = []
        for i in range(10):
            s = compute_summaries(reps[i], "S-B", t_h_arr)
            with torch.no_grad():
                samp = posterior.sample((150,), x=torch.tensor(s, dtype=torch.float32),
                                         show_progress_bars=False, reject_outside_prior=False).numpy()
            pooled.append(samp)
            pred, _ = classify(samp)
            rep_preds.append(pred)
            all_preds.append(pred)
            all_trues.append(true_unit)
        pooled_arr = np.concatenate(pooled, axis=0)
        means = pooled_arr.mean(axis=0)
        rows.append((sc.name, true_unit, np.mean([p == true_unit for p in rep_preds]),
                     dict(zip(PARAM_NAMES, np.round(means, 3))), dict(zip(PARAM_NAMES, np.round(true_theta, 3)))))
    cm, f1, macro_f1, acc = metrics(all_preds, all_trues)
    print(f"\n=== {z_score_label} seed={seed} ===  accuracy={acc:.3f}  macro-F1={macro_f1:.3f}")
    for cls, score in f1.items():
        print(f"  {cls:8s} F1={score:.3f}")
    print("  per-scenario accuracy:")
    for name, true_unit, acc_sc, est, true in rows:
        flag = "" if acc_sc >= 0.5 else "  <-- MISCLASSIFIED"
        print(f"    {name:26s} true={true_unit:8s} acc={acc_sc:.2f} est_alpha={est['alpha']:.3f} true_alpha={true['alpha']:.3f}{flag}")
