"""Phase 3 of docs/Manuscript/system_ii_analyzer_rework_plan.md: fault
classification for the S-C.2 (feed-analyzer + xi_reb-sharpening features)
posterior, seed 0 (validated, `system_ii_sbi_solidification_plan.md`: passes
both independent SBC sets, all 5 parameters, p=0.070/0.171).

Mirrors notebooks/31_wu2003_fault_classification.ipynb's methodology exactly
(same posterior-mass threshold classifier, same fault_unit() ground truth,
same 14-scenario catalogue, same N_REPS=30/N_POST=200 budget, same seeding
convention) -- this is the first-ever run of this pipeline against an S-C-family
posterior (no nb31 S-C companion existed before this).
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.scenarios import list_closed_loop, FAULT_UNITS
from cstr_sbi.recycle.simulator import nominal_warm_start, deterministic_window, noisy_replicates
from cstr_sbi.recycle.summaries import compute_summaries
from cstr_sbi.recycle.physics import PARAM_NAMES

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SBI_LOGS = ROOT / "sbi-logs"
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-C.2"
NOISE_PCT = 0.003
N_REPS = 30
N_POST = 200
THRESH = 0.85
Z0_NOM = 0.90
SEED = 20260705
POSTERIOR_PATH = SBI_LOGS / "wu2003_posterior_sc2_true_n1000_zscorenone_seed0_mixedprior_stage1bfix.pkl"


def sample_fault_unit(theta_row: np.ndarray, thresh: float = THRESH, z0_nom: float = Z0_NOM) -> str:
    """Posterior-mass classifier convention (same as nb31/nb11 M6c)."""
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


def classify_fault_recycle(samples: np.ndarray) -> dict:
    labels = [sample_fault_unit(row) for row in samples]
    probs = {u: float(np.mean([l == u for l in labels])) for u in FAULT_UNITS}
    predicted = max(probs, key=probs.__getitem__)
    return {"class": predicted, "probs": probs,
            "means": {p: float(np.mean(samples[:, i])) for i, p in enumerate(PARAM_NAMES)}}


def compute_classification_metrics_recycle(preds: list, trues: list, classes=FAULT_UNITS) -> dict:
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
    return {"confusion_matrix": cm, "class_names": classes,
            "per_class_f1": f1_per_class, "macro_f1": macro_f1, "accuracy": accuracy}


def main():
    torch.manual_seed(SEED)
    with open(POSTERIOR_PATH, "rb") as f:
        payload = pickle.load(f)
    posterior = payload["posterior"]
    print(f"Loaded posterior: {POSTERIOR_PATH.name}  "
          f"(structure={payload.get('structure')}, n_train={payload.get('n_train')}, "
          f"seed={payload.get('torch_seed')}, z_score_x={payload.get('z_score_x')})")

    y0_sb = nominal_warm_start("S-B")
    scenarios = list_closed_loop()
    print(f"Scenario catalogue: {len(scenarios)} scenarios")

    rep_rows = []
    per_scenario_rows = []
    for sc in scenarios:
        theta_np = np.asarray(sc.theta(), dtype=np.float32)
        true_class = sc.fault_unit()

        t_h, raw_obs = deterministic_window(sc, structure="S-B", y0=y0_sb)
        t_h_arr = np.asarray(t_h)

        rng = np.random.default_rng(SEED)
        reps = noisy_replicates(raw_obs, n_replicates=N_REPS, rng=rng, noise_pct=NOISE_PCT)

        pooled_samples = []
        rep_predicted = []
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
        # majority vote across the 30 per-replicate predictions, as a second aggregation view
        from collections import Counter
        majority_class = Counter(rep_predicted).most_common(1)[0][0]
        print(f"  {sc.name:22s} true={true_class:8s}  pooled_pred={pooled_result['class']:8s}  "
              f"majority_pred={majority_class:8s}  theta={np.round(theta_np, 3).tolist()}")
        per_scenario_rows.append({"scenario": sc.name, "true_class": true_class,
                                   "pooled_pred": pooled_result["class"],
                                   "majority_pred": majority_class, **pooled_result["probs"]})

    rep_df = pd.DataFrame(rep_rows)
    scenario_df = pd.DataFrame(per_scenario_rows)
    rep_df.to_csv(RESULTS / "sc2_seed0_mixedprior_classification_per_replicate.csv", index=False)
    scenario_df.to_csv(RESULTS / "sc2_seed0_mixedprior_classification_per_scenario.csv", index=False)

    print("\n=== Metrics: per-replicate predictions (30 reps x 14 scenarios = 420 samples) ===")
    metrics_rep = compute_classification_metrics_recycle(rep_df["pred_class"].tolist(), rep_df["true_class"].tolist())
    print(f"macro-F1 = {metrics_rep['macro_f1']:.3f}   accuracy = {metrics_rep['accuracy']:.3f}")
    print("per-class F1:", {k: round(v, 3) for k, v in metrics_rep["per_class_f1"].items()})
    print("confusion matrix (rows=true, cols=pred), classes =", metrics_rep["class_names"])
    print(metrics_rep["confusion_matrix"])

    print("\n=== Metrics: per-scenario pooled-posterior predictions (14 scenarios) ===")
    metrics_scenario = compute_classification_metrics_recycle(scenario_df["pooled_pred"].tolist(), scenario_df["true_class"].tolist())
    print(f"macro-F1 = {metrics_scenario['macro_f1']:.3f}   accuracy = {metrics_scenario['accuracy']:.3f}")
    print("per-class F1:", {k: round(v, 3) for k, v in metrics_scenario["per_class_f1"].items()})
    print("confusion matrix (rows=true, cols=pred), classes =", metrics_scenario["class_names"])
    print(metrics_scenario["confusion_matrix"])

    np.savez(RESULTS / "sc2_seed0_mixedprior_classification_metrics.npz",
             cm_rep=metrics_rep["confusion_matrix"], cm_scenario=metrics_scenario["confusion_matrix"],
             macro_f1_rep=metrics_rep["macro_f1"], macro_f1_scenario=metrics_scenario["macro_f1"],
             accuracy_rep=metrics_rep["accuracy"], accuracy_scenario=metrics_scenario["accuracy"],
             class_names=metrics_rep["class_names"])
    print("\nSaved results/sc2_seed0_mixedprior_classification_{per_replicate,per_scenario,metrics}.*")


if __name__ == "__main__":
    main()
