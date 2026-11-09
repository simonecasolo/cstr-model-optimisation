"""Train + SBC + classify S-B3 on the full-uniform wide-prior bank (stage1bfix, matched protocol).

Pre-registered, no seed selection by classification outcome: all seeds are reported.
 * prior: box_uniform_4d_sb3 (alpha, beta_r, eta_col, xi_reb uniform; z_A0_eff fixed nominal)
 * training: SNPE / zuko_nsf 60x3, default (baseline) z-scoring, 5 seeds
 * SBC: reduced N=120 and confirm N=400 sets, per-param KS p, anti-shrinkage slope, 90% coverage
 * classification: 12 S-B3 scenarios x 30 noisy replicates x 200 posterior draws, prediction
   threshold = ground-truth threshold = 0.90.  Primary protocol = matched per-scenario warm start
   (same pipeline as the training bank); secondary = fixed nominal warm start (nb31e protocol).

Outputs: results/sb3_wideprior_sbc_n5000.csv, results/sb3_wideprior_classification_n5000.csv,
         results/sb3_wideprior_classification_per_scenario_n5000.csv,
         sbi-logs/wu2003_posterior_sb3_wideprior_seed{k}.pkl (resumable).
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
from cstr_sbi.recycle.sb3 import (
    SB3_PARAM_NAMES, FAULT_UNITS_SB3, TRUE_UNIT_THRESH_SB3, box_uniform_4d_sb3,
    theta4_to_theta5, list_closed_loop_sb3, fault_unit_sb3_truth, fault_unit_sb3_predict,
)
from cstr_sbi.recycle.simulator import nominal_warm_start, deterministic_window, noisy_replicates
from cstr_sbi.recycle.summaries import compute_summaries
from sbi_pipeline import _structure_config, _simulate_summary, _matched_y0

DATA, RESULTS, SBI_LOGS = ROOT / "data", ROOT / "results", ROOT / "sbi-logs"
SUFFIX = "_sb3_wideprior_stage1bfix"
ARCH, HIDDEN, TRANSFORMS = "zuko_nsf", 60, 3
MAX_EPOCHS, STOP_AFTER_EPOCHS = 120, 12
SEEDS = [0, 1, 2, 3, 4]
N_REPS, N_POST, PRED_THRESH = 30, 200, TRUE_UNIT_THRESH_SB3

bank = np.load(DATA / f"wu2003_sbi_train_sb3_n5000{SUFFIX}.npz")
theta_train, summary_train = bank["thetas"].astype(np.float32), bank["summaries"].astype(np.float32)
r120, r400 = np.load(RESULTS / f"sb3_sbc_set_n120{SUFFIX}.npz"), np.load(RESULTS / f"sb3_sbc_set_n400{SUFFIX}.npz")
sbc_sets = {"reduced_n120": (r120["thetas"].astype(np.float32), r120["summaries"].astype(np.float32)),
            "confirm_n400": (r400["thetas"].astype(np.float32), r400["summaries"].astype(np.float32))}
print(f"train {theta_train.shape} {summary_train.shape}; sbc {[(k, v[0].shape) for k, v in sbc_sets.items()]}", flush=True)

prior = box_uniform_4d_sb3()
ctrl, y0_fixed_cfg = _structure_config("S-B")
y0_sb = nominal_warm_start("S-B")


def train_posterior(seed):
    torch.manual_seed(seed)
    inf = SNPE(prior=prior, density_estimator=posterior_nn(model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS))
    inf.append_simulations(torch.tensor(theta_train), torch.tensor(summary_train))
    t0 = time.time()
    trained = inf.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1,
                        stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    return inf.build_posterior(trained), time.time() - t0


def draw(posterior, s, n):
    with torch.no_grad():
        return posterior.sample((n,), x=torch.tensor(s, dtype=torch.float32), show_progress_bars=False,
                                reject_outside_prior=False).detach().cpu().numpy()


def run_sbc(posterior, th_set, sm_set, seed, n_post=N_POST):
    torch.manual_seed(seed)
    ranks = {n: [] for n in SB3_PARAM_NAMES}; means = {n: [] for n in SB3_PARAM_NAMES}
    cov = {n: [] for n in SB3_PARAM_NAMES}
    for th, s in zip(th_set, sm_set):
        samp = draw(posterior, s, n_post)
        for k, n in enumerate(SB3_PARAM_NAMES):
            ranks[n].append(int(np.sum(samp[:, k] < th[k]))); means[n].append(float(samp[:, k].mean()))
            lo, hi = np.quantile(samp[:, k], [0.05, 0.95]); cov[n].append(bool(lo <= th[k] <= hi))
    out = {"n_sbc": len(th_set)}
    for k, n in enumerate(SB3_PARAM_NAMES):
        r = np.asarray(ranks[n], float)
        out[f"{n}_ks_p"] = float(scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf).pvalue)
        out[f"{n}_slope"] = float(np.polyfit(th_set[:, k], np.asarray(means[n]), 1)[0])
        out[f"{n}_coverage90"] = float(np.mean(cov[n]))
    return out


def classify(samples):
    labels = [fault_unit_sb3_predict(r, thresh=PRED_THRESH) for r in samples]
    probs = {u: float(np.mean([l == u for l in labels])) for u in FAULT_UNITS_SB3}
    return max(probs, key=probs.__getitem__)


def metrics(preds, trues):
    cm = np.zeros((4, 4), int); c2i = {c: i for i, c in enumerate(FAULT_UNITS_SB3)}
    for t, p in zip(trues, preds):
        cm[c2i[t], c2i[p]] += 1
    f1 = {}
    for i, c in enumerate(FAULT_UNITS_SB3):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1[c] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return float(np.trace(cm) / cm.sum()), float(np.mean(list(f1.values()))), f1


# ---- scenario summaries, simulated once and shared across seeds (30 replicates per scenario)
scen_cache = RESULTS / "sb3_wideprior_scenario_summaries.pkl"
if scen_cache.exists():
    scen = pickle.load(open(scen_cache, "rb"))
else:
    scen = {"matched": {}, "fixed": {}}
    for sc in list_closed_loop_sb3():
        th4 = np.asarray(sc.theta())[:4]; th5 = theta4_to_theta5(th4)
        rng = np.random.default_rng(920001 + sc.id)
        ms = []
        # matched y0 depends only on theta, so test it once instead of retrying a deterministic failure
        n_try = 3 * N_REPS if _matched_y0(th5, ctrl) is not None else 0
        for _ in range(n_try):
            s = _simulate_summary(th5, ctrl, y0_fixed_cfg, "S-B", 0.003, rng, scenario_specific_warm_start=True)
            if s is not None and np.isfinite(s).all():
                ms.append(s)
            if len(ms) == N_REPS:
                break
        if len(ms) == N_REPS:
            scen["matched"][sc.name] = (th4, np.stack(ms).astype(np.float32))
        else:
            print(f"  {sc.name}: matched warm start FAILED ({len(ms)}/{N_REPS}); excluded from matched protocol", flush=True)
        t_h, raw = deterministic_window(sc, structure="S-B", y0=y0_sb)
        reps = noisy_replicates(np.asarray(raw), n_replicates=N_REPS, rng=np.random.default_rng(310001 + sc.id), noise_pct=0.003)
        scen["fixed"][sc.name] = (th4, np.stack([compute_summaries(reps[i], "S-B", np.asarray(t_h)) for i in range(N_REPS)]).astype(np.float32))
        print(f"simulated scenario {sc.name}", flush=True)
    pickle.dump(scen, open(scen_cache, "wb"))

sbc_rows, cls_rows, scen_rows = [], [], []
for seed in SEEDS:
    pkl = SBI_LOGS / f"wu2003_posterior_sb3_wideprior_seed{seed}.pkl"
    if pkl.exists():
        posterior = pickle.load(open(pkl, "rb"))["posterior"]; print(f"loaded {pkl.name}", flush=True)
    else:
        posterior, tr = train_posterior(seed)
        pickle.dump({"posterior": posterior, "seed": seed, "structure": "S-B3", "prior": "box_uniform_4d_sb3",
                     "bank": f"wu2003_sbi_train_sb3_n5000{SUFFIX}.npz"}, open(pkl, "wb"))
        print(f"[seed {seed}] trained in {tr:.0f}s", flush=True)
    for budget, (th_s, sm_s) in sbc_sets.items():
        res = run_sbc(posterior, th_s, sm_s, seed)
        row = dict(seed=seed, budget=budget, min_ks_p=min(res[f"{n}_ks_p"] for n in SB3_PARAM_NAMES), **res)
        sbc_rows.append(row)
        print(f"[seed {seed}] SBC {budget}: ks_p=" + str({n: round(res[f'{n}_ks_p'], 4) for n in SB3_PARAM_NAMES})
              + " slope=" + str({n: round(res[f'{n}_slope'], 2) for n in SB3_PARAM_NAMES})
              + " cov90=" + str({n: round(res[f'{n}_coverage90'], 2) for n in SB3_PARAM_NAMES}), flush=True)
    for proto in ("matched", "fixed"):
        preds, trues = [], []
        for name, (th4, sums) in scen[proto].items():
            tu = fault_unit_sb3_truth(th4); rp = []; pooled = []
            for s in sums:
                samp = draw(posterior, s, N_POST); pooled.append(samp)
                p = classify(samp); rp.append(p); preds.append(p); trues.append(tu)
            pm = np.concatenate(pooled).mean(0)
            scen_rows.append(dict(seed=seed, protocol=proto, scenario=name, true_unit=tu,
                                  scenario_acc=float(np.mean([p == tu for p in rp])),
                                  **{f"{n}_true": float(th4[k]) for k, n in enumerate(SB3_PARAM_NAMES)},
                                  **{f"{n}_mean": float(pm[k]) for k, n in enumerate(SB3_PARAM_NAMES)}))
        acc, mf1, f1 = metrics(preds, trues)
        cls_rows.append(dict(seed=seed, protocol=proto, accuracy=acc, macro_f1=mf1, **{f"f1_{c}": v for c, v in f1.items()}))
        print(f"[seed {seed}] classification ({proto}): acc={acc:.3f} macroF1={mf1:.3f} {({c: round(v, 2) for c, v in f1.items()})}", flush=True)
    pd.DataFrame(sbc_rows).to_csv(RESULTS / "sb3_wideprior_sbc_n5000.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(RESULTS / "sb3_wideprior_classification_n5000.csv", index=False)
    pd.DataFrame(scen_rows).to_csv(RESULTS / "sb3_wideprior_classification_per_scenario_n5000.csv", index=False)

print("\nDONE", flush=True)
