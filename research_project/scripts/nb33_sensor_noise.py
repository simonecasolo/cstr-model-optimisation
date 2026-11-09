"""Notebook 33 (System I) -- R1: does the sensor-noise model drive the results?
Stage functions called from notebooks/system_I/33_sensor_noise_realism.ipynb; resumable.
Outputs: results/33_* (and results/33/ for banks, test sets, posteriors).

Why: the inherited sensor model (sigma = 0.5% of each channel's window maximum) is
pessimistic for T (sigma ~1.56 K; industrial RTDs +-0.1-0.5 K) and optimistic for C
(sigma ~9e-5 mol/L; online analysers ~1-2% of full scale). alpha is inferred mainly from C
at ~98% conversion, so its precision is probably overstated.

VARIANTS (P1b, same theta draws and seeds; sigma per channel [C, T, Tc, Qc]; None = keep the
project convention, 0.5% of the channel's window maximum, for that channel):
  N0  (None, None, None, None)   current convention                         -- reference
  N1  (0.01, None, None, None)   realistic C analyser only (1% of 1 mol/L)  -- isolates alpha
  N2  (None, 0.2, 0.2, None)     realistic T, Tc sensors only (RTD class)
  N3  (0.01, 0.2, 0.2, 2.0)      realistic on all channels (Qc 0.5% of the 400 L/min span)
  N4  (1.0, 0.2, 0.2, 2.0)       as N3 but NO usable C measurement (sigma_C = 1 mol/L, ~55x the
                                 value of C itself): plants without an online analyser.
                                 Added 2026-11-08 before any training result; reported, not gated.

Each variant: Cramer-Rao bounds at the scenario points (Gaussian FIM, as nb29 E2), then NPE
(production NSF recipe) at 40k simulations x 3 seeds (each seed its own 40k draws), 3-seed
ensemble, evaluated on a test set simulated with the SAME variant (1000 SBC draws + the
closed-loop scenario grid x 50).

DECISION RULE (fixed before running; re-run plan, Phase 2b)
-----------------------------------------------------------
Compare the N3 ensemble with the N0 ensemble (same budget) on Sc1-Sc5 (Sc7 is misspecified by
design and reported only). N3 "changes a headline conclusion" if, for any of Sc1-Sc5:
  (a) class accuracy differs from N0 by more than 5 percentage points, or
  (b) beta 90% coverage at the truth falls outside [0.85, 0.95], or
  (c) the alpha-beta correlation changes regime, with regimes |r| < 0.2 ("uncorrelated"),
      r >= 0.2 ("positive"), r <= -0.2 ("negative").
If so: FLAG -- the user decides whether production switches to N3 before Phase 3.
If not: production stays N0; N1-N4 go to the SI, alpha precision reported under both.
N4 does not enter the rule: it describes a different instrumentation (no analyser), reported
as the bound on what is identifiable from temperatures and coolant flow alone.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

TAG = "33"
R = ROOT / "results"
OUT = R / "33"
PROTOCOL = "P1b"
N_TRAIN, SEEDS = 40_000, [0, 1, 2]
N_SBC, SC_REPS = 1000, 50
THETA_SEED, TEST_SEED = 3300_3, 3300_9
GATED = ["Sc1", "Sc2", "Sc3", "Sc4", "Sc5"]

VARIANTS = {
    "N0": None,                        # project convention on every channel
    "N1": (0.01, None, None, None),    # realistic C analyser only
    "N2": (None, 0.2, 0.2, None),      # realistic T / Tc sensors only
    "N3": (0.01, 0.2, 0.2, 2.0),       # realistic on all channels
    "N4": (1.0, 0.2, 0.2, 2.0),        # as N3, no usable C measurement (reported, not gated)
}


def _kw(v):
    na = VARIANTS[v]
    return {} if na is None else {"noise_abs": tuple(None if x is None else float(x) for x in na)}


def stage_fim(n_rep: int = 1500, h: float = 0.01):
    """Cramer-Rao sd and implied correlation per variant at the scenario points (29-D summaries,
    full covariance; raw trajectory with Ledoit-Wolf as the information ceiling)."""
    import jax.numpy as jnp
    import ws_protocols as ws
    import ws_eval as we
    import nb29_ws_study as s29
    from cstr_sbi.scenarios import SCENARIO_CONFIGS
    f = R / f"{TAG}_fim.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["variant"], r["point"]) for r in rows}
    pts = {k.split("_")[0]: (SCENARIO_CONFIGS[k].alpha, SCENARIO_CONFIGS[k].beta)
           for k in we.EVAL_SCENARIOS if k.split("_")[0] in GATED}
    for v in VARIANTS:
        for name, (a, b) in pts.items():
            if (v, name) in done:
                continue
            th0 = np.array([a, b], np.float32)
            var = [th0, th0 + [h, 0], th0 - [h, 0], th0 + [0, h], th0 - [0, h]]
            raws = [we.simulate(np.tile(x, (n_rep, 1)), PROTOCOL, seed=33_200, raw=True, **_kw(v)) for x in var]
            t = np.arange(1, raws[0].shape[1] + 1) * 0.5
            summ = [np.asarray(ws.summaries(jnp.asarray(r), jnp.asarray(t))) for r in raws]
            for rep, arrs, mode in (("summaries_fullcov", summ, "full"),
                                    ("raw_traj_lwcov", [r.reshape(n_rep, -1) for r in raws], "lw")):
                X0, ap, am, bp, bm = (np.asarray(x, np.float64) for x in arrs)
                G = np.stack([(ap - am).mean(0) / (2 * h), (bp - bm).mean(0) / (2 * h)], axis=1)
                F = s29._fim(G, X0, mode)
                Fi = np.linalg.pinv(F)
                rows.append({"variant": v, "point": name, "representation": rep,
                             "crb_sd_a": float(np.sqrt(max(Fi[0, 0], 0))),
                             "crb_sd_b": float(np.sqrt(max(Fi[1, 1], 0))),
                             "corr_implied": float(-F[0, 1] / np.sqrt(F[0, 0] * F[1, 1]))})
            pd.DataFrame(rows).to_csv(f, index=False)
            print(v, name, "done", flush=True)


def stage_sim():
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    th = we.prior_draws(N_TRAIN * len(SEEDS), THETA_SEED)
    for v in VARIANTS:
        f = OUT / f"bank_{v}.npz"
        if not f.exists():
            t0 = time.time()
            x = we.simulate(th, PROTOCOL, seed=THETA_SEED * 10, **_kw(v))
            we.save_npz(f, {"theta": th, "x": x})
            print(f"bank {v}: {x.shape} in {time.time() - t0:.0f}s", flush=True)
        g = OUT / f"test_{v}.npz"
        if not g.exists():
            we.save_npz(g, we.make_test_set(PROTOCOL, TEST_SEED, N_SBC, SC_REPS, **_kw(v)))
            print(f"test {v} done", flush=True)
        we.mark(TAG, f"sim_{v}_done")


def stage_train(threads: int = 4):
    import torch
    import ws_eval as we
    torch.set_num_threads(threads)
    for v in VARIANTS:
        bank = we.load_npz(OUT / f"bank_{v}.npz")
        for s in SEEDS:
            f = OUT / f"post_{v}_s{s}.pkl"
            if f.exists():
                continue
            sl = slice(s * N_TRAIN, (s + 1) * N_TRAIN)
            post, meta = we.train_npe(bank["theta"][sl], bank["x"][sl], seed=s)
            meta.update({"variant": v, "noise_abs": VARIANTS[v]})
            with open(f, "wb") as fh:
                pickle.dump({"posterior": post, "metadata": meta}, fh)
            we.mark(TAG, f"train_{v}_s{s}")
            print(v, s, round(meta["train_s"]), "s", meta["epochs"], "epochs", flush=True)


def stage_eval():
    import ws_eval as we
    f_rows, f_sc = R / f"{TAG}_metrics.csv", R / f"{TAG}_scenarios.csv"
    rows = pd.read_csv(f_rows).to_dict("records") if f_rows.exists() else []
    sc_rows = pd.read_csv(f_sc).to_dict("records") if f_sc.exists() else []
    done = {(r["variant"], str(r["member"])) for r in rows}
    for v in VARIANTS:
        test = we.load_npz(OUT / f"test_{v}.npz")
        members = {s: pickle.load(open(OUT / f"post_{v}_s{s}.pkl", "rb"))["posterior"] for s in SEEDS}
        todo = [(str(s), m) for s, m in members.items()] + [("ensemble", we.EnsemblePosterior(members.values()))]
        for name, post in todo:
            if (v, name) in done:
                continue
            row, scr = we.evaluate(post, test, seed=0)
            rows.append({"variant": v, "member": name, **row})
            sc_rows += [{"variant": v, "member": name, **r} for r in scr]
            pd.DataFrame(rows).to_csv(f_rows, index=False)
            pd.DataFrame(sc_rows).to_csv(f_sc, index=False)
            print(v, name, {k: round(row[k], 3) for k in ("ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p",
                                                          "rmse_norm", "macro_f1")}, flush=True)
        we.mark(TAG, f"eval_{v}_done")


def stage_classification(n_draws: int = 1000, keep_draws: int = 500):
    """Section 7 (added 2026-11-09, descriptive -- no decision rule): how the sensor model
    changes the *diagnosis*. For every variant's 3-seed ensemble and each seed member:
      * class confusion matrices and per-class F1 on the SBC set (1000 prior draws: every
        fault class over the whole prior) and on the scenario set (Sc1-Sc7 x 50);
      * the information gained on alpha and beta: posterior sd / prior sd per observation
        (1 = the data say nothing beyond the prior);
    and keeps ``keep_draws`` ensemble draws per scenario window for the figures.
    Outputs: results/33_class_f1.csv, 33_confusion.csv, 33_information.csv, 33/cls_samples_{v}.npz.
    Uses the same posteriors and test sets as sections 4-5; resumable per variant."""
    import ws_eval as we
    import ws_metrics as wm
    f_f1, f_cm, f_inf = R / f"{TAG}_class_f1.csv", R / f"{TAG}_confusion.csv", R / f"{TAG}_information.csv"
    f1_rows = pd.read_csv(f_f1).to_dict("records") if f_f1.exists() else []
    cm_rows = pd.read_csv(f_cm).to_dict("records") if f_cm.exists() else []
    inf_rows = pd.read_csv(f_inf).to_dict("records") if f_inf.exists() else []
    done = {(r["variant"], str(r["member"])) for r in f1_rows}
    prior_sd = (we.HI - we.LO) / np.sqrt(12.0)
    for v in VARIANTS:
        test = we.load_npz(OUT / f"test_{v}.npz")
        members = {str(s): pickle.load(open(OUT / f"post_{v}_s{s}.pkl", "rb"))["posterior"] for s in SEEDS}
        todo = list(members.items()) + [("ensemble", we.EnsemblePosterior(members.values()))]
        for name, post in todo:
            if (v, name) in done:
                continue
            t0 = time.time()
            for set_name, xk, tk in (("sbc", "x_sbc", "theta_sbc"), ("scenarios", "x_sc", "theta_sc")):
                smp, _ = we.sample_within_prior(post, test[xk], n_draws, rng_seed=7)
                y_t = wm.class_of_theta(test[tk]); y_p = wm.class_of_samples(smp)
                for c in wm.CLASSES:
                    tp = int(np.sum((y_p == c) & (y_t == c))); fp = int(np.sum((y_p == c) & (y_t != c)))
                    fn = int(np.sum((y_p != c) & (y_t == c)))
                    f1_rows.append({"variant": v, "member": name, "set": set_name, "cls": c,
                                    "n_true": int(np.sum(y_t == c)),
                                    "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                                    "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                                    "precision": tp / (tp + fp) if (tp + fp) else np.nan,
                                    "macro_f1": wm.macro_f1(y_t, y_p), "accuracy": float((y_t == y_p).mean())})
                if name == "ensemble":
                    for ct in wm.CLASSES:
                        for cp in wm.CLASSES:
                            cm_rows.append({"variant": v, "set": set_name, "true": ct, "pred": cp,
                                            "n": int(np.sum((y_t == ct) & (y_p == cp)))})
                    sd = smp.std(axis=1) / prior_sd
                    groups = (np.full(len(sd), "SBC (prior draws)") if set_name == "sbc"
                              else np.asarray(test["sc_name"]).astype(str))
                    for g in sorted(set(groups.tolist())):
                        m = groups == g
                        inf_rows.append({"variant": v, "group": g,
                                         "sd_ratio_a": float(np.median(sd[m, 0])), "sd_ratio_b": float(np.median(sd[m, 1])),
                                         "corr_ab": float(np.nanmean(wm.per_obs_corr(smp[m])))})
                    if set_name == "scenarios":
                        we.save_npz(OUT / f"cls_samples_{v}.npz",
                                    {"samples": smp[:, :keep_draws].astype(np.float32),
                                     "theta": test[tk], "sc_name": np.asarray(test["sc_name"]).astype(str)})
            pd.DataFrame(f1_rows).to_csv(f_f1, index=False)
            pd.DataFrame(cm_rows).to_csv(f_cm, index=False)
            pd.DataFrame(inf_rows).to_csv(f_inf, index=False)
            print(v, name, f"{time.time() - t0:.0f}s", flush=True)
        we.mark(TAG, f"cls_{v}_done")


def _regime(r):
    return "uncorrelated" if abs(r) < 0.2 else ("positive" if r > 0 else "negative")


def stage_decide():
    import ws_eval as we
    sc = pd.read_csv(R / f"{TAG}_scenarios.csv")
    sc["member"] = sc["member"].astype(str)
    sc["short"] = sc["scenario"].str.split("_").str[0]
    e = sc[sc.member == "ensemble"].set_index(["variant", "short"])
    checks, flag = [], False
    for s in GATED:
        n0, n3 = e.loc[("N0", s)], e.loc[("N3", s)]
        a = abs(n3.class_acc - n0.class_acc) > 0.05
        b = not (0.85 <= n3.cov90_truth_b <= 0.95)
        c = _regime(n3.corr_ab_mean) != _regime(n0.corr_ab_mean)
        flag |= bool(a or b or c)
        checks.append({"scenario": s, "acc_N0": n0.class_acc, "acc_N3": n3.class_acc, "a_acc_changed": bool(a),
                       "covb_N3": n3.cov90_truth_b, "b_cov_outside": bool(b),
                       "corr_N0": n0.corr_ab_mean, "corr_N3": n3.corr_ab_mean,
                       "regime_N0": _regime(n0.corr_ab_mean), "regime_N3": _regime(n3.corr_ab_mean),
                       "c_regime_changed": bool(c),
                       "sd_a_N0": n0.sd_a, "sd_a_N3": n3.sd_a, "sd_b_N0": n0.sd_b, "sd_b_N3": n3.sd_b})
    res = {"flag": flag,
           "outcome": ("N3 CHANGES A HEADLINE CONCLUSION: user decides whether production switches to N3 "
                       "before Phase 3" if flag else
                       "No headline change: production stays N0; report N1-N3 in the SI"),
           "checks": checks, "rule": "see docstring of scripts/nb33_sensor_noise.py (fixed before results)"}
    we.save_json(R / f"{TAG}_decision.json", res)
    print(pd.DataFrame(checks).round(3).to_string(index=False))
    print(res["outcome"])
    return res
