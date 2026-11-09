"""Notebook 35 (System I) -- R2 Part C: condition the posterior on MEASURED inlet temperatures.
Stage functions called from notebooks/system_I/35_measured_inlet.ipynb; resumable.
Outputs: results/35<suffix>_* and results/35<suffix>/.

WHY. Notebook 34 showed that the production posterior, which assumes the nominal inlet, is
confidently wrong under realistic inlet errors: a 0.5 K error in the coolant inlet temperature Tci
biases beta by 7-13 posterior sds (the coolant warms by only ~3 K in the jacket). Marginalising the
inlet as an unknown (nb34 Part B) restores calibration at a 5x loss of beta precision. A real plant
measures Tci and Ti (and its coolant supply drifts by several K, so a fixed-inlet model would fail
anyway). This study gives the posterior the MEASURED inlet temperatures and asks how accurate those
measurements must be.

DESIGN (N3 sensors, P2 protocol)
  Plant inlet varies per window over the training prior: Ti, Tci ~ U(295, 299) K (nominal 297 +- 2 K,
  daily/seasonal supply variation), Ci ~ Ci_nom (1 + N(0, 0.01^2)) (not measured; small effect, nb34).
  Inlet sensors: measured = true + calibration offset (fixed per window, N(0, eps^2)) + window-mean
  white noise N(0, (0.2 K)^2 / 120) (an N3-grade RTD averaged over the 60-min window).
  Features ("S1m", 34-D): the 29 summaries + the three plug-in estimators recomputed with the MEASURED
  Ti and Tci (nominal Ci) + the two measured temperatures (as deviations from 297 K).
  Variants (each a 3-seed ensemble at 40k, same theta / inlet draws and seeds):
    oracle      eps = 0, no white noise (the inlet is known exactly; upper bound)
    eps0.05     calibrated sensor / matched-pair measurement
    eps0.1      class-AA RTD
    eps0.2      class-A RTD (uncalibrated)
    marginal    no inlet measurement; S1 features with nominal constants; trained on the same varying
                inlet (= nb34 Part B with the wider +-2 K supply range)
  Tests: (a) SBC set, 1000 prior draws, inlet from the training distribution, each variant measured
  with its own eps; (b) scenario grid Sc1-Sc5 x 50 at fixed true inlets: nominal; Tci +-0.5, +-1,
  +-2 K; Ti +-0.5, +-1, +-2 K (common random numbers), sensor offsets drawn per window with the
  variant's eps. Reference: the N3 production posterior (nominal-inlet assumption) on the same windows.

DECISION RULE (fixed before running)
  A variant is DEPLOYABLE if (i) it is calibrated on the SBC set (interior marginal SBC KS p >= 0.01
  for alpha and beta, TARP KS p >= 0.01), and (ii) for every scenario Sc1-Sc4 and every inlet
  condition within +-2 K: beta 90% coverage at the truth >= 0.80 and class accuracy >= the oracle's
  accuracy on the same cell - 0.10.
  Instrument requirement = the largest deployable eps among {0.05, 0.1, 0.2} K.
  Production recommendation (USER decision; this notebook never changes production) [AMENDED v1.1,
  2026-11-09, after results: the recommended variant must itself be deployable, else none]:
    eps0.1 deployable -> propose the inlet-conditioned posterior at eps = 0.1 K;
    else eps0.05 deployable -> propose eps = 0.05 K and state the requirement (calibrated / matched pair);
    else -> report; propose "marginal" with its precision cost.
  Not covered: a calibration offset on the jacket-outlet sensor Tc (the error that matters is the one on
  the coolant temperature rise Tc - Tci; a matched sensor pair bounds both); feed-flow error.
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

import system_I_config as CFG
SENSOR = CFG.model()
NOISE = CFG.noise_kw()
PROTOCOL = CFG.protocol()
TAG = "35" + CFG.suffix()
R = ROOT / "results"
OUT = R / TAG

T_NOM = 297.0
INLET_RANGE = 2.0                         # K, half-width of the uniform supply-temperature prior
CI_REL_SD = 0.01
WHITE_SD = 0.2 / np.sqrt(120.0)           # window-mean error of an N3-grade RTD
VARIANTS = {"oracle": 0.0, "eps0.05": 0.05, "eps0.1": 0.1, "eps0.2": 0.2, "marginal": None}
CANDIDATE_EPS = ["eps0.05", "eps0.1", "eps0.2"]
B_N, B_SEEDS, CHUNK = 40_000, [0, 1, 2], 10_000
SC_REPS, N_DRAWS = 50, 1000
THETA_SEED, INLET_SEED, SIM_SEED, OFFSET_SEED = 3510, 3511, 3512, 3513
SBC_SEED, TEST_SEED, TEST_OFFSET_SEED = 3520, 3530, 3531
GATED = ["Sc1", "Sc2", "Sc3", "Sc4"]
SCEN = ["Sc1_closed_healthy", "Sc2_closed_fouling", "Sc3_closed_decay", "Sc4_closed_combined", "Sc5_closed_saturated"]
CONDITIONS = {"nominal": (0.0, 0.0)}
for _d in (0.5, 1.0, 2.0):
    for _s in (-1, 1):
        CONDITIONS[f"Tci{_s * _d:+.1f}"] = (0.0, _s * _d)
        CONDITIONS[f"Ti{_s * _d:+.1f}"] = (_s * _d, 0.0)
I_C, I_T, I_TC, I_QC = 0, 5, 10, 15       # positions of C_mean, T_mean, Tc_mean, Qc_mean in the 29 summaries


# ------------------------------------------------------------------ features
def estimators(Cm, Tm, Tcm, Qcm, Ti, Tci, Ci=None):
    """The three plug-in estimators of summary_variants.extra_features (ln alpha_hat, ln beta_hat_jac,
    ln beta_hat_rx) from window means, with per-window inlet values (arrays or scalars)."""
    import summary_variants as sv
    c = sv.physics_constants()
    Ci = c["CI"] if Ci is None else Ci
    eps = 1e-9
    k_T = c["K0"] * np.exp(-c["EA"] / (c["R"] * Tm))
    alpha = (c["Q"] / c["V"]) * (Ci - Cm) / (k_T * np.maximum(Cm, eps))
    dT = np.maximum(Tm - Tcm, 1e-3)
    beta_j = c["RHOC"] * c["CPC"] * Qcm * (Tcm - Tci) / (c["UA"] * dT)
    q_gen = -c["HR"] * c["Q"] * (Ci - Cm)
    q_flow = c["RHO"] * c["CP"] * c["Q"] * (Ti - Tm)
    beta_r = (q_flow + q_gen) / (c["UA"] * dT)
    lg = lambda v: np.log(np.clip(v, 1e-3, 1e3))
    return np.stack([lg(alpha), lg(beta_j), lg(beta_r)], axis=1)


def build_features(base29, inlet_meas, kind):
    """kind 'meas' -> 34-D (29 + estimators with measured Ti, Tci + measured Ti, Tci - 297);
    kind 'marg' -> 32-D (29 + estimators with the nominal inlet)."""
    b = np.asarray(base29, np.float64)
    Cm, Tm, Tcm, Qcm = b[:, I_C], b[:, I_T], b[:, I_TC], b[:, I_QC]
    if kind == "marg":
        est = estimators(Cm, Tm, Tcm, Qcm, T_NOM, T_NOM)
        return np.concatenate([b, est], axis=1).astype(np.float32)
    Ti, Tci = inlet_meas[:, 0], inlet_meas[:, 1]
    est = estimators(Cm, Tm, Tcm, Qcm, Ti, Tci)
    return np.concatenate([b, est, np.stack([Ti - T_NOM, Tci - T_NOM], axis=1)], axis=1).astype(np.float32)


def measured(true_ti_tci, eps, seed):
    """Measured (Ti, Tci) = true + calibration offset N(0, eps^2) + window-mean white noise."""
    rng = np.random.default_rng(seed)
    n = len(true_ti_tci)
    z = rng.normal(size=(n, 2))
    w = rng.normal(size=(n, 2))
    if eps == 0.0:                         # oracle: exact
        return np.asarray(true_ti_tci, np.float64)
    return np.asarray(true_ti_tci, np.float64) + eps * z + WHITE_SD * w


def features_for(variant, base29, inlet_true, seed):
    eps = VARIANTS[variant]
    if eps is None:
        return build_features(base29, None, "marg")
    return build_features(base29, measured(inlet_true[:, 1:3], eps, seed), "meas")


# ------------------------------------------------------------------ simulation
def _inlets(n, seed):
    import ws_inlet as wi
    rng = np.random.default_rng(seed)
    nom = wi.NOMINAL.astype(np.float64)
    return np.stack([nom[0] * (1 + rng.normal(0, CI_REL_SD, n)),
                     rng.uniform(T_NOM - INLET_RANGE, T_NOM + INLET_RANGE, n),
                     rng.uniform(T_NOM - INLET_RANGE, T_NOM + INLET_RANGE, n)], axis=1).astype(np.float32)


def stage_bank():
    import ws_inlet as wi
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    n = B_N * len(B_SEEDS)
    th = we.prior_draws(n, THETA_SEED)
    inl = _inlets(n, INLET_SEED)
    for c in range(n // CHUNK):
        f = OUT / f"bank_chunk{c:02d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        x = wi.simulate(th[sl], inl[sl], PROTOCOL, SIM_SEED * 10_000 + c * CHUNK, features="S0_base29", **NOISE)
        we.save_npz(f, {"theta": th[sl], "inlet": inl[sl], "base29": x})
        we.mark(TAG, f"bank_{c:02d}")
        print("bank", c, f"{time.time() - t0:.0f}s", flush=True)


def _grid():
    from cstr_sbi.scenarios import SCENARIO_CONFIGS
    th, names = [], []
    for s in SCEN:
        sc = SCENARIO_CONFIGS[s]
        th += [[sc.alpha, sc.beta]] * SC_REPS
        names += [s] * SC_REPS
    return np.asarray(th, np.float32), np.asarray(names)


def stage_test():
    import ws_inlet as wi
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "test_sbc.npz"
    if not f.exists():
        th = we.prior_draws(1000, SBC_SEED)
        inl = _inlets(1000, SBC_SEED + 1)
        raw = wi.simulate(th, inl, PROTOCOL, SBC_SEED * 1000, raw=True, **NOISE)
        we.save_npz(f, {"theta": th, "inlet": inl, "raw": raw.astype(np.float32)})
    th, names = _grid()
    for cond, (dti, dtci) in CONDITIONS.items():
        f = OUT / f"test_{cond}.npz"
        if f.exists():
            continue
        inl = np.tile(np.array([wi.NOMINAL[0], T_NOM + dti, T_NOM + dtci], np.float32), (len(th), 1))
        raw = wi.simulate(th, inl, PROTOCOL, TEST_SEED, raw=True, **NOISE)          # CRN across conditions
        we.save_npz(f, {"theta": th, "sc_name": names, "inlet": inl, "raw": raw.astype(np.float32)})
        print("test", cond, flush=True)
    we.mark(TAG, "test_done")


def _t():
    return np.arange(1, 121, dtype=np.float32) * 0.5


def _base29_of(raw):
    import ws_eval as we
    return we.features(raw, _t(), "S0_base29")


# ------------------------------------------------------------------ training
def _bank():
    import ws_eval as we
    parts = [we.load_npz(OUT / f"bank_chunk{c:02d}.npz") for c in range(B_N * len(B_SEEDS) // CHUNK)]
    return (np.concatenate([p["theta"] for p in parts]), np.concatenate([p["inlet"] for p in parts]),
            np.concatenate([p["base29"] for p in parts]))


def stage_train(threads: int = 4):
    import torch
    import ws_eval as we
    torch.set_num_threads(threads)
    th, inl, b29 = _bank()
    for v in VARIANTS:
        x = features_for(v, b29, inl, OFFSET_SEED)
        for s in B_SEEDS:
            f = OUT / f"post_{v}_s{s}.pkl"
            if f.exists():
                continue
            sl = slice(s * B_N, (s + 1) * B_N)
            post, meta = we.train_npe(th[sl], x[sl], seed=s)
            meta.update({"variant": v, "eps": VARIANTS[v], "n_features": int(x.shape[1])})
            with open(f, "wb") as fh:
                pickle.dump({"posterior": post, "metadata": meta}, fh)
            we.mark(TAG, f"train_{v}_s{s}")
            print(v, s, round(meta["train_s"]), "s", flush=True)


# ------------------------------------------------------------------ evaluation
def _post(v):
    import ws_eval as we
    posts = [pickle.load(open(OUT / f"post_{v}_s{s}.pkl", "rb"))["posterior"] for s in B_SEEDS]
    return we.EnsemblePosterior(posts)


def stage_eval():
    import ws_eval as we
    import ws_metrics as wm
    import nb31_budget_convergence as B
    f_sc, f_sbc = R / f"{TAG}_metrics.csv", R / f"{TAG}_sbc.csv"
    rows = pd.read_csv(f_sc).to_dict("records") if f_sc.exists() else []
    sbc_rows = pd.read_csv(f_sbc).to_dict("records") if f_sbc.exists() else []
    done = {(r["variant"], r["condition"]) for r in rows}
    done_sbc = {r["variant"] for r in sbc_rows}
    sbc = we.load_npz(OUT / "test_sbc.npz")
    sbc_b29 = _base29_of(sbc["raw"])
    tests = {c: we.load_npz(OUT / f"test_{c}.npz") for c in CONDITIONS}
    b29 = {c: _base29_of(d["raw"]) for c, d in tests.items()}
    prod_post, prod = B.load_production_posterior()
    evals = {v: (_post(v), None) for v in VARIANTS}
    evals["production"] = (prod_post, prod["summary_set"])
    for v, (post, sset) in evals.items():
        if v != "production" and v not in done_sbc:
            x = features_for(v, sbc_b29, sbc["inlet"], SBC_SEED + 7)
            smp, leak = we.sample_within_prior(post, x, N_DRAWS, rng_seed=1)
            r = wm.evaluate_sbc_set(smp, sbc["theta"], we.LO, we.HI, seed=0)
            sbc_rows.append({"variant": v, "eps": VARIANTS[v], **r, "leak_mean": float(leak.mean()),
                             "sd_b_median": float(np.median(smp[:, :, 1].std(1))),
                             "sd_a_median": float(np.median(smp[:, :, 0].std(1)))})
            pd.DataFrame(sbc_rows).to_csv(f_sbc, index=False)
        for c, d in tests.items():
            if (v, c) in done:
                continue
            if v == "production":
                x = we.features(d["raw"], _t(), sset)
            else:
                x = features_for(v, b29[c], d["inlet"], TEST_OFFSET_SEED)
            smp, leak = we.sample_within_prior(post, x, N_DRAWS, rng_seed=0)
            overall, sc_rows = wm.evaluate_scenario_set(smp, d["theta"], d["sc_name"], we.LO, we.HI)
            for r in sc_rows:
                rows.append({"variant": v, "eps": VARIANTS.get(v), "condition": c, **r,
                             "z_bias_b": r["bias_b"] / r["sd_b"], "macro_f1_all": overall["macro_f1"]})
            pd.DataFrame(rows).to_csv(f_sc, index=False)
        print(v, "evaluated", flush=True)
    we.mark(TAG, "eval_done")


def stage_decide():
    import ws_eval as we
    m = pd.read_csv(R / f"{TAG}_metrics.csv"); m["short"] = m.scenario.str.split("_").str[0]
    sbc = pd.read_csv(R / f"{TAG}_sbc.csv").set_index("variant")
    g = m[m.short.isin(GATED)]
    orc = g[g.variant == "oracle"].set_index(["condition", "short"]).class_acc
    out = {}
    for v in list(VARIANTS):
        s = sbc.loc[v]
        calib = bool(s.ks_p_a_interior >= 0.01 and s.ks_p_b_interior >= 0.01 and s.tarp_ks_p >= 0.01)
        bad = []
        for _, r in g[g.variant == v].iterrows():
            ref = orc.loc[(r.condition, r.short)]
            if r.cov90_truth_b < 0.80 or r.class_acc < ref - 0.10:
                bad.append({"condition": r.condition, "scenario": r.short, "cov90_b": r.cov90_truth_b,
                            "class_acc": r.class_acc, "oracle_acc": ref, "z_bias_b": r.z_bias_b})
        out[v] = {"calibrated": calib, "n_violations": len(bad), "deployable": calib and not bad,
                  "violations": bad, "sd_b_median_sbc": float(s.sd_b_median), "sd_a_median_sbc": float(s.sd_a_median)}
    ok = [v for v in CANDIDATE_EPS if out[v]["deployable"]]
    req = max(ok, key=lambda v: VARIANTS[v]) if ok else None
    # rule as fixed before running (kept for the record)
    rec_v1 = "eps0.1" if out["eps0.1"]["deployable"] else ("eps0.05" if out["eps0.05"]["deployable"] else "marginal")
    # AMENDMENT v1.1 (2026-11-09, after results, user-approved): the fallback recommended "marginal"
    # without checking it against the rule's own criteria (it was the worst variant: 29 violations).
    # Same principle as nb31 v2.1: a recommended posterior must itself be deployable; if none is, the
    # rule reports "none deployable at this budget" and recommends nothing.
    rec = rec_v1 if out[rec_v1]["deployable"] else None
    res = {"variants": out, "instrument_requirement": req,
           "requirement_eps_K": VARIANTS[req] if req else None,
           "recommended_for_production": rec, "recommended_v1_as_fixed": rec_v1,
           "amendment": {"version": "v1.1", "date": "2026-11-09", "after_results": True,
                         "approved_by": "user (Simone Casolo), 2026-11-09, in conversation with Claude",
                         "principle": "a recommended posterior must itself pass the deployability criteria"},
           "near_misses": {v: o["n_violations"] for v, o in out.items()},
           "decision": "USER (this notebook does not change production); design adopted 2026-11-09: eps = 0.1 K, "
                       "production posterior from notebook 36",
           "rule": "see docstring of scripts/nb35_measured_inlet.py (fixed before running) + amendment v1.1"}
    we.save_json(R / f"{TAG}_decision.json", res)
    print(pd.DataFrame({v: {k: o[k] for k in ("calibrated", "n_violations", "deployable", "sd_b_median_sbc")}
                        for v, o in out.items()}).T.to_string())
    print("instrument requirement:", req, "| recommended for production (user decision):", rec)
    return res
