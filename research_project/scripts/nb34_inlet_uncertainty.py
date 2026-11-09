"""Notebook 34 (System I) -- R2: is the diagnosis robust to uncertain inlet conditions?
Stage functions called from notebooks/system_I/34_inlet_uncertainty.ipynb; resumable.
Outputs: results/34<suffix>_* and results/34<suffix>/ (raw test windows, banks, posteriors).

WHY. The jacket removes only ~19% of the reaction heat (the feed absorbs ~81%), so the jacket duty,
and hence beta, is sensitive to the feed temperature: a 1 K error in Ti shifts the duty by ~27%
(~0.2 in beta) to first order. The model treats Ci, Ti and Tci as exactly known, while the N3
production posterior reports beta with sd ~0.015. If realistic inlet errors bias beta by many
posterior sds, the precision claim does not hold in practice. The N3 production summary set (S1)
contains two independent plug-in estimates of beta, one from the jacket (coolant) balance (uses
Tci) and one from the reactor balance (uses Ti and Ci), so the study also tests whether their
disagreement can serve as a misspecification alarm.

DESIGN (sensor model and protocol from system_I_config: N3, P2; production summary set S1)
  Check     ws_inlet at the nominal inlet reproduces ws_eval.simulate (raw windows, to <1% of the
            sensor sigma per channel); deterministic first-order sensitivity of the plug-in
            estimators to each inlet error (noise-free steady states).
  Part A    misspecification test. True plant inlet perturbed one factor at a time:
              Ti  +-0.5, +-1 K    Tci +-0.5, +-1 K    Ci +-1 %, +-2 %   (+ nominal)
            Scenario grid Sc1-Sc5 x 50 windows, common random numbers across conditions.
            Posteriors (all assume the nominal inlet): the N3 production posterior (nb31,
            S1, 160k ensemble) and, for S0 vs S1 at equal budget, the nb03 section-10b 3-seed
            ensembles at 20k (S0_base29 and S1_base_est).
            Metrics per scenario: alpha/beta bias, bias / posterior sd, 90% coverage at the truth,
            class accuracy, prior leakage. Alarm: d = ln beta_hat_jac - ln beta_hat_rx per window,
            z-scored against a separate nominal calibration set (pooled Sc1-Sc4); detection rate
            at |z| > 3 per condition (false-alarm rate = the nominal condition).
  Part B    robust training (run if the Part A rule triggers, or FORCE_B). Two S1 posteriors at
            40k x 3 seeds, same theta draws and seeds: "nominal" (inlet fixed) and "robust" (inlet
            drawn per sample: Ti ~ N(297, 0.5^2) K, Tci ~ N(297, 0.5^2) K, Ci ~ Ci_nom (1 + N(0, 0.01^2))),
            i.e. the inlet marginalised as a nuisance. Both evaluated on the Part A conditions and on
            SBC sets drawn under the nominal and under the nuisance inlet distribution.

DECISION RULE (fixed before running; applies to Part A and the production posterior)
  Realistic inlet uncertainty = Ti +-0.5 K, Tci +-0.5 K, Ci +-1 % (calibrated feed RTD and feed
  variability; coolant supply variability; analyser accuracy). The production posterior is called
  ROBUST if, for every realistic condition and every scenario Sc1-Sc4, beta 90% coverage at the
  truth >= 0.80 and class accuracy within 0.10 of the nominal condition. Otherwise Part B is run.
  The +-1 K / +-2 % levels are stress tests, reported, not gated. Sc5 (saturated) is reported, not
  gated. Adopting a robust posterior as production is a USER decision (Part B reports; it does
  not change production).
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
SET = CFG.summary_set()
TAG = "34" + CFG.suffix()
R = ROOT / "results"
OUT = R / TAG

SC_REPS, N_DRAWS = 50, 1000
TEST_SEED, CAL_SEED = 3400, 3401
B_THETA_SEED, B_INLET_SEED, B_SIM_SEED, B_SBC_SEED = 3410, 3411, 3412, 3420
B_N, B_SEEDS, CHUNK = 40_000, [0, 1, 2], 10_000
NUIS_SD = {"Ti": 0.5, "Tci": 0.5, "Ci_rel": 0.01}
GATED = ["Sc1", "Sc2", "Sc3", "Sc4"]
SCEN = ["Sc1_closed_healthy", "Sc2_closed_fouling", "Sc3_closed_decay", "Sc4_closed_combined", "Sc5_closed_saturated"]

# name -> (relative Ci error, Ti error [K], Tci error [K])
CONDITIONS = {
    "nominal": (0.0, 0.0, 0.0),
    "Ti-1.0": (0.0, -1.0, 0.0), "Ti-0.5": (0.0, -0.5, 0.0), "Ti+0.5": (0.0, 0.5, 0.0), "Ti+1.0": (0.0, 1.0, 0.0),
    "Tci-1.0": (0.0, 0.0, -1.0), "Tci-0.5": (0.0, 0.0, -0.5), "Tci+0.5": (0.0, 0.0, 0.5), "Tci+1.0": (0.0, 0.0, 1.0),
    "Ci-2%": (-0.02, 0.0, 0.0), "Ci-1%": (-0.01, 0.0, 0.0), "Ci+1%": (0.01, 0.0, 0.0), "Ci+2%": (0.02, 0.0, 0.0),
}
REALISTIC = ["Ti-0.5", "Ti+0.5", "Tci-0.5", "Tci+0.5", "Ci-1%", "Ci+1%"]


def inlet_of(cond: str) -> np.ndarray:
    import ws_inlet as wi
    dci, dti, dtci = CONDITIONS[cond]
    n = wi.NOMINAL.astype(np.float64)
    return np.array([n[0] * (1 + dci), n[1] + dti, n[2] + dtci], np.float32)


def _grid():
    from cstr_sbi.scenarios import SCENARIO_CONFIGS
    th, names = [], []
    for s in SCEN:
        sc = SCENARIO_CONFIGS[s]
        th += [[sc.alpha, sc.beta]] * SC_REPS
        names += [s] * SC_REPS
    return np.asarray(th, np.float32), np.asarray(names)


# ============================================================================ check
def stage_check():
    """Nominal identity of ws_inlet vs ws_eval, and deterministic sensitivities of the plug-in
    estimators (noise-free closed-loop steady states, Qc from the controller)."""
    import jax.numpy as jnp
    import ws_inlet as wi
    import summary_variants as sv
    from cstr_sbi.physics import compute_qc, NOMINAL_CTRL
    OUT.mkdir(parents=True, exist_ok=True)
    res = wi.nominal_identity_check(PROTOCOL, **NOISE)
    sig = np.array(CFG.SENSOR_MODELS[SENSOR] or (1e-4, 1.5, 1.5, 1.0), float)
    res["ratio_to_sensor_sigma"] = (np.array(res["max_abs_diff_per_channel"]) / sig).tolist()
    res["ok"] = bool(np.all(np.array(res["ratio_to_sensor_sigma"]) < 0.01))
    (R / f"{TAG}_identity_check.json").write_text(json.dumps(res, indent=2))
    print("nominal identity:", res)
    if not res["ok"]:
        raise RuntimeError("STOP: ws_inlet does not reproduce ws_eval at the nominal inlet (>= 1% of the sensor sigma)")
    const = sv.physics_constants()
    th, names = _grid()
    pts = {n.split("_")[0]: th[i] for i, n in enumerate(names) if i % SC_REPS == 0}
    rows = []
    for cond in CONDITIONS:
        inl = jnp.asarray(inlet_of(cond))
        for sc, (a, b) in pts.items():
            y = np.asarray(wi._ss(jnp.float32(a), jnp.float32(b), inl))
            qc = float(compute_qc(y[1], y[3], NOMINAL_CTRL))
            obs = np.tile(np.array([y[0], y[1], y[2], qc], np.float64), (1, 120, 1))
            est = sv.extra_features(obs, np.arange(1, 121) * 0.5, const)[0, :3]
            rows.append({"condition": cond, "scenario": sc, "alpha": float(a), "beta": float(b), "Qc_ss": qc,
                         "T_ss": float(y[1]), "dln_alpha_hat": est[0] - np.log(a),
                         "dln_beta_hat_jac": est[1] - np.log(b), "dln_beta_hat_rx": est[2] - np.log(b)})
    df = pd.DataFrame(rows)
    nom = df[df.condition == "nominal"].set_index("scenario")
    for c in ("dln_alpha_hat", "dln_beta_hat_jac", "dln_beta_hat_rx", "Qc_ss"):
        df[c + "_vs_nominal"] = df[c] - df.scenario.map(nom[c])
    df.to_csv(R / f"{TAG}_sensitivity.csv", index=False)
    return res, df


# ============================================================================ Part A
def stage_sim_A():
    import ws_inlet as wi
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    th, names = _grid()
    for cond in CONDITIONS:
        f = OUT / f"A_{cond}.npz"
        if f.exists():
            continue
        raw = wi.simulate(th, inlet_of(cond), PROTOCOL, TEST_SEED, raw=True, **NOISE)   # same seed: CRN
        we.save_npz(f, {"theta": th, "sc_name": names, "raw": raw.astype(np.float32)})
        print(cond, raw.shape, flush=True)
    f = OUT / "A_calibration_nominal.npz"
    if not f.exists():
        raw = wi.simulate(th, inlet_of("nominal"), PROTOCOL, CAL_SEED, raw=True, **NOISE)
        we.save_npz(f, {"theta": th, "sc_name": names, "raw": raw.astype(np.float32)})
    we.mark(TAG, "simA_done")


def _ensemble(paths):
    import ws_eval as we
    posts = [pickle.load(open(p, "rb"))["posterior"] for p in paths]
    return we.EnsemblePosterior(posts) if len(posts) > 1 else posts[0]


def posteriors_A() -> dict:
    """name -> (posterior, summary set). The production posterior plus the nb03 §10b 20k ensembles."""
    import nb31_budget_convergence as B
    out = {}
    post, prod = B.load_production_posterior()
    assert prod["sensor_model"] == SENSOR and prod["protocol"] == PROTOCOL, (prod["sensor_model"], prod["protocol"])
    out[f"production_{prod['summary_set']}_{prod['production_budget'] // 1000}k"] = (post, prod["summary_set"])
    d03 = R / ("03s" + CFG.suffix())
    for s in ("S0_base29", "S1_base_est"):
        paths = [d03 / f"post_{s}_n20000_s{k}.pkl" for k in (0, 1, 2)]
        if all(p.exists() for p in paths):
            out[f"nb03_{s}_20k"] = (_ensemble(paths), s)
        else:
            print(f"missing nb03 posteriors for {s}: skipped", flush=True)
    return out


def _t():
    return np.arange(1, 121, dtype=np.float32) * 0.5


def stage_eval_A():
    import ws_eval as we
    import ws_metrics as wm
    f_rows = R / f"{TAG}_A_metrics.csv"
    rows = pd.read_csv(f_rows).to_dict("records") if f_rows.exists() else []
    done = {(r["posterior"], r["condition"]) for r in rows}
    for pname, (post, sset) in posteriors_A().items():
        for cond in CONDITIONS:
            if (pname, cond) in done:
                continue
            d = we.load_npz(OUT / f"A_{cond}.npz")
            x = we.features(d["raw"], _t(), sset)
            smp, leak = we.sample_within_prior(post, x, N_DRAWS, rng_seed=0)
            overall, sc_rows = wm.evaluate_scenario_set(smp, d["theta"], d["sc_name"], we.LO, we.HI)
            names = np.asarray(d["sc_name"])
            for r in sc_rows:
                m = names == r["scenario"]
                rows.append({"posterior": pname, "summary_set": sset, "condition": cond, **r,
                             "z_bias_b": r["bias_b"] / r["sd_b"], "z_bias_a": r["bias_a"] / r["sd_a"],
                             "leak_mean": float(leak[m].mean()), "macro_f1_all": overall["macro_f1"]})
            pd.DataFrame(rows).to_csv(f_rows, index=False)
            print(pname, cond, "done", flush=True)
    we.mark(TAG, "evalA_done")


def stage_alarm_A(z_thr: float = 3.0):
    """Disagreement of the two plug-in beta estimates as a misspecification alarm."""
    import ws_eval as we
    import summary_variants as sv
    const = sv.physics_constants()

    def dis(raw):
        e = sv.extra_features(raw, _t(), const)
        return e[:, 1] - e[:, 2], e[:, 0]            # ln b_jac - ln b_rx, ln alpha_hat

    cal = we.load_npz(OUT / "A_calibration_nominal.npz")
    g = np.array([str(n).split("_")[0] in GATED for n in cal["sc_name"]])
    dc, _ = dis(cal["raw"])
    mu, sd = float(dc[g].mean()), float(dc[g].std(ddof=1))
    per_sc = {s: (float(dc[np.asarray(cal["sc_name"]) == s].mean()), float(dc[np.asarray(cal["sc_name"]) == s].std(ddof=1)))
              for s in SCEN}
    rows = []
    for cond in CONDITIONS:
        d = we.load_npz(OUT / f"A_{cond}.npz")
        dd, _ = dis(d["raw"])
        names = np.asarray(d["sc_name"])
        for s in SCEN:
            m = names == s
            z = (dd[m] - mu) / sd
            zs = (dd[m] - per_sc[s][0]) / per_sc[s][1]
            rows.append({"condition": cond, "scenario": s, "d_mean": float(dd[m].mean()),
                         "shift_in_cal_sd": float((dd[m].mean() - mu) / sd),
                         "detect_rate_pooled": float((np.abs(z) > z_thr).mean()),
                         "detect_rate_per_scenario_cal": float((np.abs(zs) > z_thr).mean())})
    df = pd.DataFrame(rows)
    df.attrs.update(mu=mu, sd=sd)
    df.to_csv(R / f"{TAG}_A_alarm.csv", index=False)
    return df


def stage_decide_A():
    import ws_eval as we
    m = pd.read_csv(R / f"{TAG}_A_metrics.csv")
    prod = [p for p in m.posterior.unique() if p.startswith("production_")]
    assert len(prod) == 1, prod
    m = m[m.posterior == prod[0]].copy()
    m["short"] = m.scenario.str.split("_").str[0]
    nom = m[m.condition == "nominal"].set_index("short")
    bad = []
    for _, r in m[m.condition.isin(REALISTIC) & m.short.isin(GATED)].iterrows():
        drop = nom.loc[r.short, "class_acc"] - r.class_acc
        if r.cov90_truth_b < 0.80 or drop > 0.10:
            bad.append({"condition": r.condition, "scenario": r.short, "cov90_b": r.cov90_truth_b,
                        "class_acc": r.class_acc, "acc_drop": drop, "bias_b": r.bias_b, "z_bias_b": r.z_bias_b})
    res = {"posterior": prod[0], "robust": not bad, "run_part_B": bool(bad), "violations": bad,
           "rule": "see docstring of scripts/nb34_inlet_uncertainty.py (fixed before running)"}
    we.save_json(R / f"{TAG}_A_decision.json", res)
    print("ROBUST to realistic inlet uncertainty" if not bad else f"NOT robust: {len(bad)} violations -> Part B")
    if bad:
        print(pd.DataFrame(bad).round(3).to_string(index=False))
    return res


# ============================================================================ Part B
def _nuisance_inlets(n, seed):
    import ws_inlet as wi
    rng = np.random.default_rng(seed)
    nom = wi.NOMINAL.astype(np.float64)
    return np.stack([nom[0] * (1 + rng.normal(0, NUIS_SD["Ci_rel"], n)),
                     nom[1] + rng.normal(0, NUIS_SD["Ti"], n),
                     nom[2] + rng.normal(0, NUIS_SD["Tci"], n)], axis=1).astype(np.float32)


def stage_bank_B():
    import ws_inlet as wi
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    n = B_N * len(B_SEEDS)
    th = we.prior_draws(n, B_THETA_SEED)
    inl = {"nominal": np.broadcast_to(wi.NOMINAL, (n, 3)), "robust": _nuisance_inlets(n, B_INLET_SEED)}
    for kind in ("nominal", "robust"):
        for c in range(n // CHUNK):
            f = OUT / f"B_bank_{kind}_{SET}_chunk{c:02d}.npz"
            if f.exists():
                continue
            t0 = time.time()
            sl = slice(c * CHUNK, (c + 1) * CHUNK)
            x = wi.simulate(th[sl], inl[kind][sl], PROTOCOL, B_SIM_SEED * 10_000 + c * CHUNK, features=SET, **NOISE)
            we.save_npz(f, {"theta": th[sl], "inlet": np.asarray(inl[kind][sl]), "x": x})
            we.mark(TAG, f"bankB_{kind}_{c:02d}")
            print(kind, c, f"{time.time() - t0:.0f}s", flush=True)
    for kind in ("nominal", "robust"):                       # SBC sets under both inlet distributions
        f = OUT / f"B_sbc_{kind}_{SET}.npz"
        if not f.exists():
            ths = we.prior_draws(1000, B_SBC_SEED)
            il = np.broadcast_to(wi.NOMINAL, (1000, 3)) if kind == "nominal" else _nuisance_inlets(1000, B_SBC_SEED + 1)
            x = wi.simulate(ths, il, PROTOCOL, B_SBC_SEED * 1000, features=SET, **NOISE)
            we.save_npz(f, {"theta": ths, "inlet": np.asarray(il), "x": x})


def _bank(kind):
    import ws_eval as we
    parts = [we.load_npz(OUT / f"B_bank_{kind}_{SET}_chunk{c:02d}.npz") for c in range(B_N * len(B_SEEDS) // CHUNK)]
    return np.concatenate([p["theta"] for p in parts]), np.concatenate([p["x"] for p in parts])


def stage_train_B(threads: int = 4):
    import torch
    import ws_eval as we
    torch.set_num_threads(threads)
    for kind in ("nominal", "robust"):
        th, x = _bank(kind)
        for s in B_SEEDS:
            f = OUT / f"B_post_{kind}_{SET}_s{s}.pkl"
            if f.exists():
                continue
            sl = slice(s * B_N, (s + 1) * B_N)
            post, meta = we.train_npe(th[sl], x[sl], seed=s)
            meta.update({"kind": kind, "summary_set": SET, "nuisance_sd": NUIS_SD if kind == "robust" else None})
            with open(f, "wb") as fh:
                pickle.dump({"posterior": post, "metadata": meta}, fh)
            we.mark(TAG, f"trainB_{kind}_s{s}")
            print(kind, s, round(meta["train_s"]), "s", flush=True)


def stage_eval_B():
    import ws_eval as we
    import ws_metrics as wm
    f_sc, f_sbc = R / f"{TAG}_B_metrics.csv", R / f"{TAG}_B_sbc.csv"
    rows = pd.read_csv(f_sc).to_dict("records") if f_sc.exists() else []
    sbc_rows = pd.read_csv(f_sbc).to_dict("records") if f_sbc.exists() else []
    done = {(r["posterior"], r["condition"]) for r in rows}
    done_sbc = {(r["posterior"], r["sbc_inlet"]) for r in sbc_rows}
    for kind in ("nominal", "robust"):
        pname = f"B_{kind}_{SET}_40k_ens"
        post = _ensemble([OUT / f"B_post_{kind}_{SET}_s{s}.pkl" for s in B_SEEDS])
        for cond in CONDITIONS:
            if (pname, cond) in done:
                continue
            d = we.load_npz(OUT / f"A_{cond}.npz")
            x = we.features(d["raw"], _t(), SET)
            smp, leak = we.sample_within_prior(post, x, N_DRAWS, rng_seed=0)
            overall, sc_rows = wm.evaluate_scenario_set(smp, d["theta"], d["sc_name"], we.LO, we.HI)
            for r in sc_rows:
                rows.append({"posterior": pname, "condition": cond, **r, "z_bias_b": r["bias_b"] / r["sd_b"],
                             "macro_f1_all": overall["macro_f1"]})
            pd.DataFrame(rows).to_csv(f_sc, index=False)
        for sk in ("nominal", "robust"):
            if (pname, sk) in done_sbc:
                continue
            d = we.load_npz(OUT / f"B_sbc_{sk}_{SET}.npz")
            smp, leak = we.sample_within_prior(post, d["x"], N_DRAWS, rng_seed=1)
            r = wm.evaluate_sbc_set(smp, d["theta"], we.LO, we.HI, seed=0)
            sbc_rows.append({"posterior": pname, "sbc_inlet": sk, **r, "leak_mean": float(leak.mean())})
            pd.DataFrame(sbc_rows).to_csv(f_sbc, index=False)
        print(pname, "evaluated", flush=True)
    we.mark(TAG, "evalB_done")
