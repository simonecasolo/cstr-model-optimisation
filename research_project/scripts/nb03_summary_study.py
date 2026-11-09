"""Notebook 03, section 10 -- can the summary statistics be improved? (fixed controller law,
P1b warm start). Stage functions called from notebooks/system_I/03_summary_statistics_design.ipynb;
resumable (each stage skips work already on disk). Outputs: results/03s_*.

Candidate sets are defined in scripts/summary_variants.py (S0 = current 29-D production set).

DECISION RULE (fixed before results; do not edit after running ``train``)
-------------------------------------------------------------------------
At the 20k budget, per set, using seed means over 3 seeds:
  calib_ok : seed-median interior marginal SBC KS p >= 0.01 for alpha AND beta,
             and seed-median TARP KS p >= 0.01.
  eff      : mean over the scenario points shared with nb29 E2 (Sc1-Sc4 at their nb29 values)
             and over (alpha, beta) of  posterior sd / raw-trajectory Cramer-Rao sd
             (nb29 E2, same protocol, Ledoit-Wolf). Lower = closer to the information in the data.
  rmse     : prior-normalised posterior-mean RMSE on the SBC set.
A candidate REPLACES S0 only if calib_ok and rmse <= rmse_S0 + 1 SE and either
  (a) eff <= 0.9 * eff_S0 with the difference larger than 2 combined SE, or
  (b) S0 is not calib_ok and the candidate's eff <= eff_S0 (a pure calibration gain).
Among winners: the lowest eff; any other winner within 1 SE with FEWER features is preferred.
No winner -> keep S0 (the paper's summaries do not change).
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
SENSOR = CFG.model()                 # set CSTR_SENSOR_MODEL before importing (notebook cell)
NOISE = CFG.noise_kw()
TAG = "03s" + CFG.suffix()           # "03s" for N0 (original run), "03s_N3" for N3
E2_TAG = "29fx" + CFG.suffix()       # nb29 E2 Fisher information for the same sensor model
R = ROOT / "results"
OUT = R / TAG
PROTOCOL = CFG.protocol()            # nb29 choice for this sensor model (N0: P1b, N3: P2)
N_PER_SEED, SEEDS, BUDGETS = 20_000, [0, 1, 2], [5_000, 20_000]
N_SBC, SC_REPS = 1000, 50
THETA_SEED, TEST_SEED = 3300, 3399
CHUNK = 5_000
E2_POINTS = {"Sc1": (1.0, 1.0), "Sc2": (1.0, 0.7), "Sc3": (0.7, 1.0), "Sc4": (0.85, 0.85)}


def _union(theta, seed, drift_T=None):
    """Simulate raw windows under PROTOCOL and return the union feature matrix (n, 29+13)."""
    import jax.numpy as jnp
    import ws_eval as we
    import ws_protocols as ws
    import summary_variants as sv
    const = sv.physics_constants()
    out = []
    n = len(theta)
    drift = np.zeros(n, np.float32) if drift_T is None else np.broadcast_to(np.asarray(drift_T, np.float32), (n,))
    for c in range(0, n, CHUNK):
        raw = we.simulate(theta[c:c + CHUNK], PROTOCOL, seed + c, raw=True, drift_T=drift[c:c + CHUNK], **NOISE)
        t = np.arange(1, raw.shape[1] + 1) * 0.5
        base = np.asarray(ws.summaries(jnp.asarray(raw), jnp.asarray(t)))
        out.append(np.concatenate([base, sv.extra_features(raw, t, const)], axis=1).astype(np.float32))
    return np.concatenate(out)


def stage_sim():
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "train_union.npz"
    if not f.exists():
        th = we.prior_draws(N_PER_SEED * len(SEEDS), THETA_SEED)
        t0 = time.time()
        x = _union(th, THETA_SEED * 1000)
        we.save_npz(f, {"theta": th, "x": x})
        print(f"training union {x.shape} in {time.time() - t0:.0f}s", flush=True)
    g = OUT / "test_union.npz"
    if not g.exists():
        th_sbc = we.prior_draws(N_SBC, TEST_SEED)
        x_sbc = _union(th_sbc, TEST_SEED + 1_000_000)
        th_sc, names, drift = we.scenario_thetas(SC_REPS)
        x_sc = _union(th_sc, TEST_SEED + 2_000_000, drift_T=drift)
        we.save_npz(g, dict(theta_sbc=th_sbc, x_sbc=x_sbc, theta_sc=th_sc, sc_name=names, x_sc=x_sc))
    we.mark(TAG, "sim_done")


def stage_fim(n_rep: int = 1500, h: float = 0.01):
    """Gaussian-FIM Cramer-Rao sd per candidate set at the nb29 E2 points (same method as nb29)."""
    import nb29_ws_study as s29
    import summary_variants as sv
    rows = []
    for name, (a, b) in E2_POINTS.items():
        th0 = np.array([a, b], np.float32)
        var = [th0, th0 + [h, 0], th0 - [h, 0], th0 + [0, h], th0 - [0, h]]
        U = [_union(np.tile(v, (n_rep, 1)), seed=33_200) for v in var]   # common random numbers
        for sname, idx in sv.SETS.items():
            X0, ap, am, bp, bm = (u[:, idx].astype(np.float64) for u in U)
            G = np.stack([(ap - am).mean(0) / (2 * h), (bp - bm).mean(0) / (2 * h)], axis=1)
            F = s29._fim(G, X0, "full")
            Fi = np.linalg.pinv(F)
            rows.append({"point": name, "set": sname, "n_features": len(idx),
                         "crb_sd_a": float(np.sqrt(max(Fi[0, 0], 0))), "crb_sd_b": float(np.sqrt(max(Fi[1, 1], 0))),
                         "corr_implied": float(-F[0, 1] / np.sqrt(F[0, 0] * F[1, 1]))})
        print(name, "done", flush=True)
    pd.DataFrame(rows).to_csv(R / f"{TAG}_fim.csv", index=False)


def stage_train(threads: int = 4):
    import torch
    import ws_eval as we
    import summary_variants as sv
    torch.set_num_threads(threads)
    d = we.load_npz(OUT / "train_union.npz")
    for sname, idx in sv.SETS.items():
        for s in SEEDS:
            block = slice(s * N_PER_SEED, (s + 1) * N_PER_SEED)
            for N in BUDGETS:
                f = OUT / f"post_{sname}_n{N}_s{s}.pkl"
                if f.exists():
                    continue
                th = d["theta"][block][:N]
                x = d["x"][block][:N][:, idx]
                post, meta = we.train_npe(th, x, seed=s)
                meta.update({"set": sname, "budget": N})
                with open(f, "wb") as fh:
                    pickle.dump({"posterior": post, "metadata": meta}, fh)
                we.mark(TAG, f"train_{sname}_n{N}_s{s}")
                print(sname, N, s, round(meta["train_s"]), "s", meta["epochs"], "epochs", flush=True)


def stage_eval():
    import ws_eval as we
    import summary_variants as sv
    test = we.load_npz(OUT / "test_union.npz")
    f_rows = R / f"{TAG}_metrics.csv"
    rows = pd.read_csv(f_rows).to_dict("records") if f_rows.exists() else []
    done = {(r["set"], int(r["budget"]), int(r["seed"])) for r in rows}
    for sname, idx in sv.SETS.items():
        sub = dict(test, x_sbc=test["x_sbc"][:, idx], x_sc=test["x_sc"][:, idx])
        for N in BUDGETS:
            for s in SEEDS:
                if (sname, N, s) in done:
                    continue
                post = pickle.load(open(OUT / f"post_{sname}_n{N}_s{s}.pkl", "rb"))["posterior"]
                row, _ = we.evaluate(post, sub, seed=s)
                rows.append({"set": sname, "n_features": len(idx), "budget": N, "seed": s, **row})
                pd.DataFrame(rows).to_csv(f_rows, index=False)
                print(sname, N, s, {k: round(row[k], 4) for k in ("ks_p_a_interior", "ks_p_b_interior",
                                                                    "tarp_ks_p", "rmse_norm", "macro_f1")}, flush=True)
    we.mark(TAG, "eval_done")


def efficiency_table(budget: int = 20_000) -> pd.DataFrame:
    """Per (set, seed): mean posterior sd / raw-trajectory CRB over the E2 points present in
    the evaluated catalogue (Sc1-Sc4 at their nb29 values)."""
    from cstr_sbi.scenarios import SCENARIO_CONFIGS
    import ws_eval as we
    m = pd.read_csv(R / f"{TAG}_metrics.csv")
    m = m[m.budget == budget]
    e2 = pd.read_csv(R / f"{E2_TAG}_e2_fim.csv")
    e2 = e2[(e2.protocol == PROTOCOL) & (e2.representation == "raw_traj_lwcov")].set_index("point")
    pts = []
    for sc, (a, b) in E2_POINTS.items():
        cfg = [k for k in we.EVAL_SCENARIOS if k.startswith(sc + "_")][0]
        if np.isclose(SCENARIO_CONFIGS[cfg].alpha, a) and np.isclose(SCENARIO_CONFIGS[cfg].beta, b):
            pts.append(sc)
    rows = []
    for _, r in m.iterrows():
        ratios = [r[f"sda_{p}"] / e2.loc[p, "crb_sd_a"] for p in pts] + [r[f"sdb_{p}"] / e2.loc[p, "crb_sd_b"] for p in pts]
        rows.append({"set": r["set"], "seed": r["seed"], "n_features": r["n_features"], "eff": float(np.mean(ratios)),
                     **{f"eff_a_{p}": r[f"sda_{p}"] / e2.loc[p, "crb_sd_a"] for p in pts},
                     **{f"eff_b_{p}": r[f"sdb_{p}"] / e2.loc[p, "crb_sd_b"] for p in pts}})
    out = pd.DataFrame(rows)
    out.attrs["points"] = pts
    return out


def stage_decide(budget: int = 20_000):
    import ws_eval as we
    m = pd.read_csv(R / f"{TAG}_metrics.csv")
    m = m[m.budget == budget]
    eff = efficiency_table(budget)
    se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
    agg = m.groupby("set").agg(n_features=("n_features", "first"),
                               ks_a_int=("ks_p_a_interior", "median"), ks_b_int=("ks_p_b_interior", "median"),
                               tarp_ks=("tarp_ks_p", "median"), rmse=("rmse_norm", "mean"), rmse_se=("rmse_norm", se),
                               f1=("macro_f1", "mean"), cov90_a=("cov90_a", "mean"), cov90_b=("cov90_b", "mean"))
    e = eff.groupby("set").eff.agg(["mean", se])
    e.columns = ["eff", "eff_se"]
    agg = agg.join(e)
    agg["calib_ok"] = (agg.ks_a_int >= 0.01) & (agg.ks_b_int >= 0.01) & (agg.tarp_ks >= 0.01)
    s0 = agg.loc["S0_base29"]
    win = []
    for name, r in agg.drop(index="S0_base29").iterrows():
        if not r.calib_ok or r.rmse > s0.rmse + r.rmse_se:
            continue
        a = (r.eff <= 0.9 * s0.eff) and (s0.eff - r.eff > 2 * np.hypot(r.eff_se, s0.eff_se))
        b = (not s0.calib_ok) and (r.eff <= s0.eff)
        if a or b:
            win.append(name)
    if win:
        best = agg.loc[win].eff.idxmin()
        near = [w for w in win if agg.loc[w].eff <= agg.loc[best].eff + agg.loc[best].eff_se]
        choice = min(near, key=lambda w: agg.loc[w].n_features)
    else:
        choice = "S0_base29"
    res = {"budget": budget, "sensor_model": SENSOR, "choice": choice, "winners": win, "eff_points": eff.attrs["points"],
           "table": agg.reset_index().to_dict("records"),
           "rule": "see docstring of scripts/nb03_summary_study.py (fixed before results)"}
    we.save_json(R / f"{TAG}_decision.json", res)
    print(agg.round(4).to_string()); print("winners:", win, "| CHOICE:", choice)
    return res
