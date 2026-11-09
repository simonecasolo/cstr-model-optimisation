"""nb29 -- warm-start (initial-state) protocol study for System I, under the FIXED
controller law (2026-11-08). Standalone, resumable, stage-by-stage (house rule for long
runs: no nbclient; progress sentinels in results/29fx_progress_*.txt).

Outputs use the prefix ``29fx_`` (N0 sensor model, original run) or ``29fx_<model>_`` (e.g.
``29fx_N3_``; sensor model from ``system_I_config``), so they can never be confused with the
superseded old-law ``29_*`` files or with each other.

SELECTION CRITERION (fixed before any result was seen; do not edit after running E3)
-------------------------------------------------------------------------------------
"Optimal" = matches the situation the monitor faces and degrades gracefully under
mismatch -- NOT best-looking accuracy on its own test data (that is circular).
The realistic deployment situation is an INCIPIENT fault (P2 test column: theta drifts at
30-day degradation rates, noise on throughout). Rule, applied by ``select``:
  1. Eligible = on the P2 test column, seed-median marginal SBC KS p >= 0.01 for BOTH
     alpha and beta AND seed-median TARP KS p >= 0.01 (joint calibration).
  2. Among eligible: lowest seed-mean normalised posterior-mean RMSE on P2.
     Every protocol within 1 combined standard error of the best is tied.
  3. Tie-break 1: smallest worst-case TARP deviation over the other INCIPIENT-fault test
     columns P1, P1b, P3 (robustness to a mis-specified initial state). Differences < 0.01
     count as ties. The abrupt-fault column P0 is a different regime (it is what the
     misspecification check of E5 is for) and is reported, not used for selection.
  4. Tie-break 2: simplicity / cost, in the order P1 < P1b < P3 < P2 < P0.
  5. If no protocol is eligible: pick the lowest P2 TARP deviation and FLAG IT LOUDLY;
     the production budget study (nb31) must not start until the user has looked.

Stages
------
  e1       time scales (deterministic, no ensembles): closed-loop eigen time constants,
           settling after an abrupt fault (noise-free, tolerance far above float32
           precision), burn-in to stationarity from the Lyapunov solution (exact, no noisy
           ensemble estimate), P2 plant lag at realistic and 10x rates.        ~1 min
  e2       Fisher information per protocol and representation (29-D summaries with full
           and diagonal covariance, raw 120x4 trajectory with Ledoit-Wolf covariance,
           4 channel means), common random numbers for the finite differences.  ~10-20 min
  e3sim    training banks (3 x 5k per protocol, same theta draws across protocols) and
           test sets (500 SBC + 6 scenarios x 30 reps per test protocol).     ~5-10 min
  e3train  5 protocols x 3 seeds NSF trainings at 5k.  Parallelise with
           --protocols / --seeds in separate processes (set --threads).       ~10 min each
  e3eval   every trained posterior on every test column.                      ~20-40 min
  e4       optional window-length check (30/60/120 min) for P0, P1, P2, seed 0.
  select   applies the rule above -> results/29fx_selection.json

Usage
-----
  python scripts/nb29_ws_study.py e1
  python scripts/nb29_ws_study.py e2
  python scripts/nb29_ws_study.py e3sim
  python scripts/nb29_ws_study.py e3train --protocols P0,P1 --seeds 0,1,2 --threads 3   # one process
  python scripts/nb29_ws_study.py e3train --protocols P1b,P2,P3 --seeds 0,1,2 --threads 3
  python scripts/nb29_ws_study.py e3eval
  python scripts/nb29_ws_study.py select
  python scripts/nb29_ws_study.py e4          # optional
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import system_I_config as CFG
SENSOR = CFG.model()                 # set CSTR_SENSOR_MODEL before importing (notebook first cell)
NOISE = CFG.noise_kw()               # sensor-noise kwargs for every stochastic simulation
TAG = "29fx" + CFG.suffix()          # "29fx" for N0 (original run), "29fx_N3" for N3
PROTOCOLS = ["P0", "P1", "P1b", "P2", "P3"]
SIMPLICITY = ["P1", "P1b", "P3", "P2", "P0"]
N_TRAIN, N_SEEDS = 5_000, 3
N_SBC, SC_REPS = 500, 30
THETA_SEED = 2900          # training theta draws (shared by all protocols)
TEST_SEED = 2990           # reserved for test sets, never used for training
R = ROOT / "results"
OUT = R / TAG


def _paths():
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


# =========================================================================== E1
def stage_e1():
    import jax, jax.numpy as jnp
    from scipy.linalg import expm, solve_continuous_lyapunov
    import ws_protocols as ws
    from cstr_sbi.physics import compute_qc, NOMINAL_CTRL
    from cstr_sbi.simulator import DEFAULT_PROCESS_SIGMA
    import ws_eval as we
    we.mark(TAG, "e1_start")
    Kp, tau_i = float(NOMINAL_CTRL[0]), float(NOMINAL_CTRL[1])
    w_qc = np.array([0.0, Kp, 0.0, Kp / tau_i])          # dQc/d[C,T,Tc,I] when unsaturated
    q = np.asarray(DEFAULT_PROCESS_SIGMA, float) ** 2
    pts = {"Sc1": (1.0, 1.0), "Sc2": (1.0, 0.7), "Sc3": (0.7, 1.0), "Sc4": (0.85, 0.85),
           "Sc5": (1.0, 0.4), "Sc7": (1.0, 0.85), "c(0.4,0.4)": (0.4, 0.4),
           "c(0.4,1.2)": (0.4, 1.2), "c(1.2,0.4)": (1.2, 0.4), "c(1.2,1.2)": (1.2, 1.2)}
    y_h = np.asarray(ws._ss_batch(jnp.array([1.0]), jnp.array([1.0]))[0])
    rows = []
    for name, (a, b) in pts.items():
        yss = np.asarray(ws._ss_batch(jnp.array([a]), jnp.array([b]))[0])
        qss = float(compute_qc(yss[1], yss[3], NOMINAL_CTRL))
        J = np.asarray(ws._jac_batch(jnp.array([a]), jnp.array([b]), jnp.asarray(yss[None]))[0], float)
        act = [k for k in range(4) if np.abs(J[k]).max() > 1e-9 or np.abs(J[:, k]).max() > 1e-9]
        Ja = J[np.ix_(act, act)]
        ev = np.linalg.eigvals(Ja)
        saturated = 3 not in act
        zeta = [(-e.real / abs(e)) for e in ev if abs(e.imag) > 1e-12]
        # --- burn-in to stationarity, exact: Var(t) = P_inf - e^{Jt} P_inf e^{J't}
        Pinf = solve_continuous_lyapunov(Ja, -np.diag(q[act]))
        Pinf = 0.5 * (Pinf + Pinf.T)
        dg = np.diag(Pinf)
        wq = w_qc[act]
        t95 = np.nan
        for t in np.arange(0.0, 601.0, 1.0):
            M = expm(Ja * t)
            Rt = M @ Pinf @ M.T
            fr = [Rt[i, i] / dg[i] for i in range(len(act)) if dg[i] > 1e-30]
            if not saturated:
                fr.append(float(wq @ Rt @ wq) / max(float(wq @ Pinf @ wq), 1e-30))
            if max(fr) <= 0.05:
                t95 = t; break
        # --- settling after an abrupt fault from the healthy steady state (noise-free)
        n = 60_000                                       # 600 min at dt=0.01
        ys = np.asarray(ws._em_ramp(jnp.asarray(y_h), jax.random.PRNGKey(0),
                                    jnp.array([a, b]), jnp.array([a, b]), n_steps=n,
                                    sigma=jnp.zeros(4)))[49::50]          # every 0.5 min
        qc = np.asarray(jax.vmap(compute_qc, in_axes=(0, 0, None))(ys[:, 1], ys[:, 3], NOMINAL_CTRL))
        tt = np.arange(1, len(ys) + 1) * 0.5
        bad = ((np.abs(ys[:, 1] - yss[1]) > 0.05) | (np.abs(qc - qss) > 1.0)
               | (np.abs(ys[:, 0] - yss[0]) > 0.01 * abs(yss[0])))
        if not bad.any():
            t_settle = 0.0
        elif bad[-1]:
            t_settle = np.inf
        else:
            t_settle = float(tt[np.where(bad)[0][-1]])
        # --- P2 lag: noise-free history ramp ending at theta, realistic and 10x rates
        lags = {}
        for mult in (1.0, 10.0):
            d = mult * 300.0 * np.array([ws.RATE_ALPHA_PER_MIN, ws.RATE_BETA_PER_MIN])
            th0 = np.clip(np.array([a, b]) + d, we.LO, we.HI)
            y0 = np.asarray(ws._ss_batch(jnp.array([th0[0]]), jnp.array([th0[1]]))[0])
            ye = np.asarray(ws._em_ramp(jnp.asarray(y0), jax.random.PRNGKey(0), jnp.asarray(th0),
                                        jnp.array([a, b]), n_steps=30_000, sigma=jnp.zeros(4)))[-1]
            qe = float(compute_qc(ye[1], ye[3], NOMINAL_CTRL))
            lags[f"lag_Qc_x{int(mult)}"] = qe - qss
            lags[f"lag_T_x{int(mult)}"] = float(ye[1] - yss[1])
            lags[f"lag_C_rel_x{int(mult)}"] = float((ye[0] - yss[0]) / yss[0])
        rows.append({"point": name, "alpha": a, "beta": b, "saturated_ss": saturated,
                     "Qc_ss": qss, "T_ss": float(yss[1]),
                     "slowest_tau_min": float(1.0 / np.min(np.abs(ev.real))),
                     "fastest_tau_min": float(1.0 / np.max(np.abs(ev.real))),
                     "min_zeta": float(min(zeta)) if zeta else 1.0,
                     "t95_stationary_min": t95, "t_settle_abrupt_min": t_settle,
                     "sd_T_stat": float(np.sqrt(Pinf[act.index(1), act.index(1)])),
                     "sd_Qc_stat": float(np.sqrt(max(float(wq @ Pinf @ wq), 0))) if not saturated else 0.0,
                     **lags})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / f"{TAG}_e1_timescales.csv", index=False)
    we.mark(TAG, "e1_done")
    print(df.to_string(index=False))


# =========================================================================== E2
def _fim(G: np.ndarray, X: np.ndarray, mode: str) -> np.ndarray:
    """G (d,2) mean-derivatives, X (N,d) replicates at theta0. Gaussian FIM G' S^-1 G
    (the dS/dtheta term is neglected, as in main.tex sec. 4.3), on standardised features,
    dropping (near-)constant ones."""
    sd = X.std(axis=0)
    keep = sd > 1e-8 * (np.abs(X).mean(axis=0) + 1e-12)
    Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    Gz = G[keep] / sd[keep][:, None]
    if mode == "diag":
        S = np.eye(Z.shape[1])
    elif mode == "lw":
        from sklearn.covariance import LedoitWolf
        S = LedoitWolf().fit(Z).covariance_
    else:
        S = np.cov(Z, rowvar=False) + 1e-9 * np.eye(Z.shape[1])
    return Gz.T @ np.linalg.solve(S, Gz)


def stage_e2(n_rep: int = 1500, h: float = 0.01):
    import jax.numpy as jnp
    import ws_protocols as ws
    import ws_eval as we
    we.mark(TAG, "e2_start")
    pts = {"Sc1": (1.0, 1.0), "Sc2": (1.0, 0.7), "Sc3": (0.7, 1.0), "Sc4": (0.85, 0.85)}
    rows = []
    for P in PROTOCOLS:
        for name, th0 in pts.items():
            th0 = np.array(th0, np.float32)
            variants = [th0, th0 + [h, 0], th0 - [h, 0], th0 + [0, h], th0 - [0, h]]
            raws = [we.simulate(np.tile(v, (n_rep, 1)), P, seed=29_200, raw=True, **NOISE) for v in variants]
            t = np.arange(1, raws[0].shape[1] + 1) * 0.5
            summ = [np.asarray(ws.summaries(jnp.asarray(r), jnp.asarray(t))) for r in raws]
            reps = {
                "summaries_fullcov": (summ, "full"),
                "summaries_diagcov": (summ, "diag"),
                "raw_traj_lwcov": ([r.reshape(n_rep, -1) for r in raws], "lw"),
                "channel_means": ([r.mean(axis=1) for r in raws], "full"),
            }
            for rep_name, (arrs, mode) in reps.items():
                X0, ap, am, bp, bm = arrs
                G = np.stack([(ap - am).mean(0) / (2 * h), (bp - bm).mean(0) / (2 * h)], axis=1)
                F = _fim(G, X0, mode)
                Finv = np.linalg.pinv(F)
                rows.append({"protocol": P, "point": name, "representation": rep_name,
                             "I_aa": F[0, 0], "I_bb": F[1, 1], "I_ab": F[0, 1],
                             "corr_implied": float(-F[0, 1] / np.sqrt(F[0, 0] * F[1, 1])),
                             "crb_sd_a": float(np.sqrt(max(Finv[0, 0], 0))),
                             "crb_sd_b": float(np.sqrt(max(Finv[1, 1], 0))),
                             "ratio_Iaa_Ibb": float(F[0, 0] / F[1, 1])})
            print(P, name, "done", flush=True)
            pd.DataFrame(rows).to_csv(R / f"{TAG}_e2_fim.csv", index=False)
    we.mark(TAG, "e2_done")
    print(pd.DataFrame(rows).pivot_table(index=["point", "representation"], columns="protocol",
                                         values="corr_implied").round(3).to_string())


# =========================================================================== E3
def stage_e3sim():
    import ws_eval as we
    out = _paths()
    we.mark(TAG, "e3sim_start")
    th = we.prior_draws(N_TRAIN * N_SEEDS, THETA_SEED)
    for P in PROTOCOLS:
        f = out / f"bank_{P}.npz"
        if not f.exists():
            t0 = time.time()
            x = we.simulate(th, P, seed=THETA_SEED * 1000, **NOISE)
            we.save_npz(f, {"theta": th, "x": x})
            print(f"bank {P}: {x.shape} in {time.time()-t0:.0f}s", flush=True)
        g = out / f"test_{P}.npz"
        if not g.exists():
            we.save_npz(g, we.make_test_set(P, TEST_SEED, N_SBC, SC_REPS, **NOISE))
            print(f"test {P} done", flush=True)
    we.mark(TAG, "e3sim_done")


def stage_e3train(protocols, seeds, threads):
    import torch
    import ws_eval as we
    torch.set_num_threads(threads)
    out = _paths()
    for P in protocols:
        bank = we.load_npz(out / f"bank_{P}.npz")
        for s in seeds:
            f = out / f"post_{P}_s{s}.pkl"
            if f.exists():
                continue
            we.mark(TAG, f"e3train_{P}_s{s}_start")
            sl = slice(s * N_TRAIN, (s + 1) * N_TRAIN)      # each seed: its own 5k draws
            post, meta = we.train_npe(bank["theta"][sl], bank["x"][sl], seed=s)
            with open(f, "wb") as fh:
                pickle.dump({"posterior": post, "metadata": {**meta, "protocol": P}}, fh)
            we.save_json(out / f"post_{P}_s{s}_meta.json", {**meta, "protocol": P})
            we.mark(TAG, f"e3train_{P}_s{s}_done")
            print(P, s, meta, flush=True)


def stage_e3eval():
    import ws_eval as we
    out = _paths()
    tests = {P: we.load_npz(out / f"test_{P}.npz") for P in PROTOCOLS}
    f_rows, f_sc = R / f"{TAG}_e3_matrix.csv", R / f"{TAG}_e3_scenarios.csv"
    rows = pd.read_csv(f_rows).to_dict("records") if f_rows.exists() else []
    sc_rows = pd.read_csv(f_sc).to_dict("records") if f_sc.exists() else []
    done = {(r["train"], int(r["seed"]), r["test"]) for r in rows}
    for P in PROTOCOLS:
        for s in range(N_SEEDS):
            pf = out / f"post_{P}_s{s}.pkl"
            if not pf.exists():
                print("MISSING", pf); continue
            post = pickle.load(open(pf, "rb"))["posterior"]
            for Q in PROTOCOLS:
                if (P, s, Q) in done:
                    continue
                row, scr = we.evaluate(post, tests[Q], seed=s)
                rows.append({"train": P, "seed": s, "test": Q, **row})
                sc_rows += [{"train": P, "seed": s, "test": Q, **r} for r in scr]
                pd.DataFrame(rows).to_csv(f_rows, index=False)
                pd.DataFrame(sc_rows).to_csv(f_sc, index=False)
                print(P, s, Q, {k: round(row[k], 3) for k in
                                ("ks_p_a", "ks_p_b", "tarp_ks_p", "rmse_norm", "macro_f1", "corr_Sc4")},
                      flush=True)
            we.mark(TAG, f"e3eval_{P}_s{s}_done")


# ======================================================================= select
def stage_select():
    import ws_eval as we
    df = pd.read_csv(R / f"{TAG}_e3_matrix.csv")
    df["min_ks"] = df[["ks_p_a", "ks_p_b"]].min(axis=1)
    p2 = df[df.test == "P2"].groupby("train").agg(
        min_ks_med=("min_ks", "median"), tarp_ks_med=("tarp_ks_p", "median"),
        tarp_dev=("tarp_dev", "mean"), rmse=("rmse_norm", "mean"),
        rmse_se=("rmse_norm", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        f1=("macro_f1", "mean"), n_seeds=("seed", "nunique"))
    inc = df[df.test.isin(["P1", "P1b", "P3"])]
    p2["worst_offcol_tarp_dev"] = inc.groupby(["train", "test"]).tarp_dev.mean().groupby("train").max()
    p2["P0col_tarp_dev"] = df[df.test == "P0"].groupby("train").tarp_dev.mean()
    p2["P0col_rmse"] = df[df.test == "P0"].groupby("train").rmse_norm.mean()
    p2["eligible"] = (p2.min_ks_med >= 0.01) & (p2.tarp_ks_med >= 0.01)
    log = []
    if p2.eligible.any():
        el = p2[p2.eligible]
        best = el.rmse.idxmin()
        se_best = el.loc[best, "rmse_se"]
        tied = el[el.rmse <= el.loc[best, "rmse"] + np.sqrt(se_best ** 2 + el.rmse_se ** 2)]
        log.append(f"eligible: {list(el.index)}; best RMSE {best}; tied within 1 SE: {list(tied.index)}")
        wmin = tied.worst_offcol_tarp_dev.min()
        tied2 = tied[tied.worst_offcol_tarp_dev <= wmin + 0.01]
        log.append(f"after robustness tie-break (<=0.01 of best worst-case TARP dev): {list(tied2.index)}")
        choice = sorted(tied2.index, key=SIMPLICITY.index)[0]
        flag = None
    else:
        choice = p2.tarp_dev.idxmin()
        flag = ("NO PROTOCOL PASSES CALIBRATION ON THE P2 COLUMN -- chosen by lowest TARP deviation. "
                f"Do NOT start nb31 until the user has reviewed results/{TAG}_e3_matrix.csv.")
        log.append(flag)
    res = {"criterion": "see docstring of scripts/nb29_ws_study.py (fixed before results)",
           "sensor_model": SENSOR,
           "choice": choice, "flag": flag, "log": log,
           "table_P2_column": p2.reset_index().to_dict("records")}
    we.save_json(R / f"{TAG}_selection.json", res)
    print(p2.round(4).to_string()); print("\n".join(log)); print("CHOICE:", choice)


# =========================================================================== E4
def stage_e4(threads):
    import torch
    import ws_eval as we
    torch.set_num_threads(threads)
    out = _paths()
    th = we.prior_draws(N_TRAIN, THETA_SEED + 4)
    rows, tests = [], {}
    for win in (30.0, 60.0, 120.0):
        for P in ("P0", "P1", "P2"):
            x = we.simulate(th, P, seed=29_400, t_window=win, **NOISE)
            post, meta = we.train_npe(th, x, seed=0)
            for Q in sorted({P, "P2"}):
                if (win, Q) not in tests:
                    tests[(win, Q)] = we.make_test_set(Q, TEST_SEED + 4, N_SBC, 20, t_window=win, **NOISE)
                row, _ = we.evaluate(post, tests[(win, Q)], seed=0)
                rows.append({"window_min": win, "train": P, "test": Q, **row})
                pd.DataFrame(rows).to_csv(R / f"{TAG}_e4_windows.csv", index=False)
                print(win, P, Q, round(row["rmse_norm"], 4), round(row["tarp_ks_p"], 3), flush=True)
    we.mark(TAG, "e4_done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["e1", "e2", "e3sim", "e3train", "e3eval", "select", "e4"])
    ap.add_argument("--protocols", default=",".join(PROTOCOLS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(N_SEEDS)))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "e1":
        stage_e1()
    elif a.stage == "e2":
        stage_e2()
    elif a.stage == "e3sim":
        stage_e3sim()
    elif a.stage == "e3train":
        stage_e3train(a.protocols.split(","), [int(s) for s in a.seeds.split(",")], a.threads)
    elif a.stage == "e3eval":
        stage_e3eval()
    elif a.stage == "select":
        stage_select()
    elif a.stage == "e4":
        stage_e4(a.threads)
