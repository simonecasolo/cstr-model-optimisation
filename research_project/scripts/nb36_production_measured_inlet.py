"""Notebook 36 (System I) -- production posterior with MEASURED inlet temperatures (N3, P2).
Stage functions called from notebooks/system_I/36_production_measured_inlet.ipynb; resumable.
Outputs: results/36<suffix>_* and results/36<suffix>/.

WHY. Notebook 34: the nominal-inlet production posterior (nb31) is confidently wrong under realistic
inlet errors. Notebook 35: conditioning on measured inlet temperatures removes that failure; the
instrument requirement is a calibration error of <= 0.1 K. The user adopted (2026-11-09) the
measured-inlet design with eps = 0.1 K (class-AA RTDs on the feed and coolant inlet) as the paper's
production model; 0.05 and 0.2 K are reported as sensitivity (nb35). nb35 trained at 40k only and
even its oracle failed TARP there, so the production posterior gets the same budget study and rule
as nb31.

DESIGN
  Plant and sensors exactly as nb35: inlet varies per window (Ti, Tci ~ U(295, 299) K; Ci +-1% sd,
  unmeasured); measured Ti, Tci = true + 0.1 K calibration offset N(0,1) + window-mean white noise.
  Features: nb35 "S1m" (34-D: 29 summaries + plug-in estimators with the measured inlet + measured
  Ti, Tci). Bank 160k (chunks of 10k), budgets 5k-160k, 5 seeds (nested subsets per seed, as nb31).
  Test: SBC set (1000 prior draws, inlet from the training distribution) and the closed-loop
  scenario grid Sc1-Sc5, Sc7 x 50 with the inlet drawn from the training distribution (Sc7 keeps its
  +2 K T-sensor drift).

DECISION RULE: nb31 rule v2.1, unchanged (scripts/nb31_budget_convergence.py: _steps_v2 and the
  v2.1 deployed-object principle). Fixed before running.
ACCEPTANCE (fixed before running): the production posterior is also evaluated on the nb35 inlet
  grid (true Tci, Ti each +-0.5, +-1, +-2 K; Sc1-Sc4) with nb35's deployability criteria (beta cov90
  >= 0.80; class accuracy >= nb35 oracle - 0.10). One violating cell is reported; more than one, or
  a flagged v2.1 decision, is a STOP for the user.
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
import nb31_budget_convergence as B
import nb35_measured_inlet as M

SENSOR, NOISE, PROTOCOL = CFG.model(), CFG.noise_kw(), CFG.protocol()
TAG = "36" + CFG.suffix()
R = ROOT / "results"
OUT = R / TAG
VARIANT = "eps0.1"
EPS = M.VARIANTS[VARIANT]
FEATURES = "S1m_measured_inlet"
BANK_N, CHUNK = 160_000, 10_000
BUDGETS, SEEDS = B.DEFAULT_BUDGETS, B.SEEDS
N_SBC, SC_REPS = B.N_SBC, B.SC_REPS
THETA_SEED, INLET_SEED, SIM_SEED, OFFSET_SEED = 3610, 3611, 3612, 3613
TEST_SEED = 3620
DECISION_RECORD = {"design": "measured inlet temperatures, eps = 0.1 K (class-AA RTD)", "date": "2026-11-09",
                   "approved_by": "user (Simone Casolo), 2026-11-09, in conversation with Claude",
                   "basis": "nb35: removes the nb34 silent failure; eps <= 0.1 K is the instrument requirement; "
                            "0.05 and 0.2 K reported as sensitivity"}


def _t():
    return np.arange(1, 121, dtype=np.float32) * 0.5


def features(base29, inlet_true, seed):
    return M.features_for(VARIANT, base29, inlet_true, seed)


def features_from_raw(raw, inlet_measured):
    """Downstream use on observed windows: raw (n, 120, 4) and MEASURED (Ti, Tci) per window."""
    import ws_eval as we
    return M.build_features(we.features(raw, _t(), "S0_base29"), np.asarray(inlet_measured, np.float64), "meas")


# ------------------------------------------------------------------ simulation
def stage_bank():
    import ws_inlet as wi
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    th = we.prior_draws(BANK_N, THETA_SEED)
    inl = M._inlets(BANK_N, INLET_SEED)
    for c in range(BANK_N // CHUNK):
        f = OUT / f"bank_chunk{c:02d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        b29 = wi.simulate(th[sl], inl[sl], PROTOCOL, SIM_SEED * 10_000 + c * CHUNK, features="S0_base29", **NOISE)
        we.save_npz(f, {"theta": th[sl], "inlet": inl[sl], "base29": b29})
        we.mark(TAG, f"bank_chunk{c:02d}")
        print(f"chunk {c}: {time.time() - t0:.0f}s", flush=True)


def load_bank():
    import ws_eval as we
    parts = [we.load_npz(OUT / f"bank_chunk{c:02d}.npz") for c in range(BANK_N // CHUNK)]
    th = np.concatenate([p["theta"] for p in parts])
    inl = np.concatenate([p["inlet"] for p in parts])
    b29 = np.concatenate([p["base29"] for p in parts])
    return th, features(b29, inl, OFFSET_SEED)


def stage_test():
    import ws_inlet as wi
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "test.npz"
    if f.exists():
        return
    th_sbc = we.prior_draws(N_SBC, TEST_SEED)
    inl_sbc = M._inlets(N_SBC, TEST_SEED + 1)
    b_sbc = wi.simulate(th_sbc, inl_sbc, PROTOCOL, TEST_SEED + 1_000_000, features="S0_base29", **NOISE)
    th_sc, names, drift = we.scenario_thetas(SC_REPS)
    inl_sc = M._inlets(len(th_sc), TEST_SEED + 2)
    raw = wi.simulate(th_sc, inl_sc, PROTOCOL, TEST_SEED + 2_000_000, raw=True, **NOISE)
    raw[:, :, 1] += drift[:, None]                    # Sc7 T-sensor drift, as ws_eval.simulate
    b_sc = we.features(raw, _t(), "S0_base29")
    we.save_npz(f, dict(theta_sbc=th_sbc, x_sbc=features(b_sbc, inl_sbc, TEST_SEED + 3),
                        theta_sc=th_sc, sc_name=names, x_sc=features(b_sc, inl_sc, TEST_SEED + 4),
                        inlet_sbc=inl_sbc, inlet_sc=inl_sc))
    we.mark(TAG, "test_done")


def _post_path(N, s):
    return OUT / f"post_n{N}_s{s}.pkl"


def stage_train(threads: int = 4):
    import torch
    import ws_eval as we
    torch.set_num_threads(threads)
    theta, x = load_bank()
    for s in SEEDS:
        perm = np.random.default_rng(36_000 + s).permutation(len(theta))
        for N in sorted(BUDGETS):
            f = _post_path(N, s)
            if f.exists():
                continue
            post, meta = we.train_npe(theta[perm[:N]], x[perm[:N]], seed=s)
            meta.update({"budget": N, "features": FEATURES, "eps": EPS, "protocol": PROTOCOL})
            with open(f, "wb") as fh:
                pickle.dump({"posterior": post, "metadata": meta}, fh)
            we.mark(TAG, f"train_n{N}_s{s}_done")
            print(N, s, round(meta["train_s"]), "s", flush=True)


def stage_eval():
    import ws_eval as we
    test = we.load_npz(OUT / "test.npz")
    f_rows, f_sc = R / f"{TAG}_metrics.csv", R / f"{TAG}_scenarios.csv"
    rows = pd.read_csv(f_rows).to_dict("records") if f_rows.exists() else []
    sc_rows = pd.read_csv(f_sc).to_dict("records") if f_sc.exists() else []
    done = {(int(r["budget"]), str(r["member"])) for r in rows}
    for N in sorted(BUDGETS):
        if not all(_post_path(N, s).exists() for s in SEEDS):
            continue
        members = {s: pickle.load(open(_post_path(N, s), "rb"))["posterior"] for s in SEEDS}
        todo = [(str(s), m) for s, m in members.items()] + [("ensemble", we.EnsemblePosterior(members.values()))]
        for name, post in todo:
            if (N, name) in done:
                continue
            row, scr = we.evaluate(post, test, seed=0)
            meta = {"budget": N, "member": name, "n_members": len(members) if name == "ensemble" else 1}
            rows.append({**meta, **row}); sc_rows += [{**meta, **r} for r in scr]
            pd.DataFrame(rows).to_csv(f_rows, index=False); pd.DataFrame(sc_rows).to_csv(f_sc, index=False)
            print(N, name, {k: round(row[k], 3) for k in ("ks_p_a_interior", "ks_p_b_interior", "tarp_ks_p", "cov90_b")}, flush=True)
    we.mark(TAG, "eval_done")


# ------------------------------------------------------------------ decision (nb31 rule v2.1)
def stage_decide():
    import ws_eval as we
    df = pd.read_csv(R / f"{TAG}_metrics.csv"); df["member"] = df["member"].astype(str)
    ens = df[df.member == "ensemble"].set_index("budget").sort_index()
    single = df[df.member != "ensemble"]
    steps = B._steps_v2(ens)
    Bs = list(ens.index)
    choice, how = None, ""
    for k, st in enumerate(steps):
        if all(x["converged"] for x in steps[k:]):
            choice, how = st["to"], f"first persistently converged step {st['from']}->{st['to']}"
            break
    if choice is None:
        calib = [N for N in Bs if B._passes_d(ens.loc[N])]
        choice = int(max(calib)) if calib else int(Bs[-1])
        how = ("no persistently converged step: largest budget passing calibration (d)" if calib
               else "no budget passes calibration: largest budget, FLAGGED")
    sb = single[single.budget == choice]
    flag = None if "FLAGGED" not in how else how
    ok = sb[[B._passes_d(r) for _, r in sb.iterrows()]]
    if B._passes_d(ens.loc[choice]):
        use_ens, members = True, [str(_post_path(choice, int(s))) for s in sb.member]
    elif len(ok):
        use_ens, members = False, [str(_post_path(choice, int(ok.loc[ok.tarp_dev.idxmin(), "member"])))]
    else:
        use_ens, members = True, [str(_post_path(choice, int(s))) for s in sb.member]
        flag = (flag or "") + f" FLAGGED: neither the ensemble nor any member passes calibration (d) at {choice}"
    res = {"protocol": PROTOCOL, "sensor_model": SENSOR, "sensor_noise": CFG.describe(), "summary_set": FEATURES,
           "inlet_model": {"mode": "measured", "eps_K": EPS, "white_sd_K": float(M.WHITE_SD),
                           "inlet_range_K": M.INLET_RANGE, "Ci_rel_sd": M.CI_REL_SD},
           "production_budget": int(choice), "use_ensemble": use_ens, "members": members, "flag": flag,
           "selection": how, "steps": steps, "rule": "nb31 v2.1 (unchanged; fixed before running)",
           "design_decision": DECISION_RECORD}
    we.save_json(R / f"{TAG}_production.json", res)
    print(pd.DataFrame(steps)[["from", "to", "a_corr", "b_cov", "c_tarp", "d_calib", "e_precision", "converged"]].to_string(index=False))
    print("production:", choice, "ensemble" if use_ens else "single", "| flag:", flag)
    return res


def load_production_posterior():
    import ws_eval as we
    res = json.loads((R / f"{TAG}_production.json").read_text())
    posts = [pickle.load(open(m, "rb"))["posterior"] for m in res["members"]]
    return (we.EnsemblePosterior(posts) if res["use_ensemble"] else posts[0]), res


# ------------------------------------------------------------------ acceptance on the nb35 inlet grid
def _review_exception(bad: list, review_path: Path):
    """If the user has recorded an approved, documented exception for exactly this outcome (same number of violating
    cells and, when the record lists them, the same cells), return the review dict, else None.
    Added 2026-11-09 so the notebook is reproducible after the user accepted the 4-cell exception
    (results/36_N3_acceptance_review.json). The acceptance criterion itself is unchanged."""
    if not review_path.exists():
        return None
    rev = json.loads(review_path.read_text())
    if rev.get("approved") is not True or rev.get("n_violations") != len(bad):
        return None
    cells = rev.get("violating_cells")
    if cells is not None and sorted(f"{b['scenario']}|{b['condition']}" for b in bad) != sorted(cells):
        return None
    return rev


def stage_accept():
    import ws_eval as we
    import ws_metrics as wm
    post, prod = load_production_posterior()
    d35 = R / ("35" + CFG.suffix())
    oracle = pd.read_csv(R / f"35{CFG.suffix()}_metrics.csv")
    oracle = oracle[oracle.variant == "oracle"].copy(); oracle["short"] = oracle.scenario.str.split("_").str[0]
    orc = oracle.set_index(["condition", "short"]).class_acc
    rows = []
    for cond in M.CONDITIONS:
        d = we.load_npz(d35 / f"test_{cond}.npz")
        x = M.features_for(VARIANT, we.features(d["raw"], _t(), "S0_base29"), d["inlet"], M.TEST_OFFSET_SEED)
        smp, leak = we.sample_within_prior(post, x, M.N_DRAWS, rng_seed=0)
        _, sc_rows = wm.evaluate_scenario_set(smp, d["theta"], d["sc_name"], we.LO, we.HI)
        for r in sc_rows:
            rows.append({"condition": cond, **r, "z_bias_b": r["bias_b"] / r["sd_b"]})
    df = pd.DataFrame(rows); df["short"] = df.scenario.str.split("_").str[0]
    df.to_csv(R / f"{TAG}_acceptance.csv", index=False)
    g = df[df.short.isin(M.GATED)]
    bad = [{"condition": r.condition, "scenario": r.short, "cov90_b": r.cov90_truth_b, "class_acc": r.class_acc,
            "oracle_acc": orc.loc[(r.condition, r.short)]}
           for _, r in g.iterrows() if r.cov90_truth_b < 0.80 or r.class_acc < orc.loc[(r.condition, r.short)] - 0.10]
    res = {"n_violations": len(bad), "violations": bad, "accepted": len(bad) <= 1 and not prod.get("flag"),
           "criteria": "nb35 deployability on the nb35 inlet grid; <= 1 violating cell and no v2.1 flag (fixed before running)"}
    we.save_json(R / f"{TAG}_acceptance.json", res)
    print(f"acceptance: {len(bad)} violating cell(s); accepted = {res['accepted']}")
    if not res["accepted"]:
        rev = _review_exception(bad, R / f"{TAG}_acceptance_review.json")
        if rev is not None:
            print("ACCEPTED WITH A DOCUMENTED EXCEPTION (user, " + str(rev.get("by")) + "):")
            print("  " + str(rev.get("exception")))
            print("  reason: " + str(rev.get("reason")))
            res = {**res, "accepted_with_documented_exception": True}      # the saved acceptance.json is not rewritten
            return df, res
        raise RuntimeError("STOP: the measured-inlet production posterior fails acceptance -- user decision")
    return df, res
