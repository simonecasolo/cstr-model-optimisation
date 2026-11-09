"""nb31 -- training-budget convergence for the System I production posterior, under the
FIXED controller law and the warm-start protocol chosen by nb29
(results/29fx_selection.json). Standalone, resumable, stage-by-stage.

Design
------
- ONE simulation bank (160k prior draws) under the chosen protocol, simulated once
  (nb29 measured ~2 min per 75k simulations with ws_protocols, so this is cheap).
- Budgets N in {5k, 10k, 20k, 40k, 80k, 160k} x seeds {0..4}. Seed s uses the first N
  rows of its own permutation of the bank, so for a fixed seed the training sets are
  NESTED across budgets (budget effects are not confounded with fresh-data noise) while
  different seeds see different data (seed spread includes data variation).
- A held-out test set (1000 SBC draws + 6 scenarios x 50 reps), simulated once from a
  reserved seed under the same protocol, never used for training.
- Every (N, seed) posterior and the equal-weight 5-seed ensemble at each N are scored with
  the same suite as nb29 (marginal SBC KS, TARP joint coverage, 90/50% coverage, rank
  shape, leakage, RMSE, per-scenario bias / coverage / Corr(alpha,beta), macro-F1).

CONVERGENCE RULE (fixed before results; do not edit after running)
-------------------------------------------------------------------
Budget step N_k -> N_{k+1} is CONVERGED if, using the ensemble posterior at both budgets:
  (a) |delta Corr(alpha,beta)| < 0.05 at each of Sc1, Sc2, Sc3, Sc4, Sc7
      (Sc5, saturated, is reported but not gated: its posterior is prior-bounded);
  (b) |delta 90% marginal coverage| < 0.02 for alpha and beta (SBC set);
  (c) |delta TARP deviation| < 0.01;
  (d) at N_{k+1}: marginal KS p >= 0.01 for alpha and beta on the INTERIOR SBC subset
      (true theta >= 5% of the prior width from every edge) and TARP KS p >= 0.01 on the
      full set. Full-set marginal KS is reported alongside.
      [Amended 2026-11-08 after nb29 and before any nb31 result: in nb29 every protocol,
      including matched train/test pairs, failed the full-set alpha KS at 5k while 90%
      coverage was 0.91-0.93; the interior gate separates estimator bias from prior-edge
      truncation, a known, separate NPE failure mode. Both are reported.]
PRODUCTION BUDGET = N_{k+1} of the first converged step (already trained and shown stable).
If no step converges, report the largest budget and flag that 80k (or more) is needed.

ENSEMBLE vs SINGLE SEED (fixed before results)
-----------------------------------------------
Use the ensemble at the production budget if its TARP deviation <= the median single-seed
TARP deviation at that budget (ensembles of NPEs are typically better calibrated,
Hermans et al. 2022); otherwise use the single seed with the lowest TARP deviation.
Write the choice and the member pickles to results/31<suffix>_production.json (suffix "" for
N0, "_N3" for N3; sensor model from system_I_config); downstream
notebooks load it with ``load_production_posterior()`` below.

Timing (measured in nb29, M-series CPU, NSF 128/5, batch 256): ~20 s per 5k training run;
expect minutes at 40k and ~10-30 min at 160k per seed. Seeds can run in parallel processes
with --seeds and --threads.

Usage
-----
  python scripts/nb31_budget_convergence.py bank                    # 160k sims, chunked, resumable
  python scripts/nb31_budget_convergence.py test
  python scripts/nb31_budget_convergence.py train --seeds 0,1 --threads 3
  python scripts/nb31_budget_convergence.py train --seeds 2,3,4 --threads 3
  python scripts/nb31_budget_convergence.py eval
  python scripts/nb31_budget_convergence.py decide
"""
from __future__ import annotations

import argparse
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
SENSOR = CFG.model()                 # set CSTR_SENSOR_MODEL before importing (notebook first cell)
NOISE = CFG.noise_kw()
TAG = "31" + CFG.suffix()            # "31" for N0 (original run), "31_N3" for N3
E2_TAG = "29fx" + CFG.suffix()       # nb29 outputs for the same sensor model
SET = CFG.summary_set()              # production summary set: N0 S0_base29 (29), N3 S1_base_est (32); notebook 03 s.10
R = ROOT / "results"
OUT = R / ("31_budget" + CFG.suffix())
BANK_N, CHUNK = 160_000, 10_000
BANK_SEED, TEST_SEED = 3100, 3199
N_SBC, SC_REPS = 1000, 50
DEFAULT_BUDGETS = [5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
SEEDS = [0, 1, 2, 3, 4]
GATED = ["Sc1", "Sc2", "Sc3", "Sc4", "Sc7"]


def _tok(P: str) -> str:
    """File-name token: the protocol, plus the summary set unless it is S0_base29. Anything built with another
    set therefore cannot be picked up by mistake; S0 keeps the legacy names so that the existing N0 results
    (built with S0, never to be overwritten) stay reproducible."""
    return P if SET == "S0_base29" else f"{P}_{SET}"


def protocol() -> str:
    f = R / f"{E2_TAG}_selection.json"
    if not f.exists():
        sys.exit(f"results/{E2_TAG}_selection.json missing: run nb29 (select) first, or pass --protocol")
    sel = json.loads(f.read_text())
    if sel.get("flag"):
        rv = R / f"{E2_TAG}_selection_review.json"     # written by nb29's review cell
        if rv.exists():
            rev = json.loads(rv.read_text())
            if rev.get("approved") and rev.get("choice"):
                print(f"nb29 selection was flagged; proceeding with user-approved {rev['choice']}: {rev.get('reason','')}")
                return rev["choice"]
        sys.exit(f"nb29 selection is FLAGGED, not starting nb31 without user review:\n{sel['flag']}\n"
                 "Review notebook 29 and set USER_APPROVES_CHOICE = True in its review cell.")
    return sel["choice"]


def stage_bank(P):
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    th = we.prior_draws(BANK_N, BANK_SEED)
    for c in range(BANK_N // CHUNK):
        f = OUT / f"bank_{_tok(P)}_chunk{c:02d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        x = we.simulate(th[sl], P, seed=BANK_SEED * 10_000 + c * CHUNK, features=SET, **NOISE)
        we.save_npz(f, {"theta": th[sl], "x": x})
        we.mark(TAG, f"bank_{_tok(P)}_chunk{c:02d}")
        print(f"chunk {c}: {time.time()-t0:.0f}s", flush=True)


def load_bank(P):
    import ws_eval as we
    parts = [we.load_npz(OUT / f"bank_{_tok(P)}_chunk{c:02d}.npz") for c in range(BANK_N // CHUNK)]
    return np.concatenate([p["theta"] for p in parts]), np.concatenate([p["x"] for p in parts])


def stage_test(P):
    import ws_eval as we
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"test_{_tok(P)}.npz"
    if not f.exists():
        we.save_npz(f, we.make_test_set(P, TEST_SEED, N_SBC, SC_REPS, features=SET, **NOISE))
    we.mark(TAG, f"test_{_tok(P)}_done")


def _post_path(P, N, s):
    return OUT / f"post_{_tok(P)}_n{N}_s{s}.pkl"


def stage_train(P, budgets, seeds, threads):
    import torch
    import ws_eval as we
    torch.set_num_threads(threads)
    theta, x = load_bank(P)
    for s in seeds:
        perm = np.random.default_rng(31_000 + s).permutation(len(theta))
        for N in sorted(budgets):
            f = _post_path(P, N, s)
            if f.exists():
                continue
            we.mark(TAG, f"train_{_tok(P)}_n{N}_s{s}_start")
            idx = perm[:N]
            post, meta = we.train_npe(theta[idx], x[idx], seed=s)
            meta.update({"protocol": P, "budget": N, "summary_set": SET})
            with open(f, "wb") as fh:
                pickle.dump({"posterior": post, "metadata": meta}, fh)
            we.save_json(f.with_suffix(".json"), meta)
            we.mark(TAG, f"train_{_tok(P)}_n{N}_s{s}_done")
            print(meta, flush=True)


def stage_eval(P):
    import ws_eval as we
    test = we.load_npz(OUT / f"test_{_tok(P)}.npz")
    f_rows, f_sc = R / f"{TAG}_metrics.csv", R / f"{TAG}_scenarios.csv"
    rows = pd.read_csv(f_rows).to_dict("records") if f_rows.exists() else []
    sc_rows = pd.read_csv(f_sc).to_dict("records") if f_sc.exists() else []
    done = {(int(r["budget"]), str(r["member"])) for r in rows}
    budgets = sorted({int(p.stem.split("_n")[-1].split("_")[0]) for p in OUT.glob(f"post_{_tok(P)}_n*_s*.pkl")})
    for N in budgets:
        missing = [s for s in SEEDS if not _post_path(P, N, s).exists()]
        if missing:
            print(f"budget {N}: waiting for seeds {missing} -- not evaluated yet (ensemble needs all)")
            continue
        members = {s: pickle.load(open(_post_path(P, N, s), "rb"))["posterior"] for s in SEEDS}
        todo = [(str(s), m) for s, m in members.items()]
        if len(members) >= 2:
            todo.append(("ensemble", we.EnsemblePosterior(members.values())))
        for name, post in todo:
            if (N, name) in done:
                continue
            row, scr = we.evaluate(post, test, seed=0)
            meta = {"budget": N, "member": name, "n_members": len(members) if name == "ensemble" else 1}
            rows.append({**meta, **row})
            sc_rows += [{**meta, **r} for r in scr]
            pd.DataFrame(rows).to_csv(f_rows, index=False)
            pd.DataFrame(sc_rows).to_csv(f_sc, index=False)
            print(N, name, {k: round(row[k], 3) for k in ("ks_p_a", "ks_p_b", "tarp_ks_p", "tarp_dev",
                                                          "cov90_a", "cov90_b", "corr_Sc4")}, flush=True)
        we.mark(TAG, f"eval_n{N}_done")


def stage_decide(P):
    import ws_eval as we
    df = pd.read_csv(R / f"{TAG}_metrics.csv")
    df["member"] = df["member"].astype(str)
    ens = df[df.member == "ensemble"].set_index("budget").sort_index()
    single = df[df.member != "ensemble"]
    steps, prod = [], None
    B = list(ens.index)
    for N0, N1 in zip(B[:-1], B[1:]):
        e0, e1 = ens.loc[N0], ens.loc[N1]
        dcorr = {s: abs(e1[f"corr_{s}"] - e0[f"corr_{s}"]) for s in GATED}
        ok_a = all(v < 0.05 for v in dcorr.values())
        ok_b = max(abs(e1.cov90_a - e0.cov90_a), abs(e1.cov90_b - e0.cov90_b)) < 0.02
        ok_c = abs(e1.tarp_dev - e0.tarp_dev) < 0.01
        ok_d = (e1.ks_p_a_interior >= 0.01) and (e1.ks_p_b_interior >= 0.01) and (e1.tarp_ks_p >= 0.01)
        ok_a, ok_b, ok_c, ok_d = bool(ok_a), bool(ok_b), bool(ok_c), bool(ok_d)
        steps.append({"from": int(N0), "to": int(N1), "max_dcorr": float(max(dcorr.values())),
                      "dcorr": {k: float(v) for k, v in dcorr.items()},
                      "a_corr": ok_a, "b_cov": ok_b, "c_tarp": ok_c, "d_calib": ok_d,
                      "ks_a_full": float(e1.ks_p_a), "ks_b_full": float(e1.ks_p_b),
                      "ks_a_interior": float(e1.ks_p_a_interior), "ks_b_interior": float(e1.ks_p_b_interior),
                      "converged": bool(ok_a and ok_b and ok_c and ok_d)})
        if prod is None and steps[-1]["converged"]:
            prod = int(N1)
    flag = None
    if prod is None:
        prod = int(B[-1]) if B else None
        flag = ("NO BUDGET STEP CONVERGED up to the largest budget -- review with the user before production "
                "(options: larger budget, ensemble of more seeds, or report the residual miscalibration).")
    use_ens, members = False, []
    if prod is not None:
        sb = single[single.budget == prod]
        e_dev = ens.loc[prod, "tarp_dev"] if prod in ens.index else np.inf
        use_ens = bool(e_dev <= sb.tarp_dev.median())
        if use_ens:
            members = [str(_post_path(P, prod, int(s))) for s in sb.member]
        else:
            members = [str(_post_path(P, prod, int(sb.loc[sb.tarp_dev.idxmin(), "member"])))]
    res = {"protocol": P, "production_budget": prod, "use_ensemble": use_ens, "members": members,
           "flag": flag, "steps": steps,
           "rule": "see docstring of scripts/nb31_budget_convergence.py (fixed before results)"}
    res["sensor_model"] = SENSOR
    res["summary_set"] = SET
    we.save_json(R / f"{TAG}_production.json", res)
    for s in steps:
        print(s)
    print(json.dumps({k: v for k, v in res.items() if k != "steps"}, indent=2))


# ----------------------------------------------------------------------------------------
# AMENDED RULE (v2) -- written 2026-11-08 AFTER seeing the nb31 results; approved by the user.
# The v1 rule above selected 20k. Three design flaws were found in it:
#   1. It accepted the FIRST converged step without requiring later steps to stay converged
#      (20k->40k then failed the TARP criterion).
#   2. It gated on Sc7, which is misspecified by design (a +2 K T-sensor drift absent from the
#      trained model): the posterior there extrapolates and is not expected to converge
#      (its Corr(alpha,beta) jumped by 0.25 between 40k and 80k).
#   3. It had no precision criterion: at 20k the alpha posterior sd was ~3.5x its Cramer-Rao
#      bound and still halving with each doubling of the budget.
# v2 = v1 with Sc7 removed from the correlation gate, plus
#   (e) precision: max relative change of the ensemble posterior sd (alpha and beta, Sc1-Sc4)
#       between the two budgets < 10%, and
#   persistence: a budget counts only if EVERY following step is also converged.
# If no budget qualifies: production = the largest budget passing calibration (d).
# Noted, not changed: criterion (c) |delta TARP dev| < 0.01 is below the sampling noise of the
# TARP deviation itself (~1.36/sqrt(1000) = 0.043), so it carries little information.
# Both results (v1, v2) are written to 31_production.json together with the approval record.
GATED_V2 = ["Sc1", "Sc2", "Sc3", "Sc4"]

# ----------------------------------------------------------------------------------------
# AMENDED RULE v2.1 (2026-11-09, after results, user-approved) -- single seed vs ensemble.
# v2 (and v1) chose the deployed object as: ensemble if its TARP deviation <= the median
# single-member TARP deviation at the chosen budget, else the single member with the lowest TARP
# deviation. Under N3 + P2 + S1 (nb31, 40k) this deployed seed 3, whose interior beta marginal KS
# p is 0.0001: a posterior that FAILS the rule's own calibration criterion (d), which the rule
# only checks on the ENSEMBLE. The tie-break itself compared 0.0200 with 0.0200, far below the
# sampling noise of the TARP deviation (~1.36/sqrt(1000) = 0.043, noted above), so it selected on
# noise. The 40k ensemble passes (d) with the same posterior widths.
# v2.1 (a general principle, not a choice of seed): "the deployed posterior must itself pass the
# calibration test (d) at the chosen budget." Use the ensemble if it passes (d); otherwise the
# single member with the lowest TARP deviation AMONG MEMBERS THAT PASS (d); if no member and not
# the ensemble passes, use the ensemble and FLAG. Nothing else in the rule changes. Under N0 the
# ensemble was deployed (160k) and passes (d), so v2.1 gives the same N0 choice (checked from
# results/31_metrics.csv, not re-written). The v2 choice is kept in the record next to v2.1.
V2_1_RECORD = {"amendment": "v2.1", "date": "2026-11-09", "after_results": True,
               "approved_by": "user (Simone Casolo), 2026-11-09, in conversation with Claude",
               "reason": ("the deployed single member (40k, seed 3) failed the rule's own calibration criterion (d): "
                          "interior beta KS p = 0.0001; the single-vs-ensemble tie-break (ensemble 0.020 vs median member "
                          "0.020) is far below the TARP-deviation sampling noise (~0.043) and selected on noise; the ensemble "
                          "passes (d) at the same precision"),
               "principle": "the deployed posterior must itself pass the calibration test (d); use the ensemble whenever it does"}


def _passes_d(r) -> bool:
    return bool(r["ks_p_a_interior"] >= 0.01 and r["ks_p_b_interior"] >= 0.01 and r["tarp_ks_p"] >= 0.01)


def _steps_v2(ens: pd.DataFrame) -> list:
    B = list(ens.index)
    steps = []
    for N0, N1 in zip(B[:-1], B[1:]):
        e0, e1 = ens.loc[N0], ens.loc[N1]
        dcorr = {s: float(abs(e1[f"corr_{s}"] - e0[f"corr_{s}"])) for s in GATED_V2}
        dsd = {f"{p}_{s}": float(abs(e1[f"sd{p}_{s}"] / e0[f"sd{p}_{s}"] - 1)) for s in GATED_V2 for p in ("a", "b")}
        ok = {"a_corr": max(dcorr.values()) < 0.05,
              "b_cov": max(abs(e1.cov90_a - e0.cov90_a), abs(e1.cov90_b - e0.cov90_b)) < 0.02,
              "c_tarp": abs(e1.tarp_dev - e0.tarp_dev) < 0.01,
              "d_calib": (e1.ks_p_a_interior >= 0.01) and (e1.ks_p_b_interior >= 0.01) and (e1.tarp_ks_p >= 0.01),
              "e_precision": max(dsd.values()) < 0.10}
        ok = {k: bool(v) for k, v in ok.items()}
        steps.append({"from": int(N0), "to": int(N1), **ok, "converged": all(ok.values()),
                      "max_dcorr": max(dcorr.values()), "max_rel_dsd": max(dsd.values()),
                      "worst_sd": max(dsd, key=dsd.get)})
    return steps


def stage_decide_amended(P, *, approve: bool = False, approved_by: str = "user", reason: str = ""):
    """Apply rule v2 next to the original v1 result and, if ``approve``, write the production
    choice with an explicit approval record. Never overwrites the v1 result."""
    import ws_eval as we
    df = pd.read_csv(R / f"{TAG}_metrics.csv"); df["member"] = df["member"].astype(str)
    ens = df[df.member == "ensemble"].set_index("budget").sort_index()
    single = df[df.member != "ensemble"]
    v1 = json.loads((R / f"{TAG}_production.json").read_text())
    v1 = v1.get("rule_v1", v1)                      # idempotent: keep the original v1 record
    steps = _steps_v2(ens)
    B = list(ens.index)
    choice, how = None, ""
    for k, st in enumerate(steps):
        if all(x["converged"] for x in steps[k:]):
            choice, how = st["to"], f"first persistently converged step {st['from']}->{st['to']}"
            break
    if choice is None:
        calib = [N for N in B if ens.loc[N, "ks_p_a_interior"] >= 0.01 and ens.loc[N, "ks_p_b_interior"] >= 0.01
                 and ens.loc[N, "tarp_ks_p"] >= 0.01]
        choice = int(max(calib)) if calib else int(B[-1])
        how = ("no persistently converged step (precision still improving): largest budget passing "
               "calibration (d)") if calib else "no budget passes calibration: largest budget, FLAGGED"
    sb = single[single.budget == choice]
    # v2 (kept for the record): ensemble if its TARP deviation <= the median member's, else the member with the lowest TARP deviation
    use_ens_v2 = bool(ens.loc[choice, "tarp_dev"] <= sb.tarp_dev.median())
    members_v2 = ([str(_post_path(P, choice, int(s))) for s in sb.member] if use_ens_v2
                  else [str(_post_path(P, choice, int(sb.loc[sb.tarp_dev.idxmin(), "member"])))])
    # v2.1 (2026-11-09): the deployed posterior must itself pass calibration (d)
    member_flag = None
    ok_members = sb[(sb.ks_p_a_interior >= 0.01) & (sb.ks_p_b_interior >= 0.01) & (sb.tarp_ks_p >= 0.01)]
    if _passes_d(ens.loc[choice]):
        use_ens = True
        members = [str(_post_path(P, choice, int(s))) for s in sb.member]
    elif len(ok_members):
        use_ens = False
        members = [str(_post_path(P, choice, int(ok_members.loc[ok_members.tarp_dev.idxmin(), "member"])))]
    else:
        use_ens = True
        members = [str(_post_path(P, choice, int(s))) for s in sb.member]
        member_flag = f"FLAGGED: neither the ensemble nor any single member passes calibration (d) at {choice}"
    eff = {}
    e2f = R / f"{E2_TAG}_e2_fim.csv"
    if e2f.exists():
        e2 = pd.read_csv(e2f)
        e2 = e2[(e2.protocol == P) & (e2.representation == "summaries_fullcov")].set_index("point")
        for s in ("Sc1", "Sc3", "Sc4"):          # points whose (alpha, beta) are unchanged since nb29
            if s in e2.index:
                eff[s] = {"sd_a_over_crb": float(ens.loc[choice, f"sda_{s}"] / e2.loc[s, "crb_sd_a"]),
                          "sd_b_over_crb": float(ens.loc[choice, f"sdb_{s}"] / e2.loc[s, "crb_sd_b"])}
    res = {"protocol": P, "sensor_model": SENSOR, "sensor_noise": CFG.describe(), "summary_set": SET,
           "production_budget": int(choice), "use_ensemble": use_ens, "members": members,
           "flag": ("; ".join(x for x in (how if "FLAGGED" in how else None, member_flag) if x) or None),
           "selection": how, "rule_v1": v1, "rule_v1_choice": v1.get("production_budget"),
           "rule_v2_steps": steps, "efficiency_vs_summary_crb": eff,
           "rule": "v2.1 (v2 amended 2026-11-09 after results; see scripts/nb31_budget_convergence.py, 'AMENDED RULE v2.1')",
           "rule_v2_choice": {"production_budget": int(choice), "use_ensemble": use_ens_v2, "members": members_v2,
                              "description": ("single seed " + str(int(sb.loc[sb.tarp_dev.idxmin(), "member"]))) if not use_ens_v2 else "ensemble"},
           "rule_v2_1": {**V2_1_RECORD, "changes_choice": bool(use_ens != use_ens_v2 or members != members_v2),
                         "member_calibration_d": {str(int(m)): _passes_d(sb[sb.member == m].iloc[0]) for m in sb.member},
                         "ensemble_calibration_d": _passes_d(ens.loc[choice])}}
    for st in steps:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()})
    print("v1 choice:", v1.get("production_budget"), "| v2 choice:", choice, "|", how,
          "| ensemble" if use_ens else "| single seed")
    print("efficiency (posterior sd / CRB of the 29 summaries):", eff)
    if approve:
        res["approval"] = {"approved": True, "by": approved_by, "time": time.ctime(), "reason": reason}
        we.save_json(R / f"{TAG}_production.json", res)
        print(f"written results/{TAG}_production.json (production = rule v2, approval recorded)")
    else:
        print("not written: pass approve=True to record the user's approval")
    return res


def load_production_posterior():
    """For downstream notebooks:  from nb31_budget_convergence import load_production_posterior"""
    import ws_eval as we
    res = json.loads((R / f"{TAG}_production.json").read_text())
    res.setdefault("sensor_model", SENSOR)
    res.setdefault("summary_set", "S0_base29")        # files written before the production feature set was introduced
    posts = [pickle.load(open(m, "rb"))["posterior"] for m in res["members"]]
    return (we.EnsemblePosterior(posts) if res["use_ensemble"] else posts[0]), res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bank", "test", "train", "eval", "decide"])
    ap.add_argument("--protocol", default=None, help="override nb29's choice (not recommended)")
    ap.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    P = a.protocol or protocol()
    if a.stage == "bank":
        stage_bank(P)
    elif a.stage == "test":
        stage_test(P)
    elif a.stage == "train":
        stage_train(P, [int(b) for b in a.budgets.split(",")], [int(s) for s in a.seeds.split(",")], a.threads)
    elif a.stage == "eval":
        stage_eval(P)
    elif a.stage == "decide":
        stage_decide(P)
