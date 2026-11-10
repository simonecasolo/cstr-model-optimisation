"""Decision rules carried over from System I, implemented once (fixed before results).

* ``budget_rule_v21``: nb31 rule v2.1 (persistent convergence over budget steps with criteria a-e, the
  deployed posterior must itself pass calibration (d); ensemble whenever it does).
* ``protocol_rule``: nb29 selection on the realistic test column (default P2): eligible if seed-median
  min marginal SBC KS p >= 0.01 and TARP KS p >= 0.01; best RMSE; ties within 1 SE -> robustness tie-break
  (worst off-column TARP deviation within 0.01) -> simplicity order.
* ``summary_rule``: nb03 rule: a candidate replaces the default only if calibrated, RMSE not worse by more
  than 1 SE, and either efficiency <= 0.9 x default (difference > 2 combined SE) or a pure calibration gain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GATED = ["Sc1", "Sc2", "Sc3", "Sc4"]


def passes_d(r) -> bool:
    return bool(r["ks_p_a_interior"] >= 0.01 and r["ks_p_b_interior"] >= 0.01 and r["tarp_ks_p"] >= 0.01)


def _steps(ens: pd.DataFrame):
    B = list(ens.index); steps = []
    for N0, N1 in zip(B[:-1], B[1:]):
        e0, e1 = ens.loc[N0], ens.loc[N1]
        dcorr = max(abs(e1[f"corr_{s}"] - e0[f"corr_{s}"]) for s in GATED)
        dsd = max(abs(e1[f"sd{p}_{s}"] / e0[f"sd{p}_{s}"] - 1) for s in GATED for p in ("a", "b"))
        ok = {"a_corr": dcorr < 0.05,
              "b_cov": max(abs(e1.cov90_a - e0.cov90_a), abs(e1.cov90_b - e0.cov90_b)) < 0.02,
              "c_tarp": abs(e1.tarp_dev - e0.tarp_dev) < 0.01,
              "d_calib": passes_d(e1), "e_precision": dsd < 0.10}
        steps.append({"from": int(N0), "to": int(N1), **{k: bool(v) for k, v in ok.items()},
                      "converged": all(ok.values()), "max_dcorr": float(dcorr), "max_rel_dsd": float(dsd)})
    return steps


def budget_rule_v21(metrics: pd.DataFrame):
    """metrics: rows (budget, member, ...) with member 'ensemble' and seed ids. Returns decision dict."""
    df = metrics.copy(); df["member"] = df["member"].astype(str)
    ens = df[df.member == "ensemble"].set_index("budget").sort_index()
    single = df[df.member != "ensemble"]
    steps = _steps(ens); choice, how, flag = None, "", None
    for k, st in enumerate(steps):
        if all(x["converged"] for x in steps[k:]):
            choice, how = st["to"], f"first persistently converged step {st['from']}->{st['to']}"
            break
    if choice is None:
        calib = [N for N in ens.index if passes_d(ens.loc[N])]
        choice = int(max(calib)) if calib else int(ens.index[-1])
        how = "largest budget passing calibration (d)" if calib else "no budget passes calibration"
        flag = None if calib else "FLAGGED: no budget passes calibration"
    sb = single[single.budget == choice]
    if passes_d(ens.loc[choice]):
        members, use_ens = [int(m) for m in sb.member], True
    else:
        ok = sb[[passes_d(r) for _, r in sb.iterrows()]]
        if len(ok):
            members, use_ens = [int(ok.loc[ok.tarp_dev.idxmin(), "member"])], False
        else:
            members, use_ens = [int(m) for m in sb.member], True
            flag = (flag or "") + " FLAGGED: neither the ensemble nor any member passes (d)"
    return {"production_budget": int(choice), "use_ensemble": use_ens, "members": members, "selection": how,
            "flag": flag, "steps": steps, "rule": "System I nb31 v2.1"}


SIMPLICITY = ["P1b", "P2", "P1", "P0"]


def protocol_rule(matrix: pd.DataFrame, test_col="P2"):
    """matrix: rows (train, test, seed, ks_p_a, ks_p_b, tarp_ks_p, tarp_dev, rmse_norm)."""
    df = matrix.copy(); df["min_ks"] = df[["ks_p_a", "ks_p_b"]].min(axis=1)
    col = df[df.test == test_col].groupby("train").agg(min_ks_med=("min_ks", "median"), tarp_ks_med=("tarp_ks_p", "median"),
                                                       rmse=("rmse_norm", "mean"),
                                                       rmse_se=("rmse_norm", lambda v: v.std(ddof=1) / np.sqrt(len(v))))
    col["worst_off_tarp"] = df[df.test != test_col].groupby(["train", "test"]).tarp_dev.mean().groupby("train").max()
    col["eligible"] = (col.min_ks_med >= 0.01) & (col.tarp_ks_med >= 0.01)
    if not col.eligible.any():
        return {"choice": col.tarp_ks_med.idxmax(), "flag": "no protocol passes calibration on the test column",
                "table": col.reset_index().to_dict("records")}
    el = col[col.eligible]; best = el.rmse.idxmin()
    tied = el[el.rmse <= el.loc[best, "rmse"] + np.sqrt(el.loc[best, "rmse_se"] ** 2 + el.rmse_se ** 2)]
    tied = tied[tied.worst_off_tarp <= tied.worst_off_tarp.min() + 0.01]
    choice = sorted(tied.index, key=lambda p: SIMPLICITY.index(p) if p in SIMPLICITY else 99)[0]
    return {"choice": choice, "flag": None, "table": col.reset_index().to_dict("records")}


def summary_rule(agg: pd.DataFrame, default="S1m"):
    """agg: index = set; columns calib_ok, rmse, rmse_se, eff, eff_se, n_features."""
    d = agg.loc[default]; win = []
    for name, r in agg.drop(index=default).iterrows():
        if not r.calib_ok or r.rmse > d.rmse + r.rmse_se:
            continue
        a = (r.eff <= 0.9 * d.eff) and (d.eff - r.eff > 2 * np.hypot(r.eff_se, d.eff_se))
        b = (not d.calib_ok) and (r.eff <= d.eff)
        if a or b:
            win.append(name)
    if not win:
        return {"choice": default, "winners": []}
    best = agg.loc[win].eff.idxmin()
    near = [w for w in win if agg.loc[w].eff <= agg.loc[best].eff + agg.loc[best].eff_se]
    return {"choice": min(near, key=lambda w: agg.loc[w].n_features), "winners": win}
