"""Augment existing S-C banks/SBC sets to S-C.2 (adds the 2 xi_reb-sharpening
features, `summaries.py::xi_reb_sharpened_features`) WITHOUT re-simulation.

The two new features are closed-form functions of channel WINDOW MEANS
(Q_reb_mean, T_reb_mean, F_R_norm_mean, F_B_norm_mean, V_norm_mean) that are
already stored in every existing S-C summary vector -- just compressed via
`_compress_heavy_tails` (`sign(x)*log1p(|x|)`). Inverting with `expm1` (exact
inverse for positive-valued means, no precision loss beyond float32) recovers
the identical raw-scale means `xi_reb_sharpened_features` would have computed
from the raw trajectory directly -- so this reconstruction is EXACT, not an
approximation, and needs zero new ODE solves. Matches this project's existing
"paired S-B bank, free, no re-simulation" pattern (nb24a Section 2).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cstr_sbi.recycle.summaries import summary_names, N_SUMMARIES_SC
from cstr_sbi.recycle.physics import F_R_NOM, F_B_NOM, F_TOT_NOM, QREB_NOM, T_REB_NOM

DATA = ROOT / "data"
RESULTS = ROOT / "results"

SC_NAMES = summary_names("S-C")
IDX = {name: SC_NAMES.index(name) for name in
       ("T_reb_mean", "Q_reb_mean", "F_R_norm_mean", "F_B_norm_mean", "V_norm_mean")}


def _inv(x):
    """Exact inverse of `_compress_heavy_tails` (sign(x)*log1p(|x|))."""
    return np.sign(x) * np.expm1(np.abs(x))


def _compress(x):
    return np.sign(x) * np.log1p(np.abs(x))


def augment_sc_to_sc2(summaries_sc: np.ndarray) -> np.ndarray:
    """(n, N_SUMMARIES_SC) S-C summaries -> (n, N_SUMMARIES_SC+2) S-C.2 summaries."""
    assert summaries_sc.shape[1] == N_SUMMARIES_SC, summaries_sc.shape

    T_reb_mean = _inv(summaries_sc[:, IDX["T_reb_mean"]])
    Q_reb_mean = _inv(summaries_sc[:, IDX["Q_reb_mean"]])
    F_R_norm_mean = _inv(summaries_sc[:, IDX["F_R_norm_mean"]])
    F_B_norm_mean = _inv(summaries_sc[:, IDX["F_B_norm_mean"]])
    V_norm_mean = _inv(summaries_sc[:, IDX["V_norm_mean"]])

    throughput_rat = (F_R_norm_mean * F_R_NOM + F_B_norm_mean * F_B_NOM) / F_TOT_NOM
    denom = QREB_NOM * throughput_rat * np.maximum(V_norm_mean, 1e-6)
    xi_reb_qreb_feat = np.log(np.maximum(Q_reb_mean, 1e-6) / np.maximum(denom, 1e-6))
    xi_reb_treb_feat = T_reb_mean - T_REB_NOM - 3.0 * (F_R_norm_mean - 1.0) - 12.0 * (V_norm_mean - 1.0)

    new_cols = np.stack([_compress(xi_reb_qreb_feat), _compress(xi_reb_treb_feat)], axis=1).astype(np.float32)
    return np.concatenate([summaries_sc, new_cols], axis=1)


def augment_file(in_path: pathlib.Path, out_path: pathlib.Path):
    if out_path.exists():
        print(f"already exists, skipping: {out_path}")
        return
    d = np.load(in_path)
    summaries_sc2 = augment_sc_to_sc2(np.asarray(d["summaries"], dtype=np.float32))
    kwargs = {k: d[k] for k in d.files if k not in ("summaries", "note")}
    kwargs["note"] = "augmented from S-C to S-C.2 (xi_reb-sharpening features), no re-simulation"
    np.savez(out_path, summaries=summaries_sc2, **kwargs)
    print(f"saved {out_path}: {summaries_sc2.shape}")


def main():
    files = [
        (DATA / "wu2003_sbi_train_sc_lognormal_true_n1000_stage1bfix.npz",
         DATA / "wu2003_sbi_train_sc2_lognormal_true_n1000_stage1bfix.npz"),
        (RESULTS / "24a_sysII_sc_lognormal_sbc_set_n120_stage1bfix.npz",
         RESULTS / "24a_sysII_sc2_lognormal_sbc_set_n120_stage1bfix.npz"),
        (RESULTS / "24a_sysII_sc_lognormal_sbc_set_n400_stage1bfix.npz",
         RESULTS / "24a_sysII_sc2_lognormal_sbc_set_n400_stage1bfix.npz"),
    ]
    for in_path, out_path in files:
        augment_file(in_path, out_path)


if __name__ == "__main__":
    main()
