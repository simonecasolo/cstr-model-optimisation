"""Candidate summary-statistic extensions for System I (notebook 03, section 10).

Pure numpy, vectorised over a batch of raw observation windows ``obs`` of shape
(B, n_t, 4) with channels [C, T, Tc, Qc]. The existing 29-D summaries come from
``cstr_sbi.summaries`` (unchanged); this module adds EXTRA features and defines the
candidate SETS as index lists into the union [base29 | extras].

Extra features
--------------
Plug-in steady-state estimators (window means; nearly linear in log theta, which is close to
the Fearnhead & Prangle (2012) optimal summary, E[theta | x]):
  ln_alpha_hat      component balance  (Q/V)(Ci - C) = alpha k(T) C
                    -> alpha = (Q/V)(Ci - C) / (k0 exp(-Ea/RT) C)
  ln_beta_hat_jac   jacket balance     beta UA (T - Tc) = rho_c Cp_c Qc (Tc - Tci)
  ln_beta_hat_rx    reactor balance    beta UA (T - Tc) = rho Cp Q (Ti - T) + (-dH) Q (Ci - C)
                    (independent of the coolant-flow measurement)
Loop-dynamics features (detrended signals; beta changes the plant gain, so the
noise-driven closed-loop dynamics carry information the window means do not):
  ac1_T, ac1_Tc, ac1_Qc   lag-1 autocorrelation
  xc0_T_Qc, xc1_T_Qc      T-Qc cross-correlation at lag 0 and with T leading by 1 sample
  xc0_T_Tc                T-Tc cross-correlation at lag 0
Detrended standard deviations (noise amplitude without the window trend):
  dstd_C, dstd_T, dstd_Tc, dstd_Qc
"""
from __future__ import annotations

import numpy as np

BASE29 = ("C_mean", "C_std", "C_slope", "C_min", "C_max", "T_mean", "T_std", "T_slope", "T_min", "T_max",
          "Tc_mean", "Tc_std", "Tc_slope", "Tc_min", "Tc_max", "Qc_mean", "Qc_std", "Qc_slope", "Qc_min",
          "Qc_max", "C_final_mean", "T_final_mean", "Tc_final_mean", "Qc_final_mean", "int_abs_T_err",
          "Qc_sat_low_frac", "Qc_sat_high_frac", "UA_eff_proxy", "k0_eff_proxy")
EST = ("ln_alpha_hat", "ln_beta_hat_jac", "ln_beta_hat_rx")
DYN = ("ac1_T", "ac1_Tc", "ac1_Qc", "xc0_T_Qc", "xc1_T_Qc", "xc0_T_Tc")
DSTD = ("dstd_C", "dstd_T", "dstd_Tc", "dstd_Qc")
EXTRA = EST + DYN + DSTD
UNION = BASE29 + EXTRA


def _ix(names):
    pos = {n: i for i, n in enumerate(UNION)}
    return [pos[n] for n in names]


SETS = {
    # current production summaries (the reference; replaced only if a candidate wins clearly)
    "S0_base29": _ix(BASE29),
    # base + the three plug-in estimators
    "S1_base_est": _ix(BASE29 + EST),
    # base + estimators + loop dynamics + detrended stds
    "S2_base_est_dyn": _ix(BASE29 + EST + DYN + DSTD),
    # compact: estimators, means, detrended stds, saturation, dynamics (no min/max/slope)
    "S3_compact": _ix(EST + ("C_mean", "T_mean", "Tc_mean", "Qc_mean") + DSTD
                      + ("Qc_sat_low_frac", "Qc_sat_high_frac") + DYN),
    # minimal: estimators + saturation fractions only
    "S4_est_only": _ix(EST + ("Qc_sat_low_frac", "Qc_sat_high_frac")),
}


def _detrend(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove the least-squares line from each (B, n_t) row."""
    tc = t - t.mean()
    slope = (x - x.mean(1, keepdims=True)) @ tc / (tc @ tc)
    return x - x.mean(1, keepdims=True) - slope[:, None] * tc[None, :]


def _corr(a: np.ndarray, b: np.ndarray, lag: int = 0) -> np.ndarray:
    """Row-wise Pearson correlation of a[t] with b[t+lag]; 0 where either side is constant."""
    if lag > 0:
        a, b = a[:, :-lag], b[:, lag:]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    out = np.zeros(len(a))
    ok = den > 1e-12 * (np.abs(num) + 1.0)
    out[ok] = num[ok] / den[ok]
    return out


def extra_features(obs: np.ndarray, t: np.ndarray, const: dict) -> np.ndarray:
    """(B, n_t, 4) raw windows -> (B, len(EXTRA)) extra features.

    ``const`` keys: Q, V, CI, TI, TCI, K0, EA, R, RHO, CP, RHOC, CPC, UA, HR (H_R < 0 for an
    exothermic reaction, same sign convention as cstr_sbi.physics)."""
    obs = np.asarray(obs, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    C, T, Tc, Qc = (obs[:, :, k] for k in range(4))
    Cm, Tm, Tcm, Qcm = C.mean(1), T.mean(1), Tc.mean(1), Qc.mean(1)
    eps = 1e-9
    k_T = const["K0"] * np.exp(-const["EA"] / (const["R"] * Tm))
    alpha = (const["Q"] / const["V"]) * (const["CI"] - Cm) / (k_T * np.maximum(Cm, eps))
    dT = np.maximum(Tm - Tcm, 1e-3)
    beta_j = const["RHOC"] * const["CPC"] * Qcm * (Tcm - const["TCI"]) / (const["UA"] * dT)
    q_gen = -const["HR"] * const["Q"] * (const["CI"] - Cm)          # cal/min released
    q_flow = const["RHO"] * const["CP"] * const["Q"] * (const["TI"] - Tm)  # cal/min, negative
    beta_r = (q_flow + q_gen) / (const["UA"] * dT)
    lg = lambda v: np.log(np.clip(v, 1e-3, 1e3))
    d = {k: _detrend(v, t) for k, v in (("C", C), ("T", T), ("Tc", Tc), ("Qc", Qc))}
    feats = [lg(alpha), lg(beta_j), lg(beta_r),
             _corr(d["T"], d["T"], 1), _corr(d["Tc"], d["Tc"], 1), _corr(d["Qc"], d["Qc"], 1),
             _corr(d["T"], d["Qc"], 0), _corr(d["T"], d["Qc"], 1), _corr(d["T"], d["Tc"], 0),
             d["C"].std(1), d["T"].std(1), d["Tc"].std(1), d["Qc"].std(1)]
    return np.stack(feats, axis=1)


def physics_constants() -> dict:
    """Constants from cstr_sbi.physics (imported lazily so this module stays numpy-only)."""
    import cstr_sbi.physics as ph
    return dict(Q=ph.Q, V=ph.V, CI=ph.CI_NOMINAL, TI=ph.TI_NOMINAL, TCI=ph.TCI_NOMINAL, K0=ph.K0_NOMINAL,
                EA=ph.E_A, R=ph.R_GAS, RHO=ph.RHO, CP=ph.C_P, RHOC=ph.RHO_C, CPC=ph.C_PC,
                UA=ph.UA_NOMINAL, HR=ph.H_R)
