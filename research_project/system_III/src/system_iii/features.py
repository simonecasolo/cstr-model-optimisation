"""Summary statistics and physics proxies for System III windows (numpy).

* ``base29``: the 29 System I summaries (per-channel mean/std/slope/min/max, final-window means, integrated
  |T - Tsp|, saturation fractions, and the two LEGACY proxies UA_eff_proxy and k0_eff_proxy), kept for
  comparability.
* ``plugins``: the physics proxies of System III:
    ln alpha_hat  = ln[ (Q/V)(Ci - C) / (k(T) C) ]                       (component balance)
    ln beta_hat_jac = ln[ rho_c Cp_c Qc (Tc - Tci_m) / (UA (T - Tc)) ]   (coolant/jacket balance)
    ln beta_hat_rx  = ln[ (rho Cp Q (Ti_m - T) - dH Q (Ci - C)) / (UA (T - Tc)) ]   (reactor balance)
  with the MEASURED inlet temperatures. The design study showed that the legacy UA_eff_proxy
  (T - Tc)/Qc is confounded with catalyst activity (it depends on the jacket duty); ln beta_hat_jac is not.
* Feature sets: "P2" (2 proxies), "S0" (29), "S1m" (29 + 3 plug-ins + 2 measured inlet temperatures,
  production default), "S1m_no_legacy" (S1m without the two legacy proxies).
"""
from __future__ import annotations

import numpy as np

from .physics import PLANT, R_GAS

BASE29 = ([f"{c}_{s}" for c in ("C", "T", "Tc", "Qc") for s in ("mean", "std", "slope", "min", "max")]
          + [f"{c}_final_mean" for c in ("C", "T", "Tc", "Qc")]
          + ["int_abs_T_err", "Qc_sat_low", "Qc_sat_high", "UA_eff_proxy", "k0_eff_proxy"])
PLUGINS = ["ln_alpha_hat", "ln_beta_hat_jac", "ln_beta_hat_rx"]
INLET = ["Ti_meas_dev", "Tci_meas_dev"]
NAMES = {"S0": BASE29, "P2": ["ln_alpha_hat", "ln_beta_hat_jac"],
         "S1m": BASE29 + PLUGINS + INLET,
         "S1m_no_legacy": [n for n in BASE29 if n not in ("UA_eff_proxy", "k0_eff_proxy")] + PLUGINS + INLET,
         # masking study (Claim C-M): what a temperature-only monitor sees, and without the coolant side
         "T_only": [f"T_{s}" for s in ("mean", "std", "slope", "min", "max")] + ["T_final_mean", "int_abs_T_err"],
         "C_T_only": [f"{c}_{s}" for c in ("C", "T") for s in ("mean", "std", "slope", "min", "max")]
                     + ["C_final_mean", "T_final_mean", "int_abs_T_err", "k0_eff_proxy", "ln_alpha_hat"]}


def base29(obs, t, plant=PLANT):
    obs = np.asarray(obs, float)
    n, nt, _ = obs.shape
    means = obs.mean(1); stds = obs.std(1); mins = obs.min(1); maxs = obs.max(1)
    tc = t - t.mean()
    slopes = (tc[None, :, None] * (obs - means[:, None, :])).sum(1) / (tc ** 2).sum()
    per = np.stack([means, stds, slopes, mins, maxs], axis=2).reshape(n, -1)
    nf = int(np.ceil(0.25 * nt))
    fin = obs[:, -nf:, :].mean(1)
    ierr = np.abs(obs[:, :, 1] - plant.Tsp).sum(1) * (t[1] - t[0])
    qc = obs[:, :, 3]
    sl = (qc <= plant.Qcmin + 1.0).mean(1); sh = (qc >= plant.Qcmax - 1.0).mean(1)
    ua_proxy = (means[:, 1] - means[:, 2]) / np.maximum(np.abs(means[:, 3]), 1e-6)
    k_proxy = np.clip(np.log(np.maximum(means[:, 0], 1e-6) / np.maximum(plant.Ci - means[:, 0], 1e-6)), -8, 2)
    return np.concatenate([per, fin, np.stack([ierr, sl, sh, ua_proxy, k_proxy], 1)], axis=1)


def plugins(obs, Ti_m, Tci_m, plant=PLANT):
    m = np.asarray(obs, float).mean(1)
    Cm, Tm, Tcm, Qcm = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    k1 = plant.k0 * np.exp(-plant.Ea / (R_GAS * Tm))
    a = (plant.Q / plant.V) * (plant.Ci - Cm) / (k1 * np.maximum(Cm, 1e-9))
    dT = np.maximum(Tm - Tcm, 1e-3)
    bj = plant.rhocCpc * Qcm * (Tcm - Tci_m) / (plant.UA * dT)
    br = (plant.rhoCp * plant.Q * (Ti_m - Tm) - plant.dH * plant.Q * (plant.Ci - Cm)) / (plant.UA * dT)
    lg = lambda v: np.log(np.clip(v, 1e-3, 1e3))
    return np.stack([lg(a), lg(bj), lg(br)], axis=1)


def features(obs, t, Ti_m, Tci_m, set_name="S1m", plant=PLANT):
    b = base29(obs, t, plant)
    full = np.concatenate([b, plugins(obs, Ti_m, Tci_m, plant),
                           np.stack([Ti_m - plant.Ti, Tci_m - plant.Tci], 1)], axis=1)
    allnames = BASE29 + PLUGINS + INLET
    idx = [allnames.index(k) for k in NAMES[set_name]]
    return full[:, idx].astype(np.float32)


def nominal_inlet_features(obs, t, set_name="S1m", plant=PLANT):
    """The NAIVE model's features: the inlet is assumed nominal (not measured)."""
    n = len(obs)
    return features(obs, t, np.full(n, plant.Ti), np.full(n, plant.Tci), set_name, plant)
