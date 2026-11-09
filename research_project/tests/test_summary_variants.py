"""Plug-in estimators must recover (alpha, beta) exactly from a steady state of the System I
balances (constants as in cstr_sbi.physics), and the dynamic features must behave."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import summary_variants as sv

CONST = dict(Q=40.0, V=500.0, CI=0.97, TI=297.0, TCI=297.0, K0=16.96e12, EA=75362.0, R=8.314,
             RHO=1000.0, CP=1.0, RHOC=1000.0, CPC=1.0, UA=1.25e4, HR=-20220.0)
TSP = 312.5

def steady_state(alpha, beta, c=CONST, T=TSP):
    k = c["K0"] * np.exp(-c["EA"] / (c["R"] * T))
    C = (c["Q"] / c["V"]) * c["CI"] / (c["Q"] / c["V"] + alpha * k)
    q_rem = -c["HR"] * c["Q"] * (c["CI"] - C) + c["RHO"] * c["CP"] * c["Q"] * (c["TI"] - T)
    Tc = T - q_rem / (beta * c["UA"])
    Qc = beta * c["UA"] * (T - Tc) / (c["RHOC"] * c["CPC"] * (Tc - c["TCI"]))
    return C, T, Tc, Qc

def test_matches_nb29_e1_coolant_flows():
    assert abs(steady_state(1.0, 1.0)[3] - 43.07) < 0.1      # nb29 E1 Sc1 Qc_ss
    assert abs(steady_state(0.85, 0.85)[3] - 92.27) < 0.1    # nb29 E1 Sc4 Qc_ss

def test_estimators_recover_theta_at_steady_state():
    t = np.arange(1, 121) * 0.5
    rng = np.random.default_rng(0)
    for a, b in [(1.0, 1.0), (0.7, 1.0), (0.85, 0.85), (1.1, 0.8)]:
        ss = np.array(steady_state(a, b))
        obs = ss[None, None, :] * (1 + 1e-6 * rng.standard_normal((1, 120, 4)))
        f = dict(zip(sv.EXTRA, sv.extra_features(obs, t, CONST)[0]))
        assert abs(np.exp(f["ln_alpha_hat"]) - a) < 1e-3
        assert abs(np.exp(f["ln_beta_hat_jac"]) - b) < 1e-3
        assert abs(np.exp(f["ln_beta_hat_rx"]) - b) < 1e-3

def test_dynamic_features():
    t = np.arange(1, 121) * 0.5
    rng = np.random.default_rng(1)
    e = rng.standard_normal((200, 121))
    ar = np.zeros((200, 120)); ar[:, 0] = e[:, 0]
    for i in range(1, 120):
        ar[:, i] = 0.8 * ar[:, i - 1] + e[:, i]
    obs = np.stack([0.02 + 1e-4 * rng.standard_normal((200, 120)), 312.5 + 0.1 * ar,
                    300 + 0.1 * rng.standard_normal((200, 120)), 50 + 5 * ar], axis=2)
    f = dict(zip(sv.EXTRA, sv.extra_features(obs, t, CONST).T))
    assert abs(f["ac1_T"].mean() - 0.8) < 0.08          # AR(1) coefficient recovered
    assert f["xc0_T_Qc"].mean() > 0.99                  # Qc = linear in T here
    assert abs(f["xc0_T_Tc"].mean()) < 0.05
    sat = obs.copy(); sat[:, :, 3] = 400.0              # saturated valve: constant Qc
    g = dict(zip(sv.EXTRA, sv.extra_features(sat, t, CONST).T))
    assert np.all(g["ac1_Qc"] == 0) and np.all(np.isfinite(g["xc0_T_Qc"]))

def test_sets_index_union():
    for name, idx in sv.SETS.items():
        assert len(idx) == len(set(idx)) and max(idx) < len(sv.UNION)
    assert len(sv.SETS["S0_base29"]) == 29
