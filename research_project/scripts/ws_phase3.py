"""Phase 3 shared helpers: the measured-inlet dataset D1 and the production / naive posteriors.

(Plan: docs/Manuscript/system_i_rerun_plan.md, section "Phase 3 data under the production model".)

D0 = data/observations.npz          nominal-inlet dataset (naive model, Claim 1 in notebook 07; unchanged)
D1 = data/observations_measured_inlet.npz
        closed-loop scenarios Sc1-Sc5, Sc7 x 50 windows (user decision 2026-11-09: Sc0/Sc6 are NOT in D1);
        per window the TRUE inlet [Ci, Ti, Tci] drawn from the nb35/nb36 training distribution
        (Ti, Tci ~ U(295, 299) K, Ci = Ci_nom (1 + N(0, 0.01^2))) and the MEASURED (Ti, Tci) =
        true + eps N(0,1) calibration offset (eps = 0.1 K, fixed per window) + window-mean white noise
        N(0, 0.2^2/120) K, exactly nb35's sensor model (nb35_measured_inlet.measured).
PM = production posterior (nb36: 20k x 5-seed ensemble, S1m features: summaries + plug-in estimators with the
     measured inlet + the measured Ti, Tci).   PN = naive posterior (nb31: S1 features, nominal inlet assumed).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

D0_PATH = ROOT / "data" / "observations.npz"
D1_PATH = ROOT / "data" / "observations_measured_inlet.npz"
SEED_BASE = 3_700_000            # new seed base: D1 never reuses nb02's or nb35's windows
N_REP = 50
EPS_K = 0.1


def _mods():
    os.environ.setdefault("CSTR_SENSOR_MODEL", "N3")
    import system_I_config as CFG
    import ws_eval as we
    import ws_inlet as wi
    import nb35_measured_inlet as M
    return CFG, we, wi, M


# ------------------------------------------------------------------ generation
def generate_d1(seed_base: int = SEED_BASE, n_rep: int = N_REP, eps: float = EPS_K) -> dict:
    """Simulate D1 (arrays only; saving and the checks are in notebook 02b)."""
    CFG, we, wi, M = _mods()
    from cstr_sbi.physics import K0_NOMINAL, UA_NOMINAL
    from cstr_sbi.scenarios import SCENARIO_CONFIGS
    th2, names, drift = we.scenario_thetas(n_rep)                       # Sc1-Sc5, Sc7 (EVAL_SCENARIOS order)
    n = len(th2)
    inlet_true = M._inlets(n, seed_base + 1)                            # (n, 3) [Ci, Ti, Tci]
    raw = wi.simulate(th2, inlet_true, M.PROTOCOL, seed_base + 2, raw=True, **M.NOISE)
    raw = np.array(raw, np.float32)
    raw[:, :, 1] += drift[:, None]                                      # Sc7: +2 K T-sensor drift after the sensor layer
    inlet_meas = M.measured(inlet_true[:, 1:3], eps, seed_base + 3).astype(np.float32)   # (n, 2) [Ti, Tci]
    theta4 = np.column_stack([np.full(n, UA_NOMINAL), np.full(n, K0_NOMINAL), th2]).astype(np.float32)
    sid = np.array([SCENARIO_CONFIGS[s].id for s in names], np.int32)
    t = np.arange(1, raw.shape[1] + 1) * 0.5
    return dict(
        theta=theta4, x=raw, scenario_id=sid, mode=np.array(["closed_loop"] * n), t=t,
        protocol=np.array(M.PROTOCOL), burn_min=np.array(float(CFG.HISTORY_MIN[M.PROTOCOL])),
        controller_law=np.array("standard PI: Qc = Qc0 + Kp*[(T-Tsp) + I/tau_i] (fixed 2026-11-08)"),
        sensor_model=np.array(CFG.model()), sensor_noise=np.array(CFG.SENSOR_MODELS[CFG.model()], float),
        inlet_true=inlet_true.astype(np.float32), inlet_measured=inlet_meas,
        inlet_model=np.array(json.dumps({"eps_K": eps, "white_sd_K": float(M.WHITE_SD), "inlet_range_K": M.INLET_RANGE,
                                         "Ti_Tci_nominal_K": M.T_NOM, "Ci_rel_sd": M.CI_REL_SD,
                                         "channels": ["Ti", "Tci"], "inlet_true_columns": ["Ci", "Ti", "Tci"]})),
        features=np.array("S1m_measured_inlet"), seeds=np.array([seed_base + 1, seed_base + 2, seed_base + 3]))


def load_d0() -> dict:
    with np.load(D0_PATH, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def load_d1() -> dict:
    with np.load(D1_PATH, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# ------------------------------------------------------------------ features and posteriors
def production_features(d: dict) -> np.ndarray:
    """S1m features of D1 windows from the STORED measured inlet (not a re-drawn measurement)."""
    import nb36_production_measured_inlet as P
    return P.features_from_raw(d["x"], d["inlet_measured"])


def naive_features(d: dict) -> np.ndarray:
    """S1 features (nominal inlet assumed) of any dataset's windows."""
    _, we, _, _ = _mods()
    t = np.asarray(d["t"], np.float32)
    return we.features(d["x"], t, "S1_base_est")


def load_production_posterior():
    """PM: nb36 production posterior (ensemble) and its record."""
    import nb36_production_measured_inlet as P
    return P.load_production_posterior()


def load_naive_posterior():
    """PN: nb31 nominal-inlet posterior (ensemble) and its record."""
    import nb31_budget_convergence as B
    return B.load_production_posterior()


# ------------------------------------------------------------------ 30-day stream (notebook 10)
STREAM_PATH = ROOT / "data" / "stream_30day_measured_inlet.npz"
STREAM_SEED_BASE = 3_900_000
N_STREAM = 720
TCRIT_H = 720.0
BETA_TAU_H = 200.0


def stream_truth(t_h, alpha_end: float = 0.85, beta_end: float = 0.85):
    """alpha, beta at window start t_h (hours): linear alpha, Kern-Seaton beta (scenarios.generate_degradation_stream)."""
    t_h = np.asarray(t_h, float)
    a = (1.0 - alpha_end) * t_h / TCRIT_H
    b = (1.0 - beta_end) / (1.0 - np.exp(-TCRIT_H / BETA_TAU_H))
    return 1.0 - a, 1.0 - b * (1.0 - np.exp(-t_h / BETA_TAU_H))


def stream_inlet(t_h):
    """True (Ti, Tci) of window start t_h: diurnal feed temperature, weekly + slow ramp coolant-supply drift (inside 295-299 K)."""
    t_h = np.asarray(t_h, float)
    ti = 297.0 + 1.0 * np.sin(2 * np.pi * t_h / 24.0)
    tci = 297.0 + 1.2 * np.sin(2 * np.pi * t_h / 168.0 + 0.7) + 0.5 * t_h / TCRIT_H
    return ti, tci


def generate_stream(seed_base: int = STREAM_SEED_BASE, n: int = N_STREAM, eps: float = EPS_K) -> dict:
    """The 30-day stream under the production model: per-window P2 history, N3 sensors, drifting inlet,
    measured inlet = true + PERSISTENT offset (drawn once for the month) + window noise."""
    CFG, we, wi, M = _mods()
    t_h = np.arange(n, dtype=float)
    al, be = stream_truth(t_h)
    ti, tci = stream_inlet(t_h)
    rng = np.random.default_rng(seed_base + 1)
    ci = (float(wi.NOMINAL[0]) * (1 + rng.normal(0, M.CI_REL_SD, n))).astype(np.float32)
    inlet_true = np.stack([ci, ti, tci], axis=1).astype(np.float32)
    theta = np.stack([al, be], axis=1).astype(np.float32)
    raw = np.array(wi.simulate(theta, inlet_true, M.PROTOCOL, seed_base + 2, raw=True, **M.NOISE), np.float32)
    rng2 = np.random.default_rng(seed_base + 3)
    offsets = eps * rng2.standard_normal(2)                              # persistent: one draw per channel for the month
    white = M.WHITE_SD * rng2.standard_normal((n, 2))
    inlet_meas = (inlet_true[:, 1:3].astype(np.float64) + offsets[None, :] + white).astype(np.float32)
    return dict(theta=theta, t_start_min=(t_h * 60.0).astype(np.float32), x=raw, t=np.arange(1, raw.shape[1] + 1) * 0.5,
                inlet_true=inlet_true, inlet_measured=inlet_meas, offsets_ti_tci=offsets.astype(np.float32),
                protocol=np.array(M.PROTOCOL), sensor_model=np.array(CFG.model()), features=np.array("S1m_measured_inlet"),
                seeds=np.array([seed_base + 1, seed_base + 2, seed_base + 3]),
                spec=np.array(json.dumps({"alpha_end": 0.85, "beta_end": 0.85, "beta_tau_h": BETA_TAU_H, "eps_K": eps,
                                          "Ti": "297+1.0 sin(2 pi t/24)", "Tci": "297+1.2 sin(2 pi t/168+0.7)+0.5 t/720"})))


def load_stream() -> dict:
    with np.load(STREAM_PATH, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
