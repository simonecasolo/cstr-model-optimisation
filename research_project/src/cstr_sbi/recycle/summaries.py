"""Summary statistics for the Wu 2003 recycle plant SBI study.

Summary vector layout
---------------------
S-B observations (9 channels):
    T_r, T_j, Q_j, T_reb, Q_reb, F_R_norm, F_B_norm, R_norm, V_norm

S-A observations (10 channels):
    T_r, T_j, Q_j, x_D, T_reb, Q_reb, F_R_norm, F_B_norm, R_norm, V_norm

Per-channel statistics (6 per channel):
    mean, std, slope (linear trend / (t_max-t_min)), min, max, final-quarter mean

Physics-informed features (12):
    See PHYSICS_FEATURE_NAMES below.

Total: S-B → 9×6 + 12 = 66-D; S-A → 10×6 + 12 = 72-D
"""

from __future__ import annotations

import numpy as np

from cstr_sbi.recycle.physics import (
    F0_NOM, F_R_NOM, F_B_NOM, QJ_NOM, QREB_NOM, T_SP, UA_NOM,
    X_D_NOM, X_B_NOM, REFLUX_RATIO, T_REB_NOM,
)


# ---------------------------------------------------------------------------
# Channel definitions
# ---------------------------------------------------------------------------

# Full 12-channel raw array from extract_observations_explicit:
#   0:T_r  1:T_j  2:Q_j  3:x_D  4:x_B  5:T_reb  6:Q_reb
#   7:F_R_norm  8:F_B_norm  9:R_norm  10:V_norm  11:z_A
RAW_CHANNELS = [
    "T_r", "T_j", "Q_j", "x_D", "x_B",
    "T_reb", "Q_reb", "F_R_norm", "F_B_norm", "R_norm", "V_norm", "z_A",
]
RAW_INDEX = {name: i for i, name in enumerate(RAW_CHANNELS)}

# S-A: include x_D (composition analyser available)
SA_CHANNELS = ["T_r", "T_j", "Q_j", "x_D", "T_reb", "Q_reb",
               "F_R_norm", "F_B_norm", "R_norm", "V_norm"]
SA_INDICES  = [RAW_INDEX[c] for c in SA_CHANNELS]
N_SA = len(SA_CHANNELS)   # 10

# S-B: no online composition measurement
SB_CHANNELS = ["T_r", "T_j", "Q_j", "T_reb", "Q_reb",
               "F_R_norm", "F_B_norm", "R_norm", "V_norm"]
SB_INDICES  = [RAW_INDEX[c] for c in SB_CHANNELS]
N_SB = len(SB_CHANNELS)   # 9

# Per-channel statistic names
STAT_NAMES = ["mean", "std", "slope", "min", "max", "q_mean"]

# Physics-informed feature names (12)
PHYSICS_FEATURE_NAMES = [
    "UA_proxy",          # Q_j / max(T_r - T_j, 1e-3) → encodes beta_r
    "recycle_ratio",     # F_R_norm / F_R_NOM → encodes alpha (snowball) + eta_col
    "col_recovery",      # F_B_norm / (F_R_norm + 1) → alpha × eta_col combined
    "reb_per_boilup",    # Q_reb / V_norm → column heat per boilup unit, eta_col-specific
    "recycle_excess",    # F_R_norm - 1 → snowball severity
    "Tr_Tj_ratio",       # T_r / T_j → encodes beta_r
    "Qj_slope",          # slope of Q_j → transient alpha response
    "Vn_final",          # V_norm final value → boilup compensation effort
    "Rn_final",          # R_norm final value → reflux compensation effort
    "corr_Qj_FR",        # corr(Q_j, F_R_norm) → snowball coupling (alpha)
    "corr_Qreb_FR",      # corr(Q_reb, F_R_norm) → column-recycle coupling (eta_col)
    "corr_Rn_Vn",        # corr(R_norm, V_norm) → coordinated column response
]
N_PHYS = len(PHYSICS_FEATURE_NAMES)   # 12

N_SUMMARIES_SB = N_SB * len(STAT_NAMES) + N_PHYS    # 9*6 + 12 = 66
N_SUMMARIES_SA = N_SA * len(STAT_NAMES) + N_PHYS    # 10*6 + 12 = 72


# ---------------------------------------------------------------------------
# Per-channel statistics
# ---------------------------------------------------------------------------

def _channel_stats(x: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """6-D statistics for a 1-D time series x of length n_t.

    Returns [mean, std, slope, min, max, final-quarter mean].
    """
    n = len(x)
    mn  = np.mean(x)
    sd  = np.std(x) + 1e-12
    # Linear trend via analytic formula (more stable than polyfit for short series)
    if t is not None and len(t) > 1:
        t_c = t - t.mean()
        t_var = np.var(t_c) + 1e-20
        slope = np.dot(t_c, x - mn) / (len(x) * t_var)
    else:
        slope = (x[-1] - x[0]) / (n - 1 + 1e-12)
    mn_q = np.mean(x[max(0, 3*n//4):])   # last-quarter mean
    return np.array([mn, sd, slope, np.min(x), np.max(x), mn_q], dtype=np.float32)


def channel_summaries(obs: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """Compute per-channel statistics for a (n_t, n_ch) observation array.

    Returns (n_ch * 6,) array.
    """
    n_ch = obs.shape[1]
    parts = [_channel_stats(obs[:, c], t) for c in range(n_ch)]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Physics-informed features
# ---------------------------------------------------------------------------

def physics_features(raw: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """Compute the 12 physics-informed features from the full 12-channel raw array.

    Parameters
    ----------
    raw : (n_t, 12) — output of extract_observations_explicit
          columns: T_r, T_j, Q_j, x_D, x_B, T_reb, Q_reb, F_R_norm, F_B_norm, R_norm, V_norm, z_A
    t   : (n_t,) time grid [h]
    """
    T_r     = raw[:, 0]
    T_j     = raw[:, 1]
    Q_j     = raw[:, 2]
    # x_D   = raw[:, 3]
    # x_B   = raw[:, 4]
    # T_reb = raw[:, 5]
    Q_reb   = raw[:, 6]
    F_R_n   = raw[:, 7]
    F_B_n   = raw[:, 8]
    R_norm  = raw[:, 9]
    V_norm  = raw[:, 10]

    # 1. UA proxy: Q_j / (T_r - T_j) — mean over window
    dT_rj   = np.maximum(T_r - T_j, 1e-3)
    UA_proxy = np.mean(Q_j / dT_rj) / (UA_NOM * 1e-6 + 1e-12)   # normalise by UA_NOM

    # 2. Recycle ratio (relative to nominal)
    recycle_ratio = np.mean(F_R_n)

    # 3. Column recovery proxy: F_B_norm / (F_R_norm + F_B_norm)
    col_recovery = np.mean(F_B_n / np.maximum(F_R_n + F_B_n, 1e-3))

    # 4. Reboiler duty per boilup unit: Q_reb / V_norm (eta_col-specific, avoids
    #    F_R_norm/alpha confounding present in the old reb_intensity feature)
    reb_per_boilup = np.mean(Q_reb / np.maximum(V_norm * QREB_NOM, 1e3))

    # 5. Recycle excess relative to nominal
    recycle_excess = np.mean(F_R_n) - 1.0

    # 6. T_r / T_j ratio
    Tr_Tj_ratio = np.mean(T_r / np.maximum(T_j, 200.0))

    # 7. Slope of Q_j
    n = len(Q_j)
    if t is not None and len(t) > 1:
        t_c = t - t.mean()
        t_var = np.var(t_c) + 1e-20
        Qj_slope = float(np.dot(t_c, Q_j - np.mean(Q_j)) / (n * t_var))
    else:
        Qj_slope = float((Q_j[-1] - Q_j[0]) / (n - 1 + 1e-12))
    Qj_slope /= (QJ_NOM + 1e-12)   # normalise

    # 8–9. Final actuator effort
    Vn_final = float(V_norm[-1])
    Rn_final = float(R_norm[-1])

    # 10. corr(Q_j, F_R_norm)
    corr_Qj_FR = _safe_corr(Q_j, F_R_n)

    # 11. corr(Q_reb, F_R_norm)
    corr_Qreb_FR = _safe_corr(Q_reb, F_R_n)

    # 12. corr(R_norm, V_norm)
    corr_Rn_Vn = _safe_corr(R_norm, V_norm)

    return np.array([
        UA_proxy, recycle_ratio, col_recovery, reb_per_boilup,
        recycle_excess, Tr_Tj_ratio, Qj_slope, Vn_final, Rn_final,
        corr_Qj_FR, corr_Qreb_FR, corr_Rn_Vn,
    ], dtype=np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns 0 if either signal is near-constant.

    Uses a range-relative threshold: if std < 1e-4 * (max-min+1e-6),
    the signal is treated as constant. This avoids NaN from np.corrcoef
    when S-A composition control holds a channel nearly flat.
    """
    a_range = float(np.max(a) - np.min(a)) + 1e-6
    b_range = float(np.max(b) - np.min(b)) + 1e-6
    if np.std(a) < 1e-4 * a_range or np.std(b) < 1e-4 * b_range:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


# ---------------------------------------------------------------------------
# Full summary vector
# ---------------------------------------------------------------------------

def _compress_heavy_tails(x: np.ndarray) -> np.ndarray:
    """Signed log1p compression to tame heavy-tailed summary statistics.

    Raw-unit channels (Q_j, Q_reb, UA_proxy) and channel stats for near-runaway
    prior draws (T_r, T_j, V_norm) span many orders of magnitude with a handful
    of extreme outliers (>10x IQR). sbi's mean/std z-scoring is dominated by
    these outliers, which destabilises SNPE flow training in a seed-dependent
    way (identical data/architecture, different random init -> pass/fail SBC).
    A variance-stabilising signed-log compression preserves sign and ordering
    while bounding outlier influence, applied uniformly so downstream code
    doesn't need to track which dimensions were heavy-tailed.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def compute_summaries(
    raw: np.ndarray,
    structure: str = "S-B",
    t: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the full summary vector for one observation window.

    Parameters
    ----------
    raw       : (n_t, 12) — raw output of extract_observations_explicit
    structure : "S-A" or "S-B"
    t         : (n_t,) time grid [h] for slope and slope-based features

    Returns
    -------
    (66,) for S-B  or  (72,) for S-A  float32 summary vector, signed-log
    compressed (see _compress_heavy_tails).
    """
    if structure == "S-A":
        obs = raw[:, SA_INDICES]
    elif structure == "S-B":
        obs = raw[:, SB_INDICES]
    else:
        raise ValueError(f"Unknown structure: {structure!r} — expected 'S-A' or 'S-B'")

    chan_stats = channel_summaries(obs, t)
    phys_feats = physics_features(raw, t)
    summary = np.concatenate([chan_stats, phys_feats]).astype(np.float32)
    return _compress_heavy_tails(summary)


def compute_summaries_batch(
    raw_batch: np.ndarray,
    structure: str = "S-B",
    t: np.ndarray | None = None,
) -> np.ndarray:
    """Batch version: (batch, n_t, 12) → (batch, n_summaries).

    NaN-safe: rows with NaN in raw are zeroed and flagged separately.
    """
    n_batch = raw_batch.shape[0]
    ex = compute_summaries(raw_batch[0], structure, t)
    out = np.zeros((n_batch, len(ex)), dtype=np.float32)
    for i in range(n_batch):
        row = raw_batch[i]
        if np.any(np.isnan(row)) or np.any(np.isinf(row)):
            continue
        s = compute_summaries(row, structure, t)
        if not (np.any(np.isnan(s)) or np.any(np.isinf(s))):
            out[i] = s
    return out


def summary_names(structure: str = "S-B") -> list[str]:
    """Return ordered list of summary statistic names."""
    channels = SA_CHANNELS if structure == "S-A" else SB_CHANNELS
    names = []
    for ch in channels:
        for stat in STAT_NAMES:
            names.append(f"{ch}_{stat}")
    names.extend(PHYSICS_FEATURE_NAMES)
    return names
