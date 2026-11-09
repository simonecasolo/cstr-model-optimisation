"""Summary statistics for the Wu 2003 recycle plant SBI study.

Summary vector layout
---------------------
S-B observations (9 channels):
    T_r, T_j, Q_j, T_reb, Q_reb, F_R_norm, F_B_norm, R_norm, V_norm

S-A observations (10 channels):
    T_r, T_j, Q_j, x_D, T_reb, Q_reb, F_R_norm, F_B_norm, R_norm, V_norm

S-C observations (S-B's 9 channels + a feed-line composition analyzer):
    Same 9 channels as S-B, plus a synthetic feed-analyzer reading (see
    `feed_analyzer_reading` below). Added per
    `docs/Manuscript/system_ii_analyzer_rework_plan.md` to resolve the confirmed
    `(alpha, z_A0_eff)`/`(eta_col, z_A0_eff)` identifiability limits found in
    `notebooks/39-42` — a reactor-outlet composition sensor (`z_A`) was tested first
    and rejected (`notebooks/43`): its own dynamics are driven by both `alpha` and
    `z_A0_eff` at once, so observing it doesn't separate them. A feed-line analyzer,
    measuring the fresh feed's `z_A0_eff` directly upstream of the reactor and
    recycle mixing point, does resolve both confounds, but only above a precision
    threshold (`notebooks/43`'s 7-point sweep): the *existing* 9-channel summaries
    already constrain both parameters to `sd~0.0003-0.0005` along a correlated
    direction, so a bolt-on sensor must beat that existing implicit precision, not
    just add "a" measurement. The settled requirement is `<=0.01%` relative noise
    (`FEED_ANALYZER_NOISE_PCT` below) — see `references.bib:KettBSS1700` for a real
    analyzer meeting this spec.

Per-channel statistics (6 per channel):
    mean, std, slope (linear trend / (t_max-t_min)), min, max, final-quarter mean

Physics-informed features (12):
    See PHYSICS_FEATURE_NAMES below.

Total: S-B → 9×6 + 12 = 66-D; S-A → 10×6 + 12 = 72-D; S-C → 66 + 6 = 72-D;
       S-C.2 → 72 + 2 = 74-D

S-C.2 observations (S-C's 72 channels + 2 engineered `xi_reb`-sharpening features):
    Added 2026-09-04 per `docs/Manuscript/system_ii_sbi_solidification_plan.md`
    Stage 2c (upgraded). `xi_reb` enters the model *only* through the `Q_reb` proxy
    (verified directly in `physics.py::_col_metrics`:
    `Q_reb = QREB_NOM*throughput_rat*V_norm*(1+2*x_B_excess+0.5*(1-eta_col))/xi_reb`),
    so all its information is spread across heavy-tailed raw-unit `Q_reb` stats.
    Two closed-form features sharpen this, computed from channels already observed
    under S-B (no new sensor):
      `xi_reb_qreb_feat` = `log(Q_reb_mean / (QREB_NOM * throughput_rat * V_norm_mean))`
          `= log(1 + 2*x_B_excess + 0.5*(1-eta_col)) - log(xi_reb)`
      `xi_reb_treb_feat` = `T_reb_mean - T_REB_NOM - 3*(F_R_norm_mean-1) - 12*(V_norm_mean-1)`
          `= 90*x_B_excess + 8*(1-eta_col)` (the `T_reb` proxy's own closed form,
          `physics.py::_col_metrics`)
    Tested empirically on `data/wu2003_sbi_train_sc_lognormal_true_n5000.npz`:
    `xi_reb_qreb_feat` alone already improves `corr(., xi_reb)` from `-0.65` (raw
    `Q_reb_mean`) to `-0.87`, but carries residual coupling with both `eta_col`
    (`-0.23`) and, found directly rather than assumed, `alpha` (`-0.38`, since
    `x_B_excess` turns out to be driven far more by `alpha`'s effect on reactor
    conversion than by `eta_col`). Both engineered features share the exact same
    `x_B_excess` term at two different, known coefficients (90 vs 2) -- an SNPE
    flow given BOTH raw ingredients (not a single pre-combined value) can learn to
    cancel it itself; a linear OLS fit of `-log(xi_reb)` on the two features
    reaches `R²=0.985` in-sample and held-out, with `corr(fitted, xi_reb)=-0.99`,
    `corr(., eta_col)=-0.11`, `corr(., alpha)=+0.02` -- both residual couplings
    collapse simultaneously. Feeding the two raw features (not the pre-fit
    combination) lets training find/refine that combination rather than baking in
    one bank's regression coefficients.
"""

from __future__ import annotations

import numpy as np

from cstr_sbi.recycle.physics import (
    F0_NOM, F_R_NOM, F_B_NOM, F_TOT_NOM, QJ_NOM, QREB_NOM, T_SP, UA_NOM,
    X_D_NOM, X_B_NOM, REFLUX_RATIO, T_REB_NOM, PARAM_NAMES,
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

# S-C: S-B's 9 channels + a synthetic feed-line composition analyzer reading.
# Not a `RAW_CHANNELS`/`RAW_INDEX` entry -- the feed-analyzer channel is synthesized
# from theta directly (see `feed_analyzer_reading`), not extracted from the raw
# trajectory. This list exists only for labeling the 10-channel raw observation view
# used in dataset-generation notebooks (nb22a+); `compute_summaries`/`summary_names`
# handle the "S-C" structure independently of this list.
SC_CHANNELS = list(SB_CHANNELS) + ["feed_analyzer"]

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
N_SUMMARIES_SC = N_SUMMARIES_SB + len(STAT_NAMES)   # 66 + 6 = 72

# xi_reb-sharpening engineered features (S-C.2 only, see module docstring)
XI_REB_FEATURE_NAMES = ["xi_reb_qreb_feat", "xi_reb_treb_feat"]
N_SUMMARIES_SC2 = N_SUMMARIES_SC + len(XI_REB_FEATURE_NAMES)   # 72 + 2 = 74

# Settled precision requirement for the S-C feed-line analyzer, see notebooks/43's
# 7-point sweep and the module docstring above. Relative noise, i.e. the reading's
# assumed std is `FEED_ANALYZER_NOISE_PCT * abs(z_A0_eff)`.
FEED_ANALYZER_NOISE_PCT = 1e-4   # 0.01%

Z_A0_EFF_PARAM_IDX = PARAM_NAMES.index("z_A0_eff")


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


def feed_analyzer_reading(
    theta: np.ndarray,
    t: np.ndarray | None,
    rng: np.random.Generator,
    feed_analyzer_noise_pct: float = FEED_ANALYZER_NOISE_PCT,
) -> np.ndarray:
    """Synthetic online feed-line composition analyzer reading (structure "S-C").

    A near-constant, noisy measurement of the true feed A mole fraction
    `theta[Z_A0_EFF_PARAM_IDX]`, taken upstream of the reactor and recycle mixing
    point -- i.e. it measures the *fresh* feed directly, not the reactor-internal
    or recycle-mixed composition. `z_A0_eff` does not vary within a single
    scenario window in this model, so the reading is a constant plus sensor noise,
    not a dynamic trajectory -- this is deliberately a different kind of channel
    from the state-derived ones in `RAW_CHANNELS` (see the module docstring for why
    a reactor-outlet composition channel was tried first and rejected).

    Parameters
    ----------
    theta : (5,) true parameter vector [alpha, beta_r, eta_col, xi_reb, z_A0_eff]
    t     : (n_t,) time grid -- only its length is used, to match the other
            channels' window length
    rng   : numpy Generator for the reading's own noise draw. Pass the same `rng`
            already used for the rest of the observation's sensor noise (e.g. in
            `sbi_pipeline._simulate_summary`) -- this just advances its stream,
            it does not need to be a separate generator.
    """
    z_a0 = float(theta[Z_A0_EFF_PARAM_IDX])
    n_t = len(t) if t is not None else 120
    noise_sd = feed_analyzer_noise_pct * max(abs(z_a0), 1e-3)
    reading = np.full(n_t, z_a0, dtype=np.float32) + rng.normal(0, noise_sd, n_t)
    return reading.astype(np.float32)


# SB_CHANNELS-relative indices for the two channels the xi_reb-sharpening
# features need (T_reb, Q_reb, F_R_norm, F_B_norm, V_norm) -- SB_CHANNELS =
# ["T_r","T_j","Q_j","T_reb","Q_reb","F_R_norm","F_B_norm","R_norm","V_norm"]
_XI_REB_FEAT_CH = {name: SB_CHANNELS.index(name)
                   for name in ("T_reb", "Q_reb", "F_R_norm", "F_B_norm", "V_norm")}


def xi_reb_sharpened_features(obs: np.ndarray) -> np.ndarray:
    """Two closed-form, `xi_reb`-sharpening features (structure "S-C.2" only).

    `obs` : (n_t, N_SB) -- the SB-indexed raw observation window (same array
    `compute_summaries` already slices via `SB_INDICES`), BEFORE
    `_compress_heavy_tails`. See the module docstring for the derivation and
    the empirical validation (`r=-0.99` for the fitted linear combination of
    these two features vs. true `xi_reb`, `R²=0.985` held-out).

    Returns `[xi_reb_qreb_feat, xi_reb_treb_feat]`.
    """
    T_reb_mean = float(np.mean(obs[:, _XI_REB_FEAT_CH["T_reb"]]))
    Q_reb_mean = float(np.mean(obs[:, _XI_REB_FEAT_CH["Q_reb"]]))
    F_R_norm_mean = float(np.mean(obs[:, _XI_REB_FEAT_CH["F_R_norm"]]))
    F_B_norm_mean = float(np.mean(obs[:, _XI_REB_FEAT_CH["F_B_norm"]]))
    V_norm_mean = float(np.mean(obs[:, _XI_REB_FEAT_CH["V_norm"]]))

    throughput_rat = (F_R_norm_mean * F_R_NOM + F_B_norm_mean * F_B_NOM) / F_TOT_NOM
    denom = QREB_NOM * throughput_rat * max(V_norm_mean, 1e-6)
    xi_reb_qreb_feat = float(np.log(max(Q_reb_mean, 1e-6) / max(denom, 1e-6)))
    xi_reb_treb_feat = T_reb_mean - T_REB_NOM - 3.0 * (F_R_norm_mean - 1.0) - 12.0 * (V_norm_mean - 1.0)
    return np.array([xi_reb_qreb_feat, xi_reb_treb_feat], dtype=np.float32)


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
    theta: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    feed_analyzer_noise_pct: float = FEED_ANALYZER_NOISE_PCT,
) -> np.ndarray:
    """Compute the full summary vector for one observation window.

    Parameters
    ----------
    raw       : (n_t, 12) — raw output of extract_observations_explicit
    structure : "S-A", "S-B", "S-C" ("S-B" + a feed-line composition analyzer), or
                "S-C.2" ("S-C" + 2 closed-form xi_reb-sharpening features, see the
                module docstring)
    t         : (n_t,) time grid [h] for slope and slope-based features
    theta     : (5,) true parameter vector — required only for structure="S-C"/"S-C.2",
                since the feed-analyzer reading is synthesized from
                `theta[Z_A0_EFF_PARAM_IDX]` directly rather than extracted from
                `raw` (see `feed_analyzer_reading`).
    rng       : numpy Generator — required only for structure="S-C"/"S-C.2", used for
                the feed-analyzer reading's own noise draw (pass the same `rng`
                already used for the rest of the observation's sensor noise).
    feed_analyzer_noise_pct : relative noise assumed for the S-C/S-C.2 feed-analyzer
                reading; ignored unless structure is "S-C"/"S-C.2". Default is the
                settled `0.01%` threshold, see `FEED_ANALYZER_NOISE_PCT`'s docstring.

    Returns
    -------
    (66,) for S-B, (72,) for S-A, (72,) for S-C, or (74,) for S-C.2 — float32
    summary vector, signed-log compressed (see _compress_heavy_tails).
    """
    if structure == "S-A":
        obs = raw[:, SA_INDICES]
    elif structure in ("S-B", "S-C", "S-C.2"):
        obs = raw[:, SB_INDICES]
    else:
        raise ValueError(f"Unknown structure: {structure!r} — expected 'S-A', 'S-B', 'S-C', or 'S-C.2'")

    chan_stats = channel_summaries(obs, t)
    phys_feats = physics_features(raw, t)
    parts = [chan_stats, phys_feats]

    if structure in ("S-C", "S-C.2"):
        if theta is None or rng is None:
            raise ValueError(
                f"structure={structure!r} requires theta and rng — the feed-analyzer "
                "reading is synthesized from theta[Z_A0_EFF_PARAM_IDX], not extracted from raw"
            )
        feed_reading = feed_analyzer_reading(theta, t, rng, feed_analyzer_noise_pct)
        parts.append(_channel_stats(feed_reading, t))

    if structure == "S-C.2":
        parts.append(xi_reb_sharpened_features(obs))

    summary = np.concatenate(parts).astype(np.float32)
    return _compress_heavy_tails(summary)


def compute_summaries_batch(
    raw_batch: np.ndarray,
    structure: str = "S-B",
    t: np.ndarray | None = None,
    theta_batch: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    feed_analyzer_noise_pct: float = FEED_ANALYZER_NOISE_PCT,
) -> np.ndarray:
    """Batch version: (batch, n_t, 12) → (batch, n_summaries).

    NaN-safe: rows with NaN in raw are zeroed and flagged separately.

    theta_batch : (batch, 5) — required only for structure="S-C" (one row per
        `raw_batch` row, see `compute_summaries`).
    """
    n_batch = raw_batch.shape[0]
    theta0 = theta_batch[0] if theta_batch is not None else None
    ex = compute_summaries(raw_batch[0], structure, t, theta=theta0, rng=rng,
                            feed_analyzer_noise_pct=feed_analyzer_noise_pct)
    out = np.zeros((n_batch, len(ex)), dtype=np.float32)
    for i in range(n_batch):
        row = raw_batch[i]
        if np.any(np.isnan(row)) or np.any(np.isinf(row)):
            continue
        theta_i = theta_batch[i] if theta_batch is not None else None
        s = compute_summaries(row, structure, t, theta=theta_i, rng=rng,
                               feed_analyzer_noise_pct=feed_analyzer_noise_pct)
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
    if structure in ("S-C", "S-C.2"):
        names.extend(f"feed_analyzer_{stat}" for stat in STAT_NAMES)
    if structure == "S-C.2":
        names.extend(XI_REB_FEATURE_NAMES)
    return names
