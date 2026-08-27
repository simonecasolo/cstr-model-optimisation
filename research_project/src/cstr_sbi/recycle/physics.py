"""Wu 2003 CSTR-column-recycle physics module.

Implements the deterministic dynamics of the Wu (2003) liquid-phase A→B
CSTR with distillation column recycle.  The column quasi-steady state (QSS)
is solved by a JAX-compiled McCabe-Thiele bisection (50 lax.scan steps),
which is physically correct and automatically differentiable.

Reference:
    Wu, K.-L., Yu, C.-C., Luyben, W. L., & Skogestad, S. (2003).
    Reactor/separator processes with recycle -- 2. Design for composition
    control. Computers & Chemical Engineering, 27(3), 401-421.

Unit system: Btu-lbmol-h-K
    Flows   : lbmol/h
    Holdup  : lbmol
    Energy  : Btu/h  (rates) or Btu (total)
    Time    : h
    Temp    : K  (Kelvin; R_GAS = 3.576 Btu/(lbmol·K))

Gas constant in Kelvin:
    Wu 2003 uses Ea = 30,841 Btu/lbmol with R = 1.987 Btu/(lbmol·°R).
    Converting to Kelvin: R_GAS = 1.987 × (9/5) = 3.576 Btu/(lbmol·K).
    Verification: k(342.26 K) = K0 × exp(−30841/(3.576×342.26)) = 0.33 /h ✓

State vectors
-------------
4-state model (diagnostic / open-loop):
    y = [z_A, T_r, T_j, I_T]

8-state explicit-loop model (primary SBI / EKF model):
    y = [z_A, T_r, T_j, I_T, R_state, V_norm_state, I_R, I_V]

Degradation parameter vector (5-D):
    theta = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import diffrax


# ---------------------------------------------------------------------------
# Physical constants — Btu-lbmol-h-K unit system
# ---------------------------------------------------------------------------

R_GAS: float = 3.576          # Btu/(lbmol·K)  [= 1.987 × 9/5]
EA: float = 30841.0           # Btu/lbmol  activation energy (Wu 2003 Table 1)
K_SS: float = 0.33            # /h  rate constant at nominal SS temperature
T_SS: float = (156.4 - 32.0) * 5.0/9.0 + 273.15   # 342.26 K  (156.4°F)
# k0 = k_ss × exp(Ea / (R_GAS × T_ss)) = 0.33 × exp(25.20) ≈ 2.91e10 /h
K0: float = K_SS * float(jnp.exp(jnp.array(EA / (R_GAS * T_SS))))

DH_RXN: float = -30000.0     # Btu/lbmol  (exothermic, Wu Table 1: 30,000)
# Cp_molar [Btu/(lbmol·K)] = 0.7 Btu/(lb·°R) × 100 lb/lbmol × (9/5 °R/K)
CP_MOLAR: float = 0.7 * 100.0 * (9.0/5.0)   # = 126.0 Btu/(lbmol·K)

# UA: 150.5 Btu/(h·ft²·°F) × 3206.8 ft² = 482,624 Btu/(h·°F)
# Btu/(h·K) = Btu/(h·°F) × (9/5)  since 1 K = 9/5 °R = 9/5 °F difference
UA_NOM: float = 150.5 * 3206.8 * (9.0/5.0)   # = 868,723 Btu/(h·K)

# Jacket heat capacity: M_j × Cp_j
# V_r = M_r × MW / ρ = 2400×100/60.05 ≈ 3995 ft³; jacket ≈ 10% → 400 ft³ water
# M_j = 400 ft³ × 62.4 lb/ft³ / 18 lb/lbmol = 1387 lbmol
# Cp_water [Btu/(lbmol·K)] = 1.0 Btu/(lb·°R) × 18 lb/lbmol × (9/5) = 32.4
MJ_CPJ: float = 1387.0 * 32.4   # ≈ 44,939 Btu/K

# Nominal operating point (Wu 2003 Table 1)
F0_NOM: float = 460.0         # lbmol/h  fresh feed
Z0_NOM: float = 0.90          # mol/mol  fresh feed A composition
T_IN: float = (70.0 - 32.0) * 5.0/9.0 + 273.15   # 294.26 K  (70°F)
MR_NOM: float = 2400.0        # lbmol    reactor holdup (constant)
T_SP: float = T_SS            # K   reactor temperature setpoint

# Nominal jacket temperature (136.1°F)
T_J_NOM: float = (136.1 - 32.0) * 5.0/9.0 + 273.15   # 330.98 K

# Column design (Wu 2003 Table 1)
ALPHA_REL: float = 2.0        # relative volatility A/B
N_TRAYS: int = 20             # number of equilibrium trays
FEED_TRAY: int = 11           # 0-indexed; feed at tray 12 of 20 (Wu Table 1)
X_D_NOM: float = 0.95         # distillate A mol-fraction
X_B_NOM: float = 0.0105       # bottoms A mol-fraction
REFLUX_RATIO: float = 2.198   # L/D  (= 1100/500.4)

# Nominal recycle/product flows
F_R_NOM: float = 500.4        # lbmol/h  recycle (= distillate)
F_B_NOM: float = 460.0        # lbmol/h  product (= bottoms)

# D_FRAC_NOM = D / (D+B) — nominal, used for operating-line coefficients
D_FRAC_NOM: float = F_R_NOM / (F_R_NOM + F_B_NOM)   # ≈ 0.521
F_TOT_NOM: float = F_R_NOM + F_B_NOM                 # ≈ 960.4

# Snowball sensitivity: D_frac increases with z_F (recycle loop amplification).
# When α decreases, z_F rises above Z_F_RECYCLE_REF; the column must send
# more A overhead to maintain bottoms purity → D_frac increases → F_R increases.
RECYCLE_SENSITIVITY: float = 0.12   # d(D_frac)/d(z_F)
Z_F_RECYCLE_REF: float = 0.500     # nominal reactor outlet composition

# Reboiler proxy baselines (used for T_reb and Q_reb observations)
T_REB_NOM: float = 372.0      # K
QREB_NOM: float = 1.2e7       # Btu/h

# PI controller for Loop 1 (reactor temperature → jacket duty)
# QJ_NOM derived from nominal energy balance at steady state:
#   Q_j_ss = M_r × Cp × (F_tot/M_r × (T_in_mix − T_sp) + (−ΔH) × k_ss × z_A_ss / Cp)
# Nominal z_A_ss ≈ 0.500, T_in_mix ≈ 322 K, gives Q_j_ss ≈ 9.35e6 Btu/h
QJ_NOM: float = 9.351e6       # Btu/h  bias term
KP1: float = 8.0e6            # Btu/(h·K)
TAU_I1: float = 0.1           # h
QJ_MIN: float = 0.0
QJ_MAX: float = 3.0e7         # Btu/h

# Explicit Loop 2/3 controller parameters
STRUCTURE_SA: float = 0.0     # flag value for S-A mode
STRUCTURE_SB: float = 1.0     # flag value for S-B mode
L_NOM: float = 1100.0         # lbmol/h  nominal reflux flow
V_NOM: float = 1600.4         # lbmol/h  nominal vapor boilup
RECYCLE_RATIO_NOM: float = F_R_NOM / F0_NOM   # ≈ 1.088

# Loop 2: S-A closes on x_D; S-B uses recycle-ratio control
KP_R_SA: float = 8.0          # [R_norm / mol-frac]
TAU_I_R_SA: float = 0.08      # h
KP_R_SB: float = 0.0          # no PI in ratio RC mode (feed-forward)
TAU_I_R_SB: float = 0.12      # h (inactive)
R_MIN: float = 1.0
R_MAX: float = 4.0
TAU_R_ACT: float = 0.04       # h  reflux actuator lag

# Loop 3: S-A closes on x_B; S-B closes on T_reb
KP_V_SA: float = 28.0
TAU_I_V_SA: float = 0.08
KP_V_SB: float = 0.055
TAU_I_V_SB: float = 0.12
V_NORM_MIN: float = 0.5
V_NORM_MAX: float = 1.8
TAU_V_ACT: float = 0.04

T_REB_SP: float = T_REB_NOM  # K  S-B reboiler temperature setpoint


# ---------------------------------------------------------------------------
# Nominal JAX arrays
# ---------------------------------------------------------------------------

NOMINAL_THETA = jnp.array([1.0, 1.0, 1.0, 1.0, Z0_NOM], dtype=jnp.float32)
NOMINAL_INLET = jnp.array([F0_NOM, T_IN], dtype=jnp.float32)
NOMINAL_CTRL  = jnp.array(
    [KP1, TAU_I1, T_SP, QJ_NOM, QJ_MIN, QJ_MAX], dtype=jnp.float32
)

# Full controller vector layout (27 elements):
# [0:6]   Loop 1: Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max
# [6]     mode flag (0=S-A, 1=S-B)
# [7:10]  S-A Loop 2: Kp_R, tau_R, x_D_sp
# [10:13] S-B Loop 2: Kp_R, tau_R, recycle_ratio_sp
# [13:17] Reflux actuator: R0, R_min, R_max, tau_R_act
# [17:20] S-A Loop 3: Kp_V, tau_V, x_B_sp
# [20:23] S-B Loop 3: Kp_V, tau_V, T_reb_sp
# [23:27] Boilup actuator: V_norm0, V_min, V_max, tau_V_act
NOMINAL_CTRL_SA = jnp.array([
    KP1, TAU_I1, T_SP, QJ_NOM, QJ_MIN, QJ_MAX,      # Loop 1  [0:6]
    STRUCTURE_SA,                                      # mode    [6]
    KP_R_SA, TAU_I_R_SA, X_D_NOM,                    # S-A L2  [7:10]
    KP_R_SB, TAU_I_R_SB, RECYCLE_RATIO_NOM,          # S-B L2  [10:13]
    REFLUX_RATIO, R_MIN, R_MAX, TAU_R_ACT,            # R act.  [13:17]
    KP_V_SA, TAU_I_V_SA, X_B_NOM,                    # S-A L3  [17:20]
    KP_V_SB, TAU_I_V_SB, T_REB_SP,                   # S-B L3  [20:23]
    1.0, V_NORM_MIN, V_NORM_MAX, TAU_V_ACT,           # V act.  [23:27]
], dtype=jnp.float32)

NOMINAL_CTRL_SB = NOMINAL_CTRL_SA.at[6].set(STRUCTURE_SB)

# Nominal 4-state warm start [z_A, T_r, T_j, I_T]
NOMINAL_Y0 = jnp.array([0.500, T_SP, T_J_NOM, 0.0], dtype=jnp.float32)

# Nominal 8-state warm start [z_A, T_r, T_j, I_T, R_state, V_norm_state, I_R, I_V]
NOMINAL_Y0_EXPLICIT = jnp.array(
    [0.500, T_SP, T_J_NOM, 0.0, REFLUX_RATIO, 1.0, 0.0, 0.0],
    dtype=jnp.float32,
)

PARAM_NAMES = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]


# ---------------------------------------------------------------------------
# QSS Distillation Column — McCabe-Thiele bisection
# ---------------------------------------------------------------------------
#
# Given reactor outlet z_F and tray efficiency eta_col, we find x_D by
# bisecting on the McCabe-Thiele residual:
#
#   residual(x_D) = x_reboiler(x_D) − x_B_mb(x_D)
#
# where:
#   x_B_mb(x_D) = (z_F − D_FRAC_NOM × x_D) / (1 − D_FRAC_NOM)
#       is the bottoms composition required by the overall material balance
#       assuming D/F is fixed at D_FRAC_NOM by the reflux-drum level control.
#
#   x_reboiler(x_D) is the liquid composition at the column bottom after
#       stepping N_TRAYS McCabe-Thiele stages downward from x_D.
#
# Bisection criterion (monotone in x_D):
#   x_reb < x_B_mb  →  column over-separates relative to MB requirement
#                    →  x_D is too low  →  lo = mid (increase x_D)
#   x_reb ≥ x_B_mb  →  column under-separates
#                    →  x_D is too high →  hi = mid (decrease x_D)
#
# 50 bisection steps give a resolution of (0.9998 − z_F) / 2^50 ≈ 1e-15.


def _col_step(x, i, alpha_eff, x_D, x_B_mb, lv, dv, lv_s, bv_s):
    """One McCabe-Thiele tray step (stepping downward from condenser)."""
    in_stripping = i >= FEED_TRAY
    y_rect  = lv  * x   + dv  * x_D
    y_strip = lv_s * x  - bv_s * x_B_mb
    y_op = jnp.where(in_stripping, y_strip, y_rect)
    y_op = jnp.clip(y_op, 1e-7, 1.0 - 1e-7)
    x_new = y_op / (alpha_eff - (alpha_eff - 1.0) * y_op)
    return jnp.clip(x_new, 1e-7, 1.0 - 1e-7), None


def _col_coeffs(R: jnp.ndarray) -> Tuple:
    """Operating-line coefficients for reflux ratio R and fixed D_FRAC_NOM."""
    R = jnp.clip(R, 0.5, 8.0)
    lv   = R / (R + 1.0)
    dv   = 1.0 / (R + 1.0)
    fod  = 1.0 / D_FRAC_NOM           # F/D ≈ 1.921
    lv_s = (R + fod) / (R + 1.0)      # Ls/Vs (stripping slope)
    bv_s = (fod - 1.0) / (R + 1.0)    # B/Vs  (stripping intercept)
    return lv, dv, lv_s, bv_s


def _col_residual(x_D, z_F, alpha_eff, lv, dv, lv_s, bv_s):
    """McCabe-Thiele residual at trial x_D: x_reboiler and x_B_mb."""
    x_B_mb = (z_F - D_FRAC_NOM * x_D) / (1.0 - D_FRAC_NOM)
    x_B_mb = jnp.clip(x_B_mb, 1e-6, z_F - 1e-5)
    x_reb, _ = jax.lax.scan(
        lambda x, i: _col_step(x, i, alpha_eff, x_D, x_B_mb, lv, dv, lv_s, bv_s),
        x_D,
        jnp.arange(N_TRAYS, dtype=jnp.int32),
    )
    return x_reb, x_B_mb


def column_qss(
    z_F: jax.typing.ArrayLike,
    eta_col: jax.typing.ArrayLike,
    reflux_ratio: jax.typing.ArrayLike | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Quasi-steady-state column: find x_D by 50-step McCabe-Thiele bisection.

    Parameters
    ----------
    z_F          : reactor outlet (= column feed) A mol-fraction
    eta_col      : overall tray-efficiency scale factor (1.0 = nominal)
    reflux_ratio : actual L/D; defaults to REFLUX_RATIO if None

    Returns
    -------
    x_D    : distillate A mol-fraction
    x_B    : bottoms A mol-fraction
    D_frac : D/F = D_FRAC_NOM (fixed by level control)
    """
    z_F     = jnp.clip(jnp.asarray(z_F,     dtype=jnp.float32), 1e-5, 1.0 - 1e-5)
    eta_col = jnp.asarray(eta_col, dtype=jnp.float32)
    R = (
        jnp.asarray(reflux_ratio, dtype=jnp.float32)
        if reflux_ratio is not None
        else jnp.array(REFLUX_RATIO, dtype=jnp.float32)
    )
    alpha_eff = jnp.clip(1.0 + eta_col * (ALPHA_REL - 1.0), 1.001, 10.0)
    lv, dv, lv_s, bv_s = _col_coeffs(R)

    lo = jnp.clip(z_F + 0.01, 0.10, 0.90)
    hi = jnp.array(0.9998, dtype=jnp.float32)

    def bisect_step(carry, _):
        lo_i, hi_i = carry
        mid = 0.5 * (lo_i + hi_i)
        x_reb, x_B_mb = _col_residual(mid, z_F, alpha_eff, lv, dv, lv_s, bv_s)
        # x_reb < x_B_mb: column over-separates → x_D too low → increase lo
        over_sep = x_reb < x_B_mb
        lo_new = jnp.where(over_sep, mid, lo_i)
        hi_new = jnp.where(over_sep, hi_i, mid)
        return (lo_new, hi_new), None

    (lo_f, hi_f), _ = jax.lax.scan(bisect_step, (lo, hi), None, length=50)
    x_D = jnp.clip(0.5 * (lo_f + hi_f), lo, 0.9998)

    # Effective D_frac: varies with z_F to model the snowball mechanism.
    # When z_F rises (α falls), the column must send more A overhead to
    # maintain bottoms purity → D/F increases → recycle flow F_R increases.
    d_frac_eff = D_FRAC_NOM + RECYCLE_SENSITIVITY * (z_F - Z_F_RECYCLE_REF)
    d_frac_eff = jnp.clip(d_frac_eff, 0.05, 0.95)

    x_B = (z_F - d_frac_eff * x_D) / (1.0 - d_frac_eff)
    x_B = jnp.clip(x_B, 1e-6, z_F - 1e-5)
    return x_D, x_B, d_frac_eff


def _col_metrics(
    z_A: jax.typing.ArrayLike,
    eta_col: jax.typing.ArrayLike,
    xi_reb: jax.typing.ArrayLike,
    R_actual: jax.typing.ArrayLike,
    V_norm: jax.typing.ArrayLike,
) -> tuple:
    """Return all column outputs for the explicit-loop model.

    Uses the McCabe-Thiele bisection with the actual reflux ratio R_actual so
    that S-A (which drives R to maintain x_D) and S-B (which ratio-controls R)
    generate physically distinct x_D and x_B trajectories.

    Returns
    -------
    x_D, x_B, d_frac, F_total, F_R, F_B, F_R_norm, F_B_norm, T_reb, Q_reb
    """
    z_A    = jnp.asarray(z_A,   dtype=jnp.float32)
    xi_reb = jnp.asarray(xi_reb, dtype=jnp.float32)
    V_norm = jnp.clip(jnp.asarray(V_norm, dtype=jnp.float32), V_NORM_MIN, V_NORM_MAX)

    x_D, x_B, d_frac = column_qss(z_A, eta_col, R_actual)
    d_frac_safe = jnp.clip(d_frac, 0.01, 0.98)

    # Total flow from fresh feed and recycle balance: F_total = F0 / (1 - d_frac)
    F_total  = F0_NOM / (1.0 - d_frac_safe)
    F_R      = d_frac_safe * F_total
    F_B      = F_total - F_R
    F_R_norm = F_R / F_R_NOM
    F_B_norm = F_B / F_B_NOM

    # Reboiler temperature and duty proxies
    x_B_excess     = x_B - X_B_NOM
    col_severity   = 1.0 - jnp.asarray(eta_col, dtype=jnp.float32)
    throughput_rat = F_total / F_TOT_NOM
    T_reb = (
        T_REB_NOM
        + 90.0 * x_B_excess
        + 8.0  * col_severity
        + 3.0  * (F_R_norm - 1.0)
        + 12.0 * (V_norm - 1.0)
    )
    Q_reb = QREB_NOM * throughput_rat * V_norm
    Q_reb = Q_reb * (1.0 + 2.0 * x_B_excess + 0.5 * col_severity)
    Q_reb = Q_reb / jnp.clip(xi_reb, 0.2, 2.0)

    return x_D, x_B, d_frac_safe, F_total, F_R, F_B, F_R_norm, F_B_norm, T_reb, Q_reb


# ---------------------------------------------------------------------------
# PI controller utility
# ---------------------------------------------------------------------------

def compute_qj(T_r, I_T, ctrl) -> jnp.ndarray:
    """Jacket duty Q_j [Btu/h] from the Loop 1 PI controller.

    ctrl[0:6] = [Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max]
    Positive Q_j = heat removed from reactor (cooling); increases when T_r > T_sp.
    """
    Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max = (
        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5]
    )
    qj_raw = Qj0 + Kp1 * (T_r - T_sp) + I_T / tau_i1
    return jnp.clip(qj_raw, Qj_min, Qj_max)


def compute_reflux_boilup(x_D, x_B, F_R, T_reb, I_R, I_V, ctrl):
    """Return Loop 2/3 commands and integration increments.

    Returns: R_cmd, V_cmd, dI_R, dI_V, e_R, e_V
    """
    mode  = ctrl[6]
    is_sa = mode < 0.5

    KpR_sa, tauR_sa, xD_sp   = ctrl[7],  ctrl[8],  ctrl[9]
    KpR_sb, tauR_sb, rr_sp   = ctrl[10], ctrl[11], ctrl[12]
    R0, R_min, R_max, tau_Ract = ctrl[13], ctrl[14], ctrl[15], ctrl[16]

    KpV_sa, tauV_sa, xB_sp   = ctrl[17], ctrl[18], ctrl[19]
    KpV_sb, tauV_sb, Treb_sp = ctrl[20], ctrl[21], ctrl[22]
    V0, V_min, V_max, tau_Vact = ctrl[23], ctrl[24], ctrl[25], ctrl[26]

    recycle_ratio = F_R / F0_NOM

    # Loop 2 error
    e_R_sa = xD_sp   - x_D
    e_R_sb = rr_sp   - recycle_ratio
    e_R    = jnp.where(is_sa, e_R_sa, e_R_sb)
    KpR    = jnp.where(is_sa, KpR_sa, KpR_sb)
    tauR   = jnp.where(is_sa, tauR_sa, tauR_sb)

    active_R    = KpR > 1e-8
    R_raw       = jnp.where(active_R, R0 + KpR * e_R + I_R / tauR, R0)
    R_cmd       = jnp.clip(R_raw, R_min, R_max)
    not_sat_R   = (R_raw > R_min) & (R_raw < R_max)
    dI_R        = jnp.where(active_R & not_sat_R, e_R, 0.0)

    # Loop 3 error
    e_V_sa = x_B   - xB_sp
    e_V_sb = Treb_sp - T_reb
    e_V    = jnp.where(is_sa, e_V_sa, e_V_sb)
    KpV    = jnp.where(is_sa, KpV_sa, KpV_sb)
    tauV   = jnp.where(is_sa, tauV_sa, tauV_sb)

    active_V    = KpV > 1e-8
    V_raw       = jnp.where(active_V, V0 + KpV * e_V + I_V / tauV, V0)
    V_cmd       = jnp.clip(V_raw, V_min, V_max)
    not_sat_V   = (V_raw > V_min) & (V_raw < V_max)
    dI_V        = jnp.where(active_V & not_sat_V, e_V, 0.0)

    return R_cmd, V_cmd, dI_R, dI_V, e_R, e_V


# ---------------------------------------------------------------------------
# 4-state ODE (diagnostic / open-loop / simplified closed-loop)
# ---------------------------------------------------------------------------

def recycle_rhs(
    t: float,
    y: jnp.ndarray,
    args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """4-state Wu 2003 ODE with Loop 1 PI only.

    State: y = [z_A, T_r, T_j, I_T]
    Args:  (theta, inlet, ctrl6)
        theta  = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]
        inlet  = [F0, T_in]
        ctrl6  = [Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max]

    Column operates at fixed REFLUX_RATIO (nominal); use recycle_rhs_explicit
    for S-A / S-B with variable reflux.
    """
    theta, inlet, ctrl = args
    alpha, beta_r, eta_col, _xi_reb, z_A0_eff = (
        theta[0], theta[1], theta[2], theta[3], theta[4]
    )
    F0, T_in = inlet[0], inlet[1]

    z_A = jnp.clip(y[0], 1e-6, 1.0 - 1e-6)
    T_r = jnp.maximum(y[1], 250.0)
    T_j = y[2]
    I_T = y[3]

    # QSS column at nominal reflux ratio
    x_D, _x_B, d_frac = column_qss(z_A, eta_col)
    d_frac  = jnp.clip(d_frac, 0.01, 0.98)
    F_total = F0 / (1.0 - d_frac)
    F_R     = d_frac * F_total

    z_A0    = jnp.clip(z_A0_eff, 0.01, 0.999)
    z_A_in  = jnp.clip((F0 * z_A0 + F_R * x_D) / F_total, 1e-6, 1.0 - 1e-6)
    T_in_mix = (F0 * T_in + F_R * T_r) / F_total

    k_eff = jnp.maximum(alpha * K0 * jnp.exp(-EA / (R_GAS * T_r)), 0.0)

    qj_raw = ctrl[3] + ctrl[0] * (T_r - ctrl[2]) + I_T / ctrl[1]
    Q_j    = jnp.clip(qj_raw, ctrl[4], ctrl[5])
    dI_T   = jnp.where((qj_raw > ctrl[4]) & (qj_raw < ctrl[5]), T_r - ctrl[2], 0.0)

    UA_eff    = beta_r * UA_NOM
    Q_transfer = UA_eff * (T_r - T_j)

    dz_A = (F_total / MR_NOM) * (z_A_in - z_A) - k_eff * z_A
    dT_r = (
        (F_total / MR_NOM) * (T_in_mix - T_r)
        + (-DH_RXN) * k_eff * z_A / CP_MOLAR
        - Q_transfer / (MR_NOM * CP_MOLAR)
    )
    # FIXED (2026-08-12, reviewer_response_plan.md Major Comment 6, Finding 3 /
    # pending_manuscript_fixes.md Stage 2): beta_r must not scale the actively-
    # commanded duty Q_j, only the conductive term Q_transfer (already beta_r-
    # scaled via UA_eff above) -- jacket fouling attenuates heat transfer through
    # the wall, not the controller's commanded duty. The previous
    # `(Q_transfer - beta_r * Q_j)` form double-counted fouling on Q_j with no
    # physical justification found anywhere in the text or code comments, and
    # was inconsistent with the sibling (unused-by-the-paper) implementation in
    # `cstr_sbi.luyben.physics`, which never scales Q_j this way. This changes
    # System II's reactor-jacket ODE: all cached System II training banks,
    # trained posteriors (sbi-logs/*.pkl), and downstream results computed
    # before this fix are stale and must be regenerated in Stage 3's matched-
    # protocol retraining, not reused as-is.
    dT_j = (Q_transfer - Q_j) / MJ_CPJ

    return jnp.array([dz_A, dT_r, dT_j, dI_T])


# ---------------------------------------------------------------------------
# 8-state ODE (primary model: explicit S-A / S-B control loops)
# ---------------------------------------------------------------------------

def recycle_rhs_explicit(
    t: float,
    y: jnp.ndarray,
    args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """8-state Wu 2003 ODE with explicit S-A / S-B control loops.

    State: y = [z_A, T_r, T_j, I_T, R_state, V_norm_state, I_R, I_V]
        R_state     : actual reflux ratio L/D
        V_norm_state: normalised vapor boilup (1.0 = nominal)
        I_R, I_V    : Loop 2/3 integral states

    Args: (theta, inlet, ctrl27) — full 27-element controller vector.
    """
    theta, inlet, ctrl = args
    alpha, beta_r, eta_col, xi_reb, z_A0_eff = (
        theta[0], theta[1], theta[2], theta[3], theta[4]
    )
    F0, T_in = inlet[0], inlet[1]

    z_A     = jnp.clip(y[0], 1e-6, 1.0 - 1e-6)
    T_r     = jnp.maximum(y[1], 250.0)
    T_j     = y[2]
    I_T     = y[3]
    R_state = jnp.clip(y[4], ctrl[14], ctrl[15])   # R_min .. R_max
    V_state = jnp.clip(y[5], ctrl[24], ctrl[25])   # V_norm_min .. V_norm_max
    I_R     = y[6]
    I_V     = y[7]

    # Column at actual R_state and V_state
    (
        x_D, x_B, d_frac, F_total, F_R, _F_B,
        _F_R_norm, _F_B_norm, T_reb, _Q_reb,
    ) = _col_metrics(z_A, eta_col, xi_reb, R_state, V_state)

    # Loop 2/3 commands
    R_cmd, V_cmd, dI_R, dI_V, _e_R, _e_V = compute_reflux_boilup(
        x_D, x_B, F_R, T_reb, I_R, I_V, ctrl
    )
    dR_state = (R_cmd - R_state) / ctrl[16]   # first-order actuator
    dV_state = (V_cmd - V_state) / ctrl[26]

    # Reactor feed
    z_A0   = jnp.clip(z_A0_eff, 0.01, 0.999)
    z_A_in = jnp.clip((F0 * z_A0 + F_R * x_D) / F_total, 1e-6, 1.0 - 1e-6)
    T_in_mix = (F0 * T_in + F_R * T_r) / F_total

    k_eff = jnp.maximum(alpha * K0 * jnp.exp(-EA / (R_GAS * T_r)), 0.0)

    # Loop 1
    qj_raw = ctrl[3] + ctrl[0] * (T_r - ctrl[2]) + I_T / ctrl[1]
    Q_j    = jnp.clip(qj_raw, ctrl[4], ctrl[5])
    dI_T   = jnp.where((qj_raw > ctrl[4]) & (qj_raw < ctrl[5]), T_r - ctrl[2], 0.0)

    UA_eff     = beta_r * UA_NOM
    Q_transfer = UA_eff * (T_r - T_j)

    dz_A = (F_total / MR_NOM) * (z_A_in - z_A) - k_eff * z_A
    dT_r = (
        (F_total / MR_NOM) * (T_in_mix - T_r)
        + (-DH_RXN) * k_eff * z_A / CP_MOLAR
        - Q_transfer / (MR_NOM * CP_MOLAR)
    )
    # FIXED (2026-08-12, reviewer_response_plan.md Major Comment 6, Finding 3 /
    # pending_manuscript_fixes.md Stage 2): beta_r must not scale the actively-
    # commanded duty Q_j, only the conductive term Q_transfer (already beta_r-
    # scaled via UA_eff above) -- jacket fouling attenuates heat transfer through
    # the wall, not the controller's commanded duty. The previous
    # `(Q_transfer - beta_r * Q_j)` form double-counted fouling on Q_j with no
    # physical justification found anywhere in the text or code comments, and
    # was inconsistent with the sibling (unused-by-the-paper) implementation in
    # `cstr_sbi.luyben.physics`, which never scales Q_j this way. This changes
    # System II's reactor-jacket ODE: all cached System II training banks,
    # trained posteriors (sbi-logs/*.pkl), and downstream results computed
    # before this fix are stale and must be regenerated in Stage 3's matched-
    # protocol retraining, not reused as-is.
    dT_j = (Q_transfer - Q_j) / MJ_CPJ

    return jnp.array([dz_A, dT_r, dT_j, dI_T, dR_state, dV_state, dI_R, dI_V])


# ---------------------------------------------------------------------------
# Observation extractors
# ---------------------------------------------------------------------------

def extract_observations_explicit(
    ys: jnp.ndarray,
    theta: jnp.ndarray,
    ctrl: jnp.ndarray,
) -> jnp.ndarray:
    """Extract 12 observable channels from an 8-state explicit-loop trajectory.

    Returns (n_t, 12):
    [T_r, T_j, Q_j, x_D, x_B, T_reb, Q_reb, F_R_norm, F_B_norm, R_norm, V_norm, z_A]

    Channel contract (indices):
        0  T_r       reactor temperature [K]
        1  T_j       jacket temperature [K]
        2  Q_j       jacket duty [Btu/h]
        3  x_D       distillate A mol-frac  (S-A measurement; excluded under S-B)
        4  x_B       bottoms A mol-frac
        5  T_reb     reboiler temperature proxy [K]
        6  Q_reb     reboiler duty proxy [Btu/h]
        7  F_R_norm  recycle flow / F_R_NOM
        8  F_B_norm  product flow / F_B_NOM
        9  R_norm    reflux ratio / REFLUX_RATIO
        10 V_norm    normalised boilup
        11 z_A       reactor A mol-frac (diagnostic; not a primary SBI input)
    """
    eta_col = theta[2]
    xi_reb  = theta[3]

    z_A_arr = ys[:, 0]
    T_r_arr = ys[:, 1]
    T_j_arr = ys[:, 2]
    I_T_arr = ys[:, 3]
    R_arr   = ys[:, 4]
    V_arr   = ys[:, 5]

    Q_j_arr = jax.vmap(compute_qj, in_axes=(0, 0, None))(T_r_arr, I_T_arr, ctrl)

    col_out = jax.vmap(
        lambda zf, rr, vv: _col_metrics(zf, eta_col, xi_reb, rr, vv)
    )(z_A_arr, R_arr, V_arr)

    x_D_arr   = col_out[0]
    x_B_arr   = col_out[1]
    F_R_norm  = col_out[6]
    F_B_norm  = col_out[7]
    T_reb_arr = col_out[8]
    Q_reb_arr = col_out[9]
    R_norm    = R_arr / REFLUX_RATIO
    V_norm    = V_arr

    return jnp.stack(
        [T_r_arr, T_j_arr, Q_j_arr, x_D_arr, x_B_arr,
         T_reb_arr, Q_reb_arr, F_R_norm, F_B_norm, R_norm, V_norm, z_A_arr],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Diffrax integrators
# ---------------------------------------------------------------------------

def _diffrax_solve(rhs, y0, args, t_final, dt0=0.001, rtol=1e-6, atol=1e-8,
                   max_steps=2_000_000, saveat=None):
    term    = diffrax.ODETerm(rhs)
    solver  = diffrax.Tsit5()
    ctrl    = diffrax.PIDController(rtol=rtol, atol=atol)
    if saveat is None:
        saveat = diffrax.SaveAt(t1=True)
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_final, dt0=dt0, y0=y0, args=args,
        stepsize_controller=ctrl, saveat=saveat, max_steps=max_steps,
        throw=False,
    )
    return sol


def simulate_to_steady_state(
    theta, inlet, ctrl=None, y0=None,
    t_final=200.0, rtol=1e-6, atol=1e-8, max_steps=2_000_000,
):
    """Integrate 4-state model to steady state."""
    ctrl = NOMINAL_CTRL    if ctrl is None else ctrl
    y0   = NOMINAL_Y0      if y0   is None else y0
    sol  = _diffrax_solve(recycle_rhs, y0, (theta, NOMINAL_INLET, ctrl),
                          t_final, rtol=rtol, atol=atol, max_steps=max_steps)
    return sol.ys[-1]


def simulate_trajectory(
    theta, inlet=None, ctrl=None, y0=None,
    t_final=50.0, n_save=201, dt0=0.001, rtol=1e-6, atol=1e-8, max_steps=2_000_000,
):
    """Integrate 4-state model at fixed save times."""
    inlet = NOMINAL_INLET  if inlet is None else inlet
    ctrl  = NOMINAL_CTRL   if ctrl  is None else ctrl
    y0    = NOMINAL_Y0     if y0    is None else y0
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_final, n_save))
    sol = _diffrax_solve(recycle_rhs, y0, (theta, inlet, ctrl),
                         t_final, dt0=dt0, rtol=rtol, atol=atol,
                         max_steps=max_steps, saveat=saveat)
    return sol.ts, sol.ys


def simulate_to_steady_state_explicit(
    theta, inlet=None, ctrl=None, y0=None,
    t_final=200.0, rtol=1e-6, atol=1e-8, max_steps=2_000_000,
):
    """Integrate 8-state explicit-loop model to steady state."""
    inlet = NOMINAL_INLET      if inlet is None else inlet
    ctrl  = NOMINAL_CTRL_SB    if ctrl  is None else ctrl
    y0    = NOMINAL_Y0_EXPLICIT if y0   is None else y0
    sol = _diffrax_solve(recycle_rhs_explicit, y0, (theta, inlet, ctrl),
                         t_final, rtol=rtol, atol=atol, max_steps=max_steps)
    return sol.ys[-1]


def simulate_trajectory_explicit(
    theta, inlet=None, ctrl=None, y0=None,
    t_final=2.0, n_save=120, dt0=0.001, rtol=1e-6, atol=1e-8, max_steps=2_000_000,
):
    """Integrate 8-state explicit-loop model at fixed save times."""
    inlet  = NOMINAL_INLET       if inlet is None else inlet
    ctrl   = NOMINAL_CTRL_SB     if ctrl  is None else ctrl
    y0     = NOMINAL_Y0_EXPLICIT if y0    is None else y0
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_final, n_save))
    sol = _diffrax_solve(recycle_rhs_explicit, y0, (theta, inlet, ctrl),
                         t_final, dt0=dt0, rtol=rtol, atol=atol,
                         max_steps=max_steps, saveat=saveat)
    return sol.ts, sol.ys


# ---------------------------------------------------------------------------
# JIT / vmap variants
# ---------------------------------------------------------------------------

simulate_to_ss_jit = jax.jit(simulate_to_steady_state)
simulate_to_ss_explicit_jit = jax.jit(simulate_to_steady_state_explicit)
simulate_trajectory_jit = jax.jit(simulate_trajectory, static_argnames=("n_save",))
simulate_trajectory_explicit_jit = jax.jit(
    simulate_trajectory_explicit, static_argnames=("n_save",)
)
column_qss_jit = jax.jit(column_qss)
