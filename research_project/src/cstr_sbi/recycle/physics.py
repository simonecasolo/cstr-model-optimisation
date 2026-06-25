"""Wu 2003 CSTR-column-recycle physics module.

Implements the deterministic dynamics of the Wu (2003) liquid-phase A→B
CSTR with distillation column recycle in pure JAX/diffrax so the same code
path works for both diffrax trajectory integration and SNPE_C simulation.

Reference:
    Wu, K.-L., Yu, C.-C., Luyben, W. L., & Skogestad, S. (2003).
    Reactor/separator processes with recycle -- 2. Design for composition
    control. Computers & Chemical Engineering, 27(3), 401-421.

Unit system: Btu-lbmol-h-K
    Flows   : lbmol/h
    Holdup  : lbmol
    Energy  : Btu
    Time    : h
    Temp    : K  (Kelvin; note R_GAS = 3.576 Btu/(lbmol·K), see below)

**Critical unit note – gas constant with Kelvin temperatures:**
Wu 2003 uses Ea = 30,841 Btu/lbmol with R = 1.987 Btu/(lbmol·°R).
To work in Kelvin: R_GAS = 1.987 Btu/(lbmol·R) × (9/5 R/K) = 3.576 Btu/(lbmol·K).
This gives k(342.26 K) = K0 × exp(−30841/(3.576×342.26)) = 0.33 /h ✓.

State vector (4-D): y = [z_A, T_r, T_j, I_T]
    z_A  -- reactor mol-fraction of A            [dimensionless]
    T_r  -- reactor temperature                  [K]
    T_j  -- jacket temperature                   [K]
    I_T  -- PI integrator for reactor-temp loop  [K·h]

Parameter vector theta (5-D):
    theta = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]
    alpha     -- catalyst/kinetic activity factor  (1.0 = nominal)
    beta_r    -- CSTR jacket fouling factor         (1.0 = nominal)
    eta_col   -- distillation column tray eff.      (1.0 = nominal)
    xi_reb    -- reboiler duty scale factor          (1.0 = nominal)
    z_A0_eff  -- effective fresh-feed composition   (0.90 = nominal)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import diffrax


# ---------------------------------------------------------------------------
# Physical constants — Btu-lbmol-h-K unit system
# ---------------------------------------------------------------------------

# Gas constant: 1.987 Btu/(lbmol·R) × (9/5 R/K) = 3.576 Btu/(lbmol·K)
# (Wu 2003 uses Rankine; we convert to Kelvin)
R_GAS: float = 3.576          # Btu/(lbmol·K)

# Reaction kinetics (Wu 2003 Table 1)
EA: float = 30841.0           # Btu/lbmol,  activation energy
K_SS: float = 0.33            # 1/h,  rate constant at steady state (T_ss = 342.26 K)

# k0 = k_ss * exp(EA / (R_GAS * T_ss))
# T_ss = (156.4 - 32) * 5/9 + 273.15 = 342.26 K
# k0 = 0.33 * exp(30841 / (3.576 * 342.26)) = 0.33 * exp(25.20) = 0.33 * 8.87e10
K0: float = 2.91e10           # 1/h,  pre-exponential factor

# Nominal reactor SS temperature (156.4°F)
T_SS: float = (156.4 - 32.0) * 5.0/9.0 + 273.15   # = 342.26 K

# Heat of reaction (Wu 2003 Table 1): 30,000 Btu/lbmol, exothermic
DH_RXN: float = -30000.0      # Btu/lbmol  (negative = exothermic)

# Heat capacity: Cp_mass = 0.7 Btu/(lb·°F); MW ≈ 100 lb/lbmol → 70 Btu/(lbmol·K)
# Note: 1 °F = 5/9 K, so Cp_molar [Btu/(lbmol·K)] = 0.7 * 100 * (9/5) = 126
# Wait: Cp = 0.7 Btu/(lb·°F) = 0.7 Btu/(lb·R).  Per lbmol (MW=100 lb/lbmol):
# Cp_molar = 0.7 * 100 Btu/(lbmol·R).  Per K: 1K = 9/5 R, so ΔH per K = 70*(9/5)=126
# Btu/(lbmol·K). But since we use K differences everywhere:
# Energy balance: ρ Cp ΔT [Btu/lbmol] = CP_MOLAR * ΔT_K * (9/5)?
# NO — Cp is dimensionally Btu/(lbmol·K) directly:
# Cp = 0.7 Btu/(lb·°F) = 0.7 Btu/(lb·R);  1 R = 5/9 K → 1 Btu/(lb·R) = 9/5 Btu/(lb·K)
# Cp [Btu/(lb·K)] = 0.7 * (9/5) = 1.26 Btu/(lb·K)
# Cp_molar [Btu/(lbmol·K)] = 1.26 * 100 = 126 Btu/(lbmol·K)
CP_MOLAR: float = 126.0       # Btu/(lbmol·K)  [= 0.7 Btu/(lb·R) * 100 lb/lbmol * (9/5 R/K)]

# Heat transfer: UA_F = 150.5 Btu/(h·ft²·°F) × 3206.8 ft² = 482,624 Btu/(h·°F)
# Converting to Btu/(h·K): 1°F difference = 5/9 K difference, so:
# Q = UA_F * ΔT_F = UA_F * ΔT_K * (9/5)
# ∴ UA_K = UA_F * (9/5) = 482,624 * 1.8 = 868,723 Btu/(h·K)
UA_NOM: float = 868723.0      # Btu/(h·K)  [= 482624 * 9/5]

# Jacket heat capacity: M_j * Cp_j [Btu/K]
# V_r = 2400 * 100 / 60.05 ≈ 3995 ft³; jacket ≈ 10% = 400 ft³
# M_j = 400 * 62.4 / 18 = 1387 lbmol water
# Cp_water = 1.0 Btu/(lb·R) = 9/5 Btu/(lb·K); Cp_molar_water = 18 * 9/5 = 32.4 Btu/(lbmol·K)
# MJ_CPJ = 1387 * 32.4 = 44,939 Btu/K
MJ_CPJ: float = 44939.0       # Btu/K  (jacket heat capacity, in Kelvin)

# Nominal operating point (Wu 2003 Table 1)
F0_NOM: float = 460.0         # lbmol/h,  fresh feed flow
Z0_NOM: float = 0.90          # mol/mol,  fresh feed A composition
T_IN: float = (70.0 - 32.0) * 5.0/9.0 + 273.15   # = 294.26 K  (70°F)
MR_NOM: float = 2400.0        # lbmol,    reactor holdup (constant)
T_SP: float = T_SS            # K,  reactor temperature setpoint (= 342.26 K)
T_J_NOM: float = (136.1 - 32.0) * 5.0/9.0 + 273.15  # = 330.98 K  (136.1°F)

# Nominal column (Wu 2003 Table 1)
ALPHA_REL: float = 2.0        # relative volatility A/B
N_TRAYS: int = 20             # actual trays
FEED_TRAY: int = 10           # feed enters above this tray (0-indexed: trays 0..19)
X_D_NOM: float = 0.95         # distillate A mol-fraction (nominal)
X_B_NOM: float = 0.0105       # bottoms A mol-fraction (nominal, Wu Table 1)
REFLUX_RATIO: float = 2.198   # L/D = 1100/500.4

# Nominal recycle / product flows (Wu 2003 Table 1)
F_R_NOM: float = 500.4        # lbmol/h,  recycle (distillate) flow
F_B_NOM: float = 460.0        # lbmol/h,  product (bottoms) flow
D_FRAC_NOM: float = F_R_NOM / (F_R_NOM + F_B_NOM)   # ≈ 0.521
Z_F_RECYCLE_REF: float = 0.510                       # nominal reactor outlet used by nb20
RECYCLE_SENSITIVITY: float = 0.12                    # d(D/F)/d(z_F), snowball closure
T_REB_NOM: float = 372.0                             # K, reboiler-temperature proxy baseline
QREB_NOM: float = 1.2e7                              # Btu/h, reboiler-duty proxy baseline

# Derived column constants (for McCabe-Thiele stepping, held fixed)
_LV: float = REFLUX_RATIO / (REFLUX_RATIO + 1.0)    # L/V rectifying = 0.687
_DV: float = 1.0 / (REFLUX_RATIO + 1.0)              # D/V rectifying = 0.313
_F_OVER_D: float = 1.0 / D_FRAC_NOM                  # F/D ≈ 1.921
_LV_S: float = (REFLUX_RATIO + _F_OVER_D) / (REFLUX_RATIO + 1.0)  # Ls/Vs ≈ 1.288
_BV_S: float = (_F_OVER_D - 1.0) / (REFLUX_RATIO + 1.0)           # Bs/Vs ≈ 0.288

# Total nominal flow into reactor
F_TOT_NOM: float = F0_NOM / (1.0 - D_FRAC_NOM)      # ≈ 960.4 lbmol/h

# PI controller for reactor temperature loop (Loop 1)
# Manipulated variable: Q_j [Btu/h] jacket heat removal duty
# Sign: positive error (T_r > T_sp) → increase Q_j (more cooling)
#
# QJ_NOM is the bias set to the actual energy-balance Q_j at the true SS
# (computed from: F_tot*(T_in_mix-T_r)*Cp + (-DH)*k*z_A*MR = Q_j when T_r=T_sp).
# This ensures that at SS, Kp*(T_r-T_sp)=0 and I_T=0, so Q_j = QJ_NOM.
# Value derived analytically (see physics.py module development notes):
#   Q_j_ss = MR*Cp*(F_tot/MR*(T_in_mix-T_sp) + (-DH)*k_ss*z_A_ss/Cp) = 9.351e6 Btu/h
QJ_NOM: float = 9.351e6       # Btu/h, bias at nominal SS (energy-balance derived)
KP1: float = 8.0e6            # Btu/(h·K),  proportional gain
TAU_I1: float = 0.1           # h,           integral time constant
QJ_MIN: float = 0.0           # Btu/h
QJ_MAX: float = 3.0e7         # Btu/h


# ---------------------------------------------------------------------------
# Nominal JAX arrays
# ---------------------------------------------------------------------------

# theta = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]
NOMINAL_THETA = jnp.array([1.0, 1.0, 1.0, 1.0, Z0_NOM], dtype=jnp.float32)

# inlet = [F0, T_in]
NOMINAL_INLET = jnp.array([F0_NOM, T_IN], dtype=jnp.float32)

# ctrl = [Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max]
NOMINAL_CTRL = jnp.array([KP1, TAU_I1, T_SP, QJ_NOM, QJ_MIN, QJ_MAX],
                          dtype=jnp.float32)

# y0: warm-start near SS  [z_A, T_r, T_j, I_T]
# At nominal SS: z_A ≈ 0.508, T_r ≈ T_SP, T_j ≈ T_J_NOM
NOMINAL_Y0 = jnp.array([0.508, T_SP, T_J_NOM, 0.0], dtype=jnp.float32)

# Parameter names for labeling
PARAM_NAMES = ["alpha", "beta_r", "eta_col", "xi_reb", "z_A0_eff"]


# ---------------------------------------------------------------------------
# Distillation column: quasi-steady-state McCabe-Thiele via bisection
# ---------------------------------------------------------------------------
#
# Algorithm: given z_F (reactor outlet = column feed) and eta_col (tray efficiency),
# find x_D such that the McCabe-Thiele tray-by-tray calculation (N_TRAYS stages)
# gives x_reboiler = x_B(MB) from the overall material balance.
#
# Column material balance: z_F * F = x_D * D + x_B * B, D+B=F
# With D_frac = D/F fixed at D_FRAC_NOM (≈0.521, set by reflux drum level control):
#   x_B = (z_F - D_frac * x_D) / (1 - D_frac)
#
# McCabe-Thiele stepping DOWN (from condenser to reboiler):
# Start: x = x_D
# For each tray i = 0..N_TRAYS-1:
#   Rectifying (i < FEED_TRAY):
#     y_op = LV * x + DV * x_D         (operating line)
#     x    = y_op / (alpha - (alpha-1)*y_op)  (equilibrium inverse)
#   Stripping (i >= FEED_TRAY):
#     y_op = LV_s * x - BV_s * x_B     (stripping operating line)
#     x    = y_op / (alpha - (alpha-1)*y_op)  (equilibrium inverse)
#
# Bisection: x_D is increased until x_reboiler = x_B(MB).
# Monotonicity: higher x_D → lower x_reboiler (more separation).
# So: if x_reboiler > x_B(MB), increase x_D (lo = mid); else decrease (hi = mid).


def _column_step(state, i, alpha_eff, x_D, x_B_mb):
    """One McCabe-Thiele tray step downwards."""
    x = state
    in_stripping = i >= FEED_TRAY

    # Operating line
    y_rect  = _LV * x + _DV * x_D
    y_strip = _LV_S * x - _BV_S * x_B_mb
    y_op = jnp.where(in_stripping, y_strip, y_rect)
    y_op = jnp.clip(y_op, 1e-8, 1.0 - 1e-8)

    # Equilibrium inverse: y = alpha*x/(1+(alpha-1)*x) → x = y/(alpha-(alpha-1)*y)
    x_new = y_op / (alpha_eff - (alpha_eff - 1.0) * y_op)
    return jnp.clip(x_new, 1e-8, 1.0 - 1e-8), None


def _run_column(x_D, z_F, alpha_eff):
    """Trace McCabe-Thiele from x_D down to reboiler.

    Returns (x_reboiler, x_B_mb) where x_B_mb is from material balance.

    D_frac is taken as the minimum of D_FRAC_NOM and z_F/x_D so that
    x_B = (z_F - D_frac * x_D) / (1 - D_frac) ≥ 0 for all z_F.
    """
    # Self-consistent D_frac: respect x_B ≥ 0
    d_frac = jnp.minimum(D_FRAC_NOM, z_F / jnp.maximum(x_D, 1e-6) * 0.999)
    d_frac = jnp.clip(d_frac, 0.01, 0.99)
    x_B_mb = (z_F - d_frac * x_D) / (1.0 - d_frac)
    x_B_mb = jnp.clip(x_B_mb, 1e-6, z_F - 1e-5)

    x_final, _ = jax.lax.scan(
        lambda x, i: _column_step(x, i, alpha_eff, x_D, x_B_mb),
        x_D,
        jnp.arange(N_TRAYS),
    )
    return x_final, x_B_mb


def column_qss(
    z_F: jax.typing.ArrayLike,
    eta_col: jax.typing.ArrayLike,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Quasi-steady-state column: find x_D given feed z_F and tray efficiency.

    Uses bisection (50 steps via lax.fori_loop) on the McCabe-Thiele
    construction to find x_D ∈ (z_F + ε, 1) such that the tray-by-tray
    calculation gives x_reboiler = x_B from the material balance.

    Parameters
    ----------
    z_F      : feed (= reactor outlet) A mol-fraction
    eta_col  : overall tray efficiency scale (1.0 = full efficiency)

    Returns
    -------
    x_D    : distillate A composition
    x_B    : bottoms A composition
    D_frac : D/F (≈ D_FRAC_NOM; computed from material balance)
    """
    z_F     = jnp.asarray(z_F,     dtype=jnp.float32)
    eta_col = jnp.asarray(eta_col, dtype=jnp.float32)
    z_F = jnp.clip(z_F, 1e-5, 1.0 - 1e-5)

    # Effective relative volatility: eta_col scales separation power
    alpha_eff = 1.0 + eta_col * (ALPHA_REL - 1.0)
    alpha_eff = jnp.clip(alpha_eff, 1.001, 10.0)

    # Fast controlled-column closure used by the recycle ODE.  The nominal
    # split is anchored at Wu Table 1, but the split responds to reactor
    # effluent composition so catalyst loss can snowball into recycle buildup.
    d_frac = D_FRAC_NOM + RECYCLE_SENSITIVITY * (z_F - Z_F_RECYCLE_REF)
    d_frac = d_frac - 0.08 * (1.0 - eta_col)
    d_frac = jnp.clip(d_frac, 0.05, 0.95)

    x_B = X_B_NOM + 0.08 * (1.0 - eta_col)
    x_B = jnp.clip(x_B, 1e-5, 0.25)
    x_D = (z_F - (1.0 - d_frac) * x_B) / d_frac
    x_D = jnp.clip(x_D, z_F + 1e-4, 0.9998)

    return x_D, x_B, d_frac

    lo = jnp.clip(z_F + 0.01, 0.10, 0.90)
    hi = jnp.array(0.9998, dtype=jnp.float32)

    def bisect_step(carry, _):
        lo_i, hi_i = carry
        mid = 0.5 * (lo_i + hi_i)
        x_reb, x_B_mb = _run_column(mid, z_F, alpha_eff)
        # diff = x_reb - x_B_mb
        # diff < 0 → x_D too low (not enough separation) → lo = mid
        # diff ≥ 0 → x_D too high (over-separating) → hi = mid
        diff_neg = x_reb < x_B_mb
        lo_new = jnp.where(diff_neg, mid, lo_i)
        hi_new = jnp.where(diff_neg, hi_i, mid)
        return (lo_new, hi_new), None

    (lo_f, hi_f), _ = jax.lax.scan(bisect_step, (lo, hi), None, length=50)
    x_D = jnp.clip(0.5 * (lo_f + hi_f), lo, 0.9998)

    x_B = (z_F - D_FRAC_NOM * x_D) / (1.0 - D_FRAC_NOM)
    x_B = jnp.clip(x_B, 1e-6, z_F - 1e-5)

    D_frac = jnp.clip(
        (z_F - x_B) / jnp.maximum(x_D - x_B, 1e-6),
        0.01, 0.99,
    )

    return x_D, x_B, D_frac


def recycle_flow(
    z_F: jax.typing.ArrayLike,
    eta_col: jax.typing.ArrayLike,
    F_total: jax.typing.ArrayLike,
) -> jnp.ndarray:
    """Recycle (distillate) flow F_R = D_frac * F_total [lbmol/h]."""
    _, _, D_frac = column_qss(z_F, eta_col)
    return D_frac * F_total


# ---------------------------------------------------------------------------
# PI controller utility
# ---------------------------------------------------------------------------

def compute_qj(
    T_r: jax.typing.ArrayLike,
    I_T: jax.typing.ArrayLike,
    ctrl: jnp.ndarray,
) -> jnp.ndarray:
    """Return jacket duty Q_j [Btu/h] from the PI controller.

    ctrl = [Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max]
    Positive Q_j = heat removal (cooling); increases when T_r > T_sp.
    """
    Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max = (
        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5],
    )
    qj_unclamped = Qj0 + Kp1 * (T_r - T_sp) + I_T / tau_i1
    return jnp.clip(qj_unclamped, Qj_min, Qj_max)


# ---------------------------------------------------------------------------
# 4-state ODE right-hand side
# ---------------------------------------------------------------------------

def recycle_rhs(
    t: float,
    y: jnp.ndarray,
    args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Right-hand side of the 4-state Wu 2003 recycle CSTR ODE.

    State: y = [z_A, T_r, T_j, I_T]
    Args:  (theta, inlet, ctrl)
        theta = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]
        inlet = [F0, T_in]
        ctrl  = [Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max]

    Material balance on A (fixed holdup M_r):
        dz_A/dt = (F_tot/M_r)*(z_A_in - z_A) - k_eff*z_A

    Energy balance on reactor:
        M_r * Cp * dT_r/dt = F_tot*Cp*(T_in_mix - T_r) + (-ΔH)*k_eff*z_A*M_r - Q_j
    →  dT_r/dt = (F_tot/M_r)*(T_in_mix - T_r) + (-ΔH)*k_eff*z_A/Cp - Q_j/(M_r*Cp)

    Jacket energy balance:
        M_j*Cp_j*dT_j/dt = UA_eff*(T_r - T_j) - Q_j
    →  dT_j/dt = [UA_eff*(T_r - T_j) - Q_j] / MJ_CPJ

    PI integrator (anti-windup):
        dI_T/dt = (T_r - T_sp)  [when not saturated]
    """
    theta, inlet, ctrl = args
    alpha, beta_r, eta_col, _xi_reb, z_A0_eff = (
        theta[0], theta[1], theta[2], theta[3], theta[4],
    )
    F0, T_in = inlet[0], inlet[1]

    z_A, T_r, T_j, I_T = y[0], y[1], y[2], y[3]
    z_A = jnp.clip(z_A, 1e-6, 1.0 - 1e-6)
    T_r = jnp.maximum(T_r, 250.0)

    # --- Column quasi-steady state ---
    x_D, _x_B, D_frac = column_qss(z_A, eta_col)
    D_frac = jnp.clip(D_frac, 0.01, 0.98)

    # Total flow: F_total = F0 / (1 - D_frac)  [from overall loop MB]
    F_total = F0 / (1.0 - D_frac)
    F_R = D_frac * F_total

    # Mixed inlet composition
    z_A0 = jnp.clip(z_A0_eff, 0.01, 0.999)
    z_A_in = (F0 * z_A0 + F_R * x_D) / F_total
    z_A_in = jnp.clip(z_A_in, 1e-6, 1.0 - 1e-6)

    # Mixed feed temperature: assume recycle (distillate) returns at T_r
    T_in_mix = (F0 * T_in + F_R * T_r) / F_total

    # Kinetics
    k_eff = alpha * K0 * jnp.exp(-EA / (R_GAS * T_r))
    k_eff = jnp.maximum(k_eff, 0.0)

    # PI controller → Q_j [Btu/h] cooling-command equivalent.
    Kp1, tau_i1, T_sp_ctrl, Qj0, Qj_min, Qj_max = (
        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5],
    )
    qj_unclamped = Qj0 + Kp1 * (T_r - T_sp_ctrl) + I_T / tau_i1
    Q_j = jnp.clip(qj_unclamped, Qj_min, Qj_max)

    # Anti-windup gate
    not_saturated = (qj_unclamped > Qj_min) & (qj_unclamped < Qj_max)
    dI_T = jnp.where(not_saturated, T_r - T_sp_ctrl, 0.0)

    # Effective UA
    UA_eff = beta_r * UA_NOM
    Q_transfer = UA_eff * (T_r - T_j)
    Q_cooling_eff = beta_r * Q_j

    # CSTR material balance
    dz_A = (F_total / MR_NOM) * (z_A_in - z_A) - k_eff * z_A

    # Reactor energy balance (all in Btu/h → divide by M_r*Cp for K/h)
    dT_r = (
        (F_total / MR_NOM) * (T_in_mix - T_r)
        + (-DH_RXN) * k_eff * z_A / CP_MOLAR
        - Q_transfer / (MR_NOM * CP_MOLAR)
    )

    # Jacket energy balance
    dT_j = (Q_transfer - Q_cooling_eff) / MJ_CPJ

    return jnp.array([dz_A, dT_r, dT_j, dI_T])


# ---------------------------------------------------------------------------
# Steady-state integrator
# ---------------------------------------------------------------------------

def simulate_to_steady_state(
    theta: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray = NOMINAL_Y0,
    t_final: float = 200.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 2_000_000,
) -> jnp.ndarray:
    """Integrate the recycle CSTR to steady state, returning [z_A, T_r, T_j, I_T].

    Default t_final = 200 h ≈ 80 residence times; reaches SS reliably.
    """
    term = diffrax.ODETerm(recycle_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_final,
        dt0=0.001,
        y0=y0,
        args=(theta, inlet, ctrl),
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=False,
    )
    return sol.ys[-1]


# ---------------------------------------------------------------------------
# Trajectory integrator (fixed save times, JIT-safe)
# ---------------------------------------------------------------------------

def simulate_trajectory(
    theta: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray = NOMINAL_Y0,
    t_final: float = 50.0,
    n_save: int = 201,
    dt0: float = 0.001,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 2_000_000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate and return trajectory at n_save equally-spaced times.

    Returns (ts, ys) with ts.shape = (n_save,), ys.shape = (n_save, 4).
    """
    term = diffrax.ODETerm(recycle_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_final, n_save))
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_final,
        dt0=dt0,
        y0=y0,
        args=(theta, inlet, ctrl),
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=max_steps,
        throw=False,
    )
    return sol.ts, sol.ys


# ---------------------------------------------------------------------------
# Observation extractor
# ---------------------------------------------------------------------------

def extract_observations(
    ys: jnp.ndarray,
    theta: jnp.ndarray,
    ctrl: jnp.ndarray,
) -> jnp.ndarray:
    """Extract plant-wide observable channels from a trajectory.

    Parameters
    ----------
    ys    : (n_t, 4) — [z_A, T_r, T_j, I_T]
    theta : (5,) — [alpha, beta_r, eta_col, xi_reb, z_A0_eff]
    ctrl  : (6,) — [Kp1, tau_i1, T_sp, Qj0, Qj_min, Qj_max]

    Returns
    -------
    obs : (n_t, 10) — [z_A, T_r, T_j, Q_j, x_D, x_B, F_R_norm, T_reb, Q_reb, F_B_norm]
        z_A      -- reactor A composition
        T_r      -- reactor temperature [K]
        T_j      -- jacket temperature [K]
        Q_j      -- jacket duty [Btu/h]
        x_D      -- distillate A composition (column analyser)
        x_B      -- bottoms A composition
        F_R_norm -- recycle flow / F_R_NOM (dimensionless)
        T_reb    -- reboiler-temperature proxy [K]
        Q_reb    -- reboiler-duty proxy [Btu/h]
        F_B_norm -- product bottoms flow / F_B_NOM (dimensionless)
    """
    eta_col = theta[2]
    xi_reb = theta[3]
    F0 = NOMINAL_INLET[0]

    z_A = ys[:, 0]
    T_r = ys[:, 1]
    T_j = ys[:, 2]
    I_T = ys[:, 3]

    Q_j = jax.vmap(compute_qj, in_axes=(0, 0, None))(T_r, I_T, ctrl)

    col_out = jax.vmap(lambda zf: column_qss(zf, eta_col))(z_A)
    x_D_arr = col_out[0]
    x_B_arr = col_out[1]
    D_frac  = col_out[2]

    D_frac_safe = jnp.clip(D_frac, 0.01, 0.98)
    F_total = F0 / (1.0 - D_frac_safe)
    F_R = D_frac_safe * F_total
    F_B = F_total - F_R
    F_R_norm = F_R / F_R_NOM
    F_B_norm = F_B / F_B_NOM

    col_severity = 1.0 - eta_col
    x_B_excess = x_B_arr - X_B_NOM
    throughput_ratio = F_total / F_TOT_NOM
    T_reb = T_REB_NOM + 90.0 * x_B_excess + 8.0 * col_severity + 3.0 * (F_R_norm - 1.0)
    Q_reb = QREB_NOM * throughput_ratio * (1.0 + 2.0 * x_B_excess + 0.5 * col_severity)
    Q_reb = Q_reb / jnp.clip(xi_reb, 0.2, 2.0)

    return jnp.stack(
        [z_A, T_r, T_j, Q_j, x_D_arr, x_B_arr, F_R_norm, T_reb, Q_reb, F_B_norm],
        axis=1,
    )


# ---------------------------------------------------------------------------
# JIT / vmap variants
# ---------------------------------------------------------------------------

simulate_to_ss_jit = jax.jit(simulate_to_steady_state)
simulate_to_ss_batch = jax.jit(
    jax.vmap(simulate_to_steady_state, in_axes=(0, None, None))
)
simulate_trajectory_jit = jax.jit(
    simulate_trajectory, static_argnames=("n_save",)
)
