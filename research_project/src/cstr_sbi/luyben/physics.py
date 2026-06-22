"""Luyben recycle plant ODE right-hand sides.

13-state closed-loop model:
    [Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s, I_T, I_Ts, I_L, I_R, I_P]

8 degradation parameters:
    theta = [alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta]

5 decentralized PI loops:
    Loop 1: Qc   -> T_r   (CSTR temperature)
    Loop 2: Q_s  -> T_s   (separator temperature)
    Loop 3: F_L  -> n_L   (separator liquid level)
    Loop 4: F_R  -> F_R   (recycle flow, valve position)
    Loop 5: F_P  -> x_purge (purge flow)

Reference: Luyben (1994) "Snowball effects in reactor/separator processes
with recycle", I&EC Research 33(2):299-305. Parameter values from
Luyben (1994) Table 1 and Luyben (2002) generic A+B->C benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import diffrax


# ---------------------------------------------------------------------------
# Physical constants (Luyben 1994 / 2002)
# ---------------------------------------------------------------------------

# Reaction A + B -> C, exothermic, bimolecular, irreversible
K0_R: float = 1.0e10          # L/(mol·min), pre-exponential factor
E_A_R: float = 69_000.0       # J/mol, activation energy
H_R_R: float = -41_800.0      # J/mol, heat of reaction (exothermic)
R_GAS: float = 8.314           # J/(mol·K)

# Reactor fluid
RHO_R: float = 800.0           # g/L
CP_R: float = 2.0              # J/(g·K)

# Coolant
RHO_C: float = 1000.0          # g/L
CP_C: float = 4.184            # J/(g·K)

# Reactor geometry
V_R: float = 100.0             # L, reactor volume
V_C_R: float = 20.0            # L, jacket volume

# Nominal feed conditions
F_A0: float = 9.0              # L/min, fresh A feed
F_B0: float = 9.0              # L/min, fresh B feed
CA_IN: float = 1.0             # mol/L, inlet A concentration
CB_IN: float = 1.0             # mol/L, inlet B concentration
T_IN: float = 320.0            # K, feed temperature
T_CI: float = 295.0            # K, coolant inlet temperature

# Nominal heat transfer coefficients
UA_R_NOM: float = 1_200.0      # J/(min·K), CSTR jacket
UA_S_NOM: float = 800.0        # J/(min·K), separator condenser

# Flash separator (Raoult-like VLE)
# Relative volatilities: A lightest, C heaviest (desired product liquid)
ALPHA_VLE_A: float = 3.0       # relative volatility of A
ALPHA_VLE_B: float = 2.0       # relative volatility of B
ALPHA_VLE_C: float = 1.0       # relative volatility of C (reference)

# Separator geometry
N_L_NOM: float = 50.0          # mol, nominal liquid holdup
T_S_NOM: float = 340.0         # K, nominal separator temperature
T_S_IN: float = 365.0          # K, separator inlet temperature (reactor outlet)
T_S_REF: float = 290.0         # K, separator coolant inlet

# Recycle / purge
F_R_NOM: float = 40.0          # L/min, nominal recycle flow
F_P_NOM: float = 2.0           # L/min, nominal purge flow
F_L_NOM: float = 18.0          # L/min, nominal liquid product flow
F_PROD_NOM: float = 18.0       # L/min, nominal product flow

# Feed preheater (energy effect captured via kappa on T_IN)
DELTA_T_PREHEAT: float = 10.0  # K, nominal preheat rise (kappa=1 -> T_IN + DELTA_T_PREHEAT)

# ---------------------------------------------------------------------------
# PI controller parameters (5 loops)
# ---------------------------------------------------------------------------

# Loop 1: CSTR temperature (T_r -> Qc)
TSP_R: float = 360.0           # K
KP1: float = 200.0             # J/(min·K·(J/min)) = dimensionless scaling factor
TAU_I1: float = 15.0           # min
QC0_R: float = 600.0           # J/min, nominal cooling duty
QC_MIN_R: float = 0.0
QC_MAX_R: float = 6000.0

# Loop 2: Separator temperature (T_s -> Q_s)
TSP_S: float = 330.0           # K
KP2: float = 150.0
TAU_I2: float = 20.0
QS0: float = 400.0             # J/min
QS_MIN: float = 0.0
QS_MAX: float = 4000.0

# Loop 3: Separator level (n_L -> F_L)
NSP_L: float = N_L_NOM         # mol
KP3: float = 1.5               # (L/min)/mol
TAU_I3: float = 30.0           # min
FL0: float = F_L_NOM
FL_MIN: float = 0.0
FL_MAX: float = 60.0

# Loop 4: Recycle flow (F_R -> F_R via pump speed)
FR_SP: float = F_R_NOM         # L/min
KP4: float = 2.0               # (L/min)/(L/min)
TAU_I4: float = 5.0            # min  (fast loop)
FR0: float = F_R_NOM
FR_MIN: float = 0.0
FR_MAX: float = 120.0

# Loop 5: Purge flow (x_purge -> F_P via valve)
# Controlled variable: mole fraction of A in recycle vapour
X_P_SP: float = 0.35           # target A fraction in purge/recycle
KP5: float = 20.0              # (L/min)/fraction
TAU_I5: float = 60.0           # min  (slowest loop)
FP0: float = F_P_NOM
FP_MIN: float = 0.0
FP_MAX: float = 20.0

# Separator vapour fraction (simple flash approximation)
V_FRAC_NOM: float = 0.6        # nominal vapour fraction at separator outlet


# ---------------------------------------------------------------------------
# Convenience arrays (nominal operating point)
# ---------------------------------------------------------------------------

#: 8-D degradation parameter vector at healthy baseline
NOMINAL_THETA = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)

#: Controller parameter vector packed for jit (see ctrl_arrays below)
NOMINAL_CTRL1 = jnp.array([KP1, TAU_I1, TSP_R,  QC0_R,  QC_MIN_R, QC_MAX_R], dtype=jnp.float32)
NOMINAL_CTRL2 = jnp.array([KP2, TAU_I2, TSP_S,  QS0,    QS_MIN,   QS_MAX],   dtype=jnp.float32)
NOMINAL_CTRL3 = jnp.array([KP3, TAU_I3, NSP_L,  FL0,    FL_MIN,   FL_MAX],   dtype=jnp.float32)
NOMINAL_CTRL4 = jnp.array([KP4, TAU_I4, FR_SP,  FR0,    FR_MIN,   FR_MAX],   dtype=jnp.float32)
NOMINAL_CTRL5 = jnp.array([KP5, TAU_I5, X_P_SP, FP0,    FP_MIN,   FP_MAX],   dtype=jnp.float32)

# Packed: (5, 6) array, one row per loop
NOMINAL_CTRL_ALL = jnp.stack(
    [NOMINAL_CTRL1, NOMINAL_CTRL2, NOMINAL_CTRL3, NOMINAL_CTRL4, NOMINAL_CTRL5],
    axis=0,
)

#: 13-D warm initial condition (near nominal steady state)
#  [Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s, I_T, I_Ts, I_L, I_R, I_P]
NOMINAL_Y0 = jnp.array(
    [0.10, 0.10, TSP_R, 310.0, N_L_NOM, 0.40, 0.30, TSP_S, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=jnp.float32,
)

PARAM_NAMES = ("alpha", "beta_r", "eta_sep", "beta_s", "eta_p", "xi", "kappa", "delta")
STATE_NAMES = ("Ca", "Cb", "T_r", "Tc", "n_L", "x_A", "x_B", "T_s",
               "I_T", "I_Ts", "I_L", "I_R", "I_P")
OBS_NAMES   = ("T_r", "Tc", "Qc", "T_s", "Q_s", "F_R", "F_P", "F_prod")

N_STATES: int = 13
N_PARAMS: int = 8
N_OBS: int = 8


# ---------------------------------------------------------------------------
# Helper: PI controller output (clamped)
# ---------------------------------------------------------------------------

def _pi_output(pv: float, I: float, ctrl: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (clamped_output, not_saturated_flag) for one PI loop.

    ctrl = [Kp, tau_i, sp, bias, min, max]
    """
    Kp, tau_i, sp, bias, lo, hi = ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5]
    unclamped = bias + Kp * (pv - sp) + I / tau_i
    output = jnp.clip(unclamped, lo, hi)
    not_sat = (unclamped > lo) & (unclamped < hi)
    return output, not_sat


# ---------------------------------------------------------------------------
# Flash VLE (simplified Raoult-like)
# ---------------------------------------------------------------------------

def _vle_equilibrium(
    x_A: float, x_B: float, eta_sep: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vapour mole fractions (y_A, y_B, y_C) from liquid (x_A, x_B, x_C=1-x_A-x_B).

    Degraded relative volatility: alpha_eff_i = 1 + eta_sep * (alpha_nom_i - 1)
    """
    x_C = jnp.clip(1.0 - x_A - x_B, 0.0, 1.0)
    x_A = jnp.clip(x_A, 0.0, 1.0)
    x_B = jnp.clip(x_B, 0.0, 1.0)

    a_A = 1.0 + eta_sep * (ALPHA_VLE_A - 1.0)
    a_B = 1.0 + eta_sep * (ALPHA_VLE_B - 1.0)
    a_C = 1.0 + eta_sep * (ALPHA_VLE_C - 1.0)  # = 1.0 always

    denom = jnp.maximum(a_A * x_A + a_B * x_B + a_C * x_C, 1e-8)
    y_A = a_A * x_A / denom
    y_B = a_B * x_B / denom
    y_C = a_C * x_C / denom
    return y_A, y_B, y_C


def _vapour_fraction(x_A: float, x_B: float, eta_sep: float) -> jnp.ndarray:
    """Approximate vapour fraction from Rachford-Rice (simplified).

    At the nominal operating point V_FRAC_NOM gives ~60% vapour. We
    model this as a linear function of the deviation in (x_A + x_B)
    from the nominal composition.
    """
    x_light = jnp.clip(x_A + x_B, 0.0, 1.0)
    x_light_nom = 0.70  # nominal light fraction at reactor outlet
    return jnp.clip(V_FRAC_NOM + 0.3 * (x_light - x_light_nom), 0.1, 0.95)


# ---------------------------------------------------------------------------
# 13-state closed-loop RHS
# ---------------------------------------------------------------------------

def luyben_rhs(
    t: float,
    y: jnp.ndarray,
    args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Right-hand side of the Luyben recycle plant ODE.

    State vector (13-D):
        y = [Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s, I_T, I_Ts, I_L, I_R, I_P]

    Args:
        theta = [alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta]
        inlet = [F_A0, F_B0, CA_IN, CB_IN, T_IN, T_CI]  (6-D)
        ctrl  = (5, 6) array, one row per PI loop
    """
    theta, inlet, ctrl = args
    alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta = (
        theta[0], theta[1], theta[2], theta[3],
        theta[4], theta[5], theta[6], theta[7],
    )

    Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s = (
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
    )
    I_T, I_Ts, I_L, I_R, I_P = y[8], y[9], y[10], y[11], y[12]

    # Unpack inlet
    F_A0_eff = inlet[0] * (1.0 + delta)    # stoichiometry shift
    F_B0_eff = inlet[1] * (1.0 - delta)
    CA_in  = inlet[2]
    CB_in  = inlet[3]
    T_in   = inlet[4] + kappa * DELTA_T_PREHEAT   # kappa degrades preheat
    T_ci   = inlet[5]

    # PI loop outputs
    Qc,   sat1 = _pi_output(T_r, I_T,  ctrl[0])
    Q_s,  sat2 = _pi_output(T_s, I_Ts, ctrl[1])
    F_L,  sat3 = _pi_output(n_L, I_L,  ctrl[2])
    F_R_ctrl, sat4 = _pi_output(F_R_NOM, I_R, ctrl[3])

    # Recycle flow degraded by pump efficiency
    F_R = jnp.clip(F_R_ctrl * eta_p, FR_MIN, FR_MAX)

    # Purge flow: controlled on recycle A fraction, restricted by xi
    # xi > 1 means erosion (more flow for same valve position); xi < 1 means blockage
    F_P_ctrl, sat5 = _pi_output(x_A, I_P, ctrl[4])
    F_P = jnp.clip(F_P_ctrl * xi, FP_MIN, FP_MAX)

    # Total reactor feed
    F_in = F_A0_eff + F_B0_eff + F_R
    # Mixed inlet concentrations
    CA_mix = (F_A0_eff * CA_in + F_R * x_A * (n_L / jnp.maximum(n_L, 1e-6))) / jnp.maximum(F_in, 1e-6)
    CB_mix = (F_B0_eff * CB_in + F_R * x_B * (n_L / jnp.maximum(n_L, 1e-6))) / jnp.maximum(F_in, 1e-6)
    # Effective inlet concentrations (recycle dilutes inlet streams)
    Ca_feed = (F_A0_eff * CA_in + F_R * jnp.clip(x_A, 0.0, 1.0)) / jnp.maximum(F_in, 1e-6)
    Cb_feed = (F_B0_eff * CB_in + F_R * jnp.clip(x_B, 0.0, 1.0)) / jnp.maximum(F_in, 1e-6)
    T_feed  = (F_A0_eff * T_in + F_B0_eff * T_in + F_R * T_s) / jnp.maximum(F_in, 1e-6)

    # Reaction rate
    k_r = alpha * K0_R * jnp.exp(-E_A_R / (R_GAS * jnp.maximum(T_r, 200.0)))
    rate = k_r * jnp.maximum(Ca, 0.0) * jnp.maximum(Cb, 0.0)

    # Effective heat transfer
    UA_r_eff = beta_r * UA_R_NOM
    UA_s_eff = beta_s * UA_S_NOM

    # ----- CSTR -----
    dCa = (F_in / V_R) * (Ca_feed - Ca) - rate
    dCb = (F_in / V_R) * (Cb_feed - Cb) - rate
    dT_r = (
        (F_in / V_R) * (T_feed - T_r)
        + (-H_R_R) * rate / (RHO_R * CP_R)
        - UA_r_eff * (T_r - Tc) / (RHO_R * CP_R * V_R)
    )
    dTc = (Qc / (RHO_C * CP_C * V_C_R)) * (T_ci - Tc) + UA_r_eff * (T_r - Tc) / (RHO_C * CP_C * V_C_R)

    # ----- Flash separator -----
    # Reactor outlet feeds separator; composition approximately equal to reactor outlet
    z_A = jnp.clip(Ca / jnp.maximum(Ca + Cb + (1.0 / jnp.maximum(V_R, 1.0)), 1e-6), 0.0, 1.0)
    z_B = jnp.clip(Cb / jnp.maximum(Ca + Cb + (1.0 / jnp.maximum(V_R, 1.0)), 1e-6), 0.0, 1.0)
    z_A = Ca / jnp.maximum(Ca + Cb + 0.1, 1e-6)  # simplified mol fraction proxy
    z_B = Cb / jnp.maximum(Ca + Cb + 0.1, 1e-6)

    y_A, y_B, y_C = _vle_equilibrium(x_A, x_B, eta_sep)
    V_frac = _vapour_fraction(z_A, z_B, eta_sep)
    F_sep = F_in   # feed to separator = CSTR outlet flow
    F_V  = F_sep * V_frac
    F_L_out = F_L  # controlled by Loop 3

    # Molar compositions entering separator (proxy: reactor outlet as mol fractions)
    # For mass balance, use concentration ratio as mole fraction proxy
    C_total = jnp.maximum(Ca + Cb + 0.1, 1e-6)
    z_A_mol = Ca / C_total
    z_B_mol = Cb / C_total
    z_C_mol = 0.1 / C_total

    # Separator balances
    dn_L = F_sep * (1.0 - V_frac) - F_L_out
    dx_A = (F_sep * z_A_mol - F_V * y_A - F_L_out * x_A) / jnp.maximum(n_L, 1.0)
    dx_B = (F_sep * z_B_mol - F_V * y_B - F_L_out * x_B) / jnp.maximum(n_L, 1.0)

    # Separator temperature (cooled by Q_s, heated by incoming stream)
    dT_s = (
        (F_sep / n_L) * (T_r - T_s)
        - UA_s_eff * (T_s - T_S_REF) / (n_L * CP_R * RHO_R * 0.01)  # simplified
        - Q_s / (n_L * CP_R * RHO_R * 0.01)
    )

    # Product flow (liquid leaving separator, minus purge from vapour)
    # F_prod = F_L (the liquid draw)

    # ----- PI integrators (anti-windup) -----
    dI_T  = jnp.where(sat1, T_r - TSP_R,    0.0)
    dI_Ts = jnp.where(sat2, T_s - TSP_S,    0.0)
    dI_L  = jnp.where(sat3, n_L - NSP_L,    0.0)
    dI_R  = jnp.where(sat4, F_R_NOM - FR_SP, 0.0)
    dI_P  = jnp.where(sat5, x_A - X_P_SP,   0.0)

    return jnp.array([
        dCa, dCb, dT_r, dTc, dn_L, dx_A, dx_B, dT_s,
        dI_T, dI_Ts, dI_L, dI_R, dI_P,
    ])


# ---------------------------------------------------------------------------
# Open-loop RHS (control loops bypassed, fixed actuator values)
# ---------------------------------------------------------------------------

def luyben_open_loop_rhs(
    t: float,
    y: jnp.ndarray,
    args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Open-loop RHS: actuators fixed at their nominal values.

    State vector (8-D, no PI integrators):
        y = [Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s]

    ctrl is ignored; nominal actuator values from module constants are used.
    """
    theta, inlet, _ctrl = args
    alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta = (
        theta[0], theta[1], theta[2], theta[3],
        theta[4], theta[5], theta[6], theta[7],
    )

    Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s = (
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
    )

    # Fixed inlet
    F_A0_eff = inlet[0] * (1.0 + delta)
    F_B0_eff = inlet[1] * (1.0 - delta)
    T_in = inlet[4] + kappa * DELTA_T_PREHEAT
    T_ci = inlet[5]

    # Fixed actuator values (nominal)
    Qc  = QC0_R
    Q_s = QS0
    F_L = FL0
    F_R = FR0 * eta_p
    F_P = FP0 * xi

    F_in = F_A0_eff + F_B0_eff + F_R
    Ca_feed = (F_A0_eff * inlet[2] + F_R * jnp.clip(x_A, 0.0, 1.0)) / jnp.maximum(F_in, 1e-6)
    Cb_feed = (F_B0_eff * inlet[3] + F_R * jnp.clip(x_B, 0.0, 1.0)) / jnp.maximum(F_in, 1e-6)
    T_feed  = (F_A0_eff * T_in + F_B0_eff * T_in + F_R * T_s) / jnp.maximum(F_in, 1e-6)

    k_r  = alpha * K0_R * jnp.exp(-E_A_R / (R_GAS * jnp.maximum(T_r, 200.0)))
    rate = k_r * jnp.maximum(Ca, 0.0) * jnp.maximum(Cb, 0.0)

    UA_r_eff = beta_r * UA_R_NOM
    UA_s_eff = beta_s * UA_S_NOM

    dCa = (F_in / V_R) * (Ca_feed - Ca) - rate
    dCb = (F_in / V_R) * (Cb_feed - Cb) - rate
    dT_r = (
        (F_in / V_R) * (T_feed - T_r)
        + (-H_R_R) * rate / (RHO_R * CP_R)
        - UA_r_eff * (T_r - Tc) / (RHO_R * CP_R * V_R)
    )
    dTc = (Qc / (RHO_C * CP_C * V_C_R)) * (T_ci - Tc) + UA_r_eff * (T_r - Tc) / (RHO_C * CP_C * V_C_R)

    y_A, y_B, y_C = _vle_equilibrium(x_A, x_B, eta_sep)
    V_frac = _vapour_fraction(Ca / jnp.maximum(Ca + Cb + 0.1, 1e-6),
                               Cb / jnp.maximum(Ca + Cb + 0.1, 1e-6), eta_sep)
    C_total = jnp.maximum(Ca + Cb + 0.1, 1e-6)

    dn_L = F_in * (1.0 - V_frac) - F_L
    dx_A = (F_in * Ca / C_total - F_in * V_frac * y_A - F_L * x_A) / jnp.maximum(n_L, 1.0)
    dx_B = (F_in * Cb / C_total - F_in * V_frac * y_B - F_L * x_B) / jnp.maximum(n_L, 1.0)
    dT_s = (
        (F_in / jnp.maximum(n_L, 1.0)) * (T_r - T_s)
        - UA_s_eff * (T_s - T_S_REF) / (jnp.maximum(n_L, 1.0) * CP_R * RHO_R * 0.01)
        - Q_s / (jnp.maximum(n_L, 1.0) * CP_R * RHO_R * 0.01)
    )

    return jnp.array([dCa, dCb, dT_r, dTc, dn_L, dx_A, dx_B, dT_s])


# ---------------------------------------------------------------------------
# Steady-state integrators (diffrax Tsit5)
# ---------------------------------------------------------------------------

def simulate_to_steady_state(
    theta: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray = None,
    y0: jnp.ndarray = None,
    t_final: float = 2000.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 2_000_000,
) -> jnp.ndarray:
    """Integrate to steady state; returns final 13-D state."""
    if ctrl is None:
        ctrl = NOMINAL_CTRL_ALL
    if y0 is None:
        y0 = NOMINAL_Y0

    term = diffrax.ODETerm(luyben_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=t_final, dt0=0.01,
        y0=y0, args=(theta, inlet, ctrl),
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=False,
    )
    return sol.ys[-1]


def simulate_trajectory(
    theta: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray = None,
    y0: jnp.ndarray = None,
    t_final: float = 120.0,
    n_save: int = 121,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 2_000_000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate and return full trajectory. Returns (ts, ys) shapes (n_save,) and (n_save, 13)."""
    if ctrl is None:
        ctrl = NOMINAL_CTRL_ALL
    if y0 is None:
        y0 = NOMINAL_Y0

    term = diffrax.ODETerm(luyben_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_final, n_save))
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=t_final, dt0=0.01,
        y0=y0, args=(theta, inlet, ctrl),
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=max_steps,
        throw=False,
    )
    return sol.ts, sol.ys


# ---------------------------------------------------------------------------
# Observation extractor: 13-state -> 8 observable channels
# ---------------------------------------------------------------------------

def extract_observations(
    ys: jnp.ndarray,
    theta: jnp.ndarray,
    ctrl: jnp.ndarray = None,
) -> jnp.ndarray:
    """Extract 8 observable channels from a (n_t, 13) trajectory.

    Returns shape (n_t, 8): [T_r, Tc, Qc, T_s, Q_s, F_R, F_P, F_prod]

    Qc, Q_s, F_R, F_P are computed from PI controller equations and
    degradation parameters at each timestep.
    """
    if ctrl is None:
        ctrl = NOMINAL_CTRL_ALL

    eta_p = theta[4]
    xi    = theta[5]

    T_r = ys[:, 2]
    Tc  = ys[:, 3]
    T_s = ys[:, 7]
    I_T  = ys[:, 8]
    I_Ts = ys[:, 9]
    I_L  = ys[:, 10]
    I_R  = ys[:, 11]
    I_P  = ys[:, 12]
    x_A  = ys[:, 5]
    n_L  = ys[:, 4]
    F_L  = ys[:, 4]  # F_L ~ dL/dt tracking, use I_L proxy

    def _obs_one(i):
        t_r_i  = T_r[i]
        tc_i   = Tc[i]
        t_s_i  = T_s[i]
        i_t_i  = I_T[i]
        i_ts_i = I_Ts[i]
        i_r_i  = I_R[i]
        i_p_i  = I_P[i]
        x_a_i  = x_A[i]
        n_l_i  = n_L[i]

        qc,   _ = _pi_output(t_r_i,  i_t_i,  ctrl[0])
        q_s,  _ = _pi_output(t_s_i,  i_ts_i, ctrl[1])
        fl,   _ = _pi_output(n_l_i,  I_L[i], ctrl[2])
        f_r_c,_ = _pi_output(F_R_NOM, i_r_i, ctrl[3])
        f_p_c,_ = _pi_output(x_a_i,  i_p_i,  ctrl[4])

        f_r = jnp.clip(f_r_c * eta_p, FR_MIN, FR_MAX)
        f_p = jnp.clip(f_p_c * xi,    FP_MIN, FP_MAX)
        f_prod = jnp.clip(fl, 0.0, FL_MAX)
        return jnp.array([t_r_i, tc_i, qc, t_s_i, q_s, f_r, f_p, f_prod])

    n_t = ys.shape[0]
    return jax.vmap(lambda i: _obs_one(i))(jnp.arange(n_t))


# ---------------------------------------------------------------------------
# Nominal inlet vector (6-D)
# ---------------------------------------------------------------------------

NOMINAL_INLET = jnp.array(
    [F_A0, F_B0, CA_IN, CB_IN, T_IN, T_CI],
    dtype=jnp.float32,
)
