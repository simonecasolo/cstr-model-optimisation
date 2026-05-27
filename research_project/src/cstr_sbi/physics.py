"""CSTR ODE right-hand sides.

This module implements the deterministic CSTR dynamics in pure JAX so that the
same code path is used by:

* the NumPyro generative model (NUTS / DiscreteHMCGibbs MCMC), and
* the ``sbi.SNPE_C`` simulator wrapper through ``jax.vmap`` + ``jax.jit``.

Constants follow ``cstr_parameters_recommended.md``: a Fogler (2016) Module 13
parameter set for the acid-catalysed hydrolysis of propylene oxide to
propylene glycol, replacing the Pilario & Cao (2018) benchmark values that
were used previously. Units remain calorie-based for the heat-related
quantities (cal/mol, cal/(g K), cal/(min K)) and SI for everything else
(K, mol/L, min, J/mol).

M0 scope: ``cstr_open_loop_rhs`` and a thin ``simulate_open_loop_to_steady_state``
helper that integrates the ODE with diffrax. M1 will add the 4-state
closed-loop RHS with the PI integral state, anti-windup, and the alpha/beta
degradation factors per spec Section 2.4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import diffrax


# ---------------------------------------------------------------------------
# Physical constants (Fogler 2016 Module 13, see cstr_parameters_recommended.md)
# ---------------------------------------------------------------------------

# Reaction kinetics
K0_NOMINAL: float = 16.96e12  # 1/min,    pre-exponential factor   (Fogler / Furusawa 1969)
E_A: float = 75362.0          # J/mol,    activation energy        (Fogler / Furusawa 1969)
H_R: float = -20220.0         # cal/mol,  heat of reaction         (Fogler 2016; -84,666 J/mol)
R_GAS: float = 8.314          # J/(K mol), universal gas constant

# Reactor fluid (dilute aqueous PO/PG, ~300-320 K)
RHO: float = 1000.0           # g/L,      density (~ liquid water)
C_P: float = 1.0              # cal/(g K), heat capacity (~ liquid water)

# Coolant (process water)
RHO_C: float = 1000.0         # g/L
C_PC: float = 1.0             # cal/(g K)

# Reactor geometry (pilot scale, Fogler M13)
V: float = 500.0              # L,        reactor volume
V_C: float = 40.0             # L,        jacket volume
Q: float = 40.0               # L/min,    feed flow rate    (tau = V/Q = 12.5 min)

# Nominal heat transfer
UA_NOMINAL: float = 1.25e4    # cal/(min K)

# Nominal feed conditions
CI_NOMINAL: float = 0.97      # mol/L,   inlet PO concentration
TI_NOMINAL: float = 297.0     # K,       feed temperature  (75 F)
TCI_NOMINAL: float = 297.0    # K,       coolant inlet temperature

# PI controller (cstr_parameters_recommended.md Section 3.2)
#
# Sign convention.  We use the spec equation
#     Qc(t) = Qc0 + Kp*(T - Tsp) + (1/tau_i) * integral(T - Tsp) dt
# (research_spec.md Section 3.2). With this convention the proportional
# gain MUST be positive for the controller to increase cooling when the
# reactor overheats: T > Tsp -> error positive -> Qc grows.
#
# cstr_parameters_recommended.md lists Kp = -150 with the verbal note
# "negative: more cooling if T rises", which only holds under the dual
# convention u = u0 - Kp*(T - Tsp). We use the magnitude (+150) of that
# value with the spec's additive convention; the physics is identical.
TSP: float = 312.5            # K,       temperature setpoint
KP: float = 150.0             # (L/min)/K, proportional gain (magnitude of doc's |-150|)
TAU_I: float = 10.0           # min,     integral time constant
QC0: float = 80.0             # L/min,   bias (nominal Qc at zero error)
QC_MIN: float = 0.0           # L/min,   valve fully closed
QC_MAX: float = 400.0         # L/min,   valve fully open

# Degradation timescale (used during data generation only; spec Section 3.4)
T_CRIT: float = 43200.0       # min, 30 days


@dataclass(frozen=True)
class InletConditions:
    """Inlet conditions for the open-loop CSTR model.

    ``Qc`` is treated as a fixed input here; in the closed-loop model (M1) it
    is computed by the PI controller and is not part of this dataclass.
    """

    Ci: float    # mol/L,  inlet concentration of reactant
    Ti: float    # K,      inlet temperature
    Tci: float   # K,      coolant inlet temperature
    Qc: float    # L/min,  fixed coolant flow rate (open-loop only)


# Default inlet for the nominal operating point (used by smoke test and notebooks).
NOMINAL_INLET = jnp.array([CI_NOMINAL, TI_NOMINAL, TCI_NOMINAL, QC0])
NOMINAL_PARAMS = jnp.array([UA_NOMINAL, K0_NOMINAL])
NOMINAL_Y0 = jnp.array([0.5, 300.0, 297.0])  # near feed conditions; ODE relaxes to ss

# Closed-loop conventions (used by M1's PI controller and the notebooks):
#   inlet_cl = jnp.array([Ci, Ti, Tci])         (no Qc -- controller computes it)
#   ctrl     = jnp.array([Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max])
#   params_cl = jnp.array([UA, k0, alpha, beta])
#   y_cl     = jnp.array([C, T, Tc, I])         (4-state)
NOMINAL_INLET_CL = jnp.array([CI_NOMINAL, TI_NOMINAL, TCI_NOMINAL])
NOMINAL_PARAMS_CL = jnp.array([UA_NOMINAL, K0_NOMINAL, 1.0, 1.0])  # alpha = beta = 1
NOMINAL_CTRL = jnp.array([KP, TAU_I, TSP, QC0, QC_MIN, QC_MAX])

# Warm-start IC near the open-loop steady state at Qc=80. A cold IC such as
# [0.5, 300, 297, 0] causes T to overshoot violently as the controller starts
# from a 12.5 K error and drives the system through a stiff Arrhenius transient
# that breaks the explicit Tsit5 solver. The warm IC keeps the trajectory
# inside the smooth operating envelope from the start. M2's full ``run_simulation_window``
# can re-introduce a cold IC by composing an open-loop "warm-up phase" first.
NOMINAL_Y0_CL = jnp.array([0.0184, 312.15, 299.05, 0.0])


@dataclass(frozen=True)
class ControllerParams:
    """Convenience container for PI controller settings.

    The simulator entry points accept either this dataclass or the equivalent
    flat ``jnp.ndarray`` via ``ControllerParams.as_array()`` -- the latter is
    needed for ``jax.jit`` / ``jax.vmap`` because dataclasses are not pytrees
    of leaves by default.
    """

    Kp: float = KP
    tau_i: float = TAU_I
    Tsp: float = TSP
    Qc0: float = QC0
    Qc_min: float = QC_MIN
    Qc_max: float = QC_MAX

    def as_array(self) -> jnp.ndarray:
        return jnp.array(
            [self.Kp, self.tau_i, self.Tsp, self.Qc0, self.Qc_min, self.Qc_max]
        )


# ---------------------------------------------------------------------------
# Open-loop 3-state RHS  (M0 deliverable: the JAX/diffrax smoke test target)
# ---------------------------------------------------------------------------

def cstr_open_loop_rhs(
    t: float,
    y: jnp.ndarray,
    args: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Right-hand side of the open-loop CSTR ODE.

    State vector: ``y = [C, T, Tc]`` (mol/L, K, K).
    Args tuple: ``(params, inlet)`` where
        ``params = jnp.array([UA, k0])`` and
        ``inlet  = jnp.array([Ci, Ti, Tci, Qc])``.

    Equations (spec Section 3.1):

        dC/dt  = (Q/V)(Ci - C) - k C
        dT/dt  = (Q/V)(Ti - T) - H_r k C / (rho C_p)
                 - UA (T - Tc) / (rho C_p V)
        dTc/dt = (Qc/Vc)(Tci - Tc) + UA (T - Tc) / (rho_c C_pc Vc)
        k      = k0 exp(-Ea / (R T))
    """
    params, inlet = args
    UA, k0 = params[0], params[1]
    Ci, Ti, Tci, Qc = inlet[0], inlet[1], inlet[2], inlet[3]
    C, T, Tc = y[0], y[1], y[2]

    k = k0 * jnp.exp(-E_A / (R_GAS * T))

    dC = (Q / V) * (Ci - C) - k * C
    dT = (
        (Q / V) * (Ti - T)
        - H_R * k * C / (RHO * C_P)
        - UA * (T - Tc) / (RHO * C_P * V)
    )
    dTc = (Qc / V_C) * (Tci - Tc) + UA * (T - Tc) / (RHO_C * C_PC * V_C)

    return jnp.array([dC, dT, dTc])


# ---------------------------------------------------------------------------
# Steady-state integrator  (M0 smoke-test entry point)
# ---------------------------------------------------------------------------

def simulate_open_loop_to_steady_state(
    params: jnp.ndarray,
    inlet: jnp.ndarray,
    y0: jnp.ndarray = NOMINAL_Y0,
    t_final: float = 200.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 1_000_000,
) -> jnp.ndarray:
    """Integrate the open-loop CSTR until ``t_final`` (minutes).

    Returns the final state vector ``[C, T, Tc]``.

    The default ``t_final = 200`` minutes is 16 residence times at the
    nominal flow rate (tau = V/Q = 12.5 min) and brings the system well into
    its asymptotic steady state for the Fogler-grounded operating point
    documented in ``cstr_parameters_recommended.md``.

    Uses ``diffrax.Tsit5`` with a PID step controller (the JAX analogue of
    scipy's adaptive ``RK45``). The CSTR ODE is moderately stiff via the
    Arrhenius term, but in the nominal operating window an explicit method is
    fast and accurate; the implicit ``Kvaerno5`` is ~100x slower on CPU and
    is only needed for extreme parameter draws far outside the prior support.
    """
    term = diffrax.ODETerm(cstr_open_loop_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_final,
        dt0=0.01,
        y0=y0,
        args=(params, inlet),
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=False,  # NaN on failure (e.g. extreme draws under vmap)
    )
    return sol.ys[-1]


# A jit-compiled, vmapped variant exposed for benchmarking and SBI training
# wrappers. Vectorises over the parameter axis, keeping the inlet fixed.
_simulate_jit = jax.jit(simulate_open_loop_to_steady_state)
simulate_open_loop_batch = jax.jit(
    jax.vmap(simulate_open_loop_to_steady_state, in_axes=(0, None))
)


def simulate_open_loop_trajectory_fixed(
    params: jnp.ndarray,
    inlet: jnp.ndarray,
    y0: jnp.ndarray = NOMINAL_Y0,
    t_final: float = 60.0,
    n_save: int = 120,
    dt: float = 0.05,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Open-loop trajectory with **fixed-step** Tsit5 — safe for NUTS/JAX JIT.

    Uses ``diffrax.ConstantStepSize`` so the number of solver steps is
    deterministic and bounded (``t_final / dt``), preventing the adaptive
    solver from hanging on stiff extreme-prior draws during NUTS warmup.

    Parameters
    ----------
    params
        2-D ``[UA_eff, k0_eff]`` — already scaled by alpha/beta.
    inlet
        4-D ``[Ci, Ti, Tci, Qc]`` — Qc is fixed (open-loop).
    y0
        Initial state ``[C, T, Tc]``. Defaults to ``NOMINAL_Y0``.
    t_final
        Window length in minutes (default 60).
    n_save
        Number of saved time points (default 120, matching observations.npz).
    dt
        Fixed integration step in minutes. At ``dt=0.05`` the CSTR ODE is
        stable for all ``[alpha, beta]`` in the prior support [0.4, 1.0]²
        at the Fogler operating point (Tsit5 stability limit >> 0.05 min).

    Returns
    -------
    ts : jnp.ndarray, shape (n_save,)
    ys : jnp.ndarray, shape (n_save, 3) — columns [C, T, Tc]
    """
    term   = diffrax.ODETerm(cstr_open_loop_rhs)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_final, n_save))
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_final,
        dt0=dt,
        y0=y0,
        args=(params, inlet),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=saveat,
        max_steps=int(t_final / dt) + 10,
        throw=False,
    )
    return sol.ts, sol.ys


# ---------------------------------------------------------------------------
# Closed-loop 4-state RHS  (M1)
# ---------------------------------------------------------------------------

def compute_qc(T: float, I: float, ctrl: jnp.ndarray) -> jnp.ndarray:
    """Return the (clamped) coolant flow rate ``Qc`` from the PI controller.

    ``ctrl`` packs ``[Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max]``. The unclamped
    output is ``Qc0 + Kp*(T - Tsp) + I/tau_i``; valve limits are applied via
    ``jnp.clip``.
    """
    Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max = ctrl
    qc_unclamped = Qc0 + Kp * (T - Tsp) + I / tau_i
    return jnp.clip(qc_unclamped, Qc_min, Qc_max)


def cstr_closed_loop_rhs(
    t: float,
    y: jnp.ndarray,
    args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Right-hand side of the closed-loop 4-state CSTR ODE.

    State vector: ``y = [C, T, Tc, I]`` where ``I`` is the PI integrator.
    Args:
        ``params = jnp.array([UA, k0, alpha, beta])``  (spec Section 3.1)
        ``inlet  = jnp.array([Ci, Ti, Tci])``          (Qc is computed here)
        ``ctrl   = jnp.array([Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max])``

    The ODE follows spec Section 2.4.3:

        k_eff   = alpha * k0 * exp(-Ea / (R T))
        UA_eff  = beta  * UA
        dC/dt   = (Q/V)(Ci - C) - k_eff C
        dT/dt   = (Q/V)(Ti - T) - H_r k_eff C / (rho Cp)
                  - UA_eff (T - Tc) / (rho Cp V)
        dTc/dt  = (Qc/Vc)(Tci - Tc) + UA_eff (T - Tc) / (rho_c Cpc Vc)
        dI/dt   = (T - Tsp)  * gate

    The anti-windup ``gate`` is 1 when the unclamped controller output is
    inside the valve range, and 0 otherwise (conditional integration). This
    keeps the integrator from drifting while the valve is saturated.
    """
    params, inlet, ctrl = args
    UA, k0, alpha, beta = params[0], params[1], params[2], params[3]
    Ci, Ti, Tci = inlet[0], inlet[1], inlet[2]
    Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max = (
        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5],
    )
    C, T, Tc, I = y[0], y[1], y[2], y[3]

    # PI controller (with anti-windup via conditional integration)
    qc_unclamped = Qc0 + Kp * (T - Tsp) + I / tau_i
    Qc = jnp.clip(qc_unclamped, Qc_min, Qc_max)
    not_saturated = (qc_unclamped > Qc_min) & (qc_unclamped < Qc_max)
    dI = jnp.where(not_saturated, T - Tsp, 0.0)

    # Reaction with degradation
    k_eff = alpha * k0 * jnp.exp(-E_A / (R_GAS * T))
    UA_eff = beta * UA

    dC = (Q / V) * (Ci - C) - k_eff * C
    dT = (
        (Q / V) * (Ti - T)
        - H_R * k_eff * C / (RHO * C_P)
        - UA_eff * (T - Tc) / (RHO * C_P * V)
    )
    dTc = (Qc / V_C) * (Tci - Tc) + UA_eff * (T - Tc) / (RHO_C * C_PC * V_C)

    return jnp.array([dC, dT, dTc, dI])


# ---------------------------------------------------------------------------
# Closed-loop integrators
# ---------------------------------------------------------------------------

def simulate_closed_loop_to_steady_state(
    params: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray = NOMINAL_Y0_CL,
    t_final: float = 1000.0,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int = 2_000_000,
) -> jnp.ndarray:
    """Integrate the closed-loop 4-state CSTR until ``t_final`` (minutes).

    Returns ``[C, T, Tc, I]`` at the final time. The realised coolant flow
    can be recovered with ``compute_qc(T, I, ctrl)``.
    """
    term = diffrax.ODETerm(cstr_closed_loop_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_final,
        dt0=0.01,
        y0=y0,
        args=(params, inlet, ctrl),
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=False,
    )
    return sol.ys[-1]


def simulate_closed_loop_trajectory(
    params: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
    y0: jnp.ndarray = NOMINAL_Y0_CL,
    t_final: float = 1000.0,
    n_save: int = 401,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int = 2_000_000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate and return the full trajectory.

    Returns ``(t, ys, qc)`` where ``ys`` has shape ``(n_save, 4)``
    (``[C, T, Tc, I]``) and ``qc`` has shape ``(n_save,)``.
    """
    term = diffrax.ODETerm(cstr_closed_loop_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_final, n_save))
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_final,
        dt0=0.01,
        y0=y0,
        args=(params, inlet, ctrl),
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=max_steps,
        throw=False,
    )
    qc = jax.vmap(compute_qc, in_axes=(0, 0, None))(
        sol.ys[:, 1], sol.ys[:, 3], ctrl
    )
    return sol.ts, sol.ys, qc


# JIT-compiled and batched variants for SBI training (M4) and benchmarks.
simulate_closed_loop_jit = jax.jit(simulate_closed_loop_to_steady_state)
simulate_closed_loop_batch = jax.jit(
    jax.vmap(simulate_closed_loop_to_steady_state, in_axes=(0, None, None))
)
simulate_closed_loop_trajectory_jit = jax.jit(
    simulate_closed_loop_trajectory, static_argnames=("n_save",)
)
