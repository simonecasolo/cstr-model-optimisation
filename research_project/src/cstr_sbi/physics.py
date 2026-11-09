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
#     Qc(t) = Qc0 + Kp*[ (T - Tsp) + (1/tau_i) * integral(T - Tsp) dt ]
#
# 2026-11-08 FIX: Kp previously did NOT multiply the integral term
# (Qc = Qc0 + Kp*(T-Tsp) + I/tau_i, which is also dimensionally inconsistent).
# Standard form now used; I = integral(T-Tsp) dt has units K*min.
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
    output is the standard (dimensionally consistent) PI law
    ``Qc0 + Kp*[(T - Tsp) + I/tau_i]`` with ``I = integral(T - Tsp) dt`` in K*min
    (controller-form fix, 2026-11-08: Kp now multiplies the integral term; the
    previous ``Kp*(T-Tsp) + I/tau_i`` had integral gain 1/tau_i instead of
    Kp/tau_i, i.e. 150x too weak, with a 1.6e3-5.5e3 min slow mode); valve limits are applied via
    ``jnp.clip``.
    """
    Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max = ctrl
    qc_unclamped = Qc0 + Kp * ((T - Tsp) + I / tau_i)
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
    qc_unclamped = Qc0 + Kp * ((T - Tsp) + I / tau_i)
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


def closed_loop_steady_state_exact(
    params: jnp.ndarray,
    inlet: jnp.ndarray,
    ctrl: jnp.ndarray = NOMINAL_CTRL,
) -> jnp.ndarray:
    """Exact closed-loop steady state via root-finding, not forward integration.

    2026-11-07 nb22 warm-start audit: ``warm_start_ic`` (via
    ``simulate_closed_loop_to_steady_state``) forward-integrates for a fixed
    ``t_warm`` and returns the endpoint. ``C``, ``T``, ``Tc`` converge quickly
    but the PI integrator state ``I`` is a genuinely slow mode and was found
    to still be drifting at the project's default ``t_warm=1500`` (and
    naively raising ``t_warm`` hits a second bug: ``diffeqsolve``'s
    ``max_steps`` silently truncates the run with ``throw=False``, no
    warning). This function instead solves the steady-state algebraic
    equations directly:

    * **Non-saturated branch.** ``dI/dt=0`` with the anti-windup gate open
      pins ``T = Tsp`` *exactly* (integral control removes steady-state
      temperature error) -- consistent with this project's own existing
      argument elsewhere about integral-control steady-state behaviour.
      Given ``T = Tsp``, ``C``, ``Tc``, ``Qc`` follow in closed form from
      ``dC/dt=0``, ``dT/dt=0``, ``dTc/dt=0`` in turn, and ``I`` follows from
      the controller law. No iteration needed.
    * **Saturated branch** (e.g. Sc2/Sc5's heavy fouling, where even the
      maximum coolant flow cannot remove enough heat to reach ``Tsp``):
      ``T`` no longer equals ``Tsp``. ``Qc`` is pinned at the active bound
      and a 1-D root-find (Arrhenius-nonlinear in ``T``) gives ``T``; ``C``
      and ``Tc`` follow in closed form from ``T``; ``I`` is set to the value
      that makes the *unclamped* controller output touch the active bound
      (the value the real dynamics approach in the limit of sustained
      saturation -- the anti-windup gate itself does not pin a unique ``I``,
      since any ``I`` keeping ``qc_unclamped`` beyond the bound satisfies
      ``dI/dt=0``; this is the natural boundary-continuous choice).

      **Which bound is active is decided by feasibility, not by the sign of
      the non-saturated candidate's implied ``Qc``.** An earlier version of
      this function picked the bound from the sign/magnitude of the
      non-saturated candidate's ``Qc`` directly, which is wrong: ``Tc`` is
      physically bounded within ``[Tci, T]`` for any ``Qc >= 0`` (as
      ``Qc -> inf``, ``Tc -> Tci``; as ``Qc -> 0``, ``Tc -> T``), so when the
      non-saturated algebra implies a ``Tc`` *below* ``Tci`` (as happens for
      Sc2: insufficient cooling capacity even at infinite flow), the correct
      response is to saturate at ``Qc_max`` (the best available cooling),
      NOT ``Qc_min`` -- picking ``Qc_min`` there is backwards and was
      verified numerically to give the wrong, far-from-``Tsp`` answer
      (``T=316.36`` vs. the correct ``T=312.87`` for Sc2's beta=0.7).
      Instead, this function always evaluates ``T`` at both ``Qc_min`` and
      ``Qc_max`` (cheap 1-D root-finds) and uses monotonicity of
      ``T_ss(Qc)`` (more cooling flow -> lower ``T``, verified numerically
      for this parameter regime) to decide: if ``Tsp`` lies between the two,
      use the non-saturated closed form; otherwise saturate at whichever
      bound's ``T_ss`` is the closer feasible point to ``Tsp``.

    2026-11-08 JAX-differentiability fix (nb23): the implementation above was
    originally written with plain Python ``float()`` and
    ``scipy.optimize.brentq``, which eager-evaluate on concrete numbers but
    cannot be traced by ``jax.grad``/``jax.jit`` -- fatal for NUTS's
    ``matched_warm_start=True`` path, which needs HMC gradients through this
    function (``jax.grad`` raised ``ConcretizationTypeError`` on the
    ``float()`` call). This is now a JAX-native reimplementation: the 1-D
    root-find is a fixed-iteration-count bisection via ``jax.lax.fori_loop``
    (replacing ``scipy.optimize.brentq``), and the saturated-vs-non-saturated
    branch selection uses ``jnp.where`` (replacing the Python ``if``) so both
    branches are always evaluated and selected differentiably. The physics
    (bracket bounds, monotonicity-based saturation-bound selection, the
    anti-windup epsilon push) is unchanged from the derivation above -- only
    *how* it is computed changed, not *what* it computes. Verified
    numerically identical to the old scipy-based implementation to within
    bisection tolerance at every closed-loop Sc0-7 point (see
    ``results/23_correctness_check.csv``); differentiability confirmed via a
    direct ``jax.grad`` probe and via ``run_mcmc_baseline(...,
    matched_warm_start=True)`` no longer crashing at Sc4/Sc5 (see
    ``notebooks/system_I/23_nuts_differentiable_warmstart_fix.ipynb``).

    Works under both eager (concrete) and traced (``jax.grad``/``jax.jit``)
    evaluation -- every call site in this codebase invokes it eagerly except
    NUTS's differentiated likelihood, which is the one path this fix exists
    for.
    """
    UA, k0, alpha, beta = params[0], params[1], params[2], params[3]
    Ci, Ti, Tci = inlet[0], inlet[1], inlet[2]
    Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max = (
        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5],
    )

    UA_eff = beta * UA

    def k_eff(T):
        return alpha * k0 * jnp.exp(-E_A / (R_GAS * T))

    def C_of_T(T):
        return (Q / V) * Ci / (Q / V + k_eff(T))

    def Tc_of_T(T, Qc):
        return (Qc * RHO_C * C_PC * Tci + UA_eff * T) / (Qc * RHO_C * C_PC + UA_eff)

    def dT_residual(T, Qc):
        ke = k_eff(T)
        C = C_of_T(T)
        Tc = Tc_of_T(T, Qc)
        return (
            (Q / V) * (Ti - T)
            - H_R * ke * C / (RHO * C_P)
            - UA_eff * (T - Tc) / (RHO * C_P * V)
        )

    def solve_T_at(Qc, n_iter: int = 60):
        # Fixed-iteration-count bisection on T in [250, 450] K, replacing
        # scipy.optimize.brentq. 60 halvings of a 200 K bracket drives the
        # bracket width to ~200/2**60 ~ 1.7e-16 K, far below the xtol=1e-12
        # the old brentq call used -- the extra iterations are cheap (each
        # is one exp() + a handful of flops) and are not spent chasing
        # unneeded precision once the bracket is already below float64 eps
        # at these T scales (~300 K * 2.2e-16 ~ 6.6e-14 K).
        T_lo0 = jnp.asarray(250.0, dtype=UA.dtype)
        T_hi0 = jnp.asarray(450.0, dtype=UA.dtype)

        def body(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            f_lo = dT_residual(lo, Qc)
            f_mid = dT_residual(mid, Qc)
            same_sign = f_lo * f_mid > 0.0
            lo_new = jnp.where(same_sign, mid, lo)
            hi_new = jnp.where(same_sign, hi, mid)
            return (lo_new, hi_new)

        lo_f, hi_f = jax.lax.fori_loop(0, n_iter, body, (T_lo0, T_hi0))
        return 0.5 * (lo_f + hi_f)

    # Feasibility check: is Tsp reachable within the actuator's range?
    # T_ss(Qc) is monotonically decreasing in Qc (more coolant flow -> lower
    # T), so the reachable range at steady state is [T_ss(Qc_max), T_ss(Qc_min)].
    T_at_Qmax = solve_T_at(Qc_max)
    T_at_Qmin = solve_T_at(Qc_min)

    not_saturated = (T_at_Qmax <= Tsp) & (Tsp <= T_at_Qmin)

    # --- Non-saturated candidate: dI/dt = 0 (gate open) forces T = Tsp exactly.
    ke_sp = k_eff(Tsp)
    C_ns = C_of_T(Tsp)
    rhs_no_cool = (Q / V) * (Ti - Tsp) - H_R * ke_sp * C_ns / (RHO * C_P)
    Tc_ns = Tsp - rhs_no_cool * (RHO * C_P * V) / UA_eff
    Qc_ns = -UA_eff * (Tsp - Tc_ns) / (RHO_C * C_PC * (Tci - Tc_ns))
    I_ns = tau_i * (Qc_ns - Qc0) / Kp
    result_ns = jnp.array([C_ns, Tsp, Tc_ns, I_ns])

    # --- Saturated candidate: Tsp unreachable within [Qc_min, Qc_max]; pick
    # whichever bound's steady state is the closer feasible point to Tsp.
    # (jnp.where in place of the Python `if Tsp < T_at_Qmax` -- both bound
    # choices are evaluated and selected differentiably.)
    use_qmax = Tsp < T_at_Qmax
    Qc_fixed = jnp.where(use_qmax, Qc_max, Qc_min)
    T_star = jnp.where(use_qmax, T_at_Qmax, T_at_Qmin)
    C_star = C_of_T(T_star)
    Tc_star = Tc_of_T(T_star, Qc_fixed)
    # Push `qc_unclamped` a hair past the active bound (not exactly onto it)
    # so the gate's strict `<`/`>` comparison in `cstr_closed_loop_rhs` is
    # robust to float32 rounding of `I` on the caller side -- verified
    # (nb22 warm-start-fix audit) that landing exactly on the boundary can
    # round the wrong way under float32 and spuriously re-open the gate,
    # giving a small but nonzero `dI` at an otherwise-exact fixed point.
    _qc_eps = 2e-2
    _sat_sign = jnp.where(use_qmax, 1.0, -1.0)
    I_star = tau_i * ((Qc_fixed - Qc0 + _sat_sign * _qc_eps) / Kp - (T_star - Tsp))
    result_sat = jnp.array([C_star, T_star, Tc_star, I_star])

    return jnp.where(not_saturated, result_ns, result_sat)


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
