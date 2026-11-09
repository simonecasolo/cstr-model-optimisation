"""Guards the System I PI controller form (fixed 2026-11-08).

The previous law Qc = Qc0 + Kp*(T-Tsp) + I/tau_i was dimensionally inconsistent
(I/tau_i is in K, not L/min) and gave an integral gain 1/tau_i instead of Kp/tau_i:
150x too weak, a closed-loop slow mode of ~1.6e3-5.5e3 min labelled 'tau_i = 10 min'.
"""
import numpy as np
import jax, jax.numpy as jnp
import pytest

from cstr_sbi import physics as ph

KP, TAU_I = float(ph.NOMINAL_CTRL[0]), float(ph.NOMINAL_CTRL[1])


def _params(a, b):
    return jnp.array([ph.UA_NOMINAL, ph.K0_NOMINAL, a, b], dtype=jnp.float32)


def test_integral_gain_is_kp_over_tau_i():
    """dQc/dI must equal Kp/tau_i (L/min per K*min), not 1/tau_i."""
    ctrl = ph.NOMINAL_CTRL
    dq = jax.grad(lambda I: ph.compute_qc(ctrl[2], I, ctrl))(0.0 * 1.0 + 1.0)  # inside valve range around T=Tsp?
    # at T=Tsp, I=1 -> unclamped = Qc0 + Kp/tau_i, well inside [0,400]
    assert float(dq) == pytest.approx(KP / TAU_I, rel=1e-6)


def test_proportional_gain_unchanged():
    ctrl = ph.NOMINAL_CTRL
    dq = jax.grad(lambda T: ph.compute_qc(T, 0.0, ctrl))(float(ctrl[2]) + 0.1)
    assert float(dq) == pytest.approx(KP, rel=1e-6)


@pytest.mark.parametrize("ab", [(1.0, 1.0), (0.85, 0.85), (0.7, 1.0), (1.2, 1.2)])
def test_steady_state_is_fixed_point_with_integrator(ab):
    p = _params(*ab)
    y = ph.closed_loop_steady_state_exact(p, ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)
    rhs = np.asarray(ph.cstr_closed_loop_rhs(0.0, y, (p, ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)))
    assert np.abs(rhs).max() < 1e-3
    # integrator state is in K*min: Qc = Qc0 + Kp*(I/tau_i) at T=Tsp
    qc = float(ph.compute_qc(y[1], y[3], ph.NOMINAL_CTRL))
    assert qc == pytest.approx(ph.QC0 + KP * float(y[3]) / TAU_I, abs=1e-2)


@pytest.mark.parametrize("ab", [(1.0, 1.0), (0.85, 0.85), (0.7, 1.0), (1.2, 1.2)])
def test_slowest_closed_loop_mode_is_minutes_not_days(ab):
    """Non-saturated points: slowest eigen-timescale must be O(tau_i), i.e. < 60 min."""
    p = _params(*ab)
    y = ph.closed_loop_steady_state_exact(p, ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)
    J = np.asarray(jax.jacfwd(lambda yy: ph.cstr_closed_loop_rhs(0.0, yy, (p, ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)))(y), float)
    ev = np.linalg.eigvals(J)
    ev = ev[np.abs(ev) > 1e-9]
    assert np.all(ev.real < 0)
    assert 1.0 / np.min(np.abs(ev.real)) < 60.0


def test_slowest_mode_healthy_is_about_tau_i():
    p = _params(1.0, 1.0)
    y = ph.closed_loop_steady_state_exact(p, ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)
    J = np.asarray(jax.jacfwd(lambda yy: ph.cstr_closed_loop_rhs(0.0, yy, (p, ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)))(y), float)
    ev = np.linalg.eigvals(J)
    assert 1.0 / np.min(np.abs(ev.real)) == pytest.approx(TAU_I, rel=0.1)


@pytest.mark.parametrize("ab", [(1.0, 0.7), (1.0, 0.4), (0.4, 0.4)])
def test_saturated_branch_is_fixed_point(ab):
    """Sc2/Sc5-type points: valve pinned, integrator gated (dI=0), rhs ~ 0."""
    p = _params(*ab)
    y = ph.closed_loop_steady_state_exact(p, ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)
    rhs = np.asarray(ph.cstr_closed_loop_rhs(0.0, y, (p, ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)))
    assert np.abs(rhs).max() < 1e-3
    qc = float(ph.compute_qc(y[1], y[3], ph.NOMINAL_CTRL))
    assert qc in (ph.QC_MIN, ph.QC_MAX)


def test_steady_state_is_differentiable():
    """NUTS (matched_warm_start=True) differentiates through this solve."""
    f = lambda a, b: ph.closed_loop_steady_state_exact(_params(a, b), ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL)[0]
    g = jax.grad(f, argnums=(0, 1))(0.85, 0.85)
    assert all(np.isfinite(float(x)) for x in g)
