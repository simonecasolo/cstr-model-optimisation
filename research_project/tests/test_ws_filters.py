"""F1 (Phase 0 / Phase 3 step a3): the EKF/UKF dynamics use the FIXED controller law of physics.py.

(1) f and h equal physics.cstr_closed_loop_rhs / compute_qc at random states (nominal inlet and with an inlet offset);
(2) analytic Jacobians equal central finite differences (unsaturated and saturated states);
(3) dQc/dI = Kp/tau_i, the integrator initialisation inverts the controller;
(4) the filters run (EKF and UKF, 6- and 8-state) and return finite estimates.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("CSTR_SENSOR_MODEL", "N3")
pytest.importorskip("jax")
import numpy as np
import jax.numpy as jnp

import cstr_sbi.physics as ph
import ws_filters as F

RNG = np.random.default_rng(20261109)


def _states(n=6, augment=False):
    out = []
    for _ in range(n):
        x = [RNG.uniform(0.008, 0.035), RNG.uniform(311.0, 314.5), RNG.uniform(297.0, 303.0), RNG.uniform(-3.0, 15.0),
             RNG.uniform(0.5, 1.2), RNG.uniform(0.5, 1.2)]
        if augment:
            x += [RNG.uniform(-0.2, 0.2), RNG.uniform(-0.2, 0.2)]
        out.append(np.array(x))
    return out


def _physics_rhs(x, inlet):
    p = jnp.array([ph.UA_NOMINAL, ph.K0_NOMINAL, x[4], x[5]], dtype=jnp.float32)
    return np.asarray(ph.cstr_closed_loop_rhs(0.0, jnp.array(x[:4], dtype=jnp.float32), (p, jnp.array(inlet, dtype=jnp.float32), ph.NOMINAL_CTRL)))


@pytest.mark.parametrize("x", _states(6))
def test_dynamics_equal_physics_nominal_inlet(x):
    f = F.Plant().rhs(x)[:4]
    assert np.allclose(f, _physics_rhs(x, [F.CI0, F.TI0, F.TCI0]), rtol=2e-3, atol=2e-5), (f, _physics_rhs(x, [F.CI0, F.TI0, F.TCI0]))


@pytest.mark.parametrize("x", _states(5, augment=True))
def test_dynamics_equal_physics_with_inlet_offset(x):
    ti, tci = 297.7, 296.4                                       # measured inlet temperatures
    f = F.Plant(ti, tci, augment=True).rhs(x)[:4]
    ref = _physics_rhs(x, [F.CI0, ti - x[6], tci - x[7]])        # plant sees measured - offset
    assert np.allclose(f, ref, rtol=2e-3, atol=2e-5), (f, ref)


@pytest.mark.parametrize("x", _states(5))
def test_measurement_equals_compute_qc(x):
    h = F.Plant().h(x)
    qc = float(ph.compute_qc(jnp.float32(x[1]), jnp.float32(x[3]), ph.NOMINAL_CTRL))
    assert np.allclose(h[:3], x[:3]) and abs(h[3] - qc) < 1e-3 * max(1.0, abs(qc))


def _fd_jac(fun, x, n_out, eps=1e-6):
    J = np.zeros((n_out, len(x)))
    for j in range(len(x)):
        d = np.zeros(len(x)); d[j] = eps * max(1.0, abs(x[j]))
        J[:, j] = (fun(x + d) - fun(x - d)) / (2 * d[j])
    return J


@pytest.mark.parametrize("augment", [False, True])
def test_jacobians_equal_finite_differences(augment):
    pl = F.Plant(297.4, 296.9, augment=augment)
    for x in _states(6, augment):
        # keep the states unsaturated, away from the valve limits
        x[3] = np.clip(x[3], -1.0, 8.0)
        qc_raw = F.QC0 + F.KP * ((x[1] - F.TSP) + x[3] / F.TAUI)
        if not (5.0 < qc_raw < 395.0):
            continue
        A, Afd = pl.jac_A(x), _fd_jac(pl.rhs, x, pl.n)
        assert np.allclose(A, Afd, rtol=1e-4, atol=1e-6 * np.abs(Afd).max()), np.abs(A - Afd).max()
        H, Hfd = pl.jac_H(x), _fd_jac(pl.h, x, 4)
        assert np.allclose(H, Hfd, rtol=1e-5, atol=1e-6), np.abs(H - Hfd).max()


def test_saturated_state_has_zero_valve_sensitivity():
    pl = F.Plant()
    x = np.array([0.018, 313.5, 298.0, 40.0, 1.0, 0.5])          # unclamped Qc far above 400
    assert F.QC0 + F.KP * ((x[1] - F.TSP) + x[3] / F.TAUI) > F.QCMAX
    A, H = pl.jac_A(x), pl.jac_H(x)
    assert A[3, 1] == 0.0 and A[2, 3] == 0.0 and np.all(H[3] == 0.0)    # gate closed, no dQc/dT, dQc/dI
    assert pl.h(x)[3] == F.QCMAX


def test_integral_gain_and_integrator_initialisation():
    pl = F.Plant()
    x = np.array([0.018, 312.5, 300.0, 1.0, 1.0, 1.0])
    d = 1e-4
    dq = (pl.h(x + [0, 0, 0, d, 0, 0])[3] - pl.h(x - [0, 0, 0, d, 0, 0])[3]) / (2 * d)
    assert dq == pytest.approx(F.KP / F.TAUI, rel=1e-6)                 # dQc/dI = Kp/tau_i (the old law had 1/tau_i)
    for qc in (20.0, 60.0, 130.0, 300.0):
        for T in (312.4, 312.5, 312.7):
            x0 = F.make_x0([0.018, T, 300.0, qc])
            assert abs(pl.h(x0)[3] - qc) < 1e-3, (qc, T, pl.h(x0)[3])


@pytest.mark.parametrize("method", ["ekf", "ukf"])
@pytest.mark.parametrize("augment", [False, True])
def test_filters_run_and_return_finite_estimates(method, augment):
    pl = F.Plant()
    ss = np.asarray(ph.closed_loop_steady_state_exact(jnp.array([ph.UA_NOMINAL, ph.K0_NOMINAL, 1.0, 1.0], dtype=jnp.float32), ph.NOMINAL_INLET_CL, ph.NOMINAL_CTRL), float)
    x = np.array([ss[0], ss[1], ss[2], ss[3], 1.0, 1.0])
    obs = np.array([pl.h(x)] * 40) + RNG.normal(size=(40, 4)) * F.N3_SIGMA
    r = F.estimate_window(obs, method, augment=augment)
    assert np.isfinite([r["alpha"], r["beta"], r["sd_alpha"], r["sd_beta"]]).all()
    assert 0.3 <= r["alpha"] <= 1.3 and 0.3 <= r["beta"] <= 1.3
