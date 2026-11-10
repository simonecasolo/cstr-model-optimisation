"""Fast numpy tests of the System III package (no torch/sbi needed)."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from system_iii.physics import PLANT, ss_unsat, ss_closed, beta_saturation, ss_open_all  # noqa: E402
from system_iii.simulator import simulate  # noqa: E402
from system_iii.features import features, plugins, base29, NAMES  # noqa: E402
from system_iii.data import make_windows, scenario_set  # noqa: E402


def test_nominal_design_point():
    s = ss_unsat(PLANT, 1.0, 1.0)
    assert abs(float(s["X"]) - 0.842) < 0.002
    assert abs(float(s["Qc"]) - PLANT.Qc0) < 0.1
    assert abs(float(s["Tc"]) - PLANT.Tci - 33.9) < 0.2          # coolant rise ~34 K
    assert abs(float(s["Qc"]) / PLANT.Qcmax - 0.35) < 0.005
    assert abs(beta_saturation(PLANT) - 0.391) < 0.005


def test_unique_safe_closed_loop_steady_states():
    for a in np.linspace(0.4, 1.2, 9):
        for b in np.linspace(0.4, 1.2, 9):
            sols = ss_closed(PLANT, a, b)
            assert len(sols) == 1, (a, b, sols)
            assert sols[0]["T"] <= PLANT.Tmax_safe


def test_open_loop_multistability_documented():
    # the reactor is multistable at a fixed coolant flow (why it needs control): see docs
    assert len(ss_open_all(PLANT, 0.75, 0.75, PLANT.Qc0)) == 3
    assert len(ss_open_all(PLANT, 1.0, 0.7, PLANT.Qc0)) == 1


def test_simulator_reproducible_and_at_setpoint():
    a = simulate(np.ones(50), np.ones(50), seed=3)
    b = simulate(np.ones(50), np.ones(50), seed=3)
    assert np.array_equal(a["obs"], b["obs"])
    assert a["obs"].shape == (50, 120, 4)
    assert abs(a["true"][:, :, 1].mean() - PLANT.Tsp) < 0.05
    assert abs(a["qc_true"].mean() - PLANT.Qc0) < 3.0


def test_control_offset_moves_plant_drift_does_not():
    off = simulate(np.ones(80), np.ones(80), seed=5, oT=0.5)
    assert abs(off["true"][:, :, 1].mean() - (PLANT.Tsp - 0.5)) < 0.05      # plant moves
    assert abs(off["obs"][:, :, 1].mean() - PLANT.Tsp) < 0.05               # reading at setpoint
    dr = simulate(np.ones(80), np.ones(80), seed=5, drift_T_monitor=2.0)
    assert abs(dr["true"][:, :, 1].mean() - PLANT.Tsp) < 0.05
    assert abs(dr["obs"][:, :, 1].mean() - PLANT.Tsp - 2.0) < 0.05


def test_open_loop_mode_holds_coolant_flow():
    o = simulate(np.ones(20), np.full(20, 0.7), seed=1, control="fixed")
    assert np.allclose(o["qc_true"], PLANT.Qc0)
    assert abs(o["true"][:, :, 1].mean() - 340.2) < 0.3


def test_plugins_exact_on_noise_free_steady_state():
    for a, b in [(1.0, 1.0), (0.7, 1.0), (1.0, 0.7), (0.8, 0.6)]:
        s = ss_unsat(PLANT, a, b)
        obs = np.tile(np.array([[[float(s["C"]), float(s["T"]), float(s["Tc"]), float(s["Qc"])]]]), (1, 4, 1))
        pl = plugins(obs, np.array([PLANT.Ti]), np.array([PLANT.Tci]))[0]
        np.testing.assert_allclose(pl, [np.log(a), np.log(b), np.log(b)], atol=1e-9)


def test_legacy_ua_proxy_is_alpha_confounded():
    t = np.arange(1, 5) * 0.5
    v = []
    for a in (1.0, 0.7):
        s = ss_unsat(PLANT, a, 1.0)
        obs = np.tile(np.array([[[float(s["C"]), float(s["T"]), float(s["Tc"]), float(s["Qc"])]]]), (1, 4, 1))
        v.append(base29(obs, t)[0, NAMES["S0"].index("UA_eff_proxy")])
    assert abs(v[0] - v[1]) / v[0] > 0.03       # changes with alpha at fixed beta


def test_feature_sets_and_windows():
    d = make_windows(np.tile([1.0, 1.0], (30, 1)), seed=2, return_raw=True)
    assert d["x"].shape == (30, len(NAMES["S1m"])) == (30, 34)
    assert np.isfinite(d["x"]).all()
    sc = scenario_set(["Sc1_healthy", "Sc10_control_offset", "Sc6_open_fouling"], 5, seed=3)
    assert sc["x"].shape[0] == 15


def test_s5_cascade_holds_setpoint_and_samples_analyser():
    import numpy as np
    from system_iii.simulator import simulate
    from system_iii.physics import PLANT
    o = simulate(np.full(40, 0.7), np.full(40, 0.7), seed=3, sensors="S5", control="cascade",
                 cascade=(4, 8, 20, 1), pv_filter_min=0.1)
    T = o["true"][:, :, 1]
    assert abs(T.mean() - PLANT.Tsp) < 0.05 and T.std(1).mean() < 0.4
    levels = np.mean([len(np.unique(r)) for r in o["obs"][:, :, 0]])
    assert 11 <= levels <= 14          # one analyser value every 5 min in a 60-min window
