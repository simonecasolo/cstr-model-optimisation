"""Production summary features (added 2026-11-09, user decision: S1_base_est for N3).

(a) features(raw, t, "S0_base29") is ws.summaries exactly;
(b) features(raw, t, "S1_base_est") is nb03's _union(...)[:, SETS["S1_base_est"]] on the same raw windows;
(c) simulate() / simulate_open_loop() with the default arguments are bit-identical to outputs stored BEFORE the
    change (logs/ref_default_path_before_S1.npz, made by scripts/make_default_path_reference.py);
plus routing: system_I_config.summary_set() is S0 for N0 and S1 for N3 and the S1 matrix has 32 columns.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
os.environ["CSTR_SENSOR_MODEL"] = "N3"          # nb03_summary_study reads the sensor model at import

pytest.importorskip("jax")
import numpy as np

import system_I_config as CFG
import ws_eval as we
import ws_protocols as ws
import summary_variants as sv
import make_default_path_reference as ref

N3 = CFG.noise_kw("N3")


def _raw(n=40, seed=91, protocol="P2"):
    th = we.prior_draws(n, 31)
    obs = we.simulate(th, protocol, seed, raw=True, **N3)
    t = np.arange(1, obs.shape[1] + 1) * 0.5
    return th, obs, t


def test_summary_set_routing():
    assert CFG.summary_set("N0") == "S0_base29"
    assert CFG.summary_set("N3") == "S1_base_est"


def test_features_S0_equals_ws_summaries():
    import jax.numpy as jnp
    _, obs, t = _raw()
    a = we.features(obs, t, "S0_base29")
    b = ws.summaries(jnp.asarray(obs), jnp.asarray(t))
    assert a.shape == (len(obs), 29)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_features_S1_equals_nb03_union():
    import nb03_summary_study as s03
    th, _, _ = _raw()
    seed = 91
    union = s03._union(th, seed)[:, sv.SETS["S1_base_est"]]
    prod = we.simulate(th, s03.PROTOCOL, seed, features="S1_base_est", **N3)
    assert prod.shape == (len(th), 32) and union.shape == prod.shape
    assert np.array_equal(union, prod)
    # the first 29 columns are the S0 summaries
    assert np.array_equal(prod[:, :29], we.simulate(th, s03.PROTOCOL, seed, **N3))


def test_unknown_set_raises():
    _, obs, t = _raw(n=4)
    with pytest.raises(ValueError):
        we.features(obs, t, "S9_nonexistent")


def test_default_path_bit_identical_to_pre_change_reference():
    assert ref.OUT.exists(), "reference outputs missing: logs/ref_default_path_before_S1.npz"
    r = np.load(ref.OUT)
    th = ref.make_theta()
    assert np.array_equal(th, r["theta"])
    for name, fn in ref.CASES.items():
        assert np.array_equal(np.asarray(fn(th)), r[name + "__summ"]), name + " (summaries)"
        assert np.array_equal(np.asarray(fn(th, raw=True)), r[name + "__raw"]), name + " (raw)"


def test_open_loop_features_dimension():
    th = we.prior_draws(8, 5)
    assert we.simulate_open_loop(th, seed=1, **N3).shape == (8, 29)
    assert we.simulate_open_loop(th, seed=1, features="S1_base_est", **N3).shape == (8, 32)
