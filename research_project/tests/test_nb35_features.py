"""nb35: the plug-in estimators recomputed from window means equal summary_variants.extra_features,
and the measured-inlet features behave as specified."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("jax")
os.environ.setdefault("CSTR_SENSOR_MODEL", "N3")

import summary_variants as sv          # noqa: E402
import nb35_measured_inlet as M        # noqa: E402


def _raw(n=64, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([0.018, 312.5, 300.0, 80.0])
    return base + rng.normal(size=(n, 120, 4)) * [0.01, 0.2, 0.2, 2.0]


def test_estimators_match_extra_features_with_per_window_inlet():
    raw = _raw()
    t = np.arange(1, 121) * 0.5
    rng = np.random.default_rng(1)
    ti, tci = 297 + rng.uniform(-2, 2, len(raw)), 297 + rng.uniform(-2, 2, len(raw))
    const = sv.physics_constants(); const["TI"], const["TCI"] = ti, tci
    ref = sv.extra_features(raw, t, const)[:, :3]
    m = raw.mean(axis=1)
    got = M.estimators(m[:, 0], m[:, 1], m[:, 2], m[:, 3], ti, tci)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_feature_shapes_and_oracle():
    b29 = np.zeros((10, 29), np.float32)
    b29[:, M.I_C], b29[:, M.I_T], b29[:, M.I_TC], b29[:, M.I_QC] = 0.018, 312.5, 300.0, 80.0
    inl = np.tile(np.array([[0.97, 297.5, 296.5]], np.float32), (10, 1))
    assert M.features_for("marginal", b29, inl, 0).shape == (10, 32)
    x = M.features_for("oracle", b29, inl, 0)
    assert x.shape == (10, 34)
    np.testing.assert_allclose(x[:, -2:], [[0.5, -0.5]] * 10, atol=1e-5)     # oracle = exact inlet
    x2 = M.features_for("eps0.2", b29, inl, 0)
    assert np.abs(x2[:, -2:] - [[0.5, -0.5]]).max() > 0.01                 # offsets applied
