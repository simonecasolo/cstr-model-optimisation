"""ws_inlet (notebook 34): nominal identity with the production simulator, and the inlet acting."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("jax")
pytest.importorskip("torch")
os.environ.setdefault("CSTR_SENSOR_MODEL", "N3")

import system_I_config as CFG   # noqa: E402
import ws_inlet as wi           # noqa: E402


@pytest.mark.parametrize("protocol", ["P1b", "P2"])
def test_nominal_identity(protocol):
    r = wi.nominal_identity_check(protocol, n=16, **CFG.noise_kw("N3"))
    sig = np.array(CFG.SENSOR_MODELS["N3"])
    assert np.all(np.array(r["max_abs_diff_per_channel"]) < 0.01 * sig), r


def test_feed_temperature_moves_the_plant():
    th = np.tile(np.array([[1.0, 1.0]], np.float32), (16, 1))
    a = wi.simulate(th, wi.NOMINAL, "P2", 7, raw=True, **CFG.noise_kw("N3"))
    hot = wi.NOMINAL + np.array([0, 1.0, 0], np.float32)
    b = wi.simulate(th, hot, "P2", 7, raw=True, **CFG.noise_kw("N3"))
    dqc = b[..., 3].mean() - a[..., 3].mean()
    assert dqc > 5.0, dqc          # a hotter feed needs more cooling: Qc rises
    assert abs(b[..., 1].mean() - a[..., 1].mean()) < 0.05   # T stays at setpoint (integral action)
