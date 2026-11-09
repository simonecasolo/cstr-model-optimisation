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
    """A +0.5 K feed error raises Qc and the integral action keeps T at the setpoint; at +1.0 K the healthy plant needs
    ~345 L/min of coolant and the jacket reaches its capacity limit (valve saturation), so T is no longer held.
    (2026-11-09: the original test used +1 K and assumed no saturation; at +1 K 27% of the samples are at Qc_max and
    mean T moves by +0.07 K. See runbook section 6.)"""
    th = np.tile(np.array([[1.0, 1.0]], np.float32), (16, 1))
    a = wi.simulate(th, wi.NOMINAL, "P2", 7, raw=True, **CFG.noise_kw("N3"))
    warm = wi.NOMINAL + np.array([0, 0.5, 0], np.float32)
    b = wi.simulate(th, warm, "P2", 7, raw=True, **CFG.noise_kw("N3"))
    dqc = b[..., 3].mean() - a[..., 3].mean()
    assert dqc > 5.0, dqc          # a warmer feed needs more cooling: Qc rises
    assert abs(b[..., 1].mean() - a[..., 1].mean()) < 0.05   # T stays at setpoint (integral action)
    hot = wi.NOMINAL + np.array([0, 1.0, 0], np.float32)
    c = wi.simulate(th, hot, "P2", 7, raw=True, **CFG.noise_kw("N3"))
    frac_sat = float(np.mean(c[..., 3] >= 395.0))
    print(f"fraction of samples at Qc >= 395 L/min with a +1.0 K feed error: {frac_sat:.3f} (nominal: {float(np.mean(a[..., 3] >= 395.0)):.3f})")
    assert frac_sat > 0.0          # the jacket-capacity limit is reached at +1 K
    assert float(np.mean(a[..., 3] >= 395.0)) == 0.0
