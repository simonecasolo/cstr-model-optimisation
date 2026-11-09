"""Smoke test of the N3 sensor model plumbing (~1-2 min). Run before the N3 notebook sequence.

Checks, for the closed-loop P1b simulator and the open-loop simulator:
  * the module-level tags resolve to the N3 names (29fx_N3, 31_N3, 03s_N3);
  * the injected sensor noise is exactly the specified one. Method: same seed with the sensor
    noise switched off (noise_abs = 0 on every channel). The process-noise path and the sensor
    key are identical, so (noisy - sensor-free) is the sensor noise alone:
      - N3 (CFG.noise_kw()): sd per channel = (0.01, 0.2, 0.2, 2.0) within 5%;
      - N0 (default path, no noise_abs): sd / (0.5% of the window maximum) = 1 within 5%,
        i.e. the inherited convention is unchanged.

History: the first version (2026-11-09) estimated the noise from first differences of the noisy
trajectory, which is invalid for closed-loop Qc: Kp = 150 turns each process-noise kick on T into
a Qc jump (first-difference sd ~7.7 L/min with no sensor noise at all). Replaced by the same-seed
difference after the agent's diagnosis (logs/N3_step0_sensor_noise_diag.py), before any N3 run.

    python scripts/smoke_N3.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
os.environ["CSTR_SENSOR_MODEL"] = "N3"

import numpy as np

import system_I_config as CFG
import ws_eval as we
import nb29_ws_study as s29
import nb31_budget_convergence as b31
import nb03_summary_study as s03

assert CFG.model() == "N3" and s29.TAG == "29fx_N3" and b31.TAG == "31_N3" and s03.TAG == "03s_N3", \
    (CFG.model(), s29.TAG, b31.TAG, s03.TAG)
assert b31.E2_TAG == "29fx_N3" and s03.E2_TAG == "29fx_N3"
print("tags OK:", s29.TAG, b31.TAG, s03.TAG)

theta = np.tile(np.array([[0.85, 0.85]], np.float32), (200, 1))
expect = np.array(CFG.SENSOR_MODELS["N3"])
OFF = {"noise_abs": (0.0, 0.0, 0.0, 0.0)}
TOL = 0.05

for name, fn in [("closed P1b", lambda **k: we.simulate(theta, "P1b", seed=11, raw=True, **k)),
                 ("open loop", lambda **k: we.simulate_open_loop(theta, seed=11, raw=True, **k))]:
    x_off, x3, x0 = fn(**OFF), fn(**CFG.noise_kw()), fn()
    assert x_off.shape == x3.shape == x0.shape and np.isfinite(x3).all(), name
    sd3 = (x3 - x_off).std(axis=(0, 1))
    mx = np.maximum(np.abs(x_off).max(axis=1, keepdims=True), 1e-12)        # per window and channel
    r0 = ((x0 - x_off) / (0.005 * mx)).std(axis=(0, 1))
    print(f"{name:11s} N3 injected sd = {np.round(sd3, 4)} (expected {expect});  "
          f"N0 injected sd / (0.5% of max) = {np.round(r0, 3)} (expected 1)")
    assert np.all(np.abs(sd3 / expect - 1) < TOL), f"{name}: N3 sensor noise {sd3} != {expect}"
    assert np.all(np.abs(r0 - 1) < TOL), f"{name}: N0 default noise path changed: {r0}"
    assert abs(float(x3[..., 1].mean() - x_off[..., 1].mean())) < 0.05, f"{name}: T mean shifted by the sensor layer"
# Production summary set (2026-11-09, user decision after nb03 section 10): N3 -> S1_base_est (32 columns), N0 -> S0_base29 (29)
assert CFG.summary_set() == "S1_base_est" and CFG.summary_set("N3") == "S1_base_est", CFG.summary_set()
assert CFG.summary_set("N0") == "S0_base29", CFG.summary_set("N0")
assert b31.SET == "S1_base_est"
th_s = theta[:20]
x_s1 = we.simulate(th_s, "P2", seed=11, features=CFG.summary_set(), **CFG.noise_kw())
x_s0 = we.simulate(th_s, "P2", seed=11, features=CFG.summary_set("N0"), **CFG.noise_kw("N0"))
x_dflt = we.simulate(th_s, "P2", seed=11, **CFG.noise_kw("N0"))
assert x_s1.shape == (20, 32) and x_s0.shape == (20, 29) and np.array_equal(x_s0, x_dflt), (x_s1.shape, x_s0.shape)
assert np.isfinite(x_s1).all()
print(f"summary sets OK: N3 -> {CFG.summary_set()} {x_s1.shape[1]} columns; N0 -> {CFG.summary_set('N0')} {x_s0.shape[1]} columns")
print("smoke test passed")
