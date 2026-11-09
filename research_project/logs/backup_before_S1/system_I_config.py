"""System I study configuration: the sensor-noise model used for training, test sets and data.

Production sensor model since 2026-11-09 (user decision after notebook 33): **N3**, realistic
sensors on every channel (sigma_C = 0.01 mol/L, sigma_T = sigma_Tc = 0.2 K, sigma_Qc = 2 L/min).
N0 (the earlier convention, 0.5% of each channel's window maximum) is kept for the SI.

The model is selected by the environment variable CSTR_SENSOR_MODEL (default "N3"). Every study
notebook sets it explicitly in its first cell and reloads the study modules, so a kernel that
imported an older setting cannot silently run the wrong model. Outputs of non-N0 runs carry the
suffix "_<model>" (e.g. results/31_N3_production.json), so N0 results are never overwritten.
"""
from __future__ import annotations

import os

SENSOR_MODELS = {
    "N0": None,                        # 0.5% of each channel's window maximum (pre-2026-11-09)
    "N3": (0.01, 0.2, 0.2, 2.0),       # realistic: analyser 1% of 1 mol/L span, RTD, flowmeter
}
DEFAULT_MODEL = "N3"
PROTOCOL = "P1b"           # N0 production protocol (kept for reference); use protocol() instead
HISTORY_MIN = {"P0": 0.0, "P1": 0.0, "P1b": 150.0, "P2": 300.0, "P3": 0.0}   # pre-window history


def protocol(m: str | None = None) -> str:
    """Warm-start protocol for this sensor model, as selected by nb29 (results/29fx<suffix>_selection.json)
    and, if the selection was flagged or overridden, as approved in ..._selection_review.json.
    N0: P1b (flagged selection, approved by the user 2026-11-08). N3: P2 (unflagged selection under
    the pre-set rule; adopted by the user 2026-11-09, with P1b reported side by side)."""
    import json
    from pathlib import Path
    R = Path(__file__).resolve().parent.parent / "results"
    tag = "29fx" + suffix(m)
    f = R / f"{tag}_selection.json"
    if not f.exists():
        raise FileNotFoundError(f"results/{tag}_selection.json missing: run nb29 for sensor model {m or model()} first")
    sel = json.loads(f.read_text())
    rv = R / f"{tag}_selection_review.json"
    rev = json.loads(rv.read_text()) if rv.exists() else {}
    if sel.get("flag") and not (rev.get("approved") and rev.get("choice")):
        raise RuntimeError(f"nb29 selection for {m or model()} is flagged and not approved: {sel['flag']}")
    choice = rev["choice"] if rev.get("approved") and rev.get("choice") else sel["choice"]
    if rev.get("choice") and rev["choice"] != sel["choice"] and not rev.get("approved"):
        raise RuntimeError("review file disagrees with the selection and is not approved")
    return choice


def model() -> str:
    m = os.environ.get("CSTR_SENSOR_MODEL", DEFAULT_MODEL)
    if m not in SENSOR_MODELS:
        raise ValueError(f"unknown CSTR_SENSOR_MODEL={m!r}; choose from {list(SENSOR_MODELS)}")
    return m


def noise_kw(m: str | None = None) -> dict:
    """Keyword arguments for ws_eval.simulate / simulate_open_loop / make_test_set."""
    na = SENSOR_MODELS[m or model()]
    return {} if na is None else {"noise_abs": tuple(None if x is None else float(x) for x in na)}


def suffix(m: str | None = None) -> str:
    m = m or model()
    return "" if m == "N0" else f"_{m}"


def describe(m: str | None = None) -> str:
    m = m or model()
    na = SENSOR_MODELS[m]
    if na is None:
        return "N0: sigma = 0.5% of each channel's window maximum"
    return f"{m}: sigma_C={na[0]} mol/L, sigma_T={na[1]} K, sigma_Tc={na[2]} K, sigma_Qc={na[3]} L/min"
