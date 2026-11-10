"""Sensor models for System III: white-noise sd per logged channel [C mol/L, T K, Tc K, Qc L/min], and the
calibration and inlet-measurement models.

S3 (production, the analogue of System I's N3): online analyser at ~1% of a 3 mol/L span, class-A/AA RTDs,
coolant flowmeter at 0.5% of the 288 L/min span.
S0 (the inherited System I convention, 0.5% of each channel's typical value, for the sensor-model fragility study).
S1: realistic C only; S2: realistic temperatures only; S4: S3 without a usable concentration measurement.
"""
import numpy as np

SENSORS = {
    "S3": (0.03, 0.2, 0.2, 1.44),
    "S0": (0.005 * 0.46, 0.005 * 337.0, 0.005 * 323.0, 0.005 * 101.0),
    "S1": (0.03, 0.005 * 337.0, 0.005 * 323.0, 0.005 * 101.0),
    "S2": (0.005 * 0.46, 0.2, 0.2, 1.44),
    "S4": (1.0, 0.2, 0.2, 1.44),
    "S5": (0.03, 0.2, 0.2, 1.44),   # S3 noise + realistic sensor dynamics (SENSOR_DYN below)
}

# Sensor dynamics (2026-11-10, realism upgrade). Sets not listed here are static (instantaneous readings).
# analyser_period / analyser_dead: online PO analyser (GC/NIR class) reports a new value every period,
#   measured on a sample taken `dead` minutes earlier, held between updates (sample-and-hold); the sampling
#   phase is random per window.
# tw_T / tw_Tc: first-order thermowell time constants (min) of the reactor and jacket-outlet RTDs. The
#   controller sees the lagged reactor reading.
SENSOR_DYN = {
    "S5": dict(analyser_period=5.0, analyser_dead=2.0, tw_T=0.5, tw_Tc=0.5),
}

# calibration offsets (sd) of the production instrument set
CAL_SD = {"oT": 0.1, "oTc": 0.1, "uQc": 0.01, "uC": 0.01}
# inlet measurements: calibration offset sd and per-sample white noise (averaged over the window)
INLET_EPS = 0.1
INLET_WHITE = 0.2
# realistic plant variation (unmeasured or measured), for training and robustness tests
INLET_RANGE = 2.0      # K, Ti and Tci ~ nominal +- INLET_RANGE
CI_REL_SD = 0.01       # feed PO concentration, unmeasured
Q_REL_SD = 0.005       # feed flow, flowmeter offset


def measured_inlet(Ti, Tci, rng, eps=INLET_EPS, n_samples=120):
    """Window-mean measured inlet temperatures: true + calibration offset + averaged white noise."""
    n = len(Ti)
    w = INLET_WHITE / np.sqrt(n_samples)
    return (Ti + eps * rng.standard_normal(n) + w * rng.standard_normal(n),
            Tci + eps * rng.standard_normal(n) + w * rng.standard_normal(n))


def realistic_conditions(n, rng, plant, *, inlet_range=INLET_RANGE, calibration=True):
    """Draw per-window true inlet and sensor calibration offsets for the production training distribution."""
    out = dict(Ti=plant.Ti + rng.uniform(-inlet_range, inlet_range, n),
               Tci=plant.Tci + rng.uniform(-inlet_range, inlet_range, n),
               Ci=plant.Ci * (1 + CI_REL_SD * rng.standard_normal(n)),
               Q=plant.Q * (1 + Q_REL_SD * rng.standard_normal(n)))
    if calibration:
        out.update(oT=CAL_SD["oT"] * rng.standard_normal(n), oTc=CAL_SD["oTc"] * rng.standard_normal(n),
                   uQc=CAL_SD["uQc"] * rng.standard_normal(n), uC=CAL_SD["uC"] * rng.standard_normal(n))
    return out
