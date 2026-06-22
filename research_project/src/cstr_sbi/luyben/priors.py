"""Prior distributions for the Luyben plant 8-D inference problem.

Parameter vector:
    theta = [alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta]

All parameters are degradation/shift factors relative to nominal.
Values of 1.0 (or 0.0 for delta) represent healthy operation.
"""

from __future__ import annotations

import numpy as np

from cstr_sbi.luyben.physics import PARAM_NAMES

# ---------------------------------------------------------------------------
# Prior bounds (single source of truth)
# ---------------------------------------------------------------------------

# alpha: catalyst activity. 1.0 = nominal; 0.4 = severe decay
ALPHA_LOW:   float = 0.40
ALPHA_HIGH:  float = 1.20

# beta_r: CSTR jacket fouling factor
BETA_R_LOW:  float = 0.40
BETA_R_HIGH: float = 1.20

# eta_sep: flash separator split efficiency
ETA_SEP_LOW:  float = 0.40
ETA_SEP_HIGH: float = 1.20

# beta_s: separator HEX fouling factor
BETA_S_LOW:  float = 0.40
BETA_S_HIGH: float = 1.20

# eta_p: recycle pump efficiency
ETA_P_LOW:  float = 0.40
ETA_P_HIGH: float = 1.20

# xi: purge valve restriction (>1 = erosion, <1 = blockage)
XI_LOW:  float = 0.40
XI_HIGH: float = 1.60   # wider upper bound: erosion can increase flow

# kappa: feed preheater fouling (1.0 = nominal preheat; 0.4 = cold feed)
KAPPA_LOW:  float = 0.40
KAPPA_HIGH: float = 1.20

# delta: feed A:B stoichiometry shift (-0.3 = A-lean, +0.3 = A-rich)
DELTA_LOW:  float = -0.30
DELTA_HIGH: float =  0.30

PRIOR_LOW_8D  = np.array(
    [ALPHA_LOW, BETA_R_LOW, ETA_SEP_LOW, BETA_S_LOW,
     ETA_P_LOW, XI_LOW, KAPPA_LOW, DELTA_LOW],
    dtype=np.float32,
)
PRIOR_HIGH_8D = np.array(
    [ALPHA_HIGH, BETA_R_HIGH, ETA_SEP_HIGH, BETA_S_HIGH,
     ETA_P_HIGH, XI_HIGH, KAPPA_HIGH, DELTA_HIGH],
    dtype=np.float32,
)

# Fault classification thresholds (used in metrics and scenarios)
HEALTHY_THRESHOLDS = {
    "alpha":   0.85,
    "beta_r":  0.85,
    "eta_sep": 0.85,
    "beta_s":  0.85,
    "eta_p":   0.85,
    "xi_lo":   0.85,  # below this = blockage fault
    "xi_hi":   1.15,  # above this = erosion fault
    "kappa":   0.85,
    "delta_abs": 0.15,  # |delta| above this = stoichiometry fault
}


# ---------------------------------------------------------------------------
# sbi BoxUniform prior
# ---------------------------------------------------------------------------

def box_uniform_8d():
    """8-D BoxUniform prior for sbi / SNPE_C.

    Returns a ``sbi.utils.BoxUniform`` over
    [alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta].
    """
    try:
        import torch
        from sbi.utils import BoxUniform
    except ImportError as e:
        raise ImportError("sbi and torch must be installed. Run: pip install sbi torch") from e

    return BoxUniform(
        low=torch.tensor(PRIOR_LOW_8D,  dtype=torch.float32),
        high=torch.tensor(PRIOR_HIGH_8D, dtype=torch.float32),
    )
