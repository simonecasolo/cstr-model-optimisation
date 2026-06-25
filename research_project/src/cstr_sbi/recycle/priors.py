"""Prior distributions for the Wu 2003 recycle CSTR 5-D inference problem.

Parameter vector:
    theta = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]

All multiplicative factors (alpha, beta_r, eta_col, xi_reb) equal 1.0 at
healthy nominal operation; z_A0_eff equals Z0_NOM = 0.90.
"""

from __future__ import annotations

import numpy as np

from cstr_sbi.recycle.physics import PARAM_NAMES, Z0_NOM


# ---------------------------------------------------------------------------
# Prior bounds
# ---------------------------------------------------------------------------

# alpha: catalyst/kinetic activity (decay lowers rate)
ALPHA_LOW:   float = 0.40
ALPHA_HIGH:  float = 1.20

# beta_r: CSTR jacket heat-transfer fouling (lower = worse)
BETA_R_LOW:  float = 0.40
BETA_R_HIGH: float = 1.20

# eta_col: distillation column tray efficiency (lower = worse separation)
ETA_COL_LOW:  float = 0.50
ETA_COL_HIGH: float = 1.00

# xi_reb: reboiler duty scale factor (1.0 = nominal)
XI_REB_LOW:  float = 0.40
XI_REB_HIGH: float = 1.20

# z_A0_eff: effective fresh-feed A composition (nominal = 0.90)
Z_A0_LOW:  float = float(Z0_NOM) - 0.20   # 0.70
Z_A0_HIGH: float = float(Z0_NOM) + 0.05   # 0.95

PRIOR_LOW_5D = np.array(
    [ALPHA_LOW, BETA_R_LOW, ETA_COL_LOW, XI_REB_LOW, Z_A0_LOW],
    dtype=np.float32,
)
PRIOR_HIGH_5D = np.array(
    [ALPHA_HIGH, BETA_R_HIGH, ETA_COL_HIGH, XI_REB_HIGH, Z_A0_HIGH],
    dtype=np.float32,
)

# Fault classification thresholds
HEALTHY_THRESHOLDS = {
    "alpha":      0.85,
    "beta_r":     0.85,
    "eta_col":    0.80,
    "xi_reb_lo":  0.85,   # below this = reboiler starvation
    "xi_reb_hi":  1.15,   # above this = reboiler flooding
    "z_A0_dev":   0.05,   # |z_A0 - 0.90| > this = feed composition fault
}


def box_uniform_5d():
    """5-D BoxUniform prior for sbi / SNPE_C.

    Returns a ``sbi.utils.BoxUniform`` over
    [alpha, beta_r, eta_col, xi_reb, z_A0_eff].
    """
    try:
        import torch
        from sbi.utils import BoxUniform
    except ImportError as e:
        raise ImportError(
            "sbi and torch must be installed. Run: pip install sbi torch"
        ) from e

    return BoxUniform(
        low=torch.tensor(PRIOR_LOW_5D,  dtype=torch.float32),
        high=torch.tensor(PRIOR_HIGH_5D, dtype=torch.float32),
    )
