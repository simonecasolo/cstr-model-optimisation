"""Prior distributions over the CSTR inference parameter vector.

M4 deliverable (BoxUniform for SBI); also consumed by M5 (NumPyro priors
for NUTS).  See plan §3 M4 and spec §3.3.

Parameter vector conventions (revised 2026-05-18)
--------------------------------------------------
**Primary — 2-D (standard inference):**
    theta = [alpha, beta]

    alpha [dimensionless] catalyst-activity factor  prior: Uniform[0.4, 1.0]
    beta  [dimensionless] jacket-conductance factor prior: Uniform[0.4, 1.0]

UA and k0 are **fixed design constants** (UA_NOMINAL, K0_NOMINAL from
``physics.py``) and are NOT inferred. They enter the ODE as constants, not
as sampled parameters.  The 4-D ``[UA, k0, alpha, beta]`` ODE parameter
vector is used internally in the simulator with UA and k0 always set to their
nominal values; the inference layer only touches ``[alpha, beta]``.

**Sensor-drift extension — 4-D (Sc7 substudy):**
    theta_ext = [alpha, beta, delta_T, delta_Ci]

    delta_T   [K]       additive T-sensor drift   prior: Uniform[-3, 3]
    delta_Ci  [mol/L]   additive Ci-sensor drift  prior: Uniform[-0.1, 0.1]

All bounds are stored as module-level constants so notebooks and the
NumPyro generative model reference a single source of truth.
"""

from __future__ import annotations

import numpy as np
import numpyro
import numpyro.distributions as dist

from cstr_sbi.physics import K0_NOMINAL, UA_NOMINAL

# ---------------------------------------------------------------------------
# Prior bounds (single source of truth)
# ---------------------------------------------------------------------------

# Primary inference parameters (both bounded [0.4, 1.0]).
ALPHA_LOW:  float = 0.40   # covers Sc5 severe-fouling / heavy decay
ALPHA_HIGH: float = 1.00

BETA_LOW:   float = 0.40
BETA_HIGH:  float = 1.00

# Sensor-drift extension bounds (Sc7 / 4-D extended vector).
DELTA_T_LOW:   float = -3.0    # K
DELTA_T_HIGH:  float =  3.0    # K

DELTA_CI_LOW:  float = -0.10   # mol/L
DELTA_CI_HIGH: float =  0.10   # mol/L

# Legacy 4-D bounds retained for backward compatibility.  UA and k0 are
# NOT inferred — these are only used by the 4-D ODE call wrappers that need
# to fill a full [UA, k0, alpha, beta] params array.
UA_LOW:   float = UA_NOMINAL
UA_HIGH:  float = UA_NOMINAL
K0_LOW:   float = K0_NOMINAL
K0_HIGH:  float = K0_NOMINAL

# Convenience numpy arrays.
PRIOR_LOW_2D  = np.array([ALPHA_LOW,  BETA_LOW],  dtype=np.float32)
PRIOR_HIGH_2D = np.array([ALPHA_HIGH, BETA_HIGH], dtype=np.float32)

PRIOR_LOW_4D_EXT  = np.array([ALPHA_LOW, BETA_LOW, DELTA_T_LOW,  DELTA_CI_LOW],  dtype=np.float32)
PRIOR_HIGH_4D_EXT = np.array([ALPHA_HIGH, BETA_HIGH, DELTA_T_HIGH, DELTA_CI_HIGH], dtype=np.float32)

# Keep old names as aliases so existing code that imports PRIOR_LOW_4D doesn't break.
PRIOR_LOW_4D  = np.array([UA_LOW,  K0_LOW,  ALPHA_LOW,  BETA_LOW],  dtype=np.float32)
PRIOR_HIGH_4D = np.array([UA_HIGH, K0_HIGH, ALPHA_HIGH, BETA_HIGH], dtype=np.float32)

PARAM_NAMES_2D = ("alpha", "beta")
PARAM_NAMES_4D = ("UA", "k0", "alpha", "beta")          # ODE internal
PARAM_NAMES_4D_EXT = ("alpha", "beta", "delta_T", "delta_Ci")  # extended inference


# ---------------------------------------------------------------------------
# sbi-compatible BoxUniform priors (M4, torch-based)
# ---------------------------------------------------------------------------

def _get_box_uniform(low_arr, high_arr):
    try:
        import torch
        from sbi.utils import BoxUniform
    except ImportError as e:
        raise ImportError("sbi and torch must be installed. Run: pip install sbi torch") from e
    return BoxUniform(
        low=torch.tensor(low_arr, dtype=torch.float32),
        high=torch.tensor(high_arr, dtype=torch.float32),
    )


def box_uniform_2d():
    """**Primary** 2-D BoxUniform prior ``[alpha, beta]`` for sbi / SNPE_C.

    alpha and beta are the degradation factors inferred from observations.
    UA and k0 are fixed at their nominal values and are not inferred.
    """
    return _get_box_uniform(PRIOR_LOW_2D, PRIOR_HIGH_2D)


def box_uniform_4d_ext():
    """4-D BoxUniform prior ``[alpha, beta, delta_T, delta_Ci]`` for the
    sensor-drift substudy (Scenario 7 / Scenario 9).
    """
    return _get_box_uniform(PRIOR_LOW_4D_EXT, PRIOR_HIGH_4D_EXT)


def box_uniform_4d():
    """Deprecated alias kept for backward compatibility.

    Previously this was the 4-D prior ``[UA, k0, alpha, beta]``.
    Now UA and k0 are fixed constants — use ``box_uniform_2d()`` instead.
    """
    import warnings
    warnings.warn(
        "box_uniform_4d() is deprecated. Use box_uniform_2d() — UA and k0 "
        "are fixed design constants and should not be inferred.",
        DeprecationWarning,
        stacklevel=2,
    )
    return box_uniform_2d()


# ---------------------------------------------------------------------------
# NumPyro priors (M5, used inside the NUTS generative model)
# ---------------------------------------------------------------------------

def sample_numpyro_prior_2d(prefix: str = "") -> dict:
    """**Primary** — sample the 2-D prior ``[alpha, beta]`` inside a NumPyro model.

    Returns a dict ``{"alpha": ..., "beta": ...}`` of NumPyro sample sites.
    UA and k0 are fixed constants; do not sample them.
    """
    return {
        "alpha": numpyro.sample(f"{prefix}alpha", dist.Uniform(ALPHA_LOW, ALPHA_HIGH)),
        "beta":  numpyro.sample(f"{prefix}beta",  dist.Uniform(BETA_LOW,  BETA_HIGH)),
    }


def sample_numpyro_prior_4d_ext(prefix: str = "") -> dict:
    """Sample the 4-D extended prior ``[alpha, beta, delta_T, delta_Ci]``."""
    d = sample_numpyro_prior_2d(prefix=prefix)
    d["delta_T"]  = numpyro.sample(f"{prefix}delta_T",  dist.Uniform(DELTA_T_LOW,  DELTA_T_HIGH))
    d["delta_Ci"] = numpyro.sample(f"{prefix}delta_Ci", dist.Uniform(DELTA_CI_LOW, DELTA_CI_HIGH))
    return d


def prior_dict_to_params_array(d: dict) -> "jnp.ndarray":
    """Pack a 2-D prior-sample dict into the 4-element ``[UA, k0, alpha, beta]``
    ODE params array, filling UA and k0 at their nominal constants.
    """
    import jax.numpy as jnp
    return jnp.array([UA_NOMINAL, K0_NOMINAL, d["alpha"], d["beta"]])
