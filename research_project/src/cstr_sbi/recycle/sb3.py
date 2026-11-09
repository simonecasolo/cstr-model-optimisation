"""Reduced-scope helpers for the Wu2003 System II S-B3 framing.

S-B3 is not a new instrumentation structure. It reuses the existing "S-B"
measurement structure and simulator exactly, but narrows the *fault taxonomy*
and scenario catalogue by treating feed quality (`z_A0_eff`) as exogenous to the
plant. In practice this means:

- exclude all closed-loop scenarios with a feed fault component
- keep only the 4 endogenous degradation parameters
  [alpha, beta_r, eta_col, xi_reb]
- evaluate fault attribution on the reduced 4-class unit taxonomy
  [healthy, reactor, column, multi]

This module centralizes that narrowed framing so notebooks can stay consistent.
"""

from __future__ import annotations

import numpy as np

from cstr_sbi.recycle.physics import PARAM_NAMES, Z0_NOM
from cstr_sbi.recycle.priors import (
    PRIOR_HIGH_5D,
    PRIOR_LOW_5D,
    LOGN_SIGMA_5D,
    NOMINAL_THETA_5D,
    box_uniform_5d,
    mixed_wide_prior,
    truncated_lognormal_5d,
)
from cstr_sbi.recycle.scenarios import CLOSED_LOOP_NAMES, RecycleScenarioConfig, list_closed_loop
from cstr_sbi.recycle.simulator import (
    DEFAULT_N_REPLICATES,
    DEFAULT_N_SAVE,
    DEFAULT_SENSOR_NOISE,
    DEFAULT_T_FINAL_H,
    deterministic_window,
    nominal_warm_start,
    noisy_replicates,
    scenario_warm_start,
)

SB3_PARAM_NAMES = tuple(PARAM_NAMES[:4])
SB3_PARAM_INDICES = (0, 1, 2, 3)
Z_A0_EFF_INDEX = PARAM_NAMES.index("z_A0_eff")

PRIOR_LOW_SB3_4D = PRIOR_LOW_5D[:4].copy()
PRIOR_HIGH_SB3_4D = PRIOR_HIGH_5D[:4].copy()
NOMINAL_THETA_SB3_4D = NOMINAL_THETA_5D[:4].copy()
LOGN_SIGMA_SB3_4D = LOGN_SIGMA_5D[:4].copy()

# Widened lognormal sigma for the large-scale SBC campaign (2026-09-25).
#
# `LOGN_SIGMA_SB3_4D` puts the worst named S-B3 fault value only ~2.2
# log-sigma into eta_col/xi_reb's tail (eta_col=0.80 in W7_col_eff_mild/
# W12_snowball_compound, xi_reb=0.70 in W9_reb_fouling). A 20-seed sweep
# (notebooks/31c, results/31c_extended_seed_sweep_confirm_sbc.csv) confirmed
# this is not enough coverage: 0/20 seeds pass full SBC, with eta_col/xi_reb
# the persistent worst offenders (KS p as low as 1e-32).
#
# `LOGN_SIGMA_SB3_4D_WIDE` widens sigma so the worst named fault sits at
# exactly 1 log-sigma instead of ~2.2. Since median=1.0, sigma = -log(worst):
#   eta_col: worst=0.80 -> sigma = -log(0.80) = 0.2231 (was 0.10)
#   xi_reb:  worst=0.70 -> sigma = -log(0.70) = 0.3567 (was 0.16)
# The alpha/beta_r entries are carried over unchanged and unused in practice,
# since those two dimensions stay Uniform under `mixed_wide_prior_4d_sb3`'s
# default `uniform_idx=(0, 1)`.
#
# This deliberately keeps the truncated-*lognormal* family (not a one-sided
# half-normal, which was considered and rejected this session) so xi_reb
# keeps prior mass above 1.0. xi_reb's box [0.40, 1.20] has nominal=1.0 in
# its interior (unlike eta_col, whose box already caps at 1.0), and 11 of the
# 12 retained S-B3 scenarios have true xi_reb=1.0 -- clipping all density
# above the mode would put the *healthy* truth exactly on the support's edge,
# trading today's tail-coverage problem for a new boundary-bias one.
LOGN_SIGMA_SB3_4D_WIDE = np.array(
    [LOGN_SIGMA_SB3_4D[0], LOGN_SIGMA_SB3_4D[1], 0.2231, 0.3567],
    dtype=np.float32,
)

TRUE_UNIT_THRESH_SB3 = 0.90
PRED_UNIT_THRESH_SB3 = 0.90
Z0_NOMINAL = float(Z0_NOM)
TRUE_Z0_TOL = 0.05

SB3_EXCLUDED_SCENARIOS = ("W10_feed_impurity", "W13_cat_feed")
CLOSED_LOOP_NAMES_SB3 = [name for name in CLOSED_LOOP_NAMES if name not in SB3_EXCLUDED_SCENARIOS]
FAULT_UNITS_SB3 = ["healthy", "reactor", "column", "multi"]


def ensure_theta4_sb3(theta4: np.ndarray) -> np.ndarray:
    """Validate and normalize an S-B3 4-D parameter vector.

    S-B3's public parameterization is strictly `[alpha, beta_r, eta_col, xi_reb]`.
    Any free `z_A0_eff` must be rejected at this layer rather than silently ignored.
    """
    theta4 = np.asarray(theta4, dtype=np.float32)
    if theta4.shape[-1] != 4:
        raise ValueError(f"S-B3 expects a 4-D theta [...,4], got shape {theta4.shape}")
    return theta4


def theta5_to_theta4(theta5: np.ndarray) -> np.ndarray:
    theta5 = np.asarray(theta5, dtype=np.float32)
    return theta5[..., :4].copy()


def theta4_to_theta5(theta4: np.ndarray, z_a0_eff: float = Z0_NOMINAL) -> np.ndarray:
    theta4 = ensure_theta4_sb3(theta4)
    z_col = np.full(theta4.shape[:-1] + (1,), float(z_a0_eff), dtype=np.float32)
    return np.concatenate([theta4, z_col], axis=-1)


def scenario_from_theta4_sb3(
    theta4: np.ndarray,
    *,
    scenario_id: int = -1,
    name: str = "sb3_synthetic",
    mode: str = "closed_loop",
    description: str = "synthetic S-B3 scenario",
) -> RecycleScenarioConfig:
    """Create a strict S-B3 scenario by injecting nominal `z_A0_eff` internally."""
    alpha, beta_r, eta_col, xi_reb = ensure_theta4_sb3(theta4)
    return RecycleScenarioConfig(
        id=scenario_id,
        name=name,
        alpha=float(alpha),
        beta_r=float(beta_r),
        eta_col=float(eta_col),
        xi_reb=float(xi_reb),
        z_A0_eff=Z0_NOMINAL,
        mode=mode,
        description=description,
    )


def coerce_scenario_sb3(scenario_or_theta4, **scenario_kwargs) -> RecycleScenarioConfig:
    """Return a scenario guaranteed to satisfy the reduced S-B3 scope.

    Accepts either an existing `RecycleScenarioConfig` with nominal `z_A0_eff`, or a
    strict 4-D S-B3 parameter vector which is promoted to a scenario on the fly.
    """
    if isinstance(scenario_or_theta4, RecycleScenarioConfig):
        if scenario_has_feed_fault(scenario_or_theta4):
            raise ValueError(
                f"Scenario {scenario_or_theta4.name!r} is outside S-B3 scope: z_A0_eff={scenario_or_theta4.z_A0_eff}"
            )
        return scenario_or_theta4
    return scenario_from_theta4_sb3(scenario_or_theta4, **scenario_kwargs)


def scenario_has_feed_fault(scenario, z0_nom: float = Z0_NOMINAL, z0_tol: float = TRUE_Z0_TOL) -> bool:
    return abs(float(scenario.z_A0_eff) - z0_nom) > z0_tol


def list_closed_loop_sb3():
    return [sc for sc in list_closed_loop() if not scenario_has_feed_fault(sc)]


def nominal_warm_start_sb3():
    """Strict S-B3 wrapper: identical to `S-B`, but with no free structure string."""
    return nominal_warm_start("S-B")


def scenario_warm_start_sb3(scenario_or_theta4, *, t_final: float = 200.0, **scenario_kwargs):
    """Strict 4-D warm-start wrapper for S-B3."""
    scenario = coerce_scenario_sb3(scenario_or_theta4, **scenario_kwargs)
    return scenario_warm_start(scenario, structure="S-B", t_final=t_final)


def deterministic_window_sb3(
    scenario_or_theta4,
    *,
    y0=None,
    t_final_h: float = DEFAULT_T_FINAL_H,
    n_save: int = DEFAULT_N_SAVE,
    **scenario_kwargs,
):
    """Strict 4-D deterministic-window wrapper for S-B3.

    The caller may pass either a reduced 4-D theta or an in-scope scenario object. In
    both cases `z_A0_eff` is fixed internally to nominal and never accepted as a free
    input.
    """
    scenario = coerce_scenario_sb3(scenario_or_theta4, **scenario_kwargs)
    if y0 is None:
        y0 = nominal_warm_start_sb3()
    return deterministic_window(
        scenario,
        structure="S-B",
        y0=y0,
        t_final_h=t_final_h,
        n_save=n_save,
    )


def noisy_replicates_sb3(
    base_obs: np.ndarray,
    *,
    n_replicates: int = DEFAULT_N_REPLICATES,
    rng: np.random.Generator,
    noise_pct: float = DEFAULT_SENSOR_NOISE,
) -> np.ndarray:
    """Noise wrapper kept only to give S-B3 callers a 4-D-specific import surface."""
    return noisy_replicates(base_obs, n_replicates=n_replicates, rng=rng, noise_pct=noise_pct)


def fault_unit_sb3_truth(theta_like, thresh: float = TRUE_UNIT_THRESH_SB3) -> str:
    theta = np.asarray(theta_like, dtype=np.float32)
    alpha, beta_r, eta_col, xi_reb = theta[:4]
    reactor_bad = (alpha < thresh) or (beta_r < thresh)
    column_bad = (eta_col < thresh) or (xi_reb < thresh)
    n_bad = int(reactor_bad) + int(column_bad)
    if n_bad == 0:
        return "healthy"
    if n_bad >= 2:
        return "multi"
    if reactor_bad:
        return "reactor"
    return "column"


def fault_unit_sb3_predict(theta_like, thresh: float = PRED_UNIT_THRESH_SB3) -> str:
    theta = np.asarray(theta_like, dtype=np.float32)
    alpha, beta_r, eta_col, xi_reb = theta[:4]
    reactor_bad = (alpha < thresh) or (beta_r < thresh)
    column_bad = (eta_col < thresh) or (xi_reb < thresh)
    n_bad = int(reactor_bad) + int(column_bad)
    if n_bad == 0:
        return "healthy"
    if n_bad >= 2:
        return "multi"
    if reactor_bad:
        return "reactor"
    return "column"


def classify_fault_sb3(samples: np.ndarray, classes=FAULT_UNITS_SB3) -> dict:
    labels = [fault_unit_sb3_predict(row) for row in samples]
    probs = {u: float(np.mean([label == u for label in labels])) for u in classes}
    predicted = max(probs, key=probs.__getitem__)
    return {
        "class": predicted,
        "probs": probs,
        "means": {p: float(np.mean(samples[:, i])) for i, p in enumerate(SB3_PARAM_NAMES)},
    }


def box_uniform_4d_sb3():
    prior5 = box_uniform_5d()
    try:
        import torch
        from sbi.utils import BoxUniform
    except ImportError as e:
        raise ImportError("sbi and torch must be installed. Run: pip install sbi torch") from e
    return BoxUniform(
        low=torch.tensor(PRIOR_LOW_SB3_4D, dtype=torch.float32),
        high=torch.tensor(PRIOR_HIGH_SB3_4D, dtype=torch.float32),
    )


def truncated_lognormal_4d_sb3():
    return truncated_lognormal_5d(
        median=NOMINAL_THETA_SB3_4D,
        sigma=LOGN_SIGMA_SB3_4D,
        low=PRIOR_LOW_SB3_4D,
        high=PRIOR_HIGH_SB3_4D,
    )


def mixed_wide_prior_4d_sb3(uniform_idx=(0, 1), sigma=None):
    """Strict 4-D S-B3 mixed-wide prior: by default `alpha, beta_r` (indices 0, 1)
    ~ independent `Uniform(low, high)`, `eta_col, xi_reb` (indices 2, 3) keep a
    truncated-lognormal marginal.

    `alpha` and `beta_r` are widened because their named-fault severities sit the
    deepest into the narrow lognormal prior's tail (down to ~3.2-3.7 log-sigma,
    <1% relative density at e.g. W4's alpha=0.55 or W6's beta_r=0.60) -- the same
    coverage gap already diagnosed for the full 5-D system in
    `cstr_sbi.recycle.priors.MixedAlphaUniformPrior`/`MixedWidePrior`.

    `sigma` selects the lognormal spread for the non-uniform dimensions
    (`eta_col`, `xi_reb` by default). Defaults to `LOGN_SIGMA_SB3_4D` (the
    original narrow calibration, ~2.2 log-sigma to the worst named fault) for
    backward compatibility. Pass `LOGN_SIGMA_SB3_4D_WIDE` (worst named fault at
    exactly 1 log-sigma) for the 2026-09-25 large-scale SBC campaign -- a 20-seed
    sweep under the narrow default (`notebooks/31c`) found 0/20 seeds pass full
    SBC, with `eta_col`/`xi_reb` the persistent worst offenders.
    """
    if sigma is None:
        sigma = LOGN_SIGMA_SB3_4D
    return mixed_wide_prior(
        uniform_idx=uniform_idx,
        low=PRIOR_LOW_SB3_4D,
        high=PRIOR_HIGH_SB3_4D,
        median=NOMINAL_THETA_SB3_4D,
        sigma=sigma,
    )
