"""Fault scenario configurations for the Wu 2003 CSTR-column-recycle plant.

16 closed-loop fault scenarios (W1–W16) + 7 open-loop counterparts (W1ol–W7ol).
Each scenario specifies the 5-D degradation parameter vector:
    theta = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]

Scenario IDs:
    W1   healthy (nominal)
    W2   catalyst decay (alpha=0.65)
    W3   CSTR jacket fouling (beta_r=0.65)
    W4   column tray efficiency loss (eta_col=0.70)
    W5   reboiler duty reduction (xi_reb=0.65)
    W6   feed composition shift lean (z_A0_eff=0.75)
    W7   feed composition shift rich (z_A0_eff=0.95)
    W8   moderate catalyst decay (alpha=0.80)
    W9   moderate jacket fouling (beta_r=0.80)
    W10  moderate column degradation (eta_col=0.80)
    W11  snowball: catalyst decay + column degradation
    W12  reactor faults: alpha + beta_r combined
    W13  column faults: eta_col + xi_reb combined
    W14  feed + catalyst: z_A0 shift + alpha decay
    W15  severe multi-unit: alpha + beta_r + eta_col degraded
    W16  recovery scenario: mild faults across all units
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from cstr_sbi.recycle.physics import (
    NOMINAL_THETA, PARAM_NAMES, Z0_NOM,
    NOMINAL_INLET, NOMINAL_CTRL, NOMINAL_Y0,
)


@dataclass(frozen=True)
class RecycleScenarioConfig:
    """One row of the Wu 2003 recycle scenario truth table."""

    id: int
    name: str
    alpha:     float
    beta_r:    float
    eta_col:   float
    xi_reb:    float
    z_A0_eff:  float
    mode: str          # "closed_loop" | "open_loop"
    description: str

    def theta(self) -> jnp.ndarray:
        """Return the 5-D degradation parameter vector."""
        return jnp.array(
            [self.alpha, self.beta_r, self.eta_col,
             self.xi_reb, self.z_A0_eff],
            dtype=jnp.float32,
        )

    def fault_unit(self) -> str:
        """Primary degraded unit (for classification labels)."""
        if self.alpha < 0.9 or self.beta_r < 0.9:
            return "reactor"
        elif self.eta_col < 0.85 or self.xi_reb < 0.85:
            return "column"
        elif abs(self.z_A0_eff - float(Z0_NOM)) > 0.05:
            return "feed"
        else:
            return "healthy"


def _sc(
    id, name,
    alpha=1.0, beta_r=1.0, eta_col=1.0, xi_reb=1.0,
    z_A0_eff=float(Z0_NOM),
    mode="closed_loop", description="",
):
    return RecycleScenarioConfig(
        id=id, name=name,
        alpha=alpha, beta_r=beta_r, eta_col=eta_col,
        xi_reb=xi_reb, z_A0_eff=z_A0_eff,
        mode=mode, description=description,
    )


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS: dict[str, RecycleScenarioConfig] = {
    # --- Closed-loop scenarios ---
    "W1_healthy": _sc(
        1, "W1_healthy",
        description="Healthy plant, all parameters nominal.",
    ),
    "W2_cat_decay": _sc(
        2, "W2_cat_decay", alpha=0.65,
        description="Severe catalyst decay (alpha=0.65). Reaction rate drops 35%.",
    ),
    "W3_rxr_fouling": _sc(
        3, "W3_rxr_fouling", beta_r=0.65,
        description="Severe CSTR jacket fouling (beta_r=0.65). UA reduced to 65%.",
    ),
    "W4_col_tray_eff": _sc(
        4, "W4_col_tray_eff", eta_col=0.70,
        description="Column tray efficiency loss (eta_col=0.70). Separation degraded.",
    ),
    "W5_reb_starve": _sc(
        5, "W5_reb_starve", xi_reb=0.65,
        description="Reboiler duty reduction (xi_reb=0.65). Under-separation at bottom.",
    ),
    "W6_feed_lean": _sc(
        6, "W6_feed_lean", z_A0_eff=0.75,
        description="Feed composition shift lean (z_A0=0.75, A-lean by 15%).",
    ),
    "W7_feed_rich": _sc(
        7, "W7_feed_rich", z_A0_eff=0.95,
        description="Feed composition shift rich (z_A0=0.95, A-rich by 5%).",
    ),
    "W8_mild_cat": _sc(
        8, "W8_mild_cat", alpha=0.80,
        description="Moderate catalyst decay (alpha=0.80). Early-stage degradation.",
    ),
    "W9_mild_fouling": _sc(
        9, "W9_mild_fouling", beta_r=0.80,
        description="Moderate jacket fouling (beta_r=0.80). Early-stage fouling.",
    ),
    "W10_mild_col": _sc(
        10, "W10_mild_col", eta_col=0.80,
        description="Moderate column degradation (eta_col=0.80).",
    ),
    "W11_snowball": _sc(
        11, "W11_snowball", alpha=0.65, eta_col=0.75,
        description=(
            "Snowball scenario: catalyst decay (alpha=0.65) reduces conversion, "
            "column degrades (eta_col=0.75), recycle builds up."
        ),
    ),
    "W12_rxr_combined": _sc(
        12, "W12_rxr_combined", alpha=0.75, beta_r=0.75,
        description="Combined reactor faults: kinetic decay + jacket fouling.",
    ),
    "W13_col_combined": _sc(
        13, "W13_col_combined", eta_col=0.70, xi_reb=0.75,
        description="Combined column faults: tray efficiency + reboiler starvation.",
    ),
    "W14_feed_cat": _sc(
        14, "W14_feed_cat", alpha=0.75, z_A0_eff=0.78,
        description="Feed composition shift + catalyst decay (confounded diagnosis).",
    ),
    "W15_severe_multi": _sc(
        15, "W15_severe_multi",
        alpha=0.60, beta_r=0.70, eta_col=0.70,
        description="Severe multi-unit: reactor kinetics + fouling + column degradation.",
    ),
    "W16_mild_all": _sc(
        16, "W16_mild_all",
        alpha=0.90, beta_r=0.90, eta_col=0.90, xi_reb=0.90, z_A0_eff=0.87,
        description="Mild uniform degradation across all units — hardest to classify.",
    ),
    # --- Open-loop counterparts (controller bypassed, T_j fixed) ---
    "W1_healthy_ol": _sc(
        101, "W1_healthy_ol", mode="open_loop",
        description="Healthy plant, open-loop (PI temperature loop bypassed).",
    ),
    "W2_cat_decay_ol": _sc(
        102, "W2_cat_decay_ol", alpha=0.65, mode="open_loop",
        description="Catalyst decay, open-loop.",
    ),
    "W3_rxr_fouling_ol": _sc(
        103, "W3_rxr_fouling_ol", beta_r=0.65, mode="open_loop",
        description="CSTR jacket fouling, open-loop.",
    ),
    "W4_col_tray_eff_ol": _sc(
        104, "W4_col_tray_eff_ol", eta_col=0.70, mode="open_loop",
        description="Column tray efficiency loss, open-loop.",
    ),
    "W5_reb_starve_ol": _sc(
        105, "W5_reb_starve_ol", xi_reb=0.65, mode="open_loop",
        description="Reboiler starvation, open-loop.",
    ),
    "W11_snowball_ol": _sc(
        111, "W11_snowball_ol", alpha=0.65, eta_col=0.75, mode="open_loop",
        description="Snowball scenario, open-loop.",
    ),
    "W15_severe_multi_ol": _sc(
        115, "W15_severe_multi_ol",
        alpha=0.60, beta_r=0.70, eta_col=0.70, mode="open_loop",
        description="Severe multi-unit fault, open-loop.",
    ),
}


def list_configs(mode: str | None = None) -> list[RecycleScenarioConfig]:
    """Return scenarios in numerical order, optionally filtered by mode."""
    configs = sorted(SCENARIO_CONFIGS.values(), key=lambda s: s.id)
    if mode is not None:
        configs = [c for c in configs if c.mode == mode]
    return configs


def list_closed_loop_configs() -> list[RecycleScenarioConfig]:
    """Return the 16 closed-loop scenario configs."""
    return list_configs(mode="closed_loop")


def list_open_loop_configs() -> list[RecycleScenarioConfig]:
    """Return the 7 open-loop scenario configs."""
    return list_configs(mode="open_loop")


# ---------------------------------------------------------------------------
# Convenience: get warm-start IC for a given scenario
# ---------------------------------------------------------------------------

def get_warm_ic(
    scenario: RecycleScenarioConfig,
    t_warm: float = 200.0,
) -> jnp.ndarray:
    """Integrate scenario to near-SS and return [z_A, T_r, T_j, I_T] as IC.

    Uses the closed-loop (or open-loop) physics as appropriate.
    t_warm in hours; default 200 h = ~80 residence times.
    """
    from cstr_sbi.recycle.physics import simulate_to_steady_state

    theta = scenario.theta()

    if scenario.mode == "open_loop":
        # For open-loop: Kp1=0 → controller inactive, jacket stays at bias Qj0
        ctrl_ol = NOMINAL_CTRL.at[0].set(0.0)
        return simulate_to_steady_state(theta, NOMINAL_INLET, ctrl_ol,
                                         NOMINAL_Y0, t_final=t_warm)
    else:
        return simulate_to_steady_state(theta, NOMINAL_INLET, NOMINAL_CTRL,
                                         NOMINAL_Y0, t_final=t_warm)
