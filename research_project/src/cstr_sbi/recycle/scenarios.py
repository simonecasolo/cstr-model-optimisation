"""Fault scenario configurations for the Wu 2003 CSTR-column-recycle plant.

16 closed-loop scenarios (W1–W16) matching the article outline exactly, plus
7 open-loop counterparts (W2-OL through W8-OL) with all PI loops bypassed.

Degradation vector: theta = [alpha, beta_r, eta_col, xi_reb, z_A0_eff]

Scenario design rationale (from project_wu2003_sbi.md):
    W1       Healthy nominal
    W2–W4    Catalyst decay (mild → threshold)
    W5–W6    Reactor jacket fouling (mild → severe)
    W7–W8    Column tray efficiency loss (mild → severe)
    W9       Reboiler HX fouling
    W10      Feed purity degradation
    W11      Combined reactor faults (alpha + beta_r)
    W12      HEADLINE: (alpha, eta_col) banana posterior under S-B
    W13      Catalyst decay + lean feed
    W14      Combined column faults (eta_col + xi_reb)
    W15      Near snowball tipping point (strong EKF failure)
    W16      Full multi-parameter degradation
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from cstr_sbi.recycle.physics import PARAM_NAMES, Z0_NOM


@dataclass(frozen=True)
class RecycleScenarioConfig:
    """One row of the Wu 2003 recycle scenario truth table."""

    id: int
    name: str
    alpha:    float
    beta_r:   float
    eta_col:  float
    xi_reb:   float
    z_A0_eff: float
    mode: str          # "closed_loop" | "open_loop"
    description: str

    def theta(self) -> jnp.ndarray:
        return jnp.array(
            [self.alpha, self.beta_r, self.eta_col, self.xi_reb, self.z_A0_eff],
            dtype=jnp.float32,
        )

    def fault_unit(self) -> str:
        """Primary degraded unit for hierarchical classification."""
        n = self.name
        if "healthy" in n:
            return "healthy"
        if "cat_" in n or "jacket_" in n or "reactor_" in n:
            return "reactor"
        if "col_" in n or "reb_" in n:
            return "column"
        if "feed_" in n:
            return "feed"
        if "snowball" in n or "multi" in n:
            return "multi"
        # fallback: check parameter deviations
        if self.alpha < 0.90 or self.beta_r < 0.90:
            return "reactor"
        if self.eta_col < 0.90 or self.xi_reb < 0.90:
            return "column"
        if abs(self.z_A0_eff - float(Z0_NOM)) > 0.05:
            return "feed"
        return "healthy"


_Z0 = float(Z0_NOM)   # 0.90


def _sc(
    id, name,
    alpha=1.0, beta_r=1.0, eta_col=1.0, xi_reb=1.0, z_A0_eff=_Z0,
    mode="closed_loop", description="",
):
    return RecycleScenarioConfig(
        id=id, name=name,
        alpha=alpha, beta_r=beta_r, eta_col=eta_col,
        xi_reb=xi_reb, z_A0_eff=z_A0_eff,
        mode=mode, description=description,
    )


# ---------------------------------------------------------------------------
# Scenario catalogue — 16 closed-loop + 7 open-loop
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS: dict[str, RecycleScenarioConfig] = {

    # ---- W1: healthy nominal -------------------------------------------
    "W1_healthy": _sc(
        1, "W1_healthy",
        description="Nominal healthy plant, all parameters at 1.0.",
    ),

    # ---- W2–W4: catalyst decay (reactor kinetics) ----------------------
    "W2_cat_mild": _sc(
        2, "W2_cat_mild", alpha=0.85,
        description="Mild catalyst decay (alpha=0.85). Early snowball onset visible in F_R.",
    ),
    "W3_cat_severe": _sc(
        3, "W3_cat_severe", alpha=0.65,
        description="Severe catalyst decay (alpha=0.65). Pronounced snowball; increased Q_c.",
    ),
    "W4_cat_threshold": _sc(
        4, "W4_cat_threshold", alpha=0.55,
        description=(
            "Catalyst near snowball critical point (alpha=0.55). "
            "Strong nonlinearity; EKF linearisation breaks down."
        ),
    ),

    # ---- W5–W6: reactor jacket fouling (beta_r) -----------------------
    "W5_jacket_mild": _sc(
        5, "W5_jacket_mild", beta_r=0.80,
        description=(
            "Mild jacket fouling (beta_r=0.80). T_r held at setpoint by Loop 1; "
            "Q_c increases; T_j drops. Analogue of PO Sc2."
        ),
    ),
    "W6_jacket_severe": _sc(
        6, "W6_jacket_severe", beta_r=0.60,
        description="Severe jacket fouling (beta_r=0.60). Large Q_c and T_j deviation.",
    ),

    # ---- W7: column tray efficiency loss (eta_col) --------------------
    # W8 (eta_col=0.65) removed: ODE blows up (T_r, z_A → inf) from the nominal
    # warm start under S-B. The QSS column bisection at eta_col=0.65 returns
    # physically inconsistent values (x_D=0.9998, x_B=1.0), which drives a
    # positive-feedback runaway in the reactor dynamics. eta_col=0.65 is outside
    # the stable operating range of the shortcut column model.
    "W7_col_eff_mild": _sc(
        7, "W7_col_eff_mild", eta_col=0.80,
        description=(
            "Mild column tray efficiency loss (eta_col=0.80). "
            "x_D drifts under S-B; x_B increases; Q_reb rises."
        ),
    ),

    # ---- W9: reboiler HX fouling (xi_reb) -----------------------------
    "W9_reb_fouling": _sc(
        9, "W9_reb_fouling", xi_reb=0.70,
        description=(
            "Reboiler HX fouling (xi_reb=0.70). More steam needed for same boilup; "
            "Q_reb increases. Loop 3 compensates; partially masked."
        ),
    ),

    # ---- W10: feed purity (z_A0_eff) ----------------------------------
    "W10_feed_impurity": _sc(
        10, "W10_feed_impurity", z_A0_eff=0.78,
        description=(
            "Feed contains 12% impurity (z_A0=0.78, lean by 0.12). "
            "Lower conversion; distinct steady-state shift in z_A and F_R."
        ),
    ),

    # ---- W11: combined reactor faults ---------------------------------
    "W11_reactor_combined": _sc(
        11, "W11_reactor_combined", alpha=0.80, beta_r=0.80,
        description=(
            "Both reactor faults: catalyst decay (alpha=0.80) + jacket fouling "
            "(beta_r=0.80). Competing signals in Q_c."
        ),
    ),

    # ---- W12: HEADLINE — (alpha, eta_col) banana posterior -----------
    "W12_snowball_compound": _sc(
        12, "W12_snowball_compound", alpha=0.75, eta_col=0.80,
        description=(
            "HEADLINE scenario. Catalyst decay (alpha=0.75) + column efficiency "
            "(eta_col=0.80) both increase recycle via snowball effect. "
            "Joint (alpha, eta_col) posterior is banana-shaped under S-B; "
            "narrows under S-A (x_D measurement breaks degeneracy). "
            "EKF collapses banana to overconfident Gaussian ellipse."
        ),
    ),

    # ---- W13: catalyst decay + lean feed ------------------------------
    "W13_cat_feed": _sc(
        13, "W13_cat_feed", alpha=0.80, z_A0_eff=0.80,
        description=(
            "Catalyst decay (alpha=0.80) + lean feed (z_A0=0.80). "
            "Confounded diagnosis: both suppress conversion."
        ),
    ),

    # W14 removed: eta_col=0.75 + xi_reb=0.75 produces Inf values in the S-B 2h
    # window (225 Inf timesteps), which propagate to NaN in noisy replicates via
    # sigma = noise_pct * max|obs| = Inf. Same QSS column boundary issue as W8.

    # ---- W15: near snowball tipping point ----------------------------
    "W15_snowball_threshold": _sc(
        15, "W15_snowball_threshold", alpha=0.58, eta_col=0.90,
        description=(
            "Near snowball tipping point (alpha=0.58). Strong nonlinearity: "
            "EKF Jacobian changes rapidly; SBI correctly widens uncertainty. "
            "EKF 90% CI achieves < 65% empirical coverage."
        ),
    ),

    # ---- W16: full multi-parameter degradation -----------------------
    "W16_full_multi": _sc(
        16, "W16_full_multi",
        alpha=0.75, beta_r=0.80, eta_col=0.80, xi_reb=0.85, z_A0_eff=0.90,
        description=(
            "All four degradation parameters simultaneously degraded. "
            "Most complex scenario for root-cause attribution."
        ),
    ),

    # ---- Open-loop variants (Loops 1/2/3 all bypassed) ---------------
    "W2_cat_mild_ol": _sc(
        102, "W2_cat_mild_ol", alpha=0.85, mode="open_loop",
        description="W2 cat_mild without PI control (open-loop for masking contrast).",
    ),
    "W3_cat_severe_ol": _sc(
        103, "W3_cat_severe_ol", alpha=0.65, mode="open_loop",
        description="W3 cat_severe without PI control.",
    ),
    "W5_jacket_mild_ol": _sc(
        105, "W5_jacket_mild_ol", beta_r=0.80, mode="open_loop",
        description="W5 jacket_mild without PI control.",
    ),
    "W6_jacket_severe_ol": _sc(
        106, "W6_jacket_severe_ol", beta_r=0.60, mode="open_loop",
        description="W6 jacket_severe without PI control.",
    ),
    "W7_col_eff_mild_ol": _sc(
        107, "W7_col_eff_mild_ol", eta_col=0.80, mode="open_loop",
        description="W7 col_eff_mild without PI control.",
    ),
    # W8_col_eff_severe_ol removed together with W8_col_eff_severe.
    "W1_healthy_ol": _sc(
        101, "W1_healthy_ol", mode="open_loop",
        description="W1 healthy without PI control (open-loop baseline).",
    ),
}


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def list_configs(mode: str | None = None) -> list[RecycleScenarioConfig]:
    """Return all scenarios in numerical order, optionally filtered by mode."""
    configs = sorted(SCENARIO_CONFIGS.values(), key=lambda s: s.id)
    if mode is not None:
        configs = [c for c in configs if c.mode == mode]
    return configs


def list_closed_loop() -> list[RecycleScenarioConfig]:
    return list_configs(mode="closed_loop")


def list_open_loop() -> list[RecycleScenarioConfig]:
    return list_configs(mode="open_loop")


def get_scenario(name: str) -> RecycleScenarioConfig:
    return SCENARIO_CONFIGS[name]


# Ordered list of the 14 closed-loop scenario names (paper order)
# Removed: W8 (eta_col=0.65) and W14 (eta_col=0.75+xi_reb=0.75) — both produce
# ODE blow-up (Inf/NaN) under S-B due to QSS column shortcut boundary instability.
CLOSED_LOOP_NAMES = [
    "W1_healthy", "W2_cat_mild", "W3_cat_severe", "W4_cat_threshold",
    "W5_jacket_mild", "W6_jacket_severe", "W7_col_eff_mild",
    "W9_reb_fouling", "W10_feed_impurity", "W11_reactor_combined",
    "W12_snowball_compound", "W13_cat_feed",
    "W15_snowball_threshold", "W16_full_multi",
]

# Fault unit labels for hierarchical classification
FAULT_UNITS = ["healthy", "reactor", "column", "feed", "multi"]
