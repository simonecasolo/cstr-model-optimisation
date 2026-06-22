"""Fault scenario configurations for the Luyben recycle plant.

12 scenarios × 2 control modes (closed_loop / open_loop) = 24 configs.
Each scenario specifies the 8-D degradation parameter vector:
    theta = [alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta]

Scenario IDs follow the plan (project_luyben_extension.md):
    L1  healthy
    L2  catalyst decay
    L3  CSTR fouling
    L4  separator efficiency loss
    L5  separator HEX fouling
    L6  recycle pump degradation
    L7  purge valve erosion
    L8  feed preheater fouling
    L9  stoichiometry shift
    L10 snowball (alpha + eta_p) -- headline experiment
    L11 reactor + separator competing faults
    L12 severe multi-unit fault
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from cstr_sbi.luyben.physics import NOMINAL_THETA, PARAM_NAMES


@dataclass(frozen=True)
class LuybenScenarioConfig:
    """One row of the Luyben scenario truth table."""

    id: int
    name: str
    alpha:   float
    beta_r:  float
    eta_sep: float
    beta_s:  float
    eta_p:   float
    xi:      float
    kappa:   float
    delta:   float
    mode: str        # "closed_loop" | "open_loop"
    description: str

    def theta(self) -> jnp.ndarray:
        """Return the 8-D degradation parameter vector."""
        return jnp.array(
            [self.alpha, self.beta_r, self.eta_sep, self.beta_s,
             self.eta_p, self.xi, self.kappa, self.delta],
            dtype=jnp.float32,
        )

    def fault_unit(self) -> str:
        """Which plant unit is primarily degraded (for classification)."""
        if self.alpha < 0.9 or self.beta_r < 0.9:
            return "reactor"
        elif self.eta_sep < 0.9 or self.beta_s < 0.9:
            return "separator"
        elif self.eta_p < 0.9 or self.xi > 1.1 or self.xi < 0.9:
            return "recycle"
        elif self.kappa < 0.9 or abs(self.delta) > 0.1:
            return "feed"
        else:
            return "healthy"


def _sc(id, name, alpha=1.0, beta_r=1.0, eta_sep=1.0, beta_s=1.0,
        eta_p=1.0, xi=1.0, kappa=1.0, delta=0.0,
        mode="closed_loop", description=""):
    return LuybenScenarioConfig(
        id=id, name=name,
        alpha=alpha, beta_r=beta_r, eta_sep=eta_sep, beta_s=beta_s,
        eta_p=eta_p, xi=xi, kappa=kappa, delta=delta,
        mode=mode, description=description,
    )


SCENARIO_CONFIGS: dict[str, LuybenScenarioConfig] = {
    # Closed-loop scenarios
    "L1_healthy": _sc(
        1, "L1_healthy",
        description="Healthy plant, all parameters nominal.",
    ),
    "L2_cat_decay": _sc(
        2, "L2_cat_decay", alpha=0.65,
        description="Catalyst decay (alpha=0.65). Primary fault.",
    ),
    "L3_rxr_fouling": _sc(
        3, "L3_rxr_fouling", beta_r=0.65,
        description="CSTR jacket fouling (beta_r=0.65).",
    ),
    "L4_sep_eff": _sc(
        4, "L4_sep_eff", eta_sep=0.65,
        description="Separator split efficiency loss (eta_sep=0.65).",
    ),
    "L5_sep_fouling": _sc(
        5, "L5_sep_fouling", beta_s=0.65,
        description="Separator HEX fouling (beta_s=0.65).",
    ),
    "L6_pump_deg": _sc(
        6, "L6_pump_deg", eta_p=0.65,
        description="Recycle pump degradation (eta_p=0.65).",
    ),
    "L7_purge_block": _sc(
        7, "L7_purge_block", xi=1.5,
        description="Purge valve erosion (xi=1.5, flow restriction increased).",
    ),
    "L8_feed_preheat": _sc(
        8, "L8_feed_preheat", kappa=0.65,
        description="Feed preheater fouling (kappa=0.65).",
    ),
    "L9_stoich_shift": _sc(
        9, "L9_stoich_shift", delta=0.25,
        description="Feed A:B stoichiometry shift (delta=0.25, A-rich).",
    ),
    "L10_snowball": _sc(
        10, "L10_snowball", alpha=0.65, eta_p=0.85,
        description=(
            "Snowball scenario: catalyst decay (alpha=0.65) triggers "
            "recycle buildup, stressing pump (eta_p=0.85). Headline experiment."
        ),
    ),
    "L11_reactor_sep": _sc(
        11, "L11_reactor_sep", alpha=0.80, beta_r=0.80, eta_sep=0.75,
        description="Competing faults: reactor decay + separator efficiency loss.",
    ),
    "L12_severe_multi": _sc(
        12, "L12_severe_multi",
        alpha=0.60, beta_r=0.70, eta_sep=0.70, beta_s=0.70, eta_p=0.80,
        description="Severe multi-unit fault: reactor + separator + pump degraded.",
    ),
    # Open-loop counterparts (for ablation study, nb35)
    "L1_healthy_ol": _sc(
        101, "L1_healthy_ol", mode="open_loop",
        description="Healthy plant, open-loop (all PI loops bypassed).",
    ),
    "L2_cat_decay_ol": _sc(
        102, "L2_cat_decay_ol", alpha=0.65, mode="open_loop",
        description="Catalyst decay, open-loop.",
    ),
    "L10_snowball_ol": _sc(
        110, "L10_snowball_ol", alpha=0.65, eta_p=0.85, mode="open_loop",
        description="Snowball scenario, open-loop.",
    ),
}


def list_configs(mode: str | None = None) -> list[LuybenScenarioConfig]:
    """Return scenarios in numerical order, optionally filtered by mode."""
    configs = sorted(SCENARIO_CONFIGS.values(), key=lambda s: s.id)
    if mode is not None:
        configs = [c for c in configs if c.mode == mode]
    return configs


def list_closed_loop_configs() -> list[LuybenScenarioConfig]:
    return list_configs(mode="closed_loop")


# ---------------------------------------------------------------------------
# 30-day sequential degradation profile (nb37)
# ---------------------------------------------------------------------------

def generate_degradation_stream(
    *,
    t_crit: float = 43200.0,     # 30 days in minutes
    dt_window: float = 120.0,    # 2-hour windows
    n_replicates_per_window: int = 1,
    seed: int = 0,
    inlet=None,
    ctrl=None,
    alpha_end: float = 0.70,     # final alpha after 30 days (10% decay/week)
    eta_sep_end: float = 0.85,   # separator efficiency drift
    beta_r_end: float = 0.90,    # mild CSTR fouling over 30 days
) -> list[dict]:
    """Generate a 30-day degradation stream (720 windows, 2h each).

    Linear decay profiles:
        alpha(t)   : 1.0 -> alpha_end   over t_crit
        eta_sep(t) : 1.0 -> eta_sep_end over t_crit
        beta_r(t)  : 1.0 -> beta_r_end  over t_crit
    All other parameters remain at 1.0 (healthy).

    Returns a list of dicts, one per window, each containing:
        "t_start"       -- window start time in minutes
        "theta_true"    -- 8-D true parameter vector
        "fault_class"   -- hierarchical fault class string
        "obs"           -- (n_replicates, n_t, 8) noisy observations
        "t"             -- (n_t,) time grid
    """
    import jax
    from cstr_sbi.luyben.simulator import generate_replicates, warm_start_ic
    from cstr_sbi.luyben.physics import NOMINAL_INLET, NOMINAL_CTRL_ALL

    if inlet is None:
        inlet = NOMINAL_INLET
    if ctrl is None:
        ctrl = NOMINAL_CTRL_ALL

    n_windows = int(t_crit / dt_window)
    master_key = jax.random.PRNGKey(seed)

    stream = []
    for win_idx in range(n_windows):
        t_start = win_idx * dt_window
        frac = t_start / t_crit

        alpha_t   = float(1.0 - (1.0 - alpha_end)   * frac)
        eta_sep_t = float(1.0 - (1.0 - eta_sep_end) * frac)
        beta_r_t  = float(1.0 - (1.0 - beta_r_end)  * frac)

        theta_t = jnp.array(
            [alpha_t, beta_r_t, eta_sep_t, 1.0, 1.0, 1.0, 1.0, 0.0],
            dtype=jnp.float32,
        )

        y0 = warm_start_ic(theta_t, inlet, ctrl)
        win_key = jax.random.fold_in(master_key, win_idx)
        t_out, obs = generate_replicates(
            theta_t, inlet, ctrl, y0,
            n_replicates=n_replicates_per_window,
            master_key=win_key,
            t_window=dt_window,
        )

        fault_class = _classify_for_stream(alpha_t, eta_sep_t, beta_r_t)

        stream.append({
            "t_start":    t_start,
            "theta_true": jnp.asarray(theta_t),
            "fault_class": fault_class,
            "obs":        jnp.asarray(obs),
            "t":          jnp.asarray(t_out),
        })

    return stream


def _classify_for_stream(alpha: float, eta_sep: float, beta_r: float) -> str:
    """Simple rule-based classification for the degradation stream."""
    from cstr_sbi.luyben.priors import HEALTHY_THRESHOLDS as H
    any_reactor   = alpha < H["alpha"] or beta_r < H["beta_r"]
    any_separator = eta_sep < H["eta_sep"]
    if any_reactor and any_separator:
        return "combined"
    elif any_reactor:
        return "reactor_fault"
    elif any_separator:
        return "separator_fault"
    else:
        return "healthy"
