"""cstr_sbi.recycle -- Wu 2003 CSTR-column-recycle plant extension.

Subpackage layout:
    physics    -- 4-state ODE, QSS column model, constants, SS integrator
    simulator  -- deterministic-window + noisy replicate data generation
    priors     -- 5-D BoxUniform prior [alpha, beta_r, eta_col, xi_reb, z_A0_eff]
    scenarios  -- 16 closed-loop + 7 open-loop fault scenarios (W1–W16)
"""

from cstr_sbi.recycle import physics, priors, scenarios, simulator

__all__ = ["physics", "priors", "scenarios", "simulator"]
