"""cstr_sbi.luyben -- Luyben recycle plant extension.

Subpackage layout:
    physics     -- 13-state ODE, VLE, 5 PI controllers, constants
    simulator   -- Euler-Maruyama integrator, sensor layer, replicate generator
    summaries   -- 65-D summary statistics for 8 observable channels
    priors      -- 8-D BoxUniform prior [alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta]
    scenarios   -- 12 fault scenarios (L1-L12) + open-loop counterparts
    inference   -- SNPE_C training wrapper, sample_posterior
    ekf         -- 21-state augmented EKF with jax.jacobian Jacobian
"""

from cstr_sbi.luyben import physics, simulator, summaries, priors, scenarios, inference, ekf

__all__ = [
    "physics", "simulator", "summaries", "priors",
    "scenarios", "inference", "ekf",
]
