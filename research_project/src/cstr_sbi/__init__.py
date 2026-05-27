"""cstr_sbi -- closed-loop CSTR fault diagnosis via simulation-based inference.

Package layout (see cstr_sbi_execution_plan.md):
    physics      -- ODE right-hand sides (open-loop and closed-loop).
    simulator    -- Trajectory simulation, noise, drift, parallel runner.
    summaries    -- Summary statistics for SBI observations.
    priors       -- BoxUniform / NumPyro priors over [UA, k0, alpha, beta].
    inference    -- SNPE training, NUTS MCMC baseline, latent <-> sim-theta maps.
    scenarios    -- Scenario 0..7 data generators, 30-day continuous stream.
    metrics      -- CRPS, Wasserstein-1, coverage, classification F1.
    plotting     -- Pairplot, time-series, identifiability, MCMC-vs-SBI scatter.
    style        -- Paper styling and save_fig.
"""

__version__ = "0.0.1"

from cstr_sbi import physics  # re-export for convenience

__all__ = ["physics", "__version__"]
