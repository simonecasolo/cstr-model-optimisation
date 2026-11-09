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

# NOTE (2026-09-04): the top-level `physics`/`plotting`/`simulator` modules
# ("propylene oxide system") referenced in the docstring above were removed in
# an earlier, uncommitted legacy-code cleanup (see `git status` -- most of
# `legacy_code/`, old notebooks 00-03/13, and their data/figures went with
# them) but this import was never updated to match, leaving a dead reference
# that broke every `import cstr_sbi` in a fresh interpreter. The project has
# moved on to the `luyben`/`recycle` subpackages below; nothing currently
# imports the old top-level `physics` module.
from cstr_sbi import luyben    # Luyben recycle plant extension
from cstr_sbi import recycle   # Wu 2003 CSTR-column-recycle extension

__all__ = ["luyben", "recycle", "__version__"]
