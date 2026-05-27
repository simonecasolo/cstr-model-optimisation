"""Publication-quality plotting routines.

M7 deliverable. Tailors the heat-exchanger template at
``../../../sbi_mcmc_heat_exchanger/src/hx_models/plotting.py`` to the CSTR
parameter set [UA, k0, alpha, beta] and the four observable channels
[C, T, Tc, Qc].

Planned API:

    plot_posterior_pairplot(samples, true_params, param_names)
    plot_timeseries_scenario(timeseries, scenario_id, fault_onset)
    plot_identifiability_analysis(posterior_grid, param_grid, param_names)
    plot_sbi_vs_mcmc_resources(...)
    plot_sbi_mcmc_comparison(sbi_samples, mcmc_samples, truth)
    hdi_bands(samples, levels=(0.5, 0.9))
"""
