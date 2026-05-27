"""Paper styling and figure persistence.

M7 deliverable. Mirrors ``../../../sbi_mcmc_heat_exchanger/src/hx_models/style.py``.

Planned API:

    apply_paper_style()                  -- serif font, Okabe-Ito color cycle.
    save_fig(fig, path_stem, formats=("pdf", "png"))
    PARAM_LABELS  -- {"UA": ..., "k0": ..., "alpha": ..., "beta": ...}
    OBS_LABELS    -- {"C": ..., "T": ..., "Tc": ..., "Qc": ...}
    SCENARIO_*    -- canonical scenario titles and color codes.
    MCMC_COLOR / SBI_COLOR / TRUE_COLOR
"""
