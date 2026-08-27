"""Paper styling and figure persistence.

M7 deliverable. Mirrors ``../../../sbi_mcmc_heat_exchanger/src/hx_models/style.py``:
serif fonts, Computer Modern math, Okabe-Ito colourblind-safe palette, seaborn
whitegrid axes. Used by ``notebooks/28_wu2003_publication_figures.ipynb`` (and any
other notebook producing a manuscript figure) so that every figure in the paper
shares one visual identity regardless of which notebook produced it.

Usage in every publication-figure notebook::

    from cstr_sbi.style import apply_paper_style, OI, save_fig
    apply_paper_style()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_style() -> None:
    """Apply consistent matplotlib styling for all paper figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,

        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,

        "figure.figsize": (7, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        "axes.prop_cycle": mpl.cycler("color", [
            "#0072B2", "#D55E00", "#009E73",
            "#E69F00", "#CC79A7", "#56B4E9",
        ]),
    })


# ---------------------------------------------------------------------------
# Okabe-Ito colourblind-safe palette (consistent across all notebooks in this
# project — same ordering used ad hoc in notebooks 20-23/28: black, orange,
# sky blue, teal, yellow, blue, vermillion, purple).
# ---------------------------------------------------------------------------

OI = ["#000000", "#E69F00", "#56B4E9", "#009E73",
      "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

MCMC_COLOR = OI[3]
SBI_COLOR = OI[5]
EKF_COLOR = OI[6]
TRUE_COLOR = OI[0]

CL_COLOR = OI[2]   # closed-loop
OL_COLOR = OI[6]   # open-loop

TR_COLOR = OI[5]   # reactor temperature T_r
TJ_COLOR = OI[6]   # jacket temperature T_j

# ---------------------------------------------------------------------------
# Paper-aligned labels (code variable -> descriptive label with math)
# ---------------------------------------------------------------------------

# System I (propylene-oxide CSTR, 2-D parameter space)
PARAM_LABELS_SYS1 = {
    "alpha": r"$\alpha$ (catalyst activity)",
    "beta": r"$\beta$ (jacket heat transfer)",
}

# System II (Wu 2003 reactor-column-recycle plant, 5-D parameter space)
PARAM_LABELS_SYS2 = {
    "alpha": r"$\alpha$",
    "beta_r": r"$\beta_r$",
    "eta_col": r"$\eta_{\mathrm{col}}$",
    "xi_reb": r"$\xi_{\mathrm{reb}}$",
    "z_A0_eff": r"$z_{A0,\mathrm{eff}}$",
}

# ---------------------------------------------------------------------------
# Helpers for saving figures in publication-ready form
# ---------------------------------------------------------------------------

def save_fig(fig, path_stem, formats=("png",)) -> None:
    """Save a figure, creating the parent directory if needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path_stem : str or Path
        Path without extension, e.g. ``figures/nb21_cl_vs_ol_masking``.
    formats : tuple of str
        File extensions to save.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path_stem.with_suffix(f".{fmt}"), bbox_inches="tight")
