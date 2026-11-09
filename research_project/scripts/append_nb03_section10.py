"""Append (or replace) section 10 -- "Can the summary statistics be improved?" -- in
notebooks/system_I/03_summary_statistics_design.ipynb, keeping sections 1-9 untouched.

Section 10 is self-contained (own setup cell, own simulations under the fixed controller law
and the P1b warm start), because sections 1-9 read data/observations.npz, which is old-law
until nb02 is re-run (re-run plan, Phase 3).

Usage (add --sensor N3 for section 10b, the N3 re-run; default N0 = section 10):
  python scripts/append_nb03_section10.py            # append/replace the cells only
  python scripts/append_nb03_section10.py --execute  # also execute section 10 (kernel cwd =
                                                     # notebooks/system_I) and store its outputs
                                                     # in notebook 03; sections 1-9 are not run
"""
import copy
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
NB = NB_DIR / "03_summary_statistics_design.ipynb"
MARKER = "<!-- section10-summary-redesign -->"

HEADER = MARKER + r"""
## 10. Can the summary statistics be improved? (fixed controller law, P1b — 2026-11-08)

**Why ask.** nb29 E2 shows the 29-D summaries already carry close to the information in the
raw trajectory (raw-trajectory Cramér–Rao bounds within ~10–20%). But at 5k simulations the
NPE posterior for α was several times wider than that bound and its full-set SBC failed. So
the headroom is mostly in how *learnable* the information is, not in how much there is.

**Candidates** (`scripts/summary_variants.py`, tested in `tests/test_summary_variants.py`):

| Set | Features | Idea |
|---|---|---|
| S0 | current 29 | reference (production) |
| S1 | S0 + 3 plug-in estimators | ln α̂ (component balance), ln β̂ (jacket balance), ln β̂ (reactor balance, independent of Qc): nearly linear in log θ, close to the Fearnhead–Prangle optimal summary E[θ|x] |
| S2 | S1 + 6 loop-dynamics + 4 detrended stds | lag-1 autocorrelations, T–Qc / T–Tc cross-correlations: β changes the plant gain, so the noise-driven closed-loop dynamics carry information the means do not |
| S3 | 19, compact | estimators, means, detrended stds, saturation, dynamics; drops noisy min/max/slope |
| S4 | 5, minimal | estimators + saturation fractions only |

**Protocol.** P1b, one set of simulations shared by all sets (features computed from the same
raw windows, so sets differ only by the features). Budgets 5k and 20k × 3 seeds, same NSF as
production. Test set: 1000 SBC draws + the scenario grid × 50 reps.

**Decision rule — fixed before results** (`scripts/nb03_summary_study.py`): at 20k, a candidate
replaces S0 only if it is calibrated (interior marginal KS p ≥ 0.01 for α and β, TARP KS p ≥ 0.01),
its RMSE is not worse, and either its efficiency (posterior sd / raw-trajectory CRB, Sc1–Sc4) is
≥ 10% better than S0's beyond 2 SE, or S0 fails calibration and the candidate does not lose
efficiency. Among winners the most efficient; a smaller set within 1 SE is preferred. No winner →
the paper keeps the 29-D summaries.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import nb03_summary_study as SS, summary_variants as sv
R10, FIGS10 = ROOT / "results", ROOT / "figures"
THREADS = 4
T0_10 = time.time()
print({k: len(v) for k, v in sv.SETS.items()})"""),
    ("md", "### 10.1 Simulate once (P1b), compute the union of all candidate features"),
    ("code", "SS.stage_sim()"),
    ("md", """### 10.2 Information content per set (Gaussian FIM, as nb29 E2)
Cramér–Rao sd per set at the nb29 points, next to nb29's raw-trajectory bound. A set whose
CRB is close to the raw bound loses little information; the NPE results below show whether
that information is also *learned*."""),
    ("code", """if not (R10 / "03s_fim.csv").exists():
    SS.stage_fim()
fim = pd.read_csv(R10 / "03s_fim.csv")
raw = pd.read_csv(R10 / "29fx_e2_fim.csv")
raw = raw[(raw.protocol == SS.PROTOCOL) & (raw.representation == "raw_traj_lwcov")][["point", "crb_sd_a", "crb_sd_b"]]
raw = raw.assign(set="raw trajectory (nb29)", n_features=480)
display(pd.concat([fim, raw]).pivot_table(index="set", columns="point", values=["crb_sd_a", "crb_sd_b"]).round(4))
display(fim.pivot_table(index="set", columns="point", values="corr_implied").round(3))"""),
    ("md", "### 10.3 Train NPE: 5 sets × 2 budgets × 3 seeds"),
    ("code", "SS.stage_train(THREADS)"),
    ("md", "### 10.4 Evaluate on the shared held-out test set"),
    ("code", "SS.stage_eval()"),
    ("code", """m10 = pd.read_csv(R10 / "03s_metrics.csv")
cols = ["n_features", "ks_p_a_interior", "ks_p_b_interior", "ks_p_a", "tarp_ks_p", "cov90_a", "cov90_b",
        "rmse_norm", "macro_f1", "leak_sbc_mean"]
display(m10.groupby(["set", "budget"])[cols].median().round(4))
eff = pd.concat([SS.efficiency_table(N).assign(budget=N) for N in SS.BUDGETS])
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for s, g in eff.groupby("set"):
    gg = g.groupby("budget").eff
    axes[0].errorbar(gg.mean().index, gg.mean(), yerr=gg.std(), marker="o", capsize=3, label=s)
axes[0].set_xscale("log"); axes[0].axhline(1, color="grey", lw=0.8)
axes[0].set_xlabel("training simulations"); axes[0].set_ylabel("posterior sd / raw-trajectory CRB")
axes[0].set_title("efficiency (Sc" + ",".join(p[2:] for p in eff.attrs.get("points", [])) + ")"); axes[0].legend(fontsize=7)
for s, g in m10.groupby("set"):
    gg = g.groupby("budget").ks_p_a_interior.median()
    axes[1].plot(gg.index, gg.values, "o-", label=s)
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].axhline(0.01, color="grey", lw=0.8)
axes[1].set_xlabel("training simulations"); axes[1].set_title("α interior SBC KS p (seed median)"); axes[1].legend(fontsize=7)
fig.savefig(FIGS10 / "03s_summary_sets.png", dpi=150); plt.show()
display(eff[eff.budget == 20_000].groupby("set").mean(numeric_only=True).drop(columns=["seed", "budget"]).round(2))"""),
    ("md", "### 10.5 Decision (pre-set rule)"),
    ("code", """res10 = SS.stage_decide()
print("CHOICE:", res10["choice"], "| winners:", res10["winners"])
print(f"section 10 wall time {(time.time() - T0_10) / 60:.1f} min")"""),
    ("md", """### 10.6 What follows from the decision
- **S0 kept:** the paper's summaries stand; report this section as evidence that the 29-D set is
  close to sufficient and that the remaining gap is estimator learning (budget, nb31).
- **A candidate wins:** promote it into `cstr_sbi.summaries` as a new, versioned feature set
  (keep the 29-D function for traceability), then nb31's bank and test set must be recomputed
  with it before the budget study is run. nb29's protocol conclusion is unaffected (protocols
  were compared with the same summaries on both sides)."""),
]


N3_HEADER_NOTE = """
> **Sensor model: {sensor}** ({desc}). Re-run of section 10 under the production sensor model
> adopted on 2026-11-09 (notebook 33); section 10 above is the N0 run. Warm-start protocol: the one
> nb29 selects for this sensor model (under N3 the pre-set rule chose P2, adopted by the user on
> 2026-11-09; under N0 it was P1b). With realistic sensors
> the information moves between channels (C becomes ~30× less informative about α, T ~2×
> more informative about β), so the comparison of summary sets must be repeated. Same rule,
> same seeds; outputs carry the prefix `{tag}_`.
> **If a candidate set wins, STOP**: promoting it into `cstr_sbi.summaries` (and rebuilding
> nb31's bank) is a user decision.
"""

INTERP_MD = """### 10b.7 Interpretation ({sensor} run) — to be written after the run
State: the decision and why (efficiency vs CRB, calibration); whether the plug-in estimators
or the loop-dynamics features help more under realistic sensors than under N0 (they might,
because T is now precise); and what the paper says about the summaries (sufficiency under both
sensor models, or a revised set). Keep claims to what the tables show."""


def _for_sensor(cells, sensor):
    import system_I_config as CFG
    if sensor == "N0":                      # default model is now N3: pin N0 explicitly
        pin = ('os.environ["CSTR_SENSOR_MODEL"] = "N0"\n'
               "import importlib, ws_protocols, ws_eval, system_I_config, nb29_ws_study, nb03_summary_study, summary_variants as sv\n"
               "for _m in (system_I_config, ws_protocols, ws_eval, nb29_ws_study, nb03_summary_study):\n"
               "    importlib.reload(_m)\n"
               "SS = nb03_summary_study\n"
               'assert SS.SENSOR == "N0", SS.SENSOR\n')
        out = []
        for k, src in cells:
            if k == "code" and 'import nb03_summary_study as SS, summary_variants as sv\n' in src:
                src = src.replace('import nb03_summary_study as SS, summary_variants as sv\n', pin)
                if "import json, os," not in src:
                    src = src.replace("import json, ", "import json, os, ", 1)
            out.append((k, src))
        return out
    tag = "03s" + CFG.suffix(sensor)
    out = []
    for k, src in cells:
        if k == "md" and src.startswith(MARKER):
            src = (src.replace("P1b", CFG.protocol(sensor)).replace(MARKER, MARKER_FOR[sensor])
                      .replace("## 10. Can the summary statistics be improved? (fixed controller law, P1b — 2026-11-08)",
                               f"## 10b. Can the summary statistics be improved? — {sensor} sensor model (2026-11-09)")
                   + N3_HEADER_NOTE.format(sensor=sensor, desc=CFG.describe(sensor), tag=tag))
        elif k == "md":
            src = src.replace("### 10.", "### 10b.").replace("P1b", CFG.protocol(sensor))
        else:
            src = (src.replace('R10 / "03s_', 'R10 / f"{SS.TAG}_').replace('FIGS10 / "03s_', 'FIGS10 / f"{SS.TAG}_')
                      .replace('R10 / "29fx_e2_fim.csv"', 'R10 / f"{SS.E2_TAG}_e2_fim.csv"'))
            if "import nb03_summary_study as SS, summary_variants as sv\n" in src:
                src = src.replace("import json, sys, time\n", "import json, os, sys, time\n").replace(
                    "import nb03_summary_study as SS, summary_variants as sv\n",
                    f'os.environ["CSTR_SENSOR_MODEL"] = "{sensor}"\n'
                    "import importlib, ws_protocols, ws_eval, system_I_config, nb29_ws_study, nb03_summary_study, summary_variants as sv\n"
                    "for _m in (system_I_config, ws_protocols, ws_eval, nb29_ws_study, nb03_summary_study):\n"
                    "    importlib.reload(_m)\n"
                    "SS = nb03_summary_study\n"
                    f'assert SS.SENSOR == "{sensor}", SS.SENSOR\n')
            if src.startswith("res10 = SS.stage_decide()"):
                src += ('\nif res10["choice"] != "S0_base29":\n'
                        '    raise RuntimeError("STOP: a candidate summary set wins under this sensor model -- "\n'
                        '                       "promoting it is a user decision (re-run plan, Phase 1b)")')
        out.append((k, src))
    out.append(("md", INTERP_MD.format(sensor=sensor)))
    return out


MARKER_FOR = {"N3": "<!-- section10-summary-redesign-N3 -->"}


def _section_cells(sensor="N0"):
    return [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in _for_sensor(CELLS, sensor)]


def main(execute: bool, sensor: str = "N0"):
    """Replace this sensor model's section (identified by its marker), keeping every other cell.
    The N0 section 10 and the N3 section 10b are independent."""
    nb = nbformat.read(NB, as_version=4)
    marker = MARKER if sensor == "N0" else MARKER_FOR[sensor]
    keep, cut, tail = [], None, []
    is_marker = lambda c, m: c.cell_type == "markdown" and "".join(c.source).lstrip().split("\n", 1)[0].strip() == m
    for i, c in enumerate(nb.cells):
        if is_marker(c, marker):
            cut = i
            break
    if cut is None:
        keep = list(nb.cells)
    else:                                    # drop this section only, up to the next section marker
        keep = list(nb.cells[:cut])
        rest = nb.cells[cut + 1:]
        nxt = next((j for j, c in enumerate(rest) if c.cell_type == "markdown"
                    and "".join(c.source).lstrip().startswith("<!-- section10")), None)
        tail = [] if nxt is None else rest[nxt:]
    new = _section_cells(sensor)
    if execute:
        from nbclient import NotebookClient
        tmp = new_notebook(); tmp.cells = new
        tmp.metadata = copy.deepcopy(nb.metadata)
        client = NotebookClient(tmp, timeout=None, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}})   # cwd = notebooks/system_I
        try:
            client.execute()
        finally:
            new = tmp.cells
            nb.cells = keep + new + tail
            nbformat.write(nb, NB)
    else:
        nb.cells = keep + new + tail
        nbformat.write(nb, NB)
    print(f"{NB}: {len(keep)} existing cells kept, {len(new)} section-10 cells written")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    _sensor = sys.argv[sys.argv.index("--sensor") + 1] if "--sensor" in sys.argv else "N0"
    main("--execute" in sys.argv, sensor=_sensor)
