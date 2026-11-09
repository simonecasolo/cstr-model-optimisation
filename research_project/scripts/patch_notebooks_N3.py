"""Adopt the N3 production sensor model (2026-11-09) in the System I notebooks that are edited
in place rather than rebuilt by a builder script.

  * 02_data_generation.ipynb       -> regenerates data/observations.npz under N3 (the N0 file is
                                      archived as data/observations_N0.npz by the notebook itself)
  * 07_failure_baseline_open_vs_closed.ipynb -> N3 production posterior + N3 open-loop posterior
  * archives the executed N0 versions as *_N0.ipynb (outputs kept, setup pinned to N0)
  * pins the executed N0 notebooks 29, 31 and 03 section 10 to N0 (the default model is now N3),
    keeping their outputs

Idempotent: every edit checks whether it has already been applied. JSON-level edits only (no
nbformat dependency). Outputs of the edited notebooks 02 and 07 are cleared (they are re-run).

    python scripts/patch_notebooks_N3.py [--nb-dir notebooks/system_I]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = Path(sys.argv[sys.argv.index("--nb-dir") + 1]) if "--nb-dir" in sys.argv else ROOT / "notebooks" / "system_I"
DATE = "2026-11-09"


def _load(p):
    return json.loads(Path(p).read_text())


def _save(nb, p):
    Path(p).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def _src(c):
    return "".join(c["source"])


def _set(c, s):
    lines = s.split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def _md(s):
    c = {"cell_type": "markdown", "metadata": {}, "source": []}
    _set(c, s)
    return c


def _sub(c, old, new, count=-1):
    s = _src(c)
    if new in s:
        return False                     # already applied (new may contain old)
    assert old in s, f"pattern not found: {old[:80]!r}"
    _set(c, s.replace(old, new) if count < 0 else s.replace(old, new, count))
    return True


def _clear_outputs(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None


def _find(nb, needle, kind=None):
    hits = [i for i, c in enumerate(nb["cells"]) if needle in _src(c) and (kind is None or c["cell_type"] == kind)]
    assert len(hits) == 1, f"{needle[:60]!r}: {len(hits)} matches"
    return hits[0]


def _pin_n0_line():
    return 'import os\nos.environ["CSTR_SENSOR_MODEL"] = "N0"   # archival N0 run (default model is N3 since 2026-11-09)\n'


def _archive(src: Path, dst: Path, setup_needle: str, title: str):
    """Copy the executed N0 notebook (outputs kept) and pin its setup cell to N0."""
    if dst.exists():
        return f"{dst.name}: exists, kept"
    nb = _load(src)
    i = _find(nb, setup_needle, "code")
    s = _src(nb["cells"][i])
    if 'CSTR_SENSOR_MODEL' not in s:
        _set(nb["cells"][i], _pin_n0_line() + s)
    nb["cells"].insert(0, _md(f"> **Archived N0 run ({DATE}).** {title} with the superseded sensor model N0 "
                              "(σ = 0.5% of each channel's window maximum). Kept for the record and for the "
                              "N0-vs-N3 comparison in the paper; the production version is the notebook without "
                              "the `_N0` suffix. Re-running this copy requires `data/observations_N0.npz` to be "
                              "copied back to `data/observations.npz`."))
    _save(nb, dst)
    return f"{dst.name}: archived"


# ----------------------------------------------------------------------------- notebook 02
NOTE02 = f"""> **{DATE} revision — production sensor model N3.** Sensor noise is now absolute per channel,
> σ = (0.01 mol/L on C, 0.2 K on T, 0.2 K on Tc, 2 L/min on Qc), chosen in notebook 33 as the
> realistic industrial sensor model (online analyser ~1% of full scale; class-A RTDs; flow meter
> ~0.5% of span). The previous dataset (N0: 0.5% of each channel's window maximum) is archived by
> this notebook as `data/observations_N0.npz`, and the executed N0 notebook as
> `02_data_generation_N0.ipynb`. `observations.npz` gains the keys `sensor_model` and `sensor_noise`;
> notebook 07 refuses to run if the data's sensor model differs from the production posterior's.
> **Numbers quoted in the markdown of sections 5, 6, 8a and 9 come from the N0 run until section 10
> is written after the N3 run.**
"""

SENSOR_MD02 = """## 2. Simulator settings

Process noise diffusion coefficients (``sigma * sqrt(dt)`` per Euler-Maruyama
step) are pulled from ``cstr_sbi.simulator``.

**Sensor model N3 (production since 2026-11-09, notebook 33).** Absolute Gaussian noise per
channel: σ_C = 0.01 mol/L (an online analyser at ~1% of a 1 mol/L span), σ_T = σ_Tc = 0.2 K
(class-A RTD), σ_Qc = 2 L/min (~0.5% of the 400 L/min span). Compared with the earlier N0
convention (0.5% of each channel's window maximum: σ_T ≈ 1.56 K, σ_C ≈ 9×10⁻⁵ mol/L), N3 makes
temperature ~8× *more* precise and concentration ~100× *less* precise. Consequences measured in
notebook 33: β becomes ~2.4× more precise, α ~10× less precise (α is read mainly from C at ~98%
conversion), and the α–β posterior correlation essentially disappears. The noise model is set in
`scripts/system_I_config.py` (environment variable `CSTR_SENSOR_MODEL`, default N3).
"""

INTERP02 = f"""## 10. N3 dataset — interpretation (to be written after the run)
Report from the outputs above: saturation fractions (Sc2 must stay < 1%, Sc5 > 50%), T_mean − Tsp
and Qc_mean per scenario (the headline "same T, different Qc" picture), and how the per-window
spread of C now compares with the α signal (C noise 0.01 mol/L vs C_ss ≈ 0.018 mol/L). Then update
the numbers quoted in sections 5, 6, 8a and 9 to the N3 values.
"""


def patch_02():
    p = NB_DIR / "02_data_generation.ipynb"
    msg = [_archive(p, NB_DIR / "02_data_generation_N0.ipynb", "import ws_eval as we", "Notebook 02 (data generation)")]
    nb = _load(p)
    if "production sensor model N3" not in _src(nb["cells"][1]):
        nb["cells"].insert(1, _md(NOTE02))
    c = nb["cells"][_find(nb, "ws_eval as we", "code")]
    _sub(c, "import ws_eval as we          # the training simulator (protocols in scripts/ws_protocols.py)\n",
         "import os, importlib\n"
         'os.environ["CSTR_SENSOR_MODEL"] = "N3"   # production sensor model (notebook 33, 2026-11-09)\n'
         "import system_I_config as CFG, ws_protocols, ws_eval as we   # training simulator (scripts/ws_protocols.py)\n"
         "for _m in (CFG, ws_protocols, we):\n"
         "    importlib.reload(_m)\n"
         "SENSOR, NOISE = CFG.model(), CFG.noise_kw()\n"
         'assert SENSOR == "N3", SENSOR\n'
         'print("Sensor model:", CFG.describe())\n')
    i = _find(nb, "## 2. Simulator settings", "markdown")
    _set(nb["cells"][i], SENSOR_MD02)
    c = nb["cells"][_find(nb, "Warm-start protocol    ", "code")]
    _sub(c, 'print(f"Sensor noise (fraction of channel max)         : {DEFAULT_SENSOR_NOISE_PCT}")',
         'print(f"Sensor model                                   : {CFG.describe()}")')
    c = nb["cells"][_find(nb, "obs_one = we.simulate(", "code")]
    _sub(c, "PROTOCOL, seed=0, raw=True)", "PROTOCOL, seed=0, raw=True, **NOISE)")
    c = nb["cells"][_find(nb, "N_REPLICATES = 50", "code")]
    _sub(c, "raw=True, drift_T=sc.drift_T)", "raw=True, drift_T=sc.drift_T, **NOISE)")
    c = nb["cells"][_find(nb, "again = we.simulate(", "code")]
    _sub(c, "raw=True, drift_T=sc4.drift_T)", "raw=True, drift_T=sc4.drift_T, **NOISE)")
    _sub(c, "# thresholds 395 / 5 L/min: sensor noise on Qc is 0.5% of 400 = 2 L/min, so a saturated\n",
         "# thresholds 395 / 5 L/min: sensor noise on Qc is 2 L/min (N3; N0 had 0.5% of 400), so a saturated\n")
    c = nb["cells"][_find(nb, 'data_dir / "observations.npz",', "code")]
    _sub(c, 'data_dir = DATA_DIR; data_dir.mkdir(parents=True, exist_ok=True)\n',
         'data_dir = DATA_DIR; data_dir.mkdir(parents=True, exist_ok=True)\n'
         '# archive the previous dataset if it was generated with another sensor model (N0 files have no key)\n'
         'import shutil\n'
         'if (data_dir / "observations.npz").exists():\n'
         '    with np.load(data_dir / "observations.npz", allow_pickle=False) as _old:\n'
         '        _old_model = str(_old["sensor_model"]) if "sensor_model" in _old.files else "N0"\n'
         '    _arch = data_dir / f"observations_{_old_model}.npz"\n'
         '    if _old_model != SENSOR and not _arch.exists():\n'
         '        shutil.copy2(data_dir / "observations.npz", _arch)\n'
         '        print(f"archived the previous {_old_model} dataset as {_arch.name}")\n')
    _sub(c, 'controller_law=np.array("standard PI: Qc = Qc0 + Kp*[(T-Tsp) + I/tau_i] (fixed 2026-11-08)"),\n',
         'controller_law=np.array("standard PI: Qc = Qc0 + Kp*[(T-Tsp) + I/tau_i] (fixed 2026-11-08)"),\n'
         '    sensor_model=np.array(SENSOR), sensor_noise=np.asarray(CFG.SENSOR_MODELS[SENSOR], np.float64),\n')
    c = nb["cells"][_find(nb, "# Programmatic dataset summary", "code")]
    _sub(c, 'print(f"  controller law      : {ld[\'controller_law\']}")\n',
         'print(f"  controller law      : {ld[\'controller_law\']}")\n'
         'print(f"  sensor model        : {ld[\'sensor_model\']}  sigma [C, T, Tc, Qc] = {ld[\'sensor_noise\']}")\n')
    if "## 10. N3 dataset" not in _src(nb["cells"][-1]):
        nb["cells"].append(_md(INTERP02))
    _clear_outputs(nb)
    _save(nb, p)
    msg.append("02_data_generation.ipynb: N3")
    return msg


# ----------------------------------------------------------------------------- notebook 07
NOTE07 = f"""> **{DATE} revision — production sensor model N3.** Both posteriors and the data now use the
> realistic sensor model N3 (σ = 0.01 mol/L, 0.2 K, 0.2 K, 2 L/min; notebook 33): the closed-loop
> posterior is the N3 production ensemble from `31_budget_convergence_N3.ipynb`
> (`results/31_N3_production.json`), the open-loop posterior is retrained on the N3 open-loop
> simulator (`results/sbi_posterior_open_p1b_N3.pkl`), and the observations come from notebook 02
> run under N3. Result files carry the suffix `_N3`. The executed N0 version (commentary of the
> 2026-11-08 run) is archived as `07_failure_baseline_open_vs_closed_N0.ipynb`.
"""

COMMENT07 = f"""## 7. Results commentary and Claim 1 assessment (N3 run, {DATE}) — to be written after the run

The N0 commentary (2026-11-08) is in `07_failure_baseline_open_vs_closed_N0.ipynb`. Under N0:
CL-SBI on Sc2 recovered β with good coverage while OL-SBI put 94% of its samples outside the prior
(fault accuracy 0.28); on Sc6 the roles reversed. Write the N3 version from sections 3, 3b and 5:
the four-cell table (β mean, cov90, W1, CRPS, fault accuracy, fraction outside the prior), whether
the reversal on Sc6 persists, and whether Claim 1 holds, is strengthened or must be qualified.
"""


def patch_07():
    p = NB_DIR / "07_failure_baseline_open_vs_closed.ipynb"
    msg = [_archive(p, NB_DIR / "07_failure_baseline_open_vs_closed_N0.ipynb",
                    "from nb31_budget_convergence import load_production_posterior",
                    "Notebook 07 (open- vs closed-loop failure baseline)")]
    nb = _load(p)
    if "production sensor model N3" not in _src(nb["cells"][1]):
        nb["cells"].insert(1, _md(NOTE07))
    c = nb["cells"][_find(nb, "load_production_posterior", "code")]
    _sub(c, "from nb31_budget_convergence import load_production_posterior\n",
         "import os, importlib\n"
         'os.environ["CSTR_SENSOR_MODEL"] = "N3"   # production sensor model (notebook 33, 2026-11-09)\n'
         "import system_I_config as CFG, ws_protocols, ws_eval, nb29_ws_study, nb31_budget_convergence\n"
         "for _m in (CFG, ws_protocols, ws_eval, nb29_ws_study, nb31_budget_convergence):\n"
         "    importlib.reload(_m)\n"
         "SENSOR, SFX = CFG.model(), CFG.suffix()\n"
         'assert SENSOR == "N3", SENSOR\n'
         "load_production_posterior = nb31_budget_convergence.load_production_posterior\n")
    _sub(c, '    "data/observations.npz is not the regenerated P1b dataset: re-run notebook 02 first"\n',
         '    "data/observations.npz is not the regenerated P1b dataset: re-run notebook 02 first"\n'
         '_data_model = str(_d["sensor_model"]) if "sensor_model" in _d.files else "N0"\n'
         'assert _data_model == PROD["sensor_model"] == SENSOR, \\\n'
         '    f"sensor model mismatch: data {_data_model}, production posterior {PROD[\'sensor_model\']}, notebook {SENSOR}"\n'
         'print(f"Sensor model: {CFG.describe()} (data and posterior agree)")\n')
    c = nb["cells"][_find(nb, "RUN_OL_TRAINING = ", "code")]
    _sub(c, 'RUN_OL_TRAINING = False   # trained 2026-11-08 (160k, seed 7); True retrains (~18 min)\n',
         'RUN_OL_TRAINING = False   # trained once per sensor model (cached); True retrains (~18 min)\n')
    _sub(c, 'OL_PATH = RESULTS / "sbi_posterior_open_p1b.pkl"', 'OL_PATH = RESULTS / f"sbi_posterior_open_p1b{SFX}.pkl"')
    _sub(c, "x_ol = we.simulate_open_loop(theta_ol, seed=7_007_000)",
         "x_ol = we.simulate_open_loop(theta_ol, seed=7_007_000, **CFG.noise_kw())")
    _sub(c, '"burn_min": 300.0}}, f)', '"burn_min": 300.0, "sensor_model": SENSOR}}, f)')
    i = _find(nb, "## 3. Apply both posteriors to Sc2 and Sc6 observations", "markdown")
    _set(nb["cells"][i], _src(nb["cells"][i]).replace("β=0.70", "β=0.85"))
    for needle, old, new in [
        ('07_cl_vs_ol_prior_leakage.csv', 'RESULTS / "07_cl_vs_ol_prior_leakage.csv"', 'RESULTS / f"07_cl_vs_ol_prior_leakage{SFX}.csv"'),
        ('07_cl_vs_ol_beta_posteriors.png", dpi', 'FIGS / "07_cl_vs_ol_beta_posteriors.png"', 'FIGS / f"07_cl_vs_ol_beta_posteriors{SFX}.png"'),
        ('df_m.to_csv(', 'RESULTS / "07_cl_vs_ol_metrics.csv"', 'RESULTS / f"07_cl_vs_ol_metrics{SFX}.csv"'),
        ('07_cl_vs_ol_joint.png", dpi', 'FIGS / "07_cl_vs_ol_joint.png"', 'FIGS / f"07_cl_vs_ol_joint{SFX}.png"'),
    ]:
        hits = [k for k, cc in enumerate(nb["cells"]) if cc["cell_type"] == "code" and (old in _src(cc) or new in _src(cc))]
        assert len(hits) == 1, (needle, hits)
        _sub(nb["cells"][hits[0]], old, new)
    i = [k for k, cc in enumerate(nb["cells"]) if cc["cell_type"] == "markdown"
         and _src(cc).startswith("## 7. Results commentary")]
    assert len(i) == 1, i
    _set(nb["cells"][i[0]], COMMENT07)
    _clear_outputs(nb)
    _save(nb, p)
    msg.append("07_failure_baseline_open_vs_closed.ipynb: N3")
    return msg


# ----------------------------------------------------------------------------- N0 pins
PINS = {
    "29_warm_start_protocol_study.ipynb": ("import nb29_ws_study as S\n", "nb29_ws_study", "S",
                                           "system_I_config, ws_protocols, ws_eval, nb29_ws_study"),
    "31_budget_convergence.ipynb": ("import nb31_budget_convergence as B\n", "nb31_budget_convergence", "B",
                                    "system_I_config, ws_protocols, ws_eval, nb29_ws_study, nb31_budget_convergence"),
    "03_summary_statistics_design.ipynb": ("import nb03_summary_study as SS, summary_variants as sv\n", "nb03_summary_study", "SS",
                                           "system_I_config, ws_protocols, ws_eval, nb29_ws_study, nb03_summary_study"),
}


def pin_n0():
    msg = []
    for name, (imp, mod, al, mods) in PINS.items():
        p = NB_DIR / name
        nb = _load(p)
        hits = [k for k, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and imp in _src(c)]
        if not hits:
            msg.append(f"{name}: already pinned"); continue
        assert len(hits) == 1, (name, hits)
        c = nb["cells"][hits[0]]
        extra = ", summary_variants as sv" if al == "SS" else ""
        pin = ('os.environ["CSTR_SENSOR_MODEL"] = "N0"   # N0 run; default model is N3 since 2026-11-09\n'
               f"import importlib, {mods}{extra}\n"
               f"for _m in ({mods}):\n"
               "    importlib.reload(_m)\n"
               f"{al} = {mod}\n"
               f'assert {al}.SENSOR == "N0", {al}.SENSOR\n')
        s = _src(c).replace(imp, pin)
        if "import json, os," not in s:
            s = s.replace("import json, ", "import json, os, ", 1)
        _set(c, s)                       # outputs kept: the code that produced them is unchanged for N0
        _save(nb, p)
        msg.append(f"{name}: pinned to N0 (outputs kept)")
    return msg


if __name__ == "__main__":
    for m in patch_02() + patch_07() + pin_n0():
        print(m)
