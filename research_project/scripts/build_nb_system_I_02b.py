"""Build / execute notebooks/system_I/02b_measured_inlet_dataset.ipynb (System I, Phase 3 step a2).

Writes data/observations_measured_inlet.npz (D1) with the helper scripts/ws_phase3.py. Checks fixed in
docs/Manuscript/system_i_N3_runbook.md section 2f. If the notebook already exists, a written section-4
commentary ("## 4. Interpretation" not a placeholder) is kept when rebuilding.

    python scripts/build_nb_system_I_02b.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "02b_measured_inlet_dataset.ipynb"

HEADER = """# 02b — Measured-inlet dataset D1 (System I, Phase 3)

**Why.** The production posterior (notebook 36) takes the *measured* feed and coolant inlet temperatures of every window as
inputs. `data/observations.npz` (notebook 02, **D0**) was generated at the nominal inlet and carries no inlet measurement, so it
cannot feed the production posterior. D0 stays as it is (the nominal-inlet dataset: naive model, Claim 1 in notebook 07).
This notebook writes **D1 = `data/observations_measured_inlet.npz`**.

**Design (fixed before running; `scripts/ws_phase3.py`).** Closed-loop scenarios Sc1–Sc5 and Sc7 × 50 windows (Sc0 and Sc6 are **not**
in D1: user decision 2026-11-09), the same θ per scenario as notebook 02, protocol P2 (300-min history), N3 sensors, Sc7's +2 K T-sensor
drift after the sensor layer. Per window: the **true** inlet [Ci, Ti, Tci] drawn from the nb35/nb36 training distribution
(Ti, Tci ~ U(295, 299) K independently; Ci = 0.97 (1 + N(0, 0.01²)) mol/L, unmeasured) and the **measured** (Ti, Tci) = true + ε N(0, 1)
calibration offset (ε = 0.1 K, fixed per window) + window-mean white noise N(0, 0.2²/120) K, exactly nb35's sensor model. New seed
base (3 700 000): D1 never reuses the windows of notebook 02 or 35.

**Checks (runbook §2f; any failure stops the notebook).** Shapes and finiteness; true Ti, Tci within [295, 299] K; Ci relative sd within 20%
of 0.01; measured − true sd per channel within 10% of √(0.1² + 0.0183²) = 0.1017 K; regeneration from the stored seeds is bit-identical;
Sc7's drift is exactly +2 K; the production (34-D) and naive (32-D) features are finite and the production features equal
`nb36.features_from_raw`. Saturation fractions per scenario are reported, not gated (with the inlet varying the β = 0.85 scenarios
saturate often, notebook 34).
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
import importlib, system_I_config, ws_protocols, ws_eval, ws_inlet, nb35_measured_inlet, nb36_production_measured_inlet, ws_phase3
for _m in (system_I_config, ws_protocols, ws_eval, ws_inlet, nb35_measured_inlet, nb36_production_measured_inlet, ws_phase3):
    importlib.reload(_m)
P3, M = ws_phase3, nb35_measured_inlet
print("Sensor model:", system_I_config.describe(), "| protocol:", M.PROTOCOL)
print("D1 will be written to", P3.D1_PATH.relative_to(ROOT))
"""),
    ("md", "## 1. Generate D1 and run the checks"),
    ("code", """t0 = time.time()
d = P3.generate_d1()
print(f"simulated {d['x'].shape} windows in {time.time() - t0:.1f}s")
n = len(d["x"]); sid = d["scenario_id"]
assert d["x"].shape == (300, 120, 4) and np.isfinite(d["x"]).all(), d["x"].shape
assert sorted(set(sid.tolist())) == [1, 2, 3, 4, 5, 7] and all((sid == s).sum() == 50 for s in (1, 2, 3, 4, 5, 7))
it, im = d["inlet_true"], d["inlet_measured"]
assert it[:, 1:].min() >= 295.0 and it[:, 1:].max() <= 299.0, (it[:, 1:].min(), it[:, 1:].max())
ci_rel = it[:, 0].std() / it[:, 0].mean()
assert abs(ci_rel / 0.01 - 1) < 0.20, ci_rel
dev = im - it[:, 1:]
want = float(np.sqrt(0.1 ** 2 + M.WHITE_SD ** 2))
assert np.all(np.abs(dev.std(axis=0) / want - 1) < 0.10), (dev.std(axis=0), want)
print(f"true Ti, Tci in [{it[:,1:].min():.2f}, {it[:,1:].max():.2f}] K | Ci relative sd {ci_rel:.4f} | measured - true sd per channel (Ti, Tci) = {np.round(dev.std(axis=0), 4)} (expected {want:.4f})")
print("corr(true Ti, true Tci) =", round(float(np.corrcoef(it[:, 1], it[:, 2])[0, 1]), 3), "(independent draws)")
"""),
    ("code", """# regeneration from the stored seeds is bit-identical; Sc7's drift is exactly +2 K
d2 = P3.generate_d1()
for k in ("x", "inlet_true", "inlet_measured", "theta", "scenario_id"):
    assert np.array_equal(d[k], d2[k]), k
import ws_inlet as wi
m7 = sid == 7
raw7 = np.asarray(wi.simulate(d["theta"][:, 2:], d["inlet_true"], M.PROTOCOL, int(d["seeds"][1]), raw=True, **M.NOISE))
drift = float((d["x"][m7][:, :, 1] - raw7[m7][:, :, 1]).mean())
assert abs(drift - 2.0) < 1e-3, drift
print(f"regeneration bit-identical: True | Sc7 drift added: {drift:.4f} K")
# features
f_prod = P3.production_features(d); f_naive = P3.naive_features(d)
assert f_prod.shape == (300, 34) and f_naive.shape == (300, 32) and np.isfinite(f_prod).all() and np.isfinite(f_naive).all()
ref = nb36_production_measured_inlet.features_from_raw(d["x"], d["inlet_measured"])
assert np.array_equal(f_prod, ref)
print("features: production", f_prod.shape, "naive", f_naive.shape, "| all finite, production equals nb36.features_from_raw")
"""),
    ("md", "## 2. Saturation per scenario and the inlet statistics (reported, not gated)"),
    ("code", """from cstr_sbi.scenarios import SCENARIO_CONFIGS
rows = []
d0 = P3.load_d0()
for s in (1, 2, 3, 4, 5, 7):
    m = sid == s; q = d["x"][m][:, :, 3]; T = d["x"][m][:, :, 1]
    m0 = d0["scenario_id"] == s; q0 = d0["x"][m0][:, :, 3]
    rows.append(dict(scenario=f"Sc{s}", alpha=float(d["theta"][m][0, 2]), beta=float(d["theta"][m][0, 3]),
                     D1_frac_samples_Qc_ge395=float((q >= 395).mean()), D1_frac_windows_any_ge395=float((q >= 395).any(axis=1).mean()),
                     D0_frac_samples_Qc_ge395=float((q0 >= 395).mean()),
                     D1_Qc_mean=float(q.mean()), D0_Qc_mean=float(q0.mean()), D1_T_minus_Tsp=float(T.mean() - 312.5)))
sat = pd.DataFrame(rows); display(sat.round(3))
print("D0 = nominal inlet (observations.npz); D1 = inlet varying (+-2 K) with measured temperatures.")
"""),
    ("md", "## 3. Save D1"),
    ("code", """P3.D1_PATH.parent.mkdir(exist_ok=True)
np.savez(P3.D1_PATH, **d)
with np.load(P3.D1_PATH, allow_pickle=False) as f:
    keys = list(f.files); chk = {k: f[k] for k in f.files}
for k in ("x", "inlet_true", "inlet_measured", "theta"):
    assert np.array_equal(chk[k], d[k]), k
print("Saved", P3.D1_PATH.relative_to(ROOT), "| keys:", keys)
print("size %.2f MB" % (P3.D1_PATH.stat().st_size / 1e6))
sat.to_csv(ROOT / "results" / "02b_N3_saturation.csv", index=False)
print(f"notebook wall time {(time.time() - t0) / 60:.1f} min")
"""),
    ("md", """## 4. Interpretation — to be written after the run
Report: the saturation fractions per scenario against D0 (how often the inlet pushes the β = 0.85 scenarios to the jacket limit),
the inlet and measurement statistics, and what D1 is for (production posterior; the naive posterior on D1 only in the sensitivity section).
Tie to the storyline (inlet conditions as a source of silent failure; the production model as the remedy)."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 4. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 4. Interpretation"):
                    c.source = kept[-1]
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    if execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print(build(execute="--execute" in sys.argv))
