"""Read-only: fraction of samples (and of windows) with Qc >= 395 L/min (the jacket-capacity limit) for every
Part A condition x scenario, from results/34_N3/A_*.npz (raw windows; channel 3 = Qc). Joins the Part A beta metrics.
Written 2026-11-09 for the notebook-34 commentary; changes no notebook, rule or result."""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(ROOT / "results/34_N3/A_*.npz"))):
    cond = os.path.basename(f)[2:-4]
    if cond == "calibration_nominal":
        continue
    d = np.load(f, allow_pickle=True); q = d["raw"][..., 3]; T = d["raw"][..., 1]
    for sc in sorted(set(d["sc_name"])):
        m = d["sc_name"] == sc
        rows.append(dict(condition=cond, scenario=sc.split("_")[0], frac_samples_ge395=float((q[m] >= 395).mean()),
                         frac_windows_any_ge395=float((q[m] >= 395).any(axis=1).mean()), frac_windows_half_ge395=float(((q[m] >= 395).mean(axis=1) > 0.5).mean()),
                         Qc_mean=float(q[m].mean()), T_minus_Tsp=float(T[m].mean() - 312.5)))
sat = pd.DataFrame(rows)
met = pd.read_csv(ROOT / "results/34_N3_A_metrics.csv")
met = met[met.posterior == met.posterior.iloc[0]]                      # production posterior only (first listed)
met = met.assign(scenario=met.scenario.str.split("_").str[0])
out = sat.merge(met[["condition", "scenario", "bias_b", "z_bias_b", "cov90_truth_b", "class_acc"]], on=["condition", "scenario"], how="left")
out.to_csv(ROOT / "logs/34_N3_saturation_fractions.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(out.round(3).to_string(index=False))
