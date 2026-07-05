"""Reload already-trained variant posteriors and rerun SBC with larger N_SBC / different rng
to test whether borderline/failing eta_col p-values are KS-test noise at N_SBC=200.

No retraining needed - this only costs SBC-sampling time (~30-60s per posterior).
"""
import pathlib
import pickle
import json
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sbi_pipeline import run_sbc

SBI_LOGS = pathlib.Path(__file__).resolve().parent.parent / "sbi-logs"

TARGETS = [
    ("variant_A_sb", "S-B", 0.003),
    ("variant_A_sb_seed1", "S-B", 0.003),
    ("variant_A_sb_seed2", "S-B", 0.003),
    ("variant_A_sb_v2_seed0", "S-B", 0.003),
    ("variant_A_sb_v2_seed1", "S-B", 0.003),
    ("variant_A_sb_v2_seed2", "S-B", 0.003),
    ("variant_A_sb_v2_seed3", "S-B", 0.003),
    ("variant_A_sb_v2_seed4", "S-B", 0.003),
]

results = {}
for name, structure, noise in TARGETS:
    pkl_path = SBI_LOGS / f"wu2003_posterior_{name}.pkl"
    if not pkl_path.exists():
        print(f"[skip] {pkl_path} not found")
        continue
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    posterior = obj["posterior"]
    print(f"\n=== {name} : re-SBC with N_SBC=400, seed=999 ===")
    sbc = run_sbc(posterior, structure=structure, noise_pct=noise, n_sbc=400, n_post=200, seed=999)
    results[name] = {k: v["ks_pvalue"] if isinstance(v, dict) and "ks_pvalue" in v else v
                      for k, v in sbc.items() if k in ("eta_col", "alpha", "n_valid")}
    out_path = SBI_LOGS / f"resbc400_{name}.json"
    with open(out_path, "w") as f:
        json.dump(sbc, f, indent=2)

print("\n\n=== SUMMARY (eta_col KS p-value, N_SBC=400 vs original N_SBC=200) ===")
orig = {
    "variant_A_sb": 0.8933, "variant_A_sb_seed1": 0.0000079, "variant_A_sb_seed2": 0.9599,
    "variant_A_sb_v2_seed0": 0.0006986, "variant_A_sb_v2_seed1": 0.01458,
    "variant_A_sb_v2_seed2": 0.14640, "variant_A_sb_v2_seed3": 0.03411, "variant_A_sb_v2_seed4": 0.26823,
}
for name in results:
    new_p = results[name]["eta_col"]
    print(f"{name:30s} orig_p={orig.get(name, float('nan')):.4f}  n400_p={new_p:.4f}")
