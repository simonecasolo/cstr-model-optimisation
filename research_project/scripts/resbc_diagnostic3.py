"""Final N=800 confirmation for the top-2 S-B v2 ensemble candidates (seed5, seed6)."""
import pathlib, pickle, json, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sbi_pipeline import run_sbc

SBI_LOGS = pathlib.Path(__file__).resolve().parent.parent / "sbi-logs"
for name in ["variant_A_sb_v2_seed5", "variant_A_sb_v2_seed6"]:
    with open(SBI_LOGS / f"wu2003_posterior_{name}.pkl", "rb") as f:
        posterior = pickle.load(f)["posterior"]
    print(f"\n=== {name}: N_SBC=800, seed=777 ===")
    sbc = run_sbc(posterior, structure="S-B", noise_pct=0.003, n_sbc=800, n_post=200, seed=777)
    with open(SBI_LOGS / f"resbc800_{name}.json", "w") as f:
        json.dump(sbc, f, indent=2)
