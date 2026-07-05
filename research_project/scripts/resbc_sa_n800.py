"""N=800 corroboration for the two least-bad S-A candidates (seed6, seed13)."""
import pathlib, pickle, json, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sbi_pipeline import run_sbc

SBI_LOGS = pathlib.Path(__file__).resolve().parent.parent / "sbi-logs"
for name in ["variant_A_sa_v2_seed6", "variant_A_sa_v2_seed13"]:
    with open(SBI_LOGS / f"wu2003_posterior_{name}.pkl", "rb") as f:
        posterior = pickle.load(f)["posterior"]
    print(f"\n=== {name}: N_SBC=800, seed=321 ===")
    sbc = run_sbc(posterior, structure="S-A", noise_pct=0.003, n_sbc=800, n_post=200, seed=321)
    with open(SBI_LOGS / f"resbc800_{name}.json", "w") as f:
        json.dump(sbc, f, indent=2)
