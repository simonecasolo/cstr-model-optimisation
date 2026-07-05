"""Push N_SBC even higher (800) on the two v2 seeds that passed at N=400, to check
whether they hold up (root-cause investigation: is 400 still underpowered?)."""
import pathlib, pickle, json, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sbi_pipeline import run_sbc

SBI_LOGS = pathlib.Path(__file__).resolve().parent.parent / "sbi-logs"
TARGETS = ["variant_A_sb_v2_seed2", "variant_A_sb_v2_seed4"]

for name in TARGETS:
    pkl_path = SBI_LOGS / f"wu2003_posterior_{name}.pkl"
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    posterior = obj["posterior"]
    print(f"\n=== {name}: N_SBC=800, seed=555 ===")
    sbc = run_sbc(posterior, structure="S-B", noise_pct=0.003, n_sbc=800, n_post=200, seed=555)
    with open(SBI_LOGS / f"resbc800_{name}.json", "w") as f:
        json.dump(sbc, f, indent=2)
