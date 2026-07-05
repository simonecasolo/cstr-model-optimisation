"""Train 3 more S-B seeds (5,6,7) on the existing compressed bank, SBC-evaluated at N=400
directly (per the N=200-underpowered finding) instead of the old N=200 protocol."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sbi_pipeline import train_posterior, run_sbc, coverage_at_90, save_variant

BANK = pathlib.Path(__file__).resolve().parent.parent / "data" / "wu2003_sbi_train_sb_n03_v2.npz"

for seed in [5, 6, 7]:
    print(f"\n=== seed {seed} ===")
    posterior = train_posterior(BANK, arch="zuko_nsf", hidden_features=60, num_transforms=3, torch_seed=seed)
    sbc = run_sbc(posterior, structure="S-B", noise_pct=0.003, n_sbc=400, n_post=200, seed=999)
    cov = coverage_at_90(posterior, structure="S-B", noise_pct=0.003, n_trials=100)
    meta = {"variant": f"A_sb_v2_seed{seed}", "structure": "S-B", "bank": str(BANK),
            "noise_pct": 0.003, "arch": "zuko_nsf", "hidden_features": 60, "num_transforms": 3,
            "n_sbc": 400}
    save_variant(posterior, sbc, cov, f"A_sb_v2_seed{seed}", meta)
