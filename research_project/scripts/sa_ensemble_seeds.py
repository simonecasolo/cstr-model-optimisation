"""Train an 8-seed S-A ensemble on the compressed bank, SBC-evaluated at N=400 directly
(N=200 shown underpowered for this problem -- see HANDOFF.md / memory finding_sbc_power_eta_col)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sbi_pipeline import train_posterior, run_sbc, coverage_at_90, save_variant

BANK = pathlib.Path(__file__).resolve().parent.parent / "data" / "wu2003_sbi_train_sa_n03_v2.npz"

for seed in range(8):
    print(f"\n=== S-A seed {seed} ===")
    posterior = train_posterior(BANK, arch="zuko_nsf", hidden_features=60, num_transforms=3, torch_seed=seed)
    sbc = run_sbc(posterior, structure="S-A", noise_pct=0.003, n_sbc=400, n_post=200, seed=999)
    cov = coverage_at_90(posterior, structure="S-A", noise_pct=0.003, n_trials=100)
    meta = {"variant": f"A_sa_v2_seed{seed}", "structure": "S-A", "bank": str(BANK),
            "noise_pct": 0.003, "arch": "zuko_nsf", "hidden_features": 60, "num_transforms": 3,
            "n_sbc": 400}
    save_variant(posterior, sbc, cov, f"A_sa_v2_seed{seed}", meta)
