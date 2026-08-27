"""Full-budget System II matched ongoing-degradation confirmation.

This is the production-scale follow-up to the reduced Stage 3 diagnostic:
S-B, n_train=15,000, eight zuko_nsf 60/3 seeds, with per-draw
scenario-specific steady-state warm starts for both training and validation.

Outputs are written without overwriting the current production posterior:

- data/wu2003_sbi_train_sb_n03_matched_full.npz
- sbi-logs/wu2003_posterior_variant_matched_full_sb_seed{seed}.pkl
- sbi-logs/variant_matched_full_sb_seed{seed}_results.json
- sbi-logs/wu2003_posterior_sb_matched_full.pkl
- results/stage3_sysII_matched_full_validation.json
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbi_pipeline import (  # noqa: E402
    DATA,
    PARAM_NAMES,
    SBI_LOGS,
    coverage_at_90,
    generate_training_bank,
    run_sbc,
    save_variant,
    train_posterior,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

STRUCTURE = "S-B"
NOISE_PCT = 0.003
N_TRAIN = 15_000
BANK_SEED = 20260826
SEEDS = list(range(8))
ARCH = "zuko_nsf"
HIDDEN = 60
TRANSFORMS = 3
N_SBC = 400
N_POST = 200
N_COVERAGE_TRIALS = 100
N_CONFIRM_SBC = 800

BANK_PATH = DATA / "wu2003_sbi_train_sb_n03_matched_full.npz"
SUMMARY_PATH = RESULTS / "stage3_sysII_matched_full_validation.json"
BEST_POSTERIOR_PATH = SBI_LOGS / "wu2003_posterior_sb_matched_full.pkl"


def min_ks(sbc: dict) -> float:
    return min(float(sbc[name]["ks_pvalue"]) for name in PARAM_NAMES)


def main() -> None:
    t_all = time.time()
    print("=== System II full-budget matched ongoing-degradation confirmation ===")
    print(f"structure={STRUCTURE}, n_train={N_TRAIN}, seeds={SEEDS}")
    print(f"bank={BANK_PATH}")

    t0 = time.time()
    generate_training_bank(
        structure=STRUCTURE,
        noise_pct=NOISE_PCT,
        out_path=BANK_PATH,
        n_train=N_TRAIN,
        seed=BANK_SEED,
        scenario_specific_warm_start=True,
        progress_every=250,
    )
    bank_wall_s = time.time() - t0

    per_seed: list[dict] = []
    posteriors = {}

    for seed in SEEDS:
        variant = f"matched_full_sb_seed{seed}"
        print(f"\n=== Training/evaluating {variant} ===")
        t_seed = time.time()
        posterior = train_posterior(
            bank_path=BANK_PATH,
            arch=ARCH,
            hidden_features=HIDDEN,
            num_transforms=TRANSFORMS,
            torch_seed=seed,
        )
        train_wall_s = time.time() - t_seed

        sbc = run_sbc(
            posterior,
            structure=STRUCTURE,
            noise_pct=NOISE_PCT,
            n_sbc=N_SBC,
            n_post=N_POST,
            seed=12345 + seed,
            scenario_specific_warm_start=True,
        )
        coverage = coverage_at_90(
            posterior,
            structure=STRUCTURE,
            noise_pct=NOISE_PCT,
            n_trials=N_COVERAGE_TRIALS,
            seed=777 + seed,
            scenario_specific_warm_start=True,
        )

        meta = {
            "variant": variant,
            "structure": STRUCTURE,
            "bank": str(BANK_PATH),
            "noise_pct": NOISE_PCT,
            "arch": ARCH,
            "hidden_features": HIDDEN,
            "num_transforms": TRANSFORMS,
            "n_train": N_TRAIN,
            "bank_seed": BANK_SEED,
            "torch_seed": seed,
            "scenario_specific_warm_start": True,
            "n_sbc": N_SBC,
            "n_post": N_POST,
            "n_coverage_trials": N_COVERAGE_TRIALS,
            "train_wall_s": train_wall_s,
        }
        save_variant(posterior, sbc, coverage, variant, meta)
        seed_summary = {
            "seed": seed,
            "variant": variant,
            "meta": meta,
            "sbc": sbc,
            "coverage": coverage,
            "min_ks_pvalue": min_ks(sbc),
        }
        per_seed.append(seed_summary)
        posteriors[seed] = posterior
        print(f"[stage3-full] seed {seed}: min KS p={seed_summary['min_ks_pvalue']:.4f}")

        with open(SUMMARY_PATH, "w") as f:
            json.dump({
                "status": "in_progress",
                "bank_path": str(BANK_PATH),
                "bank_wall_s": bank_wall_s,
                "per_seed": per_seed,
                "elapsed_wall_s": time.time() - t_all,
            }, f, indent=2)

    best = max(per_seed, key=lambda row: row["min_ks_pvalue"])
    best_seed = int(best["seed"])
    n_pass = sum(1 for row in per_seed if row["min_ks_pvalue"] > 0.05)
    print(f"\n[stage3-full] {n_pass}/{len(SEEDS)} seeds pass min KS p>0.05")
    print(f"[stage3-full] selected seed {best_seed} (min KS p={best['min_ks_pvalue']:.4f})")

    print(f"\n=== Independent N={N_CONFIRM_SBC} SBC confirmation for selected seed {best_seed} ===")
    confirm_sbc = run_sbc(
        posteriors[best_seed],
        structure=STRUCTURE,
        noise_pct=NOISE_PCT,
        n_sbc=N_CONFIRM_SBC,
        n_post=N_POST,
        seed=54321 + best_seed,
        scenario_specific_warm_start=True,
    )
    confirm_min_ks = min_ks(confirm_sbc)
    print(f"[stage3-full] selected seed {best_seed}: confirmation min KS p={confirm_min_ks:.4f}")

    best_payload = {
        "posterior": posteriors[best_seed],
        "selected_seed": best_seed,
        "N_TRAIN": N_TRAIN,
        "structure": STRUCTURE,
        "noise_pct": NOISE_PCT,
        "arch": ARCH,
        "hidden_features": HIDDEN,
        "num_transforms": TRANSFORMS,
        "bank_path": str(BANK_PATH),
        "scenario_specific_warm_start": True,
        "per_seed": per_seed,
        "selected_seed_confirmation_sbc": confirm_sbc,
    }
    with open(BEST_POSTERIOR_PATH, "wb") as f:
        pickle.dump(best_payload, f)
    print(f"[stage3-full] best posterior saved to {BEST_POSTERIOR_PATH}")

    summary = {
        "status": "complete",
        "structure": STRUCTURE,
        "noise_pct": NOISE_PCT,
        "n_train": N_TRAIN,
        "bank_seed": BANK_SEED,
        "bank_path": str(BANK_PATH),
        "bank_wall_s": bank_wall_s,
        "seeds": SEEDS,
        "arch": ARCH,
        "hidden_features": HIDDEN,
        "num_transforms": TRANSFORMS,
        "n_sbc": N_SBC,
        "n_post": N_POST,
        "n_coverage_trials": N_COVERAGE_TRIALS,
        "n_confirm_sbc": N_CONFIRM_SBC,
        "per_seed": per_seed,
        "n_pass_min_ks_gt_0p05": n_pass,
        "selected_seed": best_seed,
        "selected_min_ks_pvalue": best["min_ks_pvalue"],
        "selected_confirmation_sbc": confirm_sbc,
        "selected_confirmation_min_ks_pvalue": confirm_min_ks,
        "best_posterior_path": str(BEST_POSTERIOR_PATH),
        "elapsed_wall_s": time.time() - t_all,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stage3-full] summary saved to {SUMMARY_PATH}")
    print(f"[stage3-full] total wall time: {summary['elapsed_wall_s']:.0f}s")


if __name__ == "__main__":
    main()