"""CLI: train + SBC-verify one variant, given a pre-generated training bank.

Usage:
    python scripts/run_variant.py --variant A_sb --structure S-B --bank data/wu2003_sbi_train_sb_n03.npz \\
        --noise 0.003 --arch zuko_nsf --hidden 60 --transforms 3
"""

from __future__ import annotations

import argparse
import pathlib

from sbi_pipeline import train_posterior, run_sbc, coverage_at_90, save_variant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, help="variant name, e.g. A_sb")
    p.add_argument("--structure", choices=["S-B", "S-A"], required=True)
    p.add_argument("--bank", type=pathlib.Path, required=True)
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--arch", choices=["nsf", "zuko_nsf"], required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--transforms", type=int, required=True)
    p.add_argument("--n-sbc", type=int, default=200)
    p.add_argument("--n-post", type=int, default=200)
    p.add_argument("--n-coverage-trials", type=int, default=100)
    p.add_argument("--torch-seed", type=int, default=0)
    args = p.parse_args()

    posterior = train_posterior(
        bank_path=args.bank, arch=args.arch,
        hidden_features=args.hidden, num_transforms=args.transforms,
        torch_seed=args.torch_seed,
    )
    sbc_results = run_sbc(
        posterior, structure=args.structure, noise_pct=args.noise,
        n_sbc=args.n_sbc, n_post=args.n_post,
    )
    coverage_results = coverage_at_90(
        posterior, structure=args.structure, noise_pct=args.noise,
        n_trials=args.n_coverage_trials,
    )
    meta = {
        "variant": args.variant,
        "structure": args.structure,
        "bank": str(args.bank),
        "noise_pct": args.noise,
        "arch": args.arch,
        "hidden_features": args.hidden,
        "num_transforms": args.transforms,
    }
    save_variant(posterior, sbc_results, coverage_results, args.variant, meta)


if __name__ == "__main__":
    main()
