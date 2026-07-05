"""CLI: generate one SNPE training bank (prior draws -> summaries).

Usage:
    python scripts/gen_bank.py --structure S-B --noise 0.003 --out data/wu2003_sbi_train_sb_n03.npz
"""

from __future__ import annotations

import argparse
import pathlib

from sbi_pipeline import generate_training_bank


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", choices=["S-B", "S-A"], required=True)
    p.add_argument("--noise", type=float, required=True, help="noise_pct, e.g. 0.003 or 0.01")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--n-train", type=int, default=15000)
    p.add_argument("--seed", type=int, default=20260625)
    args = p.parse_args()

    generate_training_bank(
        structure=args.structure,
        noise_pct=args.noise,
        out_path=args.out,
        n_train=args.n_train,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
