"""Parallel worker for `generate_sb3_wideprior_bank.py`.

Generates ONE independent chunk of accepted (theta4, summary) rows by reusing
`generate_bank` from the sequential script unchanged (same prior, same
matched-protocol simulation, same checkpoint/resume convention). Each worker
process gets its own torch RNG state (torch seeds nondeterministically per
process), so chunks draw independent thetas; `--seed` only sets the sensor-noise
RNG and is kept distinct per chunk.

Usage:
    python scripts/generate_sb3_wideprior_chunk.py --out PATH.npz --n-rows N --seed S --label NAME

Idempotent: if PATH.npz exists it is loaded and nothing is regenerated; if only
PATH.partial.npz exists it resumes from it. Assemble chunks afterwards with
`scripts/assemble_sb3_wideprior_bank.py`.
"""
from __future__ import annotations

import argparse
import pathlib

from generate_sb3_wideprior_bank import generate_bank, log

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--n-rows", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    log(f"=== chunk worker '{args.label}' started: out={args.out.name}, n_rows={args.n_rows}, seed={args.seed} ===")
    generate_bank(args.out, args.n_rows, seed=args.seed, label=args.label)
    log(f"=== chunk worker '{args.label}' finished ===")
