"""Inspect SNPE training-loss history for representative "good" (seed 0, 3 --
pass both SBC sets) vs "bad" (seed 1, 8 -- catastrophic alpha failure) seeds
on the S-C.2 lognormal bank, to test whether the alpha-instability pattern is
an optimization/fit-quality problem (bad seeds converge to a visibly worse
loss) or a different-local-optimum problem (similar loss, different basin).
`inference.summary` isn't saved in the existing posterior pickles, so this
retrains fresh (same seed, same recipe -- reproducible on CPU).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from cstr_sbi.recycle.priors import truncated_lognormal_5d

DATA = ROOT / "data"
FIGURES = ROOT / "figures"

bank = np.load(DATA / "wu2003_sbi_train_sc2_lognormal_true_n1000_stage1bfix.npz")
theta_train = np.asarray(bank["thetas"], dtype=np.float32)
summary_train = np.asarray(bank["summaries"], dtype=np.float32)

SEEDS = {0: "good (pass both SBC)", 3: "good (pass both SBC)",
         1: "bad (alpha catastrophic)", 8: "bad (alpha catastrophic, worst: 1.3e-21)"}
COLORS = {0: "#009E73", 3: "#0072B2", 1: "#D55E00", 8: "#CC79A7"}

histories = {}
for seed in SEEDS:
    torch.manual_seed(seed)
    prior = truncated_lognormal_5d()
    inference = SNPE(prior=prior, density_estimator=posterior_nn(
        model="zuko_nsf", hidden_features=60, num_transforms=3, z_score_x="none"))
    inference.append_simulations(torch.tensor(theta_train), torch.tensor(summary_train))
    inference.train(max_num_epochs=120, validation_fraction=0.1, stop_after_epochs=12, show_train_summary=False)
    s = inference.summary
    histories[seed] = {"train": s["training_loss"], "val": s["validation_loss"],
                        "epochs": s["epochs_trained"][0], "best_val": s["best_validation_loss"][0]}
    print(f"seed {seed} ({SEEDS[seed]}): epochs_trained={histories[seed]['epochs']}, "
          f"best_validation_loss={histories[seed]['best_val']:.4f}, "
          f"final_train_loss={s['training_loss'][-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for seed, label in SEEDS.items():
    h = histories[seed]
    axes[0].plot(h["train"], color=COLORS[seed], label=f"seed {seed} ({label})", lw=1.6)
    axes[1].plot(h["val"], color=COLORS[seed], label=f"seed {seed} ({label})", lw=1.6)
axes[0].set_title("Training loss (SNPE, -log q(theta|x))")
axes[1].set_title("Validation loss")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8.5)
fig.suptitle("SNPE training/validation loss: 'good' vs 'bad' seeds, S-C.2 lognormal bank, z_score_x='none'")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = FIGURES / "44_training_loss_good_vs_bad_seeds.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")

print("\n=== Summary table ===")
for seed, label in SEEDS.items():
    h = histories[seed]
    print(f"seed {seed:2d} ({label:45s}): epochs={h['epochs']:4d}  best_val_loss={h['best_val']:.4f}  "
          f"final_train_loss={h['train'][-1]:.4f}")
