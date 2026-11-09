"""nb22 follow-up: re-run Theory F (raw-trajectory CNN embedding, from
notebooks/system_I/20_alpha_beta_correlation_mechanism_theories.ipynb) under
the CORRECTED warm-start and the now-determined-optimal n_sims=50,000.

Context: Theory F originally (pre-fix, capped at n=20,000 after a 7h41m
stall at n=40,000 -- see build_nb_system_I_20.py's own markdown cell) found
the CNN-raw-trajectory posterior's Corr(alpha,beta) did NOT shrink toward
zero relative to the hand-crafted-summary posterior, and instead FLIPPED
SIGN at 5/6 closed-loop scenarios (e.g. Sc4: hand-crafted -0.893 vs
CNN +0.868). That whole result predates both the warm-start fix and the
nb22 budget-convergence study (which found n=50,000 is the smallest budget
passing SBC + anti-shrinkage + Corr-convergence simultaneously, mean
Corr(alpha,beta)@Sc4 = -0.910 across 5 seeds).

Cost-checked before launch (see HANDOFF.md): benchmarked n=1000 (29s) and
n=5000 (142s) CNN training under the corrected warm-start -- scales
linearly (~28ms/sample, ~1s fixed overhead), NO sign of the old pre-fix
pathological blowup (which stalled for 7h41m at n=40,000 with zero
progress). Extrapolated n=50,000 cost: ~23-24 min/seed. 3 seeds requested
by the coordinator (not a new headline claim, a comparison check) ->
~70-75 min total, self-caffeinated background run with progress sentinels,
same as nb22's main script.
"""

from __future__ import annotations

import datetime as _dt
import pickle
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cstr_sbi.inference import train_sbi_posterior, simulation_wrapper_sbi_raw
from cstr_sbi.priors import box_uniform_2d
from cstr_sbi.physics import NOMINAL_INLET_CL, NOMINAL_CTRL
from cstr_sbi.scenarios import SCENARIO_CONFIGS
from sbi.neural_nets.embedding_nets import CNNEmbedding

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_SIMULATIONS = 50_000
SEEDS = [0, 1, 2]
GRID_SCENARIOS = [  # Sc4 is the priority; the rest are cheap add-ons (sampling only)
    "Sc4_closed_combined", "Sc1_closed_healthy", "Sc2_closed_fouling",
    "Sc3_closed_decay", "Sc5_closed_saturated", "Sc7_closed_drift",
]


def mark(stage: str) -> None:
    (RESULTS / f"22_progress_cnn_{stage}.txt").write_text(
        f"{time.time()}\n{_dt.datetime.now().isoformat()}\n"
    )


def build_embedding_net():
    """IDENTICAL architecture to 04b_embedding_net_study.ipynb / nb20's Theory F
    (same input_shape/channels/layers/output_dim) -- deliberately NOT modified,
    since the point of this recheck is to isolate the warm-start-fix effect,
    not to also change the embedding architecture."""
    return CNNEmbedding(
        input_shape=(120,), in_channels=4, out_channels_per_layer=[16, 32],
        num_conv_layers=2, num_linear_layers=2, num_linear_units=64,
        output_dim=20, kernel_size=5, pool_kernel_size=2,
    )


def corr_ab(posterior, x, n_draw: int = 5000, seed: int = 0):
    torch.manual_seed(seed)
    samp = posterior.sample((n_draw,), x=x, show_progress_bars=False).numpy()
    return float(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1]), samp


def main() -> None:
    mark("start")
    prior = box_uniform_2d()

    def obs_x_raw(sc_name: str, seed: int = 999) -> torch.Tensor:
        sc = SCENARIO_CONFIGS[sc_name]
        theta = torch.tensor([[sc.alpha, sc.beta]], dtype=torch.float32)
        return simulation_wrapper_sbi_raw(
            theta, inlet=NOMINAL_INLET_CL, ctrl=NOMINAL_CTRL,
            scenario_specific_warm_start=True, seed=seed,
        )

    posts = {}
    for seed in SEEDS:
        tag = f"seed{seed}_n{N_SIMULATIONS}"
        cache_path = RESULTS / f"22_theoryF_cnn_posterior_{tag}.pkl"
        if cache_path.exists():
            mark(f"skip_{tag}")
            print(f"{tag}: cached, loading", flush=True)
            with open(cache_path, "rb") as f:
                posts[seed] = pickle.load(f)["posterior"]
            continue
        mark(f"training_{tag}")
        print(f"{tag}: training CNN embedding (n={N_SIMULATIONS})...", flush=True)
        t0 = time.time()
        try:
            emb = build_embedding_net()
            post, meta = train_sbi_posterior(
                prior, n_simulations=N_SIMULATIONS, density_estimator="nsf",
                hidden_features=128, num_transforms=5, training_batch_size=256,
                max_num_epochs=300, seed=seed, embedding_net=emb, use_raw_obs=True,
                scenario_specific_warm_start=True,
            )
            wall = time.time() - t0
            with open(cache_path, "wb") as f:
                pickle.dump({"posterior": post, "meta": meta, "wall_time_s": wall}, f)
            posts[seed] = post
            mark(f"done_training_{tag}")
            print(f"{tag}: trained in {wall:.0f}s", flush=True)
        except Exception as e:
            mark(f"ERROR_{tag}")
            (RESULTS / f"22_error_cnn_{tag}.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
            print(f"{tag}: ERROR -- {e} -- continuing to next seed", flush=True)
            continue

    mark("training_all_done")
    print("All CNN training done. Computing Corr(alpha,beta) across scenarios...", flush=True)

    rows = []
    for sc_name in GRID_SCENARIOS:
        mark(f"corr_{sc_name}")
        x_raw = obs_x_raw(sc_name, seed=999)
        corrs = []
        for seed, post in posts.items():
            c, _ = corr_ab(post, x_raw, n_draw=5000, seed=500_000 + seed)
            corrs.append(c)
        if corrs:
            rows.append({
                "scenario": sc_name, "n_seeds": len(corrs),
                "corr_cnn_raw_mean": float(np.mean(corrs)),
                "corr_cnn_raw_std": float(np.std(corrs)),
                "corr_cnn_raw_min": float(np.min(corrs)),
                "corr_cnn_raw_max": float(np.max(corrs)),
                "corr_cnn_raw_all_seeds": corrs,
            })
            print(f"{sc_name:24s} CNN-raw Corr mean={np.mean(corrs):+.3f} "
                  f"std={np.std(corrs):.3f} (n_seeds={len(corrs)})", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "22_theoryF_cnn_recheck.csv", index=False)
    mark("finished")
    print("nb22 CNN Theory-F recheck finished.", flush=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
