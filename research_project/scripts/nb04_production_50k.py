"""nb04 production posterior at n_sims=50,000 + nb22's convergence-metric suite.

Standalone (house rule for long runs). Trains the System I NSF posterior with the
nb04 recipe (NSF, 128 hidden, 5 transforms, batch 256, 200 epochs) under the matched
per-draw warm start (scenario_specific_warm_start=True, corrected closed-loop
root-find warm start), then evaluates it with nb22's OWN functions
(imported, not re-implemented): Corr(alpha,beta)@Sc4, SBC (KS p, rank-shape),
posterior-mean-vs-truth slope, 90% coverage, posterior std over the Sc1-5,7 grid.

Does NOT overwrite results/sbi_posterior_final.pkl -- promotion is a separate,
user-gated step. Usage: python scripts/nb04_production_50k.py [seed]
"""
import sys, json, time, datetime as dt, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
import build_nb_system_I_22 as nb22   # metric functions only; no side effects on import
from cstr_sbi.inference import train_sbi_posterior
from cstr_sbi.priors import box_uniform_2d

N_SIMS = 50_000
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
R = ROOT / "results"
tag = f"04_production_n{N_SIMS}_seed{seed}"

def mark(s):
    (R / f"{tag}_progress_{s}.txt").write_text(f"{time.time()}\n{dt.datetime.now().isoformat()}\n")

pkl = R / f"{tag}.pkl"
mark("start")
prior = box_uniform_2d()
t0 = time.time()
posterior, meta = train_sbi_posterior(
    prior, n_simulations=N_SIMS, density_estimator="nsf", hidden_features=128,
    num_transforms=5, training_batch_size=256, max_num_epochs=200, seed=seed,
    scenario_specific_warm_start=True, save_to=str(pkl))
train_s = meta.get("wall_time_s", time.time() - t0)
mark("trained")

sc4_x = nb22.obs_x_for_scenario("Sc4_closed_combined", seed=999)
corr, _ = nb22.corr_ab(posterior, sc4_x, seed=seed)
mark("corr")
sbc = nb22.run_sbc_for_posterior(posterior, seed_offset=100_000 + seed * 1000)  # same offsets as nb22
mark("sbc")
grid = nb22.grid_diagnostics(posterior)
mark("grid")
out = {"budget": N_SIMS, "seed": seed, "wall_time_s": train_s, "corr_ab_sc4": corr, **sbc, **grid,
       "n_epochs_trained": meta.get("n_epochs", None), "pkl": str(pkl)}
(R / f"{tag}_metrics.json").write_text(json.dumps(out, indent=2, default=float))
mark("finished")
print(json.dumps(out, indent=2, default=float))
