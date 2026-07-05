"""Decisive check: does reject_outside_prior=False (vs True, the sbi default) itself cause
spurious SBC failures by letting wildly out-of-bounds posterior samples pollute the rank
statistic? Compare both on the SAME posterior/data at N_SBC=400."""
import pathlib, pickle, sys
import numpy as np
import torch
from scipy import stats as scipy_stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sbi_pipeline import _structure_config, _simulate_summary
from cstr_sbi.recycle.priors import box_uniform_5d
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.priors import PRIOR_LOW_5D, PRIOR_HIGH_5D

SBI_LOGS = pathlib.Path(__file__).resolve().parent.parent / "sbi-logs"

with open(SBI_LOGS / "wu2003_posterior_variant_A_sb_v2_seed6.pkl", "rb") as f:
    posterior = pickle.load(f)["posterior"]

structure, noise_pct = "S-B", 0.003
ctrl, y0 = _structure_config(structure)
prior = box_uniform_5d()
rng = np.random.default_rng(42)

N_SBC, N_POST = 400, 200
ranks_reject_true = {name: [] for name in PARAM_NAMES}
ranks_reject_false = {name: [] for name in PARAM_NAMES}
frac_oob = []

thetas = []
summaries = []
for i in range(N_SBC):
    th = prior.sample((1,)).numpy()[0]
    s = _simulate_summary(th, ctrl, y0, structure, noise_pct, rng)
    if s is None:
        continue
    thetas.append(th)
    summaries.append(s)
print(f"Valid sims: {len(thetas)}/{N_SBC}")

lo = np.asarray(PRIOR_LOW_5D)
hi = np.asarray(PRIOR_HIGH_5D)

for i, (th, s) in enumerate(zip(thetas, summaries)):
    x_obs = torch.tensor(s, dtype=torch.float32)
    with torch.no_grad():
        samp_f = posterior.sample((N_POST,), x=x_obs, reject_outside_prior=False, show_progress_bars=False).numpy()
        samp_t = posterior.sample((N_POST,), x=x_obs, reject_outside_prior=True, show_progress_bars=False).numpy()
    oob = np.mean(np.any((samp_f < lo) | (samp_f > hi), axis=1))
    frac_oob.append(oob)
    for k, name in enumerate(PARAM_NAMES):
        ranks_reject_false[name].append(int(np.sum(samp_f[:, k] < th[k])))
        ranks_reject_true[name].append(int(np.sum(samp_t[:, k] < th[k])))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(thetas)}  mean_frac_oob_so_far={np.mean(frac_oob):.3f}")

print(f"\nMean fraction of reject=False samples outside prior support: {np.mean(frac_oob):.4f}")
print(f"\n{'param':12s} {'KS p (reject=False)':22s} {'KS p (reject=True)':22s}")
for name in PARAM_NAMES:
    rf = np.array(ranks_reject_false[name]) / N_POST
    rt = np.array(ranks_reject_true[name]) / N_POST
    p_f = scipy_stats.ks_1samp(rf, scipy_stats.uniform.cdf).pvalue
    p_t = scipy_stats.ks_1samp(rt, scipy_stats.uniform.cdf).pvalue
    print(f"{name:12s} {p_f:<22.4f} {p_t:<22.4f}")
