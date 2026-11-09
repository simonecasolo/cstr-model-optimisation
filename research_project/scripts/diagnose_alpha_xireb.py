import sys, pickle
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np, torch
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.priors import PRIOR_LOW_5D, PRIOR_HIGH_5D

class TruncatedLogNormalPrior(torch.distributions.Distribution):
    arg_constraints = {}
    has_rsample = False
    def __init__(self, median, sigma, low, high, validate_args=False):
        self.median = torch.as_tensor(median, dtype=torch.float32)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32)
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.mu = torch.log(self.median)
        self.base = torch.distributions.Normal(self.mu, self.sigma)
        lo_z = (torch.log(self.low) - self.mu) / self.sigma
        hi_z = (torch.log(self.high) - self.mu) / self.sigma
        standard = torch.distributions.Normal(torch.zeros_like(self.mu), torch.ones_like(self.mu))
        self.log_norm = torch.log(standard.cdf(hi_z) - standard.cdf(lo_z)).sum()
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([len(self.median)]), validate_args=validate_args)
    def sample(self, sample_shape=torch.Size()):
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        sample_shape = torch.Size(sample_shape)
        n = int(np.prod(sample_shape)) if len(sample_shape) else 1
        accepted = []
        batch = max(1024, 2 * n)
        while sum(a.shape[0] for a in accepted) < n:
            theta = torch.exp(self.base.sample((batch,)))
            mask = ((theta >= self.low) & (theta <= self.high)).all(dim=1)
            if mask.any():
                accepted.append(theta[mask])
        out = torch.cat(accepted, dim=0)[:n]
        return out.reshape(sample_shape + self.event_shape)
    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=torch.float32)
        inside = ((value >= self.low) & (value <= self.high)).all(dim=-1)
        safe_value = torch.clamp(value, min=torch.finfo(torch.float32).tiny)
        logpdf = self.base.log_prob(torch.log(safe_value)).sum(dim=-1) - torch.log(safe_value).sum(dim=-1) - self.log_norm
        return torch.where(inside, logpdf, torch.full_like(logpdf, -torch.inf))

d = np.load('results/24a_sysII_sc_lognormal_sbc_set_n400.npz')
theta_sbc, summary_sbc = d['thetas'], d['summaries']
ALPHA_IDX, BETA_R_IDX, ETA_COL_IDX, XI_REB_IDX, Z_A0_IDX = range(5)

with open('sbi-logs/wu2003_posterior_sc_lognormal_true_n5000_zscorenone_seed0.pkl', 'rb') as f:
    post0 = pickle.load(f)['posterior']
with open('sbi-logs/wu2003_posterior_sc_lognormal_true_n5000_zscorenone_seed1.pkl', 'rb') as f:
    post1 = pickle.load(f)['posterior']

print("=== ALPHA diagnostic on seed 0 (failed: p=2.09e-5) ===")
torch.manual_seed(0)
for i in range(10):
    th_true = theta_sbc[i]
    x_obs = summary_sbc[i]
    with torch.no_grad():
        samp = post0.sample((300,), x=torch.tensor(x_obs, dtype=torch.float32),
                             show_progress_bars=False, reject_outside_prior=False).detach().numpy()
    a_true = th_true[ALPHA_IDX]
    a_samp = samp[:, ALPHA_IDX]
    lo, hi = PRIOR_LOW_5D[ALPHA_IDX], PRIOR_HIGH_5D[ALPHA_IDX]
    frac_out = np.mean((a_samp < lo) | (a_samp > hi))
    print(f"draw {i}: true={a_true:.4f}  mean={a_samp.mean():.4f}  std={a_samp.std():.4f}  "
          f"frac(samp<true)={np.mean(a_samp<a_true):.3f}  frac_out_of_box={frac_out:.3f}  "
          f"min={a_samp.min():.3f} max={a_samp.max():.3f}")

print("\n=== XI_REB diagnostic on seed 1 (failed: p=8e-6) ===")
torch.manual_seed(0)
for i in range(10):
    th_true = theta_sbc[i]
    x_obs = summary_sbc[i]
    with torch.no_grad():
        samp = post1.sample((300,), x=torch.tensor(x_obs, dtype=torch.float32),
                             show_progress_bars=False, reject_outside_prior=False).detach().numpy()
    x_true = th_true[XI_REB_IDX]
    x_samp = samp[:, XI_REB_IDX]
    lo, hi = PRIOR_LOW_5D[XI_REB_IDX], PRIOR_HIGH_5D[XI_REB_IDX]
    frac_out = np.mean((x_samp < lo) | (x_samp > hi))
    print(f"draw {i}: true={x_true:.4f}  mean={x_samp.mean():.4f}  std={x_samp.std():.4f}  "
          f"frac(samp<true)={np.mean(x_samp<x_true):.3f}  frac_out_of_box={frac_out:.3f}  "
          f"min={x_samp.min():.3f} max={x_samp.max():.3f}")

print("\n=== Full 5-param leakage check, seed 0, draw 0 ===")
x_obs = summary_sbc[0]
with torch.no_grad():
    samp = post0.sample((500,), x=torch.tensor(x_obs, dtype=torch.float32),
                         show_progress_bars=False, reject_outside_prior=False).detach().numpy()
for k, name in enumerate(PARAM_NAMES):
    lo, hi = PRIOR_LOW_5D[k], PRIOR_HIGH_5D[k]
    frac_out = np.mean((samp[:,k] < lo) | (samp[:,k] > hi))
    print(f"  {name:10s}: true={theta_sbc[0][k]:.4f}  mean={samp[:,k].mean():.4f}  std={samp[:,k].std():.4f}  "
          f"box=[{lo:.2f},{hi:.2f}]  frac_out_of_box={frac_out:.3f}")
