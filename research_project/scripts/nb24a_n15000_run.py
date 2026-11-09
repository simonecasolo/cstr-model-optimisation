import sys, pathlib, pickle, time
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, torch
from scipy import stats as scipy_stats
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from cstr_sbi.recycle.physics import PARAM_NAMES
from cstr_sbi.recycle.priors import PRIOR_LOW_5D, PRIOR_HIGH_5D
from cstr_sbi.recycle.summaries import N_SUMMARIES_SC
from sbi_pipeline import _structure_config, _simulate_summary

print("=== START n_train=15000 (z_score_x='none') ===", flush=True)

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

NOMINAL_THETA_NP = np.array([1.0, 1.0, 1.0, 1.0, 0.90], dtype=np.float32)
LOG_SIGMA = np.array([0.16, 0.16, 0.10, 0.16, 0.055], dtype=np.float32)
prior_logn = TruncatedLogNormalPrior(NOMINAL_THETA_NP, LOG_SIGMA, PRIOR_LOW_5D, PRIOR_HIGH_5D)
print("prior ready", flush=True)

DATA = pathlib.Path('data'); RESULTS = pathlib.Path('results'); SBI_LOGS = pathlib.Path('sbi-logs')

STRUCTURE = 'S-C'
NOISE_PCT = 0.003
ARCH = 'zuko_nsf'
HIDDEN = 60
TRANSFORMS = 3
MAX_EPOCHS = 120
STOP_AFTER_EPOCHS = 12
N_TRAIN = 15000
SEEDS = [0, 1, 2]
N_POST = 120

SC_BANK_PATH = DATA / f'wu2003_sbi_train_sc_lognormal_true_n{N_TRAIN}.npz'

def generate_lognormal_bank_sc(out_path, n_rows, seed):
    if out_path.exists():
        bank = np.load(out_path)
        print(f'loading existing bank: {out_path}', flush=True)
        return np.asarray(bank['thetas'], dtype=np.float32), np.asarray(bank['summaries'], dtype=np.float32)
    ctrl, y0 = _structure_config(STRUCTURE)
    rng = np.random.default_rng(seed)
    theta_rows, summary_rows = [], []
    t0 = time.time()
    attempts = 0
    while len(theta_rows) < n_rows:
        attempts += 1
        th = prior_logn.sample((1,)).numpy()[0]
        s = _simulate_summary(th, ctrl, y0, STRUCTURE, NOISE_PCT, rng, scenario_specific_warm_start=True)
        if s is None or not np.isfinite(s).all():
            continue
        theta_rows.append(th)
        summary_rows.append(s)
        if len(theta_rows) % 250 == 0:
            elapsed = time.time() - t0
            rate = elapsed / len(theta_rows)
            eta = rate * (n_rows - len(theta_rows))
            print(f'[bank] {len(theta_rows)}/{n_rows}, attempts={attempts}, '
                  f'elapsed={elapsed:.0f}s, ~{rate:.2f}s/draw, ETA={eta/60:.1f}min', flush=True)
            np.savez(out_path.with_suffix('.partial.npz'),
                     thetas=np.asarray(theta_rows, dtype=np.float32),
                     summaries=np.asarray(summary_rows, dtype=np.float32), attempts=attempts)
    theta_arr = np.asarray(theta_rows, dtype=np.float32)
    summary_arr = np.asarray(summary_rows, dtype=np.float32)
    np.savez(out_path, thetas=theta_arr, summaries=summary_arr, attempts=attempts, seed=seed,
             structure=STRUCTURE, prior='truncated_lognormal_centered_nominal',
             scenario_specific_warm_start=True)
    print(f'saved bank: {out_path}, theta={theta_arr.shape}, summaries={summary_arr.shape}, '
          f'wall_s={time.time()-t0:.0f}', flush=True)
    return theta_arr, summary_arr

print("generating/loading bank (n=15000, this is the long step)...", flush=True)
theta_train, summary_train_sc = generate_lognormal_bank_sc(SC_BANK_PATH, N_TRAIN, seed=20260901)
print(f'theta_train: {theta_train.shape}, summary_train_sc: {summary_train_sc.shape}', flush=True)
assert summary_train_sc.shape[1] == N_SUMMARIES_SC

print("loading SBC sets (reused from n=5000 run, same fixed validation sets)...", flush=True)
sbc120 = np.load(RESULTS / '24a_sysII_sc_lognormal_sbc_set_n120.npz')
theta_sbc120, summary_sbc120 = sbc120['thetas'], sbc120['summaries']
sbc400 = np.load(RESULTS / '24a_sysII_sc_lognormal_sbc_set_n400.npz')
theta_sbc400, summary_sbc400 = sbc400['thetas'], sbc400['summaries']
print(f"SBC120: {theta_sbc120.shape}, SBC400: {theta_sbc400.shape}", flush=True)

def train_posterior(seed, z_score_x='independent'):
    torch.manual_seed(seed)
    kwargs = dict(model=ARCH, hidden_features=HIDDEN, num_transforms=TRANSFORMS)
    if z_score_x != 'independent':
        kwargs['z_score_x'] = z_score_x
    inference = SNPE(prior=prior_logn, density_estimator=posterior_nn(**kwargs))
    inference.append_simulations(torch.tensor(theta_train, dtype=torch.float32), torch.tensor(summary_train_sc, dtype=torch.float32))
    t0 = time.time()
    trained = inference.train(max_num_epochs=MAX_EPOCHS, validation_fraction=0.1, stop_after_epochs=STOP_AFTER_EPOCHS, show_train_summary=False)
    return inference.build_posterior(trained), time.time() - t0

def run_sbc(posterior, theta_set, summary_set, seed, n_post=N_POST):
    torch.manual_seed(seed)
    ranks = {name: [] for name in PARAM_NAMES}
    for th, x in zip(theta_set, summary_set):
        with torch.no_grad():
            samp = posterior.sample((n_post,), x=torch.tensor(x, dtype=torch.float32),
                                     show_progress_bars=False, reject_outside_prior=False).detach().numpy()
        for k, name in enumerate(PARAM_NAMES):
            ranks[name].append(int(np.sum(samp[:, k] < th[k])))
    out = {}
    for name in PARAM_NAMES:
        r = np.array(ranks[name], dtype=float)
        ks = scipy_stats.ks_1samp(r / n_post, scipy_stats.uniform.cdf)
        out[name] = {'ks_pvalue': float(ks.pvalue), 'mean_rank_frac': float(r.mean() / n_post)}
    out['min_ks_pvalue'] = min(out[name]['ks_pvalue'] for name in PARAM_NAMES)
    return out

print("\n=== Training z_score_x='none' posteriors at n_train=15000, 3 seeds ===", flush=True)
posteriors = {}
for seed in SEEDS:
    p_path = SBI_LOGS / f'wu2003_posterior_sc_lognormal_true_n{N_TRAIN}_zscorenone_seed{seed}.pkl'
    if p_path.exists():
        with open(p_path, 'rb') as f:
            posteriors[seed] = pickle.load(f)['posterior']
        print(f"seed {seed}: loaded existing", flush=True)
    else:
        post, wall = train_posterior(seed, z_score_x='none')
        with open(p_path, 'wb') as f:
            pickle.dump({'posterior': post, 'n_train': N_TRAIN, 'torch_seed': seed, 'z_score_x': 'none'}, f)
        posteriors[seed] = post
        print(f"seed {seed}: trained in {wall:.1f}s, saved", flush=True)

print("\n=== SBC at N=120 (reduced) ===", flush=True)
rows120 = []
for seed in SEEDS:
    sbc = run_sbc(posteriors[seed], theta_sbc120, summary_sbc120, seed=470000 + seed)
    rows120.append({'seed': seed, **{f'{k}_ks': v['ks_pvalue'] for k, v in sbc.items() if k != 'min_ks_pvalue'}, 'min_ks_pvalue': sbc['min_ks_pvalue']})
    print(f"seed {seed}: min_ks_pvalue={sbc['min_ks_pvalue']:.4e}", flush=True)
df120 = pd.DataFrame(rows120)
df120.to_csv(RESULTS / f'24a_sysII_sc_zscorenone_reduced_sbc_n120_ntrain{N_TRAIN}.csv', index=False)
print(df120.to_string(index=False), flush=True)

print("\n=== SBC at N=400 (confirmation) ===", flush=True)
rows400 = []
for seed in SEEDS:
    sbc = run_sbc(posteriors[seed], theta_sbc400, summary_sbc400, seed=480000 + seed)
    rows400.append({'seed': seed, **{f'{k}_ks': v['ks_pvalue'] for k, v in sbc.items() if k != 'min_ks_pvalue'}, 'min_ks_pvalue': sbc['min_ks_pvalue']})
    print(f"seed {seed}: min_ks_pvalue={sbc['min_ks_pvalue']:.4e}", flush=True)
df400 = pd.DataFrame(rows400)
df400.to_csv(RESULTS / f'24a_sysII_sc_zscorenone_confirm_sbc_n400_ntrain{N_TRAIN}.csv', index=False)
print(df400.to_string(index=False), flush=True)

n_pass_120 = int((df120['min_ks_pvalue'] > 0.05).sum())
n_pass_400 = int((df400['min_ks_pvalue'] > 0.05).sum())
print(f"\n=== FINAL: {n_pass_120}/3 pass at N=120, {n_pass_400}/3 pass at N=400 (n_train={N_TRAIN}) ===", flush=True)
print("=== DONE ===", flush=True)
