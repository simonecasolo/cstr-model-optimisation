"""NPE training, prior-truncated sampling, equal-weight ensembles and the evaluation suite for System III.
Same recipe as System I (scripts/ws_eval.py): NSF 128 hidden / 5 transforms, z-scored x, batch 256,
<= 200 epochs. Requires torch + sbi (the project .venv); everything else in the package is numpy."""
from __future__ import annotations

import contextlib
import io
import time

import numpy as np

from . import metrics as wm
from .simulator import ALPHA_LO, ALPHA_HI, BETA_LO, BETA_HI

LO = np.array([ALPHA_LO, BETA_LO]); HI = np.array([ALPHA_HI, BETA_HI])
NSF_KW = dict(hidden_features=128, num_transforms=5)
TRAIN_KW = dict(training_batch_size=256, max_num_epochs=200)


def prior():
    import torch
    from sbi.utils import BoxUniform
    return BoxUniform(low=torch.tensor(LO, dtype=torch.float32), high=torch.tensor(HI, dtype=torch.float32))


def train_npe(theta, x, seed, *, max_num_epochs=None):
    import torch
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    ok = np.all(np.isfinite(x), axis=1)
    torch.manual_seed(seed); np.random.seed(seed)
    build = posterior_nn(model="nsf", z_score_x="independent", **NSF_KW)
    inf = NPE(prior=prior(), density_estimator=build, show_progress_bars=False)
    inf.append_simulations(torch.as_tensor(theta[ok], dtype=torch.float32), torch.as_tensor(x[ok], dtype=torch.float32))
    kw = dict(TRAIN_KW)
    if max_num_epochs is not None:
        kw["max_num_epochs"] = max_num_epochs
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        de = inf.train(show_train_summary=False, **kw)
    post = inf.build_posterior(de)
    summ = getattr(inf, "_summary", {})
    last = lambda v: None if v is None else float(np.asarray(v).ravel()[-1])
    return post, {"n_train": int(ok.sum()), "n_invalid_dropped": int((~ok).sum()), "seed": seed,
                  "train_s": time.time() - t0, "epochs": last(summ.get("epochs_trained")),
                  "best_val_loss": last(summ.get("best_validation_loss")), **NSF_KW, **kw}


def _draw_raw(post, x, n):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = post.sample_batched((n,), x=x, show_progress_bars=False, reject_outside_prior=False)
    return np.transpose(s.detach().cpu().numpy(), (1, 0, 2))


def sample_within_prior(post, X, n, *, chunk=100, max_rounds=8, rng_seed=0):
    """(samples (B, n, 2), leak (B,)): draws restricted to the prior box; leak = raw mass outside it."""
    import torch
    torch.manual_seed(rng_seed); rng = np.random.default_rng(rng_seed)
    B = len(X); out = np.empty((B, n, 2), np.float32); leak = np.empty(B)
    for i0 in range(0, B, chunk):
        xb = torch.as_tensor(X[i0:i0 + chunk], dtype=torch.float32); b = len(xb)
        pools = [[] for _ in range(b)]; n_raw = np.zeros(b); n_ok = np.zeros(b); need = np.arange(b)
        for _ in range(max_rounds):
            if len(need) == 0:
                break
            raw = _draw_raw(post, xb[need], n)
            inside = np.all((raw >= LO) & (raw <= HI), axis=2)
            for j, k in enumerate(need):
                pools[k].append(raw[j][inside[j]]); n_raw[k] += n; n_ok[k] += inside[j].sum()
            need = np.array([k for k in range(b) if n_ok[k] < n], dtype=int)
        for k in range(b):
            v = np.concatenate(pools[k]) if pools[k] else np.empty((0, 2))
            out[i0 + k] = (v[:n] if len(v) >= n else v[rng.integers(0, len(v), n)] if len(v)
                           else np.clip(_draw_raw(post, xb[k:k + 1], n)[0], LO, HI))
            leak[i0 + k] = 1.0 - n_ok[k] / max(n_raw[k], 1)
    return out, leak


class EnsemblePosterior:
    """Equal-weight mixture of independently trained posteriors."""

    def __init__(self, members):
        self.members = list(members)

    def sample_batched(self, sample_shape, x, **kw):
        import torch
        n = int(np.prod(sample_shape)); K = len(self.members)
        counts = [n // K + (1 if i < n % K else 0) for i in range(K)]
        s = torch.cat([m.sample_batched((c,), x=x, **kw) for m, c in zip(self.members, counts) if c > 0], 0)
        return s[torch.randperm(len(s))]

    def sample(self, sample_shape=(1,), x=None, **kw):
        import torch
        n = int(np.prod(sample_shape)); K = len(self.members)
        counts = [n // K + (1 if i < n % K else 0) for i in range(K)]
        s = torch.cat([m.sample((c,), x=x, **kw) for m, c in zip(self.members, counts) if c > 0], 0)
        return s[torch.randperm(len(s))].reshape(*tuple(sample_shape), -1)


def evaluate(post, test, *, n_sbc_draws=1000, n_sc_draws=2000, seed=0, thr=0.90):
    s_sbc, leak_sbc = sample_within_prior(post, test["x_sbc"], n_sbc_draws, rng_seed=seed)
    row = wm.evaluate_sbc_set(s_sbc, test["theta_sbc"], LO, HI, seed=seed)
    row["leak_sbc_mean"] = float(leak_sbc.mean())
    s_sc, leak_sc = sample_within_prior(post, test["x_sc"], n_sc_draws, rng_seed=seed + 1)
    overall, sc_rows = wm.evaluate_scenario_set(s_sc, test["theta_sc"], test["sc_name"], LO, HI, thr=thr)
    row.update(overall); row["leak_sc_mean"] = float(leak_sc.mean())
    for r in sc_rows:
        k = r["scenario"].split("_")[0]
        for a, b in (("corr", "corr_ab_mean"), ("cov90b", "cov90_truth_b"), ("biasb", "bias_b"),
                     ("biasa", "bias_a"), ("sda", "sd_a"), ("sdb", "sd_b"), ("acc", "class_acc")):
            if b in r:
                row[f"{a}_{k}"] = r[b]
    return row, sc_rows


def three_way(samples, thr=0.90, p=0.95):
    """Three-way decision per window and fault: 'fault' if P(param < thr) >= p, 'healthy' if <= 1 - p,
    else 'undetermined'. Returns two arrays of {-1 healthy, 0 undetermined, +1 fault} for alpha, beta."""
    out = []
    for j in (0, 1):
        q = (samples[:, :, j] < thr).mean(1)
        out.append(np.where(q >= p, 1, np.where(q <= 1 - p, -1, 0)))
    return out
