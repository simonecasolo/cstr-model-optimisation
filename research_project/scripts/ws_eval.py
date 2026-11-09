"""Simulation, training and sampling helpers shared by nb29 (warm-start study) and
nb31 (budget convergence). Imports physics / ws_protocols READ-ONLY; edits nothing shared.

Everything simulates through ``ws_protocols`` (vectorised jit+vmap, fixed controller law
via ``cstr_sbi.physics``), so training data and test data come from one simulator and the
only thing that differs between protocols is the initial state of the window.

Posterior quality metrics live in ``ws_metrics`` (pure numpy, unit-tested).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import jax.numpy as jnp
import torch

import ws_protocols as ws
import ws_metrics as wm
from cstr_sbi.priors import ALPHA_LOW, ALPHA_HIGH, BETA_LOW, BETA_HIGH, box_uniform_2d
from cstr_sbi.scenarios import SCENARIO_CONFIGS

LO = np.array([ALPHA_LOW, BETA_LOW], dtype=float)
HI = np.array([ALPHA_HIGH, BETA_HIGH], dtype=float)
RESULTS = ROOT / "results"

# Closed-loop evaluation scenarios (open-loop Sc0/Sc6 need an open-loop simulator and are
# out of scope for a closed-loop-trained posterior).
EVAL_SCENARIOS = ["Sc1_closed_healthy", "Sc2_closed_fouling", "Sc3_closed_decay",
                  "Sc4_closed_combined", "Sc5_closed_saturated", "Sc7_closed_drift"]

# NSF recipe of nb04 (do not change between protocols / budgets).
NSF_KW = dict(hidden_features=128, num_transforms=5)
TRAIN_KW = dict(training_batch_size=256, max_num_epochs=200)


def mark(tag: str, stage: str) -> None:
    """Progress sentinel (house rule: never tail tqdm logs; watch these files instead)."""
    (RESULTS / f"{tag}_progress_{stage}.txt").write_text(f"{time.time()}\n{time.ctime()}\n")


# ------------------------------------------------------------------ production features
def features(obs_raw, t, set_name: str = "S0_base29") -> np.ndarray:
    """Raw windows (n, n_t, 4) [C, T, Tc, Qc] -> summary-statistic matrix (n, d) for a named set.

    "S0_base29" : ``ws.summaries(obs, t)``, the 29 summaries, exactly as before (N0 production).
    other names : the union [29 summaries | plug-in estimators, loop dynamics, detrended stds] from
                  ``summary_variants`` indexed with ``summary_variants.SETS[set_name]``, built exactly
                  as ``nb03_summary_study._union`` builds and indexes it (so the production features
                  are bit-for-bit the ones notebook 03 section 10 evaluated). "S1_base_est" is the
                  N3 production set (29 + ln alpha_hat, ln beta_hat_jac, ln beta_hat_rx; d = 32).
    The extra features use the nominal inlet conditions explicitly (physics_constants())."""
    if set_name == "S0_base29":
        return ws.summaries(jnp.asarray(obs_raw), t)
    import summary_variants as sv
    if set_name not in sv.SETS:
        raise ValueError(f"unknown summary set {set_name!r}; choose from {list(sv.SETS)}")
    obs = np.asarray(obs_raw)
    t_np = np.asarray(t, dtype=np.float64)
    base = np.asarray(ws.summaries(jnp.asarray(obs), jnp.asarray(t_np)))
    union = np.concatenate([base, sv.extra_features(obs, t_np, sv.physics_constants())], axis=1).astype(np.float32)
    return union[:, sv.SETS[set_name]]


_features = features      # simulate()/simulate_open_loop() take a keyword named ``features``, which shadows the function


# ------------------------------------------------------------------ simulation
_SIM_CACHE: dict = {}


def _sim(protocol: str, **kw):
    key = (protocol, tuple(sorted(kw.items())))
    if key not in _SIM_CACHE:
        _SIM_CACHE[key] = ws.make_batch_simulator(protocol, **kw)
    return _SIM_CACHE[key]


def simulate(theta: np.ndarray, protocol: str, seed: int, *, drift_T=None, raw: bool = False,
             batch: int = 200, features: str = "S0_base29", **kw) -> np.ndarray:
    """theta (n,2) -> summaries (n, d), or raw observations (n, n_t, 4) if ``raw``.

    ``features``: summary set (see ``features()``); the default "S0_base29" gives the 29 summaries,
    bit-identical to the version before the production summary set was introduced. It is a named
    keyword, so it never reaches the protocol simulator through ``**kw``.

    ``drift_T``: None, a scalar, or an (n,) array of additive T-sensor offsets [K], applied
    after the sensor layer, as in the scenario generator (Sc7). Chunked in ``batch``; chunk i
    uses seed ``seed + i`` (so a run is reproducible and chunks are independent)."""
    sim = _sim(protocol, **kw)
    theta = np.asarray(theta, dtype=np.float32)
    n = len(theta)
    drift = np.zeros(n, np.float32) if drift_T is None else np.broadcast_to(
        np.asarray(drift_T, np.float32), (n,)).copy()
    out = []
    for i in range(0, n, batch):
        obs, t = sim(theta[i:i + batch], seed + i)
        obs = np.asarray(obs).copy()
        obs[:, :, 1] += drift[i:i + batch, None]
        out.append(obs if raw else _features(obs, t, features))
    return np.concatenate(out)


# ------------------------------------------------------------- open loop (Sc0, Sc6, nb07)
_OL_CACHE: dict = {}


def _open_loop_batch(n_burn: int, n_win: int, stride: int, noise_abs=None):
    """jit+vmap open-loop window simulator (3 states, Qc fixed at QC0), cached per shape and
    sensor model. ``noise_abs`` as in ws_protocols: per-channel absolute sigma, None entries
    (or None overall) keep the 0.5%-of-maximum convention."""
    key = (n_burn, n_win, stride, noise_abs)
    if key in _OL_CACHE:
        return _OL_CACHE[key]
    import jax
    from cstr_sbi.physics import K0_NOMINAL, UA_NOMINAL, NOMINAL_INLET, QC0, simulate_open_loop_to_steady_state
    from cstr_sbi.simulator import (_em_scan_open_loop, apply_sensor_layer, DEFAULT_PROCESS_SIGMA,
                                    DEFAULT_SENSOR_NOISE_PCT, DEFAULT_DT_INT)
    inlet_ol = jnp.asarray(NOMINAL_INLET)
    y_h = simulate_open_loop_to_steady_state(jnp.array([UA_NOMINAL, K0_NOMINAL]), inlet_ol, t_final=2000.0)

    def one(theta, drift, key):
        pk, sk = jax.random.split(key)
        params = jnp.array([theta[1] * UA_NOMINAL, theta[0] * K0_NOMINAL])
        ys = _em_scan_open_loop(y_h, pk, params, inlet_ol, DEFAULT_DT_INT, DEFAULT_PROCESS_SIGMA,
                                n_burn + n_win, stride)[n_burn // stride:]
        obs = jnp.stack([ys[:, 0], ys[:, 1], ys[:, 2], jnp.full((ys.shape[0],), QC0)], axis=1)
        if noise_abs is None:
            return apply_sensor_layer(obs, key=sk, noise_pct=DEFAULT_SENSOR_NOISE_PCT, drift_T=drift)
        na = np.array([np.nan if v is None else float(v) for v in noise_abs], dtype=np.float32)
        mx = jnp.maximum(jnp.max(jnp.abs(obs), axis=0, keepdims=True), 1e-12)
        sig = jnp.where(jnp.isnan(na)[None, :], DEFAULT_SENSOR_NOISE_PCT * mx, jnp.asarray(np.nan_to_num(na))[None, :])
        out = obs + jax.random.normal(sk, obs.shape) * sig
        return out.at[:, 1].add(drift)

    f = jax.jit(jax.vmap(one))
    _OL_CACHE[key] = (f, np.asarray(y_h))
    return _OL_CACHE[key]


def simulate_open_loop(theta: np.ndarray, seed: int, *, burn_min: float = 300.0, t_window: float = 60.0,
                       drift_T=None, raw: bool = False, batch: int = 200, noise_abs=None,
                       features: str = "S0_base29") -> np.ndarray:
    """Open-loop counterpart of ``simulate`` (fixed Qc = QC0, no controller).

    Protocol: every window starts from the healthy open-loop steady state and runs a noisy
    burn-in of ``burn_min`` (default 300 min, >= 10 open-loop time constants) at the sampled
    theta before the window is recorded, so it starts from theta's own stationary
    distribution -- the open-loop analogue of P1b. Used for BOTH the Sc0/Sc6 observations
    (nb02) and the open-loop posterior's training data (nb07), so they cannot drift apart."""
    import jax
    from cstr_sbi.simulator import DEFAULT_DT_INT, DEFAULT_DT_OUT
    n_burn = int(round(burn_min / DEFAULT_DT_INT))
    n_win = int(round(t_window / DEFAULT_DT_INT))
    stride = int(round(DEFAULT_DT_OUT / DEFAULT_DT_INT))
    f, _ = _open_loop_batch(n_burn, n_win, stride, None if noise_abs is None else tuple(noise_abs))
    theta = np.asarray(theta, dtype=np.float32)
    n = len(theta)
    drift = np.zeros(n, np.float32) if drift_T is None else np.broadcast_to(
        np.asarray(drift_T, np.float32), (n,)).copy()
    t = jnp.arange(1, n_win // stride + 1) * DEFAULT_DT_OUT
    out = []
    for i in range(0, n, batch):
        keys = jax.random.split(jax.random.PRNGKey(seed + i), len(theta[i:i + batch]))
        obs = f(jnp.asarray(theta[i:i + batch]), jnp.asarray(drift[i:i + batch]), keys)
        out.append(np.asarray(obs) if raw else _features(obs, t, features))
    return np.concatenate(out)


def prior_draws(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(LO, HI, size=(n, 2)).astype(np.float32)


def scenario_thetas(reps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta (6*reps,2), scenario names, drift_T) for the closed-loop evaluation grid."""
    th, names, drift = [], [], []
    for s in EVAL_SCENARIOS:
        sc = SCENARIO_CONFIGS[s]
        th += [[sc.alpha, sc.beta]] * reps
        names += [s] * reps
        drift += [sc.drift_T] * reps
    return np.asarray(th, np.float32), np.asarray(names), np.asarray(drift, np.float32)


def make_test_set(protocol: str, seed: int, n_sbc: int, sc_reps: int, **kw) -> dict:
    """Held-out test set under ``protocol``: an SBC set drawn from the prior plus the
    closed-loop scenario grid. Same theta draws for every protocol (seeded), so test
    columns differ only by the initial-state protocol."""
    th_sbc = prior_draws(n_sbc, seed)
    x_sbc = simulate(th_sbc, protocol, seed + 1_000_000, **kw)
    th_sc, names, drift = scenario_thetas(sc_reps)
    x_sc = simulate(th_sc, protocol, seed + 2_000_000, drift_T=drift, **kw)
    return dict(theta_sbc=th_sbc, x_sbc=x_sbc, theta_sc=th_sc, sc_name=names, x_sc=x_sc)


def save_npz(path: Path, d: dict) -> None:
    np.savez_compressed(path, **d)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


# -------------------------------------------------------------------- training
def train_npe(theta: np.ndarray, x: np.ndarray, seed: int, *, max_num_epochs=None) -> tuple:
    """Train NPE (NSF 128/5, z-scored x) on pre-simulated (theta, x). Returns (posterior, meta).

    The seed sets network initialisation and the train/validation split."""
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    ok = np.all(np.isfinite(x), axis=1)
    theta_t = torch.as_tensor(theta[ok], dtype=torch.float32)
    x_t = torch.as_tensor(x[ok], dtype=torch.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    build = posterior_nn(model="nsf", z_score_x="independent", **NSF_KW)
    inf = NPE(prior=box_uniform_2d(), density_estimator=build, show_progress_bars=False)
    inf.append_simulations(theta_t, x_t)
    kw = dict(TRAIN_KW)
    if max_num_epochs is not None:
        kw["max_num_epochs"] = max_num_epochs
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        de = inf.train(show_train_summary=False, **kw)
    post = inf.build_posterior(de)
    summ = getattr(inf, "_summary", {})
    meta = {"n_train": int(ok.sum()), "n_invalid_dropped": int((~ok).sum()), "seed": seed,
            "train_s": time.time() - t0,
            "epochs": _last(summ.get("epochs_trained")),
            "best_val_loss": _last(summ.get("best_validation_loss")),
            **NSF_KW, **kw}
    return post, meta


def _last(v):
    if v is None:
        return None
    try:
        return float(v[-1])
    except (TypeError, IndexError):
        return float(v)


# -------------------------------------------------------------------- sampling
def _draw_raw(post, x: torch.Tensor, n: int) -> np.ndarray:
    """(B,n,2) unrestricted draws from the flow for a batch of observations."""
    import warnings
    with warnings.catch_warnings():          # out-of-prior draws are rejected and counted below
        warnings.simplefilter("ignore")
        s = post.sample_batched((n,), x=x, show_progress_bars=False, reject_outside_prior=False)
    s = s.detach().cpu().numpy()          # (n, B, 2)
    return np.transpose(s, (1, 0, 2))


def sample_within_prior(post, X: np.ndarray, n: int, *, chunk: int = 100, max_rounds: int = 8,
                        rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Posterior draws restricted to the prior box, for each row of X.

    Rejection outside the box (equivalent to the truncated posterior sbi's own rejection
    sampler targets), done in vectorised rounds. Returns (samples (B,n,2), leak (B,)) where
    ``leak`` is the fraction of raw flow mass outside the prior, a diagnostic worth reporting.
    If an observation still lacks n valid draws after ``max_rounds`` (extreme leakage), the
    valid draws are resampled with replacement and its leak is reported as is (flagged >0.95)."""
    torch.manual_seed(rng_seed)
    rng = np.random.default_rng(rng_seed)
    B = len(X)
    out = np.empty((B, n, 2), np.float32)
    leak = np.empty(B, float)
    for i0 in range(0, B, chunk):
        xb = torch.as_tensor(X[i0:i0 + chunk], dtype=torch.float32)
        b = len(xb)
        pools = [[] for _ in range(b)]
        n_raw = np.zeros(b); n_ok = np.zeros(b)
        need = np.arange(b)
        for _ in range(max_rounds):
            if len(need) == 0:
                break
            raw = _draw_raw(post, xb[need], n)
            inside = np.all((raw >= LO) & (raw <= HI), axis=2)
            for j, k in enumerate(need):
                pools[k].append(raw[j][inside[j]])
                n_raw[k] += n; n_ok[k] += inside[j].sum()
            need = np.array([k for k in range(b) if n_ok[k] < n], dtype=int)
        for k in range(b):
            v = np.concatenate(pools[k]) if pools[k] else np.empty((0, 2))
            if len(v) >= n:
                out[i0 + k] = v[:n]
            elif len(v) > 0:
                out[i0 + k] = v[rng.integers(0, len(v), size=n)]
            else:                           # no valid draw at all: fall back to raw, clipped
                out[i0 + k] = np.clip(_draw_raw(post, xb[k:k + 1], n)[0], LO, HI)
            leak[i0 + k] = 1.0 - n_ok[k] / max(n_raw[k], 1)
    return out, leak


class EnsemblePosterior:
    """Equal-weight mixture of independently trained posteriors (Hermans et al. 2022).

    Duck-types the parts of sbi's DirectPosterior that downstream notebooks use:
    ``sample(shape, x=..., **kw)``, ``sample_batched`` and ``log_prob``."""

    def __init__(self, members):
        self.members = list(members)

    def sample(self, sample_shape=(1,), x=None, **kw):
        n = int(np.prod(sample_shape))
        K = len(self.members)
        counts = [n // K + (1 if i < n % K else 0) for i in range(K)]
        parts = [m.sample((c,), x=x, **kw) for m, c in zip(self.members, counts) if c > 0]
        s = torch.cat(parts, dim=0)
        return s[torch.randperm(len(s))].reshape(*tuple(sample_shape), -1)

    def sample_batched(self, sample_shape, x, **kw):
        n = int(np.prod(sample_shape))
        K = len(self.members)
        counts = [n // K + (1 if i < n % K else 0) for i in range(K)]
        parts = [m.sample_batched((c,), x=x, **kw) for m, c in zip(self.members, counts) if c > 0]
        s = torch.cat(parts, dim=0)                       # (n, B, 2)
        return s[torch.randperm(len(s))]

    def log_prob(self, theta, x=None, **kw):
        lp = torch.stack([m.log_prob(theta, x=x, **kw) for m in self.members])
        return torch.logsumexp(lp, dim=0) - np.log(len(self.members))


# ------------------------------------------------------------------ evaluation
def evaluate(post, test: dict, *, n_sbc_draws=1000, n_sc_draws=2000, seed=0, thr=0.90) -> tuple[dict, list]:
    """Full metric suite for one posterior on one test set. Returns (summary_row, scenario_rows)."""
    s_sbc, leak_sbc = sample_within_prior(post, test["x_sbc"], n_sbc_draws, rng_seed=seed)
    row = wm.evaluate_sbc_set(s_sbc, test["theta_sbc"], LO, HI, seed=seed)
    row["leak_sbc_mean"] = float(leak_sbc.mean())
    row["leak_sbc_frac_obs_gt50pct"] = float((leak_sbc > 0.5).mean())
    s_sc, leak_sc = sample_within_prior(post, test["x_sc"], n_sc_draws, rng_seed=seed + 1)
    overall, sc_rows = wm.evaluate_scenario_set(s_sc, test["theta_sc"], test["sc_name"], LO, HI, thr=thr)
    row.update(overall)
    row["leak_sc_mean"] = float(leak_sc.mean())
    # headline per-scenario numbers flattened into the summary row for easy tables
    for r in sc_rows:
        short = r["scenario"].split("_")[0]
        row[f"corr_{short}"] = r["corr_ab_mean"]
        row[f"cov90b_{short}"] = r["cov90_truth_b"]
        row[f"biasb_{short}"] = r["bias_b"]
        row[f"biasa_{short}"] = r["bias_a"]
        row[f"sda_{short}"] = r["sd_a"]
        row[f"sdb_{short}"] = r["sd_b"]
    return row, sc_rows


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))
