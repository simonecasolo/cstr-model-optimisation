"""Notebook 09 (System I, Phase 3 step f): the drift-aware production posterior, theta = [alpha, beta, dT].

The production posterior of notebook 36 knows nothing about a sensor offset; under Sc7 (+2 K on the temperature sensor) it is
confidently wrong (beta bias -0.395, coverage 0.002). Here the temperature-sensor offset dT is a third parameter of the posterior.

Design (rules fixed in docs/Manuscript/system_i_N3_runbook.md section 2f, row f):
  prior alpha, beta ~ U(0.4, 1.2) (as production); dT ~ U(-3, 3) K (the nb09 prior; Sc7's +2 K inside);
  simulator = the nb36 production model: protocol P2, N3 sensors, true inlet ~ nb35 training distribution, MEASURED Ti, Tci
  (eps = 0.1 K offset + window noise), dT added to the measured T channel after the sensor layer (as the Sc7 generation);
  features S1m_measured_inlet computed from the drifted raw window; budget 20,000 simulations x 5 seeds, ensemble (as nb36).
The delta-Ci term of the old nb09 is dropped (user decision 4): Ci is the unmeasured +-1% nuisance of the simulator.

    python scripts/nb09_drift_aware.py bank | train      (cached; the notebook calls the same functions)
"""
from __future__ import annotations

import contextlib
import io
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("CSTR_SENSOR_MODEL", "N3")

import numpy as np
import torch

import nb35_measured_inlet as M
import nb36_production_measured_inlet as P36
import ws_eval as we
import ws_inlet as wi

R = ROOT / "results"
TAG = "09_N3"
OUT = R / "09_drift_aware_N3"
PROTOCOL, NOISE, EPS = M.PROTOCOL, M.NOISE, 0.1
VARIANT = P36.VARIANT
FEATURES = "S1m_measured_inlet"
BANK_N, CHUNK = 20_000, 5_000
SEEDS = (0, 1, 2, 3, 4)
N_SBC = 1_000
DT_LO, DT_HI = -3.0, 3.0
LO = np.array([0.4, 0.4, DT_LO]); HI = np.array([1.2, 1.2, DT_HI])
THETA_SEED, DT_SEED, INLET_SEED, SIM_SEED, OFFSET_SEED = 9_100_001, 9_100_002, 9_100_003, 9_100_004, 9_100_005
SBC_SEED = 9_200_000


def _t():
    return np.arange(1, 121, dtype=np.float32) * 0.5


def draw_theta(n, seed):
    """(n, 3) [alpha, beta, dT] from the 3-D prior."""
    th2 = we.prior_draws(n, seed)
    dt = np.random.default_rng(seed + 7).uniform(DT_LO, DT_HI, n)
    return np.column_stack([th2, dt]).astype(np.float32)


def simulate_drifted(theta3, inlet, seed, feat_seed):
    """Raw windows under the production model with the T-sensor offset dT, then S1m features (measured inlet)."""
    raw = np.array(wi.simulate(theta3[:, :2], inlet, PROTOCOL, seed, raw=True, **NOISE), np.float32)
    raw[:, :, 1] += theta3[:, 2][:, None]
    base29 = we.features(raw, _t(), "S0_base29")
    return P36.features(base29, inlet, feat_seed)


# ------------------------------------------------------------------ bank / test / training (cached)
def stage_bank():
    OUT.mkdir(parents=True, exist_ok=True)
    th = draw_theta(BANK_N, THETA_SEED)
    inl = M._inlets(BANK_N, INLET_SEED)
    for c in range(BANK_N // CHUNK):
        f = OUT / f"bank_chunk{c:02d}.npz"
        if f.exists():
            continue
        t0 = time.time(); sl = slice(c * CHUNK, (c + 1) * CHUNK)
        x = simulate_drifted(th[sl], inl[sl], SIM_SEED * 10_000 + c * CHUNK, OFFSET_SEED + c)
        we.save_npz(f, {"theta": th[sl], "inlet": inl[sl], "x": x})
        print(f"bank chunk {c}: {time.time() - t0:.0f}s", flush=True)


def load_bank():
    parts = [we.load_npz(OUT / f"bank_chunk{c:02d}.npz") for c in range(BANK_N // CHUNK)]
    return np.concatenate([p["theta"] for p in parts]), np.concatenate([p["x"] for p in parts])


def stage_sbc_set():
    f = OUT / "sbc_set.npz"
    if f.exists():
        return we.load_npz(f)
    th = draw_theta(N_SBC, SBC_SEED); inl = M._inlets(N_SBC, SBC_SEED + 1)
    x = simulate_drifted(th, inl, SBC_SEED + 2_000_000, SBC_SEED + 3)
    d = {"theta": th, "inlet": inl, "x": x}; we.save_npz(f, d); return d


def train_npe3(theta, x, seed):
    """as ws_eval.train_npe but with the 3-D box prior"""
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform
    ok = np.all(np.isfinite(x), axis=1)
    theta_t = torch.as_tensor(theta[ok], dtype=torch.float32); x_t = torch.as_tensor(x[ok], dtype=torch.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    build = posterior_nn(model="nsf", z_score_x="independent", **we.NSF_KW)
    prior = BoxUniform(low=torch.as_tensor(LO, dtype=torch.float32), high=torch.as_tensor(HI, dtype=torch.float32))
    inf = NPE(prior=prior, density_estimator=build, show_progress_bars=False)
    inf.append_simulations(theta_t, x_t)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        de = inf.train(show_train_summary=False, **dict(we.TRAIN_KW))
    post = inf.build_posterior(de)
    return post, {"n_train": int(ok.sum()), "n_invalid_dropped": int((~ok).sum()), "seed": seed, "train_s": time.time() - t0}


def post_path(s):
    return OUT / f"post_n{BANK_N}_s{s}.pkl"


def stage_train(threads: int = 4):
    torch.set_num_threads(threads)
    th, x = load_bank()
    for s in SEEDS:
        f = post_path(s)
        if f.exists():
            continue
        perm = np.random.default_rng(9_300 + s).permutation(len(th))
        post, meta = train_npe3(th[perm], x[perm], seed=s)
        meta.update({"features": FEATURES, "eps": EPS, "protocol": PROTOCOL, "budget": BANK_N})
        with open(f, "wb") as fh:
            pickle.dump({"posterior": post, "metadata": meta}, fh)
        print("train seed", s, round(meta["train_s"]), "s", flush=True)


def load_members():
    return [pickle.load(open(post_path(s), "rb"))["posterior"] for s in SEEDS]


def ensemble():
    return we.EnsemblePosterior(load_members())


# ------------------------------------------------------------------ sampling restricted to the prior box
def sample_box(post, X, n, *, chunk=100, seed=0):
    """(B, n, 3) draws inside the prior box and the fraction of raw draws outside (the leakage)."""
    torch.manual_seed(seed)
    B = len(X); out = np.empty((B, n, 3), np.float32); leak = np.empty(B)
    for i0 in range(0, B, chunk):
        xb = torch.as_tensor(X[i0:i0 + chunk], dtype=torch.float32)
        raw = we._draw_raw(post, xb, int(n * 1.3) + 20)            # (b, m, 3)
        inside = np.all((raw >= LO) & (raw <= HI), axis=2)
        for j in range(len(xb)):
            ok = raw[j][inside[j]]
            leak[i0 + j] = 1.0 - inside[j].mean()
            if len(ok) >= n:
                out[i0 + j] = ok[:n]
            else:                                                   # extreme leakage: resample with replacement, flagged by leak
                idx = np.random.default_rng(seed + i0 + j).integers(0, max(len(ok), 1), n)
                out[i0 + j] = ok[idx] if len(ok) else raw[j][:n]
    return out, leak


if __name__ == "__main__":
    st = sys.argv[1]
    {"bank": stage_bank, "train": stage_train}[st]()
