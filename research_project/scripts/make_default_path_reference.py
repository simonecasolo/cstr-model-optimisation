"""Saves reference outputs of the DEFAULT simulate()/simulate_open_loop() path (summaries and raw) for a fixed set of
inputs, to logs/ref_default_path_before_S1.npz. Run once BEFORE the production-feature change (2026-11-09) so
tests/test_production_features.py can check that the default path stays bit-identical. Re-running overwrites nothing
unless --force."""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import ws_eval as we
import system_I_config as CFG

OUT = ROOT / "logs" / "ref_default_path_before_S1.npz"
CASES = {
    "closed_P2_N3": lambda th, **k: we.simulate(th, "P2", seed=123, **CFG.noise_kw("N3"), **k),
    "closed_P1b_N0": lambda th, **k: we.simulate(th, "P1b", seed=124, **CFG.noise_kw("N0"), **k),
    "closed_P2_N3_drift": lambda th, **k: we.simulate(th, "P2", seed=125, drift_T=2.0, **CFG.noise_kw("N3"), **k),
    "open_N3": lambda th, **k: we.simulate_open_loop(th, seed=126, **CFG.noise_kw("N3"), **k),
    "open_N0": lambda th, **k: we.simulate_open_loop(th, seed=127, **CFG.noise_kw("N0"), **k),
}

def make_theta():
    return we.prior_draws(48, 5).astype(np.float32)

if __name__ == "__main__":
    if OUT.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUT} exists; not overwriting (use --force)")
    th = make_theta(); out = {"theta": th}
    for name, fn in CASES.items():
        out[name + "__summ"] = np.asarray(fn(th)); out[name + "__raw"] = np.asarray(fn(th, raw=True))
        print(name, out[name + "__summ"].shape, out[name + "__raw"].shape, flush=True)
    np.savez(OUT, **out); print("saved", OUT)
