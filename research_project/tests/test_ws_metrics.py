"""Sanity tests for scripts/ws_metrics.py on an exactly-known Gaussian posterior."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ws_metrics as wm

LO, HI = np.array([0.4, 0.4]), np.array([1.2, 1.2])

def _gauss_problem(B=600, n=1000, rho_true=-0.8, rho_used=None, scale=1.0, seed=0):
    """theta ~ N(0.8, S) (clipped prior irrelevant: S small), x = theta + noise, posterior exact Gaussian."""
    rng = np.random.default_rng(seed)
    s = 0.05
    C = s**2 * np.array([[1, rho_true], [rho_true, 1]])
    post_mean = rng.uniform(0.6, 1.0, size=(B, 2))
    theta = post_mean + rng.multivariate_normal([0, 0], C, size=B)   # truth drawn from posterior
    r = rho_true if rho_used is None else rho_used
    Cu = (scale * s) ** 2 * np.array([[1, r], [r, 1]])
    samples = post_mean[:, None, :] + rng.multivariate_normal([0, 0], Cu, size=(B, n))
    return samples, theta

def test_correct_posterior_is_calibrated():
    smp, th = _gauss_problem()
    m = wm.evaluate_sbc_set(smp, th, LO, HI)
    assert m["ks_p_a"] > 0.01 and m["ks_p_b"] > 0.01
    assert abs(m["cov90_a"] - 0.9) < 0.04 and abs(m["cov90_b"] - 0.9) < 0.04
    assert m["tarp_ks_p"] > 0.001 and m["tarp_dev"] < 0.08   # max-dev noise ~1.36/sqrt(B)
    assert abs(m["corr_ab_mean"] + 0.8) < 0.02

def test_overconfident_posterior_fails_coverage():
    smp, th = _gauss_problem(scale=0.5)
    m = wm.evaluate_sbc_set(smp, th, LO, HI)
    assert m["cov90_a"] < 0.75 and m["frac_edge20_a"] > 0.55

def test_wrong_correlation_seen_by_tarp_not_marginals():
    smp, th = _gauss_problem(rho_true=-0.9, rho_used=+0.9)
    m = wm.evaluate_sbc_set(smp, th, LO, HI)
    assert abs(m["cov90_a"] - 0.9) < 0.04          # marginals still fine
    assert m["tarp_dev"] > 0.08 and m["tarp_ks_p"] < 0.01   # joint test catches the sign error

def test_classification_and_f1():
    th = np.array([[1.0, 1.0], [1.0, 0.7], [0.7, 1.0], [0.85, 0.85]])
    smp = th[:, None, :] + 0.001 * np.random.default_rng(0).standard_normal((4, 200, 2))
    overall, rows = wm.evaluate_scenario_set(smp, th, ["Sc1", "Sc2", "Sc3", "Sc4"], LO, HI)
    assert overall["macro_f1"] == 1.0 and len(rows) == 4
    assert list(wm.class_of_theta(th)) == ["healthy", "fouling_dominant", "decay_dominant", "combined"]


def test_interior_ks_and_scenario_sd_reported():
    smp, th = _gauss_problem()
    m = wm.evaluate_sbc_set(smp, th, LO, HI)
    assert m["n_interior"] > 50 and m["ks_p_a_interior"] > 0.001
    overall, rows = wm.evaluate_scenario_set(smp[:4], th[:4], ["S"] * 4, LO, HI)
    assert abs(rows[0]["sd_a"] - 0.05) < 0.01
