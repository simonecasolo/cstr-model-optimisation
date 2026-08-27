"""Shared 9-state augmented EKF core for the Wu 2003 recycle plant (System II).

Extracted from `notebooks/27_wu2003_sequential_tracking.ipynb` cell 22
(`run_ekf_single_window`) plus its supporting jax-jacobian helpers (cell 12: `rhs_fn`,
`obs_fn`, `_step_predict`, `_obs_and_jac`) — the cleanest of the two independent,
mutually-duplicated 9-state EKF implementations already in the repo (the other lives,
un-refactored, in `notebooks/26_wu2003_headline_banana_ekf_failure.ipynb`). `nb26`/`nb27`
are deliberately left untouched by this refactor (see `notebooks/34_ekf_tuning_sensitivity.ipynb`,
which is the third, new consumer that motivated extracting this module rather than
copy-pasting a third time).

State vector (9): [z_A, T_r, T_j, I_T, R_state, V_state, alpha, beta_r, eta_col].
Observed channels (3, `obs_idx = [0, 1, 7]` into the raw S-B trajectory): T_r, T_j,
F_R_norm. xi_reb and z_A0_eff are fixed at their healthy nominal values (1.0, 0.90) inside
`rhs_fn` -- this EKF's augmented state does not track them (matching both nb26 and nb27).
"""

import jax
import jax.numpy as jnp
import numpy as np

from cstr_sbi.recycle.physics import (
    NOMINAL_CTRL_SB, NOMINAL_INLET, recycle_rhs_explicit, column_qss, F0_NOM, F_R_NOM,
)
from cstr_sbi.recycle.simulator import nominal_warm_start

CTRL_J = jnp.array(NOMINAL_CTRL_SB, dtype=jnp.float32)
OBS_IDX = [0, 1, 7]
N_STATE = 9


def rhs_fn(y_aug, ctrl):
    z_A, T_r, T_j, I_T, R_st, V_st, alpha, beta_r, eta_col = y_aug
    xi_reb = 1.0
    z_A0_eff = 0.90
    theta = jnp.array([alpha, beta_r, eta_col, xi_reb, z_A0_eff], dtype=jnp.float32)
    y8 = jnp.array([z_A, T_r, T_j, I_T, R_st, V_st, 0.0, 0.0], dtype=jnp.float32)
    dy8 = recycle_rhs_explicit(0.0, y8, (theta, NOMINAL_INLET, ctrl))
    return jnp.array([dy8[0], dy8[1], dy8[2], dy8[3], dy8[4], dy8[5], 0.0, 0.0, 0.0])


def obs_fn(x_pred):
    z_A, T_r, T_j, I_T, R_st, V_st, alpha, beta_r, eta_col = x_pred
    _, _, d_frac = column_qss(z_A, eta_col)
    d_safe = jnp.clip(d_frac, 0.01, 0.98)
    FR = d_safe * F0_NOM / (1.0 - d_safe) / F_R_NOM
    return jnp.array([T_r, T_j, FR])


@jax.jit
def _step_predict(x, ctrl):
    f0 = rhs_fn(x, ctrl)
    F = jax.jacfwd(rhs_fn, argnums=0)(x, ctrl)
    return f0, F


@jax.jit
def _obs_and_jac(x_pred):
    y_pred = obs_fn(x_pred)
    H_full = jax.jacfwd(obs_fn)(x_pred)
    return y_pred, H_full


def run_ekf_single_window(raw_window, P_init, Q_diag, R_diag, ctrl_j=CTRL_J):
    """One 2h window, fresh cold-start (fault parameters initialised at 1.0) -- mirrors
    nb26's per-replicate protocol exactly, just with a swappable P/Q/R tuning. Identical to
    nb27 cell 22's function of the same name; extracted here so a third caller
    (`notebooks/34_ekf_tuning_sensitivity.ipynb`) does not need a third copy-paste."""
    y0_np = np.asarray(nominal_warm_start("S-B"))
    x = np.array([*y0_np[:6], 1.0, 1.0, 1.0])
    P = P_init.copy(); Q = Q_diag.copy(); R = R_diag.copy()
    for k in range(len(raw_window)):
        dt_h = 2.0 / 120
        x_j = jnp.array(x, dtype=jnp.float32)
        f0_j, F_j = _step_predict(x_j, ctrl_j)
        f0 = np.asarray(f0_j); F = np.asarray(F_j)
        x_pred = x + f0 * dt_h
        x_pred[0] = np.clip(x_pred[0], 1e-4, 0.999); x_pred[1] = max(x_pred[1], 250.0)
        x_pred[4] = np.clip(x_pred[4], 1.0, 4.0); x_pred[5] = np.clip(x_pred[5], 0.5, 1.8)
        x_pred[6] = np.clip(x_pred[6], 0.40, 1.20); x_pred[7] = np.clip(x_pred[7], 0.40, 1.20)
        x_pred[8] = np.clip(x_pred[8], 0.50, 1.00)
        A = np.eye(N_STATE) + F * dt_h
        P_pred = A @ P @ A.T + Q * dt_h
        y_pred_j, H_j = _obs_and_jac(jnp.array(x_pred, dtype=jnp.float32))
        y_pred = np.asarray(y_pred_j); H = np.asarray(H_j)
        innov = raw_window[k, OBS_IDX] - y_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.solve(S, np.eye(3))
        x = x_pred + K @ innov
        x[0] = np.clip(x[0], 1e-4, 0.999); x[6] = np.clip(x[6], 0.40, 1.20)
        x[7] = np.clip(x[7], 0.40, 1.20); x[8] = np.clip(x[8], 0.50, 1.00)
        P = (np.eye(N_STATE) - K @ H) @ P_pred
    return x, P


def coverage90(estimates, stds, truth):
    """90% Gaussian-interval coverage (z=1.645), the convention used throughout nb26/nb27."""
    estimates = np.asarray(estimates); stds = np.asarray(stds)
    return float(np.mean(np.abs(estimates - truth) <= 1.645 * stds))
