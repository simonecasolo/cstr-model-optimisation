"""21-state augmented EKF for the Luyben recycle plant.

State vector (21-D):
    [Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s,       (8 plant states, no integrators)
     I_T, I_Ts, I_L, I_R, I_P,                   (5 PI integrators)
     alpha, beta_r, eta_sep, beta_s,               (4 params)
     eta_p, xi, kappa, delta]                      (4 params)

Key design: Jacobian computed via jax.jacobian(luyben_rhs) at each EKF step,
eliminating the need to hand-derive the 21x21 matrix. This is feasible because
luyben_rhs is a pure JAX function.

Degradation parameters are modelled as random-walk states:
    d(param)/dt = 0  +  process noise

Reference: standard augmented EKF (Jazwinski 1970, Ch. 8).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from cstr_sbi.luyben.physics import (
    NOMINAL_CTRL_ALL,
    NOMINAL_INLET,
    NOMINAL_THETA,
    NOMINAL_Y0,
    N_STATES,
    N_PARAMS,
    luyben_rhs,
    extract_observations,
    PARAM_NAMES,
)


N_AUG: int = N_STATES + N_PARAMS   # 21

# Default noise covariances
# Q: process noise for plant states + degradation parameters
DEFAULT_Q_DIAG = np.array([
    1e-4, 1e-4, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-1,   # plant states
    1e-3, 1e-3, 1e-3, 1e-3, 1e-3,                       # PI integrators
    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,    # degradation params
], dtype=np.float64)

# R: measurement noise covariance for [T_r, Tc, Qc, T_s, Q_s, F_R, F_P, F_prod]
DEFAULT_R_DIAG = np.array([
    0.1, 0.1, 25.0, 0.1, 25.0, 1.0, 0.25, 1.0,
], dtype=np.float64)

# Initial state covariance
DEFAULT_P0_DIAG = np.array([
    1e-2, 1e-2, 1.0, 1.0, 10.0, 1e-3, 1e-3, 1.0,    # plant states
    0.1, 0.1, 0.1, 0.1, 0.1,                          # PI integrators
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,          # degradation params
], dtype=np.float64)


def _augmented_rhs(y_aug, theta_aug, inlet, ctrl):
    """Augmented RHS: plant states + parameter random-walk (d_theta/dt = 0).

    y_aug = [Ca, Cb, T_r, Tc, n_L, x_A, x_B, T_s, I_T, I_Ts, I_L, I_R, I_P,
             alpha, beta_r, eta_sep, beta_s, eta_p, xi, kappa, delta]
    """
    y_plant = y_aug[:N_STATES]
    theta   = y_aug[N_STATES:]
    dplant  = luyben_rhs(0.0, y_plant, (theta, inlet, ctrl))
    dtheta  = jnp.zeros(N_PARAMS)
    return jnp.concatenate([dplant, dtheta])


# JIT-compiled Jacobian of augmented RHS w.r.t. augmented state
_jac_fn = jax.jit(jax.jacobian(_augmented_rhs, argnums=0))


def _measurement_fn(y_aug, theta_aug, ctrl):
    """Map augmented state -> 8 observable channels."""
    y_plant = y_aug[:N_STATES]
    theta   = y_aug[N_STATES:]
    obs = extract_observations(y_plant[None, :], theta, ctrl)[0]  # (8,)
    return obs


_meas_jac_fn = jax.jit(jax.jacobian(_measurement_fn, argnums=0))


class LuybenEKF:
    """Augmented EKF for the 21-state Luyben plant.

    Parameters
    ----------
    Q_diag
        21-D process noise diagonal (default ``DEFAULT_Q_DIAG``).
    R_diag
        8-D measurement noise diagonal (default ``DEFAULT_R_DIAG``).
    P0_diag
        21-D initial covariance diagonal (default ``DEFAULT_P0_DIAG``).
    inlet, ctrl
        Plant operating conditions.
    dt
        Euler discretisation step for the prediction step (minutes).
    """

    def __init__(
        self,
        Q_diag: np.ndarray = None,
        R_diag: np.ndarray = None,
        P0_diag: np.ndarray = None,
        inlet: jnp.ndarray = None,
        ctrl: jnp.ndarray = None,
        dt: float = 0.1,
    ):
        self.Q = np.diag(Q_diag if Q_diag is not None else DEFAULT_Q_DIAG)
        self.R = np.diag(R_diag if R_diag is not None else DEFAULT_R_DIAG)
        self.P = np.diag(P0_diag if P0_diag is not None else DEFAULT_P0_DIAG)
        self.inlet = inlet if inlet is not None else NOMINAL_INLET
        self.ctrl  = ctrl  if ctrl  is not None else NOMINAL_CTRL_ALL
        self.dt    = dt

        # Augmented state: concatenate y0 + nominal theta
        self.x = np.concatenate([
            np.asarray(NOMINAL_Y0, dtype=np.float64),
            np.ones(N_PARAMS, dtype=np.float64),
        ])
        self.x[N_STATES + 7] = 0.0  # delta starts at 0

    def set_initial_state(self, y0: np.ndarray, theta0: np.ndarray = None):
        """Initialise the filter state from a warm-start IC."""
        self.x[:N_STATES] = np.asarray(y0, dtype=np.float64)
        if theta0 is not None:
            self.x[N_STATES:] = np.asarray(theta0, dtype=np.float64)

    def predict(self):
        """EKF prediction step: propagate mean and covariance."""
        x_jnp = jnp.array(self.x, dtype=jnp.float32)
        # Jacobian A = df/dx at current state
        A = np.asarray(_jac_fn(x_jnp, x_jnp[N_STATES:], self.inlet, self.ctrl), dtype=np.float64)

        # Euler discretisation: F = I + A * dt (first-order)
        F = np.eye(N_AUG) + A * self.dt

        # Propagate mean (Euler step on augmented RHS)
        f_val = np.asarray(
            _augmented_rhs(x_jnp, x_jnp[N_STATES:], self.inlet, self.ctrl),
            dtype=np.float64,
        )
        # Clip states to physical bounds
        x_pred = self.x + f_val * self.dt
        x_pred[0]  = max(x_pred[0],  0.0)   # Ca >= 0
        x_pred[1]  = max(x_pred[1],  0.0)   # Cb >= 0
        x_pred[4]  = max(x_pred[4],  1.0)   # n_L >= 1
        x_pred[5]  = np.clip(x_pred[5], 0.0, 1.0)  # x_A
        x_pred[6]  = np.clip(x_pred[6], 0.0, 1.0)  # x_B

        # Propagate covariance
        P_pred = F @ self.P @ F.T + self.Q

        self.x = x_pred
        self.P = P_pred

    def update(self, z: np.ndarray):
        """EKF update step with 8-D measurement z = [T_r, Tc, Qc, T_s, Q_s, F_R, F_P, F_prod]."""
        x_jnp = jnp.array(self.x, dtype=jnp.float32)

        # Predicted measurement
        z_pred = np.asarray(
            _measurement_fn(x_jnp, x_jnp[N_STATES:], self.ctrl),
            dtype=np.float64,
        )

        # Measurement Jacobian H = dh/dx
        H = np.asarray(_meas_jac_fn(x_jnp, x_jnp[N_STATES:], self.ctrl), dtype=np.float64)

        # Innovation
        inn = np.asarray(z, dtype=np.float64) - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.solve(S, np.eye(8))
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # State update
        self.x = self.x + K @ inn

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(N_AUG) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_theta_estimate(self) -> np.ndarray:
        """Return current degradation parameter estimates (8-D)."""
        return self.x[N_STATES:].copy()

    def get_theta_std(self) -> np.ndarray:
        """Return posterior std for degradation parameters (8-D)."""
        return np.sqrt(np.diag(self.P)[N_STATES:])


# ---------------------------------------------------------------------------
# Convenience: run EKF on a single observation window
# ---------------------------------------------------------------------------

def run_ekf_on_window(
    obs: np.ndarray,
    t: np.ndarray,
    *,
    y0: np.ndarray = None,
    theta0: np.ndarray = None,
    inlet: jnp.ndarray = None,
    ctrl: jnp.ndarray = None,
    Q_diag: np.ndarray = None,
    R_diag: np.ndarray = None,
) -> dict:
    """Run the augmented EKF over a single (n_t, 8) observation window.

    Returns a dict with:
        "theta_mean"  -- (n_t, 8) parameter estimates over time
        "theta_std"   -- (n_t, 8) parameter posterior std
        "final_mean"  -- (8,) final estimate
        "final_std"   -- (8,) final std
    """
    if y0 is None:
        y0 = np.asarray(NOMINAL_Y0)
    if theta0 is None:
        theta0 = np.ones(N_PARAMS)
        theta0[7] = 0.0  # delta

    ekf = LuybenEKF(Q_diag=Q_diag, R_diag=R_diag, inlet=inlet, ctrl=ctrl)
    ekf.set_initial_state(y0, theta0)

    n_t = obs.shape[0]
    theta_hist = np.zeros((n_t, N_PARAMS))
    std_hist   = np.zeros((n_t, N_PARAMS))

    dt_obs = float(t[1] - t[0]) if n_t > 1 else 1.0
    n_predict_steps = max(1, int(round(dt_obs / ekf.dt)))
    ekf.dt = dt_obs / n_predict_steps

    for i in range(n_t):
        for _ in range(n_predict_steps):
            ekf.predict()
        ekf.update(obs[i])
        theta_hist[i] = ekf.get_theta_estimate()
        std_hist[i]   = ekf.get_theta_std()

    return {
        "theta_mean":  theta_hist,
        "theta_std":   std_hist,
        "final_mean":  theta_hist[-1],
        "final_std":   std_hist[-1],
    }
