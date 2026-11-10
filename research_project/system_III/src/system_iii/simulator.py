"""System III window simulator (vectorised numpy Euler-Maruyama).

One call simulates a batch of independent windows, each with its own fault parameters (alpha, beta),
its own true inlet (Ci, Ti, Tci, Q) and its own sensor calibration offsets. Returns the logged
measurements and the true states.

Control modes
-------------
* ``control="closed"`` (default): digital PI on the MEASURED reactor temperature, executed every
  ``ctrl_dt`` (default: every integration step, 1.2 s, as a DCS) with a first-order PV filter
  (``pv_filter_min``, default 0.2 min), anti-windup by
  conditional integration, valve clamp [Qcmin, Qcmax]. The control sensor carries noise and the
  calibration offset ``oT`` (so an offset moves the plant: true T settles at Tsp - oT).
  ``control_sensor="true"`` reproduces System I's convention (continuous PI on the true T).
* ``control="cascade"``: outer PI (reactor T, measured, PV filter) sets the jacket-outlet temperature
  setpoint; inner PI (measured jacket-outlet Tc, filter ``inner_filter_min``) moves the coolant valve.
  ``cascade=(Ko K/K, tau_o min, Ki (L/min)/K, tau_i min)``. The standard industrial structure for jacketed
  exothermic reactors; needed once the thermowell lags are modelled (see design/verify_s5.md).
* ``control="fixed"``: coolant flow fixed at ``qc_level`` (open-loop operation; multistable for
  some (alpha, beta), see docs).
* ``control="exogenous"``: coolant flow = qc_level * (1 + OU fluctuation): a controller-agnostic
  simulator driven by an external manipulated-variable trajectory (revised Claim 1).

Warm-start protocols (as System I)
----------------------------------
P0: steady state of the healthy plant (alpha = beta = 1), fault applied as a step at the window start.
P1: exact steady state of (alpha, beta), no burn-in (noise-free start).
P1b: steady state of (alpha, beta) + ``burn_min`` noisy burn-in.
P2 (production): (alpha, beta) ramp linearly over ``hist_min`` from a healthier start
     (theta + U(0,1) * rate * hist_min, clipped to the prior) to the current value, then the window.

Sensor dynamics (``sensors="S5"`` or ``sensor_dyn=dict(...)``): first-order thermowells on T and Tc (the
controller sees the lagged T) and an online analyser that reports C every ``analyser_period`` min from a
sample taken ``analyser_dead`` min earlier, held between updates. Static sets (S0-S4) are unchanged.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .physics import PLANT, Plant, k_of, ss_unsat, ss_closed
from .sensors import SENSORS, SENSOR_DYN

ALPHA_LO, ALPHA_HI, BETA_LO, BETA_HI = 0.4, 1.2, 0.4, 1.2
RATE = (0.0005, 0.0005)          # per min, P2 history drift rates (alpha, beta)
PROC_SIGMA = (0.0015, 0.1, 0.1)  # per sqrt(min), process noise on C, T, Tc


def _bc(v, default, n):
    return np.full(n, float(default)) if v is None else np.broadcast_to(np.asarray(v, float), (n,)).copy()


def initial_state(p: Plant, alpha, beta, Ci, Ti, Tci, Q, *, true_T_target=None):
    """Closed-loop steady state per sample; true_T_target (n,) lets the reactor settle at Tsp - oT
    (control-sensor offset). Saturated rows fall back to the open-loop root nearest the target."""
    n = len(alpha)
    tgt = np.full(n, p.Tsp) if true_T_target is None else true_T_target
    y = np.zeros((n, 4)); qc = np.zeros(n)
    for grp in np.unique(tgt):                       # usually one value
        m = tgt == grp
        pg = replace(p, Tsp=float(grp))
        s = ss_unsat(pg, alpha[m], beta[m], Ci=Ci[m], Ti=Ti[m], Tci=Tci[m], Q=Q[m])
        ok = (s["Qc"] >= p.Qcmin) & (s["Qc"] <= p.Qcmax) & np.isfinite(s["Qc"])
        idx = np.flatnonzero(m)
        y[idx[ok], 0], y[idx[ok], 1], y[idx[ok], 2] = s["C"][ok], s["T"][ok], s["Tc"][ok]
        qc[idx[ok]] = s["Qc"][ok]
        y[idx[ok], 3] = (s["Qc"][ok] - p.Qc0) * p.taui / p.Kp
        for j in idx[~ok]:
            sols = ss_closed(pg, alpha[j], beta[j], Ci=Ci[j], Ti=Ti[j], Tci=Tci[j], Q=Q[j])
            s1 = min(sols, key=lambda x: abs(x["T"] - grp))
            y[j] = [s1["C"], s1["T"], s1["Tc"], 0.0]; qc[j] = s1["Qc"]
    return y, qc


def simulate(alpha, beta, *, seed: int, plant: Plant = PLANT, sensors="S3", protocol="P2",
             Ci=None, Ti=None, Tci=None, Q=None,
             oT=0.0, oTc=0.0, uQc=0.0, uC=0.0, drift_T_monitor=0.0,
             control="closed", control_sensor="measured", qc_level=None, qc_ou=(0.05, 10.0),
             pv_filter_min=0.2, ctrl_dt=None, hist_min=300.0, burn_min=150.0, t_window=60.0, dt=0.02, dt_out=0.5,
             sensor_dyn=None, cascade=(6.0, 6.0, 30.0, 1.0), inner_filter_min=0.05):
    """Simulate a batch of windows. Returns dict(obs (n, nt, 4) logged [C, T, Tc, Qc],
    true (n, nt, 3) [C, T, Tc], qc_true (n, nt), t (nt,), inlet (n, 4) true [Ci, Ti, Tci, Q])."""
    rng = np.random.default_rng(seed)
    alpha = np.atleast_1d(np.asarray(alpha, float)); beta = np.atleast_1d(np.asarray(beta, float))
    n = max(len(alpha), len(beta)); alpha = _bc(alpha, 1, n); beta = _bc(beta, 1, n)
    Ci, Ti, Tci, Q = _bc(Ci, plant.Ci, n), _bc(Ti, plant.Ti, n), _bc(Tci, plant.Tci, n), _bc(Q, plant.Q, n)
    oT, oTc, uQc, uC = (_bc(v, 0, n) for v in (oT, oTc, uQc, uC))
    dmon = _bc(drift_T_monitor, 0, n)
    sm = np.asarray(SENSORS[sensors] if isinstance(sensors, str) else sensors, float)   # [C, T, Tc, Qc]
    dyn = dict(SENSOR_DYN.get(sensors, {}) if isinstance(sensors, str) else {})
    if sensor_dyn is not None:
        dyn.update(sensor_dyn)
    p = plant

    # ---- start state and fault history
    if protocol == "P2":
        u = rng.uniform(0, 1, size=(n, 2))
        a0 = np.clip(alpha + u[:, 0] * RATE[0] * hist_min, ALPHA_LO, ALPHA_HI)
        b0 = np.clip(beta + u[:, 1] * RATE[1] * hist_min, BETA_LO, BETA_HI)
        pre = int(round(hist_min / dt))
    elif protocol == "P1b":
        a0, b0, pre = alpha.copy(), beta.copy(), int(round(burn_min / dt))
    elif protocol == "P1":
        a0, b0, pre = alpha.copy(), beta.copy(), 0
    elif protocol == "P0":
        a0, b0, pre = np.ones(n), np.ones(n), 0
    else:
        raise ValueError(protocol)
    tgt = p.Tsp - oT if (control in ("closed", "cascade") and control_sensor == "measured") else None
    y, qc = initial_state(p, a0, b0, Ci, Ti, Tci, Q, true_T_target=tgt)
    if control == "fixed" or control == "exogenous":
        lvl = _bc(qc_level, p.Qc0, n)
        qc = lvl.copy()
    I = y[:, 3].copy()
    Tf = (y[:, 1] + oT) if control_sensor == "measured" else y[:, 1].copy()   # filtered PV
    ou = np.zeros(n)
    if control == "cascade":                             # outer T -> jacket-outlet setpoint, inner Tc -> Qc
        Ko, To, Ki, Tin_ = cascade
        Tc_nom = float(ss_unsat(p, np.ones(1), np.ones(1))["Tc"][0])
        Tcf = y[:, 2] + oTc                              # filtered inner PV (measured jacket outlet)
        Io = (Tc_nom - Tcf) * To / Ko                    # outer integrator: setpoint = current measured Tc
        Ii = (qc - p.Qc0) * Tin_ / Ki                    # inner integrator: flow = steady-state flow
    # ---- sensor dynamics (thermowells, sampled analyser with dead time); static if not configured
    tw_T, tw_Tc = float(dyn.get("tw_T", 0.0)), float(dyn.get("tw_Tc", 0.0))
    Tw, Tcw = y[:, 1].copy(), y[:, 2].copy()            # thermowell (sensed) temperatures
    an_per = float(dyn.get("analyser_period", 0.0))
    if an_per > 0:
        an_steps = max(int(round(an_per / dt)), 1)
        dead_steps = int(round(float(dyn.get("analyser_dead", 0.0)) / dt))
        cbuf = np.repeat(y[None, :, 0], dead_steps + 1, axis=0)   # ring buffer of true C
        phase = rng.integers(0, an_steps, size=n)
        C_held = y[:, 0] * (1 + uC) + sm[0] * rng.standard_normal(n)

    nwin = int(round(t_window / dt)); stride = int(round(dt_out / dt))
    sig = np.asarray(PROC_SIGMA) * np.sqrt(dt)
    ctrl_dt = dt if ctrl_dt is None else ctrl_dt               # DCS execution period (default: every step, 1.2 s)
    cstride = max(int(round(ctrl_dt / dt)), 1)
    a_pf = ctrl_dt / (pv_filter_min + ctrl_dt) if pv_filter_min > 0 else 1.0
    a_inf = ctrl_dt / (inner_filter_min + ctrl_dt) if inner_filter_min > 0 else 1.0
    obs, tru, qtr = [], [], []
    for s in range(pre + nwin):
        frac = min((s + 1) / pre, 1.0) if pre else 1.0
        if protocol == "P2":
            a, b = a0 + (alpha - a0) * frac, b0 + (beta - b0) * frac
        else:                                               # P0: step at t = 0; P1, P1b: constant
            a, b = alpha, beta
        # ---- controller / manipulated variable
        if control == "closed" and control_sensor == "true":
            qun = p.Qc0 + p.Kp * ((y[:, 1] - p.Tsp) + I / p.taui)
            qc = np.clip(qun, p.Qcmin, p.Qcmax)
            I = I + np.where((qun > p.Qcmin) & (qun < p.Qcmax), y[:, 1] - p.Tsp, 0.0) * dt
        elif control == "closed" and s % cstride == 0:         # digital PI on the measured T
            Tm = Tw + oT + sm[1] * rng.standard_normal(n)
            Tf = Tf + a_pf * (Tm - Tf)
            qun = p.Qc0 + p.Kp * ((Tf - p.Tsp) + I / p.taui)
            unsat = (qun > p.Qcmin) & (qun < p.Qcmax)
            I = I + np.where(unsat, Tf - p.Tsp, 0.0) * ctrl_dt
            qc = np.clip(qun, p.Qcmin, p.Qcmax)
        elif control == "cascade" and s % cstride == 0:
            Tm = Tw + oT + sm[1] * rng.standard_normal(n)
            Tf = Tf + a_pf * (Tm - Tf)
            Tcm = Tcw + oTc + sm[2] * rng.standard_normal(n)
            Tcf = Tcf + a_inf * (Tcm - Tcf)
            tcsp = Tc_nom - Ko * ((Tf - p.Tsp) + Io / To)
            qun = p.Qc0 + Ki * ((Tcf - tcsp) + Ii / Tin_)
            unsat = (qun > p.Qcmin) & (qun < p.Qcmax)
            Ii = Ii + np.where(unsat, Tcf - tcsp, 0.0) * ctrl_dt
            Io = Io + np.where(unsat, Tf - p.Tsp, 0.0) * ctrl_dt      # outer anti-windup: freeze when valve clamps
            qc = np.clip(qun, p.Qcmin, p.Qcmax)
        elif control == "exogenous" and s % stride == 0:
            ou = ou + (-ou / qc_ou[1]) * dt_out + qc_ou[0] * np.sqrt(2 * dt_out / qc_ou[1]) * rng.standard_normal(n)
            qc = np.clip(lvl * (1 + ou), p.Qcmin, p.Qcmax)
        # ---- plant
        C, T, Tc = y[:, 0], y[:, 1], y[:, 2]
        k = k_of(p, T, a)
        UAe = b * p.UA
        dC = Q / p.V * (Ci - C) - k * C
        dT = Q / p.V * (Ti - T) + (-p.dH) * k * C / p.rhoCp - UAe * (T - Tc) / (p.rhoCp * p.V)
        dTc = qc / p.Vc * (Tci - Tc) + UAe * (T - Tc) / (p.rhocCpc * p.Vc)
        y[:, :3] = y[:, :3] + np.stack([dC, dT, dTc], 1) * dt + sig * rng.standard_normal((n, 3))
        y[:, 0] = np.maximum(y[:, 0], 0.0)
        if tw_T > 0:
            Tw = Tw + (y[:, 1] - Tw) * (dt / tw_T)
        else:
            Tw = y[:, 1]
        if tw_Tc > 0:
            Tcw = Tcw + (y[:, 2] - Tcw) * (dt / tw_Tc)
        else:
            Tcw = y[:, 2]
        if an_per > 0:
            cbuf[s % (dead_steps + 1)] = y[:, 0]
            delayed = cbuf[(s + 1) % (dead_steps + 1)]       # true C dead_steps ago (initial C before that)
            upd = (s + phase) % an_steps == 0
            if upd.any():
                C_held = np.where(upd, delayed * (1 + uC) + sm[0] * rng.standard_normal(n), C_held)
        # ---- logging at the scan instants inside the window
        if s >= pre and (s - pre + 1) % stride == 0:
            Tlog = Tw + oT + sm[1] * rng.standard_normal(n)       # logged reading of the control sensor
            Clog = C_held if an_per > 0 else y[:, 0] * (1 + uC) + sm[0] * rng.standard_normal(n)
            obs.append(np.stack([Clog,
                                 Tlog + dmon,
                                 Tcw + oTc + sm[2] * rng.standard_normal(n),
                                 qc * (1 + uQc) + sm[3] * rng.standard_normal(n)], 1))
            tru.append(y[:, :3].copy()); qtr.append(qc.copy())
    t = np.arange(1, len(obs) + 1) * dt_out
    return dict(obs=np.stack(obs, 1), true=np.stack(tru, 1), qc_true=np.stack(qtr, 1), t=t,
                inlet=np.stack([Ci, Ti, Tci, Q], 1))


def prior_draws(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform([ALPHA_LO, BETA_LO], [ALPHA_HI, BETA_HI], size=(n, 2))
