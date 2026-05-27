# `cstr_sbi.simulator` — design and reference

This document describes the M2 stochastic simulator
([`src/cstr_sbi/simulator.py`](../src/cstr_sbi/simulator.py)) in enough
detail that downstream milestones (M3 summary statistics, M4 SBI training,
M5 NUTS MCMC, M6 scenario batches) can use it as a black box, and that
anyone modifying it can do so without violating the conventions used by
the rest of the package.

The simulator's job is, given a parameter vector `(UA, k0, alpha, beta)`,
an inlet, a controller setting and an integration window, to produce
**realistic noisy time series** of the four observable channels
`[C, T, Tc, Qc]` that an industrial sensor layer would see. Every
downstream consumer treats the output of this module as "the data".

For the underlying ODE physics (deterministic CSTR + PI controller), see
[`physics.py`](../src/cstr_sbi/physics.py); for the eight scenarios that
exercise the simulator, see [`scenarios.py`](../src/cstr_sbi/scenarios.py)
and [`notebooks/02_data_generation.ipynb`](../notebooks/02_data_generation.ipynb).

---

## 1. Overview

```
                +-------------------------+
   theta=[UA,   |                         |  ys = [C, T, Tc, I](t)
   k0, alpha,   | closed-loop / open-loop |
   beta]        |  Euler-Maruyama scan    |--+
                | (process noise)         |  |
   inlet,       |                         |  |
   ctrl, IC --> +-------------------------+  |
                                             v
                                  +--------------------+
                                  | sensor layer       |
                                  |  - Gaussian noise  |
                                  |  - additive drift  |
                                  +---------+----------+
                                            |
                                            v
                                  observations [C, T, Tc, Qc] (t)
```

* The plant model is the 4-state SDE `[C, T, Tc, I]` (or 3-state
  `[C, T, Tc]` in open loop). The integrator state `I` carries no
  process noise.
* The controller is **inside** the SDE: at every internal step we
  recompute `Qc(T, I)` via `compute_qc` (clipped to `[Qc_min, Qc_max]`)
  and gate the integrator with conditional integration so the windup
  stays bounded.
* The sensor layer is applied **after** integration on the four
  observable channels `[C, T, Tc, Qc]`. Sensor noise is i.i.d. across
  channels and timestamps; sensor drift is a constant additive offset
  applied to `T` (and, where exercised, to `Ci` at the inlet level).
* Every public entry point is `jax.jit`-friendly and `jax.vmap`-friendly,
  which is what makes the M4 training-set generation and M5 likelihood
  evaluation tractable on a single CPU.

---

## 2. Mathematical model

### 2.1 Closed-loop SDE

The continuous-time SDE solved by `simulate_em_window` is

$$
\begin{aligned}
\mathrm{d}C  &= \Big[\tfrac{Q}{V}(C_i - C) - \alpha\,k_0\,e^{-E_a/(RT)}\,C\Big]\,\mathrm{d}t + \sigma_C\,\mathrm{d}W_C \\[2pt]
\mathrm{d}T  &= \Big[\tfrac{Q}{V}(T_i - T) - \tfrac{H_r\,\alpha\,k\,C}{\rho\,C_p} - \tfrac{\beta\,UA\,(T-T_c)}{\rho\,C_p\,V}\Big]\,\mathrm{d}t + \sigma_T\,\mathrm{d}W_T \\[2pt]
\mathrm{d}T_c &= \Big[\tfrac{Q_c}{V_c}(T_{ci}-T_c) + \tfrac{\beta\,UA\,(T-T_c)}{\rho_c\,C_{pc}\,V_c}\Big]\,\mathrm{d}t + \sigma_{T_c}\,\mathrm{d}W_{T_c} \\[2pt]
\mathrm{d}I  &= (T - T_{sp})\cdot\mathbb{1}\{Q_{c,\min}<Q_c^{\mathrm{u}}<Q_{c,\max}\}\,\mathrm{d}t \\[2pt]
Q_c &= \mathrm{clip}\Big(Q_{c0} + K_p(T-T_{sp}) + I/\tau_i,\;Q_{c,\min},\;Q_{c,\max}\Big)
\end{aligned}
$$

where `(W_C, W_T, W_{T_c})` are independent Wiener processes with
diffusion coefficients `(sigma_C, sigma_T, sigma_{Tc})` from
`DEFAULT_PROCESS_SIGMA`. The integrator state `I` is purely deterministic.

### 2.2 Discretisation (Euler–Maruyama)

We use an explicit Euler–Maruyama (EM) scheme with step `dt = 0.01 min`:

$$
y_{n+1} = y_n + f(y_n)\,\Delta t + \sigma\,\sqrt{\Delta t}\,\xi_n,\qquad
\xi_n \sim \mathcal{N}(0, I_4).
$$

`f(y_n)` is `cstr_closed_loop_rhs` from `physics.py`; `sigma` is the
4-vector `[sigma_C, sigma_T, sigma_{Tc}, 0]` (no noise on the integrator
state). The full inner loop is a `jax.lax.scan` so the whole 6 000-step
integration JIT-compiles into one XLA call.

### 2.3 Output subsampling

After integrating at the EM cadence (`dt = 0.01 min`), the trajectory is
strided down to the user-facing output cadence (`dt_out = 0.5 min`
gives 120 samples for a 60-minute window). The deterministic output
quantity `Qc(t)` is recomputed from `T(t)` and `I(t)` at the output
cadence using `compute_qc` from `physics.py`, ensuring that the recorded
`Qc` is consistent with the controller equation including saturation
clipping.

### 2.4 Sensor layer

Given the noiseless 4-channel trajectory `Y(t) = [C, T, Tc, Qc](t)`,
the sensor layer applies

$$
Y_{\text{obs}}(t) = Y(t) + \eta(t) + d
$$

where
* `eta(t) ~ N(0, diag(sigma_obs^2))`, with
  `sigma_obs[k] = noise_pct * max_t |Y(t)[k]|` (channel-wise scaling
  from research spec §3.5), and
* `d = (0, drift_T, 0, 0)` is a constant additive offset on the T
  channel (Sc 7's drift substudy), applied **after** the noise so the
  drift is exactly what an SBI parameter would have to recover.

Sensor drift on `Ci` is handled at the **inlet** layer (it perturbs the
inlet that the simulator sees); see `scenarios.perturb_inlet` for the
M6 hook.

---

## 3. Module map

| Symbol | Kind | Purpose |
|---|---|---|
| `DEFAULT_PROCESS_SIGMA` | `jnp.ndarray (4,)` | Diffusion coefficients `[sigma_C, sigma_T, sigma_{Tc}, 0]` in `[unit]/sqrt(min)`. |
| `DEFAULT_SENSOR_NOISE_PCT` | `float` | Gaussian sensor noise as a fraction of channel max (0.005 = 0.5 %). |
| `DEFAULT_DRIFT_T`, `DEFAULT_DRIFT_CI` | `float` | Constant additive offsets on the T sensor and on Ci. |
| `DEFAULT_DT_INT`, `DEFAULT_DT_OUT` | `float` | Internal EM step (0.01 min) and output grid (0.5 min). |
| `_em_scan_closed_loop`, `_em_scan_open_loop` | `jit`'d helpers | The Euler–Maruyama inner loops; not part of the public API. |
| `simulate_em_window` | function | One closed-loop EM trajectory. |
| `simulate_em_window_open_loop` | function | One open-loop EM trajectory (3-state, fixed `Qc`). |
| `apply_sensor_layer` | function | Add Gaussian noise + drift to a 4-channel trajectory. |
| `generate_replicates` | function | `vmap`-batched closed-loop replicate generator. |
| `warm_start_ic` | function | Compute the deterministic steady state of `(params, inlet, ctrl)` for use as an EM warm-start IC. |

The two underscored helpers are `jax.jit`'d with `n_steps` and `stride`
as static arguments, so changing `t_window` or `dt_out` triggers a fresh
trace; reusing the same horizon across many parameter draws (the SBI
training-set case) reuses the cached compilation.

---

## 4. Public API

### 4.1 `simulate_em_window`

```python
simulate_em_window(
    params,        # (4,) [UA, k0, alpha, beta]
    inlet,         # (3,) [Ci, Ti, Tci]
    ctrl,          # (6,) [Kp, tau_i, Tsp, Qc0, Qc_min, Qc_max]
    y0,            # (4,) [C0, T0, Tc0, I0]
    *,
    key,           # jax.random.PRNGKey
    t_window=60.0, # min
    dt=0.01,       # internal EM step
    dt_out=0.5,    # output stride
    sigma=DEFAULT_PROCESS_SIGMA,  # (4,) /sqrt(min)
) -> (t, ys, qc)
```

* `t`  — `(n_out,)` minutes, **excluding** `t = 0` (only post-step
  states are saved).
* `ys` — `(n_out, 4)`, state trajectory `[C, T, Tc, I]`.
* `qc` — `(n_out,)`, the realised (clipped) coolant flow recomputed from
  `T(t)` and `I(t)` via `compute_qc`.

### 4.2 `simulate_em_window_open_loop`

```python
simulate_em_window_open_loop(
    params,        # (2,) [UA_eff, k0_eff]
    inlet_ol,      # (4,) [Ci, Ti, Tci, Qc]
    y0,            # (3,) [C0, T0, Tc0]
    *,
    key,
    t_window=60.0, dt=0.01, dt_out=0.5,
    sigma=DEFAULT_PROCESS_SIGMA,
) -> (t, ys, qc_const)
```

* For the open-loop scenarios (Sc 0 healthy, Sc 6 fault baseline),
  `Qc` is held fixed at `inlet_ol[3]` and the fault is encoded by
  scaling `[UA, k0]` directly: `params = [beta * UA_nominal, alpha * k0_nominal]`.
* `qc_const` is the constant `Qc` repeated for shape parity with the
  closed-loop output.

### 4.3 `apply_sensor_layer`

```python
apply_sensor_layer(
    ys_obs,                # (n_t, 4) [C, T, Tc, Qc]
    *,
    key,
    noise_pct=0.005,
    drift_T=0.0,
) -> ys_obs_with_noise     # (n_t, 4)
```

Channel-wise `sigma_obs[k] = noise_pct * max_t |ys_obs[t, k]|`; noise
drawn as `N(0, sigma_obs^2)` independently per `(t, k)`. `drift_T` is
added to the T channel after noise.

### 4.4 `generate_replicates`

```python
generate_replicates(
    params, inlet, ctrl, y0,
    n_replicates,
    master_key,
    *,
    t_window=60.0,
    dt=0.01, dt_out=0.5,
    sigma_proc=DEFAULT_PROCESS_SIGMA,
    noise_pct=0.005,
    drift_T=0.0,
) -> (t, observations)
```

* `observations` — `(n_replicates, n_t, 4)` of `[C, T, Tc, Qc]` with
  process noise + sensor noise + (optional) drift. Each replicate
  uses an independent stream split from `master_key`.
* Internally `vmap`'s the per-replicate work over the 2*N seed splits
  (`N` for process noise, `N` for sensor noise) and runs in a single
  JIT-compiled XLA call.

### 4.5 `warm_start_ic`

```python
warm_start_ic(
    params,                # (4,) [UA, k0, alpha, beta]
    inlet,                 # (3,) [Ci, Ti, Tci]
    ctrl=NOMINAL_CTRL,
    t_warm=1500.0,
) -> y0                    # (4,) [C, T, Tc, I]
```

Solves the deterministic closed-loop ODE to (near) steady state and
returns the resulting state. This is the M2-side answer to the M1
finding that a cold IC such as `[0.5 mol/L, 300 K, 297 K, 0]` causes
Tsit5 to break: each scenario gets its own warm-start, computed once
and reused across all replicates.

---

## 5. Default noise levels — what they mean

### 5.1 Process noise

`DEFAULT_PROCESS_SIGMA = [0.0005, 0.1, 0.1, 0.0]` in `[unit]/sqrt(min)`.

For a stable Ornstein–Uhlenbeck process `dy = -k(y - y*) dt + sigma dW`
the steady-state std is `sigma / sqrt(2k)`. With the Fogler PI tuning,
the effective T restoration rate is dominated by the residence-time
term `Q/V = 0.08 /min` and the controller; numerically we observe a
per-replicate std in T of about **0.12 K** (clean signal, noise small
relative to the ~1 K fault signature in the Sc 5 saturation regime).

The literal spec values (`0.1 mol/L/min` on C, `10 K/min` on T and Tc;
research spec §3.5) are documented in the module docstring as a
historical reference but are **not** the runtime defaults: applied as
proper SDE diffusion they accumulate variance of order `100 * 60 = 6000 K^2`
over a 60-minute window — orders of magnitude above the signal.
The legacy code almost certainly meant a discrete-time interpretation
(noise increment per fixed step equals `sigma * dt`, not `sigma * sqrt(dt)`).

### 5.2 Sensor noise

`DEFAULT_SENSOR_NOISE_PCT = 0.005` (0.5 % of each channel's running max).
On T this is approximately **1.6 K** at the Fogler operating point (T_max
~ 318 K). This is the dominant source of observation noise in the M2
output and is the realistic industrial-grade level prescribed by the
spec.

### 5.3 Drift

`DEFAULT_DRIFT_T = 0.0` (only Sc 7 turns this on, with `+2 K`).
`DEFAULT_DRIFT_CI = 0.0` likewise; the inlet drift hook lives in
`scenarios.perturb_inlet` and is wired into the M6 30-day stream
generator.

---

## 6. Usage examples

### 6.1 One closed-loop replicate, healthy reactor

```python
import jax
from cstr_sbi.physics import NOMINAL_INLET_CL, NOMINAL_CTRL
from cstr_sbi.scenarios import SCENARIO_CONFIGS
from cstr_sbi.simulator import generate_replicates, warm_start_ic

sc = SCENARIO_CONFIGS["Sc1_closed_healthy"]
y0 = warm_start_ic(sc.params(), NOMINAL_INLET_CL, NOMINAL_CTRL)

t, obs = generate_replicates(
    sc.params(), NOMINAL_INLET_CL, NOMINAL_CTRL, y0,
    n_replicates=1, master_key=jax.random.PRNGKey(0),
)
# obs.shape == (1, 120, 4)   -- one replicate, 120 timestamps, [C, T, Tc, Qc]
```

### 6.2 Many replicates of a fouling scenario for an SBI training batch

```python
from cstr_sbi.scenarios import SCENARIO_CONFIGS
from cstr_sbi.simulator import generate_replicates, warm_start_ic
from cstr_sbi.physics import NOMINAL_INLET_CL, NOMINAL_CTRL
import jax

sc = SCENARIO_CONFIGS["Sc2_closed_fouling"]
y0 = warm_start_ic(sc.params(), NOMINAL_INLET_CL, NOMINAL_CTRL)

t, obs = generate_replicates(
    sc.params(), NOMINAL_INLET_CL, NOMINAL_CTRL, y0,
    n_replicates=2_000, master_key=jax.random.PRNGKey(42),
    t_window=60.0,
)
# obs.shape == (2000, 120, 4)
```

After JIT warmup the second call at the same `(t_window, dt, dt_out)`
takes ~0.05 s for 25 replicates and ~1 s for 2 000 (single CPU); see
the M0 baseline benchmark write-up for the deterministic per-sim cost.

### 6.3 Sensor drift substudy (Sc 7)

```python
sc = SCENARIO_CONFIGS["Sc7_closed_drift"]
y0 = warm_start_ic(sc.params(), NOMINAL_INLET_CL, NOMINAL_CTRL)

t, obs = generate_replicates(
    sc.params(), NOMINAL_INLET_CL, NOMINAL_CTRL, y0,
    n_replicates=50, master_key=jax.random.PRNGKey(7),
    drift_T=sc.drift_T,         # +2 K
)
```

The M4 SBI training will infer `drift_T` as part of the extended 6-D
parameter vector `theta = [UA, k0, alpha, beta, delta_T, delta_Ci]`.

### 6.4 Open-loop variant (Sc 0 / Sc 6)

```python
import jax, jax.numpy as jnp
from cstr_sbi.simulator import simulate_em_window_open_loop
from cstr_sbi.physics import (
    NOMINAL_INLET, K0_NOMINAL, UA_NOMINAL, QC0,
)

# Sc 6: open-loop simulator with a fault encoded by scaling UA by beta = 0.7
params_ol = jnp.array([0.7 * UA_NOMINAL, 1.0 * K0_NOMINAL])
inlet_ol  = jnp.asarray(NOMINAL_INLET).at[3].set(QC0)
y0_ol     = jnp.array([0.018, 312.15, 299.05])

t, ys, qc = simulate_em_window_open_loop(
    params_ol, inlet_ol, y0_ol, key=jax.random.PRNGKey(0),
)
```

---

## 7. Performance

* **JIT compile cost (cold).** ~0.3 s for 25 closed-loop replicates the
  first time `generate_replicates` is invoked at a new `(t_window, dt,
  dt_out)`; cached calls are ~10x faster.
* **Throughput (warm).** On a single Apple-silicon CPU we observe ~1 200
  closed-loop 60-minute replicates per second after JIT, with 50 000
  M4-style samples taking under a minute. Per-sim cost rises sub-linearly
  with `n_replicates` because XLA amortises kernel-launch overhead.
* **Memory.** A 50-replicate, 60-minute, 0.5-min-cadence batch is
  `50 * 120 * 4 * 4 bytes = 96 kB`. The full M2 dataset (8 scenarios,
  50 replicates) is under 1 MB.
* **Determinism.** Identical `master_key` → identical observations.
  Each replicate consumes two seed splits (`proc_key`, `sens_key`),
  which are derived from `jax.random.split(master_key, 2 * n_replicates)`.

---

## 8. Known limitations and follow-ups

* **Inlet perturbation** (research spec §3.5: a fresh `[Ci, Ti, Tci]`
  draw every 60 minutes) is implemented in
  [`scenarios.perturb_inlet`](../src/cstr_sbi/scenarios.py) but not
  yet wired into `generate_replicates`. M6's 30-day stream generator
  will use it.
* **EM is first-order.** For tighter accuracy at fixed `dt` we will
  consider Milstein or `diffrax.MultiTerm` SDE solvers in M4 if SBI
  coverage diagnostics on Sc 1 fall short of target.
* **Process-noise scaling** has been calibrated for clean visualisation
  (sigma_T = 0.1 K/sqrt(min)). The literal spec amplitudes are recorded
  in the module docstring; M4 may revisit after running the SBI
  coverage check on Sc 1.
* **Cold ICs.** The current contract is "the user supplies a warm IC".
  In M4 we may compose a brief deterministic warm-up phase inside
  `generate_replicates` to make the API safer with arbitrary ICs.
* **Sc 0 / Sc 6 Qc channel** is constant by construction (open-loop).
  We still record it for shape parity with closed-loop scenarios; M3's
  summary statistics will see it as "no Qc fault signal" which is
  exactly what the failure-baseline narrative wants.

---

## 9. Where to look next

* [`physics.py`](../src/cstr_sbi/physics.py) — the deterministic ODE
  RHS and PI controller, plus the Tsit5 helpers used by
  `warm_start_ic` and the M1 demonstration.
* [`scenarios.py`](../src/cstr_sbi/scenarios.py) — the
  `SCENARIO_CONFIGS` truth table consumed by notebook 02 and the M6
  batch generators.
* [`notebooks/02_data_generation.ipynb`](../notebooks/02_data_generation.ipynb)
  — the rendered end-to-end use of every public function in this
  module, including the headline Qc-vs-scenario figure.
* [`docs/m0_baseline_benchmarks.md`](m0_baseline_benchmarks.md) — the
  CPU throughput reference for the deterministic open-loop simulator
  that underlies `warm_start_ic`.
