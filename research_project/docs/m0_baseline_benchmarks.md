# M0 Baseline Benchmarks

Result of [`scripts/m0_smoke_test.py`](../scripts/m0_smoke_test.py) executed on
the open-loop CSTR ODE before any closed-loop / degradation work begins.

The acceptance criterion is the maximum relative error of the JAX/diffrax
integrator against a high-tolerance scipy reference at the nominal operating
point of [`../../cstr_parameters_recommended.md`](../../cstr_parameters_recommended.md);
it must be below `1e-3`.

Raw output is also persisted to [`m0_smoke_test_results.json`](m0_smoke_test_results.json)
on every run.

## Hardware / software

| Item | Value |
|---|---|
| Host | macOS (darwin 25.1.0) |
| Python | 3.10.11 (pyenv) |
| JAX devices | `[CpuDevice(id=0)]` (no GPU/Metal backend installed) |
| JAX | `>=0.4.30,<0.5` |
| diffrax | `>=0.6.0` |
| Solver | `diffrax.Tsit5` + `PIDController(rtol=1e-6, atol=1e-8)` |
| Integration window | `t_final = 200 min` (16 residence times at tau=12.5 min) |

## Steady-state correctness (Goal 1)

Nominal operating point from [`cstr_parameters_recommended.md`](../../cstr_parameters_recommended.md)
(Fogler 2016 Module 13 / Furusawa 1969 chemistry):

```
inlet  = (Ci=0.97 mol/L, Ti=297.0 K, Tci=297.0 K, Qc=80.0 L/min)
params = (UA=1.25e4 cal/min/K, k0=16.96e12 1/min)
        Ea = 75362 J/mol, Hr = -20220 cal/mol
        V  = 500 L,  Vc = 40 L,  Q = 40 L/min  (tau = V/Q = 12.5 min)
```

| Source | C [mol/L] | T [K] | Tc [K] |
|---|---|---|---|
| Doc Section 4 (approximate) | `~0.021` | `~312.5` | `~302` |
| `scipy.solve_ivp` Radau, `rtol=1e-9` | `0.018349` | `312.1483` | `299.0471` |
| `cstr_sbi.physics` (JAX/diffrax) | `0.018349` | `312.1483` | `299.0472` |

Maximum relative error of JAX vs scipy reference: **`9.4e-6`**, two orders of
magnitude below the `1e-3` acceptance threshold. The doc's `~0.021 / ~312.5 /
~302` figures are order-of-magnitude estimates from an analytical balance;
the numerical truth (consistent across JAX and scipy) is `0.0183 / 312.15 /
299.05`. The conversion is `1 - 0.0183/0.97 = 98.1%`, matching Fogler's
Module 13 result of ~97.8% conversion.

## Batched simulation throughput (Goal 2)

`jax.jit(jax.vmap(...))` over a batch of parameter draws (uniform on
`UA in [0.5, 2.0] x UA_nominal` and `k0 in [0.5, 2.0] x K0_nominal`),
single CPU device, after JIT warmup:

| N parameters | wall time | per-sim |
|---:|---:|---:|
|     1 000 | 0.45 s | 0.45 ms |
|    10 000 | 2.57 s | 0.26 ms |
|    50 000 | 5.33 s | 0.11 ms |

Per-sim cost drops by roughly an order of magnitude going from N=1000 to
N=50 000 because the JIT-compiled XLA program amortises its per-launch
overhead over the batch. All 50 000 simulations returned finite output for
every prior draw (no integration failures even at the wide +-2x prior).

The slightly higher per-sim cost compared with the previous Pilario-baseline
benchmark (~0.09 ms/sim) is explained by the longer integration window
(`t_final = 200 min` vs `20 min`) needed to reach steady state at the
12.5-minute residence time. Wall-clock is still well within budget for M4
training-set generation.

## Implications for downstream milestones

- **M4 (SBI training).** A 50 000-sim training set takes ~5 seconds on this
  CPU. Adding the closed-loop ODE, process noise, and a 60-min observation
  window in M1/M2 will stretch this 10--30x; even a generous 5-minute upper
  bound per training set on this hardware leaves abundant headroom for the
  M4 sensitivity sweep over `n_simulations in {1k, 5k, 10k, 20k}`.
- **M5 (NUTS MCMC).** With ~0.1 ms per simulator call, a NUTS chain of 500
  samples x 4 chains x ~50 leapfrog steps (approx 100 000 ODE evaluations)
  fits in tens of seconds per observation, making the per-observation MCMC
  baseline tractable without further engineering.
- **GPU.** Not exercised here. Once the closed-loop simulator lands and the
  per-sim cost rises, a Metal/JAX-CUDA backend (or the SINTEF cluster) will
  be revisited.
- **Adiabatic temperature rise** at the new operating point is `Delta_ad =
  (-Hr) Ci / (rho Cp) = 20220 * 0.97 / 1000 = 19.6 K` (vs ~200 K under the
  Pilario set). The energy balance is therefore far less stiff and the
  closed-loop dynamics will be smoother -- a point that will affect SBI
  identifiability of UA and beta and which the publication should discuss
  explicitly.

## Running it again

```bash
cd cstr-model-optimisation/research_project
.venv/bin/python scripts/m0_smoke_test.py
```

The script writes both the human-readable line output above and a JSON
record to `docs/m0_smoke_test_results.json`.
