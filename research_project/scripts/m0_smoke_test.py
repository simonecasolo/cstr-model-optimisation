"""M0 smoke test.

Goals (from cstr_sbi_execution_plan.md, M0):

1. Reproduce the nominal steady state of the Fogler-grounded parameter set
   (cstr_parameters_recommended.md) with the JAX/diffrax open-loop
   simulator within ``rtol=1e-3`` of a high-tolerance scipy reference.

       inlet  = (Ci=0.97, Ti=297.0, Tci=297.0, Qc=80.0)
       params = (UA=1.25e4 cal/min/K, k0=16.96e12 1/min)
       --> [C, T, Tc] approx [0.0184, 312.15, 299.05]

2. Time a ``jit`` + ``vmap`` batched simulation over N in {1000, 10000, 50000}
   parameter draws around the nominal point. Numbers feed into
   ``research_project/docs/m0_baseline_benchmarks.md``.

Run from repo root after a minimal install (``pip install jax diffrax numpy``):

    python research_project/scripts/m0_smoke_test.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cstr_sbi.physics import (
    NOMINAL_INLET,
    NOMINAL_PARAMS,
    NOMINAL_Y0,
    simulate_open_loop_batch,
    simulate_open_loop_to_steady_state,
)


# Nominal operating point from cstr_parameters_recommended.md.
INLET = NOMINAL_INLET
PARAMS = NOMINAL_PARAMS

# Approximate steady state quoted in the parameters doc (Section 4):
# [C, T, Tc] approx [0.021, 312.5, 302]. The high-tolerance scipy reference
# below gives the actual physical truth, which we use as the acceptance gate.
DOC_QUOTED = np.array([0.021, 312.5, 302.0])


def _scipy_reference() -> np.ndarray:
    """High-tolerance scipy integration of the same ODE -- ground truth."""
    from scipy.integrate import solve_ivp

    from cstr_sbi.physics import (
        C_P,
        C_PC,
        E_A,
        H_R,
        Q,
        R_GAS,
        RHO,
        RHO_C,
        V,
        V_C,
    )

    def rhs(t, y, params, inlet):
        Ci, Ti, Tci, Qc = inlet
        UA, k0 = params
        C, T, Tc = y
        k = k0 * np.exp(-E_A / (R_GAS * T))
        return [
            (Q / V) * (Ci - C) - k * C,
            (Q / V) * (Ti - T) - H_R * k * C / (RHO * C_P) - UA * (T - Tc) / (RHO * C_P * V),
            (Qc / V_C) * (Tci - Tc) + UA * (T - Tc) / (RHO_C * C_PC * V_C),
        ]

    sol = solve_ivp(
        rhs,
        (0.0, 1000.0),
        np.asarray(NOMINAL_Y0),
        method="Radau",
        args=(np.asarray(PARAMS), np.asarray(INLET)),
        rtol=1e-9,
        atol=1e-12,
    )
    return sol.y[:, -1]


def steady_state_check() -> dict:
    """Goal 1: open-loop steady state matches the scipy reference."""
    y_jax = np.asarray(simulate_open_loop_to_steady_state(PARAMS, INLET))
    y_scipy = _scipy_reference()
    rel_err_vs_scipy = np.abs(y_jax - y_scipy) / np.abs(y_scipy)
    rel_err_vs_doc = np.abs(y_jax - DOC_QUOTED) / np.abs(DOC_QUOTED)
    print(f"  JAX/diffrax steady state    = {y_jax}")
    print(f"  scipy reference (Radau)     = {y_scipy}")
    print(f"  doc quoted (parameters md)  = {DOC_QUOTED}")
    print(f"  rel.err vs scipy reference  = {rel_err_vs_scipy}")
    print(f"  rel.err vs doc quoted       = {rel_err_vs_doc}")
    return {
        "y_jax": y_jax.tolist(),
        "y_scipy_reference": y_scipy.tolist(),
        "doc_quoted": DOC_QUOTED.tolist(),
        "rel_err_vs_scipy": rel_err_vs_scipy.tolist(),
        "rel_err_vs_doc": rel_err_vs_doc.tolist(),
        "max_rel_err_vs_scipy": float(rel_err_vs_scipy.max()),
        "max_rel_err_vs_doc": float(rel_err_vs_doc.max()),
        "passes_rtol_1e-3_vs_scipy": bool(rel_err_vs_scipy.max() < 1e-3),
        "passes_rtol_1e-5_vs_scipy": bool(rel_err_vs_scipy.max() < 1e-5),
    }


def benchmark(n_samples: int, *, n_warmup: int = 1) -> float:
    """Goal 2: time a single batched (jit+vmap) call over ``n_samples`` params.

    Prior support: UA in [0.5, 2.0] x UA_NOMINAL, k0 in [0.5, 2.0] x K0_NOMINAL.
    These are placeholder bounds for the M0 benchmark only; the M4 prior in
    spec Section 3.1 will be tighter and includes the alpha and beta axes.
    """
    from cstr_sbi.physics import K0_NOMINAL, UA_NOMINAL

    rng = np.random.default_rng(0)
    UAs = rng.uniform(0.5 * UA_NOMINAL, 2.0 * UA_NOMINAL, size=n_samples)
    k0s = rng.uniform(0.5 * K0_NOMINAL, 2.0 * K0_NOMINAL, size=n_samples)
    params_batch = jnp.stack([jnp.asarray(UAs), jnp.asarray(k0s)], axis=1)

    for _ in range(n_warmup):
        y = simulate_open_loop_batch(params_batch[:8], INLET)
        y.block_until_ready()

    t0 = time.perf_counter()
    y = simulate_open_loop_batch(params_batch, INLET)
    y.block_until_ready()
    elapsed = time.perf_counter() - t0

    n_finite = int(np.isfinite(np.asarray(y)).all(axis=1).sum())
    print(
        f"  N = {n_samples:>6d}: {elapsed:7.3f} s  "
        f"({1e3 * elapsed / n_samples:6.3f} ms / sim, "
        f"{n_finite}/{n_samples} finite)"
    )
    return elapsed


def main() -> int:
    print("JAX devices:", jax.devices())
    print("\n[1/2] Steady-state check (Fogler-grounded nominal point)")
    ss = steady_state_check()

    print("\n[2/2] vmap+jit benchmark (open-loop, integrate to t_final=200 min)")
    sizes = [1_000, 10_000, 50_000]
    timings = {n: benchmark(n) for n in sizes}

    out_path = Path(__file__).resolve().parent.parent / "docs" / "m0_smoke_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "steady_state": ss,
                "benchmark_seconds": {str(k): v for k, v in timings.items()},
                "jax_devices": [str(d) for d in jax.devices()],
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")

    ok = ss["passes_rtol_1e-3_vs_scipy"]
    print(
        f"\nM0 acceptance (max rel.err vs scipy reference < 1e-3): "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
