# System III — a CSTR designed for diagnosability (main system of the paper)

The plant is Fogler's industrial propylene-glycol CSTR (Example 9-4, 3rd ed.; COMSOL "Startup of a
CSTR"), designed for diagnosability. Every decision is checked against a scorecard fixed before
computing. System I (`../notebooks/system_I`) is the marginal-design contrast.

## Layout

| Path | Content |
|---|---|
| `docs/system_III_design.md` | the design: scorecard results, information, proxies, revised Claim 1, decisions |
| `docs/SCORECARD.md` | the design criteria, fixed before any candidate was computed |
| `docs/system_III_plan.md` | storyline, claims ledger, notebook map, run order, pre-registration checklist |
| `docs/decision_log.md` | every decision with its rationale and whether the user confirmed it |
| `design/` | the numpy design toolkit and its stored results (reproduces the scorecard) |
| `src/system_iii/` | the package: `physics`, `simulator` (numpy EM, digital PI on the measured T, protocols P0–P2, offsets, drift, open-loop and exogenous-flow modes), `sensors`, `features` (29 summaries + physics proxies), `scenarios`, `data` (window generation), `information` (Bayesian CRB), `rules` (System I decision rules), `inference` (NPE, ensembles, evaluation; needs torch + sbi), `metrics` (System I metrics), `config` |
| `tests/` | `python -m pytest system_III/tests -q` (numpy only, ~10 s) |
| `notebooks/` | 20 notebooks (00–19) mirroring System I, one per claim; `build_notebooks.py` regenerates them |
| `data/`, `results/`, `figures/` | outputs (empty until run) |

## Status (2026-11-10)

- **Run so far:** nothing in `notebooks/`; the outputs (`data/`, `results/`, `figures/`) are empty.
- **Dry runs on a copy:** notebooks 00, 01, 02, 04 and the numpy part of 03 ran end to end (04 with
  reduced sample sizes).
- **Not yet run:** the notebooks that train posteriors (03 §3, 05–08, 10–12, 15). They need the
  project `.venv` (torch, sbi).
- **Specified but not implemented:** notebooks 09 (EKF/UKF baselines), 13 (L-C2ST) and 14 (30-day
  tracking) stop with a clear message.
- **Rules to fix before running:** every notebook header lists its pre-registration items; fix and
  date them first.

## Run order

See `docs/system_III_plan.md` §4. Each notebook runs from `system_III/notebooks` (the first cell asserts it).
