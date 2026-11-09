# S-B3 widened-lognormal campaign: what to copy to a new generation machine

This is the file/dependency manifest for running `notebooks/22e_wu2003_data_generation_sb3_widelognormal.ipynb`
(the per-machine training-chunk generator) on a fresh machine. It does **not** need any existing
`data/*.npz` bank — generation starts from scratch and creates its own output directories.

## 1. Files to copy

```
pyproject.toml

src/cstr_sbi/__init__.py
src/cstr_sbi/luyben/__init__.py
src/cstr_sbi/luyben/physics.py
src/cstr_sbi/luyben/priors.py
src/cstr_sbi/luyben/scenarios.py
src/cstr_sbi/luyben/simulator.py
src/cstr_sbi/luyben/summaries.py
src/cstr_sbi/luyben/inference.py
src/cstr_sbi/luyben/ekf.py
src/cstr_sbi/recycle/__init__.py
src/cstr_sbi/recycle/physics.py
src/cstr_sbi/recycle/priors.py
src/cstr_sbi/recycle/scenarios.py
src/cstr_sbi/recycle/simulator.py
src/cstr_sbi/recycle/summaries.py
src/cstr_sbi/recycle/sb3.py

notebooks/22e_wu2003_data_generation_sb3_widelognormal.ipynb
```

**Why the whole `luyben/` subpackage is required even though S-B3 never uses it**: `src/cstr_sbi/__init__.py`
unconditionally does `from cstr_sbi import luyben` at package-import time (line ~24). Any
`import cstr_sbi...` — including the S-B3 notebook's own imports — triggers this, so `luyben/`'s
files must exist and be importable even though nothing in the generation pipeline calls into them.

Everything else in the repo (`legacy_code/`, other notebooks, `docs/`, `figures/`, existing
`data/`/`results/`/`sbi-logs/` contents) is **not** needed to run this notebook. The notebook
creates its own `data/`, `results/`, and `logs/` directories on first run (`.mkdir(exist_ok=True)`).

If the machine will also do the **held-out confirm set** (`notebooks/22f`), **pooling**
(`notebooks/22g`), or **final training/classification** (`notebooks/24d`), copy those specific
notebook files too — they use the same `src/` subset above, nothing additional.

## 2. Python environment

Pin to the exact versions this project was developed and validated against (from the working
`.venv` as of 2026-09-25), not just `pyproject.toml`'s looser floor constraints — `sbi` in
particular has had breaking API changes across minor versions (e.g. `reject_outside_prior` was
removed from `posterior.sample()`'s default path between versions used earlier in this project):

| Package | pyproject.toml constraint | Validated working version |
|---|---|---|
| python | `^3.10` | 3.10.11 |
| numpy | `>=1.26,<2.1` | 2.0.2 |
| scipy | `>=1.11` | 1.15.3 |
| jax | `>=0.4.30,<0.5` | 0.4.38 |
| jaxlib | `>=0.4.30,<0.5` | (matching jax build) |
| diffrax | `>=0.6.0` | 0.7.0 |
| sbi | `>=0.23.0` | **0.26.1 — pin this exactly** |
| torch | `>=2.2` | 2.12.0 |
| pandas | `>=2.1` | 2.3.3 |
| jupyter / ipykernel / nbconvert | (dev group) | needed to run/execute the `.ipynb` |

**Not needed for this specific notebook** (declared in `pyproject.toml` for other parts of the
project, safe to skip on a dedicated generation machine): `matplotlib`, `scikit-learn`, `numpyro`,
`joblib`, `tqdm`, and especially **`OMPython`** (requires a full OpenModelica install — unrelated
to the S-B3/Wu2003 pipeline, only used by the separate Modelica/biodiesel branch of this repo).

Minimal install (once `python 3.10` + the repo files above are in place):

```bash
pip install "numpy==2.0.2" "scipy==1.15.3" "jax==0.4.38" "jaxlib==0.4.38" "diffrax==0.7.0" \
            "sbi==0.26.1" "torch==2.12.0" "pandas==2.3.3" jupyter ipykernel nbconvert
```

(`torch==2.12.0` may need a platform-specific index URL depending on the machine's OS/CUDA — use
whatever this project's other machines already use for `torch`, or the CPU-only wheel if no GPU is
available; this generation workload is CPU-bound ODE solving, not GPU-bound, so CPU-only `torch` is
fine.)

## 3. Per-machine launch

```bash
cd research_project
export SB3_GEN_SEED=<distinct integer per machine, e.g. 0, 1, 2, 3, 4, 5>
caffeinate jupyter nbconvert --to notebook --execute --inplace \
    notebooks/22e_wu2003_data_generation_sb3_widelognormal.ipynb
```

(`caffeinate` is macOS-specific — use `nohup ... &` or `systemd-run`/`tmux` on Linux machines
instead, per whatever this project's other long-running jobs already use on that platform.)

Run the notebook's own smoke-test cell's output first (it prints automatically early in the run) —
if the accept rate has collapsed well below the ~91-95% baseline noted in the notebook, stop and
report back before letting the full multi-hour run continue.

## 4. Collecting results

Copy back only:
- `data/wu2003_sbi_train_sb3_widelognormal_seed{SEED}_n{N}.npz` (the final bank; the `.partial.npz`
  checkpoint sibling is only needed if the run was interrupted and needs resuming on that same
  machine)
- `logs/notebook22e_progress_seed{SEED}.log` (optional, for debugging)

into one shared `data/` folder before running `notebooks/22g`.
