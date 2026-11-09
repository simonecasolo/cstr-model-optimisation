"""Build and execute notebook 35: S-B wide-prior bank generation (n_train=5000).

Documents and smoke-tests `scripts/generate_sb_wideprior_bank.py` -- the
actual multi-day generation campaign is launched separately, as a background
process, not inside this notebook (this project's established convention:
`jupyter nbconvert` has hung for multi-hour jobs before, see HANDOFF.md /
notebooks/legacy_systemII/24a's own history). This notebook only runs a small,
bounded smoke test (a handful of draws) to prove the recipe is correct and to
get a fresh, honest cost estimate before committing days of compute to it.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    return new_markdown_cell(source, metadata={"language": "markdown"})


def code(source: str):
    return new_code_cell(source, metadata={"language": "python"})


CELLS = [
    md(
        r"""# Notebook 35 — S-B wide-prior bank generation (n_train=5000)

**Goal.** Generate a clean, post-warm-start-fix S-B training bank with a
widened training prior, so that the catastrophic classification failure
found for the narrow-lognormal-prior S-B posterior
(`scripts/eval_sb_armc_classification.py`: macro-F1=0.089) can be properly
fixed rather than only partially fixed, as the alpha-only widening pilot did
(macro-F1=0.211 at `n_train=200` — see
`docs/Manuscript/system_ii_sb_identifiability_story.md` §0.4).

**This notebook does three things:**
1. Evaluates `notebooks/legacy_systemII/44_wu2003_wideprior_multiround_classification.ipynb`'s
   recipe — is it the right starting point?
2. Checks prior-mass coverage against the actual 14-scenario fault catalogue
   — does the chosen prior put real probability mass on every scenario's
   parameter values, not just a box that technically contains them?
3. Smoke-tests `scripts/generate_sb_wideprior_bank.py` (a handful of draws)
   to confirm correctness and get a fresh cost estimate, **before** launching
   the real `n_train=5000` campaign as a long-running background process.
"""
    ),
    md(
        r"""## 1. Does nb44's recipe make sense?

`notebooks/legacy_systemII/44_wu2003_wideprior_multiround_classification.ipynb`
("Option 3: wide-prior round 1 + multi-round refinement round 2") was written
2026-09-06, never executed (no cached output, no bank under its exact
filenames), and targets **S-C.2** (the feed-analyzer structure), not S-B.

**What it gets right, and is worth reusing as-is:**
- Its "Step 1" prior choice — `mixed_wide_prior()`, parameterised by
  `uniform_idx` — is the right generalisation of the alpha-only pilot, not an
  overreach, for however many parameters actually need it. Section 2 below
  checks this quantitatively against the fault catalogue: `beta_r`, `eta_col`,
  and `z_A0_eff` are all *at least as far* from their nominal values at their
  fault scenarios as `alpha` is at its own (max deviation, in the narrow
  lognormal's own log-sigma units: alpha 3.74, beta_r 3.19, z_A0_eff 2.60,
  eta_col 2.23, xi_reb 2.23) — so there is no principled magnitude-based
  reason to widen only a subset. **All five are widened here**
  (`UNIFORM_IDX=(0,1,2,3,4)`, equivalent to `box_uniform_5d()`), which
  nb44's own default (`uniform_idx=(0,1,2,3)`) stopped one short of. This is
  also supported by this project's own evidence, not just a priori reasoning:
  the alpha-only-widened pilot (§0.4 of the identifiability-story doc) still
  misclassified `W5_jacket_mild` and `W6_jacket_severe` — pure jacket-fouling
  scenarios where `alpha` is nominal and only `beta_r` is degraded — because
  `beta_r`'s own prior was never widened.

  **`z_A0_eff` deserves a separate note, because an earlier draft of this
  reasoning got it wrong.** It was first excluded on the claim that its fault
  deviations were "proportionally small" — that does not survive the
  quantitative check (2.60σ, larger than eta_col's or xi_reb's). The real,
  still-valid reason its *classification* value is expected to improve less
  than the other four is mechanistic, not magnitude-based: `z_A0_eff` has a
  confirmed, representation-robust confound with `alpha` and `eta_col`
  (Artefact 3 — checked 5 independent ways, survives the raw-trajectory FIM
  test that collapses genuine representation artefacts). That is an
  information-theoretic limit of S-B's instrumentation (no feed-composition
  sensor) — widening its training prior does not resolve it, so `feed`-class
  classification is expected to stay unreliable regardless of this campaign.
  What widening it still buys: honest, non-shrunk `z_A0_eff` point estimates
  and correct SBC behaviour for it, instead of the same prior-mean-shrinkage
  artifact `alpha` showed before being widened — worth having even though it
  doesn't fix the underlying confound.
- Its bank-generation machinery (checkpointed, resumable, loads-existing-
  partial-and-continues) is exactly the right shape and is reused here
  (`scripts/generate_sb_wideprior_bank.py`), just re-pointed at S-B instead
  of S-C.2 and checkpointed every 50 rows instead of every 250 (this run is
  expensive enough, and unattended enough, that losing 250 rows of a
  multi-day job is a real cost, not a rounding error).

**What is out of scope here, deliberately:**
- nb44's Step 4 ("round-2 multi-round refinement") — retraining a fresh
  per-scenario posterior via `posterior1.set_default_x(x_o)`-conditioned
  resampling — is a separate, more experimental idea stacked on top of the
  wide-prior bank itself. It was never validated (the notebook's own title
  cell frames it as untested: "if round-2 is worse ... that would mean
  per-scenario refinement isn't adding value"). This notebook only builds
  the wide-prior bank (nb44's Step 1 equivalent); multi-round refinement is
  a candidate follow-up, not part of this task.
- `N_TRAIN=1000` with only an `N=120` SBC *screen* (nb44's own choice) is
  lower rigor than this project's established confirmation protocol. This
  notebook targets the full `n_train=5000` requested, plus a proper
  `N=120`-reduced/`N=400`-confirm SBC pair, matching the standard
  `*_stage1bfix` convention used throughout `system_ii_sbi_solidification_plan.md`.
"""
    ),
    md("## 2. Setup"),
    code(
        """import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cstr_sbi.recycle.scenarios import list_closed_loop
from cstr_sbi.recycle.priors import PRIOR_LOW_5D, PRIOR_HIGH_5D, NOMINAL_THETA_5D, LOGN_SIGMA_5D
from cstr_sbi.recycle.physics import PARAM_NAMES

DATA = ROOT / "data"
RESULTS = ROOT / "results"
"""
    ),
    md(
        r"""## 3. Prior-mass coverage check

Two questions, not one: (a) does the **box** (`PRIOR_LOW_5D`/`PRIOR_HIGH_5D`,
the hard support every prior in this project respects) contain every
scenario's parameter values at all — a necessary but very weak condition;
and (b) does the **currently-used narrow lognormal training prior** put
*real, non-negligible probability density* near those values — the condition
that actually determines whether SNPE ever sees training examples anywhere
close to the fault severities it will later be asked to classify. Reports
both, per parameter, across the full 14-scenario catalogue."""
    ),
    code(
        """rows = []
for sc in list_closed_loop():
    th = np.asarray(sc.theta())
    rows.append((sc.name, *th))

theta_arr = np.array([r[1:] for r in rows], dtype=np.float64)
names = [r[0] for r in rows]

df = pd.DataFrame(theta_arr, columns=PARAM_NAMES, index=names)
print("Scenario catalogue (theta):")
print(df.round(3).to_string())

print("\\nPRIOR_LOW_5D :", PRIOR_LOW_5D)
print("PRIOR_HIGH_5D:", PRIOR_HIGH_5D)
print("\\nBox containment check (all scenarios must be strictly inside [low, high]):")
in_box = (theta_arr >= PRIOR_LOW_5D) & (theta_arr <= PRIOR_HIGH_5D)
print(pd.DataFrame(in_box, columns=PARAM_NAMES, index=names).all(axis=0))
"""
    ),
    code(
        """# Narrow-lognormal "effective coverage": how many log-sigma away from the
# lognormal prior's own median is each scenario's value? This is the quantity
# that actually explains the catastrophic classification failure (§0.4 of the
# identifiability-story doc) -- a scenario sitting 2-3+ log-sigma out is
# essentially never sampled during training, regardless of the box being
# technically wide enough to contain it.
print("NOMINAL_THETA_5D (lognormal median):", NOMINAL_THETA_5D)
print("LOGN_SIGMA_5D (lognormal log-sigma):", LOGN_SIGMA_5D)

log_dev = np.abs(np.log(theta_arr) - np.log(NOMINAL_THETA_5D)) / LOGN_SIGMA_5D
log_dev_df = pd.DataFrame(log_dev, columns=PARAM_NAMES, index=names)
print("\\nDeviation from the narrow lognormal's median, in units of its own log-sigma")
print("(>2-3 means 'essentially never sampled under the narrow prior'):")
print(log_dev_df.round(2).to_string())

print("\\nMax deviation reached per parameter across the whole catalogue:")
print(log_dev_df.max(axis=0).round(2))
"""
    ),
    md(
        r"""**Reading this table.** Every parameter whose fault scenarios push it more
than ~2 log-sigma from the narrow lognormal's median is a parameter the
current training prior effectively never samples near its real fault
severities — exactly the mechanism already confirmed empirically for
`alpha` (§0.3/§0.4, `system_ii_sb_identifiability_story.md`). All five
parameters clear that ~2σ bar somewhere in the catalogue (max: alpha 3.74,
beta_r 3.19, z_A0_eff 2.60, eta_col 2.23, xi_reb 2.23) — the direct,
quantitative justification for widening all five, not a subset. `z_A0_eff`'s
inclusion specifically corrects an earlier, weaker draft of this reasoning
that excluded it on an unverified "proportionally small" claim (§1 above has
the full correction and the separate, still-valid mechanistic caveat about
why its *classification* behaviour — as opposed to its point estimates and
SBC — is expected to improve less than the other four)."""
    ),
    md("## 4. Smoke test: `scripts/generate_sb_wideprior_bank.py`"),
    code(
        """from generate_sb_wideprior_bank import generate_bank, prior, UNIFORM_IDX, STRUCTURE, NOISE_PCT, ctrl, y0

print(f"Structure={STRUCTURE}  widened indices={UNIFORM_IDX} -> "
      f"{[PARAM_NAMES[i] for i in UNIFORM_IDX]}  (kept narrow: "
      f"{[PARAM_NAMES[i] for i in range(5) if i not in UNIFORM_IDX]})")

import time
t0 = time.time()
smoke_theta, smoke_summary = generate_bank(
    RESULTS / "35_sb_wideprior_smoketest.npz", n_rows=10, seed=999001, label="smoketest"
)
print(f"\\nSmoke test done in {time.time()-t0:.0f}s for {smoke_theta.shape[0]} accepted rows.")
print("theta sample (first 3 rows):")
print(pd.DataFrame(smoke_theta[:3], columns=PARAM_NAMES).round(3).to_string())
"""
    ),
    md(
        r"""## 5. Cost estimate and how the real campaign is launched

The smoke test above gives a fresh, direct measurement of accepted-draw cost
and accept rate for *this exact* recipe (S-B, all 4 of alpha/beta_r/eta_col/
xi_reb widened) on this machine, right now — more trustworthy than reusing an
older number from a different session. Multiply by the number of accepted
rows actually needed (`N_TRAIN=5000 + N_SBC_REDUCED=120 + N_SBC_CONFIRM=400
= 5520`) for a wall-clock estimate below.

**The real generation is run as a background script, not inside this
notebook** — this project's established convention
(`notebooks/legacy_systemII/24a`'s own history: `jupyter nbconvert` hung for
2+ hours on a comparable job). Checkpointing is every 50 accepted rows
(`scripts/generate_sb_wideprior_bank.py`'s `CHECKPOINT_EVERY`), so the run is
safe to interrupt (process kill, machine sleep, reboot) and resume at any
point — re-running the script picks up exactly where it left off via the
`.partial.npz` checkpoint, losing at most 49 rows of progress."""
    ),
    code(
        """n_total_needed = 5000 + 120 + 400
elapsed_smoke = time.time() - t0
sec_per_row = elapsed_smoke / smoke_theta.shape[0]
print(f"Measured: {sec_per_row:.1f}s/accepted-row from this smoke test.")
print(f"Projected wall-clock for the full campaign ({n_total_needed} accepted rows): "
      f"{sec_per_row * n_total_needed / 3600:.1f}h")
print()
print("To launch the real campaign (run from the repo's research_project/ directory,")
print("in the background, under caffeinate so the machine does not sleep):")
print()
print("  caffeinate -i -s python3 scripts/generate_sb_wideprior_bank.py")
print()
print("Safe to Ctrl-C and re-run at any time -- it resumes from the last checkpoint")
print("(every 50 accepted rows) rather than starting over.")
"""
    ),
]


def main() -> int:
    nb = new_notebook()
    nb.cells = CELLS
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
    )

    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / "35_wu2003_sb_wideprior_bank_generation.ipynb"

    print(f"Executing notebook -> {nb_path}")
    client = NotebookClient(
        nb,
        kernel_name="python3",
        timeout=1800,
        resources={"metadata": {"path": str(repo_root / "notebooks")}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"Wrote {nb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
