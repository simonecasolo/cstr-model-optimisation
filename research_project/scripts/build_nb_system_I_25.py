"""Build and execute notebook 25 (System I): verification of a claimed 60x unit
error in K0_NOMINAL (src/cstr_sbi/physics.py).

**Context.** An external literature-review document, pasted into chat (not in this
repo), raised a specific checkable claim about this codebase's propylene-oxide
hydrolysis kinetics: ``K0_NOMINAL = 16.96e12`` is commented as ``1/min`` but COMSOL's
own Fogler-derived reproduction of this exact textbook example states a
pre-exponential factor of ``A1 = 4.71e9 s^-1``, which converts to
``4.71e9 * 3600 = 1.696e13 hr^-1`` -- numerically identical to this codebase's
constant, but in **hr^-1**, not **min^-1**. If true, K0_NOMINAL as coded is 60x too
large, which would push the reactor to ~98% conversion (tiny alpha-sensitivity)
instead of Fogler's intended ~47% (much larger alpha-sensitivity) -- a candidate
independent explanation for why every representation in this project's long-running
(alpha,beta) banana investigation has struggled to pin down alpha.

**This notebook's job**: verify the claim against primary/independent secondary
sources (not just trust the one COMSOL citation), cross-check with the codebase's
own E_A constant via the Arrhenius equation, and if confirmed, quantify the before/
after impact on conversion and alpha-sensitivity with a cheap simulation check using
the project's own ``physics.py`` functions. This is a verification-and-impact-
assessment task, not a new experimental campaign -- no SBI training, no NUTS, no
multi-hour runs. Does NOT touch main.tex/SI/article_outline_CChE.md, and does NOT
touch any nb19-24 files (nb24's CNN investigation runs concurrently in the
background in this same session).
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
        r"""# Notebook 25 (System I) -- K0_NOMINAL units verification

**Task.** An external literature-review document claimed that `K0_NOMINAL` in
`src/cstr_sbi/physics.py` (`16.96e12`, commented `1/min`) may actually be a value
quoted in **hr^-1** in its original source (Fogler's propylene-oxide hydrolysis CSTR
example), making the coded constant 60x too large. This would push the reactor's
nominal operating point to ~98% conversion (where alpha-sensitivity is tiny) instead
of Fogler's intended ~47% (where it is much larger) -- a candidate explanation,
independent of any SBI architecture/training issue, for this project's long struggle
to extract alpha information (see `finding_system1_no_alpha_beta_banana.md`'s nb24
entry: the CNN posterior for alpha has "essentially collapsed", slope~0.05).

**Method**: (1) read the code's own citation comment and E_A for an internal
cross-check candidate; (2) independently verify the claimed pre-exponential factor
and its units against 2+ secondary/primary sources, not just the one COMSOL page the
literature document cited; (3) check internal consistency via the Arrhenius equation
using the codebase's own E_A; (4) if confirmed, quantify the corrected operating
point and alpha-sensitivity using this project's own `physics.py` functions (cheap,
no training). Verdict given explicitly at the end."""
    ),
    md("## 1. The code as it stands"),
    code(
        """import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import inspect
from src.cstr_sbi import physics as p

print("K0_NOMINAL =", p.K0_NOMINAL, " (coded comment: '1/min')")
print("E_A        =", p.E_A, " J/mol (coded comment: 'Fogler / Furusawa 1969')")
print("R_GAS      =", p.R_GAS, "J/(K mol)")
print("TSP        =", p.TSP, "K  (closed-loop setpoint)")
print("tau = V/Q  =", p.V / p.Q, "min")
print()
print("Source comment on line 37 of physics.py:")
src_lines = inspect.getsource(p).splitlines()
for ln in src_lines:
    if "K0_NOMINAL" in ln or ("E_A" in ln and "=" in ln and "J/mol" in ln):
        print(" ", ln)
"""
    ),
    md(
        r"""The in-code comment attributes both `K0_NOMINAL` and `E_A` to "Fogler /
Furusawa 1969" -- i.e. the codebase's own authors already knew this constant traces
back to the Furusawa et al. (1969) kinetic study of propylene oxide hydrolysis that
Fogler's textbook cites, not an independently-chosen value. This is useful: it means
the *numeric value* 16.96e12 is very likely sourced correctly (it is the literature
number), and the only thing in question is whether the **unit label attached to it in
this file** (`1/min`) is correct, or should be `1/hr`."""
    ),
    md("## 2. Independent source verification (not just the COMSOL citation)"),
    md(
        r"""Three separately-phrased web queries were run and cross-checked against
each other (full transcripts kept in this session's tool log; summarised here):

**Source 1 -- COMSOL's own "Startup of a Continuous Stirred Tank Reactor" model
documentation** (`doc.comsol.com/6.4/doc/.../cstr_startup.html`), fetched directly:

> Pre-exponential factor: **A1 = 4.71e9 s^-1**
> Activation energy: **E1 = 75.358 kJ/mol**
> Reference: *"H.S. Fogler, Elements of Chemical Reaction Engineering, 3rd ed.,
> Prentice Hall PTR, **Example 9-4**, pp. 553-559, 1999."*

This is the exact worked example this codebase's model structure matches (jacket
cooling, startup dynamics, PO + water -> propylene glycol), not a different, simpler
Fogler example. Converting A1 to hourly units: `4.71e9 s^-1 * 3600 s/hr = 1.696e13
hr^-1` -- i.e. `16.96e12 hr^-1`, numerically identical to this codebase's
`K0_NOMINAL`.

**Source 2 -- an independent secondary citation of the same Furusawa-sourced value**,
surfaced by a separately-worded search (not the COMSOL page), explicitly in the units
this codebase is attributed to:

> "...under conditions similar to those in the example, the reaction is first-order
> in propylene oxide concentration and apparent zero-order in excess of water with
> the preexponential factor **A = 16.96E12 hr^-1** and activation energy **E =
> 75362.4 J/mol**." (attributed to Furusawa et al.)

This states the units explicitly as **hr^-1**, and its activation energy
(75,362.4 J/mol) matches this codebase's `E_A = 75362.0 J/mol` to 6 significant
figures -- effectively an exact match, not a rounding coincidence. This is strong
independent confirmation that this source and this codebase's constants are the
*same* literature values, and that source states the pre-exponential factor's units
as hr^-1.

**Source 3 -- a third, independently-phrased hit** for a different downstream
application of the same kinetics:

> "...the kinetic data is given as **A = 16.96 x 10^12 hr^-1** and **E = 75,200
> kJ/kmol**..." (= 75.2 kJ/mol, consistent with the ~75.3-75.4 kJ/mol value within
> normal secondary-source rounding)

Three independently-phrased queries, three independent pages, zero disagreement:
every source that states explicit units for the pre-exponential factor **16.96e12**
gives **hr^-1**, never min^-1. No source found anywhere in this search gives a
min^-1 value matching this constant. Confidence: **high** that the correct unit for
the literature value numerically equal to `K0_NOMINAL` is per hour, not per minute."""
    ),
    md("## 3. Internal consistency check (independent of any citation)"),
    md(
        r"""If `K0_NOMINAL` were genuinely `16.96e12 min^-1` (as currently commented),
its hourly equivalent would be `16.96e12 * 60 = 1.018e15 hr^-1` -- a pre-exponential
factor with no match in any of the three independently-sourced values above (all of
which cluster tightly around `1.696e13 hr^-1`, not `1.018e15 hr^-1`). Conversely, if
the *coded* value is meant to already be the correct per-minute number, it is a
curious coincidence that it would then equal the hourly literature number exactly
scaled by 60 -- i.e. the "coincidence" required for the current code to be right is
that an entirely unrelated per-minute measurement happens to equal 60x the
well-attested per-hour literature value, to 4 significant figures. That is not a
plausible coincidence; the far simpler explanation is a 60x unit slip (hr -> min
conversion omitted or mis-applied) when this constant was transcribed into the code.

A genuinely independent check -- using only the codebase's own `E_A` and the
Arrhenius equation, with no external citation at all -- is run below: compute the
reactor's Damkohler number and conversion at the setpoint temperature under both the
current (as-coded) value and the candidate-corrected (/60) value, and see which one
is consistent with a "well-designed CSTR teaching example" (moderate conversion,
non-trivial sensitivity to kinetics) vs. an operating point pinned at the asymptote
of full conversion."""
    ),
    code(
        """import math

Ea = p.E_A
R = p.R_GAS
T = p.TSP  # 312.5 K, the closed-loop setpoint -- also very close to the open-loop ss T

k0_current = p.K0_NOMINAL          # as literally used by the code today (treated as 1/min)
k0_corrected = p.K0_NOMINAL / 60.0  # candidate fix if the true value is 1/hr

exp_term = math.exp(-Ea / (R * T))
k_cur = k0_current * exp_term
k_cor = k0_corrected * exp_term

tau = p.V / p.Q  # residence time, min

Da_cur = k_cur * tau
Da_cor = k_cor * tau

X_cur = Da_cur / (1 + Da_cur)   # well-mixed 1st-order CSTR: X = Da/(1+Da)
X_cor = Da_cor / (1 + Da_cor)

sens_cur = Da_cur / (1 + Da_cur) ** 2   # dX/d(ln alpha) = Da/(1+Da)^2 since alpha multiplies k0 -> Da
sens_cor = Da_cor / (1 + Da_cor) ** 2

print(f"T = {T} K, tau = {tau} min, exp(-Ea/RT) = {exp_term:.6e}")
print()
print(f"{'':28s}{'current (as coded)':>22s}{'corrected (/60)':>22s}")
print(f"{'k (1/min)':28s}{k_cur:>22.5f}{k_cor:>22.5f}")
print(f"{'Da = k*tau':28s}{Da_cur:>22.4f}{Da_cor:>22.4f}")
print(f"{'X (analytic CSTR formula)':28s}{X_cur:>22.4f}{X_cor:>22.4f}")
print(f"{'dX/dln(alpha) (analytic)':28s}{sens_cur:>22.5f}{sens_cor:>22.5f}")
print()
print(f"Sensitivity ratio corrected/current: {sens_cor/sens_cur:.2f}x")
"""
    ),
    md(
        r"""The current (as-coded) value gives **X = 98.2%** -- matching, almost
exactly, this project's own `main.tex` (`sec:identifiability` context), which states
*"corresponding to a single-pass conversion of approximately 98%"* at this nominal
operating point. This is not independent evidence about which unit is correct (both
numbers come from the same codebase), but it does confirm that the claimed ~98%
operating point is real and currently baked into the manuscript's own description of
the system -- not a hypothetical. The candidate-corrected value gives **X = 47.2%**,
matching the external literature document's "Fogler's intended ~47% conversion"
claim almost exactly. Both numbers landing within a fraction of a percent of the two
external claims, from a totally independent closed-form calculation using only this
project's own constants, is a strong internal-consistency corroboration."""
    ),
    md("## 4. Cross-check with the actual ODE model (open-loop and closed-loop)"),
    md(
        r"""The hand calculation above uses the simple analytic 1st-order CSTR formula
`X = Da/(1+Da)` evaluated at `T = Tsp`. This section repeats the check using this
project's own `physics.py` integrators directly -- first the open-loop steady state
(no controller, fixed coolant flow), then the actual closed-loop fixed point via
`closed_loop_steady_state_exact` (root-finding, the function nb22's warm-start audit
verified is the single source of truth for this project's closed-loop steady state).
No retraining, no SBI, no NUTS -- just evaluating the existing deterministic model at
two values of `k0`."""
    ),
    code(
        """import jax.numpy as jnp

Ci = p.CI_NOMINAL
inlet_ol = p.NOMINAL_INLET
inlet_cl = p.NOMINAL_INLET_CL
ctrl = p.NOMINAL_CTRL

print("--- Open-loop steady state (fixed Qc=80 L/min, no controller) ---")
for label, k0 in [("current (as coded)", p.K0_NOMINAL), ("corrected (/60)", p.K0_NOMINAL / 60.0)]:
    params = jnp.array([p.UA_NOMINAL, k0])
    C, T_ss, Tc_ss = p.simulate_open_loop_to_steady_state(params, inlet_ol, t_final=2000.0)
    X = 1 - float(C) / Ci
    print(f"  {label:22s} C={float(C):.5f}  T={float(T_ss):.3f} K  Tc={float(Tc_ss):.3f} K  X={X:.4f}")

print()
print("--- Closed-loop steady state (PI controller active, alpha=beta=1) ---")
for label, k0 in [("current (as coded)", p.K0_NOMINAL), ("corrected (/60)", p.K0_NOMINAL / 60.0)]:
    params = jnp.array([p.UA_NOMINAL, k0, 1.0, 1.0])
    C, T_ss, Tc_ss, I_ss = p.closed_loop_steady_state_exact(params, inlet_cl, ctrl)
    X = 1 - float(C) / Ci
    print(f"  {label:22s} C={float(C):.5f}  T={float(T_ss):.3f} K  Tc={float(Tc_ss):.3f} K  I={float(I_ss):.1f}  X={X:.4f}")
"""
    ),
    md(
        r"""**Important downstream subtlety, found here and not in the simple hand
calculation**: under the full closed-loop model, the corrected `k0` does **not**
reach `T = Tsp = 312.5 K` at all -- the reactor settles at `T ~ 301.7 K` with the
coolant valve **fully closed** (`Qc = Qc_min = 0`), because the lower heat release at
the corrected (lower) reaction rate is insufficient to warm the reactor up to the
setpoint even with zero cooling. This is a *saturated* branch of
`closed_loop_steady_state_exact`, not the simple `T=Tsp` branch the hand calculation
implicitly assumed. In other words: naively dividing `K0_NOMINAL` by 60 in isolation,
with every other constant (`Tsp=312.5`, `Qc0=80`, `Ci=0.97`, ...) left exactly as
they are today, drives the closed-loop system into permanent actuator saturation at
the *healthy* operating point -- not a physically sensible "fixed" nominal regime.
This means the PI-controller tuning (`Tsp`, `Kp`, `Qc0`, ...) in this codebase was
almost certainly tuned/chosen to work with the as-coded (and apparently 60x-too-high)
`k0`, not against Fogler's original per-hour value. Correcting the unit bug properly
is therefore not a one-line fix -- it requires re-deriving a consistent nominal
operating point (setpoint and/or feed conditions) alongside the kinetic constant, which
is out of scope for this verification-only notebook."""
    ),
    code(
        """import numpy as np

def X_closed_loop(k0, alpha):
    params = jnp.array([p.UA_NOMINAL, k0, alpha, 1.0])
    C, *_ = p.closed_loop_steady_state_exact(params, inlet_cl, ctrl)
    return 1 - float(C) / Ci

da = 0.01
print(f"{'':28s}{'current':>14s}{'corrected (/60)':>18s}")
for label, k0 in [("current", p.K0_NOMINAL), ("corrected (/60)", p.K0_NOMINAL / 60.0)]:
    Xp = X_closed_loop(k0, np.exp(da))
    Xm = X_closed_loop(k0, np.exp(-da))
    dXdlna = (Xp - Xm) / (2 * da)
    print(f"dX/dln(alpha) at alpha=1, {label:14s} {dXdlna:14.4f}")
"""
    ),
    md(
        r"""Even accounting for the corrected operating point landing in the
saturated (`Qc_min`) branch rather than the simple `T=Tsp` branch, the alpha-
sensitivity of the measured concentration channel is **~14-16x larger at the
corrected `k0`** than at the current, as-coded value -- fully consistent with the
simple analytic estimate in Section 3 (`13.9x`) and with the external literature
document's `~14x` claim."""
    ),
    md("## 5. Verdict"),
    md(
        r"""**CONFIRMED BUG, with a caveat about what "fixing" it actually requires.**

- **The unit error is real.** `K0_NOMINAL = 16.96e12` is correctly attributed in the
  code's own comment to Fogler/Furusawa(1969), and the *number itself* is right --
  but three independently-phrased, independently-sourced citations of that exact
  literature value (COMSOL's documented reproduction of Fogler 3rd ed. Example 9-4;
  a second, separately-worded citation of the Furusawa data with an activation
  energy matching this codebase's `E_A` to 6 significant figures; a third,
  unrelated downstream application) all state its units as **hr^-1**, never
  min^-1. No source anywhere in this search states 16.96e12 as a per-minute value.
  The comment on physics.py line 37 (`# 1/min`) is wrong; it should read `# 1/hr`,
  and the value actually used in the ODE (where `k = k0 * exp(-Ea/RT)` multiplies a
  rate expressed per simulation-minute) needs to be `16.96e12 / 60 ~= 2.827e11`
  per minute, not `16.96e12` per minute.
- **Internal consistency corroborates this independently of the citations**: using
  only this codebase's own `E_A` and the Arrhenius equation, the as-coded value
  gives X~98.2% (matching `main.tex`'s own stated operating point almost exactly)
  and the corrected value gives X~47.2% (matching the external document's claim
  about Fogler's intended operating point almost exactly) -- two separate
  external claims, both independently reproduced from this project's own numbers.
- **Quantified impact**: the corrected operating point has **~14-16x larger**
  local alpha-sensitivity (`dX/dln(alpha)`) than the current one, both by the
  simple analytic CSTR formula and by the project's actual closed-loop
  steady-state solver.
- **Important caveat, found only by running the actual closed-loop model (Section
  4), not by the external document or the simple hand calculation**: simply
  dividing `K0_NOMINAL` by 60 in isolation drives the closed-loop healthy operating
  point into permanent coolant-valve saturation (`Qc=Qc_min=0`, `T` settling at
  ~301.7 K instead of the `Tsp=312.5 K` setpoint) -- the PI-controller tuning in
  this codebase appears to have been chosen to match the (apparently buggy)
  as-coded `k0`, not Fogler's actual value. A correct fix is therefore **not a
  one-line constant change**; it requires re-deriving a consistent nominal
  operating point (setpoint, feed conditions, and/or controller tuning) alongside
  the kinetic constant -- explicitly out of scope for this verification-only
  notebook, and flagged as the next concrete step if this is acted on.
- **Confidence level**: HIGH that the unit label is wrong (3-for-3 independent
  sources, exact `E_A` digit match, two independent external conversion claims
  both reproduced from this codebase's own numbers to within a fraction of a
  percent). MEDIUM on the correct remediation path, since fixing the label alone
  creates a new, different problem (actuator saturation) that would need its own
  resolution before any retraining.

**Downstream relevance to this project's (alpha,beta) investigation** (not
re-litigated here, flagged as a hypothesis for a future session): this project's
long-running CNN alpha-information-deficit finding (`notebooks/system_I/24_...`,
summarised in `finding_system1_no_alpha_beta_banana.md`'s newest entry: CNN
posterior-mean slope on alpha ~0.05-0.053 vs hand-crafted's ~0.99 at the same
training budget) was diagnosed purely at the representation level (CNN embedding
vs. hand-crafted summary). This notebook's finding offers an independent, purely
physics-level contributing factor that was never checked in that investigation: if
the nominal operating point is pinned at ~98% conversion (the asymptotic flat part
of the Da/(1+Da) curve), *any* representation -- hand-crafted or learned -- has
~14-16x less genuine alpha signal in the trajectory to work with than it would at
Fogler's intended ~47% operating point. This does not contradict the nb24 finding
(the hand-crafted summary evidently still extracts *some* alpha signal even at the
current operating point, since its posterior-mean slope is ~0.99, not ~0; the CNN's
near-total failure to do the same at the identical operating point is still a real,
separate representation-level deficiency) -- but it is a plausible *additional*
headwind that was not previously on this project's radar, and would predict that
re-tuning the operating point should improve absolute alpha identifiability for
*every* representation, not just explain away the CNN-specific gap. **Not tested
here** -- this would require the operating-point re-derivation flagged above plus a
fresh SBI retrain, out of scope for this notebook.

**Relevance to `main.tex`'s existing ~236x FIM ratio
(`eq:fim_ratio`, Section~`sec:identifiability`)**: that ratio measures
`I_alpha,alpha / I_beta,beta` -- a relative comparison between the two parameters at
the *same*, currently-buggy operating point, not an absolute measure of how much
alpha information is actually available. This notebook's finding does not contradict
or invalidate that ratio (alpha's sensitivity could still dominate beta's by a
similar relative margin at the corrected operating point -- not tested here, since it
would require redoing the FIM calculation under a re-derived operating point). What
it does suggest is that the *absolute* magnitude of alpha's local sensitivity may be
closer to the practical limit of what is recoverable from a finite, noisy, amortised
estimator than the existing 236x relative-ratio framing on its own implies -- a
reason for caution before citing that ratio as evidence that alpha is "easy" to
identify in an absolute sense, separate from whether it dominates beta in a relative
sense. This is a hypothesis for a follow-up, not a correction to the existing
manuscript number.

**Explicitly NOT done in this notebook (by design, per scope)**: no edits to
`main.tex`/the SI/`article_outline_CChE.md`; no edits to `K0_NOMINAL` or any other
constant in `src/cstr_sbi/physics.py` (verification only -- changing a live constant
used by every other System I notebook, several of them with results currently cited
in the manuscript, needs an explicit decision by the user, not a unilateral
verification-notebook edit); no SBI retraining; no touching of nb19-24's own files."""
    ),
]


def build_and_execute() -> int:
    nb = new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }

    out_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "system_I"
        / "25_k0_units_verification.ipynb"
    )

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    print(f"Executing notebook -> {out_path}")
    client.execute()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    print(f"Wrote {out_path}")
    return 0


def main() -> int:
    return build_and_execute()


if __name__ == "__main__":
    raise SystemExit(main())
