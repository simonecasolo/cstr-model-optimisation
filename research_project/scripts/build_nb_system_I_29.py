"""Build (and optionally execute) notebooks/system_I/29_warm_start_protocol_study.ipynb.

The notebook IS the experiment: every stage of the study runs in a notebook cell, with the
kernel's working directory set to notebooks/system_I. The stage functions live in
scripts/nb29_ws_study.py (one tested implementation, imported by the notebook); each
stage skips work already on disk, so re-executing the notebook after an interruption
resumes where it stopped.

Usage:  python scripts/build_nb_system_I_29.py [--sensor N3] [--execute]
  --sensor N0 (default) rebuilds the original notebook; --sensor N3 writes the separate
  29_warm_start_protocol_study_N3.ipynb (production sensor model since 2026-11-09).
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "29_warm_start_protocol_study.ipynb"

CRITERION = r"""# 29 — Warm-start (initial-state) protocol study, System I (fixed controller law)

## Selection criterion — fixed before any result was computed

The initial state of each simulated window is part of the generative model: it encodes
an assumption about how the fault evolves relative to the 60-min window. Choosing the
protocol that looks best on data generated with that same protocol is circular.

**"Optimal" = matches the situation the monitor faces, and degrades gracefully under
mismatch.** The realistic deployment case is an *incipient* fault (test column **P2**:
θ drifts at the 30-day degradation rates, noise on throughout). Rule (implemented in
`scripts/nb29_ws_study.py::stage_select`):

1. **Eligible** = on the P2 column, seed-median marginal SBC KS p ≥ 0.01 for α and β
   **and** seed-median TARP KS p ≥ 0.01 (joint calibration, sensitive to a wrong α–β correlation).
2. Among eligible: lowest seed-mean normalised posterior-mean RMSE on P2; protocols within
   one combined standard error are tied.
3. Tie-break 1: smallest worst-case TARP deviation over the other incipient-fault columns
   (P1, P1b, P3). The abrupt-fault column P0 is a different regime and is reported only.
4. Tie-break 2: simplicity / cost, P1 < P1b < P3 < P2 < P0.
5. If nothing is eligible: lowest P2 TARP deviation, **flagged**; nb31 does not start.

| Protocol | Initial state | Assumption |
|---|---|---|
| P0 | healthy steady state for every θ | abrupt fault at window start |
| P1 | deterministic steady state at θ | incipient fault, noise-free start |
| P1b | P1 + 150-min noisy burn-in | incipient fault, stationary start |
| P2 | θ ramps over a 300-min noisy history at 30-day rates | incipient fault, realistic |
| P3 | steady state at θ′ = θ + N(0, 0.03²) | initial state uncertain (nuisance) |

**Running this notebook.** Execute top to bottom from `notebooks/system_I`. Every stage
skips work already saved in `results/29fx/` and `results/29fx_*.csv`, so after an
interruption simply re-run. Progress sentinels: `results/29fx_progress_*.txt`.
"""

CELLS = [
    ("md", CRITERION),
    ("code", """import json, os, sys, time, subprocess
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", \\
    f"execute this notebook from notebooks/system_I (cwd is {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import nb29_ws_study as S
R, FIGS = ROOT / "results", ROOT / "figures"
P = S.PROTOCOLS

# ---- run switches -----------------------------------------------------------------
PARALLEL_TRAINING = False   # True: train protocol groups in parallel subprocesses (cwd = this folder)
RUN_E4 = False              # optional window-length check (~1-2 h)
THREADS = 4
T0 = time.time()

def load(name):
    f = R / name
    return pd.read_csv(f) if f.exists() else None"""),

    ("md", """## E1 — time scales (deterministic)
Closed-loop eigen time constants, burn-in to stationarity from the exact Lyapunov
solution (no noisy ensemble estimate), settling after an abrupt fault (noise-free,
tolerances far above float32 precision), and the P2 plant lag at realistic and 10× rates.
Replaces the superseded old-law `29_e1_*` files.

**Check:** slowest τ 10–26 min at unsaturated points; `t95_stationary_min` below the
150-min P1b burn-in; finite settling; `lag_Qc_x1` ≲ 3 L/min."""),
    ("code", """if not (R / "29fx_e1_timescales.csv").exists():
    S.stage_e1()
e1 = load("29fx_e1_timescales.csv"); display(e1.round(4))
assert e1.loc[~e1.saturated_ss, "t95_stationary_min"].max() < 150, \\
    "P1b burn-in (150 min) shorter than the time to stationarity: raise burn_min in ws_protocols before E3"
"""),
    ("md", """**Reading E1 (2026-11-08 run).** The eigen time constants (10–26 min) and the exact
Lyapunov burn-in (`t95_stationary_min` 14–56 min, all below the 150-min P1b burn-in) are
the reliable E1 numbers. The two noise-free columns are **not**: `t_settle_abrupt_min = inf`
at Sc4/Sc7 and the `lag_Qc_*` offsets (−3 to −5 L/min, not proportional to the rate
multiplier) come from float32 stalling of the integrator in a run with *no* noise (state
increments below one ulp of T ≈ 312 K). The stochastic simulator is unaffected (each
noise increment, ~0.01 K, is far above the ulp). Do not cite those two columns; if needed,
recompute them in a float64 subprocess."""),

    ("md", """## E2 — where the information is (Fisher information per protocol and representation)
Gaussian FIM with common random numbers. `corr_implied` is the correlation of the inverse
FIM (the local posterior correlation). Representations: 29-D summaries with full vs diagonal
covariance, the raw 120×4 trajectory (Ledoit–Wolf covariance), and the four channel means
only (the "level" information)."""),
    ("code", """if not (R / "29fx_e2_fim.csv").exists():
    S.stage_e2()
e2 = load("29fx_e2_fim.csv")
display(e2.pivot_table(index=["point", "representation"], columns="protocol", values="corr_implied").round(3))
display(e2.pivot_table(index=["point", "representation"], columns="protocol", values="crb_sd_b").round(4))"""),

    ("md", """## E3 — train-protocol × test-protocol matrix (5k sims, 3 seeds, equal budget)
### E3a — simulate training banks and test sets
Same θ draws (and seeds) for every protocol, so protocols differ only by the initial state."""),
    ("code", "S.stage_e3sim()"),

    ("md", """### E3b — train 5 protocols × 3 seeds
Sequential by default (~2–3 h). With `PARALLEL_TRAINING = True`, two subprocesses are
launched from this folder and the cell waits for both."""),
    ("code", """if PARALLEL_TRAINING:
    groups = [("P0,P1", THREADS), ("P1b,P2,P3", THREADS)]
    procs = [subprocess.Popen([sys.executable, str(ROOT / "scripts" / "nb29_ws_study.py"), "e3train",
                               "--protocols", g, "--threads", str(t)], cwd=NB_DIR,
                              stdout=open(R / f"29fx_train_{g.replace(',', '_')}.log", "w"),
                              stderr=subprocess.STDOUT) for g, t in groups]
    codes = [p.wait() for p in procs]
    assert all(c == 0 for c in codes), f"training subprocess failed: {codes} (see results/29fx_train_*.log)"
else:
    S.stage_e3train(P, list(range(S.N_SEEDS)), THREADS)
print(sorted(f.name for f in (R / "29fx").glob("post_*.pkl")))"""),

    ("md", "### E3c — evaluate every posterior on every test column"),
    ("code", "S.stage_e3eval()"),
    ("code", """m = load("29fx_e3_matrix.csv")
metrics = [("tarp_dev", "TARP deviation (joint calibration, lower better)"),
           ("rmse_norm", "posterior-mean RMSE / prior width"),
           ("macro_f1", "macro-F1 (threshold 0.90)"),
           ("corr_Sc4", "posterior Corr(α,β) at Sc4")]
fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
for ax, (k, title) in zip(axes, metrics):
    tab = m.groupby(["train", "test"])[k].mean().unstack().reindex(index=P, columns=P)
    im = ax.imshow(tab.values, cmap="viridis")
    ax.set_xticks(range(5), P); ax.set_yticks(range(5), P)
    ax.set_xlabel("test protocol (data)"); ax.set_ylabel("train protocol"); ax.set_title(title, fontsize=9)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{tab.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(FIGS / "29fx_e3_matrix.png", dpi=150); plt.show()
cols = ["ks_p_a", "ks_p_b", "tarp_ks_p", "tarp_dev", "cov90_a", "cov90_b", "rmse_norm", "macro_f1",
        "leak_sbc_mean", "corr_Sc1", "corr_Sc2", "corr_Sc3", "corr_Sc4", "corr_Sc7"]
display(m[m.test == "P2"].groupby("train")[cols].agg(["mean", "std"]).round(3))"""),
    ("md", "### Per-scenario detail on the decision column (P2)"),
    ("code", """sc = load("29fx_e3_scenarios.csv")
display(sc[sc.test == "P2"].groupby(["train", "scenario"])[
    ["alpha_hat", "beta_hat", "bias_a", "bias_b", "cov90_truth_a", "cov90_truth_b",
     "corr_ab_mean", "class_acc"]].mean().round(3))"""),

    ("md", "## Selection (applies the pre-set rule)"),
    ("code", """S.stage_select()
sel = json.loads((R / "29fx_selection.json").read_text())
print("CHOICE:", sel["choice"])
if sel["flag"]:
    print("FLAG:", sel["flag"])"""),

    ("md", """## Conclusions (2026-11-08 run)

**1. The selection rule was flagged, and the cause is not the protocol.** No protocol is
eligible because every one fails the *full-set α* SBC KS test, including the matched
diagonal (P1→P1 p≈10⁻¹⁴, P2→P2 p≈10⁻¹⁰). β KS and joint TARP pass for the realistic
protocols, and for P1b the α 90% coverage is 0.91–0.93. The α failure is a small shift of
the α posterior whose sign changes between seeds and protocols (mean rank 0.43–0.59, ~0.1
posterior sd) for a parameter
the data pin down very tightly (posterior sd ≈ 1.5% of the prior width): an estimator
accuracy limit at 5k simulations, shared by all protocols. It is nb31's job (budget), not
a reason to prefer one protocol. The rule's fallback (lowest P2 TARP deviation) picks
**P1b**, which is also best on every other P2-column calibration measure (TARP KS p 0.82,
β KS 0.40, 90% coverage 0.91/0.89, RMSE tied with P2).

**2. The noise-free deterministic start (P1) is not innocuous.** Trained on P1 and tested on
realistic data, β is overconfident (90% coverage 0.76–0.78; 0.40 at Sc4) and the Fisher
information shows a spurious *negative* α–β correlation (−0.41 at Sc1, −0.69 at Sc4) that
disappears with a stationary start (P1b: −0.04, +0.04). An exactly-steady initial state
leaks θ through the deterministic steady-state map.

**3. The old β bias is reproduced exactly by the protocol mismatch.** Train P0 (abrupt
fault) → test P2 (incipient): Sc2 β̂ = 0.555 (bias −0.145, coverage 0), against the old
manuscript value 0.551 (bias −0.149). The bias was a training/evaluation initial-state
mismatch, not physics.

**4. The α–β correlation is now physically interpretable.** With the fixed controller and a
stationary start, α and β are nearly uncorrelated while the loop is in control (SBI: Sc1
+0.03, Sc3 +0.03, Sc4 +0.02; FIM: −0.04, −0.06, +0.04) and positively correlated when the
coolant valve saturates (Sc2 +0.42, Sc5 +0.39; FIM Sc2 +0.48). Saturation opens the loop,
and the open-loop heat-balance compensation (less heat generated ≈ less heat removed)
appears with the expected positive sign. The near-unity negative correlation of the earlier
work came from the controller bug plus the noise-free start.

**5. The hand-crafted summaries are close to sufficient here.** Under P1b the raw-trajectory
Cramér–Rao bounds are within ~20% of the 29-D summary bounds (β: 0.033 vs 0.036 at Sc1), and
the implied correlations agree in sign and roughly in size. The diagonal-covariance FIM is
wrong (Sc2 I_αα/I_ββ 4.4 vs 36 with full covariance).

**6. Scenario-design issues found** (user decisions, see the re-run plan): Sc5's true β = 0.40
sits *on* the prior's lower edge, so its 0% β coverage is a boundary artefact; Sc2 (β = 0.70)
saturates under the fixed law, so there is no unsaturated fouling scenario; Sc7's β bias
(−0.29) is the known temperature-drift confound (nb09)."""),
    ("md", """## Review — user sign-off required before nb31

The selection is flagged, so nb31 will not start until the user approves a protocol here.
Set `USER_APPROVES_CHOICE = True` (and edit `REASON` if needed) and run this cell."""),
    ("code", """USER_APPROVES_CHOICE = False
CHOICE = "P1b"
REASON = ("Flag caused by full-set alpha SBC failing for every protocol incl. matched pairs "
          "(5k-budget estimator limit, handled by nb31); P1b best on all other P2-column calibration measures.")
if USER_APPROVES_CHOICE:
    (R / "29fx_selection_review.json").write_text(json.dumps(
        {"choice": CHOICE, "approved": True, "reason": REASON, "time": time.ctime()}, indent=2))
    print("approved:", CHOICE)
else:
    print("not approved yet -- nb31 will refuse to start")"""),
    ("md", """## E4 (optional) — window length
If the protocols converge as the window grows, the initial state matters only for short
windows: a practical rule for readers. Off by default (`RUN_E4`)."""),
    ("code", """if RUN_E4:
    S.stage_e4(THREADS)
e4 = load("29fx_e4_windows.csv")
if e4 is not None:
    display(e4.pivot_table(index=["train", "test"], columns="window_min", values=["rmse_norm", "tarp_dev"]).round(4))"""),

    ("md", """## Reading guide for the manuscript
- The off-diagonal cell **train P0 → test P1** reproduces, under the fixed controller, the
  mechanism behind the old β bias (training/evaluation initial-state mismatch).
- Main text: the E3 heatmap. SI: E1, E2, E4.
- The chosen protocol is used by nb31 (budget) and every downstream notebook."""),
    ("code", 'print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")'),
]


N3_NOTE = """> **Sensor model: {sensor}** ({desc}). This notebook re-runs the protocol study under the
> production sensor model adopted on 2026-11-09 after notebook 33. The N0 run is
> `29_warm_start_protocol_study.ipynb` (kept for the SI). Same criterion, same seeds; outputs
> carry the prefix `{tag}_`. Purpose: show that the protocol choice (P1b) does not depend on
> the sensor model, and report the N3 protocol-mismatch effects for the paper."""

AUTO_REVIEW_MD = """## Review — automatic under the pre-set conditions (proposed 2026-11-09, user confirmation in the N3 runbook)
Under N0 the selection was flagged because the full-set α SBC test failed for *every* protocol,
including matched train/test pairs (an estimator-budget limit, not a protocol property), and the
user approved the fallback P1b. For this re-run the same resolution is pre-approved **only
if all of the following hold**; otherwise the cell stops the pipeline for user review:
1. the selection is P1b;
2. if flagged, the cause is the same: the full-set α KS test fails (p < 0.01, seed median) on
   the matched diagonal of every protocol;
3. on the P2 column, P1b passes β (KS p ≥ 0.01) and the joint TARP test (p ≥ 0.01)."""

AUTO_REVIEW_CODE = """sel = json.loads((R / f"{S.TAG}_selection.json").read_text())
m = load(f"{S.TAG}_e3_matrix.csv")
diag_alpha_fails = all(m[(m.train == p) & (m.test == p)].ks_p_a.median() < 0.01 for p in P)
p1b = m[(m.train == "P1b") & (m.test == "P2")]
p1b_ok = (p1b.ks_p_b.median() >= 0.01) and (p1b.tarp_ks_p.median() >= 0.01)
print("choice", sel["choice"], "| flagged", bool(sel["flag"]), "| alpha fails on every matched pair", diag_alpha_fails,
      "| P1b passes beta and TARP on P2", p1b_ok)
_rv = R / f"{S.TAG}_selection_review.json"
_rev = json.loads(_rv.read_text()) if _rv.exists() else {}
if _rev.get("approved") and _rev.get("choice") == sel["choice"] and "user" in str(_rev.get("by", "")).lower():
    print(f"user decision on file: {_rev['choice']} approved by {_rev['by']} ({_rev.get('time', '')})")
    print("reason:", _rev.get("reason", ""))
elif sel["choice"] != "P1b":
    raise RuntimeError("STOP: under this sensor model the rule selects " + sel["choice"] + ", not P1b -- user review needed")
elif sel["flag"]:
    if not (diag_alpha_fails and p1b_ok):
        raise RuntimeError("STOP: selection flagged for a different reason than under N0 -- user review needed")
    (R / f"{S.TAG}_selection_review.json").write_text(json.dumps({
        "choice": "P1b", "approved": True, "time": time.ctime(),
        "by": "pre-approval conditions proposed 2026-11-09 (runbook docs/Manuscript/system_i_N3_runbook.md); all three met",
        "reason": "flag caused by full-set alpha SBC failing on every matched pair (budget limit), as under N0"}, indent=2))
    print("pre-approved: P1b (review file written)")
else:
    print("not flagged: P1b selected directly")"""

INTERP_MD = """## Interpretation ({sensor} run) — to be written after the run
Compare with the N0 notebook, cell by cell, and state for the paper:
1. **E1** is deterministic and identical to N0 (sensor noise does not enter it).
2. **E2**: how the Fisher-information correlations and Cramér–Rao bounds change with realistic
   sensors (expect α bounds ~30× wider, β ~2× tighter; check whether the P1 spurious negative
   correlation survives).
3. **E3**: is P1b still the choice, and is it still best on every P2-column calibration measure?
   Does train P0 → test P2 still reproduce a large β bias (the old manuscript's bias)? Does the
   noise-free start (P1) still make β overconfident?
4. **What changes in the paper's protocol section** when it is presented under N3 (main text)
   with N0 in the SI. Keep claims to what the tables show; quote numbers with their seeds/spread."""


def _for_sensor(cells, sensor):
    """N0: the original notebook. Other models: a separate notebook with the model set in the
    first code cell, all result names prefixed with S.TAG, the auto-review cell and an
    interpretation cell in place of the N0 conclusions and the manual review cell."""
    import system_I_config as CFG
    if sensor == "N0":                      # default model is now N3: pin N0 explicitly
        pin = ('os.environ["CSTR_SENSOR_MODEL"] = "N0"\n'
               "import importlib, ws_protocols, ws_eval, system_I_config, nb29_ws_study\n"
               "for _m in (system_I_config, ws_protocols, ws_eval, nb29_ws_study):\n"
               "    importlib.reload(_m)\n"
               "S = nb29_ws_study\n"
               'assert S.SENSOR == "N0", S.SENSOR\n')
        out = []
        for k, src in cells:
            if k == "code" and 'import nb29_ws_study as S\n' in src:
                src = src.replace('import nb29_ws_study as S\n', pin)
                if "import json, os," not in src:
                    src = src.replace("import json, ", "import json, os, ", 1)
            out.append((k, src))
        return out
    out = []
    for k, src in cells:
        if k == "md" and src.startswith("## Conclusions (2026-11-08 run)"):
            continue                                   # replaced by INTERP_MD after the review cell
        if k == "md" and src.startswith("## Review — user sign-off required before nb31"):
            out.append(("md", AUTO_REVIEW_MD))
            continue
        if k == "code" and src.startswith("USER_APPROVES_CHOICE"):
            out.append(("code", AUTO_REVIEW_CODE))
            out.append(("md", INTERP_MD.format(sensor=sensor)))
            continue
        if k == "code":
            src = (src.replace('load("29fx_', 'load(f"{S.TAG}_').replace('(R / "29fx_', '(R / f"{S.TAG}_')
                      .replace('R / "29fx_', 'R / f"{S.TAG}_').replace('FIGS / "29fx_', 'FIGS / f"{S.TAG}_')
                      .replace('(R / "29fx").glob', 'S.OUT.glob').replace('f"29fx_train_', 'f"{S.TAG}_train_')
                      .replace('see results/29fx_train_', 'see results/{S.TAG}_train_'))
            if src.startswith("import json, os, sys, time, subprocess"):
                src = src.replace("import nb29_ws_study as S\n",
                                  f'os.environ["CSTR_SENSOR_MODEL"] = "{sensor}"\n'
                                  "import importlib, ws_protocols, ws_eval, system_I_config, nb29_ws_study\n"
                                  "for _m in (system_I_config, ws_protocols, ws_eval, nb29_ws_study):\n"
                                  "    importlib.reload(_m)\n"
                                  "S = nb29_ws_study\n"
                                  f'assert S.SENSOR == "{sensor}", S.SENSOR\n')
        out.append((k, src))
        if k == "md" and src is CRITERION:
            tag = "29fx" + CFG.suffix(sensor)
            out.append(("md", N3_NOTE.format(sensor=sensor, desc=CFG.describe(sensor), tag=tag)))
    return out


def build(execute: bool = False, sensor: str = "N0") -> Path:
    out_path = OUT if sensor == "N0" else OUT.with_name(OUT.stem + f"_{sensor}.ipynb")
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in _for_sensor(CELLS, sensor)]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)             # write first, so the notebook exists even if execution stops
    if execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=None, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}})   # cwd = notebooks/system_I
        try:
            client.execute()
        finally:
            nbformat.write(nb, out_path)     # keep partial outputs if a cell fails
    return out_path


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    _sensor = sys.argv[sys.argv.index("--sensor") + 1] if "--sensor" in sys.argv else "N0"
    print(build(execute="--execute" in sys.argv, sensor=_sensor))
