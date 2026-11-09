"""Build / execute notebooks/system_I/14_claims_and_conclusions_N3.ipynb (System I, Phase 3 step g; last).

Rebuilt from the claims ledger of docs/Manuscript/system_i_rerun_plan.md (read at BUILD time, so the ledger text is the current one) and a numeric
dashboard that loads every headline number from the stored results (nothing typed). A written "## 5. Conclusions" is kept on rebuild.

    python scripts/build_nb_system_I_14.py [--execute]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "14_claims_and_conclusions_N3.ipynb"
PLAN = ROOT / "docs" / "Manuscript" / "system_i_rerun_plan.md"


def _section(text: str, head: str) -> str:
    m = re.search(rf"^## {re.escape(head)}.*?$", text, re.M)
    if not m:
        return f"(section '{head}' not found in the plan)"
    rest = text[m.end():]
    n = re.search(r"^## ", rest, re.M)
    return m.group(0) + rest[: n.start() if n else len(rest)]


def cells() -> list:
    plan = PLAN.read_text()
    story = _section(plan, "Working storyline of the paper").replace("## Working storyline", "### Working storyline", 1)
    ledger = _section(plan, "Claims ledger").replace("## Claims ledger (fill in as notebooks are re-run)", "### Claims ledger (as of the build of this notebook)", 1)
    si = _section(plan, "SI material").replace("## SI material", "### SI material", 1)
    header = """# 14 — Claims, conclusions and the paper outline (System I, N3)

The old notebook 14 synthesised the notebooks 02–12 of the pre-fix pipeline. This one is **rebuilt from the claims ledger** of `docs/Manuscript/system_i_rerun_plan.md` (copied in at build time, sections 1 and 3) and from a **dashboard that loads every headline number from the stored results** (section 2; nothing is typed here).
Section 4 lists what is not settled. Section 5 (the conclusions) is written after the last run.
"""
    dash = r'''import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent; R = ROOT / "results"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
def need(*names):
    miss = [n for n in names if not (R / n).exists()]
    assert not miss, f"missing result files: {miss}"
need("36_N3_production.json", "36_N3_metrics.csv", "11_N3_macro_f1.csv", "11_N3_L1_three_numbers.csv", "10_N3_tracking_metrics.csv", "16_N3_tuned_vs_untuned_pooled_Sc1_5.csv", "09_N3_decision.json",
     "38_N3_gate_rates.csv", "08_N3_fim.csv", "18_N3_ablation.csv", "11b_N3_zT_guarded_rule.csv")
rows = []
j = json.loads((R / "36_N3_production.json").read_text()); m36 = pd.read_csv(R / "36_N3_metrics.csv"); e = m36[(m36.member == "ensemble") & (m36.budget == j["production_budget"])].iloc[0]
rows.append(("Production posterior (nb36)", f"{j['production_budget']} simulations x {len(j['members'])}-member ensemble; features {j['summary_set']}; interior SBC KS p alpha {e.ks_p_a_interior:.3f}, beta {e.ks_p_b_interior:.3f}; TARP p {e.tarp_ks_p:.3f}; 90% coverage {e.cov90_a:.3f} / {e.cov90_b:.3f}; sd {e.sd_a:.3f} / {e.sd_b:.3f}"))
mf = pd.read_csv(R / "11_N3_macro_f1.csv"); r = mf[(mf.set.str.startswith("L1")) & (mf.scenarios.str.startswith("Sc1-Sc5"))].iloc[0]; r2 = mf[(mf.set.str.startswith("L1")) & (mf.scenarios == "all six")].iloc[0]
rows.append(("Macro-F1 (nb11, L1 = 6,000 windows)", f"Sc1-Sc5 {r.macro_f1:.3f} (accuracy {r.accuracy:.3f}); with Sc7 {r2.macro_f1:.3f}"))
t = pd.read_csv(R / "11_N3_L1_three_numbers.csv"); rows.append(("Per-scenario accuracy / beta coverage / beta sd (nb11, L1)", "; ".join(f"{a.scenario} {a.class_acc:.3f} / {a.cov90_b:.3f} / {a.sd_b:.3f}" for a in t.itertuples())))
tr = pd.read_csv(R / "10_N3_tracking_metrics.csv").iloc[0]; rows.append(("30-day tracking (nb10)", f"alpha MAE {tr.alpha_MAE:.3f}, beta MAE {tr.beta_MAE:.3f}; 90% coverage {tr.alpha_cov90:.3f} / {tr.beta_cov90:.3f}; beta bias {tr.beta_bias:+.3f} (persistent inlet offsets); class accuracy {tr.class_acc:.3f}"))
pooled = pd.read_csv(R / "16_N3_tuned_vs_untuned_pooled_Sc1_5.csv")
rows.append(("EKF/UKF vs SBI (nb16, D1 Sc1-Sc5 pooled)", "; ".join(f"{a.method} {a.variant}: beta coverage {a.cov90_b:.2f}, RMSE {a.rmse_b:.3f}, tail windows {int(a.n_tail_b)}" for a in pooled.itertuples()) + f"; SBI production: beta coverage {t[t.scenario != 'Sc7'].cov90_b.mean():.2f}, sd {t[t.scenario != 'Sc7'].sd_b.mean():.3f}"))
dec = json.loads((R / "09_N3_decision.json").read_text()); rows.append(("Drift-aware posterior (nb09)", f"verdict {dec['verdict']}; macro-F1 Sc1-Sc5 {dec['macro_f1_Sc1_5']:.4f}; Sc7 beta coverage {dec['sc7']['cov90_b']:.3f}, bias {dec['sc7']['bias_b']:+.3f}"))
g = pd.read_csv(R / "38_N3_gate_rates.csv"); rows.append(("Gate (nb38, D1)", f"false-alarm rate (all windows) {', '.join(f'{a.scenario} {a.flag_rate_all_windows:.2f}' for a in g[g.scenario != 'Sc7'].itertuples())}; Sc7 detection {g[g.scenario == 'Sc7'].flag_rate_all_windows.iloc[0]:.2f} overall, {g[g.scenario == 'Sc7'].flag_rate_unsaturated.iloc[0]:.2f} unsaturated"))
f = pd.read_csv(R / "08_N3_fim.csv"); n_ = f[f["mode"].str.startswith("nominal")]; u_ = f[f["mode"].str.startswith("nuisance")]
rows.append(("Information (nb08)", f"CRB beta nominal {n_.crb_sd_b.min():.3f}-{n_.crb_sd_b.max():.3f}; inlet as a nuisance {u_.crb_sd_b.min():.3f}-{u_.crb_sd_b.max():.3f}; alpha {u_.crb_sd_a.min():.3f}-{u_.crb_sd_a.max():.3f} (unchanged)"))
ab = pd.read_csv(R / "18_N3_ablation.csv").set_index("set"); rows.append(("Feature ablation (nb18)", "; ".join(f"{k.split(' (')[0]}: beta sd {v.sd_b:.3f}, macro-F1 {v.macro_f1:.2f}" for k, v in ab.iloc[[0, 1, 3, 4, 5]].iterrows())))
for nm, fn, fmt in (("Sbi timing (nb12)", "12_N3_sbi_timing.csv", None), ("Break-even (nb12)", "12_N3_break_even.csv", None)):
    if (R / fn).exists():
        d_ = pd.read_csv(R / fn); rows.append((nm, d_.round(3).to_string(index=False)))
for fn, nm in (("37_N3_lc2st_positive_controls.csv", "L-C2ST positive controls (nb37)"),):
    if (R / fn).exists():
        d_ = pd.read_csv(R / fn); rows.append((nm, "; ".join(f"{a.variant}|{a.posterior}: p<0.05 in {a.frac_p_below_0_05:.2f}" for a in d_.itertuples())))
dash = pd.DataFrame(rows, columns=["claim", "value (loaded from results/)"])
for _, r_ in dash.iterrows():
    print(f"* {r_['claim']}\n    {r_.iloc[1]}\n")
dash.to_csv(R / "14_N3_dashboard.csv", index=False)'''
    open_items = """## 4. What is not settled (read before the conclusions)

* **No NUTS baseline** (decision C, plan SI-1): the posterior is checked by SBC, TARP, the Cramér–Rao bound, the EKF/UKF comparison and L-C2ST. The notebooks 05, 20, 21 and 23 still carry the pre-fix NUTS numbers; only notebook 21 carries the banner that they are not valid (notebooks 05, 20 and 23 need the same banner, pending your decision).
* **Drift-aware posterior (nb09):** pre-set verdict FIXED with a macro-F1 margin of 0.0001 and an ensemble TARP p of 0.017; the deployment gate (nb38) repairs the unsaturated Sc7 windows only (saturated windows remain wrong unless the guard is dropped, at a cost in the well-specified scenarios).
* **The L-C2ST (nb37)** — see its commentary: the pre-set default is not interpretable on its own; the corrected positive controls decide how far it can be read.
* **System II / Luyben** carry the same malformed PI form as System I had; not touched.
* **CNN vs hand-crafted (nb04b/24)** is still pending; the old "within 1%" claim is retired.
* **The k0 = 60× Fogler** (about 98% conversion) is kept and must be disclosed.
* **Controller-form correction, k0 disclosure and the warm-start protocol** are decided but their manuscript text is pending (ledger).
* **SI figures (nb17):** not regenerated (nb17 rewrites `data/observations.npz` and `data/scenario_configs.csv`, which would overwrite the regenerated D0).
"""
    concl = """## 5. Conclusions — to be written after the last run
(The core message, the sources of silent failure with their numbers, the practitioner checklist and the open items, written against the dashboard above.)"""
    return [
        ("md", header),
        ("md", "## 1. The storyline and the SI decisions (from the plan)\n\n" + story + "\n\n" + si),
        ("md", "## 2. The headline numbers, loaded from the stored results"),
        ("code", dash),
        ("md", "## 3. The claims ledger\n\n" + ledger),
        ("md", open_items),
        ("md", concl),
    ]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in cells()]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 5. Conclusions")
                and "to be written after the last run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 5. Conclusions"):
                    c.source = kept[-1]
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    if execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print(build(execute="--execute" in sys.argv))
