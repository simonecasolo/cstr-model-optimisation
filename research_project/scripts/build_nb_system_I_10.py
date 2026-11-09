"""Build / execute notebooks/system_I/10_sequential_degradation_tracking_N3.ipynb (System I, Phase 3 step d; includes c2 on the stream and c4).

30-day stream under the production model; production posterior (nb36) per window. Rules and checks fixed in
docs/Manuscript/system_i_N3_runbook.md section 2f (row d). A written section-"8. Interpretation" is kept on rebuild.
The old notebook 10 is left untouched.

    python scripts/build_nb_system_I_10.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks" / "system_I"
OUT = NB_DIR / "10_sequential_degradation_tracking_N3.ipynb"

HEADER = """# 10 — Sequential degradation tracking under the production model (System I, N3)

**Claim 2** (amortised monitoring): one posterior, trained once, applied to every window of a 30-day degradation with no retraining.
The old notebook 10 used the pre-fix simulator, a nominal inlet and per-window steady-state warm starts; it is left untouched.

**Stream (fixed before running; runbook §2f row d).** 720 windows, window *w* starting at *t* = *w* hours. α(t) = 1 − 0.15 t/720 h (linear);
β(t) = 1 − b(1 − e^(−t/200 h)), b = 0.15/(1 − e^(−3.6)) (Kern–Seaton): both reach 0.85 at day 30; α crosses 0.90 on day 20 and β on day 9 (window 210). Each window is simulated independently with the production protocol **P2** (300-min noisy history) at its own (α, β), N3 sensors.
True inlet: Ti = 297 + 1.0 sin(2πt/24 h); Tci = 297 + 1.2 sin(2πt/168 h + 0.7) + 0.5 t/720 h (a drifting coolant-supply temperature, inside the
training range [295, 299] K); Ci = 0.97(1 + N(0, 0.01²)) per window, unmeasured. **Measured Ti, Tci = true + a persistent offset for the whole month**
(ε N(0, 1) per channel, ε = 0.1 K, drawn once) + window noise N(0, 0.2²/120). The production posterior (notebook 36) was trained with offsets drawn
*per window*; a persistent offset is the realistic case and a deliberate mismatch, so a month-long β bias is reported rather than assumed away.
Seed base 3 900 000 (new). Posterior: nb36 ensemble, 1,000 draws per window.

**Also here.** **c2**: the drift detector z_T = (mean measured T − T_sp)/(0.2 K/√120) per window on the stream (D1 and L1: separate step).
**c4**: SBI inference time per window (features + 1,000 draws from the 5-member ensemble).

**Checks (stop if violated):** finite; α(0) = β(0) = 1 and end values from the formulas; true Ti, Tci in [295.5, 298.7] K; regeneration bit-identical;
production features equal `nb36.features_from_raw`; α and β 90% coverage ≥ 0.80 over the 720 windows. No decision rule.
"""

CELLS = [
    ("md", HEADER),
    ("code", """import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NB_DIR = Path.cwd().resolve()
assert NB_DIR.name == "system_I" and NB_DIR.parent.name == "notebooks", f"run from notebooks/system_I (cwd {NB_DIR})"
ROOT = NB_DIR.parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
os.environ["CSTR_SENSOR_MODEL"] = "N3"
import importlib, system_I_config, ws_eval, ws_metrics, ws_inlet, nb35_measured_inlet, nb36_production_measured_inlet, ws_phase3
for _m in (system_I_config, ws_eval, ws_metrics, ws_inlet, nb35_measured_inlet, nb36_production_measured_inlet, ws_phase3):
    importlib.reload(_m)
we, wm, P3 = ws_eval, ws_metrics, ws_phase3
T0 = time.time(); R = ROOT / "results"; FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)
post, prod = P3.load_production_posterior()
print("Production posterior:", prod["production_budget"], "simulations, ensemble of", len(prod["members"]), "| summary set", prod["summary_set"])
"""),
    ("md", "## 1. Generate the stream and run the checks"),
    ("code", """t0 = time.time(); s = P3.generate_stream(); print(f"simulated {s['x'].shape} in {time.time() - t0:.1f}s")
n = len(s["x"]); th = s["theta"]; it = s["inlet_true"]; im = s["inlet_measured"]
assert s["x"].shape == (720, 120, 4) and np.isfinite(s["x"]).all() and np.isfinite(im).all()
assert abs(th[0, 0] - 1) < 1e-6 and abs(th[0, 1] - 1) < 1e-6
a_end, b_end = P3.stream_truth(719.0)
assert abs(th[-1, 0] - a_end) < 1e-6 and abs(th[-1, 1] - b_end) < 1e-6, (th[-1], a_end, b_end)
assert it[:, 1:].min() >= 295.5 and it[:, 1:].max() <= 298.7, (it[:, 1:].min(), it[:, 1:].max())
s2 = P3.generate_stream()
for k in ("x", "inlet_true", "inlet_measured", "theta", "offsets_ti_tci"):
    assert np.array_equal(s[k], s2[k]), k
f_prod = P3.production_features(s)
assert f_prod.shape == (720, 34) and np.isfinite(f_prod).all()
assert np.array_equal(f_prod, nb36_production_measured_inlet.features_from_raw(s["x"], s["inlet_measured"]))
dev = im - it[:, 1:]
print(f"alpha {th[0,0]:.3f} -> {th[-1,0]:.4f}, beta {th[0,1]:.3f} -> {th[-1,1]:.4f} | true Ti in [{it[:,1].min():.2f}, {it[:,1].max():.2f}], Tci in [{it[:,2].min():.2f}, {it[:,2].max():.2f}] K")
print("persistent offsets drawn for the month (Ti, Tci):", np.round(s["offsets_ti_tci"], 4), "K | mean measured - true:", np.round(dev.mean(0), 4), "| window-noise sd:", np.round((dev - s["offsets_ti_tci"]).std(0), 4))
print("checks passed (shapes, finite, truth, inlet range, bit-identical, features)")
P3.STREAM_PATH.parent.mkdir(exist_ok=True); np.savez(P3.STREAM_PATH, **s); print("saved", P3.STREAM_PATH.relative_to(ROOT))
"""),
    ("md", "## 2. Production posterior on every window (1,000 draws)"),
    ("code", """t0 = time.time()
smp, leak = we.sample_within_prior(post, f_prod, 1000, rng_seed=11)
print(f"{n} windows x 1000 draws in {time.time() - t0:.1f}s | prior leakage mean {leak.mean():.4f}, max {leak.max():.3f}")
mean = smp.mean(1); sd = smp.std(1); q05 = np.percentile(smp, 5, axis=1); q95 = np.percentile(smp, 95, axis=1)
a_true, b_true = th[:, 0].astype(float), th[:, 1].astype(float)
cov = lambda lo, hi, t: float(((t >= lo) & (t <= hi)).mean())
cov_a, cov_b = cov(q05[:, 0], q95[:, 0], a_true), cov(q05[:, 1], q95[:, 1], b_true)
print(f"90% coverage over the 720 windows: alpha {cov_a:.3f}, beta {cov_b:.3f}")
assert cov_a >= 0.80 and cov_b >= 0.80, "coverage check failed"
yt = wm.class_of_theta(th.astype(float)); yp = wm.class_of_samples(smp)
"""),
    ("md", "## 3. Tracking metrics (overall and by phase)"),
    ("code", """day = np.arange(n) / 24.0
def block(m, label):
    ea, eb = mean[m, 0] - a_true[m], mean[m, 1] - b_true[m]
    return dict(period=label, n=int(m.sum()), alpha_MAE=float(np.abs(ea).mean()), alpha_bias=float(ea.mean()), alpha_RMSE=float(np.sqrt((ea ** 2).mean())),
                alpha_cov90=cov(q05[m, 0], q95[m, 0], a_true[m]), alpha_sd=float(sd[m, 0].mean()),
                beta_MAE=float(np.abs(eb).mean()), beta_bias=float(eb.mean()), beta_RMSE=float(np.sqrt((eb ** 2).mean())),
                beta_cov90=cov(q05[m, 1], q95[m, 1], b_true[m]), beta_sd=float(sd[m, 1].mean()), class_acc=float((yt[m] == yp[m]).mean()))
rows = [block(np.ones(n, bool), "all 30 days")] + [block((day >= lo) & (day < hi), f"days {lo}-{hi}") for lo, hi in ((0, 10), (10, 20), (20, 30))]
tab = pd.DataFrame(rows); display(tab.round(3)); tab.to_csv(R / "10_N3_tracking_metrics.csv", index=False)
# month-mean beta / alpha bias with a standard error from the 30 daily means (windows within a day are strongly correlated)
dm = lambda e: np.array([e[(day >= d) & (day < d + 1)].mean() for d in range(30)])
for nm, e in (("beta", mean[:, 1] - b_true), ("alpha", mean[:, 0] - a_true)):
    x = dm(e); print(f"month-mean {nm} bias {x.mean():+.4f} +- {x.std(ddof=1) / np.sqrt(30):.4f} (SE from 30 daily means)")
print("expected under per-window offsets: month-mean beta bias ~ 0; realised persistent offsets (Ti, Tci):", np.round(s["offsets_ti_tci"], 3), "K")
"""),
    ("md", "## 4. Classification and detection delay"),
    ("code", """CLS = list(wm.CLASSES)
cm = pd.crosstab(pd.Series(yt, name="true"), pd.Series(yp, name="pred")).reindex(index=CLS, columns=CLS, fill_value=0); display(cm)
def first_sustained(p, k=24):
    ok = (p > 0.5).astype(int); run = np.convolve(ok, np.ones(k, int), "valid") == k
    return int(np.argmax(run)) if run.any() else None
p_a = (smp[:, :, 0] < 0.90).mean(1); p_b = (smp[:, :, 1] < 0.90).mean(1)
true_a = int(np.argmax(a_true < 0.90)) if (a_true < 0.90).any() else None
true_b = int(np.argmax(b_true < 0.90)) if (b_true < 0.90).any() else None
det = pd.DataFrame([dict(parameter="alpha", true_crossing_window=true_a, detected_window=first_sustained(p_a)),
                    dict(parameter="beta", true_crossing_window=true_b, detected_window=first_sustained(p_b))])
det["delay_windows(h)"] = det.detected_window - det.true_crossing_window
display(det); det.to_csv(R / "10_N3_detection_delay.csv", index=False)
cm.to_csv(R / "10_N3_confusion.csv")
"""),
    ("md", "## 5. c2 — drift detector on the stream: z_T = (mean measured T − T_sp) / (0.2 K/√120)"),
    ("code", """TSP = 312.5
zT = (s["x"][:, :, 1].mean(1) - TSP) / (0.2 / np.sqrt(120.0))
sat = (s["x"][:, :, 3] >= 395).mean(1)
print(f"z_T on the stream: median {np.median(zT):+.2f}, 5-95% [{np.percentile(zT, 5):+.2f}, {np.percentile(zT, 95):+.2f}], |z_T| > 5 in {(np.abs(zT) > 5).mean() * 100:.1f}% of windows")
print(f"windows with > 50% of samples at Qc >= 395: {(sat > 0.5).mean() * 100:.1f}%; |z_T| > 5 among them: {(np.abs(zT[sat > 0.5]) > 5).mean() * 100:.1f}%, among the others: {(np.abs(zT[sat <= 0.5]) > 5).mean() * 100:.1f}%")
pd.DataFrame(dict(window=np.arange(n), z_T=zT, frac_saturated=sat)).to_csv(R / "10_N3_zT_stream.csv", index=False)
"""),
    ("md", "## 6. c4 — SBI inference time per window"),
    ("code", """import torch
raw_one = s["x"][:1]; im_one = s["inlet_measured"][:1]
def one_window(i):
    f = nb36_production_measured_inlet.features_from_raw(s["x"][i:i + 1], s["inlet_measured"][i:i + 1])
    return we.sample_within_prior(post, f, 1000, rng_seed=i)
one_window(0)                                         # warm-up
tt = []
for i in range(50):
    t = time.perf_counter(); one_window(i); tt.append(time.perf_counter() - t)
t = time.perf_counter(); f_all = nb36_production_measured_inlet.features_from_raw(s["x"], s["inlet_measured"]); t_feat = time.perf_counter() - t
t = time.perf_counter(); _ = we.sample_within_prior(post, f_all, 1000, rng_seed=1); t_batch = time.perf_counter() - t
tim = pd.DataFrame([dict(item="single window: features + 1,000 draws (median of 50), ms", value=1000 * float(np.median(tt))),
                    dict(item="single window (5th-95th percentile), ms", value=f"{1000 * np.percentile(tt, 5):.0f}-{1000 * np.percentile(tt, 95):.0f}"),
                    dict(item="720-window batch: features, ms per window", value=1000 * t_feat / n),
                    dict(item="720-window batch: 1,000 draws, ms per window", value=1000 * t_batch / n),
                    dict(item="torch threads", value=torch.get_num_threads())])
display(tim); tim.to_csv(R / "10_N3_sbi_timing.csv", index=False)
"""),
    ("md", "## 7. Figures and per-window results"),
    ("code", """fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
for ax, k, nm, tr, c in ((axes[0], 0, "α", a_true, "steelblue"), (axes[1], 1, "β", b_true, "darkorange")):
    ax.fill_between(day, q05[:, k], q95[:, k], color=c, alpha=0.25, label="90% credible band"); ax.plot(day, mean[:, k], color=c, lw=1, label=f"posterior mean {nm}")
    ax.plot(day, tr, "k--", lw=1.5, label=f"true {nm}"); ax.axhline(0.9, color="tomato", ls=":", lw=1); ax.set_ylabel(nm); ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
axes[2].plot(day, zT, lw=0.7, color="purple"); axes[2].axhspan(-5, 5, color="grey", alpha=0.15); axes[2].set_ylabel("z_T"); axes[2].set_xlabel("day"); axes[2].grid(alpha=0.3)
fig.suptitle("Production posterior on the 30-day stream (persistent inlet offsets, drifting inlet)"); fig.tight_layout(); fig.savefig(FIGS / "10_N3_tracking.png", dpi=150); plt.show()
pd.DataFrame(dict(window=np.arange(n), day=day, alpha_true=a_true, beta_true=b_true, alpha_mean=mean[:, 0], beta_mean=mean[:, 1], alpha_q05=q05[:, 0], alpha_q95=q95[:, 0],
                  beta_q05=q05[:, 1], beta_q95=q95[:, 1], true_class=yt, pred_class=yp, z_T=zT, frac_saturated=sat)).to_csv(R / "10_N3_window_results.csv", index=False)
print(f"notebook wall time {(time.time() - T0) / 60:.1f} min")
"""),
    ("md", """## 8. Interpretation — to be written after the run
Toward the storyline: what the production posterior does over a month of drifting inlet with persistent offsets (calibration, bias by phase, the persistent
β bias against the per-window-offset training assumption, detection delay of the α and β threshold crossings); whether the z_T detector is quiet on the
well-specified stream and what saturation does to it; inference time. Which of this is evidence for "reliable if the simulator tells the truth"."""),
]


def build(execute: bool = False) -> Path:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if OUT.exists():
        old = nbformat.read(OUT, as_version=4)
        kept = [c.source for c in old.cells if c.cell_type == "markdown" and c.source.startswith("## 8. Interpretation")
                and "to be written after the run" not in c.source.splitlines()[0]]
        if kept:
            for c in nb.cells:
                if c.cell_type == "markdown" and c.source.startswith("## 8. Interpretation"):
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
