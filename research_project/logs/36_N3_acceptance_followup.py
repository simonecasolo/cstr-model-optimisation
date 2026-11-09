"""Read-only follow-up of the notebook-36 acceptance check (user decision 2026-11-09, option 3).
For the four failing cells (Sc2 x {nominal, Tci-0.5, Tci+0.5}, Sc4 x Tci+0.5) and the nominal inlet of Sc1-Sc4,
simulate N_WIN fresh windows per cell (new seeds) with ws_inlet under the nb35 protocol, N3 sensors and inlet, then
evaluate on IDENTICAL windows: the nb36 production ensemble (20k x 5, eps 0.1 K), and the nb35 oracle, eps 0.05 and
eps 0.1 ensembles (40k x 3). Reports class accuracy +- binomial SE, beta sd, beta 90% coverage, beta bias.
Writes only results/36_N3_acceptance_followup.csv."""
import os, sys
from pathlib import Path
os.environ["CSTR_SENSOR_MODEL"] = "N3"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
import system_I_config as CFG, ws_eval as we, ws_metrics as wm, ws_inlet as wi
import nb35_measured_inlet as M, nb36_production_measured_inlet as P
from cstr_sbi.scenarios import SCENARIO_CONFIGS

N_WIN = 1000
CELLS = [("Sc2_closed_fouling", "nominal"), ("Sc2_closed_fouling", "Tci-0.5"), ("Sc2_closed_fouling", "Tci+0.5"),
         ("Sc4_closed_combined", "Tci+0.5"), ("Sc4_closed_combined", "nominal"),
         ("Sc1_closed_healthy", "nominal"), ("Sc3_closed_decay", "nominal")]
FAILING = {("Sc2_closed_fouling", "nominal"), ("Sc2_closed_fouling", "Tci-0.5"), ("Sc2_closed_fouling", "Tci+0.5"), ("Sc4_closed_combined", "Tci+0.5")}
SEED0, OFFSET0 = 3_690_000, 3_691_000

posts = {"nb36 production (20k x5, eps 0.1)": (P.load_production_posterior()[0], "eps0.1"),
         "nb35 oracle (40k x3)": (M._post("oracle"), "oracle"),
         "nb35 eps0.05 (40k x3)": (M._post("eps0.05"), "eps0.05"),
         "nb35 eps0.1 (40k x3)": (M._post("eps0.1"), "eps0.1")}
rows = []
for k, (scn, cond) in enumerate(CELLS):
    sc = SCENARIO_CONFIGS[scn]; dti, dtci = M.CONDITIONS[cond]
    th = np.tile(np.array([[sc.alpha, sc.beta]], np.float32), (N_WIN, 1))
    inl = np.tile(np.array([wi.NOMINAL[0], M.T_NOM + dti, M.T_NOM + dtci], np.float32), (N_WIN, 1))
    raw = wi.simulate(th, inl, M.PROTOCOL, SEED0 + 1000 * k, raw=True, **M.NOISE)
    b29 = we.features(raw, M._t(), "S0_base29")
    names = np.array([scn] * N_WIN)
    for pname, (post, variant) in posts.items():
        x = M.features_for(variant, b29, inl, OFFSET0 + k)          # same offsets for every variant of the cell
        smp, leak = we.sample_within_prior(post, x, M.N_DRAWS, rng_seed=0)
        _, sc_rows = wm.evaluate_scenario_set(smp, th, names, we.LO, we.HI)
        r = sc_rows[0]; acc = r["class_acc"]
        rows.append(dict(scenario=scn.split("_")[0], condition=cond, failing_cell=(scn, cond) in FAILING, posterior=pname, n_windows=N_WIN,
                         class_acc=acc, class_acc_se=float((acc * (1 - acc) / N_WIN) ** 0.5), sd_b=r["sd_b"], sd_a=r["sd_a"],
                         cov90_b=r["cov90_truth_b"], bias_b=r["bias_b"], bias_a=r["bias_a"], leak=float(leak.mean())))
    print("done", scn, cond, flush=True)
df = pd.DataFrame(rows); df.to_csv(ROOT / "results/36_N3_acceptance_followup.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(df.round(3).to_string(index=False))
