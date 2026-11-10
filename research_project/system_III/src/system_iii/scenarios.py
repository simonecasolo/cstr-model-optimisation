"""System III scenario catalogue (docs/system_III_design.md section 5). Placement rule: faults >= 3 posterior
sd (~0.05) from the 0.90 class thresholds and inside the unsaturated band (beta_sat = 0.39 at alpha = 1),
plus deliberate near-threshold, sensor-drift and control-sensor-offset cases. Valve saturation is not a
catalogue scenario (decision 14: inside the prior box it exists only in a thin strip at the beta = 0.4 edge);
it is tested in notebook 12. Fixed before any SBI run; changing a value later is a logged decision."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    alpha: float
    beta: float
    control: str = "closed"         # closed | fixed (open loop)
    drift_T_monitor: float = 0.0    # K, monitoring T reading drifts (control loop unaffected)
    oT: float = 0.0                 # K, control-sensor offset (in the loop)
    note: str = ""


SCENARIOS = {s.name: s for s in [
    # numbering and meaning aligned with System I (src/cstr_sbi/scenarios.py) Sc0-Sc7, plus Sc8-Sc10
    Scenario("Sc0_open_healthy", 1.00, 1.00, control="fixed", note="open loop at the nominal Qc: unique SS 337 K"),
    Scenario("Sc1_healthy", 1.00, 1.00, note="Qc ~101 L/min"),
    Scenario("Sc2_fouling", 1.00, 0.70, note="Qc ~123"),
    Scenario("Sc3_decay", 0.70, 1.00, note="Qc ~76, X 79%"),
    Scenario("Sc4_combined", 0.75, 0.75, note="Qc ~91"),
    Scenario("Sc5_severe_fouling", 1.00, 0.50, note="Qc ~177, unsaturated (valve at the clamp 0.8% of the time "
             "under production inlet variation); same beta as System I's saturated Sc5"),
    Scenario("Sc6_open_fouling", 1.00, 0.70, control="fixed", note="open-loop twin of Sc2: unique SS 340 K"),
    Scenario("Sc7_fouling_drift", 1.00, 0.70, drift_T_monitor=2.0, note="Sc2 + 2 K drift on the logged T, outside the loop"),
    Scenario("Sc8_near_fouling", 1.00, 0.87, note="near-threshold: ~1.9 sd (S5) from the beta threshold"),
    Scenario("Sc9_near_decay", 0.87, 1.00, note="near-threshold: ~1.5 sd (S5) from the alpha threshold"),
    Scenario("Sc10_control_offset", 1.00, 1.00, oT=0.5, note="+0.5 K on the control T sensor, in the loop"),
]}
CLOSED = [k for k, s in SCENARIOS.items() if s.control == "closed"]
OPEN = [k for k, s in SCENARIOS.items() if s.control == "fixed"]
GATED = ["Sc1_healthy", "Sc2_fouling", "Sc3_decay", "Sc4_combined"]
THRESHOLD = 0.90
