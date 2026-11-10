"""Production configuration of System III (fixed in docs/system_III_plan.md; change only by a logged decision)."""
SENSORS = "S3"            # realistic instruments
PROTOCOL = "P2"           # warm start: fault ramp history (System I production choice; re-tested in nb05)
FEATURE_SET = "S1m"       # 29 summaries + 3 plug-in proxies (measured inlet) + 2 measured inlet temperatures
CONDITIONS = "production" # inlet varies +-2 K (measured), Ci, Q and every sensor calibration offset drawn
CONTROL_SENSOR = "measured"
CONTROL = "closed"         # decision 12 pending: "cascade" (realism upgrade)
CASCADE = (4.0, 8.0, 20.0, 1.0)   # (Ko K/K, tau_o min, Ki (L/min)/K, tau_i min), used when CONTROL == "cascade"
BUDGETS = [5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
SEEDS = [0, 1, 2, 3, 4]
N_SBC, SC_REPS = 1000, 50
