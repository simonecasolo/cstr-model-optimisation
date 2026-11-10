# System R design scorecard — criteria fixed BEFORE any candidate is computed (2026-11-10)

Fault model: alpha (catalyst activity, k = alpha k0 exp(-E/RT)), beta (jacket fouling, UA_eff = beta UA).
Prior box: alpha, beta in [0.4, 1.2]. Class thresholds 0.90 (as System I).
Realistic plant variation for all tests: true Ti, Tci in nominal +-2 K (measured, calibration offset sd 0.1 K,
white noise 0.2 K); feed PO concentration Ci +-1% sd (unmeasured); feed flow Q +-0.5% sd (flowmeter offset).

C1 Unique steady state: unique closed-loop steady state at every (alpha, beta) of a 17x17 grid over the prior box
   and at the inlet corners; when the valve saturates, the open-loop steady state at the clamp is unique.
C2 Safety: every steady state on the grid (saturated or not) has T <= 355 K (Fogler's limit for this reactor).
C3 Controller: closed loop locally stable at every unsaturated grid point; slowest closed-loop time constant
   <= 15 min (so the 300-min P2 history is >= 20 time constants and a 60-min window >= 4).
C4 Cooling margin: at alpha = beta = 1, Qc <= 40% of Qc_max; valve saturation onset beta_sat(alpha=1) <= 0.55;
   no low-side saturation for alpha >= 0.4 at any beta in the box.
C5 Conversion at nominal: 70-90%.
C6 Information: Bayesian Cramer-Rao sd at nominal (alpha=beta=1), production-type features (29 summaries + 3 plug-in
   estimators with measured inlet + 2 measured inlet temperatures), nuisances marginalised with the priors above:
   sd_alpha <= 0.03 and sd_beta <= 0.02.
C7 Inlet robustness of the physics: deterministic error of the jacket-balance beta estimate per 0.5 K error in Tci,
   and of the reactor-balance beta estimate per 0.5 K error in Ti, each <= 1 x (C6 sd_beta).
C8 Proxy separation: the two physics proxies (alpha_hat from the component balance, beta_hat from the jacket balance)
   with a plain 0.90-threshold rule classify scenarios placed >= 3 C6-sd from both thresholds with accuracy >= 0.95
   under the full realistic variation; cross-sensitivity |d beta_hat / d alpha| and |d alpha_hat / d beta|
   <= 0.10 over the unsaturated prior box.
C9 Summary robustness (descriptive, not gated): degenerate/constant features, clipping frequency, collinearity
   (condition number), shift of each feature between warm starts P1b and P2 in units of its sd.

Selection: candidates passing C1-C8; tie-break = largest minimum normalised margin over C3-C8.
Scenario placement (after selection): faults >= 3 sd from the thresholds and inside the unsaturated band, plus one
deliberate near-threshold case (1 sd), one saturation case, one T-sensor-drift case.
