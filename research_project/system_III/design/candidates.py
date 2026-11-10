"""Candidate designs. Fogler Example 9-4 (3rd ed.; COMSOL 'Startup of a CSTR'): propylene-glycol production by
acid-catalysed hydrolysis of propylene oxide in a 1.89 m3 CSTR with a heat exchanger.
COMSOL values: V = 1.89 m3, v = 3.47e-3 m3/s, c_PO = 2903 mol/m3, c_W = 36291, c_MeOH = 3629, Tf = 297 K,
A = 4.71e9 1/s, E = 75.358 kJ/mol, UA = 8441 W/K, coolant 126 mol/s water at 289 K, SS T = 337 K, c_PO = 472.
Heat capacity of the feed per litre: 2.903*146.5 + 36.291*75.4 + 3.629*81.6 = 3458 J/(L K).
dH = -84 666 J/mol (Fogler: -36 400 Btu/lbmol).
"""
from dataclasses import replace

from design_lib import Design, ss_unsat

RHOCP_FOGLER = 2.903 * 146.5 + 36.291 * 75.4 + 3.629 * 81.6     # J/(L K)

SYS_I = Design(name="R0_SystemI", V=500.0, Vc=40.0, Q=40.0, Ci=0.97, Ti=297.0, Tci=297.0,
               k0=16.96e12, Ea=75362.0, dH=-20220.0 * 4.184, rhoCp=4184.0, rhocCpc=4184.0,
               UA=1.25e4 * 4.184, Tsp=312.5, Kp=150.0, taui=10.0, Qc0=80.0, Qcmin=0.0, Qcmax=400.0,
               sig_proc=(0.0005, 0.1, 0.1, 0.0), sig_meas=(0.01, 0.2, 0.2, 2.0))

FOGLER = Design(name="Fogler94", V=1890.0, Vc=190.0, Q=3.47e-3 * 60_000, Ci=2.903, Ti=297.0, Tci=289.0,
                k0=4.71e9 * 60, Ea=75362.0, dH=-84666.0, rhoCp=RHOCP_FOGLER, rhocCpc=4184.0,
                UA=8441.0 * 60, Tsp=337.0, Qcmin=0.0, Qcmax=1.0,
                sig_proc=(0.0015, 0.1, 0.1, 0.0), sig_meas=(0.03, 0.2, 0.2, 2.0))


def sized(d, qc_frac=0.35):
    """Valve sized so that the nominal coolant flow is qc_frac of Qc_max; Qc sensor 0.5% of span."""
    s = ss_unsat(d, 1.0, 1.0)
    qmax = s["Qc"] / qc_frac
    sm = list(d.sig_meas); sm[3] = 0.005 * qmax
    return replace(d, Qcmax=qmax, Qc0=s["Qc"], sig_meas=tuple(sm))


CANDIDATES = {
    "R1_fogler_UA1.0": sized(replace(FOGLER, name="R1_fogler_UA1.0")),
    "R2_fogler_UA1.5": sized(replace(FOGLER, name="R2_fogler_UA1.5", UA=FOGLER.UA * 1.5)),
    "R3_fogler_UA2.0": sized(replace(FOGLER, name="R3_fogler_UA2.0", UA=FOGLER.UA * 2.0)),
    "R4_fogler_UA2.0_Ti310": sized(replace(FOGLER, name="R4_fogler_UA2.0_Ti310", UA=FOGLER.UA * 2.0, Ti=310.0)),
    "R5_fogler_UA2.5": sized(replace(FOGLER, name="R5_fogler_UA2.5", UA=FOGLER.UA * 2.5)),
}
