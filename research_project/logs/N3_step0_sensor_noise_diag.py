# READ-ONLY diagnostic (not project code): isolate the sensor noise exactly. Same seed => same process-noise
# path and same sensor-noise key; N3 minus (sensor sigma ~ 0) leaves only the injected sensor noise.
import os, sys, numpy as np
R="/Users/simo/inso-po-RD/cstr-model-optimisation/research_project"
sys.path.insert(0,R+"/src"); sys.path.insert(0,R+"/scripts"); os.environ["CSTR_SENSOR_MODEL"]="N3"
import system_I_config as CFG, ws_eval as we
theta=np.tile(np.array([[0.85,0.85]],np.float32),(200,1))
print("expected N3 sigma:",CFG.SENSOR_MODELS["N3"])
for name,fn in [("closed P1b",lambda **k: we.simulate(theta,"P1b",seed=11,raw=True,**k)),("open loop",lambda **k: we.simulate_open_loop(theta,seed=11,raw=True,**k))]:
    x3=fn(**CFG.noise_kw()); x_tiny=fn(noise_abs=(1e-12,1e-12,1e-12,1e-12))
    d=x3-x_tiny
    print(f"{name:11s} sd(N3 - noiseless-sensor, same seed) per channel [C,T,Tc,Qc] =",np.round(d.std(axis=(0,1)),4),
          "| total sd of Qc (process + sensor) =",round(float(x3[...,3].std(axis=1).mean()),2),
          "| mean |first-diff|-based Qc sd from sensor-free run =",round(float(np.sqrt(np.var(np.diff(x_tiny,axis=1),axis=(0,1))/2)[3]),3))
