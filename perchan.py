import sys, time, numpy as np
from exclurad_py import models
from exclurad_py.core.kinematics import EventKin
from exclurad_py.core.radiative import sigma_F
CH = [("pi0.amp2609", -0.30, (2.0, 6.0, -0.5, 0.7)),
      ("eta.amp2609", -0.30, (2.0, 6.0, -0.5, 0.7)),
      ("phi.diffrad_vm", None, (2.0, 8.0, -0.5, 0.7)),
      ("rho.diffrad_vm", None, (2.0, 8.0, -0.5, 0.7)),
      ("jpsi.diffrad_vm", None, (1.5, 19.0, -1.5, 0.7))]
for name, tn, (Q2, W2, t, phi) in CH:
    try:
        sf = models.get(name, t_nucl=tn) if tn is not None else models.get(name)
        lim = EventKin.t_limits(Q2, W2, sf.mV, sf.mU)
        if lim is None or not (min(lim) <= t <= max(lim)):
            print(f"{name:18s} skipped (t outside limits)"); continue
        kin = EventKin(10.6, Q2, W2, t, phi, sf.mV, mU=sf.mU)
        sigma_F(kin, sf, +1, 0.20)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); sigma_F(kin, sf, +1, 0.20)
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"{name:18s} {min(ts):8.2f} ms")
    except Exception as e:
        print(f"{name:18s} FAILED {type(e).__name__}: {str(e)[:50]}")
