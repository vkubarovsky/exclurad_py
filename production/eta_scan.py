#!/usr/bin/env python3
"""eta = sigma_obs/sigma_Born at random points of the box, exactly, no Monte Carlo.

    python3 production/eta_scan.py <npoints> <model> [model ...]

Answers one question: is a large stage-2 bound a property of the MODEL or of the
sampler?  A median eta near 1.5 with a bound of 8 says the model is not the
cause.  Written after the Launch generator session measured amp2609 this way and
found pi0n no worse than pi0p, which removed "the neutron channel is
pathological" as an explanation.
"""
import os, sys
import numpy as np
from exclurad_py import models
from exclurad_py.core.rc import rc_factor

N = int(sys.argv[1]); NAMES = sys.argv[2:]
# the RG-A production box
Q2LO, Q2HI, W2LO, W2HI, TLO = 0.80, 10.0, 3.24, 16.0, -2.50
rng = np.random.default_rng(20261005)
pts = []
while len(pts) < N:
    q = rng.uniform(Q2LO, Q2HI); w = rng.uniform(W2LO, W2HI); t = rng.uniform(TLO, -0.001)
    pts.append((q, w, t))

print(f"{N} points, Q2 {Q2LO}-{Q2HI}, W2 {W2LO}-{W2HI}, t {TLO}..-0.001, Ebeam 10.6, h=+1")
print(f"{'model':18} {'n ok':>5} {'median':>8} {'90%':>8} {'99%':>8} {'max':>9} {'eta>5':>7}")
for nm in NAMES:
    m = models.get(nm, t_nucl=-0.30)
    e = []
    for q, w, t in pts:
        try:
            d = rc_factor(10.6, q, w, t, 0.0, m, h=+1)
            v = d["eta"]
            if np.isfinite(v) and v > 0: e.append(v)
        except Exception:
            continue
    if not e: print(f"{nm:18} {'--':>5}"); continue
    e = np.array(e)
    print(f"{nm:18} {len(e):5d} {np.median(e):8.2f} {np.percentile(e,90):8.2f} "
          f"{np.percentile(e,99):8.2f} {e.max():9.1f} {100*(e>5).mean():6.0f} %")
