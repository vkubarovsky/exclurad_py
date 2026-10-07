#!/usr/bin/env python3
"""eta = sigma_obs/sigma_Born at random points of the box, exactly, no Monte Carlo.

    python3 production/eta_scan.py <npoints> <model> [model ...]

Answers one question: is a large stage-2 bound a property of the MODEL or of the
sampler?  A median eta near 1.5 with a bound of 8 says the model is not the
cause.  Written after the Launch generator session measured amp2609 this way and
found pi0n no worse than pi0p, which removed "the neutron channel is
pathological" as an explanation.

THIS IS NOT THE STAGE-2 BOUND, AND THE TWO ARE EASY TO CONFUSE.  The bound is
the weighted maximum of r = eta_exact / min(eta_table, ETA_PROPOSAL_CAP): it
measures how badly the TABLE interpolates eta.  This script measures eta
itself.  A heavier eta with a well-resolved table sits under a SMALLER bound,
so the two numbers can move in opposite directions and both be right.  Measured
on the two campaigns:

    model     eta 90th pct (pi0p)   bound    b_viol rate      worst excess
    amp2609          4.42           1.947      0.0044 %          x1.87
    amp2613          ~              2.64       0.0147 %          x5.41
    amp2614          5.52           1.819      0.0038 %          x1.69

amp2614 has the heaviest eta of the three and the lowest violation rate.  Read
this scan to decide whether a model is samplable at all; read b_viol out of
prov/<mode>_<NNN>.config.json to decide whether a finished sample is exactly
unweighted.  Do not substitute one for the other.

SAMPLE SIZE.  The rng is seeded, so a smaller N is a prefix of the same point
set and the percentiles converge rather than scatter.  They converge slowly:
pi0.amp2614 gives 5.03 at N=300, 5.43 at N=500, 5.52 at N=1200.  N=220 gave
4.96, which was quoted in a decision and was wrong by half a unit.  Use 1000 or
more for a number anyone will act on.
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
