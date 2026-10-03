"""Timing benchmark for sigma_F -- the dominant cost of rad generation.

Reports the MINIMUM over repeats (least contaminated by other load) and the
median, in ms per sigma_F call.  Always print `uptime` alongside: a number
from a loaded machine is worthless.

    python3 -m production.speed.bench_sigf [repeats]
"""
import os, sys, time, statistics, subprocess
import numpy as np
from exclurad_py import models
from exclurad_py.core.kinematics import EventKin
from exclurad_py.core.radiative import sigma_F, quad_default

PTS = [(2.0, 6.0, -0.5, 0.7), (4.0, 9.0, -1.0, 2.3),
       (1.5, 4.5, -0.3, 4.0), (6.0, 12.0, -1.6, 5.2)]


def main(reps=7):
    sf = models.get("pi0.amp2609", t_nucl=-0.30)
    kins = [EventKin(10.6, Q2, W2, t, phi, sf.mV, mU=sf.mU) for Q2, W2, t, phi in PTS]
    for k in kins:                      # warm
        sigma_F(k, sf, +1, 0.20)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for k in kins:
            sigma_F(k, sf, +1, 0.20)
        ts.append((time.perf_counter() - t0) / len(kins) * 1e3)
    print(f"quad={quad_default()}  numpy={np.__version__}  py={sys.version.split()[0]}")
    print(f"uptime: {subprocess.run(['uptime'],capture_output=True,text=True).stdout.strip()}")
    print(f"sigma_F: min {min(ts):7.2f} ms   median {statistics.median(ts):7.2f} ms"
          f"   max {max(ts):7.2f} ms   (n={reps})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
