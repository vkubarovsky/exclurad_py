"""Bit-identity across CHANNELS, helicities and the polarised branch.

`sigf_grid.py` checks the pi0 rad card the brief named.  But the speed work
touches SHARED core code -- spinors, hadronic, radiative -- which every
channel goes through, so a regression could hide in eta, in a vector meson,
or in the lam != 0 branch of `hadronic_tensor` that the pi0 card never
exercises.  This covers those.

    python3 -m production.speed.channels_grid dump out.npz
    python3 -m production.speed.channels_grid cmp a.npz b.npz
"""
import sys
import numpy as np

from exclurad_py import models
from exclurad_py.core.kinematics import EventKin
from exclurad_py.core.radiative import sigma_F

# (model, t_nucl, lam).  lam != 0 selects the POL_KEYS branch of the
# hadronic tensor -- four extra structures the unpolarised card never builds.
CASES = [
    ("pi0.amp2609", -0.30, 0.0), ("pi0.vpk2021", -0.30, 0.0),
    ("eta.amp2609", -0.30, 0.0), ("eta.tableB1", -0.30, 0.0),
    ("pi0.amp2609", -0.30, 0.8), ("eta.amp2609", -0.30, 0.8),
    ("phi.diffrad_vm", None, 0.0), ("rho.diffrad_vm", None, 0.0),
    ("jpsi.diffrad_vm", None, 0.0),
]
# J/psi needs W2 above (3.097+0.938)^2 = 16.3, so it gets its own points;
# without them it silently contributes zero values and looks "checked".
PTS_LIGHT = [(2.0, 6.0, -0.5, 0.7), (4.0, 9.0, -1.0, 2.3), (1.5, 4.5, -0.3, 4.0),
             (6.0, 12.0, -1.6, 5.2), (1.0, 14.0, -0.2, 1.1)]
PTS_HEAVY = [(0.5, 18.0, -0.6, 0.4), (1.5, 19.0, -1.5, 3.5),
             (3.0, 19.8, -0.6, 0.4), (1.5, 18.0, -0.6, 3.5)]


def dump(path):
    out = {}
    for name, tn, lam in CASES:
        try:
            sf = models.get(name, t_nucl=tn) if tn is not None else models.get(name)
        except Exception:
            continue
        pts = PTS_HEAVY if name.startswith("jpsi") else PTS_LIGHT
        vals = []
        for (Q2, W2, t, phi) in pts:
            lim = EventKin.t_limits(Q2, W2, sf.mV, sf.mU)
            if lim is None or not (min(lim) <= t <= max(lim)):
                continue
            try:
                kin = EventKin(10.6, Q2, W2, t, phi, sf.mV, mU=sf.mU)
                for h in (+1, -1):
                    v = sigma_F(kin, sf, h, 0.20, lam=lam)
                    if np.isfinite(v):
                        vals.append(v)
            except Exception:
                continue
        out[f"{name}|lam={lam}"] = np.array(vals, dtype=np.float64)
    np.savez(path, **out)
    for k, v in out.items():
        print(f"  {k:26s} {len(v):3d} values")
    if any(len(v) == 0 for v in out.values()):
        print("  WARNING: a case produced no points -- it is NOT being checked")


def cmp(a, b):
    A, B = np.load(a), np.load(b)
    tot = bad = 0
    for k in A.files:
        x, y = A[k], B[k]
        d = -1 if x.shape != y.shape else int(
            np.count_nonzero(x.view(np.uint64) != y.view(np.uint64)))
        tot += len(x)
        bad += max(d, 0)
        print(f"  {k:26s} {len(x):3d} values, {d} bits differ")
    print(f"TOTAL {tot} values, {bad} bit-different")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        sys.exit(cmp(sys.argv[2], sys.argv[3]))
