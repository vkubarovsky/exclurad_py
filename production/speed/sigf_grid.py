"""Reference grid for sigma_F: bit-exactness harness for the speed work.

Computes sigma_F at a grid of points spanning the production box and writes
the raw float64 BITS (not a formatted number) so that two revisions can be
compared for bit-identity, not merely for agreement to 1e-12.

    python3 -m production.speed.sigf_grid dump  out.npz
    python3 -m production.speed.sigf_grid cmp   a.npz b.npz
"""
import sys
import numpy as np

from exclurad_py import models
from exclurad_py.core.kinematics import EventKin
from exclurad_py.core.radiative import sigma_F

EBEAM = 10.6
MODEL = "pi0.amp2609"
T_NUCL = -0.30


def grid_points():
    """Points spanning the production card's box (W2 3.24-16, Q2 0.8-10,
    t -2.5..-0.001), plus the selftest point.  Kept deterministic."""
    pts = []
    for W2 in (3.6, 5.0, 8.0, 12.0, 15.0):
        for Q2 in (1.0, 2.0, 4.0, 7.0):
            for t in (-0.15, -0.5, -1.2, -2.2):
                for phi in (0.3, 2.1, 4.7):
                    pts.append((Q2, W2, t, phi))
    return pts


def dump(path):
    sf = models.get(MODEL, t_nucl=T_NUCL)
    vals, keep = [], []
    for (Q2, W2, t, phi) in grid_points():
        try:
            lim = EventKin.t_limits(Q2, W2, sf.mV, sf.mU)
            if lim is None or not (min(lim) <= t <= max(lim)):
                continue
            kin = EventKin(EBEAM, Q2, W2, t, phi, sf.mV, mU=sf.mU)
            v = sigma_F(kin, sf, +1, 0.20)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        vals.append(v)
        keep.append((Q2, W2, t, phi))
    vals = np.array(vals, dtype=np.float64)
    np.savez(path, vals=vals, pts=np.array(keep, dtype=np.float64))
    print(f"{len(vals)} points -> {path}")
    return vals


def cmp(a, b):
    A, B = np.load(a), np.load(b)
    va, vb = A["vals"], B["vals"]
    if va.shape != vb.shape:
        print(f"SHAPE MISMATCH {va.shape} vs {vb.shape}")
        return 1
    ba = va.view(np.uint64)
    bb = vb.view(np.uint64)
    nbit = int(np.count_nonzero(ba != bb))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(np.where(vb != 0, va / vb - 1.0, va - vb))
    print(f"points          : {len(va)}")
    print(f"bit-different   : {nbit}")
    print(f"max |rel diff|  : {np.nanmax(rel):.3e}")
    if nbit:
        i = int(np.argmax(rel))
        p = A["pts"][i]
        print(f"worst at Q2={p[0]} W2={p[1]} t={p[2]} phi={p[3]}: "
              f"{va[i]!r} vs {vb[i]!r}")
    return 0 if nbit == 0 else 2


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        sys.exit(cmp(sys.argv[2], sys.argv[3]))
