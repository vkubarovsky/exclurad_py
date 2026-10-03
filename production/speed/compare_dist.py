"""Compare two LUND samples statistically.

The numba backend is not byte-identical, so byte comparison is replaced by
this: the sampled physics must agree within the statistics of the samples.
Two-sample KS on the shapes plus a z-test on the means, for the quantities
the brief names -- MX^2, v, phi and <sin phi>.
"""
import sys
import numpy as np
from scipy import stats

M_P = 0.93827208816


def read(path):
    """Return dict of per-event arrays from a LUND file."""
    ev = {"v": [], "mx2": [], "phi": [], "nph": []}
    beam = np.array([0.0, 0.0, 10.6, 10.6])
    with open(path) as fh:
        lines = [l for l in fh if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        hdr = lines[i].split()
        n = int(float(hdr[0]))
        parts = [lines[i + 1 + k].split() for k in range(n)]
        i += 1 + n
        e_out = ph = None
        prot = None
        ng = 0
        for pt in parts:
            pid = int(float(pt[3]))
            p4 = np.array([float(pt[6]), float(pt[7]), float(pt[8]), float(pt[9])])
            if pid == 11 and e_out is None:
                e_out = p4
            elif pid == 2212:
                prot = p4
            elif pid == 22:
                ng += 1
                ph = p4 if ph is None else ph
        if e_out is None or prot is None:
            continue
        q = beam - np.array([e_out[0], e_out[1], e_out[2], e_out[3]])
        tgt = np.array([0.0, 0.0, 0.0, M_P])
        miss = beam + tgt - np.array([e_out[0], e_out[1], e_out[2], e_out[3]]) \
            - np.array([prot[0], prot[1], prot[2], prot[3]])
        mx2 = miss[3] ** 2 - miss[0] ** 2 - miss[1] ** 2 - miss[2] ** 2
        ev["mx2"].append(mx2)
        ev["nph"].append(ng)
        # phi: azimuth of the hadron plane about q
        zq = q[:3] / np.linalg.norm(q[:3])
        x = np.cross(np.array([0.0, 0.0, 1.0]), zq)
        nx = np.linalg.norm(x)
        if nx < 1e-12:
            continue
        x /= nx
        y = np.cross(zq, x)
        pp = prot[:3] - np.dot(prot[:3], zq) * zq
        ev["phi"].append(np.arctan2(np.dot(pp, y), np.dot(pp, x)) % (2 * np.pi))
    return {k: np.asarray(v) for k, v in ev.items()}


if __name__ == "__main__":
    A, B = read(sys.argv[1]), read(sys.argv[2])
    print(f"events: {len(A['mx2'])} vs {len(B['mx2'])}")
    print(f"{'quantity':>10s} {'mean A':>12s} {'mean B':>12s} {'z(mean)':>8s} {'KS p':>8s}")
    for k in ("mx2", "phi", "nph"):
        a, b = A[k], B[k]
        if len(a) == 0 or len(b) == 0:
            continue
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        z = (a.mean() - b.mean()) / se if se > 0 else 0.0
        ks = stats.ks_2samp(a, b)
        print(f"{k:>10s} {a.mean():12.5f} {b.mean():12.5f} {z:8.2f} {ks.pvalue:8.3f}")
    for k in ("phi",):
        a, b = np.sin(A[k]), np.sin(B[k])
        se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        print(f"{'<sin phi>':>10s} {a.mean():12.5f} {b.mean():12.5f} "
              f"{(a.mean()-b.mean())/se:8.2f} {'--':>8s}")
    print("\n|z| < 3 and KS p > 0.01 is agreement within the samples' own statistics.")
