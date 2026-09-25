"""LUND -> per-event arrays (W2, Q2, -t, v), parsed where the files live.

Reduce first, move the arrays, plot afterwards: a million LUND events are
~150 MB of text but 32 MB of float64, and the parse is local I/O.

    python3 production/lund_kinematics.py "<glob>" out.npz <workers>

v = MX^2(e'p) - m_pi0^2 is the variable an offline missing-mass cut acts on;
see production/shape_steps.py for what it is used for.
"""
import glob, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
M_P, M_PI0, EB = 0.93827208816, 0.1349768, 10.604

def one(path):
    o = []
    with open(path) as fh:
        for line in fh:
            tk = line.split()
            if len(tk) != 10:
                continue
            e = p = None
            for _ in range(int(tk[0])):
                r = next(fh).split()
                pid = int(r[3]); f4 = tuple(float(r[i]) for i in (6, 7, 8, 9))
                if pid == 11 and e is None: e = f4
                elif pid == 2212 and p is None: p = f4
            if e is None or p is None:
                continue
            nu = EB - e[3]
            q2 = 2 * EB * e[3] - 2 * EB * e[2]
            w2 = M_P ** 2 + 2 * M_P * nu - q2
            mt = -((p[3] - M_P) ** 2 - p[0] ** 2 - p[1] ** 2 - p[2] ** 2)
            mx = (-e[0] - p[0], -e[1] - p[1], EB - e[2] - p[2], EB + M_P - e[3] - p[3])
            mx2 = mx[3] ** 2 - mx[0] ** 2 - mx[1] ** 2 - mx[2] ** 2
            o.append((w2, q2, mt, mx2 - M_PI0 ** 2))
    return np.array(o) if o else np.zeros((0, 4))

if __name__ == "__main__":
    fs = sorted(glob.glob(sys.argv[1]))
    with ProcessPoolExecutor(int(sys.argv[3])) as pool:
        a = np.concatenate([r for r in pool.map(one, fs) if len(r)])
    np.savez_compressed(sys.argv[2], W2=a[:,0], Q2=a[:,1], mt=a[:,2], v=a[:,3])
    print(f"  {len(fs)} файлов, {len(a)} событий -> {sys.argv[2]}")
    print(f"  v: медиана {np.median(a[:,3]):.5f}  макс {a[:,3].max():.4f}"
          f"   доля v>0.17: {100*(a[:,3]>0.17).mean():.2f}%")
