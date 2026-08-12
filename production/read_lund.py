"""Minimal LUND reader for the comparison (handles the diffrad_vm samples)."""
import numpy as np

PID_MESON = {"phi": 333, "jpsi": 443, "rho": 113, "pi0": 111, "eta": 221}


def read(path, meson_pid, max_events=None):
    """Return dict of arrays: e' 4-vec, meson 4-vec, proton 4-vec, photon E, weight."""
    e_, me, pr, eg, wt = [], [], [], [], []
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 10:
                continue
            try:
                npart = int(f[0]); w = float(f[9])
            except ValueError:
                continue
            ee = mm = pp = None; g = 0.0
            for _ in range(npart):
                p = fh.readline().split()
                if len(p) < 11:
                    continue
                pid = int(p[3]); status = int(p[2])
                px, py, pz, E = map(float, p[6:10])
                v4 = (E, px, py, pz)
                if pid == 11 and status == 1 and ee is None:
                    ee = v4
                elif pid == meson_pid:
                    mm = v4
                elif pid == 2212:
                    pp = v4
                elif pid == 22:
                    g = max(g, E)
            if ee is None or pp is None:
                continue
            e_.append(ee); pr.append(pp); eg.append(g); wt.append(w)
            me.append(mm if mm is not None else (0., 0., 0., 0.))
            if max_events and len(e_) >= max_events:
                break
    return dict(scattered=np.array(e_), meson=np.array(me),
                recoil=np.array(pr), E_gamma=np.array(eg),
                weight=np.array(wt))


def invariants(d, Ebeam=10.6, M=0.93827208816):
    k1 = np.array([Ebeam, 0., 0., np.sqrt(Ebeam**2 - 0.511e-3**2)])
    k2 = d["scattered"]
    q = k1[None, :] - k2
    g = np.array([1., -1., -1., -1.])
    Q2 = -np.einsum('ni,i,ni->n', q, g, q)
    p = np.array([M, 0., 0., 0.])
    W2 = M*M + 2*np.einsum('i,i,ni->n', p, g, q) - Q2
    pr = d["recoil"]
    dt = p[None, :] - pr
    t = np.einsum('ni,i,ni->n', dt, g, dt)
    return Q2, W2, t
