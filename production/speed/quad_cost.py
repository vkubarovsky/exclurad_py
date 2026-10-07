"""What a coarser photon quadrature costs, in eta, where events actually land.

The radiative correction needs an integral over the emitted photon's energy
and direction.  It is done on a grid: (n_v, n_core, n_tail, n_phi), default
(20,12,16,12).  A coarser grid is proportionally faster and slightly less
accurate.  Unlike the two stage-2 levers, this one DOES change the physics,
so it is measured and never adopted here.

Accuracy is quoted as |d eta| against a dense reference, weighted by the
Born cross section -- i.e. by how many events actually land at that point,
which is the only weighting that means anything for a sample.

    python3 -m production.speed.quad_cost [n_points]
"""
import sys, time
import numpy as np

from exclurad_py import models
from exclurad_py.core.kinematics import EventKin
from exclurad_py.core.rc import rc_factor

EBEAM, MODEL, T_NUCL = 10.6, "pi0.amp2609", -0.30
REFERENCE = (50, 24, 40, 24)
CANDIDATES = [(20, 12, 16, 12),      # production today
              (16, 10, 14, 10),
              (15, 10, 12, 10),      # the proposal
              (12, 8, 10, 8)]


def box_points(n, seed=0):
    """Points drawn over the production box (card rad_pi0_rga)."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        Q2 = np.exp(rng.uniform(np.log(0.8), np.log(10.0)))
        W2 = rng.uniform(3.24, 16.0)
        phi = rng.uniform(0.0, 2 * np.pi)
        sf = _SF
        lim = EventKin.t_limits(Q2, W2, sf.mV, sf.mU)
        if lim is None:
            continue
        tlo, thi = min(lim), max(lim)
        lo, hi = max(tlo, -2.5), min(thi, -0.001)
        if not (lo < hi):
            continue
        pts.append((Q2, W2, rng.uniform(lo, hi), phi))
    return pts


if __name__ == "__main__":
    _SF = models.get(MODEL, t_nucl=T_NUCL)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    pts = box_points(n)
    print(f"{n} points over the production box, reference {REFERENCE}\n")

    ref, wts = [], []
    for (Q2, W2, t, phi) in pts:
        try:
            q = dict(zip(("n_v", "n_core", "n_tail", "n_phi"), REFERENCE))
            r = rc_factor(EBEAM, Q2, W2, t, phi, _SF, +1, **q)
            ref.append(r["eta"]); wts.append(max(r["sigma_born"], 0.0))
        except Exception:
            ref.append(np.nan); wts.append(0.0)
    ref = np.array(ref); wts = np.array(wts)
    ok = np.isfinite(ref) & (wts > 0)
    wts = wts / wts[ok].sum()

    print(f"{'quadrature':>18s} {'points':>7s} {'rel cost':>9s} "
          f"{'wtd mean|d eta|':>16s} {'wtd 95%':>9s} {'worst':>8s}  s/point")
    base_pts = np.prod(REFERENCE)
    for cand in CANDIDATES:
        t0 = time.perf_counter(); cur = []
        for (Q2, W2, t, phi) in pts:
            try:
                q = dict(zip(("n_v", "n_core", "n_tail", "n_phi"), cand))
                cur.append(rc_factor(EBEAM, Q2, W2, t, phi, _SF, +1, **q)["eta"])
            except Exception:
                cur.append(np.nan)
        dt = (time.perf_counter() - t0) / len(pts)
        cur = np.array(cur)
        m = ok & np.isfinite(cur)
        d = np.abs(cur[m] / ref[m] - 1.0)
        w = wts[m] / wts[m].sum()
        order = np.argsort(d)
        c95 = d[order][np.searchsorted(np.cumsum(w[order]), 0.95)]
        print(f"{str(cand):>18s} {np.prod(cand):>7d} "
              f"{np.prod(cand)/base_pts:>8.2f}x {100*np.sum(w*d):>15.3f}% "
              f"{100*c95:>8.3f}% {100*d.max():>7.2f}%  {dt:6.3f}")
