"""Where does the eta table fail, and what does it cost?

r = eta_exact / min(eta_table, cap) is exactly what calibrate() probes:
B_cell = 1.3 * max(r) over the cell.  A large global B means some region needs
a loose bound; this says WHICH region, and how much Born cross section lives
there -- i.e. whether the whole sample should pay for it.

    python3 production/bound_map.py card.txt <n_points> <workers>

On the 2026-09-25 wide box it showed r > 1.5 at 7.4 % of points carrying only
0.4 % of the Born cross section: the reason a per-cell bound is worth keeping
(a global B would have to be 8.75) and therefore the reason the proposal has
to carry B_cell rather than the acceptance alone.  See commit 225d4d9.
"""
import os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from exclurad_py import models
from exclurad_py.generator import Generator, Box
from exclurad_py.core.rc import rc_factor
from exclurad_py.run import _needs_t_nucl
from exclurad_py.runcard import parse

CARD, NTOT, NW = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def make():
    cfg = parse(CARD)
    kw = dict(t_nucl=cfg["t_nucl"]) if _needs_t_nucl(cfg["model"]) else {}
    sf = models.get(cfg["model"], **kw)
    sf.Ebeam = float(cfg["ebeam"])
    mx2 = cfg["vcut"] + sf.mU ** 2 if cfg["vcut"] > 0 else None
    g = Generator(sf, Ebeam=cfg["ebeam"],
                  box=Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"]),
                  mx2_cut=mx2, helicity=cfg["ihel"], ptarg=cfg.get("ptarg", 0.0),
                  seed=cfg["seed"], born_only=bool(cfg["iborn"]),
                  cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"]),
                  itarg=cfg["itarg"], vz=cfg["vz"], v_min=cfg["vmin"] or None,
                  maj_cache=os.environ.get("MAJ_TABLE"),
                  cache=os.environ.get("ETA_TABLE"))
    g.setup(verbose=False)
    return g


def one(seed):
    g = make()
    rng = np.random.default_rng(seed)
    lQ = np.log(np.abs(g.box.Q2)); lt = np.log(np.abs(g.box.t))
    out = []
    for _ in range(max(NTOT // NW, 1)):
        Q2 = float(np.exp(rng.uniform(*sorted(lQ))))
        W2 = float(rng.uniform(*g.box.W2))
        t = -float(np.exp(rng.uniform(*sorted(lt))))
        ph = float(rng.uniform(0.0, 2 * np.pi))
        try:
            g._set_event_t(t)
            ex = float(rc_factor(M=g._mT, Ebeam=g._E, Q2=Q2, W2=W2, t=t, phi=ph,
                                 sfmodel=g.sf, h=+1, lam=g.ptarg,
                                 mx2_cut=g.mx2_cut, t_kind=g._t_kind,
                                 mV=g._mV)["eta"])
            ip = float(g.table(Q2, W2, t, ph)[0])
            b = g._born(Q2, W2, t, ph, +1)
        except Exception:
            continue
        if not (np.isfinite(ex) and np.isfinite(ip)) or ip <= 0:
            continue
        if b is None or not np.isfinite(b) or b <= 0:
            b = 0.0
        cap = g.eta_proposal_cap
        ipc = min(ip, cap) if cap else ip
        out.append((Q2, W2, t, ph, ex, ip, ex / ipc, b))
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(NW) as pool:
        res = sum(pool.map(one, range(1000, 1000 + NW)), [])
    a = np.array(res)
    Q2, W2, t, ph, e, i, r, b = a.T
    w = b / b.sum() if b.sum() > 0 else np.ones(len(b)) / len(b)
    print(f"\n  {len(a)} точек в боксе")
    print(f"  eta точное   : {e.min():.3f} .. {e.max():.3f}   медиана {np.median(e):.3f}")
    print(f"  eta таблица  : {i.min():.3f} .. {i.max():.3f}   медиана {np.median(i):.3f}")
    print(f"  r = eta_ex / min(eta_tab, cap):")
    print(f"      медиана {np.median(r):.3f}   95% {np.quantile(r,0.95):.3f}"
          f"   макс {r.max():.3f}   ->  B = {1.3*r.max():.2f}")
    print(f"      взвешенная по борну медиана {np.sum(w*r)/1:.3f}\n")
    for name, v, ed in (("Q2", Q2, [0.8, 1.15, 2, 3, 5, 10.01]),
                        ("-t", -t, [0.001, 0.1, 0.3, 0.6, 1.0, 1.8, 2.001]),
                        ("W", np.sqrt(W2), [1.8, 2.2, 2.6, 3.0, 3.5, 4.001])):
        print(f"  r по {name}:")
        for lo, hi in zip(ed[:-1], ed[1:]):
            m = (v >= lo) & (v < hi)
            if m.sum() > 3:
                print(f"    {lo:6.3f}-{hi:6.3f}  n={m.sum():5d}  медиана {np.median(r[m]):6.3f}"
                      f"  макс {r[m].max():7.2f}   доля борна {100*b[m].sum()/b.sum():5.1f}%")
        print()
    for thr in (1.5, 2.0, 3.0):
        m = r > thr
        print(f"  r > {thr}: {m.sum():5d} точек ({100*m.mean():4.1f}%), "
              f"несут {100*b[m].sum()/b.sum():4.1f}% борновского сечения")
