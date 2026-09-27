#!/usr/bin/env python3
"""Build the majorant and a STRATIFIED stage-2 bound, once, for a box.

    python3 production/build_bound.py card.txt out.npz <workers> <probes per cell>

Why stratified.  calibrate() probes the bound by drawing from the PROPOSAL,
which is concentrated where the cross section is.  Cells in the tail are
therefore never probed however many probes you buy: 20000 draws over 384
cells left the median cell unmeasured, taking the global fallback of 8.67,
which turns a per-cell bound into a global one and costs a factor three in
stage-2 acceptance for nothing.  Here every cell is probed the same number of
times, by construction, so B is tight where the events are (about 1.5) and
large only in the high-Q2/high-W corner that needs it.

Why it exists at all.  B is a property of the box and was being rebuilt from
300 proposal-drawn probes inside every job -- 200 independent lotteries whose
outcome, measured across 100 workers of 200 probes, ranged from 2.01 to 8.67
against a true maximum of 8.67.  The 2026-09-25 production duly exceeded the
bound in 198 of 200 chunks, worst x3.35.  A breach is not harmless: the event
is accepted with probability 1 instead of r, so that point is
under-represented by 1/r.  See events.py:1258 for the x45 precedent.

r is evaluated exactly as the generator evaluates it -- eta_exact divided by
the capped table value, at a drawn vertex energy -- so the number this script
writes is the number the acceptance will compare against.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from exclurad_py import models
from exclurad_py.generator import Generator, Box
from exclurad_py.core.rc import rc_factor
from exclurad_py.run import _needs_t_nucl
from exclurad_py.runcard import parse

CARD, OUT, NW, PER = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
BSHAPE = sys.argv[5] if len(sys.argv) > 5 else ""
MARGIN = 1.3


def build(seed, maj_cache=None):
    cfg = parse(CARD)
    kw = dict(t_nucl=cfg["t_nucl"]) if _needs_t_nucl(cfg["model"]) else {}
    sf = models.get(cfg["model"], **kw)
    sf.Ebeam = float(cfg["ebeam"])
    mx2 = cfg["vcut"] + sf.mU ** 2 if cfg["vcut"] > 0 else None
    g = Generator(sf, Ebeam=cfg["ebeam"],
                  box=Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"]),
                  mx2_cut=mx2, helicity=cfg["ihel"], ptarg=cfg.get("ptarg", 0.0),
                  seed=seed, born_only=False,
                  cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"]),
                  itarg=cfg["itarg"], vz=cfg["vz"], v_min=cfg["vmin"] or None,
                  maj_cache=maj_cache, cache=os.environ.get("ETA_TABLE"))
    g.setup(verbose=False)
    return g


def probe_cells(arg):
    """Max of r over PER uniform draws in each of the assigned bound cells."""
    lo, hi, bshape, seed = arg
    g = build(seed, maj_cache=OUT + ".maj.npz")
    e = g.maj["edges"]
    ms = g.majorant_shape
    rng = np.random.default_rng(seed)
    out, nfail = {}, 0
    for flat in range(lo, hi):
        bc = np.unravel_index(flat, bshape)
        # the majorant cells this bound cell covers, via the same integer map
        rngs = []
        for ax, k in enumerate(("Q2", "W2", "t", "phi")):
            idx = [i for i in range(ms[ax]) if (i * bshape[ax]) // ms[ax] == bc[ax]]
            if not idx:
                idx = [min(ms[ax] - 1, bc[ax])]
            rngs.append((e[k][min(idx)], e[k][max(idx) + 1]))
        best = 0.0
        for _ in range(PER):
            if not g._draw_vertex():
                continue
            u = rng.random(4)
            Q2 = rngs[0][0] + u[0] * (rngs[0][1] - rngs[0][0])
            W2 = rngs[1][0] + u[1] * (rngs[1][1] - rngs[1][0])
            t = rngs[2][0] + u[2] * (rngs[2][1] - rngs[2][0])
            ph = rngs[3][0] + u[3] * (rngs[3][1] - rngs[3][0])
            if not g._pass_yxb(Q2, W2):
                continue
            if not g._draw_mass(Q2, W2, t):
                continue
            h = +1 if rng.random() < 0.5 else -1
            b = g._born(Q2, W2, t, ph, h)
            if b is None or not np.isfinite(b) or b <= 0:
                continue
            eta = float(g.table(Q2, W2, t, ph)[0])
            if g.eta_proposal_cap:
                eta = min(eta, g.eta_proposal_cap)
            try:
                g._set_event_t(t)
                so = float(rc_factor(M=g._mT, Ebeam=g._E, Q2=Q2, W2=W2, t=t, phi=ph,
                                     sfmodel=g.sf, h=h, lam=g.ptarg,
                                     mx2_cut=g.mx2_cut, t_kind=g._t_kind,
                                     mV=g._mV)["eta"])
            except Exception:
                nfail += 1
                continue
            if np.isfinite(so) and so > 0:
                best = max(best, so / eta)
        out[flat] = best
    return out, nfail


if __name__ == "__main__":
    t0 = time.time()
    g0 = build(950001)
    g0.calibrate(n_probe=30000, margin=2.0, verbose=False)
    g0.save_majorant(OUT + ".maj.npz")
    ms = g0.majorant_shape
    bshape = (tuple(int(x) for x in BSHAPE.split(",")) if BSHAPE
              else g0._resolve_b_shape(ms, per_cell=1))
    ncell = int(np.prod(bshape))
    print(f"  majorant {ms} built in {time.time()-t0:.0f} s")
    print(f"  bound grid {bshape} = {ncell} cells x {PER} probes = {ncell*PER} total")

    edges = np.linspace(0, ncell, NW + 1).astype(int)
    jobs = [(int(edges[i]), int(edges[i + 1]), bshape, 960000 + i * 7919)
            for i in range(NW) if edges[i + 1] > edges[i]]
    t1 = time.time()
    with ProcessPoolExecutor(len(jobs)) as pool:
        res = list(pool.map(probe_cells, jobs))
    print(f"  probed in {time.time()-t1:.0f} s")

    bmax = np.zeros(ncell)
    nfail = 0
    for d, nf in res:
        nfail += nf
        for k, v in d.items():
            bmax[k] = max(bmax[k], v)
    bmax = bmax.reshape(bshape)
    nz = bmax > 0
    Bg = float(max(bmax.max() * MARGIN, 1.05))
    B = np.where(nz, bmax * MARGIN, Bg)
    print(f"  cells with no usable probe: {int((~nz).sum())} of {ncell}"
          f"   rc_factor failures: {nfail}")
    print(f"  B: min {B.min():.2f}  median {np.median(B):.2f}  "
          f"mean {B.mean():.2f}  max {B.max():.2f}   global {Bg:.2f}")
    print(f"  cells above 3: {int((B > 3).sum())} of {ncell}")

    d0 = np.load(OUT + ".maj.npz", allow_pickle=True)
    np.savez_compressed(OUT, M=d0["M"], cdf=d0["cdf"], shape=d0["shape"],
                        eta_hi=d0["eta_hi"], key=d0["key"], B=B, B_global=Bg,
                        b_shape=np.array(bshape),
                        **{k: d0[k] for k in d0.files if k.startswith("e_")})
    os.remove(OUT + ".maj.npz")
    print(f"  wrote {OUT}   total {time.time()-t0:.0f} s")
