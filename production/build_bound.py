#!/usr/bin/env python3
"""Build the majorant and a STRATIFIED stage-2 bound, once, for a box.

    python3 production/build_bound.py card.txt out.npz <workers> <probes per cell>

The MODE comes from the card (`iborn`), not from an argument, because it is
part of the majorant's identity: `_maj_key` carries `born_only`, so a born run
CANNOT use a rad majorant.  It does not say so -- the majorant cache falls
through silently on a key mismatch and rebuilds, unlike the eta table, which
raises.  Pointing both modes at one file therefore costs 25 s in every one of
200 born jobs and nothing warns you.  Build one file per mode.

For a born card the stage-2 probing is skipped: stage 2 IS the RC rejection,
and `B` is not read when `born_only` is set.

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


MAXCELL = int(os.environ.get("B_MAXCELL", "400"))
MINUSE = int(os.environ.get("B_MINUSE", "20"))


def b_shape_for_budget(shape, maxcell):
    """The finest bound grid, at the majorant's aspect, within `maxcell` cells.

    NOT `Generator._resolve_b_shape`.  That one sizes the grid from the
    GENERATOR's probe budget (B_PROBES, default 300, at 20 probes per cell =
    15 cells) and, worse, calibrate() has already called it by the time this
    script asks -- so it returns the cached answer and ignores the per_cell
    we pass.  On the production box that gave (1,1,1,2): two cells, which is
    a global bound wearing a grid's clothes.

    This script's budget is its own: `ncell x PER` probes over NW workers,
    both given on the command line.  So the grid is chosen here, from the
    number of cells we are willing to probe, and written into the npz -- the
    jobs restore it from there and never re-resolve it.
    """
    for f in range(1, max(shape) + 1):
        cand = tuple(max(1, -(-m // f)) for m in shape)
        if int(np.prod(cand)) <= maxcell:
            return cand
    return tuple(1 for _ in shape)


def build(seed, maj_cache=None):
    cfg = parse(CARD)
    kw = dict(t_nucl=cfg["t_nucl"]) if _needs_t_nucl(cfg["model"]) else {}
    sf = models.get(cfg["model"], **kw)
    sf.Ebeam = float(cfg["ebeam"])
    mx2 = cfg["vcut"] + sf.mU ** 2 if cfg["vcut"] > 0 else None
    # theta_e and fermi are NOT decoration here.  Both are in _maj_key or
    # change what the majorant describes, and neither was passed: a majorant
    # built without `fermi` is a free-nucleon majorant, and a deuteron run
    # would have rejected it on the key and rebuilt silently -- 200 times.
    g = Generator(sf, Ebeam=cfg["ebeam"],
                  box=Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"]),
                  mx2_cut=mx2, helicity=cfg["ihel"], ptarg=cfg.get("ptarg", 0.0),
                  seed=seed, born_only=bool(cfg["iborn"]),
                  cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"],
                            theta_e=cfg.get("theta_e", (0.0, 180.0))),
                  itarg=cfg["itarg"], vz=cfg["vz"], v_min=cfg["vmin"] or None,
                  fermi=cfg.get("fermi", "none"),
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
    # `best` alone is not enough.  A cell where 3 of 400 draws survived the
    # guards gets a bound from 3 samples -- the same lottery as before, just
    # confined to one cell, and THAT is where a violation comes from.  So the
    # usable count travels with the max and decides whether it is trusted.
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
        best, nuse = 0.0, 0
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
                nuse += 1
                best = max(best, so / eta)
        out[flat] = (best, nuse)
    return out, nfail


if __name__ == "__main__":
    t0 = time.time()
    g0 = build(950001)
    g0.calibrate(n_probe=30000, margin=2.0, verbose=False)
    if g0.born_only:
        g0.save_majorant(OUT)
        print(f"  born majorant {g0.majorant_shape} -> {OUT}"
              f"   {time.time()-t0:.0f} s   (no stage-2 bound: born has no stage 2)")
        sys.exit(0)
    g0.save_majorant(OUT + ".maj.npz")
    ms = g0.majorant_shape
    bshape = (tuple(int(x) for x in BSHAPE.split(",")) if BSHAPE
              else b_shape_for_budget(ms, MAXCELL))
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
    nuse = np.zeros(ncell, dtype=int)
    nfail = 0
    for d, nf in res:
        nfail += nf
        for k, (v, n) in d.items():
            bmax[k] = max(bmax[k], v)
            nuse[k] += n
    bmax, nuse = bmax.reshape(bshape), nuse.reshape(bshape)
    # A bound is a max over probes, so it is biased LOW when there are few.
    # Cells under MINUSE keep the global fallback instead of a number drawn
    # from a handful of draws.  Measured on the -2.5 box at 400 probes per
    # cell: a third of the grid is kinematically EMPTY (0 usable probes, and
    # x10 more probes bought only 18 of them back), so this is not a budget
    # problem and cannot be bought out -- it has to be handled.
    ok = nuse >= MINUSE
    Bg = float(max(bmax.max() * MARGIN, 1.05))
    B = np.where(ok, bmax * MARGIN, Bg)
    B_nomin = np.where(nuse > 0, bmax * MARGIN, Bg)   # the rule switched off
    print(f"  usable probes per cell: 0 in {int((nuse == 0).sum())} cells, "
          f"1-{MINUSE-1} in {int(((nuse > 0) & ~ok).sum())}, "
          f">={MINUSE} in {int(ok.sum())} of {ncell}"
          f"   rc_factor failures: {nfail}")
    print(f"  cells on the global fallback: {int((~ok).sum())} of {ncell}")
    # NOT max(B).  It is identically Bg: every measured cell gets
    # bmax*MARGIN <= bmax.max()*MARGIN = Bg and every fallback cell gets Bg,
    # so "max B" is one probe -- the largest r found anywhere -- times the
    # margin, and it moved 12.72 / 9.79 / 9.87 across three runs of the same
    # box for that reason.  It is not a property of the grid and comparing
    # runs by it compares one sample point.  Reported as what it is.
    print(f"  B over cells: min {B.min():.2f}  median {np.median(B):.2f}  "
          f"mean {B.mean():.2f}")
    print(f"  fallback Bg = {Bg:.2f}  (largest single probe x {MARGIN}, "
          f"not a grid property -- expect it to move run to run)")

    # THE number, and the only one worth deciding on.  A median over CELLS
    # counts the empty corner of the box the same as the cell every event
    # lands in, so it moves a lot while nothing that matters moves at all.
    # Weight by the majorant mass -- M x cell volume, which is what the
    # proposal draws from -- and the comparison is between two bounds the
    # generator will actually meet.
    e = g0.maj["edges"]
    vol = 1.0
    for ax, k in enumerate(("Q2", "W2", "t", "phi")):
        d = np.diff(e[k])
        vol = vol * d.reshape([-1 if i == ax else 1 for i in range(4)])
    w = np.asarray(g0.maj["M"]) * vol
    idx = np.ix_(*[np.minimum((np.arange(ms[a]) * bshape[a]) // ms[a],
                              bshape[a] - 1) for a in range(4)])
    tot = float(w.sum())
    for name, arr in (("with the >=%d rule" % MINUSE, B), ("rule off", B_nomin)):
        print(f"  proposal-weighted bound, {name:>18}: "
              f"{float((w * arr[idx]).sum()) / tot:.3f}")
    print(f"  majorant mass sitting on the global fallback: "
          f"{100.0 * float(w[np.where(~ok[idx], 1.0, 0.0) > 0].sum()) / tot:.2f} %")

    d0 = np.load(OUT + ".maj.npz", allow_pickle=True)
    np.savez_compressed(OUT, M=d0["M"], cdf=d0["cdf"], shape=d0["shape"],
                        eta_hi=d0["eta_hi"], key=d0["key"], B=B, B_global=Bg,
                        b_shape=np.array(bshape),
                        diag_bmax=bmax, diag_nuse=nuse,
                        **{k: d0[k] for k in d0.files if k.startswith("e_")})
    os.remove(OUT + ".maj.npz")
    print(f"  wrote {OUT}   total {time.time()-t0:.0f} s")
