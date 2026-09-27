#!/usr/bin/env python3
"""Build the majorant AND a well-measured stage-2 bound, once, for a box.

    python3 production/build_bound.py card.txt out_majorant.npz <workers> <probes>

Why this exists.  The stage-2 bound B is a property of the BOX -- it is
1.3 x max over a cell of sigma_exact / (born * eta_table) -- and it was being
rebuilt from 300 fresh probes inside every production job.  Two hundred jobs
therefore ran two hundred independent 300-draw lotteries for the same
quantity, on a grid of 6144 cells, so under 5 % of cells were ever measured.
The 2026-09-25 production came out with the bound exceeded in 198 of 200
chunks, worst x3.35, and the excess is not harmless: where the bound fails
the event is accepted with probability 1 instead of r, so the point is
under-represented by 1/r.  See events.py:1258 for the 2026-08-18 precedent,
where a x45 breach put the delta profile 17-29 % low at high W.

Workers are independent draws of the same quantity, so the combination is an
element-wise maximum -- no averaging, a bound is a max.  Each worker also
re-probes the stage-1 majorant, which is redundant but keeps this script off
the calibrate() code path: it calls the same function the production calls,
rather than a copy of it that can drift.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from exclurad_py import models
from exclurad_py.generator import Generator, Box
from exclurad_py.run import _needs_t_nucl
from exclurad_py.runcard import parse

CARD, OUT, NW, NPROBE = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
BSHAPE = sys.argv[5] if len(sys.argv) > 5 else ""


def _grid_for(total, shape, per_cell=20):
    """Size the grid from the TOTAL probe budget, not one worker's share.

    First run of this script got (1,1,1,2): each worker sized the grid from
    its own 200 probes instead of the 20000 the run had between them, so the
    combination was a two-cell bound -- correct, but throwing away the whole
    point of a per-cell bound, which is to keep B = 8.7 in the corner that
    needs it instead of over the whole box.
    """
    budget = max(1, total // per_cell)
    for f in range(1, max(shape) + 1):
        cand = tuple(max(1, -(-m // f)) for m in shape)
        if int(np.prod(cand)) <= budget:
            return cand
    return tuple(1 for _ in shape)


def make(seed, probes, tmp):
    os.environ["B_PROBES"] = str(probes)
    if BSHAPE:
        os.environ["B_SHAPE"] = BSHAPE
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
                  maj_cache=None, cache=os.environ.get("ETA_TABLE"))
    g.setup(verbose=False)
    g.calibrate(n_probe=30000, margin=2.0, verbose=False)
    g.save_majorant(tmp)
    return tmp


def one(i):
    t0 = time.time()
    # np.savez_compressed appends .npz unless the name already ends in it,
    # so the temp name has to carry the suffix or the file is written next to
    # where it is looked for.
    tmp = f"{OUT}.w{i:03d}.npz"
    make(900000 + i * 7919, max(1, NPROBE // NW), tmp)
    return tmp, time.time() - t0


if __name__ == "__main__":
    if not BSHAPE:
        cfg0 = parse(CARD)
        _ms = os.environ.get("MAJ_SHAPE", "")
        mshape = tuple(int(x) for x in _ms.split(",")) if _ms else (8, 8, 8, 12)
        BSHAPE = ",".join(str(x) for x in _grid_for(NPROBE, mshape))
        print(f"  grid sized from the TOTAL budget: {BSHAPE}")
    print(f"  {NW} workers x {max(1, NPROBE//NW)} probes = {NW*max(1, NPROBE//NW)} total")
    with ProcessPoolExecutor(NW) as pool:
        res = list(pool.map(one, range(NW)))
    files = [f for f, _ in res]
    print(f"  built in {max(t for _, t in res):.0f} s")

    d0 = np.load(files[0], allow_pickle=True)
    if "B" not in d0.files:
        sys.exit("FATAL: workers produced no B -- was EXACT_ACCEPT set?")
    B = np.array(d0["B"], dtype=float)
    Bg = float(d0["B_global"])
    bsh = tuple(int(x) for x in d0["b_shape"])
    for f in files[1:]:
        d = np.load(f, allow_pickle=True)
        assert tuple(int(x) for x in d["b_shape"]) == bsh, "workers disagree on the grid"
        B = np.maximum(B, np.array(d["B"], dtype=float))   # a bound is a max
        Bg = max(Bg, float(d["B_global"]))
    print(f"  grid {bsh} = {int(np.prod(bsh))} cells")
    print(f"  B per cell: min {B.min():.2f}  median {np.median(B):.2f}  max {B.max():.2f}")
    print(f"  B_global:   {Bg:.2f}")
    print(f"  spread across workers tells you whether the probes were enough:")
    per = [float(np.load(f, allow_pickle=True)["B"].max()) for f in files]
    print(f"    worst cell per worker: min {min(per):.2f}  max {max(per):.2f}")

    np.savez_compressed(OUT, M=d0["M"], cdf=d0["cdf"], shape=d0["shape"],
                        eta_hi=d0["eta_hi"], key=d0["key"], B=B, B_global=Bg,
                        b_shape=np.array(bsh),
                        **{k: d0[k] for k in d0.files if k.startswith("e_")})
    for f in files:
        os.remove(f)
    print(f"  wrote {OUT}  ({os.path.getsize(OUT)} bytes)")
