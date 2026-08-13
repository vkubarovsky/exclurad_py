"""
Run exclurad_py from a run card.

    python3 -m exclurad_py.run gen_input.txt [--jobs N]

`--jobs N` splits the requested events across N processes, each with its own
seed, writing N LUND part files exactly as exclugen's OSG runs do
(`..._01.lund` ... `..._NN.lund`).  Parts are statistically independent and
can be concatenated or fed to GEMC separately.  Default: all cores but one.
The eta table is built ONCE in the parent (itself parallel) and cached, so
workers only rebuild the cheap majorant.

Writes, under `outroot`, the exclugen-style layout:

    <outroot>/<channel>/<date>_<Ebeam>_<born|rad>_v1/
        README.md
        inputs/gen_input.txt      <- the run card, copied verbatim
        inputs/config.json        <- parsed card + provenance + run report
        lund/<channel>_exclurad_py_<mode>_<E>GeV.lund

The run card is copied into `inputs/` unchanged so a sample can always be
regenerated from what is sitting next to it.
"""

import json
import os
import shutil
import sys
import time

import numpy as np

from . import models, stamp
from .generator import Generator, Box, write_lund, decay_two_body, DECAY
from .generator.tables import RCTable
from .runcard import parse, echo


def _outdir(cfg, date=None, version="v1"):
    mode = "born" if cfg["iborn"] else "rad"
    ch = cfg["model"].split(".")[0]
    date = date or time.strftime("%Y%m%d")
    d = os.path.join(cfg["outroot"], ch,
                     f"{date}_{cfg['ebeam']:g}_{mode}_{version}")
    for sub in ("lund", "inputs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d, mode, ch


def _worker(payload):
    card_path, date, ipart, npart, seed = payload
    return run(card_path, date=date, verbose=False, _part=(ipart, npart, seed))


def run_parallel(card_path, jobs, date=None, verbose=True):
    """Split the run across `jobs` processes, one LUND part each."""
    from concurrent.futures import ProcessPoolExecutor
    cfg = parse(card_path)
    if verbose:
        echo(cfg)
        print(f" splitting {cfg['nev']} events across {jobs} processes")

    # Build (and cache) the eta table once in the parent -- it is the same
    # for every part, and rebuilding it per worker would dominate.
    if not cfg["iborn"]:
        d, _, _ = _outdir(cfg, date)
        ch = cfg["model"].split(".")[0]
        kw = dict(t_nucl=cfg["t_nucl"]) if ch in ("pi0", "eta") else {}
        sf = models.get(cfg["model"], **kw)
        mx2 = cfg["vcut"] + sf.mU ** 2 if cfg["vcut"] > 0 else None
        cache = os.path.join(d, "etatable.npz")
        if not os.path.exists(cache):
            box = Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"])
            h = cfg["ihel"] if cfg["ihel"] != 0 else +1
            t0 = time.time()
            RCTable(sf, cfg["ebeam"], box, mx2, h, (5, 5, 5, 8)).build(
                verbose=verbose).save(cache)
            if verbose:
                print(f" eta table built and cached in {time.time()-t0:.0f}s")

    # Build AND CALIBRATE the majorant once here, then let the workers load
    # it.  Without this every worker repeats the ~19 s prescan, which made
    # --jobs slower than a single core (226 s vs 176 s for 100k).
    d, _, _ = _outdir(cfg, date)
    majc = os.path.join(d, "majorant.npz")
    if not os.path.exists(majc):
        ch = cfg["model"].split(".")[0]
        kw = dict(t_nucl=cfg["t_nucl"]) if ch in ("pi0", "eta") else {}
        sf0 = models.get(cfg["model"], **kw)
        mx20 = cfg["vcut"] + sf0.mU ** 2 if cfg["vcut"] > 0 else None
        g0 = Generator(sf0, Ebeam=cfg["ebeam"],
                       box=Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"]),
                       mx2_cut=mx20, helicity=cfg["ihel"], seed=cfg["seed"],
                       born_only=bool(cfg["iborn"]),
                       cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"]),
                       v_min=cfg["vmin"] or None,
                       cache=None if cfg["iborn"] else os.path.join(d, "etatable.npz"))
        t0 = time.time()
        g0.setup(verbose=verbose)
        g0.calibrate(n_probe=30000, margin=2.0, verbose=verbose)
        g0.calibrate(n_probe=30000, margin=1.5, verbose=verbose)
        g0.save_majorant(majc)
        if verbose:
            print(f" majorant built+calibrated once in {time.time()-t0:.0f}s, shared")

    args = [(card_path, date, i + 1, jobs, cfg["seed"] + 1000 * i)
            for i in range(jobs)]
    # record the split so the sample can be reproduced exactly
    d, _, _ = _outdir(cfg, date)
    with open(os.path.join(d, "inputs", "parallel.json"), "w") as fh:
        json.dump(dict(jobs=jobs, base_seed=cfg["seed"],
                       part_seeds=[a[4] for a in args],
                       nev_total=cfg["nev"]), fh, indent=2)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(_worker, args))
    n = sum(r["n"] for r in res)
    if verbose:
        print(f" {n} events in {time.time()-t0:.0f}s across {jobs} parts")
        for r in res:
            if r["health"]:
                print("  ", os.path.basename(r["lund"]), "->", "; ".join(r["health"]))
    return res


def run(card_path, date=None, verbose=True, _part=None):
    cfg = parse(card_path)
    if verbose:
        echo(cfg)
    part_tag = ""
    if _part is not None:
        ipart, npart, seed = _part
        cfg = dict(cfg)
        cfg["nev"] = cfg["nev"] // npart + (1 if ipart <= cfg["nev"] % npart else 0)
        cfg["seed"] = seed
        part_tag = f"_{ipart:02d}"

    ch = cfg["model"].split(".")[0]
    kw = {}
    if ch in ("pi0", "eta"):
        kw["t_nucl"] = cfg["t_nucl"]
    sf = models.get(cfg["model"], **kw)

    mx2 = None
    if cfg["vcut"] > 0:
        mx2 = cfg["vcut"] + sf.mU ** 2

    box = Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"])
    d, mode, ch = _outdir(cfg, date)

    g = Generator(sf, Ebeam=cfg["ebeam"], box=box, mx2_cut=mx2,
                  helicity=cfg["ihel"], seed=cfg["seed"],
                  born_only=bool(cfg["iborn"]),
                  cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"]),
                  v_min=cfg["vmin"] or None,
                  maj_cache=os.path.join(d, "majorant.npz"),
                  cache=None if cfg["iborn"] else os.path.join(d, "etatable.npz"))
    t0 = time.time()
    g.setup(verbose=verbose)
    t_setup = time.time() - t0

    interp = None
    if not cfg["iborn"]:
        interp = g.table.check_interpolation(n=25)
        if verbose:
            print(f" eta-table interpolation: {interp}")
        if interp.get("rms", 9) > 0.02:
            print(f" *** WARNING: eta interpolation rms {interp['rms']:.3f} "
                  f"> 0.02 -- the soft/hard split will be unreliable. "
                  f"Narrow the box or refine table_shape. ***")

    # two passes: the first finds the gross misses, the second the ones
    # only reachable once the cell probabilities have shifted.
    if _part is None:
        cal = g.calibrate(n_probe=30000, margin=2.0, verbose=verbose)
        cal2 = g.calibrate(n_probe=30000, margin=1.5, verbose=verbose)
        cal = dict(pass1=cal, pass2=cal2)
    else:
        cal = "shared calibrated majorant loaded from majorant.npz"

    t0 = time.time()
    ev = g.generate(cfg["nev"], progress=max(cfg["nev"] // 5, 1) if verbose else None)
    t_gen = time.time() - t0

    # optional isotropic decay of the vector meson, as diffrad_vm does
    if cfg["idecay"] and ch in DECAY:
        rng = np.random.default_rng(cfg["seed"] + 1)
        m_d = DECAY[ch][2]
        n_bad = 0
        for e in ev:
            a, b = decay_two_body(e["meson"], m_d, rng)
            if a is None:
                n_bad += 1
            e["daughters"] = (a, b)
        if verbose and n_bad:
            print(f" {n_bad} events below the {ch} decay threshold, left undecayed")

    st = stamp(sf, extra=dict(runcard=os.path.abspath(card_path), mode=mode,
                              box=box.as_dict(), vcut=cfg["vcut"],
                              mx2_cut=mx2, Ebeam=cfg["ebeam"],
                              n_events=len(ev), seed=cfg["seed"],
                              eta_interp=interp))
    lund = os.path.join(d, "lund",
                        f"{ch}_exclurad_py_{mode}_{cfg['ebeam']:g}GeV{part_tag}.lund")
    n = write_lund(lund, ev, channel=ch, topology=sf.topology, stamp=st)

    shutil.copyfile(card_path, os.path.join(d, "inputs", "gen_input.txt"))
    rep = {k: (float(v) if isinstance(v, float) else v)
           for k, v in g.report().items()}
    cfgname = f"config{part_tag or ''}.json"
    with open(os.path.join(d, "inputs", cfgname), "w") as fh:
        json.dump(dict(card=cfg, stamp=st, report=rep, health=g.health(),
                       calibration=cal, setup_s=t_setup, gen_s=t_gen),
                  fh, indent=2, default=str)
    with open(os.path.join(d, "README.md"), "w") as fh:
        fh.write(f"# exclurad_py {ch} {mode.upper()}, {cfg['ebeam']:g} GeV, "
                 f"{n} events\n\nModel `{cfg['model']}`. "
                 f"{'Born only, no RC.' if cfg['iborn'] else 'Full RC, exact one-photon tail.'}\n"
                 f"Unweighted: every event equally probable, LUND weight column 1.0.\n\n"
                 f"Regenerate with `python3 -m exclurad_py.run inputs/gen_input.txt`.\n")

    if verbose:
        print(f" wrote {n} events -> {lund}")
        print(f" setup {t_setup:.0f}s, generate {t_gen:.0f}s "
              f"({1e3*t_gen/max(len(ev),1):.2f} ms/event)")
        print(" health:", "; ".join(g.health()) or "clean")
    return dict(outdir=d, lund=lund, n=n, report=rep, health=g.health())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    # DEFAULT 1, deliberately. Parallel runs split the events into parts
    # with seed + 1000*i, so the part COUNT changes the sample. Defaulting to
    # cpu_count() would make the same run card produce different events on a
    # 14-core and an 8-core machine -- machine-dependent output from a
    # physics generator. Parallelism must be asked for explicitly and is
    # recorded in config.json.
    jobs = 1
    if "--jobs" in sys.argv:
        i = sys.argv.index("--jobs")
        jobs = int(sys.argv[i + 1])
    elif os.environ.get("EXRU_JOBS"):
        jobs = int(os.environ["EXRU_JOBS"])
    if jobs > 1:
        run_parallel(sys.argv[1], jobs)
    else:
        run(sys.argv[1])
