"""
Run exclurad_py from a run card.

    python3 -m exclurad_py.run gen_input.txt

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


def run(card_path, date=None, verbose=True):
    cfg = parse(card_path)
    if verbose:
        echo(cfg)

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
                        f"{ch}_exclurad_py_{mode}_{cfg['ebeam']:g}GeV.lund")
    n = write_lund(lund, ev, channel=ch, topology=sf.topology, stamp=st)

    shutil.copyfile(card_path, os.path.join(d, "inputs", "gen_input.txt"))
    rep = {k: (float(v) if isinstance(v, float) else v)
           for k, v in g.report().items()}
    with open(os.path.join(d, "inputs", "config.json"), "w") as fh:
        json.dump(dict(card=cfg, stamp=st, report=rep, health=g.health(),
                       setup_s=t_setup, gen_s=t_gen), fh, indent=2, default=str)
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
    run(sys.argv[1])
