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
import socket
import sys
import time

import numpy as np

from . import models, stamp
from .generator import (Generator, Box, write_lund, decay_two_body, DECAY,
                        meson_slot)
from .generator.decay import decay_meson
from .core.radiative import quad_default
from .core.integrand import USE_NUMBA as _USE_NUMBA
from .generator.tables import RCTable
from .runcard import parse, echo


_FERMI_WF = {
    "none": None,
    # Wiringa's variational Monte Carlo deuteron momentum distribution,
    # rho(k)/(2 pi)^3 from phy.anl.gov/theory/research/momenta (h2.momentum),
    # normalised per nucleon.  Cite Wiringa et al., Phys. Rev. C 89, 024305.
    "av18": "AV18 (Wiringa VMC), per-nucleon, k <= 3.5601 fm^-1",
}


def _cache_paths(d):
    """Where this run reads/writes its eta table and majorant.

    Default: inside the run directory, so every run builds its own.  ETA_TABLE
    and MAJ_TABLE in the environment override that with one shared file --
    the table depends only on model, box, E, vcut and the two spins, so a
    run that differs in seed and event count reuses it verbatim.  Safety is
    in RCTable.load, which refuses a key mismatch; a wrong path fails loudly.
    Motivation: 120 slurm jobs of 2026-09-20 sat ten hours behind one job
    that rebuilt a table already built four times that week.
    """
    eta = os.environ.get("ETA_TABLE") or os.path.join(d, "etatable.npz")
    maj = os.environ.get("MAJ_TABLE") or os.path.join(d, "majorant.npz")
    return eta, maj


def _needs_t_nucl(model_name):
    """True if the model is constructed with the bin's nucleon transfer.

    Asked of the TOPOLOGY, not of a class or a channel list.  A hardcoded
    channel list silently missed pi0n/etan when they were added; asking
    `issubclass(..., ProtonDetectedModel)` then missed the proton-detected
    VECTOR models, which deliberately do NOT inherit that base class -- it
    carries the Hand-convention kf conversion the pi0/eta fits need and the
    vector models must not have.  What actually decides it is
    sf_t == FIXED_NUCLEON: the model is a function of t_N, which the core
    kinematics does not build from the detected hadron, so the caller has to
    supply it.
    """
    from .topology import FIXED_NUCLEON
    topo = getattr(models.get_class(model_name), "topology", None)
    return getattr(topo, "sf_t", None) == FIXED_NUCLEON


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
        kw = dict(t_nucl=cfg["t_nucl"]) if _needs_t_nucl(cfg["model"]) else {}
        # The beam energy has to reach the MODEL, not only the generator.
        # StructureFunctions.Ebeam is a class attribute defaulting to 10.6,
        # and _epsilon uses it to turn SLT_OVER_SU / SLTP_OVER_SU into
        # structure functions.  Nothing ever assigned it from the card, so a
        # 6.6 GeV run would have built sigma_LT from epsilon(10.6): 0.910
        # against the correct 0.728, a 25% error, silently.  Latent while
        # every card says 10.6.
        sf = models.get(cfg["model"], **kw)
        sf.Ebeam = float(cfg["ebeam"])
        mx2 = cfg["vcut"] + sf.mU ** 2 if cfg["vcut"] > 0 else None
        cache, _ = _cache_paths(d)
        if not os.path.exists(cache):
            box = Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"])
            h = cfg["ihel"] if cfg["ihel"] != 0 else +1
            t0 = time.time()
            RCTable(sf, cfg["ebeam"], box, mx2, h, (5, 5, 5, 8),
                    ptarg=cfg.get("ptarg", 0.0)).build(
                verbose=verbose).save(cache)
            if verbose:
                print(f" eta table built and cached in {time.time()-t0:.0f}s")

    # Build AND CALIBRATE the majorant once here, then let the workers load
    # it.  Without this every worker repeats the ~19 s prescan, which made
    # --jobs slower than a single core (226 s vs 176 s for 100k).
    d, _, _ = _outdir(cfg, date)
    _, majc = _cache_paths(d)
    if not os.path.exists(majc):
        ch = cfg["model"].split(".")[0]
        kw = dict(t_nucl=cfg["t_nucl"]) if _needs_t_nucl(cfg["model"]) else {}
        sf0 = models.get(cfg["model"], **kw)
        sf0.Ebeam = float(cfg["ebeam"])
        mx20 = cfg["vcut"] + sf0.mU ** 2 if cfg["vcut"] > 0 else None
        g0 = Generator(sf0, Ebeam=cfg["ebeam"],
                       box=Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"]),
                       mx2_cut=mx20, helicity=cfg["ihel"],
                       ptarg=cfg.get("ptarg", 0.0), seed=cfg["seed"],
                       born_only=bool(cfg["iborn"]),
                       cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"],
                                 theta_e=cfg.get("theta_e", (0.0, 180.0))),
                       itarg=cfg["itarg"], vz=cfg["vz"],
                       fermi=cfg.get("fermi", "none"),
                       fermi_m_star_min=cfg.get("fermi_mstar_min", 0.30),
                       v_min=cfg["vmin"] or None,
                       cache=None if cfg["iborn"] else _cache_paths(d)[0])
        t0 = time.time()
        g0.setup(verbose=verbose)
        g0.calibrate(n_probe=30000, margin=2.0, verbose=verbose)
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
    if _needs_t_nucl(cfg["model"]):
        kw["t_nucl"] = cfg["t_nucl"]
    sf = models.get(cfg["model"], **kw)
    sf.Ebeam = float(cfg["ebeam"])          # see the note above

    mx2 = None
    if cfg["vcut"] > 0:
        mx2 = cfg["vcut"] + sf.mU ** 2

    box = Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"])
    d, mode, ch = _outdir(cfg, date)

    g = Generator(sf, Ebeam=cfg["ebeam"], box=box, mx2_cut=mx2,
                  helicity=cfg["ihel"], ptarg=cfg.get("ptarg", 0.0),
                  seed=cfg["seed"],
                  born_only=bool(cfg["iborn"]),
                  cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"],
                                 theta_e=cfg.get("theta_e", (0.0, 180.0))),
                  itarg=cfg["itarg"], vz=cfg["vz"],
                  fermi=cfg.get("fermi", "none"),
                  fermi_m_star_min=cfg.get("fermi_mstar_min", 0.30),
                  decay_fsr=(cfg["idecay"] == 2),
                  v_min=cfg["vmin"] or None,
                  maj_cache=_cache_paths(d)[1],
                  cache=None if cfg["iborn"] else _cache_paths(d)[0])
    # Announce the soft/hard split MODE, not just its value.  A card with
    # vmin = 0 -- or with the field omitted, which is the DEFAULT_OPT -- puts
    # the generator into per-event selection by choose_vmin, and nothing said
    # so anywhere: not the log, not the LUND header.  Story's 2026-09-28
    # diagnostic sample had no step at 1e-4 in its v spectrum and a pile-up at
    # 2.5e-3 instead, which is the smeared boundary, and the only way to find
    # out was to plot the spectrum and notice.  A mode switch that a sample
    # cannot report about itself is not a setting, it is a trap.
    if not cfg["iborn"]:
        if cfg["vmin"]:
            print(f" soft/hard split: FIXED at vmin = {cfg['vmin']:.3g} GeV^2")
        else:
            print(" soft/hard split: PER EVENT (vmin = 0 in the card, so "
                  "choose_vmin picks it per event, targeting soft_frac 0.35 "
                  "within 1e-5..0.1 of v_max).  The v spectrum will have no "
                  "step: set vmin explicitly if you want one.")
    t0 = time.time()
    g.setup(verbose=verbose)
    t_setup = time.time() - t0

    interp = None
    if not cfg["iborn"]:
        interp = g.table.check_interpolation(n=25)
        if verbose:
            print(f" eta-table interpolation: {interp}")
        # WHAT THIS DOES AND DOES NOT MEAN.  The old text said "the soft/hard
        # split will be unreliable".  That is true of the CELL-BASED photon
        # path, where the table drives the split.  Production runs the
        # PER-EVENT path (EXACT_ACCEPT/EXPO_TAIL both force exact_per_event),
        # where rc_factor is recomputed at the event's own kinematics and the
        # split inverts the EXACT sigma_obs.  There the table is only the
        # stage-1 proposal and stage 2 corrects it exactly, so interpolation
        # error costs ACCEPTANCE EFFICIENCY, not correctness -- provided the
        # stage-2 bound holds, which is what b_viol in the report measures.
        # Measured on the 2026-09-28 million: absolute rms 0.9-2.3 with
        # bound violations 0.0044 % and worst excess x1.87.
        _exact = os.environ.get("EXACT_ACCEPT", "") == "1" or \
                 os.environ.get("EXPO_TAIL", "") == "1"
        if interp.get("rms_rel", interp.get("rms", 9)) > 0.02:
            print(f" *** WARNING: eta interpolation rms(rel) "
                  f"{interp.get('rms_rel', float('nan')):.3f}, rms(abs) "
                  f"{interp['rms']:.3f} over {interp.get('n')} uniform points. "
                  + ("Per-event path: this costs stage-1 acceptance "
                     "efficiency, NOT correctness -- check b_viol in the "
                     "report, that is the number that would bite. ***"
                     if _exact else
                     "Cell-based path: the soft/hard split USES the table "
                     "and will be unreliable.  Narrow the box or refine "
                     "table_shape. ***"))

    # two passes: the first finds the gross misses, the second the ones
    # only reachable once the cell probabilities have shifted.
    if _part is None:
        # ONE pass only. A second calibration pass corrupts the sampled t
        # spectrum: pi0 |t| mean goes 0.42 (0 or 1 pass, matching exclugen's
        # 0.39) -> 3.67 with two passes. Pass 2 probes with the cdf already
        # reshaped by pass 1, so it re-inflates an over-sampled subset of
        # cells and compounds; the mechanism is not yet fully understood, so
        # the second pass stays OFF until it is.
        cal = g.calibrate(n_probe=30000, margin=2.0, verbose=verbose)
    else:
        # НЕ утверждать, что загружена общая откалиброванная мажоранта.
        # В боевой раскладке majorant.npz пишет только run_parallel, а
        # run_prod_v1.sh его не зовёт -- файла нет, каждая часть строит
        # свою и не калибрует её.  Спрашиваем генератор, что было на самом
        # деле: выборка, врущая про себя, хуже выборки без описания.
        cal = g.majorant_provenance()

    t0 = time.time()
    ev = g.generate(cfg["nev"], progress=max(cfg["nev"] // 5, 1) if verbose else None)
    t_gen = time.time() - t0

    # optional isotropic decay of the meson: vector -> h+ h- as diffrad_vm
    # does, pi0/eta -> gamma gamma as exclugen does.
    if cfg["idecay"] and ch in DECAY:
        rng = np.random.default_rng(cfg["seed"] + 1)
        # NOT always ev["meson"] -- slots are named by role, and under the
        # EXCLURAD pi0/eta topology the detected hadron is the PROTON.
        slot = meson_slot(getattr(sf, "topology", None), ch)
        n_bad = 0
        for e in ev:
            # An event the generator already decayed (acceptance preset with
            # daughter cuts) keeps THAT decay: it is the one the cut saw.
            if e.get("daughters") is not None:
                continue
            a, b, gd = decay_meson(ch, e[slot], rng, g.fsr)
            if a is None:
                n_bad += 1
            e["daughters"] = (a, b)
            e["decay_photon"] = gd
        if verbose and n_bad:
            print(f" {n_bad} events below the {ch} decay threshold, left undecayed")

    # ihel and ptarg belong in the header for the same reason the model does:
    # a polarised study generates several samples that differ ONLY in the two
    # spins, and without them the files are indistinguishable once they leave
    # the directory that named them.  That is exactly how 36 GB of vpk2021
    # came to be labelled amp2609.
    rep0 = g.report()
    st = stamp(sf, extra=dict(runcard=os.path.abspath(card_path), mode=mode,
                              box=box.as_dict(), vcut=cfg["vcut"],
                              mx2_cut=mx2, Ebeam=cfg["ebeam"],
                              ihel=cfg["ihel"], ptarg=cfg.get("ptarg", 0.0),
                              # 2 = the decay leptons radiate; a sample with and
                              # one without differ only in M(ee), so the file
                              # has to say which it is
                              idecay=cfg["idecay"],
                              decay_rad=(None if g.fsr is None else dict(
                                  model="exact tree ME V->l+l-gamma, Ehlotzky "
                                        "spectrum exponentiated (YFS)",
                                  eps_GeV=g.fsr.eps, beta=g.fsr.spec.beta,
                                  bound_violations=g.fsr.n_viol)),
                              # The target is part of what a sample IS.  A
                              # deuteron file and a free-nucleon file are
                              # otherwise indistinguishable from their
                              # contents -- same channel, same box, same
                              # model -- except for one extra particle per
                              # event.  Record the flag AND the wave
                              # function; "none" for a free nucleon.
                              fermi=cfg.get("fermi", "none"),
                              fermi_wavefunction=(
                                  _FERMI_WF.get(cfg.get("fermi", "none"))),
                              # the machine is provenance: once files are merged
                              # across hosts only the header can still say
                              host=socket.gethostname().split(".")[0],
                              # the sigma_F quadrature: samples made before and
                              # after 2026-09-24 differ by ~0.13% in eta, and
                              # nothing else in the file would say which
                              rc_quad=",".join(str(x) for x in quad_default()),
                              # WHICH INTEGRAND BACKEND made this sample.
                              # numpy and numba agree to ~1e-11 per event and
                              # exactly on sigma_nb, but a sample must still
                              # be able to say which produced it -- the same
                              # reason rc_quad is stamped above.
                              rc_backend=("numba" if _USE_NUMBA else "numpy"),
                              # the CARD's t_nucl -- sf.t_nucl in the stamp is
                              # the live attribute, i.e. the last event's t
                              t_nucl_card=cfg.get("t_nucl"),
                              # the MODE, not the card value: "fixed" or
                              # "per-event".  A sample has to be able to say
                              # which soft/hard split made it, or two samples
                              # get compared in the soft region without anyone
                              # knowing they were sliced differently.
                              vmin_mode=("fixed" if cfg["vmin"] else "per-event"),
                              n_events=len(ev), seed=cfg["seed"],
                              eta_interp=interp,
                              # THE STAGE-2 BOUND, IN THE FILE THAT SURVIVES.
                              # b_viol decides whether the sample is exactly
                              # unweighted, and it lived only in
                              # inputs/config.json -- which a farm pipeline
                              # that moves the LUND out and drops the run
                              # directory throws away.  The 2026-10-05
                              # amp2614 campaign did exactly that, and the
                              # only surviving evidence was health(), which
                              # warns at 0.1 % of accepts and is SILENT below
                              # it: our own measured 0.0044 % printed "health:
                              # clean".  Absence of the warning was then read
                              # as zero violations.  It is not; now the count
                              # travels with the LUND.
                              bound=dict(b_viol=rep0.get("b_viol", 0),
                                         max_b_viol=rep0.get("max_b_viol", 0.0),
                                         accepted=rep0.get("accepted", 0)),
                              ))
    lund = os.path.join(d, "lund",
                        f"{ch}_exclurad_py_{mode}_{cfg['ebeam']:g}GeV{part_tag}.lund")
    # Header column 8 is the TARGET.  With Fermi motion the target is the
    # deuteron, not the nucleon the reaction happened on: the file already
    # carries the struck nucleon's partner as a particle line, and a header
    # saying 2212 would contradict it.  PDG nuclear code for the deuteron is
    # 1000010020 (10LZZZAAAI with Z = 1, A = 2).
    n = write_lund(lund, ev, channel=ch, topology=sf.topology, stamp=st,
                   beam_energy=float(cfg["ebeam"]),   # header column 7; was the 10.6 default for every run
                   target_pid=(1000010020 if cfg.get("fermi", "none") != "none"
                               else None),
                   pol_target=float(cfg.get("ptarg", 0.0)))   # header column 4; was 0.000 for every run

    shutil.copyfile(card_path, os.path.join(d, "inputs", "gen_input.txt"))
    rep = {k: (float(v) if isinstance(v, float) else v)
           for k, v in g.report().items()}
    cfgname = f"config{part_tag or ''}.json"
    with open(os.path.join(d, "inputs", cfgname), "w") as fh:
        # `interp` beside `report`, so the two numbers that must be read
        # TOGETHER are in one place: the interpolation error only matters
        # through the stage-2 bound, and b_viol lives in `report`.
        #
        # CORRECTION 2026-09-29.  An earlier version of this comment said
        # there was no record of the check on a finished run.  That was
        # wrong: the .info sidecar has carried `run.eta_interp.rms` all
        # along, in all 200 chunks of the 2026-09-28 million.  I had claimed
        # a missing record without looking for it, and the optimization
        # session found it by reading the sidecars.  What was genuinely
        # missing is only the job-log line (gen_array.sbatch's grep), which
        # is why the warning went unnoticed at the time.
        json.dump(dict(card=cfg, stamp=st, report=rep, health=g.health(),
                       interp=interp,
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
        # the count ALWAYS, not only when it crosses the warning threshold:
        # a health line that says "clean" must not be readable as "zero".
        nb, na = rep.get("b_viol", 0), rep.get("accepted", 0)
        print(" health:", "; ".join(g.health()) or "clean",
              f"| stage-2 b_viol {nb}/{na}"
              + (f" max x{rep.get('max_b_viol', 0.0):.2f}" if nb else ""))
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
