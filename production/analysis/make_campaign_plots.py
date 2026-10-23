#!/usr/bin/env python3
"""Build a campaign plot book: radiative against Born, plus data if there is any.

    python3 make_campaign_plots.py <ntuple root> <out.pdf> [mariana dir]

<ntuple root> holds rad/out_inb, rad/out_outb, born/out_*, rad/prov, born/prov
exactly as the campaign directory does, so it runs equally on /volatile or on
a copy.  One LUND feeds all four GEMC configurations of a mode, so the
generator-level pages are built once, from the inbending-with-background set,
and are the full million either way.
"""
import gc
import glob
import os
import re
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as cl                                          # noqa: E402
import campaign_plots as cp                                        # noqa: E402

ROOT_DIR = sys.argv[1]
OUT = sys.argv[2]
MARIANA = sys.argv[3] if len(sys.argv) > 3 else None


def _card(path):
    """The box, read from the card the campaign actually ran.

    Not retyped into this script.  A plot book whose header is a constant in
    the plotting code is a plot book that will one day describe a different
    run than the one it shows, and nothing will say so.
    """
    vals = []
    for line in open(path):
        line = line.split("!")[0].strip()
        if line:
            vals.append(line)
    return vals


CARD = _card(os.path.join(ROOT_DIR, "cards", "rga_rad.txt"))
CH = CARD[0].split(".")[0]
TITLES = {"jpsi": "J/$\\psi$", "phi": "$\\phi$", "rho": "$\\rho$",
          "pi0": "$\\pi^0$", "eta": "$\\eta$"}
PRETTY = TITLES.get(CH, CH)
def _discover(root):
    """Find the (polarity, background) sets that are actually on disk.

    A fixed list of four was wrong the first time it met a campaign that was
    not ours: Story's amp2614 run is inbending at 50 nA, and a list naming
    only (inb, 45nA) and (outb, 50nA) matched none of it and reported that
    nothing was there.  The names carry the configuration -- read them.
    """
    seen = {}
    for f in glob.glob(os.path.join(root, "*", "out_*", "*.f32")):
        d = os.path.basename(os.path.dirname(f))          # out_inb / out_outb
        pol = d[4:]
        m = re.search(r"_bg([A-Za-z0-9]+)_\d+\.f32$", os.path.basename(f))
        if m:
            seen[(pol, m.group(1))] = True
    def lab(pol, bg):
        p = "inbending" if pol == "inb" else "outbending"
        return f"{p}, " + ("no background" if bg == "none" else f"{bg} background")
    order = {"inb": 0, "outb": 1}
    return [(pol, bg, lab(pol, bg))
            for pol, bg in sorted(seen, key=lambda k: (order.get(k[0], 9), k[1] == "none", k[1]))]
BOX = ("RG-A Fall 2018  |  gemc 5.13 + coatjava 10.0.7  |  "
       f"E = {CARD[1]} GeV  |  $W^2$ {CARD[4]},  $Q^2$ {CARD[5]},  "
       f"$t$ {CARD[6]} GeV$^2$,  vcut {CARD[7]}")

# THE NORMALISATION COMES FROM THE CHUNKS THAT ARE ACTUALLY HERE.
# NGEN was 200 * 5000, a remembered number, and sigma averaged every sidecar
# in prov/.  Both were right only while a campaign had exactly 200 chunks and
# ntuples for all of them.  On 2026-10-23 a second million was generated as
# chunks 201-400 with no GEMC behind it yet: sigma then averaged 400 sidecars
# for a 200-chunk sample -- the peer session caught that in this book's own
# header, 0.94679 printed for a ratio that is 0.94658 -- and NGEN would have
# been wrong by a factor two the moment the new ntuples appeared, silently
# halving every per-event normalisation.
CHUNKS = {m: cl.chunks_present(ROOT_DIR, m) for m in ("rad", "born")}
NEV1 = {m: cl.events_per_chunk(ROOT_DIR, m, CH) for m in ("rad", "born")}
SIG = {m: cl.sigma_nb(os.path.join(ROOT_DIR, m, "prov"), m, chunks=CHUNKS[m])
       for m in ("rad", "born")}
NGEN = {m: len(CHUNKS[m]) * NEV1[m] for m in ("rad", "born")}
for m in ("rad", "born"):
    c = CHUNKS[m]
    print("  %-5s %d chunks (%d-%d) x %d events = %d generated, sigma over %d sidecars"
          % (m, len(c), min(c), max(c), NEV1[m], NGEN[m], SIG[m]["n"]))
log = []


def _exists(pol, bg):
    return bool(glob.glob(pat("rad", pol, bg))) and bool(glob.glob(pat("born", pol, bg)))




def pat(mode, pol, bg):
    return os.path.join(ROOT_DIR, mode, f"out_{pol}", f"*_bg{bg}_*.f32")


PRESENT = None          # filled in below, once pat() exists


def sets_for(pol, bg, rec, require):
    out = {}
    for m in ("rad", "born"):
        d = cl.load(pat(m, pol, bg), CH, rec=rec, require=require)
        out[m] = dict(d=d, sigma=SIG[m]["mean"], n_gen=NGEN[m])
    return out


# The final state, spelled once.  Every title and every log line used to
# carry a hand-typed "e' N h+ h-", which is right for no channel at all and
# wrong on a pi0 page in the same way the axis labels were.
_C = cl.CHANNELS[CH]
_H1, _H2 = _C["names"]
_N = _C["baryon_name"]
FS_FULL = f"e' {_N} ${_H1}$ ${_H2}$".replace("$$", "")
FS_NOBAR = f"e' ${_H1}$ ${_H2}$".replace("$$", "")
FS_FULL_TXT = f"e' {_N} {_H1} {_H2}".replace("\\", "").replace("^", "")
FS_NOBAR_TXT = f"e' {_H1} {_H2}".replace("\\", "").replace("^", "")

PRESENT = [c for c in _discover(ROOT_DIR) if _exists(c[0], c[1])]
if not PRESENT:
    raise SystemExit("no rad/born pair of f32 ntuples found under " + ROOT_DIR)
log.append("configurations on disk: " + "; ".join(c[2] for c in PRESENT))
S3POL, S3BG, S3LAB = PRESENT[0]

with PdfPages(OUT) as pdf:
    # ---- title page -------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 8.0))
    fig.text(0.5, 0.80, f"{PRETTY} electroproduction at CLAS12", ha="center",
             fontsize=23, fontweight="semibold")
    fig.text(0.5, 0.735, "exclurad_py campaign   " + os.path.basename(os.path.realpath(ROOT_DIR))
             + "   |   model " + CARD[0], ha="center", fontsize=11, color="#555555")
    lines = [
        "Eight samples of one million events: radiative and Born, each passed through",
        "GEMC four ways -- inbending and outbending, with and without background merging.",
        "",
        "Every distribution is normalised to its own CROSS SECTION, not to unit area:",
        "the pair was generated to measure a change in rate as well as in shape, and",
        "area normalisation would hide half of the answer.",
        "",
        f"sigma(radiative) = {SIG['rad']['mean']:.5g} nb   "
        f"+- {SIG['rad']['err']:.2g} over {SIG['rad']['n']} chunks",
        f"sigma(Born)      = {SIG['born']['mean']:.5g} nb   "
        f"+- {SIG['born']['err']:.2g} over {SIG['born']['n']} chunks",
        f"ratio            = {SIG['rad']['mean'] / SIG['born']['mean']:.5f}",
        "",
        BOX.replace("  |  ", "\n"),
    ]
    fig.text(0.09, 0.62, "\n".join(lines), fontsize=10.2, va="top", linespacing=1.55)
    fig.text(0.09, 0.07, "The chunk-to-chunk spread of sigma is a health check, not a decoration: a "
             "stale majorant\ncache once moved it by a factor <B> ~ 2 with no other symptom.  Here it is "
             f"{100 * SIG['rad']['spread_rel']:.2f} %\n(radiative) and "
             f"{100 * SIG['born']['spread_rel']:.2f} % (Born) peak to peak.",
             fontsize=8.4, color="#555555", va="bottom", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- generator level --------------------------------------------------
    g = sets_for(S3POL, S3BG, rec=False, require=())
    log.append(f"generated: rad {g['rad']['d']['n_kept']}, born {g['born']['d']['n_kept']}")
    cp.page(pdf, g, "Generator level - the full million", BOX)

    # The vertex, every time.  Six panels: where the event was generated,
    # where the electron was reconstructed in each polarity against the
    # note's cuts, the resolution, the electron-proton difference, and what
    # the cuts keep.  Nothing else plots any of it -- the f32 record carried
    # momenta only until 2026-10-22, which is why a Born arm sat at a point
    # for a day unseen.
    try:
        _vz_card = [float(x) for x in CARD[18].split()[:2]] if len(CARD) > 18 else None
    except Exception:                                              # noqa: BLE001
        _vz_card = None
    cp.page_vertex(pdf, ROOT_DIR, PRESENT, CH,
                   cell=tuple(sorted(_vz_card)) if _vz_card and len(_vz_card) == 2 else None)

    # The pseudoscalar ntuple keeps only the two photons whose MC::Lund parent
    # is the meson row, so the radiated photon is not in the record and this
    # page cannot be built from it.  Say so rather than draw an empty one.
    if g["rad"]["d"].get("has_gprod", True):
        cp.page_radiation(pdf, g, "Generator level - where the radiation goes")
    else:
        log.append("radiation page skipped: this record has no radiated-photon slot")
    cp.page_ratio(pdf, g, "Generator level - radiative / Born, bin by bin",
                  "a flat line means radiation only rescales; a slope means it reshapes")
    del g
    gc.collect()

    # ---- reconstructed, four configurations, exclusive --------------------
    yields = []
    for pol, bg, lab in PRESENT:
        s = sets_for(pol, bg, rec=True, require=cl.TOPOLOGY["full"])
        n = {m: s[m]["d"]["n_kept"] for m in s}
        yields.append((lab, n["rad"], n["born"]))
        zr = s["rad"]["d"].get("n_dp_rejected", {})
        log.append(f"rec {FS_FULL_TXT} {lab}: rad {n['rad']}, born {n['born']}"
                   + (f"   [rad dropped on the momentum match: "
                      + ", ".join(f"{k} {v}" for k, v in zr.items() if v) + "]" if any(zr.values()) else ""))
        cp.page(pdf, s, f"Reconstructed   {FS_FULL}   -   {lab}",
                f"matched, carrying momentum, and agreeing with the generated momentum "
                f"to {int(100 * cl.DP_MAX['charged'])} % (charged) / "
                f"{int(100 * cl.DP_MAX['photon'])} % (photons)   |   "
                f"rad {n['rad']} of 1 000 000"
                f" ({100 * n['rad'] / 1e6:.3f} %),  Born {n['born']}"
                f" ({100 * n['born'] / 1e6:.3f} %)")
        if bg == "none":
            cp.page_ratio(pdf, s, f"Reconstructed - radiative / Born - {lab}",
                          "acceptance divides out of this ratio; what is left is radiation")
        del s
        gc.collect()

    # ---- reconstructed, three-track topology ------------------------------
    s3 = sets_for(S3POL, S3BG, rec=True, require=cl.TOPOLOGY["no_baryon"])
    log.append(f"rec {FS_NOBAR_TXT} {S3LAB}: rad {s3['rad']['d']['n_kept']}, "
               f"born {s3['born']['d']['n_kept']}")
    cp.page(pdf, s3, f"Reconstructed   {FS_NOBAR}   (recoil baryon not required)   -   " + S3LAB,
            f"the recoil {_N} is dropped: no t measurement, and for J/psi roughly four "
            "times the data statistics")

    # ---- data -------------------------------------------------------------
    # One page per (topology, period).  The exclusive topology is the four-track
    # sample our MC was cut to; the three-track one drops the proton and holds
    # about four times the data events, at the price of the t measurement.
    if MARIANA and CH == "jpsi":
        plan = [("ee+e-p", cl.TOPOLOGY["full"], "inb", "45nA", "F18in"),
                ("ee+e-p", cl.TOPOLOGY["full"], "outb", "50nA", "F18out"),
                ("ee+e-", cl.TOPOLOGY["no_baryon"], "inb", "45nA", "F18in"),
                ("ee+e-", cl.TOPOLOGY["no_baryon"], "outb", "50nA", "F18out")]
        for topol, req, pol, bg, period in plan:
            try:
                dat = cp.load_mariana(MARIANA, topol, period)
            except Exception as e:                                 # noqa: BLE001
                log.append(f"data {topol} {period}: NOT read ({e})")
                continue
            w = (dat["m_pair"] > 2.9) & (dat["m_pair"] < 3.3)
            sig = {k: (v[w] if isinstance(v, np.ndarray) else v) for k, v in dat.items()}
            sig["n_kept"] = int(w.sum())
            s = sets_for(pol, bg, rec=True, require=req)
            log.append(f"data {topol} {period}: {dat['n_kept']} baseline, "
                       f"{int(w.sum())} in the J/psi window;  MC rad "
                       f"{s['rad']['d']['n_kept']}, Born {s['born']['d']['n_kept']}")
            cp.page_data(pdf, s, dat, sig,
                         f"MC against CLAS12 data   -   {topol}   -   {period}",
                         "SHAPE ONLY: no run_number, and beamCharge is a cumulative scaler, "
                         "so there is no luminosity in these files\n"
                         f"data cuts  E$_\\gamma \\geq$ 8.1 GeV,  |t$_{{ee}}$| $\\leq$ 2.5 ns,  "
                         f"FT fiducial r $\\leq$ 16.065 cm       "
                         f"{dat['n_kept']} events, {int(w.sum())} in 2.9 < M < 3.3")
            del s
            gc.collect()

    # ---- closing table ----------------------------------------------------
    fig = plt.figure(figsize=(11.0, 8.0))
    fig.text(0.5, 0.92, "Yields", ha="center", fontsize=16, fontweight="semibold")
    rows = ["%-34s %12s %12s %10s" % ("configuration", "radiative", "Born", "rad/Born"), ""]
    for lab, nr, nb in yields:
        rows.append("%-34s %12d %12d %10.4f" % (lab, nr, nb, nr / nb if nb else float("nan")))
    rows += ["", "%-34s %12s %12s" % ("", "of 1 000 000 generated", "")]
    for lab, nr, nb in yields:
        rows.append("%-34s %11.3f %% %11.3f %%" % (lab, 100 * nr / 1e6, 100 * nb / 1e6))
    fig.text(0.07, 0.80, "\n".join(rows), fontsize=10, va="top", family="monospace",
             linespacing=1.6)
    fig.text(0.07, 0.22, "\n".join(log), fontsize=8, va="top", family="monospace",
             color="#555555", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)

print("wrote", OUT)
for L in log:
    print(" ", L)
