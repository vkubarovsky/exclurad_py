#!/usr/bin/env python3
"""Build the J/psi campaign plot book: radiative against Born, plus the data.

    python3 make_jpsi_plots.py <ntuple root> <out.pdf> [mariana dir]

<ntuple root> holds rad/out_inb, rad/out_outb, born/out_*, rad/prov, born/prov
exactly as the campaign directory does, so it runs equally on /volatile or on
a copy.  One LUND feeds all four GEMC configurations of a mode, so the
generator-level pages are built once, from the inbending-with-background set,
and are the full million either way.
"""
import gc
import os
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
CH = "jpsi"
CONFIGS = [("inb", "45nA", "inbending, 45 nA background"),
           ("outb", "50nA", "outbending, 50 nA background"),
           ("inb", "none", "inbending, no background"),
           ("outb", "none", "outbending, no background")]
BOX = ("RG-A Fall 2018  |  gemc 5.13 + coatjava 10.0.7  |  E = 10.6 GeV  |  "
       r"$Q^2$ 0.001-0.25, $W^2$ 16.29-20.80, $t$ -5.00..-0.30 GeV$^2$, "
       "vcut 0, FT electron")

SIG = {m: cl.sigma_nb(os.path.join(ROOT_DIR, m, "prov"), m) for m in ("rad", "born")}
NGEN = {m: 200 * 5000 for m in ("rad", "born")}
log = []


def pat(mode, pol, bg):
    return os.path.join(ROOT_DIR, mode, f"out_{pol}", f"*_bg{bg}_*.f32")


def sets_for(pol, bg, rec, require):
    out = {}
    for m in ("rad", "born"):
        d = cl.load(pat(m, pol, bg), CH, rec=rec, require=require)
        out[m] = dict(d=d, sigma=SIG[m]["mean"], n_gen=NGEN[m])
    return out


with PdfPages(OUT) as pdf:
    # ---- title page -------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 8.0))
    fig.text(0.5, 0.80, "J/$\\psi$ photoproduction at CLAS12", ha="center",
             fontsize=23, fontweight="semibold")
    fig.text(0.5, 0.735, "exclurad_py campaign  prod_jpsi_f18in_20261009", ha="center",
             fontsize=12, color="#555555")
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
    g = sets_for("inb", "45nA", rec=False, require=())
    log.append(f"generated: rad {g['rad']['d']['n_kept']}, born {g['born']['d']['n_kept']}")
    cp.page(pdf, g, "Generator level - the full million", BOX)
    cp.page_radiation(pdf, g, "Generator level - where the radiation goes")
    cp.page_ratio(pdf, g, "Generator level - radiative / Born, bin by bin",
                  "a flat line means radiation only rescales; a slope means it reshapes")
    del g
    gc.collect()

    # ---- reconstructed, four configurations, exclusive --------------------
    yields = []
    for pol, bg, lab in CONFIGS:
        s = sets_for(pol, bg, rec=True, require=cl.TOPOLOGY["ee+e-p"])
        n = {m: s[m]["d"]["n_kept"] for m in s}
        yields.append((lab, n["rad"], n["born"]))
        log.append(f"rec e'pe+e- {lab}: rad {n['rad']}, born {n['born']}")
        cp.page(pdf, s, f"Reconstructed  e' p e$^+$ e$^-$  -  {lab}",
                f"all four tracks matched   |   rad {n['rad']} of 1 000 000"
                f" ({100 * n['rad'] / 1e6:.3f} %),  Born {n['born']}"
                f" ({100 * n['born'] / 1e6:.3f} %)")
        if bg == "none":
            cp.page_ratio(pdf, s, f"Reconstructed - radiative / Born - {lab}",
                          "acceptance divides out of this ratio; what is left is radiation")
        del s
        gc.collect()

    # ---- reconstructed, three-track topology ------------------------------
    s3 = sets_for("inb", "45nA", rec=True, require=cl.TOPOLOGY["ee+e-"])
    log.append(f"rec e'e+e- inb 45nA: rad {s3['rad']['d']['n_kept']}, "
               f"born {s3['born']['d']['n_kept']}")
    cp.page(pdf, s3, "Reconstructed  e' e$^+$ e$^-$  (no proton required)  -  inbending, 45 nA",
            "the proton is dropped: no t measurement, roughly four times the data statistics")

    # ---- data -------------------------------------------------------------
    # One page per (topology, period).  The exclusive topology is the four-track
    # sample our MC was cut to; the three-track one drops the proton and holds
    # about four times the data events, at the price of the t measurement.
    if MARIANA:
        plan = [("ee+e-p", cl.TOPOLOGY["ee+e-p"], "inb", "45nA", "F18in"),
                ("ee+e-p", cl.TOPOLOGY["ee+e-p"], "outb", "50nA", "F18out"),
                ("ee+e-", cl.TOPOLOGY["ee+e-"], "inb", "45nA", "F18in"),
                ("ee+e-", cl.TOPOLOGY["ee+e-"], "outb", "50nA", "F18out")]
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
