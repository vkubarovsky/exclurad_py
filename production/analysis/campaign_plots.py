#!/usr/bin/env python3
"""Campaign plots: radiative against non-radiative, on one canvas, in nanobarns.

    python3 campaign_plots.py <ntuple root> <channel> <out.pdf> [--mariana DIR]

The two samples of a campaign are a PAIR: same box, same seedless geometry,
one field different in the card.  So they belong on one canvas, and they are
normalised to their own CROSS SECTIONS rather than to unit area -- area
normalisation hides exactly the thing the pair was generated to measure, the
change in rate.  sigma_nb comes from each chunk's config.json and the chunk
spread is printed, because a stale majorant cache once moved sigma by a
factor <B> ~ 2 with no other symptom.

Colours: one blue, one vermilion, far apart in both hue and lightness, and
the two series also differ in line style, so the pair never relies on colour
alone.  Labels are English throughout -- VPK's standing rule.
"""
import argparse
import os
import sys

import numpy as np

MP = 0.9382720813      # proton mass, GeV
MV = 3.0969            # J/psi mass, GeV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as cl                                          # noqa: E402

C_RAD, C_BORN, C_DATA = "#1b4f8f", "#d1562f", "#1a1a1a"
STY = {"rad": dict(color=C_RAD, ls="-", lw=1.8, label="radiative"),
       "born": dict(color=C_BORN, ls="--", lw=1.8, label="non-radiative (Born)")}

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.frameon": False,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

def panels(channel):
    """The twelve panels, with the particles called by their own names.

    The slot names hplus/hminus are right inside the code and wrong on a
    page: for pi0 and eta the two slots are photons, and "p_h+" for a photon
    is simply false.  The names come from the channel, so a pi0 page says
    gamma_1 and a phi page says K+.
    """
    c = cl.CHANNELS[channel]
    h1, h2 = c["names"]
    V, N = c["meson_name"], c["baryon_name"]
    return [
        ("Q2",          r"$Q^2$  [GeV$^2$]",                None, 50, True),
        ("W",           r"$W$  [GeV]",                      None, 50, False),
        ("mt",          r"$-t$  [GeV$^2$]",                 None, 50, True),
        ("nu",          r"$E_\gamma = \nu$  [GeV]",         None, 50, False),
        ("m_pair",      rf"$M({h1}{h2})$  [GeV]",           None, 60, True),
        ("escat_theta", r"$\theta_{e'}$  [deg]",            None, 50, False),
        ("escat_p",     r"$|p_{e'}|$  [GeV]",               None, 50, False),
        ("prot_theta",  rf"$\theta_{{{N}}}$  [deg]",         None, 50, False),
        ("prot_p",      rf"$|p_{{{N}}}|$  [GeV]",            None, 50, False),
        ("hplus_theta", rf"$\theta_{{{h1}}}$  [deg]",        None, 50, False),
        ("hplus_p",     rf"$|p_{{{h1}}}|$  [GeV]",           None, 50, False),
        ("meson_theta", rf"$\theta_{{{V}}}$  [deg]",         None, 50, False),
    ]


def _edges(arrs, rng, nb):
    """Bin edges: the card's range when given, else 0.2-99.8 percentiles.

    Percentiles rather than min/max, because one event at |t| = 5 would
    otherwise set the axis for a million events living below 1.
    """
    if rng is not None:
        return np.linspace(rng[0], rng[1], nb + 1)
    good = [a[np.isfinite(a)] for a in arrs if a is not None and a.size]
    if not good:
        return np.linspace(0, 1, nb + 1)
    v = np.concatenate(good)
    if v.size == 0:
        return np.linspace(0, 1, nb + 1)
    lo, hi = np.percentile(v, 0.2), np.percentile(v, 99.8)
    if hi <= lo:
        lo, hi = v.min(), max(v.max(), v.min() + 1e-6)
    pad = 0.04 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, nb + 1)


def _clip_log(ax, decades=4.0):
    """Hold a log axis to `decades` below the tallest bin.

    Left alone, matplotlib honours a bin holding a single event -- 1e-12 fb
    against a peak of 1e2 -- and fourteen decades of empty paper then squash
    the distribution into the top centimetre of the panel.
    """
    top = 0.0
    for L in ax.get_lines() + list(ax.collections):
        try:
            y = np.asarray(L.get_ydata() if hasattr(L, "get_ydata") else [])
        except Exception:                                          # noqa: BLE001
            continue
        y = y[np.isfinite(y) & (y > 0)]
        if y.size:
            top = max(top, y.max())
    if top > 0:
        ax.set_ylim(top / 10 ** decades, top * 2.2)


NB_TO_FB = 1.0e6
YLAB = "d$\\sigma$ / bin   [fb]"


def _hist_nb(x, edges, sigma, n_gen):
    """Histogram in FEMTOBARNS per bin: each generated event carries sigma/N.

    fb and not nb because the J/psi cross section over this box is 0.33 pb:
    in nanobarns every axis reads 1e-5 and the exponent, not the number, is
    what the eye lands on.
    """
    h, _ = np.histogram(x[np.isfinite(x)], bins=edges)
    w = sigma * NB_TO_FB / n_gen
    return h * w, np.sqrt(h) * w


def _panels_for(sets, channel):
    """The panel list, with the mass panel dropped when the mass is a delta.

    A meson that decays to hadrons is generated with idecay 1, so M(h+h-) is
    the pole mass to float precision and the panel renders as
    "1e-8 + 1.019461" -- an axis showing rounding error.  It is replaced by
    -t', which is the variable the forward edge actually lives in.
    """
    out = []
    for p in panels(channel):
        if p[0] == "m_pair":
            spread = max(float(np.ptp(s["d"]["m_pair"])) for s in sets.values())
            if spread < 1e-3:
                out.append(("mtprime", r"$-t' = -t - |t_{min}|$  [GeV$^2$]", None, 50, True))
                continue
        out.append(p)
    return out


# Panels that only mean something when the recoil baryon was REQUIRED to
# match.  On a no-baryon page the proton slot was never matched, so
# MC::GenMatch is free to attach a REC row with pid 0, 22 or 2112 and momentum
# exactly zero -- which, read as a four-vector, gives |p| = 0 and theta = 90
# deg exactly.  Measured on born inbending 45 nA, no_baryon, N = 18875: 3177
# events, 16.83 %, sit in that spike in prot_p, prot_theta AND mt, because t
# is built from the proton.  t' is worse rather than better: t' = t - t_min
# moves the spike off zero, so it lands among the real entries and stops
# looking like an artefact at all (0 events exactly at zero).
#
# The data page already guarded against this and the MC pages did not.  These
# panels are not thin on a no-baryon page, they are undefined, so they are
# dropped and the page says so instead of drawing them.  Found by the analysis
# session, 2026-10-23.
_NEEDS_BARYON = ("prot", "mt", "mtprime")


def _drop_baryon_panels(pl):
    return [q for q in pl if not (q[0].startswith("prot") or q[0] in ("mt", "mtprime"))]


def page(pdf, sets, title, subtitle, require=None):
    """One page: every panel, both samples, femtobarns per bin.

    `require` is the topology tuple the sample was loaded with.  When slot 1
    (the recoil baryon) is not in it, the proton and t panels are dropped --
    see _NEEDS_BARYON above.  Passing None keeps every panel, which is correct
    only for a sample where the baryon was required.
    """
    channel = next(iter(sets.values()))["d"]["channel"]
    pl = _panels_for(sets, channel)
    dropped = []
    if require is not None and 1 not in tuple(require):
        before = [q[0] for q in pl]
        pl = _drop_baryon_panels(pl)
        dropped = [k for k in before if k not in [q[0] for q in pl]]
        if dropped:
            subtitle = subtitle + ("\n" if "\n" not in subtitle else "  ") + \
                ("the recoil baryon was not required, so the proton slot is unmatched: "
                 + ", ".join(dropped) + " are undefined here and are not drawn")
    nrow = max(1, -(-len(pl) // 3))
    fig, axes = plt.subplots(nrow, 3, figsize=(11.0, 3.1 * nrow), squeeze=False)
    if not cl.CHANNELS[channel]["ordered"]:
        subtitle = subtitle + ("\n" if "\n" not in subtitle else "  ") + \
            "the two decay photons are in MC::Lund order, not sorted by energy "\
            "(slot 1 is the harder one in 50.4 % of events)"
    for ax, (key, lab, rng, nb, logy) in zip(axes.ravel(), pl):
        ed = _edges([s["d"][key] for s in sets.values()], rng, nb)
        c = 0.5 * (ed[1:] + ed[:-1])
        for nm, s in sets.items():
            y, e = _hist_nb(s["d"][key], ed, s["sigma"], s["n_gen"])
            ax.step(c, y, where="mid", **STY[nm])
            ax.fill_between(c, np.maximum(y - e, 1e-12), y + e, step="mid",
                            color=STY[nm]["color"], alpha=0.16, lw=0)
        ax.set_xlabel(lab)
        ax.set_ylabel(YLAB)
        if logy:
            ax.set_yscale("log")
            _clip_log(ax)
    axes.ravel()[0].legend(loc="upper right")
    for ax in axes.ravel()[len(pl):]:
        ax.set_visible(False)
    # THE HEADER IS MEASURED IN INCHES, NOT IN FIGURE FRACTIONS.
    #
    # These were y=0.997, 0.9755 - 0.0105*i and rect top 0.958 -- fractions
    # calibrated by eye against the one figure height this function used to
    # have, 12.4 in.  Dropping the proton panels made the height depend on the
    # panel count (3.1 in per row), and on a 9-row-less page of 9.3 in the same
    # fractions are 25 % fewer inches: the subtitle line spacing fell from
    # 0.130 to 0.098 in against a line height of about 0.11, so two subtitle
    # lines printed on top of each other and the first ran into the title.
    # Found by the analysis session on the merged tip, 2026-10-23, in an
    # end-to-end render -- neither branch shows it alone.
    #
    # The constants below are the old fractions times 12.4, so a 12.4 in page
    # is laid out exactly as before and every other height now matches it
    # physically instead of proportionally.
    H = fig.get_figheight()
    lines = subtitle.split("\n")
    fig.suptitle(title, y=1.0 - 0.037 / H, fontsize=12.5, fontweight="semibold")
    for i, line in enumerate(lines):
        fig.text(0.5, 1.0 - (0.304 + 0.130 * i) / H, line,
                 ha="center", fontsize=8.0, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 1.0 - (0.391 + 0.130 * len(lines)) / H))
    pdf.savefig(fig)
    plt.close(fig)


# The analysis vertex cuts from VPK's note, Tables 7-10.  Read literally,
# |vz| < cut keeps only the downstream end of a cell that sits at -5.5..-0.5,
# which is worth seeing on the plot rather than discovering in a table.
VZ_CUT = {"inb": 2.2, "outb": 2.5}


def page_vertex(pdf, root, configs, channel, cell=None, max_files=8):
    """Everything about the vertex, on one page.

    Six panels, because the vertex is the one distribution nothing plotted
    and therefore the one that went wrong: a Born arm spent a day at a single
    point, z = 0, outside the target cell, and no figure could have shown it.
    VPK, 2026-10-22: plot this after every generator run.

    GEMC keeps the LUND vertex exactly -- BEAM_V and SPREAD_V in the gcard are
    not applied to LUND input, checked on 40k events per arm -- so the
    generated vertex is also the true vertex in the detector, and
    reconstructed minus generated is the vertex resolution.
    """
    import glob as _g
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    ax = axes.ravel()
    lo, hi = cell if cell else (-5.5, -0.5)

    # ---- 1. the generated vertex -----------------------------------------
    pad = max(0.6, 0.25 * (hi - lo))
    ed = np.linspace(lo - pad, hi + pad, 80)
    pol0 = configs[0][0] if configs else "inb"
    bg0 = configs[0][1] if configs else "none"
    for nm in ("born", "rad"):
        pat = os.path.join(root, nm, "out_" + pol0, f"*_bg{bg0}_*.f32")
        if not _g.glob(pat):
            continue
        d = cl.load(pat, channel, rec=False, require=(), max_files=max_files)
        if not d.get("has_vz"):
            ax[0].text(0.5, 0.5 - 0.08 * (nm == "rad"),
                       f"{nm}: record has no vertex (pre-2026-10-22 ntuple)",
                       transform=ax[0].transAxes, ha="center", fontsize=8,
                       color="#aa3333")
            continue
        v = d["vz_gen"]
        ax[0].hist(v, bins=ed, histtype="step", density=True, color=STY[nm]["color"],
                   ls=STY[nm]["ls"], lw=STY[nm]["lw"],
                   label=f"{STY[nm]['label']}: {v.min():.2f} .. {v.max():.2f} cm")
    for x in (lo, hi):
        ax[0].axvline(x, color="#333333", lw=1.1)
    ax[0].set_xlabel("generated vertex $z$  [cm]")
    ax[0].set_ylabel("normalised")
    ax[0].set_title(f"Generated vertex: uniform over the {hi - lo:.0f} cm cell", fontsize=9)
    ax[0].legend(loc="lower center", fontsize=7)

    # ---- 2, 3. the reconstructed electron vertex, per polarity -----------
    for i, pol in enumerate(("inb", "outb")):
        a = ax[1 + i]
        cfg = [c for c in configs if c[0] == pol]
        if not cfg:
            a.text(0.5, 0.5, f"no {pol} configuration on disk", transform=a.transAxes,
                   ha="center", fontsize=8.5, color="#777777")
            a.set_xticks([]); a.set_yticks([]); continue
        bg = cfg[0][1]
        edr = np.linspace(lo - 8, hi + 8, 110)
        for nm in ("born", "rad"):
            pat = os.path.join(root, nm, "out_" + pol, f"*_bg{bg}_*.f32")
            if not _g.glob(pat):
                continue
            d = cl.load(pat, channel, rec=True, require=cl.TOPOLOGY["full"],
                        max_files=max_files)
            if not d.get("has_vz"):
                continue
            v = d["escat_vz"]
            a.hist(v, bins=edr, histtype="step", density=True, color=STY[nm]["color"],
                   ls=STY[nm]["ls"], lw=STY[nm]["lw"], label=f"{STY[nm]['label']}, N={len(v)}")
        for x in (lo, hi):
            a.axvline(x, color="#333333", lw=1.0)
        c = VZ_CUT.get(pol)
        if c:
            for x in (-c, c):
                a.axvline(x, color=C_BORN, lw=1.0, ls="--")
        a.set_xlabel(f"reconstructed electron $v_z$, {pol}bending  [cm]")
        a.set_title(f"solid: target;  dashed: note |vz| < {c}", fontsize=8.5)
        a.legend(loc="upper right", fontsize=7)

    # ---- 4. resolution ----------------------------------------------------
    a = ax[3]
    edres = np.linspace(-6, 6, 90)
    rows = []
    for pol in ("inb", "outb"):
        cfg = [c for c in configs if c[0] == pol]
        if not cfg:
            continue
        bg = cfg[0][1]
        for nm in ("born", "rad"):
            pat = os.path.join(root, nm, "out_" + pol, f"*_bg{bg}_*.f32")
            if not _g.glob(pat):
                continue
            d = cl.load(pat, channel, rec=True, require=cl.TOPOLOGY["full"],
                        max_files=max_files)
            if not d.get("has_vz"):
                continue
            r = d["escat_vz"] - d["vz_gen"]
            q = np.percentile(r, [16, 50, 84])
            a.hist(r, bins=edres, histtype="step", density=True, color=STY[nm]["color"],
                   ls="-" if pol == "inb" else "--", lw=1.4,
                   label=f"{nm} {pol}: med {q[1]:+.2f}, $\\sigma_{{68}}$ {0.5 * (q[2] - q[0]):.2f}")
            keep_cut = (np.abs(d["escat_vz"]) < VZ_CUT.get(pol, 99)).mean()
            keep_cell = ((d["escat_vz"] > lo) & (d["escat_vz"] < hi)).mean()
            rows.append((f"{nm} {pol}", 100 * keep_cut, 100 * keep_cell, len(r)))
    a.set_xlabel("electron $v_z$: reconstructed $-$ generated  [cm]")
    a.set_title("Vertex resolution, electron", fontsize=9)
    if a.has_data():
        a.legend(loc="upper left", fontsize=6.6)

    # ---- 5. electron minus proton ----------------------------------------
    a = ax[4]
    edd = np.linspace(-10, 10, 90)
    for pol in ("inb", "outb"):
        cfg = [c for c in configs if c[0] == pol]
        if not cfg:
            continue
        bg = cfg[0][1]
        for nm in ("born", "rad"):
            pat = os.path.join(root, nm, "out_" + pol, f"*_bg{bg}_*.f32")
            if not _g.glob(pat):
                continue
            d = cl.load(pat, channel, rec=True, require=cl.TOPOLOGY["full"],
                        max_files=max_files)
            if not d.get("has_vz"):
                continue
            dv = d["escat_vz"] - d["prot_vz"]
            a.hist(dv, bins=edd, histtype="step", density=True, color=STY[nm]["color"],
                   ls="-" if pol == "inb" else "--", lw=1.4,
                   label=f"{nm} {pol}, N={len(dv)}")
    a.set_xlabel("$v_z(e') - v_z(p')$ reconstructed  [cm]")
    a.set_title("Electron$-$proton vertex difference", fontsize=9)
    if a.has_data():
        a.legend(loc="upper left", fontsize=6.6)

    # ---- 6. the table -----------------------------------------------------
    a = ax[5]; a.axis("off")
    if rows:
        txt = ["Fraction of reconstructed electrons kept", "",
               "%-11s %9s %11s" % ("", "|vz|<cut", f"{lo:.1f}..{hi:.1f}")]
        for nm, kc, kcell, n in rows:
            txt.append("%-11s %8.1f%% %10.1f%%" % (nm, kc, kcell))
        txt += ["", "cut = %.1f cm inbending, %.1f cm outbending" % (VZ_CUT["inb"], VZ_CUT["outb"]),
                "(note Tables 7-10).  Read literally, |vz| < cut",
                "keeps only the downstream end of a cell that",
                "sits at %.1f .. %.1f cm." % (lo, hi)]
        a.text(0.0, 0.95, "\n".join(txt), transform=a.transAxes, va="top",
               fontsize=8.2, family="monospace")

    fig.suptitle("Interaction point, radiative and Born", y=0.995, fontsize=12.5,
                 fontweight="semibold")
    fig.text(0.5, 0.963, "GEMC keeps the LUND vertex exactly, so the generated vertex is the "
             "true one in the detector and reconstructed minus generated is the resolution",
             ha="center", fontsize=8.0, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.952))
    pdf.savefig(fig)
    plt.close(fig)


def page_radiation(pdf, sets, title):
    """Where the radiation went.

    The Born sample has no production photon at all, and no decay-lepton
    radiation either (its card sets idecay 1 against the radiative sample's
    idecay 2).  Both absences are deliberate -- Born means no radiation of
    any kind -- and together they are the cleanest check that the two
    samples are what their cards say.  It is also why the two M(e+e-)
    lineshapes differ by more than the production radiation alone.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    specs = [("gprod_p", r"$E_\gamma$  production radiation  [GeV]", True),
             ("gdec_p", r"$E_\gamma$  decay-lepton radiation  [GeV]", True),
             ("y", r"$y = \nu / E$", False)]
    for ax, (key, lab, logy) in zip(axes, specs):
        arrs = [s_["d"][key][s_["d"][key] > 1e-6] for s_ in sets.values()]
        arrs = [a for a in arrs if a.size]
        ed = _edges(arrs if key != "y" else [s_["d"][key] for s_ in sets.values()], None, 45)
        for nm, s_ in sets.items():
            x = s_["d"][key]
            if key != "y":
                x = x[x > 1e-6]
            if x.size == 0:
                continue
            y, _ = _hist_nb(x, ed, s_["sigma"], s_["n_gen"])
            ax.step(0.5 * (ed[1:] + ed[:-1]), y, where="mid", **STY[nm])
        ax.set_xlabel(lab)
        ax.set_ylabel(YLAB)
        if logy:
            ax.set_yscale("log")
            _clip_log(ax)
    for ax, key in zip(axes[:2], ("gprod_p", "gdec_p")):
        fr = {nm: 100.0 * (s_["d"][key] > 1e-6).mean() for nm, s_ in sets.items()}
        if max(fr.values()) == 0.0:
            ax.text(0.5, 0.5, "no such photon in either sample\n"
                              "(idecay 1: the meson decays to hadrons)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, color="#777777")
            ax.set_yscale("linear")
            ax.set_yticks([])
        else:
            ax.set_title("fraction of events carrying one\n"
                         + "      ".join(f"{STY[nm]['label']}:  {v:.1f} %" for nm, v in fr.items()),
                         fontsize=7.6, color="#444444", pad=5)
    axes[2].legend(loc="upper left")
    fig.suptitle(title, y=0.995, fontsize=12.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    pdf.savefig(fig)
    plt.close(fig)


def page_ratio(pdf, sets, title, subtitle):
    """radiative / Born, bin by bin.  This is eta-bar in that binning: a flat
    line means radiation only rescales, a slope means it reshapes."""
    ch = next(iter(sets.values()))["d"]["channel"]
    h1, h2 = cl.CHANNELS[ch]["names"]
    N = cl.CHANNELS[ch]["baryon_name"]
    keys = [("Q2", r"$Q^2$  [GeV$^2$]", 24), ("mt", r"$-t$  [GeV$^2$]", 24),
            ("nu", r"$E_\gamma$  [GeV]", 24), ("W", r"$W$  [GeV]", 24),
            ("mtprime", r"$-t'$  [GeV$^2$]", 24), (f"prot_theta", rf"$\theta_{{{N}}}$  [deg]", 24)]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    r, b = sets["rad"], sets["born"]
    for ax, (key, lab, nb) in zip(axes.ravel(), keys):
        ed = _edges([r["d"][key], b["d"][key]], None, nb)
        c = 0.5 * (ed[1:] + ed[:-1])
        yr, er = _hist_nb(r["d"][key], ed, r["sigma"], r["n_gen"])
        yb, eb = _hist_nb(b["d"][key], ed, b["sigma"], b["n_gen"])
        ok = (yb > 0) & (yr > 0)
        rat = np.where(ok, yr / np.where(yb > 0, yb, 1), np.nan)
        err = np.where(ok, rat * np.sqrt((er / np.where(yr > 0, yr, 1)) ** 2
                                         + (eb / np.where(yb > 0, yb, 1)) ** 2), np.nan)
        ax.errorbar(c, rat, yerr=err, fmt="o", ms=3.0, color=C_RAD, lw=1.0, capsize=0)
        ax.axhline(1.0, color="#999999", lw=0.9, ls=":")
        tot = r["sigma"] / b["sigma"]
        ax.axhline(tot, color=C_BORN, lw=1.2, ls="--")
        ax.set_xlabel(lab)
        ax.set_ylabel("radiative / Born")
    axes.ravel()[0].text(0.03, 0.06,
                         f"dashed: integrated ratio {sets['rad']['sigma'] / sets['born']['sigma']:.4f}",
                         transform=axes.ravel()[0].transAxes, fontsize=7.5, color=C_BORN)
    fig.suptitle(title, y=0.995, fontsize=12.5, fontweight="semibold")
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=8.2, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------- Mariana ---
def load_mariana(d, topology, period, ebeam=10.6, fiducial=(8.0, 15.75)):
    """Mariana Tenorio's CLAS12 J/psi ntuples, with her event selection.

    Three traps, all of them measured and all of them silent:
      - angles here are in RADIANS, unlike the upstream toroot_v2 ntuple;
      - the FT fiducial is NOT the branch called radius_FT (that is a cluster
        shape, 0..3.8); it is sqrt(x_FT^2 + y_FT^2), and the thesis cuts
        8 < r < 15.75 cm.  A cut on radius_FT > 8 returns zero events and
        reads as an empty sample;
      - t_prime reaches -3.6e8 because t_min ~ 1/E_photon^2 and E_photon
        passes through zero, so E_photon >= 8.1 has to come FIRST.
    These files carry no run_number and beamCharge is a cumulative scaler, so
    there is NO luminosity in them: data and MC can be compared in SHAPE only.
    """
    import ROOT                                                    # noqa: E402
    path = os.path.join(d, "by_topology", topology, period + ".root")
    if not os.path.exists(path):
        raise RuntimeError("no such data file: " + path)
    br = ["E_photon", "Q2", "W", "t_prime", "invariantMass",
          "scattered_theta", "scattered_p", "proton_theta", "proton_p",
          "positron_theta", "positron_p", "electron_theta", "electron_p",
          "time_ee", "x_FT", "y_FT"]
    df = ROOT.RDataFrame("results", path)
    v = {k: np.asarray(a, dtype=float) for k, a in df.AsNumpy(br).items()}
    n = len(v["E_photon"])
    r = np.hypot(v["x_FT"], v["y_FT"])
    # THE FT FIDUCIAL, with where each number comes from.
    #
    # The thesis (p. 73) uses 8 < r < 15.75 cm: the nominal 7.25 .. 16.5
    # trimmed by half a crystal width at each edge.  This code had r <= 16.065
    # with NO lower bound, and I do not know the provenance of 16.065 -- it is
    # not the thesis number and nothing here records where it came from.  The
    # thesis values are therefore the default, because the point of this
    # loader is to reproduce HER selection; `fiducial` overrides it for anyone
    # who wants the older behaviour and wants to say so.
    #
    # The two differ by 5.4 % of the baseline sample, measured on ee+e-p
    # F18in: 6452 events under r <= 16.065, 6102 under 8 < r < 15.75.
    rlo, rhi = fiducial
    sel = (v["E_photon"] >= 8.1) & (np.abs(v["time_ee"]) <= 2.5) & (r > rlo) & (r < rhi)
    out = dict(n_read=n, n_kept=int(sel.sum()), path=path)
    out["nu"] = v["E_photon"][sel]
    out["Q2"] = v["Q2"][sel]
    out["W"] = v["W"][sel]
    out["m_pair"] = v["invariantMass"][sel]
    # HER t_prime IS OUR mtprime, AND IT IS POSITIVE.
    #
    # This line was  out["mt"] = -v["t_prime"]  and "mt" is the panel labelled
    # -t, which campaign_lib fills with |t| > 0.  Her t_prime is already
    # |t| - |t_min| >= 0 by the thesis definition (Eq. 43, t' = -(t - t_min)
    # with t and t_min both negative and |t| > |t_min|), so negating it put a
    # negative number on an axis the MC fills with a positive one.  Measured
    # on ee+e-p F18in against born inbending 45 nA: data median -1.765 over
    # -7.43 .. +0.83 against MC +1.435 over +0.51 .. +5.12.  The two never
    # overlapped, so that panel has never compared anything.  Two faults at
    # once: the sign, and the variable.  Found by the analysis session,
    # 2026-10-23.
    #
    # |t| CANNOT BE FORMED FROM THESE FILES.  They carry t_prime and no t_min,
    # so there is no |t| to put on an -t axis; the comparable quantity is t'
    # on both sides.  page_data therefore plots mtprime when data is present.
    #
    # AND THE TWO t' ARE NOT BUILT WITH THE SAME t_min.  Hers comes from her
    # code, ours from campaign_lib, and the thesis Eq. 44 prints a t_min that
    # is dimensionless rather than a GeV^2 -- so a disagreement on this panel
    # may be a convention and not physics.  Measured in 2.9 < M < 3.3: data t'
    # median +1.202 over 55 events, MC +0.671 over 1170, a factor 1.8, with
    # our |t_min| median 0.679 at the same kinematics.  Do not read that as a
    # physics result until both sides use one t_min.
    # |t| IS RECOVERABLE FROM HER OWN PROTON, AND NEEDS NO t_min.
    # For t taken at the target vertex, t = (p - p')^2 = 2 m_p^2 - 2 m_p E_p',
    # so |t| = 2 m_p (sqrt(p_p^2 + m_p^2) - m_p) -- the recoil momentum and
    # nothing else.  That is the quantity to compare, because it removes both
    # t_min conventions from the comparison instead of reconciling them.
    #
    # DO NOT USE HER PUBLISHED t_prime.  On the baseline it EXCEEDS the |t|
    # rebuilt from her own proton -- medians 1.762 against 1.028 -- which
    # requires a negative |t_min| and so cannot be |t| - |t_min| there at all.
    # In the J/psi window it is 1.202 against |t| 1.220, implying |t_min|
    # = 0.018 where ours at her own nu and Q2 is 0.559.  Measured against our
    # Born MC in 2.9 < M < 3.3, data/MC is
    #     1.79  with her t_prime as published
    #     1.02  with t' built from OUR t_min on both sides
    #     0.87  with |t|, no t_min anywhere
    # so the factor was a convention and not physics, and it vanishes the
    # moment one definition is used.  Found by the analysis session and
    # reproduced here, 2026-10-23.
    #
    # mtprime is still provided, from OUR t_min on her kinematics, for anyone
    # who wants t'; it is not what the data page plots.
    pp = v["proton_p"][sel]
    out["mt"] = 2 * MP * (np.sqrt(pp ** 2 + MP ** 2) - MP)
    out["t_prime_published"] = v["t_prime"][sel]
    _nu, _Q2 = out["nu"], out["Q2"]
    _W2 = MP ** 2 + 2 * MP * _nu - _Q2
    with np.errstate(invalid="ignore", divide="ignore"):
        _W = np.where(_W2 > 0, np.sqrt(np.abs(_W2)), np.nan)
        _Eg = (_W2 - _Q2 - MP ** 2) / (2 * _W)
        _pg = np.sqrt(np.maximum(_Eg ** 2 + _Q2, 0))
        _EV = (_W2 + MV ** 2 - MP ** 2) / (2 * _W)
        _pV = np.sqrt(np.maximum(_EV ** 2 - MV ** 2, 0))
        _t0 = np.abs(-_Q2 + MV ** 2 - 2 * (_Eg * _EV - _pg * _pV))
    out["mtprime"] = out["mt"] - _t0
    for a, b in (("escat", "scattered"), ("prot", "proton"),
                 ("hplus", "positron"), ("hminus", "electron")):
        out[a + "_theta"] = np.degrees(v[b + "_theta"][sel])
        out[a + "_p"] = v[b + "_p"][sel]
    return out


def page_data(pdf, sets, data_all, data_sig, title, subtitle):
    """MC against data, SHAPE ONLY -- these files carry no luminosity.

    The mass panel shows the WHOLE baseline sample, because that is where the
    J/psi peak is visible sitting on its Bethe-Heitler background, and the
    background is most of it.  Every other panel shows only the mass window:
    comparing J/psi MC to a BH-dominated sample would disagree everywhere and
    mean nothing.  The two data sets are labelled separately for that reason.
    """
    keys = [("nu", r"$E_\gamma$  [GeV]", (8.0, 10.6), 20),
            ("Q2", r"$Q^2$  [GeV$^2$]", (0.0, 0.30), 18),
            ("W", r"$W$  [GeV]", (4.05, 4.60), 18),
            ("escat_theta", r"$\theta_{e'}$  [deg]", (2.0, 5.0), 18),
            ("prot_theta", r"$\theta_{p'}$  [deg]", (0.0, 50.0), 20),
            ("hplus_theta", r"$\theta_{e^+}$  [deg]", (0.0, 40.0), 20),
            ("hplus_p", r"$|p_{e^+}|$  [GeV]", (0.0, 9.0), 20),
            ("mt", r"$-t$  [GeV$^2$]  (from the recoil proton)", (0.0, 4.0), 20)]
    # HAS A PROTON MEANS MOST EVENTS HAVE ONE, not that the array is finite.
    # This test was `size > 0 and isfinite(...).any()`, and a zero is finite:
    # on the ee+e- data, 21385 of 27701 entries -- 77.2 % -- are exactly 0
    # because those events have no detected proton, and the panel was drawn
    # with a spike at zero made of absences.  That is the same artefact as the
    # unmatched-proton slot on the MC no-baryon page, fixed this morning, and
    # the data page had it too.  The proton and t panels are defined only
    # where there is a proton, so they are dropped unless most events have
    # one, and the fraction is said rather than left to be noticed.
    _pt = np.asarray(data_sig.get("prot_theta", np.array([])), dtype=float)
    _have = float(np.count_nonzero(np.isfinite(_pt) & (_pt != 0.0))) / max(_pt.size, 1)
    has_proton = _pt.size > 0 and _have > 0.5
    if not has_proton:
        keys = [k for k in keys if not k[0].startswith("prot")]
        if _pt.size and _have > 0:
            subtitle = subtitle + ("\n" if "\n" not in subtitle else "  ") + \
                ("a recoil proton is reconstructed in only %.0f %% of these events, "
                 "so the proton and t panels are not drawn" % (100.0 * _have))
    # t WAS NOT ON THIS PAGE AT ALL until 2026-10-23, which is a worse gap
    # than the loader bug that led me here: t is the variable the analysis
    # bins in, and the only data-against-MC comparison in the book did not
    # contain it.  It is plotted as |t| - |t_min| and labelled so, rather than
    # as t' -- her thesis calls that quantity t' and campaign_lib calls it
    # -t', the same number under two names, and writing either would have one
    # reader in two reading a sign that is not there.
    #

    if not has_proton or "mt" not in data_sig or not np.isfinite(
            np.asarray(data_sig.get("mt", np.array([])))).any():
        keys = [k for k in keys if k[0] != "mt"]

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 9.8))
    ax = axes.ravel()[0]
    ed = np.linspace(2.0, 3.6, 40)
    c = 0.5 * (ed[1:] + ed[:-1])
    h, _ = np.histogram(data_all["m_pair"], bins=ed)
    n = h.sum()
    ax.errorbar(c, h / n, yerr=np.sqrt(h) / n, fmt="o", ms=3.2, color=C_DATA, lw=1.0,
                capsize=0, label=f"data, all {data_all['n_kept']}")
    for nm, s_ in sets.items():
        hh, _ = np.histogram(s_["d"]["m_pair"], bins=ed)
        if hh.sum():
            ax.step(c, hh / hh.sum(), where="mid", **STY[nm])
    ax.axvspan(2.9, 3.3, color="#1b4f8f", alpha=0.07, lw=0)
    ax.set_xlabel(r"$M(e^+e^-)$  [GeV]")
    ax.set_ylabel("fraction / bin")
    ax.set_yscale("log")
    _clip_log(ax, 3.5)
    ax.legend(loc="upper left", fontsize=7.0)
    ax.set_title("whole baseline sample; shaded = the window used below",
                 fontsize=7.4, color="#444444", pad=4)

    for ax, (key, lab, rng, nb) in zip(axes.ravel()[1:], keys):
        ed = np.linspace(rng[0], rng[1], nb + 1)
        c = 0.5 * (ed[1:] + ed[:-1])
        hd, _ = np.histogram(data_sig[key][np.isfinite(data_sig[key])], bins=ed)
        nd = hd.sum()
        if nd:
            ax.errorbar(c, hd / nd, yerr=np.sqrt(hd) / nd, fmt="o", ms=3.4,
                        color=C_DATA, lw=1.0, capsize=0,
                        label=f"data, {int(nd)} in window")
        for nm, s_ in sets.items():
            h, _ = np.histogram(s_["d"][key][np.isfinite(s_["d"][key])], bins=ed)
            if h.sum():
                ax.step(c, h / h.sum(), where="mid", **STY[nm])
        ax.set_xlabel(lab)
        ax.set_ylabel("fraction / bin")
    axes.ravel()[1].legend(loc="upper left", fontsize=7.0)
    for ax in axes.ravel()[1 + len(keys):]:
        ax.set_visible(False)
    fig.suptitle(title, y=0.997, fontsize=12.5, fontweight="semibold")
    # two lines, explicitly: one line of this ran off both edges of the page
    for i, line in enumerate(subtitle.split("\n")):
        fig.text(0.5, 0.9715 - 0.0135 * i, line, ha="center", fontsize=7.9,
                 color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.947))
    pdf.savefig(fig)
    plt.close(fig)
