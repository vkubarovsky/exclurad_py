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


def page(pdf, sets, title, subtitle):
    """One page: every panel, both samples, femtobarns per bin."""
    channel = next(iter(sets.values()))["d"]["channel"]
    fig, axes = plt.subplots(4, 3, figsize=(11.0, 12.4))
    pl = _panels_for(sets, channel)
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
    fig.suptitle(title, y=0.997, fontsize=12.5, fontweight="semibold")
    for i, line in enumerate(subtitle.split("\n")):
        fig.text(0.5, 0.9755 - 0.0105 * i, line, ha="center", fontsize=8.0, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.958))
    pdf.savefig(fig)
    plt.close(fig)


def page_vertex(pdf, sets, lund_patterns, title, cell=None):
    """Where along the target the events happen.

    Its own page, not a panel, because this is the one distribution that was
    never looked at and therefore the one that went wrong: on 2026-10-21 a
    Born arm came out as a point at z = 0, outside the cell, because itarg
    gated the vertex draw as well as the radiator.  GEMC keeps the LUND
    vertex exactly, so a point target there is a point target in the
    detector, and acceptance follows the vertex.

    VPK, the same day: plot the vertex after every generator run.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ed = None
    for nm, pat in lund_patterns.items():
        try:
            v = cl.lund_vz(pat)
        except Exception as e:                                     # noqa: BLE001
            axes[0].text(0.5, 0.5 - 0.1 * list(lund_patterns).index(nm),
                         f"{nm}: {e}", transform=axes[0].transAxes,
                         ha="center", fontsize=7.5, color="#aa3333")
            continue
        if ed is None:
            lo, hi = (cell if cell else (v.min(), v.max()))
            pad = max(0.35, 0.08 * (hi - lo))
            ed = np.linspace(lo - pad, hi + pad, 90)
        axes[0].hist(v, bins=ed, histtype="step", density=True,
                     color=STY[nm]["color"], ls=STY[nm]["ls"], lw=STY[nm]["lw"],
                     label=f"{STY[nm]['label']}   {v.min():.2f} .. {v.max():.2f} cm")
        axes[1].text(0.02, 0.80 - 0.3 * list(lund_patterns).index(nm),
                     f"{STY[nm]['label']}\n  span {v.max() - v.min():.3f} cm,"
                     f"  median {np.median(v):.3f}\n  n = {len(v)}",
                     transform=axes[1].transAxes, fontsize=8.5,
                     color=STY[nm]["color"], va="top", family="monospace")
    if cell:
        for x in cell:
            axes[0].axvline(x, color="#888888", lw=0.9, ls=":")
        axes[1].text(0.02, 0.12, f"cell from the card: {cell[0]:.2f} .. {cell[1]:.2f} cm",
                     transform=axes[1].transAxes, fontsize=8.5, color="#555555",
                     family="monospace")
    axes[0].set_xlabel("vertex $z$  [cm]")
    axes[0].set_ylabel("events / bin, normalised")
    axes[0].legend(loc="upper center", fontsize=7.4)
    axes[1].axis("off")
    fig.suptitle(title, y=0.99, fontsize=12.5, fontweight="semibold")
    fig.text(0.5, 0.925, "GEMC keeps the LUND vertex exactly -- a point here is a point "
             "in the detector, and acceptance follows the vertex",
             ha="center", fontsize=8.0, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
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
def load_mariana(d, topology, period, ebeam=10.6):
    """Mariana Tenorio's CLAS12 J/psi ntuples, with her event selection.

    Three traps, all of them measured and all of them silent:
      - angles here are in RADIANS, unlike the upstream toroot_v2 ntuple;
      - the FT fiducial is NOT the branch called radius_FT (that is a cluster
        shape, 0..3.8); it is sqrt(x_FT^2 + y_FT^2) <= 16.065 cm.  A cut on
        radius_FT > 8 returns zero events and reads as an empty sample;
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
    sel = (v["E_photon"] >= 8.1) & (np.abs(v["time_ee"]) <= 2.5) & (r <= 16.065)
    out = dict(n_read=n, n_kept=int(sel.sum()), path=path)
    out["nu"] = v["E_photon"][sel]
    out["Q2"] = v["Q2"][sel]
    out["W"] = v["W"][sel]
    out["m_pair"] = v["invariantMass"][sel]
    out["mt"] = -v["t_prime"][sel]
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
            ("hplus_p", r"$|p_{e^+}|$  [GeV]", (0.0, 9.0), 20)]
    has_proton = data_sig["prot_theta"].size > 0 and np.isfinite(data_sig["prot_theta"]).any()
    if not has_proton:
        keys = [k for k in keys if not k[0].startswith("prot")]

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
