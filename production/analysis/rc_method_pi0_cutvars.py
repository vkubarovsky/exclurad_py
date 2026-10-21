#!/usr/bin/env python3
"""Every variable we cut on, N-1: each with all the OTHER cuts applied, rad and Born MC.

    python3 production/analysis/rc_method_pi0_cutvars.py <figdir>

Lines:  solid black   the analysis note's cut (Sec. 3.5; M(gg) from Sec. 3.4)
        dashed blue   the cut applied here, where it differs from the note's
        dotted grey   Tables 7-10 values that Sec. 3.5 does not list (not applied)
The coplanarity angle is theta(X, pi0): X is the missing particle of
e p -> e' p' X, the pi0 is the reconstructed gamma-gamma pair.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402
import rc_method_pi0 as R  # noqa: E402

FIG = sys.argv[1] if len(sys.argv) > 1 else "docs/rc_method_pi0"
RAD, BORN, INK, MUTED, APPLIED = "#2a78d6", "#eb6834", "#222222", "#888888", "#1c5cab"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False})

# name: (label, plot range, cut test, note lines, applied lines if different, Tables-only lines)
MPI2 = R.MPI ** 2
VARS = {
    "Q2":     ("Q$^2$ (GeV$^2$)", (0, 8), lambda k, c: k["Q2"] > 1, [1.0], None, None),
    "W":      ("W (GeV)", (1.5, 4.2), lambda k, c: k["W"] > 2, [2.0], None, None),
    "pe":     ("p$_{e'}$ (GeV)", (0, 10), lambda k, c: k["pe"] > 2, [2.0], None, None),
    "mgg":    ("M$_{\\gamma\\gamma}$ (GeV)", (0.05, 0.22), None, [0.096, 0.168], "mgg3s", None),
    "open":   ("$\\gamma\\gamma$ opening angle (deg)", (0, 20), lambda k, c: k["open"] > 1, [1.0], None, None),
    "me":     ("ME(e'p$\\gamma\\gamma$) (GeV)", (-1.0, 2.0), lambda k, c: np.abs(k["me"]) < 0.2, [-0.2, 0.2], None, None),
    "mm2tot": ("MM$^2$(e'p$\\gamma\\gamma$) (GeV$^2$)", (-0.04, 0.04), lambda k, c: np.abs(k["mm2tot"]) < 0.004,
               [-0.004, 0.004], None, [-0.006, 0.0025]),
    "mpt":    ("MP$_T$(e'p$\\gamma\\gamma$) (GeV)", (0, 1.0), lambda k, c: k["mpt"] < 0.75, [0.75], None, None),
    "cone":   ("coplanarity $\\theta$(X, $\\pi^0$) (deg), X from ep$\\to$e'p'X", (0, 15),
               lambda k, c: k["cone"] < 4, [4.0], None, None),
    "mm2ep":  ("MM$^2$(e'p) (GeV$^2$)", (-0.6, 1.0), None, [], None, [MPI2 - 3 * 0.0861, MPI2 + 3 * 0.0861]),
    "megg":   ("M(e'$\\gamma\\gamma$) (GeV)", (0.2, 2.0), None, [], None, [0.75, 1.2]),
}


def mgg_window(k, cal):
    c = np.where(k["pdet"] == "FD", cal["FD"]["mgg"][0], cal["CD"]["mgg"][0])
    s = np.where(k["pdet"] == "FD", cal["FD"]["mgg"][1], cal["CD"]["mgg"][1])
    return np.abs(k["mgg"] - c) < 3 * s


rng = np.random.default_rng(20261021)
sig = {a: C.sigma_nb(f"{R.DATA}/prov_{a}", a)["mean"] for a in ("rad", "born")}
data = {a: {v: [] for v in list(VARS) + ["w"]} for a in sig}
masks = {a: {v: [] for v in list(VARS) + ["all"]} for a in sig}
cal_all = {}
for pol in ("inb", "outb"):
    S = {a: R.load(a, pol, rng) for a in sig}
    cal = R.calibrate(S["born"]["rec"]); cal_all[pol] = cal
    for a in S:
        k = S[a]["rec"]
        tests = {v: (spec[2](k, cal) if spec[2] else None) for v, spec in VARS.items()}
        tests["mgg"] = mgg_window(k, cal)
        applied = [v for v in tests if tests[v] is not None]
        allc = np.logical_and.reduce([tests[v] for v in applied])
        for v in VARS:
            data[a][v].append(k[v])
            others = [tests[u] for u in applied if u != v]
            masks[a][v].append(np.logical_and.reduce(others))
        masks[a]["all"].append(allc)
        data[a]["w"].append(np.full(len(k["Q2"]), sig[a] / S[a]["n_gen"]))
for a in sig:
    for v in data[a]:
        data[a][v] = np.concatenate(data[a][v])
    for v in masks[a]:
        masks[a][v] = np.concatenate(masks[a][v])

# the applied M(gg) window, for the plot: FD and CD proton, inbending (outbending within 1 MeV)
mw = [(cal_all["inb"][d]["mgg"][0] - 3 * cal_all["inb"][d]["mgg"][1],
       cal_all["inb"][d]["mgg"][0] + 3 * cal_all["inb"][d]["mgg"][1]) for d in ("FD", "CD")]

with PdfPages(os.path.join(FIG, "rc_method_pi0_cutvars.pdf")) as pdf:
    for scale in ("linear", "log"):
        fig, axs = plt.subplots(3, 4, figsize=(16, 10.5))
        for ax, (v, (lbl, rg, test, note, appl, tables)) in zip(axs.ravel(), VARS.items()):
            bins = np.linspace(*rg, 81)
            for a, col in (("born", BORN), ("rad", RAD)):
                m = masks[a][v]
                ax.hist(data[a][v][m], bins=bins, weights=data[a]["w"][m] * 1e6, histtype="step", lw=1.3,
                        color=col, label=f"{a} MC")
                if test is not None or v == "mgg":
                    keep = masks[a]["all"].sum() / m.sum()
            for x in note:
                ax.axvline(x, color=INK, lw=1.1)
            if appl == "mgg3s":
                for lo, hi in mw:
                    ax.axvline(lo, color=APPLIED, lw=0.9, ls="--"); ax.axvline(hi, color=APPLIED, lw=0.9, ls="--")
            for x in tables or []:
                ax.axvline(x, color=MUTED, lw=0.9, ls=":")
            # fraction of each arm the cut keeps, given all the others
            if test is not None or v == "mgg":
                kr = masks["rad"]["all"].sum() / masks["rad"][v].sum()
                kb = masks["born"]["all"].sum() / masks["born"][v].sum()
                ax.set_title(f"keeps: Born {100 * kb:.0f} %, rad {100 * kr:.0f} %", fontsize=8.5)
            else:
                ax.set_title("not cut (all cuts applied)", fontsize=8.5)
            ax.set_xlabel(lbl, fontsize=8.5); ax.set_xlim(rg); ax.set_yscale(scale)
            if scale == "log":
                ax.set_ylim(bottom=0.5)
        ax = axs.ravel()[-1]; ax.axis("off")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=BORN, lw=1.5, label="Born MC"),
                           Line2D([], [], color=RAD, lw=1.5, label="rad MC"),
                           Line2D([], [], color=INK, lw=1.2, label="analysis note cut (Sec. 3.5; M$_{\\gamma\\gamma}$ Sec. 3.4)"),
                           Line2D([], [], color=APPLIED, lw=1, ls="--", label="applied here: M$_{\\gamma\\gamma}$ 3$\\sigma$ of MC (FD, CD)"),
                           Line2D([], [], color=MUTED, lw=1, ls=":", label="Tables 7-10 only, not applied")],
                  loc="center", fontsize=9)
        ax.text(0.5, 0.12, "Each variable with all OTHER cuts applied (N$-$1).\nYield in fb per bin, "
                "all kinematics,\ninbending + outbending.", ha="center", fontsize=8.5, transform=ax.transAxes)
        for a_ in axs[:, 0]:
            a_.set_ylabel("yield (fb per bin)")
        fig.suptitle(f"Every cut variable, N$-$1, rad and Born MC ({scale} scale)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"n1_cutvars_{scale}.png"), dpi=140, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
print("applied M(gg) windows (inb):", [tuple(round(x, 4) for x in w) for w in mw])
