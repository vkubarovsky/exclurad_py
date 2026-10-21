#!/usr/bin/env python3
"""Plots for rc_method_pi0_combos.py.   python3 rc_method_pi0_combos_plots.py <run outdir> <figdir>"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

RUN, FIG = sys.argv[1], sys.argv[2]
os.makedirs(FIG, exist_ok=True)
r = json.load(open(os.path.join(RUN, "rc_method_pi0_combos.json")))
INK, MUTED = "#222222", "#888888"
V0_COL, VB_COL = "#1baf7a", "#6250d6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "legend.frameon": False})
V = np.array(r["vcuts"]); V0 = r["v0"]; E = r["edges"]
NAME = {"Q2": "Q$^2$ (GeV$^2$)", "xB": "x$_B$", "mt": "$-t$ (GeV$^2$)"}
CUTS_TXT = ("cuts as in the note, Sec. 3.5: Q$^2$>1, W>2, p$_{e'}$>2; |ME|<0.2, |MM$^2$(e'p$\\gamma\\gamma$)|<0.004, "
            "MP$_T$<0.75, cone<4$^\\circ$; M$_{\\gamma\\gamma}$ 3$\\sigma$, opening>1$^\\circ$")


def arr(cl):
    return (np.array([c["B"] for c in cl]), np.array([c["B_err"] for c in cl]), np.array([c["A"] for c in cl]))


def A_at(A, v):
    return np.array([np.interp(v, V, a) for a in A])


def chi2_curve(cl):
    B, Eb, A = arr(cl)
    return np.array([np.sum(((B - A[:, j]) / Eb) ** 2) for j in range(len(V))])


def best_v(cl):
    ch = chi2_curve(cl); i = int(np.argmin(ch)); s = slice(max(i - 1, 0), min(i + 2, len(V)))
    if s.stop - s.start < 3:
        return float(V[i])
    p = np.polyfit(V[s], ch[s], 2)
    return float(-p[1] / (2 * p[0])) if p[0] > 0 else float(V[i])


# one global v_cut: the chi2 minimum over the (Q2, xB) cells -- finer than 1D, better filled than 3D
VB = best_v(r["combos"]["Q2-xB"])


def save(fig, name, pdf):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


summary = {"v_best_global": VB, "v0": V0, "chi2": {}, "per_combo_best": {}}
with PdfPages(os.path.join(FIG, "rc_method_pi0_combos.pdf")) as pdf:
    # ---- C1: one variable at a time, the other two integrated ----------------
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.4), sharex="col", gridspec_kw=dict(height_ratios=(2, 1.2)))
    for j, d in enumerate(("Q2", "xB", "mt")):
        cl = r["combos"][d]; B, Eb, A = arr(cl)
        x = np.array([c["mean"][d] for c in cl])
        xe = np.array([[c["mean"][d] - c["ranges"][d][0], c["ranges"][d][1] - c["mean"][d]] for c in cl]).T
        ax, ar = axs[0, j], axs[1, j]
        ax.errorbar(x, B, xerr=xe, yerr=Eb, fmt="o", ms=4, color=INK, lw=0.8, label="B: rad/Born MC, note cuts")
        ax.plot(x, A_at(A, V0), "s-", ms=4, color=V0_COL, lw=1.5, label=f"A: rc_factor, v_cut = {V0}")
        ax.plot(x, A_at(A, VB), "^-", ms=4, color=VB_COL, lw=1.5, label=f"A: rc_factor, v_cut = {VB:.3f}")
        ax.set_ylabel(r"$\delta_{RC}$" if j == 0 else ""); ax.set_ylim(0.65, 1.05)
        ar.axhline(0, color=MUTED, lw=0.8)
        for v, col, mk, dx in ((V0, V0_COL, "s", -0.004), (VB, VB_COL, "^", 0.004)):
            Av = A_at(A, v)
            ar.errorbar(x * (1 + dx), 100 * (B / Av - 1), yerr=100 * Eb / Av, fmt=mk, ms=4, color=col, lw=0.8)
        ar.set_ylim(-25, 15); ar.set_xlabel(NAME[d] + ", other two integrated")
        ar.set_ylabel("B/A $-$ 1 (%)" if j == 0 else "")
        if d == "Q2":
            ar.set_xscale("log"); ar.set_xticks([1, 2, 3, 4, 6]); ar.set_xticklabels(["1", "2", "3", "4", "6"])
    axs[0, 0].legend(fontsize=7.5)
    fig.suptitle(CUTS_TXT, fontsize=8.5)
    fig.tight_layout()
    save(fig, "c1_one_variable", pdf)

    # ---- C2: two variables, the third integrated ------------------------------
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20); cmap = plt.get_cmap("RdBu_r")
    for v, tag in ((V0, "v0"), (VB, "vbest")):
        fig, axs = plt.subplots(1, 3, figsize=(17, 5.6))
        for ax, (dx, dy) in zip(axs, (("xB", "Q2"), ("mt", "Q2"), ("mt", "xB"))):
            key = f"{dy}-{dx}" if f"{dy}-{dx}" in r["combos"] else f"{dx}-{dy}"
            cl = r["combos"][key]; B, Eb, A = arr(cl); Av = A_at(A, v)
            for c, b, e, a in zip(cl, B, Eb, Av):
                x0, x1 = c["ranges"][dx]; y0, y1 = c["ranges"][dy]
                d = 100 * (b / a - 1); de = 100 * e / a
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=cmap(norm(d)), ec="white", lw=1.2))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{d:+.0f}\n±{de:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(d) > 12 else INK)
            ax.set_xlim(E[dx][0], E[dx][-1]); ax.set_ylim(E[dy][0], min(E[dy][-1], 7.0)); ax.grid(False)
            ax.set_xlabel(NAME[dx]); ax.set_ylabel(NAME[dy])
            third = ({"Q2", "xB", "mt"} - {dx, dy}).pop()
            ch = np.sum(((B - Av) / Eb) ** 2)
            ax.set_title(f"{NAME[third].split(' (')[0]} integrated;  $\\chi^2$ = {ch:.0f} / {len(cl)}", fontsize=9)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axs, label=f"B/A $-$ 1 (%),  A at v_cut = {v:.3f} GeV$^2$", shrink=0.85)
        fig.suptitle(f"Two variables at a time, A at v_cut = {v:.3f}.  " + CUTS_TXT, fontsize=8.5)
        save(fig, f"c2_maps_{tag}", pdf)

    # ---- C3: all three ---------------------------------------------------------
    cl = r["combos"]["Q2-xB-mt"]; B, Eb, A = arr(cl); x = np.arange(len(cl))
    fig, (ax, ar) = plt.subplots(2, 1, figsize=(max(10, 0.3 * len(cl) + 3), 6.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=(2, 1.3)))
    ax.errorbar(x, B, yerr=Eb, fmt="o", ms=4, color=INK, label="B: rad/Born MC, note cuts")
    ax.plot(x, A_at(A, V0), "_", ms=12, mew=2, color=V0_COL, label=f"A, v_cut = {V0}")
    ax.plot(x, A_at(A, VB), "_", ms=12, mew=2, color=VB_COL, label=f"A, v_cut = {VB:.3f}")
    ax.set_ylabel(r"$\delta_{RC}$"); ax.legend(fontsize=8, ncol=3)
    ax.set_title("The note's (Q$^2$, x$_B$, $-t$) cells with $\\geq$%s events per arm" % r["nmin"]["3"], fontsize=9)
    ar.axhline(0, color=MUTED, lw=0.8)
    for v, col, mk, dx in ((V0, V0_COL, "s", -0.15), (VB, VB_COL, "^", 0.15)):
        Av = A_at(A, v)
        ar.errorbar(x + dx, 100 * (B / Av - 1), yerr=100 * Eb / Av, fmt=mk, ms=4, color=col, lw=0.8,
                    label=f"v_cut = {v:.3f}: $\\chi^2$ = {np.sum(((B - Av) / Eb) ** 2):.0f} / {len(cl)}")
    ar.set_ylabel("B/A $-$ 1 (%)"); ar.set_ylim(-35, 25); ar.legend(fontsize=7.5, ncol=2, loc="lower left")
    ar.set_xticks(x)
    ar.set_xticklabels([" | ".join(f"{c['ranges'][d][0]}-{c['ranges'][d][1]}" for d in ("Q2", "xB", "mt"))
                        for c in cl], rotation=90, fontsize=6)
    fig.tight_layout()
    save(fig, "c3_cells_3d", pdf)

    # ---- C4: chi2 against the one v_cut, every combination ---------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    shades = plt.get_cmap("Greys")(np.linspace(0.35, 0.95, len(r["combos"])))
    for (k, cl), col in zip(r["combos"].items(), shades):
        ch = chi2_curve(cl)
        vb = best_v(cl)
        summary["chi2"][k] = dict(zip(map(str, V), map(float, ch))); summary["per_combo_best"][k] = vb
        axs[0].plot(V, ch / len(cl), "o-", ms=3, color=col, label=f"{k}: {len(cl)} cells, best v_cut {vb:.3f}")
        veq = np.array([c["v_eq"] for c in cl])
        axs[1].hist(veq[np.isfinite(veq)], bins=np.linspace(0, 0.45, 19), histtype="step", color=col, lw=1.2,
                    label=f"{k}: {np.isfinite(veq).sum()} of {len(cl)}")
    axs[0].axhline(1, color=MUTED, ls="--", lw=0.8)
    axs[0].set_xlabel("one v_cut for every cell (GeV$^2$)"); axs[0].set_ylabel("$\\chi^2$ / cells")
    axs[0].set_yscale("log"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel("v_cut at which A equals B, cell by cell (GeV$^2$)"); axs[1].set_ylabel("cells")
    axs[1].legend(fontsize=7)
    fig.suptitle(CUTS_TXT, fontsize=8.5)
    fig.tight_layout()
    save(fig, "c4_chi2", pdf)

json.dump(summary, open(os.path.join(FIG, "combos_summary.json"), "w"), indent=1)
print(json.dumps({"v_best_global": VB, "per_combo_best": summary["per_combo_best"]}, indent=1))
