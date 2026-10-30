#!/usr/bin/env python3
"""Plots for rc_method_pi0_grid.py.   python3 rc_method_pi0_grid_plots.py <run outdir> <figdir>"""
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
r = json.load(open(os.path.join(RUN, "rc_method_pi0_grid.json")))
INK, MUTED, A_COL = "#222222", "#888888", "#1baf7a"
MP, EB = 0.938272, 10.6
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "legend.frameon": False})
cuts = r["cuts"]
cv = np.array([float(c) for c in cuts if c != "None"])


def arrays(cl):
    B = np.array([c["B"] for c in cl]); E = np.array([c["B_err"] for c in cl])
    A = np.array([c["A"] for c in cl])
    return B, E, A


# one global cut: the chi2 minimum over the 2D cells, by a parabola through the three lowest points
B2, E2, A2 = arrays(r["2d"])
chi2_2 = np.array([np.sum(((B2 - A2[:, i]) / E2) ** 2) for i in range(len(cv))])
i0 = int(np.argmin(chi2_2)); sl = slice(max(i0 - 1, 0), min(i0 + 2, len(cv)))
pp = np.polyfit(cv[sl], chi2_2[sl], 2); c_best = float(-pp[1] / (2 * pp[0]))


def A_at(A, c):
    """A interpolated in the cut, per cell."""
    return np.array([np.interp(c, cv, a[:len(cv)]) for a in A])


def label(c):
    return f"{c['Q2'][0]}-{c['Q2'][1]} | {c['xB'][0]}-{c['xB'][1]}" + (
        f" | {c['mt'][0]}-{c['mt'][1]}" if c["it"] >= 0 else "")


def save(fig, name, pdf):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


with PdfPages(os.path.join(FIG, "rc_method_pi0_grid.pdf")) as pdf:
    # ---- G1: the (Q2, xB) plane, -t integrated ------------------------------
    cl = r["2d"]
    Ab = A_at(A2, c_best)
    dev = 100 * (B2 / Ab - 1); deve = 100 * E2 / Ab
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2))
    norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
    cmap = plt.get_cmap("RdBu_r")
    for ax, what in zip(axs, ("dev", "ceq")):
        for c, d, de in zip(cl, dev, deve):
            x0, x1 = c["xB"]; q0, q1 = c["Q2"]
            if what == "dev":
                fc = cmap(norm(d)); txt = f"{d:+.1f}\n±{de:.1f}"
            else:
                v = c["c_eq"]
                fc = plt.get_cmap("Greens")(0.15 + 0.7 * min(v, 0.5) / 0.5) if np.isfinite(v) else "#dddddd"
                txt = (f"{v:.2f}\n({c['c_lo']:.2f}-{c['c_hi']:.2f})" if np.isfinite(c["c_lo"]) and np.isfinite(c["c_hi"])
                       else (f"{v:.2f}" if np.isfinite(v) else ">0.5"))
            ax.add_patch(Rectangle((x0, q0), x1 - x0, q1 - q0, fc=fc, ec="white", lw=1.5))
            dark = (what == "dev" and abs(d) > 9) or (what == "ceq" and np.isfinite(c["c_eq"]) and c["c_eq"] > 0.3)
            ax.text((x0 + x1) / 2, (q0 + q1) / 2, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if dark else INK)
        ax.set_xlim(0.1, 0.65); ax.set_ylim(1, 6); ax.grid(False)
        ax.set_xlabel("x$_B$"); ax.set_ylabel("Q$^2$ (GeV$^2$)")
    axs[0].set_title(f"B/A $-$ 1 (%), A at one global mx2_cut = {c_best:.3f} GeV$^2$", fontsize=9)
    axs[1].set_title("mx2_cut at which A equals B (GeV$^2$), with its 1$\\sigma$ range", fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); fig.colorbar(sm, ax=axs[0], label="B/A $-$ 1 (%)")
    fig.suptitle(f"The analysis note's (Q$^2$, x$_B$) cells, $-t$ integrated (0.09-2.0 GeV$^2$), "
                 f"$\\phi$ integrated; cells with $\\geq${r['nmin']} events per arm after the full chain", fontsize=9)
    fig.tight_layout()
    save(fig, "g1_map", pdf)

    # ---- G2: B and A per cell, ratio in percent below ------------------------
    for key, title in (("2d", "(Q$^2$, x$_B$) cells, $-t$ integrated"), ("3d", "(Q$^2$, x$_B$, $-t$) cells")):
        cl = r[key]; B, E, A = arrays(cl); Ab = A_at(A, c_best)
        x = np.arange(len(cl))
        fig, (ax, ar) = plt.subplots(2, 1, figsize=(max(9, 0.32 * len(cl) + 3), 6.4), sharex=True,
                                     gridspec_kw=dict(height_ratios=(2, 1.2)))
        ax.errorbar(x, B, yerr=E, fmt="o", ms=4, color=INK, label="B: rad/Born MC, full chain")
        ax.plot(x, Ab, "_", ms=14, mew=2.2, color=A_COL, label=f"A: rc_factor, mx2_cut = {c_best:.3f}")
        ax.plot(x, A_at(A, 0.08), "_", ms=10, mew=1, color="#9ee3c8", label="A, mx2_cut = 0.08")
        ax.plot(x, A_at(A, 0.25), "_", ms=10, mew=1, color="#117a55", label="A, mx2_cut = 0.25")
        ax.set_ylabel(r"$\delta_{RC}$ ($\phi$ integrated)"); ax.legend(fontsize=7.5, ncol=2)
        ax.set_title(f"{title}.  Cell label: Q$^2$ | x$_B$" + (" | $-t$" if key == "3d" else ""), fontsize=9)
        dv = 100 * (B / Ab - 1)
        ar.axhline(0, color=MUTED, lw=0.8)
        ar.errorbar(x, dv, yerr=100 * E / Ab, fmt="o", ms=4, color=INK)
        chi = np.sum((dv / (100 * E / Ab)) ** 2)
        ar.text(0.01, 0.88, f"$\\chi^2$ = {chi:.0f} / {len(cl)} cells", transform=ar.transAxes, fontsize=8)
        ar.set_ylabel("B/A $-$ 1 (%)"); ar.set_ylim(-25, 25)
        ar.set_xticks(x); ar.set_xticklabels([label(c) for c in cl], rotation=90, fontsize=6.5)
        fig.tight_layout()
        save(fig, f"g2_cells_{key}", pdf)

    # ---- G3: the deviation against y, W and -t -------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for key, mk, lab in (("2d", "o", "$-t$ integrated"), ("3d", "s", "per $-t$ cell")):
        cl = r[key]; B, E, A = arrays(cl); Ab = A_at(A, c_best)
        cen = np.array([c["centroid"] for c in cl])
        y = cen[:, 0] / (2 * MP * EB * cen[:, 1]); W = np.sqrt(MP ** 2 + cen[:, 0] * (1 / cen[:, 1] - 1))
        dv, de = 100 * (B / Ab - 1), 100 * E / Ab
        for ax, xv in zip(axs, (y, W, cen[:, 2])):
            ax.errorbar(xv, dv, yerr=de, fmt=mk, ms=4, color=INK, mfc="white" if key == "3d" else INK,
                        lw=0.7, label=lab)
    for ax, xl in zip(axs, ("y (cell centroid)", "W (GeV, cell centroid)", "$-t$ (GeV$^2$, cell centroid)")):
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_xlabel(xl)
    axs[0].set_ylabel(f"B/A $-$ 1 (%), mx2_cut = {c_best:.3f}"); axs[0].legend(fontsize=8)
    axs[0].set_ylim(-30, 30)
    fig.suptitle("Where one cut fails: the deviation against the kinematic variables", fontsize=9)
    fig.tight_layout()
    save(fig, "g3_vs_kin", pdf)

    # ---- G4: chi2 of one global cut -----------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, mk in (("2d", "o-"), ("3d", "s--")):
        B, E, A = arrays(r[key])
        ch = [np.sum(((B - A[:, i]) / E) ** 2) for i in range(len(cv))]
        ax.plot(cv, ch, mk, ms=4, color=INK if key == "2d" else MUTED,
                label=f"{key.upper()}: {len(B)} cells")
        ax.axhline(len(B), color=INK if key == "2d" else MUTED, lw=0.6, ls=":")
    ax.axvline(c_best, color=A_COL, lw=1)
    ax.text(c_best + 0.01, 400, f"best global\nmx2_cut = {c_best:.3f}", fontsize=8)
    ax.set_xlabel("one mx2_cut for every cell (GeV$^2$)"); ax.set_ylabel("$\\chi^2$ of A against B")
    ax.set_ylim(0, 650); ax.legend(fontsize=8)
    ax.set_title("Dotted: number of cells", fontsize=8)
    fig.tight_layout()
    save(fig, "g4_chi2", pdf)

summ = {"c_best": c_best, "chi2_2d": dict(zip(map(str, cv), map(float, chi2_2))), "n2d": len(r["2d"]),
        "n3d": len(r["3d"])}
json.dump(summ, open(os.path.join(FIG, "grid_summary.json"), "w"), indent=1)
print(summ)
