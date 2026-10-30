#!/usr/bin/env python3
"""Interaction point in the rad and Born arms: generated, true (GEMC), reconstructed.

    python3 production/analysis/rc_method_pi0_vertex.py <figdir>

Input: /Volumes/wd_14tb/rc_method_pi0/vz/vz_<arm>_<pol>_NNN.txt, written on ifarm by
vz.groovy from 8 DSTs per arm and polarity (40k events each):
    MC::Lund vz   MC::Particle vz   REC electron vz (trigger, pid 11)   REC proton vz
-999 = not reconstructed.  The cards draw vz uniform in -5.5 .. -0.5 cm (5 cm LH2
cell centred at -3), itarg = 1, in BOTH arms; GEMC keeps the LUND vertex.
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = sys.argv[1] if len(sys.argv) > 1 else "docs/rc_method_pi0"
D = "/Volumes/wd_14tb/rc_method_pi0/vz"
RAD, BORN, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#888888"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False})


def load(a, pol):
    return np.concatenate([np.loadtxt(f) for f in sorted(glob.glob(f"{D}/vz_{a}_{pol}_*.txt"))])


V = {(a, p): load(a, p) for a in ("rad", "born") for p in ("inb", "outb")}
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

ax = axs[0, 0]
for a, col in (("born", BORN), ("rad", RAD)):
    x = np.concatenate([V[(a, p)][:, 1] for p in ("inb", "outb")])
    ax.hist(x, bins=np.linspace(-7, 1, 81), histtype="step", lw=1.4, color=col, density=True,
            label=f"{a}: {x.min():.2f} .. {x.max():.2f} cm")
for z in (-5.5, -0.5):
    ax.axvline(z, color=INK, lw=0.9)
ax.set_xlabel("true vz, MC::Particle = MC::Lund (cm)"); ax.set_ylabel("normalised")
ax.set_title("Generated vertex: uniform over the 5 cm LH$_2$ cell", fontsize=9); ax.legend(fontsize=8)

for j, (pol, pid_win, tab) in enumerate((("inb", (-13, 12), 2.2), ("outb", (-18, 10), 2.5))):
    ax = axs[0, 1 + j]
    for a, col in (("born", BORN), ("rad", RAD)):
        d = V[(a, pol)]; e = d[:, 2] > -900
        ax.hist(d[e, 2], bins=np.linspace(-14, 8, 111), histtype="step", lw=1.4, color=col, density=True,
                label=f"{a}, N = {e.sum()}")
    for z in (-5.5, -0.5):
        ax.axvline(z, color=INK, lw=0.9)
    for z in (-tab, tab):
        ax.axvline(z, color="#e34948", lw=1, ls="--")
    for z in pid_win:
        ax.axvline(z, color=MUTED, lw=1, ls=":")
    ax.set_xlim(-14, 8)
    ax.set_xlabel(f"reconstructed electron vz, {pol}ending (cm)")
    ax.set_title(f"solid: target; dashed: note Tables |vz|<{tab}; dotted: note Sec. 2 {pid_win}", fontsize=8)
    ax.legend(fontsize=8)

ax = axs[1, 0]
for pol, ls in (("inb", "-"), ("outb", "--")):
    for a, col in (("born", BORN), ("rad", RAD)):
        d = V[(a, pol)]; e = d[:, 2] > -900
        ax.hist(d[e, 2] - d[e, 1], bins=np.linspace(-6, 6, 121), histtype="step", lw=1.2, ls=ls, color=col,
                density=True, label=f"{a} {pol}: med {np.median(d[e, 2] - d[e, 1]):+.2f}, "
                                    f"$\\sigma_{{68}}$ {0.5 * np.subtract(*np.percentile(d[e, 2] - d[e, 1], [84, 16])):.2f}")
ax.set_xlabel("electron vz: reconstructed $-$ true (cm)"); ax.legend(fontsize=7.5)
ax.set_title("Vertex resolution, electron", fontsize=9)

ax = axs[1, 1]
for pol, ls, s in (("inb", "-", 1.16), ("outb", "--", 1.244)):
    for a, col in (("born", BORN), ("rad", RAD)):
        d = V[(a, pol)]; e = (d[:, 2] > -900) & (d[:, 3] > -900)
        ax.hist(d[e, 2] - d[e, 3], bins=np.linspace(-10, 10, 101), histtype="step", lw=1.2, ls=ls, color=col,
                density=True, label=f"{a} {pol}: N = {e.sum()}")
    for z in (-3 * s, 3 * s):
        ax.axvline(z, color=MUTED, lw=0.8, ls=ls)
ax.set_xlabel("vz(e') $-$ vz(p') reconstructed (cm)"); ax.legend(fontsize=7.5)
ax.set_title("Electron-proton vertex difference; lines: note 3$\\sigma$ (FD values)", fontsize=9)

# what the two readings of the note's electron-vertex cut keep
ax = axs[1, 2]; ax.axis("off")
rows = []
for pol, tab in (("inb", 2.2), ("outb", 2.5)):
    for a in ("born", "rad"):
        d = V[(a, pol)]; e = d[:, 2] > -900; z = d[e, 2]
        rows.append((f"{a} {pol}", 100 * np.mean(np.abs(z) < tab), 100 * np.mean(np.abs(z + 3) < tab),
                     100 * np.mean((z > -5.5) & (z < -0.5))))
txt = ("Fraction of reconstructed electrons kept\n\n"
       f"{'':10}{'|vz|<cut':>10}{'|vz+3|<cut':>12}{'-5.5..-0.5':>12}\n" +
       "\n".join(f"{r[0]:10}{r[1]:9.1f}%{r[2]:11.1f}%{r[3]:11.1f}%" for r in rows) +
       "\n\ncut = 2.2 cm inbending, 2.5 cm outbending (note Tables 7-10).\n"
       "Read literally, |vz| < cut keeps only the downstream\nend of a cell that sits at -5.5 .. -0.5 cm.")
ax.text(0.0, 0.95, txt, va="top", family="monospace", fontsize=8.5, transform=ax.transAxes)

fig.suptitle("Interaction point, rad and Born MC (8 DSTs per arm and polarity, 40k events each). "
             "Cards: vz uniform -5.5 .. -0.5 cm, itarg = 1, in both arms; GEMC keeps the LUND vertex.", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "v5_vertex.png"), dpi=150, bbox_inches="tight")
for r in rows:
    print(f"{r[0]:10} |vz|<cut {r[1]:5.1f}%   |vz+3|<cut {r[2]:5.1f}%   -5.5..-0.5 {r[3]:5.1f}%")
