#!/usr/bin/env python3
"""Electron acceptance against GENERATED theta_e, inbending vs outbending.

    python3 production/gemc_check/plot_theta_e_acceptance.py <theta_fine.npz> <out.pdf>

The argument is written by the companion scan over the campaign's .f32 ntuples
(columns [1..3] generated electron, [13] found, [14] rec pid).  This figure is
the justification for the card's theta_e = 5.0: the window sits BELOW the box's
own kinematic floor, so it removes nothing, and no larger value is admissible
because one card feeds both polarities and outbending has real efficiency right
down to that floor.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else "theta_e_acceptance.pdf"
e = z["edges"]
c = 0.5 * (e[1:] + e[:-1])
THETA_FLOOR = 5.273          # 2 asin sqrt(Q2min / 4 E E'max), Q2min=0.80, W2min=3.24

def eff(mode, pol, kind):
    g = z["%s_%s_gen" % (mode, pol)].astype(float)
    n = z["%s_%s_%s" % (mode, pol, kind)].astype(float)
    r = np.where(g > 0, n / np.maximum(g, 1), np.nan)
    err = np.where(g > 0, np.sqrt(np.maximum(r * (1 - r), 0) / np.maximum(g, 1)), np.nan)
    return 100 * r, 100 * err

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
STY = dict(inb=("#1b6ca8", "o", "inbending, torus $-1$"),
           outb=("#c1440e", "s", "outbending, torus $+1$"))
for k, (mode, title) in enumerate((("born", "Born sample"), ("rad", "Radiative sample"))):
    a = ax[k]
    for pol in ("inb", "outb"):
        col, mk, lab = STY[pol]
        y, dy = eff(mode, pol, "trk")
        a.plot(c, y, "--", color=col, lw=1.2, alpha=0.55)
        y, dy = eff(mode, pol, "id")
        a.errorbar(c, y, yerr=dy, color=col, marker=mk, ms=4, lw=1.6,
                   capsize=0, label=lab)
    a.axvline(THETA_FLOOR, color="0.35", lw=1.0, ls=":")
    a.text(THETA_FLOOR + 0.15, 96, r"kinematic floor $5.27^\circ$",
           fontsize=8.5, color="0.3", va="top")
    a.axvspan(e[0], THETA_FLOOR, color="0.9", zorder=0)
    a.set_xlabel(r"generated $\theta_{e'}$  (deg)")
    a.set_title(title, fontsize=11)
    a.set_xlim(e[0], 14)
    a.set_ylim(0, 100)
    a.grid(alpha=0.25, lw=0.5)
ax[0].set_ylabel("efficiency  (%)")
ax[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.88), fontsize=9, frameon=False)
for a in ax:
    a.text(0.97, 0.04, "solid: found AND identified as $e^-$\n"
                       "dashed: track found, any PID",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8.5,
           color="0.25")
fig.suptitle(r"CLAS12 RG-A fall 2018, $\pi^0$ off the proton, 50 nA background"
             "\n"
             r"electron acceptance vs generated $\theta_{e'}$ — one card, two field directions",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
