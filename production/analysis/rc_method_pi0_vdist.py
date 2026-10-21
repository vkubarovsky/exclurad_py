#!/usr/bin/env python3
"""v = MM2(e'p) - m_pi0^2 after the OTHER exclusivity cuts: what the cut chain does to radiation.

    python3 production/analysis/rc_method_pi0_vdist.py <figdir>

The note's Sec. 3.5 list has no cut on MM2(e'p), so "the other cuts" is the
whole chain of rc_method_pi0_combos.note_cuts.  Two views:
  reconstructed v, rad and Born arms, as the analysis sees it;
  GENERATED v of the rad events (the radiated photon's real size), before the
  exclusivity cuts and after them.  Their ratio is the chain's efficiency
  against v -- the soft edge that A replaces by a step at v_cut.
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
from rc_method_pi0_combos import note_cuts  # noqa: E402

FIG = sys.argv[1] if len(sys.argv) > 1 else "docs/rc_method_pi0"
RAD, BORN, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#888888"
V_EQ, V0 = 0.06, 0.20
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False})

rng = np.random.default_rng(20261021)
sig = {a: C.sigma_nb(f"{R.DATA}/prov_{a}", a)["mean"] for a in ("rad", "born")}
vr = {"rad": [], "born": []}; wr = {"rad": [], "born": []}
vg_pre, vg_post, w_pre, w_post = [], [], [], []
for pol in ("inb", "outb"):
    S = {a: R.load(a, pol, rng) for a in ("rad", "born")}
    cal = R.calibrate(S["born"]["rec"])
    for a in S:
        k = S[a]["rec"]; w = sig[a] / S[a]["n_gen"]
        full = note_cuts(k, cal)
        vr[a].append(k["mm2ep"][full] - R.MPI ** 2); wr[a].append(np.full(full.sum(), w))
        if a == "rad":
            vgen = S[a]["gen"]["mm2ep"][S[a]["ok"]] - R.MPI ** 2      # generated v of the reconstructed events
            dis = (k["Q2"] > 1) & (k["W"] > 2) & (k["pe"] > 2)
            vg_pre.append(vgen[dis]); w_pre.append(np.full(dis.sum(), w))
            vg_post.append(vgen[full]); w_post.append(np.full(full.sum(), w))
cat = np.concatenate
vr = {a: cat(v) for a, v in vr.items()}; wr = {a: cat(v) for a, v in wr.items()}
vg_pre, vg_post, w_pre, w_post = cat(vg_pre), cat(vg_post), cat(w_pre), cat(w_post)

with PdfPages(os.path.join(FIG, "rc_method_pi0_vdist.pdf")) as pdf:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    # reconstructed v, both arms
    ax = axs[0]; bins = np.linspace(-0.6, 1.2, 91)
    for a, col in (("born", BORN), ("rad", RAD)):
        ax.hist(vr[a], bins=bins, weights=wr[a] * 1e6, histtype="step", lw=1.4, color=col,
                label=f"{a} MC, N = {len(vr[a])}")
    for v, ls, t in ((V_EQ, "-", f"v_cut = {V_EQ}"), (V0, "--", f"v_cut = {V0}")):
        ax.axvline(v, color=INK, lw=0.9, ls=ls, label=t)
    ax.set_yscale("log"); ax.set_xlabel("reconstructed v = MM$^2$(e'p) $-$ m$_{\\pi^0}^2$ (GeV$^2$)")
    ax.set_ylabel("yield (fb per bin)"); ax.legend(fontsize=7.5)
    ax.set_title("Reconstructed, after all other cuts", fontsize=9)
    # generated v of the rad events, before / after the exclusivity cuts
    ax = axs[1]; bins = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 60)])
    h_pre = np.histogram(vg_pre, bins=bins, weights=w_pre)[0]
    h_post = np.histogram(vg_post, bins=bins, weights=w_post)[0]
    cen = np.sqrt(bins[1:] * np.maximum(bins[:-1], 5e-4))
    ax.step(cen, h_pre * 1e6 / np.diff(bins), where="mid", color=MUTED, lw=1.2, label="rad, topology + DIS")
    ax.step(cen, h_post * 1e6 / np.diff(bins), where="mid", color=RAD, lw=1.6, label="rad, + all exclusivity cuts")
    for v, ls in ((V_EQ, "-"), (V0, "--")):
        ax.axvline(v, color=INK, lw=0.9, ls=ls)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("generated v of the rad events (GeV$^2$)"); ax.set_ylabel("d(yield)/dv (fb/GeV$^2$)")
    ax.legend(fontsize=7.5); ax.set_title("True radiation in the accepted events", fontsize=9)
    # efficiency of the chain against generated v
    ax = axs[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = h_post / h_pre
    n_pre = np.histogram(vg_pre, bins=bins)[0]
    err = np.sqrt(eff * (1 - eff) / np.maximum(n_pre, 1))
    ok = n_pre >= 20
    soft = eff[ok][cen[ok] < 0.005].mean() if np.any(cen[ok] < 0.005) else eff[ok][0]
    ax.errorbar(cen[ok], eff[ok] / soft, yerr=err[ok] / soft, fmt="o", ms=3, color=INK, lw=0.7,
                label="all other exclusivity cuts")
    ax.plot([1e-3, V_EQ, V_EQ, 5], [1, 1, 0, 0], color=RAD, lw=1.6, label=f"step at v_cut = {V_EQ} (A)")
    ax.plot([1e-3, V0, V0, 5], [1, 1, 0, 0], color=RAD, lw=1, ls="--", label=f"step at v_cut = {V0}")
    # the v at which the measured efficiency falls to one half
    fe = eff[ok] / soft; ce = cen[ok]
    i = np.argmax(fe < 0.5) if np.any(fe < 0.5) else None
    if i:
        v_half = float(np.exp(np.interp(0.5, [fe[i], fe[i - 1]], [np.log(ce[i]), np.log(ce[i - 1])])))
        ax.axvline(v_half, color=MUTED, lw=0.8, ls=":")
        ax.text(v_half * 1.1, 0.55, f"50 % at v = {v_half:.3f}", fontsize=8)
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel("generated v (GeV$^2$)"); ax.set_ylabel("efficiency, relative to v $\\to$ 0")
    ax.legend(fontsize=7.5, loc="upper right"); ax.set_title("The cut chain as a cut on v", fontsize=9)
    fig.suptitle("v after the note's cuts (|ME|<0.2, |MM$^2$(e'p$\\gamma\\gamma$)|<0.004, MP$_T$<0.75, cone<4$^\\circ$, "
                 "Q$^2$>1, W>2, p$_{e'}$>2); all kinematics, inbending + outbending", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v1_vdist.png"), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- the reconstructed v alone, linear and log, finer bins --------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    bins = np.linspace(-0.6, 1.0, 161)
    for ax, scale in zip(axs, ("linear", "log")):
        for a, col in (("born", BORN), ("rad", RAD)):
            h, _, _ = ax.hist(vr[a], bins=bins, weights=wr[a] * 1e6, histtype="step", lw=1.4, color=col,
                              label=f"{a} MC, N = {len(vr[a])}")
            if scale == "linear":
                c = 0.5 * (bins[1:] + bins[:-1]); top = h.argmax()
                core = slice(max(top - 4, 0), top + 5)          # parabola through the top +-0.04
                pp = np.polyfit(c[core], np.log(np.maximum(h[core], 1e-12)), 2)
                peak = -pp[1] / (2 * pp[0])
                print(f"{a}: peak of reconstructed v (log-parabola on the top) = {peak:+.4f} GeV^2, "
                      f"median {np.median(vr[a]):+.4f}")
                ax.text(0.98, 0.70 - 0.07 * (a == "rad"), f"{a} peak {peak:+.3f}", transform=ax.transAxes,
                        ha="right", fontsize=8, color=col)
        ax.axvline(0, color=MUTED, lw=0.8)
        for v, ls in ((V_EQ, "-"), (V0, "--")):
            ax.axvline(v, color=INK, lw=0.9, ls=ls, label=f"v_cut = {v}")
        ax.set_yscale(scale); ax.set_xlim(-0.6, 1.0)
        ax.set_xlabel("reconstructed v = MM$^2$(e'p) $-$ m$_{\\pi^0}^2$ (GeV$^2$)"); ax.set_ylabel("yield (fb per 0.01 GeV$^2$)")
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Reconstructed v after all the note's other cuts; all kinematics, inbending + outbending", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_vrec_lin_log.png"), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# area check: how much of the rad yield after the chain lies above each v
tot = w_post.sum()
for v in (0.02, 0.066, 0.1, 0.2, 0.3):
    print(f"generated v > {v:5.3f}: {100 * w_post[vg_post > v].sum() / tot:5.2f} % of accepted rad yield")
print("50 % point", v_half if i else None)
