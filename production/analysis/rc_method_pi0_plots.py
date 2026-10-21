#!/usr/bin/env python3
"""Plots for rc_method_pi0.py.   python3 rc_method_pi0_plots.py <outdir-of-run> <figdir>"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RUN, FIG = sys.argv[1], sys.argv[2]
os.makedirs(FIG, exist_ok=True)
r = json.load(open(os.path.join(RUN, "rc_method_pi0.json")))
H = np.load(os.path.join(RUN, "hists.npy"), allow_pickle=True).item()
C = r["combined"]
NPHI = r["nphi"]
pc = (np.arange(NPHI) + 0.5) * 360 / NPHI
RAD, BORN, A_COL, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#888888"
FULL = "+MM2(e'pgg), ME: full chain"
LEVELS = ["topology+DIS", "+M(gg), MM2(e'p)", "+M(e'gg), cone, MPT", FULL]
b = r["bin"]
BINLBL = (f"{b['Q2'][0]}<Q$^2$<{b['Q2'][1]} GeV$^2$,  {b['xB'][0]}<x$_B$<{b['xB'][1]},  "
          f"{b['mt'][0]}<$-t$<{b['mt'][1]} GeV$^2$;  inbending + outbending")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False})

cuts = [k for k in r["A"]]
cval = np.array([float(k) if k != "None" else np.nan for k in cuts])
A = np.array([r["A"][k] for k in cuts])
# phi weights for an integral of A: the generated Born yield in phi_p, no cut
wphi = np.array(C["B_gen"]["None|phip"]["n_born"]); wphi = wphi / wphi.sum()
A_int = (A * wphi).sum(1) / 1.0
finite = ~np.isnan(cval)


def save(fig, name, pdf):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pooled(arm, var, sel="base"):
    xs, ws = [], []
    for pol in ("inb", "outb"):
        h = H[f"{pol}|{arm}"]
        m = np.ones(len(h[var]), bool) if sel == "base" else h["full"]
        xs.append(h[var][m]); ws.append(np.full(m.sum(), h["w"]))
    return np.concatenate(xs), np.concatenate(ws)


summary = {}
with PdfPages(os.path.join(FIG, "rc_method_pi0.pdf")) as pdf:
    # ---- 1. the data in the bin: exclusivity variables, rad vs Born ----------
    spec = [("mgg", r"M$_{\gamma\gamma}$ (GeV)", (0.06, 0.21)),
            ("mm2ep", r"MM$^2$(e'p) (GeV$^2$)", (-0.6, 1.2)),
            ("megg", r"M(e'$\gamma\gamma$) (GeV)", (0.2, 2.2)),
            ("mm2tot", r"MM$^2$(e'p$\gamma\gamma$) (GeV$^2$)", (-0.05, 0.05)),
            ("me", r"ME(e'p$\gamma\gamma$) (GeV)", (-1.0, 2.5)),
            ("mpt", r"MP$_T$ (GeV)", (0, 0.6)),
            ("cone", r"cone angle $\pi^0$ vs X (deg)", (0, 15)),
            ("open", r"$\gamma\gamma$ opening angle (deg)", (0, 20))]
    cal = r["pol"]["outb"]["cal"]["CD"]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, (v, lbl, rg) in zip(axs.ravel(), spec):
        for arm, col in (("born", BORN), ("rad", RAD)):
            x, w = pooled(arm, v)
            ax.hist(x, bins=60, range=rg, weights=w * 1e6, histtype="step", lw=1.4, color=col,
                    label=f"{arm} MC")
        if v in cal:
            c0, s0 = cal[v]
            for e in (c0 - 3 * s0, c0 + 3 * s0):
                ax.axvline(e, color=INK, lw=0.8, ls="--")
        lim = {"cone": 4, "mpt": 0.75, "open": 1}.get(v)
        if lim is not None:
            ax.axvline(lim, color=INK, lw=0.8, ls="--")
        ax.set_xlabel(lbl); ax.set_xlim(rg)
    axs[0, 0].set_ylabel("yield (fb per bin, before cuts)")
    axs[1, 0].set_ylabel("yield (fb per bin, before cuts)")
    axs[0, 0].legend(loc="upper left")
    fig.suptitle("The MC in the bin after topology + DIS cuts, before exclusivity cuts.  "
                 "Dashed: the cut (3$\\sigma$ windows from the Born MC, outbending CD proton shown)\n" + BINLBL,
                 fontsize=9)
    fig.tight_layout()
    save(fig, "p1_exclusivity", pdf)

    # ---- 2. yields per phi after the full chain ------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, lv in zip(axs, ("topology+DIS", FULL)):
        for arm, col in (("born", BORN), ("rad", RAD)):
            n = np.array(C["B_rec"][f"{lv}|phi"][f"n_{arm}"])
            ax.errorbar(pc, n, yerr=np.sqrt(n), fmt="o", ms=4, color=col, label=f"{arm} MC, N={int(n.sum())}")
        ax.set_xlabel(r"$\phi_{\gamma\gamma}$ (deg, Trento)"); ax.set_title(lv, fontsize=9)
        ax.set_xticks(range(0, 361, 60)); ax.legend()
    axs[0].set_ylabel("reconstructed events per 30$^\\circ$")
    fig.suptitle("Reconstructed MC counts in the bin.  " + BINLBL, fontsize=9)
    fig.tight_layout()
    save(fig, "p2_counts", pdf)

    # ---- 3. closure: A against B at generator level, same MM2(e'p) cut -------
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, c in zip(axs, ("0.05", "0.17", "0.5", "None")):
        ia = cuts.index(c)
        ax.plot(pc, A[ia], color=A_COL, lw=2, label="A: rc_factor")
        for pv, mk, lab in (("phip", "o", r"B gen, $\phi_p+180$"), ("phi", "s", r"B gen, $\phi_{\gamma\gamma}$")):
            d = C["B_gen"][f"{c}|{pv}"]
            ax.errorbar(pc, d["delta"], yerr=d["err"], fmt=mk, ms=4, mfc="white" if pv == "phi" else INK,
                        color=INK, lw=0.8, label=lab)
        ax.set_title(f"MM$^2$(e'p) < {c} GeV$^2$" if c != "None" else "no MM$^2$(e'p) cut", fontsize=9)
        ax.set_xlabel(r"$\phi$ (deg)"); ax.set_xticks(range(0, 361, 90))
    axs[0].set_ylabel(r"$\delta = \sigma_{obs}/\sigma_{Born}$"); axs[0].legend(fontsize=8)
    fig.suptitle("Closure at generator level: same engine, same cut, no detector.  "
                 "A's $\\phi$ is the proton's azimuth + 180$^\\circ$; with that definition B follows A.", fontsize=9)
    fig.tight_layout()
    save(fig, "p3_closure", pdf)

    # ---- 4. the comparison: B with the real chain against A for several cuts --
    Bf = C["B_rec"][f"{FULL}|phi"]; Bp = C["B_rec"][f"{FULL}|phip"]
    Bd, Be = np.array(Bf["delta"]), np.array(Bf["err"])
    chi2 = np.array([np.nansum(((Bd - A[i]) / Be) ** 2) for i in range(len(cuts))])
    B_int = Bf["delta_int"]
    B_int_err = B_int * np.sqrt(1 / sum(Bf["n_rad"]) + 1 / sum(Bf["n_born"]))
    # the cut at which A's integral equals B's
    xs, ys = cval[finite], A_int[finite]
    c_eq = float(np.interp(B_int, ys, xs)) if ys.min() <= B_int <= ys.max() else np.nan
    c_lo = float(np.interp(B_int - B_int_err, ys, xs)); c_hi = float(np.interp(B_int + B_int_err, ys, xs))
    ibest = int(np.nanargmin(chi2))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.errorbar(pc, Bd, yerr=Be, fmt="o", ms=5, color=INK, label=r"B: rad/Born MC, full chain, $\phi_{\gamma\gamma}$")
    ax.errorbar(pc + 4, Bp["delta"], yerr=Bp["err"], fmt="o", ms=4, mfc="white", color=INK, lw=0.6,
                label=r"B: same, $\phi_p+180$")
    shades = {"0.08": "#9ee3c8", "0.17": "#1baf7a", "0.3": "#117a55", "None": "#0b4d36"}
    for c, col in shades.items():
        ia = cuts.index(c)
        ax.plot(pc, A[ia], color=col, lw=1.8,
                label=f"A, mx2_cut={c}" + (f"  ($\\chi^2$={chi2[ia]:.0f}/12)"))
    ax.set_xlabel(r"$\phi$ (deg)"); ax.set_ylabel(r"$\delta_{RC}$ = N$_{rad}$/N$_{Born}$ (B),  $\sigma_{obs}/\sigma_{Born}$ (A)")
    ax.set_xticks(range(0, 361, 60)); ax.legend(fontsize=7.5, loc="upper center", ncol=2)
    ax.set_ylim(0.6, 1.55)
    ax.set_title("The two corrections in one bin.  " + BINLBL, fontsize=8)
    fig.tight_layout()
    save(fig, "p4_A_vs_B", pdf)

    # ---- 5. integral vs the cut, and chi2 ------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ax = axs[0]
    ax.plot(xs, ys, color=A_COL, lw=2, label="A: rc_factor, integrated over the bin")
    ax.axhline(A_int[~finite][0], color=A_COL, lw=1, ls=":", label=f"A, no cut: {A_int[~finite][0]:.3f}")
    gx = [float(k.split("|")[0]) for k in C["B_gen"] if k.endswith("|phip") and not k.startswith("None")]
    gy = [C["B_gen"][f"{x}|phip"]["delta_int"] for x in gx]
    ax.plot(gx, gy, "o", color=INK, ms=4, mfc="white", label="B at generator level, same cut (closure)")
    for lv, ls in zip(LEVELS, (":", "-.", "--", "-")):
        d = C["B_rec"][f"{lv}|phi"]["delta_int"]
        ax.axhline(d, color=INK, lw=1, ls=ls, label=f"B reconstructed, {lv}: {d:.3f}")
    ax.axhspan(B_int - B_int_err, B_int + B_int_err, color="#dddddd", zorder=0)
    if np.isfinite(c_eq):
        ax.axvline(c_eq, color=MUTED, lw=0.8)
        ax.text(c_eq + 0.01, 1.03, f"A = B at\nmx2_cut = {c_eq:.3f}\n({c_lo:.3f}-{c_hi:.3f})", fontsize=8)
    ax.set_xlabel("mx2_cut = upper limit on MM$^2$(e'p) (GeV$^2$)"); ax.set_ylabel(r"$\delta_{RC}$, bin integral")
    ax.legend(fontsize=7, loc="lower right"); ax.set_ylim(0.65, 1.3)
    ax = axs[1]
    ax.plot(xs, chi2[finite], "o-", color=A_COL, ms=4)
    ax.axhline(NPHI, color=MUTED, ls="--", lw=0.8); ax.text(xs[-1], NPHI + 1, "ndf = 12", ha="right", fontsize=8)
    ax.set_xlabel("mx2_cut (GeV$^2$)"); ax.set_ylabel(r"$\chi^2$ of A against B (full chain), 12 $\phi$ bins")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save(fig, "p5_scan", pdf)

summary = dict(B_full_int=B_int, B_full_int_err=B_int_err,
               B_levels={lv: C["B_rec"][f"{lv}|phi"]["delta_int"] for lv in LEVELS},
               A_int={k: float(v) for k, v in zip(cuts, A_int)},
               c_equal_integral=c_eq, c_range=(c_lo, c_hi),
               best_chi2_cut=cuts[ibest], best_chi2=float(chi2[ibest]),
               chi2={k: float(v) for k, v in zip(cuts, chi2)},
               B_full_phi=Bd.tolist(), B_full_err=Be.tolist(),
               A_best_phi=A[ibest].tolist())
json.dump(summary, open(os.path.join(FIG, "summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k not in ("chi2", "A_int")}, indent=1))
