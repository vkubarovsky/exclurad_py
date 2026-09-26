#!/usr/bin/env python3
"""One standard page of plots and one standard record for a production run.

    python3 production/run_report.py born.npz rad.npz <tag> <outdir>

The arrays come from production/lund_kinematics.py.  The point of fixing the
figure is comparability: every run gets the same six panels on the same axes,
so two runs can be laid side by side and the difference read off, instead of
each run being plotted however it happened to be plotted that week.

Panel 6 is the shape test that found the 2026-09-25 sampler bug: rad/born is
smooth in the physics, so a step at a majorant cell edge is the sampler.  It
belongs on every production page, not in a one-off script.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_P = 0.93827208816
W2BOX, MAJ_NW = (3.24, 16.0), 8       # majorant W2 cells, for panel 6

BORN_C, RAD_C, RATIO_C = "#1f4e9c", "#d1600f", "#4a3aa7"


def load(p):
    d = np.load(p)
    return dict(W=np.sqrt(np.maximum(d["W2"], 0)), Q2=d["Q2"],
                mt=d["mt"], v=d["v"] if "v" in d.files else None)


def panel(ax, b, r, key, bins, rng, xlabel, logy=True, logx=False):
    hb, ed = np.histogram(b[key], bins=bins, range=rng)
    hr, _ = np.histogram(r[key], bins=bins, range=rng)
    x = 0.5 * (ed[1:] + ed[:-1])
    ax.step(ed[:-1], hb / hb.sum(), where="post", color=BORN_C, lw=1.4, label="born")
    ax.step(ed[:-1], hr / hr.sum(), where="post", color=RAD_C, lw=1.4, label="rad")
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction / bin")
    ax.grid(alpha=0.22, lw=0.5)


def main():
    bp, rp, tag, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    os.makedirs(out, exist_ok=True)
    b, r = load(bp), load(rp)
    nb, nr = len(b["W"]), len(r["W"])

    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.4))
    fig.suptitle(f"exclurad_py production {tag}    —    {nb/1e6:.2f} M born, "
                 f"{nr/1e6:.2f} M rad", fontsize=13.5, y=0.985)

    panel(ax[0, 0], b, r, "W", 120, (1.75, 4.3), "$W$  [GeV]")
    panel(ax[0, 1], b, r, "Q2", 120, (0.7, 10.5), "$Q^2$  [GeV$^2$]")
    panel(ax[0, 2], b, r, "mt", 120, (0.0, 2.05), "$-t$  [GeV$^2$]")
    ax[0, 0].legend(loc="upper right", fontsize=10)

    # v = MX^2 - m_pi0^2: what an offline missing-mass cut acts on
    a = ax[1, 0]
    if r["v"] is not None:
        vv = r["v"][np.isfinite(r["v"])]
        pos = vv[vv > 1e-5]
        a.hist(np.log10(pos), bins=110, range=(-5, 1.1),
               histtype="step", color=RAD_C, lw=1.4,
               weights=np.full(len(pos), 1.0 / len(vv)))
        a.set_yscale("log")
        a.set_xlabel(r"$\log_{10}\,v$,   $v=M_X^2-m_{\pi^0}^2$  [GeV$^2$]")
        a.set_ylabel("fraction / bin")
        a.grid(alpha=0.22, lw=0.5)
        keep = [(c, 100.0 * (vv < c).mean()) for c in (0.05, 0.1, 0.17, 0.3, 0.5)]
        a.text(0.03, 0.95, "kept by $v<v_{\\rm cut}$:\n" +
               "\n".join(f"  {c:.2f}: {k:.0f}\\%" for c, k in keep),
               transform=a.transAxes, va="top", fontsize=9, color="0.3")
        a.set_title("radiative tail, generator level (generated with vcut = 0)",
                    fontsize=10.5, loc="left")

    # rad/born in Q2 -- physics, should be smooth
    for a, key, bins, rng, xl in ((ax[1, 1], "Q2", 40, (0.8, 10.0), "$Q^2$  [GeV$^2$]"),):
        hb, ed = np.histogram(b[key], bins=bins, range=rng)
        hr, _ = np.histogram(r[key], bins=bins, range=rng)
        x = 0.5 * (ed[1:] + ed[:-1])
        ok = (hb > 20) & (hr > 20)
        rt = np.where(ok, (hr / hr.sum()) / np.where(hb > 0, hb / hb.sum(), 1), np.nan)
        er = np.where(ok, rt * np.sqrt(1 / np.maximum(hb, 1) + 1 / np.maximum(hr, 1)), np.nan)
        a.errorbar(x[ok], rt[ok], yerr=er[ok], fmt="o", ms=3, lw=0.8, color=RATIO_C)
        a.set_xlabel(xl); a.set_ylabel("rad / born"); a.grid(alpha=0.22, lw=0.5)
        a.set_title("shape ratio vs $Q^2$", fontsize=10.5, loc="left")

    # the sampler test: rad/born vs W with the majorant cell edges marked
    a = ax[1, 2]
    NB, RNG = 220, (1.80, 4.00)
    hb, ed = np.histogram(b["W"], bins=NB, range=RNG)
    hr, _ = np.histogram(r["W"], bins=NB, range=RNG)
    x = 0.5 * (ed[1:] + ed[:-1])
    ok = (hb > 20) & (hr > 20)
    rt = np.where(ok, (hr / hr.sum()) / np.where(hb > 0, hb / hb.sum(), 1), np.nan)
    er = np.where(ok, rt * np.sqrt(1 / np.maximum(hb, 1) + 1 / np.maximum(hr, 1)), np.nan)
    for w in np.sqrt(np.linspace(*W2BOX, MAJ_NW + 1)):
        a.axvline(w, color=RAD_C, lw=0.9, ls="--", alpha=0.7)
    a.errorbar(x[ok], rt[ok], yerr=er[ok], fmt="o", ms=2.6, lw=0.7, color=RATIO_C, zorder=3)
    a.set_xlabel("$W$  [GeV]"); a.set_ylabel("rad / born"); a.grid(alpha=0.22, lw=0.5)
    a.set_title("sampler test: steps at majorant cell edges (dashed)?",
                fontsize=10.5, loc="left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(out, f"{tag}_overview.png")
    fig.savefig(png, dpi=140)
    print("wrote", png)

    # the numbers that belong in the registry, derived not typed
    rec = dict(tag=tag, n_born=int(nb), n_rad=int(nr),
               W=[float(np.min(b["W"])), float(np.max(b["W"]))],
               Q2=[float(np.min(b["Q2"])), float(np.max(b["Q2"]))],
               mt=[float(np.nanmin(b["mt"])), float(np.nanmax(b["mt"]))])
    if r["v"] is not None:
        vv = r["v"][np.isfinite(r["v"])]
        rec["v_median"] = float(np.median(vv))
        rec["kept"] = {str(c): round(100.0 * float((vv < c).mean()), 1)
                       for c in (0.05, 0.1, 0.17, 0.2, 0.3, 0.5)}
    with open(os.path.join(out, f"{tag}_record.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
