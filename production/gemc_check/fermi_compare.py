#!/usr/bin/env python3
"""What Fermi motion does, measured through the full GEMC + reconstruction chain.

    python3 production/gemc_check/fermi_compare.py none.f32 av18.f32 outdir \
            [--ebeam 10.6] [--nucleon p|n]

Two samples that differ in ONE card field.  Everything else -- model, box,
gcard, torus, software, number of events -- is identical, so every difference
below is Fermi motion and nothing else.

The question is not "does Fermi smear things" (it does, by construction) but
"does it survive the detector".  Detector resolution and acceptance are folded
into both arms equally; where Fermi is small compared with resolution it is
invisible and can be ignored, where it is large it has to be modelled.  Each
panel is built to answer that for one observable.

Definitions follow VPK's terminology (2026-09-26):
    generated           all generated events, before GEMC
    generated-detected  generated AND matched to a reconstructed track
                        (MC::GenMatch), i.e. an efficiency
    detected            reconstructed without reference to the generator,
                        i.e. what an analysis actually sees
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

M_P, M_N, M_PI0, NFLD = 0.93827208816, 0.93956542, 0.1349768, 50

# THE STRUCK NUCLEON'S MASS, and it is not decoration.  It enters the target
# mass assumed at rest (W, xB, t) and the detected hadron's energy in
# MX^2(e'h) -- the exclusivity variable this whole comparison is about.  It was
# hardcoded to the proton, which is right for RG-A and silently wrong for a
# neutron target: 1.3 MeV in the mass is nothing, but it is used in four places
# and nothing would have complained.  Set once, from the command line, and
# printed in the output so a table cannot be read without knowing which it was.
M_T = M_P
NUCLEON = "p"


def load(path):
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(-1, NFLD)


def block(d, k):
    """(found, pid, charge, px, py, pz) for k = 0 e', 1 p, 2 g1, 3 g2."""
    b = 13 + 8 * k
    return d[:, b], d[:, b + 1], d[:, b + 2], d[:, b + 5], d[:, b + 6], d[:, b + 7]


def gen_p(d, k):
    b = 1 + 3 * k
    return d[:, b], d[:, b + 1], d[:, b + 2]


def kin(px, py, pz, eb):
    """Q2, W, xB from the scattered electron, target assumed AT REST -- which is
    the assumption an analysis makes and the one Fermi motion violates."""
    p = np.sqrt(px**2 + py**2 + pz**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        q2 = 2.0 * eb * (p - pz)
        nu = eb - p
        w2 = M_T**2 + 2.0 * M_T * nu - q2
        xb = np.where(nu > 0, q2 / (2.0 * M_T * nu), np.nan)
    return q2, np.sqrt(np.maximum(w2, 0.0)), xb, nu


def t_of(px, py, pz):
    """t = (p_target - p')^2 with the target at rest."""
    e = np.sqrt(M_T**2 + px**2 + py**2 + pz**2)
    return 2.0 * M_T**2 - 2.0 * M_T * e


def mx2_ep(epx, epy, epz, ppx, ppy, ppz, eb):
    """MX^2(e'p): the exclusivity variable, target assumed at rest."""
    ee = np.sqrt(epx**2 + epy**2 + epz**2)
    ep_ = np.sqrt(M_T**2 + ppx**2 + ppy**2 + ppz**2)
    e_m = eb + M_T - ee - ep_
    x_m = -epx - ppx
    y_m = -epy - ppy
    z_m = eb - epz - ppz
    return e_m**2 - x_m**2 - y_m**2 - z_m**2


def analyse(d, eb):
    """Everything both arms need, computed identically."""
    o = {}
    fe, pide, che, epx, epy, epz = block(d, 0)
    fp, pidp, chp, ppx, ppy, ppz = block(d, 1)
    o["n"] = len(d)
    # generated-detected: matched, right charge, right pid
    o["ep_det"] = ((fe > 0) & (che == -1) & (pide == 11)
                   & (fp > 0) & (chp == 1) & (pidp == 2212))
    # detected (analysis-level): the naive counts, no generator truth used
    o["ep_naive"] = (d[:, 45] >= 1) & (d[:, 46] >= 1)
    o["epgg_naive"] = o["ep_naive"] & (d[:, 47] >= 2)

    gex, gey, gez = gen_p(d, 0)
    gpx, gpy, gpz = gen_p(d, 1)
    o["q2_g"], o["w_g"], o["xb_g"], _ = kin(gex, gey, gez, eb)
    o["t_g"] = t_of(gpx, gpy, gpz)
    o["mx2_g"] = mx2_ep(gex, gey, gez, gpx, gpy, gpz, eb)

    o["q2_r"], o["w_r"], o["xb_r"], _ = kin(epx, epy, epz, eb)
    o["t_r"] = t_of(ppx, ppy, ppz)
    o["mx2_r"] = mx2_ep(epx, epy, epz, ppx, ppy, ppz, eb)
    o["v_r"] = o["mx2_r"] - M_PI0**2
    o["v_g"] = o["mx2_g"] - M_PI0**2
    return o


def band(ax, x, ya, yb, la, lb):
    ax.step(x, ya, where="mid", label=la, color="C0")
    ax.step(x, yb, where="mid", label=lb, color="C3")


def main():
    global M_T, NUCLEON
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    eb = 10.6
    for a in sys.argv[1:]:
        if a.startswith("--ebeam"):
            eb = float(a.split("=")[1]) if "=" in a else eb
        if a.startswith("--nucleon"):
            NUCLEON = a.split("=")[1] if "=" in a else NUCLEON
    if NUCLEON not in ("p", "n"):
        sys.exit(f"--nucleon must be p or n, not {NUCLEON!r}")
    M_T = M_P if NUCLEON == "p" else M_N
    f_none, f_av18, outdir = args[0], args[1], Path(args[2])
    outdir.mkdir(parents=True, exist_ok=True)

    N = analyse(load(f_none), eb)
    A = analyse(load(f_av18), eb)
    print(f"# Fermi motion through GEMC + reconstruction\n")
    print(f"struck nucleon: {NUCLEON}   (M_T = {M_T:.6f} GeV, "
          f"used for W, xB, t and MX^2)")
    print(f"free nucleon : {N['n']:,} events")
    print(f"deuteron     : {A['n']:,} events\n")

    # ---------------------------------------------------------------- yields
    print("| selection | free nucleon | deuteron | ratio |")
    print("|---|---:|---:|---:|")
    for lab, key in (("generated", None),
                     ("generated-detected  e'p", "ep_det"),
                     ("detected (naive)    e'p", "ep_naive"),
                     ("detected (naive)    e'p gamma gamma", "epgg_naive")):
        if key is None:
            print(f"| {lab} | {N['n']:,} | {A['n']:,} | {A['n']/N['n']:.4f} |")
            continue
        a, b = N[key].sum(), A[key].sum()
        print(f"| {lab} | {a:,} ({a/N['n']:.2%}) | {b:,} ({b/A['n']:.2%}) | {b/max(a,1):.4f} |")

    # -------------------------------------------- is Fermi bigger than resolution?
    print("\n## Is Fermi larger than the detector resolution?\n")
    print("| observable | resolution (free, rec-gen) | Fermi shift (gen) | Fermi total (rec) | verdict |")
    print("|---|---:|---:|---:|---|")
    for lab, kg, kr, unit in (("W  [GeV]", "w_g", "w_r", 1.0),
                              ("Q2 [GeV^2]", "q2_g", "q2_r", 1.0),
                              ("-t [GeV^2]", "t_g", "t_r", 1.0),
                              ("v = MX^2-m^2 [GeV^2]", "v_g", "v_r", 1.0)):
        sel_n, sel_a = N["ep_det"], A["ep_det"]
        res = np.nanstd((N[kr] - N[kg])[sel_n])          # detector only
        fer = np.nanstd(A[kg][sel_a]) - np.nanstd(N[kg][sel_n])  # generator-level extra width
        tot = np.nanstd((A[kr] - A[kg])[sel_a])
        verdict = "Fermi dominates" if abs(fer) > res else "resolution dominates"
        print(f"| {lab} | {res:10.4f} | {fer:+10.4f} | {tot:10.4f} | {verdict} |")

    # ---------------------------------------------------------------- figures
    sel_n, sel_a = N["ep_det"], A["ep_det"]

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))

    # 1. the exclusivity variable, reconstructed -- the plot that matters most
    a0 = ax[0, 0]
    bins = np.linspace(-1.0, 2.0, 160)
    a0.hist(N["v_r"][sel_n], bins=bins, histtype="step", density=True,
            color="C0", label="free nucleon")
    a0.hist(A["v_r"][sel_a], bins=bins, histtype="step", density=True,
            color="C3", label="deuteron (AV18)")
    a0.axvline(0.17, color="k", ls="--", lw=1, label="a hydrogen-tuned cut")
    a0.set_xlabel(r"$v = M_X^2(e'p) - m_{\pi^0}^2$  [GeV$^2$]  reconstructed")
    a0.set_ylabel("density")
    a0.set_title("Exclusivity: where a cut is applied")
    a0.legend(fontsize=8)

    # 2. reconstructed W, both arms
    a1 = ax[0, 1]
    bw = np.linspace(1.6, 4.2, 140)
    a1.hist(N["w_r"][sel_n], bins=bw, histtype="step", density=True, color="C0")
    a1.hist(A["w_r"][sel_a], bins=bw, histtype="step", density=True, color="C3")
    a1.axvline(np.sqrt(3.24), color="k", ls="--", lw=1)
    a1.set_xlabel(r"$W$ reconstructed [GeV]")
    a1.set_ylabel("density")
    a1.set_title("W: the box edge stops being an edge")

    # 3-5. ratio deuteron/free vs each variable -- bin migration made visible
    for a, (key, lab, rng) in zip(
            (ax[0, 2], ax[1, 0], ax[1, 1]),
            (("w_r", r"$W$ [GeV]", (1.8, 4.0)),
             ("q2_r", r"$Q^2$ [GeV$^2$]", (0.8, 8.0)),
             ("t_r", r"$t$ [GeV$^2$]", (-2.0, 0.0)))):
        e = np.linspace(*rng, 41)
        hn, _ = np.histogram(N[key][sel_n], bins=e)
        ha, _ = np.histogram(A[key][sel_a], bins=e)
        c = 0.5 * (e[1:] + e[:-1])
        ok = (hn > 20) & (ha > 20)
        r = np.where(ok, ha / np.where(hn > 0, hn, 1), np.nan)
        er = np.where(ok, r * np.sqrt(1 / np.maximum(hn, 1) + 1 / np.maximum(ha, 1)), np.nan)
        a.errorbar(c[ok], r[ok], yerr=er[ok], fmt="o", ms=3, color="C2")
        a.axhline(np.nanmedian(r), color="k", lw=0.8, ls=":")
        a.set_xlabel(lab)
        a.set_ylabel("deuteron / free")
        a.set_title(f"Bin migration in {lab.split('$')[1] if '$' in lab else lab}")
        a.grid(alpha=0.3)

    # 6. survival under a hydrogen-tuned cut, vs Fermi momentum proxy
    a5 = ax[1, 2]
    cuts = np.linspace(0.02, 1.0, 60)
    sn = [np.mean(np.abs(N["v_r"][sel_n]) < c) for c in cuts]
    sa = [np.mean(np.abs(A["v_r"][sel_a]) < c) for c in cuts]
    a5.plot(cuts, sn, color="C0", label="free nucleon")
    a5.plot(cuts, sa, color="C3", label="deuteron (AV18)")
    a5.axvline(0.17, color="k", ls="--", lw=1)
    a5.set_xlabel(r"cut on $|v|$  [GeV$^2$]")
    a5.set_ylabel("fraction surviving")
    a5.set_title("What an exclusivity cut keeps")
    a5.legend(fontsize=8)
    a5.grid(alpha=0.3)

    for row in ax:
        for a in row:
            a.grid(alpha=0.25)
    fig.suptitle("Fermi motion through GEMC + reconstruction: deuteron vs free nucleon, "
                 "identical in every other respect", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "fermi_influence.png", dpi=150)
    print(f"\nwrote {outdir/'fermi_influence.png'}")

    # survival at the canonical cut
    print(f"\nat |v| < 0.17: free {np.mean(np.abs(N['v_r'][sel_n]) < 0.17):.2%}, "
          f"deuteron {np.mean(np.abs(A['v_r'][sel_a]) < 0.17):.2%}")


if __name__ == "__main__":
    main()
