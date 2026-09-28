#!/usr/bin/env python3
"""Generated against reconstructed, one page per sample.

    python3 production/gemc_check/gen_vs_rec.py <ntuple.f32> <tag> <outdir> [--log]

The ntuple is the 50-float record from hipo_to_ntuple.groovy: generated
four-vectors for e', p and the two pi0 photons, and for each the REC::Particle
row MC::GenMatch assigns.  "Reconstructed" here means found AND identified with
the right pid -- an unidentified track is not a reconstructed particle for an
analysis, and on this sample that distinction is worth a factor five: without a
trigger electron the start time is taken from the fastest particle, assumed to
be a pion, and time-of-flight PID collapses (the proton is identified in 5.6 %
of events against 68.3 % when the electron is there).

Counts, not normalised fractions: the reconstructed histogram sits INSIDE the
generated one and the gap between them is the efficiency.  Normalising would
hide exactly what the page is for.

The electron condition is specific to ELECTROPRODUCTION.  Photoproduction has
no scattered electron and needs a different anchor -- the proton, or the pair
itself -- so do not carry the 'no electron, drop the event' rule across.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_P, M_PI0, EB = 0.93827208816, 10.604 * 0 + 0.1349768, 10.604
# Transparent fills, no edge line: the two histograms are nested (the
# reconstructed one is a subset), so an outline draws a border around a
# shape that is not a separate object, and a solid fill hides the one
# underneath.  Alpha lets the overlap read as overlap.
GEN_FACE = "#4b7fc4"
REC_FACE = "#e07b39"
REC_EDGE = "#c2560c"


def kin(e, p):
    Ee = np.linalg.norm(e, axis=1)
    Ep = np.sqrt((p ** 2).sum(axis=1) + M_P ** 2)
    nu = EB - Ee
    Q2 = 2 * EB * Ee - 2 * EB * e[:, 2]
    W2 = M_P ** 2 + 2 * M_P * nu - Q2
    mt = 2 * M_P * (Ep - M_P)
    xB = np.where(nu > 0, Q2 / np.maximum(2 * M_P * nu, 1e-9), np.nan)
    Emx = EB + M_P - Ee - Ep
    mm2 = Emx ** 2 - ((-e[:, 0] - p[:, 0]) ** 2 + (-e[:, 1] - p[:, 1]) ** 2
                      + (EB - e[:, 2] - p[:, 2]) ** 2)
    return Q2, np.sqrt(np.maximum(W2, 0)), mt, xB, mm2


def mag(v):
    return np.linalg.norm(v, axis=1)


def theta(v):
    return np.degrees(np.arctan2(np.hypot(v[:, 0], v[:, 1]), v[:, 2]))


def main():
    f, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
    logy = "--log" in sys.argv
    os.makedirs(out, exist_ok=True)
    a = np.fromfile(f, dtype=np.float32).reshape(-1, 50).astype(np.float64)
    n = len(a)

    gen = [a[:, 1 + 3 * k:4 + 3 * k] for k in range(4)]
    rec = [a[:, 18 + 8 * k:21 + 8 * k] for k in range(4)]
    ok = [(a[:, 13 + 8 * k] > 0) & (a[:, 14 + 8 * k].astype(int) == pid)
          for k, pid in enumerate((11, 2212, 22, 22))]
    ep = ok[0] & ok[1]
    gg = ok[2] & ok[3]

    gQ2, gW, gmt, gxB, gmm2 = kin(gen[0], gen[1])
    rQ2, rW, rmt, rxB, rmm2 = kin(rec[0], rec[1])

    def pair_mass(x, y):
        return np.sqrt(np.maximum(2 * (mag(x) * mag(y) - (x * y).sum(axis=1)), 0))

    def opening(x, y):
        c = (x * y).sum(axis=1) / np.maximum(mag(x) * mag(y), 1e-9)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    gpi0, rpi0 = gen[2] + gen[3], rec[2] + rec[3]

    P = [
        (r"$e'$  $|p|$ [GeV]", mag(gen[0]), mag(rec[0]), ok[0], (0, 9), 90),
        (r"$e'$  $\theta$ [deg]", theta(gen[0]), theta(rec[0]), ok[0], (0, 40), 80),
        (r"$p$  $|p|$ [GeV]", mag(gen[1]), mag(rec[1]), ok[1], (0, 2.2), 88),
        (r"$p$  $\theta$ [deg]", theta(gen[1]), theta(rec[1]), ok[1], (0, 80), 80),
        (r"$\gamma_1$  $|p|$ [GeV]", mag(gen[2]), mag(rec[2]), ok[2], (0, 7), 90),
        (r"$\gamma_1$  $\theta$ [deg]", theta(gen[2]), theta(rec[2]), ok[2], (0, 45), 90),
        (r"$\gamma_2$  $|p|$ [GeV]", mag(gen[3]), mag(rec[3]), ok[3], (0, 7), 90),
        (r"$\gamma_2$  $\theta$ [deg]", theta(gen[3]), theta(rec[3]), ok[3], (0, 45), 90),
        (r"$M(\gamma\gamma)$ [GeV]", pair_mass(gen[2], gen[3]),
         pair_mass(rec[2], rec[3]), gg, (0.0, 0.30), 90),
        (r"$\gamma\gamma$ opening [deg]", opening(gen[2], gen[3]),
         opening(rec[2], rec[3]), gg, (0, 60), 90),
        (r"$\pi^0$  $|p|$ [GeV]", mag(gpi0), mag(rpi0), gg, (0, 9), 90),
        (r"$\pi^0$  $\theta$ [deg]", theta(gpi0), theta(rpi0), gg, (0, 60), 90),
        (r"$Q^2$ [GeV$^2$]", gQ2, rQ2, ok[0], (0.5, 10), 95),
        (r"$W$ [GeV]", gW, rW, ok[0], (1.7, 4.3), 90),
        (r"$-t$ [GeV$^2$]", gmt, rmt, ok[1], (0, 2.2), 88),
        (r"$MM^2(e'p)$ [GeV$^2$]", gmm2, rmm2, ep, (-0.2, 1.0), 96),
    ]

    fig, ax = plt.subplots(4, 4, figsize=(18, 13))
    fig.suptitle(f"{tag}   —   generated against reconstructed,   {n/1e6:.2f} M events"
                 "   (reconstructed = matched AND identified)"
                 + ("   [log scale]" if logy else ""), fontsize=14, y=0.988)
    for a_, (lab, g, r, m, rng, nb) in zip(ax.ravel(), P):
        a_.hist(g, bins=nb, range=rng, color=GEN_FACE, alpha=0.55,
                edgecolor="none", label="generated")
        a_.hist(r[m], bins=nb, range=rng, color=REC_FACE, alpha=0.65,
                edgecolor="none", label="reconstructed")
        if logy:
            a_.set_yscale("log")
            # a filled bar on a log axis runs to the bottom of the frame and
            # the panel turns into a solid block; start the axis just under
            # the first count so the shape is visible again
            top = max(np.histogram(g, bins=nb, range=rng)[0].max(), 1)
            a_.set_ylim(0.5, top * 3)
        a_.set_xlabel(lab, fontsize=10)
        a_.set_ylabel("events / bin", fontsize=9)
        a_.tick_params(labelsize=8.5)
        a_.grid(alpha=0.2, lw=0.5)
        a_.text(0.97, 0.94, f"{100*m.mean():.1f} %", transform=a_.transAxes,
                ha="right", va="top", fontsize=10, color=REC_EDGE, weight="bold")
    ax[0, 0].legend(loc="upper left", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.968])
    png = os.path.join(out, f"{tag}_gen_vs_rec{'_log' if logy else ''}.png")
    fig.savefig(png, dpi=130)
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
