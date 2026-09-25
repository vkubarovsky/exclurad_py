#!/usr/bin/env python3
"""Acceptance criterion 5: what Fermi motion does to the OBSERVED W.

    python3 validation/fermi_w_smearing.py free.npz deuteron.npz [out.png]

The arrays come from production/lund_kinematics.py, which reconstructs W2 the
way an analysis does -- from the scattered electron with the target assumed to
be a PROTON AT REST.  That is the point: the generator draws the physical
gamma* N invariants inside the card's box, and the smearing is what an
experiment sees on top of it.  A W box edge at 1.8 stops being a sharp edge of
the sample, which §6 of the brief says to expect and not to treat as a bug.
"""

import sys

import numpy as np

M_P = 0.93827208816


def load(path):
    d = np.load(path)
    w2 = d["W2"]
    return np.sqrt(np.maximum(w2, 0.0)), d


def main():
    free, _ = load(sys.argv[1])
    deut, _ = load(sys.argv[2])
    print(f"free nucleon : n = {len(free):7d}  <W> = {free.mean():.4f}  "
          f"rms = {free.std():.4f}  range {free.min():.3f} .. {free.max():.3f}")
    print(f"deuteron     : n = {len(deut):7d}  <W> = {deut.mean():.4f}  "
          f"rms = {deut.std():.4f}  range {deut.min():.3f} .. {deut.max():.3f}")

    # The smearing is the quadratic difference of the two widths: the free
    # sample already has the box's own spread, which is not smearing.
    extra = np.sqrt(max(deut.var() - free.var(), 0.0))
    print(f"\nsmearing (quadrature difference of the widths): {extra*1000:.0f} MeV")
    print(f"shift of the mean: {(deut.mean()-free.mean())*1000:+.0f} MeV")

    lo = np.sqrt(4.5)      # the card's W2 lower edge
    print(f"\nbox edge W = sqrt(W2_min) = {lo:.4f} GeV")
    print(f"  free     below the edge: {np.mean(free < lo):8.4%}")
    print(f"  deuteron below the edge: {np.mean(deut < lo):8.4%}")
    print(f"  deuteron below 1.8 GeV : {np.mean(deut < 1.8):8.4%}")

    for q in (1, 5, 50, 95, 99):
        print(f"  {q:3d}% quantile   free {np.percentile(free, q):.4f}   "
              f"deuteron {np.percentile(deut, q):.4f}")

    ok = 0.050 < extra < 0.250
    print("\nsmearing in the expected 100-150 MeV range: "
          + ("YES" if 0.100 <= extra <= 0.150 else
             f"{extra*1000:.0f} MeV (brief says ~100-150)"))

    if len(sys.argv) > 3:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        bins = np.linspace(min(free.min(), deut.min()) - 0.05,
                           max(free.max(), deut.max()) + 0.05, 160)
        for a, log in zip(ax, (False, True)):
            a.hist(free, bins=bins, histtype="step", density=True, log=log,
                   label="free nucleon")
            a.hist(deut, bins=bins, histtype="step", density=True, log=log,
                   label="deuteron (AV18)")
            a.axvline(lo, color="k", lw=0.9, ls="--",
                      label=r"box edge $\sqrt{W^2_{\min}}$")
            a.set_xlabel(r"observed $W$ [GeV]  (target assumed at rest)")
            a.set_ylabel("probability density [GeV$^{-1}$]")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        ax[0].set_title(f"Fermi smearing of the observed $W$: "
                        f"{extra*1000:.0f} MeV")
        ax[1].set_title("Same, logarithmic -- the box edge is no longer sharp")
        fig.tight_layout()
        fig.savefig(sys.argv[3], dpi=150)
        print(f"wrote {sys.argv[3]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
