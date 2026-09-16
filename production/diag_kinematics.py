"""Single-sample generator sanity plots for one channel/mode.

    python3 production/diag_kinematics.py <channel> <born|rad> <lund-glob> [out.png] [Ebeam]

Reads the LUND written by exclurad_py and draws, in one figure:
  rows 1-3  per particle (e', p, meson): |p|, theta, theta vs |p| (2D)
  row  4    event-level Q2, xB, |t|
  row  5    W, MM^2 (missing mass^2 vs e'+meson), v = MM^2 - M_p^2 with v_cut

All labels English.  Histograms weighted by the LUND weight column and filled.
This verifies we generate the right stuff; it is NOT a two-generator compare.
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_lund as RL

MESON_PID = {"pi0": 111, "eta": 221}
MESON_SYM = {"pi0": r"\pi^0", "eta": r"\eta"}          # for embedding in math
MESON_TEX = {k: "$%s$" % v for k, v in MESON_SYM.items()}  # standalone
M_MESON = {"pi0": 0.1349768, "eta": 0.547862}          # missing-mass reference
VCUT = {"pi0": 0.20, "eta": 0.30}                      # cards rad_<ch>_amp2609
M_P = 0.93827208816
BLUE, FILL = "#0072B2", "#9ecae1"


def theta_deg(v):
    p = np.sqrt(v[:, 1] ** 2 + v[:, 2] ** 2 + v[:, 3] ** 2)
    with np.errstate(invalid="ignore"):
        th = np.degrees(np.arccos(np.clip(v[:, 3] / np.where(p > 0, p, 1), -1, 1)))
    return p, th


def hist(ax, x, w, xlabel, bins=60, rng=None, logy=False):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if rng is not None:
        m &= (x >= rng[0]) & (x <= rng[1])
    if rng is None and m.any():
        lo, hi = np.percentile(x[m], [0.2, 99.8])
        if hi <= lo:
            hi = lo + 1
        rng = (lo, hi)
    ax.hist(x[m], bins=bins, range=rng, weights=w[m], color=BLUE,
            histtype="stepfilled", facecolor=FILL, edgecolor=BLUE, lw=1.0)
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")


def hist2d(ax, x, y, w, xlabel, ylabel, yr=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if yr is not None:
        m &= (y >= yr[0]) & (y <= yr[1])
    rng = [[np.nanmin(x[m]), np.nanmax(x[m])], list(yr)] if yr is not None else None
    ax.hist2d(x[m], y[m], bins=60, weights=w[m], cmap="viridis", range=rng)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    ch, mode, gl = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[4] if len(sys.argv) > 4 else f"diag_{ch}_{mode}.png"
    E0 = float(sys.argv[5]) if len(sys.argv) > 5 else 10.6
    files = sorted(glob.glob(gl))
    if not files:
        sys.exit("no LUND files match: " + gl)

    acc = None
    for f in files:
        d = RL.read(f, MESON_PID[ch])
        if d["scattered"].size == 0:
            continue
        acc = d if acc is None else {k: np.concatenate([acc[k], d[k]])
                                     for k in d}
    n = acc["scattered"].shape[0]
    w = acc["weight"]
    Q2, W2, t = RL.invariants(acc, Ebeam=E0, M=M_P)
    W = np.sqrt(np.maximum(W2, 0))
    nu = (W2 + Q2 - M_P ** 2) / (2 * M_P)
    xB = Q2 / (2 * M_P * nu)
    # The RC/generator cut is EXCLURAD baryon_detected: detected = e' + recoil
    # PROTON, the MESON is the missing system.  So the exclusivity variable is
    # MX^2(e' p) = (k1 + p_target - k2 - p_recoil)^2 -- the missing meson mass^2
    # -- and v = MX^2(e'p) - m_meson^2 < vcut (base.py mU = m_meson).  At Born
    # MX^2(e'p) = m_meson^2 exactly; radiation pushes it up by v.
    k1 = np.array([E0, 0, 0, np.sqrt(E0 ** 2 - 0.511e-3 ** 2)])
    ptar = np.array([M_P, 0, 0, 0])
    miss = k1[None, :] + ptar[None, :] - acc["scattered"] - acc["recoil"]
    g = np.array([1., -1., -1., -1.])
    MM2 = np.einsum("ni,i,ni->n", miss, g, miss)
    v = MM2 - M_MESON[ch] ** 2

    # e' scattered at small angle -- cap theta for a readable 2D panel
    THETA_MAX = {"scattered": 30.0}
    parts = [("scattered", r"$e'$"), ("recoil", r"$p$"),
             ("meson", MESON_TEX[ch])]
    fig, ax = plt.subplots(5, 3, figsize=(13, 16))
    for r, (key, tex) in enumerate(parts):
        p, th = theta_deg(acc[key])
        thmax = THETA_MAX.get(key)
        trng = (0.0, thmax) if thmax else None
        hist(ax[r, 0], p, w, r"%s $|p|$ [GeV]" % tex)
        hist(ax[r, 1], th, w, r"%s $\theta$ [deg]" % tex, rng=trng)
        hist2d(ax[r, 2], p, th, w, r"%s $|p|$ [GeV]" % tex,
               r"%s $\theta$ [deg]" % tex, yr=trng)

    hist(ax[3, 0], Q2, w, r"$Q^2$ [GeV$^2$]")
    hist(ax[3, 1], xB, w, r"$x_B$")
    hist(ax[3, 2], -t, w, r"$-t$ [GeV$^2$]")
    hist(ax[4, 0], W, w, r"$W$ [GeV]")
    hist(ax[4, 1], MM2, w, r"$M_X^2(e'p)$ [GeV$^2$]  (missing %s)" % MESON_TEX[ch],
         logy=True)
    ax[4, 1].axvline(M_MESON[ch] ** 2, color="grey", ls=":", lw=1)
    hist(ax[4, 2], v, w, r"$v = M_X^2(e'p) - m_{%s}^2$ [GeV$^2$]" % MESON_SYM[ch],
         logy=True)
    ax[4, 2].axvline(VCUT[ch], color="#D55E00", ls="--", lw=1.2,
                     label=r"$v_{\rm cut}=%.2f$" % VCUT[ch])
    ax[4, 2].legend(fontsize=8, loc="upper right")

    fig.suptitle(r"%s  %s  --  exclurad_py amp2609, %d events, $E=%.1f$ GeV"
                 % (MESON_TEX[ch], mode, n, E0), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=130)
    print("wrote %s  (%d events from %d files)" % (out, n, len(files)))


if __name__ == "__main__":
    main()
