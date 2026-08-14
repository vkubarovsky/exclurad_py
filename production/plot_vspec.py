"""v-spectrum of the radiative tail, exclurad_py vs the reference generator.

v = MX^2(e'p) - mU^2 is the quantity the soft/hard split is defined on, so it
is the sharpest test of the radiative branch -- sharper than any particle
distribution, because a broken split shows up here directly as a missing tail.

Reads the LUND files and recomputes v from the four-vectors, so it is
independent of the generator's own bookkeeping.

    python3 production/plot_vspec.py <channel> [mode]
    MINE_GLOB=... python3 production/plot_vspec.py pi0 rad     # scratch sample
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hard_fraction import v_spectrum, MESON_M2, REF, REFMODE, THRESHOLDS  # noqa: E402

import glob  # noqa: E402

BLUE, VERM, GREY = "#0072B2", "#D55E00", "#666666"
OUT_DIR = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-JeffersonLab/Jlab_OneDrive/Work/exclurad_py")


def main(ch, mode="rad"):
    m2 = MESON_M2[ch]
    mine_glob = os.environ.get(
        "MINE_GLOB",
        f"/Volumes/wd_14tb/mc/exclurad_py/{ch}/*_{mode}_v1/lund/*.lund")
    a = v_spectrum(sorted(glob.glob(mine_glob)), m2)
    if not len(a):
        sys.exit(f"no exclurad_py events for {ch} {mode}")

    b = np.array([])
    if ch in REF:
        ref = sorted(glob.glob(REF[ch].format(m=REFMODE.get(mode, mode))))
        if ref:
            b = v_spectrum(ref, m2, drop_straggled=True)

    fig, (AX, RX) = plt.subplots(
        2, 1, figsize=(7.6, 6.4), sharex=True, dpi=140,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))

    lo = max(1e-5, min(a[a > 0].min() if (a > 0).any() else 1e-5, 1e-4))
    hi = max(a.max(), b.max() if len(b) else 0.0)
    bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), 55)
    ha, _ = np.histogram(a, bins=bins)
    wa = ha / max(ha.sum(), 1)
    ctr = np.sqrt(bins[:-1] * bins[1:])

    if len(b):
        hb, _ = np.histogram(b, bins=bins)
        wb = hb / max(hb.sum(), 1)
        AX.step(ctr, wb, where="mid", lw=2.4, color=VERM,
                label=f"exclugen  (N={len(b)})")
    AX.step(ctr, wa, where="mid", lw=1.7, ls="--", color=BLUE,
            label=f"exclurad_py  (N={len(a)})")
    AX.set_xscale("log")
    AX.set_yscale("log")
    AX.set_ylabel("normalized / bin")
    AX.set_title(f"{ch} {mode.upper()} — radiative tail  "
                 r"$v = M_X^2(e'p) - m_U^2$",
                 loc="left", fontsize=11, fontweight="bold")
    AX.legend(frameon=False, fontsize=9)
    AX.grid(alpha=0.25, lw=0.5)

    if len(b):
        ok = (ha > 3) & (hb > 3)
        r = np.full(len(ctr), np.nan)
        e = np.full(len(ctr), np.nan)
        r[ok] = wa[ok] / wb[ok]
        e[ok] = r[ok] * np.sqrt(1 / ha[ok] + 1 / hb[ok])
        RX.axhline(1, color=GREY, lw=1, ls="--")
        RX.errorbar(ctr[ok], r[ok], yerr=e[ok], fmt="o", ms=3, lw=0.8,
                    color=BLUE, capsize=0)
        RX.set_ylim(0, 2)
    RX.set_ylabel("new / ref", fontsize=9)
    RX.set_xlabel(r"$v$  [GeV$^2$]")
    RX.set_xscale("log")
    RX.grid(alpha=0.25, lw=0.5)

    txt = []
    for thr in THRESHOLDS:
        fa = 100.0 * (a > thr).mean()
        line = f"v>{thr:<6.3f}  {fa:6.2f}%"
        if len(b):
            line += f"  vs {100.0*(b>thr).mean():6.2f}%"
        txt.append(line)
    AX.text(0.985, 0.03, "\n".join(txt), transform=AX.transAxes,
            ha="right", va="bottom", family="monospace", fontsize=8,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.9, pad=5))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"vspec_{ch}_{mode}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pi0",
         sys.argv[2] if len(sys.argv) > 2 else "rad")
