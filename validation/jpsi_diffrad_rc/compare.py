"""J/psi radiative correction: exclurad_py vs DIFFRAD, SAME cross-section model.

Both codes carry the Bardin-Shumeiko RC scheme; the point of this test is to
run them on the IDENTICAL J/psi cross section so the comparison isolates the
RC machinery, not the model.  The model is DIFFRAD's own two-gluon `difflt`
(idiffrad.f, ivec=4): it lives in exclurad_py as the SF model `jpsi.diffrad`
(an exact line-by-line port), and it is what DIFFRAD uses internally.  So the
same sigma_T(Q2,W2,t), sigma_L = Q2/m^2 sigma_T feed both RC engines.

What is compared.  The multiplicative RC factor
    eta = sigma_observed / sigma_Born
which exclurad_py returns as `eta` and DIFFRAD prints as the `de` column of
its "main" line (de = dev + der = soft+virtual/Born + radiative-tail/Born).

Soft-photon cut matching.  DIFFRAD caps its inelasticity at `cutv`
(idiffrad/mdiffrad line 452: vmax = min(vmax, cutv)); exclurad_py caps
v_m = mx2_cut - M_p^2.  Both use v = M_X^2 - M_p^2, so cutv <-> mx2_cut =
M_p^2 + cutv.  Verified: exclurad_py reports v_max = cutv exactly.

DIFFRAD input convention (decoded from rankin): a NEGATIVE xmas flags a W2
range (|xmas| bounds), a NEGATIVE ymas flags a Q2 range, tmas is the actual
t range.  We use a degenerate (min=max) bin per point, so DIFFRAD evaluates
at the single kinematic point.

    python3 compare.py            # runs the grid, writes csv + png + md

Requires mdiffrad.exe built in ~/RC_vector_mesons_DIFFRAD.
"""
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exclurad_py.models.registry import get
from exclurad_py.core.rc import rc_factor
from exclurad_py.core.constants import M_P

HERE = os.path.dirname(os.path.abspath(__file__))
DIFFRAD_DIR = os.path.expanduser("~/RC_vector_mesons_DIFFRAD")
EXE = os.path.join(DIFFRAD_DIR, "mdiffrad.exe")

EBEAM = 10.6           # GeV, CLAS12 near-threshold J/psi
CUTV = 0.03            # GeV^2, soft-photon cut, active at every grid point
SEED = 333522
ANN = (200000.0, 200000.0, 100000.0)   # Born, soft+virt, tail sample counts
NEV = 4                # MC repetitions per point (for the DIFFRAD error bar)

# three 1-D scans crossing at (W2=18.5, Q2=1.0, t=-1.8)
CROSS = dict(W2=18.5, Q2=1.0, t=-1.8)
SCANS = {
    "t":  [(18.5, 1.0, tt) for tt in (-1.2, -1.5, -1.8, -2.1, -2.4)],
    "Q2": [(18.5, qq, -1.8) for qq in (0.5, 1.0, 1.5)],
    "W2": [(ww, 1.0, -1.8) for ww in (17.5, 18.0, 18.5, 19.0)],
}


def run_diffrad(W2, Q2, t):
    """Return (eta_mean, eta_err) from DIFFRAD at one J/psi point."""
    inp = os.path.join(DIFFRAD_DIR, "inmdi.dat")
    with open(inp, "w") as fh:
        fh.write("%f\n0.0\n1\n4\n" % EBEAM)          # bmom, tmom, e-, ivec=J/psi
        fh.write("%f %f %f\n" % ANN)
        fh.write("%f\n%d\n%d\n1\n" % (CUTV, NEV, SEED))
        # npoi=1: xmas1,xmas2 (=-W2), ymas1,ymas2 (=-Q2), tmas1,tmas2 (=t)
        fh.write("%f\n%f\n%f\n%f\n%f\n%f\n"
                 % (-W2, -W2, -Q2, -Q2, t, t))
    out = subprocess.run([EXE], cwd=DIFFRAD_DIR, capture_output=True,
                         text=True, timeout=600).stdout
    de = [float(ln.split()[4]) for ln in out.splitlines()
          if ln.strip().startswith("main")]
    if not de:
        raise RuntimeError("DIFFRAD returned no 'main' line (point rejected?)"
                           " W2=%.2f Q2=%.2f t=%.2f" % (W2, Q2, t))
    a = np.array(de)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)


def run_exclurad(m, W2, Q2, t):
    r = rc_factor(EBEAM, Q2, W2, t, 0.0, m, mx2_cut=M_P ** 2 + CUTV)
    return r


def main():
    m = get("jpsi.diffrad")
    rows = []          # W2,Q2,t, eta_D,err_D, eta_x, d_inf,d_VR,d_vac,tail
    seen = set()
    order = []
    for scan in ("t", "Q2", "W2"):
        for (W2, Q2, t) in SCANS[scan]:
            key = (W2, Q2, t)
            if key in seen:
                continue
            seen.add(key)
            order.append(key)
            eD, errD = run_diffrad(W2, Q2, t)
            r = run_exclurad(m, W2, Q2, t)
            rows.append((W2, Q2, t, eD, errD, r["eta"],
                         r["delta_inf"], r["delta_VR"], r["delta_vac"],
                         r["sigma_F"] / r["sigma_born"]))
            print("W2=%4.1f Q2=%3.1f t=%+.2f | DIFFRAD eta=%.4f+-%.4f | "
                  "exclurad eta=%.4f | diff=%+.4f"
                  % (W2, Q2, t, eD, errD, r["eta"], r["eta"] - eD))
    rows = np.array(rows)

    hdr = ("W2,Q2,t,eta_diffrad,err_diffrad,eta_exclurad,"
           "d_inf,d_VR,d_vac,tail_over_born")
    np.savetxt(os.path.join(HERE, "jpsi_rc_compare.csv"), rows,
               delimiter=",", header=hdr, comments="", fmt="%.6g")

    _plots(rows)
    _report(rows)


def _plots(rows):
    keyed = {(r[0], r[1], r[2]): r for r in rows}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        ("t",  [(18.5, 1.0, tt) for tt in (-1.2, -1.5, -1.8, -2.1, -2.4)],
         2, r"$t\;[\mathrm{GeV}^2]$",
         r"$W^2=18.5,\;Q^2=1.0\;\mathrm{GeV}^2$"),
        ("Q2", [(18.5, qq, -1.8) for qq in (0.5, 1.0, 1.5)],
         1, r"$Q^2\;[\mathrm{GeV}^2]$",
         r"$W^2=18.5\;\mathrm{GeV}^2,\;t=-1.8\;\mathrm{GeV}^2$"),
        ("W2", [(ww, 1.0, -1.8) for ww in (17.5, 18.0, 18.5, 19.0)],
         0, r"$W^2\;[\mathrm{GeV}^2]$",
         r"$Q^2=1.0\;\mathrm{GeV}^2,\;t=-1.8\;\mathrm{GeV}^2$"),
    ]
    for a, (name, pts, xi, xlab, sub) in zip(ax, panels):
        xs = [p[xi] for p in pts]
        eD = [keyed[p][3] for p in pts]
        erD = [keyed[p][4] for p in pts]
        eX = [keyed[p][5] for p in pts]
        a.errorbar(xs, eD, yerr=erD, fmt="s-", color="#c0392b",
                   label="DIFFRAD (Fortran)", capsize=3, ms=6)
        a.plot(xs, eX, "o-", color="#2471a3", label="exclurad_py", ms=6)
        a.set_xlabel(xlab)
        a.set_title(sub, fontsize=9)
        a.grid(alpha=0.3)
    ax[0].set_ylabel(r"RC factor  $\eta=\sigma_{\rm obs}/\sigma_{\rm Born}$")
    ax[0].legend(fontsize=9, loc="best")
    fig.suptitle(r"J/$\psi$ radiative correction, same model (difflt), "
                 r"$E=10.6$ GeV, $v_{\rm cut}=%.2f$ GeV$^2$" % CUTV,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, "jpsi_rc_compare.png"), dpi=140)
    print("wrote jpsi_rc_compare.png")


def _report(rows):
    d = rows[:, 5] - rows[:, 3]           # eta_x - eta_D
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("# J/psi RC: exclurad_py vs DIFFRAD (same difflt model)\n\n")
        f.write("Beam %.1f GeV, v_cut %.2f GeV^2, model `jpsi.diffrad` "
                "(exact port of difflt, ivec=4) in both codes.\n\n" %
                (EBEAM, CUTV))
        f.write("eta = sigma_obs/sigma_Born (DIFFRAD: `de` column).\n\n")
        f.write("| W2 | Q2 | t | eta DIFFRAD | eta exclurad_py | diff |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write("| %.1f | %.1f | %+.2f | %.4f +- %.4f | %.4f | %+.4f |\n"
                    % (r[0], r[1], r[2], r[3], r[4], r[5], r[5] - r[3]))
        f.write("\nMean diff (exclurad - DIFFRAD): %+.4f, "
                "RMS %.4f, max |diff| %.4f.\n"
                % (d.mean(), np.sqrt((d ** 2).mean()), np.abs(d).max()))
        f.write("\nBoth carry the same vacuum polarization "
                "(alpha/pi * vacpol = d_vac = 0.033 at Q2=1). The residual "
                "sits in the soft+virtual factor: DIFFRAD's deltavr is the "
                "inclusive/DIS vertex form with shifted Mandelstams, "
                "exclurad_py uses the exclusive vertex.\n")
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
