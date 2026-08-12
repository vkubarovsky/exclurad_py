"""mesonrc vs the PUBLISHED CLAS6 pi0 radiative corrections (PRC 90, 025205).

Reference: ~/excl2013/radcor_all.vec -- the original July 2013 EXCLURAD
production over the 7840-point analysis grid.  Per
~/exclurad-2026/report/report.pdf that archive is reproduced byte-identically
by a rerun of the 2013 source AND by the restructured 2026 code, with
max|Delta delta| = 0.0000 over 2920 physical bins.  It is therefore a much
tighter reference than the eta table.

Settings, from the same report: E = 5.75 GeV (CLAS6),
v_cut = MM_X^2 - m_pi^2 = 0.17 GeV^2, model = pi0.vpk2013 (sigma_L dominant).
t is measured from the recoil proton, so t_kind="target" (set by the
baryon_detected topology).
"""
import numpy as np
from mesonrc import models
from mesonrc.core.rc import rc_factor
from mesonrc.validation.legacy import (load_published_vec, compare,
                                       format_comparison)

M, EBEAM, VCUT = 0.93827208816, 5.75, 0.17
M_PI_EXCL = 0.13957          # EXCLURAD uses the charged pion mass for amu
GRID = "/Users/vpk/excl2013/radcor_all.vec"
NBINS = int(__import__("os").environ.get("NBINS", "20"))

d = load_published_vec(GRID)
# rows with delta == 0 are the unphysical (flag=0) grid slots
phys = d["delta"] > 1e-6
Q2, xb, tabs, phid, dpub = (d["q2"][phys][:NBINS], d["xb"][phys][:NBINS],
                            d["t"][phys][:NBINS], d["phi"][phys][:NBINS],
                            d["delta"][phys][:NBINS])
tN = -tabs
W2 = Q2 * (1 / xb - 1) + M * M
phik = np.radians(phid) + np.pi

rows, eta = [], []
for i in range(len(Q2)):
    sf = models.get("pi0.vpk2013", t_nucl=float(tN[i]))
    try:
        r = rc_factor(Ebeam=EBEAM, Q2=Q2[i], W2=W2[i], t=tN[i], phi=phik[i],
                      sfmodel=sf, h=+1, mx2_cut=VCUT + M_PI_EXCL ** 2)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        r = dict(eta=np.nan, v_max=np.nan)
    eta.append(r["eta"])
    rows.append((phid[i], Q2[i], xb[i], tN[i], dpub[i], r["eta"],
                 r["eta"] - dpub[i], r["v_max"]))
    if NBINS > 100 and (i + 1) % 250 == 0:
        print(f"  {i+1}/{len(Q2)}", flush=True)

np.savetxt("pi0_published_direct.csv", np.array(rows), delimiter=",",
           header="phi_deg,Q2,xB,t_N,delta_2013,eta_mesonrc,diff,v_max",
           comments="", fmt="%.6g")
eta = np.array(eta)
if NBINS <= 40:
    print("%6s %9s %10s %+10s %8s" % ("phi", "delta_13", "mesonrc", "diff", "v_max"))
    for r in rows:
        print("%6.1f %9.3f %10.4f %+10.4f %8.4f" % (r[0], r[4], r[5], r[6], r[7]))
    print()
print(format_comparison(compare(eta, dpub, "mesonrc", "2013 production"),
                        tol_rel=1e-2))
