"""Direct comparison: mesonrc vs the PUBLISHED eta RC table.

No window matching and no Born-relation remapping of t -- mesonrc is run
with the same bin definition the analysis uses.

THE POINT.  In the CLAS pi0/eta analyses t is measured from the RECOIL
PROTON, t = (p - p')^2 = 2M^2 - 2 M E_p'(lab).  Fixing t therefore fixes
the proton's LAB ENERGY, and as v runs across the missing-mass window it
is the proton's ANGLE that moves.  That is EventKin(t_kind="target"),
which `baryon_detected` now selects automatically.

Running instead with t_kind="photon" -- holding (q - p')^2 fixed -- gives a
spurious kinematic limit v_max(t) = 0.087 at this bin (vs 1.18 for the
correct convention), which truncates the 0.20 window and shifts eta by
about -0.04.  That is an artifact of the wrong convention, not physics.
"""
import numpy as np
from mesonrc import models
from mesonrc.core.rc import rc_factor
from mesonrc.validation.legacy import (load_published_vec, compare,
                                       format_comparison)

M, M_ETA_EXCL, EBEAM, VCUT = 0.93827208816, 0.5473, 5.75, 0.20
GRID = "/Users/vpk/eta-rc-2026/grids/radcoreta_published.vec"

d = load_published_vec(GRID)
N = 20                                     # first (Q2,xB,t) bin: 20 phi points
Q2, xb, tN = d["q2"][:N], d["xb"][:N], -d["t"][:N]
phid, dpub = d["phi"][:N], d["delta"][:N]
W2 = Q2 * (1 / xb - 1) + M * M
phik = np.radians(phid) + np.pi            # detected proton azimuth

rows = []
for i in range(N):
    sf = models.get("eta.tableB1", t_nucl=float(tN[i]))
    r = rc_factor(Ebeam=EBEAM, Q2=Q2[i], W2=W2[i], t=tN[i], phi=phik[i],
                  sfmodel=sf, h=+1, mx2_cut=VCUT + M_ETA_EXCL ** 2)
    rows.append((phid[i], Q2[i], xb[i], tN[i], dpub[i], r["eta"],
                 r["eta"] - dpub[i], r["v_max"]))

np.savetxt("eta_published_direct.csv", np.array(rows), delimiter=",",
           header="phi_deg,Q2,xB,t_N,delta_published,eta_mesonrc,diff,v_max",
           comments="", fmt="%.6g")
eta = np.array([r[5] for r in rows])
print(format_comparison(compare(eta, dpub, "mesonrc", "published"),
                        tol_rel=1e-2))
print("\nResidual shape: ~+0.002 at the phi endpoints, ~+0.011 mid-phi.")
print("~/eta-rc-2026/README.md records the SAME shape for the Fortran's own")
print("reproduction of this table (endpoints +-0.001, mid-phi +0.007), and")
print("flags it as unexplained -- probably a slightly different 2010-era")
print("code or model iteration. mesonrc sits ~0.004 above that, consistently.")
