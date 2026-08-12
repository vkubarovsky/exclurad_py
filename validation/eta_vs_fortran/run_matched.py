"""Matched-window validation: mesonrc vs EXCLURAD on the published eta bin.

Reproduces the numbers in ../../REPORT.md section 3.2.

    gfortran -std=legacy -w -O2 -ffixed-line-length-132 exclurad.F fint.F -o exclurad.exe
    ./exclurad.exe            # writes radcor_all.vec (20 points, vcut from input.dat)
    python3 run_matched.py

WHY A MATCHED WINDOW.  EXCLURAD's inelasticity limit (exclurad.F bornkin,
`vmax = w2 - amu2 + amh2 - 2 sqrt(w2 amh2)`) is the GLOBAL bound at fixed
detected-hadron ANGLE -- 3.58 GeV^2 at this bin, so the 0.20 cut binds.
mesonrc works at fixed observed t, where the kinematic limit is 0.0845.
Comparing those directly measures the bin definition, not the physics
(it gives -0.043 on eta here).  Setting vcut = 0.08, below both limits,
makes the two integrate the identical v window.
"""
import numpy as np
from mesonrc import models
from mesonrc.validation.legacy import (excl_to_mesonrc, rc_over_points,
                                       compare, format_comparison)

EBEAM, VCUT, M_ETA_EXCL = 5.75, 0.08, 0.5473

a = np.loadtxt("radcor_all.vec")
Q2, xb, tabs, phid, delta_f = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 5]
tN, phi = -tabs, np.radians(phid)
W2, tk, phik = excl_to_mesonrc(Q2, xb, tN, phi, M_ETA_EXCL)

rows = []
for i in range(len(Q2)):
    sf = models.get("eta.tableB1", t_nucl=float(tN[i]))
    r = rc_over_points(sf, [(Q2[i], W2[i], tk[i], phik[i])], EBEAM,
                       mx2_cut=VCUT + M_ETA_EXCL ** 2)[0]
    rows.append((phid[i], Q2[i], xb[i], tN[i], delta_f[i], r["eta"],
                 r["eta"] - delta_f[i], r["v_max"]))

hdr = "phi_deg,Q2,xB,t_N,delta_exclurad,eta_mesonrc,diff,v_max"
np.savetxt("eta_matched.csv", np.array(rows), delimiter=",", header=hdr,
           comments="", fmt="%.6g")
eta = np.array([r[5] for r in rows])
print(format_comparison(compare(eta, delta_f, "mesonrc", "EXCLURAD matched"),
                        tol_rel=5e-3))
print("\nNote: EXCLURAD writes delta with f8.3, i.e. 0.001 granularity "
      "= 0.11% at delta~0.9, comparable to the residual itself.")
