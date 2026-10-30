"""Matched-window validation: exclurad_py vs EXCLURAD on the published eta bin.

Reproduces the numbers in ../../REPORT.md section 3.2.

    gfortran -std=legacy -w -O2 -ffixed-line-length-132 exclurad.F fint.F -o exclurad.exe
    ./exclurad.exe            # writes radcor_all.vec (20 points, vcut from input.dat)
    python3 run_matched.py

WHY A MATCHED WINDOW.  EXCLURAD's inelasticity limit (exclurad.F bornkin,
`vmax = w2 - amu2 + amh2 - 2 sqrt(w2 amh2)`) is the GLOBAL bound at fixed
detected-hadron ANGLE -- 3.58 GeV^2 at this bin, so the 0.20 cut binds.
exclurad_py works at fixed observed t, where the kinematic limit is 0.0845.
Comparing those directly measures the bin definition, not the physics
(it gives -0.043 on eta here).  Setting vcut = 0.08, below both limits,
makes the two integrate the identical v window.
"""
import numpy as np
from exclurad_py import models
from exclurad_py.validation.legacy import (rc_over_points, compare,
                                           format_comparison)

EBEAM, VCUT, M_ETA_EXCL = 5.75, 0.08, 0.5473
M_P = 0.93827208816

a = np.loadtxt("radcor_all.vec")
Q2, xb, tabs, phid, delta_f = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 5]
tN, phi = -tabs, np.radians(phid)
# t IS PASSED STRAIGHT THROUGH -- DO NOT CALL excl_to_mesonrc HERE.  Until
# 2026-10-08 this line read
#     W2, tk, phik = excl_to_mesonrc(Q2, xb, tN, phi, M_ETA_EXCL)
# and that converted t ONCE TOO OFTEN.  excl_to_mesonrc returns the MESON-side
# transfer t_kin = (q - p')^2, and its docstring says why: "mesonrc's EventKin,
# in the baryon_detected topology, is built from the DETECTED PROTON, so it
# needs t_kin = (q - p')^2".  That was true until 72f8dc6 (2026-08-12) gave
# `baryon_detected` t_kind="target", after which the topology takes t_N
# DIRECTLY -- and this driver was not updated.  So an already-converted t was
# handed to a topology expecting the unconverted one, the comparison read
# -7.99 % instead of +0.20 %, and it stayed that way for eight weeks.
#
# Measured against the Fortran column at vcut 0.08, 20 points, both t inputs
# crossed against all three t_kinds:
#
#     excl_to_mesonrc t + photon    +0.199 %     raw t_N + target    +0.198 %  <- this
#     excl_to_mesonrc t + target    -7.992 %     raw t_N + cm_angle  +0.198 %
#     excl_to_mesonrc t + cm_angle  -7.991 %     raw t_N + photon    -7.991 %
#
# The 8 % was never a disagreement about which invariant is right: every
# CONSISTENT pairing agrees with EXCLURAD and every inconsistent one fails by
# the same amount.  Mechanically it is delta_inf via S' = 2 k1.Lam (0.425
# against 7.630), because Lam is the Born recoil that t_kind builds from
# whichever t it is given.  'photon' appeared to be the one that worked only
# because it is the correct partner of the conversion -- and it stops working
# as soon as its own kinematic limit (0.0862 here) binds, which is why it
# agrees at vcut 0.02-0.08 and walks away above.
#
# Over a Fortran vcut sweep with this fix, t_kind='target' throughout:
#
#     vcut   0.02   0.05   0.08   0.12   0.17   0.20   0.30   0.50
#     diff  +0.01  +0.16  +0.20  +0.24  +0.28  +0.30  +0.39  +0.51  %
#
W2 = Q2 * (1.0 / xb - 1.0) + M_P ** 2
tk, phik = tN, phi

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
print(format_comparison(compare(eta, delta_f, "exclurad_py", "EXCLURAD matched"),
                        tol_rel=5e-3))
print("\nNote: EXCLURAD writes delta with f8.3, i.e. 0.001 granularity "
      "= 0.11% at delta~0.9, comparable to the residual itself.")
