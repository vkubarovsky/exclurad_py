#!/usr/bin/env python3
"""The t input and the topology's t_kind must be a CONSISTENT pair.

Run from this directory.  Uses the committed eta_matched.csv for the bin and
for the Fortran column, so it needs no Fortran executable.

    raw t_N  pairs with  t_kind='target'   (and 'cm_angle')
    (q-p')^2 pairs with  t_kind='photon'

Either pair reproduces EXCLURAD to 0.2 %.  Crossing them costs 8 %, which is
how run_matched.py came to be 8 % off for eight weeks after 72f8dc6 gave
baryon_detected t_kind='target' without updating this driver.
"""
import dataclasses
import numpy as np
from exclurad_py import models
from exclurad_py.core.rc import rc_factor
from exclurad_py.validation.legacy import excl_to_mesonrc

M_P, M_ETA_EXCL, EBEAM, VCUT = 0.93827208816, 0.5473, 5.75, 0.08
TOL_OK, TOL_BAD = 1.0, 5.0          # per cent


def _eta(Q2, W2, t, phi, tN, mode):
    out = []
    for i in range(len(Q2)):
        sf = models.get("eta.tableB1", t_nucl=float(tN[i]))
        try:
            sf.topology = dataclasses.replace(sf.topology, t_kind=mode)
        except Exception:
            object.__setattr__(sf.topology, "t_kind", mode)
        r = rc_factor(Ebeam=EBEAM, Q2=Q2[i], W2=W2[i], t=t[i], phi=phi[i],
                      sfmodel=sf, h=+1, mx2_cut=VCUT + M_ETA_EXCL ** 2)
        out.append(float(np.asarray(r["eta"]).ravel()[0]))
    return np.array(out)


def main():
    c = np.genfromtxt("eta_matched.csv", delimiter=",", names=True)
    Q2, xb, tN = c["Q2"], c["xB"], c["t_N"]
    phi, dF = np.radians(c["phi_deg"]), c["delta_exclurad"]
    W2_raw = Q2 * (1.0 / xb - 1.0) + M_P * M_P
    W2_cnv, t_cnv, phi_cnv = excl_to_mesonrc(Q2, xb, tN, phi, M_ETA_EXCL)

    cases = [
        ("raw t_N            + target  ", W2_raw, tN, phi, "target", True),
        ("raw t_N            + cm_angle", W2_raw, tN, phi, "cm_angle", True),
        ("excl_to_mesonrc t  + photon  ", W2_cnv, t_cnv, phi_cnv, "photon", True),
        ("raw t_N            + photon  ", W2_raw, tN, phi, "photon", False),
        ("excl_to_mesonrc t  + target  ", W2_cnv, t_cnv, phi_cnv, "target", False),
    ]
    bad = 0
    print("%-32s %10s %8s %s" % ("pairing", "mean %", "rms %", "verdict"))
    for lab, W2, t, ph, mode, consistent in cases:
        d = 100.0 * (_eta(Q2, W2, t, ph, tN, mode) - dF) / dF
        ok = (abs(d.mean()) < TOL_OK) if consistent else (abs(d.mean()) > TOL_BAD)
        bad += not ok
        print("%-32s %+10.3f %8.3f %s" % (lab, d.mean(), d.std(),
              ("ok" if ok else "FAIL") + ("" if consistent else "  (must disagree)")))
    print("FAILURES:", bad)
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
