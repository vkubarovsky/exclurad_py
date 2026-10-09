#!/usr/bin/env python3
"""Born identity: the meson bin and the proton bin are the same cross section.

For 2 -> 2, q - p_V = p' - p identically, so (q - p_V)^2 = (p - p')^2 and the
two bins coincide at v = 0.  The test matters most for rho, where
SLT_OVER_SU = -0.30 makes sigma_LT non-zero: baryon_detected flips lt_sign AND
sets phi_offset = pi, and only the two TOGETHER reproduce the cross section.
A sign flip without the azimuth shift would mirror the phi profile, which is
invisible at phi = 0 and pi -- hence the scan.

    python3 validation/vector_proton_t/born_identity.py
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/validation/", 1)[0])
from exclurad_py import models                                    # noqa: E402
from exclurad_py.core.rc import rc_factor                         # noqa: E402

EB = 10.6
CASES = [("jpsi.diffrad_vm", "jpsi.diffrad_vm_pdet", 0.10, 18.0, -1.50),
         ("phi.diffrad_vm",  "phi.diffrad_vm_pdet",  2.00,  8.0, -0.50),
         ("rho.diffrad_vm",  "rho.diffrad_vm_pdet",  2.00,  8.0, -0.50)]
PHIS = np.radians([0, 30, 60, 90, 120, 150, 180, 240, 300])


def main():
    bad = 0
    for mnam, pnam, Q2, W2, t in CASES:
        worst = 0.0
        vals = []
        for ph in PHIS:
            r = []
            for nm in (mnam, pnam):
                m = models.get(nm) if "pdet" not in nm else models.get(nm, t_nucl=t)
                x = rc_factor(EB, Q2, W2, t, ph, m, h=+1)
                r.append(float(np.asarray(x["sigma_born"]).ravel()[0]))
            if r[0] <= 0:
                continue
            d = abs(r[1] / r[0] - 1.0)
            worst = max(worst, d)
            vals.append((np.degrees(ph), r[0], r[1]))
        ok = worst < 1e-10
        bad += not ok
        print("%-20s vs %-22s Q2 %.2f W2 %.1f t %+.2f : worst |ratio-1| = %.2e  %s"
              % (mnam, pnam, Q2, W2, t, worst, "ok" if ok else "FAIL"))
        if not ok:
            for p, a, b in vals:
                print("      phi %5.0f  meson %.8g  proton %.8g  ratio %.9f"
                      % (p, a, b, b / a))
    # rho carries sigma_LT: show that the phi profile is not flat, so the
    # test above has something to constrain.
    m = models.get("rho.diffrad_vm")
    s = [float(np.asarray(rc_factor(EB, 2.0, 8.0, -0.50, ph, m, h=+1)["sigma_born"]
                          ).ravel()[0]) for ph in PHIS]
    print("rho phi profile (must vary, else the lt_sign test is vacuous): "
          "min/max = %.4f" % (min(s) / max(s)))
    print("FAILURES:", bad)
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
