"""The guard must be silent in BOTH conventions and loud on a frozen t_nucl."""
import os, warnings, numpy as np
from exclurad_py import models
from exclurad_py.core.rc import rc_factor
import exclurad_py.core.rc as RC
from exclurad_py.validation.legacy import excl_to_mesonrc

def count(fn):
    RC._WARNED_TNUCL = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
        return len([x for x in w if "t_nucl" in str(x.message)])

def generator_convention():
    m = models.get("pi0.amp2609", t_nucl=-0.30)
    for t in (-0.25, -0.9, -1.8):
        m.t_nucl = float(t)
        rc_factor(10.6, 2.0, 9.0, t, 0.7, m, h=+1)

def exclurad_convention():
    ME = 0.5473
    # radcor_all.vec was an external ~/excl2013 reference file and is not
    # present in a clean checkout.  eta_matched.csv is the committed matched
    # subset derived from it and carries these same inputs explicitly.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "validation/eta_matched/eta_matched.csv")
    a = np.genfromtxt(path, delimiter=",", names=True)
    Q2, xb = a["Q2"][:5], a["xB"][:5]
    tN, phid = a["t_N"][:5], a["phi_deg"][:5]
    W2, tk, phik = excl_to_mesonrc(Q2, xb, tN, np.radians(phid), ME)
    for i in range(len(Q2)):
        sf = models.get("eta.tableB1", t_nucl=float(tN[i]))
        rc_factor(Ebeam=5.75, Q2=Q2[i], W2=W2[i], t=tk[i], phi=phik[i],
                  sfmodel=sf, h=+1, mx2_cut=0.08 + ME**2)

def frozen():
    m = models.get("pi0.amp2609", t_nucl=-0.30)
    for t in (-0.25, -0.9, -1.8):
        rc_factor(10.6, 2.0, 9.0, t, 0.7, m, h=+1)

def main():
    for tag, fn, want in (("generator convention (t_nucl == t)", generator_convention, 0),
                          ("EXCLURAD convention (t = meson-side)", exclurad_convention, 0),
                          ("FROZEN t_nucl, the real bug", frozen, 1)):
        n = count(fn)
        print(f"  {tag:38} warnings={n}  expected={want}  "
              f"{'OK' if n == want else 'FAIL'}")

if __name__ == "__main__":
    main()
