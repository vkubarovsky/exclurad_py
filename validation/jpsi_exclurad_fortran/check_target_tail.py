#!/usr/bin/env python3
"""RC check of the PROTON-DETECTED J/psi bin (t_kind="target").

Compares exclurad_py's jpsi.diffrad_vm_pdet against the independent
first-order tail of independent_tail.py in its new "target" mode, which
shares nothing with exclurad_py except the model and the |q|-flux
convention.

    python3 validation/jpsi_exclurad_fortran/check_target_tail.py [jobs] [quality]

Reports the tail added between successive v cuts, independent against
exclurad_py non-exponentiated, so the comparison is of the FIRST-ORDER hard
term alone -- the virtual and soft pieces are identical by construction and
would mask a tail disagreement if the exponentiated eta were compared.
"""
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from independent_tail import Point, tail_between, MJ, EBEAM          # noqa: E402

CUTS = [0.03, 0.1, 0.2, 0.5]
POINTS = [(0.03, 17.0, -2.592475, 20.0),
          (0.03, 18.5, -1.331910, 20.0),
          (0.10, 18.0, -1.500000, 60.0)]


def py_eta(Q2, W2, tN, phi, cuts):
    os.environ["RC_QUAD"] = "160,64,96,64"
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    sf = models.get("jpsi.diffrad_vm_pdet", t_nucl=tN)
    out = []
    for vc in cuts:
        cut = None if vc == 0 else vc + MJ ** 2
        r = [rc_factor(Ebeam=EBEAM, Q2=Q2, W2=W2, t=tN, phi=np.radians(phi),
                       sfmodel=sf, h=h, mx2_cut=cut, exponentiate=False)
             for h in (1, -1)]
        out.append((r[0]["sigma_obs"] + r[1]["sigma_obs"])
                   / (r[0]["sigma_born"] + r[1]["sigma_born"]))
    return np.array([float(np.asarray(x).ravel()[0]) for x in out])


def main(jobs=8, quality="mid"):
    nv, nt, nps = {"low": (8, 96, 32), "mid": (12, 160, 48),
                   "high": (16, 256, 64)}[quality]
    pool = mp.Pool(jobs)
    worst = 0.0
    for Q2, W2, tN, phi in POINTS:
        pt = Point(Q2, W2, tN, phi, mode="target")
        print("\nQ2 %.2f  W2 %.2f  t_N %+.4f  phi %.0f   v_max(target) %.4f"
              % (Q2, W2, tN, phi, pt.vmax))
        print("  soft check |M_R|^2/|M_B|^2 over the eikonal: %.6f  (must be 1)"
              % pt.soft_check())
        tasks = [(Q2, W2, tN, phi, CUTS[i], CUTS[i + 1], nv, nt, nps, "target")
                 for i in range(len(CUTS) - 1)]
        ind = pool.map(tail_between, tasks)
        py = py_eta(Q2, W2, tN, phi, CUTS)
        dpy = np.diff(py)
        print("  %-16s %14s %14s %10s" % ("v interval", "independent",
                                          "exclurad_py", "diff %"))
        for i, d in enumerate(ind):
            rel = 100.0 * (dpy[i] - d) / abs(d) if d else float("nan")
            worst = max(worst, abs(rel))
            print("  %6.2f - %-7.2f %14.6g %14.6g %+10.2f"
                  % (CUTS[i], CUTS[i + 1], d, dpy[i], rel))
    pool.close()
    print("\nworst |diff| over all intervals: %.2f %%" % worst)
    return 0 if worst < 5.0 else 1


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8,
                          sys.argv[2] if len(sys.argv) > 2 else "mid"))
