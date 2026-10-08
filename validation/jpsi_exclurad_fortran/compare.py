"""J/psi radiative correction: exclurad_py against the Fortran EXCLURAD.

    python3 validation/jpsi_exclurad_fortran/compare.py <EXCLURAD dir> [matched|native]

matched (default) -- the IDENTICAL integrand in both codes: EXCLURAD's
    parametrisation of the photon phase space (hadron CM angle frozen as v
    runs; exclurad_py t_kind="cm_angle") and the J/psi model at the observed
    t_N in both (Fortran JPSI_FIXT=1).  Tests the radiative machinery with the
    J/psi masses and nothing else.
native -- each code as it is: EXCLURAD lets the model see the t that its
    frozen CM angle implies at the shifted kinematics (t drifts, -1.29 ->
    -2.31 GeV^2 at the first point); exclurad_py holds t_N = (p - p')^2 fixed
    (t_kind="target", the physical bin definition for a measured proton).

<EXCLURAD dir> holds the binary `exclurad_jpsi` built from branch
jpsi-check of ~/exclurad-2026 (ivec = 4, iphy = 12: J/psi missing, proton
detected, the jpsi.diffrad_vm model in jpsi_sf.F).

Why the PROTON-detected topology.  EXCLURAD only knows that one: the observed
t is the nucleon transfer (p - p')^2, held fixed under radiation.  Our
generator observes t through the meson.  At "the same point" the two define
different cross sections, so the machinery is compared in EXCLURAD's
topology, with the identical model on both sides.  The meson-detected eta of
the generator is a second, internal step (same engine, different topology).

Grid: inside the generator's box (rad_jpsi.txt), no v cut (full photon phase
space), unpolarised (EXCLURAD ipol = 1; exclurad_py averaged over h = +-1).
Writes jpsi_points.vec, runs the binary, reads jpsi_eta.dat (full precision),
and prints the comparison and the decomposition of the difference.
"""
import dataclasses
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exclurad_py import rc_factor                                   # noqa: E402
from exclurad_py.models.vector import Jpsi_DiffradVM                # noqa: E402
from exclurad_py.topology import baryon_detected                    # noqa: E402
from exclurad_py.core.constants import M_P, M_JPSI                  # noqa: E402

EBEAM = 10.6


class JpsiProtonDetected(Jpsi_DiffradVM):
    """jpsi.diffrad_vm seen from the detected proton: t = t_N fixed, only
    (W2, Q2) shift in the radiative integral.  Validation only."""
    name = "jpsi.diffrad_vm_pdet"
    topology = baryon_detected(M_JPSI, meson="J/psi")

    def __init__(self, t_nucl, t_kind="target"):
        super().__init__()
        self.topology = dataclasses.replace(type(self).topology, t_kind=t_kind)
        self.t_nucl = float(t_nucl)
        self.kf_W2Q2 = None

    def _sfs(self, W2, Q2, t):
        return super()._sfs(W2, Q2, np.full_like(np.asarray(W2, float), self.t_nucl))


def tmin(W2, Q2, m=M_JPSI, M=M_P):
    W = np.sqrt(W2)
    pi = np.sqrt(((W2 + Q2 + M * M) / (2 * W)) ** 2 - M * M)
    ev = (W2 + m * m - M * M) / (2 * W)
    pf = np.sqrt(ev * ev - m * m)
    return (pi - pf) ** 2 - ((Q2 + m * m) / (2 * W)) ** 2      # |t|min, positive


def grid():
    pts = []
    for Q2 in (0.03, 0.1, 0.2):
        for W2 in (17.0, 18.5, 20.0):
            xB = Q2 / (W2 + Q2 - M_P ** 2)
            for dt in (0.1, 0.6, 1.5):
                for ph in (20.0, 100.0, 200.0, 300.0):
                    pts.append((Q2, xB, tmin(W2, Q2) + dt, ph, W2))
    return pts


def main(exdir, mode="matched"):
    pts = grid()
    with open(os.path.join(exdir, "jpsi_points.vec"), "w") as f:
        for Q2, xB, ta, ph, _ in pts:
            f.write(f"1 {Q2:.6f} {xB:.8f} {ta:.6f} {ph:.3f} 0.0\n")
    with open(os.path.join(exdir, "input.dat"), "w") as f:
        f.write("12\n0\n%.4f\n0.0\n1\n4\n0.0\n1\n" % EBEAM)
    env = dict(os.environ, JPSI_FIXT="1" if mode == "matched" else "0")
    r = subprocess.run(["./exclurad_jpsi"], cwd=exdir, capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("EXCLURAD failed")
    F = np.loadtxt(os.path.join(exdir, "jpsi_eta.dat"), ndmin=2)
    if len(F) != len(pts):
        sys.exit(f"EXCLURAD wrote {len(F)} points of {len(pts)}")
    rows = []
    for (Q2, xB, ta, ph, W2), fr in zip(pts, F):
        sf = JpsiProtonDetected(-ta, "cm_angle" if mode == "matched" else "target")
        res = [rc_factor(Ebeam=EBEAM, Q2=Q2, W2=W2, t=-ta, phi=np.radians(ph),
                         sfmodel=sf, h=h, mx2_cut=None) for h in (+1, -1)]
        eta_py = 0.5 * (res[0]["sigma_obs"] + res[1]["sigma_obs"]) / \
            (0.5 * (res[0]["sigma_born"] + res[1]["sigma_born"]))
        rows.append((Q2, W2, ta, ph, fr[4], eta_py, res[0]))
    eta_f = np.array([r[4] for r in rows])
    eta_p = np.array([r[5] for r in rows])
    d = eta_p - eta_f
    print(f"{'Q2':>5} {'W2':>5} {'|t|':>6} {'phi':>5} {'eta EXCLURAD':>13} {'eta py':>9} {'py-F':>9}")
    for (Q2, W2, ta, ph, ef, ep, _), x in zip(rows, d):
        print(f"{Q2:5.2f} {W2:5.1f} {ta:6.3f} {ph:5.0f} {ef:13.5f} {ep:9.5f} {x:+9.5f}")
    print(f"\n{len(d)} points: mean(py-F) {d.mean():+.5f}, rms {d.std():.5f}, "
          f"max |py-F| {np.abs(d).max():.5f};  mean eta EXCLURAD {eta_f.mean():.4f}, "
          f"exclurad_py {eta_p.mean():.4f}")
    rel = d / eta_f
    print(f"relative: mean {rel.mean()*100:+.3f} %, max |.| {np.abs(rel).max()*100:.3f} %")
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"jpsi_vs_exclurad_{mode}.csv"),
               np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows]),
               delimiter=",", header="Q2,W2,abs_t,phi_deg,eta_exclurad,eta_exclurad_py",
               fmt="%.6f")
    # pieces, for the first point of each (Q2, W2)
    print("\nEXCLURAD pieces (vmax, deltavr, delinf, vacpol, tail/sib):")
    for fr, row in list(zip(F, rows))[::12]:
        print(f"  Q2 {row[0]:.2f} W2 {row[1]:.1f}: vmax {fr[6]:.4f} dVR {fr[7]:.4f} "
              f"dinf {fr[8]:.4f} vac {fr[9]:.4f} tail {fr[10]:.5f} | py keys "
              + " ".join(f"{k}={row[6][k]:.4f}" for k in ("delta_VR", "delta_inf", "delta_vac")
                         if k in row[6]))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "matched")
