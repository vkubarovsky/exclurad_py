"""Where in v do exclurad_py and EXCLURAD part?  eta(v_cut) ladders.

    python3 validation/jpsi_exclurad_fortran/diag_vladder.py <EXCLURAD build dir> <work dir> [jobs]

The 0.2 % pi0/eta agreement with EXCLURAD (REPORT.md 3.2, eta_matched/) was
established at v_cut = 0.08 GeV^2 only, so the hard tail at large v was never
compared.  With no cut the J/psi comparison differs by up to 2.6 %.  This
script computes eta(v_cut) for a ladder of cuts in both codes:

  1. J/psi, identical integrand (EXCLURAD JPSI_FIXT=1, exclurad_py
     t_kind="cm_angle" with the model at the observed t_N), at three
     (W2, t') points with large/medium/small difference, at Q2 = 0.03 and 1.5.
  2. pi0, EXCLURAD's own dvmpw model (= exclurad_py pi0.vpk2021), Q2 = 0.03
     and 1.5.  CAVEAT: EXCLURAD's pi0 model sees EXCLURAD's drifting t
     (frozen CM angle); exclurad_py's sees the fixed t_N.  The difference
     between the two codes therefore includes that drift for pi0 -- read the
     Q2 dependence and the v dependence, not the absolute offset.

<EXCLURAD build dir> holds exclurad.F etc. of branch jpsi-check; a diagnostic
binary that writes full-precision results for every ivec (unit 95) is built
there.  Each EXCLURAD run gets its own directory and a 900 s timeout.
"""
import dataclasses
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

EBEAM = 10.6
M_P = 0.93827208816
M_JPSI = 3.0969
M_PI_EXCL = 0.13957          # EXCLURAD's amu for ivec 1 (see pi0_published/run_direct.py)
M_PI0 = 0.134976             # what dvmpw uses for the pi0 kinematics
VCUTS = [0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 0.0]     # 0 = no cut


def tmin(W2, Q2, m):
    W = np.sqrt(W2)
    pi = np.sqrt(((W2 + Q2 + M_P ** 2) / (2 * W)) ** 2 - M_P ** 2)
    ev = (W2 + m * m - M_P ** 2) / (2 * W)
    pf = np.sqrt(ev * ev - m * m)
    return (pi - pf) ** 2 - ((Q2 + m * m) / (2 * W)) ** 2


def cases():
    out = []
    for Q2 in (0.03, 1.5):
        for W2, tp in ((17.0, 1.4), (18.5, 0.6), (20.0, 0.1)):
            out.append(("jpsi", Q2, W2, tmin(W2, Q2, M_JPSI) + tp, 20.0))
        for tp in (0.2, 0.6):
            out.append(("pi0", Q2, 5.0, tmin(5.0, Q2, M_PI0) + tp, 20.0))
    return out


def build(exdir):
    src = open(os.path.join(exdir, "exclurad.F")).read()
    src = src.replace("              if(ivec.eq.4) write(95,", "              write(95,", 1)
    src = src.replace("	 open(unit=98,file='radcor.vec',status='unknown')",
                      "	 open(unit=98,file='radcor.vec',status='unknown')\n"
                      "	 open(unit=95,file='jpsi_eta.dat',status='unknown')", 1)
    src = src.replace("      if(ivec.eq.4) close(95)", "      close(95)", 1)
    open(os.path.join(exdir, "ex_diag.F"), "w").write(src)
    subprocess.run(["gfortran", "-std=legacy", "-w", "ex_diag.F", "fint.F", "-o", "ex_diag"],
                   cwd=exdir, check=True)
    return os.path.join(exdir, "ex_diag")


def run_fortran(args):
    binary, work, tag, ch, Q2, W2, ta, ph, vc = args
    d = os.path.join(work, f"F_{tag}_v{vc:g}")
    os.makedirs(d, exist_ok=True)
    xB = Q2 / (W2 + Q2 - M_P ** 2)
    line = f"1 {Q2:.6f} {xB:.8f} {ta:.6f} {ph:.3f} 0.0\n"
    ivec, iphy, pts = (4, 12, "jpsi_points.vec") if ch == "jpsi" else (1, 11, "gn_aver_eta.vec")
    open(os.path.join(d, pts), "w").write(line)
    if ch == "pi0":
        open(os.path.join(d, "jpsi_points.vec"), "w").write("")
    open(os.path.join(d, "input.dat"), "w").write(f"{iphy}\n0\n{EBEAM}\n0.0\n1\n{ivec}\n{vc}\n1\n")
    env = dict(os.environ, JPSI_FIXT="1")
    try:
        subprocess.run([binary], cwd=d, env=env, capture_output=True, timeout=900)
        a = np.loadtxt(os.path.join(d, "jpsi_eta.dat"), ndmin=2)
        return float(a[0, 4])
    except Exception:
        return float("nan")


def run_python(args):
    tag, ch, Q2, W2, ta, ph, vc = args
    os.environ["RC_QUAD"] = "160,64,96,64"
    from exclurad_py import rc_factor, models
    from compare import JpsiProtonDetected
    if ch == "jpsi":
        sf = JpsiProtonDetected(-ta, "cm_angle")
        mU = M_JPSI
    else:
        sf = models.get("pi0.vpk2021", t_nucl=-ta)
        sf.topology = dataclasses.replace(sf.topology, t_kind="cm_angle")
        mU = M_PI_EXCL
    cut = None if vc == 0 else vc + mU ** 2
    try:
        r = [rc_factor(Ebeam=EBEAM, Q2=Q2, W2=W2, t=-ta, phi=np.radians(ph), sfmodel=sf,
                       h=h, mx2_cut=cut) for h in (1, -1)]
        return (r[0]["sigma_obs"] + r[1]["sigma_obs"]) / (r[0]["sigma_born"] + r[1]["sigma_born"])
    except Exception as e:
        return float("nan")


def main(exdir, work, jobs):
    os.makedirs(work, exist_ok=True)
    binary = build(exdir)
    C = cases()
    tasksF, tasksP = [], []
    for i, (ch, Q2, W2, ta, ph) in enumerate(C):
        tag = f"{i:02d}_{ch}"
        for vc in VCUTS:
            tasksF.append((binary, work, tag, ch, Q2, W2, ta, ph, vc))
            tasksP.append((tag, ch, Q2, W2, ta, ph, vc))
    with mp.Pool(jobs) as pool:
        resF = pool.map(run_fortran, tasksF, chunksize=1)
        resP = pool.map(run_python, tasksP, chunksize=1)
    n = len(VCUTS)
    rows = []
    for i, (ch, Q2, W2, ta, ph) in enumerate(C):
        tp = ta - tmin(W2, Q2, M_JPSI if ch == "jpsi" else M_PI0)
        print(f"\n{ch:4s} Q2={Q2:.2f} W2={W2:.1f} |t|={ta:.3f} (t'={tp:.2f})")
        print(f"   {'v_cut':>6}  {'EXCLURAD':>9}  {'exclurad_py':>11}  {'py - F':>8}")
        for k, vc in enumerate(VCUTS):
            f, p = resF[i * n + k], resP[i * n + k]
            rows.append((i, Q2, W2, ta, vc, f, p))
            print(f"   {'none' if vc == 0 else f'{vc:g}':>6}  {f:9.5f}  {p:11.5f}  {p - f:+8.5f}")
    np.savetxt(os.path.join(HERE, "diag_vladder.csv"), np.array(rows), delimiter=",",
               header="case,Q2,W2,abs_t,vcut(0=none),eta_exclurad,eta_exclurad_py", fmt="%.6f")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]),
         int(sys.argv[3]) if len(sys.argv) > 3 else 32)
