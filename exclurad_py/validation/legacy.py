"""
Comparison against the legacy Fortran codes.

EXCLURAD (pi0/eta) and DIFFRAD (vector mesons) are kept as FROZEN
references, not as production paths.  They are never edited; this module
runs them and compares.  A regression that changes `exclurad_py.core` must move
these residuals, which is the point.

Two traps, both of which have bitten this comparison before:

1. BIN DEFINITION.  exclurad_py works at fixed observed t, where the kinematic
   limit v_max(t) can truncate the inelasticity window BELOW the nominal
   cut; EXCLURAD bins at fixed detected-hadron ANGLE, where v runs to the
   full cut.  At one CLAS12 pi0 point this alone is a ~1% effect on eta.
   Always compare at MATCHED windows (set the Fortran's cutv to mesonrc's
   v_max(t)) and report the unmatched number separately.

2. TOPOLOGY.  EXCLURAD detects the PROTON and misses the meson.  The
   corresponding exclurad_py models are ProtonDetectedModel subclasses with
   `baryon_detected` topology; using a meson-detected model instead gives
   plausible, wrong numbers.

Known residual: exclurad_py puts the undetected mass mU in the Li2 argument of
delta_VR where EXCLURAD uses the target mass (~0.1% on eta).  The covariant
derivation says mU.  Do not "fix" either code to make them agree; quantify
and report.
"""

import os
import subprocess

import numpy as np

GFORTRAN_FLAGS = ["-std=legacy", "-w", "-O2", "-ffixed-line-length-132"]


def compile_fortran(srcdir, sources, exe="a.out", workdir=None):
    """Compile a legacy Fortran setup into `workdir` (never into srcdir)."""
    workdir = workdir or srcdir
    os.makedirs(workdir, exist_ok=True)
    for s in sources:
        src = os.path.join(srcdir, s)
        dst = os.path.join(workdir, s)
        if os.path.abspath(src) != os.path.abspath(dst):
            with open(src, "rb") as a, open(dst, "wb") as b:
                b.write(a.read())
    cmd = ["gfortran"] + GFORTRAN_FLAGS + list(sources) + ["-o", exe]
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gfortran failed:\n{r.stderr[-3000:]}")
    return os.path.join(workdir, exe)


def run_exe(exe, workdir, timeout=3600):
    r = subprocess.run([f"./{os.path.basename(exe)}"], cwd=workdir,
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr, r.returncode


# --------------------------------------------------------------------------- #
def compare(delta_a, delta_b, label_a="exclurad_py", label_b="legacy"):
    """Residual statistics between two RC tables aligned point-by-point."""
    a = np.asarray(delta_a, dtype=float)
    b = np.asarray(delta_b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return dict(n=0)
    d = a[m] - b[m]
    rel = d / np.where(np.abs(b[m]) > 1e-12, b[m], np.nan)
    return dict(n=int(m.sum()),
                n_dropped=int((~m).sum()),
                mean=float(d.mean()), rms=float(d.std()),
                max_abs=float(np.abs(d).max()),
                mean_rel=float(np.nanmean(rel)),
                rms_rel=float(np.nanstd(rel)),
                max_abs_rel=float(np.nanmax(np.abs(rel))),
                label_a=label_a, label_b=label_b)


def format_comparison(stats, tol_rel=3e-3):
    if stats.get("n", 0) == 0:
        return "  no comparable points"
    ok = abs(stats["mean_rel"]) < tol_rel and stats["rms_rel"] < tol_rel
    lines = [
        f"  points compared : {stats['n']} (dropped {stats.get('n_dropped',0)})",
        f"  mean  residual  : {stats['mean']:+.5f}  "
        f"({100*stats['mean_rel']:+.3f} %)",
        f"  rms   residual  : {stats['rms']:.5f}  "
        f"({100*stats['rms_rel']:.3f} %)",
        f"  max |residual|  : {stats['max_abs']:.5f}  "
        f"({100*stats['max_abs_rel']:.3f} %)",
        f"  verdict         : {'CONSISTENT' if ok else 'DISCREPANT'} "
        f"at {100*tol_rel:.1f} % ",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def rc_over_points(sfmodel, points, Ebeam, mx2_cut=None, helicity=+1,
                   match_window=False, progress=None):
    """Evaluate exclurad_py eta over a list of (Q2, W2, t, phi) points.

    match_window=True reports, alongside eta, the fixed-t kinematic limit
    v_max(t) so a legacy run can be repeated with cutv set to it.
    """
    from ..core.kinematics import EventKin
    from ..core.rc import rc_factor

    out = []
    for n, (Q2, W2, t, phi) in enumerate(points):
        row = dict(Q2=Q2, W2=W2, t=t, phi=phi, eta=np.nan, v_max=np.nan)
        try:
            vk = EventKin.vmax_kin(Ebeam, Q2, W2, t, sfmodel.mV, mU=sfmodel.mU)
            row["v_max_kin"] = vk
            vm = vk if mx2_cut is None else min(vk, mx2_cut - sfmodel.mU ** 2)
            row["v_max"] = vm
            if vm > 0:
                r = rc_factor(Ebeam=Ebeam, Q2=Q2, W2=W2, t=t, phi=phi,
                              sfmodel=sfmodel, h=helicity, mx2_cut=mx2_cut)
                row["eta"] = r["eta"]
                row["sigma_born"] = r["sigma_born"]
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
            row["error"] = str(exc)
        out.append(row)
        if progress and (n + 1) % progress == 0:
            print(f"  {n+1}/{len(points)}")
    return out


# --------------------------------------------------------------------------- #
#  EXCLURAD grid variables -> exclurad_py arguments
# --------------------------------------------------------------------------- #
def excl_to_mesonrc(Q2, xB, t_nucl, phi_meson, m_meson, M=0.93827208816):
    """Map published-grid variables onto the detected-proton kinematics.

    Published pi0/eta grids are given in (Q2, xB, t_N, phi_meson) where
    t_N = (p - p')^2 is the NUCLEON transfer and phi_meson is the meson's
    azimuth.  mesonrc's EventKin, in the `baryon_detected` topology, is
    built from the DETECTED PROTON, so it needs

        t_kin = (q - p')^2   and   phi_kin = phi_meson + pi.

    For the 2->2 Born process gamma*(-Q2) p(M^2) -> h(m^2) p'(M^2),

        s + t_N + u = -Q2 + 2 M^2 + m^2 ,    s = W^2,  u = (q - p')^2

    so t_kin = -Q2 + 2 M^2 + m^2 - W^2 - t_N.  Getting this wrong (passing
    t_N straight through) silently shifts every point.

    Returns (W2, t_kin, phi_kin).
    """
    import numpy as _np
    Q2 = _np.asarray(Q2, dtype=float)
    xB = _np.asarray(xB, dtype=float)
    t_nucl = _np.asarray(t_nucl, dtype=float)
    W2 = Q2 * (1.0 / xB - 1.0) + M * M
    t_kin = -Q2 + 2.0 * M * M + m_meson ** 2 - W2 - t_nucl
    phi_kin = _np.asarray(phi_meson, dtype=float) + _np.pi
    return W2, t_kin, phi_kin


def load_published_vec(path, cols=("q2", "xb", "t", "phi", "X", "delta",
                                   "one", "delta8")):
    """Read a radcor*.vec table into a dict of arrays."""
    import numpy as _np
    a = _np.loadtxt(path)
    return {c: a[:, i] for i, c in enumerate(cols) if i < a.shape[1]}
