"""exclurad_py against Fortran EXCLURAD as a function of vcut, on the published
eta bin, with an explicit NaN/inf hunt on both sides.

The matched comparison (validation/eta_matched) is a SINGLE vcut = 0.08, chosen
to sit below every fixed-t kinematic limit so the two codes integrate the
identical v window.  This sweeps vcut instead, which is the only way to see
WHERE the two definitions part company rather than taking it on trust:

    EXCLURAD    bins at fixed detected-hadron ANGLE; its vmax is the global
                bound (exclurad.F bornkin), 3.58 GeV^2 at this bin, so a cut
                below that always binds.
    exclurad_py works at fixed observed t, where v_max(t) is 0.0845 here.

So below ~0.0845 the windows match and any residual is physics; above it
exclurad_py truncates at v_max(t) while EXCLURAD keeps integrating to the cut,
and the difference is the bin definition.  Both regimes are measured here.

vcut = 0 means NO cut in BOTH codes (exclurad.F:338 guards on cutv > 1d-12;
exclurad_py takes mx2_cut=None), but "no cut" is a different window in each,
so that row is the fully unmatched baseline.

RESULT, 2026-10-08.  Two things, and the second is the important one.

1. NaN.  exclurad_py produces ZERO non-finite eta at every vcut from 0.01 to
   2.0 and at vcut = 0 (no cut).  The Fortran produces 2 non-finite points of
   20, at vcut = 0 only.  So the NaN question is clean on our side.

2. THE ETA VALIDATION IS CURRENTLY BROKEN, AND REPORT.md SECTION 3.2 IS STALE.
   At the matched vcut = 0.08 this comparison gave +0.199 % when the CSV was
   committed (d88f0ea, 2026-08-12).  It now gives -7.992 %, verdict
   DISCREPANT.  The Fortran column is BYTE-IDENTICAL to the committed CSV, so
   the whole move is ours: every exclurad_py point is lower by 0.066-0.085.

   CAUSE: `eta.tableB1` violates positivity at low W2~ and drives sigma_F
   NEGATIVE.  Scanned down in W2~ at Q2 = 1.15, t_nucl = -0.12 -- the path the
   radiative integral takes -- sigma_TT and sigma_LT are exactly -0 between
   W2~ = 2.5 and 4.0, then jump to large negative values below 2.0:

       W2~     sT      sL      sTT      sLT     |sTT|/sT
       4.00   56.4   440.7      -0       -0
       2.50   83.5   651.9      -0       -0
       2.00  109.8   857.7  -177.9   -50.41       1.62
       1.50  181.2  1415.0  -305.4   -83.16       1.69
       1.30     --      --      --       --       1.72   (worst)

   |sigma_TT| > sigma_T makes sigma_T + eps*sigma_TT*cos(2phi) negative at some
   phi, so the Born cross section goes negative inside the photon phase space
   and sigma_F comes out negative: -4.5e-06 against a Born of 2.1e-03.  eta is
   then just the virtual-plus-soft factor exp(d_inf)(1+d_VR+d_vac) = 0.848,
   which is the whole 8 %.  The model's declared validity is
   W2 (1.15, 20.0), so the pathological region is INSIDE the window it claims.

   NO PRODUCTION MODEL IS AFFECTED, and this is the reason not to panic.
   Worst |sigma_TT|/sigma_T on the same scan, with sigma_F's sign:

       eta.tableB1     1.718   sigma_F -4.3e-05   <-- the only violator
       eta.amp2609     0.007   sigma_F +6.6e-04
       eta.amp2614     0.035   sigma_F +7.3e-04
       etan.amp2614    0.428   sigma_F +3.5e-04
       pi0.amp2609     0.401   sigma_F +9.1e-03
       pi0.amp2614     0.409   sigma_F +1.3e-02

   `eta.tableB1` is the frozen Table-B.1 reference used ONLY by this
   validation; it is not a production model and no campaign uses it.  So the
   defect is in the reference, not in the RC engine.

   STILL OPEN: it passed on 2026-08-12 and fails now, with the Fortran
   unchanged, so either the table interpolation moved or the v integration
   bounds did and the integral now reaches W2~ it did not before.  25
   core/model commits sit in between; `67f74f0` ("lo was shadowing the log-v
   bound") and `8d28335` (quadrature 20,12,16,12) are the candidates to bisect
   first.  Do not edit either code to make them agree -- quantify and report.

3. WHERE THE TWO DEFINITIONS PART COMPANY, which is what the sweep was for.
   exclurad_py's eta stops moving above vcut ~ 0.15 because v_max(t) binds,
   while EXCLURAD keeps integrating to the cut:

       vcut    eta EXCL   eta py   mean rel
       0.08      0.9092   0.8365    -7.99 %
       0.15      0.9462   0.8709    -7.95 %
       0.20      0.9632   0.8807    -8.56 %
       0.50      1.0169   0.8807   -13.39 %
       2.00      1.0855   0.8807   -18.85 %
       0 (none)  1.0942   0.8807   -19.61 %

   Below 0.15 the residual is flat at -8.0 %, which is the sigma_F defect
   alone; above it the growth is the bin definition on top.  That flatness is
   what says the two effects are separable.

NOTE a second, smaller inconsistency found on the way: `rc_over_points` in
validation/legacy.py computes its reported v_max_kin WITHOUT passing t_kind,
so it uses the default 'photon' while rc_factor uses the model's 'target'.
The two disagree (0.0861 against 0.1787 at this bin).  Only the REPORTED
v_max is affected, not any eta, but it is why the plateau above reads 0.0861.
"""
import os, re, subprocess, sys
import numpy as np
from exclurad_py import models
from exclurad_py.validation.legacy import rc_over_points

EBEAM, M_ETA = 5.75, 0.5473
HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_VCUT = 0.08   # restored on exit so the committed input.dat is unchanged
VCUTS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.0845, 0.10, 0.15, 0.20, 0.30, 0.50,
         1.00, 2.00, 0.0]


def set_vcut(v):
    p = os.path.join(HERE, "input.dat")
    lines = open(p).read().split("\n")
    for i, l in enumerate(lines):
        if "vcut" in l:
            lines[i] = f"{v:<8g}!  vcut - set by scan.py"
            break
    else:
        raise RuntimeError("no vcut line in input.dat")
    open(p, "w").write("\n".join(lines))


def run_fortran():
    r = subprocess.run([os.path.join(HERE, "exclurad.exe")], cwd=HERE,
                       capture_output=True, text=True, timeout=1800)
    out = os.path.join(HERE, "radcor_all.vec")
    if r.returncode != 0:
        return None, f"exit {r.returncode}: {r.stderr.strip()[:120]}"
    a = np.loadtxt(out)
    return a, None


def main():
    print(__doc__)
    print(f"{'vcut':>7} {'nF':>3} {'npy':>4} | {'<eta> EXCL':>10} "
          f"{'<eta> py':>9} {'mean rel':>9} {'max rel':>8} | "
          f"{'v_max(t)':>8} {'binds':>6} | {'NaN F':>5} {'NaN py':>6}")
    rows = []
    for v in VCUTS:
        set_vcut(v)
        a, err = run_fortran()
        if a is None:
            print(f"{v:7.4f}  FORTRAN FAILED: {err}")
            continue
        Q2, xb, tabs, phid, dF = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 5]
        tN, phi = -tabs, np.radians(phid)
        from exclurad_py.validation.legacy import excl_to_mesonrc
        W2, tk, phik = excl_to_mesonrc(Q2, xb, tN, phi, M_ETA)
        etas, vmaxes = [], []
        for i in range(len(Q2)):
            sf = models.get("eta.tableB1", t_nucl=float(tN[i]))
            r = rc_over_points(sf, [(Q2[i], W2[i], tk[i], phik[i])], EBEAM,
                               mx2_cut=None if v == 0.0 else v + M_ETA ** 2)[0]
            etas.append(r["eta"]); vmaxes.append(r["v_max"])
        etas = np.array(etas); vmaxes = np.array(vmaxes)
        nanF = int(np.sum(~np.isfinite(dF)))
        nanpy = int(np.sum(~np.isfinite(etas)))
        ok = np.isfinite(etas) & np.isfinite(dF) & (dF != 0)
        rel = (etas[ok] - dF[ok]) / dF[ok] if ok.any() else np.array([np.nan])
        vk = np.nanmedian(vmaxes)
        binds = "py" if (v > 0 and vk < v - 1e-9) else ("cut" if v > 0 else "none")
        print(f"{v:7.4f} {len(dF):3d} {int(ok.sum()):4d} | {np.nanmean(dF):10.4f} "
              f"{np.nanmean(etas):9.4f} {100*np.nanmean(rel):8.3f}% "
              f"{100*np.nanmax(np.abs(rel)):7.3f}% | {vk:8.4f} {binds:>6} | "
              f"{nanF:5d} {nanpy:6d}")
        rows.append((v, np.nanmean(dF), np.nanmean(etas), np.nanmean(rel),
                     np.nanmax(np.abs(rel)), vk, nanF, nanpy))
    set_vcut(ORIG_VCUT)   # leave input.dat as committed
    np.savetxt(os.path.join(HERE, "vcut_scan.csv"), np.array(rows),
               delimiter=",", fmt="%.6g",
               header="vcut,eta_exclurad,eta_py,mean_rel,max_abs_rel,"
                      "v_max_t,nan_fortran,nan_py", comments="")
    print("\n-> vcut_scan.csv")


if __name__ == "__main__":
    main()
