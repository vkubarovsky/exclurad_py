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

2. THE DRIVER CONVERTED t TWICE.  FOUND AND FIXED 2026-10-08.
   `excl_to_mesonrc` predates t_kind and converts t_N to the meson side;
   72f8dc6 made baryon_detected select t_kind='target', which takes t_N
   directly.  The driver kept converting, so for eight weeks this read
   -7.992 % instead of +0.198 %.  Crossing both t inputs with all three
   conventions: consistent pairings give +0.198/+0.199 %, inconsistent ones
   -7.99 %.  Not a convention question -- 'target' is correct, and 3.2/3.2b
   kept passing because their drivers were always consistent.

   The 8 % is delta_inf alone, via S' = 2 k1.Lam moving 0.425 -> 7.630 while
   X' and v_m hold (core/virtual.py:77); Lam is the Born recoil built from
   whichever t it is handed.  Fixed, the sweep is monotone and sub-percent:
   -0.00 % at vcut 0.02 rising to +0.51 % at 0.50, target and cm_angle
   identical throughout.

2b. A SEPARATE DEFECT OF THE REFERENCE MODEL: tableB1 violates positivity.
   Independent of the convention question and much smaller here.  Scanned down
   in W2~ at Q2 = 1.15, t_nucl = -0.12, sigma_TT and sigma_LT are exactly -0
   between W2~ 2.5 and 4.0 and large negative below 2.0:

       W2~     sT      sL      sTT      sLT     |sTT|/sT
       4.00   56.4   440.7      -0       -0
       2.00  109.8   857.7  -177.9   -50.41       1.62
       1.50  181.2  1415.0  -305.4   -83.16       1.69
       1.30     --      --      --       --       1.72   (worst)

   |sigma_TT| > sigma_T makes sigma_T + eps*sigma_TT*cos(2phi) negative at
   some phi, so the Born cross section goes negative inside the photon phase
   space and sigma_F comes out NEGATIVE.  Its declared validity is
   W2 (1.15, 20.0), so it is pathological inside the window it claims.

   BUT sigma_F is negative on BOTH paths, 20/20 points each (-4.53e-06 on
   'target', -4.77e-06 on 'photon'), i.e. also on the path that AGREES with
   the Fortran.  It is -0.2 % of Born, not 8 %.  An earlier version of this
   docstring claimed sigma_F's sign was the cause of the 8 %; it is not, and
   the two findings are independent rather than composed.

   NO PRODUCTION MODEL IS AFFECTED.  Worst |sigma_TT|/sigma_T with sigma_F's
   sign at a production point:

       eta.tableB1     1.718   sigma_F -4.3e-05   <-- the only violator
       eta.amp2609     0.007   sigma_F +6.6e-04
       eta.amp2614     0.035   sigma_F +7.3e-04
       etan.amp2614    0.428   sigma_F +3.5e-04
       pi0.amp2609     0.401   sigma_F +9.1e-03
       pi0.amp2614     0.409   sigma_F +1.3e-02

   tableB1 is the frozen Table-B.1 reference this validation uses and nothing
   else.  Taken with the convention result: it is a poor reference for a
   t-binned radiative correction, because it is unphysical at the W2~ the
   integral reaches on ANY path.

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

   Below 0.15 the residual is flat at -8.0 %, which is the convention alone;
   above it the growth is the v-range mismatch on top.  That flatness is what
   says the two effects are separable.

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
        # raw t_N: t_kind='target' takes the NUCLEON transfer directly.
        # excl_to_mesonrc would convert it a second time -- see run_matched.py.
        W2 = Q2 * (1.0 / xb - 1.0) + 0.93827208816 ** 2
        tk, phik = tN, phi
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
