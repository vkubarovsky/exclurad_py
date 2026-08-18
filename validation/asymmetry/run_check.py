"""Beam-spin asymmetry chain check, with sigma_LT' switched ON.

sigma_LT' is zero in every fit we currently carry, so the whole fifth-SF
path -- flux conversion, topology sign, tensor contraction, helicity
handling, radiative correction -- is exercised by nothing.  This script
switches it on artificially, in the pi0 (detected-proton) topology where
the sign conventions are hardest, and checks four things:

  1. CLOSURE.  The covariant tensor must agree with the canonical form
     sigma ~ ... + h sqrt(2 eps (1-eps)) sin(phi) sigma_LT', at BOTH
     helicities.  A wrong NU_LTP, or a missing lt_sign, shows up here.

  2. SHAPE.  A_LU(phi) must be ODD about phi = 180 deg.  This is the check
     that catches a sin(2phi) mix-up -- exclugen's own dvmpw.F:62 has one
     (it multiplies S_LTP by SIN(2*PHI_G)), dormant only because
     dvmpw.F:158 sets SIGMA_LTP = 0.  A sin(2phi) term is EVEN about 180,
     so this test separates the two unambiguously.

  3. AMPLITUDE.  The Born A_LU must equal the analytic ratio built from
     the model's own structure functions.

  4. RADIATIVE CORRECTION.  A_LU(observed) / A_LU(Born) must be finite and
     smooth in phi, and the ABSOLUTE shift |A_obs - A_Born| must stay small.
     Deliberately not "the ratio is below 1": that looked true at the point
     this script was first calibrated on, but it is not general.  At
     t_N = -0.80, v_cut = 0.17 the ratio reaches 1.097 at phi = 7.5 deg --
     an enhancement -- simply because A_LU itself is near its node there
     (0.0091), so a 0.0009 absolute shift is 10% in relative terms.  Near
     the nodes the ratio is the wrong figure of merit; the absolute shift
     is the one that means anything.

  5. SOFT LIMIT.  As the cut tightens, A_obs / A_Born -> 1, because soft
     emission is blind to the beam helicity and so cancels in the ratio.
     This is the one test here that does NOT lean on the canonical formula
     of check 1: if the tensor and the reference were wrong in the same
     way, checks 1-3 would still pass, but this one would not.  It is also
     not trivially satisfied -- eta itself runs 0.46 to 1.04 across the
     same range while the ratio stays at 1.

    python3 validation/asymmetry/run_check.py [sLTp_fraction]

The fraction (default 0.15) is sigma_LT' expressed as a fraction of
sigma_T; the real fit will supply its own.
"""
import sys

import numpy as np

from exclurad_py import models
from exclurad_py.models.pseudoscalar import Pi0_VPK2021
from exclurad_py.core.kinematics import EventKin
from exclurad_py.core.born import born_xsec, born_reference
from exclurad_py.core.rc import rc_factor

EB, Q2, W2, TN = 10.6, 2.0, 9.0, -0.30
PHID = np.arange(7.5, 360.0, 15.0)


def make_model(frac):
    """pi0.vpk2021 with sigma_LT' = frac * sigma_T bolted on."""

    class Pi0WithLTp(Pi0_VPK2021):
        name = "pi0.vpk2021+ltp"

        def _sfs_tn(self, t_nucl, xB, Q2_):
            sT, sL, sTT, sLT = super()._sfs_tn(t_nucl, xB, Q2_)
            return sT, sL, sTT, sLT, frac * sT

    return Pi0WithLTp(t_nucl=TN)


def kin_at(sf, phi):
    return EventKin(EB, Q2, W2, TN, phi, sf.mV, v=0.0, mU=sf.mU,
                    t_kind=getattr(sf.topology, "t_kind", "photon"),
                    phi_offset=getattr(sf.topology, "phi_offset", 0.0))


def main(frac=0.15):
    sf = make_model(frac)
    base = models.get("pi0.vpk2021", t_nucl=TN)
    d = sf(np.array([W2]), np.array([Q2]), np.array([TN]))
    d0 = base(np.array([W2]), np.array([Q2]), np.array([TN]))
    print(f"sigma_LT' = {frac} * sigma_T включена искусственно")
    print(f"  sLTp = {float(d['sLTp'][0]):+.5e}   "
          f"(в штатной модели {float(d0['sLTp'][0]):+.1e})")
    print(f"  проверка знака: sLTp/sT = {float(d['sLTp'][0]/d['sT'][0]):+.4f}, "
          f"задано {frac:+.4f} с lt_sign={sf.topology.lt_sign:+.0f}")

    # ---- 1. closure ---------------------------------------------------- #
    ratios = []
    for h in (+1, -1):
        for pd in PHID:
            k = kin_at(sf, np.radians(pd))
            a = float(np.atleast_1d(born_xsec(k, sf, h))[0])
            comb, _ = born_reference(k, sf, h)
            ratios.append(a / float(np.atleast_1d(comb)[0]))
    r = np.array(ratios)
    spread = r.std() / r.mean()
    ok1 = spread < 1e-8
    print(f"\n1. замыкание тензор == канон (оба знака h): разброс {spread:.2e} "
          f"-> {'PASS' if ok1 else 'FAIL'}")

    # ---- 2 & 3. shape and amplitude ------------------------------------ #
    eps = kin_at(sf, 0.0).epsilon()
    sT, sL, sTT, sLT, sLTp = (float(d[k][0]) for k in
                              ("sT", "sL", "sTT", "sLT", "sLTp"))
    ab, ao = [], []
    for pd in PHID:
        ph = np.radians(pd)
        v = {}
        for h in (+1, -1):
            rr = rc_factor(Ebeam=EB, Q2=Q2, W2=W2, t=TN, phi=ph, sfmodel=sf,
                           h=h, mx2_cut=None)
            v[h] = (rr["sigma_born"], rr["sigma_obs"])
        ab.append((v[1][0] - v[-1][0]) / (v[1][0] + v[-1][0]))
        ao.append((v[1][1] - v[-1][1]) / (v[1][1] + v[-1][1]))
    ab, ao = np.array(ab), np.array(ao)

    # odd about 180 deg: A(360 - phi) = -A(phi).  PHID is symmetric, so the
    # reversed array is exactly the mirrored one.
    odd = np.abs(ab + ab[::-1]).max() / np.abs(ab).max()
    ok2 = odd < 1e-6
    # a sin(2phi) term would instead be EVEN about 180 -- report that too
    even = np.abs(ab - ab[::-1]).max() / np.abs(ab).max()
    print(f"2. A_LU нечётна около 180 град: остаток {odd:.2e} "
          f"-> {'PASS' if ok2 else 'FAIL'}   "
          f"(чётная часть {even:.3f}; у sin(2phi) было бы наоборот)")

    # same azimuth the tensor uses: the DETECTED hadron's, i.e. Trento phi
    # plus topology.phi_offset (pi here, since the proton is detected)
    pd_ = np.radians(PHID) + sf.topology.phi_offset
    den = (sT + eps * sL + eps * np.cos(2 * pd_) * sTT
           + np.sqrt(2 * eps * (1 + eps)) * np.cos(pd_) * sLT)
    ana = np.sqrt(2 * eps * (1 - eps)) * np.sin(pd_) * sLTp / den
    # relative, not absolute: A_LU is ~0.07 here, and the tensor contraction
    # accumulates float64 noise at the 1e-9 level, so an absolute 1e-9 gate
    # fails on arithmetic rather than on physics.
    dev = np.abs(ab - ana).max() / np.abs(ab).max()
    ok3 = dev < 1e-6
    print(f"3. амплитуда против аналитики: max отн.разн {dev:.2e} "
          f"-> {'PASS' if ok3 else 'FAIL'}")

    # ---- 4. radiative correction --------------------------------------- #
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(np.abs(ab) > 1e-12, ao / ab, np.nan)
    good = np.isfinite(corr)
    shift = np.abs(ao - ab).max()
    ok4 = good.all() and (corr[good] > 0).all() and shift < 0.05
    print(f"4. поправка на асимметрию A_obs/A_Born: "
          f"{np.nanmin(corr):.4f} .. {np.nanmax(corr):.4f}; "
          f"max |A_obs - A_Born| = {shift:.4f} -> {'PASS' if ok4 else 'FAIL'}")
    print(f"\n{'phi':>6} {'A_Born':>10} {'A_obs':>10} {'поправка':>10}")
    for pd, x, y, c in zip(PHID, ab, ao, corr):
        print(f"{pd:6.1f} {x:10.5f} {y:10.5f} {c:10.4f}")

    # ---- 5. soft limit -------------------------------------------------- #
    mU2 = sf.mU ** 2
    print(f"\n5. мягкий предел (phi = 75 град):")
    print(f"{'v_cut':>9} {'eta(h=+1)':>11} {'eta(h=-1)':>11} {'A_obs/A_Born':>13}")
    soft = None
    for vc in (1e-5, 1e-3, 1e-1, 0.5):
        v = {}
        for h in (+1, -1):
            rr = rc_factor(Ebeam=EB, Q2=Q2, W2=W2, t=TN, phi=np.radians(75.0),
                           sfmodel=sf, h=h, mx2_cut=vc + mU2)
            v[h] = (rr["sigma_born"], rr["sigma_obs"], rr["eta"])
        x = (v[1][0] - v[-1][0]) / (v[1][0] + v[-1][0])
        y = (v[1][1] - v[-1][1]) / (v[1][1] + v[-1][1])
        if soft is None:
            soft = y / x
        print(f"{vc:9.0e} {v[1][2]:11.5f} {v[-1][2]:11.5f} {y/x:13.6f}")
    ok5 = abs(soft - 1.0) < 1e-4
    print(f"   A_obs/A_Born при v_cut=1e-5: {soft:.6f} "
          f"-> {'PASS' if ok5 else 'FAIL'}")

    n_ok = sum((ok1, ok2, ok3, ok4, ok5))
    print(f"\nVERDICT: {n_ok}/5 PASS")
    return 0 if n_ok == 5 else 1


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.15))
