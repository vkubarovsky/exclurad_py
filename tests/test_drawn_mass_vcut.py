"""The exponentiated tail must reference THIS event's mass, not the pole.

A channel with a width draws m_U per event (events.py, _draw_mass).
rc_factor forms its own v from the mU it is handed, as

    v_m = mx2_cut - mU**2                            (core/rc.py)

so a caller asking for sigma_obs at v = x must pass mx2_cut = x + m_drawn^2.
Asking with the POLE mass instead leaves v_m = x + m_pole^2 - m_drawn^2,
negative for every mass drawn above the pole: rc_factor raises "no phase
space for photon emission", the generator's blanket `except` swallowed it,
and the event was still written -- as a soft, radiation-free one.  That is
where half the hard photons of the 2026-10-21 and 2026-11-11 phi samples
went (drop_except 48.6 %, hard fraction 14.5 % against the true 43 %).

Both halves are checked: that the pole spelling really does raise on a mass
above the pole, so this test would have caught the bug, and that the drawn
spelling is well behaved on either side of it.  The last case records why
only phi was hit -- pi0 and eta draw no mass at all, and J/psi's width is
below DELTA_WIDTH, so there m_drawn == m_pole and the two spellings agreed.
"""
import sys

import numpy as np

from exclurad_py import models
from exclurad_py.core.rc import rc_factor
from exclurad_py.generator.breit_wigner import WIDTH, DELTA_WIDTH, sample_mass

# A point inside the RG-A Fall-2018 phi box (cards/rga_rad.txt).  t is the
# model's own t_nucl default only to keep rc_factor's fixed-t_N warning
# quiet -- a vector-meson model ignores t_nucl, and nothing here depends on
# the value: the v bookkeeping under test is pure kinematics.
KIN = dict(Ebeam=10.6, Q2=2.0, W2=9.0, t=-0.30, phi=0.7, h=+1)
VMN = 1.0e-4
XS = (VMN, 1.0e-3, 1.0e-2, 0.04)


def sobs(sf, m_u, mx2_cut):
    return rc_factor(sfmodel=sf, mx2_cut=mx2_cut, mV=float(sf.mV),
                     mU=float(m_u), t_kind="target", **KIN)


def pole_spelling_raises():
    """Above the pole the pole-referenced cut has no phase space at all."""
    sf = models.get("phi.diffrad_vm_pdet")
    m_pole = float(sf.mU)
    m_high = m_pole + 3.0 * WIDTH["phi"]      # inside the sampling window
    try:
        sobs(sf, m_high, VMN + m_pole ** 2)
    except ValueError as exc:
        ok = "no phase space" in str(exc)
        print(f"  pole spelling, m = m0 + 3*Gamma -> ValueError"
              f"           {'ok' if ok else 'FAIL (wrong message)'}")
        return 0 if ok else 1
    print("  pole spelling, m = m0 + 3*Gamma -> no error       FAIL")
    return 1


def drawn_spelling_works():
    """mx2_cut = x + m_drawn^2 puts rc_factor at exactly v = x."""
    sf = models.get("phi.diffrad_vm_pdet")
    fail = 0
    for n_gamma in (-5.0, -1.0, 0.0, +1.0, +5.0):
        m_drawn = float(sf.mU) + n_gamma * WIDTH["phi"]
        bad = [x for x in XS
               if not np.isfinite(sobs(sf, m_drawn, x + m_drawn ** 2)
                                  ["sigma_obs"]) > 0.0]
        fail += bool(bad)
        print(f"  drawn spelling, m = m0 {n_gamma:+.0f}*Gamma -> "
              f"{len(XS) - len(bad)}/{len(XS)} v values finite and positive"
              f"   {'ok' if not bad else 'FAIL'}")
    return fail


def only_wide_channels_differ():
    """pi0/eta draw no mass, J/psi is a delta: there the spellings agree."""
    fail = 0
    if not WIDTH["jpsi"] < DELTA_WIDTH < WIDTH["phi"]:
        print("  jpsi below DELTA_WIDTH, phi above              FAIL")
        fail += 1
    else:
        print("  jpsi below DELTA_WIDTH, phi above              ok")
    rng = np.random.default_rng(0)
    drawn = {sample_mass("jpsi", 3.0969, 5.0, rng) for _ in range(50)}
    if drawn == {3.0969}:
        print("  jpsi mass is a delta at the pole               ok")
    else:
        print("  jpsi mass is a delta at the pole               FAIL")
        fail += 1
    return fail


def main():
    print("drawn mass vs pole mass in the exponentiated tail")
    fail = pole_spelling_raises()
    fail += drawn_spelling_works()
    fail += only_wide_channels_differ()
    print(f"FAILURES: {fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
