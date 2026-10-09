#!/usr/bin/env python3
"""The acceptance block must read the meson by ROLE, not by slot name.

Under baryon_detected the slots are swapped: ev["meson"] holds the DETECTED
NUCLEON.  Reading it directly decayed the proton and applied the proton
acceptance window to the meson.

Run:  PYTHONPATH=<repo> python3 tests/test_acceptance_slot.py
"""
import numpy as np

from exclurad_py.generator.decay import (meson_slot, DECAY, decay_meson,
                                         in_acceptance)
from exclurad_py.topology import baryon_detected, meson_detected

M_P, M_JPSI, M_PI0 = 0.93827208816, 3.0969, 0.1349768


def _slots(topo, ch):
    m = meson_slot(topo, ch)
    return m, ("recoil" if m == "meson" else "meson")


def main():
    bad = 0

    # 1. the slot map itself
    cases = [
        (meson_detected(M_JPSI, meson="J/psi"), "jpsi", "meson", "recoil"),
        (baryon_detected(M_JPSI, meson="J/psi"), "jpsi", "recoil", "meson"),
        (baryon_detected(M_PI0, meson="pi0"), "pi0", "recoil", "meson"),
        (baryon_detected(M_PI0, meson="pi0", m_recoil=0.93956542,
                         recoil="n"), "pi0n", "recoil", "meson"),
    ]
    for topo, ch, want_m, want_b in cases:
        m, b = _slots(topo, ch)
        ok = (m == want_m and b == want_b)
        bad += not ok
        print("%-16s %-5s meson->%-7s baryon->%-7s  %s"
              % (topo.name, ch, m, b, "ok" if ok else "FAIL"))

    # 2. the consequence: decaying the wrong slot gives daughters that do not
    #    reconstruct the meson mass.
    topo = baryon_detected(M_JPSI, meson="J/psi")
    mslot, bslot = _slots(topo, "jpsi")
    # chosen to sit inside the jpsi card's windows where they should:
    #   electron (0.1,2.5, 2,20 deg)  proton (0.4,4.5, 5,30)  daughter (1,9, 5,40)
    PROTON = np.array([1.374, 0.21, 0.00, 1.00])     # |p| 1.02, theta 11.9 deg
    MESON = np.array([6.49, 0.60, 0.00, 5.70])       # |p| 5.73 -> ABOVE the
                                                      # proton window's 4.5
    ev = {"meson": PROTON,            # the PROTON slot under baryon_detected
          "recoil": MESON,            # the MESON slot under baryon_detected
          "scattered": np.array([2.03, 0.30, 0.00, 2.00])}   # |p| 2.02, 8.5 deg
    rng = np.random.default_rng(1)
    a, b, _ = decay_meson("jpsi", ev[mslot], rng, None)
    s = np.asarray(a) + np.asarray(b)
    m_rec = np.sqrt(max(s[0] ** 2 - float(np.dot(s[1:], s[1:])), 0.0))
    ok = abs(m_rec - np.sqrt(max(ev[mslot][0] ** 2
             - float(np.dot(ev[mslot][1:], ev[mslot][1:])), 0.0))) < 1e-6
    bad += not ok
    print("decay of the role-correct slot reconstructs its mass: %s (%.4f)"
          % ("ok" if ok else "FAIL", m_rec))

    # 3. in_acceptance's third argument is the BARYON: a proton-like vector
    #    passes the jpsi proton window, a 4 GeV meson-like one does not.
    p_ok = in_acceptance("jpsi", ev["scattered"], ev[bslot])[0]   # baryon
    p_no = in_acceptance("jpsi", ev["scattered"], ev[mslot])[0]   # meson
    ok = bool(p_ok) and not bool(p_no)
    bad += not ok
    print("proton window accepts the baryon and rejects the meson: %s"
          % ("ok" if ok else "FAIL"))

    print("FAILURES:", bad)
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
