"""
Kinematics for exclusive meson electroproduction

    e(k1) p(p) -> e'(k2) h(pv) B(pu) [gamma(k)]

with a detected meson h of mass mV (vector: rho/phi/J/psi, or pseudoscalar:
pi0/pi+/eta, ...) and an undetected recoil baryon B of mass mU
(proton for rho/phi/J/psi/pi0/eta off the proton, neutron for pi+).

All four-vectors are built in the CM frame of the OBSERVED virtual photon
q = k1 - k2 and the target p ("hadronic CM"):

    q  along +z,  p along -z,  leptons in the x-z plane (k1_x > 0),
    the vector meson at azimuth phi (measured from the lepton plane,
    right-handed system y = z cross x -- Trento-like convention).

Observed (reconstructed) variables:
    Q2 = -q^2,  W2 = (p+q)^2,  t = (q - pv)^2,  phi.

Inelasticity of the radiative process:
    v = Lambda^2 - mU^2,  Lambda = p + q - pv    (missing-mass shift:
    v = MX^2 - mU^2, so a missing-mass cut MX^2 < MX2cut means v < MX2cut - mU^2).
At the Born level v = 0.
"""

import numpy as np
from .constants import M_P, M_E
from .spinors import dot4


def kallen(a, b, c):
    return (a - b - c) ** 2 - 4.0 * b * c


class EventKin:
    """Event kinematics at fixed (E_beam, Q2, W2, t, phi) and given v."""

    T_PHOTON = "photon"     # t = (q - p_detected)^2
    T_TARGET = "target"     # t = (p - p_detected)^2   <-- CLAS pi0/eta binning

    def __init__(self, Ebeam, Q2, W2, t, phi, mV, v=0.0, M=M_P, m=M_E, mU=None,
                 t_kind="photon"):
        self.Ebeam, self.Q2, self.W2, self.t, self.phi = Ebeam, Q2, W2, t, phi
        self.mV, self.M, self.m, self.v = mV, M, m, v
        self.mU = M if mU is None else mU        # undetected recoil baryon
        mU = self.mU

        S = 2.0 * M * Ebeam                     # S = 2 k1.p (target at rest)
        Sx = W2 + Q2 - M * M                    # 2 p.q
        X = S - Sx                              # 2 k2.p
        self.S, self.X, self.Sx = S, X, Sx
        self.y = Sx / S
        W = np.sqrt(W2)
        self.W = W

        lam_q = Sx * Sx + 4.0 * M * M * Q2      # lambda(W2,-Q2,M2)
        self.lam_q = lam_q
        self.lam_S = S * S - 4.0 * m * m * M * M

        # --- CM frame of (p + q) --------------------------------------------
        q0 = (W2 - Q2 - M * M) / (2.0 * W)
        qv = np.sqrt(lam_q) / (2.0 * W)
        Ep = (W2 + Q2 + M * M) / (2.0 * W)      # target energy in CM
        self.q = np.array([q0, 0.0, 0.0, qv])
        self.p = np.array([Ep, 0.0, 0.0, -qv])

        u1 = S - Q2                             # 2 k1.(p+q)
        u2 = X + Q2                             # 2 k2.(p+q)
        E1 = u1 / (2.0 * W)
        E2 = u2 / (2.0 * W)
        p1 = np.sqrt(max(E1 * E1 - m * m, 0.0))
        p2 = np.sqrt(max(E2 * E2 - m * m, 0.0))
        # angle of k1 wrt z (q direction):  vec(k1).vec(q) = E1 q0 + Q2/2
        c1 = (E1 * q0 + Q2 / 2.0) / (p1 * qv)
        if abs(c1) > 1.0 + 1e-9:
            raise ValueError("unphysical lepton kinematics (|cos| > 1)")
        c1 = np.clip(c1, -1.0, 1.0)
        s1 = np.sqrt(1.0 - c1 * c1)
        self.k1 = np.array([E1, p1 * s1, 0.0, p1 * c1])
        self.k2 = self.k1 - self.q
        self.k2[0] = E2                          # enforce exact energy

        # --- detected meson at inelasticity v -------------------------------
        Ev = (W2 + mV * mV - mU * mU - v) / (2.0 * W)
        if Ev < mV:
            raise ValueError("v too large: E_h < m_h")
        pv = np.sqrt(Ev * Ev - mV * mV)
        # WHICH invariant the bin holds fixed as v runs across the
        # missing-mass window.  E_detected is fixed by v alone (above); only
        # cos(theta) is free, and the conventions differ in how they fix it:
        #
        #   t_kind="photon"  t = (q - p_det)^2   -- DVMP convention
        #   t_kind="target"  t = (p - p_det)^2   -- CLAS pi0/eta: t is measured
        #                                          from the recoil PROTON, so
        #                                          fixing t fixes E_p'(lab).
        #
        # These agree at v = 0 and separate as v grows; the choice is part of
        # the bin definition, not a detail.
        self.t_kind = t_kind
        if t_kind == self.T_PHOTON:
            cv = (t + Q2 - mV * mV + 2.0 * q0 * Ev) / (2.0 * qv * pv)
        elif t_kind == self.T_TARGET:
            # t = M^2 + mV^2 - 2 (Ep Ev + qv pv cv)   (target moves along -z)
            cv = (M * M + mV * mV - t - 2.0 * Ep * Ev) / (2.0 * qv * pv)
        else:
            raise ValueError(f"unknown t_kind {t_kind!r}")
        if abs(cv) > 1.0 + 1e-9:
            raise ValueError("unphysical (t,v): |cos theta_V| > 1")
        cv = np.clip(cv, -1.0, 1.0)
        sv = np.sqrt(1.0 - cv * cv)
        self.pv = np.array([Ev,
                            pv * sv * np.cos(phi),
                            pv * sv * np.sin(phi),
                            pv * cv])
        self.Lam = self.p + self.q - self.pv     # undetected system (+ photon)
        # consistency
        assert abs(dot4(self.Lam, self.Lam) - (mU * mU + v)) < 1e-6 * max(1.0, W2)

    # ------------------------------------------------------------------ #
    @staticmethod
    def vmax_kin(Ebeam, Q2, W2, t, mV, M=M_P, m=M_E, mU=None,
                 t_kind="photon"):
        """Largest v allowed at fixed (Q2,W2,t): |cos theta_V(v)| <= 1."""
        if mU is None:
            mU = M
        W = np.sqrt(W2)
        vglob = (W - mV) ** 2 - mU * mU          # E_h >= m_h bound
        if vglob <= 0.0:
            return 0.0

        Sx = W2 + Q2 - M * M
        lam_q = Sx * Sx + 4.0 * M * M * Q2
        q0 = (W2 - Q2 - M * M) / (2.0 * W)
        qv = np.sqrt(lam_q) / (2.0 * W)
        Ep = (W2 + Q2 + M * M) / (2.0 * W)

        def cos_v(v):
            Ev = (W2 + mV * mV - mU * mU - v) / (2.0 * W)
            if Ev <= mV:
                return None
            pv = np.sqrt(Ev * Ev - mV * mV)
            if t_kind == "target":
                return (M * M + mV * mV - t - 2.0 * Ep * Ev) / (2.0 * qv * pv)
            return (t + Q2 - mV * mV + 2.0 * q0 * Ev) / (2.0 * qv * pv)

        c0 = cos_v(0.0)
        if c0 is None or abs(c0) > 1.0:
            raise ValueError("Born point itself unphysical at this t")
        lo, hi = 0.0, vglob
        # find first v where cos leaves [-1,1] (or E_V < m_V)
        nscan = 400
        vprev = 0.0
        for i in range(1, nscan + 1):
            v = vglob * i / nscan
            c = cos_v(v)
            if c is None or abs(c) > 1.0:
                lo, hi = vprev, v
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    c = cos_v(mid)
                    if c is None or abs(c) > 1.0:
                        hi = mid
                    else:
                        lo = mid
                return lo
            vprev = v
        return vglob

    # ------------------------------------------------------------------ #
    def epsilon(self):
        """Standard virtual-photon polarization parameter (massless-lepton)."""
        x = self.Q2 / self.Sx
        g2 = 4.0 * self.M ** 2 * x * x / self.Q2
        y = self.y
        num = 1.0 - y - 0.25 * g2 * y * y
        den = 1.0 - y + 0.5 * y * y + 0.25 * g2 * y * y
        return num / den
