"""
Demo structure-function models for rho0, phi, J/psi.

*** ILLUSTRATIVE ONLY ***  These are simple, smooth parameterizations with
roughly realistic magnitudes, W/Q2 slopes and t-slopes, meant to make the
code runnable and testable out of the box.  For physics analyses replace
them with your own model / fit by subclassing StructureFunctions.

Units: all sigma_i are d sigma_i/dt in nb/GeV^2; W2, Q2, t in GeV^2.

Built-in kinematic behaviour of the interference SFs: sigma_LT, sigma_LT'
vanish like sqrt(-t') and sigma_TT like (-t') as t -> t_min (as required by
angular momentum conservation), where t' = t - t_min(W2,Q2).
"""

import numpy as np
from .constants import (M_P, M_N, M_RHO, M_PHI, M_JPSI,
                        M_PI0, M_PIP, M_ETA)
from .kinematics import kallen


def tmin(W2, Q2, mV, M=M_P, mU=None):
    """t_min (least negative t) for gamma*(Q2) p -> h B at given W2
    (h: detected meson of mass mV, B: recoil baryon of mass mU)."""
    if mU is None:
        mU = M
    W = np.sqrt(W2)
    E_g = (W2 - Q2 - M * M) / (2.0 * W)
    E_v = (W2 + mV * mV - mU * mU) / (2.0 * W)
    q = np.sqrt(np.maximum(kallen(W2, -Q2, M * M), 0.0)) / (2.0 * W)
    pv = np.sqrt(np.maximum(E_v * E_v - mV * mV, 0.0))
    return -Q2 + mV * mV - 2.0 * (E_g * E_v - q * pv)


class DemoSF:
    """sigma_T = sig0 (W2/W02)^delta (mV^2/(mV^2+Q2))^n exp(b t')
       R = sigma_L/sigma_T = r0 (Q2/mV^2)^p / (1 + r1 Q2/mV^2)
       sigma_TT  = -cTT sigma_T   * ft2
       sigma_LT  =  cLT sqrt(sigma_T sigma_L) * ft1
       sigma_LT' =  cLTp sqrt(sigma_T sigma_L) * ft1
       ft1 = sqrt(-t'/GeV^2)/(1 + (-t'/GeV^2)),  ft2 = (-t'/GeV^2)/(1+(-t'/GeV^2))^2
    """

    def __init__(self, mV, sig0, W02, delta, n, b,
                 r0, p, r1, cTT=0.15, cLT=0.2, cLTp=0.1, M=M_P, mU=None):
        self.mV, self.M = mV, M
        self.mU = mU                     # undetected recoil baryon (None=proton)
        self.sig0, self.W02, self.delta, self.n, self.b = sig0, W02, delta, n, b
        self.r0, self.p, self.r1 = r0, p, r1
        self.cTT, self.cLT, self.cLTp = cTT, cLT, cLTp

    def __call__(self, W2, Q2, t):
        W2 = np.asarray(W2, dtype=float)
        Q2 = np.asarray(Q2, dtype=float)
        t = np.asarray(t, dtype=float)
        mV2 = self.mV ** 2

        t0 = tmin(W2, Q2, self.mV, self.M, self.mU)
        tp = np.minimum(t - t0, 0.0)              # t' <= 0
        prop = (mV2 / (mV2 + Q2)) ** self.n
        sT = self.sig0 * (W2 / self.W02) ** self.delta * prop * np.exp(self.b * tp)
        R = self.r0 * (Q2 / mV2) ** self.p / (1.0 + self.r1 * Q2 / mV2)
        sL = R * sT

        mt = -tp
        ft1 = np.sqrt(mt) / (1.0 + mt)
        ft2 = mt / (1.0 + mt) ** 2
        sqTL = np.sqrt(sT * sL)
        return {'sT': sT, 'sL': sL,
                'sTT': -self.cTT * sT * ft2,
                'sLT': self.cLT * sqTL * ft1,
                'sLTp': self.cLTp * sqTL * ft1}


def demo_rho():
    return DemoSF(M_RHO, sig0=8.0e3, W02=10.0, delta=0.22, n=2.2, b=7.0,
                  r0=0.6, p=0.61, r1=0.14)


def demo_phi():
    return DemoSF(M_PHI, sig0=1.0e3, W02=10.0, delta=0.28, n=2.2, b=5.5,
                  r0=0.5, p=0.6, r1=0.1)


def demo_jpsi():
    return DemoSF(M_JPSI, sig0=8.0, W02=100.0, delta=0.8, n=2.4, b=4.5,
                  r0=0.3, p=0.7, r1=0.05)


# ---------------------------------------------------------------------------
# pseudoscalar mesons (same 5-SF formalism; EXCLURAD's original case).
# Demo parameterizations only: non-diffractive falling W-dependence,
# large sigma_TT (as observed for pi0), recoil baryon mU (neutron for pi+).
# ---------------------------------------------------------------------------
def demo_pi0():
    """gamma* p -> pi0 p (recoil proton undetected)."""
    return DemoSF(M_PI0, sig0=350.0, W02=5.0, delta=-1.1, n=1.6, b=1.8,
                  r0=0.12, p=0.7, r1=0.10,
                  cTT=0.55, cLT=0.15, cLTp=0.10)


def demo_pip():
    """gamma* p -> pi+ n (recoil NEUTRON undetected: mU = M_N)."""
    return DemoSF(M_PIP, sig0=600.0, W02=5.0, delta=-1.0, n=1.4, b=3.0,
                  r0=0.8, p=0.8, r1=0.25,
                  cTT=0.25, cLT=0.20, cLTp=0.15, mU=M_N)


def demo_eta():
    """gamma* p -> eta p (recoil proton undetected)."""
    return DemoSF(M_ETA, sig0=120.0, W02=5.0, delta=-1.1, n=1.6, b=1.5,
                  r0=0.10, p=0.7, r1=0.10,
                  cTT=0.35, cLT=0.12, cLTp=0.08)
