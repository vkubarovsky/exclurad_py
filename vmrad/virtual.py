"""
Virtual and soft corrections (Bardin-Shumeiko covariant IR separation).

Master formula (EXCLURAD, Afanasev-Akushevich-Burkert-Joo, PRD 66 (2002) 074004):

    sigma_obs = sigma_0 * exp(delta_inf) * (1 + delta_VR + delta_vac) + sigma_F

with

    delta_inf = (alpha/pi) (l_m - 1) ln( v_m^2 / (S'0 X'0) )
    delta_VR  = (alpha/pi) [ 3/2 l_m - 2 - 1/2 ln^2(X'0/S'0)
                             + Li2( 1 - Q2 mU^2/(S'0 X'0) ) - pi^2/6 ]

    l_m  = ln(Q2/m^2),  mU = undetected recoil-baryon mass
    S'0  = 2 k1 . p'_Born,   X'0 = 2 k2 . p'_Born
    p'_Born = p + q - pv  (recoil baryon at Born kinematics)
    v_m  = upper limit of the inelasticity integration (missing-mass cut).

delta_vac: leptonic loops computed exactly from the one-loop integral
representation; hadronic part from the standard parameterization
(as used in POLRAD/HAPRAD-family codes).
"""

import numpy as np
from scipy.special import spence, roots_legendre
from .constants import ALPHA, M_E, M_MU, M_TAU
from .spinors import dot4


def Li2(x):
    return spence(1.0 - x)


# ---------------------------------------------------------------- vacuum pol.
_gl_x, _gl_w = roots_legendre(64)
_gl_x = 0.5 * (_gl_x + 1.0)          # map to (0,1)
_gl_w = 0.5 * _gl_w


def _pibar_lepton(Q2, ml):
    """Renormalized photon self-energy Pi(-Q2) for one lepton loop:
       Pi = (2 alpha/pi) int_0^1 dx x(1-x) ln[1 + Q2 x(1-x)/m^2]."""
    x = _gl_x
    val = np.sum(_gl_w * x * (1 - x) * np.log1p(Q2 * x * (1 - x) / ml ** 2))
    return 2.0 * ALPHA / np.pi * val


def delta_vac_lepton(Q2):
    return 2.0 * sum(_pibar_lepton(Q2, ml) for ml in (M_E, M_MU, M_TAU))


def delta_vac_hadron(Q2):
    """Standard fit (Burkhardt et al. style, as in POLRAD 2.0)."""
    if Q2 < 1.0:
        a, b, c = -1.345e-9, -2.302e-3, 4.091
    elif Q2 < 64.0:
        a, b, c = -1.512e-3, -2.822e-3, 1.218
    else:
        a, b, c = -1.1344e-3, -3.0680e-3, 0.99992
    return -2.0 * (a + b * np.log(1.0 + c * Q2))


def delta_vac(Q2):
    return delta_vac_lepton(Q2) + delta_vac_hadron(Q2)


# ---------------------------------------------------------- vertex + soft real
def delta_VR_inf(kin, v_m):
    """Returns (delta_VR, delta_inf)."""
    Q2, m, mU = kin.Q2, kin.m, kin.mU   # mU: undetected recoil-baryon mass
    lm = np.log(Q2 / m ** 2)
    pB = kin.Lam if kin.v == 0.0 else None
    if pB is None:
        raise ValueError("delta_VR_inf must be called with Born kinematics (v=0)")
    Sp = 2.0 * dot4(kin.k1, pB)
    Xp = 2.0 * dot4(kin.k2, pB)
    d_inf = ALPHA / np.pi * (lm - 1.0) * np.log(v_m ** 2 / (Sp * Xp))
    d_VR = ALPHA / np.pi * (1.5 * lm - 2.0
                            - 0.5 * np.log(Xp / Sp) ** 2
                            + Li2(1.0 - Q2 * mU ** 2 / (Sp * Xp))
                            - np.pi ** 2 / 6.0)
    return d_VR, d_inf
