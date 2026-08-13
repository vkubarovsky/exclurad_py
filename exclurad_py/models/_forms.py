"""
Shared functional forms for the pi0 / eta cross-section vintages.

Two functional families are in use:

  * `dvmp2013_form`  -- the 2013/2014 EXCLURAD built-in parameterization,
    P(1..14): a plain t- and xB-factorized fit.  Used by the 2014 vintage
    and by the eta analysis-note Appendix B ("Table B.1") production model,
    which is the SAME functional form with a different parameter column.

  * `vpk2021_form`   -- the March-2021 GT/transversity fit (H_T, E_T,
    HTEbar convolutions), P(1..12).

Both are ports of Fortran in the user's own trees; the port is verified
numerically against the Fortran by
`mesonrc/validation/verify_ports.py` (see PORTS.md for achieved digits).

IMPORTANT conventions carried over verbatim from the Fortran:
  * `t` arguments are NEGATIVE (spacelike); the forms use T = -t > 0.
  * `tminq` returns a quantity T0 such that the fits use (T - T0); it is
    computed with the *detected proton* in the final state and the meson
    mass of the channel, exactly as dvmpx.F does.
  * EXCLURAD's `dvmpw` wrapper multiplies sigma_LT (and sigma_LT') by 2
    for the 2021 model.  That factor belongs to the *wrapper*, not to the
    fit, and is applied by the model class, not here.
"""

import numpy as np

MP_FIT = 0.93827          # proton mass as hardcoded in dvmpx.F
_ALPHA_FIT = 0.00729927   # alpha as hardcoded in dvmpx.F
_HC2 = 389379.36          # GeV^2 -> nb, as hardcoded in dvmpx.F


def tminq(Q2, xB, m_meson):
    """Port of dvmpx.F tminq(Q2,X) with MPI0 = m_meson.

    Note this is *not* the usual |t_min|: it is the Fortran's T0, computed
    with the recoil PROTON momenta in the CM.  Reproduced exactly, including
    the use of MP_FIT rather than the PDG proton mass.
    """
    Q2 = np.asarray(Q2, dtype=float)
    xB = np.asarray(xB, dtype=float)
    Mp = 0.9382723                       # value used inside tminq itself
    with np.errstate(divide="ignore", invalid="ignore"):
        W2 = Q2 * (1.0 / xB - 1.0) + Mp ** 2
    W = np.sqrt(np.maximum(W2, 1e-30))
    E1cm = (W2 + Q2 + Mp ** 2) / (2.0 * W)
    P1cm = np.sqrt(np.maximum(E1cm ** 2 - Mp ** 2, 0.0))
    E3cm = (W2 - m_meson ** 2 + Mp ** 2) / (2.0 * W)
    P3cm = np.sqrt(np.maximum(E3cm ** 2 - Mp ** 2, 0.0))
    T0 = -((Q2 + m_meson ** 2) ** 2 / 4.0 / W2 - (P1cm - P3cm) ** 2)
    bad = (xB <= 0) | (xB >= 1) | (W < Mp + m_meson)
    return np.where(bad, 0.0, T0)


def phase(Q2, xB):
    """Port of dvmpx.F PHASE(Q2,X)."""
    Q2 = np.asarray(Q2, dtype=float)
    xB = np.asarray(xB, dtype=float)
    MP = 0.938272
    with np.errstate(divide="ignore", invalid="ignore"):
        W2 = Q2 * (1.0 / xB - 1.0) + MP ** 2
    lam = (W2 ** 2 + Q2 ** 2 + MP ** 4 + 2 * W2 * Q2
           - 2 * W2 * MP ** 2 + 2 * Q2 * MP ** 2)
    out = 16.0 * np.pi * (W2 - MP ** 2) * np.sqrt(np.maximum(lam, 0.0))
    return np.where((xB >= 1) | (lam <= 0), np.nan, out)


# --------------------------------------------------------------------------- #
def dvmp2013_form(t, xB, Q2, P, m_meson):
    """XSIGMA_T/L/TT/LT of dvmpx.F (2013/2014 family), P = P(1..14).

    Returns (sT, sL, sTT, sLT) in nb/GeV^2.  `t` negative.
    """
    t = np.asarray(t, dtype=float)
    xB = np.asarray(xB, dtype=float)
    Q2 = np.asarray(Q2, dtype=float)
    T = -t
    T0 = tminq(Q2, xB, m_meson)
    dT = np.maximum(T - T0, 0.0)
    slope = 2.0 * 1.1 * np.log(np.maximum(xB, 1e-300))
    common = (xB ** P[10] * (1.0 - xB) ** P[11]
              / (Q2 + P[12]) ** P[13])
    sT = (P[0] + P[1] * np.sqrt(dT)) * np.exp(slope * T * P[2]) * common
    sL = Q2 * (P[3] + P[4] * dT) * np.exp(slope * T * P[5]) * common
    sTT = P[6] * dT * np.exp(slope * T * P[7]) * common
    sLT = P[8] * np.sqrt(dT) * np.exp(slope * T * P[9]) * common
    return sT, sL, sTT, sLT


def vpk2021_form(t, xB, Q2, P, m_meson):
    """H_T / E_T / HTEbar (March-2021) form, P = P(1..12).

    Returns (sT, sL, sTT, sLT) in nb/GeV^2 with sL = 0 (the 2021 fit has no
    longitudinal piece).  The EXCLURAD dvmpw factor 2 on sigma_LT is NOT
    applied here.
    """
    t = np.asarray(t, dtype=float)
    xB = np.asarray(xB, dtype=float)
    Q2 = np.asarray(Q2, dtype=float)
    T = -t
    dT = np.maximum(T - tminq(Q2, xB, m_meson), 0.0)
    lx = np.log(np.maximum(xB, 1e-300)) - np.log(0.15)

    HT = P[0] * np.exp(-(P[1] + P[2] * lx) * T) * Q2 ** (P[3] / 2.0)
    ET = P[5] * np.exp(-(P[6] + P[7] * lx) * T) * Q2 ** (P[8] / 2.0)
    HTEB = P[10] * np.exp(-P[11] * T)

    ksi = xB / (2.0 - xB) * (1.0 + MP_FIT ** 2 / Q2)
    ph = phase(Q2, xB)

    pref = 4.0 * np.pi * _ALPHA_FIT / 2.0 / ph / Q2 ** 2
    sT = pref * ((1.0 - ksi ** 2) * HT ** 2
                 + dT / (8.0 * MP_FIT ** 2) * ET ** 2) * _HC2
    sTT = -pref * dT / (8.0 * MP_FIT ** 2) * ET ** 2 * _HC2
    sLT = (4.0 * np.pi * _ALPHA_FIT / np.sqrt(2.0) / ph / Q2 ** 1.5
           * ksi * np.sqrt(np.maximum(1.0 - ksi ** 2, 0.0))
           * np.sqrt(dT) / (2.0 * MP_FIT) * HTEB ** 2 * _HC2)

    bad = (ksi >= 1.0) | ~np.isfinite(ph)
    z = np.zeros_like(np.asarray(sT, dtype=float))
    sT = np.where(bad, 0.0, sT)
    sTT = np.where(bad, 0.0, sTT)
    sLT = np.where(bad, 0.0, sLT)
    return sT, z, sTT, sLT


# --------------------------------------------------------------------------- #
# Parameter columns, transcribed from the Fortran sources.
# --------------------------------------------------------------------------- #

# pi0eta.par of exclurad-2014 / eta-rc-2026 model2013note, column k=1 (pi0).
P2013_PI0 = [846.416, 0.0, 0.450431,
             1396.68, 0.0, 0.69104,
             -5362.34, 1.01158,
             673.505, 1.00505,
             0.0, 0.0,
             1.08767, 1.69202]

# pi0eta.par of eta-rc-2026/model2013note, column k=2 (eta) == analysis
# note CLAS-2016-101 Appendix B Table B.1, the eta PRODUCTION model.
P_ETA_TABLE_B1 = [204.99, 0.0, 0.5,
                  1392.1, 0.0, 0.5,
                  -3083.0, 1.101,
                  271.6, 0.5,
                  0.0, 0.0,
                  1.0, 2.0]

# pi0eta.par of the eta-rc-2026 repo default (a LATER eta refit, kept for
# reference -- NOT the published production model).
P_ETA_2013_REFIT = [42.56526, 0.000000, 0.6235287e-01,
                    1743.521, 0.000000, 0.5497253,
                    -3453.083, 1.102218,
                    363.1541, 0.6164739,
                    0.000000, 0.000000,
                    1.012193, 2.165608]

# dvmpx.F XSINIT March-2021 P(12,2): column 1 = pi0, column 2 = eta.
P2021_PI0 = [6.2054, 0.8020, -0.1066, 1.8364, 0.00,
             94.6474, 3.4426, -1.9769, -0.1324, 0.0000, 17.0001, 2.1228]
P2021_ETA = [6.8774, 1.7856, -1.3181, 0.7827, 0.00,
             17.7821, 1.1451, 0.0053, 1.5816, 0.0000, 21.9122, 4.0069]
