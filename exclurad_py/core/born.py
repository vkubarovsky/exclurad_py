"""
Born cross section  d sigma / (dQ2 dW2 dt dphi)  via exact tensor contraction:

    d sigma_0 = alpha^2 / (32 pi^2 lam_S lam_q^{1/2} Q^4) * L0_{mu nu} H0^{mu nu}

(derived from  dsigma = |M|^2 dPhi_3 / F,  F = 2 lam_S^{1/2},
 dPhi_3 -> dQ2 dW2 dt dphi / (2^9 pi^5 lam_S^{1/2} lam_q^{1/2}) * (2 pi)_phi_e ).

Units: input SFs in nb/GeV^2, output in nb/GeV^6.
"""

import numpy as np
from .constants import ALPHA, NB_TO_GEV2, GEV2_TO_NB
from .spinors import born_L, contractLH
from .hadronic import hadronic_tensor


def _sf_to_gev(sf):
    return {k: np.atleast_1d(np.asarray(v, dtype=float)) * NB_TO_GEV2
            for k, v in sf.items()}


def born_tensors(kin, h):
    """L0 (4,4) and the scalar prefactor common to Born/radiative parts."""
    L0 = born_L(kin.k1, kin.k2, kin.m, h)
    pref = ALPHA ** 2 / (32.0 * np.pi ** 2 * kin.lam_S *
                         np.sqrt(kin.lam_q) * kin.Q2 ** 2)
    return L0, pref


def born_H(kin, sfmodel):
    """Born hadronic tensor H0 (1,4,4) from the SF model (units GeV^-4)."""
    sf = _sf_to_gev(sfmodel(kin.W2, kin.Q2, kin.t))
    qt = kin.q[None, :]
    p = kin.p[None, :]
    pv = kin.pv[None, :]
    W2t = np.array([kin.W2])
    return hadronic_tensor(qt, p, pv, W2t, sf, M=kin.M)


def born_xsec(kin, sfmodel, h):
    """d sigma_Born/(dQ2 dW2 dt dphi) in nb/GeV^6."""
    if hasattr(sfmodel, 'kf_W2Q2'):
        sfmodel.kf_W2Q2 = (float(kin.W2), float(kin.Q2))
    L0, pref = born_tensors(kin, h)
    H0 = born_H(kin, sfmodel)[0]
    val = pref * np.real(contractLH(L0, H0))
    return val * GEV2_TO_NB


def born_reference(kin, sfmodel, h):
    """Canonical-form Born cross section
       Gamma_v/(2pi) [ sT + eps sL + eps cos2phi sTT
                       + sqrt(2eps(1+eps)) cosphi sLT
                       + h sqrt(2eps(1-eps)) sinphi sLTp ]
    with Gamma_v computed from the same contraction machinery at unit SFs.
    Used by the closure test: must equal born_xsec identically once the
    conventions are consistent."""
    eps = kin.epsilon()
    sf = sfmodel(kin.W2, kin.Q2, kin.t)
    # The harmonics are in the azimuth of the DETECTED hadron, because that
    # is the plane the tensor's X and Y basis vectors are built on (see
    # hadronic.photon_basis, which takes pv).  Since 2026-08-17 kin.phi is
    # the Trento azimuth of the produced MESON, so the detected hadron sits
    # at kin.phi + kin.phi_offset -- pi for pi0/eta, 0 for the vector
    # mesons.  Leaving the offset out here does not change born_xsec, which
    # goes through the tensor, but it silently breaks THIS function, whose
    # only job is to be the independent check on it: the closure came apart
    # at the 18% level the first time sigma_LT' was switched on.
    phi_det = kin.phi + getattr(kin, 'phi_offset', 0.0)
    comb = (sf['sT'] + eps * sf['sL']
            + eps * np.cos(2 * phi_det) * sf['sTT']
            + np.sqrt(2 * eps * (1 + eps)) * np.cos(phi_det) * sf['sLT']
            + h * np.sqrt(2 * eps * (1 - eps)) * np.sin(phi_det) * sf['sLTp'])
    return comb, eps
