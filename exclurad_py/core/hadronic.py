"""
Hadronic tensor for exclusive vector meson production gamma*(qt) p -> V p'
parameterized by the five structure functions

    sigma_T, sigma_L, sigma_TT, sigma_LT, sigma_LT'   (each = d sigma_i / dt),

defined as Hand-convention virtual-photoproduction cross sections:

    d sigma_lam,lam' / dt = e^2 (eps_lam . H . eps_lam') / (16 pi (W2 - M^2)^2)

so that the Born ep cross section takes the canonical form

    d sigma / (dQ2 dW2 dt dphi) = Gamma_v/(2pi) [ sigma_T + eps sigma_L
        + eps cos(2phi) sigma_TT + sqrt(2 eps(1+eps)) cos(phi) sigma_LT
        + h sqrt(2 eps(1-eps)) sin(phi) sigma_LT' ].

The tensor (with X = transverse basis vector in the hadron plane,
Y = normal to the hadron plane, L = longitudinal polarization vector):

    H^{mu nu} = (kappa/2) [ sT (X X + Y Y) + sL L L + sTT (X X - Y Y)
                + nu_LT sLT (L X + X L) + i nu_LTp sLTp (L X - X L) ],
    kappa = 8 (W2 - M^2)^2 / alpha .

nu_LT, nu_LTp are pure numbers fixed once and for all by the Born closure
test (test suite verifies them to machine precision).
"""

import numpy as np
from .constants import ALPHA, M_P
from .spinors import dot4

# Fixed once and for all by the Born closure test (see tests/run_checks.py):
# with these values the contraction reproduces the canonical decomposition
# ... + sqrt(2 eps(1+eps)) cos(phi) sigma_LT + h sqrt(2 eps(1-eps)) sin(phi)
# sigma_LT'  with + signs in the geometry defined in kinematics.py.
NU_LT = -1.0
NU_LTP = -1.0


# --------------------------------------------------------------------------- #
#  photon polarization basis attached to the (possibly shifted) hadron vertex
# --------------------------------------------------------------------------- #
def _boost_matrix(beta):
    """Pure boost with velocity beta (N,3) -> (N,4,4) (active on 4-vectors)."""
    b2 = np.einsum('ni,ni->n', beta, beta)
    g = 1.0 / np.sqrt(1.0 - b2)
    N = len(beta)
    Lam = np.zeros((N, 4, 4))
    Lam[:, 0, 0] = g
    Lam[:, 0, 1:] = -g[:, None] * beta
    Lam[:, 1:, 0] = -g[:, None] * beta
    fac = np.where(b2 > 1e-30, (g - 1.0) / np.where(b2 > 1e-30, b2, 1.0), 0.0)
    Lam[:, 1:, 1:] = np.eye(3)[None] + fac[:, None, None] * \
        np.einsum('ni,nj->nij', beta, beta)
    return Lam


def photon_basis(qt, p, pv):
    """Basis 4-vectors (X, Y, L) for a batch of shifted kinematics.

    qt : (N,4) virtual photon at the hadronic vertex (q - k)
    p  : (N,4) target
    pv : (N,4) vector meson
    Returns X, Y, L each (N,4), built in the (p+qt) CM frame with z || qt,
    x along the transverse part of pv, y = z cross x, then boosted back.
    """
    P = p + qt
    beta = P[:, 1:] / P[:, [0]]
    B = _boost_matrix(beta)          # to CM
    Binv = _boost_matrix(-beta)      # back

    qt_c = np.einsum('nij,nj->ni', B, qt)
    pv_c = np.einsum('nij,nj->ni', B, pv)

    Q2t = -dot4(qt, qt)
    zhat = qt_c[:, 1:] / np.linalg.norm(qt_c[:, 1:], axis=1, keepdims=True)
    hperp = pv_c[:, 1:] - np.einsum('ni,ni->n', pv_c[:, 1:], zhat)[:, None] * zhat
    hn = np.linalg.norm(hperp, axis=1)
    # fallback direction for degenerate (collinear) configurations:
    fb = np.cross(zhat, np.tile(np.array([0.0, 1.0, 0.0]), (len(qt), 1)))
    fbn = np.linalg.norm(fb, axis=1)
    bad = hn < 1e-12 * pv_c[:, 0]
    xhat = np.where(bad[:, None],
                    fb / np.where(fbn[:, None] > 0, fbn[:, None], 1.0),
                    hperp / np.where(hn[:, None] > 0, hn[:, None], 1.0))
    yhat = np.cross(zhat, xhat)

    N = len(qt)
    X_c = np.zeros((N, 4)); X_c[:, 1:] = xhat
    Y_c = np.zeros((N, 4)); Y_c[:, 1:] = yhat
    L_c = np.zeros((N, 4))
    aq = np.linalg.norm(qt_c[:, 1:], axis=1)
    sq = np.sqrt(Q2t)
    L_c[:, 0] = aq / sq
    L_c[:, 1:] = (qt_c[:, 0] / sq)[:, None] * zhat

    X = np.einsum('nij,nj->ni', Binv, X_c)
    Y = np.einsum('nij,nj->ni', Binv, Y_c)
    L = np.einsum('nij,nj->ni', Binv, L_c)
    return X, Y, L


def hadronic_tensor(qt, p, pv, W2t, sf, M=M_P):
    """H^{mu nu} (N,4,4) complex from structure functions.

    sf : dict with arrays 'sT','sL','sTT','sLT','sLTp' (N,), in GeV^-4
         (i.e. d sigma_i/dt in natural units) evaluated at the shifted point.
    W2t : (N,) shifted W^2 entering the Hand flux factor of the SF definition.
    """
    X, Y, L = photon_basis(qt, p, pv)
    # NOTE on flux convention: kap*(the 1/sqrt(lam_q) in born_tensors' pref)
    # normalises H with an effective covariant K = |q| photon flux.  That is
    # the diffrad_vm convention (VM models validated against it as-is).
    # HAND-defined SFs (CLAS pi0/eta fits, PRC Eq. 4) are converted at the
    # model boundary -- see models.base.ProtonDetectedModel._sfs.  Do not
    # "fix" the convention here without re-validating every VM channel.
    kap = 8.0 * (W2t - M * M) ** 2 / ALPHA

    def op(a, b):
        return a[:, :, None] * b[:, None, :]

    XX = op(X, X); YY = op(Y, Y); LL = op(L, L)
    LX = op(L, X); XL = op(X, L)

    H = (0.5 * kap)[:, None, None] * (
        sf['sT'][:, None, None] * (XX + YY)
        + sf['sL'][:, None, None] * LL
        + sf['sTT'][:, None, None] * (XX - YY)
        + NU_LT * sf['sLT'][:, None, None] * (LX + XL)
        + 1j * NU_LTP * sf['sLTp'][:, None, None] * (LX - XL)
    )
    return H

# NOTE: the StructureFunctions interface and the TwoSF wrapper used to live
# here.  They now live in exclurad_py.models.base -- core/ must not import or
# know about models.
