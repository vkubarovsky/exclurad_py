"""
Exact leptonic tensors from numerical Dirac algebra.

Conventions:
  * metric (+,-,-,-), all four-vectors contravariant, index order (E, x, y, z)
  * Dirac representation
  * helicity spinors u(p,lambda), lambda = +-1  (helicity +-1/2),
    normalized  ubar u = 2m,  sum_lam u ubar = pslash + m
  * Born current      j0^mu       = ubar(k2,l2) gamma^mu u(k1,h)
  * radiative current jR^mu       = ubar(k2,l2) [ gamma^mu (k1s-ks+m)/(-2k.k1) eps*_s
                                     + eps*_s (k2s+ks+m)/(2k.k2) gamma^mu ] u(k1,h)
    (real photon k, physical transverse polarizations; the coupling e is
     factored out of the currents).

  L0^{mu nu}(h) = sum_{l2} j0^mu j0^nu*
  LR^{mu nu}(h) = sum_{l2,pol} jR^mu jR^nu*
"""

import numpy as np

SIGN = np.array([1.0, -1.0, -1.0, -1.0])

_s0 = np.eye(2, dtype=complex)
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)

GAMMA = np.zeros((4, 4, 4), dtype=complex)          # GAMMA[mu] = gamma^mu
GAMMA[0][:2, :2] = _s0
GAMMA[0][2:, 2:] = -_s0
for _i, _s in enumerate((_sx, _sy, _sz), start=1):
    GAMMA[_i][:2, 2:] = _s
    GAMMA[_i][2:, :2] = -_s
ID4 = np.eye(4, dtype=complex)


def dot4(a, b):
    """Minkowski dot product, broadcasting over leading axes."""
    return np.einsum('...m,m,...m->...', a, SIGN, b)


def fslash(p):
    """p^mu -> pslash = p_mu gamma^mu ; p: (...,4) -> (...,4,4)."""
    return np.einsum('...m,m,mij->...ij', p, SIGN, GAMMA)


def uspinor(p, m, lam):
    """Helicity spinor u(p,lam), p: (4,) -> (4,) complex."""
    E, px, py, pz = p
    pp = np.sqrt(px * px + py * py + pz * pz)
    if pp < 1e-300:
        ct, phi = 1.0, 0.0
    else:
        ct = np.clip(pz / pp, -1.0, 1.0)
        phi = np.arctan2(py, px)
    c2 = np.sqrt(0.5 * (1.0 + ct))
    s2 = np.sqrt(0.5 * (1.0 - ct))
    eip = np.exp(1j * phi)
    if lam == +1:
        chi = np.array([c2, s2 * eip], dtype=complex)
    else:
        chi = np.array([-s2 * np.conj(eip), c2], dtype=complex)
    a = np.sqrt(max(E + m, 0.0))
    b = lam * np.sqrt(max(E - m, 0.0))
    return np.concatenate((a * chi, b * chi))


def _ubar(u):
    return np.conj(u) @ GAMMA[0]


def born_current(k1, k2, m, h):
    """j0^mu for both final helicities: returns (2,4) complex."""
    u1 = uspinor(k1, m, h)
    out = np.zeros((2, 4), dtype=complex)
    for i, l2 in enumerate((+1, -1)):
        ub2 = _ubar(uspinor(k2, m, l2))
        for mu in range(4):
            out[i, mu] = ub2 @ GAMMA[mu] @ u1
    return out


def born_L(k1, k2, m, h):
    """Born leptonic tensor L0^{mu nu}(h): (4,4) complex."""
    j = born_current(k1, k2, m, h)
    return np.einsum('am,an->mn', j, np.conj(j))


def photon_pols(k):
    """Two real transverse polarization 4-vectors for photons k: (N,4).
    Returns (N,2,4)."""
    n = k[:, 1:] / np.linalg.norm(k[:, 1:], axis=1, keepdims=True)
    # reference axis least aligned with n
    ref = np.zeros_like(n)
    idx = np.argmin(np.abs(n), axis=1)
    ref[np.arange(len(n)), idx] = 1.0
    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    eps = np.zeros((len(k), 2, 4))
    eps[:, 0, 1:] = e1
    eps[:, 1, 1:] = e2
    return eps


def rad_L(k1, k2, m, h, k):
    """Radiative leptonic tensor LR^{mu nu}(h) for a batch of photon momenta.

    k1,k2: (4,) beam / scattered lepton;  k: (N,4) photon momenta.
    Returns (N,4,4) complex, plus kappa1=2k.k1, kappa2=2k.k2 (N,).
    """
    N = k.shape[0]
    u1 = uspinor(k1, m, h)                       # (4,)
    ub2 = np.array([_ubar(uspinor(k2, m, l2)) for l2 in (+1, -1)])   # (2,4)
    UB2G = np.einsum('ai,mij->amj', ub2, GAMMA)  # (2,4mu,4)  = ubar2 gamma^mu
    GU1 = np.einsum('mij,j->mi', GAMMA, u1)      # (4mu,4)    = gamma^mu u1

    kap1 = 2.0 * dot4(k, k1[None, :])            # (N,)
    kap2 = 2.0 * dot4(k, k2[None, :])
    prop1 = (fslash(k1[None, :] - k) + m * ID4) / (-kap1)[:, None, None]
    prop2 = (fslash(k2[None, :] + k) + m * ID4) / (kap2)[:, None, None]

    eps = photon_pols(k)                         # (N,2,4) real
    L = np.zeros((N, 4, 4), dtype=complex)
    for a in range(2):
        es = fslash(eps[:, a])                   # (N,4,4)
        v1 = np.einsum('nij,j->ni', es, u1)      # eps* u1
        v2 = np.einsum('nij,nj->ni', prop1, v1)  # prop1 eps* u1
        for il2 in range(2):
            A = np.einsum('mj,nj->nm', UB2G[il2], v2)          # ubar2 g^mu prop1 es u1
            w1 = np.einsum('i,nij->nj', ub2[il2], es)          # ubar2 es
            w2 = np.einsum('ni,nij->nj', w1, prop2)            # ubar2 es prop2
            B = np.einsum('ni,mi->nm', w2, GU1)                # ... g^mu u1
            j = A + B                                          # (N,4)
            L += j[:, :, None] * np.conj(j)[:, None, :]
    return L, kap1, kap2


def contractLH(L, H):
    """L^{mu nu} H_{mu nu} with metric lowering of H's indices; L,H (...,4,4)."""
    return np.einsum('...mn,m,n,...mn->...', L, SIGN, SIGN, H)


def F_IR(Q2, m, kap1, kap2):
    """Soft (eikonal) factor:  LR -> 4 F_IR L0  as photon momentum -> 0."""
    return (Q2 + 2.0 * m * m) / (kap1 * kap2) - m * m / kap1**2 - m * m / kap2**2
