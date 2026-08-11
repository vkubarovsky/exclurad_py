"""
Finite (IR-subtracted) hard-photon contribution sigma_F.

    sigma_F = alpha^3 / (512 pi^4 lam_S lam_q^{1/2}) *
              int_0^{v_m} dv int dOmega_k (v/z^2) *
              [ LR:H(qt; sigma_i(W2t,Q2t,tt)) / Q2t^2
                - 4 F_IR L0:H0 / Q2^2 ]                      (BS subtraction)

with z = (k.Lam)/omega = Lam0 - vec(Lam).n,  omega = v/(2z),
qt = q - k,  Q2t = Q2 + 2 k.q,  W2t = W2 - 2 omega W (CM),  tt = t - 2 k.(q-pv).

Photon angular integral: two-channel decomposition (channels centered on the
incident / scattered lepton directions, weights w_c = kappa_c^-2 / sum),
with substitution x = ln(a_c - cos theta_c) flattening the collinear peaks;
the collinear core (m^2/kappa^2 region) gets its own Gauss-Legendre panel.

Units: SF model in nb/GeV^2 -> sigma_F in nb/GeV^6.
"""

import numpy as np
from scipy.special import roots_legendre
from .constants import ALPHA, NB_TO_GEV2, GEV2_TO_NB
from .spinors import dot4, rad_L, contractLH, F_IR
from .hadronic import hadronic_tensor
from .kinematics import EventKin
from .born import born_tensors, born_H, _sf_to_gev


def _orthobasis(zaxis):
    ref = np.zeros(3)
    ref[np.argmin(np.abs(zaxis))] = 1.0
    x = np.cross(ref, zaxis)
    x /= np.linalg.norm(x)
    y = np.cross(zaxis, x)
    return x, y


def _channel_nodes(kvec, m, n_core, n_tail, n_phi):
    """Angular nodes for one channel: returns nhat (N,3), weights (N,),
    and the stable own-lepton kappa_hat = E - |k| cos(theta_c) (N,)."""
    ak = np.linalg.norm(kvec)
    E = np.sqrt(ak * ak + m * m)
    zc = kvec / ak
    xc, yc = _orthobasis(zc)
    am1 = m * m / (ak * (E + ak))            # a - 1 (stable), a = E/|k|
    x_min = np.log(am1)
    x_max = np.log(E / ak + 1.0)
    x_split = min(x_min + 4.0, 0.5 * (x_min + x_max))

    xs, ws = [], []
    for (lo, hi, n) in ((x_min, x_split, n_core), (x_split, x_max, n_tail)):
        r, w = roots_legendre(n)
        xs.append(0.5 * (hi - lo) * r + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * w)
    x = np.concatenate(xs)
    wx = np.concatenate(ws)

    ex = np.exp(x)                           # = a - cos(theta_c)
    one_m_c = ex - am1                       # 1 - cos(theta_c), stable
    ct = 1.0 - one_m_c
    st = np.sqrt(np.maximum(one_m_c * (1.0 + ct), 0.0))
    kap_own = ak * ex                        # E - |k| cos(theta_c), stable

    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi

    nx = len(x)
    CT = np.repeat(ct, n_phi)
    ST = np.repeat(st, n_phi)
    KO = np.repeat(kap_own, n_phi)
    WW = np.repeat(wx * ex, n_phi) * (2.0 * np.pi / n_phi)   # dcos dphi
    PH = np.tile(phi, nx)

    nhat = (CT[:, None] * zc[None, :]
            + ST[:, None] * (np.cos(PH)[:, None] * xc[None, :]
                             + np.sin(PH)[:, None] * yc[None, :]))
    return nhat, WW, KO


def sigma_F(kin0, sfmodel, h, v_m,
            n_v=30, n_core=16, n_tail=24, n_phi=16, return_diag=False,
            v_lo=None, subtract=True):
    """IR-finite hard-photon integral (nb/GeV^6). kin0: Born EventKin (v=0).

    v_lo / subtract: consistency-test and generator hooks -- integrate over
    [v_lo, v_m] and optionally WITHOUT the Bardin-Shumeiko subtraction, which
    gives the positive-definite (samplable) slicing form. Defaults reproduce
    the BS-subtracted integral used by rc_factor."""
    m, M, mV = kin0.m, kin0.M, kin0.mV
    Q2, W2, W, t = kin0.Q2, kin0.W2, kin0.W, kin0.t
    k1, k2, p, q = kin0.k1, kin0.k2, kin0.p, kin0.q
    E1, E2 = k1[0], k2[0]

    L0, _ = born_tensors(kin0, h)
    H0 = born_H(kin0, sfmodel)[0]
    LH0 = np.real(contractLH(L0, H0))                 # Born contraction (GeV^-4 SF units)

    # angular node sets for the two channels (v-independent)
    channels = []
    for ci, kv in enumerate((k1, k2)):
        nhat, wang, kap_own = _channel_nodes(kv[1:], m, n_core, n_tail, n_phi)
        kh1 = np.maximum(E1 - nhat @ k1[1:], 1e-30)
        kh2 = np.maximum(E2 - nhat @ k2[1:], 1e-30)
        if ci == 0:
            kh1 = kap_own
        else:
            kh2 = kap_own
        wchan = (1.0 / (kh1 if ci == 0 else kh2) ** 2) / \
                (1.0 / kh1 ** 2 + 1.0 / kh2 ** 2)
        channels.append(dict(nhat=nhat, wang=wang, kh1=kh1, kh2=kh2,
                             wchan=wchan))

    # v grid: Gauss-Legendre in ln v
    r, wv = roots_legendre(n_v)
    lo, hi = np.log(v_m) - 18.0, np.log(v_m)
    if v_lo is not None:
        lo = np.log(v_lo)
    lv = 0.5 * (hi - lo) * r + 0.5 * (hi + lo)
    vs = np.exp(lv)
    wvs = 0.5 * (hi - lo) * wv * vs                   # dv = v dln(v)

    total = 0.0
    n_bad = 0
    for v, wv_i in zip(vs, wvs):
        kinv = EventKin(kin0.Ebeam, Q2, W2, t, kin0.phi, mV, v=v, M=M, m=m,
                        mU=kin0.mU)
        pv, Lam = kinv.pv, kinv.Lam
        for chd in channels:
            nhat, wang = chd['nhat'], chd['wang']
            z = Lam[0] - nhat @ Lam[1:]
            om = v / (2.0 * z)
            k4 = np.concatenate([om[:, None], om[:, None] * nhat], axis=1)

            kap1 = 2.0 * om * chd['kh1']
            kap2 = 2.0 * om * chd['kh2']

            # shifted kinematics
            qt = q[None, :] - k4
            Q2t = Q2 + 2.0 * dot4(k4, q[None, :])
            W2t = W2 - 2.0 * om * W
            tt = t - 2.0 * dot4(k4, (q - pv)[None, :])

            good = Q2t > 1e-6
            n_bad += np.count_nonzero(~good)

            sf = _sf_to_gev(sfmodel(W2t, np.maximum(Q2t, 1e-6), tt))
            H = hadronic_tensor(qt, p[None, :] * np.ones_like(om)[:, None],
                                pv[None, :] * np.ones_like(om)[:, None],
                                W2t, sf, M=M)

            LR, _, _ = rad_L(k1, k2, m, h, k4)
            LRH = np.real(contractLH(LR, H))

            fir = F_IR(Q2, m, kap1, kap2)
            exact = np.where(good, LRH / Q2t ** 2, 0.0)
            sub = 4.0 * fir * LH0 / Q2 ** 2 if subtract else 0.0
            integrand = (v / z ** 2) * (exact - sub)
            total += wv_i * np.sum(wang * chd['wchan'] * integrand)

    pref = ALPHA ** 3 / (512.0 * np.pi ** 4 * kin0.lam_S * np.sqrt(kin0.lam_q))
    out = pref * total * GEV2_TO_NB
    if return_diag:
        return out, dict(n_bad=int(n_bad))
    return out
