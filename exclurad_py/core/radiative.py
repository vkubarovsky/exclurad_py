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


import os as _os


def quad_default():
    """Nodes for the sigma_F quadrature: (n_v, n_core, n_tail, n_phi).

    (20, 12, 16, 12) since 2026-09-24, from (30, 16, 24, 16).  Measured over
    740 points of the production box (Q2 1-6, W2 3.24-16, |t| 0.10-2.00, four
    phi), against a dense (50, 24, 40, 24) reference, with each point weighted
    by its Born cross section -- i.e. by how many events actually land there:

        grid              weighted mean |d eta|   weighted 95%   worst point
        (30,16,24,16)              0.092%              0.48%         3.7%
        (20,12,16,12)              0.132%              0.50%         5.3%

    So the change costs 0.04% on eta -- less than the old default's own
    distance from the dense reference -- and the 95th percentile does not move
    at all.  It buys 2.5x: sigma_F is 73% of generation time, and the whole
    rad mode goes from ~1.9 s to ~0.8 s per event on the mini.

    Worst-case points sit at |t| ~ 2 and Q2 ~ 6, outside amp2609 validity and
    where the cross section has fallen by a factor 20; that is why the plain
    maximum is much larger than anything the sample carries.

    RC_QUAD=n_v,n_core,n_tail,n_phi overrides, for scans and for reproducing
    an older sample.  The value used is stamped in the LUND header.
    """
    e = _os.environ.get("RC_QUAD", "")
    if e:
        v = tuple(int(x) for x in e.split(","))
        if len(v) != 4:
            raise ValueError(f"RC_QUAD wants four integers, got {e!r}")
        return v
    return (20, 12, 16, 12)


def sigma_F(kin0, sfmodel, h, v_m, lam=0.0,
            n_v=None, n_core=None, n_tail=None, n_phi=None, return_diag=False,
            v_lo=None, subtract=True):
    """IR-finite hard-photon integral (nb/GeV^6). kin0: Born EventKin (v=0).

    v_lo / subtract: consistency-test and generator hooks -- integrate over
    [v_lo, v_m] and optionally WITHOUT the Bardin-Shumeiko subtraction, which
    gives the positive-definite (samplable) slicing form. Defaults reproduce
    the BS-subtracted integral used by rc_factor."""
    _q = quad_default()
    n_v    = _q[0] if n_v    is None else n_v
    n_core = _q[1] if n_core is None else n_core
    n_tail = _q[2] if n_tail is None else n_tail
    n_phi  = _q[3] if n_phi  is None else n_phi
    m, M, mV = kin0.m, kin0.M, kin0.mV
    Q2, W2, W, t = kin0.Q2, kin0.W2, kin0.W, kin0.t
    k1, k2, p, q = kin0.k1, kin0.k2, kin0.p, kin0.q
    E1, E2 = k1[0], k2[0]

    L0, _ = born_tensors(kin0, h)
    H0 = born_H(kin0, sfmodel, lam)[0]
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

    # --- The v grid is evaluated in ONE batch per channel. -------------- #
    # Every quantity below is computed per angular node and never couples
    # nodes to each other, so stacking the n_v v-values into one array of
    # n_v*N rows does bitwise the same arithmetic on each row -- it only
    # stops paying numpy's per-call dispatch 20 times over on arrays of 336
    # elements, which is where the time actually went.  The reduction is the
    # one place order matters, so it is still done per v (a row reduction of
    # the (n_v, N) block, which numpy performs row by row with the same
    # pairwise algorithm) and accumulated in the original v-outer,
    # channel-inner order.
    total = 0.0
    n_bad = 0
    n_v_act = len(vs)
    N = len(channels[0]['nhat'])
    # Rows per batch.  Caps the working set -- fine quadratures (the selftest
    # uses N = 1344) would otherwise build several hundred MB of temporaries
    # at once -- and sits at the measured cache optimum: on the mini a scan
    # over 336...6720 rows was flat to ~3% with a shallow minimum near 4000.
    # The VALUE DOES NOT AFFECT THE RESULT: every chunk size tested gave
    # bit-identical sigma_F, which is the point of reducing per v.
    v_chunk = max(1, 4096 // max(N, 1))

    kinvs = [EventKin(kin0.Ebeam, Q2, W2, t, kin0.phi, mV, v=v, M=M, m=m,
                      phi_offset=getattr(kin0, 'phi_offset', 0.0),
                      mU=kin0.mU, t_kind=getattr(kin0, "t_kind", "photon"))
             for v in vs]

    sums = np.empty((len(channels), n_v_act))
    for ci, chd in enumerate(channels):
        nhat, wang = chd['nhat'], chd['wang']
        wch = chd['wchan']
        for i0 in range(0, n_v_act, v_chunk):
            i1 = min(i0 + v_chunk, n_v_act)
            nb = i1 - i0
            # per-v scalar setup stays a (cheap) loop: it is a handful of
            # flops each, and keeping `nhat @ Lam[1:]` in its original
            # (N,3)@(3,) form keeps that matvec bit-for-bit what it was.
            z_b = np.empty((nb, N))
            om_b = np.empty((nb, N))
            pv_b = np.empty((nb, 4))
            for j in range(nb):
                v = vs[i0 + j]
                Lam = kinvs[i0 + j].Lam
                z = Lam[0] - nhat @ Lam[1:]
                z_b[j] = z
                om_b[j] = v / (2.0 * z)
                pv_b[j] = kinvs[i0 + j].pv
            z = z_b.reshape(-1)
            om = om_b.reshape(-1)
            v_f = np.repeat(vs[i0:i1], N)
            nh = np.tile(nhat, (nb, 1))
            pv_f = np.repeat(pv_b, N, axis=0)
            k4 = np.concatenate([om[:, None], om[:, None] * nh], axis=1)

            kap1 = 2.0 * om * np.tile(chd['kh1'], nb)
            kap2 = 2.0 * om * np.tile(chd['kh2'], nb)

            # shifted kinematics
            qt = q[None, :] - k4
            Q2t = Q2 + 2.0 * dot4(k4, q[None, :])
            W2t = W2 - 2.0 * om * W
            tt = t - 2.0 * dot4(k4, q[None, :] - pv_f)

            good = Q2t > 1e-6
            n_bad += np.count_nonzero(~good)

            sf = _sf_to_gev(sfmodel(W2t, np.maximum(Q2t, 1e-6), tt))
            H = hadronic_tensor(qt, np.tile(p, (len(om), 1)), pv_f,
                                W2t, sf, M=M, lam=lam)

            LR, _, _ = rad_L(k1, k2, m, h, k4)
            LRH = np.real(contractLH(LR, H))

            fir = F_IR(Q2, m, kap1, kap2)
            exact = np.where(good, LRH / Q2t ** 2, 0.0)
            sub = 4.0 * fir * LH0 / Q2 ** 2 if subtract else 0.0
            integrand = (v_f / z ** 2) * (exact - sub)
            sums[ci, i0:i1] = (wang * wch * integrand.reshape(nb, N)).sum(axis=1)

    # accumulate in the original order: v outer, channel inner
    for iv in range(n_v_act):
        for ci in range(len(channels)):
            total += wvs[iv] * sums[ci, iv]

    pref = ALPHA ** 3 / (512.0 * np.pi ** 4 * kin0.lam_S * np.sqrt(kin0.lam_q))
    out = pref * total * GEV2_TO_NB
    if return_diag:
        return out, dict(n_bad=int(n_bad))
    return out
