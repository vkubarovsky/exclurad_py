"""Fused numba kernel for the radiative integrand.

The whole cost of rad generation is one quantity, evaluated ~232 000 times
per written event (see docs/cost_model.md):

    LRH = Re( L_R^{mu nu} H_{mu nu} )

computed today as `rad_L` -> `hadronic_tensor` -> `contractLH`, each
materialising (N,4,4) complex temporaries.  This computes it row by row with
nothing bigger than a 4x4 on the stack, which is the thing numpy cannot do
however well it is vectorised.

NOT bit-identical to the numpy path: same arithmetic, different order, and
LLVM may contract multiply-adds.  Agreement is checked numerically.

The structure-function model stays outside -- it is a Python object and
cannot be jitted -- which is a clean cut because it is already evaluated
once per block.
"""
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                    # pragma: no cover
    HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not a or not callable(a[0]) else a[0]
    prange = range

SIGN = np.array([1.0, -1.0, -1.0, -1.0])
ALPHA = 1.0 / 137.035999084
NU_LT = -1.0
NU_LTP = -1.0


@njit(cache=True, inline='always')
def _pslash(p, out):
    """out[4,4] complex <- p_mu gamma^mu, Dirac representation.

    The 12 nonzero entries written directly; mirrors core.spinors.fslash.
    """
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    for i in range(4):
        for j in range(4):
            out[i, j] = 0.0 + 0.0j
    out[0, 0] = p0
    out[1, 1] = p0
    out[2, 2] = -p0
    out[3, 3] = -p0
    a = complex(-p1, p2)      # -(p1 - i p2)
    b = complex(-p1, -p2)     # -(p1 + i p2)
    out[0, 2] = -p3
    out[0, 3] = a
    out[1, 2] = b
    out[1, 3] = p3
    out[2, 0] = p3
    out[2, 1] = -a
    out[3, 0] = -b
    out[3, 1] = -p3


@njit(cache=True, inline='always')
def _boost(beta, Lam):
    """Pure boost with velocity beta[3] -> Lam[4,4] real."""
    b2 = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2]
    g = 1.0 / np.sqrt(1.0 - b2)
    for i in range(4):
        for j in range(4):
            Lam[i, j] = 0.0
    Lam[0, 0] = g
    for i in range(3):
        Lam[0, i + 1] = -g * beta[i]
        Lam[i + 1, 0] = -g * beta[i]
    fac = (g - 1.0) / b2 if b2 > 1e-30 else 0.0
    for i in range(3):
        Lam[i + 1, i + 1] += 1.0
        for j in range(3):
            Lam[i + 1, j + 1] += fac * beta[i] * beta[j]


@njit(cache=True, inline='always')
def _mv(Lam, v, out):
    for i in range(4):
        s = 0.0
        for j in range(4):
            s += Lam[i, j] * v[j]
        out[i] = s


@njit(cache=True, inline='always')
def _dot4(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


@njit(cache=True, parallel=False)
def lrh_kernel(k1, k2, m, k4, qt, p, pv, W2t,
               sT, sL, sTT, sLT, sLTp, M,
               u1, ub2, UB2G, GU1, out):
    """LRH[n] = Re( L_R^{mu nu} H_{mu nu} ) for a batch of photons.

    k1,k2 (4,) real; k4,qt,p,pv (N,4) real; W2t and the five sf (N,) real;
    u1 (4,), ub2 (2,4), UB2G (2,4,4), GU1 (4,4) complex -- all photon-
    independent, built once in numpy by the caller.
    out (N,) real, written in place.

    Everything between here and the return is per row: nothing bigger than a
    4x4 is allocated, and L_R and H are never materialised as arrays of N.
    """
    N = k4.shape[0]
    # per-row scratch
    Bm = np.empty((4, 4)); Bi = np.empty((4, 4))
    qtc = np.empty(4); pvc = np.empty(4)
    Xc = np.empty(4); Yc = np.empty(4); Lc = np.empty(4)
    X = np.empty(4); Y = np.empty(4); Lv = np.empty(4)
    beta = np.empty(3); zhat = np.empty(3); hperp = np.empty(3)
    xhat = np.empty(3); yhat = np.empty(3); fb = np.empty(3)
    nh = np.empty(3); e1 = np.empty(3); e2 = np.empty(3); refv = np.empty(3)
    es = np.empty((4, 4), dtype=np.complex128)
    pr1 = np.empty((4, 4), dtype=np.complex128)
    pr2 = np.empty((4, 4), dtype=np.complex128)
    tmp = np.empty((4, 4), dtype=np.complex128)
    epsv = np.empty(4)
    v1 = np.empty(4, dtype=np.complex128)
    v2 = np.empty(4, dtype=np.complex128)
    w1 = np.empty(4, dtype=np.complex128)
    w2 = np.empty(4, dtype=np.complex128)
    jc = np.empty(4, dtype=np.complex128)
    Lt = np.empty((4, 4), dtype=np.complex128)
    H = np.empty((4, 4), dtype=np.complex128)
    km = np.empty(4)

    for n in range(N):
        # ---------------- photon basis X, Y, L at this shifted point -----
        P0 = p[n, 0] + qt[n, 0]
        for i in range(3):
            beta[i] = (p[n, i + 1] + qt[n, i + 1]) / P0
        _boost(beta, Bm)
        for i in range(4):
            for j in range(4):
                Bi[i, j] = Bm[i, j]
        for i in range(3):
            Bi[0, i + 1] = -Bm[0, i + 1]
            Bi[i + 1, 0] = -Bm[i + 1, 0]
        _mv(Bm, qt[n], qtc)
        _mv(Bm, pv[n], pvc)

        Q2t = -_dot4(qt[n], qt[n])
        aq = np.sqrt(qtc[1] ** 2 + qtc[2] ** 2 + qtc[3] ** 2)
        for i in range(3):
            zhat[i] = qtc[i + 1] / aq
        pz = pvc[1] * zhat[0] + pvc[2] * zhat[1] + pvc[3] * zhat[2]
        for i in range(3):
            hperp[i] = pvc[i + 1] - pz * zhat[i]
        hn = np.sqrt(hperp[0] ** 2 + hperp[1] ** 2 + hperp[2] ** 2)
        # fallback direction: cross(zhat, (0,1,0))
        fb[0] = zhat[1] * 0.0 - zhat[2] * 1.0
        fb[1] = zhat[2] * 0.0 - zhat[0] * 0.0
        fb[2] = zhat[0] * 1.0 - zhat[1] * 0.0
        fbn = np.sqrt(fb[0] ** 2 + fb[1] ** 2 + fb[2] ** 2)
        if hn < 1e-12 * pvc[0]:
            d = fbn if fbn > 0.0 else 1.0
            for i in range(3):
                xhat[i] = fb[i] / d
        else:
            d = hn if hn > 0.0 else 1.0
            for i in range(3):
                xhat[i] = hperp[i] / d
        yhat[0] = zhat[1] * xhat[2] - zhat[2] * xhat[1]
        yhat[1] = zhat[2] * xhat[0] - zhat[0] * xhat[2]
        yhat[2] = zhat[0] * xhat[1] - zhat[1] * xhat[0]

        sq = np.sqrt(Q2t)
        Xc[0] = 0.0; Yc[0] = 0.0; Lc[0] = aq / sq
        for i in range(3):
            Xc[i + 1] = xhat[i]
            Yc[i + 1] = yhat[i]
            Lc[i + 1] = (qtc[0] / sq) * zhat[i]
        _mv(Bi, Xc, X); _mv(Bi, Yc, Y); _mv(Bi, Lc, Lv)

        # ---------------- hadronic tensor H ------------------------------
        kap = 8.0 * (W2t[n] - M * M) ** 2 / ALPHA
        half = 0.5 * kap
        a_T, a_L, a_TT = sT[n], sL[n], sTT[n]
        a_LT, a_LTp = NU_LT * sLT[n], NU_LTP * sLTp[n]
        for i in range(4):
            for j in range(4):
                re = (a_T * (X[i] * X[j] + Y[i] * Y[j])
                      + a_L * Lv[i] * Lv[j]
                      + a_TT * (X[i] * X[j] - Y[i] * Y[j])
                      + a_LT * (Lv[i] * X[j] + X[i] * Lv[j]))
                im = a_LTp * (Lv[i] * X[j] - X[i] * Lv[j])
                H[i, j] = complex(half * re, half * im)

        # ---------------- radiative leptonic tensor ----------------------
        kap1 = 2.0 * _dot4(k4[n], k1)
        kap2 = 2.0 * _dot4(k4[n], k2)
        for i in range(4):
            km[i] = k1[i] - k4[n, i]
        _pslash(km, pr1)
        for i in range(4):
            pr1[i, i] += m
        inv1 = 1.0 / (-kap1)
        for i in range(4):
            for j in range(4):
                pr1[i, j] *= inv1
        for i in range(4):
            km[i] = k2[i] + k4[n, i]
        _pslash(km, pr2)
        for i in range(4):
            pr2[i, i] += m
        inv2 = 1.0 / kap2
        for i in range(4):
            for j in range(4):
                pr2[i, j] *= inv2

        # two real transverse polarizations of this photon
        kn = np.sqrt(k4[n, 1] ** 2 + k4[n, 2] ** 2 + k4[n, 3] ** 2)
        for i in range(3):
            nh[i] = k4[n, i + 1] / kn
        imin = 0
        amin = abs(nh[0])
        if abs(nh[1]) < amin:
            imin = 1; amin = abs(nh[1])
        if abs(nh[2]) < amin:
            imin = 2
        refv[0] = 0.0; refv[1] = 0.0; refv[2] = 0.0
        refv[imin] = 1.0
        e1[0] = refv[1] * nh[2] - refv[2] * nh[1]
        e1[1] = refv[2] * nh[0] - refv[0] * nh[2]
        e1[2] = refv[0] * nh[1] - refv[1] * nh[0]
        en = np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
        for i in range(3):
            e1[i] /= en
        e2[0] = nh[1] * e1[2] - nh[2] * e1[1]
        e2[1] = nh[2] * e1[0] - nh[0] * e1[2]
        e2[2] = nh[0] * e1[1] - nh[1] * e1[0]

        for i in range(4):
            for j in range(4):
                Lt[i, j] = 0.0 + 0.0j

        for a in range(2):
            epsv[0] = 0.0
            for i in range(3):
                epsv[i + 1] = e1[i] if a == 0 else e2[i]
            _pslash(epsv, es)
            # v1 = es u1 ; v2 = prop1 v1
            for i in range(4):
                s = 0.0 + 0.0j
                for j in range(4):
                    s += es[i, j] * u1[j]
                v1[i] = s
            for i in range(4):
                s = 0.0 + 0.0j
                for j in range(4):
                    s += pr1[i, j] * v1[j]
                v2[i] = s
            for il2 in range(2):
                # w1 = ubar2 es ; w2 = w1 prop2
                for j in range(4):
                    s = 0.0 + 0.0j
                    for i in range(4):
                        s += ub2[il2, i] * es[i, j]
                    w1[j] = s
                for j in range(4):
                    s = 0.0 + 0.0j
                    for i in range(4):
                        s += w1[i] * pr2[i, j]
                    w2[j] = s
                for mu in range(4):
                    sA = 0.0 + 0.0j
                    sB = 0.0 + 0.0j
                    for j in range(4):
                        sA += UB2G[il2, mu, j] * v2[j]
                        sB += w2[j] * GU1[mu, j]
                    jc[mu] = sA + sB
                for i in range(4):
                    for j in range(4):
                        Lt[i, j] += jc[i] * np.conj(jc[j])

        # ---------------- contraction ------------------------------------
        acc = 0.0
        for i in range(4):
            si = SIGN[i]
            for j in range(4):
                acc += (Lt[i, j] * (si * SIGN[j]) * H[i, j]).real
        out[n] = acc
