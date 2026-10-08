"""An independent first-order calculation of the hard-photon tail, to decide
between exclurad_py and EXCLURAD for J/psi (they agree for v <~ 0.03 GeV^2 and
part by up to 16 % of the tail above it, see diag_vladder.csv).

    python3 validation/jpsi_exclurad_fortran/independent_tail.py <diag_vladder work dir> [jobs] [quality]

Shares NOTHING with either code except
  * the J/psi model jpsi.diffrad_vm (sigma_T, sigma_L at W~2, Q~2, the
    observed t_N -- the matched-integrand mode of the comparison), and
  * the convention mapping the model to a hadronic tensor,
        H^{mu nu} = N(W2,Q2) [ sigma_T (-g + q q/q^2 + L L) + sigma_L L L ],
        N = (W2 - M^2)^2 / sqrt(lambda_q),   lambda_q = (W2+Q2-M^2)^2 + 4 M^2 Q2,
    i.e. the |q|-flux convention on which both codes already agree
    (exclurad_py hadronic.py; EXCLURAD jpsi_sf.F reduces to the same W, Q2
    dependence).

Everything else is written here from scratch:
  * the leptonic side: the exact one-photon amplitude e(k1) -> e(k2) gamma(k)
    with both diagrams and the full electron mass, from explicit Dirac
    spinors (no analytic tensor from either code), squared numerically and
    summed over spins and the two physical photon polarisations;
  * the phase space, in the bin definition the codes are compared in
    (EXCLURAD's: detected proton at a FROZEN direction in the observed
    gamma* p CM, its energy following v = MX^2 - m_J/psi^2):

        d sigma_R/dX       /p_h(v)        omega              |M_R|^2
        ------------  =  | --------- dv dOmega_k ------------ -------
        d sigma_B/dX       J p_h(0)        4 D (2 pi)^3      |M_B|^2

    D = Lambda^0 - n.Lambda, Lambda = p + q - p_h, omega = v / (2 D);
  * the photon-angle integral: two channels around the beam and scattered-
    electron directions, partition of unity a/(a+b) with a = 1/(k1.k),
    Gauss-Legendre in ln(1 - beta cos) into each collinear peak.

It returns the FIRST-ORDER tail added between successive v cuts, and
compares it with both codes non-exponentiated (EXCLURAD: 1 + a/pi(dVR + vac
+ dinf(v)) + tail(v)/sib from jpsi_eta.dat; exclurad_py: rc_factor with
exponentiate=False), and with EXCLURAD's own leading-log tail (qqtll), which
is the peaking approximation of its formalism.
"""
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

ALPHA = 1.0 / 137.035999
E2 = 4 * np.pi * ALPHA
ME = 0.51099895e-3
MP = 0.93827208816
MJ = 3.0969
EBEAM = 10.6
G = np.diag([1.0, -1.0, -1.0, -1.0])

# Dirac matrices (Dirac representation)
_I2 = np.eye(2)
_Z2 = np.zeros((2, 2))
_SIG = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex),
        np.array([[1, 0], [0, -1]], complex)]
GAM = np.array([np.block([[_I2, _Z2], [_Z2, -_I2]]).astype(complex)]
               + [np.block([[_Z2, s], [-s, _Z2]]) for s in _SIG])


def dot(a, b):
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def slash(p):
    return np.einsum("...m,m,mij->...ij", p, np.diag(G), GAM)


def spinors(p, m):
    """u(p, s) for s = up, down: (..., 2, 4)."""
    E = p[..., 0]
    sp = np.einsum("...i,ijk->...jk", p[..., 1:], np.array(_SIG))     # sigma.p
    out = []
    for chi in (np.array([1, 0], complex), np.array([0, 1], complex)):
        up = np.broadcast_to(chi, E.shape + (2,)) * np.sqrt(E + m)[..., None]
        lo = np.einsum("...ij,j->...i", sp, chi) / np.sqrt(E + m)[..., None]
        out.append(np.concatenate([up, lo], axis=-1))
    return np.stack(out, axis=-2)


def boost_to(p4, beta):
    """Lorentz boost of 4-vectors p4 (...,4) by velocity beta (3,)."""
    b2 = beta @ beta
    g = 1 / np.sqrt(1 - b2)
    bp = p4[..., 1:] @ beta
    E = g * (p4[..., 0] - bp)
    v = p4[..., 1:] + ((g - 1) * bp / b2 - g * p4[..., 0])[..., None] * beta
    return np.concatenate([E[..., None], v], axis=-1)


class Point:
    """Observed kinematics in the gamma* p CM, q along +z, beam in the x-z plane."""

    def __init__(self, Q2, W2, tN, phi_deg, mode="cm_angle"):
        from exclurad_py.models.vector import Jpsi_DiffradVM
        self.mode = mode
        self.model = Jpsi_DiffradVM()
        self.Q2, self.W2, self.tN = Q2, W2, tN
        nu = (W2 - MP ** 2 + Q2) / (2 * MP)
        Ep = EBEAM - nu
        p1 = np.sqrt(EBEAM ** 2 - ME ** 2)
        p2 = np.sqrt(Ep ** 2 - ME ** 2)
        c = (EBEAM * Ep - ME ** 2 - Q2 / 2) / (p1 * p2)
        s = np.sqrt(1 - c * c)
        k1 = np.array([EBEAM, 0, 0, p1])
        k2 = np.array([Ep, p2 * s, 0, p2 * c])
        p = np.array([MP, 0, 0, 0.0])
        q = k1 - k2
        beta = q[1:] / (MP + q[0])
        k1, k2, q, p = (boost_to(x, beta) for x in (k1, k2, q, p))
        # rotate q to +z, keep the beam in the x-z plane with k1_x > 0
        ez = q[1:] / np.linalg.norm(q[1:])
        ex = k1[1:] - (k1[1:] @ ez) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        R = np.array([ex, ey, ez])
        rot = lambda x: np.concatenate([[x[0]], R @ x[1:]])         # noqa: E731
        self.k1, self.k2, self.q, self.p = (rot(x) for x in (k1, k2, q, p))
        self.W = np.sqrt(W2)
        self.phi = np.radians(phi_deg)
        self.qv = np.linalg.norm(self.q[1:])
        if mode == "meson_t":
            # generator bin: J/psi detected, t = (q - p_V)^2 fixed, proton missing
            self.mdet, self.mmis = MJ, MP
            Ev0 = (W2 + MJ ** 2 - MP ** 2) / (2 * self.W)
            self.ph0 = np.sqrt(Ev0 ** 2 - MJ ** 2)
            self.nh = self._dir_meson_t(0.0)
            self.vmax = self._vmax_meson_t()
            self.LH_B = self._LH(self._LB(), self.q[None, :], np.array([True]),
                                 t_sf=np.array([tN]))[0]
            return
        self.mdet, self.mmis = MP, MJ
        # detected proton: direction frozen at its v = 0 value (t_N), azimuth
        # phi + pi (the meson goes to phi, Trento), energy follows v
        Ep0 = self.p[0]
        qv = np.linalg.norm(self.q[1:])
        Eh0 = (W2 + MP ** 2 - MJ ** 2) / (2 * self.W)
        ph0 = np.sqrt(Eh0 ** 2 - MP ** 2)
        cth = (2 * MP ** 2 - tN - 2 * Ep0 * Eh0) / (2 * qv * ph0)
        sth = np.sqrt(1 - cth * cth)
        ph = np.radians(phi_deg) + np.pi
        self.nh = np.array([sth * np.cos(ph), sth * np.sin(ph), cth])
        self.ph0 = ph0
        self.vmax = (self.W - MP) ** 2 - MJ ** 2
        # Born leptonic . hadronic at the observed point
        self.LH_B = self._LH(self._LB(), self.q[None, :], np.array([True]))[0]

    def _dir_meson_t(self, v):
        """J/psi direction at this v with t = (q - p_V)^2 held fixed (None if unphysical)."""
        W = self.W
        Ev = (self.W2 + MJ ** 2 - MP ** 2 - v) / (2 * W)
        if Ev <= MJ:
            return None
        pv = np.sqrt(Ev ** 2 - MJ ** 2)
        q0 = self.q[0]
        c = (self.tN + self.Q2 - MJ ** 2 + 2 * q0 * Ev) / (2 * self.qv * pv)
        if abs(c) > 1:
            return None
        s_ = np.sqrt(1 - c * c)
        return np.array([s_ * np.cos(self.phi), s_ * np.sin(self.phi), c])

    def _vmax_meson_t(self):
        lo, hi = 0.0, (self.W - MJ) ** 2 - MP ** 2
        if self._dir_meson_t(hi * (1 - 1e-12)) is not None:
            return hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self._dir_meson_t(mid) is None:
                hi = mid
            else:
                lo = mid
        return lo

    # -- leptonic tensors --------------------------------------------------
    def _LB(self):
        u1, u2 = spinors(self.k1, ME), spinors(self.k2, ME)
        ub2 = np.conj(u2) @ GAM[0]
        J = np.einsum("ai,mij,bj->abm", ub2, GAM, u1)          # (s2, s1, mu)
        return np.einsum("abm,abn->mn", J, np.conj(J))[None]

    def _LR(self, k):
        """Radiative leptonic tensor for photons k (N,4): (N,4,4)."""
        k1, k2 = self.k1, self.k2
        u1, u2 = spinors(k1, ME), spinors(k2, ME)
        ub2 = np.conj(u2) @ GAM[0]
        n = k[:, 1:] / k[:, :1]
        a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
        e1 = np.cross(n, a)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(n, e1)
        S_f = (slash(k2[None] + k) + ME * np.eye(4)) / (2 * dot(k2[None], k))[:, None, None]
        S_i = (slash(k1[None] - k) + ME * np.eye(4)) / (-2 * dot(k1[None], k))[:, None, None]
        L = np.zeros((len(k), 4, 4), complex)
        for e in (e1, e2):
            eps = np.concatenate([np.zeros((len(k), 1)), e], axis=1)
            es = slash(eps)                         # real polarisation, eps* = eps
            O = (np.einsum("nij,njk,mkl->nmil", es, S_f, GAM)
                 + np.einsum("mij,njk,nkl->nmil", GAM, S_i, es))
            J = np.einsum("ai,nmij,bj->nabm", ub2, O, u1)
            L += np.einsum("nabm,nabk->nmk", J, np.conj(J))
        return L

    # -- hadronic contraction ------------------------------------------------
    def _LH(self, L, qt, ok, t_sf=None):
        """L_{mu nu} H^{mu nu}(qt) with sigma at (W~2, Q~2, t_N); (N,)."""
        p = self.p
        Q2t = -dot(qt, qt)
        W2t = dot(p[None] + qt, p[None] + qt)
        pq = dot(p[None], qt)
        P = p[None] - (pq / dot(qt, qt))[:, None] * qt
        Lv = P / np.sqrt(dot(P, P))[:, None]
        tt = np.full_like(W2t, self.tN) if t_sf is None else np.broadcast_to(t_sf, W2t.shape)
        sf = self.model._sfs(W2t, np.maximum(Q2t, 1e-9), tt)
        sT, sL = sf["sT"], sf["sL"]
        lamq = (W2t + Q2t - MP ** 2) ** 2 + 4 * MP ** 2 * Q2t
        N = (W2t - MP ** 2) ** 2 / np.sqrt(lamq)
        # lower both tensor indices of L, contract with H^{mu nu}
        Ll = np.einsum("mi,nij,jk->nmk", G, L, G).real if L.ndim == 3 else None
        gg = np.einsum("nmk,mk->n", Ll, np.linalg.inv(G))      # L_{mu nu} g^{mu nu}
        qLq = np.einsum("nm,nmk,nk->n", qt, Ll, qt)
        lLl = np.einsum("nm,nmk,nk->n", Lv, Ll, Lv)
        HT = -gg + qLq / dot(qt, qt) + lLl
        out = N * (sT * HT + sL * lLl)
        return np.where(ok, out, 0.0)

    # -- the tail at fixed v, integrated over the photon direction -----------
    def dtail_dv(self, v, nt=160, nps=48):
        """d(eta)/dv, first order, at this v (photon angles integrated)."""
        W = self.W
        if self.mode == "meson_t":
            nd = self._dir_meson_t(v)
            if nd is None:
                return 0.0
            Eh = (self.W2 + MJ ** 2 - MP ** 2 - v) / (2 * W)
            phv = np.sqrt(Eh ** 2 - MJ ** 2)
            pvec = phv * nd
            # at fixed t, d3p/2E = (p/2) dE dcos dphi and dcos = dt/(2|q|p):
            # p cancels, the detected-particle factor is 1 (not p(v)/p(0))
            jacv = 1.0
            pdet = np.concatenate([[Eh], pvec])
        else:
            Eh = (self.W2 + MP ** 2 - MJ ** 2 - v) / (2 * W)
            if Eh <= MP:
                return 0.0
            phv = np.sqrt(Eh ** 2 - MP ** 2)
            pvec = phv * self.nh
            jacv = phv / self.ph0
            pdet = None
        Lam = np.concatenate([[W - Eh], -pvec])            # p + q - p_det in the CM
        tot = 0.0
        xg, wg = np.polynomial.legendre.leggauss(nt)
        psi = (np.arange(nps) + 0.5) * 2 * np.pi / nps
        for axis_k, other_k in ((self.k1, self.k2), (self.k2, self.k1)):
            beta = np.linalg.norm(axis_k[1:]) / axis_k[0]
            ez = axis_k[1:] / np.linalg.norm(axis_k[1:])
            tmp = np.array([1.0, 0, 0]) if abs(ez[0]) < 0.9 else np.array([0, 1.0, 0])
            ex = np.cross(ez, tmp); ex /= np.linalg.norm(ex)
            ey = np.cross(ez, ex)
            lo, hi = np.log(1 - beta), np.log(1 + beta)
            # nodes: tt = ln(1 - beta c) on [lo, hi]
            tt = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
            wt = 0.5 * (hi - lo) * wg
            c = (1 - np.exp(tt)) / beta
            jac = np.exp(tt) / beta                         # |dc/dtt|
            C, PS = np.meshgrid(c, psi, indexing="ij")
            Wc = np.outer(wt * jac, np.full(nps, 2 * np.pi / nps))
            S = np.sqrt(np.clip(1 - C * C, 0, None))
            n = (S[..., None] * (np.cos(PS)[..., None] * ex + np.sin(PS)[..., None] * ey)
                 + C[..., None] * ez).reshape(-1, 3)
            w = Wc.reshape(-1)
            D = Lam[0] - n @ Lam[1:]
            om = v / (2 * D)
            k = np.concatenate([om[:, None], om[:, None] * n], axis=1)
            # partition of unity: this channel's share
            a = 1.0 / dot(axis_k[None], k)
            b = 1.0 / dot(other_k[None], k)
            share = a / (a + b)
            qt = self.q[None] - k
            W2t = dot(self.p[None] + qt, self.p[None] + qt)
            ok = W2t > (MP + MJ) ** 2
            if self.mode == "meson_t":
                # the model's kinematic t at the shifted point: (q~ - p_V)^2
                d_ = qt - pdet[None]
                LH = self._LH(self._LR(k), qt, ok, t_sf=dot(d_, d_))
            else:
                LH = self._LH(self._LR(k), qt, ok)
            Q2t = -dot(qt, qt)
            f = jacv * om / (4 * D * (2 * np.pi) ** 3) * E2 \
                * (self.Q2 / Q2t) ** 2 * LH / self.LH_B
            tot += np.sum(w * share * f)
        # dv: the proton-energy integral, d E_h = dv / (2W), is the J of the
        # docstring together with the Born's 1/(2W) -- they cancel exactly
        return tot

    def soft_check(self, v=1e-7):
        if self.mode == "meson_t":
            return float("nan")
        """|M_R|^2/|M_B|^2 against the eikonal factor at a soft photon."""
        n = np.array([[0.3, 0.4, np.sqrt(1 - 0.25)]])
        om = 1e-6
        k = np.concatenate([[[om]], om * n], axis=1)
        LH = self._LH(self._LR(k), self.q[None] - k, np.array([True]))
        r = E2 * LH / self.LH_B * (self.Q2 / -dot(self.q[None] - k, self.q[None] - k)) ** 2
        eik = E2 * (2 * dot(self.k1, self.k2) / (dot(self.k1[None], k) * dot(self.k2[None], k))
                    - ME ** 2 / dot(self.k1[None], k) ** 2 - ME ** 2 / dot(self.k2[None], k) ** 2)
        return float(r[0] / eik[0])


def tail_between(args):
    Q2, W2, tN, phi, va, vb, nv, nt, nps = args[:9]
    pt = Point(Q2, W2, tN, phi, mode=args[9] if len(args) > 9 else "cm_angle")
    vb = min(vb, pt.vmax * (1 - 1e-9))
    if vb <= va:
        return 0.0
    x, w = np.polynomial.legendre.leggauss(nv)
    # integrate in ln v: the tail falls like 1/v
    la, lb = np.log(va), np.log(vb)
    lv = 0.5 * (lb - la) * x + 0.5 * (lb + la)
    vv = np.exp(lv)
    return float(sum(0.5 * (lb - la) * wi * v * pt.dtail_dv(v, nt, nps) for v, wi in zip(vv, w)))


def code_values(work, case, cuts):
    """Non-exponentiated eta and LL eta of EXCLURAD at each cut (from the diag
    runs), and exclurad_py non-exponentiated at the same cuts."""
    A = 0.729735e-2
    ex, ll = [], []
    for vc in cuts:
        f = os.path.join(work, f"F_{case}_v{vc:g}", "jpsi_eta.dat")
        r = np.loadtxt(f, ndmin=2)[0]
        dvr, dinf, vac, tail, extai = r[7], r[8], r[9], r[10], r[12]
        ex.append(1 + A / np.pi * (dvr + vac + dinf) + tail)
        taill = r[5] - extai * (1 + A / np.pi * (dvr + vac))
        ll.append(1 + A / np.pi * (dvr + vac + dinf) + taill)
    return np.array(ex), np.array(ll)


def py_values(Q2, W2, tN, phi, cuts):
    os.environ["RC_QUAD"] = "160,64,96,64"
    from exclurad_py import rc_factor
    sys.path.insert(0, HERE)
    from compare import JpsiProtonDetected
    sf = JpsiProtonDetected(tN, "cm_angle")
    out = []
    for vc in cuts:
        cut = None if vc == 0 else vc + MJ ** 2
        r = [rc_factor(Ebeam=EBEAM, Q2=Q2, W2=W2, t=tN, phi=np.radians(phi), sfmodel=sf,
                       h=h, mx2_cut=cut, exponentiate=False) for h in (1, -1)]
        out.append((r[0]["sigma_obs"] + r[1]["sigma_obs"]) / (r[0]["sigma_born"] + r[1]["sigma_born"]))
    return np.array(out)


def main(work, jobs, quality):
    nv, nt, nps = {"low": (8, 96, 32), "mid": (12, 160, 48), "high": (16, 256, 64)}[quality]
    # (case tag in diag_vladder, Q2, W2, |t|, phi)
    points = [("00_jpsi", 0.03, 17.0, 2.592475, 20.0),
              ("01_jpsi", 0.03, 18.5, 1.331910, 20.0),
              ("05_jpsi", 1.50, 17.0, None, 20.0)]
    import csv
    rows = list(csv.reader(open(os.path.join(HERE, "diag_vladder.csv"))))
    t05 = [float(r[3]) for r in rows[1:] if int(float(r[0])) == 5][0]
    points[2] = ("05_jpsi", 1.50, 17.0, t05, 20.0)
    cuts = [0.03, 0.1, 0.2, 0.5, 0.0]
    edges = [0.03, 0.1, 0.2, 0.5, 10.0]
    for tag, Q2, W2, ta, phi in points:
        pt = Point(Q2, W2, -ta, phi)
        print(f"\n=== {tag}: Q2={Q2} W2={W2} |t|={ta:.4f} phi={phi}   v_max(frozen angle) = {pt.vmax:.4f}")
        print(f"    soft-photon check |M_R|^2/|M_B|^2 / eikonal = {pt.soft_check():.6f}")
        tasks = [(Q2, W2, -ta, phi, edges[i], edges[i + 1], nv, nt, nps) for i in range(4)]
        with mp.Pool(min(jobs, 4)) as pool:
            ind = pool.map(tail_between, tasks)
        exv, llv = code_values(work, tag, cuts)
        pyv = py_values(Q2, W2, -ta, phi, cuts)
        print(f"    {'v interval':>14} {'independent':>12} {'exclurad_py':>12} {'EXCLURAD':>10} {'EXCL. LL':>10}"
              "   (first-order tail added in the interval, units of sigma_Born)")
        names = ["0.03-0.1", "0.1-0.2", "0.2-0.5", "0.5-vmax"]
        for i in range(4):
            print(f"    {names[i]:>14} {ind[i]:12.5f} {pyv[i+1]-pyv[i]:12.5f} {exv[i+1]-exv[i]:10.5f} {llv[i+1]-llv[i]:10.5f}")
        print(f"    {'total >0.03':>14} {sum(ind):12.5f} {pyv[-1]-pyv[0]:12.5f} {exv[-1]-exv[0]:10.5f} {llv[-1]-llv[0]:10.5f}")


def generator_mode(jobs, quality):
    """J/psi as the generator sees it: compare with rc_factor(models.get('jpsi.diffrad_vm'))."""
    nv, nt, nps = {"low": (8, 96, 32), "mid": (12, 160, 48), "high": (16, 256, 64)}[quality]
    from exclurad_py import rc_factor, models
    sf = models.get("jpsi.diffrad_vm")
    tmin = lambda W2, Q2: (np.sqrt(((W2 + Q2 + MP**2) / (2*np.sqrt(W2)))**2 - MP**2)
                           - np.sqrt(((W2 + MJ**2 - MP**2) / (2*np.sqrt(W2)))**2 - MJ**2))**2 \
        - ((Q2 + MJ**2) / (2*np.sqrt(W2)))**2
    pts = [(0.007, 20.3, 0.1), (0.007, 20.3, 0.8), (0.007, 18.5, 1.4), (0.02, 19.5, 0.5), (1.5, 17.0, 1.4)]
    edges = [0.03, 0.1, 0.2, 0.5, 100.0]
    for Q2, W2, tp in pts:
        t = -(tmin(W2, Q2) + tp)
        pt = Point(Q2, W2, t, 20.0, mode="meson_t")
        tasks = [(Q2, W2, t, 20.0, edges[i], edges[i + 1], nv, nt, nps, "meson_t") for i in range(4)]
        with mp.Pool(min(jobs, 4)) as pool:
            ind = pool.map(tail_between, tasks)
        pyv = []
        for vc in [0.03, 0.1, 0.2, 0.5, 0.0]:
            cut = None if vc == 0 else vc + MP ** 2
            r = [rc_factor(Ebeam=EBEAM, Q2=Q2, W2=W2, t=t, phi=np.radians(20.0), sfmodel=sf,
                           h=h, mx2_cut=cut, exponentiate=False) for h in (1, -1)]
            pyv.append((r[0]["sigma_obs"] + r[1]["sigma_obs"]) / (r[0]["sigma_born"] + r[1]["sigma_born"]))
        print(f"\n=== generator mode: Q2={Q2} W2={W2} t'={tp} (t={t:.4f})   v_max(fixed t) = {pt.vmax:.4f}")
        print(f"    {'v interval':>14} {'independent':>12} {'exclurad_py':>12}")
        names = ["0.03-0.1", "0.1-0.2", "0.2-0.5", "0.5-vmax"]
        for i in range(4):
            print(f"    {names[i]:>14} {ind[i]:12.5f} {pyv[i+1]-pyv[i]:12.5f}")
        print(f"    {'total >0.03':>14} {sum(ind):12.5f} {pyv[-1]-pyv[0]:12.5f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generator":
        generator_mode(int(sys.argv[2]) if len(sys.argv) > 2 else 16,
                       sys.argv[3] if len(sys.argv) > 3 else "mid")
        sys.exit()
    main(os.path.abspath(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 16,
         sys.argv[3] if len(sys.argv) > 3 else "mid")
