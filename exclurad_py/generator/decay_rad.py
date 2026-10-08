"""
Radiation of the decay leptons, V -> l+ l- (gamma), for J/psi -> e+ e-.

What this is
------------
The production step (exclurad_py proper) radiates from the beam and the
scattered electron.  The J/psi then decays, and its leptons radiate too.  The
two steps are treated as independent: the J/psi is narrow, so a photon of
energy k emitted in production and one emitted in the decay put the J/psi
propagator at virtualities that differ by ~2 M k, and their interference is
suppressed by ~Gamma/k.  That is negligible for every photon that changes
M(ee) by more than the resolution (k >~ 1 MeV against Gamma = 93 keV).

Normalisation: NONE is applied.  The decay is a redistribution in M(ee), not
a change in the number of decays.  The analysis divides by the measured
B(J/psi -> e+e-), which is radiation-inclusive (BESIII PRD 88 032007 counts
M(ee) in [2.7, 3.2] GeV and restores the tail below with PHOTOS), so it
already contains the virtual + real correction (1 + delta_total).

Two pieces
----------
1. PHOTON ENERGY k (J/psi rest frame), from the exact first-order spectrum,
   exponentiated.  delta(K) = (soft + virtual) + hard(k < K) is the
   closed-form result of Ehlotzky & Mitter (Nuovo Cim. 55A (1968) 181), as
   implemented in ~/jpsi-rc-2026/rc_vector_mesons.py:delta_vm (with the
   rmax(rmax-6)/16 term restored in Sept 2026).  Its K-dependence starts as
   beta ln(K/E), beta = (2 alpha/pi)(ln(M^2/m^2) - 1) = 0.076 for J/psi -> ee.
   Exponentiating that log (YFS) gives the cumulative distribution

        F(K) = (K/E)^beta [1 + delta(K) - beta ln(K/E)] / [1 + delta(E)],

   F(E) = 1.  F(eps) is the fraction with no photon above eps (the "A" of
   the shape-normalised form), 1 - F(eps) the tail ("B"); A + B = 1.
   As usual for an exponentiated single-photon generator, the whole radiated
   energy is given to one photon -- multiple photons are collinear with the
   leptons anyway and M(ee) depends only on the total.

2. ANGLES at that k, from the EXACT tree-level matrix element of
   V -> l-(p1) l+(p2) gamma(k): photon from either lepton, full lepton mass,
   summed over all spins and polarisations (unpolarised V, i.e. the
   isotropic decay chosen for production).  At fixed M(ee) the three-body
   phase space is flat in cos(theta*) -- the angle between l- and the photon
   in the l+l- rest frame -- so the matrix element IS the density in
   cos(theta*).  Sampled by accept/reject against the eikonal antenna
   1/(1 - b c) + 1/(1 + b c), which carries the collinear peaks.

The matrix element is evaluated numerically from Dirac traces, not from a
transcribed formula: validation/jpsi_decay_fsr.py checks gauge invariance,
current conservation, and that its angle integral reproduces the Ehlotzky
spectrum used in step 1 -- which is what makes 1 and 2 one model.
"""

import numpy as np
from scipy.special import spence

ALPHA = 1.0 / 137.035999
M_E = 0.51099895e-3

# ---------------------------------------------------------------- spectrum
# Copied from ~/jpsi-rc-2026/rc_vector_mesons.py (delta_vm), the source of
# truth for these formulas; kept here so the farm needs no second checkout.


def _li2(x):
    return spence(1.0 - x)


def delta_vm(kmax, mV, m=M_E):
    """First-order correction to Gamma(V -> l+l-) with photons k < kmax
    (soft + virtual + hard), relative to the lowest-order width."""
    E = mV / 2.0
    r = np.clip(np.asarray(kmax, float) / E, 1e-300, 1.0)
    L = np.log(2.0 * E / m)
    R = (r - r * r / 4.0 - 0.75) * (L + 0.5 * np.log(np.clip(1.0 - r, 1e-300, None)) - 0.5)
    dd = r * (r - 6.0) / 16.0
    return -4.0 * ALPHA / np.pi * (dd + (0.5 - L) * (np.log(r) + 1.0 / 3.0)
                                   + 0.5 * (_li2(r) - np.pi ** 2 / 6.0) + 17.0 / 72.0 + R)


def beta_ir(mV, m=M_E):
    return 2.0 * ALPHA / np.pi * (np.log(mV * mV / (m * m)) - 1.0)


class PhotonEnergy:
    """Inverse-CDF sampler of the decay-photon energy k in the V rest frame."""

    def __init__(self, mV, m=M_E, n=4000):
        self.mV, self.m = mV, m
        E = mV / 2.0
        self.beta = b = beta_ir(mV, m)
        # physical endpoint: M(ll) = 2m
        self.kmax = (mV * mV - 4.0 * m * m) / (2.0 * mV)
        K = np.geomspace(1e-12 * E, self.kmax, n)
        d_full = delta_vm(E, mV, m)
        F = (K / E) ** b * (1.0 + delta_vm(K, mV, m) - b * np.log(K / E)) / (1.0 + d_full)
        F[-1] = max(F[-1], F[-2])
        if np.any(np.diff(F) < 0):
            raise RuntimeError("decay_rad: F(K) not monotonic -- spectrum broken")
        self.K, self.F = K, F / F[-1]

    def cdf(self, k):
        return np.interp(k, self.K, self.F)

    def sample(self, rng):
        return float(np.interp(rng.random(), self.F, self.K))


# ---------------------------------------------------------- matrix element
_G0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], complex)
_SX = np.array([[0, 1], [1, 0]], complex)
_SY = np.array([[0, -1j], [1j, 0]], complex)
_SZ = np.array([[1, 0], [0, -1]], complex)


def _gk(s):
    z = np.zeros((2, 2), complex)
    return np.block([[z, s], [-s, z]])


GAM = np.array([_G0, _gk(_SX), _gk(_SY), _gk(_SZ)])     # gamma^mu, Dirac rep
METRIC = np.array([1.0, -1.0, -1.0, -1.0])
ONE = np.eye(4, dtype=complex)


def _slash(p):
    """p-slash for a batch of 4-vectors (..., 4) -> (..., 4, 4)."""
    return np.einsum("...m,m,mij->...ij", p, METRIC, GAM)


def _dot(a, b):
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def amp_tensor(p1, p2, k, m=M_E):
    """O^{mu nu} (..., 4, 4, 4, 4): mu = V index, nu = photon index.
    p1 = l- (spinor u), p2 = l+ (spinor v)."""
    d1 = 2.0 * _dot(p1, k)
    d2 = 2.0 * _dot(p2, k)
    A = (_slash(p1 + k) + m * ONE) / d1[..., None, None]
    B = (-_slash(p2 + k) + m * ONE) / d2[..., None, None]
    # photon off l-:  gamma^nu S(p1+k) gamma^mu ;  off l+:  gamma^mu S(-p2-k) gamma^nu
    t1 = np.einsum("nab,...bc,mcd->...mnad", GAM, A, GAM)
    t2 = np.einsum("mab,...bc,ncd->...mnad", GAM, B, GAM)
    return t1 + t2


def me2_ab(a, b, M2, m2):
    """Spin- and polarisation-summed |M|^2 of V -> l-(p1) l+(p2) gamma(k) in
    closed form, a = p1.k, b = p2.k, M2 = V mass^2, m2 = lepton mass^2.

    Obtained from me2_trace (below) by fitting the exact polynomial
    |M|^2 a^2 b^2 at 50-digit precision (relative residual 1e-45); massless
    part = 8 (x1^2 + x2^2)/((1-x1)(1-x2)), the textbook gamma* -> f f gamma.
    The trace form is NOT usable in the generator: near the collinear peak it
    cancels terms ~1/(p.k)^2 against a result ~1/(p.k) and double precision
    loses everything (the angle-integrated spectrum came out x90 too high at
    k = 1 GeV).  This form has no such cancellation."""
    return (4.0 * (M2 * M2 - 2.0 * (a + b) * M2 + 2.0 * (a * a + b * b)
                   - 4.0 * (a + b) * m2 - 4.0 * m2 * m2) / (a * b)
            - 4.0 * m2 * (M2 + 2.0 * m2) * (1.0 / (a * a) + 1.0 / (b * b)))


def me2_trace(p1, p2, k, m=M_E):
    """Same quantity from the Dirac traces -- the DEFINITION, used only to
    validate me2_ab away from the collinear region.  Polarisation sums -g for
    both the V (current conserved) and the photon."""
    O = amp_tensor(p1, p2, k, m)
    Ob = np.einsum("ab,...mncb,cd->...mnad", _G0, O.conj(), _G0)  # gamma0 O^dag gamma0
    P1 = _slash(p1) + m * ONE
    P2 = _slash(p2) - m * ONE
    tr = np.einsum("...ab,...mnbc,...cd,...mnda->...mn", P1, O, P2, Ob)
    return np.real(np.einsum("m,n,...mn->...", METRIC, METRIC, tr))


def _frame_t(s, mV, t1, m=M_E):
    """l+l- rest frame, photon along +z, l- at angle theta* in the x-z plane,
    parametrised by t1 = 1 - beta* cos(theta*) so that the collinear region
    keeps its precision.  Returns p1 (l-), p2 (l+), k, a = p1.k, b = p2.k."""
    rs = np.sqrt(s)
    w = (mV * mV - s) / (2.0 * rs)
    Es = rs / 2.0
    ps = np.sqrt(max(Es * Es - m * m, 0.0))
    beta = ps / Es
    t1 = np.asarray(t1, float)
    t2 = 2.0 - t1
    c = (1.0 - t1) / beta
    # 1 - c^2 = (1-c)(1+c), each side computed from t without cancellation
    omc = (t1 - (1.0 - beta)) / beta
    opc = (t2 - (1.0 - beta)) / beta
    sn = np.sqrt(np.clip(omc * opc, 0.0, None))
    z = np.zeros_like(t1)
    k = np.stack([np.full_like(t1, w), z, z, np.full_like(t1, w)], -1)
    p1 = np.stack([np.full_like(t1, Es), ps * sn, z, ps * c], -1)
    p2 = np.stack([np.full_like(t1, Es), -ps * sn, z, -ps * c], -1)
    return p1, p2, k, Es * w * t1, Es * w * t2, beta


# ------------------------------------------------------------- the decay
def _boost(p4, beta):
    b2 = float(beta @ beta)
    if b2 < 1e-30:
        return p4.copy()
    g = 1.0 / np.sqrt(1.0 - b2)
    bp = float(beta @ p4[1:])
    E = g * (p4[0] + bp)
    v = p4[1:] + beta * ((g - 1.0) * bp / b2 + g * p4[0])
    return np.concatenate([[E], v])


def _basis(n):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


class DecayFSR:
    """V -> l- l+ (gamma) in the lab.  One instance per (mV, m); thread the
    caller's own rng so the production stream is untouched."""

    EPS = 1.0e-3        # GeV: below this the photon is merged into the leptons

    def __init__(self, mV, m=M_E, eps=None):
        self.mV, self.m = mV, m
        self.eps = self.EPS if eps is None else eps
        self.spec = PhotonEnergy(mV, m)
        self.n_try = self.n_acc = self.n_viol = 0
        self.max_viol = 1.0

    def _t_star(self, s, M, rng, safety=1.3):
        """Draw t1 = 1 - beta* cos(theta*) at fixed s from the exact |M|^2.

        Phase space at fixed s is flat in cos(theta*), hence in t1.  Proposal:
        the antenna 1/t1 + 1/t2 (t2 = 2 - t1), i.e. 1/(1-bc) + 1/(1+bc),
        sampled exactly as a 50/50 mixture of log-uniform t1 and t2."""
        m = self.m
        rs = np.sqrt(s)
        Es = rs / 2.0
        beta = np.sqrt(max(1.0 - 4.0 * m * m / s, 0.0))
        tlo, thi = 1.0 - beta, 1.0 + beta
        lo, hi = np.log(tlo), np.log(thi)
        M2, m2 = M * M, m * m

        def weight(t1):
            w = (M2 - s) / (2.0 * rs)
            a, b = Es * w * t1, Es * w * (2.0 - t1)
            return me2_ab(a, b, M2, m2) / (1.0 / t1 + 1.0 / (2.0 - t1))

        u = (np.arange(512) + 0.5) / 512
        tg = np.exp(lo + u * (hi - lo))
        tg = np.concatenate([tg, 2.0 - tg, [tlo, thi, 1.0]])
        C = safety * float(weight(tg).max())
        if not (np.isfinite(C) and C > 0):
            raise ValueError(f"decay_rad: no valid matrix-element bound at s = {s!r}")
        while True:
            t = np.exp(lo + rng.random(64) * (hi - lo))
            t = np.where(rng.random(64) < 0.5, t, 2.0 - t)
            wt = weight(t)
            self.n_try += len(t)
            if np.any(wt > C):
                self.n_viol += int((wt > C).sum())
                self.max_viol = max(self.max_viol, float((wt / C).max()))
            ok = np.nonzero(rng.random(len(t)) * C < wt)[0]
            if len(ok):
                self.n_acc += 1
                return float(t[ok[0]])

    def decay(self, pV, rng):
        """(l+ , l-, gamma or None) lab 4-vectors -- l+ first, as DECAY lists
        the jpsi daughters (-11, 11)."""
        mV, m = self.mV, self.m
        M2 = pV[0] ** 2 - float(pV[1:] @ pV[1:])
        if not np.isfinite(M2) or M2 <= 4.0 * m * m * (1.0 + 1e-9):
            # an accept/reject loop on NaN weights never terminates -- fail loudly
            raise ValueError(f"decay_rad: meson four-vector below the l+l- "
                             f"threshold (M^2 = {M2!r})")
        M = np.sqrt(M2)
        k = self.spec.sample(rng)
        k = min(k, 0.999999 * (M * M - 4.0 * m * m) / (2.0 * M))
        nV = rng.normal(size=3); nV /= np.linalg.norm(nV)       # photon dir, V frame
        if k < self.eps:
            # no resolvable photon: plain isotropic two-body
            ps = np.sqrt(M * M / 4.0 - m * m)
            Es = M / 2.0
            lm = np.concatenate([[Es], ps * nV])
            lp = np.concatenate([[Es], -ps * nV])
            bV = pV[1:] / pV[0]
            return _boost(lp, bV), _boost(lm, bV), None
        s = M * M - 2.0 * M * k
        t1 = self._t_star(s, M, rng)
        p1, p2, kk = _frame_t(s, M, np.array([t1]), m)[:3]
        p1, p2, kk = p1[0], p2[0], kk[0]
        # rotate: photon axis z -> nV, random azimuth of the lepton plane
        e1, e2 = _basis(nV)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        ex = np.cos(ph) * e1 + np.sin(ph) * e2
        ey = np.cross(nV, ex)
        R = np.stack([ex, ey, nV], axis=1)
        rot = lambda q: np.concatenate([[q[0]], R @ q[1:]])
        p1, p2, kk = rot(p1), rot(p2), rot(kk)
        # l+l- frame -> V frame: the pair recoils against the photon
        Ell = np.sqrt(s + k * k)
        b_ll = -k * nV / Ell
        p1, p2 = _boost(p1, b_ll), _boost(p2, b_ll)
        g = np.concatenate([[k], k * nV])
        bV = pV[1:] / pV[0]
        return _boost(p2, bV), _boost(p1, bV), _boost(g, bV)
