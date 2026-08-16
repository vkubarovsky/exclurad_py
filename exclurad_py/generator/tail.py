"""
Radiative tail in the SLICING organization -- the form an event generator
can sample from.

Why not the Bardin-Shumeiko form
--------------------------------
`core.radiative.sigma_F` returns the IR-SUBTRACTED hard-photon integral.
That is the right object for computing an RC factor, but its integrand

    (v/z^2) [ LR:H/Q2~^2  -  4 F_IR L0:H0/Q2^2 ]

is not positive definite -- the subtraction cancels the collinear
divergence pointwise -- so it is not a probability density and cannot be
sampled.  For generation we re-organize the SAME total at an auxiliary
slicing point v_min.

The decomposition (and a trap)
------------------------------
The naive slicing form

    sigma_soft = sigma_Born exp(delta_inf(v_min)) (1 + delta_VR + delta_vac)   # WRONG

is incorrect, and badly so.  Exponentiation resums multiple soft emission
up to the PHYSICAL cut v_max; re-exponentiating at an arbitrary v_min
resums to the wrong place and double-counts.  At a typical CLAS12 eta
point delta_inf(v_max) = -0.26 while delta_inf(1e-6 v_max) = -1.13, so
exp() is nowhere near linear and sigma_obs acquires a ~20% dependence on
the arbitrary v_min.

The correct split keeps the exponentiation at v_max and moves only the
IR-subtraction bookkeeping across the slicing point:

    sigma_hard(v_min) = int_{v_min}^{v_max} dv dOmega (v/z^2) LR:H/Q2~^2
                        [positive definite -- this is the density]

    sigma_soft(v_min) = sigma_obs(BS) - sigma_hard(v_min)
                        [everything else: virtual, soft real, and the
                         unresolved v < v_min slice]

which reproduces sigma_obs exactly, for any v_min, by construction.

Choosing v_min
--------------
v_min is not free in practice: sigma_soft must stay POSITIVE, or the
"no resolved photon" branch has negative probability.  sigma_soft falls
roughly like sigma_Born [exp(delta_inf) (1+delta_VR+delta_vac)
- 2(alpha/pi)(l_m - 1) ln(v_max/v_min)], so it goes negative for
v_max/v_min beyond ~1e5 at CLAS12 kinematics.  `choose_vmin` picks the
largest ratio that keeps the soft fraction above a floor, and generation
refuses to proceed if none exists.
"""

import numpy as np

from ..core.constants import ALPHA, GEV2_TO_NB
from ..core.spinors import dot4, rad_L, contractLH
from ..core.hadronic import hadronic_tensor
from ..core.kinematics import EventKin
from ..core.born import _sf_to_gev
from ..core.radiative import sigma_F
from ..core.virtual import delta_VR_inf, delta_vac
from .grid import ChannelGeometry, CellGrid

DEFAULT_VMIN_FRAC = 1e-3
MIN_SOFT_FRACTION = 0.02


class TailSampler:
    """Hard-photon density at one observed point, with sampling.

    Parameters
    ----------
    kin0     : Born EventKin (v = 0) at the observed (Q2, W2, t, phi)
    sfmodel  : structure-function model
    h        : beam helicity
    v_max    : upper inelasticity limit (missing-mass cut and/or kinematics)
    v_min    : slicing point; default DEFAULT_VMIN_FRAC * v_max
    safety   : majorant inflation factor per grid cell
    """

    def __init__(self, kin0, sfmodel, h, v_max, v_min=None,
                 n_v=20, n_x=32, n_phi=12, safety=3.0, n_sub=12, seed=12345):
        self.kin0, self.sf, self.h = kin0, sfmodel, h
        if hasattr(sfmodel, 'kf_W2Q2'):
            sfmodel.kf_W2Q2 = (float(kin0.W2), float(kin0.Q2))
        self.v_max = float(v_max)
        self.v_min = float(v_min) if v_min is not None \
            else DEFAULT_VMIN_FRAC * self.v_max
        if not (0.0 < self.v_min < self.v_max):
            raise ValueError("need 0 < v_min < v_max")
        self.n_v, self.n_x, self.n_phi = n_v, n_x, n_phi
        self.safety = float(safety)
        self.n_sub = int(n_sub)
        self._seed = int(seed)

        self.n_overweight = 0
        self.max_overweight = 1.0
        self.n_trials = 0
        self.n_accept = 0

        self._build_grid()

    # ------------------------------------------------------------------ #
    def _build_grid(self):
        k0 = self.kin0
        self.geoms = [ChannelGeometry(k0.k1[1:], k0.m),
                      ChannelGeometry(k0.k2[1:], k0.m)]
        self.grid = CellGrid(self.geoms, self.v_min, self.v_max,
                             self.n_v, self.n_x, self.n_phi,
                             safety=self.safety, n_sub=self.n_sub)
        self._raw_integral = self.grid.build(self._weighted_density,
                                             np.random.default_rng(self._seed))

    def _chan_weight(self, nhat, ic):
        """Importance weight of channel ic at direction nhat.

        The two channels each cover the whole sphere; the weights
        kappa_c^-2 / sum_c kappa_c^-2 sum to one, so the mixture reproduces
        the full angular integral exactly.
        """
        k0 = self.kin0
        E1, E2 = k0.k1[0], k0.k2[0]
        kh1 = np.maximum(E1 - nhat @ k0.k1[1:], 1e-30)
        kh2 = np.maximum(E2 - nhat @ k0.k2[1:], 1e-30)
        i1 = 1.0 / kh1 ** 2
        i2 = 1.0 / kh2 ** 2
        num = np.where(np.asarray(ic) == 0, i1, i2)
        return num / (i1 + i2)

    def _weighted_density(self, v, nhat, ic):
        f, _, _ = self._density_batch(v, nhat)
        return f * self._chan_weight(nhat, ic)

    # ------------------------------------------------------------------ #
    def _meson_at_v(self, v):
        """Detected-hadron four-vector(s) at inelasticity v -- vectorized.

        This is the v-dependent part of EventKin, inlined so that a batch of
        different v values costs one array operation instead of one Python
        object per value.  Constructing an EventKin per candidate photon is
        what makes a naive sampler ~100x too slow: numpy is fast on large
        arrays and ruinous on scalar-sized ones.
        """
        k0 = self.kin0
        v = np.atleast_1d(np.asarray(v, dtype=float))
        W, W2, Q2, t, mV, mU = k0.W, k0.W2, k0.Q2, k0.t, k0.mV, k0.mU
        q0, qv = k0.q[0], k0.q[3]
        M, Ep = k0.M, k0.p[0]

        Ev = (W2 + mV * mV - mU * mU - v) / (2.0 * W)
        if np.any(Ev < mV):
            raise ValueError("v too large: E_h < m_h")
        pvm = np.sqrt(np.maximum(Ev * Ev - mV * mV, 0.0))

        # The t convention MUST match EventKin's -- the three agree at v = 0 and
        # separate as v grows, which is precisely the regime this function is
        # for.  Hardcoding the photon form here while the pi0/eta topology
        # declares t_kind="target" walked the recoil proton off its measured t
        # as soon as a photon was emitted: soft events came out at <-t> = 0.308
        # (exact) and hard events at 2.968, with only 0.3% below -t = 0.5.
        # Born and soft events never showed it because they take their hadron
        # from EventKin, which does branch on t_kind.
        tk = getattr(k0, "t_kind", EventKin.T_PHOTON)
        if tk == EventKin.T_PHOTON:
            cv = (t + Q2 - mV * mV + 2.0 * q0 * Ev) / (2.0 * qv * pvm)
        elif tk == EventKin.T_TARGET:
            cv = (M * M + mV * mV - t - 2.0 * Ep * Ev) / (2.0 * qv * pvm)
        elif tk == EventKin.T_CMANGLE:
            # angle frozen at its v = 0 value; E and |p| still follow v
            Ev0 = (W2 + mV * mV - mU * mU) / (2.0 * W)
            pv0 = np.sqrt(max(Ev0 * Ev0 - mV * mV, 0.0))
            cv = np.full_like(pvm,
                              (M * M + mV * mV - t - 2.0 * Ep * Ev0)
                              / (2.0 * qv * pv0))
        else:
            raise ValueError(f"unknown t_kind {tk!r}")
        cv = np.clip(cv, -1.0, 1.0)
        sv = np.sqrt(np.maximum(1.0 - cv * cv, 0.0))
        pv = np.stack([Ev,
                       pvm * sv * np.cos(k0.phi),
                       pvm * sv * np.sin(k0.phi),
                       pvm * cv], axis=1)                      # (N,4)
        Lam = (k0.p + k0.q)[None, :] - pv
        return pv, Lam

    def _density_batch(self, v, nhat):
        """Positive integrand (v/z^2) LR:H/Q2~^2 for arrays v (N,), nhat (N,3).

        Fully vectorized: one call handles a whole block of candidates.
        """
        k0 = self.kin0
        q, p = k0.q, k0.p
        v = np.atleast_1d(np.asarray(v, dtype=float))
        nhat = np.atleast_2d(np.asarray(nhat, dtype=float))

        pv, Lam = self._meson_at_v(v)
        z = Lam[:, 0] - np.einsum('ni,ni->n', nhat, Lam[:, 1:])
        om = v / (2.0 * z)
        k4 = np.concatenate([om[:, None], om[:, None] * nhat], axis=1)

        qt = q[None, :] - k4
        Q2t = k0.Q2 + 2.0 * dot4(k4, q[None, :])
        W2t = k0.W2 - 2.0 * om * k0.W
        tt = k0.t - 2.0 * dot4(k4, q[None, :] - pv)

        good = (Q2t > 1e-6) & (z > 0) & (W2t > 0)
        sf = _sf_to_gev(self.sf(W2t, np.maximum(Q2t, 1e-6), tt))
        H = hadronic_tensor(qt, np.broadcast_to(p, pv.shape).copy(), pv,
                            W2t, sf, M=k0.M)
        LR, _, _ = rad_L(k0.k1, k0.k2, k0.m, self.h, k4)
        LRH = np.real(contractLH(LR, H))
        f = np.where(good, (v / z ** 2) * LRH / Q2t ** 2, 0.0)
        return np.maximum(f, 0.0), k4, om

    # ------------------------------------------------------------------ #
    @property
    def _prefactor(self):
        k0 = self.kin0
        return (ALPHA ** 3 / (512.0 * np.pi ** 4 * k0.lam_S
                              * np.sqrt(k0.lam_q))) * GEV2_TO_NB

    def sigma_hard(self):
        """Hard-photon cross section above v_min (nb/GeV^6), >= 0.

        This is the integral of the sampling density, so the branching
        fraction and the generated spectrum are automatically consistent.
        """
        return self._prefactor * self._raw_integral

    def sigma_hard_reference(self):
        """Same quantity from the core quadrature -- a cross-check of the
        sampler's own grid (they use the same nodes, different assembly)."""
        return sigma_F(self.kin0, self.sf, self.h, self.v_max,
                       v_lo=self.v_min, subtract=False)

    def split(self, sigma_obs):
        """(soft, hard, hard_fraction) given the exact sigma_obs from rc_factor.

        sigma_soft is defined as the remainder, so soft + hard == sigma_obs
        identically for any v_min.  A negative soft part means v_min is too
        small -- see choose_vmin.
        """
        hard = self.sigma_hard()
        soft = sigma_obs - hard
        frac = hard / sigma_obs if sigma_obs != 0 else 0.0
        return soft, hard, frac

    # ------------------------------------------------------------------ #
    def sample(self, rng, n=1):
        """Draw n photons from the hard density.

        Returns dict with v (n,), nhat (n,3), omega (n,), k4 (n,4).
        """
        if self.grid.cdf is None:
            raise RuntimeError("hard density is identically zero at this point")

        v_acc, n_acc, k4_acc, om_acc = [], [], [], []
        got = 0
        # Oversample per pass: one vectorized pass over a block costs about
        # the same as a single scalar evaluation, so the block IS the unit
        # of work -- never loop event by event.
        while got < n:
            need = n - got
            block = max(1024, int(2.0 * need / max(self._eff_guess(), 0.02)))
            block = min(block, 100_000)

            v, nhat, ic, M, jac = self.grid.propose(rng, block)

            f, k4, om = self._density_batch(v, nhat)
            f = f * self._chan_weight(nhat, ic) * jac     # same measure as build
            self.n_trials += block

            over = f > M
            if np.any(over):
                self.n_overweight += int(np.count_nonzero(over))
                self.max_overweight = max(
                    self.max_overweight,
                    float(np.max(f[over] / np.maximum(M[over], 1e-300))))

            keep = rng.random(block) * M <= f
            k = np.where(keep)[0][:need]
            if len(k):
                v_acc.append(v[k]); n_acc.append(nhat[k])
                k4_acc.append(k4[k]); om_acc.append(om[k])
                got += len(k)
                self.n_accept += len(k)

        return dict(v=np.concatenate(v_acc)[:n],
                    nhat=np.concatenate(n_acc)[:n],
                    omega=np.concatenate(om_acc)[:n],
                    k4=np.concatenate(k4_acc)[:n])

    def _eff_guess(self):
        if self.n_trials > 0 and self.n_accept > 0:
            return self.n_accept / self.n_trials
        return 0.15

    def report(self):
        eff = self.n_accept / self.n_trials if self.n_trials else float("nan")
        return dict(v_min=self.v_min, v_max=self.v_max,
                    sigma_hard=self.sigma_hard(),
                    trials=self.n_trials, accepted=self.n_accept,
                    efficiency=eff,
                    n_overweight=self.n_overweight,
                    max_overweight=self.max_overweight)


# --------------------------------------------------------------------------- #
def choose_vmin(kin0, sfmodel, h, v_max, sigma_obs, sigma_born=None,
                target_soft=0.35, min_soft=MIN_SOFT_FRACTION,
                min_ratio=1e-5, max_ratio=1e-1):
    """Pick the slicing point ANALYTICALLY -- no quadrature.

    Building TailSamplers to search for v_min costs ~0.4 s each and is
    unnecessary: the soft branch has a closed form.  Since

        sigma_hard(v_min) = int_{v_min}^{v_max} exact
                          ~ sigma_F(BS) + sigma_Born * [delta_inf(v_max)
                                                        - delta_inf(v_min)]

    and delta_inf is logarithmic with slope c = 2 (alpha/pi) (l_m - 1),

        soft_frac(r) ~ 1 - [sigma_F + sigma_Born c ln(1/r)] / sigma_obs,
        r = v_min / v_max,

    which inverts directly for the r giving a target soft fraction.  The
    result is clipped to [min_ratio, max_ratio] and rejected if the implied
    soft fraction falls below `min_soft` (a negative soft branch means
    negative probability -- see the module docstring).

    Returns (v_min, soft_fraction) or (None, None).
    """
    from ..core.born import born_xsec

    if sigma_born is None:
        sigma_born = born_xsec(kin0, sfmodel, h)
    if sigma_born <= 0 or sigma_obs <= 0:
        return None, None

    lm = np.log(kin0.Q2 / kin0.m ** 2)
    c = 2.0 * (ALPHA / np.pi) * (lm - 1.0)
    if c <= 0:
        return None, None

    # sigma_hard at r -> 1 is just sigma_F(BS); everything above it is the
    # subtraction moved across the slicing point.
    sigma_F_bs = sigma_obs - sigma_born * _virtual_factor(kin0, v_max)

    def soft_frac(r):
        hard = sigma_F_bs + sigma_born * c * np.log(1.0 / r)
        return 1.0 - hard / sigma_obs

    # invert soft_frac(r) = target_soft
    want_hard = (1.0 - target_soft) * sigma_obs
    ln_inv_r = (want_hard - sigma_F_bs) / (sigma_born * c)
    r = float(np.exp(-ln_inv_r)) if np.isfinite(ln_inv_r) else max_ratio
    r = float(np.clip(r, min_ratio, max_ratio))

    sfrac = soft_frac(r)
    if not np.isfinite(sfrac) or sfrac < min_soft:
        # walk up towards larger v_min (smaller hard branch)
        for rr in (3e-2, 1e-1, 3e-1):
            if soft_frac(rr) >= min_soft:
                return rr * v_max, float(soft_frac(rr))
        return None, None
    return r * v_max, float(sfrac)


def _virtual_factor(kin0, v_max):
    """exp(delta_inf) (1 + delta_VR + delta_vac) at the physical cut."""
    d_VR, d_inf = delta_VR_inf(kin0, v_max)
    return float(np.exp(d_inf) * (1.0 + d_VR + delta_vac(kin0.Q2)))


def check_vmin_independence(kin0, sfmodel, h, v_max, sigma_obs,
                            fracs=(1e-2, 1e-3, 1e-4, 1e-5), **kw):
    """sigma_soft + sigma_hard must equal sigma_obs for every v_min.

    With the remainder definition of sigma_soft this is exact by
    construction, so what this actually tests is whether the soft branch
    stays POSITIVE -- i.e. which slicing points are usable.  Returns rows
    of (v_min, soft, hard, soft_fraction, usable).
    """
    rows = []
    for f in fracs:
        ts = TailSampler(kin0, sfmodel, h, v_max, v_min=f * v_max, **kw)
        soft, hard, hf = ts.split(sigma_obs)
        rows.append((ts.v_min, soft, hard, 1.0 - hf, soft > 0))
    return rows


# --------------------------------------------------------------------------- #
def photon_at(kin, v, nhat):
    """Rebuild the photon and detected hadron at THIS event's kinematics.

    A TailSampler built at a nearby point supplies the *distribution* of
    (v, nhat), but the photon energy omega = v / (2 z) depends on
    z = Lam0 - vec(Lam).nhat, which is a property of the event, not of the
    cell the sampler was built in.  Using the sampler's own omega for a
    different event breaks four-momentum conservation -- in testing, 65% of
    events failed closure until this was applied.

    Returns (k4, meson4) in the CM, or None if the point is unphysical.
    """
    ts = TailSampler.__new__(TailSampler)
    ts.kin0 = kin
    try:
        pv, Lam = ts._meson_at_v(v)
    except ValueError:
        return None
    nhat = np.asarray(nhat, dtype=float)
    z = float(Lam[0, 0] - nhat @ Lam[0, 1:])
    if z <= 0:
        return None
    om = v / (2.0 * z)
    return np.concatenate([[om], om * nhat]), pv[0]
