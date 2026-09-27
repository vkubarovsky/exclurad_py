"""Fermi motion of the struck nucleon in a deuteron target (PWIA).

Scope: unpolarised deuteron, spectator on shell, struck nucleon off shell by
the spectator prescription, no final-state interactions.

The deuteron has exactly two constituents and nothing else, so drawing the
spectator momentum fixes everything:

    spectator   on shell    E_s = sqrt(M_N^2 + p_s^2),  p_s ~ n(k), isotropic
    struck      off shell   E*  = M_d - E_s,            p*  = -p_s

    struck + spectator = (M_d, 0, 0, 0)          exactly, by construction

so the generator never invents energy and the written event balances at
machine precision.  The struck nucleon must be off shell: the deuteron has no
third body to carry the binding away, and forcing both nucleons on shell adds
~9 MeV to the initial state (which is what the RG-C java does; see the note).

WHY m* ENTERS THE KINEMATICS.  frames.assemble_event builds the event in a
frame with the target AT REST with mass M, and closes four-momentum on the
recoil.  A Lorentz map from that frame onto a lab nucleon of invariant mass m*
exists only if M = m*.  Keeping M = M_N while placing an off-shell nucleon in
the lab would break beam + d = e' + meson + N' + N_s by ~20 MeV.  So the
KINEMATICS use m* (per event) while the CROSS SECTION and the RADIATIVE
CORRECTIONS stay at M_N -- standard PWIA with on-shell extrapolation, which is
the only thing the amp2609 models know.  That split is deliberate; see
FERMI_NOTE.md.

The wave function is the Argonne v18 deuteron momentum distribution, tabulated
by R. B. Wiringa (phy.anl.gov/theory/research/momenta, h2.momentum) as
rho(k)/(2 pi)^3.  It is normalised per NUCLEON, Int 4 pi k^2 n(k) dk = 1, which
is what makes sigma_nb a per-nucleon cross section.
"""

from pathlib import Path

import numpy as np

from ..core.constants import M_N, M_P
from .frames import boost_matrix, rotation_to_z

M_D = 1.87561294                  # deuteron mass, CODATA
HBARC = 0.19732                   # GeV*fm -- the value the n(k) table assumes

_TABLE = Path(__file__).resolve().parent.parent / "data" / "nk_d2_av18.dat"

# struck -> (m_struck, m_spectator, pid_spectator)
CONFIG = {"p": (M_P, M_N, 2112),
          "n": (M_N, M_P, 2212)}

WAVEFUNCTIONS = ("av18",)


def spectator_limit(m_spec):
    """Largest |p_s| for which the struck nucleon is not spacelike.

    m*^2 = (M_d - E_s)^2 - p_s^2 falls with p_s and crosses zero at
    E_s = (M_d^2 + m_spec^2) / (2 M_d).  Above that the spectator prescription
    has no meaning at all, so this is a hard ceiling, not a preference.  It
    costs 0.4% of the AV18 distribution.
    """
    e = (M_D ** 2 + m_spec ** 2) / (2.0 * M_D)
    return np.sqrt(e ** 2 - m_spec ** 2)


# The smaller of the two configurations (spectator neutron, 0.70247 GeV/c),
# so one ceiling is safe whichever nucleon is struck.
P_MAX = min(spectator_limit(M_N), spectator_limit(M_P))
K_MAX = P_MAX / HBARC             # 3.5601 fm^-1


class FermiSampler:
    """Draw |p_s| from 4 pi k^2 n(k) by inverting the cumulative.

    The table is piecewise linear in F(k) = 4 pi k^2 n(k), so the cumulative is
    piecewise quadratic and the inversion solves a quadratic exactly.  Building
    the cdf from the same trapezoid areas keeps the two consistent.
    """

    def __init__(self, name="av18", k_max=K_MAX, path=None):
        if name not in WAVEFUNCTIONS:
            raise ValueError(f"unknown deuteron wave function {name!r}; "
                             f"known: {', '.join(WAVEFUNCTIONS)}")
        self.name = name
        k, nk = np.loadtxt(path or _TABLE, unpack=True)
        if k_max is not None:
            # Interpolate an endpoint exactly AT k_max rather than snapping to
            # the last grid node below it: the ceiling is a physics boundary
            # (m* -> 0), not a property of the table's 0.1 fm^-1 grid.
            keep = k <= k_max - 1e-12
            if keep.sum() < 2:
                raise ValueError(f"k_max = {k_max} leaves too few grid points")
            n_end = float(np.interp(k_max, k, nk))
            k = np.append(k[keep], float(k_max))
            nk = np.append(nk[keep], n_end)
        self.k, self.nk, self.k_max = k, nk, float(k[-1])
        F = 4.0 * np.pi * k ** 2 * nk
        cell = 0.5 * (F[:-1] + F[1:]) * np.diff(k)
        cdf = np.concatenate(([0.0], np.cumsum(cell)))
        self.total = float(cdf[-1])
        self.cdf = cdf / self.total
        self.f = F / self.total

    # ------------------------------------------------------------------ #
    def draw_k(self, rng):
        """One k [fm^-1].  Consumes exactly ONE uniform."""
        r = float(rng.random())
        i = int(np.searchsorted(self.cdf, r, side="right")) - 1
        i = min(max(i, 0), len(self.k) - 2)
        f1, f2 = self.f[i], self.f[i + 1]
        h = self.k[i + 1] - self.k[i]
        s = (f2 - f1) / h
        target = r - self.cdf[i]
        denom = f1 + np.sqrt(max(f1 * f1 + 2.0 * s * target, 0.0))
        d = 2.0 * target / denom if denom > 0.0 else 0.0
        return self.k[i] + min(max(d, 0.0), h)

    def draw(self, rng):
        """One spectator three-momentum (3,) in GeV/c, isotropic.

        Consumes exactly THREE uniforms: magnitude, cos(theta), phi.  The count
        is fixed and documented because the byte-identity test depends on the
        RNG stream being untouched when the flag is off -- and on it being
        reproducible when it is on.
        """
        p = self.draw_k(rng) * HBARC
        cos_t = 2.0 * float(rng.random()) - 1.0
        sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        phi = 2.0 * np.pi * float(rng.random())
        return np.array([p * sin_t * np.cos(phi),
                         p * sin_t * np.sin(phi),
                         p * cos_t])


# --------------------------------------------------------------------------- #
def flux_factor(struck4):
    """Moller flux factor for a massless beam along +z on a moving nucleon.

        F = sqrt((k.p)^2 - m_k^2 m_N^2) / (E_k E_N)  ->  (E_N - p_N,z) / E_N
          = 1 - beta_z

    It is the relative velocity of the two colliding particles: the rate per
    beam particle per nucleon in the laboratory is sigma * F, not sigma alone.
    A nucleon running toward the beam meets more beam particles per unit time.
    For a nucleon at rest F is identically 1, which is why a free-nucleon run
    is untouched -- and why acceptance criterion 1 still holds.

    Why it cannot be dropped although <F> = 1 exactly over an isotropic
    distribution: F is strongly correlated with the collision energy, through

        s - m*^2 = 2 E_beam E* F                     (exact)

    so F enters s linearly -- measured correlation +0.947 over AV18 at
    10.6 GeV.  A nucleon moving toward the beam has both a larger F and a
    larger s, hence a larger sigma wherever sigma rises with s, and
    <F sigma> != <F> <sigma>.  Reported from branch jpsi-fermi on 2026-09-26,
    where near the J/psi threshold sigma(W) is steep and the correction is
    +4% on the deuteron yield; at the pi0/eta W of 2-3 GeV sigma varies slowly
    and the effect is at the per-cent level, concentrated in the tail where
    F departs most from 1.
    """
    a = np.atleast_2d(np.asarray(struck4, dtype=float))
    out = (a[..., 0] - a[..., 3]) / a[..., 0]
    return out[0] if np.ndim(struck4) == 1 else out


def flux_max(m_star_min, struck="p"):
    """Largest F reachable with m* >= m_star_min -- the rejection ceiling.

    F is maximal head on (spectator downstream, struck nucleon against the
    beam) and grows as m* falls, because beta -> 1.  With no floor on m* it
    reaches exactly 2 at the spacelike limit; with the generator's default
    floor of 0.30 GeV it is 1.911.  The ceiling matters because the flux
    factor is applied by rejection: too low silently truncates the tail, too
    high only costs draws.
    """
    _, m_spec, _ = CONFIG[struck]
    lo, hi = 0.0, spectator_limit(m_spec)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if split_deuteron([0.0, 0.0, mid], struck)[2] > m_star_min:
            lo = mid
        else:
            hi = mid
    return float(flux_factor(split_deuteron([0.0, 0.0, lo], struck)[0]))


def split_deuteron(p_spec, struck="p"):
    """(struck4, spectator4, m_star) from the spectator three-momentum.

    struck = 'p' -> struck proton, spectator neutron; 'n' the other way round.
    struck4 + spectator4 = (M_d, 0, 0, 0) exactly.
    """
    try:
        _, m_spec, _ = CONFIG[struck]
    except KeyError:
        raise ValueError(f"struck must be 'p' or 'n', not {struck!r}") from None
    p_spec = np.asarray(p_spec, dtype=float)
    e_spec = np.sqrt(m_spec ** 2 + float(np.dot(p_spec, p_spec)))
    spectator4 = np.array([e_spec, *p_spec])
    struck4 = np.array([M_D - e_spec, *(-p_spec)])
    m2 = struck4[0] ** 2 - float(np.dot(struck4[1:], struck4[1:]))
    return struck4, spectator4, np.sqrt(max(m2, 0.0))


def working_frame(E_lab, struck4):
    """Map the lab onto the frame the generator already knows how to work in.

    Returns (E_work, to_lab) where

      E_work  is the beam energy in the struck nucleon's rest frame -- the
              value that goes into self._E, exactly as the target straggling
              already supplies a per-event energy;
      to_lab  is the 4x4 matrix taking a working-frame four-vector back to the
              laboratory.

    The working frame is reached by boosting to the struck nucleon's rest frame
    and then rotating the boosted beam onto +z, which is precisely the
    configuration frames.assemble_event delivers: target at rest, beam along z.
    """
    beam_lab = np.array([E_lab, 0.0, 0.0, E_lab])       # electron mass ignored
    beta = struck4[1:] / struck4[0]
    B = boost_matrix(beta)                              # lab -> nucleon rest
    k_rest = B @ beam_lab
    R4 = np.eye(4)
    R4[1:, 1:] = rotation_to_z(k_rest[1:])              # beam -> +z
    to_lab = boost_matrix(-beta) @ R4.T
    return float(k_rest[0]), to_lab
