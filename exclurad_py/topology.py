"""
Detection topology: which hadron is DETECTED and which is MISSING.

This is the third axis of the package (channel x model-vintage x topology)
and the one that silently produces plausible-but-wrong numbers when it is
wrong, so it is explicit here rather than buried inside a model adapter.

Background
----------
The core RC machinery is written for

    e(k1) p(p) -> e'(k2) h(pv) B(pu) [gamma(k)]

where `h` (mass mV) is the hadron whose momentum is *used to define the
kinematics*, and `B` (mass mU) is the *undetected* system whose mass enters

    v = MX^2 - mU^2

and the Li2 term of delta_VR.  Two conventions are in use in CLAS
analyses and they are NOT interchangeable:

MESON_DETECTED  (DVMP convention, used by DIFFRAD and by most vector-meson
                 analyses)
    detected   = the meson         -> mV = m_meson
    undetected = the recoil baryon -> mU = m_p (or m_n for pi+ n)
    t passed to the SF model is the kinematic t = (q - p_meson)^2.

BARYON_DETECTED (EXCLURAD convention for pi0 / eta)
    detected   = the recoil proton -> mV = m_p
    undetected = the meson         -> mU = m_meson
    the missing-mass cut is on the meson:  v = MX^2 - m_meson^2.
    The SF model is still a function of the NUCLEON transfer
    t_N = (p - p')^2, which is NOT the t that the core kinematics builds
    from the detected proton, so the model must be evaluated at a t_N
    supplied independently (`sf_t = FIXED_NUCLEON`).
    In addition the hadron-plane x axis is defined by the detected
    hadron, so swapping meson <-> proton flips it: sigma_LT and
    sigma_LT' change sign, sigma_T/L/TT do not.

Getting `lt_sign` or `sf_t` wrong changes the phi-dependence of the RC by
percent-level amounts with no other symptom, which is why both are
recorded on the Topology object and stamped into every output file.
"""

from dataclasses import dataclass, field
from math import pi

from .core.constants import M_P, M_N

# how the SF model's t argument is obtained
KINEMATIC_T = "kinematic_t"        # t of the core kinematics (detected hadron)
FIXED_NUCLEON = "fixed_nucleon_t"  # t_N = (p-p')^2, supplied by the caller


@dataclass(frozen=True)
class Topology:
    """Which hadron is detected, which is missing, and the SF conventions.

    Attributes
    ----------
    name        : short label, stamped into outputs (e.g. "meson_detected")
    m_detected  : mass of the hadron defining the kinematics  -> EventKin mV
    m_missing   : mass of the undetected system               -> EventKin mU
    sf_t        : KINEMATIC_T or FIXED_NUCLEON (see module docstring)
    lt_sign     : +1 or -1, multiplies sigma_LT and sigma_LT' to account for
                  the hadron-plane x-axis flip
    detected    : human-readable name of the detected hadron
    missing     : human-readable name of the missing hadron
    note        : free-text provenance / caveat
    """

    name: str
    m_detected: float
    m_missing: float
    sf_t: str = KINEMATIC_T
    lt_sign: float = +1.0
    phi_offset: float = 0.0
    detected: str = ""
    missing: str = ""
    t_kind: str = "photon"     # which invariant the t bin holds fixed
    note: str = ""

    def __post_init__(self):
        if self.sf_t not in (KINEMATIC_T, FIXED_NUCLEON):
            raise ValueError(f"unknown sf_t: {self.sf_t!r}")
        if self.lt_sign not in (+1.0, -1.0):
            raise ValueError("lt_sign must be +1 or -1")

    @property
    def needs_external_t(self) -> bool:
        """True if the SF model cannot use the core kinematic t."""
        return self.sf_t == FIXED_NUCLEON

    def stamp(self) -> dict:
        """Provenance dict written into result files."""
        return dict(topology=self.name, detected=self.detected,
                    missing=self.missing, m_detected=self.m_detected,
                    m_missing=self.m_missing, sf_t=self.sf_t,
                    lt_sign=self.lt_sign, phi_offset=self.phi_offset,
                    t_kind=self.t_kind)

    def __str__(self):
        return (f"{self.name}(detected={self.detected} m={self.m_detected:.6f}, "
                f"missing={self.missing} m={self.m_missing:.6f}, "
                f"sf_t={self.sf_t}, lt_sign={self.lt_sign:+.0f})")


# --------------------------------------------------------------------------- #
# factories
# --------------------------------------------------------------------------- #
def meson_detected(m_meson, meson="meson", m_recoil=M_P, recoil="p"):
    """DVMP convention: meson detected, recoil baryon missing."""
    return Topology(name="meson_detected",
                    m_detected=m_meson, m_missing=m_recoil,
                    sf_t=KINEMATIC_T, lt_sign=+1.0, phi_offset=0.0,
                    detected=meson, missing=recoil,
                    note="t of the core kinematics is the meson transfer; "
                         "hadron plane defined by the meson.")


def baryon_detected(m_meson, meson="meson", m_recoil=M_P, recoil="p"):
    """EXCLURAD convention for pi0/eta: recoil proton detected, meson missing.

    The SF model must be evaluated at the nucleon transfer t_N, supplied by
    the caller (see exclurad_py.models.base.ProtonDetectedModel), and sigma_LT,
    sigma_LT' flip sign relative to the meson frame.
    """
    return Topology(name="baryon_detected",
                    m_detected=m_recoil, m_missing=m_meson,
                    sf_t=FIXED_NUCLEON, lt_sign=-1.0, phi_offset=pi,
                    detected=recoil, missing=meson,
                    # DEFAULT = the physical bin definition.
                    #
                    #   t = (p - p')^2 with p the target at rest and p' the
                    #   MEASURED recoil proton.  Radiation is from the lepton
                    #   line only (hadron-line radiation is neglected here as
                    #   in EXCLURAD/DIFFRAD, and the proton is heavy), so the
                    #   photon does not touch p'.  t is therefore fixed by the
                    #   measurement and cannot shift: only Q2~ and W2~ do.
                    #   At fixed (Q2, W2, t, phi) it is the proton's DIRECTION
                    #   that moves as v runs, not its lab energy.
                    #
                    # t_kind="cm_angle" is an EXCLURAD-COMPATIBILITY mode: it
                    # freezes cos(theta_CM) instead, which lets t drift ~5.5 %
                    # of |t| across the window.  Use it to reproduce published
                    # tables (exclurad_py matches EXCLURAD to 0.15-0.20 % there);
                    # do not use it for new physics.  The two differ by up to
                    # 13-17 % in the ~10 % of bins where v_max(t) < v_cut.
                    t_kind="target",
                    note="EXCLURAD pi0/eta configuration: v = MX^2 - m_meson^2; "
                         "SFs at fixed t_N=(p-p')^2; hadron-plane x axis "
                         "flipped -> sigma_LT, sigma_LT' change sign.")


def pip_neutron(m_pip):
    """e p -> e' pi+ n : pi+ detected, NEUTRON missing (mU = M_N)."""
    return meson_detected(m_pip, meson="pi+", m_recoil=M_N, recoil="n")
