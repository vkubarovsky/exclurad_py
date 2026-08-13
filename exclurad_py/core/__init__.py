"""exclurad_py.core -- channel- and model-agnostic RC machinery.

FROZEN.  This subpackage implements the exact covariant (Bardin-Shumeiko)
QED radiative correction for

    e(k1) p(p) -> e'(k2) h(pv) B(pu) [gamma(k)]

with the hadronic side described by the five structure functions
sigma_T, sigma_L, sigma_TT, sigma_LT, sigma_LT' and beam helicity h.

Nothing here knows about a particular meson, a particular cross-section
model, or a particular detection topology: those live in exclurad_py.topology
and exclurad_py.models.  Change this subpackage only for *physics* reasons
(hadron-line radiation, two-photon exchange, exponentiation scheme) --
never to accommodate a new channel or a refit.  Every change here
invalidates the validation suite and must be re-validated.

Provenance: imported unchanged from the `vmrad` package delivered
2026-08-11 (2026_full_pseudoscalar_RC_model), except that the structure
function *interface* classes were moved to exclurad_py.models.base.
"""

from .constants import *          # noqa: F401,F403
from .kinematics import EventKin, kallen
from .hadronic import hadronic_tensor, photon_basis, NU_LT, NU_LTP
from .born import born_xsec, born_reference, born_tensors, born_H
from .virtual import delta_VR_inf, delta_vac, delta_vac_lepton, delta_vac_hadron
from .radiative import sigma_F
from .rc import rc_factor

CORE_VERSION = "1.0.0"

__all__ = [
    "EventKin", "kallen", "hadronic_tensor", "photon_basis",
    "NU_LT", "NU_LTP", "born_xsec", "born_reference", "born_tensors",
    "born_H", "delta_VR_inf", "delta_vac", "delta_vac_lepton",
    "delta_vac_hadron", "sigma_F", "rc_factor", "CORE_VERSION",
]
