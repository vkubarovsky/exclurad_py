"""
pi0 and eta structure-function models, all vintages.

Registry keys
-------------
    pi0.vpk2014     2013/2014 EXCLURAD built-in fit (dvmpx.F, k=1)
    pi0.vpk2021     March-2021 GT fit (dvmpx.F XSINIT, k=1)
    eta.tableB1     analysis-note CLAS-2016-101 App. B Table B.1 -- the
                    PUBLISHED eta production model (PRC 95, 035202)
    eta.refit2013   later 2013-form eta refit (reference only, NOT published)
    eta.vpk2021     March-2021 GT fit (dvmpx.F XSINIT, k=2)

All are in the EXCLURAD topology: the recoil PROTON is detected, the meson
is missing, v = MX^2 - m_meson^2, and the fits are functions of the nucleon
transfer t_N -- so they are constructed with the bin's t_N:

    sf = models.get("eta.tableB1", t_nucl=-0.2)

The sigma_LT sign flip of the detected-proton hadron plane is applied by
the base class from `topology.lt_sign`, not here.
"""

from . import _forms as F
from .base import ProtonDetectedModel
from .registry import register
from ..core.constants import M_PI0, M_ETA
from ..topology import baryon_detected

# EXCLURAD-2014 uses the CHARGED pion mass for the missing-mass variable amu
# even for pi0 -- reproduced here so the v definition matches the Fortran.
M_PI_EXCL = 0.13957

# meson mass as hardcoded in dvmpx.F AM(2) -- used inside the fit's tminq
M_ETA_FIT = 0.547300
M_PI0_FIT = 0.134976

_PI0_TOPO = baryon_detected(M_PI_EXCL, meson="pi0")
_ETA_TOPO = baryon_detected(M_ETA, meson="eta")

# The fits were made on the CLAS12 pi0/eta grids; outside them the radiative
# tail would extrapolate a power law with no data behind it.
_PS_VALIDITY = dict(W2=(1.15, 20.0), Q2=(0.5, 12.0), mt=(0.0, 5.0))


# --------------------------------------------------------------------------- #
#  pi0
# --------------------------------------------------------------------------- #
@register
class Pi0_VPK2014(ProtonDetectedModel):
    name = "pi0.vpk2014"
    channel = "pi0"
    version = "2014"
    topology = _PI0_TOPO
    validity = _PS_VALIDITY
    provenance = (
        "Port of dvmpx.F XSIGMA_T/L/TT/LT with pi0eta.par column k=1, as "
        "built into exclurad-2014 (copy: ~/eta-rc-2026/model2013note/). "
        "Verified against the Fortran by validation/verify_ports.py.")
    reference = "EXCLURAD-2014 built-in fit; ~/OneDrive/Work/2026_RC_test_comparison/exclurad/"

    def _sfs_tn(self, t_nucl, xB, Q2):
        return F.dvmp2013_form(t_nucl, xB, Q2, F.P2013_PI0, M_PI0_FIT)


@register
class Pi0_VPK2021(ProtonDetectedModel):
    name = "pi0.vpk2021"
    channel = "pi0"
    version = "2021"
    topology = _PI0_TOPO
    validity = _PS_VALIDITY
    provenance = (
        "Port of the March-2021 GT/transversity fit (dvmpx.F XSINIT P(12,2) "
        "column 1) as built into exclurad-2026 / ~/exclugen/stock_run. "
        "EXCLURAD's dvmpw wrapper DOUBLES sigma_LT; that factor is applied "
        "here so the model matches the Fortran as run. sigma_L = 0 in this fit.")
    reference = "dvmpx.F March 2021; ~/exclurad-2026/"

    def _sfs_tn(self, t_nucl, xB, Q2):
        sT, sL, sTT, sLT = F.vpk2021_form(t_nucl, xB, Q2, F.P2021_PI0, M_PI0_FIT)
        return sT, sL, sTT, 2.0 * sLT      # dvmpw wrapper factor 2


# --------------------------------------------------------------------------- #
#  eta
# --------------------------------------------------------------------------- #
@register
class Eta_TableB1(ProtonDetectedModel):
    name = "eta.tableB1"
    channel = "eta"
    version = "2016note"
    topology = _ETA_TOPO
    validity = _PS_VALIDITY
    provenance = (
        "2013 functional form with the eta column of pi0eta.par set to the "
        "CLAS-2016-101 analysis-note Appendix B Table B.1 PRODUCTION values "
        "(sigT=204.99, sigL=1392.1, sigTT=-3083.0, sigLT=271.6, "
        "alphaTT=1.101, common /(Q2+1)^2). This is the model that reproduces "
        "the published RC table of PRC 95, 035202 at vcut=0.20 "
        "(endpoints +-0.001, mid-phi +0.007; see ~/eta-rc-2026/README.md).")
    reference = ("Bedlinskiy et al., PRC 95, 035202 (2017); analysis note "
                 "CLAS-2016-101 App. B Table B.1; ~/eta-rc-2026/model2013note/")

    def _sfs_tn(self, t_nucl, xB, Q2):
        return F.dvmp2013_form(t_nucl, xB, Q2, F.P_ETA_TABLE_B1, M_ETA_FIT)


@register
class Eta_Refit2013(ProtonDetectedModel):
    name = "eta.refit2013"
    channel = "eta"
    version = "2013refit"
    topology = _ETA_TOPO
    validity = _PS_VALIDITY
    provenance = (
        "2013 functional form with the eta column of the eta-rc-2026 repo "
        "default pi0eta.par -- a LATER refit than Table B.1 "
        "(sigT=42.565, alphaT=0.0624, sigL=1743.5, n=2.1656). "
        "REFERENCE ONLY: this is not the model behind the published RC.")
    reference = "~/eta-rc-2026/model2013/pi0eta.par"

    def _sfs_tn(self, t_nucl, xB, Q2):
        return F.dvmp2013_form(t_nucl, xB, Q2, F.P_ETA_2013_REFIT, M_ETA_FIT)


@register
class Eta_VPK2021(ProtonDetectedModel):
    name = "eta.vpk2021"
    channel = "eta"
    version = "2021"
    topology = _ETA_TOPO
    validity = _PS_VALIDITY
    provenance = (
        "Port of the March-2021 GT/transversity fit (dvmpx.F XSINIT P(12,2) "
        "column 2, eta). EXCLURAD's dvmpw wrapper factor 2 on sigma_LT is "
        "applied. sigma_L = 0 in this fit.")
    reference = "dvmpx.F March 2021; ~/eta-rc-2026/model2021/"

    def _sfs_tn(self, t_nucl, xB, Q2):
        sT, sL, sTT, sLT = F.vpk2021_form(t_nucl, xB, Q2, F.P2021_ETA, M_ETA_FIT)
        return sT, sL, sTT, 2.0 * sLT
