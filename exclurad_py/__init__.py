"""
exclurad_py -- QED radiative corrections and event generation for exclusive
meson electroproduction at CLAS12.

    e(k1) p(p) -> e'(k2) h(pv) B(pu) [gamma(k)]      h = rho0, phi, J/psi, pi0, eta

One RC engine (exact covariant Bardin-Shumeiko, full five structure
functions sigma_T, sigma_L, sigma_TT, sigma_LT, sigma_LT', beam helicity)
serving all channels, with the cross-section model, the meson, and the
detection topology as plug-in axes rather than compile-time choices.

    from exclurad_py import models, rc_factor
    sf  = models.get("eta.tableB1", t_nucl=-0.2)
    res = rc_factor(Ebeam=10.6, Q2=1.5, W2=5.0, t=-0.2, phi=1.1,
                    sfmodel=sf, mx2_cut=0.5)
    print(res["eta"])

Layout
------
    core/        frozen RC machinery; knows nothing about channels or models
    topology.py  which hadron is detected, which is missing
    models/      the versioned axis: pi0.vpk2014, eta.tableB1, jpsi.diffrad...
    generator/   unweighted event generation + LUND output
    validation/  comparison against EXCLURAD / DIFFRAD / published grids
"""

from .core import (EventKin, rc_factor, born_xsec, sigma_F,
                   delta_VR_inf, delta_vac, CORE_VERSION)
from . import models
from . import topology
from .provenance import stamp, VERSION

__version__ = VERSION

__all__ = ["EventKin", "rc_factor", "born_xsec", "sigma_F", "delta_VR_inf",
           "delta_vac", "models", "topology", "stamp",
           "CORE_VERSION", "__version__"]
