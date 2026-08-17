"""
Driver: radiative correction factor for  e p -> e' p V  with the full set of
structure functions sigma_T, sigma_L, sigma_TT, sigma_LT, sigma_LT'.

    sigma_obs = sigma_Born * exp(delta_inf) * (1 + delta_VR + delta_vac) + sigma_F
    eta = sigma_obs / sigma_Born ,   delta = eta - 1 .
"""

import numpy as np
from .constants import M_P, M_E
from .kinematics import EventKin
from .born import born_xsec
from .virtual import delta_VR_inf, delta_vac
from .radiative import sigma_F


def rc_factor(Ebeam, Q2, W2, t, phi, sfmodel, h=+1,
              mx2_cut=None, M=M_P, m=M_E,
              n_v=30, n_core=16, n_tail=24, n_phi=16,
              exponentiate=True, t_kind=None):
    """Radiative correction at fixed observed (Q2, W2, t, phi), beam helicity h.

    mx2_cut : missing-mass-squared cut MX^2 < mx2_cut  (GeV^2), where the
              missing (undetected) particle is the recoil baryon of mass
              sfmodel.mU (proton by default, neutron for pi+ n).
              If None, integrate the full exclusive-channel phase space.
    Returns dict with sigma_born, sigma_obs (nb/GeV^6), eta, delta and the
    individual pieces.
    """
    if hasattr(sfmodel, 'kf_W2Q2'):
        sfmodel.kf_W2Q2 = (float(W2), float(Q2))
    mU = getattr(sfmodel, 'mU', None)
    if t_kind is None:
        topo = getattr(sfmodel, 'topology', None)
        t_kind = getattr(topo, 't_kind', 'photon') if topo else 'photon'
    _topo = getattr(sfmodel, "topology", None)
    _off = float(getattr(_topo, "phi_offset", 0.0))
    kin = EventKin(Ebeam, Q2, W2, t, phi, sfmodel.mV, v=0.0, M=M, m=m, mU=mU,
                   phi_offset=_off,
                   t_kind=t_kind)

    v_kin = EventKin.vmax_kin(Ebeam, Q2, W2, t, sfmodel.mV, M=M, m=m, mU=mU,
                              t_kind=t_kind)
    v_m = v_kin
    if mx2_cut is not None:
        v_m = min(v_m, mx2_cut - kin.mU ** 2)
    if v_m <= 0:
        raise ValueError("no phase space for photon emission (v_m <= 0)")

    sig_b = born_xsec(kin, sfmodel, h)
    d_VR, d_inf = delta_VR_inf(kin, v_m)
    d_vac = delta_vac(Q2)
    sig_f = sigma_F(kin, sfmodel, h, v_m,
                    n_v=n_v, n_core=n_core, n_tail=n_tail, n_phi=n_phi)

    if exponentiate:
        virt = np.exp(d_inf) * (1.0 + d_VR + d_vac)
    else:
        virt = 1.0 + d_inf + d_VR + d_vac

    sig_obs = sig_b * virt + sig_f
    return dict(sigma_born=sig_b, sigma_obs=sig_obs,
                eta=sig_obs / sig_b, delta=sig_obs / sig_b - 1.0,
                delta_VR=d_VR, delta_inf=d_inf, delta_vac=d_vac,
                sigma_F=sig_f, v_max=v_m, v_max_kin=v_kin)
