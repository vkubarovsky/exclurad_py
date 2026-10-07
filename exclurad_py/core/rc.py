"""
Driver: radiative correction factor for  e p -> e' p V  with the full set of
structure functions sigma_T, sigma_L, sigma_TT, sigma_LT, sigma_LT'.

    sigma_obs = sigma_Born * exp(delta_inf) * (1 + delta_VR + delta_vac) + sigma_F
    eta = sigma_obs / sigma_Born ,   delta = eta - 1 .
"""

import warnings as _warnings

import numpy as np
from .constants import M_P, M_E
from .kinematics import EventKin
from .born import born_xsec
from .virtual import delta_VR_inf, delta_vac
from .radiative import sigma_F


_WARNED_TNUCL = False


def rc_factor(Ebeam, Q2, W2, t, phi, sfmodel, h=+1, lam=0.0,
              mx2_cut=None, M=M_P, m=M_E,
              n_v=None, n_core=None, n_tail=None, n_phi=None,
              exponentiate=True, t_kind=None, mV=None):
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
    # mV -- масса ЭТОГО события, если она разыграна по Брейту-Вигнеру.
    # По умолчанию номинальная, как было: у pi0/eta и J/psi ширина
    # ничтожна, и там ничего не меняется.
    mV = sfmodel.mV if mV is None else float(mV)
    kin = EventKin(Ebeam, Q2, W2, t, phi, mV, v=0.0, M=M, m=m, mU=mU,
                   phi_offset=_off,
                   t_kind=t_kind)

    # THE FROZEN-t_nucl GUARD.
    #
    # A FIXED_NUCLEON model (pi0/eta, ProtonDetectedModel -- base.py:299
    # REQUIRES that topology, so hasattr(t_nucl) is a reliable test) carries
    # its t dependence in self.t_nucl, never in the `t` argument, which inside
    # the radiative integral is the SHIFTED transfer.  At THIS entry point t is
    # still the observed one, so t_nucl must equal it; every correct caller
    # arranges that via Generator._set_event_t.
    #
    # Build the model once with t_nucl=-0.30, loop over t, and you integrate a
    # cross section with NO t dependence -- finite and plausible at every
    # point, and nothing raised.  It cost 9 % on
    # integral(sigma_obs)/integral(sigma_Born) over the RG-A box (1.143 frozen
    # against 1.294 per point, campaign 1.25010 +- 0.00175) and invalidated
    # every percentile production/eta_scan.py had ever printed.  Warning
    # rather than raising, because a false positive here would stop a campaign.
    _tn = getattr(sfmodel, "t_nucl", None)
    if _tn is not None and abs(float(_tn) - float(t)) > 1e-9:
        global _WARNED_TNUCL
        if not _WARNED_TNUCL:
            _WARNED_TNUCL = True
            _warnings.warn(
                f"rc_factor: {type(sfmodel).__name__}.t_nucl = {float(_tn):.6g} "
                f"but the observed t = {float(t):.6g}.  This model's t "
                f"dependence lives in t_nucl, so it is being evaluated at the "
                f"wrong |t| and the result is finite but wrong.  Set "
                f"model.t_nucl = t (or call Generator._set_event_t(t)) before "
                f"each call.  Warned once per process.",
                RuntimeWarning, stacklevel=2)

    v_kin = EventKin.vmax_kin(Ebeam, Q2, W2, t, mV, M=M, m=m, mU=mU,
                              t_kind=t_kind)
    v_m = v_kin
    if mx2_cut is not None:
        v_m = min(v_m, mx2_cut - kin.mU ** 2)
    if v_m <= 0:
        raise ValueError("no phase space for photon emission (v_m <= 0)")

    sig_b = born_xsec(kin, sfmodel, h, lam)
    d_VR, d_inf = delta_VR_inf(kin, v_m)
    d_vac = delta_vac(Q2)
    sig_f = sigma_F(kin, sfmodel, h, v_m, lam,
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
