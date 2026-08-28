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

# EXCLURAD-2014 used the CHARGED pion mass for the missing-mass variable amu
# even for pi0.  We do NOT reproduce that here, for two reasons.
#
# mU is a single number: it defines both  v = MX^2 - mU^2  and the four-vector
# of the missing meson written to LUND.  Giving those two different masses
# would emit events whose missing mass contradicts the v they were sampled at,
# so "keep the convention for v, fix only the output" is not available -- it is
# one choice, not two.  And eta already runs on the physical mass in the same
# slot (_ETA_TOPO below), so the charged mass made pi0 the odd channel out.
#
# Cost of the change: v_cut moves by m_pi+^2 - m_pi0^2 = 0.00126 GeV^2, i.e.
# 0.74 % of the published v_cut = 0.17.  Measured effect on agreement with the
# 2013 production is recorded in validation/pi0_published/.
#
# This package generates pi0 only.  A future pi- n generator should start from
# a working copy of this one and change the masses there, not carry a dormant
# second channel here.

# meson mass as hardcoded in dvmpx.F AM(2) -- used inside the fit's tminq
M_ETA_FIT = 0.547300
M_PI0_FIT = 0.134976

_PI0_TOPO = baryon_detected(M_PI0, meson="pi0")
_ETA_TOPO = baryon_detected(M_ETA, meson="eta")

# The fits were made on the CLAS12 pi0/eta grids; outside them the radiative
# tail would extrapolate a power law with no data behind it.
_PS_VALIDITY = dict(W2=(1.15, 20.0), Q2=(0.5, 12.0), mt=(0.0, 5.0))


# --------------------------------------------------------------------------- #
#  pi0
# --------------------------------------------------------------------------- #
@register
class Pi0_VPK2013(ProtonDetectedModel):
    name = "pi0.vpk2013"
    channel = "pi0"
    version = "2013"
    topology = _PI0_TOPO
    validity = _PS_VALIDITY
    provenance = (
        "Port of XSIGMA_T/L/TT/LT with pi0eta.par column k=1: the sigma_L-"
        "dominant model used for the PUBLISHED CLAS6 pi0 radiative "
        "corrections (PRC 90, 025205). Parameters byte-identical in "
        "~/excl2013/pi0eta.par (July 2013 production) and "
        "~/eta-rc-2026/model2013note/. NOTE: some directories label this "
        "same model '2014' (e.g. 'dvmpx-2014'); it is the 2013 production "
        "model. Alias pi0.vpk2014 kept for old references.")
    reference = ("PRC 90, 025205 (2014); ~/excl2013/ (July 2013 production, "
                 "radcor_all.vec); ~/exclurad-2026/report/report.pdf")

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
        "EXCLURAD's dvmpw wrapper doubles sigma_LT (dvmpw.F:55), but its "
        "tensor absorbs a compensating 1/2 on the LT term (exclurad_core.F:"
        "618-619), so the physical LT strength is the bare XSIGMA_LT. Our "
        "tensor is canonical (NU_LT=-1, == born_reference), so NO factor 2 "
        "here -- applying it doubled the cos(phi) harmonic of sigma_Born "
        "(11.41% vs exclugen's 5.71% at G1). sigma_L = 0 in this fit.")
    reference = "dvmpx.F March 2021; ~/exclurad-2026/"

    # sigma_LT' as a fixed fraction of the UNPOLARISED cross section
    # (VPK, 2026-08-19).  The CLAS measurement of sigma_LT'/sigma_0 for pi0
    # sits around +0.05..+0.18 across the forward bins and is roughly flat in
    # -t, so one number is a fair first model; +0.10 is the middle of it.
    # No eta measurement exists yet, so eta carries the same value until one
    # does -- change SLTP_OVER_SU in one place when it arrives.
    #
    # sigma_U = sigma_T + eps sigma_L, and it is written that way on purpose.
    # The March-2021 fits have sigma_L identically zero, so today this equals
    # 0.10 * sigma_T and the shortcut would give the same number -- but the
    # shortcut would stop being the stated physics the moment a fit grows a
    # longitudinal piece, and would do so silently.  epsilon is available
    # here because the model is constructed with the beam energy.
    SLTP_OVER_SU = 0.10

    def _sfs_tn(self, t_nucl, xB, Q2):
        sT, sL, sTT, sLT = F.vpk2021_form(t_nucl, xB, Q2, F.P2021_PI0, M_PI0_FIT)
        return sT, sL, sTT, sLT


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
        "column 2, eta). EXCLURAD's dvmpw factor 2 on sigma_LT is NOT applied "
        "-- it is cancelled inside EXCLURAD's own tensor, and ours is "
        "canonical. See Pi0_VPK2021. sigma_L = 0 in this fit.")
    reference = "dvmpx.F March 2021; ~/eta-rc-2026/model2021/"

    SLTP_OVER_SU = 0.10       # no eta measurement yet; see Pi0_VPK2021

    def _sfs_tn(self, t_nucl, xB, Q2):
        sT, sL, sTT, sLT = F.vpk2021_form(t_nucl, xB, Q2, F.P2021_ETA, M_ETA_FIT)
        return sT, sL, sTT, sLT


# Deprecated alias: this model is labelled "2014" in several older
# directories, but the parameters are the July-2013 production set.
@register
class Pi0_VPK2014(Pi0_VPK2013):
    name = "pi0.vpk2014"
    version = "2013"
    provenance = ("DEPRECATED ALIAS of pi0.vpk2013 -- identical parameters. "
                  "Kept so older scripts and notes resolve; prefer "
                  "pi0.vpk2013, which names the production year correctly.")


# --------------------------------------------------------------------------- #
#  2021 functional form, refitted 2026 in the helicity-amplitude basis
# --------------------------------------------------------------------------- #
#
# The forms are those of the March-2021 GT fit -- exp[(b + b' (ln xB - ln 0.15))
# t] (Q2)^(c/2), one curvature term added -- which is why the vintage tag is
# "amp2021".  The PARAMETERS are new: a 27-parameter fit made in 2026 to 639
# CLAS points (pi0 and eta sigma_U/sigma_TT/sigma_LT, pi0 and eta beam-spin
# asymmetries at 5.78 GeV, pi0 BSA at 10.6 GeV, eg1-dvcs target and double-spin
# moments), chi2/ndf = 1.48.
#
# What is different from pi0.vpk2021 / eta.tableB1, and why it matters here:
#
#   * the fit parameters are the six helicity amplitudes T00^(+/-), T01^(+/-),
#     U01^(+/-) per quark flavour, not the structure functions.  The five SFs
#     below are entries of the Gram matrix of the channel vectors (u,v,w), so
#     positivity (|sigma_TT| <= sigma_T, and the LT minor) holds identically
#     and the generator's negative-weight clamp is never reached;
#   * sigma_L is NOT zero.  It is separated from sigma_T without a Rosenbluth
#     measurement, through the interferences that carry T00 linearly;
#     sigma_L/sigma_T ~ 0.056 at Q2=1.94, xB=0.25, -t=0.4;
#   * sigma_LT' is fitted to the beam-spin asymmetries, not set to a fixed
#     fraction of sigma_U.  NOTE: a nonzero sigma_LT' makes the generator warn
#     about EXACT_ACCEPT (the eta table is built at one helicity); that warning
#     is about polarised samples only;
#   * sigma_LT comes out with the measured (small, sign-changing) size because
#     its sign is Re<T00|T01> with a fitted phase, not a positive-definite form;
#   * ONE parameter set serves all four target/meson combinations below.
#
# Source of the parameters: ~/pi0_eta_amplitude_model (amplitudes.py, the fits
# fit_global.py -> fit_level3.py -> fit_clas12.py, fitpar_amp27c.npy) and the
# draft paper paper/paper.tex there.  Frozen copy: models/amp2021_par.npy.

import os as _os
import numpy as _np
from . import _amplitude_fit as _AF

_AMP_PAR = _np.load(_os.path.join(_os.path.dirname(__file__), "amp2021_par.npy"))

# The CLAS6 fit domain.  Outside it the analytic forms extrapolate smoothly
# (exponential in t, power in Q2); OutOfRange.REPORT counts the excursions.
# W2 floor: the fit domain is W2 > 4, but the radiative tail samples below it.
# Measured (production/tail_reach.py, v_cut = 0.17/0.20 of the published
# analyses): the tail NODES reach W~2 ~ 1.4 for pi0, but the WEIGHT is
# concentrated within ~0.6 GeV^2 of the observed W2.  Zeroing the model below
# W2 = 3.6 moves eta by < 0.25% in the worst (lowest-W2) bin and by < 0.06% in
# the bulk; below 4.0 it moves eta by up to 5.6%, so 4.0 is too tight a floor
# and 3.6 is enough.  Declared 3.3 for margin -- 0.7 GeV^2 of smooth
# extrapolation, not a resonance model.  For GENERATION with a loose v_cut the
# tail is far longer and this floor does not apply; see the run cards.
_AMP_VALIDITY = dict(W2=(3.3, 20.0), Q2=(1.0, 6.5), mt=(0.0, 2.5))

_AMP_REF = ("PRC 90 025205; PRC 95 035202; PRC 77 042201; PLB 789 426; "
            "PLB 849 138459; PLB 768 168; PoS SPIN2018 034")

_AMP_PROV = (
    "2021 functional form, REFITTED 2026 in the helicity-amplitude basis: "
    "27 parameters, 639 CLAS points, chi2/ndf=1.48. Not the 2021 parameters "
    "-- see pi0.vpk2021 for those. sigma_L != 0, sigma_LT' fitted, positivity "
    "exact, one parameter set for pi0/eta on p and n. ")


def _amp_sfs(channel, t_nucl, xB, Q2):
    """(sT, sL, sTT, sLT, sLTp) in nb/GeV^2, canonical (meson-plane) sign."""
    t = _np.atleast_1d(_np.asarray(t_nucl, dtype=float))
    x = _np.atleast_1d(_np.asarray(xB, dtype=float))
    q = _np.atleast_1d(_np.asarray(Q2, dtype=float))
    t, x, q = _np.broadcast_arrays(t, x, q)
    out = _np.zeros((5,) + t.shape)
    o = out.reshape(5, -1)
    for i, (ti, xi, qi) in enumerate(zip(t.ravel(), x.ravel(), q.ravel())):
        s = _AF.structure(_AMP_PAR, channel, float(ti), float(xi), float(qi))
        if s is None:                      # below meson threshold in t
            continue
        o[0, i], o[1, i] = s["T"], s["L"]
        o[2, i], o[3, i], o[4, i] = s["TT"], s["LT"], s["LTp"]
    if out.shape[1:] == (1,):
        return tuple(float(v) for v in out[:, 0])
    return tuple(out)


@register
class Pi0_Amp2021(ProtonDetectedModel):
    name = "pi0.amp2021"
    channel = "pi0"
    version = "amp2021"
    topology = _PI0_TOPO
    validity = _AMP_VALIDITY
    provenance = _AMP_PROV + "pi0 off the PROTON: (2F^u + F^d)/(3 sqrt2)."
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp_sfs("pi0p", t_nucl, xB, Q2)


@register
class Eta_Amp2021(ProtonDetectedModel):
    name = "eta.amp2021"
    channel = "eta"
    version = "amp2021"
    topology = _ETA_TOPO
    validity = _AMP_VALIDITY
    provenance = (_AMP_PROV + "eta off the PROTON: (2F^u - F^d)/(3 sqrt6 k), "
                  "k = 0.863 transverse (chiral condensate + eta8-eta1 mixing), "
                  "0.695 longitudinal (decay constants only). No free parameter.")
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp_sfs("etap", t_nucl, xB, Q2)


@register
class Pi0n_Amp2021(ProtonDetectedModel):
    name = "pi0n.amp2021"
    channel = "pi0"
    version = "amp2021-n"
    topology = _PI0_TOPO
    validity = _AMP_VALIDITY
    provenance = (_AMP_PROV + "pi0 off the NEUTRON: isospin rotation "
                  "(F^u + 2F^d)/(3 sqrt2). PREDICTION -- no neutron data was "
                  "used in the fit. Compare Hall-A PRL 118 222002.")
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp_sfs("pi0n", t_nucl, xB, Q2)


@register
class Etan_Amp2021(ProtonDetectedModel):
    name = "etan.amp2021"
    channel = "eta"
    version = "amp2021-n"
    topology = _ETA_TOPO
    validity = _AMP_VALIDITY
    provenance = (_AMP_PROV + "eta off the NEUTRON: isospin rotation of the "
                  "eta combination, k (2F^d - F^u)/(3 sqrt6) -- i.e. u <-> d "
                  "relative to eta.amp2021. PREDICTION, never measured.")
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp_sfs("etan", t_nucl, xB, Q2)


# --------------------------------------------------------------------------- #
#  amp2026: the self-consistent fixed point of the RC iteration
# --------------------------------------------------------------------------- #
#
# Same amplitude parameterisation as amp2021, but fitted to CLAS6 data whose
# radiative correction was recomputed WITH the amplitude model itself, iterated
# to a fixed point (~/rc_iter1).  amp2021 is fitted to the published data, whose
# RC assumed a sigma_L-dominant model; the July 2026 study
# (~/pi0eta-rc-2026-paper) showed the RC depends significantly on that
# assumption.  amp2026 removes the assumption: sigma_L is measured, not guessed.
#
# Use amp2026 for production.  Keep amp2021 to reproduce the published data.

_AMP26_PAR = _np.load(_os.path.join(_os.path.dirname(__file__), "amp2026_par.npy"))
_AMP26_PROV = (
    "Self-consistent fixed point of the RC iteration: RC recomputed with the "
    "amplitude model over the full published grids, cross sections re-corrected "
    "(sigma_new = sigma_pub / r, r = eta(new)/eta(published), both from "
    "exclurad_py so the code cancels), sigma(phi) refitted, amplitudes refitted. "
    "See ~/rc_iter1/README.md. ")


def _amp26_sfs(channel, t_nucl, xB, Q2):
    t = _np.atleast_1d(_np.asarray(t_nucl, dtype=float))
    x = _np.atleast_1d(_np.asarray(xB, dtype=float))
    q = _np.atleast_1d(_np.asarray(Q2, dtype=float))
    t, x, q = _np.broadcast_arrays(t, x, q)
    out = _np.zeros((5,) + t.shape)
    o = out.reshape(5, -1)
    for i, (ti, xi, qi) in enumerate(zip(t.ravel(), x.ravel(), q.ravel())):
        s = _AF.structure(_AMP26_PAR, channel, float(ti), float(xi), float(qi))
        if s is None:
            continue
        o[0, i], o[1, i] = s["T"], s["L"]
        o[2, i], o[3, i], o[4, i] = s["TT"], s["LT"], s["LTp"]
    if out.shape[1:] == (1,):
        return tuple(float(v) for v in out[:, 0])
    return tuple(out)


@register
class Pi0_Amp2026(ProtonDetectedModel):
    name = "pi0.amp2026"
    channel = "pi0"
    version = "amp2026"
    topology = _PI0_TOPO
    validity = _AMP_VALIDITY
    provenance = _AMP26_PROV + "pi0 off the proton."
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp26_sfs("pi0p", t_nucl, xB, Q2)


@register
class Eta_Amp2026(ProtonDetectedModel):
    name = "eta.amp2026"
    channel = "eta"
    version = "amp2026"
    topology = _ETA_TOPO
    validity = _AMP_VALIDITY
    provenance = _AMP26_PROV + "eta off the proton, flavour-rotated."
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp26_sfs("etap", t_nucl, xB, Q2)


@register
class Pi0n_Amp2026(ProtonDetectedModel):
    name = "pi0n.amp2026"
    channel = "pi0"
    version = "amp2026-n"
    topology = _PI0_TOPO
    validity = _AMP_VALIDITY
    provenance = _AMP26_PROV + "pi0 off the NEUTRON. Prediction."
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp26_sfs("pi0n", t_nucl, xB, Q2)


@register
class Etan_Amp2026(ProtonDetectedModel):
    name = "etan.amp2026"
    channel = "eta"
    version = "amp2026-n"
    topology = _ETA_TOPO
    validity = _AMP_VALIDITY
    provenance = _AMP26_PROV + "eta off the NEUTRON. Prediction."
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp26_sfs("etan", t_nucl, xB, Q2)


# --------------------------------------------------------------------------- #
#  amp2026s: amp2026 with the physical t-slope constraints imposed
# --------------------------------------------------------------------------- #
#
# Same data and same RC fixed point as amp2026; what changed is that the fit is
# no longer allowed to produce form factors that GROW with |t|.  amp2026 had a
# negative H_T^d slope over the whole measured xB range.  Imposed here:
#   b + b' ln xB >= 0 for every block on xB in [0.1, 0.6];
#   slope(H_T^d) >= slope(H_T^u), the ordering of the global GPD fits;
#   and the b2 t^2 curvature of Ebar_T is REMOVED (26 parameters, not 27), so
#   every form factor is a pure falling exponential and no slope can reverse at
#   any t.  b2 was a near-tmin fudge: dropping it costs 71 chi2 on the full set
#   but only 11 once -t' > 0.3, where sigma_TT is not dominated by its own
#   kinematic t' factor.
# The fit also AVERAGES THE MODEL OVER THE ACCEPTED BIN VOLUME (PRC 90 025205
# and PRC 95 035202 Tables I-III, cut by W > 2, E' > 0.8, 21-45 deg and by the
# tmin boundary), instead of evaluating it at the published bin mean.  The
# reconstruction reproduces the published bin means to 0.001-0.02 GeV^2.
# Cost of removing b2, bin-averaged: chi2 1090.7 -> 1159.0 (chi2/ndf 1.5604 ->
# 1.6557), and the R_ET prior ends pulled +4.3 sigma (R_ET = 1.18) - the fit
# compensates the lost curvature in the d/u ratio.  Proper bin averaging did NOT
# remove the need for b2, and did not remove the negative H_T^d slope of the
# free fit either.  Both are physics choices, not binning artefacts.
# See ~/rc_iter1/README.md.
#
# Use amp2026s for production.  amp2026 is kept frozen because the OneDrive
# generator baseline (Work/2026_pi0_amplitudes/generator/) was produced with it.

_AMP26S_PAR = _np.load(_os.path.join(_os.path.dirname(__file__), "amp2026s_par.npy"))
_AMP26S_PROV = (
    "Slope-constrained refit of the amp2026 fixed point: form factors required "
    "to fall with |t| over xB in [0.1, 0.6], H_T^d at least as steep as H_T^u, "
    "Ebar_T curvature not reversing the fall inside -t <= 2.5. "
    "See ~/rc_iter1/README.md. ")


def _amp26s_sfs(channel, t_nucl, xB, Q2):
    t = _np.atleast_1d(_np.asarray(t_nucl, dtype=float))
    x = _np.atleast_1d(_np.asarray(xB, dtype=float))
    q = _np.atleast_1d(_np.asarray(Q2, dtype=float))
    t, x, q = _np.broadcast_arrays(t, x, q)
    out = _np.zeros((5,) + t.shape)
    o = out.reshape(5, -1)
    for i, (ti, xi, qi) in enumerate(zip(t.ravel(), x.ravel(), q.ravel())):
        s = _AF.structure(_AMP26S_PAR, channel, float(ti), float(xi), float(qi))
        if s is None:
            continue
        o[0, i], o[1, i] = s["T"], s["L"]
        o[2, i], o[3, i], o[4, i] = s["TT"], s["LT"], s["LTp"]
    if out.shape[1:] == (1,):
        return tuple(float(v) for v in out[:, 0])
    return tuple(out)


@register
class Pi0_Amp2026s(ProtonDetectedModel):
    name = "pi0.amp2026s"
    channel = "pi0"
    version = "amp2026s"
    topology = _PI0_TOPO
    validity = _AMP_VALIDITY
    provenance = _AMP26S_PROV + "pi0 off the proton."
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp26s_sfs("pi0p", t_nucl, xB, Q2)


@register
class Eta_Amp2026s(ProtonDetectedModel):
    name = "eta.amp2026s"
    channel = "eta"
    version = "amp2026s"
    topology = _ETA_TOPO
    validity = _AMP_VALIDITY
    provenance = _AMP26S_PROV + "eta off the proton, flavour-rotated."
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp26s_sfs("etap", t_nucl, xB, Q2)


@register
class Pi0n_Amp2026s(ProtonDetectedModel):
    name = "pi0n.amp2026s"
    channel = "pi0"
    version = "amp2026s-n"
    topology = _PI0_TOPO
    validity = _AMP_VALIDITY
    provenance = _AMP26S_PROV + "pi0 off the NEUTRON. Prediction."
    reference = _AMP_REF

    def _sfs_tn(self, t_nucl, xB, Q2):
        return _amp26s_sfs("pi0n", t_nucl, xB, Q2)
