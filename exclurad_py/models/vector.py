"""
Vector-meson structure-function models: rho0, phi, J/psi.

Registry keys
-------------
    jpsi.diffrad      DIFFRAD's internal two-gluon model (port of difflt)
    jpsi.roberts2026  sigma_tot(W) fit from 2026_Jpsi_total_cross_section
    jpsi.diffrad_vm   the tuned model of diffrad_vm.f90 (v2 defaults)
    jpsi.vmd2026      roberts2026 sigma(W) x VMD (1+Q2/m^2)^-2 x dipole t
    rho.demo          illustrative rho0 parameterization
    phi.demo          illustrative phi parameterization

All use the DVMP topology (meson detected, recoil proton missing).

TWO-STRUCTURE-FUNCTION MODE (the intended configuration for vector mesons).
Every vector model here supplies sigma_T and sigma_L and returns
sigma_TT = sigma_LT = sigma_LT' = 0 as deliberate zero placeholders.  That
is not a missing model: the rho, phi and J/psi cross sections in
~/MC_vector_mesons/diffrad_vm.f90 are complete two-SF models, and the
vector program is being tested in two-SF mode on purpose.  SDMEs are not
yet in a ready-to-use form and are deliberately out of scope here.

The RC engine does not require the zeros -- it carries all five SFs -- so
when interference SFs do become available they drop straight into the same
classes.  Until then the zeros are the model, not a gap.
"""

import numpy as np

from .base import StructureFunctions
from .registry import register
from ..core.constants import M_P, M_N, M_RHO, M_PHI, M_JPSI
from ..core.kinematics import kallen
from ..topology import meson_detected, baryon_detected

_MP = 0.938272          # value used in the ported Fortran

_RHO_TOPO = meson_detected(M_RHO, meson="rho0")
_PHI_TOPO = meson_detected(M_PHI, meson="phi")
_JPSI_TOPO = meson_detected(M_JPSI, meson="J/psi")
_JPSI_TOPO_PDET = baryon_detected(M_JPSI, meson="J/psi")
_JPSIN_TOPO = meson_detected(M_JPSI, meson="J/psi", m_recoil=M_N, recoil="n")


def tmin_dvmp(W2, Q2, mV, M=M_P, mU=None):
    """t_min (least negative t) for gamma*(Q2) p -> V B."""
    if mU is None:
        mU = M
    W = np.sqrt(np.maximum(W2, 1e-30))
    E_g = (W2 - Q2 - M * M) / (2.0 * W)
    E_v = (W2 + mV * mV - mU * mU) / (2.0 * W)
    q = np.sqrt(np.maximum(kallen(W2, -Q2, M * M), 0.0)) / (2.0 * W)
    pv = np.sqrt(np.maximum(E_v * E_v - mV * mV, 0.0))
    return -Q2 + mV * mV - 2.0 * (E_g * E_v - q * pv)


# --------------------------------------------------------------------------- #
#  J/psi
# --------------------------------------------------------------------------- #
_ALPHA_F = 0.729735e-2
_GGAM4 = 5.36e-6
_P02 = 0.5
_AL_S = 0.25
_GEV2_NB = 0.389379e6


@register
class Jpsi_Diffrad(StructureFunctions):
    name = "jpsi.diffrad"
    channel = "jpsi"
    version = "diffrad"
    topology = _JPSI_TOPO
    validity = dict(W2=(16.4, 400.0), Q2=(1e-4, 30.0), mt=(0.0, 12.0))
    # (was "freeze"; clamping removed -- see models.base.OutOfRange)
    provenance = (
        "Exact port of the internal two-gluon J/psi model `difflt` of "
        "idiffrad.f (ivec=4), the model DIFFRAD uses for its own RC. "
        "Overall normalization is DIFFRAD's; only the (Q2,W2,t) shape "
        "matters for RC ratios. sigma_L = Q2/m^2 * sigma_T (DIFFRAD's choice).")
    reference = "~/OneDrive/Work/2026_DIFFRAD/RC_vector_mesons_DIFFRAD/idiffrad.f"

    def _sfs(self, W2, Q2, t):
        amp2 = _MP * _MP
        amv2 = M_JPSI ** 2
        Wth2 = (_MP + M_JPSI) ** 2
        ok = W2 > Wth2
        W2s = np.where(ok, W2, Wth2 + 1e-3)

        sx = W2s + Q2 - amp2
        anu = sx / (2 * _MP)
        eh = (sx + t) / (2 * _MP)
        ff2 = np.exp(5.0 * t)
        tqt = t + Q2 - amv2
        denom = 4.0 * (anu ** 2 + Q2)
        pt2 = -(4 * (anu ** 2 + Q2) * amv2 + 4 * anu * eh * tqt
                - 4 * eh ** 2 * Q2 + tqt ** 2) / denom
        pt2 = np.maximum(pt2, 0.0)
        xsb = (Q2 + amv2 + pt2) / W2s
        q2b = (Q2 + amv2 + pt2) / 4.0
        fm = np.where(pt2 <= _P02,
                      np.log((4 * q2b - pt2 + _P02) / (pt2 + _P02)),
                      np.log((pt2 + _P02) / (4 * q2b - pt2 + _P02)
                             * 4 * q2b ** 2 / np.maximum(pt2, 1e-30) ** 2))
        xsbgm = 3.0 * np.maximum(1.0 - xsb, 0.0) ** 5
        sk = xsbgm * fm / (2 * q2b * (2 * q2b - pt2) * np.log(8 * q2b / _P02))
        sigt = (_AL_S ** 2 * _GGAM4 * M_JPSI ** 3 / 3.0 / _ALPHA_F
                * np.pi ** 3 * sk ** 2 * ff2)
        sigl = Q2 / amv2 * sigt

        sT = np.where(ok, sigt, 0.0) * _GEV2_NB
        sL = np.where(ok, sigl, 0.0) * _GEV2_NB
        z = np.zeros_like(sT)
        return {"sT": sT, "sL": sL, "sTT": z, "sLT": z, "sLTp": z}


@register
class Jpsi_Roberts2026(StructureFunctions):
    name = "jpsi.roberts2026"
    channel = "jpsi"
    version = "2026"
    topology = _JPSI_TOPO
    validity = dict(W2=(16.4, 400.0), Q2=(1e-4, 30.0), mt=(0.0, 12.0))
    provenance = (
        "sigma_tot(W) = A (W-Wth)^alpha exp(-B (W-Wth)) nb with "
        "A=1.0184032, alpha=0.70676531, B=-0.92829968, from "
        "2026_Jpsi_total_cross_section/fit_Roberts.py (GlueX/CLAS12 points). "
        "ASSUMPTIONS added on top of the total-cross-section fit, not fitted: "
        "dsigma_T/dt = sigma_tot * b exp(b t') with b = 1.67 GeV^-2 "
        "(GlueX-like); Q2 dependence = VMD dipole (m^2/(m^2+Q2))^2.4; "
        "R = sigma_L/sigma_T = Q2/m^2. Replace when a near-threshold t-slope "
        "and R are settled.")
    reference = "~/OneDrive/Work/2026_Jpsi_total_cross_section/fit_Roberts.py"

    A, ALPH, B = 1.0184032, 0.70676531, -0.92829968
    WTH = 0.939 + 3.097
    b = 1.67
    ndip = 2.4

    def _sfs(self, W2, Q2, t):
        W = np.sqrt(np.maximum(W2, 1e-12))
        ok = W > self.WTH + 1e-9
        W2s = np.where(ok, W2, (self.WTH + 1e-3) ** 2)
        t0 = tmin_dvmp(W2s, Q2, M_JPSI)
        tp = np.minimum(t - t0, 0.0)
        x = np.maximum(np.sqrt(W2s) - self.WTH, 0.0)
        sig = np.where(x > 0, self.A * x ** self.ALPH * np.exp(-self.B * x), 0.0)
        dip = (M_JPSI ** 2 / (M_JPSI ** 2 + Q2)) ** self.ndip
        sT = np.where(ok, sig * self.b * np.exp(self.b * tp) * dip, 0.0)
        sL = Q2 / M_JPSI ** 2 * sT
        z = np.zeros_like(sT)
        return {"sT": sT, "sL": sL, "sTT": z, "sLT": z, "sLTp": z}


@register
class Jpsin_Roberts2026(Jpsi_Roberts2026):
    """gamma* n -> J/psi n: the proton model relabelled, sigma_n = sigma_p.

    Deliberate baseline, not an oversight (JPSI_FERMI_BRIEF.md section 5): the
    exchange is gluonic and isoscalar, so at leading order the neutron and
    proton cross sections are equal.  That equality is also the null
    hypothesis a near-threshold neutron measurement tests, so nothing is
    invented here that could bury the signal.  What differs is the topology:
    a neutron initial state (m_T = M_N, threshold 7 MeV lower), a neutron
    recoil in the LUND, and -- on a deuteron -- a struck neutron with a
    PROTON spectator.  Everything assumed in the parent (t-slope b, dipole,
    R) stays assumed here.
    """
    name = "jpsin.roberts2026"
    channel = "jpsin"
    version = "2026"
    topology = _JPSIN_TOPO
    provenance = ("Relabelled jpsi.roberts2026 for a neutron target: sigma_n = "
                  "sigma_p (isoscalar two-gluon exchange, leading order), same "
                  "sigma(W) function and threshold constant as the parent; only "
                  "the topology (neutron target and recoil) differs. "
                  + Jpsi_Roberts2026.provenance)


@register
class Jpsi_DiffradVM(StructureFunctions):
    name = "jpsi.diffrad_vm"
    channel = "jpsi"
    version = "vm2026"
    topology = _JPSI_TOPO
    validity = dict(W2=(16.4, 400.0), Q2=(1e-4, 30.0), mt=(0.0, 12.0))
    provenance = (
        "Tuned J/psi model of diffrad_vm.f90 (v2 defaults, June/July 2026): "
        "alf1=2.33e-8, alf2=0.5, alf3=7.8, nuT=3.114, cR=0.4, mg2=2.637; "
        "Mariana-fit anchored, power-law Q2, dipole t-form. VERIFIED against "
        "diffrad_vm.f90 itself to 8 significant digits (validation/vm_models/).")
    reference = "~/RC_vector_mesons_DIFFRAD/diffrad_vm.f90 (v2)"

    alf1, alf2, alf3 = 2.33e-8, 0.500, 7.800
    nuT, cR, mg2 = 3.114, 0.400, 2.637

    def _sfs(self, W2, Q2, t):
        amp2 = _MP * _MP
        m2 = M_JPSI ** 2
        Wth2 = (_MP + M_JPSI) ** 2
        ok = (W2 > Wth2) & (Q2 > 0)
        W2s = np.where(ok, W2, Wth2 * 1.001)
        W = np.sqrt(W2s)
        ecm_i = (W2s + Q2 + amp2) / (2 * W)
        pcm_i = np.sqrt(np.maximum(ecm_i ** 2 - amp2, 0))
        ecm_f = (W2s + m2 - amp2) / (2 * W)
        pcm_f = np.sqrt(np.maximum(ecm_f ** 2 - m2, 0))
        tmin_k = (pcm_i - pcm_f) ** 2 - ((Q2 + m2) / (2 * W)) ** 2
        tmax_k = (pcm_i + pcm_f) ** 2 - ((Q2 + m2) / (2 * W)) ** 2
        ok = ok & (-t >= tmin_k) & (-t <= tmax_k)
        cT = (self.alf1 * np.maximum(1 - Wth2 / W2s, 0) ** self.alf2
              * W ** self.alf3)
        sT = (cT / Q2 ** (self.nuT / 2.0)
              * 3.0 * (self.mg2 + tmin_k) ** 3 / (self.mg2 - t) ** 4)
        sT = np.where(ok, sT, 0.0)
        sL = self.cR * Q2 / m2 * sT
        z = np.zeros_like(sT)
        return {"sT": sT, "sL": sL, "sTT": z, "sLT": z, "sLTp": z}


@register
class Jpsi_VMD2026(StructureFunctions):
    """sigma_T = sigma_gamma(W) (1 + Q2/m^2)^-k  x  dipole t-shape normalised to 1.

    Replaces jpsi.diffrad_vm for quasi-real (FT) kinematics.  That model put
    Q2^(-nuT/2) with nuT = 3.114 into sigma_T, but 3.114 is M. Tenorio's fit
    exponent of dsigma_ep/dQ2dW, which already contains the flux Gamma ~ 1/Q2;
    sigma_gamma* therefore fell as Q^-3.1 instead of being flat, and its alf1
    was anchored to a superseded version of her cross section.
    """

    name = "jpsi.vmd2026"
    channel = "jpsi"
    version = "2026"
    topology = _JPSI_TOPO
    validity = dict(W2=(16.4, 400.0), Q2=(0.0, 30.0), mt=(0.0, 12.0))
    provenance = (
        "sigma_gamma(W) = the existing total-cross-section fit of "
        "jpsi.roberts2026 (A, alpha, B taken from that class, not refitted). "
        "Q2: pure VMD (m^2/(m^2+Q2))^k, k=2, Lambda2=m_J/psi^2. t: the "
        "diffrad_vm dipole 3(mg2+|t|min)^3/(mg2-t)^4, mg2=2.637 (Tenorio "
        "dsigma/dt fit), normalised to unit integral over the kinematic t "
        "range, so the t-integrated sigma_T equals sigma_gamma(W) x VMD. "
        "sigma_L = cR Q2/m^2 sigma_T, cR=0.4 (as diffrad_vm).")
    reference = "this file; VPK 29.10.2026"

    A, ALPH, B = Jpsi_Roberts2026.A, Jpsi_Roberts2026.ALPH, Jpsi_Roberts2026.B
    LAM2 = M_JPSI ** 2
    k = 2.0
    mg2, cR = 2.637, 0.400

    def _sfs(self, W2, Q2, t):
        amp2 = _MP * _MP
        m2 = M_JPSI ** 2
        Wth = _MP + M_JPSI
        ok = (W2 > Wth ** 2) & (Q2 >= 0)
        W2s = np.where(ok, W2, Wth ** 2 * 1.001)
        W = np.sqrt(W2s)
        ecm_i = (W2s + Q2 + amp2) / (2 * W)
        pcm_i = np.sqrt(np.maximum(ecm_i ** 2 - amp2, 0))
        ecm_f = (W2s + m2 - amp2) / (2 * W)
        pcm_f = np.sqrt(np.maximum(ecm_f ** 2 - m2, 0))
        tmin_k = (pcm_i - pcm_f) ** 2 - ((Q2 + m2) / (2 * W)) ** 2
        tmax_k = (pcm_i + pcm_f) ** 2 - ((Q2 + m2) / (2 * W)) ** 2
        ok = ok & (-t >= tmin_k) & (-t <= tmax_k)
        x = np.maximum(W - Wth, 0.0)
        sig = self.A * x ** self.ALPH * np.exp(-self.B * x)
        vmd = (1.0 + Q2 / self.LAM2) ** (-self.k)
        a = self.mg2 + tmin_k
        norm = 1.0 - (a / (self.mg2 + tmax_k)) ** 3
        shape = 3.0 * a ** 3 / (self.mg2 - t) ** 4 / np.maximum(norm, 1e-300)
        sT = np.where(ok, sig * vmd * shape, 0.0)
        sL = self.cR * Q2 / m2 * sT
        z = np.zeros_like(sT)
        return {"sT": sT, "sL": sL, "sTT": z, "sLT": z, "sLTp": z}


# --------------------------------------------------------------------------- #
#  rho0, phi -- illustrative only until a real fit is plugged in
# --------------------------------------------------------------------------- #
class _DemoVector(StructureFunctions):
    """Smooth parameterization with roughly realistic magnitudes and slopes.

    sigma_T = sig0 (W2/W02)^delta (mV^2/(mV^2+Q2))^n exp(b t')
    R = sigma_L/sigma_T = r0 (Q2/mV^2)^p / (1 + r1 Q2/mV^2)
    sigma_TT, sigma_LT, sigma_LT' carry the correct t' -> 0 threshold
    behaviour (sqrt(-t') for LT/LT', -t' for TT).
    """

    sig0 = W02 = delta = n = b = r0 = p = r1 = None
    cTT, cLT, cLTp = 0.15, 0.20, 0.10

    def _sfs(self, W2, Q2, t):
        mV = self.topology.m_detected
        mV2 = mV ** 2
        t0 = tmin_dvmp(W2, Q2, mV)
        tp = np.minimum(t - t0, 0.0)
        prop = (mV2 / (mV2 + Q2)) ** self.n
        sT = self.sig0 * (W2 / self.W02) ** self.delta * prop * np.exp(self.b * tp)
        R = self.r0 * (Q2 / mV2) ** self.p / (1.0 + self.r1 * Q2 / mV2)
        sL = R * sT
        mt = -tp
        ft1 = np.sqrt(mt) / (1.0 + mt)
        ft2 = mt / (1.0 + mt) ** 2
        sqTL = np.sqrt(np.maximum(sT * sL, 0.0))
        return {"sT": sT, "sL": sL,
                "sTT": -self.cTT * sT * ft2,
                "sLT": self.cLT * sqTL * ft1,
                "sLTp": self.cLTp * sqTL * ft1}


_DEMO_WARNING = (
    "*** ILLUSTRATIVE ONLY -- DO NOT USE *** invented smooth parameterization "
    "kept solely as a self-test fixture with non-zero sigma_TT/LT/LT' (nothing "
    "else in the registry exercises the interference SFs). The REAL models for "
    "this channel are rho.diffrad_vm / phi.diffrad_vm, transcribed from "
    "~/MC_vector_mesons/diffrad_vm.f90. Never use a demo for physics.")


@register
class Rho_Demo(_DemoVector):
    name = "rho.demo"
    channel = "rho"
    version = "demo"
    topology = _RHO_TOPO
    validity = dict(W2=(4.0, 400.0), Q2=(0.05, 20.0), mt=(0.0, 5.0))
    provenance = _DEMO_WARNING
    reference = "vmrad/models.py demo_rho()"
    sig0, W02, delta, n, b = 8.0e3, 10.0, 0.22, 2.2, 7.0
    r0, p, r1 = 0.6, 0.61, 0.14


@register
class Phi_Demo(_DemoVector):
    name = "phi.demo"
    channel = "phi"
    version = "demo"
    topology = _PHI_TOPO
    validity = dict(W2=(4.4, 400.0), Q2=(0.05, 20.0), mt=(0.0, 5.0))
    provenance = _DEMO_WARNING
    reference = "vmrad/models.py demo_phi()"
    sig0, W02, delta, n, b = 1.0e3, 10.0, 0.28, 2.2, 5.5
    r0, p, r1 = 0.5, 0.6, 0.1


# --------------------------------------------------------------------------- #
#  phi -- the tuned model of diffrad_vm.f90 (what generated samples_v2)
# --------------------------------------------------------------------------- #
@register
class Phi_DiffradVM(StructureFunctions):
    """dsigma_T/dt = sigma_T(W,Q2) * bt * exp(bt t), exponential t-form.

        cT   = alf1 (1 - Wth^2/W^2)^alf2 W^alf3
        sT   = cT / (1 + Q2/m_phi^2)^nuT
        sL   = cR * Q2/m_phi^2 * sT
    """

    name = "phi.diffrad_vm"
    channel = "phi"
    version = "vm2026"
    topology = meson_detected(1.019461, meson="phi")  # PDG (VPK 15.08: PDG everywhere)
    validity = dict(W2=(3.85, 400.0), Q2=(0.05, 30.0), mt=(0.0, 12.0))
    provenance = (
        "Transcribed from `sigma_T_phi` / `sigma_L_phi` of "
        "~/MC_vector_mesons/diffrad_vm.f90 with the table defaults: "
        "alf1=400, alf2=2.000, alf3=0.200, nuT=2.253, bt=1.326, cR=1.000. "
        "W-shape retuned 2026-06-30 (alf2 fixed at the physical +2, "
        "Donnachie-Landshoff; alf3=0.20 Pomeron exponent from Roberts et al. "
        "arXiv:2510.08845). This is the model behind samples_v2/phi_*_100k. "
        "VERIFIED against diffrad_vm.f90 itself to 9 significant digits "
        "(validation/vm_models/).")
    reference = "~/MC_vector_mesons/diffrad_vm.f90 (sigma_T_phi)"

    alf1, alf2, alf3 = 400.0, 2.000, 0.200
    nuT, bt, cR = 2.253, 1.326, 1.000
    M_PHI_VM = 1.019461  # PDG; was 1.019412 (diffrad_vm transcription) until 2026-08-15
    WTH = 1.96

    def _sfs(self, W2, Q2, t):
        m2 = self.M_PHI_VM ** 2
        Wth2 = self.WTH ** 2
        ok = (W2 > Wth2) & (Q2 > 0)
        W2s = np.where(ok, W2, Wth2 * 1.001)
        W = np.sqrt(W2s)
        cT = self.alf1 * np.maximum(1.0 - Wth2 / W2s, 0.0) ** self.alf2 \
            * W ** self.alf3
        sigT = cT / (1.0 + Q2 / m2) ** self.nuT
        sT = np.where(ok, sigT * self.bt * np.exp(self.bt * t), 0.0)
        sL = self.cR * Q2 / m2 * sT
        z = np.zeros_like(sT)
        return {"sT": sT, "sL": sL, "sTT": z, "sLT": z, "sLTp": z}


# --------------------------------------------------------------------------- #
#  rho0 -- the model of diffrad_vm.f90 (two structure functions)
# --------------------------------------------------------------------------- #
@register
class Rho_DiffradVM(StructureFunctions):
    """sigma_T = AT exp(bT t) xB/(1-xB) / Q2^3 ,  sigma_L = AL exp(bL t) xB / Q2^2."""

    name = "rho.diffrad_vm"
    channel = "rho"
    SLT_OVER_SU = -0.30       # CLAS sigma_LT/sigma_0, forward bins -0.2..-0.4
    version = "vm2026"
    topology = meson_detected(M_RHO, meson="rho0")
    validity = dict(W2=(4.0, 400.0), Q2=(0.3, 30.0), mt=(0.0, 12.0))
    provenance = (
        "Transcribed from `sigma_T_rho` / `sigma_L_rho` of "
        "~/MC_vector_mesons/diffrad_vm.f90 with the defaults set at line 281: "
        "AT=15, bT=2.0, AL=25, bL=4.0. Two-SF model; sigma_TT/LT/LT' are "
        "deliberate zeros. VERIFIED against diffrad_vm.f90 itself to 9 "
        "significant digits (validation/vm_models/).")
    reference = "~/MC_vector_mesons/diffrad_vm.f90 (sigma_T_rho, line 1614)"

    AT, bT, AL, bL = 15.0, 2.0, 25.0, 4.0

    def _sfs(self, W2, Q2, t):
        xB = Q2 / np.maximum(W2 + Q2 - _MP ** 2, 1e-12)
        ok = (xB > 0) & (xB < 1) & (Q2 > 0)
        xs = np.where(ok, xB, 0.5)
        sT = np.where(ok, self.AT * np.exp(self.bT * t) * xs / (1.0 - xs) / Q2 ** 3, 0.0)
        sL = np.where(ok, self.AL * np.exp(self.bL * t) * xs / Q2 ** 2, 0.0)
        z = np.zeros_like(sT)
        return {"sT": sT, "sL": sL, "sTT": z, "sLT": z, "sLTp": z}


class _PdetVector:
    """Mixin: the same vector model seen from the DETECTED PROTON.

    What changes is the BIN, not the physics:

        detected   = the recoil proton  -> t = (p - p')^2 held fixed,
                                           i.e. t = -2 M T_p in the lab
        undetected = the meson          -> v = MX^2(e'p) - m_V^2

    At Born the two bins coincide: for 2 -> 2, q - p_V = p' - p identically,
    so (q - p_V)^2 = (p - p')^2 and a card's t box carries over unchanged.
    They part only for radiative events, where the measured q is not the
    vertex q~.  That is the point: the proton bin's t is built from the
    measured proton alone and lepton radiation cannot move it, while the
    meson-side t computed from the measured lepton arm is displaced by
    exactly v.

    DELIBERATELY NOT ProtonDetectedModel.  That base class carries the
    Hand-convention kf conversion the pi0/eta fits need at the model
    boundary; the vector models are already in the core's |q|-flux
    convention and must not have it.

    baryon_detected also flips lt_sign and sets phi_offset = pi, because the
    hadron plane's x axis follows the detected hadron.  For J/psi and phi
    sigma_TT = sigma_LT = sigma_LT' = 0 so neither matters; for rho,
    SLT_OVER_SU = -0.30 makes sigma_LT non-zero and the two act together --
    verified in validation/vector_proton_t/ to reproduce the Born cross
    section at every phi.

    t_nucl is the OBSERVED nucleon transfer, supplied per event by
    Generator._set_event_t exactly as for pi0/eta.  The `t` argument of
    _sfs is the SHIFTED transfer inside the radiative integral and must not
    be used (see models/base.py and the rc_factor guard).
    """

    def __init__(self, t_nucl=-0.30, **kw):
        super().__init__(**kw)
        if t_nucl > 0:
            raise ValueError("t_nucl must be negative (spacelike)")
        self.t_nucl = float(t_nucl)

    def _sfs(self, W2, Q2, t):
        tn = np.full_like(np.asarray(W2, dtype=float), self.t_nucl)
        return super()._sfs(W2, Q2, tn)


@register
class Jpsi_DiffradVM_Pdet(_PdetVector, Jpsi_DiffradVM):
    name = "jpsi.diffrad_vm_pdet"
    topology = baryon_detected(M_JPSI, meson="J/psi")


@register
class Phi_DiffradVM_Pdet(_PdetVector, Phi_DiffradVM):
    name = "phi.diffrad_vm_pdet"
    topology = baryon_detected(M_PHI, meson="phi")


@register
class Rho_DiffradVM_Pdet(_PdetVector, Rho_DiffradVM):
    name = "rho.diffrad_vm_pdet"
    topology = baryon_detected(M_RHO, meson="rho0")


@register
class Jpsi_VMD2026_Pdet(_PdetVector, Jpsi_VMD2026):
    name = "jpsi.vmd2026_pdet"
    topology = baryon_detected(M_JPSI, meson="J/psi")
