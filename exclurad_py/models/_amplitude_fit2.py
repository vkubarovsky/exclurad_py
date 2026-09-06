"""Amplitude-level model for pi0/eta electroproduction.

Variables: the channel amplitudes of the pi0_amplitude note,
    T00p = T_00^(+)  longitudinal, recoil non-flip   (= 0 at this level)
    T00m = T_00^(-)  longitudinal, recoil flip       (complex: the fitted phase)
    T01p = T_01^(+)  natural transverse, non-flip    (= 0: the C=E relation)
    T01m = T_01^(-)  natural transverse, flip        (= D/2, real)
    U01p = U_01^(+)  unnatural transverse, non-flip  (real at this level)
    U01m = U_01^(-)  unnatural transverse, flip      (= D/2 = T01m, forward kinematics)

Fourier dictionary (verified in the note against the QED photon density matrix):
    sigma_L        = sum |T00|^2
    sigma_T        = sum |T01|^2 + sum |U01|^2
    sigma_TT       = sum |T01|^2 - sum |U01|^2
    sigma_LT + i sigma_LT'      = -sqrt2 * sum T00^* T01
    sigma_LT'^LL + i sigma_LT^UL = -sqrt2 * sum T00^* U01
    sigma_T' + i sigma_TT^UL     =  2     * sum T01^* U01

Flavour layer: every amplitude is (2X_u + X_d)/(3 sqrt2) for pi0 p,
(2X_u - X_d)/(3 sqrt6 k) for eta p, (X_u + 2X_d)/(3 sqrt2) for pi0 n.
k = 0.863 for the transverse twist-3 sector (chiral condensate factors),
k = 0.695 for the longitudinal twist-2 sector (decay constants only).

Parameter vector p (21):
  0-3   HT_u  N,b,b',c   slope(xB) = b + b' ln xB        }  D-sector (T01m=U01m), P2 seed
  4-7   HT_d  N,b,b',c        }
  8-12  ET_u  N,b,b',c,b2     }  U01p sector, P2 seed
  13    R_ET  (ET_d = R*ET_u*exp(db*t), prior 0.54+-0.15)
  14    db
  15-17 L_u   N,b,c           }  T00m modulus (sqrt(-t') prefactor)
  18    R_L   (L_d = R_L*L_u)
  19-20 delta0, delta1        }  arg T00m = delta0 + delta1*t, flavour-common
  21    L_bx  (optional xB-slope of the L sector; 0 if absent)
Level-3 unlock (present only if len(p) > 22):
  22    rho_CE   |T01p| = rho_CE * |U01p|      (C != E)
  23    phi_CE   arg T01p
  24    phi_w    arg U01p                      (w-sector phase)
  25    rho_nf   |T00p| = rho_nf * |T00m|
  26    phi_nf   arg T00p
"""
import math, cmath, os
import numpy as np

_PT = None
def _phi_du(xi):
    """Computed relative u-d phase, interpolated from phase_table.npz."""
    global _PT
    if _PT is None:
        d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_table.npz"))
        _PT = (d["xi"], d["ET"])
    return float(np.interp(xi, _PT[0], _PT[1]))

Mp, Mpi0, Meta = 0.938272, 0.1349766, 0.547862
ALPHA, HC2, PI = 0.00729927, 389379.36, math.pi
# Slope convention (2026-08-28): the t-slopes are written as b + b' ln xB,
# i.e. b is the slope extrapolated to xB = 1 and -b' is the Regge alpha'.
# The earlier convention had L = ln xB - ln 0.15, so that b was the slope at
# xB = 0.15; convert old parameter files with reparam.py (b_new = b + 1.897 b').
# NOTE exclurad_py/models/_amplitude_fit.py and its amp2021/amp2026 .npy files
# are still in the OLD convention - convert on install, never copy raw.
K_T = 0.863
K_L = 1.0/((math.cos(math.radians(-21.2))
            - math.sqrt(2)*(1.17/1.26)*math.sin(math.radians(-9.2)))*1.26)  # = 0.695
S2, S6 = math.sqrt(2), math.sqrt(6)

def tmin(mM, Q2, xB):
    W2 = Q2*(1/xB - 1) + Mp*Mp
    W = math.sqrt(W2)
    E1 = (W2 + Q2 + Mp*Mp)/(2*W); P1 = math.sqrt(E1*E1 - Mp*Mp)
    E3 = (W2 - mM*mM + Mp*Mp)/(2*W); P3 = math.sqrt(E3*E3 - Mp*Mp)
    return (Q2 + mM*mM)**2/(4*W2) - (P1 - P3)**2   # true (negative) t_min

def phase_factor(Q2, xB):
    W2 = Q2*(1/xB - 1) + Mp*Mp
    lam = W2*W2 + Q2*Q2 + Mp**4 + 2*W2*Q2 - 2*W2*Mp*Mp + 2*Q2*Mp*Mp
    return 16*PI*(W2 - Mp*Mp)*math.sqrt(lam)

def Afac(Q2, xB):
    return 4*PI*ALPHA/2/phase_factor(Q2, xB)/(Q2*Q2)*HC2

def ksi(xB, Q2):
    return xB/(2 - xB)*(1 + Mp*Mp/Q2)

def epsilon(xB, Q2, E):
    y = Q2/(2*Mp*xB*E); g2 = 4*Mp*Mp*xB*xB/Q2
    return (1 - y - 0.25*g2*y*y)/(1 - y + y*y/2 + 0.25*g2*y*y)

def _flavour(p, t, xB, Q2):
    L = math.log(xB)
    HTu = p[0]*math.exp((p[1]+p[2]*L)*t)*Q2**(p[3]/2)
    HTd = p[4]*math.exp((p[5]+p[6]*L)*t)*Q2**(p[7]/2)
    if len(p) > 27:
        # Independent Ebar_T^d block (VPK, 2026-08-28): same functional form as u,
        # its own N, b, b', nQ.  b2 is gone; its slot p[12] now carries b'_d.
        #   p[13] = N_d   p[14] = b_d   p[12] = b'_d   p[27] = nQ_d
        ETu = p[8]*math.exp((p[9]+p[10]*L)*t)*Q2**(p[11]/2)
        ETd = p[13]*math.exp((p[14]+p[12]*L)*t)*Q2**(p[27]/2)
        if len(p) > 31:
            # optional xB shape F(xB) = xB^alpha (1-xB)^n, separately for u and d
            #   p[28] alpha_u  p[29] n_u   p[30] alpha_d  p[31] n_d
            ETu *= xB**p[28]*(1.0-xB)**p[29]
            ETd *= xB**p[30]*(1.0-xB)**p[31]
    else:
        # legacy 27-parameter layout: ET_d tied to ET_u by a ratio and a slope shift
        ETu = p[8]*math.exp((p[9]+p[10]*L)*t + p[12]*t*t)*Q2**(p[11]/2)
        ETd = p[13]*ETu*math.exp(p[14]*t)
    bxL = p[21] if len(p) > 21 else 0.0
    Lu  = p[15]*math.exp((p[16] + bxL*L)*t)*Q2**(p[17]/2)
    Ld  = p[18]*Lu
    return (HTu, HTd), (ETu, ETd), (Lu, Ld)

def _combine(pair, ch, k):
    u, d = pair
    if ch == "pi0p": return (2*u + d)/(3*S2)
    if ch == "etap": return (2*u - d)/(3*S6*k)
    if ch == "pi0n": return (u + 2*d)/(3*S2)
    raise ValueError(ch)

def amplitudes(p, ch, t, xB, Q2):
    """Return dict of channel amplitudes (complex) at (t<0, xB, Q2)."""
    mM = Meta if ch == "etap" else Mpi0
    tp = t - tmin(mM, Q2, xB)
    if tp >= 0: return None
    HT, ET, LL = _flavour(p, t, xB, Q2)
    if len(p) > 36 and p[35] <= -9.0:
        # phases COMPUTED from the convolution (phase_table.npz), not fitted:
        # phi_du(xi) = arg<F^d> - arg<F^u> for a GK-like x shape.
        ph = _phi_du(ksi(xB, Q2))
        HT = (HT[0], HT[1]*cmath.exp(1j*ph))
        ET = (ET[0], ET[1]*cmath.exp(1j*ph))
    elif len(p) > 36:
        # Relative u-d phases (2026-08-29).  Each flavour convolution is complex
        # in its own right: Im = pi F(xi,xi,t) from the pole, Re from the
        # principal value, and the two have different t-slopes, so u and d do not
        # share a phase.  The overall phase of a channel is unobservable, so u is
        # kept real and only d carries one.  The flavour sum is then complex, and
        # the near-cancellation 2u - d in the eta channel is no longer exact.
        HT = (HT[0], HT[1]*cmath.exp(1j*p[34]))
        ET = (ET[0], ET[1]*cmath.exp(1j*p[35]))
        LL = (LL[0], LL[1]*cmath.exp(1j*p[36]))
    A  = Afac(Q2, xB)
    xi = ksi(xB, Q2)
    kin = -tp/(8*Mp*Mp)
    D    = math.sqrt(max(2*A*(1 - xi*xi), 0.0))*_combine(HT, ch, K_T)
    U01p = math.sqrt(A*kin)*_combine(ET, ch, K_T)
    Lmag = math.sqrt(A*kin)*_combine(LL, ch, K_L)
    delta = p[19] + p[20]*t
    T00m = Lmag*cmath.exp(1j*delta)
    if len(p) > 22:
        rho_CE, phi_CE, phi_w, rho_nf, phi_nf = p[22], p[23], p[24], p[25], p[26]
        T01p = rho_CE*abs(U01p)*cmath.exp(1j*phi_CE)
        U01c = U01p*cmath.exp(1j*phi_w)
        T00p = rho_nf*abs(Lmag)*cmath.exp(1j*phi_nf)
    else:
        T01p, U01c, T00p = 0.0+0j, U01p+0j, 0.0+0j
    return dict(T00p=T00p, T00m=T00m,
                T01p=T01p, T01m=D/2+0j,
                U01p=U01c, U01m=D/2+0j)

def structure(p, ch, t, xB, Q2):
    """All nine structure functions from the Gram dictionary."""
    a = amplitudes(p, ch, t, xB, Q2)
    if a is None: return None
    T00 = (a["T00p"], a["T00m"]); T01 = (a["T01p"], a["T01m"]); U01 = (a["U01p"], a["U01m"])
    s2 = lambda v: sum(abs(x)**2 for x in v)
    ip = lambda x, y: sum(xc.conjugate()*yc for xc, yc in zip(x, y))
    sL, sT01, sU01 = s2(T00), s2(T01), s2(U01)
    uv = -S2*ip(T00, T01)
    uw = -S2*ip(T00, U01)
    vw =  2*ip(T01, U01)
    return dict(L=sL, T=sT01+sU01, TT=sT01-sU01,
                LT=uv.real, LTp=uv.imag,
                LTpLL=uw.real, LTUL=uw.imag,
                Tp=vw.real, TTUL=vw.imag)

def sigma_U(p, ch, t, xB, Q2, E):
    s = structure(p, ch, t, xB, Q2)
    if s is None: return None
    return s["T"] + epsilon(xB, Q2, E)*s["L"]

def bsa_sinphi(p, ch, t, xB, Q2, E):
    s = structure(p, ch, t, xB, Q2)
    if s is None: return None
    e = epsilon(xB, Q2, E)
    return math.sqrt(2*e*(1 - e))*s["LTp"]/(s["T"] + e*s["L"])
