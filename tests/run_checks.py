"""
Validation suite for vmrad.

1. Born closure: the exact tensor contraction must reproduce the canonical
   angular decomposition  sT + eps sL + eps c2phi sTT + sqrt(2eps(1+eps)) cphi sLT
   + h sqrt(2eps(1-eps)) sphi sLTp  -- this fixes/verifies all conventions
   (also determines NU_LT, NU_LTP on first run).
2. Gauge invariance & soft factorization of the radiative leptonic tensor.
3. IR consistency: the angular integral of the subtraction term at small v
   must behave as (2 alpha/pi)(l_m - 1)/v * sigma_Born  (the coefficient of
   delta_inf) -- non-circular test of the eikonal normalization.
4. TwoSF reduction: switching off TT/LT/LT' reproduces the (sigma_T,sigma_L)
   result (DIFFRAD-like mode) exactly at phi-averaged level.
5. Grid convergence of sigma_F.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from vmrad.constants import ALPHA, M_P, M_E, GEV2_TO_NB
from vmrad.kinematics import EventKin
from vmrad.spinors import born_L, rad_L, contractLH, F_IR, dot4
from vmrad.hadronic import photon_basis, hadronic_tensor, TwoSF
from vmrad.born import born_xsec, born_tensors, born_H, _sf_to_gev
from vmrad.models import demo_rho
from vmrad.radiative import sigma_F, _channel_nodes
from vmrad.rc import rc_factor

EB, Q2, W2, T, PHI = 10.6, 2.5, 12.0, -0.4, 1.1
SF = demo_rho()


def hdr(s):
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)


# ------------------------------------------------------------------ 1. closure
hdr("1. BORN CLOSURE (fixes conventions)")
kin = EventKin(EB, Q2, W2, T, PHI, SF.mV, v=0.0)
eps = kin.epsilon()
L0p = born_L(kin.k1, kin.k2, kin.m, +1)
L0m = born_L(kin.k1, kin.k2, kin.m, -1)

# contraction with each unit structure separately
qt = kin.q[None, :]; pp = kin.p[None, :]; pv = kin.pv[None, :]
W2t = np.array([kin.W2])
UNIT = {'sT': 0, 'sL': 1, 'sTT': 2, 'sLT': 3, 'sLTp': 4}
C = {}
for key in UNIT:
    sf = {k: np.array([1.0 if k == key else 0.0]) for k in UNIT}
    H = hadronic_tensor(qt, pp, pv, W2t, sf, M=kin.M)[0]
    C[key] = (contractLH(L0p, H), contractLH(L0m, H))

CT = np.real(C['sT'][0])
print(f"eps (from y-formula)        = {eps:.10f}")
print(f"C_L/C_T                     = {np.real(C['sL'][0])/CT:.10f}")
print(f"C_TT/(C_T eps cos2phi)      = "
      f"{np.real(C['sTT'][0])/CT/(eps*np.cos(2*PHI)):.10f}")
print(f"C_LT/(C_T sq2e(1+e) cosphi) = "
      f"{np.real(C['sLT'][0])/CT/(np.sqrt(2*eps*(1+eps))*np.cos(PHI)):.10f}")
den = np.sqrt(2 * eps * (1 - eps)) * np.sin(PHI)
print(f"C_LTp/(C_T sq2e(1-e) sinphi), h=+1: "
      f"{np.real(C['sLTp'][0])/CT/den:.10f}")
print(f"C_LTp/(C_T sq2e(1-e) sinphi), h=-1: "
      f"{np.real(C['sLTp'][1])/CT/den:.10f}")
print(f"(imag parts, should be ~0): "
      f"{max(abs(np.imag(C[k][0])) for k in C)/CT:.2e}")

# full closure vs canonical combination
from vmrad.born import born_reference
for h in (+1, -1):
    sig = born_xsec(kin, SF, h)
    comb, _ = born_reference(kin, SF, h)
    H0 = born_H(kin, SF)[0]
    L0 = L0p if h == +1 else L0m
    _, pref = born_tensors(kin, h)
    # Gamma_eff/(2pi) from unit-T contraction:
    gam_eff = pref * CT
    print(f"h={h:+d}:  dsig/dQ2dW2dtdphi = {sig:.6e} nb/GeV^6 ; "
          f"sig/(Gamma_eff*comb) = {sig/(gam_eff*comb):.10f}")

# ------------------------------------------------- 2. gauge + soft factorization
hdr("2. GAUGE INVARIANCE & SOFT FACTORIZATION of L^R")
rng = np.random.default_rng(1)
nh = rng.normal(size=(6, 3)); nh /= np.linalg.norm(nh, axis=1, keepdims=True)
om = 1e-4 * np.ones(6)
k4 = np.concatenate([om[:, None], om[:, None] * nh], axis=1)
LR, kap1, kap2 = rad_L(kin.k1, kin.k2, kin.m, +1, k4)
L0 = L0p
fir = F_IR(kin.Q2, kin.m, kap1, kap2)
rat = []
for i in range(6):
    r = np.real(contractLH(LR[i], born_H(kin, SF)[0])) / \
        (4 * fir[i] * np.real(contractLH(L0, born_H(kin, SF)[0])))
    rat.append(r)
print("LR:H / (4 F_IR L0:H) at omega=1e-4  :", np.array_str(np.array(rat), precision=6))
# current conservation at the virtual-photon vertex: qt_mu LR^{mu nu} = 0
S4 = np.array([1., -1., -1., -1.])
gau, scale = [], []
for i in range(6):
    qt = kin.q - k4[i]
    gau.append(abs(np.einsum('m,m,n,n,mn->', qt, S4, qt, S4, LR[i])))
    scale.append(abs(np.einsum('m,m,n,n,mn->', kin.k1, S4, kin.k1, S4, LR[i])))
print("  |qt.LR.qt| / |k1.LR.k1|           :",
      np.array_str(np.array(gau) / np.array(scale), precision=3))

# ------------------------------------------------------------- 3. IR slope
hdr("3. IR SLOPE: angular integral of subtraction term vs (2a/pi)(l_m-1)/v")
lm = np.log(kin.Q2 / kin.m ** 2)
L0, prefB = born_tensors(kin, +1)
H0 = born_H(kin, SF)[0]
LH0 = np.real(contractLH(L0, H0))
for v in (1e-4, 1e-3, 1e-2):
    kinv = EventKin(EB, Q2, W2, T, PHI, SF.mV, v=v)
    tot = 0.0
    for ci, kv in enumerate((kin.k1, kin.k2)):
        nhat, wang, kap_own = _channel_nodes(kv[1:], kin.m, 24, 32, 24)
        kh1 = np.maximum(kin.k1[0] - nhat @ kin.k1[1:], 1e-30)
        kh2 = np.maximum(kin.k2[0] - nhat @ kin.k2[1:], 1e-30)
        if ci == 0: kh1 = kap_own
        else:       kh2 = kap_own
        wch = (1 / (kh1 if ci == 0 else kh2) ** 2) / (1 / kh1 ** 2 + 1 / kh2 ** 2)
        z = kinv.Lam[0] - nhat @ kinv.Lam[1:]
        om = v / (2 * z)
        fir = F_IR(kin.Q2, kin.m, 2 * om * kh1, 2 * om * kh2)
        tot += np.sum(wang * wch * (v / z ** 2) * 4 * fir * LH0 / kin.Q2 ** 2)
    # expected (from d delta_inf/d v_m = (2 alpha/pi)(l_m-1)/v_m and the
    # prefactor bookkeeping): I_ang = 32 pi (l_m - 1)/v  as v -> 0
    got = tot / (LH0 / kin.Q2 ** 2)
    exp = 32 * np.pi * (lm - 1) / v
    print(f"v={v:8.1e}:  I_ang = {got:.6e},  32pi(l_m-1)/v = {exp:.6e},  "
          f"ratio = {got/exp:.6f}")

# ------------------------------------------------------------ 4. TwoSF reduction
hdr("4. FULL-SF vs TwoSF (DIFFRAD-like) and RC factors")
res_full = rc_factor(EB, Q2, W2, T, PHI, SF, h=+1, mx2_cut=(M_P + 0.3) ** 2)
res_two = rc_factor(EB, Q2, W2, T, PHI, TwoSF(SF), h=+1, mx2_cut=(M_P + 0.3) ** 2)
for name, r in (("full SF set", res_full), ("sigma_T/L only", res_two)):
    print(f"{name:>16}:  eta = {r['eta']:.5f}   delta_VR={r['delta_VR']:+.5f} "
          f"delta_inf={r['delta_inf']:+.5f} delta_vac={r['delta_vac']:+.5f} "
          f"sigF/sigB={r['sigma_F']/r['sigma_born']:+.5f}")

# ------------------------------------------------------------- 5. convergence
hdr("5. GRID CONVERGENCE of sigma_F")
kin0 = EventKin(EB, Q2, W2, T, PHI, SF.mV, v=0.0)
vm = res_full['v_max']
base = None
for (nv, nc, nt, nph) in ((20, 12, 16, 12), (30, 16, 24, 16), (45, 24, 36, 24)):
    val = sigma_F(kin0, SF, +1, vm, n_v=nv, n_core=nc, n_tail=nt, n_phi=nph)
    print(f"n_v={nv:3d} n_core={nc:3d} n_tail={nt:3d} n_phi={nph:3d} : "
          f"sigma_F = {val:.8e}")
    base = val
print("\nAll checks done.")
