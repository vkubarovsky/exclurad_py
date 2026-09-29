"""
Internal consistency suite -- the checks that do not need external codes.

These are non-circular in the sense that each compares two independent
routes to the same number:

 1. Born closure         exact tensor contraction vs the canonical angular
                         decomposition; all five SF coefficients must be 1.
 2. Current conservation q_mu LR^{mu nu} = 0 for the radiative tensor.
 3. Soft factorization   LR -> 4 F_IR L0 as omega -> 0.
 4. IR bookkeeping       the angular integral of the BS subtraction term
                         must reproduce the analytic d(delta_inf)/d ln v^2.
 5. Slicing vs BS        the generator's positive-definite tail plus the
                         soft remainder must reproduce rc_factor's sigma_obs
                         for every slicing point v_min.
 6. Grid convergence     sigma_F stable against quadrature refinement.
 7. Sampler closure      the sampled photon spectrum must reproduce the
                         density it was drawn from (chi2 against the grid).

Run:  python3 -m exclurad_py.validation.selftest
"""

import numpy as np

from ..core import EventKin, born_xsec, rc_factor
from ..core.born import born_tensors, born_H, born_reference
from ..core.spinors import born_L, rad_L, contractLH, dot4, F_IR
from ..core.radiative import sigma_F
from ..core.virtual import delta_VR_inf
from ..core.hadronic import hadronic_tensor
from ..models import get, TwoSF
from ..generator import TailSampler, choose_vmin

# Tolerances. The quadrature-limited ones (ir, slicing, grid) sit at a few
# times the sigma_F grid error; the algebraic ones (born, current) are at
# machine level. Tightening the former means refining the quadrature, not
# fixing physics.
# `slicing` and `grid` were 2e-3 and had NOT been touched since the suite was
# written.  The default quadrature was coarsened on 2026-09-24 (8d28335,
# (30,16,24,16) -> (20,12,16,12), 2.4x on the radiative mode) and these two
# checks have FAILED on every commit since -- at 3.372e-03 and 3.237e-03
# against a 2e-03 bar.  Verified to be the quadrature and nothing else:
# RC_QUAD=30,16,24,16 restores them to 6.066e-04 and 6.616e-04.
#
# So this was a stale alarm, not a defect, and a permanently red suite is
# worse than no suite -- for five days nobody could use it as a gate, over a
# period in which a million rad events were generated.
#
# 5e-3 is the coarsened quadrature's own error at the std point plus about
# 1.5x headroom.  It is consistent with what that change was accepted on:
# weighted |d eta| 0.132 %, 95th percentile 0.50 % over 740 points of the
# production box (see the note beside quad_default).  IF YOU COARSEN THE
# QUADRATURE AGAIN, these two numbers move and you must revisit this line
# rather than widen it again.
TOL = dict(born_closure=1e-6, current=1e-8, soft=1e-3,
           ir=3e-4, slicing=5e-3, grid=5e-3, sampler_chi2=3.0)


def _std_point():
    MP = 0.938272
    E0, W2, xb = 10.6, 4.0, 0.15
    Q2 = xb * (W2 - MP ** 2) / (1 - xb)
    t, phi = -0.2, 1.0
    sf = get("eta.tableB1", t_nucl=t)
    kin = EventKin(E0, Q2, W2, t, phi, sf.mV, v=0.0, mU=sf.mU)
    return E0, Q2, W2, t, phi, sf, kin


def check_born_closure():
    """Exact contraction must equal the canonical five-term decomposition."""
    _, _, _, _, _, sf, kin = _std_point()
    worst = 0.0
    for h in (+1, -1):
        exact = born_xsec(kin, sf, h)
        comb, eps = born_reference(kin, sf, h)
        # Gamma_v extracted from the machinery itself at unit SFs
        class Unit:
            mV, mU = sf.mV, sf.mU
            def __call__(self, W2, Q2, t):
                o = np.ones_like(np.atleast_1d(np.asarray(W2, float)))
                z = np.zeros_like(o)
                return dict(sT=o, sL=z, sTT=z, sLT=z, sLTp=z)
        gam = born_xsec(kin, Unit(), h)
        pred = gam * float(np.atleast_1d(comb)[0])
        worst = max(worst, abs(exact / pred - 1.0))
    return worst, worst < TOL["born_closure"]


def check_current_conservation():
    _, Q2, _, _, _, sf, kin = _std_point()
    rng = np.random.default_rng(0)
    n = rng.normal(size=(20, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    om = 0.05
    k4 = np.concatenate([np.full((20, 1), om), om * n], axis=1)
    LR, _, _ = rad_L(kin.k1, kin.k2, kin.m, +1, k4)
    qt = kin.q[None, :] - k4
    num = np.abs(np.einsum('nm,m,nmk,k,nk->n', qt, np.array([1., -1, -1, -1]),
                           LR, np.array([1., -1, -1, -1]), qt))
    den = np.abs(np.einsum('m,m,nmk,k,k->n', kin.k1, np.array([1., -1, -1, -1]),
                           LR, np.array([1., -1, -1, -1]), kin.k1))
    r = float(np.max(num / den))
    return r, r < TOL["current"]


def check_soft_factorization():
    _, Q2, _, _, _, sf, kin = _std_point()
    rng = np.random.default_rng(1)
    n = rng.normal(size=(8, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    om = 1e-4
    k4 = np.concatenate([np.full((8, 1), om), om * n], axis=1)
    LR, kap1, kap2 = rad_L(kin.k1, kin.k2, kin.m, +1, k4)
    L0 = born_L(kin.k1, kin.k2, kin.m, +1)
    H0 = born_H(kin, sf)[0]
    lhs = np.real(contractLH(LR, np.broadcast_to(H0, (8, 4, 4))))
    rhs = 4.0 * F_IR(Q2, kin.m, kap1, kap2) * np.real(contractLH(L0, H0))
    r = float(np.max(np.abs(lhs / rhs - 1.0)))
    return r, r < TOL["soft"]


def check_ir_bookkeeping():
    """int_{v_lo}^{v_hi} of the subtraction term == sigma_Born * d(delta_inf)."""
    E0, Q2, W2, t, phi, sf, kin = _std_point()
    sb = born_xsec(kin, sf, +1)
    v_hi = 0.20
    worst = 0.0
    for f in (1e-4, 1e-5, 1e-6):
        v_lo = f * v_hi
        # fine quadrature: this must test the IR identity, not the grid
        fine = dict(n_v=44, n_core=24, n_tail=32, n_phi=24)
        ex = sigma_F(kin, sf, +1, v_hi, v_lo=v_lo, subtract=False, **fine)
        bs = sigma_F(kin, sf, +1, v_hi, v_lo=v_lo, subtract=True, **fine)
        _, di_hi = delta_VR_inf(kin, v_hi)
        _, di_lo = delta_VR_inf(kin, v_lo)
        pred = sb * (di_hi - di_lo)
        worst = max(worst, abs((ex - bs) / pred - 1.0))
    return worst, worst < TOL["ir"]


def check_slicing_vs_bs():
    """Generator decomposition must reproduce rc_factor for every v_min."""
    E0, Q2, W2, t, phi, sf, kin = _std_point()
    v_max = 0.20
    r = rc_factor(Ebeam=E0, Q2=Q2, W2=W2, t=t, phi=phi, sfmodel=sf, h=+1,
                  mx2_cut=v_max + sf.mU ** 2)
    s_obs = r["sigma_obs"]
    worst = 0.0
    for f in (1e-2, 1e-3, 1e-4):
        ts = TailSampler(kin, sf, +1, v_max, v_min=f * v_max)
        ref = ts.sigma_hard_reference()
        worst = max(worst, abs(ts.sigma_hard() / ref - 1.0))
        soft, hard, _ = ts.split(s_obs)
        assert abs((soft + hard) / s_obs - 1.0) < 1e-12
        if soft <= 0:
            return worst, False
    return worst, worst < TOL["slicing"]


def check_grid_convergence():
    E0, Q2, W2, t, phi, sf, kin = _std_point()
    base = sigma_F(kin, sf, +1, 0.20)
    fine = sigma_F(kin, sf, +1, 0.20, n_v=44, n_core=24, n_tail=32, n_phi=24)
    r = abs(base / fine - 1.0)
    return r, r < TOL["grid"]


def check_sampler_closure(n=40000):
    """Sampled ln v spectrum must match the grid density it came from."""
    E0, Q2, W2, t, phi, sf, kin = _std_point()
    v_max = 0.20
    ts = TailSampler(kin, sf, +1, v_max, v_min=2e-3)
    s = ts.sample(np.random.default_rng(11), n)

    nb = 10
    edges = np.linspace(np.log(ts.v_min), np.log(v_max), nb + 1)
    obs, _ = np.histogram(np.log(s["v"]), bins=edges)
    # expected: the grid's own integral in each ln v slab
    I = ts.grid.I.sum(axis=(0, 2, 3))          # per v-cell, summed over rest
    ce = ts.grid.lv_edges
    exp = np.zeros(nb)
    for i in range(len(I)):
        lo, hi = ce[i], ce[i + 1]
        for b in range(nb):
            ov = max(0.0, min(hi, edges[b + 1]) - max(lo, edges[b]))
            if ov > 0:
                exp[b] += I[i] * ov / (hi - lo)
    exp = exp / exp.sum() * obs.sum()
    m = exp > 20
    chi2 = float(np.sum((obs[m] - exp[m]) ** 2 / exp[m]) / max(m.sum() - 1, 1))
    return chi2, chi2 < TOL["sampler_chi2"]


def check_twosf_reduction():
    """TwoSF must zero the interference SFs and change the RC measurably."""
    E0, Q2, W2, t, phi, sf, kin = _std_point()
    full = rc_factor(Ebeam=E0, Q2=Q2, W2=W2, t=t, phi=phi, sfmodel=sf,
                     h=+1, mx2_cut=0.20 + sf.mU ** 2)["eta"]
    two = rc_factor(Ebeam=E0, Q2=Q2, W2=W2, t=t, phi=phi, sfmodel=TwoSF(sf),
                    h=+1, mx2_cut=0.20 + sf.mU ** 2)["eta"]
    return abs(full - two), True     # informational, not a pass/fail


CHECKS = [
    ("Born closure (5 SF coefficients)", check_born_closure),
    ("current conservation of L_R", check_current_conservation),
    ("soft factorization L_R -> 4 F_IR L_0", check_soft_factorization),
    ("IR bookkeeping (subtraction == d delta_inf)", check_ir_bookkeeping),
    ("slicing form vs Bardin-Shumeiko", check_slicing_vs_bs),
    ("sigma_F grid convergence", check_grid_convergence),
    ("sampler closure (chi2/dof of ln v)", check_sampler_closure),
    ("TwoSF: RC shift from interference SFs", check_twosf_reduction),
]


def main():
    print("=" * 70)
    print("exclurad_py internal consistency suite")
    print("=" * 70)
    n_fail = 0
    for label, fn in CHECKS:
        try:
            val, ok = fn()
        except Exception as exc:                       # noqa: BLE001
            print(f"  ERROR  {label}: {type(exc).__name__}: {exc}")
            n_fail += 1
            continue
        tag = "ok  " if ok else "FAIL"
        print(f"  {tag}  {label:<46} {val:.3e}")
        if not ok:
            n_fail += 1
    print("=" * 70)
    print("all checks passed" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())
