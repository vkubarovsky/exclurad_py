"""Validation of the decay-lepton radiation, exclurad_py/generator/decay_rad.py.

    python3 validation/jpsi_decay_fsr.py

Five checks, each a pass/fail line.  Results 2026-10-07 in the comments.

1. Ward identities of the trace matrix element: k_nu O^{mu nu} = 0 (photon
   gauge invariance) and P_mu O^{mu nu} = 0 (vector current conservation),
   which justify the -g polarisation sums.            7e-13, 2e-9 of |M|^2
2. The closed form me2_ab equals the trace form away from the collinear
   region (where the trace loses precision).          ratio 1 to 1e-12
3. Integrated over the angle, the exact matrix element reproduces the
   Ehlotzky-Mitter photon spectrum used for the energy -- the energy and the
   angles are one model.                              1.00000, k = 1 MeV..1.5 GeV
4. Sampled M(ee) fractions against the analytic exponentiated spectrum.
                                                      within ~1 sigma at 2M draws
5. Sampled lepton-photon angle at fixed k against the matrix element, and
   four-momentum closure of full lab decays.          chi2/ndf ~ 1, closure 1e-12
"""
import os
import functools
print = functools.partial(print, flush=True)
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exclurad_py.generator.decay_rad import (                    # noqa: E402
    ALPHA, M_E, METRIC, ONE, DecayFSR, PhotonEnergy, _G0, _frame_t, _slash,
    amp_tensor, me2_ab, me2_trace)

M = 3.0969
m = M_E
E = M / 2.0
ok_all = True


def report(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# 1 -------------------------------------------------------------------------
s = M * M - 2 * M * 0.3
p1, p2, k, a, b, _ = _frame_t(s, M, np.array([0.2, 0.7, 1.0, 1.4, 1.9]))
O = amp_tensor(p1, p2, k)


def tnorm(X):
    Xb = np.einsum("ab,...mcb,cd->...mad", _G0, X.conj(), _G0)
    return np.real(np.einsum("...ab,...mbc,...cd,...mda->...",
                             _slash(p1) + m * ONE, X, _slash(p2) - m * ONE, Xb))


scale = np.abs(me2_trace(p1, p2, k)).max()
w_k = np.abs(tnorm(np.einsum("...n,...mnab->...mab", k * METRIC, O))).max() / scale
w_P = np.abs(tnorm(np.einsum("...m,...mnab->...nab", (p1 + p2 + k) * METRIC, O))).max() / scale
report("Ward identities", w_k < 1e-10 and w_P < 1e-10, f"photon {w_k:.1e}, current {w_P:.1e}")

# 2 -------------------------------------------------------------------------
r = me2_ab(a, b, M * M, m * m) / me2_trace(p1, p2, k)
report("closed form = traces", np.allclose(r, 1, atol=1e-9), f"max |ratio-1| {np.abs(r - 1).max():.1e}")


# 3 -------------------------------------------------------------------------
def ehlotzky(kk):
    x = kk / E
    return (2 * ALPHA / np.pi) * (2 * np.log(2 * E / m * np.sqrt(1 - x)) - 1) / kk * (1 - x + x * x / 2)


def me_integral(kk):
    """int |M|^2 dx1 at fixed photon energy; both collinear peaks on log grids."""
    s = M * M - 2 * M * kk
    rs = np.sqrt(s)
    beta = np.sqrt(1 - 4 * m * m / s)
    w = (M * M - s) / (2 * rs)
    Es = rs / 2
    y = np.linspace(np.log(1 - beta), 0.0, 40001)
    t = np.exp(y)
    I = sum(np.trapezoid(me2_ab(Es * w * t1, Es * w * t2, M * M, m * m) * t, y)
            for t1, t2 in ((t, 2 - t), (2 - t, t)))
    return I * 2 * Es * w / M ** 2


ks = np.array([0.001, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 1.3, 1.5])
rr = np.array([me_integral(x) / ehlotzky(x) for x in ks])
rr /= rr[0]
report("angle-integrated ME = Ehlotzky spectrum", np.abs(rr - 1).max() < 1e-3,
       "ratios " + " ".join(f"{x:.5f}" for x in rr))

# 4 -------------------------------------------------------------------------
sp = PhotonEnergy(M)
rng = np.random.default_rng(1)
N = 2_000_000
mee = np.sqrt(M * M - 2 * M * np.interp(rng.random(N), sp.F, sp.K))
pulls = []
for mc in (3.09, 3.0, 2.9, 2.7):
    p = 1 - sp.cdf((M * M - mc * mc) / (2 * M))
    pulls.append(((mee < mc).mean() - p) / np.sqrt(p * (1 - p) / N))
report("M(ee) fractions vs analytic", max(abs(x) for x in pulls) < 3.5,
       "pulls " + " ".join(f"{x:+.1f}" for x in pulls)
       + f";  P(M<3.0) = {1 - sp.cdf((M*M-9.0)/(2*M)):.4f}, A = F(1 MeV) = {sp.cdf(1e-3):.3f}")

# 5 -------------------------------------------------------------------------
d = DecayFSR(M)
rng = np.random.default_rng(3)
chis = []
for kk in (0.01, 0.3, 1.0):
    s = M * M - 2 * M * kk
    rs = np.sqrt(s); Es = rs / 2; w = (M * M - s) / (2 * rs); beta = np.sqrt(1 - 4 * m * m / s)
    t = np.array([d._t_star(s, M, rng) for _ in range(20000)])
    u = np.minimum(t, 2 - t)
    edges = np.linspace(np.log(1 - beta), 0, 25)
    h, _ = np.histogram(np.log(u), edges)
    yy = np.linspace(edges[0], 0, 200001); tt = np.exp(yy)
    dens = (me2_ab(Es * w * tt, Es * w * (2 - tt), M * M, m * m)
            + me2_ab(Es * w * (2 - tt), Es * w * tt, M * M, m * m)) * tt
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(yy))])
    exp = np.diff(np.interp(edges, yy, cdf / cdf[-1])) * len(t)
    chis.append(((h - exp) ** 2 / np.maximum(exp, 1)).sum() / (len(h) - 1))
pV = np.array([np.sqrt(M * M + 36.25), 0.5, 0.0, 6.0])
clos = max(np.abs(sum(x for x in d.decay(pV, rng) if x is not None) - pV).max()
           for _ in range(20000))
report("angle sampler and closure", max(chis) < 2.0 and clos < 1e-9 and d.n_viol == 0,
       "chi2/ndf " + " ".join(f"{x:.2f}" for x in chis)
       + f";  closure {clos:.1e};  bound violations {d.n_viol}")

print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
