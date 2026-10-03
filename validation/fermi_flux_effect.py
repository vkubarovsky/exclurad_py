#!/usr/bin/env python3
"""How much the Moller flux factor moves sigma_total, measured as a MEAN.

    python3 validation/fermi_flux_effect.py [channel ...]      # pi0, jpsi

WHY NOT SIMPLY RUN BOTH ARMS AND DIVIDE.  Because the honest answer is a
fraction of a per cent and the difference of two Monte Carlo integrals cannot
see it: 4 x 15000 events buy ~0.4 % on the ratio, 3 x 4000 buy 0.8 %.  The
first attempt at this comparison returned -1.9 % from a single seed, which
looked like a real suppression and was noise.

The flux rejection sits inside _draw_fermi and returns BEFORE `n_trials += 1`,
so a thinned draw enters neither the numerator nor the denominator of
sigma = (n_accept / n_trials) * I_maj.  Conditional on reaching the counter the
Fermi momentum is then distributed as n(k) F(p) instead of n(k), and with
eps(p) the probability that the rest of the chain accepts,

    sigma_on / sigma_off = <eps F> / (<F> <eps>) = <F>_accepted / <F>_drawn

both of which one run measures.  <F>_accepted is an average over accepted
events, so its error is rms(F)/sqrt(N_acc) -- 0.04 % on 24-32k events, twenty
times better than the difference of the integrals, from a quarter of the cpu.
F is recomputed exactly from the spectator the event already carries, so
nothing has to be instrumented.

NORMALISE BY <F>_drawn, NOT BY 1.  Over an isotropic distribution <1 - beta_z>
= 1 identically, and for pi0 the measured value is 0.99997.  Near the J/psi
threshold it is 1.0055: the m* and s guards cut exactly the directions with the
smallest F, so the identity no longer holds where the guards bite.  Dividing by
1 there would report +1.65 % instead of +1.10 %.

MEASURED 2026-09-27, Born, AV18, 10.6 GeV:

    pi0.amp2609    Q2 0.8-10,  W2 3.24-16    +0.20 +- 0.04 %
    jpsi.roberts   Q2 .01-2.0, W2 16.5-20.0  +1.10 +- 0.04 %

CROSS-CHECKED against the thing it estimates.  Running both arms for real,
pi0, 4 x 15000 each, gave -0.66 +- 0.49 %, which is 1.8 sigma from the +0.20 %
above and settles nothing by itself -- that is the point.  Its four per-seed
ratios were 1.0023, 1.0012, 0.9867, 0.9832: a single seed anywhere in that
spread can read -1.9 %, which is exactly what the first attempt did.  Note the
per-seed pairing does NOT cancel the noise, although both arms start from the
same seed: with the rejection live the thinned draws consume uniforms, so the
two streams diverge on the first rejected draw and the seeds are common in name
only.

Positive in both, and that is forced: E* = E F E_N / m*, so a larger F is a
larger collision energy, and both cross sections rise with it.  The size
follows how steeply sigma depends on s -- slowly over the pi0 W range, steeply
at the J/psi threshold, a factor 5.5 between them.  Any future channel should
come out positive as well; a negative result means a bookkeeping error, not a
new effect.
"""

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from exclurad_py.models import get                                # noqa: E402
from exclurad_py.generator.events import Box, Generator           # noqa: E402
from exclurad_py.generator.fermi import M_D, flux_factor          # noqa: E402

CASES = {
    "pi0": dict(model="pi0.amp2609", model_kw=dict(t_nucl=-0.30), E=10.6,
                box=Box(Q2=(0.8, 10.), W2=(3.24, 16.), t=(-2., -0.001)),
                n=8000, seeds=(101, 202, 303, 404), gen_kw={}),
    "jpsi": dict(model="jpsi.roberts2026", model_kw={}, E=10.6,
                 box=Box(Q2=(0.01, 2.0), W2=(16.5, 20.0), t=(-2., -0.05)),
                 n=6000, seeds=(11, 22, 33, 44), gen_kw=dict(safety=4.0)),
}


def measure(c):
    sf = get(c["model"], **c["model_kw"])
    sf.Ebeam = c["E"]
    F, drawn_sum, drawn_n = [], 0.0, 0
    for seed in c["seeds"]:
        g = Generator(sf, Ebeam=c["E"], box=c["box"], mx2_cut=None, helicity=0,
                      seed=seed, born_only=True, fermi="av18",
                      majorant_shape=(4, 4, 4, 6), **c["gen_kw"])
        g.setup()
        # flux OFF: the rejection always passes, and the same uniform is still
        # drawn, so the two arms would share an RNG stream if both were run.
        g._flux_max = 1e-12
        for ev in g.generate(c["n"]):
            s = np.asarray(ev["spectator"], float)
            F.append(float(flux_factor(np.array([M_D - s[0], -s[1], -s[2], -s[3]]))))
        drawn_sum += g.sum_flux
        drawn_n += g.n_flux_drawn
        print(f"    seed {seed}: {c['n']} events", flush=True)
    F = np.asarray(F)
    mean_drawn = drawn_sum / drawn_n
    ratio = F.mean() / mean_drawn
    err = F.std(ddof=1) / np.sqrt(len(F)) / mean_drawn
    return ratio, err, F, mean_drawn


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(CASES)):
        c = CASES[name]
        print(f"\n  {name}: {c['model']}, {len(c['seeds'])} x {c['n']}, Born, AV18")
        ratio, err, F, mean_drawn = measure(c)
        print(f"  N_acc = {len(F)}   <F> разыгранных = {mean_drawn:.6f}")
        print(f"  <F> принятых = {F.mean():.5f} +- {F.std(ddof=1)/np.sqrt(len(F)):.5f}"
              f"   (rms {F.std():.3f})")
        print(f"  sigma(вкл)/sigma(выкл) = {ratio:.5f} +- {err:.5f}"
              f"   ->  {100*(ratio-1):+.2f} +- {100*err:.2f} %")
        if ratio < 1.0 - 3 * err:
            print("  <<< ОТРИЦАТЕЛЬНО: этого быть не должно, см. заголовок")
