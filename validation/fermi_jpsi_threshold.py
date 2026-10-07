#!/usr/bin/env python3
"""Acceptance criterion 4 of JPSI_FERMI_BRIEF.md: sub-threshold J/psi yield on a deuteron.

Two independent checks of the same physics.

A. Kinematics only, no cross section.  Draw AV18 spectator momenta, form the struck
   neutron by the spectator prescription, and count the fraction of configurations for
   which gamma n -> J/psi n is open at a fixed photon energy (s > (M_N + M_J/psi)^2, photon
   along +z).  Reproduces the table of the brief: 0.84 % at 7.0 GeV, 24.6 % at 8.0 GeV,
   44.4 % at the free threshold 8.20 GeV, with <|p_s|> and the fraction above 300 MeV/c.
   Also prints how much of the allowed set an m* floor removes, which is the number the
   choice fermi_m_star_min = 0.10 rests on.

B. The generator.  jpsin.roberts2026 with fermi = av18, Born, the whole quasi-real box
   (no y cut: the generator's y is the struck-nucleon-frame value, so the LAB photon
   energy E_gamma = E_beam - E' is read off the written electron), with the J/psi floor
   0.10, with the pi0 default floor 0.30, and on a free neutron.  What must hold: events
   below 8.20 GeV exist only with Fermi motion, and their spectators are hard, as the
   kinematics of A demand (A is unweighted by the cross section, B is weighted, so the
   means are compared by eye, not asserted).

    python3 validation/fermi_jpsi_threshold.py [n_kin] [n_gen]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exclurad_py import models                                          # noqa: E402
from exclurad_py.core.constants import M_JPSI, M_N                       # noqa: E402
from exclurad_py.generator import Box, Generator                         # noqa: E402
from exclurad_py.generator.fermi import FermiSampler, split_deuteron    # noqa: E402

EB = 10.6
WTH2 = (M_N + M_JPSI) ** 2
E_GAMMAS = (7.0, 7.5, 8.0, 8.2, 9.0, 10.0)


def kinematic_table(n):
    rng = np.random.default_rng(20260926)
    fs = FermiSampler("av18")
    P = np.array([fs.draw(rng) for _ in range(n)])
    E, pz, ms = np.array([(s4[0], s4[3], m) for s4, _, m in
                          (split_deuteron(p, "n") for p in P)]).T
    ps = np.linalg.norm(P, axis=1)
    rows = {}
    print("A. AV18 kinematics, gamma n -> J/psi n, photon along +z, %d configurations" % n)
    print("   E_gamma   allowed   <|p_s|>   p_s>300   removed by m* floor: 0.30   0.20   0.10 GeV")
    for Eg in E_GAMMAS:
        s = ms ** 2 + 2 * Eg * (E - pz)
        ok = s > WTH2
        row = dict(allowed=ok.mean(), ps_mean=ps[ok].mean() if ok.any() else np.nan,
                   ps300=(ps[ok] > 0.3).mean() if ok.any() else np.nan,
                   f030=(ms[ok] < 0.30).mean(), f020=(ms[ok] < 0.20).mean(), f010=(ms[ok] < 0.10).mean())
        rows[Eg] = row
        print("   %5.2f    %6.2f %%   %4.0f MeV   %5.1f %%                      %5.2f %%  %5.2f %%  %5.2f %%"
              % (Eg, 100 * row["allowed"], 1e3 * row["ps_mean"], 100 * row["ps300"],
                 100 * row["f030"], 100 * row["f020"], 100 * row["f010"]))
    return rows


def generate(model, fermi, floor, nev, seed=7):
    """Whole quasi-real box, NO y cut: with Fermi motion the generator's y is the
    struck-nucleon-frame value, so the lab photon energy has to be read off the
    written electron afterwards.  Born, no radiator, so E_gamma = E_beam - E'."""
    sf = models.get(model)
    sf.Ebeam = EB
    g = Generator(sf, Ebeam=EB, box=Box(Q2=(0.01, 0.10), W2=(16.29, 30.0), t=(-6.0, -0.001)),
                  mx2_cut=None, helicity=0, seed=seed, born_only=True, fermi=fermi,
                  fermi_m_star_min=floor, majorant_shape=(6, 6, 6, 8))
    g.setup()
    ev = g.generate(nev)
    return g, ev, g.report()


BINS = ((6.5, 7.5), (7.5, 8.0), (8.0, 8.2), (8.2, 9.0), (9.0, 10.6))


def main():
    n_kin = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    rows = kinematic_table(n_kin)

    print("\nB. Generator, jpsin.roberts2026, Born, Q2 0.01-0.1, whole box, %d events per run;" % n_gen)
    print("   events binned in LAB E_gamma = E_beam - E'_lab.  Free threshold 8.20 GeV.")
    print("   target     floor   " + "  ".join("%4.1f-%4.1f" % b for b in BINS) + "   killed draws  <|p_s|> below thr.  p_s>300 below thr.")
    failures, res = [], {}
    for label, fermi, floor in (("deuteron", "av18", 0.10), ("deuteron", "av18", 0.30), ("free n", "none", 0.10)):
        g, ev, rep = generate("jpsin.roberts2026", fermi, floor, n_gen)
        Eg = np.array([EB - e["scattered"][0] for e in ev])
        ps = np.array([e.get("p_fermi", 0.0) for e in ev])
        counts = [int(((Eg >= a) & (Eg < b)).sum()) for a, b in BINS]
        below = Eg < 8.20
        killed = g.n_fermi_reject / max(rep.get("trials", 0) + g.n_fermi_reject, 1)
        res[(label, floor)] = (counts, below.sum())
        print("   %-9s  %.2f    " % (label, floor) + "  ".join("%9d" % c for c in counts)
              + "   %8.4f      %4.0f MeV           %5.1f %%" % (
                  killed, 1e3 * ps[below].mean() if below.any() else float("nan"),
                  100 * (ps[below] > 0.3).mean() if below.any() else float("nan")))
    c_free = res[("free n", 0.10)]
    if c_free[1] > 0:
        failures.append("free neutron produced %d events below the threshold" % c_free[1])
    c10 = res[("deuteron", 0.10)]
    if c10[1] == 0:
        failures.append("deuteron with floor 0.10 produced no sub-threshold events")
    if c10[0][0] == 0:
        failures.append("deuteron with floor 0.10 produced nothing in 6.5-7.5 GeV")
    # sub-threshold spectators must be hard: kinematics say <p_s> ~ 0.2-0.37 GeV/c there
    g, ev, rep = None, None, None
    print()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
