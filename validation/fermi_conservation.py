#!/usr/bin/env python3
"""Acceptance criterion 3: per-event four-momentum conservation.

    beam + d = e' + meson + N' + N_spectator (+ gamma)

evaluated on the LABORATORY vectors the LUND record actually carries, at
machine precision.  This is the test that catches a boost error immediately:
any mistake in working_frame(), in _boost_to_lab(), or in the order of the
rotation and the boost shows up here as a residual of order the momenta
themselves, not of order 1e-15.

Also checks the free-nucleon case (beam + N = ...) so the same script covers
both configurations, and the invariant masses of the written particles.

    python3 validation/fermi_conservation.py [nev] [model]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exclurad_py import models                                    # noqa: E402
from exclurad_py.core.constants import M_N, M_P                   # noqa: E402
from exclurad_py.generator.events import Box, Generator           # noqa: E402
from exclurad_py.generator.fermi import CONFIG, M_D               # noqa: E402

TOL = 1e-9


def run(model, fermi, nev, born_only):
    kw = dict(t_nucl=-0.30) if "pi0" in model or "eta" in model else {}
    sf = models.get(model, **kw)
    sf.Ebeam = 10.6
    g = Generator(sf, Ebeam=10.6,
                  box=Box(Q2=(1.0, 5.0), W2=(4.5, 12.0), t=(-2.0, -0.10)),
                  mx2_cut=sf.mU ** 2 + 0.20, helicity=0, seed=4242,
                  born_only=born_only, fermi=fermi,
                  majorant_shape=(4, 4, 4, 6))
    g.setup()
    return sf, g, g.generate(nev)


def main():
    nev = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    models_to_try = sys.argv[2:] or ["pi0.amp2609", "pi0n.amp2021",
                                     "eta.amp2609", "etan.amp2609"]
    bad = 0
    for model in models_to_try:
        for fermi in ("none", "av18"):
            for born_only in (True, False):
                try:
                    sf, g, evs = run(model, fermi, nev, born_only)
                except Exception as exc:
                    print(f"{model:15s} {fermi:5s} "
                          f"{'born' if born_only else 'rad ':4s}  "
                          f"ERROR {type(exc).__name__}: {exc}")
                    bad += 1
                    continue
                if not evs:
                    print(f"{model:15s} {fermi:5s} no events")
                    bad += 1
                    continue
                worst = 0.0
                worst_m = 0.0
                n_rad = 0
                for ev in evs:
                    if fermi == "none":
                        m_t = float(getattr(sf, "mT", M_P))
                        initial = ev["beam"] + np.array([m_t, 0.0, 0.0, 0.0])
                    else:
                        initial = ev["beam"] + np.array([M_D, 0.0, 0.0, 0.0])
                    final = ev["scattered"] + ev["meson"] + ev["recoil"]
                    if ev.get("photon") is not None:
                        final = final + ev["photon"]
                        n_rad += 1
                    if ev.get("spectator") is not None:
                        final = final + ev["spectator"]
                    worst = max(worst, float(np.abs(initial - final).max()))
                    # the spectator must be ON shell, it is a real particle
                    sp = ev.get("spectator")
                    if sp is not None:
                        _, m_spec, _ = CONFIG[g._struck]
                        m2 = sp[0] ** 2 - float(np.dot(sp[1:], sp[1:]))
                        worst_m = max(worst_m, abs(m2 - m_spec ** 2))
                flag = "ok " if worst < TOL and worst_m < TOL else "FAIL"
                if flag == "FAIL":
                    bad += 1
                print(f"{model:15s} {fermi:5s} "
                      f"{'born' if born_only else 'rad ':4s} "
                      f"n={len(evs):4d} rad={n_rad:4d}  "
                      f"max|P_in - P_out| = {worst:.2e}  "
                      f"spectator off shell by {worst_m:.2e}   {flag}")
    print()
    print("all conserved" if bad == 0 else f"{bad} configuration(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
