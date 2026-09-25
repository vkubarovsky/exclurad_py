#!/usr/bin/env python3
"""Checks on the Fermi module alone, before any generator wiring.

    python3 validation/fermi_module.py

Covers acceptance criterion 2 (the sampled p_F reproduces |psi(p)|^2) and the
exactness the rest of the chain depends on: the deuteron splits with zero
residual, and the working-frame map is a genuine Lorentz transformation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exclurad_py.generator.fermi import (CONFIG, HBARC, K_MAX, M_D, P_MAX,  # noqa: E402
                                         FermiSampler, split_deuteron,
                                         spectator_limit, working_frame)
from exclurad_py.core.constants import M_N, M_P  # noqa: E402

N = 400_000


def main():
    fail = []
    s = FermiSampler("av18")
    rng = np.random.default_rng(20260925)
    P = np.array([s.draw(rng) for _ in range(N)])
    p = np.linalg.norm(P, axis=1)

    print(f"wave function      av18, {len(s.k)} nodes, "
          f"k_max = {s.k_max:.4f} fm^-1 = {s.k_max * HBARC * 1000:.1f} MeV/c")
    print(f"table norm         {s.total:.6f}   (per nucleon: Int 4 pi k^2 n dk)")
    print(f"<|p_F|>            {p.mean() * 1000:8.2f} MeV/c")
    print(f"rms |p_F|          {np.sqrt((p ** 2).mean()) * 1000:8.2f} MeV/c")
    print(f"max |p_F|          {p.max() * 1000:8.2f} MeV/c  (ceiling "
          f"{P_MAX * 1000:.2f})")

    # --- criterion 2: shape against the table, per table cell -------------
    k = p / HBARC
    counts, _ = np.histogram(k, bins=s.k)
    expect = np.diff(s.cdf) * N
    keep = expect > 20
    chi2 = np.sum((counts[keep] - expect[keep]) ** 2 / expect[keep])
    ndf = int(keep.sum())
    print(f"\nshape vs |psi(p)|^2   chi2/ndf = {chi2 / ndf:.3f}  ({ndf} cells)")
    if chi2 / ndf > 2.0:
        fail.append(f"sampled shape does not follow the table: {chi2/ndf:.2f}")

    # --- isotropy ---------------------------------------------------------
    cos_t = P[:, 2] / p
    if abs(cos_t.mean()) > 5.0 / np.sqrt(3.0 * N):
        fail.append(f"direction not isotropic: <cos t> = {cos_t.mean():.5f}")
    print(f"isotropy              <cos t> = {cos_t.mean():+.5f}, "
          f"<cos^2 t> = {np.mean(cos_t ** 2):.5f}  (1/3)")

    # --- the deuteron splits exactly, both configurations -----------------
    print()
    for struck in ("p", "n"):
        st, sp, _ = zip(*[split_deuteron(x, struck) for x in P[:20_000]])
        st, sp = np.array(st), np.array(sp)
        res = np.abs(st + sp - np.array([M_D, 0.0, 0.0, 0.0])).max()
        m2 = st[:, 0] ** 2 - np.sum(st[:, 1:] ** 2, axis=1)
        _, m_spec, pid = CONFIG[struck]
        msp = np.abs(sp[:, 0] ** 2 - np.sum(sp[:, 1:] ** 2, axis=1)
                     - m_spec ** 2).max()
        print(f"struck {struck} / spectator pid {pid}: "
              f"|struck+spec-(M_d,0)| = {res:.1e}, "
              f"spectator on shell to {msp:.1e}, "
              f"m* in [{np.sqrt(m2).min():.4f}, {np.sqrt(m2).max():.4f}] GeV")
        if res > 1e-12:
            fail.append(f"struck {struck}: deuteron does not balance ({res:.1e})")
        if msp > 1e-12:
            fail.append(f"struck {struck}: spectator off shell ({msp:.1e})")
        if m2.min() <= 0:
            fail.append(f"struck {struck}: m*^2 went negative ({m2.min():.3e})")

    # --- the working-frame map is a Lorentz transformation ----------------
    print()
    worst_beam = worst_nucl = 0.0
    for x in P[:5_000]:
        st, _, ms = split_deuteron(x, "p")
        E, to_lab = working_frame(10.6, st)
        beam_back = to_lab @ np.array([E, 0.0, 0.0, E])
        worst_beam = max(worst_beam,
                         np.abs(beam_back - np.array([10.6, 0, 0, 10.6])).max())
        worst_nucl = max(worst_nucl,
                         np.abs(to_lab @ np.array([ms, 0.0, 0.0, 0.0]) - st).max())
    print(f"working frame -> lab: beam residual {worst_beam:.1e}, "
          f"nucleon residual {worst_nucl:.1e}")
    if worst_beam > 1e-9 or worst_nucl > 1e-9:
        fail.append("working-frame map is not exact")

    Es = [working_frame(10.6, split_deuteron(x, "p")[0])[0] for x in P[:20_000]]
    print(f"beam energy in the struck-nucleon frame: "
          f"{np.min(Es):.3f} .. {np.max(Es):.3f} GeV  (lab 10.600)")

    print()
    if fail:
        for f in fail:
            print("FAIL " + f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
