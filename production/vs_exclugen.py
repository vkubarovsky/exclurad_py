"""exclurad_py against exclugen, both freshly generated, one matched box.

    OUR_RAD=... OUR_NORAD=... EXG_RAD=... EXG_BORN=... \
        python3 production/vs_exclugen.py pi0

This is the only EVENT-LEVEL external check the project has for pi0 and eta:
everything else compares the generator against our own quadrature, which
shares the RC core with it and therefore cannot see an error in the core.

The comparison from 2026-08-13 is not usable as a result any more.  It
predates the Trento phi convention (17.08), the removal of the factor 2 on
sigma_LT, the geometric t axis (19.08), sigma_LT' and every box change --
and its exclugen sample was generated with the PRE-VERTEX EXTERNAL RADIATOR
ON, which is stated in compare_exclugen.py as an uncorrected difference.

Matching, which is where such comparisons usually go wrong:

  * exclugen's box is (W, Q2, cos theta*) and it generates the full t range
    at each point; ours is a (Q2, W2, t) rectangle.  So the box is matched
    by CUTTING both samples to the same rectangle, not by trying to make
    exclugen generate one.
  * exclugen generates to the kinematic v limit and expects the analysis to
    apply the exclusivity cut (its own header says so); our card applies
    v_cut at generation.  So the same v cut is applied to exclugen here.
  * the missing mass is reconstructed with the NOMINAL beam energy in both,
    which is what an experiment does.
  * itarg must be 0 in the exclugen run.  With the radiator on, exclugen is
    modelling something we do not have, and the difference is physics, not
    a discrepancy.

Absolute cross sections are comparable, not only shapes: exclugen's events
are unweighted and distributed as sigma_Born over its own range, so the
fraction passing our rectangle is an unbiased estimate of the ratio of the
two volumes' cross sections.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_channel import (expand_glob, parse_card, read_events,  # noqa
                              MESON_M, MESON_PID)

M_P = 0.93827208816
E0 = 10.6


def read_exclugen(paths, ch):
    """(Q2, W2, t, phi_Trento_deg, v) per event from an exclugen LUND.

    Header is 15 fields (ours is 10), particle lines 14 in both.  The
    invariants sit in header columns 11-15 -- xB, y, W2, Q2, nu -- but t,
    phi and v are rebuilt from the four-vectors so that both samples go
    through the SAME arithmetic and a convention error cannot hide in the
    difference between two ways of computing the same number.
    """
    mU2 = MESON_M[ch] ** 2
    out = []
    for path in paths:
        el = pr = ms = None
        for line in open(path):
            q = line.split()
            if len(q) != 14:
                el = pr = ms = None
                continue
            pid, st = int(q[3]), int(q[2])
            p4 = np.array([float(q[6]), float(q[7]), float(q[8]), float(q[9])])
            if pid == 11 and st == 1:
                el = p4
            elif pid == 2212 and st == 1:
                pr = p4
            elif pid == MESON_PID[ch]:
                ms = p4
            if el is not None and pr is not None and ms is not None:
                out.append(np.concatenate([el, pr, ms]))
                el = pr = ms = None
    if not out:
        return np.zeros((0, 5))
    a = np.array(out)
    el, pr, ms = a[:, 0:4], a[:, 4:8], a[:, 8:12]
    nu = E0 - el[:, 3]
    pmag = np.linalg.norm(el[:, :3], axis=1)
    Q2 = 2.0 * E0 * el[:, 3] * (1.0 - el[:, 2] / np.maximum(pmag, 1e-12))
    W2 = M_P ** 2 + 2.0 * M_P * nu - Q2
    t = 2.0 * M_P * (M_P - pr[:, 3])
    X = np.array([0.0, 0.0, E0, E0 + M_P])[None, :] - el - pr
    v = X[:, 3] ** 2 - np.einsum("ij,ij->i", X[:, :3], X[:, :3]) - mU2
    beam = np.array([0.0, 0.0, E0])
    qv = beam[None, :] - el[:, :3]
    qh = qv / np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
    aa = np.cross(qh, beam[None, :])
    bb = np.cross(qh, ms[:, :3])
    den = np.maximum(np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1), 1e-12)
    cp = np.einsum("ij,ij->i", aa, bb) / den
    sp = np.einsum("ij,ij->i", np.cross(beam[None, :], ms[:, :3]), qh) / den
    phi = np.degrees(np.arctan2(sp, cp)) % 360.0
    return np.stack([Q2, W2, t, phi, v], axis=1)


def box_mask(ev, card, vcut):
    """The card's rectangle plus its inelasticity cut, applied identically."""
    Q2, W2, t, _, v = (ev[:, i] for i in range(5))
    m = ((Q2 >= card["Q2"][0]) & (Q2 <= card["Q2"][1])
         & (W2 >= card["W2"][0]) & (W2 <= card["W2"][1])
         & (t >= card["t"][0]) & (t <= card["t"][1]))
    if vcut > 0:
        m &= (v <= vcut)
    return m


AX = (("Q2", 0, r"Q^2"), ("W2", 1, r"W^2"), ("mt", 2, r"-t"),
      ("phi", 3, r"phi"))


def shapes(a, b, card, label_a, label_b):
    """Normalised histograms of both samples on the same edges, with pulls."""
    for name, col, lab in AX:
        if name == "mt":
            lo, hi = -card["t"][1], -card["t"][0]
            xa, xb = -a[:, col], -b[:, col]
        elif name == "phi":
            lo, hi = 0.0, 360.0
            xa, xb = a[:, col], b[:, col]
        else:
            lo, hi = card["Q2"] if name == "Q2" else card["W2"]
            xa, xb = a[:, col], b[:, col]
        edges = np.linspace(lo, hi, 13)
        ha, _ = np.histogram(xa, bins=edges)
        hb, _ = np.histogram(xb, bins=edges)
        if ha.sum() == 0 or hb.sum() == 0:
            continue
        fa, fb = ha / ha.sum(), hb / hb.sum()
        ea = np.sqrt(np.maximum(ha, 1)) / ha.sum()
        eb = np.sqrt(np.maximum(hb, 1)) / hb.sum()
        ok = (ha > 20) & (hb > 20)
        pull = (fa - fb) / np.sqrt(ea ** 2 + eb ** 2)
        chi2 = float(np.sum(pull[ok] ** 2) / max(ok.sum() - 1, 1))
        print(f"  {lab:>5}: max |pull| {np.abs(pull[ok]).max():5.1f}   "
              f"chi2/N {chi2:5.2f}   бинов {ok.sum():2d}   "
              f"({label_a} {ha.sum()}, {label_b} {hb.sum()})", flush=True)


def main(ch):
    card = parse_card(ch)
    vcut = card["vcut"]
    print(f"[{ch}] коробка Q2 {card['Q2']}, W2 {card['W2']}, "
          f"t {card['t']}, vcut {vcut}", flush=True)

    ours = {}
    for tag, env in (("rad", "OUR_RAD"), ("born", "OUR_NORAD")):
        ev, _ = read_events(expand_glob(os.environ[env]), ch)
        # read_events returns (Q2, W2, t, phi, v); same layout as above
        ours[tag] = ev[:, :5]
    exg = {}
    for tag, env in (("rad", "EXG_RAD"), ("born", "EXG_BORN")):
        p = os.environ.get(env)
        if p:
            exg[tag] = read_exclugen(expand_glob(p), ch)

    for tag in ("born", "rad"):
        if tag not in exg:
            continue
        a = ours[tag][box_mask(ours[tag], card, vcut)]
        b = exg[tag][box_mask(exg[tag], card, vcut)]
        print(f"\n[{ch}] {tag}: наших {len(a)}, exclugen {len(b)} "
              f"(из {len(exg[tag])}, доля в коробке "
              f"{100*len(b)/max(len(exg[tag]),1):.2f}%)", flush=True)
        shapes(a, b, card, "наш", "exg")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "pi0"))
