#!/usr/bin/env python3
"""Acceptance criterion 6, adapted to a deuteron sample.

    python3 validation/fermi_shape_steps.py born.npz rad.npz [label]

`production/shape_steps.py` checks the rad/born ratio for discontinuities at
the majorant cell edges: the ratio is smooth in the physics (it is eta), so a
step in it belongs to the sampler.  Two changes are needed here.

1. ITS EDGES ARE HARDCODED for a different box (`linspace(3.24, 16.0, 9)`).
   The cards used for this branch have `W2 = 4.5 .. 12`, so the edges are
   recomputed from the card's box.

2. W IS THE WRONG VARIABLE FOR A DEUTERON SAMPLE.  The estimator fits within
   HALF = 0.11 GeV of each edge, and Fermi motion smears the OBSERVED W by
   131 MeV -- more than the window.  Any step at a majorant W2 edge is washed
   across the fit, so the test loses most of its power exactly where it is
   being asked to work.

   Q2 does not have that problem.  It is built from the beam and the scattered
   electron alone, both laboratory quantities, and it is a Lorentz invariant --
   so the reconstructed Q2 IS the generated Q2, to the precision of the beam
   energy, whatever the target was doing.  A step at a majorant Q2 cell edge
   is therefore visible at full strength.  That is the test that carries the
   information for a deuteron sample; the W one is reported alongside it for
   continuity with production, not as the discriminating check.

The majorant's Q2 axis is GEOMETRIC (see `_maj_key`, tag "geom-q2t-cutmask"),
so the edges are geomspace, not linspace.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "production"))

import shape_steps as ss  # noqa: E402

# the cards used on this branch
BOX_W2 = (4.5, 12.0)
BOX_Q2 = (1.0, 5.0)
NCELL = 8                      # majorant_shape[0] and [1]


def main():
    db, dr = np.load(sys.argv[1]), np.load(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    print(f"  born {len(db['W2'])}   rad {len(dr['W2'])}   {label}")

    # ---- Q2: unsmeared, the discriminating test ----------------------------
    q2b, q2r = db["Q2"], dr["Q2"]
    edges_q2 = np.geomspace(*BOX_Q2, NCELL + 1)[1:-1]
    ss.HALF = 0.11
    ss.jumps(q2b, q2r, edges_q2, nb=180, rng=BOX_Q2,
             label=f"Q2: MAJORANT cell edges (geometric) {label}")
    # controls: midpoints between the edges, which are not edges
    ctrl_q2 = np.sqrt(edges_q2[:-1] * edges_q2[1:])
    ss.jumps(q2b, q2r, ctrl_q2, nb=180, rng=BOX_Q2,
             label="Q2: CONTROL, same windows off the edges (should be quiet)")

    # ---- W: reported for continuity, with its power spelled out -------------
    wb, wr = np.sqrt(np.maximum(db["W2"], 0)), np.sqrt(np.maximum(dr["W2"], 0))
    edges_w = np.sqrt(np.linspace(*BOX_W2, NCELL + 1))[1:-1]
    ss.jumps(wb, wr, edges_w, nb=220, rng=(np.sqrt(BOX_W2[0]) - 0.4,
                                           np.sqrt(BOX_W2[1]) + 0.4),
             label=f"W: MAJORANT cell edges -- BLUNTED by the 131 MeV "
                   f"Fermi smearing {label}")
    ctrl_w = np.sqrt(edges_w[:-1] * edges_w[1:])
    ss.jumps(wb, wr, ctrl_w, nb=220, rng=(np.sqrt(BOX_W2[0]) - 0.4,
                                          np.sqrt(BOX_W2[1]) + 0.4),
             label="W: CONTROL, off the edges")


if __name__ == "__main__":
    main()
