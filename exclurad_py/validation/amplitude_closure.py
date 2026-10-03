"""The three copies of the amplitude model must agree exactly.

`exclurad_py` does not import the fit's amplitude code; it keeps its own copies.
That is why the H~ block, added to the fit on 2026-09-30, was invisible here:
dropping a 43-slot fitpar into models/ raised no error and gave neither the old
model nor the new one (sigma_L out by x5.0, sigma_LT by x10.6, because p[25] is
now zero and the refitted <E~> is calibrated to work alongside <H~>).

This test closes that hole.  It compares, over a grid covering the campaign box
and all four channels, for EVERY vector length in use:

    pi0eta_ampmodel_fit/amplitudes.py   the reference the parameters were fitted with
    models/_amplitude_fit2.py           the scalar copy
    models/_amplitude_fit2_vec.py       the vectorised copy, the production path

Two axes have to hold and both are checked: reference-vs-generator catches a
botched port, scalar-vs-vectorised catches a port done in only one of them.

The reference lives outside this repository.  If it is not found the test says
so and SKIPS that axis rather than passing quietly -- a silent pass is exactly
the failure mode this file exists to prevent.

    python3 -m exclurad_py.validation.amplitude_closure
"""
import os
import sys

import numpy as np

from ..models import _amplitude_fit2 as S
from ..models import _amplitude_fit2_vec as V

FIT_REPO = os.environ.get("AMPFIT_REPO",
                          os.path.expanduser("~/pi0eta_ampmodel_fit"))
VECTORS = [("amp2609 (37 slots, no H~)", "runs/C1_with_compass/fitpar.npy"),
           ("Htil_free2 (43 slots, H~)", "runs/Htil_free2/fitpar.npy")]
CHANNELS = ("pi0p", "etap", "pi0n", "etan")
KEYS = ("L", "T", "TT", "LT", "LTp", "LTpLL", "LTUL", "Tp", "TTUL")
TOL = 1e-12


def _grid():
    for q in (1.0, 1.5, 2.5, 4.0, 6.0):
        for x in (0.10, 0.15, 0.20, 0.30, 0.40, 0.55):
            for mt in (0.10, 0.25, 0.5, 1.0, 1.6, 2.0):
                yield -mt, x, q


def _reference():
    if not os.path.isdir(FIT_REPO):
        return None
    sys.path.insert(0, FIT_REPO)
    try:
        import amplitudes as FIT
        return FIT
    except Exception:
        return None


def run(verbose=True):
    FIT = _reference()
    bad = []
    skipped = FIT is None
    for label, rel in VECTORS:
        path = os.path.join(FIT_REPO, rel)
        if not os.path.exists(path):
            raise SystemExit(f"closure test cannot run: {path} is missing")
        p = np.load(path)
        n = 0
        worst_sv = worst_rf = 0.0
        for ch in CHANNELS:
            for t, x, q in _grid():
                try:
                    # below threshold the scalar module raises instead of
                    # returning None -- a known flaw, handled the same way
                    # _amplitude_fit2_vec.self_check handles it
                    a = S.structure(p, ch, t, x, q)
                except ValueError:
                    a = None
                r, ok = V.structure_arr(p, ch, np.array([t]), np.array([x]),
                                        np.array([q]))
                if a is None:
                    if ok[0]:
                        bad.append(f"{label} {ch} t={t} x={x} Q2={q}: "
                                   "scalar says unphysical, vector says fine")
                    continue
                if not ok[0]:
                    bad.append(f"{label} {ch} t={t} x={x} Q2={q}: "
                               "vector says unphysical, scalar says fine")
                    continue
                n += 1
                for k in KEYS:
                    d = abs(float(r[k][0]) - a[k])/max(abs(a[k]), 1e-30)
                    worst_sv = max(worst_sv, d)
                    if d > TOL:
                        bad.append(f"{label} {ch} {k} t={t} x={x} Q2={q}: "
                                   f"scalar {a[k]:.12g} vs vector "
                                   f"{float(r[k][0]):.12g}  rel {d:.2e}")
                if FIT is not None:
                    try:
                        b = FIT.structure(p, ch, t, x, q)
                    except ValueError:
                        b = None
                    if b is None:
                        bad.append(f"{label} {ch} t={t} x={x} Q2={q}: "
                                   "reference says unphysical")
                        continue
                    for k in KEYS:
                        d = abs(a[k] - b[k])/max(abs(b[k]), 1e-30)
                        worst_rf = max(worst_rf, d)
                        if d > TOL:
                            bad.append(f"{label} {ch} {k} t={t} x={x} Q2={q}: "
                                       f"generator {a[k]:.12g} vs reference "
                                       f"{b[k]:.12g}  rel {d:.2e}")
        if verbose:
            ref = (f"vs reference {worst_rf:.2e}" if FIT is not None
                   else "vs reference SKIPPED (fit repo not found)")
            print(f"  {label:28}  {n:5d} points   "
                  f"scalar vs vector {worst_sv:.2e}   {ref}")
    if skipped and verbose:
        print(f"  WARNING: {FIT_REPO} not found, the reference axis did not run."
              "\n           Set AMPFIT_REPO.  This test is half blind without it.")
    if bad:
        for line in bad[:20]:
            print("  FAIL", line)
        raise SystemExit(f"amplitude closure FAILED: {len(bad)} disagreements")
    return not skipped


if __name__ == "__main__":
    full = run()
    print("amplitude closure PASSED" + ("" if full else " (reference axis skipped)"))
