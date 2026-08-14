"""Hard-photon fraction straight from LUND files -- the RAD acceptance test.

Measures v = MX^2(e'p) - mU^2 event by event and reports the fraction above a
set of thresholds, for our sample and for a reference sample, side by side.

This is deliberately independent of the generator's own bookkeeping: it reads
the written file and recomputes v from the four-vectors, so it cannot inherit a
mistake from the code under test.  If the generator's reported hard fraction
and this one ever disagree, that disagreement itself localises the bug.

    python3 production/hard_fraction.py <channel> [mode]
"""
import glob
import os
import sys

import numpy as np

M_P = 0.93827208816
E0 = 10.6
MESON_M2 = {"pi0": 0.1349768 ** 2, "eta": 0.547862 ** 2,
            "rho": 0.77526 ** 2, "phi": 1.019461 ** 2, "jpsi": 3.096900 ** 2}
THRESHOLDS = (0.005, 0.01, 0.05, 0.10, 0.30)

REF = {
    "pi0": "/Volumes/wd_14tb/mc/exclugen/pi0/20260810_10.6_{m}_osg_v2/osg/*.lund",
    "eta": "/Volumes/wd_14tb/mc/exclugen/eta/20260810_10.6_{m}_osg_v2/osg/*.lund",
}
REFMODE = {"rad": "rad", "born": "born"}


def v_spectrum(paths, m2, drop_straggled=False, limit=None):
    """v = MX^2 - mU^2 per event, from the scattered electron and the proton."""
    out = []
    for path in paths[:limit]:
        with open(path) as fh:
            while True:
                line = fh.readline()
                if not line:
                    break
                if line.startswith("#"):
                    continue
                f = line.split()
                if len(f) < 10:
                    continue
                try:
                    npart = int(f[0])
                except ValueError:
                    continue
                el = pr = None
                beam = np.array([0.0, 0.0, E0, E0])
                for i in range(npart):
                    q = fh.readline().split()
                    if len(q) < 11:
                        continue
                    pid, st, par = int(q[3]), int(q[2]), int(q[4])
                    p4 = np.array([float(q[6]), float(q[7]),
                                   float(q[8]), float(q[9])])
                    if i == 0 and st == 0 and pid == 11:
                        beam = p4                    # straggled beam, if any
                        continue
                    if par != 0:                     # decay product
                        continue
                    if pid == 11 and st == 1 and el is None:
                        el = p4
                    elif pid == 2212 and st == 1:
                        pr = p4
                if el is None or pr is None:
                    continue
                if drop_straggled and abs(beam[3] - E0) > 1e-3:
                    continue
                X = beam + np.array([0.0, 0.0, 0.0, M_P]) - el - pr
                out.append(X[3] ** 2 - X[:3] @ X[:3] - m2)
    return np.array(out)


def main(ch, mode="rad"):
    m2 = MESON_M2[ch]
    # MINE_GLOB lets this run against a scratch sample without touching the
    # shared tree, which is how a fix is checked before anything is regenerated
    mine = sorted(glob.glob(os.environ.get(
        "MINE_GLOB",
        f"/Volumes/wd_14tb/mc/exclurad_py/{ch}/*_{mode}_v1/lund/*.lund")))
    if not mine:
        sys.exit(f"no exclurad_py {ch} {mode} samples")
    a = v_spectrum(mine, m2)

    b = None
    if ch in REF:
        ref = sorted(glob.glob(REF[ch].format(m=REFMODE.get(mode, mode))))
        if ref:
            # exclugen runs itarg=1; drop pre-vertex-radiated events so the
            # two samples are compared on the same beam energy
            b = v_spectrum(ref, m2, drop_straggled=True)

    print(f"\n  {ch} {mode}:  exclurad_py N={len(a)}"
          + (f"   reference N={len(b)}" if b is not None else "   (no reference)"))
    print(f"  {'threshold':>16} {'exclurad_py':>13} "
          + (f"{'exclugen':>11} {'ratio':>8}" if b is not None else ""))
    for thr in THRESHOLDS:
        fa = 100.0 * (a > thr).mean() if len(a) else float("nan")
        row = f"  v > {thr:<11.3f} {fa:12.3f}%"
        if b is not None and len(b):
            fb = 100.0 * (b > thr).mean()
            ratio = (fa / fb) if fb > 0 else float("nan")
            row += f" {fb:10.3f}% {ratio:8.3f}"
        print(row)
    print(f"  {'mean v':>16} {a.mean():12.5f} "
          + (f"{b.mean():11.5f}" if b is not None and len(b) else ""))
    print(f"  {'max v':>16} {a.max():12.5f} "
          + (f"{b.max():11.5f}" if b is not None and len(b) else ""))
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pi0",
         sys.argv[2] if len(sys.argv) > 2 else "rad")
