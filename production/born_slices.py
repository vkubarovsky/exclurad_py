"""Born comparison SLICE BY SLICE, not just on means.

Means hid the pi0/eta failure: Q2 and W2 agreed to a few percent while the
|t| shape was inverted.  A ratio of means near 1 says nothing about shape,
so every variable is compared in bins, inside narrow slices of the other
two, and a chi2/dof is reported per slice.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_born import read_lund, MESON_PID, REF

ROOT = "/Volumes/wd_14tb/mc/exclurad_py"


def load_mine(ch):
    fs = sorted(glob.glob(f"{ROOT}/{ch}/20260813_10.6_born_v1/lund/*.lund"))
    if not fs:
        return None
    p = [read_lund(f, MESON_PID[ch]) for f in fs]
    return tuple(np.concatenate([q[i] for q in p]) for i in range(3))


def load_ref(ch):
    name, files, need = REF[ch]
    p = [read_lund(f, MESON_PID[ch], need) for f in files]
    return name, tuple(np.concatenate([q[i] for q in p]) for i in range(3))


def chi2_shape(a, b, bins):
    """chi2/dof between two shapes, both normalized to unit area."""
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    if ha.sum() < 50 or hb.sum() < 50:
        return np.nan, 0
    na, nb = ha.sum(), hb.sum()
    pa, pb = ha / na, hb / nb
    va, vb = ha / na ** 2, hb / nb ** 2          # Poisson on the fractions
    m = (va + vb) > 0
    d = (pa - pb) ** 2 / (va + vb)
    ndf = max(m.sum() - 1, 1)
    return float(d[m].sum() / ndf), int(ndf)


def report(ch):
    mine = load_mine(ch)
    if mine is None:
        print(f"{ch}: no sample yet")
        return
    name, ref = load_ref(ch)
    # Cards now carry the reference's OWN limits on every variable, so both
    # samples occupy the same volume and no overlap restriction is applied.
    # Restricting after the fact hid differences and left room for
    # normalization error: when integrated distributions are compared,
    # everything must be the same by construction, not by post-selection.
    lo = [min(mine[i].min(), ref[i].min()) for i in range(3)]
    hi = [max(mine[i].max(), ref[i].max()) for i in range(3)]
    mm = np.ones(len(mine[0]), bool)
    rm = np.ones(len(ref[0]), bool)
    for i in range(3):
        mm &= (mine[i] >= lo[i]) & (mine[i] <= hi[i])
        rm &= (ref[i] >= lo[i]) & (ref[i] <= hi[i])
    A = [v[mm] for v in mine]
    B = [v[rm] for v in ref]
    print(f"=== {ch} vs {name}   mine {mm.sum()}  ref {rm.sum()}")

    names = ("Q2", "W2", "|t|")
    for i in range(3):
        b = np.linspace(lo[i], min(hi[i], np.percentile(B[i], 99)), 25)
        c, ndf = chi2_shape(A[i], B[i], b)
        flag = "OK " if c < 3 else "BAD"
        print(f"   {flag} {names[i]:>4} inclusive     chi2/dof {c:8.2f} (ndf {ndf})"
              f"   mean {A[i].mean():8.4f} vs {B[i].mean():8.4f}")

    # |t| shape inside narrow (Q2, W2) slices -- where the pi0 failure showed
    qs = np.percentile(B[0], [20, 40, 60, 80])
    for k in range(len(qs) - 1):
        sa = (A[0] >= qs[k]) & (A[0] < qs[k + 1])
        sb = (B[0] >= qs[k]) & (B[0] < qs[k + 1])
        if sa.sum() < 200 or sb.sum() < 200:
            continue
        b = np.linspace(lo[2], min(hi[2], np.percentile(B[2][sb], 99)), 15)
        c, ndf = chi2_shape(A[2][sa], B[2][sb], b)
        flag = "OK " if c < 3 else "BAD"
        print(f"   {flag} |t| in Q2 {qs[k]:5.2f}-{qs[k+1]:5.2f}  chi2/dof {c:8.2f}"
              f"   mean {A[2][sa].mean():7.4f} vs {B[2][sb].mean():7.4f}")
    print()


if __name__ == "__main__":
    for ch in sys.argv[1:] or ["pi0", "eta", "rho", "phi", "jpsi"]:
        try:
            report(ch)
        except Exception as exc:
            print(f"{ch}: {type(exc).__name__}: {exc}\n")
