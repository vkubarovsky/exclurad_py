"""Is the sampler putting steps into the SHAPE of the sample?

The rad/born ratio is smooth in the physics (it is eta), so any discontinuity
in it belongs to the sampler.  This measures the jump at each majorant cell
edge, and -- as a control -- at points that are not cell edges.

    python3 production/shape_steps.py born.npz rad.npz [label]

(arrays from production/lund_kinematics.py)

Found the 2026-09-25 stage-2 composition bug (commit 225d4d9): 19.2 sigma at
the W = 2.20 edge, against 2.2 sigma on the control points.  After the fix,
1.4 sigma against 1.9 sigma.  Keep it as a regression test for any change to
the proposal, the majorant or the stage-2 bound.

The estimator fits a straight line on each side of the edge and compares the
two extrapolations AT the edge.  A plain difference of means does NOT work
here: the ratio rises by a factor 4 across the W range, so a perfectly
continuous but sloping curve registers a 26 % "step".
"""
import sys
import numpy as np

HALF = 0.11          # fit window on each side, GeV


def load(p):
    d = np.load(p)
    return np.sqrt(np.maximum(d["W2"], 0))


def jumps(Wb, Wr, edges, nb=220, rng=(1.80, 4.00), label=""):
    hb, ed = np.histogram(Wb, bins=nb, range=rng)
    hr, _ = np.histogram(Wr, bins=nb, range=rng)
    x = 0.5 * (ed[1:] + ed[:-1])
    fb, fr = hb / hb.sum(), hr / hr.sum()
    ok = (hb > 20) & (hr > 20)
    ratio = np.where(ok, fr / np.where(fb > 0, fb, 1), np.nan)
    # per-bin error of the ratio
    err = np.where(ok, ratio * np.sqrt(1 / np.maximum(hb, 1) + 1 / np.maximum(hr, 1)),
                   np.nan)
    out = []
    for w in edges:
        res = {}
        for side, m in (("L", ok & (x > w - HALF) & (x < w)),
                        ("R", ok & (x >= w) & (x < w + HALF))):
            if m.sum() < 4:
                res = None
                break
            dx = x[m] - w
            wt = 1.0 / err[m] ** 2
            # weighted straight-line fit, value and its error at dx = 0
            S = wt.sum(); Sx = (wt * dx).sum(); Sxx = (wt * dx * dx).sum()
            Sy = (wt * ratio[m]).sum(); Sxy = (wt * dx * ratio[m]).sum()
            det = S * Sxx - Sx * Sx
            a = (Sxx * Sy - Sx * Sxy) / det            # intercept = value at w
            va = Sxx / det                              # its variance
            res[side] = (a, np.sqrt(va))
        if res is None:
            continue
        (l, sl), (r, sr) = res["L"], res["R"]
        s = np.hypot(sl, sr)
        out.append((w, l, r, 100 * (r - l) / l, abs(r - l) / max(s, 1e-12)))
    print(f"\n  {label}")
    print(f"  {'W':>6} {'слева':>8} {'справа':>8} {'разрыв':>9} {'значимость':>11}")
    for w, l, r, pc, sg in out:
        flag = "  <<<" if sg > 3 else ""
        print(f"  {w:6.2f} {l:8.3f} {r:8.3f} {pc:+8.1f}% {sg:9.1f} sigma{flag}")
    worst = max((o[4] for o in out), default=0.0)
    print(f"  худшая значимость: {worst:.1f} sigma")
    return out, worst


if __name__ == "__main__":
    Wb, Wr = load(sys.argv[1]), load(sys.argv[2])
    print(f"  born {len(Wb)}   rad {len(Wr)}")
    maj = np.sqrt(np.linspace(3.24, 16.0, 9))[1:-1]
    tab = np.sqrt(np.linspace(3.24, 16.0, 6))[1:-1]
    jumps(Wb, Wr, maj, label=f"границы МАЖОРАНТНЫХ ячеек W2  ({sys.argv[3] if len(sys.argv)>3 else ''})")
    # control: the same estimator on points that are NOT cell edges must be quiet
    ctrl = np.array([2.05, 2.35, 2.70, 2.95, 3.22, 3.47, 3.70])
    jumps(Wb, Wr, ctrl, label="КОНТРОЛЬ: те же окна на не-границах (должно быть тихо)")
