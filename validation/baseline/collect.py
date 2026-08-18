"""Freeze a numerical contract for the current code, and check against it.

    python3 validation/baseline/collect.py write   # record baseline.json
    python3 validation/baseline/collect.py check   # compare, exit 1 on drift

Why this exists: VPK, 2026-08-18 -- "у нас не с чем сравнивать, устойчивой
версии нет".  Speed work is about to start, and every optimisation is a
chance to move a number by accident.  A git tag alone does not help: it
records the code, not what the code produced.

What is in here is deterministic ONLY -- quadrature, no Monte Carlo.  MC
validation (acceptance gates, delta profiles) needs samples and hours;
this runs in a couple of minutes and is meant to be run before and after
every change that is supposed to be physics-neutral.

Tolerance is 1e-10 relative: these are the same arithmetic on the same
inputs, so anything that moves at all moved for a reason and should be
looked at, not averaged away.
"""
import json
import os
import sys

import numpy as np

from exclurad_py import models
from exclurad_py.core.rc import rc_factor

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "baseline.json")
RTOL = 1e-10

# (label, model, kwargs, Ebeam, Q2, W2, t, phi_deg, mx2_cut)
POINTS = [
    ("pi0.2021 @G1", "pi0.vpk2021", dict(t_nucl=-0.12), 5.75, 1.1708,
     2.91 ** 2, -0.12, 75.0, 0.094 + 0.1349768 ** 2),
    ("pi0.2021 wide", "pi0.vpk2021", dict(t_nucl=-0.30), 10.6, 2.0, 9.0,
     -0.30, 40.0, 0.50 + 0.1349768 ** 2),
    ("pi0.2013 publ", "pi0.vpk2013", dict(t_nucl=-0.30), 5.75, 1.8, 6.0,
     -0.30, 155.0, 0.17 + 0.1349768 ** 2),
    ("eta.2021", "eta.vpk2021", dict(t_nucl=-0.30), 10.6, 2.0, 12.0,
     -0.30, 40.0, 0.20 + 0.547862 ** 2),
    ("rho.diffrad_vm", "rho.diffrad_vm", {}, 10.6, 2.0, 9.0, -0.30, 40.0,
     None),
    ("phi.diffrad_vm", "phi.diffrad_vm", {}, 10.6, 2.0, 9.0, -0.30, 40.0,
     None),
    ("jpsi.diffrad_vm", "jpsi.diffrad_vm", {}, 10.6, 2.0, 17.0, -1.60,
     40.0, None),
]


def collect():
    out = {}
    for lab, name, kw, EB, Q2, W2, t, phid, cut in POINTS:
        sf = models.get(name, **kw)
        rec = {}
        for h in (+1, -1):
            r = rc_factor(Ebeam=EB, Q2=Q2, W2=W2, t=t, phi=np.radians(phid),
                          sfmodel=sf, h=h, mx2_cut=cut)
            rec[f"h{h:+d}"] = {k: float(r[k]) for k in
                               ("sigma_born", "sigma_obs", "eta")}
        d = sf(np.array([W2]), np.array([Q2]), np.array([t]))
        rec["sf"] = {k: float(d[k][0]) for k in
                     ("sT", "sL", "sTT", "sLT", "sLTp")}
        rec["topology"] = dict(lt_sign=sf.topology.lt_sign,
                               phi_offset=sf.topology.phi_offset,
                               t_kind=sf.topology.t_kind)
        out[lab] = rec
        print(f"  {lab:18s} eta = {rec['h+1']['eta']:.9f}", flush=True)
    return out


def walk(a, b, path=""):
    """Yield (path, old, new) for every numeric leaf that moved."""
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield (path + "/" + k, a.get(k, "—"), b.get(k, "—"))
            else:
                yield from walk(a[k], b[k], path + "/" + k)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b and (abs(a - b) > RTOL * max(abs(a), abs(b), 1e-300)):
            yield (path, a, b)
    elif a != b:
        yield (path, a, b)


def main(mode):
    print(f"базовая линия: {len(POINTS)} точек, {PATH}")
    cur = collect()
    if mode == "write":
        json.dump(cur, open(PATH, "w"), indent=1, sort_keys=True)
        print(f"\nзаписано {PATH}")
        return 0
    if not os.path.exists(PATH):
        sys.exit("базовой линии нет -- сначала `collect.py write`")
    old = json.load(open(PATH))
    drift = list(walk(old, cur))
    if not drift:
        print("\nVERDICT: PASS -- ни одно число не сдвинулось")
        return 0
    print(f"\nVERDICT: FAIL -- сдвинулось {len(drift)} величин:")
    for p, a, b in drift[:40]:
        if isinstance(a, float) and isinstance(b, float) and a != 0:
            print(f"  {p:44s} {a:.12g} -> {b:.12g}  ({(b/a-1):+.2e})")
        else:
            print(f"  {p:44s} {a} -> {b}")
    if len(drift) > 40:
        print(f"  ... ещё {len(drift)-40}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "check"))
