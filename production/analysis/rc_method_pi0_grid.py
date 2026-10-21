#!/usr/bin/env python3
"""A against B on the analysis note's grid, phi-integrated per cell.

    python3 production/analysis/rc_method_pi0_grid.py [outdir]

Two levels, because the sample (1M rad + 1M Born per polarity) cannot fill the
note's grid: after the full chain it holds 56k Born and 40k rad events, and
only 44 of the 316 occupied (Q2, xB, -t) cells have >= 300 in each arm.
    2D  the note's (Q2, xB) cells, -t integrated over 0.09-2.0
    3D  the note's (Q2, xB, -t) cells with >= NMIN events in each arm
B per cell is the note's Eq. 20, phi-integrated: (N_rad sigma_rad/N_gen) / (N_born sigma_born/N_gen).
A per cell is rc_factor at the cell centroid -- the mean RECONSTRUCTED rad-MC
kinematics, as the note centres its bins on data -- summed over 12 phi points
with its own sigma_Born as the weight.  No Born MC enters A.
phi is the EXCLURAD one, phi_p' + 180, as in rc_method_pi0.py.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402
import rc_method_pi0 as R  # noqa: E402

# the note's Tables 11-13
QE = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.6, 6.0, 7.0, 8.5]
XE = [0.10, 0.15, 0.20, 0.25, 0.30, 0.38, 0.48, 0.58, 0.65]
TE = [0.09, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00, 1.50, 2.00]
NMIN = 300
CUTS = [0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.5, None]
FULL = "+MM2(e'pgg), ME: full chain"


def collect():
    rng = np.random.default_rng(20261021)
    sig = {a: C.sigma_nb(f"{R.DATA}/prov_{a}", a)["mean"] for a in ("rad", "born")}
    ev = {"rad": [], "born": []}
    for pol in ("inb", "outb"):
        S = {a: R.load(a, pol, rng) for a in ("rad", "born")}
        cal = R.calibrate(S["born"]["rec"])
        for a in S:
            k = S[a]["rec"]
            m = R.cut_levels(k, cal, binned=False)[FULL]
            w = sig[a] / S[a]["n_gen"]
            ev[a].append(np.column_stack([k["Q2"][m], k["xB"][m], k["mt"][m], np.full(m.sum(), w)]))
    return {a: np.concatenate(v) for a, v in ev.items()}


def cells(ev, dims):
    """Per cell: counts, weighted yields, rad centroid.  dims = 2 or 3."""
    out = []
    edges = [QE, XE, TE]
    idx = {}
    for a, e in ev.items():
        iq = np.digitize(e[:, 0], QE) - 1
        ix = np.digitize(e[:, 1], XE) - 1
        it = np.digitize(e[:, 2], TE) - 1
        ok = (iq >= 0) & (iq < len(QE) - 1) & (ix >= 0) & (ix < len(XE) - 1) & (it >= 0) & (it < len(TE) - 1)
        idx[a] = (iq, ix, it if dims == 3 else np.zeros_like(it), ok)
    nt = len(TE) - 1 if dims == 3 else 1
    for i in range(len(QE) - 1):
        for j in range(len(XE) - 1):
            for l in range(nt):
                sel = {}
                for a in ev:
                    iq, ix, it, ok = idx[a]
                    sel[a] = ok & (iq == i) & (ix == j) & (it == l)
                nr, nb = int(sel["rad"].sum()), int(sel["born"].sum())
                if nr < NMIN or nb < NMIN:
                    continue
                er, eb = ev["rad"][sel["rad"]], ev["born"][sel["born"]]
                B = er[:, 3].sum() / eb[:, 3].sum()
                out.append(dict(iq=i, ix=j, it=l if dims == 3 else -1,
                                Q2=(QE[i], QE[i + 1]), xB=(XE[j], XE[j + 1]),
                                mt=(TE[l], TE[l + 1]) if dims == 3 else (TE[0], TE[-1]),
                                n_rad=nr, n_born=nb, B=float(B), B_err=float(B * np.sqrt(1 / nr + 1 / nb)),
                                centroid=[float(er[:, c].mean()) for c in range(3)]))
    return out


def eval_A(cent):
    """sum sigma_obs / sum sigma_Born over 12 phi points at the centroid, for each cut."""
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    Q2, xB, mt = cent
    m = models.get("pi0.amp2609", t_nucl=-mt)
    m.t_nucl = -mt
    W2 = R.MP ** 2 + Q2 * (1 / xB - 1)
    sb = np.zeros(len(CUTS)); so = np.zeros(len(CUTS)); nok = 0
    for ph in (np.arange(12) + 0.5) * 30:
        try:
            row = [rc_factor(R.EB, Q2, W2, -mt, np.radians(ph), m, h=+1, mx2_cut=c) for c in CUTS]
        except Exception:
            continue
        b = np.array([d["sigma_born"] for d in row]); o = np.array([d["sigma_obs"] for d in row])
        if np.all(np.isfinite(b)) and np.all(np.isfinite(o)):
            sb += b; so += o; nok += 1
    return (so / sb).tolist() if nok else [float("nan")] * len(CUTS), nok


def equivalent_cut(A, B):
    xs = np.array([c for c in CUTS if c is not None]); ys = np.array(A[:len(xs)])
    if not np.all(np.isfinite(ys)) or not (ys.min() <= B <= ys.max()):
        return float("nan")
    return float(np.interp(B, ys, xs))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R.DATA, "out")
    os.makedirs(outdir, exist_ok=True)
    ev = collect()
    res = {"cuts": [str(c) for c in CUTS], "nmin": NMIN, "QE": QE, "XE": XE, "TE": TE}
    for dims in (2, 3):
        cl = cells(ev, dims)
        with Pool(max(1, (os.cpu_count() or 2) - 2)) as p:
            As = p.map(eval_A, [c["centroid"] for c in cl])
        for c, (A, nok) in zip(cl, As):
            c["A"] = A; c["A_nphi"] = nok
            c["c_eq"] = equivalent_cut(A, c["B"])
            for lo, hi in ((c["B"] - c["B_err"], 0), (c["B"] + c["B_err"], 1)):
                c[("c_lo", "c_hi")[hi]] = equivalent_cut(A, lo)
        res[f"{dims}d"] = cl
        print(f"{dims}D: {len(cl)} cells with >= {NMIN} per arm")
    with open(os.path.join(outdir, "rc_method_pi0_grid.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
