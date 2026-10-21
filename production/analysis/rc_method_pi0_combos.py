#!/usr/bin/env python3
"""A against B with the note's cuts as printed, in every combination of Q2, xB, -t.

    python3 production/analysis/rc_method_pi0_combos.py [outdir]

Cuts, the note's Sec. 3.5 list, literally, on both MC arms:
    DIS        Q2 > 1, W > 2, p_e' > 2
    pi0        M(gg) within 3 sigma (sigma from the Born MC), gamma-gamma opening > 1 deg
    exclusive  -0.2 < ME < 0.2,  |MM2(e'p gg)| < 0.004,  MP_T < 0.75,  cone < 4 deg
These windows are data windows after corrections; on this uncorrected MC they
keep ~40 % of Born.  They act on both arms alike, which is all B needs.

A has ONE exclusivity cut, v_cut (MM2(e'p) < v_cut + m_pi0^2): rc_factor knows
nothing of ME, MM2(e'pgg), MP_T or the cone.  A per bin is

    A_bin = N / sum_i 1/eta(x_i)      over the accepted RAD events i in the bin,

eta = sigma_obs/sigma_Born at each event's own reconstructed (Q2, W2, t, phi).
Rad events are distributed as sigma_Born eta Acc, so this is the Born- and
acceptance-weighted mean of eta, for any binning, without the Born MC.
phi is the EXCLURAD one, phi_p' + 180.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402
import rc_method_pi0 as R  # noqa: E402

QE = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.6, 6.0, 7.0, 8.5]
XE = [0.10, 0.15, 0.20, 0.25, 0.30, 0.38, 0.48, 0.58, 0.65]
TE = [0.09, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00, 1.50, 2.00]
EDGES = {"Q2": QE, "xB": XE, "mt": TE}
VCUTS = [0.02, 0.04, 0.06, 0.08, 0.11, 0.14, 0.20, 0.30, 0.45]          # v_cut; mx2_cut = v_cut + m_pi0^2
V0 = 0.20
NMIN = {1: 100, 2: 200, 3: 200}
COMBOS = [("Q2",), ("xB",), ("mt",), ("Q2", "xB"), ("Q2", "mt"), ("xB", "mt"), ("Q2", "xB", "mt")]


def note_cuts(k, cal):
    def mgg_ok():
        c = np.where(k["pdet"] == "FD", cal["FD"]["mgg"][0], cal["CD"]["mgg"][0])
        s = np.where(k["pdet"] == "FD", cal["FD"]["mgg"][1], cal["CD"]["mgg"][1])
        return np.abs(k["mgg"] - c) < 3 * s
    return ((k["Q2"] > 1) & (k["W"] > 2) & (k["pe"] > 2) & mgg_ok() & (k["open"] > 1) &
            (np.abs(k["me"]) < 0.2) & (np.abs(k["mm2tot"]) < 0.004) & (k["mpt"] < 0.75) & (k["cone"] < 4))


def collect():
    rng = np.random.default_rng(20261021)
    sig = {a: C.sigma_nb(f"{R.DATA}/prov_{a}", a)["mean"] for a in ("rad", "born")}
    ev = {"rad": [], "born": []}
    for pol in ("inb", "outb"):
        S = {a: R.load(a, pol, rng) for a in ("rad", "born")}
        cal = R.calibrate(S["born"]["rec"])
        for a in S:
            k = S[a]["rec"]; m = note_cuts(k, cal)
            ev[a].append(np.column_stack([k["Q2"][m], k["xB"][m], k["mt"][m], k["phi"][m],
                                          np.full(m.sum(), sig[a] / S[a]["n_gen"])]))
    return {a: np.concatenate(v) for a, v in ev.items()}


def eta_chunk(rows):
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    m = models.get("pi0.amp2609", t_nucl=-0.3)
    out = np.full((len(rows), len(VCUTS)), np.nan)
    for i, (Q2, xB, mt, ph) in enumerate(rows):
        m.t_nucl = -mt
        W2 = R.MP ** 2 + Q2 * (1 / xB - 1)
        try:
            for j, v in enumerate(VCUTS):
                out[i, j] = rc_factor(R.EB, Q2, W2, -mt, np.radians(ph), m, h=+1,
                                      mx2_cut=v + R.MPI ** 2)["eta"]
        except Exception:
            pass
    return out


def per_event_eta(rad, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        if z["eta"].shape == (len(rad), len(VCUTS)) and np.allclose(z["kin"], rad[:, :4]):
            return z["eta"]
    chunks = np.array_split(rad[:, :4], 400)
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as p:
        eta = np.concatenate(p.map(eta_chunk, chunks))
    np.savez(cache, eta=eta, kin=rad[:, :4])
    return eta


def cell_index(e, dims):
    col = {"Q2": 0, "xB": 1, "mt": 2}
    idx, ok = [], np.ones(len(e), bool)
    for d in dims:
        i = np.digitize(e[:, col[d]], EDGES[d]) - 1
        ok &= (i >= 0) & (i < len(EDGES[d]) - 1)
        idx.append(i)
    # integrated variables still live inside the note's full range
    for d in set(col) - set(dims):
        x = e[:, col[d]]
        ok &= (x >= EDGES[d][0]) & (x < EDGES[d][-1])
    return np.column_stack(idx), ok


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R.DATA, "out")
    os.makedirs(outdir, exist_ok=True)
    ev = collect()
    eta = per_event_eta(ev["rad"], os.path.join(outdir, "eta_per_event.npz"))
    good = np.all(np.isfinite(eta) & (eta > 0), axis=1)
    print(f"rad {len(ev['rad'])}, born {len(ev['born'])}; eta finite for {good.mean():.4f} of rad events")
    res = {"vcuts": VCUTS, "v0": V0, "edges": EDGES, "nmin": {str(k): v for k, v in NMIN.items()},
           "n_rad": len(ev["rad"]), "n_born": len(ev["born"]), "eta_ok": float(good.mean()), "combos": {}}
    for dims in COMBOS:
        ir, okr = cell_index(ev["rad"], dims)
        ib, okb = cell_index(ev["born"], dims)
        cells = []
        for key in sorted(set(map(tuple, ir[okr]))):
            sr = okr & np.all(ir == key, axis=1); sb = okb & np.all(ib == key, axis=1)
            nr, nb = int(sr.sum()), int(sb.sum())
            if nr < NMIN[len(dims)] or nb < NMIN[len(dims)]:
                continue
            B = ev["rad"][sr, 4].sum() / ev["born"][sb, 4].sum()
            ge = sr & good
            A = (ge.sum() / (1.0 / eta[ge]).sum(axis=0)).tolist()
            xs, ys = np.array(VCUTS), np.array(A)
            veq = float(np.interp(B, ys, xs)) if ys.min() <= B <= ys.max() else float("nan")
            cells.append(dict(key=[int(k) for k in key],
                              ranges={d: (EDGES[d][k], EDGES[d][k + 1]) for d, k in zip(dims, key)},
                              n_rad=nr, n_born=nb, B=float(B), B_err=float(B * np.sqrt(1 / nr + 1 / nb)),
                              A=A, v_eq=veq,
                              mean={d: float(ev["rad"][sr, c].mean()) for d, c in (("Q2", 0), ("xB", 1), ("mt", 2))}))
        res["combos"]["-".join(dims)] = cells
        i0 = VCUTS.index(V0)
        dev = [100 * (c["B"] / c["A"][i0] - 1) for c in cells]
        print(f"{'-'.join(dims):10} {len(cells):3d} cells   B/A(v=0.2)-1: "
              f"{np.min(dev) if dev else 0:+.1f} .. {np.max(dev) if dev else 0:+.1f} %")
    with open(os.path.join(outdir, "rc_method_pi0_combos.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
