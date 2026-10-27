#!/usr/bin/env python3
"""J/psi C_rad two ways, M. Tenorio's selection, every binning, both polarities.

    python3 production/analysis/rc_method_jpsi_combos.py [outdir]

B  MONTE CARLO.  Rad and Born arms of the GEMC 5.14 campaign, the same cuts:
       B_bin = [N_rad(bin) sigma_rad / N_gen,rad] / [N_born(bin) sigma_Born / N_gen,born]
   N_gen and sigma from the chunks actually read.  Binned in RECONSTRUCTED
   kinematics.

A  ANALYTIC.  No Born MC.  For every accepted rad event i, at its own
   reconstructed (Q2, W2, t, phi),
       eta_i(v) = eta_int(v) x eta_ext(v) x F_fsr
   eta_int  rc_factor (exclurad_py/core/rc.py), MX^2 < m_U^2 + v, the model
            whose missing particle is the one the topology misses:
              3 tracks  jpsi.diffrad_vm       (missing proton,  t from e'J/psi)
              4 tracks  jpsi.diffrad_vm_pdet  (missing J/psi,   t from p'; t_nucl = t per event)
   eta_ext  the pre-vertex radiator (rc_method_jpsi.eta_ext), which rc_factor
            does not contain and the rad arm does (itarg = 1)
   F_fsr    the decay-lepton radiation, PhotonEnergy.cdf(K) at the M(ee) > 2.95
            edge, K = (M^2 - 2.95^2)/2M: 0.8784, her 0.88
       A_bin = N / sum_i 1/eta_i
   Rad events are distributed as sigma_Born eta Acc, so A is the Born- and
   acceptance-weighted mean of eta in the bin.  A_flat is the same eta averaged
   with sigma_Born alone on a grid (no acceptance): A - A_flat is the
   acceptance weighting.

The decomposition of B - A, per cell:
   migration     B(rec bins) - B(gen bins), the same selected events binned by
                 their generated kinematics
   acceptance    A(per event) - A_flat
   cut           A(v at her cut) against A(v_eq), v_eq where A = B
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_method_jpsi as R  # noqa: E402

VCUTS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.43, 0.60, 1.0, None]
EDGES = {
    "W": [4.04, 4.11, 4.18, 4.25, 4.32, 4.39, 4.46, 4.53, 4.60],
    "Q2": [0.001, 0.02, 0.035, 0.05, 0.07, 0.09, 0.11, 0.143],
    "mt": [0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.2, 3.0, 5.0],
}
RANGE = {"W": (4.04, 4.60), "Q2": (0.001, 0.143), "mt": (0.3, 5.0)}
QSLICES = [(0.02, 0.05), (0.05, 0.09), (0.09, 0.13)]
MAP = {"W": [4.04, 4.20, 4.30, 4.40, 4.60], "mt": [0.3, 0.7, 1.1, 1.7, 5.0]}
MARIANA = dict(Q2=(0.02, 0.12), W=[4.15, 4.29, 4.40, 4.48],
               gen10M=[(0.9017, 0.0027), (0.9250, 0.0023), (0.9333, 0.0023)])
MODEL = {"3trk": "jpsi.diffrad_vm", "4trk": "jpsi.diffrad_vm_pdet"}
MU2 = {"3trk": R.MP ** 2, "4trk": R.MJ ** 2}
NMIN = 100
NPROC = 10


def fsr_factor():
    from exclurad_py.generator.decay_rad import PhotonEnergy
    K = (R.MJ ** 2 - R.MEE[0] ** 2) / (2 * R.MJ)
    return float(PhotonEnergy(R.MJ).cdf(K))


# ------------------------------------------------------------------ events
COLS = ("W", "Q2", "mt", "phi", "minus")


def columns(k, topo):
    """The five numbers per event that binning and eta need, topology-aware."""
    if topo == "3trk":
        return np.column_stack([k["W"], k["Q2"], k["mt_V"], k["phi_V"], k["xminus"]])
    return np.column_stack([k["W"], k["Q2"], k["mt_p"], k["phi_p"], k["vminus"]])


def collect(cache):
    """Selected events of both arms, rec and gen columns, per topology and polarity."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["ev"].item()
    ev = {}
    for topo in R.TOPOS:
        for pol in ("inb", "outb"):
            S = {a: R.load(a, pol, topo) for a in ("rad", "born")}
            cal = None
            if topo == "3trk":
                cal = R.calibrate_mx3(S["born"]["rec"], R.pre_mx(R.cuts(S["born"]["rec"], pol, topo)))
            for a in S:
                L = R.cuts(S[a]["rec"], pol, topo, cal)
                m = list(L.values())[-1]
                ev[(topo, pol, a)] = dict(
                    rec=columns(S[a]["rec"], topo)[m], gen=columns(S[a]["gen"], topo)[m],
                    n_gen=S[a]["n_gen"], n_chunks=S[a]["n_chunks"],
                    sigma=S[a]["sigma_nb"], sigma_err=S[a]["sigma_err"], w=S[a]["w"],
                    flow={n: int(v.sum()) for n, v in L.items()}, cal=cal)
                print(topo, pol, a, "selected", int(m.sum()), flush=True)
    np.savez(cache, ev=np.array(ev, dtype=object))
    return ev


# --------------------------------------------------------------------- eta
def _eta_rows(args):
    topo, rows = args
    import warnings
    warnings.filterwarnings("ignore")
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    m = models.get(MODEL[topo], t_nucl=-1.0) if topo == "4trk" else models.get(MODEL[topo])
    out = np.full((len(rows), len(VCUTS)), np.nan)
    for i, (W, Q2, mt, ph) in enumerate(rows):
        if topo == "4trk":
            m.t_nucl = -mt
        for j, v in enumerate(VCUTS):
            try:
                out[i, j] = rc_factor(R.EB, Q2, W * W, -mt, np.radians(ph), m, h=+1,
                                      mx2_cut=None if v is None else MU2[topo] + v)["eta"]
            except Exception:
                pass
    return out


def eta_int(topo, kin, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        if z["eta"].shape == (len(kin), len(VCUTS)) and np.allclose(z["kin"], kin):
            return z["eta"]
    chunks = np.array_split(kin, max(1, len(kin) // 200))
    with Pool(NPROC) as p:
        eta = np.concatenate(p.map(_eta_rows, [(topo, c) for c in chunks]))
    np.savez(cache, eta=eta, kin=kin)
    return eta


def eta_ext_matrix(minus):
    return np.column_stack([R.eta_ext(v if v is not None else 1e9, minus) for v in VCUTS])


# ----------------------------------------------------------------- cells
def cell_masks(x, spec):
    """spec: list of (col, lo, hi).  One mask."""
    m = np.ones(len(x), bool)
    for c, lo, hi in spec:
        m &= (x[:, c] >= lo) & (x[:, c] < hi)
    return m


def cell_specs():
    """Every cell of every binning, as (binning name, label, spec)."""
    ci = {n: COLS.index(n) for n in ("W", "Q2", "mt")}
    full = {n: (ci[n], *RANGE[n]) for n in ci}
    out = []
    for var in ("W", "Q2", "mt"):
        e = EDGES[var]
        for i in range(len(e) - 1):
            spec = [(ci[var], e[i], e[i + 1])] + [full[n] for n in ci if n != var]
            out.append((var, {var: (e[i], e[i + 1])}, spec))
    for qs in QSLICES:
        for i in range(len(MAP["W"]) - 1):
            for j in range(len(MAP["mt"]) - 1):
                spec = [(ci["Q2"], *qs), (ci["W"], MAP["W"][i], MAP["W"][i + 1]),
                        (ci["mt"], MAP["mt"][j], MAP["mt"][j + 1])]
                out.append((f"map_Q2_{qs[0]}_{qs[1]}", {"Q2": qs, "W": (MAP["W"][i], MAP["W"][i + 1]),
                                                        "mt": (MAP["mt"][j], MAP["mt"][j + 1])}, spec))
    w = MARIANA["W"]
    for i in range(len(w) - 1):
        spec = [(ci["Q2"], *MARIANA["Q2"]), (ci["W"], w[i], w[i + 1]), full["mt"]]
        out.append(("mariana", {"Q2": MARIANA["Q2"], "W": (w[i], w[i + 1])}, spec))
    return out


# ---------------------------------------------------------- flat A (grid)
def _flat_cell(args):
    topo, rng = args
    import warnings
    warnings.filterwarnings("ignore")
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    m = models.get(MODEL[topo], t_nucl=-1.0) if topo == "4trk" else models.get(MODEL[topo])
    mid = lambda lo, hi, n: lo + (hi - lo) * (np.arange(n) + 0.5) / n
    sb = np.zeros(len(VCUTS)); so = np.zeros(len(VCUTS)); n_ok = n_all = 0
    for W in mid(*rng["W"], 3):
        for Q2 in mid(*rng["Q2"], 3):
            for mt in mid(*rng["mt"], 4):
                for ph in mid(0, 360, 4):
                    n_all += 1
                    if topo == "4trk":
                        m.t_nucl = -mt
                    b, o = [], []
                    try:
                        for v in VCUTS:
                            d = rc_factor(R.EB, Q2, W * W, -mt, np.radians(ph), m, h=+1,
                                          mx2_cut=None if v is None else MU2[topo] + v)
                            b.append(d["sigma_born"]); o.append(d["sigma_obs"])
                    except Exception:
                        continue
                    if np.all(np.isfinite(b)) and np.all(np.isfinite(o)):
                        sb += b; so += o; n_ok += 1
    return (so / sb).tolist() if n_ok else [float("nan")] * len(VCUTS), n_ok, n_all


# ------------------------------------------------------------------- main
def ratio_B(nr, nb, wr, wb):
    B = nr * wr / (nb * wb)
    return B, B * np.sqrt(1.0 / nr + 1.0 / nb)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R.DATA, "out")
    os.makedirs(outdir, exist_ok=True)
    F = fsr_factor()
    ev = collect(os.path.join(outdir, "events.npz"))
    specs = cell_specs()
    res = dict(vcuts=VCUTS, F_fsr=F, edges=EDGES, map=MAP, qslices=QSLICES, mariana=MARIANA,
               model=MODEL, sample={}, cells={}, flat={})

    # flat A per cell and topology -- no acceptance, so one per topology
    for topo in R.TOPOS:
        cache = os.path.join(outdir, f"flat_{topo}.json")
        if os.path.exists(cache):
            res["flat"][topo] = json.load(open(cache))
            continue
        uniq = {json.dumps(r, sort_keys=True): r for _, r, _ in specs}
        rngs = []
        for key, r in uniq.items():
            full = dict(r)
            for n in ("W", "Q2", "mt"):
                full.setdefault(n, RANGE[n])
            rngs.append((key, full))
        with Pool(NPROC) as p:
            got = p.map(_flat_cell, [(topo, r) for _, r in rngs])
        res["flat"][topo] = {k: dict(A=g[0], n_ok=g[1], n_all=g[2]) for (k, _), g in zip(rngs, got)}
        json.dump(res["flat"][topo], open(cache, "w"))
        print("flat", topo, "done", flush=True)

    for topo in R.TOPOS:
        for pol in ("inb", "outb"):
            rad, born = ev[(topo, pol, "rad")], ev[(topo, pol, "born")]
            res["sample"][f"{topo}_{pol}"] = {a: dict(n_gen=e["n_gen"], n_chunks=e["n_chunks"], sigma_nb=e["sigma"],
                                                      sigma_err=e["sigma_err"], flow=e["flow"],
                                                      n_sel=int(len(e["rec"])),
                                                      cal=e["cal"]) for a, e in (("rad", rad), ("born", born))}
            ei = eta_int(topo, rad["rec"][:, :4], os.path.join(outdir, f"eta_{topo}_{pol}.npz"))
            ex = eta_ext_matrix(rad["rec"][:, 4])
            eta = ei * ex * F
            good = np.all(np.isfinite(ei) & (ei > 0), axis=1)
            res["sample"][f"{topo}_{pol}"]["eta_ok"] = float(good.mean())
            print(f"{topo} {pol}: eta finite for {good.mean():.4f} of {len(good)} rad events", flush=True)
            if topo == "3trk":
                mu, sg = rad["cal"] if rad["cal"] else born["cal"]
                res["sample"][f"{topo}_{pol}"]["v_her"] = (mu + R.NSIG_MX * sg) ** 2 - R.MP ** 2
            cells = []
            for name, rng, spec in specs:
                sr, sb = cell_masks(rad["rec"], spec), cell_masks(born["rec"], spec)
                gr, gb = cell_masks(rad["gen"], spec), cell_masks(born["gen"], spec)
                nr, nb = int(sr.sum()), int(sb.sum())
                c = dict(binning=name, range=rng, n_rad=nr, n_born=nb)
                if nr >= NMIN and nb >= NMIN:
                    c["B"], c["B_err"] = ratio_B(nr, nb, rad["w"], born["w"])
                    ngr, ngb = int(gr.sum()), int(gb.sum())
                    c["B_gen"], c["B_gen_err"] = ratio_B(ngr, ngb, rad["w"], born["w"])
                    ge = sr & good
                    c["A"] = (ge.sum() / (1.0 / eta[ge]).sum(axis=0)).tolist()
                    c["A_int"] = (ge.sum() / (1.0 / ei[ge]).sum(axis=0)).tolist()
                    c["ext"] = (ge.sum() / (1.0 / ex[ge]).sum(axis=0)).tolist()
                    ys = np.array(c["A"][:-1]); xs = np.array(VCUTS[:-1])
                    c["v_eq"] = float(np.interp(c["B"], ys, xs)) if ys.min() <= c["B"] <= ys.max() else None
                    c["mean"] = {n: float(rad["rec"][sr, COLS.index(n)].mean()) for n in ("W", "Q2", "mt")}
                cells.append(c)
            res["cells"][f"{topo}_{pol}"] = cells
            mc = [c for c in cells if c["binning"] == "mariana"]
            print("  Mariana bins B:", ["%.4f+-%.4f" % (c["B"], c["B_err"]) if "B" in c else "-" for c in mc])

    with open(os.path.join(outdir, "rc_method_jpsi_combos.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("written", os.path.join(outdir, "rc_method_jpsi_combos.json"))


if __name__ == "__main__":
    main()
