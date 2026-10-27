#!/usr/bin/env python3
"""Generator-level closure: B from the GENERATED events of both arms, no cut, no detector,
against A = rc_factor with no cut, in 1-D W, Q2, |t|.

    python3 production/analysis/rc_method_jpsi_closure.py [outdir]

Same engine on both sides, so B_gen/A - 1 must be what A leaves out:
  - the pre-vertex radiator (rad itarg = 1), which at no cut changes only
    where events land, and
  - with t from e'J/psi, the decay-lepton radiation, which takes momentum out
    of the pair and moves t_V; with t from the proton it cannot.
Binned with the proton-side t (the box the campaign was drawn in, model
jpsi.diffrad_vm_pdet) and with the meson-side t (jpsi.diffrad_vm).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402
import rc_method_jpsi as R  # noqa: E402
import rc_method_jpsi_combos as K  # noqa: E402

NSUB = 8000


def _eta(args):
    model, rows = args
    import warnings
    warnings.filterwarnings("ignore")
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    m = models.get(model, t_nucl=-1.0) if model.endswith("pdet") else models.get(model)
    out = np.full(len(rows), np.nan)
    for i, (W, Q2, mt, ph) in enumerate(rows):
        if model.endswith("pdet"):
            m.t_nucl = -mt
        try:
            out[i] = rc_factor(R.EB, Q2, W * W, -mt, np.radians(ph), m, h=+1)["eta"]
        except Exception:
            pass
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R.DATA, "out")
    pol = "inb"
    G = {}
    for a in ("rad", "born"):
        g = C.load(f"{R.DATA}/ntuple/{a}_{pol}/*.f32", "jpsi", rec=False)
        G[a] = dict(k=R.kinematics(g), w=R.sigma(a, pol)["mean"] / g["n_read"], n=g["n_read"])
    rng = np.random.default_rng(20261027)
    res = dict(pol=pol, n_gen={a: G[a]["n"] for a in G}, nsub=NSUB, variants={})
    for label, model, tcol, pcol in (("t_proton", "jpsi.diffrad_vm_pdet", "mt_p", "phi_p"),
                                     ("t_meson", "jpsi.diffrad_vm", "mt_V", "phi_V")):
        kr = G["rad"]["k"]
        cols = lambda k: np.column_stack([k["W"], k["Q2"], k[tcol], k[pcol]])
        xr, xb = cols(kr), cols(G["born"]["k"])
        idx = rng.choice(len(xr), NSUB, replace=False)
        with Pool(K.NPROC) as p:
            eta = np.concatenate(p.map(_eta, [(model, c) for c in np.array_split(xr[idx], 80)]))
        ok = np.isfinite(eta) & (eta > 0)
        sub, eta = xr[idx][ok], eta[ok]
        V = dict(model=model, eta_ok=float(ok.mean()), bins={})
        for var, ci in (("W", 0), ("Q2", 1), ("mt", 2)):
            e = K.EDGES[var]
            rows = []
            for i in range(len(e) - 1):
                inr = (xr[:, ci] >= e[i]) & (xr[:, ci] < e[i + 1])
                inb = (xb[:, ci] >= e[i]) & (xb[:, ci] < e[i + 1])
                ins = (sub[:, ci] >= e[i]) & (sub[:, ci] < e[i + 1])
                if inr.sum() < 200 or inb.sum() < 200 or ins.sum() < 30:
                    continue
                B, Be = K.ratio_B(inr.sum(), inb.sum(), G["rad"]["w"], G["born"]["w"])
                A = ins.sum() / (1.0 / eta[ins]).sum()
                Ae = A * np.std(eta[ins]) / np.mean(eta[ins]) / np.sqrt(ins.sum())
                rows.append(dict(lo=e[i], hi=e[i + 1], B=B, B_err=Be, A=A, A_err=Ae,
                                 dev=100 * (B / A - 1), n_rad=int(inr.sum()), n_sub=int(ins.sum())))
            V["bins"][var] = rows
            print(label, var, " ".join("%+.2f" % r["dev"] for r in rows), flush=True)
        tot_B = K.ratio_B(len(xr), len(xb), G["rad"]["w"], G["born"]["w"])
        V["integrated"] = dict(B=tot_B[0], A=float(len(eta) / (1.0 / eta).sum()))
        print(label, "integrated B %.5f  A(sub) %.5f" % (V["integrated"]["B"], V["integrated"]["A"]))
        res["variants"][label] = V
    with open(os.path.join(outdir, "rc_method_jpsi_closure.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
