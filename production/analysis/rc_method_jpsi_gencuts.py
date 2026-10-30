#!/usr/bin/env python3
"""Her kinematic cuts on the GENERATED momenta, no detector: how much of
method 1 - method 2 is physics of the selection, and how much is detector.

    python3 production/analysis/rc_method_jpsi_gencuts.py <figdir>

Both arms, F18 inbending files (generation does not know the torus), all
generated events.  The chain is the part of her selection that is kinematics:
M(ee), Delta p_z, and the missing-mass cut, built from the generated e', e+,
e-, p' with the nominal beam -- the photons are NOT added, they are what a
detector does not see.  Binned in her window (Q2 0.02-0.12, W 4.15-4.48, from
the generated e') and per W bin.

Also the decay-photon truth in the rad arm: the fraction with k* below the
M(ee) edge, which is F_fsr at generator level, and how many of THOSE the
missing-mass window still removes -- the decay radiation that F_fsr alone does
not count.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402
import rc_method_jpsi as R  # noqa: E402
import rc_method_jpsi_combos as K  # noqa: E402

FIG = sys.argv[1] if len(sys.argv) > 1 else "."


def main():
    pol = "inb"
    G, W8 = {}, {}
    for a in ("rad", "born"):
        g = C.load(f"{R.DATA}/ntuple/{a}_{pol}/*.f32", "jpsi", rec=False)
        k = R.kinematics(g)
        L = R.four(g["hplus_p3"], R.ME_) + R.four(g["hminus_p3"], R.ME_)
        mee = np.sqrt(np.clip(R.mdot(L, L), 0, None))
        k["kstar"] = (R.MJ ** 2 - mee ** 2) / (2 * R.MJ)
        G[a] = k
        W8[a] = R.sigma(a, pol)["mean"] / g["n_read"]
    # the MX window as the study uses it, from the reconstructed Born calibration
    ev = np.load(os.path.join(R.DATA, "out", "events.npz"), allow_pickle=True)["ev"].item()
    mu, sg = ev[("3trk", pol, "born")]["cal"]
    vher = (mu + R.NSIG_MX * sg) ** 2 - R.MP ** 2
    Kedge = (R.MJ ** 2 - R.MEE[0] ** 2) / (2 * R.MJ)
    out = dict(pol=pol, mx_window=(mu - R.NSIG_MX * sg, mu + R.NSIG_MX * sg), v_her=vher, chains={})

    def chains(k, topo):
        c = {"no cut": np.ones(len(k["W"]), bool)}
        c["M(ee)"] = (k["mee"] > R.MEE[0]) & (k["mee"] < R.MEE[1])
        if topo == "3trk":
            c["+ dpz"] = c["M(ee)"] & (k["dpz3"] > R.DPZ3_MIN)
            c["+ MX window"] = c["+ dpz"] & (np.abs(k["mx3"] - mu) < R.NSIG_MX * sg)
            c["MX window only"] = np.abs(k["mx3"] - mu) < R.NSIG_MX * sg
        else:
            c["+ dpz"] = c["M(ee)"] & (k["dpz4"] > R.DPZ4[0]) & (k["dpz4"] < R.DPZ4[1])
            c["+ MX2 total"] = c["+ dpz"] & (np.abs(k["mx2_tot"]) < R.MX2TOT)
            c["dpz only"] = (k["dpz4"] > R.DPZ4[0]) & (k["dpz4"] < R.DPZ4[1])
        return c

    for topo in R.TOPOS:
        tcol = "mt_V" if topo == "3trk" else "mt_p"
        res = {}
        for where, rng in [("her window", (4.15, 4.48))] + [(f"W {a}-{b}", (a, b)) for a, b in
                                                            zip(K.MARIANA["W"][:-1], K.MARIANA["W"][1:])]:
            win = {a: (G[a]["Q2"] >= 0.02) & (G[a]["Q2"] < 0.12) & (G[a]["W"] >= rng[0]) & (G[a]["W"] < rng[1])
                   & (G[a][tcol] >= 0.3) & (G[a][tcol] < 5.0) for a in G}
            ch = {a: chains(G[a], topo) for a in G}
            res[where] = {}
            for name in ch["rad"]:
                nr = int((win["rad"] & ch["rad"][name]).sum()); nb = int((win["born"] & ch["born"][name]).sum())
                B, Be = K.ratio_B(nr, nb, W8["rad"], W8["born"])
                res[where][name] = dict(B=B, B_err=Be, n_rad=nr, n_born=nb)
            print(topo, where, "  ".join(f"{n}: {v['B']:.4f}" for n, v in res[where].items()), flush=True)
        out["chains"][topo] = res
    # decay-photon truth, rad arm, her window, 3-track variables
    k = G["rad"]
    win = (k["Q2"] >= 0.02) & (k["Q2"] < 0.12) & (k["W"] >= 4.15) & (k["W"] < 4.48)
    soft = k["kstar"] < Kedge
    inmx = np.abs(k["mx3"] - mu) < R.NSIG_MX * sg
    out["fsr_truth"] = dict(F_gen=float(soft[win].mean()),
                            mx_keeps_soft=float(inmx[win & soft].mean()),
                            mx_keeps_hard=float(inmx[win & ~soft].mean()))
    print("decay photon: F_gen = %.4f (formula %.4f); MX window keeps %.4f of k*<K and %.4f of k*>K" %
          (out["fsr_truth"]["F_gen"], K.fsr_factor(), out["fsr_truth"]["mx_keeps_soft"],
           out["fsr_truth"]["mx_keeps_hard"]))
    with open(os.path.join(FIG, "gencuts.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
