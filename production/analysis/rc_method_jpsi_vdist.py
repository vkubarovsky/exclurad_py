#!/usr/bin/env python3
"""Which of her cuts removes which radiation: each cut's efficiency against the
GENERATED production v and the GENERATED decay-photon energy k*.

    python3 production/analysis/rc_method_jpsi_vdist.py <figdir>

Rad arm only.  For the events with the reconstructed topology:
    v_gen  = the v of the model method 2 uses for the topology, at generator level
             with the nominal beam -- production radiation plus the pre-vertex photon:
               3 tracks  MX^2(e' J/psi) - m_p^2, J/psi = e+ + e- + gamma_dec
               4 tracks  MX^2(e' p') - m_J^2
    k*     = (M_J^2 - M(ee)_gen^2) / 2 M_J, the decay photon in the J/psi frame
The cut flow is rc_method_jpsi.cuts(); each curve is the fraction that
survives the cumulative chain up to that cut, so the drop from one curve to
the next is what that cut removes, as a function of the radiation.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402
import rc_method_jpsi as R  # noqa: E402

FIG = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(FIG, exist_ok=True)


def gen_rad(pol, topo):
    pat = f"{R.DATA}/ntuple/rad_{pol}/*.f32"
    g = C.load(pat, "jpsi", rec=False)
    rec = C.load(pat, "jpsi", rec=True, require=C.TOPOLOGY[R.TOPOS[topo]])
    keep = rec["keep"]
    k = np.array([R.EB, 0, 0, R.EB]); tgt = np.array([R.MP, 0, 0, 0])
    E = R.four(g["escat_p3"], R.ME_)
    L = R.four(g["hplus_p3"], R.ME_) + R.four(g["hminus_p3"], R.ME_)
    G = R.four(g["gdec_p3"], 0.0)
    V = L + G
    if topo == "3trk":
        # the missing proton: MX^2(e' J/psi) - m_p^2, jpsi.diffrad_vm's v
        X = k + tgt - E - V
        v = R.mdot(X, X) - R.MP ** 2
    else:
        # the missing J/psi: MX^2(e' p') - m_J^2, jpsi.diffrad_vm_pdet's v
        X = k + tgt - E - R.four(g["prot_p3"], R.MP)
        v = R.mdot(X, X) - R.MJ ** 2
    mee = np.sqrt(np.clip(R.mdot(L, L), 0, None))
    kstar = (R.MJ ** 2 - mee ** 2) / (2 * R.MJ)
    return dict(v=v[keep], kstar=kstar[keep], egprod=g["gprod_p"][keep]), R.kinematics(rec)


def main():
    pol = "inb"
    out = {}
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.4))
    for row, topo in enumerate(R.TOPOS):
        gen, rec = gen_rad(pol, topo)
        cal = None
        if topo == "3trk":
            # the same mu, sigma as the study: from Born
            b = R.load("born", pol, topo)["rec"]
            cal = R.calibrate_mx3(b, R.pre_mx(R.cuts(b, pol, topo)))
        L = R.cuts(rec, pol, topo, cal)
        for col, (var, edges, lab) in enumerate((
                ("v", np.concatenate([[-0.05], np.geomspace(0.002, 3.0, 24)]), "generated v (GeV$^2$) of the topology's model, production + pre-vertex"),
                ("kstar", np.concatenate([[0.0], np.geomspace(0.0005, 1.2, 24)]), "generated decay photon k* (GeV), J/$\\psi$ frame"))):
            ax = axs[row, col]
            x = gen[var]
            idx = np.digitize(x, edges) - 1
            xc = np.sqrt(np.abs(edges[1:] * edges[:-1])); xc[0] = edges[1] / 2
            n0 = np.bincount(idx[(idx >= 0) & (idx < len(edges) - 1)], minlength=len(edges) - 1)[:len(edges) - 1]
            cm = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(L)))
            out.setdefault(topo, {})[var] = dict(edges=edges.tolist(), n=n0.tolist(), eff={})
            for (name, m), cc in zip(L.items(), cm):
                if name == "topology":
                    continue
                nk = np.bincount(idx[m & (idx >= 0) & (idx < len(edges) - 1)], minlength=len(edges) - 1)[:len(edges) - 1]
                with np.errstate(invalid="ignore", divide="ignore"):
                    eff = nk / n0
                ax.plot(xc, eff, "o-", ms=3, lw=1, color=cc, label=name)
                out[topo][var]["eff"][name] = eff.tolist()
            ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.set_xlabel(lab)
            ax.set_ylabel("fraction surviving the chain")
            ax.set_title(f"rad arm, F18 {pol}, {topo}: {len(x)} events with the topology", fontsize=9)
            if var == "kstar":
                K = (R.MJ ** 2 - R.MEE[0] ** 2) / (2 * R.MJ)
                ax.axvline(K, color="#888888", ls="--", lw=0.8)
                ax.text(K, 0.05, " M(ee)=2.95", fontsize=7, color="#888888")
        axs[row, 0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v1_cuts_vs_radiation.png"), dpi=150, bbox_inches="tight")
    json.dump(out, open(os.path.join(FIG, "vdist.json"), "w"), default=float)
    print("written", FIG)


if __name__ == "__main__":
    main()
