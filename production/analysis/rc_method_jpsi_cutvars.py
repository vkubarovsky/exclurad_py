#!/usr/bin/env python3
"""N-1 per cut: how much of C_rad each of her cuts carries, MC against formula.

    python3 production/analysis/rc_method_jpsi_cutvars.py <figdir>

For every cut X, with all OTHER cuts applied,
    r_X = [N_rad(all) / N_rad(all but X)] / [N_born(all) / N_born(all but X)]
the radiation-specific part of X's survival.  The product over X is not B
(the chain is not factorised), but each r_X says what that one cut does:
  - a cut rc_factor knows nothing of (PID, |p|, dpz, vz, theta_e, nu) must give
    r_X = 1 unless radiation acts through it; r_X - 1 is then a piece of
    method 1 that method 2 cannot contain;
  - M(ee) must give r = F_fsr = 0.878 if the decay radiation were all it does;
  - the MX window (3 tracks) must give eta_int(v_her) / eta_int(no cut).
In her kinematic window, Q2 0.02-0.12 and W 4.15-4.48, and over the box.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_method_jpsi as R  # noqa: E402
import rc_method_jpsi_combos as K  # noqa: E402

FIG = sys.argv[1] if len(sys.argv) > 1 else "."


def single_masks(k, pol, topo, cal):
    """Each of her cuts on its own (not cumulative), name -> mask."""
    inward = "hminus_vz" if pol == "inb" else "hplus_vz"
    m = {
        "FT theta_e": (k["theta_e"] > R.THETA_E[0]) & (k["theta_e"] < R.THETA_E[1]),
        "nu > 8.1": k["nu"] > R.NU_MIN,
        "lepton PID": (k["hminus_pid"] == 11) & (k["hplus_pid"] == -11),
        "lepton |p|": (k["p_em"] > R.P_LEP_MIN) & (k["p_em"] < R.P_EM_MAX) & (k["p_ep"] > R.P_LEP_MIN),
        "vz": (k[inward] > R.VZ[0]) & (k[inward] < R.VZ[1]),
        "M(ee)": (k["mee"] > R.MEE[0]) & (k["mee"] < R.MEE[1]),
    }
    if topo == "3trk":
        m["dpz"] = k["dpz3"] > R.DPZ3_MIN
        m["MX window"] = np.abs(k["mx3"] - cal[0]) < R.NSIG_MX * cal[1]
    else:
        m["proton"] = (k["prot_pid"] == 2212) & (k["p_p"] > R.P_PROT_MIN)
        m["dpz"] = (k["dpz4"] > R.DPZ4[0]) & (k["dpz4"] < R.DPZ4[1])
        m["MX2 total"] = np.abs(k["mx2_tot"]) < R.MX2TOT
    return m


def window(k, topo, where):
    if where == "box":
        return np.ones(len(k["W"]), bool)
    mt = k["mt_V"] if topo == "3trk" else k["mt_p"]
    return ((k["Q2"] >= 0.02) & (k["Q2"] < 0.12) & (k["W"] >= 4.15) & (k["W"] < 4.48)
            & (mt >= 0.3) & (mt < 5.0))


def surv(all_, nm1):
    n, k = nm1.sum(), all_.sum()
    f = k / n
    return f, np.sqrt(f * (1 - f) / n)


def main():
    out = {}
    combos = json.load(open(os.path.join(R.DATA, "out", "rc_method_jpsi_combos.json")))
    for topo in R.TOPOS:
        for pol in ("inb", "outb"):
            S = {a: R.load(a, pol, topo) for a in ("rad", "born")}
            cal = R.calibrate_mx3(S["born"]["rec"], R.pre_mx(R.cuts(S["born"]["rec"], pol, topo))) if topo == "3trk" else None
            res = {}
            for where in ("mariana", "box"):
                M = {a: single_masks(S[a]["rec"], pol, topo, cal) for a in S}
                W = {a: window(S[a]["rec"], topo, where) for a in S}
                allm = {a: W[a] & np.logical_and.reduce(list(M[a].values())) for a in S}
                rows = {}
                for X in M["rad"]:
                    nm1 = {a: W[a] & np.logical_and.reduce([v for n, v in M[a].items() if n != X]) for a in S}
                    fr, er = surv(allm["rad"], nm1["rad"]); fb, eb = surv(allm["born"], nm1["born"])
                    rows[X] = dict(f_rad=fr, f_born=fb, r=fr / fb, r_err=(fr / fb) * np.hypot(er / fr, eb / fb))
                nr, nb = allm["rad"].sum(), allm["born"].sum()
                B, Be = K.ratio_B(nr, nb, S["rad"]["w"], S["born"]["w"])
                res[where] = dict(B=B, B_err=Be, n_rad=int(nr), n_born=int(nb), cuts=rows)
                print(f"\n{topo} {pol} {where}: B = {B:.4f} +- {Be:.4f}")
                for X, q in rows.items():
                    print(f"   {X:12}  rad keeps {q['f_rad']:.4f}  born {q['f_born']:.4f}   r = {q['r']:.4f} +- {q['r_err']:.4f}")
            if topo == "3trk":
                # what rc_factor says the MX window should do, over the selected rad events
                z = np.load(os.path.join(R.DATA, "out", f"eta_{topo}_{pol}.npz"))
                e = z["eta"]; ok = np.all(np.isfinite(e) & (e > 0), axis=1)
                vh = (cal[0] + R.NSIG_MX * cal[1]) ** 2 - R.MP ** 2
                a_v = np.array([np.interp(vh, K.VCUTS[:-1], row[:-1]) for row in e[ok]])
                res["A_mx_ratio"] = float((1 / e[ok][:, -1]).sum() / (1 / a_v).sum())
                print(f"   rc_factor: eta(v_her={vh:.3f}) / eta(no cut) = {res['A_mx_ratio']:.4f}")
            res["F_fsr"] = combos["F_fsr"]
            out[f"{topo}_{pol}"] = res
    with open(os.path.join(FIG, "cutvars.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
