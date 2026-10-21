#!/usr/bin/env python3
"""Two ways to correct pi0 for radiation, in one (Q2, xB, -t) bin, phi by phi.

    python3 production/analysis/rc_method_pi0.py [outdir]

B  rad/Born MC, as the CLAS12 pi0 note does it (Korover et al., Eq. 20):
       delta_B(phi) = [N_rad(phi) sigma_rad / N_gen,rad] / [N_born(phi) sigma_born / N_gen,born]
   after the same reconstruction and the same cut chain on both arms.
A  the same RC engine used as a formula, no Born sample:
       delta_A(phi; mx2_cut) = sum sigma_obs / sum sigma_Born over a grid inside the bin,
   where the selection enters only as MX^2(e'p) < mx2_cut.

Cut chain from the note, Tables 7-10 (per polarity, per proton detector) and
Sec. 3.5, with the Sec. 4.7 proton smearing applied to the MC.  The ntuple has
no vertex and no sector, so the vertex cuts and the same-sector veto are NOT
applied; both act identically on the two arms.

Data: /volatile/clas12/vpk/prod_pi0_rga_t25_20260928, merged plots/*.f32,
copied to /Volumes/wd_14tb/rc_method_pi0.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402

DATA = "/Volumes/wd_14tb/rc_method_pi0"
EB = 10.6
MP, ME_, MPI = 0.938272, 0.000511, 0.1349768
NF, REC0 = 50, 13

# the bin: merged from the note's Tables 11-13 in the dense region
BIN = dict(Q2=(1.5, 2.5), xB=(0.15, 0.30), mt=(0.20, 0.60))
NPHI = 12

# Tables 7-10 as printed: DATA widths after corrections.  Kept for the record;
# the MC windows are calibrated instead (see calibrate()).
CUTS = {
    ("inb", "FD"):  dict(s_mgg=0.0073,  s_mm2ep=0.0861, me=(-0.02, 0.02)),
    ("inb", "CD"):  dict(s_mgg=0.00626, s_mm2ep=0.0842, me=(-0.015, 0.015)),
    ("outb", "FD"): dict(s_mgg=0.00742, s_mm2ep=0.0743, me=(-0.02, 0.02)),
    ("outb", "CD"): dict(s_mgg=0.00619, s_mm2ep=0.06811, me=(-0.015, 0.02)),
}
MM2TOT = (-0.006, 0.0025)
MEGG = (0.75, 1.2)
# Sec. 4.7, proton only: sigma_p (GeV), sigma_theta, sigma_phi (rad).  OFF by
# default: the note smears because its MC was NARROWER than data; this GEMC
# sample is already WIDER than the note's data (sigma M(gg) 0.011 vs 0.0073,
# MM2(e'p) ~0.12 vs 0.086), so smearing would move it further away.
DO_SMEAR = os.environ.get("RCM_SMEAR", "0") == "1"
SMEAR = {"FD": (0.020, 0.0050, 0.042), "CD": (0.025, 0.028, 0.015)}


def four(p3, m):
    return np.column_stack([np.sqrt((p3 ** 2).sum(1) + m * m), p3])


def mdot(a, b):
    return a[:, 0] * b[:, 0] - (a[:, 1:] * b[:, 1:]).sum(1)


def trento_phi(q3, k3, h3):
    """Trento azimuth of the hadron h around the virtual photon q, lepton plane from k (beam).

    phi is the azimuth of the PRODUCED MESON, h = p(g1) + p(g2): the note's
    definition and the one rc_factor documents (core/kinematics.py).  For a
    radiative event the pi0 is not opposite the proton, so phi_p + 180 is a
    different variable, and it is not used.
    """
    qn = q3 / np.linalg.norm(q3, axis=1)[:, None]
    n_l = np.cross(qn, k3)
    n_h = np.cross(qn, h3)
    cos = (n_l * n_h).sum(1) / (np.linalg.norm(n_l, axis=1) * np.linalg.norm(n_h, axis=1))
    sin = (np.cross(n_l, n_h) * qn).sum(1) / (np.linalg.norm(n_l, axis=1) * np.linalg.norm(n_h, axis=1))
    return np.degrees(np.arctan2(sin, cos)) % 360.0


def kinematics(pe, pp, g1, g2):
    """Everything the cut chain and the binning need, from four three-momenta."""
    k = np.array([EB, 0, 0, EB])
    tgt = np.array([MP, 0, 0, 0])
    E = four(pe, ME_); P = four(pp, MP); G1 = four(g1, 0.0); G2 = four(g2, 0.0)
    q = k - E
    Q2 = -mdot(q, q)
    nu = q[:, 0]
    W2 = MP ** 2 + 2 * MP * nu - Q2
    xB = Q2 / (2 * MP * nu)
    t = 2 * MP ** 2 - 2 * MP * P[:, 0]
    pi = G1 + G2
    X_ep = k + tgt - E - P                     # missing pi0
    X_egg = k + tgt - E - pi                   # missing proton
    X_tot = X_ep - pi
    out = dict(Q2=Q2, xB=xB, W=np.sqrt(np.maximum(W2, 0)), W2=W2, mt=-t, t=t,
               pe=np.linalg.norm(pe, axis=1),
               mgg=np.sqrt(np.maximum(mdot(pi, pi), 0)),
               mm2ep=mdot(X_ep, X_ep), megg=np.sqrt(np.maximum(mdot(X_egg, X_egg), 0)),
               mm2tot=mdot(X_tot, X_tot), me=X_tot[:, 0],
               mpt=np.hypot(X_tot[:, 1], X_tot[:, 2]),
               phi=trento_phi(q[:, 1:], np.tile([0, 0, EB], (len(q), 1)), pi[:, 1:]))
    c = (X_ep[:, 1:] * pi[:, 1:]).sum(1) / (np.linalg.norm(X_ep[:, 1:], axis=1) *
                                             np.linalg.norm(pi[:, 1:], axis=1))
    out["cone"] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    c = (g1 * g2).sum(1) / (np.linalg.norm(g1, axis=1) * np.linalg.norm(g2, axis=1))
    out["open"] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


def load(arm, pol, rng):
    a = np.fromfile(f"{DATA}/ntuple/{arm}_{pol}.f32", "<f4").reshape(-1, NF).astype(np.float64)
    n_gen = len(a)
    gen = kinematics(*(a[:, 1 + 3 * k:4 + 3 * k] for k in range(4)))

    # reconstruction: found, momentum, |dp/p| (campaign_lib rule), PID, detector
    ok = np.ones(n_gen, bool)
    want_pid = (11, 2212, 22, 22)
    for k in range(4):
        b = REC0 + 8 * k
        pk = np.linalg.norm(a[:, b + 5:b + 8], axis=1)
        pg = np.linalg.norm(a[:, 1 + 3 * k:4 + 3 * k], axis=1)
        thr = C.DP_MAX["charged" if k < 2 else "photon"]
        ok &= (a[:, b] > 0.5) & (pk > 0) & (np.abs(pk - pg) < thr * pg)
        ok &= np.rint(a[:, b + 1]) == want_pid[k]
    det = lambda k: (np.abs(a[:, REC0 + 8 * k + 3]) // 1000).astype(int)
    ok &= det(0) == 2                                     # electron in FD
    ok &= np.isin(det(1), (2, 4))                         # proton FD or CD
    ok &= np.isin(det(2), (1, 2)) & np.isin(det(3), (1, 2))   # photons FT/FD (calorimeters)
    r = a[ok]
    pdet = np.where(det(1)[ok] == 2, "FD", "CD")

    # Sec. 4.7 proton smearing
    pp = r[:, REC0 + 8 + 5:REC0 + 8 + 8].copy()
    pm = np.linalg.norm(pp, axis=1)
    th = np.arccos(pp[:, 2] / pm)
    ph = np.arctan2(pp[:, 1], pp[:, 0])
    for d, (sp, st, sf) in (SMEAR.items() if DO_SMEAR else ()):
        m = pdet == d
        pm[m] += rng.normal(0, sp, m.sum())
        th[m] += rng.normal(0, st, m.sum())
        ph[m] += rng.normal(0, sf, m.sum())
    pp = np.column_stack([pm * np.sin(th) * np.cos(ph), pm * np.sin(th) * np.sin(ph), pm * np.cos(th)])

    rec = kinematics(r[:, REC0 + 5:REC0 + 8], pp,
                     r[:, REC0 + 16 + 5:REC0 + 16 + 8], r[:, REC0 + 24 + 5:REC0 + 24 + 8])
    rec["pdet"] = pdet
    return dict(n_gen=n_gen, gen=gen, rec=rec, n_rec=int(ok.sum()))


def in_bin(k):
    return ((k["Q2"] >= BIN["Q2"][0]) & (k["Q2"] < BIN["Q2"][1]) &
            (k["xB"] >= BIN["xB"][0]) & (k["xB"] < BIN["xB"][1]) &
            (k["mt"] >= BIN["mt"][0]) & (k["mt"] < BIN["mt"][1]))


# Windows, centred and sized on the Born MC itself.  The note's numbers
# (Tables 7-10) are data widths AFTER momentum and energy-loss corrections; the
# uncorrected GEMC sample has its ME peak at +0.1 GeV and MM2(e'pgg) three times
# wider, so the literal windows keep ~3 % of Born.  What carries over is the
# rule: each window is +-NSIG robust sigma around the peak, per polarity and per
# proton detector, exactly as the note builds its own from data.
NSIG = 3.0
CAL_VARS = ("mgg", "mm2ep", "megg", "mm2tot", "me")


def calibrate(k):
    """Peak and robust sigma of each exclusivity variable, from Born, per proton detector."""
    dis = (k["Q2"] > 1) & (k["W"] > 2) & (k["pe"] > 2)
    cal = {}
    for d in ("FD", "CD"):
        m = dis & (k["pdet"] == d)
        cal[d] = {}
        for v in CAL_VARS:
            x = k[v][m]
            for _ in range(3):                      # iterate the core: 2.5 sigma clip
                q16, q50, q84 = np.percentile(x, [16, 50, 84])
                s = 0.5 * (q84 - q16)
                x = x[np.abs(x - q50) < 2.5 * s]
            q16, q50, q84 = np.percentile(x, [16, 50, 84])
            cal[d][v] = (float(q50), float(0.5 * (q84 - q16)))
    return cal


def cut_levels(k, cal):
    """Cumulative selections.  Returns an ordered dict name -> mask."""
    dis = (k["Q2"] > 1) & (k["W"] > 2) & (k["pe"] > 2)
    def win(v):
        c = np.where(k["pdet"] == "FD", cal["FD"][v][0], cal["CD"][v][0])
        s = np.where(k["pdet"] == "FD", cal["FD"][v][1], cal["CD"][v][1])
        return np.abs(k[v] - c) < NSIG * s
    L = {}
    L["topology+DIS"] = dis & in_bin(k)
    L["+M(gg), MM2(e'p)"] = L["topology+DIS"] & win("mgg") & win("mm2ep") & (k["open"] > 1)
    L["+M(e'gg), cone, MPT"] = (L["+M(gg), MM2(e'p)"] &
                                win("megg") &
                                (k["cone"] < 4) & (k["mpt"] < 0.75))
    L["+MM2(e'pgg), ME: full chain"] = L["+M(e'gg), cone, MPT"] & win("mm2tot") & win("me")
    return L


def phi_hist(phi, m):
    return np.histogram(phi[m], bins=NPHI, range=(0, 360))[0].astype(float)


def ratio(nr, nb, wr, wb):
    """delta = (nr wr)/(nb wb) with its binomial-free Poisson error."""
    with np.errstate(invalid="ignore", divide="ignore"):
        d = nr * wr / (nb * wb)
        e = d * np.sqrt(1 / nr + 1 / nb)
    return d, e


# ------------------------------------------------------------- method A ---
def method_A(mx2_cuts, nq=3, nx=3, nt=3, nphi_sub=3):
    """sum sigma_obs / sum sigma_Born over a grid in the bin, per phi bin, per cut."""
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    m = models.get("pi0.amp2609", t_nucl=-0.3)
    mid = lambda lo, hi, n: lo + (hi - lo) * (np.arange(n) + 0.5) / n
    Qs, Xs, Ts = mid(*BIN["Q2"], nq), mid(*BIN["xB"], nx), mid(*BIN["mt"], nt)
    sb = np.zeros((len(mx2_cuts), NPHI)); so = np.zeros_like(sb)
    n_ok = 0; n_all = 0
    for ip in range(NPHI):
        for phd in mid(ip * 360 / NPHI, (ip + 1) * 360 / NPHI, nphi_sub):
            for Q2 in Qs:
                for xB in Xs:
                    W2 = MP ** 2 + Q2 * (1 / xB - 1)
                    for mt in Ts:
                        n_all += 1
                        m.t_nucl = -mt
                        row_b, row_o = [], []
                        try:
                            for c in mx2_cuts:
                                d = rc_factor(EB, Q2, W2, -mt, np.radians(phd), m, h=+1, mx2_cut=c)
                                row_b.append(d["sigma_born"]); row_o.append(d["sigma_obs"])
                        except Exception:
                            continue
                        if not (np.all(np.isfinite(row_b)) and np.all(np.isfinite(row_o))):
                            continue
                        n_ok += 1
                        sb[:, ip] += row_b; so[:, ip] += row_o
    return so / sb, n_ok, n_all


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "out")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20261021)
    sig = {a: C.sigma_nb(f"{DATA}/prov_{a}", a)["mean"] for a in ("rad", "born")}
    res = dict(bin=BIN, nphi=NPHI, sigma_nb=sig, pol={})
    hists = {}
    for pol in ("inb", "outb"):
        S = {a: load(a, pol, rng) for a in ("rad", "born")}
        w = {a: sig[a] / S[a]["n_gen"] for a in S}
        P = dict(n_gen={a: S[a]["n_gen"] for a in S}, n_rec={a: S[a]["n_rec"] for a in S}, w=w)

        # B at reconstructed level, per cut level
        cal = calibrate(S["born"]["rec"])
        P["cal"] = cal
        levels = {a: cut_levels(S[a]["rec"], cal) for a in S}
        P["B_rec"] = {}
        for name in levels["rad"]:
            nr = phi_hist(S["rad"]["rec"]["phi"], levels["rad"][name])
            nb = phi_hist(S["born"]["rec"]["phi"], levels["born"][name])
            d, e = ratio(nr, nb, w["rad"], w["born"])
            P["B_rec"][name] = dict(n_rad=nr.tolist(), n_born=nb.tolist(), delta=d.tolist(), err=e.tolist(),
                                    delta_int=float(nr.sum() * w["rad"] / (nb.sum() * w["born"])))

        # B at generated level with a cut on generated MM2(e'p): the closure of A
        P["B_gen"] = {}
        for c in GEN_CUTS:
            sel = {a: in_bin(S[a]["gen"]) & ((S[a]["gen"]["mm2ep"] < c) if c else True) for a in S}
            nr = phi_hist(S["rad"]["gen"]["phi"], sel["rad"]); nb = phi_hist(S["born"]["gen"]["phi"], sel["born"])
            d, e = ratio(nr, nb, w["rad"], w["born"])
            P["B_gen"][str(c)] = dict(delta=d.tolist(), err=e.tolist(), n_rad=nr.tolist(), n_born=nb.tolist())

        # distributions for the "show the data" page
        full = "+MM2(e'pgg), ME: full chain"
        for a in S:
            k = S[a]["rec"]; base = levels[a]["topology+DIS"]
            hists[(pol, a)] = {v: k[v][base] for v in ("mgg", "mm2ep", "megg", "mm2tot", "me", "mpt", "cone", "open", "phi")}
            hists[(pol, a)]["full"] = levels[a][full][base]
            hists[(pol, a)]["pdet"] = k["pdet"][base]
            hists[(pol, a)]["w"] = w[a]
            hists[(pol, a)]["gen_mm2ep"] = S[a]["gen"]["mm2ep"][in_bin(S[a]["gen"])]
        res["pol"][pol] = P
        print(pol, "n_rec", P["n_rec"], {n: round(v["delta_int"], 4) for n, v in P["B_rec"].items()})

    # both polarities together: each arm's yield in barns-per-event, summed
    comb = {}
    for key in ("B_rec", "B_gen"):
        comb[key] = {}
        for name in res["pol"]["inb"][key]:
            nr = sum(np.array(res["pol"][p][key][name]["n_rad"]) * res["pol"][p]["w"]["rad"] for p in res["pol"])
            nb = sum(np.array(res["pol"][p][key][name]["n_born"]) * res["pol"][p]["w"]["born"] for p in res["pol"])
            cr = sum(np.array(res["pol"][p][key][name]["n_rad"]) for p in res["pol"])
            cb = sum(np.array(res["pol"][p][key][name]["n_born"]) for p in res["pol"])
            with np.errstate(invalid="ignore", divide="ignore"):
                d = nr / nb
                e = d * np.sqrt(1 / cr + 1 / cb)
            comb[key][name] = dict(n_rad=cr.tolist(), n_born=cb.tolist(), delta=d.tolist(), err=e.tolist(),
                                   delta_int=float(nr.sum() / nb.sum()))
    res["combined"] = comb
    print("combined", {n: (int(sum(v["n_rad"])), int(sum(v["n_born"])), round(v["delta_int"], 4))
                       for n, v in comb["B_rec"].items()})

    A, n_ok, n_all = method_A(A_CUTS)
    res["A"] = {str(c): A[i].tolist() for i, c in enumerate(A_CUTS)}
    res["A_points"] = [n_ok, n_all]
    print("A grid points ok", n_ok, "of", n_all)
    with open(os.path.join(outdir, "rc_method_pi0.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.save(os.path.join(outdir, "hists.npy"), {f"{p}|{a}": v for (p, a), v in hists.items()}, allow_pickle=True)


GEN_CUTS = [0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.5, None]
A_CUTS = [0.03, 0.05, 0.065, 0.08, 0.10, 0.12, 0.145, 0.17, 0.21, 0.25, 0.30, 0.35, 0.42, 0.5, 0.6, None]

if __name__ == "__main__":
    main()
