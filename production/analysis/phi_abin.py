#!/usr/bin/env python3
"""phi(1020): A(bin) = acceptance x radiative correction, for the Born cross section.

    python3 production/analysis/phi_abin.py ROOT POL BG [--vcut 0.286] [--out DIR]
                                            [--ndata counts.csv --lumi L_nb^-1]

ROOT is a campaign directory holding  rad/ and born/,  each with  out_<POL>/*.f32
(one ntuple row per WRITTEN generated event) and  prov/<arm>_NNN.config.json.
BG picks the background-merging set from the file name (bgnone, bg45nA, ...).

    sigma_Born(bin) = N_data(bin) / ( L * Br * A(bin) )

    A(bin)   = w_rad R_rad(bin) / ( w_born G_born(bin) )       w = sigma_nb / N_written
             = eps_acc x eta_rec
    eps_acc  = R_born(bin) / G_born(bin)                        Born acceptance
    eta_rec  = w_rad R_rad(bin) / ( w_born R_born(bin) )        radiative correction

    R_arm    RECONSTRUCTED events of that arm in the bin, after the full selection
    G_born   GENERATED Born events in the bin, no detector, no cut

R_rad and N_data MUST be counted with the identical selection; that is the whole
content of A.  Both arms ran with vcut = 0 (fully inclusive in v), so the
exclusivity cut exists only here, on RECONSTRUCTED quantities:
    v = MX^2(e' K+ K-) - M_p^2         (core/kinematics.py: v = MX^2 - mU^2)
so v < v_cut is MX^2(e'K+K-) < M_p^2 + v_cut, with detector resolution in it.

The R_born/G_born split is the same algebra checked on J/psi (one bin,
2026-11-13): the two-step N/(A_B C_RC) and the one-step sigma_B(model) N/N_rec(rad)
agree to 1e-16.  The Born arm is what lets eps_acc and eta_rec be quoted apart.

phi is the Trento azimuth of the PRODUCED MESON (K+K- pair), the generator's
convention (core/kinematics.py:84).  If the analysis bins by the proton, set
PHI_FROM = "proton": phi_p = phi_meson + 180 for a Born event.  The generated
and the reconstructed side always use the same definition.

Br(phi -> K+K-) = 0.491 is applied here; the generator decays every phi to K+K-.

=====================================================================
THE SELECTION BELOW IS A PLACEHOLDER.  Every cut marked PLACEHOLDER is to be
replaced by the analysis code's own cut, value for value, before any number
from this script is used.  The output says so in its header until
PLACEHOLDER_CUTS is set to False.
=====================================================================
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402

EB = 10.6
MP, ME_, MK = 0.938272, 0.000511, 0.493677
BR_KK = 0.491

PLACEHOLDER_CUTS = True

# ---- binning: PLACEHOLDER edges, to be replaced by the analysis's ----------
EDGES = dict(
    Q2=np.array([1.0, 1.5, 2.0, 3.0, 5.0]),
    xB=np.array([0.10, 0.20, 0.30, 0.45, 0.70]),
    mt=np.array([0.1, 0.4, 0.7, 1.0, 1.5, 2.5]),        # -t, GeV^2, from the proton
    phi=np.linspace(0.0, 360.0, 13),
)
VARS = ("Q2", "xB", "mt", "phi")
PHI_FROM = "meson"          # "meson" (generator convention) or "proton"

# ---- selection: PLACEHOLDER values ----------------------------------------
TOPOLOGY = (0, 1, 2, 3)     # e', p, K+, K- all reconstructed          PLACEHOLDER
PID = {"escat": 11, "prot": 2212, "hplus": 321, "hminus": -321}  #     PLACEHOLDER
Q2_MIN, W_MIN = 1.0, 2.0    #                                          PLACEHOLDER
MKK = (0.99, 1.05)          # M(K+K-) window, GeV                      PLACEHOLDER
VCUTS = (0.04, 0.15, 0.286)  # v < v_cut on reconstructed MX^2(e'K+K-) - M_p^2


def four(p3, m):
    return np.column_stack([np.sqrt((p3 ** 2).sum(1) + m * m), p3])


def mdot(a, b):
    return a[:, 0] * b[:, 0] - (a[:, 1:] * b[:, 1:]).sum(1)


def trento_phi(q3, h3):
    """Trento azimuth of h around q, lepton plane from the beam along +z, degrees 0..360."""
    k3 = np.tile([0.0, 0.0, EB], (len(q3), 1))
    qn = q3 / np.linalg.norm(q3, axis=1)[:, None]
    n_l, n_h = np.cross(qn, k3), np.cross(qn, h3)
    nn = np.linalg.norm(n_l, axis=1) * np.linalg.norm(n_h, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (n_l * n_h).sum(1) / nn
        sin = (np.cross(n_l, n_h) * qn).sum(1) / nn
    return np.degrees(np.arctan2(sin, cos)) % 360.0


def kinematics(d, idx=None):
    """Binning and selection variables from the four three-momenta, nominal beam."""
    s = (lambda x: x) if idx is None else (lambda x: x[idx])
    pe, pp, kp, km = (s(d[n + "_p3"]) for n in ("escat", "prot", "hplus", "hminus"))
    k = np.array([EB, 0, 0, EB])
    tgt = np.array([MP, 0, 0, 0])
    E, P, K1, K2 = four(pe, ME_), four(pp, MP), four(kp, MK), four(km, MK)
    V = K1 + K2
    q = k - E
    Q2 = -mdot(q, q)
    nu = q[:, 0]
    W2 = MP ** 2 + 2 * MP * nu - Q2
    X_eV = k + tgt - E - V                     # missing proton
    phi_m = trento_phi(q[:, 1:], V[:, 1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = dict(Q2=Q2, nu=nu, xB=Q2 / (2 * MP * nu), W=np.sqrt(np.clip(W2, 0, None)),
                   mt=-mdot(P - tgt, P - tgt),
                   mkk=np.sqrt(np.clip(mdot(V, V), 0, None)),
                   v=mdot(X_eV, X_eV) - MP ** 2,
                   phi=phi_m if PHI_FROM == "meson" else (phi_m + 180.0) % 360.0)
    for n in PID:
        if n + "_pid" in d:
            out[n + "_pid"] = s(d[n + "_pid"])
    return out


def selection(k, v_cut):
    """The reconstructed-level selection.  ALL VALUES ARE PLACEHOLDERS."""
    m = np.ones(len(k["Q2"]), bool)
    for n, pid in PID.items():
        if n + "_pid" in k:
            m &= k[n + "_pid"] == pid
    m &= (k["Q2"] > Q2_MIN) & (k["W"] > W_MIN)
    m &= (k["mkk"] > MKK[0]) & (k["mkk"] < MKK[1])
    if v_cut is not None:
        m &= k["v"] < v_cut
    return m


def hist(k, m):
    x = np.column_stack([k[v][m] for v in VARS])
    return np.histogramdd(x, bins=[EDGES[v] for v in VARS])[0]


def chunk_ids(files):
    return sorted({int(re.search(r"_(\d+)\.f32$", f).group(1)) for f in files})


def load_arm(root, arm, pol, bg):
    """Generated and reconstructed kinematics of one arm, and its per-event weight."""
    pat = f"{root}/{arm}/out_{pol}/*_{pol}_{bg}_*.f32"
    files = sorted(glob.glob(pat))
    if not files:
        raise RuntimeError(f"no files match {pat}")
    chunks = chunk_ids(files)
    s = C.sigma_nb(f"{root}/{arm}/prov", arm, chunks=chunks)
    if s["preset_unknown"]:
        raise RuntimeError(f"{arm}: {s['preset_unknown']} sidecars with unverified normalisation")
    gen = C.load(pat, "phi", rec=False)
    rec = C.load(pat, "phi", rec=True, require=TOPOLOGY)
    return dict(n_written=gen["n_read"], n_chunks=len(chunks), sigma_nb=s["mean"], sigma_err=s["err"],
                w=s["mean"] / gen["n_read"], gen=kinematics(gen), rec=kinematics(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("pol"); ap.add_argument("bg")
    ap.add_argument("--vcut", type=float, action="append")
    ap.add_argument("--out", default=".")
    ap.add_argument("--ndata", help="npy with N_data in the same 4-D binning")
    ap.add_argument("--lumi", type=float, help="integrated luminosity, nb^-1")
    a = ap.parse_args()
    vcuts = a.vcut or list(VCUTS)
    os.makedirs(a.out, exist_ok=True)
    if PLACEHOLDER_CUTS:
        print("*** PLACEHOLDER CUTS: these A(bin) are not for the cross section ***")

    S = {arm: load_arm(a.root, arm, a.pol, a.bg) for arm in ("rad", "born")}
    for arm, s in S.items():
        print(f"{arm:4s} chunks {s['n_chunks']}  written {s['n_written']}  "
              f"sigma {s['sigma_nb']:.6g} +- {s['sigma_err']:.2g} nb  rec(topology) {len(s['rec']['Q2'])}")

    G = hist(S["born"]["gen"], np.ones(S["born"]["n_written"], bool))
    wr, wb = S["rad"]["w"], S["born"]["w"]
    res = dict(placeholder_cuts=PLACEHOLDER_CUTS, root=a.root, pol=a.pol, bg=a.bg, phi_from=PHI_FROM,
               br=BR_KK, edges={v: EDGES[v].tolist() for v in VARS},
               arms={arm: {k: s[k] for k in ("n_written", "n_chunks", "sigma_nb", "sigma_err", "w")}
                     for arm, s in S.items()},
               vcut={})
    arrays = dict(G_born=G)
    for vc in vcuts:
        Rr = hist(S["rad"]["rec"], selection(S["rad"]["rec"], vc))
        Rb = hist(S["born"]["rec"], selection(S["born"]["rec"], vc))
        with np.errstate(invalid="ignore", divide="ignore"):
            A = wr * Rr / (wb * G)
            eps = Rb / G
            eta = wr * Rr / (wb * Rb)
            dA = A * np.sqrt(1 / Rr + 1 / G)
            deps = np.sqrt(eps * (1 - eps) / G)
            deta = eta * np.sqrt(1 / Rr + 1 / Rb)
        tag = f"v{vc:g}"
        arrays.update({f"R_rad_{tag}": Rr, f"R_born_{tag}": Rb, f"A_{tag}": A, f"dA_{tag}": dA,
                       f"eps_{tag}": eps, f"deps_{tag}": deps, f"eta_{tag}": eta, f"deta_{tag}": deta})
        # the factorisation is an identity, checked rather than assumed
        ok = (Rb > 0) & (Rr > 0) & (G > 0)
        dev = float(np.nanmax(np.abs(eps[ok] * eta[ok] / A[ok] - 1))) if ok.any() else float("nan")
        tot = wr * Rr.sum() / (wb * Rb.sum())
        res["vcut"][tag] = dict(R_rad=int(Rr.sum()), R_born=int(Rb.sum()), eta_integrated=tot,
                                max_dev_factorisation=dev)
        print(f"v < {vc:<6g} R_rad {int(Rr.sum()):8d}  R_born {int(Rb.sum()):8d}  "
              f"eta(all bins) {tot:.4f}  max|eps*eta/A-1| {dev:.1e}")
        if a.ndata:
            N = np.load(a.ndata)
            if N.shape != A.shape:
                raise RuntimeError(f"N_data shape {N.shape} != binning {A.shape}")
            if not a.lumi:
                raise RuntimeError("--lumi is required with --ndata")
            with np.errstate(invalid="ignore", divide="ignore"):
                sig = N / (a.lumi * BR_KK * A)
                dsig = sig * np.sqrt(1 / N + (dA / A) ** 2)
            arrays[f"sigma_born_nb_{tag}"] = sig
            arrays[f"dsigma_born_nb_{tag}"] = dsig

    base = os.path.join(a.out, f"phi_abin_{a.pol}_{a.bg}")
    np.savez(base + ".npz", **arrays)
    with open(base + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", base + ".npz", base + ".json")


if __name__ == "__main__":
    main()
