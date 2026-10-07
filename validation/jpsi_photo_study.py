#!/usr/bin/env python3
"""Checks and results for the untagged bremsstrahlung photoproduction mode (generator/photo.py).

    python3 validation/jpsi_photo_study.py <d2 run dir> <free-p run dir> [<free-n run dir>] [outdir]

Checks (printed, and the script exits 1 if one fails):
  1. four-momentum: photon + target = V + N' (+ N_s) per event, from the LUND lab vectors;
  2. the free-nucleon E_gamma spectrum follows n_gamma(k) * sigma_tot(W(k)) -- the analytic
     bremsstrahlung shape times the model's t-integrated cross section -- within statistics;
  3. the t' distribution of the free-nucleon sample at fixed E_gamma has the model's slope.
Results (figures + jpsi_photo_study.md): yield per incident electron and per target nucleon
versus E_gamma for deuteron and free nucleons, the sub-threshold yield, the t distributions,
the spectator momentum and m* of the deuteron sample.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exclurad_py import models                                             # noqa: E402
from exclurad_py.generator.photo import PhotoGenerator, brems_shape        # noqa: E402

M_P, M_N, M_D, M_JPSI = 0.93827208816, 0.93956542, 1.87561294, 3.0969
E_TH = ((M_N + M_JPSI) ** 2 - M_N ** 2) / (2 * M_N)
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#777777"


def read_run(rundir):
    cfgs = sorted(glob.glob(os.path.join(rundir, "inputs", "config*.json")))
    C = [json.load(open(c)) for c in cfgs]
    card, rep = C[0]["card"], C[0]["report"]
    yield_nb = float(np.mean([c["report"]["yield_per_electron_per_nucleon_nb"] for c in C]))
    k, V, R, S, vz = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(rundir, "lund", "*.lund"))):
        with open(f) as fh:
            while True:
                line = fh.readline()
                if not line:
                    break
                if line.startswith("#"):
                    continue
                w = line.split()
                if len(w) < 10:
                    continue
                npart = int(w[0]); k.append(float(w[6]))
                vv = rr = ss = None
                for _ in range(npart):
                    p = fh.readline().split()
                    pid, st = int(p[3]), int(p[2])
                    q4 = tuple(map(float, (p[9], p[6], p[7], p[8])))
                    if pid == 443:
                        vv = q4; vz.append(float(p[13]))
                    elif pid in (2212, 2112) and st == 1:
                        if rr is None:
                            rr = q4
                        else:
                            ss = q4
                V.append(vv); R.append(rr); S.append(ss if ss is not None else (0, 0, 0, 0))
    k, V, R, S, vz = (np.array(x, float) for x in (k, V, R, S, vz))
    return dict(k=k, V=V, R=R, S=S, vz=vz, card=card, rep=rep, yield_nb=yield_nb, n=len(k), rundir=rundir)


def mink2(a):
    return a[:, 0] ** 2 - (a[:, 1:] ** 2).sum(1)


def main():
    args = sys.argv[1:]
    d2, p = read_run(args[0]), read_run(args[1])
    n = read_run(args[2]) if len(args) > 2 and os.path.isdir(args[2]) else None
    outdir = args[-1] if not os.path.isdir(args[-1]) or len(args) == 4 else "plots/jpsi_photo"
    if len(args) == 3 and os.path.isdir(args[2]):
        outdir = "plots/jpsi_photo"
    os.makedirs(outdir, exist_ok=True)
    failures, lines = [], ["# Untagged bremsstrahlung J/psi photoproduction: checks and results", ""]
    runs = [("deuteron (AV18), gamma n", d2, BLUE), ("free proton", p, ORANGE)] + ([("free neutron", n, GREEN)] if n else [])

    # ---- 1. conservation
    for lab, d, _ in runs:
        fermi = d["card"]["fermi"] != "none"
        Mt = M_D if fermi else (M_N if d["card"]["model"].startswith("jpsin") else M_P)
        k4 = np.column_stack([d["k"], 0 * d["k"], 0 * d["k"], d["k"]])
        init = k4 + np.array([Mt, 0, 0, 0])
        fin = d["V"] + d["R"] + (d["S"] if fermi else 0)
        res = np.abs(init - fin).max()
        lines.append("- %s: %d events, max |P_in - P_out| = %.1e GeV%s" % (lab, d["n"], res, "" if res < 1e-6 else "  FAIL"))
        if res > 1e-6:
            failures.append("conservation %s: %.1e" % (lab, res))
    lines.append("")

    # ---- 2. E_gamma spectrum of the free proton against brems * sigma_tot
    sf = models.get(p["card"]["model"])
    g = PhotoGenerator(sf, p["card"]["ebeam"], p["card"]["kmin"], p["card"]["kmax"], (p["card"]["tmin"], p["card"]["tmax"]))
    g.setup()
    edges = np.linspace(p["card"]["kmin"], p["card"]["kmax"], 42)
    cen = 0.5 * (edges[1:] + edges[:-1])
    h, _ = np.histogram(p["k"], edges)
    model = np.array([brems_shape(c, p["card"]["ebeam"]) * g.sigma_tot_fast(M_P ** 2 + 2 * M_P * c) for c in cen])
    model *= h.sum() / model.sum()
    m = h > 20
    chi2 = float((((h - model) ** 2) / np.maximum(model, 1))[m].sum()); ndf = int(m.sum() - 1)
    lines.append("## Free-proton E_gamma spectrum against n_gamma(k) sigma_tot(W(k)): chi2/ndf = %.1f/%d%s" % (chi2, ndf, "" if chi2 < 2.5 * ndf else "  FAIL"))
    if chi2 > 2.5 * ndf:
        failures.append("E_gamma spectrum chi2/ndf = %.1f/%d" % (chi2, ndf))

    # ---- 3. t' slope of the free proton in a narrow E_gamma window
    sel = (p["k"] > 9.4) & (p["k"] < 9.6)
    W2 = M_P ** 2 + 2 * M_P * p["k"][sel]
    t = mink2(np.column_stack([p["k"][sel], 0 * W2, 0 * W2, p["k"][sel]]) - p["V"][sel])
    lo, hi = g.t_limits(float(W2.mean()))
    tp = t - hi                                  # t' = t - t_min (hi is t_min, the least negative)
    tp = tp[(tp < 0) & (tp > -1.5)]
    hh, ee = np.histogram(tp, bins=15, range=(-1.5, 0))
    cc = 0.5 * (ee[1:] + ee[:-1]); ok = hh > 10
    slope = np.polyfit(cc[ok], np.log(hh[ok]), 1)[0]
    lines.append("## t' slope of the free proton at E_gamma 9.4-9.6 GeV: b = %.2f GeV^-2 (model %.2f)%s" % (slope, sf.b, "" if abs(slope - sf.b) < 0.15 else "  FAIL"))
    if abs(slope - sf.b) > 0.15:
        failures.append("t slope %.2f vs %.2f" % (slope, sf.b))
    lines.append("")

    # ---- results: yields
    lines.append("## Yield per incident electron and per target nucleon, 5 cm cell (nb x photons/electron), by E_gamma")
    lines.append("")
    lines.append("| E_gamma bin | " + " | ".join(lab for lab, _, _ in runs) + " |")
    lines.append("|---|" + "---:|" * len(runs))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    hists = {}
    for lab, d, col in runs:
        hk, _ = np.histogram(d["k"], edges)
        hists[lab] = hk * d["yield_nb"] / d["n"]
        ax[0].step(cen, hists[lab], where="mid", color=col, lw=1.6, label=lab)
    for lo_, hi_ in ((6.5, 7.5), (7.5, 8.0), (8.0, 8.2), (8.2, 9.0), (9.0, 10.6)):
        vals = []
        for lab, d, col in runs:
            mm = (d["k"] >= lo_) & (d["k"] < hi_)
            vals.append("%.3g (%d ev)" % (mm.sum() * d["yield_nb"] / d["n"], mm.sum()))
        lines.append("| %.1f-%.1f | " % (lo_, hi_) + " | ".join(vals) + " |")
    for lab, d, col in runs:
        lines.append("- %s: total yield %.4g nb x photons/e, photons per electron in the k range %.4g, flux-weighted <sigma> %.4g nb; below the free threshold: %.2f %% of events" % (
            lab, d["yield_nb"], d["rep"]["photons_per_electron"], d["rep"]["sigma_flux_weighted_nb"], 100 * (d["k"] < E_TH).mean()))
    ax[0].axvline(E_TH, color=GREY, ls="--", lw=1); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$E_\gamma$ [GeV]"); ax[0].set_ylabel("yield per bin [nb $\\times$ photons/e, per nucleon]"); ax[0].legend(fontsize=8)
    ax[0].set_title("bremsstrahlung in the 5 cm cell, untagged", fontsize=10)
    ref = hists["free proton"]
    for lab, d, col in runs:
        if lab == "free proton":
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ax[1].step(cen, np.where(ref > 0, hists[lab] / ref, np.nan), where="mid", color=col, lw=1.6, label="%s / free proton" % lab)
    ax[1].axvline(E_TH, color=GREY, ls="--", lw=1); ax[1].axhline(1, color=GREY, lw=0.6); ax[1].set_ylim(0, 2)
    ax[1].set_xlabel(r"$E_\gamma$ [GeV]"); ax[1].set_ylabel("ratio"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "jpsi_photo_egamma.png"), dpi=130); plt.close(fig)

    # ---- t distributions and spectator
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for lab, d, col in runs:
        k4 = np.column_stack([d["k"], 0 * d["k"], 0 * d["k"], d["k"]])
        mt = -mink2(k4 - d["V"])
        ax[0].hist(mt, bins=60, range=(0, 4), histtype="step", color=col, lw=1.5, density=True, label=lab, log=True)
    ax[0].set_xlabel(r"$-t = -(k - p_{J/\psi})^2$ [GeV$^2$]"); ax[0].legend(fontsize=8)
    ps = np.linalg.norm(d2["S"][:, 1:], axis=1); below = d2["k"] < E_TH
    ax[1].hist(ps[~below], bins=60, range=(0, 0.7), histtype="step", color=BLUE, lw=1.5, density=True, label="above threshold")
    if below.sum() > 10:
        ax[1].hist(ps[below], bins=60, range=(0, 0.7), histtype="step", color=ORANGE, lw=1.5, density=True, label="below threshold")
    ax[1].set_xlabel(r"spectator $|p_s|$ [GeV/c]"); ax[1].legend(fontsize=8); ax[1].set_title("deuteron", fontsize=10)
    struck = d2["V"] + d2["R"] - np.column_stack([d2["k"], 0 * d2["k"], 0 * d2["k"], d2["k"]])
    mstar = np.sqrt(np.maximum(mink2(struck), 0))
    ax[2].hist(mstar, bins=70, range=(0, 1.0), histtype="step", color=BLUE, lw=1.5, log=True)
    ax[2].axvline(0.10, color=ORANGE, ls="--", lw=1, label="floor 0.10 GeV"); ax[2].set_xlabel(r"$m^*$ [GeV]"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "jpsi_photo_t_spectator.png"), dpi=130); plt.close(fig)
    lines.append("")
    lines.append("## Deuteron spectator")
    lines.append("")
    for nm, mm in (("above threshold", ~below), ("below threshold", below), ("E_gamma < 8.0", d2["k"] < 8.0), ("E_gamma < 7.5", d2["k"] < 7.5)):
        if mm.any():
            lines.append("- %s (%d events): <|p_s|> = %.0f MeV/c, above 300 MeV/c %.1f %%, <m*> %.3f GeV" % (nm, mm.sum(), 1e3 * ps[mm].mean(), 100 * (ps[mm] > 0.3).mean(), mstar[mm].mean()))
    lines.append("")
    lines.append("Vertex z of the deuteron sample: mean %.2f cm (cell %s; density grows linearly along the cell as the upstream radiator does)." % (d2["vz"].mean(), d2["card"]["vz"]))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(outdir, "jpsi_photo_study.md"), "w") as f:
        f.write(txt + "\n")
    print("figures in", outdir)
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures)); sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
