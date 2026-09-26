#!/usr/bin/env python3
"""Publication figures and numbers for the J/psi photoproduction Fermi-motion report.

    python3 validation/jpsi_photo_report_figs.py <root with d2/ n/ p/ run trees> <outdir>

Writes PDF and PNG of each figure and numbers.json / numbers.md used by the report.
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
from exclurad_py.generator.photo import (brems_shape, brems_shape_cs, screening_factor,   # noqa: E402
                                         BremsSampler, PhotoGenerator)
from exclurad_py import models                                                            # noqa: E402

M_P, M_N, M_D, M_JPSI = 0.93827208816, 0.93956542, 1.87561294, 3.0969
E_TH = ((M_N + M_JPSI) ** 2 - M_N ** 2) / (2 * M_N)
E0 = 10.6
C = dict(d2="#0072B2", n="#009E73", p="#D55E00", grey="#6c6c6c", acc="#CC79A7")
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 10,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "lines.linewidth": 2.0, "axes.linewidth": 1.0,
                     "figure.dpi": 130, "savefig.bbox": "tight", "pdf.fonttype": 42, "axes.grid": False})


def read(rundir):
    """Photon energy and the J/psi, recoil and spectator four-vectors of one run tree.

    The parsed arrays are cached as float32 in <rundir>/cache_photo.npz (a 20 M-event
    sample is 1.3 GB in memory this way; lists of Python tuples would need ten times more).
    """
    cfgs = sorted(glob.glob(os.path.join(rundir, "inputs", "config*.json")))
    Cj = [json.load(open(c)) for c in cfgs]
    y = float(np.mean([c["report"]["yield_per_electron_per_nucleon_nb"] for c in Cj]))
    ng = float(Cj[0]["report"]["photons_per_electron"])
    sig = float(np.mean([c["report"]["sigma_flux_weighted_nb"] for c in Cj]))
    files = sorted(glob.glob(os.path.join(rundir, "lund", "*.lund")))
    cache = os.path.join(rundir, "cache_photo.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > max(os.path.getmtime(f) for f in files):
        z = np.load(cache); k, V, R, S = z["k"], z["V"], z["R"], z["S"]
    else:
        import array
        k, V, R, S = array.array("f"), array.array("f"), array.array("f"), array.array("f")
        for f in files:
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
                        pp = fh.readline().split()
                        pid, st = int(pp[3]), int(pp[2])
                        q4 = (float(pp[9]), float(pp[6]), float(pp[7]), float(pp[8]))
                        if pid == 443:
                            vv = q4
                        elif pid in (2212, 2112) and st == 1:
                            if rr is None:
                                rr = q4
                            else:
                                ss = q4
                    V.extend(vv); R.extend(rr); S.extend(ss if ss is not None else (0.0, 0.0, 0.0, 0.0))
        k = np.frombuffer(k, dtype=np.float32).copy()
        V, R, S = (np.frombuffer(a, dtype=np.float32).reshape(-1, 4).copy() for a in (V, R, S))
        np.savez(cache, k=k, V=V, R=R, S=S)
    return dict(k=k, V=V.astype(np.float64), R=R.astype(np.float64), S=S.astype(np.float64),
                yield_nb=y, ng=ng, sig=sig, n=len(k), card=Cj[0]["card"])


def mink2(a):
    return a[:, 0] ** 2 - (a[:, 1:] ** 2).sum(1)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=170); plt.close(fig)


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    D = {x: read(glob.glob(os.path.join(root, x, "*", "*_photo_v1"))[0]) for x in ("d2", "n", "p")}
    num = {}
    for x, d in D.items():
        num[x] = dict(n=d["n"], yield_nb=d["yield_nb"], photons_per_e=d["ng"], sigma_flux_weighted_nb=d["sig"],
                      frac_below=float((d["k"] < E_TH).mean()), frac_below_75=float((d["k"] < 7.5).mean()),
                      frac_below_80=float((d["k"] < 8.0).mean()))

    # ---------- Fig 1: photon spectrum (flux) ----------
    X_LD2, X_LH2 = 2.5 / 769.1, 2.5 / 890.4
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    Eg = np.linspace(0.5, E0 - 1e-6, 4000)
    ax[0].plot(Eg, X_LD2 * brems_shape(Eg, E0), color=C["d2"], label=r"5 cm LD$_2$, $X/X_0 = 3.25\times10^{-3}$")
    ax[0].plot(Eg, X_LH2 * brems_shape(Eg, E0), color=C["p"], label=r"5 cm LH$_2$, $X/X_0 = 2.81\times10^{-3}$")
    ax[0].plot(Eg, X_LD2 * brems_shape_cs(Eg, E0), color=C["grey"], ls="--", lw=1.3, label="complete screening (no endpoint)")
    ax[0].axvspan(6.5, E0, color=C["grey"], alpha=0.10); ax[0].axvline(E_TH, color=C["grey"], ls=":", lw=1.2)
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"$E_\gamma$ [GeV]"); ax[0].set_ylabel(r"$dN_\gamma/dE_\gamma$ per incident electron [GeV$^{-1}$]")
    ax[0].legend(loc="lower left"); ax[0].text(6.7, 4.5e-3, "generated range", fontsize=10, color=C["grey"])
    Eg2 = np.linspace(10.3, E0 - 1e-7, 4000)
    ax[1].plot(Eg2, 1e3 * X_LD2 * brems_shape_cs(Eg2, E0), color=C["grey"], ls="--", lw=1.3, label="complete screening")
    ax[1].plot(Eg2, 1e3 * X_LD2 * brems_shape(Eg2, E0), color=C["d2"], label="Schiff, intermediate screening")
    ax[1].set_xlabel(r"$E_\gamma$ [GeV]"); ax[1].set_ylabel(r"$dN_\gamma/dE_\gamma$ [$10^{-3}$ GeV$^{-1}$]"); ax[1].legend(loc="lower left"); ax[1].set_ylim(0, 0.34)
    ax[1].set_title("the last 300 MeV", fontsize=12)
    dk = np.logspace(-2, 3, 600); ax[2].semilogx(dk, screening_factor(E0 - dk * 1e-3, E0), color=C["d2"], label=r"$a_s$ = Bohr radius (H, D)")
    ax[2].semilogx(dk, screening_factor(E0 - dk * 1e-3, E0, 111.0), color=C["acc"], ls="--", lw=1.4, label=r"$a_s$ = Thomas-Fermi, $111\,Z^{-1/3}$")
    ax[2].set_xlabel(r"$E_0 - E_\gamma$ [MeV]"); ax[2].set_ylabel(r"$S(E_\gamma)$ = ratio to complete screening"); ax[2].set_ylim(0, 1.05)
    ax[2].grid(alpha=0.3, which="both"); ax[2].legend(loc="upper left")
    save(fig, out, "fig_flux")
    bs = BremsSampler(E0, 6.5, E0)
    num["flux"] = dict(photons_per_e_LD2_6p5_10p6=bs.norm * X_LD2, photons_per_e_LD2_above_th=float(BremsSampler(E0, E_TH, E0).norm * X_LD2),
                       endpoint_reduction_6p5_10p6=1 - bs.norm / bs.norm_cs,
                       S_at=dict((str(d), float(screening_factor(E0 - d * 1e-3, E0))) for d in (300, 100, 30, 10, 3, 1)))

    # ---------- Fig 2: sigma_tot(E_gamma) of the model and dsigma/dt ----------
    sf = models.get("jpsin.roberts2026")
    g = PhotoGenerator(sf, E0, 6.5, E0, (-6.0, -0.001)); g.setup()
    Egm = np.linspace(E_TH + 1e-4, 10.6, 300)
    W2 = M_N ** 2 + 2 * M_N * Egm
    sig = np.array([g.sigma_tot(w) for w in W2])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(Egm, sig, color=C["n"]); ax[0].set_xlabel(r"$E_\gamma$ [GeV]"); ax[0].set_ylabel(r"$\sigma_{\rm tot}(\gamma N \to J/\psi\, N)$ [nb]")
    ax[0].axvline(E_TH, color=C["grey"], ls=":", lw=1.2); ax[0].text(0.05, 0.9, r"$\sigma = A\,(W-W_{\rm th})^{\alpha}\,e^{-B(W-W_{\rm th})}$", transform=ax[0].transAxes)
    ax[0].text(0.05, 0.82, r"$A = 1.018$ nb, $\alpha = 0.707$, $B = -0.928$ GeV$^{-1}$", transform=ax[0].transAxes, fontsize=10)
    for Egv, col in ((8.5, C["acc"]), (9.5, C["d2"]), (10.5, C["n"])):
        w2 = M_N ** 2 + 2 * M_N * Egv; lo, hi = g.t_limits(w2); t = np.linspace(lo, hi, 300)
        ax[1].plot(-t, g.dsigma_dt(w2, t), color=col, label=r"$E_\gamma$ = %.1f GeV, $|t|_{\min}$ = %.2f GeV$^2$" % (Egv, -hi))
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"$-t$ [GeV$^2$]"); ax[1].set_ylabel(r"$d\sigma/dt$ [nb/GeV$^2$]"); ax[1].legend(); ax[1].set_xlim(0, 4)
    ax[1].text(0.35, 0.9, r"$d\sigma/dt = \sigma_{\rm tot}\, b\, e^{b\,t'}$, $b = 1.67$ GeV$^{-2}$", transform=ax[1].transAxes)
    save(fig, out, "fig_model")

    # ---------- Fig 3: yield vs E_gamma, D vs free n, and precise ratio ----------
    d2, n = D["d2"], D["n"]
    edges = np.arange(6.5, 10.61, 0.1); cen = 0.5 * (edges[1:] + edges[:-1])
    hd, _ = np.histogram(d2["k"], edges); hn, _ = np.histogram(n["k"], edges)
    Yd, Yn = hd * d2["yield_nb"] / d2["n"], hn * n["yield_nb"] / n["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(hn > 0, Yd / Yn, np.nan); er = np.where(hn > 0, r * np.sqrt(1 / np.maximum(hd, 1) + 1 / np.maximum(hn, 1)), np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].step(cen, Yd, where="mid", color=C["d2"], label=r"deuteron, $\gamma n \to J/\psi\, n$ with Fermi motion (AV18)")
    ax[0].step(cen, Yn, where="mid", color=C["n"], label="neutron at rest, same photon flux")
    ax[0].axvline(E_TH, color=C["grey"], ls=":", lw=1.2); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$E_\gamma$ [GeV]"); ax[0].set_ylabel(r"yield per 100 MeV [nb $\times$ photons/e]"); ax[0].legend(loc="upper left")
    ax[0].text(E_TH + 0.05, 2e-8, "free-neutron\nthreshold\n8.20 GeV", fontsize=10, color=C["grey"])
    sel = cen > 8.5
    ax[1].errorbar(cen[sel], r[sel], yerr=er[sel], fmt="o", ms=4, color=C["d2"], capsize=2, label="Fermi on / Fermi off, 100 MeV bins")
    # the same ratio without the Moller flux factor (1 - beta_z) of the struck nucleon: each
    # deuteron event was accepted with weight sigma(s) (1 - beta_z), so 1/(1 - beta_z) removes it
    Es = np.sqrt(M_P ** 2 + (d2["S"][:, 1:] ** 2).sum(1)); EN = M_D - Es; pNz = -d2["S"][:, 3]
    inv_flux = EN / (EN - pNz)
    hd_nf, _ = np.histogram(d2["k"], edges, weights=inv_flux)
    Yd_nf = hd_nf * d2["yield_nb"] / d2["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        r_nf = np.where(hn > 0, Yd_nf / Yn, np.nan)
    ax[1].plot(cen[sel], r_nf[sel], "s", ms=4, mfc="none", color=C["p"], label=r"same, without the flux factor $(1-\beta_z)$: binding only")
    # the ratio above 8.6 GeV: its mean (yield ratio 8.6-10.6), and its extreme bins
    m9 = cen > 8.6
    on86, off86 = hd[m9].sum() * d2["yield_nb"] / d2["n"], hn[m9].sum() * n["yield_nb"] / n["n"]
    c9 = float(on86 / off86); ec9 = float(c9 * np.sqrt(1 / hd[m9].sum() + 1 / hn[m9].sum()))
    imin, imax = int(np.nanargmin(np.where(m9, r, np.nan))), int(np.nanargmax(np.where(m9, r, np.nan)))
    ax[1].axhspan(0.99, 1.01, color=C["grey"], alpha=0.12, lw=0, label=r"$\pm 1\,\%$")
    ax[1].axhline(c9, color=C["acc"], ls="--", lw=1.4, label=r"mean above 8.6 GeV: $%.4f \pm %.4f$" % (c9, ec9))
    ax[1].axhline(1, color=C["grey"], lw=0.8); ax[1].set_ylim(0.90, 1.12); ax[1].set_xlim(8.5, 10.6)
    ax[1].set_xlabel(r"$E_\gamma$ [GeV]"); ax[1].set_ylabel("yield ratio  Fermi on / Fermi off"); ax[1].grid(alpha=0.3); ax[1].legend(loc="upper right")
    save(fig, out, "fig_yield_ratio")
    num["ratio"] = dict(bins=[dict(lo=round(float(edges[i]), 2), hi=round(float(edges[i + 1]), 2), ratio=float(r[i]), err=float(er[i])) for i in range(len(cen)) if cen[i] > 8.2],
                        mean_above_8p6=c9, mean_err=ec9, min_bin=dict(lo=round(float(edges[imin]), 2), hi=round(float(edges[imin + 1]), 2), ratio=float(r[imin]), err=float(er[imin])),
                        max_bin=dict(lo=round(float(edges[imax]), 2), hi=round(float(edges[imax + 1]), 2), ratio=float(r[imax]), err=float(er[imax])),
                        total_ratio=d2["yield_nb"] / n["yield_nb"],
                        noflux_bins=[dict(lo=round(float(edges[i]), 2), hi=round(float(edges[i + 1]), 2), ratio=float(r_nf[i])) for i in range(len(cen)) if cen[i] > 8.2],
                        noflux_total=float(inv_flux.sum() * d2["yield_nb"] / d2["n"] / n["yield_nb"]),
                        mean_flux_factor=float(1 / inv_flux.mean()))
    for lo, hi in ((8.2, 8.4), (8.4, 8.6), (8.6, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.6)):
        md = (d2["k"] >= lo) & (d2["k"] < hi); mn = (n["k"] >= lo) & (n["k"] < hi)
        a, b = md.sum() * d2["yield_nb"] / d2["n"], mn.sum() * n["yield_nb"] / n["n"]
        num["ratio"]["%.1f-%.1f" % (lo, hi)] = dict(on=a, off=b, ratio=a / b, err=a / b * np.sqrt(1 / md.sum() + 1 / mn.sum()),
                                                  noflux_ratio=float(inv_flux[md].sum() * d2["yield_nb"] / d2["n"] / b))

    # ---------- Fig 4: near and below threshold ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    e2 = np.arange(6.5, 9.01, 0.1); c2 = 0.5 * (e2[1:] + e2[:-1])
    h2d, _ = np.histogram(d2["k"], e2); h2n, _ = np.histogram(n["k"], e2)
    ax[0].step(c2, h2d * d2["yield_nb"] / d2["n"], where="mid", color=C["d2"], label="deuteron (AV18)")
    ax[0].step(c2, h2n * n["yield_nb"] / n["n"], where="mid", color=C["n"], label="neutron at rest")
    ax[0].axvline(E_TH, color=C["grey"], ls=":", lw=1.2); ax[0].set_yscale("log"); ax[0].set_xlabel(r"$E_\gamma$ [GeV]")
    ax[0].set_ylabel(r"J/$\psi$ yield per 100 MeV [nb $\times$ photons/e]"); ax[0].legend(loc="upper left")
    below = d2["k"] < E_TH
    num["below"] = dict(frac=float(below.mean()), yield_nb=float(below.sum() * d2["yield_nb"] / d2["n"]),
                        counts={"6.5-7.0": int(((d2["k"] >= 6.5) & (d2["k"] < 7.0)).sum()), "7.0-7.5": int(((d2["k"] >= 7.0) & (d2["k"] < 7.5)).sum()),
                                "7.5-8.0": int(((d2["k"] >= 7.5) & (d2["k"] < 8.0)).sum()), "8.0-8.2": int(((d2["k"] >= 8.0) & (d2["k"] < E_TH)).sum())})
    # fraction of the total yield below E for the deuteron (cumulative)
    ks = np.sort(d2["k"]); frac = np.arange(1, len(ks) + 1) / len(ks)
    ax[1].plot(ks, 100 * frac, color=C["d2"]); ax[1].set_xlim(6.5, 9.0); ax[1].set_ylim(0, 12); ax[1].axvline(E_TH, color=C["grey"], ls=":", lw=1.2)
    ax[1].set_xlabel(r"$E_\gamma$ [GeV]"); ax[1].set_ylabel(r"fraction of the deuteron J/$\psi$ yield below $E_\gamma$ [%]"); ax[1].grid(alpha=0.3)
    ax[1].text(6.6, 10.5, "below the free threshold: %.2f %%\nbelow 8.0 GeV: %.2f %%\nbelow 7.5 GeV: %.2f %%" % (
        100 * below.mean(), 100 * (d2["k"] < 8.0).mean(), 100 * (d2["k"] < 7.5).mean()), fontsize=11)
    save(fig, out, "fig_threshold")

    # ---------- Fig 5: spectator and m* ----------
    ps = np.linalg.norm(d2["S"][:, 1:], axis=1)
    k4 = np.column_stack([d2["k"], 0 * d2["k"], 0 * d2["k"], d2["k"]])
    mstar = np.sqrt(np.maximum(mink2(d2["V"] + d2["R"] - k4), 0))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(wspace=0.45))
    for nm, m, col in (("above threshold", ~below, C["d2"]), ("below threshold", below, C["p"]), (r"$E_\gamma < 7.5$ GeV", d2["k"] < 7.5, C["acc"])):
        ax[0].hist(ps[m], bins=70, range=(0, 0.7), histtype="step", color=col, density=True, label="%s: %d MeV/c" % (nm, 1e3 * ps[m].mean()))
    ax[0].set_xlabel(r"spectator proton momentum $|p_s|$ [GeV/c]"); ax[0].set_ylabel("probability density"); ax[0].legend(title=r"$\langle|p_s|\rangle$", fontsize=9)
    egb = np.arange(6.5, 10.61, 0.1); cb = 0.5 * (egb[1:] + egb[:-1])
    mp = np.array([ps[(d2["k"] >= a) & (d2["k"] < b)].mean() if ((d2["k"] >= a) & (d2["k"] < b)).sum() > 20 else np.nan for a, b in zip(egb[:-1], egb[1:])])
    f3 = np.array([(ps[(d2["k"] >= a) & (d2["k"] < b)] > 0.3).mean() if ((d2["k"] >= a) & (d2["k"] < b)).sum() > 20 else np.nan for a, b in zip(egb[:-1], egb[1:])])
    ax[1].plot(cb, 1e3 * mp, color=C["d2"], label=r"$\langle|p_s|\rangle$ [MeV/c]"); ax[1].set_xlabel(r"$E_\gamma$ [GeV]"); ax[1].set_ylabel(r"$\langle|p_s|\rangle$ [MeV/c]", color=C["d2"])
    ax1b = ax[1].twinx(); ax1b.plot(cb, 100 * f3, color=C["p"]); ax1b.set_ylabel(r"fraction with $|p_s| > 300$ MeV/c [%]", color=C["p"])
    ax[1].axvline(E_TH, color=C["grey"], ls=":", lw=1.2); ax[1].grid(alpha=0.3)
    ax[2].hist(mstar, bins=90, range=(0.1, 0.95), histtype="step", color=C["d2"], log=True, label="all")
    ax[2].hist(mstar[below], bins=90, range=(0.1, 0.95), histtype="step", color=C["p"], log=True, label="below threshold")
    ax[2].axvline(0.10, color=C["acc"], ls="--", lw=1.4, label="floor 0.10 GeV"); ax[2].set_xlabel(r"$m^*$ of the struck neutron [GeV]"); ax[2].set_ylabel("events"); ax[2].legend(fontsize=9)
    save(fig, out, "fig_spectator")
    num["spectator"] = {}
    for nm, m in (("above", ~below), ("below", below), ("below_8.0", d2["k"] < 8.0), ("below_7.5", d2["k"] < 7.5)):
        num["spectator"][nm] = dict(n=int(m.sum()), ps_mean_MeV=float(1e3 * ps[m].mean()), frac_gt300=float((ps[m] > 0.3).mean()),
                                    mstar_mean=float(mstar[m].mean()), frac_mstar_lt_03=float((mstar[m] < 0.3).mean()))

    # ---------- Fig 6: t distributions ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for x, lab in (("d2", "deuteron (AV18)"), ("n", "neutron at rest"), ("p", "proton at rest")):
        d = D[x]; kk = np.column_stack([d["k"], 0 * d["k"], 0 * d["k"], d["k"]]); mt = -mink2(kk - d["V"])
        ax[0].hist(mt, bins=80, range=(0, 4), histtype="step", color=C[x], density=True, label=lab, log=True)
        if x == "d2":
            sel_hi = d["k"] > 9.5
            for nm, m, col in (((r"$E_\gamma > 9.5$ GeV"), sel_hi, C["d2"]), ((r"$E_\gamma < 8.2$ GeV (sub-threshold)"), d["k"] < E_TH, C["p"])):
                ax[1].hist(mt[m], bins=40, range=(0, 4), histtype="step", color=col, density=True, label=nm, log=True)
    ax[0].set_xlabel(r"$-t = -(k - p_{J/\psi})^2$ [GeV$^2$]"); ax[0].set_ylabel("probability density"); ax[0].legend()
    ax[1].set_xlabel(r"$-t$ [GeV$^2$]"); ax[1].set_ylabel("probability density"); ax[1].legend(title="deuteron"); ax[0].set_ylim(1e-3, 3); ax[1].set_ylim(1e-3, 3)
    save(fig, out, "fig_t")

    json.dump(num, open(os.path.join(out, "numbers.json"), "w"), indent=1, default=float)
    print(json.dumps(num, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
