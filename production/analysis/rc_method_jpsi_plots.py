#!/usr/bin/env python3
"""Figures and summary.json for the J/psi RC-method study.

    python3 production/analysis/rc_method_jpsi_plots.py <run outdir> <figdir>

Reads rc_method_jpsi_combos.json (and rc_method_jpsi_closure.json if present)
and writes every number the report quotes into <figdir>/summary.json, so the
text and the figures come from one place.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

RUN, FIG = sys.argv[1], sys.argv[2]
os.makedirs(FIG, exist_ok=True)
r = json.load(open(os.path.join(RUN, "rc_method_jpsi_combos.json")))
CL = os.path.join(RUN, "rc_method_jpsi_closure.json")
closure = json.load(open(CL)) if os.path.exists(CL) else None

INK, MUTED = "#222222", "#888888"
COL = {"inb": "#1f5fbf", "outb": "#d0582a", "A": "#1baf7a", "Av": "#6250d6", "flat": "#999999"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "legend.frameon": False})
VC = r["vcuts"]
VN = np.array([v for v in VC if v is not None])          # numeric cuts; the last entry is "no cut"
NAME = {"W": "W (GeV)", "Q2": "Q$^2$ (GeV$^2$)", "mt": "|t| (GeV$^2$)"}
TOPO_TXT = {"3trk": "e'e$^+$e$^-$ (proton missing)", "4trk": "e'e$^+$e$^-$p' (all detected)"}
POLS = ("inb", "outb")
CUTS_TXT = ("M. Tenorio's selection: FT e' ($\\theta_e$ 2.45-6.45$^\\circ$), $\\nu$>8.1, lepton PID, |p|>1.95, "
            "vz on the inward lepton, 2.95<M(ee)<3.20;  3 tracks: $\\Delta p_z$>-2.5, MX within 4$\\sigma$;  "
            "4 tracks: proton, -0.7<$\\Delta p_z$<0.5, |MX$^2$|<0.036")


def flat_A(topo, rng):
    return r["flat"][topo][json.dumps(rng, sort_keys=True)]["A"]


def A_at(A, v):
    """A at v_cut v (GeV^2), linear in v between the scanned cuts; v=None means no cut."""
    A = np.asarray(A, float)
    if v is None:
        return float(A[-1])
    return float(np.interp(v, VN, A[:len(VN)]))


def cells(topo, pol, binning):
    return [c for c in r["cells"][f"{topo}_{pol}"] if c["binning"] == binning and "B" in c]


def chi2_curve(cl):
    B = np.array([c["B"] for c in cl]); E = np.array([c["B_err"] for c in cl])
    return np.array([np.sum(((B - np.array([A_at(c["A"], v) for c in cl])) / E) ** 2) for v in VN])


def best_v(cl):
    ch = chi2_curve(cl)
    fine = np.linspace(VN[0], VN[-1], 400)
    return float(fine[np.argmin(np.interp(fine, VN, ch))]), float(ch.min())


def v_her(topo, pol):
    return r["sample"][f"{topo}_{pol}"].get("v_her")


summary = dict(F_fsr=r["F_fsr"], vcuts=VC, sample=r["sample"], topo={}, mariana={}, closure=None)

# one reference cut per topology and polarity: her MX window for 3 tracks (where
# rc_factor has the same variable), the chi2-best v over the maps for 4 tracks
VREF = {}
for topo in r["model"]:
    for pol in POLS:
        mapcells = sum((cells(topo, pol, f"map_Q2_{a}_{b}") for a, b in r["qslices"]), [])
        vb, chmin = best_v(mapcells)
        vh = v_her(topo, pol)
        VREF[(topo, pol)] = dict(v_best=vb, chi2_best=chmin, n_cells=len(mapcells), v_her=vh,
                                 v_ref=vh)
        summary["topo"][f"{topo}_{pol}"] = VREF[(topo, pol)]


def vtxt(v):
    return "no cut" if v is None else f"{v:.3f}"


# THE EFFECTIVE CUTS.  Her chain does not cut where its nominal edges are: the
# 50 % points of its efficiency against the GENERATED decay-photon energy k* and
# the generated production v (rc_method_jpsi_vdist.py) sit below them, because
# the M(ee) and MX resolutions act on the radiative tails.  Method 2 with those
# two numbers -- still rc_factor, still no Born MC -- is A_eff.  The pre-vertex
# radiator's hard part, which eta_ext drops, enters as the campaign's measured
# itarg=1/itarg=0 Born sigma ratio.
HARD = 0.345396 / 0.349740
EFF = {}
_vd = os.path.join(FIG, "vdist.json")
if os.path.exists(_vd):
    from exclurad_py.generator.decay_rad import PhotonEnergy
    _pe = PhotonEnergy(3.0969)
    _d = json.load(open(_vd))
    for _t in _d:
        _x50 = {}
        for _var, _plat in (("kstar", 0.03), ("v", 0.05)):
            _e = np.array(_d[_t][_var]["edges"]); _xc = np.sqrt(np.abs(_e[1:] * _e[:-1])); _xc[0] = _e[1] / 2
            _n = np.array(_d[_t][_var]["n"]); _eff = np.array(list(_d[_t][_var]["eff"].values())[-1])
            _ok = (_xc < _plat) & (_xc > 9e-4) & (_n > 200)
            _rr = _eff / np.average(_eff[_ok], weights=_n[_ok])
            _i = np.where((_xc > _plat) & (_rr < 0.5))[0][0]
            _x50[_var] = float(np.exp(np.interp(0.5, [_rr[_i], _rr[_i - 1]], [np.log(_xc[_i]), np.log(_xc[_i - 1])])))
        EFF[_t] = dict(K50=_x50["kstar"], v50=_x50["v"], F50=float(_pe.cdf(_x50["kstar"])))
summary["effective"] = dict(EFF, hard=HARD)


def A_eff(c, topo):
    if topo not in EFF:
        return float("nan")
    e = EFF[topo]
    return A_at(c["A_int"], e["v50"]) * A_at(c["ext"], e["v50"]) * HARD * e["F50"]


def save(fig, name, pdf):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


with PdfPages(os.path.join(FIG, "rc_method_jpsi.pdf")) as pdf:
    # ---- F1: 1-D, per topology ------------------------------------------------
    for topo in r["model"]:
        fig, axs = plt.subplots(2, 3, figsize=(14, 6.6), sharex="col", gridspec_kw=dict(height_ratios=(2, 1.2)))
        for j, var in enumerate(("W", "Q2", "mt")):
            ax, ar = axs[0, j], axs[1, j]
            for k, pol in enumerate(POLS):
                cl = cells(topo, pol, var)
                x = np.array([c["mean"][var] for c in cl])
                xe = np.array([[c["mean"][var] - c["range"][var][0], c["range"][var][1] - c["mean"][var]] for c in cl]).T
                B = np.array([c["B"] for c in cl]); E = np.array([c["B_err"] for c in cl])
                vr = VREF[(topo, pol)]["v_ref"]
                A = np.array([A_at(c["A"], vr) for c in cl])
                ax.errorbar(x, B, xerr=xe, yerr=E, fmt="o", ms=4, color=COL[pol], lw=0.8,
                            label=f"method 1 (MC), F18 {pol}")
                ax.plot(x, A, "-", color=COL[pol], lw=1.2, alpha=0.6,
                        label=f"method 2, nominal cuts (v {vtxt(vr)})" if j == 0 else None)
                Ae = np.array([A_eff(c, topo) for c in cl])
                ax.plot(x, Ae, "--", color=COL[pol], lw=1.4,
                        label="method 2, effective cuts" if j == 0 else None)
                ar.errorbar(x * (1 + (k - 0.5) * 0.004 + 0.002), 100 * (B / Ae - 1), yerr=100 * E / Ae,
                            fmt="o", mfc="white", ms=3.5, color=COL[pol], lw=0.6)
                fl = np.array([A_at(flat_A(topo, c["range"]), vr) * c["ext"][-1] * r["F_fsr"] for c in cl])
                if k == 0:
                    ax.plot(x, fl, ":", color=COL["flat"], lw=1.4, label="method 2, flat (no acceptance)" if j == 0 else None)
                ar.errorbar(x * (1 + (k - 0.5) * 0.004), 100 * (B / A - 1), yerr=100 * E / A,
                            fmt="o", ms=3.5, color=COL[pol], lw=0.8)
            ax.set_ylim(0.6, 1.0)
            ax.set_ylabel("C$_{rad}$" if j == 0 else "")
            ar.axhline(0, color=MUTED, lw=0.8); ar.set_ylim(-15, 15)
            ar.set_xlabel(NAME[var] + ", the other two integrated")
            ar.set_ylabel("method 1 / method 2 $-$ 1 (%)" if j == 0 else "")
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(f"{TOPO_TXT[topo]}.  " + CUTS_TXT, fontsize=7.5)
        fig.tight_layout()
        save(fig, f"f1_1d_{topo}", pdf)

    # ---- F2: (W, |t|) maps at three Q2 -----------------------------------------
    norm = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12); cmap = plt.get_cmap("RdBu_r")
    for topo in r["model"]:
        fig, axs = plt.subplots(len(POLS) + 1, 3, figsize=(20, 14))
        for row, pol in enumerate(POLS + ("A",)):
            for col, (qa, qb) in enumerate(r["qslices"]):
                ax = axs[row, col]
                if pol == "A":
                    cl = cells(topo, "inb", f"map_Q2_{qa}_{qb}")
                else:
                    cl = cells(topo, pol, f"map_Q2_{qa}_{qb}")
                for c in cl:
                    x0, x1 = c["range"]["W"]; y0, y1 = c["range"]["mt"]
                    if pol == "A":
                        v = A_eff(c, topo); txt = f"{v:.3f}"
                        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=plt.cm.viridis((v - 0.6) / 0.4),
                                               ec="white", lw=1.2))
                        tc = "white"
                    else:
                        A = A_eff(c, topo); d = 100 * (c["B"] / A - 1)
                        de = 100 * c["B_err"] / A
                        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=cmap(norm(d)), ec="white", lw=1.2))
                        txt = f"{c['B']:.3f}±{c['B_err']:.3f}\n{d:+.1f}±{de:.1f}%"
                        tc = "white" if abs(d) > 8 else INK
                    ax.text((x0 + x1) / 2, (y0 + y1) / 2, txt, ha="center", va="center", fontsize=5.5, color=tc)
                ax.set_xlim(r["map"]["W"][0], r["map"]["W"][-1]); ax.set_ylim(r["map"]["mt"][0], r["map"]["mt"][-1])
                ax.grid(False)
                ax.set_xlabel("W (GeV)"); ax.set_ylabel("|t| (GeV$^2$)")
                ttl = (f"method 2, effective cuts, acceptance-weighted (inb)" if pol == "A" else
                       f"F18 {pol}: method 1 and (1/2 $-$ 1)")
                ax.set_title(f"Q$^2$ {qa}-{qb}:  {ttl}", fontsize=8.5)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axs[:2, :], label="method 1 / method 2 (effective cuts) $-$ 1 (%)", shrink=0.8)
        fig.suptitle(f"C$_{{rad}}$(W, |t|), {TOPO_TXT[topo]}.  Upper rows: method-1 value $\\pm$ stat, "
                     f"and its deviation from method 2.  " + CUTS_TXT, fontsize=7.5)
        save(fig, f"f2_maps_{topo}", pdf)

    # ---- F3: her three W bins ---------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    gen = r["mariana"]["gen10M"]; wb = r["mariana"]["W"]
    xc = np.array([(wb[i] + wb[i + 1]) / 2 for i in range(3)])
    for ax, topo in zip(axs, r["model"]):
        ax.errorbar(xc - 0.012, [g[0] for g in gen], yerr=[g[1] for g in gen], fmt="D", ms=5, color=MUTED,
                    label="generated 10M, no cuts (Launch generator)")
        ax.axhline(0.75, color=INK, ls="--", lw=0.9, label="M. Tenorio: 0.88 x 0.85")
        rows = []
        for k, pol in enumerate(POLS):
            cl = cells(topo, pol, "mariana")
            vr = VREF[(topo, pol)]["v_ref"]
            B = np.array([c["B"] for c in cl]); E = np.array([c["B_err"] for c in cl])
            A = np.array([A_at(c["A"], vr) for c in cl])
            fl = np.array([A_at(flat_A(topo, c["range"]), vr) * c["ext"][-1] * r["F_fsr"] for c in cl])
            ax.errorbar(xc + (k - 0.5) * 0.012, B, yerr=E, fmt="o", ms=5, color=COL[pol], label=f"method 1, {pol}")
            ax.plot(xc + (k - 0.5) * 0.012, A, "_", ms=16, mew=1.2, color=COL[pol], label=f"method 2 nominal cuts, {pol}")
            Ae = np.array([A_eff(c, topo) for c in cl])
            ax.plot(xc + (k - 0.5) * 0.012, Ae, "_", ms=16, mew=2.6, color=COL[pol], alpha=0.55,
                    label=f"method 2 effective cuts, {pol}")
            if k == 0:
                ax.plot(xc + 0.024, fl, "x", ms=7, color=COL["flat"], label="method 2 flat")
            for c, b, e, a, f, ae in zip(cl, B, E, A, fl, Ae):
                rows.append(dict(pol=pol, W=c["range"]["W"], n_rad=c["n_rad"], n_born=c["n_born"], B=b, B_err=e,
                                 B_gen=c["B_gen"], A=a, A_flat=f, v_ref=vr, v_eq=c["v_eq"],
                                 A_int=A_at(c["A_int"], vr), ext=A_at(c["ext"], vr), A_eff=ae))
        summary["mariana"][topo] = rows
        for i in range(4):
            ax.axvline(wb[i], color="#dddddd", lw=0.8)
        ax.set_xlabel("W (GeV), Q$^2$ 0.02-0.12"); ax.set_title(TOPO_TXT[topo], fontsize=9)
        ax.set_ylim(0.6, 1.0)
    axs[0].set_ylabel("C$_{rad}$"); axs[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(CUTS_TXT, fontsize=7.5)
    fig.tight_layout()
    save(fig, "f3_mariana", pdf)

    # ---- F4: decomposition, 1-D W ----------------------------------------------
    # Every term in % of method 2 at the nominal cuts, additive to first order:
    #   hard       pre-vertex radiator, energy shift, not in eta_ext   (HARD - 1)
    #   FSR edge   decay photon cut at K50, not at the M(ee) edge       (F50/F - 1)
    #   prod v     production radiation cut at v50, not at v_her        (A_int ext at v50 / at v_ref - 1)
    #   migration  rec bins against gen bins, the same selected events  (B - B_gen)/A
    #   residual   what is left: the detector's response to radiation beyond
    #              those two edges -- lepton PID and |p| on FSR leptons, mainly
    # (c) acceptance weighting is shown but is NOT a term of the sum: method 2 is
    # already acceptance-weighted per event, A - A_flat is what a flat average
    # would have got wrong.
    for topo in r["model"]:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        dec = {}
        for ax, pol in zip(axs, POLS):
            cl = cells(topo, pol, "W")
            x = np.array([c["mean"]["W"] for c in cl])
            vr = VREF[(topo, pol)]["v_ref"]
            B = np.array([c["B"] for c in cl]); Bg = np.array([c["B_gen"] for c in cl]); E = np.array([c["B_err"] for c in cl])
            A = np.array([A_at(c["A"], vr) for c in cl])
            fl = np.array([A_at(flat_A(topo, c["range"]), vr) * c["ext"][-1] * r["F_fsr"] for c in cl])
            e = EFF[topo]
            terms = dict(
                hard=np.full(len(cl), 100 * (HARD - 1)),
                fsr_edge=np.full(len(cl), 100 * (e["F50"] / r["F_fsr"] - 1)),
                prod_v=np.array([100 * (A_at(c["A_int"], e["v50"]) * A_at(c["ext"], e["v50"]) /
                                        (A_at(c["A_int"], vr) * A_at(c["ext"], vr)) - 1) for c in cl]),
                migration=100 * (B - Bg) / A)
            total = 100 * (B / A - 1)
            terms["residual"] = total - sum(terms.values())
            acc = 100 * (A - fl) / A
            ax.errorbar(x, total, yerr=100 * E / A, fmt="o", color=INK, ms=4.5, label="total: method 1 / method 2 (nominal) $-$ 1")
            sty = dict(hard=("#999999", "pre-vertex radiator, hard part"),
                       fsr_edge=(COL["outb"], "decay FSR: effective edge K$_{50}$"),
                       prod_v=(COL["Av"], "production: effective v$_{50}$"),
                       migration=(COL["inb"], "migration: rec $-$ gen bins"),
                       residual=("#b8860b", "residual: detector response (PID, |p|)"))
            for kname, (cc, lab) in sty.items():
                ax.plot(x, terms[kname], "-", marker="s", ms=3, lw=1, color=cc, label=lab)
            ax.plot(x, acc, ":", color=COL["A"], lw=1.2, label="(info) acceptance weighting A $-$ A$_{flat}$")
            ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylim(-12, 12)
            ax.set_xlabel("W (GeV)"); ax.set_title(f"F18 {pol}, nominal v {vtxt(vr)}", fontsize=9)
            dec[pol] = [dict(W=c["range"]["W"], total=float(t), total_err=float(100 * ee / a), acceptance=float(ac),
                             **{kk: float(vv[i]) for kk, vv in terms.items()})
                        for i, (c, t, ee, a, ac) in enumerate(zip(cl, total, E, A, acc))]
        axs[0].set_ylabel("% of C$_{rad}$"); axs[0].legend(fontsize=6.5, loc="lower left")
        fig.suptitle(f"Decomposition of method 1 $-$ method 2, {TOPO_TXT[topo]}", fontsize=9)
        fig.tight_layout()
        save(fig, f"f4_decomp_{topo}", pdf)
        summary["topo"][f"{topo}_decomp_W"] = dec

    # ---- F5: polarity -----------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, topo in zip(axs, r["model"]):
        pulls, diffs, labs = [], [], []
        for b in ["W", "Q2", "mt"] + [f"map_Q2_{a}_{bb}" for a, bb in r["qslices"]] + ["mariana"]:
            ci = {json.dumps(c["range"], sort_keys=True): c for c in cells(topo, "inb", b)}
            co = {json.dumps(c["range"], sort_keys=True): c for c in cells(topo, "outb", b)}
            for k in ci:
                if k in co:
                    d = ci[k]["B"] - co[k]["B"]; e = np.hypot(ci[k]["B_err"], co[k]["B_err"])
                    pulls.append(d / e); diffs.append(100 * d / co[k]["B"]); labs.append(b)
        pulls = np.array(pulls)
        ax.hist(pulls, bins=np.linspace(-4, 4, 33), color=COL["inb"], alpha=0.7)
        xx = np.linspace(-4, 4, 200)
        ax.plot(xx, len(pulls) * 0.25 * np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1)
        chi2 = float(np.sum(pulls ** 2))
        ax.set_title(f"{TOPO_TXT[topo]}: (inb $-$ outb)/$\\sigma$, all cells;  "
                     f"$\\chi^2$ = {chi2:.0f} / {len(pulls)}, mean {pulls.mean():+.2f}", fontsize=8.5)
        ax.set_xlabel("pull"); ax.set_ylabel("cells")
        summary["topo"][f"{topo}_polarity"] = dict(n=len(pulls), chi2=chi2, mean_pull=float(pulls.mean()),
                                                   rms_pull=float(pulls.std()),
                                                   mean_diff_pct=float(np.mean(diffs)), rms_diff_pct=float(np.std(diffs)))
    fig.suptitle("Method 2 has no torus in it: any inbending/outbending difference of method 1 is acceptance", fontsize=9)
    fig.tight_layout()
    save(fig, "f5_polarity", pdf)

    # ---- F6: chi2 against one v, and the cell-by-cell v_eq -------------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for topo, ls in zip(r["model"], ("-", "--")):
        for pol in POLS:
            for b in ("W", "Q2", "mt", "maps"):
                cl = (sum((cells(topo, pol, f"map_Q2_{a}_{bb}") for a, bb in r["qslices"]), [])
                      if b == "maps" else cells(topo, pol, b))
                if b != "maps":
                    continue
                axs[0].plot(VN, chi2_curve(cl) / len(cl), ls, marker="o", ms=3, color=COL[pol],
                            label=f"{topo} {pol} maps, {len(cl)} cells")
                veq = np.array([c["v_eq"] if c["v_eq"] is not None else np.nan for c in cl])
                axs[1].hist(veq[np.isfinite(veq)], bins=np.linspace(0, 1.0, 26), histtype="step", ls=ls,
                            color=COL[pol], lw=1.2, label=f"{topo} {pol}: {np.isfinite(veq).sum()} of {len(cl)}")
    axs[0].axhline(1, color=MUTED, ls=":", lw=0.8); axs[0].set_yscale("log")
    axs[0].set_xlabel("one v for every cell (GeV$^2$)"); axs[0].set_ylabel("$\\chi^2$ / cells"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel("v at which method 2 equals method 1, cell by cell (GeV$^2$)"); axs[1].set_ylabel("cells")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "f6_chi2", pdf)

    # ---- F7: generator-level closure ---------------------------------------------
    if closure:
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
        for j, var in enumerate(("W", "Q2", "mt")):
            for lab, col in (("t_proton", COL["inb"]), ("t_meson", COL["outb"])):
                rows = closure["variants"][lab]["bins"][var]
                x = [(q["lo"] + q["hi"]) / 2 for q in rows]
                axs[j].errorbar(x, [q["dev"] for q in rows],
                                yerr=[100 * np.hypot(q["B_err"] / q["B"], q["A_err"] / q["A"]) for q in rows],
                                fmt="o", ms=4, color=col, label=f"{lab}, {closure['variants'][lab]['model']}")
            axs[j].axhline(0, color=MUTED, lw=0.8); axs[j].set_xlabel(NAME[var] + ", generated")
        axs[0].set_ylabel("B$_{gen}$ / A $-$ 1 (%), no cut"); axs[0].legend(fontsize=7)
        fig.suptitle("Generator level, no detector, no cut: what rc_factor leaves out (pre-vertex radiator; "
                     "decay FSR when t is built from e$^+$e$^-$)", fontsize=8.5)
        fig.tight_layout()
        save(fig, "f7_closure", pdf)
        summary["closure"] = {k: dict(integrated=v["integrated"], eta_ok=v["eta_ok"]) for k, v in closure["variants"].items()}

# method 1 against method 2 with the effective cuts, every cell of every binning
summary["chi2_eff"] = {}
for topo in r["model"]:
    for pol in POLS:
        for name in ("W", "Q2", "mt", "maps", "mariana"):
            cl = (sum((cells(topo, pol, f"map_Q2_{a}_{b}") for a, b in r["qslices"]), [])
                  if name == "maps" else cells(topo, pol, name))
            pul = np.array([(c["B"] - A_eff(c, topo)) / c["B_err"] for c in cl])
            dev = np.array([100 * (c["B"] / A_eff(c, topo) - 1) for c in cl])
            summary["chi2_eff"][f"{topo}_{pol}_{name}"] = dict(chi2=float((pul ** 2).sum()), n=len(cl),
                                                               mean_dev_pct=float(dev.mean()),
                                                               wmean_dev_pct=float(np.average(dev, weights=1 / np.array([c["B_err"] for c in cl]) ** 2)))
# the side studies, written by their own scripts into FIG, carried into the one summary
for name in ("cutvars", "gencuts", "vdist"):
    p = os.path.join(FIG, name + ".json")
    if os.path.exists(p) and name != "vdist":
        summary[name] = json.load(open(p))
# the pre-vertex radiator's hard part, which eta_ext leaves out: the campaign's
# own itarg=1 Born run against itarg=0 (NOTES.txt), and the closure at no cut
summary["itarg_hard"] = dict(sigma_born_itarg1=0.345396, sigma_born_itarg0=0.349740,
                             ratio=0.345396 / 0.349740)
json.dump(summary, open(os.path.join(FIG, "summary.json"), "w"), indent=1, default=float)
print(json.dumps(dict(topo={k: v for k, v in summary["topo"].items() if "decomp" not in k}), indent=1, default=float))
for topo, rows in summary["mariana"].items():
    for q in rows:
        print(topo, q["pol"], q["W"], "B %.4f+-%.4f  Bgen %.4f  A %.4f  A_eff %.4f  Aflat %.4f  v_eq %s" %
              (q["B"], q["B_err"], q["B_gen"], q["A"], q["A_eff"], q["A_flat"], q["v_eq"]))
