#!/usr/bin/env python3
"""RC-corrected phi distributions in a wide bin, fitted for sigma_U, sigma_TT, sigma_LT, against the model.

    python3 production/analysis/rc_method_pi0_phifit.py <figdir> [outdir]

Pseudo-data: the reconstructed RAD MC after the note's cuts, EVEN event numbers.
The ODD half builds delta_B, so the B correction is not the data divided by itself.
    sigma(phi) = N_data(phi) w / ( Acc_Born(phi) delta(phi) )
    Acc_Born(phi) = N_Born,rec(phi) / N_Born,gen(phi)   (rec kinematics / gen kinematics, in the bin)
    delta_B(phi)  = N_rad,odd(phi) w_odd / ( N_Born,rec(phi) w_Born )
    delta_A(phi)  = N / sum 1/eta over the ODD rad events in the phi bin (rc_factor, one v_cut)
Truth: the GENERATED Born phi distribution in the bin, fitted the same way.
Fit: dsigma/dphi = sigma_U + eps sigma_TT c2 + sqrt(2 eps (1+eps)) sigma_LT c1, with c_n the
cosine averaged over each phi bin.  eps is the mean over the data events.
phi is the EXCLURAD one, phi_p' + 180, in data, MC and rc_factor alike.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402
import rc_method_pi0 as R  # noqa: E402
from rc_method_pi0_combos import note_cuts  # noqa: E402

FIG = sys.argv[1] if len(sys.argv) > 1 else "docs/rc_method_pi0"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R.DATA, "out")
BINS = {
    "wide1": dict(Q2=(1.5, 2.5), xB=(0.15, 0.30), mt=(0.20, 0.60)),
    "wide2": dict(Q2=(1.0, 2.5), xB=(0.10, 0.30), mt=(0.09, 1.00)),
}
NPHI = 12
VCUTS = [0.06, 0.20]
INK, MUTED = "#222222", "#888888"
COLS = {"truth": INK, "B": "#eb6834", "A0.06": "#6250d6", "A0.2": "#1baf7a"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False})
edges = np.linspace(0, 360, NPHI + 1); pc = 0.5 * (edges[1:] + edges[:-1])
half = np.radians(360 / NPHI) / 2
c1 = np.cos(np.radians(pc)) * np.sin(half) / half
c2 = np.cos(2 * np.radians(pc)) * np.sin(2 * half) / (2 * half)


def in_bin(Q2, xB, mt, b):
    return ((Q2 >= b["Q2"][0]) & (Q2 < b["Q2"][1]) & (xB >= b["xB"][0]) & (xB < b["xB"][1]) &
            (mt >= b["mt"][0]) & (mt < b["mt"][1]))


def eps_of(Q2, xB):
    nu = Q2 / (2 * R.MP * xB); y = nu / R.EB; g2 = Q2 / nu ** 2
    return (1 - y - 0.25 * y * y * g2) / (1 - y + 0.5 * y * y + 0.25 * y * y * g2)


def eta_chunk(rows):
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    m = models.get("pi0.amp2609", t_nucl=-0.3)
    out = np.full((len(rows), len(VCUTS)), np.nan)
    for i, (Q2, xB, mt, ph) in enumerate(rows):
        m.t_nucl = -mt
        try:
            out[i] = [rc_factor(R.EB, Q2, R.MP ** 2 + Q2 * (1 / xB - 1), -mt, np.radians(ph), m, h=+1,
                                mx2_cut=v + R.MPI ** 2)["eta"] for v in VCUTS]
        except Exception:
            pass
    return out


def fit(y, e, eps):
    """Weighted LSQ for (sigma_U, sigma_TT, sigma_LT); returns values, errors, chi2."""
    X = np.column_stack([np.ones(NPHI), eps * c2, np.sqrt(2 * eps * (1 + eps)) * c1])
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
    Wt = 1 / e[ok] ** 2
    Cov = np.linalg.inv(X[ok].T @ (X[ok] * Wt[:, None]))
    p = Cov @ (X[ok].T @ (Wt * y[ok]))
    chi = float(np.sum(Wt * (y[ok] - X[ok] @ p) ** 2))
    return p, np.sqrt(np.diag(Cov)), chi, X @ p


def main():
    rng = np.random.default_rng(20261021)
    sig = {a: C.sigma_nb(f"{R.DATA}/prov_{a}", a)["mean"] for a in ("rad", "born")}
    raw = {}
    for pol in ("inb", "outb"):
        S = {a: R.load(a, pol, rng) for a in sig}
        cal = R.calibrate(S["born"]["rec"])
        raw[pol] = {}
        for a in S:
            k = S[a]["rec"]; g = S[a]["gen"]
            cut = note_cuts(k, cal)
            idx = np.flatnonzero(S[a]["ok"])                 # event number of each reconstructed row
            raw[pol][a] = dict(rec=np.column_stack([k["Q2"], k["xB"], k["mt"], k["phi"]])[cut],
                               evn=idx[cut], gen=np.column_stack([g["Q2"], g["xB"], g["mt"], g["phi"]]),
                               w=sig[a] / S[a]["n_gen"])

    res = {}
    with PdfPages(os.path.join(FIG, "rc_method_pi0_phifit.pdf")) as pdf:
        for bname, b in BINS.items():
            per_pol = {}
            for pol in raw:
                r, bo = raw[pol]["rad"], raw[pol]["born"]
                rb = in_bin(*r["rec"][:, :3].T, b); even = r["evn"] % 2 == 0
                data, odd = r["rec"][rb & even], r["rec"][rb & ~even]
                brec = bo["rec"][in_bin(*bo["rec"][:, :3].T, b)]
                bgen = bo["gen"][in_bin(*bo["gen"][:, :3].T, b)]
                H = lambda x: np.histogram(x[:, 3], bins=edges)[0].astype(float)
                per_pol[pol] = dict(nd=H(data), no=H(odd), nbr=H(brec), nbg=H(bgen), odd=odd, data=data,
                                    w_half=2 * r["w"], wb=bo["w"])
            # delta_A per phi bin from the odd rad events, both polarities together
            odd_all = np.concatenate([per_pol[p]["odd"] for p in per_pol])
            with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
                eta = np.concatenate(pool.map(eta_chunk, np.array_split(odd_all, 120)))
            good = np.all(np.isfinite(eta) & (eta > 0), axis=1)
            ib = np.clip(np.digitize(odd_all[:, 3], edges) - 1, 0, NPHI - 1)
            dA = {v: np.array([good[ib == i].sum() / (1 / eta[good & (ib == i), j]).sum() for i in range(NPHI)])
                  for j, v in enumerate(VCUTS)}
            data_all = np.concatenate([per_pol[p]["data"] for p in per_pol])
            eps = float(np.mean(eps_of(data_all[:, 0], data_all[:, 1])))

            # corrected cross sections per polarity, then combined by inverse variance
            curves = {"B": [], **{f"A{v:g}": [] for v in VCUTS}}
            truth_y, truth_n = 0, 0
            for pol, P in per_pol.items():
                acc = P["nbr"] / P["nbg"]
                dB = (P["no"] * P["w_half"]) / (P["nbr"] * P["wb"])
                ys = 1e3 * P["nd"] * P["w_half"] / acc                 # pb per phi bin, before RC
                rel_d = 1 / P["nd"] + 1 / P["nbr"] - 1 / P["nbg"]      # data + acceptance (binomial-ish)
                curves["B"].append((ys / dB, ys / dB * np.sqrt(rel_d + 1 / P["no"] + 1 / P["nbr"])))
                for v in VCUTS:
                    curves[f"A{v:g}"].append((ys / dA[v], ys / dA[v] * np.sqrt(rel_d)))
                truth_y = truth_y + 1e3 * P["nbg"] * P["wb"]; truth_n = truth_n + P["nbg"]
            comb = {}
            for kname, lst in curves.items():
                y = np.array([l[0] for l in lst]); e = np.array([l[1] for l in lst])
                wv = 1 / e ** 2
                comb[kname] = ((wv * y).sum(0) / wv.sum(0), 1 / np.sqrt(wv.sum(0)))
            # truth: generated Born in the bin; per polarity it is the same sample twice -> average
            t = truth_y / 2; te = t / np.sqrt(truth_n / 2)
            comb = {"truth": (t, te), **comb}

            fits = {k: fit(*v, eps) for k, v in comb.items()}
            res[bname] = dict(bin=b, eps=eps, n_data=int(len(data_all)), n_odd=int(len(odd_all)),
                              eta_ok=float(good.mean()),
                              fits={k: dict(p=f[0].tolist(), e=f[1].tolist(), chi2=f[2]) for k, f in fits.items()},
                              delta_B=None)

            # ---- page: distributions, fits, ratio to truth --------------------------
            fig = plt.figure(figsize=(14, 6.4))
            gs = fig.add_gridspec(2, 2, width_ratios=(1.6, 1), height_ratios=(2, 1.1))
            ax = fig.add_subplot(gs[0, 0]); ar = fig.add_subplot(gs[1, 0], sharex=ax)
            off = {"truth": 0, "B": -4, "A0.06": 0, "A0.2": 4}
            lab = {"truth": "model (generated Born)", "B": "corrected with $\\delta_B$ (rad/Born MC)",
                   "A0.06": "corrected with $\\delta_A$, v_cut = 0.06", "A0.2": "corrected with $\\delta_A$, v_cut = 0.2"}
            for kname, (y, e) in comb.items():
                ph = np.linspace(0, 360, 361)
                p = fits[kname][0]
                curve = p[0] + eps * p[1] * np.cos(2 * np.radians(ph)) + np.sqrt(2 * eps * (1 + eps)) * p[2] * np.cos(np.radians(ph))
                if kname == "truth":
                    ax.plot(ph, curve, color=INK, lw=2, label=lab[kname])
                    ax.errorbar(pc, y, yerr=e, fmt="none", ecolor=INK, lw=0.8)
                else:
                    ax.errorbar(pc + off[kname], y, yerr=e, fmt="o", ms=4, color=COLS[kname], lw=0.8, label=lab[kname])
                    ax.plot(ph, curve, color=COLS[kname], lw=1, ls="--")
                    ar.errorbar(pc + off[kname], 100 * (y / comb["truth"][0] - 1), yerr=100 * e / comb["truth"][0],
                                fmt="o", ms=4, color=COLS[kname], lw=0.8)
            ax.set_ylabel("d$\\sigma$ in the bin (pb per 30$^\\circ$)"); ax.legend(fontsize=7.5)
            ax.set_title(f"{b['Q2'][0]}<Q$^2$<{b['Q2'][1]}, {b['xB'][0]}<x$_B$<{b['xB'][1]}, "
                         f"{b['mt'][0]}<$-t$<{b['mt'][1]};  $\\varepsilon$ = {eps:.3f};  pseudo-data: "
                         f"{len(data_all)} rad events (even half)", fontsize=8.5)
            ar.axhline(0, color=MUTED, lw=0.8); ar.set_ylabel("corrected / model $-$ 1 (%)")
            ar.set_xlabel("$\\phi = \\phi_{p'} + 180^\\circ$ (deg, Trento)"); ar.set_xticks(range(0, 361, 60))
            ar.set_ylim(-30, 30)
            # structure functions
            axs = fig.add_subplot(gs[:, 1]); axs.grid(False)
            names = ["$\\sigma_U$ = $\\sigma_T$ + $\\varepsilon\\sigma_L$", "$\\sigma_{TT}$", "$\\sigma_{LT}$"]
            tp, tpe = fits["truth"][0], fits["truth"][1]
            rows = []
            for i, nm in enumerate(names):
                for j, kname in enumerate(("B", "A0.06", "A0.2")):
                    p, pe = fits[kname][0][i], fits[kname][1][i]
                    y0 = 3 * (2 - i) + (1 - j) * 0.6
                    axs.errorbar((p - tp[i]) / abs(tp[0]) * 100, y0, xerr=pe / abs(tp[0]) * 100, fmt="o",
                                 color=COLS[kname], ms=5)
                    rows.append((nm, kname, p, pe))
                axs.text(-38, 3 * (2 - i) + 1.1, f"{nm}: model {tp[i]:.2f} pb" + ("" if i == 0 else f" = {tp[i] / tp[0]:+.3f} $\\sigma_U$"), fontsize=8)
            axs.axvline(0, color=MUTED, lw=0.8)
            axs.set_xlim(-40, 40); axs.set_ylim(-0.8, 8.5); axs.set_yticks([])
            axs.set_xlabel("(fit $-$ model) / $\\sigma_U$(model)  (%)")
            axs.set_title("Structure functions: shift from the model, in % of $\\sigma_U$", fontsize=8.5)
            for kname in ("B", "A0.06", "A0.2"):
                axs.plot([], [], "o", color=COLS[kname], label=lab[kname].replace("corrected with ", ""))
            axs.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, f"f_{bname}.png"), dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print(f"\n{bname} {b}  eps={eps:.3f}  data={len(data_all)}  odd={len(odd_all)}  eta ok {good.mean():.4f}")
            for kname, f in fits.items():
                p, pe, ch, _ = f
                print(f"  {kname:6} sU {p[0]:8.2f}±{pe[0]:5.2f}  sTT {p[1]:8.2f}±{pe[1]:5.2f}  sLT {p[2]:7.2f}±{pe[2]:5.2f}"
                      f"   TT/U {p[1]/p[0]:+.3f}  LT/U {p[2]/p[0]:+.3f}  chi2 {ch:.1f}/9")
    json.dump(res, open(os.path.join(OUT, "rc_method_pi0_phifit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
