#!/usr/bin/env python3
"""Which exclusivity cut removes the radiation: efficiency against the generated v, one cut at a time.

    python3 production/analysis/rc_method_pi0_vcuts_each.py <figdir>

For a radiative event with an unseen photon k:
    v  = MM2(e'p) - m_pi^2 = 2 p_pi.k = 2 E_k E_pi (1 - beta_pi cos theta_pi,k)
    ME = E_k        MM2(e'p pi0) = k^2 = 0        MP_T = k_T
so ME cuts the photon ENERGY and maps onto v through the photon's angle to the
pi0; MM2(e'p pi0) cuts only through a resolution that grows with E_k.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402
import rc_method_pi0 as R  # noqa: E402

FIG = sys.argv[1] if len(sys.argv) > 1 else "docs/rc_method_pi0"
INK, MUTED = "#222222", "#888888"
COL = {"ME": "#2a78d6", "MM2tot": "#eb6834", "cone": "#1baf7a", "MPT": "#eda100", "all": INK}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False})

rng = np.random.default_rng(20261021)
rows = {k: [] for k in ("v", "Ek", "thk", "base", "ME", "MM2tot", "cone", "MPT", "w")}
sig = C.sigma_nb(f"{R.DATA}/prov_rad", "rad")["mean"]
for pol in ("inb", "outb"):
    S = R.load("rad", pol, rng); k = S["rec"]; ok = S["ok"]
    cal = R.calibrate(R.load("born", pol, rng)["rec"])
    a = np.fromfile(f"{R.DATA}/ntuple/rad_{pol}.f32", "<f4").reshape(-1, R.NF).astype(np.float64)[ok]
    pe, pp, g1, g2 = (a[:, 1 + 3 * j:4 + 3 * j] for j in range(4))
    # the radiated photon, generated: beam + target - e' - p - g1 - g2 (beam energy nominal)
    E = lambda p, m: np.sqrt((p ** 2).sum(1) + m * m)
    Ek = R.EB + R.MP - E(pe, R.ME_) - E(pp, R.MP) - E(g1, 0) - E(g2, 0)
    pk = np.array([0, 0, R.EB]) - pe - pp - g1 - g2
    ppi = g1 + g2
    cth = (pk * ppi).sum(1) / np.maximum(np.linalg.norm(pk, axis=1) * np.linalg.norm(ppi, axis=1), 1e-12)
    vgen = S["gen"]["mm2ep"][ok] - R.MPI ** 2
    c = np.where(k["pdet"] == "FD", cal["FD"]["mgg"][0], cal["CD"]["mgg"][0])
    s = np.where(k["pdet"] == "FD", cal["FD"]["mgg"][1], cal["CD"]["mgg"][1])
    base = (k["Q2"] > 1) & (k["W"] > 2) & (k["pe"] > 2) & (np.abs(k["mgg"] - c) < 3 * s) & (k["open"] > 1)
    for key, val in (("v", vgen), ("Ek", Ek), ("thk", np.degrees(np.arccos(np.clip(cth, -1, 1)))),
                     ("base", base), ("ME", np.abs(k["me"]) < 0.2), ("MM2tot", np.abs(k["mm2tot"]) < 0.004),
                     ("cone", k["cone"] < 4), ("MPT", k["mpt"] < 0.75),
                     ("w", np.full(len(vgen), sig / S["n_gen"]))):
        rows[key].append(val)
d = {k: np.concatenate(v) for k, v in rows.items()}
d["all"] = d["ME"] & d["MM2tot"] & d["cone"] & d["MPT"]
b = d["base"]

bins = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 50)])
cen = np.sqrt(bins[1:] * np.maximum(bins[:-1], 5e-4))
n0 = np.histogram(d["v"][b], bins=bins)[0]
okb = n0 >= 30
res = {}
with PdfPages(os.path.join(FIG, "rc_method_pi0_vcuts_each.pdf")) as pdf:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    for cut in ("MPT", "cone", "MM2tot", "ME", "all"):
        n1 = np.histogram(d["v"][b & d[cut]], bins=bins)[0]
        eff = n1 / np.maximum(n0, 1); soft = eff[okb][cen[okb] < 0.005].mean()
        fe = eff / soft
        i = np.argmax((fe < 0.5) & okb) if np.any((fe < 0.5) & okb) else 0
        vh = float(np.exp(np.interp(0.5, [fe[i], fe[i - 1]], [np.log(cen[i]), np.log(cen[i - 1])]))) if i else np.nan
        res[cut] = dict(v_half=vh, keep_soft=float(soft))
        ax.plot(cen[okb], fe[okb], "o-", ms=3, lw=1.4 if cut == "all" else 1.0, color=COL[cut],
                label=f"{cut}: 50 % at v = {vh:.3f}" if np.isfinite(vh) else f"{cut}: never below 50 %")
    for v, ls in ((0.06, "-"), (0.2, "--")):
        ax.axvline(v, color=MUTED, lw=0.8, ls=ls)
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("generated v (GeV$^2$)"); ax.set_ylabel("efficiency, relative to v $\\to$ 0")
    ax.legend(fontsize=7.5, loc="lower left"); ax.set_title("Each cut alone (after DIS + $\\pi^0$ ID)", fontsize=9)

    ax = axs[1]
    m = b & (d["v"] > 1e-3)
    h = ax.hist2d(np.log10(d["v"][m]), d["Ek"][m], bins=(60, 60), range=((-3, 0.7), (0, 3)), cmap="Greys",
                  cmin=1)
    ax.axhline(0.2, color=COL["ME"], lw=1.2, label="ME = E$_k$ = 0.2 GeV")
    for v, ls in ((0.06, "-"), (0.2, "--")):
        ax.axvline(np.log10(v), color=MUTED, lw=0.8, ls=ls)
    ax.set_xlabel("log$_{10}$ generated v (GeV$^2$)"); ax.set_ylabel("E$_k$, radiated photon energy (GeV)")
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(False)
    ax.set_title("v = 2 E$_k$ E$_\\pi$(1 $-$ $\\beta$cos$\\theta_{\\pi k}$): the band is the photon angle", fontsize=9)

    ax = axs[2]
    m = b & (d["Ek"] > 0.02)
    ax.hist2d(d["thk"][m], np.log10(np.maximum(d["v"][m], 1e-4) / d["Ek"][m]), bins=(60, 60),
              range=((0, 60), (-2.5, 1.0)), cmap="Greys", cmin=1)
    ax.set_xlabel("angle between radiated photon and $\\pi^0$ (deg)")
    ax.set_ylabel("log$_{10}$(v / E$_k$)  (GeV)")
    ax.grid(False); ax.set_title("v per GeV of photon energy against the photon's angle", fontsize=9)
    fig.suptitle("Rad MC, all kinematics, inbending + outbending.  Cuts: |ME|<0.2, |MM$^2$(e'p$\\gamma\\gamma$)|<0.004, "
                 "cone<4$^\\circ$, MP$_T$<0.75", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v3_each_cut.png"), dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

for cut, r in res.items():
    print(f"{cut:7} 50 % point v = {r['v_half']:.3f}   keeps {100 * r['keep_soft']:.1f} % of soft (v->0) events")
# where are the photons: share of radiated energy along the beam vs along e'
m = b & (d["Ek"] > 0.05)
print("photon-pi0 angle, E_k>0.05: median %.1f deg" % np.median(d["thk"][m]))
