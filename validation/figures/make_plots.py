"""Validation figures: exclurad_py vs the published CLAS6 RC tables."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette, fixed order (never cycled)
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})

P = "/Users/vpk/exclurad_py/validation/"
pi0 = np.genfromtxt(P+"pi0_published/pi0_published_direct_cm_angle.csv",
                    delimiter=",", names=True)
eta = np.genfromtxt(P+"eta_published_direct/eta_published_sweep_cm_angle.csv",
                    delimiter=",", names=True)
pi0_t = np.genfromtxt(P+"pi0_published/pi0_published_direct.csv",
                      delimiter=",", names=True)

fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.4))

# --- A: delta vs phi, first bin, pi0 -------------------------------------- #
a = ax[0, 0]
s = slice(0, 20)
a.plot(pi0["phi_deg"][s], pi0["delta_2013"][s], "o", ms=6, color=GREY,
       label="published (PRC 90 025205)", zorder=3)
a.plot(pi0["phi_deg"][s], pi0["eta_mesonrc"][s], "-", lw=2, color=BLUE,
       label="exclurad_py", zorder=2)
a.set_xlabel(r"$\phi$  [deg]"); a.set_ylabel(r"RC factor  $\delta$")
a.set_title(r"(a)  $\pi^0$, first bin: $Q^2$=1.17, $x_B$=0.134, $t$=$-$0.12",
            loc="left", fontsize=9)
a.legend(frameon=False, fontsize=8)

# --- B: delta vs phi, first bin, eta --------------------------------------- #
b = ax[0, 1]
b.plot(eta["phi_deg"][s], eta["delta_published"][s], "o", ms=6, color=GREY,
       label="published (PRC 95 035202)", zorder=3)
b.plot(eta["phi_deg"][s], eta["eta_mesonrc"][s], "-", lw=2, color=VERM,
       label="exclurad_py", zorder=2)
b.set_xlabel(r"$\phi$  [deg]"); b.set_ylabel(r"RC factor  $\delta$")
b.set_title(r"(b)  $\eta$, first bin: $Q^2$=1.16, $x_B$=0.133, $t$=$-$0.12",
            loc="left", fontsize=9)
b.legend(frameon=False, fontsize=8)

# --- C: mean residual vs phi over the FULL grids --------------------------- #
c = ax[1, 0]
edges = np.linspace(0, 180, 10)
mid = 0.5*(edges[:-1]+edges[1:])
for d, col, lab in ((pi0, BLUE, r"$\pi^0$  (3104 bins)"),
                    (eta, VERM, r"$\eta$  (1814 bins)")):
    fold = np.minimum(d["phi_deg"], 360-d["phi_deg"])
    m = np.isfinite(d["diff"])
    mu = [d["diff"][m & (fold >= lo) & (fold < hi)].mean()
          for lo, hi in zip(edges[:-1], edges[1:])]
    sd = [d["diff"][m & (fold >= lo) & (fold < hi)].std()
          for lo, hi in zip(edges[:-1], edges[1:])]
    c.errorbar(mid, mu, yerr=sd, fmt="-o", ms=5, lw=2, color=col, capsize=2,
               label=lab)
c.axhline(0, color=GREY, lw=1, ls="--")
c.set_xlabel(r"$\phi$ folded about 180$^\circ$  [deg]")
c.set_ylabel(r"$\delta_{\rm mesonrc}-\delta_{\rm published}$")
c.set_title("(c)  residual vs $\\phi$, full grids (bars = rms across bins)",
            loc="left", fontsize=9)
c.legend(frameon=False, fontsize=8)

# --- D: the truncation effect, pi0 ----------------------------------------- #
d_ = ax[1, 1]
mt = pi0_t["v_max"] < 0.17-1e-6
d_.plot(pi0_t["v_max"][mt], pi0_t["diff"][mt], "o", ms=3.5, color=GREEN,
        alpha=0.55, mew=0, label=r"$t$ fixed (analysis binning)",
        zorder=2)
d_.plot(pi0_t["v_max"][mt], pi0["diff"][mt], "o", ms=4.5, color=BLUE, alpha=0.9,
        mew=0.4, mec="white",
        label=r'$\cos\theta_{\rm CM}$ frozen (EXCLURAD)', zorder=3)
d_.set_ylim(-0.145, 0.022)
d_.axhline(0, color=GREY, lw=1, ls="--")
d_.set_xlabel(r"$v_{\max}(t)$  [GeV$^2$]   (cut $v<0.17$)")
d_.set_ylabel(r"$\delta_{\rm mesonrc}-\delta_{\rm published}$")
d_.set_title(r"(d)  $\pi^0$: the 302 bins where $v_{\max}(t)<v_{\rm cut}$",
             loc="left", fontsize=9)
d_.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig("/Users/vpk/exclurad_py/validation/figures/validation.png",
            bbox_inches="tight")
print("wrote validation.png")
