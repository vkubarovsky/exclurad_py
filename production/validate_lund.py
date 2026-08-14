"""Single-sample validation page, in the layout of
~/MC_vector_mesons/<ch>/plot_compare.py `make_plot`.

Same 3x4 grid and the same twelve quantities, read straight from a LUND
file so it works on exclurad_py output and on the reference generators
alike:

  Q2 (with the A/(Q2)^(n/2) fit) | xB | t | y
  W | missing mass Mx | meson pT2 | meson invariant mass
  inelasticity v | radiated photon energy | missing energy | nu

    python3 production/validate_lund.py <file.lund> [-o out.png] [-c channel]

`-c` is only needed when the channel cannot be read off the file name.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Mp = 0.938272
E0 = 10.6
MESON = {"pi0": (111, 0.1349768, r"\pi^0", None),
         "eta": (221, 0.547862, r"\eta", None),
         "rho": (113, 0.77526, r"\rho^0", (211, -211, r"\pi^+\pi^-")),
         "phi": (333, 1.019461, r"\phi", (321, -321, r"K^+K^-")),
         "jpsi": (443, 3.096900, r"J/\psi", (-11, 11, r"e^+e^-"))}


def read(path, channel):
    """-> dict of arrays. Daughters fall back to the meson when undecayed."""
    mpid, mmass, mlab, dec = MESON[channel]
    ev = []
    with open(path) as fh:
        while True:
            l = fh.readline()
            if not l:
                break
            if l.startswith("#"):
                continue
            f = l.split()
            if len(f) < 10:
                continue
            try:
                npart = int(f[0])
            except ValueError:
                continue
            scat = mes = rec = None
            gam = np.zeros(4)
            da = db = None
            n_el = 0
            for i in range(npart):
                q = fh.readline().split()
                if len(q) < 11:
                    continue
                pid, st, par = int(q[3]), int(q[2]), int(q[4])
                px, py, pz, E = map(float, q[6:10])
                p4 = np.array([E, px, py, pz])
                if i == 0 and st == 0 and pid == 11:
                    continue                      # beam documentation line
                if pid == 22 and par == 0:
                    gam = p4                      # radiated photon
                elif pid == mpid:
                    mes = p4
                elif pid == 2212:
                    rec = p4
                elif pid == 11 and n_el == 0 and (dec is None or par == 0):
                    scat = p4; n_el += 1
                elif dec is not None and pid == dec[0] and da is None:
                    da = p4
                elif dec is not None and pid == dec[1] and db is None:
                    db = p4
                elif pid == 11 and dec is not None and dec[1] == 11:
                    db = p4                       # J/psi decay electron
            if scat is None or rec is None:
                continue
            if da is None or db is None:          # meson written undecayed
                da, db = (mes, np.zeros(4)) if mes is not None else (None, None)
                if da is None:
                    continue
            ev.append((scat, rec, gam, da, db))

    scat = np.array([e[0] for e in ev]); rec = np.array([e[1] for e in ev])
    gam = np.array([e[2] for e in ev])
    da = np.array([e[3] for e in ev]); db = np.array([e[4] for e in ev])

    nu = E0 - scat[:, 0]
    Q2 = 2 * E0 * scat[:, 0] - 2 * E0 * scat[:, 3]
    W2 = Mp ** 2 + 2 * Mp * nu - Q2
    xB = Q2 / np.maximum(2 * Mp * nu, 1e-12)
    y = nu / E0
    t = 2 * Mp ** 2 - 2 * Mp * rec[:, 0]

    def m2(p):
        return p[:, 0] ** 2 - np.einsum("ij,ij->i", p[:, 1:], p[:, 1:])

    mfour = da + db
    Mrec = np.sqrt(np.maximum(0, m2(mfour)))
    pT2 = mfour[:, 1] ** 2 + mfour[:, 2] ** 2
    Mmiss = np.sqrt(np.maximum(0, m2(rec + gam)))
    E_miss = nu + Mp - da[:, 0] - db[:, 0] - rec[:, 0] - gam[:, 0]
    has_ph = gam[:, 0] > 1e-9
    v = Mmiss[has_ph] ** 2 - Mp ** 2
    return dict(Q2=Q2, xB=xB, t=t, y=y, W=np.sqrt(np.maximum(W2, 0)),
                Mmiss=Mmiss, pT2=pT2, Mrec=Mrec, nu=nu, E_miss=E_miss,
                Egam=gam[has_ph, 0], v=v, has_ph=has_ph, N=len(ev),
                mmass=mmass, mlab=mlab, dlab=dec[2] if dec else mlab)


def annotate(ax, data, unit="", fmt=".4f", loc="right"):
    data = np.asarray(data)[np.isfinite(data)]
    n, mean, sig = len(data), np.mean(data) if len(data) else 0, \
        np.std(data) if len(data) else 0
    w = max(len(f"{mean:{fmt}}"), len(f"{sig:{fmt}}"), len(str(n)))
    txt = "\n".join([f"N     = {n:>{w}}", f"Mean = {mean:>{w}{fmt}}{unit}",
                     f"Sigma = {sig:>{w}{fmt}}{unit}"])
    x, ha = (0.97, "right") if loc == "right" else (0.03, "left")
    ax.text(x, 0.95, txt, transform=ax.transAxes, ha=ha, va="top",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                      ec="gray", alpha=0.9))


def fit_q2(ax, edges, counts, rng=(1.2, 4.5)):
    c = 0.5 * (edges[:-1] + edges[1:])
    m = (c >= rng[0]) & (c <= rng[1]) & (counts > 0)
    if m.sum() < 4:
        return
    q, n_ = c[m], counts[m].astype(float)
    try:
        popt, pcov = curve_fit(lambda x, A, n: A * x ** (-n / 2), q, n_,
                               sigma=np.maximum(np.sqrt(n_), 1.0),
                               p0=[n_[0] * q[0] ** 1.5, 3.0],
                               bounds=([0, 0.2], [1e9, 12.0]))
        e = np.sqrt(np.diag(pcov))
        xl = np.linspace(*rng, 300)
        ax.plot(xl, popt[0] * xl ** (-popt[1] / 2), "-", color="red", lw=2,
                label=fr"Fit $A/(Q^2)^{{n/2}}$: $n={popt[1]:.2f}\pm{e[1]:.2f}$"
                      f"\n  range [{rng[0]}, {rng[1]}] GeV²")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.97, 0.75),
                  bbox_transform=ax.transAxes)
    except Exception:
        pass


def page(d, title, outfile, fc="#F4A582", ec="black"):
    bkw = dict(histtype="stepfilled", color=fc, edgecolor=ec)
    fig, ax = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    c, e, _ = ax[0, 0].hist(d["Q2"], bins=50, **bkw)
    ax[0, 0].set(xlabel=r"$Q^2$ (GeV$^2$)", ylabel="Events",
                 title=r"$Q^2$ distribution", yscale="log")
    annotate(ax[0, 0], d["Q2"], " GeV²"); fit_q2(ax[0, 0], e, c)

    ax[0, 1].hist(d["xB"], bins=50, **bkw)
    ax[0, 1].set(xlabel=r"$x_B$", ylabel="Events",
                 title=r"$x_B$ distribution", yscale="log")
    annotate(ax[0, 1], d["xB"])

    mt = -d["t"]
    ax[0, 2].hist(mt, bins=60, range=(np.floor(mt.min() * 20) / 20,
                                      np.ceil(mt.max() * 20) / 20), **bkw)
    ax[0, 2].set(xlabel=r"$-t$ (GeV$^2$)", ylabel="Events",
                 title=r"$t$ distribution (exponential)", yscale="log")
    annotate(ax[0, 2], mt, " GeV²")

    ax[0, 3].hist(d["y"], bins=50, range=(0, 1), **bkw)
    ax[0, 3].set(xlabel=r"$y$", ylabel="Events", title=r"$y$ distribution")
    annotate(ax[0, 3], d["y"])

    ax[1, 0].hist(d["W"], bins=50, **bkw)
    ax[1, 0].set(xlabel=r"$W$ (GeV)", ylabel="Events", title=r"$W$ distribution")
    annotate(ax[1, 0], d["W"], " GeV")

    ax[1, 1].hist(d["Mmiss"], bins=60, range=(0.85, 1.20), **bkw)
    ax[1, 1].axvline(Mp, color="red", lw=1.5, ls="--",
                     label=f"$M_p$={Mp:.4f} GeV")
    ax[1, 1].set(xlabel=r"$M_x$ (GeV)", ylabel="Events",
                 title=r"Missing mass $M_x$", yscale="log")
    ax[1, 1].legend(fontsize=9); annotate(ax[1, 1], d["Mmiss"], " GeV")

    ax[1, 2].hist(d["pT2"], bins=50, **bkw)
    ax[1, 2].set(xlabel=r"$p_T^2$ (GeV$^2/c^2$)", ylabel="Events",
                 title=rf"${d['mlab']}$  $p_T^2$ (lab)", yscale="log")
    annotate(ax[1, 2], d["pT2"], " GeV²")

    m0 = d["mmass"]
    ax[1, 3].hist(d["Mrec"], bins=60,
                  range=(m0 - 0.05 * max(m0, 1), m0 + 0.05 * max(m0, 1)), **bkw)
    ax[1, 3].axvline(m0, color="red", lw=1.5, ls="--",
                     label=f"${d['mlab']}$={m0:.4f} GeV")
    ax[1, 3].set(xlabel=rf"$M_{{{d['dlab']}}}$ (GeV)", ylabel="Events",
                 title=rf"${d['mlab']}$ invariant mass")
    ax[1, 3].legend(fontsize=9); annotate(ax[1, 3], d["Mrec"], " GeV")

    if len(d["v"]):
        ax[2, 0].hist(d["v"], bins=40, **bkw)
        ax[2, 0].set(xlabel=r"$v = M_X^2 - M_p^2$ (GeV$^2$)",
                     ylabel="Hard events",
                     title=r"Inelasticity $v$ distribution", yscale="log")
        annotate(ax[2, 0], d["v"], " GeV²")
        ax[2, 1].hist(d["Egam"], bins=50, **bkw)
        ax[2, 1].set(xlabel=r"$E_\gamma$ (GeV)", ylabel="Hard events",
                     title=r"Radiated photon energy", yscale="log")
        annotate(ax[2, 1], d["Egam"], " GeV")
    else:
        for a, tt in ((ax[2, 0], "Inelasticity v"),
                      (ax[2, 1], "Radiated photon energy")):
            a.text(0.5, 0.5, "No hard events\n(Born only)", ha="center",
                   va="center", transform=a.transAxes, fontsize=12)
            a.set_title(tt)

    ax[2, 2].hist(d["E_miss"], bins=80, range=(-0.07, 0.07), **bkw)
    ax[2, 2].set(xlabel=r"$E_{\rm miss} = E_{\rm init}-\sum E_{\rm final}$ (GeV)",
                 ylabel="Events",
                 title=r"Missing energy (should be $\delta(0)$)", yscale="log")
    annotate(ax[2, 2], d["E_miss"], " GeV")

    ax[2, 3].hist(d["nu"], bins=50, **bkw)
    ax[2, 3].set(xlabel=r"$\nu$ (GeV)", ylabel="Events",
                 title=r"Energy transfer $\nu$")
    annotate(ax[2, 3], d["nu"], " GeV")

    for a in ax.flat:
        a.tick_params(direction="in", which="both")
        a.grid(True, alpha=0.4, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    a = sys.argv[1:]
    path = a[0]
    out = a[a.index("-o") + 1] if "-o" in a else path.rsplit(".", 1)[0] + "_validation.png"
    ch = a[a.index("-c") + 1] if "-c" in a else next(
        (k for k in MESON if k in os.path.basename(path)), None)
    if ch is None:
        sys.exit("cannot infer channel from the file name; pass -c <channel>")
    d = read(path, ch)
    nh = int(d["has_ph"].sum())
    page(d, f"{os.path.basename(path)}  |  ${MESON[ch][2]}$  |  {d['N']} events  |  "
            f"Hard: {nh} ({100*nh/max(d['N'],1):.1f}%)  Soft: {d['N']-nh}", out)
