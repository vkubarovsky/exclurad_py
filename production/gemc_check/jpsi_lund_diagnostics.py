"""Generator-level diagnostics of a J/psi production: Born and rad superimposed.

    python3 jpsi_lund_diagnostics.py <production dir> <output dir> [jobs]

<production dir> holds born/ and rad/, each with lund_osg/*.lund (gen_array
layout) and prov/*.config.json (one per chunk: card + run report).

Every 1D histogram is d(sigma) per bin in nb: each event carries
sigma_written/N_written of its own chunk, where sigma_written = sigma_nb *
written / accepted (sigma_nb is counted before the hadron/lepton acceptance
cut; accept2 for the exact rad path).  Born and rad are drawn on the same
axes with a rad/Born ratio panel.  2D maps are drawn side by side.

Particles, from LUND pid and parent (1-based): e' = 11 with parent 0,
p = 2212, J/psi = 443 (status 0), e+/e- = -11/11 with the J/psi as parent,
production photon = 22 with parent 0, decay photon = 22 with the J/psi as
parent.  Kinematics with the nominal beam (header column 7) and a proton
target at rest; t_N = (p - p')^2 is the proton t of the production cards
(jpsi.diffrad_vm_pdet), t_meson = (q - p_psi)^2 is shown for comparison.
"""
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np

MP, ME, MJ = 0.93827208816, 0.51099895e-3, 3.0969
G = np.array([1.0, -1.0, -1.0, -1.0])
FIELDS = ["Q2", "W", "xB", "y", "nu", "tN", "tmeson", "tprime", "phi",
          "pe", "the", "phe", "pp", "thp", "ppsi", "thpsi",
          "pep", "thep", "pem", "them", "Mee",
          "MX2_ep", "MX2_epsi", "MX2_all", "Emiss",
          "Eg_prod", "thg_prod", "Eg_dec", "ang_dec", "vz", "hel", "npart"]


def mass2(a):
    return a[0] * a[0] - a[1:] @ a[1:]


def pt(a):
    p = np.linalg.norm(a[1:])
    return p, np.degrees(np.arccos(np.clip(a[3] / p, -1, 1))) if p > 0 else 0.0


def tmin(W2, Q2, m=MJ, M=MP):
    W = np.sqrt(W2)
    pi = np.sqrt(np.maximum(((W2 + Q2 + M * M) / (2 * W)) ** 2 - M * M, 0))
    ev = (W2 + m * m - M * M) / (2 * W)
    pf = np.sqrt(np.maximum(ev * ev - m * m, 0))
    return (pi - pf) ** 2 - ((Q2 + m * m) / (2 * W)) ** 2


def trento_phi(k, kp, q, h):
    """Angle between the lepton plane (k, k') and the hadron plane (q, h)."""
    nl = np.cross(k[1:], kp[1:])
    nh = np.cross(q[1:], h[1:])
    c = nl @ nh / (np.linalg.norm(nl) * np.linalg.norm(nh) + 1e-30)
    s = np.sign(np.cross(nl, nh) @ q[1:]) or 1.0
    ph = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return ph if s > 0 else 360.0 - ph


def one_file(path):
    rows = []
    with open(path) as fh:
        lines = [l.split() for l in fh if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        h = lines[i]
        n = int(h[0])
        P = lines[i + 1:i + 1 + n]
        i += 1 + n
        E = float(h[6])
        hel = float(h[4])
        v4 = lambda p: np.array([float(p[9]), float(p[6]), float(p[7]), float(p[8])])  # noqa: E731
        psirow = next((j + 1 for j, p in enumerate(P) if p[3] == "443"), -1)
        e = pp = psi = ep = em = gp = gd = None
        vz = 0.0
        for p in P:
            pid, par = p[3], int(p[4])
            if pid == "11" and par == 0 and e is None:
                e = v4(p)
                vz = float(p[13]) if len(p) > 13 else 0.0
            elif pid == "2212":
                pp = v4(p)
            elif pid == "443":
                psi = v4(p)
            elif pid == "-11" and par == psirow:
                ep = v4(p)
            elif pid == "11" and par == psirow:
                em = v4(p)
            elif pid == "22" and par == 0:
                gp = v4(p)
            elif pid == "22" and par == psirow:
                gd = v4(p)
        if e is None or pp is None or psi is None or ep is None or em is None:
            continue
        k = np.array([E, 0.0, 0.0, np.sqrt(E * E - ME * ME)])
        tgt = np.array([MP, 0.0, 0.0, 0.0])
        q = k - e
        Q2 = -mass2(q)
        nu = q[0]
        W2 = MP * MP + 2 * MP * nu - Q2
        tN = mass2(tgt - pp)
        tme = mass2(q - psi)
        tp = -tN - tmin(W2, Q2)
        pe, the = pt(e)
        pP, thP = pt(pp)
        pJ, thJ = pt(psi)
        pEp, thEp = pt(ep)
        pEm, thEm = pt(em)
        Mee = np.sqrt(max(mass2(ep + em), 0))
        mx_ep = mass2(k + tgt - e - pp)
        mx_epsi = mass2(k + tgt - e - psi)
        miss = k + tgt - e - pp - ep - em
        mx_all = mass2(miss)
        if gp is not None:
            egp, thgp = gp[0], pt(gp)[1]
        else:
            egp, thgp = np.nan, np.nan
        if gd is not None:
            egd = gd[0]
            # angle to the nearer decay lepton
            cang = [np.degrees(np.arccos(np.clip(gd[1:] @ l[1:] / (np.linalg.norm(gd[1:]) * np.linalg.norm(l[1:])), -1, 1))) for l in (ep, em)]
            angd = min(cang)
        else:
            egd, angd = np.nan, np.nan
        rows.append((Q2, np.sqrt(max(W2, 0)), Q2 / (2 * MP * nu), nu / E, nu, -tN, -tme, tp,
                     trento_phi(k, e, q, psi), pe, the, np.degrees(np.arctan2(e[2], e[1])),
                     pP, thP, pJ, thJ, pEp, thEp, pEm, thEm, Mee,
                     mx_ep, mx_epsi, mx_all, miss[0], egp, thgp, egd, angd, vz, hel, n))
    return np.array(rows, dtype=float)


def chunk_sigma(prov_json, n_written):
    d = json.load(open(prov_json))
    r = d["report"]
    acc = r.get("accept2", 0) or r["accepted"]
    return r["sigma_nb"] * n_written / acc


def load_arm(proddir, arm, jobs):
    lunds = sorted(glob.glob(os.path.join(proddir, arm, "lund_osg", "*.lund")))
    with mp.Pool(jobs) as pool:
        arrs = pool.map(one_file, lunds, chunksize=1)
    provs = sorted(glob.glob(os.path.join(proddir, arm, "prov", "*.config.json")))
    if len(provs) != len(lunds):
        print(f"WARNING {arm}: {len(lunds)} lund files but {len(provs)} provenance files")
    sig = np.array([chunk_sigma(p, 5000) for p in provs])
    # the chunks estimate the same sigma: the sample's cross section is their
    # mean, and each event carries sigma / N_total
    data = np.concatenate([a for a in arrs if len(a)])
    w = np.full(len(data), sig.mean() / len(data))
    return data, w, sig


VARS = [  # name, label, bins, range, log-y
    ("Q2", r"$Q^2$ (GeV$^2$)", 60, (0, 0.25), True),
    ("W", r"$W$ (GeV)", 60, (4.0, 4.6), False),
    ("xB", r"$x_B$", 60, (0, 0.016), True),
    ("y", r"$y$", 60, (0.75, 1.0), False),
    ("nu", r"$\nu$ (GeV)", 60, (7.9, 10.6), False),
    ("tN", r"$-t$ from proton, $-(p-p')^2$ (GeV$^2$)", 60, (0, 6), True),
    ("tmeson", r"$-t$ from J/$\psi$, $-(q-p_\psi)^2$ (GeV$^2$)", 60, (0, 6), True),
    ("tprime", r"$t' = |t_N| - |t|_{min}$ (GeV$^2$)", 60, (-0.2, 5), True),
    ("phi", r"$\phi$ Trento (deg)", 36, (0, 360), False),
    ("pe", r"$p_{e'}$ (GeV)", 60, (0, 2.6), True),
    ("the", r"$\theta_{e'}$ (deg)", 60, (1.5, 9.5), False),
    ("phe", r"$\phi_{e'}$ (deg)", 36, (-180, 180), False),
    ("pp", r"$p_{p'}$ (GeV)", 60, (0, 5), False),
    ("thp", r"$\theta_{p'}$ (deg)", 60, (0, 40), False),
    ("ppsi", r"$p_{J/\psi}$ (GeV)", 60, (4, 11), False),
    ("thpsi", r"$\theta_{J/\psi}$ (deg)", 60, (0, 30), False),
    ("pep", r"$p_{e^+}$ (GeV)", 60, (0, 10), False),
    ("thep", r"$\theta_{e^+}$ (deg)", 60, (0, 45), False),
    ("pem", r"$p_{e^-}$ decay (GeV)", 60, (0, 10), False),
    ("them", r"$\theta_{e^-}$ decay (deg)", 60, (0, 45), False),
    ("Mee", r"$M(e^+e^-)$ (GeV)", 80, (2.0, 3.2), True),
    ("MX2_ep", r"$M_X^2(e'p')$ (GeV$^2$)", 80, (7, 12), True),
    ("MX2_epsi", r"$M_X^2(e'J/\psi)$ (GeV$^2$)", 80, (0, 3), True),
    ("MX2_all", r"$M_X^2(e'p'e^+e^-)$ (GeV$^2$)", 80, (-0.2, 0.8), True),
    ("Emiss", r"$E_{miss}(e'p'e^+e^-)$ (GeV)", 80, (-0.1, 2.0), True),
    ("Eg_prod", r"production photon $E_\gamma$ (GeV)", 80, (0, 4), True),
    ("thg_prod", r"production photon $\theta_\gamma$ (deg)", 60, (0, 30), True),
    ("Eg_dec", r"decay photon $E_\gamma$ (GeV)", 80, (0, 4), True),
    ("ang_dec", r"decay photon to nearer lepton (deg)", 60, (0, 15), True),
    ("vz", r"vertex $z$ (cm)", 60, (-6, 0), False),
    ("hel", "beam helicity (LUND col 5)", 3, (-1.5, 1.5), False),
    ("npart", "particles per event", 6, (4.5, 10.5), False),
]
MAPS = [("W", "Q2", (4.0, 4.6), (0, 0.25)), ("pe", "the", (0, 2.6), (1.5, 9.5)),
        ("pp", "thp", (0, 5), (0, 40)), ("pep", "thep", (0, 10), (0, 45)),
        ("W", "tN", (4.0, 4.6), (0, 6))]


def plot(born, rad, outdir, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    (db, wb, sb), (dr, wr, sr) = born, rad
    idx = {f: i for i, f in enumerate(FIELDS)}
    os.makedirs(os.path.join(outdir, "png"), exist_ok=True)
    pdf = PdfPages(os.path.join(outdir, "jpsi_born_rad_diagnostics.pdf"))
    # title page
    fig = plt.figure(figsize=(8.5, 6))
    fig.text(0.05, 0.92, "J/psi production f18in 2026-10-09: generator-level Born vs rad", fontsize=13, weight="bold")
    txt = (f"Born: {len(db)} events, sigma = {sb.mean():.5e} nb (chunk rms {sb.std():.1e})\n"
           f"Rad : {len(dr)} events, sigma = {sr.mean():.5e} nb (chunk rms {sr.std():.1e})\n"
           f"sigma_rad / sigma_Born = {sr.mean()/sb.mean():.4f}\n\n" + meta +
           "\n\nHistograms: d(sigma) per bin in nb (each event weighted sigma/N);"
           "\nratio panel: rad / Born.  Photons: only events that carry one.")
    fig.text(0.05, 0.85, txt, fontsize=9, family="monospace", va="top")
    pdf.savefig(fig); plt.close(fig)
    for name, lab, nb, rng, logy in VARS:
        xb, xr = db[:, idx[name]], dr[:, idx[name]]
        mb, mr = np.isfinite(xb), np.isfinite(xr)
        edges = np.linspace(*rng, nb + 1)
        hb, _ = np.histogram(xb[mb], edges, weights=wb[mb])
        hr, _ = np.histogram(xr[mr], edges, weights=wr[mr])
        nb_, _ = np.histogram(xb[mb], edges)
        nr_, _ = np.histogram(xr[mr], edges)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
        c = 0.5 * (edges[1:] + edges[:-1])
        a1.step(edges[:-1], hb, where="post", color="C0", label=f"Born ({mb.sum()} ev)")
        a1.step(edges[:-1], hr, where="post", color="C3", label=f"rad ({mr.sum()} ev)")
        a1.set_ylabel(r"$d\sigma$ per bin (nb)")
        if logy:
            a1.set_yscale("log")
        out_b = (~((xb[mb] >= rng[0]) & (xb[mb] <= rng[1]))).mean() if mb.sum() else 0
        out_r = (~((xr[mr] >= rng[0]) & (xr[mr] <= rng[1]))).mean() if mr.sum() else 0
        a1.set_title(f"{name}   (outside range: Born {100*out_b:.2f} %, rad {100*out_r:.2f} %)", fontsize=9)
        a1.legend(fontsize=8)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = hr / hb
            err = ratio * np.sqrt(1 / np.maximum(nr_, 1) + 1 / np.maximum(nb_, 1))
        ok = (nb_ > 0) & (nr_ > 0)
        a2.errorbar(c[ok], ratio[ok], yerr=err[ok], fmt=".", color="k", ms=3)
        a2.axhline(1, color="gray", lw=0.8)
        a2.axhline(sr.mean() / sb.mean(), color="C3", lw=0.8, ls="--")
        a2.set_ylabel("rad / Born")
        a2.set_xlabel(lab)
        if ok.any():
            lo, hi = np.nanpercentile(ratio[ok], [2, 98])
            a2.set_ylim(max(0, lo - 0.1), hi + 0.1)
        fig.savefig(os.path.join(outdir, "png", f"{name}.png"), dpi=110, bbox_inches="tight")
        pdf.savefig(fig); plt.close(fig)
    for xn, yn, xr_, yr_ in MAPS:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, (d, w), t in zip(axs, ((db, wb), (dr, wr)), ("Born", "rad")):
            h = ax.hist2d(d[:, idx[xn]], d[:, idx[yn]], bins=60, range=(xr_, yr_), weights=w,
                          norm=LogNorm(), cmap="viridis")
            fig.colorbar(h[3], ax=ax, label="nb per cell")
            ax.set_title(t); ax.set_xlabel(xn)
        axs[0].set_ylabel(yn)
        fig.savefig(os.path.join(outdir, "png", f"map_{xn}_{yn}.png"), dpi=110, bbox_inches="tight")
        pdf.savefig(fig); plt.close(fig)
    pdf.close()


def main(proddir, outdir, jobs):
    born = load_arm(proddir, "born", jobs)
    rad = load_arm(proddir, "rad", jobs)
    np.savez_compressed(os.path.join(outdir, "jpsi_born_rad_events.npz"),
                        born=born[0], wborn=born[1], rad=rad[0], wrad=rad[1], fields=np.array(FIELDS))
    card = open(os.path.join(proddir, "cards", "rga_rad.txt")).read()
    meta = "rad card (fields):\n" + " | ".join(l.split("!")[0].strip() for l in card.split("\n")
                                               if l.split("!")[0].strip())
    plot(born, rad, outdir, meta)
    db, dr = born[0], rad[0]
    i = {f: k for k, f in enumerate(FIELDS)}
    print(f"Born {len(db)} ev sigma {born[2].mean():.5e} nb; rad {len(dr)} ev sigma {rad[2].mean():.5e} nb; "
          f"ratio {rad[2].mean()/born[2].mean():.4f}")
    for nm, d in (("Born", db), ("rad", dr)):
        print(f"{nm}: production photon in {np.isfinite(d[:, i['Eg_prod']]).mean():.3f}, "
              f"decay photon in {np.isfinite(d[:, i['Eg_dec']]).mean():.3f}, "
              f"M(ee)<3.0 {np.mean(d[:, i['Mee']] < 3.0):.4f}, theta_e {d[:, i['the']].min():.2f}-{d[:, i['the']].max():.2f}, "
              f"p_e median {np.median(d[:, i['pe']]):.3f}, vz {d[:, i['vz']].min():.2f}..{d[:, i['vz']].max():.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
