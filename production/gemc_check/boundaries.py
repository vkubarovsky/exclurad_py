"""Acceptance boundaries of every variable from a GEMC-reconstructed sample, per configuration.

    python3 production/gemc_check/boundaries.py <outdir> <label1>=<ntuple1.f32>[@Ebeam] [<label2>=<ntuple2.f32>[@Ebeam] ...]

Ebeam defaults to 10.604 GeV (RG-A fa18); give it per file for other data sets.

For each configuration (e.g. inbending, outbending) and each variable:
  * reconstructed distribution (REC::Particle via MC::GenMatch, identified e' and p; pi0 and
    photons from the e' p gamma gamma subset), overlaid across configurations,
    with the generated distribution (all events, scaled) for reference;
  * acceptance maps rec/gen in the planes that define a generator box:
    (theta_e', E'), (Q2, W), (Q2, xB), (theta_p, p_p), (theta_gamma, E_gamma);
  * a table of reconstructed percentiles (0.1, 1, 5, 50, 95, 99, 99.9 %) and, for the
    one-sided edges, the value where the identified/generated fraction first passes 10 % and 50 %.
Output: rec_<config>.png, rec_overlay.png, accmap_<config>.png, boundaries.md.
All samples unweighted; MC::Event.weight is nu, not a weight.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NFLD = 50
EB = 10.604
M_P = 0.93827208816
M_PI0 = 0.1349768
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
FILLS = ["#9ecae1", "#fdd0a2", "#a1d99b", "#e7b6d4"]
GREY = "#777777"


def mink(a, b):
    return a[:, 0] * b[:, 0] - (a[:, 1:] * b[:, 1:]).sum(1)


def four(v3, m):
    v3 = np.asarray(v3, float)
    return np.column_stack([np.sqrt((v3 ** 2).sum(1) + m * m), v3])


def angles(v4):
    p3 = v4[:, 1:]
    pm = np.linalg.norm(p3, axis=1)
    th = np.degrees(np.arccos(np.clip(p3[:, 2] / np.maximum(pm, 1e-30), -1, 1)))
    ph = np.degrees(np.arctan2(p3[:, 1], p3[:, 0]))
    return pm, th, ph


def event_vars(e4, p4, EB=EB):
    k1 = np.array([EB, 0.0, 0.0, EB])
    P = np.array([M_P, 0.0, 0.0, 0.0])
    q = k1[None, :] - e4
    Q2 = -mink(q, q)
    nu = q[:, 0]
    W = np.sqrt(np.maximum(M_P ** 2 + 2 * M_P * nu - Q2, 0))
    xB = Q2 / (2 * M_P * nu)
    y = nu / EB
    d = P[None, :] - p4
    t = mink(d, d)
    miss = k1[None, :] + P[None, :] - e4 - p4
    return dict(Q2=Q2, W=W, xB=xB, y=y, mt=-t, MM2=mink(miss, miss), Ebeam=np.full(len(Q2), EB))


def load(fname, EB=EB):
    d = np.memmap(fname, dtype=np.float32, mode="r").reshape(-1, NFLD)
    out = {}
    ge, gp, g1, g2 = (np.asarray(d[:, 1 + 3 * k:4 + 3 * k], float) for k in range(4))
    e4, p4 = four(ge, 0.0), four(gp, M_P)
    pi4 = four(g1, 0.0) + four(g2, 0.0)
    gen = event_vars(e4, p4, EB)
    for nm, v4 in (("e", e4), ("p", p4), ("pi", pi4), ("g1", four(g1, 0.0)), ("g2", four(g2, 0.0))):
        pm, th, ph = angles(v4)
        gen[nm + "_p"], gen[nm + "_th"], gen[nm + "_ph"] = pm, th, ph
    gen["e_E"] = gen["e_p"]
    out["gen"] = gen
    # reconstructed via GenMatch
    blk = {k: 13 + 8 * i for i, k in enumerate(["e", "p", "g1", "g2"])}
    want = {"e": (11, -1), "p": (2212, 1), "g1": (22, 0), "g2": (22, 0)}
    ident, r3 = {}, {}
    for k, b in blk.items():
        v3 = np.asarray(d[:, b + 5:b + 8], float)
        pr = np.linalg.norm(v3, axis=1)
        ident[k] = (d[:, b] > 0) & (d[:, b + 1] == want[k][0]) & (d[:, b + 2] == want[k][1]) & (pr > 0)
        r3[k] = v3
    sel_ep = ident["e"] & ident["p"]
    sel_all = sel_ep & ident["g1"] & ident["g2"]
    re4, rp4 = four(r3["e"][sel_ep], 0.0), four(r3["p"][sel_ep], M_P)
    rec = event_vars(re4, rp4, EB)
    rec["e_dp"] = np.linalg.norm(r3["e"][sel_ep], axis=1) / gen["e_p"][sel_ep] - 1.0
    rec["p_dp"] = np.linalg.norm(r3["p"][sel_ep], axis=1) / gen["p_p"][sel_ep] - 1.0
    for nm, v4 in (("e", re4), ("p", rp4)):
        pm, th, ph = angles(v4)
        rec[nm + "_p"], rec[nm + "_th"], rec[nm + "_ph"] = pm, th, ph
    rec["e_E"] = rec["e_p"]
    rg1, rg2 = four(r3["g1"][sel_all], 0.0), four(r3["g2"][sel_all], 0.0)
    rpi = rg1 + rg2
    recpi = {}
    for nm, v4 in (("pi", rpi), ("g1", rg1), ("g2", rg2)):
        pm, th, ph = angles(v4)
        recpi[nm + "_p"], recpi[nm + "_th"], recpi[nm + "_ph"] = pm, th, ph
    recpi["mgg"] = np.sqrt(np.maximum(mink(rpi, rpi), 0))
    out.update(rec=rec, recpi=recpi, sel_ep=sel_ep, sel_all=sel_all, ident=ident, n=d.shape[0])
    return out


# variable catalogue: key, label, range, which rec set, log-y
VARS = [
    ("Q2", r"$Q^2$ [GeV$^2$]", (0.8, 8.0), "rec", True),
    ("W", r"$W$ [GeV]", (1.0, 4.1), "rec", False),
    ("xB", r"$x_B$", (0.0, 1.0), "rec", False),
    ("y", r"$y=\nu/E$", (0.0, 0.9), "rec", False),
    ("e_E", r"$E'$ [GeV]", (1.0, 10.0), "rec", False),
    ("e_th", r"$\theta_{e'}$ [deg]", (0.0, 40.0), "rec", False),
    ("e_ph", r"$\phi_{e'}$ [deg]", (-180, 180), "rec", False),
    ("mt", r"$-t$ [GeV$^2$]", (0.0, 5.0), "rec", True),
    ("p_p", r"$p_p$ [GeV]", (0.0, 4.0), "rec", False),
    ("p_th", r"$\theta_p$ [deg]", (0.0, 80.0), "rec", False),
    ("p_ph", r"$\phi_p$ [deg]", (-180, 180), "rec", False),
    ("MM2", r"$M_X^2(e'p)$ [GeV$^2$]", (-0.5, 0.6), "rec", True),
    ("g1_p", r"$E_{\gamma_1}$ [GeV]", (0.0, 8.0), "recpi", True),
    ("g1_th", r"$\theta_{\gamma_1}$ [deg]", (0.0, 40.0), "recpi", False),
    ("pi_p", r"$p_{\pi^0}$ [GeV]", (0.0, 9.0), "recpi", False),
    ("pi_th", r"$\theta_{\pi^0}$ [deg]", (0.0, 40.0), "recpi", False),
    ("pi_ph", r"$\phi_{\pi^0}$ [deg]", (-180, 180), "recpi", False),
    ("mgg", r"$M(\gamma\gamma)$ [GeV]", (0.0, 0.3), "recpi", False),
]
PCT = [0.1, 1, 5, 50, 95, 99, 99.9]
# one-sided edges: variable, direction ("low" = acceptance opens going up), which numerator
EDGES = [("e_th", "low", "e"), ("e_E", "low", "e"), ("e_E", "high", "e"), ("Q2", "low", "ep"), ("W", "high", "ep"),
         ("y", "high", "ep"), ("y", "low", "ep"), ("p_p", "low", "p"), ("p_th", "low", "p"), ("p_th", "high", "p"),
         ("mt", "low", "ep"), ("g1_th", "low", "g"), ("g1_th", "high", "g"), ("g1_p", "low", "g"), ("pi_th", "high", "all")]


def edge(x, num_mask, direction, rng, nb=200):
    """Value where the fraction num/gen first exceeds 10 % and 50 % coming from the closed side."""
    edges = np.linspace(rng[0], rng[1], nb + 1)
    h0, _ = np.histogram(x, edges)
    h1, _ = np.histogram(x[num_mask], edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(h0 > 50, h1 / h0, np.nan)
    cen = 0.5 * (edges[1:] + edges[:-1])
    order = range(nb) if direction == "low" else range(nb - 1, -1, -1)
    out = {}
    for thr in (0.10, 0.50):
        val = np.nan
        for i in order:
            if np.isfinite(f[i]) and f[i] >= thr:
                val = cen[i]
                break
        out[thr] = val
    return out


def rec_figure(cfgs, out, overlay=False):
    n = len(VARS)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(14, 3.3 * nrow))
    ax = ax.ravel()
    for i, (key, lab, rng, which, logy) in enumerate(VARS):
        a = ax[i]
        edges = np.linspace(rng[0], rng[1], 60)
        for j, (name, D) in enumerate(cfgs):
            x = D[which][key]
            h, _ = np.histogram(x, edges)
            dens = h / max(h.sum(), 1) / np.diff(edges) if overlay else h
            a.step(0.5 * (edges[1:] + edges[:-1]), dens, where="mid", color=COLORS[j], lw=1.3,
                   label="%s rec (%d)" % (name, len(x)))
            if not overlay:
                a.fill_between(0.5 * (edges[1:] + edges[:-1]), dens, step="mid", color=FILLS[j], alpha=0.8)
            if key in D["gen"] and not overlay:
                xg = D["gen"][key]
                hg, _ = np.histogram(xg, edges)
                sc = h.sum() / max(hg.sum(), 1)
                a.step(0.5 * (edges[1:] + edges[:-1]), hg * sc, where="mid", color=GREY, lw=1.0, ls="--",
                       label="generated, scaled" if j == 0 else None)
        a.set_xlabel(lab)
        if logy:
            a.set_yscale("log")
        a.legend(fontsize=6.5)
    for k in range(n, len(ax)):
        ax[k].axis("off")
    ttl = "GEMC output, reconstructed (identified e'p; photon/pi0 panels from e'p gamma gamma)"
    fig.suptitle(ttl + (" -- normalized overlay of configurations" if overlay else " -- " + cfgs[0][0]), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


GEN_VARS = [("Q2", r"generated $Q^2$ [GeV$^2$]", (0.8, 8.0), True), ("W", r"generated $W$ [GeV]", (1.0, 4.1), False),
            ("xB", r"generated $x_B$", (0.0, 1.0), False), ("y", r"generated $y$", (0.0, 0.9), False),
            ("e_E", r"generated $E'$ [GeV]", (1.5, 10.0), False), ("e_th", r"generated $\theta_{e'}$ [deg]", (3.0, 40.0), False),
            ("mt", r"generated $-t$ [GeV$^2$]", (0.0, 5.0), True), ("p_p", r"generated $p_p$ [GeV]", (0.0, 4.0), False),
            ("p_th", r"generated $\theta_p$ [deg]", (0.0, 80.0), False), ("g1_p", r"generated $E_{\gamma_1}$ [GeV]", (0.0, 8.0), True),
            ("g1_th", r"generated $\theta_{\gamma_1}$ [deg]", (0.0, 60.0), False), ("pi_th", r"generated $\theta_{\pi^0}$ [deg]", (0.0, 60.0), False)]


BOXES = {
    "card on disk (Q2 1.2-8, -t 0.001-1.8, y 0.25-0.90, xB 0.13-0.60)": dict(Q2=(1.2, 8.0), mt=(0.001, 1.8), y=(0.25, 0.90), xB=(0.13, 0.60)),
    "18.09 box (Q2 1-6, -t 0.10-2.0, y 0.25-0.90, xB 0.13-0.60)": dict(Q2=(1.0, 6.0), mt=(0.10, 2.0), y=(0.25, 0.90), xB=(0.13, 0.60)),
    "Q2 1-8, -t 0-3, no y/xB cut": dict(Q2=(1.0, 8.0), mt=(0.0, 3.0), y=(0.0, 1.0), xB=(0.0, 1.0)),
    "Q2 0.9-10, -t 0-4, no y/xB cut": dict(Q2=(0.9, 10.0), mt=(0.0, 4.0), y=(0.0, 1.0), xB=(0.0, 1.0)),
}


def coverage(D, sel, box):
    g = D["gen"]
    m = sel & (g["W"] >= 1.8) & (g["W"] <= 4.0)
    inb = m.copy()
    for k, (lo, hi) in box.items():
        inb &= (g[k] >= lo) & (g[k] <= hi)
    return inb.sum() / max(m.sum(), 1)


def gen_of_accepted_figure(cfgs, out):
    """Generated kinematics of the events that end up identified: what a generator box must cover."""
    fig, ax = plt.subplots(4, 3, figsize=(14, 13))
    for a, (key, lab, rng, logy) in zip(ax.ravel(), GEN_VARS):
        edges = np.linspace(rng[0], rng[1], 60)
        cen = 0.5 * (edges[1:] + edges[:-1])
        for j, (name, D) in enumerate(cfgs):
            sel = D["sel_all"] if key.startswith(("g1", "pi")) else D["sel_ep"]
            x = D["gen"][key]
            h0, _ = np.histogram(x, edges)
            h1, _ = np.histogram(x[sel], edges)
            if j == 0:
                a.step(cen, h0 / max(h0.sum(), 1), where="mid", color=GREY, lw=1.0, ls="--", label="generated, all (norm.)")
            a.step(cen, h1 / max(h1.sum(), 1), where="mid", color=COLORS[j], lw=1.4, label="%s: gen. of accepted (%d)" % (name, sel.sum()))
        a.set_xlabel(lab)
        if logy:
            a.set_yscale("log")
        a.legend(fontsize=6.5)
    fig.suptitle("Generated kinematics of the events that are reconstructed and identified (e'p; photon/pi0 panels: e'p gamma gamma). "
                 "Normalized to unit area.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def accmap_figure(name, D, out):
    gen, id_ = D["gen"], D["ident"]
    planes = [
        ("e_E", "e_th", r"$E'$ [GeV]", r"$\theta_{e'}$ [deg]", (1.8, 10.0), (4, 40), id_["e"], "e' identified"),
        ("Q2", "W", r"$Q^2$ [GeV$^2$]", r"$W$ [GeV]", (0.9, 8), (1.1, 4.05), D["sel_ep"], "e'p identified"),
        ("xB", "Q2", r"$x_B$", r"$Q^2$ [GeV$^2$]", (0.05, 0.95), (0.9, 8), D["sel_ep"], "e'p identified"),
        ("p_p", "p_th", r"$p_p$ [GeV]", r"$\theta_p$ [deg]", (0.2, 4.0), (0, 75), id_["p"], "p identified"),
        ("g1_p", "g1_th", r"$E_{\gamma_1}$ [GeV]", r"$\theta_{\gamma_1}$ [deg]", (0, 6), (0, 60), id_["g1"], "gamma identified"),
        ("W", "mt", r"$W$ [GeV]", r"$-t$ [GeV$^2$]", (1.1, 4.05), (0, 4), D["sel_all"], "e'p gamma gamma identified"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for a, (kx, ky, lx, ly, rx, ry, num, ttl) in zip(ax.ravel(), planes):
        x, y = gen[kx], gen[ky]
        h0, xe, ye = np.histogram2d(x, y, bins=[50, 50], range=[rx, ry])
        h1, _, _ = np.histogram2d(x[num], y[num], bins=[xe, ye])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(h0 > 30, h1 / h0, np.nan)
        im = a.pcolormesh(xe, ye, r.T, cmap="viridis", vmin=0, vmax=1)
        a.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), np.nan_to_num(r.T), levels=[0.1, 0.5], colors=["white", "red"], linewidths=0.8)
        a.set_xlabel(lx)
        a.set_ylabel(ly)
        a.set_title("%s: %s / generated" % (name, ttl), fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("%s: acceptance maps (fraction of generated events reconstructed and identified); contours 10 %% (white), 50 %% (red)" % name, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    cfgs = []
    for a in sys.argv[2:]:
        name, f = a.split("=", 1)
        eb = EB
        if "@" in f:
            f, eb = f.split("@", 1)
            eb = float(eb)
        cfgs.append((name, load(f, eb)))
        print("loaded", name, cfgs[-1][1]["n"], "events; e'p identified", cfgs[-1][1]["sel_ep"].sum(), "e'p gg", cfgs[-1][1]["sel_all"].sum())

    for name, D in cfgs:
        rec_figure([(name, D)], os.path.join(outdir, "rec_%s.png" % name))
        accmap_figure(name, D, os.path.join(outdir, "accmap_%s.png" % name))
    if len(cfgs) > 1:
        rec_figure(cfgs, os.path.join(outdir, "rec_overlay.png"), overlay=True)
    gen_of_accepted_figure(cfgs, os.path.join(outdir, "gen_of_accepted.png"))

    lines = ["# Acceptance boundaries from the GEMC-reconstructed sample(s)", "",
             "Generated box of the aao_norad production (applies to all configurations): Q2 0.9-10.6 GeV2, "
             "E' 1.9-9.95 GeV (y <= 0.82), W 1.10-4.04 GeV, no cut on t, xB, theta.  Reconstructed edges that "
             "coincide with these are generator limits, not acceptance.", ""]
    for name, D in cfgs:
        lines.append("## %s: %d generated, %d e'p identified (%.2f %%), %d e'p gamma gamma (%.2f %%)" % (
            name, D["n"], D["sel_ep"].sum(), 100 * D["sel_ep"].mean(), D["sel_all"].sum(), 100 * D["sel_all"].mean()))
        lines.append("")
        lines.append("### Reconstructed percentiles")
        lines.append("")
        lines.append("| variable | sample | " + " | ".join("%g %%" % p for p in PCT) + " |")
        lines.append("|---|---|" + "---:|" * len(PCT))
        for key, lab, rng, which, _ in VARS:
            x = D[which][key]
            pc = np.percentile(x, PCT)
            lines.append("| %s | %s | " % (key, "e'p" if which == "rec" else "e'p gg") + " | ".join("%.3f" % v for v in pc) + " |")
        lines.append("")
        lines.append("### Generated kinematics of the accepted events (percentiles) -- what the generator box must cover")
        lines.append("")
        lines.append("| variable | sample | " + " | ".join("%g %%" % p for p in PCT) + " |")
        lines.append("|---|---|" + "---:|" * len(PCT))
        for key, lab, rng, logy in GEN_VARS:
            sel = D["sel_all"] if key.startswith(("g1", "pi")) else D["sel_ep"]
            pc = np.percentile(D["gen"][key][sel], PCT)
            lines.append("| %s | %s | " % (key, "e'p" if not key.startswith(("g1", "pi")) else "e'p gg") + " | ".join("%.3f" % v for v in pc) + " |")
        lines.append("")
        lines.append("### Same, restricted to W >= 1.8 GeV (the exclurad_py model region)")
        lines.append("")
        lines.append("| variable | sample | " + " | ".join("%g %%" % p for p in PCT) + " |")
        lines.append("|---|---|" + "---:|" * len(PCT))
        wm = D["gen"]["W"] >= 1.8
        for key, lab, rng, logy in GEN_VARS:
            sel = (D["sel_all"] if key.startswith(("g1", "pi")) else D["sel_ep"]) & wm
            pc = np.percentile(D["gen"][key][sel], PCT)
            lines.append("| %s | %s | " % (key, "e'p" if not key.startswith(("g1", "pi")) else "e'p gg") + " | ".join("%.3f" % v for v in pc) + " |")
        lines.append("")
        lines.append("Accepted e'p events with W >= 1.8: %d (%.1f %% of accepted); e'p gg: %d" % (
            (D["sel_ep"] & wm).sum(), 100 * (D["sel_ep"] & wm).sum() / max(D["sel_ep"].sum(), 1), (D["sel_all"] & wm).sum()))
        lines.append("")
        lines.append("### Coverage of the accepted phase space (W 1.8-4.0) by candidate generator boxes")
        lines.append("")
        lines.append("| box | e'p accepted inside | e'p gg accepted inside |")
        lines.append("|---|---:|---:|")
        for bname, box in BOXES.items():
            lines.append("| %s | %.1f %% | %.1f %% |" % (bname, 100 * coverage(D, D["sel_ep"], box), 100 * coverage(D, D["sel_all"], box)))
        lines.append("")
        lines.append("Loss from each cut alone (accepted e'p with W 1.8-4.0 falling outside):")
        lines.append("")
        g = D["gen"]
        m = D["sel_ep"] & (g["W"] >= 1.8) & (g["W"] <= 4.0)
        singles = [("Q2 < 1.0", g["Q2"] < 1.0), ("Q2 < 1.2", g["Q2"] < 1.2), ("Q2 > 6", g["Q2"] > 6), ("Q2 > 8", g["Q2"] > 8),
                   ("y < 0.25", g["y"] < 0.25), ("y > 0.80", g["y"] > 0.80), ("xB < 0.13", g["xB"] < 0.13), ("xB > 0.60", g["xB"] > 0.60),
                   ("-t < 0.10", g["mt"] < 0.10), ("-t > 1.8", g["mt"] > 1.8), ("-t > 2.0", g["mt"] > 2.0), ("-t > 3.0", g["mt"] > 3.0)]
        lines.append("| cut | e'p lost |")
        lines.append("|---|---:|")
        for nm, bad in singles:
            lines.append("| %s | %.1f %% |" % (nm, 100 * (m & bad).sum() / max(m.sum(), 1)))
        lines.append("")
        dpe, dpp = D["rec"]["e_dp"], D["rec"]["p_dp"]
        lines.append("Mismeasured among identified: e' with |p_rec/p_gen - 1| > 0.2: %.2f %% ; > 0.5: %.2f %% ; p with > 0.2: %.2f %%" % (
            100 * (np.abs(dpe) > 0.2).mean(), 100 * (np.abs(dpe) > 0.5).mean(), 100 * (np.abs(dpp) > 0.2).mean()))
        lines.append("")
        lines.append("### One-sided edges of the identified/generated fraction")
        lines.append("")
        lines.append("| variable | side | numerator | fraction reaches 10 % at | reaches 50 % at |")
        lines.append("|---|---|---|---:|---:|")
        nummask = {"e": D["ident"]["e"], "p": D["ident"]["p"], "g": D["ident"]["g1"], "ep": D["sel_ep"], "all": D["sel_all"]}
        rngs = {k: r for k, _, r, _, _ in VARS}
        for key, side, num in EDGES:
            e = edge(D["gen"][key], nummask[num], side, rngs[key])
            lines.append("| %s | %s | %s | %.3f | %.3f |" % (key, side, num, e[0.10], e[0.50]))
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(outdir, "boundaries.md"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
