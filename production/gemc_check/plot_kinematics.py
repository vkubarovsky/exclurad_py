"""Kinematic plots for a CLAS12 GEMC sample ntupled with hipo_to_ntuple.groovy (any generator).

    python3 production/gemc_check/plot_kinematics.py <ntuple.f32> [outdir] [--ebeam E]

Input: flat little-endian float32 records (NFLD = 50), see aao_ntuple.groovy:
  [0] MC::Event.weight -- for aao_norad this is NOT a weight: the LUND header column 10
      holds nu = E - E' (GeV); aao events are unweighted
  [1..12] generated px,py,pz of e', p, gamma1, gamma2 (MC::Lund)
  [13+8k ..] for k = e', p, g1, g2: found pid charge status quality px py pz
             of the REC::Particle row that MC::GenMatch assigns to that MC particle
  [45] n(pid 11) [46] n(pid 2212) [47] n(pid 22) in REC::Particle  [48] n rows

Figures (style follows exclurad_py/production/diag_kinematics.py, all labels English):
  gen_kinematics.png      generator level, 5x3 diag layout (all events)
  rec_kinematics.png      reconstructed level, same layout (e'p identified; pi0 row from e'p gg)
  gen_vs_rec.png          shape comparison gen (scaled) vs rec, event and particle variables
  efficiency.png          tracked / identified fraction vs generated theta, |p|, and event variables
  header_nu_check.png     MC::Event.weight (LUND header col 10) against nu = E - E' computed from e'
  results.md              yield table
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
BLUE, FILL = "#0072B2", "#9ecae1"
ORANGE = "#D55E00"
GREEN = "#009E73"
PI0 = r"$\pi^0$"


# ----------------------------------------------------------------------------- io
def load(fname):
    d = np.memmap(fname, dtype=np.float32, mode="r")
    n = d.size // NFLD
    return d[: n * NFLD].reshape(n, NFLD)


def p3(a):
    return np.asarray(a, dtype=np.float64)


def four(v3, m):
    v3 = p3(v3)
    E = np.sqrt((v3 ** 2).sum(1) + m * m)
    return np.column_stack([E, v3])


def mag(v3):
    return np.sqrt((p3(v3) ** 2).sum(1))


def theta_deg(v3):
    v3 = p3(v3)
    p = mag(v3)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip(v3[:, 2] / np.where(p > 0, p, 1), -1, 1)))


def phi_deg(v3):
    v3 = p3(v3)
    return np.degrees(np.arctan2(v3[:, 1], v3[:, 0]))


def minkowski(a, b):
    return a[:, 0] * b[:, 0] - (a[:, 1:] * b[:, 1:]).sum(1)


def invariants(e4, p4):
    """Q2, W, xB from e', t from the recoil proton, MM2(e'p) of the missing meson."""
    k1 = np.array([EB, 0.0, 0.0, EB])
    P = np.array([M_P, 0.0, 0.0, 0.0])
    q = k1[None, :] - e4
    Q2 = -minkowski(q, q)
    nu = q[:, 0]
    W2 = M_P * M_P + 2 * M_P * nu - Q2
    W = np.sqrt(np.maximum(W2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        xB = Q2 / (2 * M_P * nu)
    d = P[None, :] - p4
    t = minkowski(d, d)
    miss = k1[None, :] + P[None, :] - e4 - p4
    MM2 = minkowski(miss, miss)
    return Q2, W, xB, t, MM2


# ----------------------------------------------------------------------------- style
def hist(ax, x, w, xlabel, bins=60, rng=None, logy=False, color=BLUE, fill=FILL,
         label=None, filled=True, density=False):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if rng is not None:
        m &= (x >= rng[0]) & (x <= rng[1])
    if rng is None and m.any():
        lo, hi = np.percentile(x[m], [0.2, 99.8])
        if hi <= lo:
            hi = lo + 1
        rng = (lo, hi)
    ww = None if w is None else np.asarray(w, float)[m]
    if filled:
        ax.hist(x[m], bins=bins, range=rng, weights=ww, color=color,
                histtype="stepfilled", facecolor=fill, edgecolor=color, lw=1.0,
                label=label, density=density)
    else:
        ax.hist(x[m], bins=bins, range=rng, weights=ww, color=color,
                histtype="step", lw=1.3, label=label, density=density)
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
    return rng


def hist2d(ax, x, y, w, xlabel, ylabel, yr=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if yr is not None:
        m &= (y >= yr[0]) & (y <= yr[1])
    if not m.any():
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return
    rng = [[np.nanmin(x[m]), np.nanmax(x[m])], list(yr)] if yr is not None else None
    ww = None if w is None else np.asarray(w, float)[m]
    ax.hist2d(x[m], y[m], bins=60, weights=ww, cmap="viridis", range=rng)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def diag_figure(parts, Q2, xB, t, W, MM2, mgg, w, title, out, n_parts_note=None):
    """The 5x3 layout of diag_kinematics.py; last panel is M(gg) instead of v."""
    THETA_MAX = {"e": 30.0}
    fig, ax = plt.subplots(5, 3, figsize=(13, 16))
    for r, (key, tex, v3, wk) in enumerate(parts):
        p, th = mag(v3), theta_deg(v3)
        thmax = THETA_MAX.get(key)
        trng = (0.0, thmax) if thmax else None
        hist(ax[r, 0], p, wk, r"%s $|p|$ [GeV]" % tex)
        hist(ax[r, 1], th, wk, r"%s $\theta$ [deg]" % tex, rng=trng)
        hist2d(ax[r, 2], p, th, wk, r"%s $|p|$ [GeV]" % tex, r"%s $\theta$ [deg]" % tex, yr=trng)
        if n_parts_note and key in n_parts_note:
            ax[r, 0].set_title(n_parts_note[key], fontsize=9, loc="left")
    hist(ax[3, 0], Q2, w, r"$Q^2$ [GeV$^2$]")
    hist(ax[3, 1], xB, w, r"$x_B$")
    hist(ax[3, 2], -t, w, r"$-t$ [GeV$^2$]")
    hist(ax[4, 0], W, w, r"$W$ [GeV]")
    hist(ax[4, 1], MM2, w, r"$M_X^2(e'p)$ [GeV$^2$]  (missing %s)" % PI0, logy=True)
    ax[4, 1].axvline(M_PI0 ** 2, color="grey", ls=":", lw=1)
    hist(ax[4, 2], mgg, parts[2][3], r"$M(\gamma\gamma)$ [GeV]")
    ax[4, 2].axvline(M_PI0, color="grey", ls=":", lw=1)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------- main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fname = args[0]
    outdir = args[1] if len(args) > 1 else "plots"
    global EB
    if "--ebeam" in sys.argv:
        EB = float(sys.argv[sys.argv.index("--ebeam") + 1])
    tag = os.path.basename(fname)
    os.makedirs(outdir, exist_ok=True)

    d = load(fname)
    n = d.shape[0]
    hdr_nu = p3(d[:, 0])          # LUND header column 10 = nu for aao_norad, not a weight
    w = np.ones(n)                # aao events are unweighted
    wtag = "unweighted"
    print("events", n)

    # ---- generator
    ge, gp, gg1, gg2 = d[:, 1:4], d[:, 4:7], d[:, 7:10], d[:, 10:13]
    e4, p4 = four(ge, 0.0), four(gp, M_P)
    g1_4, g2_4 = four(gg1, 0.0), four(gg2, 0.0)
    pi4 = g1_4 + g2_4
    Q2, W, xB, t, MM2 = invariants(e4, p4)
    mgg = np.sqrt(np.maximum(minkowski(pi4, pi4), 0))
    gen = dict(Q2=Q2, W=W, xB=xB, t=t, MM2=MM2, mgg=mgg,
               e=p3(ge), p=p3(gp), pi=pi4[:, 1:])

    # ---- reconstructed, through MC::GenMatch
    blk = {k: 13 + 8 * i for i, k in enumerate(["e", "p", "g1", "g2"])}
    want_pid = {"e": 11, "p": 2212, "g1": 22, "g2": 22}
    want_ch = {"e": -1, "p": 1, "g1": 0, "g2": 0}
    found, tracked, ident, rec3 = {}, {}, {}, {}
    for k, b in blk.items():
        found[k] = d[:, b] > 0
        v3 = d[:, b + 5:b + 8]
        pr = mag(v3)
        tracked[k] = found[k] & (d[:, b + 2] == want_ch[k]) & (pr > 0)
        ident[k] = tracked[k] & (d[:, b + 1] == want_pid[k])
        rec3[k] = v3
    sel_ep = ident["e"] & ident["p"]
    sel_all = sel_ep & ident["g1"] & ident["g2"]
    naive_e, naive_p, naive_g = d[:, 45] > 0, d[:, 46] > 0, d[:, 47] >= 2

    re4, rp4 = four(rec3["e"][sel_ep], 0.0), four(rec3["p"][sel_ep], M_P)
    rQ2, rW, rxB, rt, rMM2 = invariants(re4, rp4)
    rg1, rg2 = four(rec3["g1"][sel_all], 0.0), four(rec3["g2"][sel_all], 0.0)
    rpi4 = rg1 + rg2
    rmgg = np.sqrt(np.maximum(minkowski(rpi4, rpi4), 0))
    rec = dict(Q2=rQ2, W=rW, xB=rxB, t=rt, MM2=rMM2, mgg=rmgg,
               e=p3(rec3["e"][sel_ep]), p=p3(rec3["p"][sel_ep]), pi=rpi4[:, 1:])
    w_ep, w_all = w[sel_ep], w[sel_all]

    # ---- yield table
    rows = [
        ("generated events", n),
        ("e' tracked (matched, charge -1)", tracked["e"].sum()),
        ("e' identified (pid 11)", ident["e"].sum()),
        ("p tracked (matched, charge +1)", tracked["p"].sum()),
        ("p identified (pid 2212)", ident["p"].sum()),
        ("gamma1 identified (pid 22)", ident["g1"].sum()),
        ("gamma2 identified (pid 22)", ident["g2"].sum()),
        ("e' and p tracked", (tracked["e"] & tracked["p"]).sum()),
        ("e' and p identified  [rec level, event variables]", sel_ep.sum()),
        ("e' p gamma gamma identified  [rec level, pi0]", sel_all.sum()),
        ("naive: any pid 11 and any pid 2212 in REC::Particle", (naive_e & naive_p).sum()),
        ("naive: ... and >= 2 pid 22", (naive_e & naive_p & naive_g).sum()),
        ("p identified given e' identified", ident["p"][ident["e"]].sum()),
        ("p tracked given e' identified", tracked["p"][ident["e"]].sum()),
    ]
    lines = ["# aao_norad fa18_inb 0000: yields (%s)" % os.path.basename(fname), "",
             "| selection | events | fraction |", "|---|---:|---:|"]
    for name, cnt in rows:
        den = ident["e"].sum() if name.startswith("p identified given") or name.startswith("p tracked given") else n
        lines.append("| %s | %d | %.4f |" % (name, cnt, cnt / max(den, 1)))
    lines += ["", "Fractions relative to generated events, except the last two rows (relative to identified e')."]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(outdir, "results.md"), "w") as f:
        f.write(txt + "\n")

    # ---- figure 1: generator
    diag_figure([("e", r"$e'$", gen["e"], w), ("p", r"$p$", gen["p"], w), ("pi", PI0, gen["pi"], w)],
                gen["Q2"], gen["xB"], gen["t"], gen["W"], gen["MM2"], gen["mgg"], w,
                r"%s, generator level (MC::Lund), %d events, $E=%.3f$ GeV, %s" % (tag, n, EB, wtag),
                os.path.join(outdir, "gen_kinematics.png"))

    # ---- figure 2: reconstructed
    diag_figure([("e", r"$e'$", rec["e"], w_ep), ("p", r"$p$", rec["p"], w_ep), ("pi", PI0, rec["pi"], w_all)],
                rec["Q2"], rec["xB"], rec["t"], rec["W"], rec["MM2"], rec["mgg"], w_ep,
                r"%s, reconstructed (REC::Particle via MC::GenMatch): " % tag + 
                r"$e'p$ identified %d ev, $e'p\gamma\gamma$ %d ev, %s" % (sel_ep.sum(), sel_all.sum(), wtag),
                os.path.join(outdir, "rec_kinematics.png"),
                n_parts_note={"pi": r"from $e'p\gamma\gamma$ sample"})

    # ---- figure 3: gen vs rec shapes
    fig, ax = plt.subplots(5, 3, figsize=(13, 16))
    ev_panels = [("Q2", r"$Q^2$ [GeV$^2$]", 1, sel_ep), ("W", r"$W$ [GeV]", 1, sel_ep), ("xB", r"$x_B$", 1, sel_ep),
                 ("t", r"$-t$ [GeV$^2$]", -1, sel_ep), ("MM2", r"$M_X^2(e'p)$ [GeV$^2$]", 1, sel_ep),
                 ("mgg", r"$M(\gamma\gamma)$ [GeV]", 1, sel_all)]
    for i, (key, lab, sgn, sel) in enumerate(ev_panels):
        a = ax[i // 3, i % 3]
        xg, xr = sgn * gen[key][sel], sgn * rec[key]
        rng = hist(a, xr, w[sel], lab, color=ORANGE, fill="#fdd0a2", label="rec (%d)" % sel.sum())
        if key == "MM2":
            rng = (-0.5, 0.5)
            a.cla()
            hist(a, xr, w[sel], lab, rng=rng, color=ORANGE, fill="#fdd0a2", label="rec (%d)" % sel.sum())
        # generated distribution of the SAME events (acceptance-selected truth) and of all events, scaled
        hist(a, xg, w[sel], lab, rng=rng, color=BLUE, filled=False, label="gen, same events")
        m = np.isfinite(gen[key]) & (sgn * gen[key] >= rng[0]) & (sgn * gen[key] <= rng[1])
        mr = np.isfinite(xr) & (xr >= rng[0]) & (xr <= rng[1])
        scale = w[sel][mr].sum() / max(w[m].sum(), 1e-12)
        hist(a, sgn * gen[key], w * scale, lab, rng=rng, color=GREEN, filled=False, label="gen, all, scaled")
        a.legend(fontsize=7)
    for r, (key, tex, sel) in enumerate([("e", r"$e'$", sel_ep), ("p", r"$p$", sel_ep), ("pi", PI0, sel_all)], start=2):
        for c, (fn, lab) in enumerate([(mag, r"$|p|$ [GeV]"), (theta_deg, r"$\theta$ [deg]"), (phi_deg, r"$\phi$ [deg]")]):
            a = ax[r, c]
            xr, xg = fn(rec[key]), fn(gen[key][sel])
            rng = (-180, 180) if c == 2 else None
            rng = hist(a, xr, w[sel], "%s %s" % (tex, lab), rng=rng, color=ORANGE, fill="#fdd0a2", label="rec (%d)" % sel.sum())
            hist(a, xg, w[sel], "%s %s" % (tex, lab), rng=rng, color=BLUE, filled=False, label="gen, same events")
            xa = fn(gen[key])
            m = np.isfinite(xa) & (xa >= rng[0]) & (xa <= rng[1])
            mr = np.isfinite(xr) & (xr >= rng[0]) & (xr <= rng[1])
            scale = w[sel][mr].sum() / max(w[m].sum(), 1e-12)
            hist(a, xa, w * scale, "%s %s" % (tex, lab), rng=rng, color=GREEN, filled=False, label="gen, all, scaled")
            a.legend(fontsize=7)
    fig.suptitle("%s: reconstructed vs generated (%s). Rows 1-2: $e'p$ sample; $M(\\gamma\\gamma)$ and $\\pi^0$ row: $e'p\\gamma\\gamma$ sample"
                 % (tag, wtag), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(outdir, "gen_vs_rec.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)

    # ---- figure 4: efficiencies
    def eff(ax_, x, num_masks, labels, xlabel, edges):
        h0, _ = np.histogram(x, edges, weights=w)
        cen = 0.5 * (edges[1:] + edges[:-1])
        for msk, lab, col in zip(num_masks, labels, [BLUE, ORANGE, GREEN]):
            h1, _ = np.histogram(x[msk], edges, weights=w[msk])
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(h0 > 0, h1 / h0, np.nan)
            ax_.step(cen, r, where="mid", color=col, lw=1.4, label=lab)
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel("fraction")
        ax_.set_ylim(0, 1.05)
        ax_.grid(alpha=0.3)
        ax_.legend(fontsize=8)

    fig, ax = plt.subplots(4, 3, figsize=(13, 13))
    parts_eff = [("e", r"$e'$", gen["e"], (0, 40, 41), (0, 11, 45)),
                 ("p", r"$p$", gen["p"], (0, 90, 46), (0, 4, 41)),
                 ("g1", r"$\gamma_1$", p3(gg1), (0, 180, 46), (0, 6, 31))]
    for r, (k, tex, v3, thr, prr) in enumerate(parts_eff):
        th, p = theta_deg(v3), mag(v3)
        eff(ax[r, 0], th, [tracked[k], ident[k]], ["tracked", "identified"],
            r"generated %s $\theta$ [deg]" % tex, np.linspace(*thr))
        eff(ax[r, 1], p, [tracked[k], ident[k]], ["tracked", "identified"],
            r"generated %s $|p|$ [GeV]" % tex, np.linspace(*prr))
        hist(ax[r, 2], th, w, r"generated %s $\theta$ [deg]" % tex, rng=thr[:2])
        hist(ax[r, 2], th[ident[k]], w[ident[k]], r"generated %s $\theta$ [deg]" % tex, rng=thr[:2],
             color=ORANGE, fill="#fdd0a2", label="identified")
        ax[r, 2].set_yscale("log")
        ax[r, 2].legend(["all generated", "identified"], fontsize=8)
    eff(ax[3, 0], gen["Q2"], [tracked["e"] & tracked["p"], sel_ep, sel_all],
        [r"$e'p$ tracked", r"$e'p$ identified", r"$e'p\gamma\gamma$ identified"], r"generated $Q^2$ [GeV$^2$]", np.linspace(0.9, 8, 36))
    eff(ax[3, 1], gen["W"], [tracked["e"] & tracked["p"], sel_ep, sel_all],
        [r"$e'p$ tracked", r"$e'p$ identified", r"$e'p\gamma\gamma$ identified"], r"generated $W$ [GeV]", np.linspace(1.1, 4.0, 30))
    eff(ax[3, 2], -gen["t"], [tracked["e"] & tracked["p"], sel_ep, sel_all],
        [r"$e'p$ tracked", r"$e'p$ identified", r"$e'p\gamma\gamma$ identified"], r"generated $-t$ [GeV$^2$]", np.linspace(0, 4, 41))
    fig.suptitle("%s: reconstruction fractions vs generated quantities (MC::GenMatch), %s" % (tag, wtag), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(outdir, "efficiency.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)

    # ---- figure 5: LUND header column 10 is nu, not a weight
    nu = EB - mag(gen["e"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    hist(ax[0], hdr_nu, None, "MC::Event.weight (LUND header col 10) [GeV]")
    hist2d(ax[1], nu, hdr_nu, None, r"$\nu = E - E'$ from generated $e'$ [GeV]", "MC::Event.weight [GeV]")
    hist(ax[2], hdr_nu - nu, None, r"MC::Event.weight $- \nu$ [GeV]", rng=(-2e-3, 2e-3))
    fig.suptitle("%s: MC::Event.weight against $\\nu$ (aao_norad writes nu there; exclurad_py writes 1), " % tag + 
                 "not an event weight (events are unweighted)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(outdir, "header_nu_check.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
