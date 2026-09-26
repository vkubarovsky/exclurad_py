#!/usr/bin/env python3
"""What Fermi motion does to quasi-real J/psi photoproduction near threshold.

    python3 validation/jpsi_fermi_study.py <d2 run dir> <free-proton run dir> [<free-neutron run dir>] [outdir]

Each run dir is an exclurad_py output directory (<outroot>/<channel>/<date>_..._v1) with
lund/*.lund and inputs/config.json (sigma_nb).  Reads the LUND directly: electron, J/psi
(status 0 with its e+e- daughters), recoil nucleon, and -- on a deuteron -- the spectator.

Figures (all labels English), in outdir:
  jpsi_fermi_egamma.png   dN/dE_gamma (lab, E - E') per nucleon, deuteron vs free; sub-threshold yield
  jpsi_fermi_w.png        observed W (target at rest) minus true gamma N W, and the observed W spectrum
  jpsi_fermi_spectator.png  |p_s| of accepted events below / above the free threshold, and m*
  jpsi_fermi_t.png        -t distributions deuteron vs free
  jpsi_fermi_study.md     the numbers
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_P, M_N, M_D = 0.93827208816, 0.93956542, 1.87561294
M_JPSI = 3.0969
E_TH_N = ((M_N + M_JPSI) ** 2 - M_N ** 2) / (2 * M_N)      # free-neutron photoproduction threshold
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#777777"


def read_run(rundir):
    """Return dict of arrays and the run's sigma_nb, E_beam, target."""
    cfgs = sorted(glob.glob(os.path.join(rundir, "inputs", "config*.json")))
    cfg = json.load(open(cfgs[0]))
    sigma = float(np.mean([json.load(open(c))["report"]["sigma_nb"] for c in cfgs]))
    EB = float(cfg["card"]["ebeam"])
    fermi = cfg["card"].get("fermi", "none")
    files = sorted(glob.glob(os.path.join(rundir, "lund", "*.lund")))
    e4, v4, r4, s4 = [], [], [], []
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
                npart = int(w[0])
                ee = vv = rr = ss = None
                for _ in range(npart):
                    p = fh.readline().split()
                    pid, status, parent = int(p[3]), int(p[2]), int(p[4])
                    px, py, pz, E = map(float, p[6:10])
                    q4 = (E, px, py, pz)
                    if pid == 11 and status == 1 and parent == 0 and ee is None:
                        ee = q4
                    elif pid == 443:
                        vv = q4
                    elif pid in (2212, 2112) and status == 1:
                        if rr is None:
                            rr = q4          # first nucleon line = the recoil (written before the spectator)
                        else:
                            ss = q4          # second nucleon line = the spectator
                if ee is None or vv is None or rr is None:
                    continue
                e4.append(ee); v4.append(vv); r4.append(rr)
                s4.append(ss if ss is not None else (0.0, 0.0, 0.0, 0.0))
    e4, v4, r4, s4 = (np.array(x, float) for x in (e4, v4, r4, s4))
    return dict(e=e4, V=v4, R=r4, S=s4, sigma_nb=sigma, EB=EB, fermi=fermi, n=len(e4), rundir=rundir)


def mink2(a):
    return a[:, 0] ** 2 - (a[:, 1:] ** 2).sum(1)


def kin(d):
    EB = d["EB"]
    k = np.zeros_like(d["e"]); k[:, 0] = EB; k[:, 3] = EB
    q = k - d["e"]
    Eg = q[:, 0]
    Q2 = -mink2(q)
    M = M_N if d["fermi"] != "none" or d["rundir"].rstrip("/").split("/")[-2].startswith("jpsin") else M_P
    # observed W: target nucleon assumed at rest
    W_obs = np.sqrt(np.maximum(M * M + 2 * M * Eg - Q2, 0))
    # true gamma* N W: from the final state J/psi + recoil
    W_true = np.sqrt(np.maximum(mink2(d["V"] + d["R"]), 0))
    t = mink2(q - d["V"])
    ps = np.linalg.norm(d["S"][:, 1:], axis=1)
    struck = d["V"] + d["R"] - q           # the struck nucleon four-vector (= target if free)
    mstar = np.sqrt(np.maximum(mink2(struck), 0))
    return dict(Eg=Eg, Q2=Q2, W_obs=W_obs, W_true=W_true, mt=-t, ps=ps, mstar=mstar)


def main():
    args = [a for a in sys.argv[1:]]
    d2_dir, p_dir = args[0], args[1]
    n_dir = args[2] if len(args) > 2 and os.path.isdir(args[2]) else None
    outdir = args[3] if len(args) > 3 else (args[2] if len(args) > 2 and not os.path.isdir(args[2]) else "plots/jpsi_fermi")
    os.makedirs(outdir, exist_ok=True)
    runs = [("deuteron (AV18), gamma n", read_run(d2_dir), BLUE), ("free proton, gamma p", read_run(p_dir), ORANGE)]
    if n_dir:
        runs.append(("free neutron, gamma n", read_run(n_dir), GREEN))
    K = [(lab, d, kin(d), col) for lab, d, col in runs]
    lines = ["# Fermi motion in quasi-real J/psi photoproduction near threshold", ""]
    for lab, d, k, _ in K:
        lines.append("- %s: %d events, sigma_nb = %.4f (per nucleon, box of the card), E_beam %.3f" % (lab, d["n"], d["sigma_nb"], d["EB"]))
    lines.append("")
    lines.append("Free-neutron photoproduction threshold E_gamma = %.3f GeV." % E_TH_N)
    lines.append("")

    # ---- figure 1: E_gamma spectrum, absolute per-nucleon normalisation (sigma_nb * fraction)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    edges = np.linspace(6.0, 10.6, 47)
    cen = 0.5 * (edges[1:] + edges[:-1])
    lines.append("## Yield by lab photon energy, per nucleon, normalised to sigma_nb x fraction of events (nb)")
    lines.append("")
    lines.append("| E_gamma bin | " + " | ".join(lab for lab, _, _, _ in K) + " |")
    lines.append("|---|" + "---:|" * len(K))
    hists = {}
    for lab, d, k, col in K:
        h, _ = np.histogram(k["Eg"], edges)
        hists[lab] = h * d["sigma_nb"] / d["n"]
        ax[0].step(cen, hists[lab], where="mid", color=col, lw=1.6, label=lab)
    for lo, hi in ((6.5, 7.5), (7.5, 8.0), (8.0, 8.2), (8.2, 9.0), (9.0, 10.0)):
        vals = []
        for lab, d, k, col in K:
            m = (k["Eg"] >= lo) & (k["Eg"] < hi)
            vals.append("%.4g nb (%d ev)" % (m.sum() * d["sigma_nb"] / d["n"], m.sum()))
        lines.append("| %.1f-%.1f | " % (lo, hi) + " | ".join(vals) + " |")
    ax[0].axvline(E_TH_N, color=GREY, ls="--", lw=1)
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"$E_\gamma = E - E'$ (lab) [GeV]"); ax[0].set_ylabel(r"$\sigma$ per bin [nb per nucleon]")
    ax[0].legend(fontsize=8); ax[0].set_title("quasi-real J/$\\psi$ photoproduction, $Q^2$ 0.01-0.1 GeV$^2$", fontsize=10)
    ref = hists[K[1][0]]
    for lab, d, k, col in K[:1] + K[2:]:
        with np.errstate(invalid="ignore", divide="ignore"):
            ax[1].step(cen, np.where(ref > 0, hists[lab] / ref, np.nan), where="mid", color=col, lw=1.6, label="%s / free proton" % lab)
    ax[1].axvline(E_TH_N, color=GREY, ls="--", lw=1); ax[1].axhline(1, color=GREY, lw=0.6)
    ax[1].set_ylim(0, 2); ax[1].set_xlabel(r"$E_\gamma$ (lab) [GeV]"); ax[1].set_ylabel("ratio"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "jpsi_fermi_egamma.png"), dpi=130); plt.close(fig)

    # ---- figure 2: W smearing
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    lines.append(""); lines.append("## Observed W (target at rest) against the true gamma N invariant mass")
    lines.append("")
    for lab, d, k, col in K:
        dW = k["W_obs"] - k["W_true"]
        ax[0].hist(dW, bins=80, range=(-0.6, 0.6), histtype="step", color=col, lw=1.5, label="%s: rms %.0f MeV" % (lab, 1e3 * dW.std()), density=True)
        ax[1].hist(k["W_obs"], bins=70, range=(3.6, 5.4), histtype="step", color=col, lw=1.5, label=lab, density=True)
        lines.append("- %s: W_obs - W_true mean %+.1f MeV, rms %.1f MeV; W_obs below the free threshold %.3f GeV: %.2f %%" % (
            lab, 1e3 * dW.mean(), 1e3 * dW.std(), M_N + M_JPSI, 100 * (k["W_obs"] < M_N + M_JPSI).mean()))
    ax[0].set_xlabel(r"$W_{\rm obs} - W_{\rm true}$ [GeV]"); ax[0].legend(fontsize=8); ax[0].set_yscale("log")
    ax[1].axvline(M_N + M_JPSI, color=GREY, ls="--", lw=1); ax[1].set_xlabel(r"$W_{\rm obs}$ (nucleon at rest) [GeV]"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "jpsi_fermi_w.png"), dpi=130); plt.close(fig)

    # ---- figure 3: spectator momentum below / above threshold, m*
    lab, d, k, col = K[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    below, above = k["Eg"] < E_TH_N, k["Eg"] >= E_TH_N
    ax[0].hist(k["ps"][above], bins=60, range=(0, 0.7), histtype="step", color=BLUE, lw=1.5, density=True, label=r"$E_\gamma$ above threshold")
    ax[0].hist(k["ps"][below], bins=60, range=(0, 0.7), histtype="step", color=ORANGE, lw=1.5, density=True, label=r"$E_\gamma$ below threshold")
    ax[0].set_xlabel(r"spectator $|p_s|$ [GeV/c]"); ax[0].legend(fontsize=8); ax[0].set_title("deuteron sample", fontsize=10)
    lines.append(""); lines.append("## Spectator momentum in the deuteron sample")
    lines.append("")
    for nm, m in (("above threshold", above), ("below threshold", below), ("E_gamma < 8.0", k["Eg"] < 8.0), ("E_gamma < 7.5", k["Eg"] < 7.5)):
        if m.any():
            lines.append("- %s (%d events): <|p_s|> = %.0f MeV/c, fraction above 300 MeV/c = %.1f %%, m* mean %.3f GeV, m* < 0.3: %.2f %%" % (
                nm, m.sum(), 1e3 * k["ps"][m].mean(), 100 * (k["ps"][m] > 0.3).mean(), k["mstar"][m].mean(), 100 * (k["mstar"][m] < 0.3).mean()))
    egb = np.linspace(6.5, 10.6, 42)
    mean_ps = [k["ps"][(k["Eg"] >= a) & (k["Eg"] < b)].mean() if ((k["Eg"] >= a) & (k["Eg"] < b)).sum() > 5 else np.nan for a, b in zip(egb[:-1], egb[1:])]
    ax[1].step(0.5 * (egb[1:] + egb[:-1]), np.array(mean_ps) * 1e3, where="mid", color=BLUE, lw=1.6)
    ax[1].axvline(E_TH_N, color=GREY, ls="--", lw=1); ax[1].set_xlabel(r"$E_\gamma$ (lab) [GeV]"); ax[1].set_ylabel(r"$\langle |p_s| \rangle$ [MeV/c]"); ax[1].grid(alpha=0.3)
    ax[2].hist(k["mstar"], bins=70, range=(0.0, 1.0), histtype="step", color=BLUE, lw=1.5, log=True)
    ax[2].axvline(0.10, color=ORANGE, ls="--", lw=1, label="floor 0.10 GeV"); ax[2].axvline(0.30, color=GREY, ls=":", lw=1, label=r"$\pi^0$ floor 0.30")
    ax[2].set_xlabel(r"$m^*$ of the struck neutron [GeV]"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "jpsi_fermi_spectator.png"), dpi=130); plt.close(fig)

    # ---- figure 4: t
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for lab, d, k, col in K:
        ax[0].hist(k["mt"], bins=60, range=(0, 4), histtype="step", color=col, lw=1.5, density=True, label=lab, log=True)
    lab, d, k, col = K[0]
    ax[1].hist(k["mt"][above], bins=40, range=(0, 4), histtype="step", color=BLUE, lw=1.5, density=True, label="deuteron, above threshold", log=True)
    if below.sum() > 20:
        ax[1].hist(k["mt"][below], bins=40, range=(0, 4), histtype="step", color=ORANGE, lw=1.5, density=True, label="deuteron, below threshold", log=True)
    for a in ax:
        a.set_xlabel(r"$-t = -(q - p_{J/\psi})^2$ [GeV$^2$]"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "jpsi_fermi_t.png"), dpi=130); plt.close(fig)

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(outdir, "jpsi_fermi_study.md"), "w") as f:
        f.write(txt + "\n")
    print("figures in", outdir)


if __name__ == "__main__":
    main()
