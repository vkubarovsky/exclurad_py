#!/usr/bin/env python3
"""Photon-energy estimators for gamma d -> J/psi n (p_s) with the J/psi and the neutron measured.

    python3 validation/jpsi_egamma_estimators.py <d2 run dir with cache_photo.npz> <out.png>

Five ways to get E_gamma from the final state, compared with the true photon energy of the
generator, with ideal and with smeared momenta.  Eq. 10 of Liu et al., PRC 109, 065206 (2024),
written for the proton channel, is used here with the roles exchanged (undetected spectator proton).
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_N, M_P, M_D, M_PSI = 0.93956542, 0.93827208816, 1.87561294, 3.0969
C = dict(A="#0072B2", E="#009E73", B="#D55E00", D="#CC79A7", C="#000000")
LABEL = dict(C=r"(1) Eq. 10: spectator on shell, exact",
             A=r"(2) $E_\psi + E_n - M_n$",
             E=r"(3) $p_{z,\psi} + p_{z,n}$",
             D=r"(4) J/$\psi$ alone, neutron at rest",
             B=r"(5) $(s - M_n^2)/2M_n$, $s=(p_\psi+p_n)^2$")


def estimators(V, R):
    Ev, pv = V[:, 0], V[:, 1:]; En, pn = R[:, 0], R[:, 1:]
    P = pv + pn; Esum = Ev + En
    A = Esum - M_N
    E = P[:, 2]
    s = Esum ** 2 - (P ** 2).sum(1); B = (s - M_N ** 2) / (2 * M_N)
    D = (2 * M_N * Ev - M_PSI ** 2) / (2 * (M_N - Ev + pv[:, 2]))
    Cq = ((Esum - M_D) ** 2 - (P ** 2).sum(1) - M_P ** 2) / (2 * (Esum - P[:, 2] - M_D))
    return dict(C=Cq, A=A, E=E, D=D, B=B)


def smear(X, m, dp, dth, dph, rng):
    """Gaussian smearing of |p| (relative), theta and phi (absolute, rad); energy recomputed with mass m."""
    p = X[:, 1:]; pm = np.linalg.norm(p, axis=1); th = np.arccos(p[:, 2] / pm); ph = np.arctan2(p[:, 1], p[:, 0])
    pm = pm * (1 + dp * rng.standard_normal(len(pm))); th = th + dth * rng.standard_normal(len(pm)); ph = ph + dph * rng.standard_normal(len(pm))
    out = np.empty_like(X)
    out[:, 1] = pm * np.sin(th) * np.cos(ph); out[:, 2] = pm * np.sin(th) * np.sin(ph); out[:, 3] = pm * np.cos(th)
    out[:, 0] = np.sqrt(m * m + pm * pm)
    return out


def main():
    rundir, out = sys.argv[1], sys.argv[2]
    z = np.load(glob.glob(os.path.join(rundir, "cache_photo.npz"))[0])
    n = 2_000_000
    k = z["k"][:n].astype(float); V = z["V"][:n].astype(float); R = z["R"][:n].astype(float)
    rng = np.random.default_rng(1)
    # illustrative resolutions: J/psi from e+e- in the forward detector (1 % momentum, 1 mrad, 4 mrad);
    # neutron 10 % momentum from time of flight, 10 mrad angles
    Vs = smear(V, M_PSI, 0.01, 1e-3, 4e-3, rng); Rs = smear(R, M_N, 0.10, 1e-2, 1e-2, rng)
    sets = (("ideal momenta", estimators(V, R)), (r"smeared: J/$\psi$ 1 %, 1 mrad; neutron 10 %, 10 mrad", estimators(Vs, Rs)))
    edges = np.arange(8.2, 10.61, 0.2); cen = 0.5 * (edges[1:] + edges[:-1])
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 9.5, "lines.linewidth": 2.0, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    sel = k > 8.6
    table = {}
    for row, (title, est) in enumerate(sets):
        for key in ("B", "D", "E", "A", "C"):
            d = est[key] - k
            ax[row, 0].hist(d[sel], bins=200, range=(-1.0, 1.0), histtype="step", color=C[key], density=True,
                            label=LABEL[key] + ": rms %.0f MeV" % (1e3 * d[sel].std()))
            rms = np.array([d[(k >= a) & (k < b)].std() for a, b in zip(edges[:-1], edges[1:])])
            bias = np.array([d[(k >= a) & (k < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
            ax[row, 1].plot(cen, 1e3 * rms, color=C[key], label=LABEL[key])
            ax[row, 1].plot(cen, 1e3 * np.abs(bias), color=C[key], ls=":", lw=1.3)
            table[(row, key)] = (1e3 * d[sel].mean(), 1e3 * d[sel].std())
        ax[row, 0].set_yscale("log"); ax[row, 0].set_ylim(3e-3, 300); ax[row, 0].set_xlabel(r"$E_\gamma^{\rm rec} - E_\gamma^{\rm true}$ [GeV]")
        ax[row, 0].set_ylabel("probability density"); ax[row, 0].set_title(title + r", $E_\gamma > 8.6$ GeV", fontsize=12); ax[row, 0].legend(loc="upper left")
        ax[row, 1].set_yscale("log"); ax[row, 1].set_ylim(1, 2000); ax[row, 1].set_xlabel(r"true $E_\gamma$ [GeV]")
        ax[row, 1].set_ylabel("rms (solid) and |mean shift| (dotted) [MeV]"); ax[row, 1].grid(alpha=0.3, which="both")
        ax[row, 1].set_title(title, fontsize=12); ax[row, 1].legend(loc="upper right")
        ax[row, 1].axvline(8.2, color="grey", ls=":", lw=1)
    fig.suptitle(r"$\gamma\, d \to J/\psi\, n\, (p_s)$: photon energy from the measured J/$\psi$ and neutron, deuteron with AV18 Fermi motion", fontsize=13)
    fig.subplots_adjust(hspace=0.32, wspace=0.25)
    fig.savefig(out, dpi=170)
    print("%-45s %-22s %-22s" % ("estimator", "ideal: mean / rms [MeV]", "smeared: mean / rms [MeV]"))
    for key in ("C", "A", "E", "D", "B"):
        print("%-45s %+7.1f / %6.1f          %+7.1f / %6.1f" % (LABEL[key].replace("$", "").replace("\\", ""), *table[(0, key)], *table[(1, key)]))


if __name__ == "__main__":
    main()
