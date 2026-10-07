#!/usr/bin/env python3
"""Effective photon flux per W bin on the deuteron against the free-neutron bremsstrahlung flux.

    python3 validation/jpsi_effective_flux_W.py <d2 run dir> <out.png>

Phi_d(W) = sum over generated events of 1/sigma(s) in the bin, scaled to photons per electron per
neutron: the number of photons that produced a given final-state W, folded with Fermi motion, binding
and the flux factor.  Dividing an acceptance-corrected yield in a W bin by this is what turns a
deuteron measurement into sigma(gamma n -> J/psi n) at that W.
"""
import glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exclurad_py import models
from exclurad_py.generator.photo import PhotoGenerator, brems_shape

M_N, M_D, M_PSI, E0 = 0.93956542, 1.87561294, 3.0969, 10.6
run, out = sys.argv[1], sys.argv[2]
z = np.load(run + "/cache_photo.npz"); k = z["k"].astype(float); V = z["V"].astype(float); R = z["R"].astype(float); S = z["S"].astype(float)
cfg = json.load(open(glob.glob(run + "/inputs/config*.json")[0]))["report"]; Y = cfg["yield_per_electron_per_nucleon_nb"]; N = len(k)
P = V + R; s = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1)
ps = np.linalg.norm(S[:, 1:], axis=1)
g = PhotoGenerator(models.get("jpsin.roberts2026"), E0, 6.5, E0, (-6.0, -0.001)); g.setup()
sig = np.interp(s, g._W2g, g._sigg)
Eeff = (s - M_N ** 2) / (2 * M_N)
edges = np.arange(7.0, 11.61, 0.2); cen = 0.5 * (edges[1:] + edges[:-1])


def phi(mask):
    h, _ = np.histogram(Eeff[mask], edges, weights=1.0 / sig[mask]); return h * Y / N


phi_all = phi(np.ones(N, bool)); phi_cut = phi(ps < 0.2)
X_LD2 = 2.5 / 769.1
phi_free = np.array([np.trapezoid(X_LD2 * brems_shape(np.linspace(a, min(b, E0 - 1e-6), 200), E0), np.linspace(a, min(b, E0 - 1e-6), 200)) if a < E0 else 0.0 for a, b in zip(edges[:-1], edges[1:])])
frac_cut = np.array([(ps[(Eeff >= a) & (Eeff < b)] < 0.2).mean() if ((Eeff >= a) & (Eeff < b)).sum() > 50 else np.nan for a, b in zip(edges[:-1], edges[1:])])
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10, "lines.linewidth": 2.0, "savefig.bbox": "tight"})
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].step(cen, phi_free, where="mid", color="#009E73", label="free neutron: bremsstrahlung photons in the bin")
ax[0].step(cen, phi_all, where="mid", color="#0072B2", label=r"deuteron: effective photons $\Phi_d(W)$, all events")
ax[0].step(cen, phi_cut, where="mid", color="#D55E00", label=r"deuteron, $|p_s| < 0.2$ GeV/c")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-8, 3e-4); ax[0].set_xlabel(r"$E_\gamma^{\rm eff} = (W^2 - M_n^2)/2M_n$ [GeV]   ($W$ from $p_\psi + p_n$)")
ax[0].set_ylabel("photons per electron per neutron, 200 MeV bins"); ax[0].axvline(8.2, color="grey", ls=":"); ax[0].axvline(E0, color="grey", ls=":"); ax[0].legend(loc="lower left")
ax[0].text(8.25, 1.8e-4, "free threshold", fontsize=9, color="grey"); ax[0].text(10.65, 1.8e-4, r"$E_0$", fontsize=9, color="grey")
with np.errstate(divide="ignore", invalid="ignore"):
    r_all = phi_all / phi_free; r_cut = phi_cut / phi_free
ax[1].plot(cen, r_all, "o-", color="#0072B2", ms=4, label="all events"); ax[1].plot(cen, r_cut, "s-", color="#D55E00", ms=4, label=r"$|p_s| < 0.2$ GeV/c")
ax[1].axhline(1, color="grey", lw=0.8); ax[1].set_ylim(0.5, 1.2); ax[1].set_xlim(8.2, 10.6); ax[1].set_xlabel(r"$E_\gamma^{\rm eff}$ [GeV]")
ax[1].set_ylabel(r"$\Phi_d(W)\, /\, \Phi_{\rm free}(W)$"); ax[1].grid(alpha=0.3); ax[1].legend(loc="lower left")
ax2 = ax[1].twinx(); ax2.plot(cen, 100 * frac_cut, color="#CC79A7", ls="--", lw=1.5); ax2.set_ylabel(r"events with $|p_s| < 0.2$ GeV/c [%]", color="#CC79A7"); ax2.set_ylim(0, 100)
fig.suptitle(r"Binning in $W$: effective photon flux on the deuteron against the free-neutron flux (AV18, PWIA, flux factor included)", fontsize=12)
fig.savefig(out, dpi=170); print(out)
print("events beyond the free-nucleon endpoint (E_eff > E0): %.2f %%; below the free threshold: %.2f %%" % (100 * (Eeff > E0).mean(), 100 * (Eeff < 8.2).mean()))
