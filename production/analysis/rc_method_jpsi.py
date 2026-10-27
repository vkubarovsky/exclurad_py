#!/usr/bin/env python3
"""J/psi: the reader, the kinematics and M. Tenorio's selection, for both MC arms.

    python3 production/analysis/rc_method_jpsi.py        # prints the cut flow

Everything else in the study (rc_method_jpsi_combos.py, _plots.py) imports
this module, so the selection exists once.

THE SAMPLE.  prod_jpsi_f18in_20261009, GEMC 5.14, copied to
/Volumes/wd_14tb/rc_method_jpsi/ntuple and checked against ifarm by sha256
(2177 files, 0 mismatches).  The arms differ in exactly three card fields,
and all three are part of the correction being measured:
    iborn   rad 0 / Born 1     production radiation (beam and e')
    itarg   rad 1 / Born 0     pre-vertex radiation in the LH2 cell
    idecay  rad 2 / Born 1     J/psi -> e+e-(gamma), the decay leptons radiate
So C_rad from the MC is the WHOLE correction, which is what Mariana's
0.88 x 0.85 stands for: the Born arm is the cross section at nominal energy.

THE SELECTION, her note Secs. 3-4, as far as the ntuple carries the variable:
    e'       identified FT electron (campaign_lib), theta_e 2.45-6.45 deg
             in place of her FT-face radius 8 < r < 15.75 cm (the ntuple has
             no FT hit position; that window reproduces her r cut on 91 % of
             her own F18in e'e+e- events, event by event)
    nu       > 8.1 GeV
    e-, e+   pid -11 / 11, |p| > 1.95 GeV, e- |p| < 10.6 GeV
    vertex   -8 < vz < 4 cm on the inward-bending lepton (e- inb, e+ outb)
    M(ee)    2.95 - 3.20 GeV
    3 tracks dpz > -2.5 GeV,  MX(e'e+e-) within mu +- 4 sigma, mu and sigma
             from a Gaussian core fit to the BORN MC (her rule, our sample)
    4 tracks proton pid 2212, |p| > 0.4 GeV,  -0.7 < dpz < 0.5 GeV,
             |MX2(e'e+e-p)| < 0.036 GeV^2
NOT applied, because the ntuple does not carry them: PCAL energy and
fiducial (V, W > 9 cm), BDT scores, the triangular sampling-fraction cut,
chi2_PID, beta, the theta(E) cut on two-hit FT clusters, and lepton timing.
They act on the same detector response in both arms; they reach C_rad only
through the lepton momenta that FSR changes, which the M(ee) window already
dominates.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_lib as C  # noqa: E402

DATA = "/Volumes/wd_14tb/rc_method_jpsi"
EB = 10.6
MP, ME_, MJ = 0.938272, 0.000511, 3.096900
TOPOS = {"3trk": "ee+e-", "4trk": "ee+e-p"}

THETA_E = (2.45, 6.45)
NU_MIN = 8.1
P_LEP_MIN, P_EM_MAX = 1.95, 10.6
VZ = (-8.0, 4.0)
MEE = (2.95, 3.20)
DPZ3_MIN = -2.5
DPZ4 = (-0.7, 0.5)
MX2TOT = 0.036
P_PROT_MIN = 0.4
NSIG_MX = 4.0

# pre-vertex radiator, as in generator/events.py: P(loss < y) = y^bt,
# bt = BRAD (z - z_min) / X0, the vertex uniform in the cell
BRAD, X0_LH2, CELL = 4.0 / 3.0, 890.4, (-5.5, -0.5)


def four(p3, m):
    return np.column_stack([np.sqrt((p3 ** 2).sum(1) + m * m), p3])


def mdot(a, b):
    return a[:, 0] * b[:, 0] - (a[:, 1:] * b[:, 1:]).sum(1)


def trento_phi(q3, h3):
    """Trento azimuth of h around q, the lepton plane spanned by q and the beam."""
    k3 = np.tile([0.0, 0.0, 1.0], (len(q3), 1))
    qn = q3 / np.linalg.norm(q3, axis=1)[:, None]
    n_l = np.cross(qn, k3)
    n_h = np.cross(qn, h3)
    nn = np.linalg.norm(n_l, axis=1) * np.linalg.norm(n_h, axis=1)
    cos = (n_l * n_h).sum(1) / nn
    sin = (np.cross(n_l, n_h) * qn).sum(1) / nn
    return np.degrees(np.arctan2(sin, cos)) % 360.0


def kinematics(d, idx=None):
    """The variables of the binning and the selection, from campaign_lib's dict.

    idx picks rows (the generated partners of the accepted events).  Every
    quantity is built from the four three-momenta, with the NOMINAL beam
    energy, exactly as an analysis of data would build it.
    """
    s = (lambda x: x) if idx is None else (lambda x: x[idx])
    pe, pp, ppos, pel = (s(d[n + "_p3"]) for n in ("escat", "prot", "hplus", "hminus"))
    k = np.array([EB, 0, 0, EB])
    tgt = np.array([MP, 0, 0, 0])
    E = four(pe, ME_); P = four(pp, MP); L1 = four(ppos, ME_); L2 = four(pel, ME_)
    V = L1 + L2
    q = k - E
    Q2 = -mdot(q, q)
    nu = q[:, 0]
    W2 = MP ** 2 + 2 * MP * nu - Q2
    X3 = k + tgt - E - V                      # missing proton
    Xep = k + tgt - E - P                     # missing J/psi
    Xt = X3 - P                               # nothing should be missing
    ez_bal = EB - (E[:, 3])                   # Ebeam - E' cos(theta') for an electron, |p| >> m
    out = dict(Q2=Q2, nu=nu, W2=W2, W=np.sqrt(np.clip(W2, 0, None)),
               mee=np.sqrt(np.clip(mdot(V, V), 0, None)),
               mx3=np.sqrt(np.clip(mdot(X3, X3), 0, None)), mx2_3=mdot(X3, X3),
               mx2_ep=mdot(Xep, Xep), mx2_tot=mdot(Xt, Xt),
               dpz3=V[:, 3] - ez_bal, dpz4=V[:, 3] + P[:, 3] - ez_bal,
               mt_V=-mdot(q - V, q - V), mt_p=-mdot(P - tgt, P - tgt),
               xminus=X3[:, 0] - X3[:, 3],    # E - p_z of the missing proton (pre-vertex photon -> v)
               pminus=P[:, 0] - P[:, 3],
               vminus=V[:, 0] - V[:, 3],      # the same for the missing J/psi of the 4-track bin
               phi_V=trento_phi(q[:, 1:], V[:, 1:]),
               phi_p=(trento_phi(q[:, 1:], P[:, 1:]) + 180.0) % 360.0,
               theta_e=np.degrees(np.arccos(np.clip(pe[:, 2] / np.linalg.norm(pe, axis=1), -1, 1))),
               p_em=np.linalg.norm(pel, axis=1), p_ep=np.linalg.norm(ppos, axis=1),
               p_p=np.linalg.norm(pp, axis=1))
    for n in ("escat", "prot", "hplus", "hminus"):
        for f in ("pid", "vz"):
            if n + "_" + f in d:
                out[n + "_" + f] = s(d[n + "_" + f])
    return out


def chunk_ids(arm, pol):
    return sorted(int(re.search(r"_(\d+)\.f32$", f).group(1))
                  for f in glob.glob(f"{DATA}/ntuple/{arm}_{pol}/*.f32"))


def sigma(arm, pol):
    """sigma_nb over exactly the chunks that are read (campaign_lib rule)."""
    return C.sigma_nb(f"{DATA}/prov/{arm}", arm, chunks=chunk_ids(arm, pol))


def load(arm, pol, topo):
    """One arm, one polarity, one topology: reconstructed kinematics of the
    events with the topology, their generated partners, and N_gen."""
    pat = f"{DATA}/ntuple/{arm}_{pol}/*.f32"
    gen = C.load(pat, "jpsi", rec=False)
    rec = C.load(pat, "jpsi", rec=True, require=C.TOPOLOGY[TOPOS[topo]])
    keep = np.flatnonzero(rec["keep"])
    s = sigma(arm, pol)
    return dict(arm=arm, pol=pol, topo=topo,
                n_gen=gen["n_read"], n_chunks=gen["n_files"], sigma_nb=s["mean"], sigma_err=s["err"],
                rec=kinematics(rec), gen=kinematics(gen, keep),
                gen_all=gen, w=s["mean"] / gen["n_read"])


def calibrate_mx3(k, base):
    """mu and sigma of MX(e'e+e-) on Born: a Gaussian fit to the core, her mu +- 4 sigma rule."""
    from scipy.optimize import curve_fit
    x = k["mx3"][base]
    x = x[(x > 0.4) & (x < 1.6)]
    h, e = np.histogram(x, bins=120, range=(0.4, 1.6))
    c = 0.5 * (e[1:] + e[:-1])
    i = np.argmax(h)
    sel = np.abs(c - c[i]) < 0.15
    g = lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2)
    p, _ = curve_fit(g, c[sel], h[sel], p0=(h[i], c[i], 0.05))
    return float(p[1]), float(abs(p[2]))


def cuts(k, pol, topo, cal=None):
    """Ordered cumulative cut flow.  Returns {name: mask}; the last one is the selection."""
    inward = "hminus_vz" if pol == "inb" else "hplus_vz"
    L = {}
    m = np.ones(len(k["Q2"]), bool)
    def add(name, c):
        nonlocal m
        m = m & c
        L[name] = m.copy()
    add("topology", m)
    add("FT theta_e", (k["theta_e"] > THETA_E[0]) & (k["theta_e"] < THETA_E[1]))
    add("nu > 8.1", k["nu"] > NU_MIN)
    add("lepton PID", (k["hminus_pid"] == 11) & (k["hplus_pid"] == -11))
    add("lepton |p|", (k["p_em"] > P_LEP_MIN) & (k["p_em"] < P_EM_MAX) & (k["p_ep"] > P_LEP_MIN))
    add("vz", (k[inward] > VZ[0]) & (k[inward] < VZ[1]))
    if topo == "4trk":
        add("proton", (k["prot_pid"] == 2212) & (k["p_p"] > P_PROT_MIN))
    add("M(ee)", (k["mee"] > MEE[0]) & (k["mee"] < MEE[1]))
    if topo == "3trk":
        add("dpz", k["dpz3"] > DPZ3_MIN)
        if cal is not None:
            mu, sg = cal
            add("MX window", np.abs(k["mx3"] - mu) < NSIG_MX * sg)
    else:
        add("dpz", (k["dpz4"] > DPZ4[0]) & (k["dpz4"] < DPZ4[1]))
        add("MX2 total", np.abs(k["mx2_tot"]) < MX2TOT)
    return L


def pre_mx(L):
    """The selection without the exclusivity window, for calibrating it."""
    return L["dpz"]


def eta_ext(v_cut, xminus):
    """Pre-vertex radiation, analytic: probability that the beam photon keeps v below v_cut.

    A collinear beam photon of energy y E moves MX^2 of the missing proton by
    2 y E (E - p_z)_X, so it survives when y < y_max = v_cut / (2 E (E-p_z)_X).
    With P(loss < y) = y^bt and bt = b0 z' (z' uniform in 0..1 along the cell)
    the average is (y^b0 - 1) / (b0 ln y).  Peaking approximation: the hard
    part, which changes the Born cross section at the vertex, is dropped.
    """
    b0 = BRAD * (CELL[1] - CELL[0]) / X0_LH2
    y = np.clip(v_cut / (2 * EB * np.clip(xminus, 1e-6, None)), 1e-12, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(y < 1.0, (y ** b0 - 1.0) / (b0 * np.log(y)), 1.0)
    return r


def main():
    for topo in TOPOS:
        for pol in ("inb", "outb"):
            S = {a: load(a, pol, topo) for a in ("rad", "born")}
            cal = None
            if topo == "3trk":
                Lb = cuts(S["born"]["rec"], pol, topo)
                cal = calibrate_mx3(S["born"]["rec"], pre_mx(Lb))
            print(f"\n{topo} {pol}  N_gen rad {S['rad']['n_gen']} ({S['rad']['n_chunks']} chunks) "
                  f"born {S['born']['n_gen']} ({S['born']['n_chunks']})  "
                  f"sigma_rad/sigma_born = {S['rad']['sigma_nb'] / S['born']['sigma_nb']:.5f}"
                  + (f"   MX cal mu {cal[0]:.4f} sigma {cal[1]:.4f}" if cal else ""))
            L = {a: cuts(S[a]["rec"], pol, topo, cal) for a in S}
            for name in L["rad"]:
                nr, nb = L["rad"][name].sum(), L["born"][name].sum()
                B = nr * S["rad"]["w"] / (nb * S["born"]["w"])
                print(f"  {name:12} rad {nr:7d}  born {nb:7d}   B = {B:.4f} +- {B * np.sqrt(1 / nr + 1 / nb):.4f}")


if __name__ == "__main__":
    main()
