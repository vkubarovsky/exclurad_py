"""Untagged real-photon production: bremsstrahlung in the target, gamma N -> V N.

    python3 -m exclurad_py.generator.photo card.txt [--jobs N]

What is generated.  The beam electron radiates in the target material upstream of the
production vertex (thin-target bremsstrahlung, complete screening), the photon strikes a
nucleon at rest -- or, on a deuteron, a nucleon with an AV18 Fermi momentum, spectator
prescription as in generator/fermi.py -- and produces the vector meson.  The scattered beam
electron is NOT part of the event: it continues down the beam line.  The hadronic cross
section is the electroproduction model at Q2 = 0, whose Hand-convention sigma_T IS the
real-photon dsigma/dt (the Hand flux K = (W2 - M2)/2M is the real-photon energy), so nothing
new is assumed about the hadron side.

Event weight (per incident electron, per target nucleon):

    dN/dk dt = n_gamma(k) * dsigma_T/dt(W2(k, p_N), Q2 = 0, t)

    n_gamma(k) dk = (X/X0) (1/k) [ 4/3 (1 - y) + y^2 ] dk,   y = k / E0        (Tsai, thin target)

with X the radiator thickness the photon has seen: the material between the entrance of the
target and the vertex.  The vertex is drawn along the cell with density proportional to that
upstream thickness (the photon flux grows linearly along the cell), and the per-event flux
factor is then the cell-averaged one, X = L/2.  The 5 cm LH2 / LD2 cell is 0.006 / 0.007 X0,
so the thin-target formula is exact to that order and the electron's own energy loss before
radiating is neglected.

Sampling.  k from the analytic bremsstrahlung cumulative in [kmin, kmax]; the nucleon; then
accept with probability sigma_tot(W) / sigma_max, where sigma_tot(W) is the t-integral of the
model at Q2 = 0 over the card's t range and sigma_max a scanned majorant with a safety factor
(overweights are counted and reported, as in the main generator).  t from the exact inverse
cumulative of dsigma/dt on a 400-point grid for the event's W.  phi uniform.  The meson and
the recoil are built in the gamma N centre of mass and boosted to the laboratory with the
actual photon and nucleon four-vectors, so on a deuteron the event is exact in the lab and
photon + deuteron = V + N' + N_spectator to machine precision.

Card (key = value, one per line, '#' comments):

    model  = jpsin.roberts2026     # any registered vector-meson model; the target follows its topology
    ebeam  = 10.6                  # electron beam energy, GeV
    kmin   = 6.5                   # photon energy range, GeV (kmax defaults to ebeam)
    kmax   = 10.6
    tmin   = -6.0                  # t range (both negative, tmin < tmax)
    tmax   = -0.001
    nev    = 100000
    seed   = 1
    fermi  = av18                  # none | av18
    fermi_mstar_min = 0.10
    target_length_cm = 5.0
    target_X0_cm     = 769.1       # LD2; LH2: 890.4
    vz     = -5.5 -0.5             # cell in the lab, cm (upstream end first)
    idecay = 1
    maj_safety = 1.5
    outroot = /Volumes/wd_14tb/...

Output: <outroot>/<channel>/<date>_<E>_photo_v1/lund/*.lund and inputs/config.json with the
photon flux integral, <sigma>, sigma_max, overweights and timing.  LUND header per event:
beam pid 22, beam energy = k, target pid 2212 / 2112 / 1000010020.
"""
import json
import os
import socket
import sys
import time
from datetime import date as _date

import numpy as np
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

from ..core.constants import M_P, M_N
from .. import models
from .decay import DECAY, decay_two_body
from .fermi import CONFIG, FermiSampler, split_deuteron, M_D
from .frames import boost_matrix
from .lund import PID, CHANNEL_PID, DECAY_PID

X0_CM = {"LH2": 890.4, "LD2": 769.1}


# --------------------------------------------------------------------------- #
# bremsstrahlung
# --------------------------------------------------------------------------- #
M_E = 0.51099895e-3          # GeV


def brems_shape_cs(k, E0):
    """Complete screening: (1/k)[4/3 (1-y) + y^2], y = k/E0, per unit X/X0.  Finite at the tip."""
    y = k / E0
    return (4.0 / 3.0 * (1.0 - y) + y * y) / k


def brems_cdf_cs(k, E0):
    """Indefinite integral of brems_shape_cs: 4/3 ln k - 4/3 k/E0 + k^2/(2 E0^2)."""
    return 4.0 / 3.0 * np.log(k) - 4.0 / 3.0 * k / E0 + k * k / (2.0 * E0 * E0)


def screening_factor(k, E0, a_s=137.0):
    """Schiff intermediate-screening spectrum divided by its complete-screening limit.

    Schiff (1951) with a screening radius a_s in units of the electron Compton wavelength
    (Thomas-Fermi: 111 Z^-1/3; for hydrogen and deuterium the Bohr radius, 137, is the
    physical one and is the default).  Away from the endpoint the factor is 1; as k -> E0 the
    minimum momentum transfer m^2 k / (2 E E') outgrows 1/a_s, screening switches off, and the
    logarithm ln M + 1 - (2/b) arctan b falls through zero: the spectrum goes to zero within a
    fraction of an MeV of the endpoint, after starting to bend down about 30 MeV below it for
    a 10.6 GeV beam.  The formula is cut at zero where Schiff's bracket turns negative.
    """
    k = np.asarray(k, float)
    E = E0 / M_E
    kk = np.clip(k / M_E, 1e-9, E - 1e-9)
    Ep = E - kk
    r = Ep / E
    b = 2.0 * E * Ep / (a_s * kk)
    Minv = (kk / (2.0 * E * Ep)) ** 2 + 1.0 / a_s ** 2
    lnM = -np.log(Minv)
    atb = np.arctan(b)
    A = 1.0 + r * r - 2.0 / 3.0 * r
    F = A * (lnM + 1.0 - 2.0 * atb / b) \
        + r * (2.0 / b ** 2 * np.log1p(b * b) + 4.0 * (2.0 - b * b) / (3.0 * b ** 3) * atb - 8.0 / (3.0 * b * b) + 2.0 / 9.0)
    Finf = A * (2.0 * np.log(a_s) + 1.0) + 2.0 / 9.0 * r
    return np.clip(F / Finf, 0.0, None)


def brems_shape(k, E0, a_s=137.0):
    """Thin-target photon spectrum per unit X/X0 with the Schiff endpoint behaviour."""
    return brems_shape_cs(k, E0) * screening_factor(k, E0, a_s)


class BremsSampler:
    """Draw k from the screened spectrum in [kmin, kmax] by a tabulated cumulative."""

    def __init__(self, E0, kmin, kmax, a_s=137.0, n=20001):
        self.E0, self.kmin, self.kmax = float(E0), float(kmin), float(kmax)
        self._k = np.linspace(self.kmin, self.kmax, n)
        f = brems_shape(self._k, self.E0, a_s)
        c = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(self._k))))
        self.norm = float(c[-1])                   # photons per electron per unit X/X0
        self.norm_cs = float(brems_cdf_cs(self.kmax, self.E0) - brems_cdf_cs(self.kmin, self.E0))
        self._c = c / self.norm

    def draw(self, rng):
        return float(np.interp(rng.random(), self._c, self._k))


# --------------------------------------------------------------------------- #
# kinematics helpers
# --------------------------------------------------------------------------- #
def mink(a):
    return a[0] ** 2 - float(np.dot(a[1:], a[1:]))


def two_body_cm(W, m1, m2):
    E1 = (W * W + m1 * m1 - m2 * m2) / (2.0 * W)
    p = np.sqrt(max(E1 * E1 - m1 * m1, 0.0))
    return E1, p


def rotation_from_z(n):
    """Rotation matrix taking +z onto the unit vector n."""
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n); s = np.linalg.norm(v); c = float(np.dot(z, n))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


class PhotoGenerator:
    def __init__(self, sf, E0, kmin, kmax, t_range, fermi="none", fermi_m_star_min=0.10,
                 seed=1, maj_safety=1.5, target_length_cm=5.0, target_X0_cm=769.1,
                 vz=(-5.5, -0.5)):
        self.sf, self.E0 = sf, float(E0)
        self.kmin, self.kmax = float(kmin), float(kmax if kmax else E0)
        self.t_lo, self.t_hi = float(t_range[0]), float(t_range[1])      # both negative, t_lo < t_hi
        self.rng = np.random.default_rng(seed)
        self.brems = BremsSampler(self.E0, self.kmin, self.kmax)
        self.fermi = str(fermi or "none")
        self.floor = float(fermi_m_star_min)
        self._fermi = FermiSampler(self.fermi) if self.fermi != "none" else None
        self.struck = str(getattr(sf.topology, "target", "p"))
        self.mN = M_N if self.struck == "n" else M_P
        self.mV = float(sf.topology.m_detected)
        self.mR = float(sf.topology.m_missing)
        self.X_over_X0 = 0.5 * float(target_length_cm) / float(target_X0_cm)   # cell-averaged upstream thickness
        self.vz = (float(vz[0]), float(vz[1]))
        self.safety = float(maj_safety)
        self.n_trials = self.n_fermi_reject = self.n_below = self.n_over = 0
        self.max_over = 1.0

    # -- cross section at Q2 = 0 --------------------------------------------- #
    def dsigma_dt(self, W2, t):
        """nb/GeV^2 from the model's Hand-convention sigma_T at Q2 = 0."""
        out = self.sf._sfs(np.asarray(W2, float), np.zeros_like(np.asarray(t, float)) + 1e-9, np.asarray(t, float))
        return np.asarray(out["sT"], float)

    def t_limits(self, W2):
        W = np.sqrt(W2)
        Eg = (W2 - self.mN ** 2) / (2 * W)                    # real photon in the gamma N CM
        Ev, pv = two_body_cm(W, self.mV, self.mR)
        t0 = self.mV ** 2 - 2 * Eg * (Ev - pv)                 # forward
        t1 = self.mV ** 2 - 2 * Eg * (Ev + pv)                 # backward
        return max(t1, self.t_lo), min(t0, self.t_hi)         # lower (more negative), upper

    def sigma_tot(self, W2):
        lo, hi = self.t_limits(W2)
        if hi <= lo:
            return 0.0
        t = np.linspace(lo, hi, 401)
        return float(_trapz(self.dsigma_dt(W2, t), t))

    def draw_t(self, W2):
        lo, hi = self.t_limits(W2)
        t = np.linspace(lo, hi, 401)
        f = self.dsigma_dt(W2, t)
        c = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))))
        if c[-1] <= 0:
            return None
        return float(np.interp(self.rng.random() * c[-1], c, t))

    # -- majorant of sigma_tot over the reachable W ---------------------------- #
    def setup(self):
        # W2 reachable: free nucleon at the highest k, plus the Fermi head-on gain
        smax = self.mN ** 2 + 2 * self.kmax * self.mN
        if self._fermi is not None:
            # struck nucleon moving toward the photon at the ceiling momentum
            from .fermi import P_MAX
            ps = 0.99 * P_MAX
            struck4, _, mstar = split_deuteron(np.array([0.0, 0.0, -ps]), self.struck)
            smax = max(smax, mstar ** 2 + 2 * self.kmax * (struck4[0] + ps))
        # sigma_tot(W2) once, on a grid; per trial it is an interpolation.  The majorant is
        # taken over W2 reachable with spectator momenta up to p_s = 0.40 GeV/c head-on
        # (99.9 % of the AV18 weight); rarer configurations beyond it are overweights,
        # counted and reported, exactly as the main generator does.
        self._W2g = np.linspace((self.mV + self.mR) ** 2 * 1.0001, smax, 800)
        self._sigg = np.array([self.sigma_tot(w) for w in self._W2g])
        s_maj = self.mN ** 2 + 2 * self.kmax * self.mN
        if self._fermi is not None:
            struck4, _, mstar = split_deuteron(np.array([0.0, 0.0, -0.40]), self.struck)
            s_maj = mstar ** 2 + 2 * self.kmax * (struck4[0] + 0.40)
        inside = self._W2g <= s_maj
        self.sigma_max = float(self._sigg[inside].max()) * self.safety
        self._W2_at_max = float(self._W2g[inside][int(self._sigg[inside].argmax())])
        self._s_maj = float(s_maj)
        return self.sigma_max

    def sigma_tot_fast(self, W2):
        return float(np.interp(W2, self._W2g, self._sigg))

    # -- one event -------------------------------------------------------------- #
    def _nucleon(self):
        if self._fermi is None:
            return np.array([self.mN, 0.0, 0.0, 0.0]), None, self.mN
        p_spec = self._fermi.draw(self.rng)
        struck4, spec4, mstar = split_deuteron(p_spec, self.struck)
        if mstar < self.floor:
            self.n_fermi_reject += 1
            return None, None, None
        return struck4, spec4, mstar

    def generate(self, nev, progress=None):
        ev = []
        while len(ev) < nev:
            self.n_trials += 1
            k = self.brems.draw(self.rng)
            k4 = np.array([k, 0.0, 0.0, k])
            N4, S4, mstar = self._nucleon()
            if N4 is None:
                continue
            P = k4 + N4
            s = mink(P)
            if s <= (self.mV + self.mR) ** 2:
                self.n_below += 1
                continue
            sig = self.sigma_tot_fast(s)
            if sig <= 0:
                self.n_below += 1
                continue
            r = sig / self.sigma_max
            if r > 1.0:
                self.n_over += 1
                self.max_over = max(self.max_over, r)
            if self.rng.random() > r:
                continue
            t = self.draw_t(s)
            if t is None:
                continue
            # build V and N' in the gamma N CM, then boost to the lab
            W = np.sqrt(s)
            beta = P[1:] / P[0]
            B = boost_matrix(beta)              # lab -> CM
            k_cm = B @ k4
            Ev, pv = two_body_cm(W, self.mV, self.mR)
            Eg = k_cm[0]
            # cos(theta) between photon and meson in the CM from t = mV^2 - 2 Eg (Ev - pv cos)
            cos_th = (self.mV ** 2 - t - 2 * Eg * Ev) / (-2 * Eg * pv)
            cos_th = min(1.0, max(-1.0, cos_th))
            sin_th = np.sqrt(1 - cos_th ** 2)
            phi = 2 * np.pi * self.rng.random()
            R = rotation_from_z(k_cm[1:])
            pV_cm = R @ (pv * np.array([sin_th * np.cos(phi), sin_th * np.sin(phi), cos_th]))
            V_cm = np.array([Ev, *pV_cm])
            Rc_cm = np.array([W - Ev, *(-pV_cm)])
            Bi = boost_matrix(-beta)            # CM -> lab
            V4, Rc4 = Bi @ V_cm, Bi @ Rc_cm
            # vertex: density proportional to the upstream thickness
            u = self.rng.random()
            vz = self.vz[0] + (self.vz[1] - self.vz[0]) * np.sqrt(u)
            e = dict(beam=k4, target=N4, meson=V4, recoil=Rc4, k=k, W2=s, t=t, vz=vz)
            if S4 is not None:
                e["spectator"], e["spectator_pid"] = S4, CONFIG[self.struck][2]
                e["p_fermi"], e["m_star"] = float(np.linalg.norm(S4[1:])), float(mstar)
            ev.append(e)
            if progress and len(ev) % progress == 0:
                print(f"  {len(ev)}/{nev}", flush=True)
        return ev

    def report(self):
        acc = 1.0 if self.n_trials == 0 else None
        return dict(trials=self.n_trials, fermi_rejected=self.n_fermi_reject, below_threshold=self.n_below,
                    sigma_max_nb=self.sigma_max, W2_at_sigma_max=self._W2_at_max,
                    overweight=self.n_over, max_overweight=self.max_over,
                    photons_per_electron=self.brems.norm * self.X_over_X0,
                    X_over_X0=self.X_over_X0, brems_norm_per_X0=self.brems.norm)


# --------------------------------------------------------------------------- #
# LUND
# --------------------------------------------------------------------------- #
def write_photo_lund(path, events, channel, sf, target_pid, comments):
    det_pid = CHANNEL_PID[channel]
    rec_pid = PID["n"] if sf.topology.missing == "n" else PID["p"]
    with open(path, "w") as fh:
        for line in comments:
            fh.write("# " + line + "\n")
        for e in events:
            parts = []
            d = e.get("daughters")
            if d is not None and d[0] is not None and channel in DECAY_PID:
                d1, d2, md = DECAY_PID[channel]
                parts.append([det_pid, sf.topology.m_detected, e["meson"], 0, 0, 3 if "spectator" not in e else 4])
                parts.append([rec_pid, sf.topology.m_missing, e["recoil"], 1, 0, 0])
                if "spectator" in e:
                    parts.append([e["spectator_pid"], M_P if e["spectator_pid"] == 2212 else M_N, e["spectator"], 1, 0, 0])
                parts.append([d1, md, d[0], 1, 1, 0])
                parts.append([d2, md, d[1], 1, 1, 0])
            else:
                parts.append([det_pid, sf.topology.m_detected, e["meson"], 1, 0, 0])
                parts.append([rec_pid, sf.topology.m_missing, e["recoil"], 1, 0, 0])
                if "spectator" in e:
                    parts.append([e["spectator_pid"], M_P if e["spectator_pid"] == 2212 else M_N, e["spectator"], 1, 0, 0])
            fh.write(f"{len(parts):4d} {1:4d} {1:4d} {0.0:6.3f} {0.0:6.3f} {22:5d} {e['k']:12.8f} {target_pid:5d} {0:4d} {1.0:12.6e}\n")
            for i, (pid, m, p4, status, parent, daughter) in enumerate(parts, 1):
                fh.write(f"{i:4d} {0.0:6.2f} {status:3d} {pid:8d} {parent:3d} {daughter:3d} "
                         f"{p4[1]:17.10e} {p4[2]:17.10e} {p4[3]:17.10e} {p4[0]:17.10e} {m:12.8f} "
                         f"{0.0:8.4f} {0.0:8.4f} {e['vz']:8.4f}\n")


# --------------------------------------------------------------------------- #
# card and driver
# --------------------------------------------------------------------------- #
def parse_card(path):
    cfg = dict(kmax=None, tmin=-6.0, tmax=-0.001, nev=10000, seed=1, fermi="none", fermi_mstar_min=0.10,
               target_length_cm=5.0, target_X0_cm=X0_CM["LD2"], vz="-5.5 -0.5", idecay=1, maj_safety=1.5,
               outroot=".")
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = (x.strip() for x in line.split("=", 1))
        cfg[key] = val
    for key in ("ebeam", "kmin", "kmax", "tmin", "tmax", "fermi_mstar_min", "target_length_cm", "target_X0_cm", "maj_safety"):
        if cfg.get(key) is not None:
            cfg[key] = float(cfg[key])
    for key in ("nev", "seed", "idecay"):
        cfg[key] = int(cfg[key])
    cfg["vz"] = tuple(float(x) for x in str(cfg["vz"]).split())
    if cfg["kmax"] is None:
        cfg["kmax"] = cfg["ebeam"]
    return cfg


def run(card_path, part=None, nev=None, seed_offset=0):
    cfg = parse_card(card_path)
    sf = models.get(cfg["model"])
    ch = cfg["model"].split(".")[0]
    g = PhotoGenerator(sf, cfg["ebeam"], cfg["kmin"], cfg["kmax"], (cfg["tmin"], cfg["tmax"]),
                       fermi=cfg["fermi"], fermi_m_star_min=cfg["fermi_mstar_min"], seed=cfg["seed"] + seed_offset,
                       maj_safety=cfg["maj_safety"], target_length_cm=cfg["target_length_cm"],
                       target_X0_cm=cfg["target_X0_cm"], vz=cfg["vz"])
    t0 = time.time(); g.setup(); t_setup = time.time() - t0
    n = nev or cfg["nev"]
    t0 = time.time(); ev = g.generate(n, progress=max(n // 5, 1) if part is None else None); t_gen = time.time() - t0
    if cfg["idecay"] and ch in DECAY:
        rng = np.random.default_rng(cfg["seed"] + seed_offset + 1)
        for e in ev:
            e["daughters"] = decay_two_body(e["meson"], DECAY[ch][2], rng)
    tag = "" if part is None else f"_{part:02d}"
    d = os.path.join(cfg["outroot"], ch, f"{_date.today():%Y%m%d}_{cfg['ebeam']:g}_photo_v1")
    for sub in ("lund", "inputs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    target_pid = 1000010020 if cfg["fermi"] != "none" else (2112 if g.struck == "n" else 2212)
    rep = g.report(); rep.update(accepted=len(ev), setup_s=t_setup, gen_s=t_gen,
                                 mean_sigma_nb=None)
    # <sigma> over the accepted photon spectrum: sigma_tot weighted by the brems flux = accepted/trials * sigma_max
    rep["sigma_flux_weighted_nb"] = g.sigma_max * len(ev) / max(g.n_trials, 1)
    rep["yield_per_electron_per_nucleon_nb"] = rep["sigma_flux_weighted_nb"] * rep["photons_per_electron"]
    comments = ["code: exclurad_py photo (untagged bremsstrahlung photoproduction)",
                f"model: {cfg['model']}", f"ebeam: {cfg['ebeam']}", f"k range: {cfg['kmin']} {cfg['kmax']}",
                f"t range: {cfg['tmin']} {cfg['tmax']}", f"fermi: {cfg['fermi']}", f"fermi_mstar_min: {cfg['fermi_mstar_min']}",
                f"target: length {cfg['target_length_cm']} cm, X0 {cfg['target_X0_cm']} cm, X/X0 used {g.X_over_X0:.5f}",
                f"photons per electron in k range: {rep['photons_per_electron']:.4e}",
                f"sigma_max_nb: {g.sigma_max:.5g}  flux-weighted sigma_nb: {rep['sigma_flux_weighted_nb']:.5g}",
                f"overweight: {g.n_over} of {g.n_trials} trials (max x{g.max_over:.2f})",
                "header per event: beam pid 22, beam energy = k; events are UNWEIGHTED",
                f"host: {socket.gethostname().split('.')[0]}"]
    lund = os.path.join(d, "lund", f"{ch}_exclurad_py_photo_{cfg['ebeam']:g}GeV{tag}.lund")
    write_photo_lund(lund, ev, ch, sf, target_pid, comments)
    with open(os.path.join(d, "inputs", f"config{tag}.json"), "w") as fh:
        json.dump(dict(card=cfg, report=rep), fh, indent=1, default=str)
    return lund, rep


def _worker(args):
    card, part, nev = args
    return run(card, part=part, nev=nev, seed_offset=1000 * part)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    card = sys.argv[1]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    cfg = parse_card(card)
    if jobs <= 1:
        lund, rep = run(card)
        print(json.dumps(rep, indent=1, default=str)); print("wrote", lund)
        return
    from concurrent.futures import ProcessPoolExecutor
    per = cfg["nev"] // jobs
    print(f" splitting {cfg['nev']} events across {jobs} processes")
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for lund, rep in ex.map(_worker, [(card, i + 1, per) for i in range(jobs)]):
            print(f" {os.path.basename(lund)}: {rep['accepted']} events, {1e3 * rep['gen_s'] / max(rep['accepted'], 1):.2f} ms/event, "
                  f"overweight {rep['overweight']} (max x{rep['max_overweight']:.2f}), <sigma> {rep['sigma_flux_weighted_nb']:.4g} nb")


if __name__ == "__main__":
    main()
