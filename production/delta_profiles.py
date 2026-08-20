"""Radiative correction delta from the GENERATOR vs the exclurad quadrature.

delta(bin) = sigma_obs(bin) / sigma_Born(bin), measured end-to-end:

    delta(bin) = [n_rad(bin)/N_rad * sigma_obs] / [n_norad(bin)/N_norad * sigma_born]

Both samples come from the SAME generator (iborn=0 / iborn=1), so the ratio
tests the radiative machinery only -- not two different codes.  Sequences are
uncorrelated (rad draws extra randoms per event), so errors add:

    sigma(delta)/delta = sqrt(1/n_rad + 1/n_norad)

Hence: 1-D profiles with 10-15 bins (not the published 4-D grid -- 1% there
would need ~1e8 events), and an ASYMMETRIC sample size.  Born generation is
~100x cheaper per event, so a 10x larger norad sample makes the error almost
purely radiative: sqrt(1e-4 + 1e-5) = 1.05% instead of 1.41%.

The reference is the quadrature itself, evaluated in the same bins with the
production conventions (scheme A, frozen kf).  It carries no statistics, so
the error band shown is entirely the generator's.

    python3 production/delta_profiles.py <channel>
Env: RAD_GLOB, NORAD_GLOB, SIG_OBS, SIG_BORN (nb), NQUAD (points/bin, dflt 60)
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_channel import expand_glob, parse_card, get_model, read_events, MESON_M  # noqa

M_P = 0.93827208816
E0 = 10.6
BLUE, GREEN = "#0072B2", "#009E73"
OUT_DIR = os.path.expanduser("~/Library/CloudStorage/OneDrive-JeffersonLab/"
                             "Jlab_OneDrive/Work/exclurad_py")


def observables(ev, ch):
    """(Q2, W, -t, phi_deg) per event -- the four profile axes.

    phi is column 3: the core's phi, which since 2026-08-17 is the TRENTO
    azimuth of the produced meson in every channel (topology.phi_offset
    absorbs the pi that pi0/eta pick up from detecting the proton).
    read_events recovers it by undoing that offset on the detected
    particle, which is exact for radiative events too.

    Do NOT swap in column 5 (the measured meson azimuth).  It equals
    column 3 only when nothing is radiated; for rad events the photon
    pushes the meson off the plane, so the reference -- which merely
    mirrors -- and the generator stop describing the same quantity.
    Tried on 2026-08-17: max pull went 3.4 -> 15.4.
    """
    Q2 = ev[:, 0]
    W = np.sqrt(np.maximum(ev[:, 1], 0.0))
    mt = -ev[:, 2]
    phi = np.degrees(ev[:, 3]) % 360.0
    return dict(Q2=Q2, W=W, mt=mt, phi=phi)


def _quad_chunk(args):
    """Exact delta at sampled points inside one bin (sigma_obs weighted)."""
    ch, card, pts = args
    from exclurad_py.core.rc import rc_factor
    sf = get_model(card, ch)
    tk = getattr(sf.topology, "t_kind", "photon")
    mU2 = MESON_M[ch] ** 2 if ch in ("pi0", "eta") else M_P ** 2
    mx2 = (card["vcut"] + mU2) if card["vcut"] > 0 else None
    so = sb = 0.0
    for Q2, W2, t, ph in pts:
        if hasattr(sf, "t_nucl"):
            sf.t_nucl = float(t)
        try:
            r = rc_factor(Ebeam=card["ebeam"], Q2=Q2, W2=W2, t=t, phi=ph,
                          sfmodel=sf, h=+1, mx2_cut=mx2, t_kind=tk)
        except Exception:
            continue
        o, b = r["sigma_obs"], r["sigma_born"]
        if np.isfinite(o) and np.isfinite(b) and b > 0:
            so += o
            sb += b
    return so, sb


def quad_at_points(ch, card, pts):
    """Exact delta at GIVEN kinematic points, parallelised.

    The points are the generator's own NORAD events, i.e. already distributed
    proportional to sigma_Born -- the exact measure the bin-averaged delta is
    defined with.  So a bin's reference value is the PLAIN MEAN of delta over
    its points, whose error is driven by the spread of delta itself (~10%),
    not by the heavy tail of sigma_obs.  Sampling uniformly and weighting by
    sigma_obs (the first version) let a single outlier dominate a bin: the
    phi profile came out scattering 0.7-1.85 where delta must be nearly flat.
    """
    jobs = max(1, (os.cpu_count() or 2) - 1)
    chunks = [pts[i::jobs] for i in range(jobs)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(_delta_chunk, [(ch, card, c) for c in chunks]))
    out = np.full(len(pts), np.nan)
    for j, r in enumerate(res):
        out[j::jobs] = r
    return out


def _delta_chunk(args):
    ch, card, pts = args
    from exclurad_py.core.rc import rc_factor
    sf = get_model(card, ch)
    tk = getattr(sf.topology, "t_kind", "photon")
    mU2 = MESON_M[ch] ** 2 if ch in ("pi0", "eta") else M_P ** 2
    mx2 = (card["vcut"] + mU2) if card["vcut"] > 0 else None
    out = []
    for Q2, W2, t, ph in pts:
        if hasattr(sf, "t_nucl"):
            sf.t_nucl = float(t)
        try:
            r = rc_factor(Ebeam=card["ebeam"], Q2=Q2, W2=W2, t=t, phi=ph,
                          sfmodel=sf, h=+1, mx2_cut=mx2, t_kind=tk)
            o, b = r["sigma_obs"], r["sigma_born"]
            out.append(o / b if (np.isfinite(o) and np.isfinite(b) and b > 0)
                       else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def main(ch):
    card = parse_card(ch)
    rad = expand_glob(os.environ["RAD_GLOB"])
    nor = expand_glob(os.environ["NORAD_GLOB"])
    if not rad or not nor:
        sys.exit("need RAD_GLOB and NORAD_GLOB")
    ev_r, _ = read_events(rad, ch)
    ev_n, _ = read_events(nor, ch)
    Nr, Nn = len(ev_r), len(ev_n)
    s_obs = float(os.environ.get("SIG_OBS", "1"))
    s_born = float(os.environ.get("SIG_BORN", "1"))
    shape_mode = os.environ.get("SIG_OBS") is None
    norm = (s_obs / Nr) / (s_born / Nn)
    print(f"[{ch}] rad {Nr}, norad {Nn}; sigma_obs/sigma_born = "
          f"{s_obs/s_born:.4f}", flush=True)

    obs_r = observables(ev_r, ch)
    obs_n = observables(ev_n, ch)
    AX = [("Q2", r"$Q^2$ [GeV$^2$]", np.linspace(*card["Q2"], 13)),
          ("W", r"$W$ [GeV]", np.linspace(np.sqrt(card["W2"][0]),
                                          np.sqrt(card["W2"][1]), 13)),
          ("mt", r"$-t$ [GeV$^2$]", np.linspace(0.0,
                                                min(-card["t"][0], 3.0), 13)),
          ("phi", r"$\phi_h$ (Тренто, мезон) [deg]", np.linspace(0, 360, 13))]

    rng = np.random.default_rng(20260816)
    nq = int(os.environ.get("NQUAD", "6000"))
    sub = rng.choice(len(ev_n), size=min(nq, len(ev_n)), replace=False)
    pts = ev_n[sub, :4]
    print(f"[{ch}] опорные точки: {len(pts)} norad-событий", flush=True)
    dref = quad_at_points(ch, card, pts.tolist())
    obs_ref = observables(ev_n[sub], ch)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=140)
    res = {}
    for ax, (name, lab, edges) in zip(axes.ravel(), AX):
        hr, _ = np.histogram(obs_r[name], bins=edges)
        hn, _ = np.histogram(obs_n[name], bins=edges)
        ok = (hr > 20) & (hn > 20)
        d = np.full(len(hr), np.nan)
        e = np.full(len(hr), np.nan)
        d[ok] = norm * hr[ok] / hn[ok]
        e[ok] = d[ok] * np.sqrt(1.0 / hr[ok] + 1.0 / hn[ok])
        dq = np.full(len(hr), np.nan)
        dqe = np.full(len(hr), np.nan)
        idx = np.digitize(obs_ref[name], edges) - 1
        for j in range(len(hr)):
            sel = (idx == j) & np.isfinite(dref)
            if sel.sum() > 10:
                dq[j] = dref[sel].mean()
                dqe[j] = dref[sel].std() / np.sqrt(sel.sum())
        if shape_mode:
            # no absolute cross sections available -> compare SHAPES: both
            # curves normalised to their own mean over populated bins.  The
            # ABSOLUTE level is validated elsewhere (published-2013 RC, 0.88%).
            #
            # Both means MUST be taken over the same bins.  The generator
            # populates the sparse tail of the box; the 3000 reference points
            # do not reach it, so `ok` and `isfinite(dq)` are different sets --
            # for J/psi, six of eleven Q2 bins had events but no reference.
            # Normalising each curve over its own set slid them against each
            # other and charged the difference to the most populated bin: on
            # 2026-08-20 that produced max |pull| of 8.0 (J/psi), 6.8 (phi)
            # and 12.0 (rho) in Q2, all of which fell to 1.1, 2.5 and 7.9 the
            # moment the two means were taken over the common bins.
            both = ok & np.isfinite(dq)
            g = np.nanmean(d[both]); q = np.nanmean(dq[both])
            if g > 0 and q > 0:
                d, e, dq = d / g, e / g, dq / q
        ctr = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(ctr, dq, yerr=dqe, fmt="*-", ms=11, color=GREEN,
                    mec="k", mew=.4, capsize=2,
                    label="квадратура exclurad", zorder=5)
        ax.errorbar(ctr[ok], d[ok], yerr=e[ok], fmt="o", ms=5, color=BLUE,
                    capsize=2, label="генератор rad/norad")
        ax.set_xlabel(lab, fontsize=9)
        ax.set_ylabel(r"$\delta = \sigma_{obs}/\sigma_{Born}$", fontsize=9)
        ax.grid(alpha=.3, lw=.5)
        ax.tick_params(labelsize=8)
        # quad_err too: without it a plot rebuilt from the json can only
        # form (gen-quad)/err, which overstates the pull -- the reference is
        # a mean over a finite number of points and carries its own error.
        res[name] = dict(centres=ctr.tolist(), gen=d.tolist(),
                         err=e.tolist(), quad=dq.tolist(),
                         quad_err=dqe.tolist(),
                         n_rad=hr.tolist(), n_norad=hn.tolist())
        with np.errstate(invalid="ignore"):
            pull = (d - dq) / np.sqrt(e**2 + np.nan_to_num(dqe)**2)
        good = np.isfinite(pull)
        if good.any():
            nref = int((ok & ~np.isfinite(dq)).sum())
            print(f"[{ch}] {name}: max |pull| = {np.abs(pull[good]).max():.1f}"
                  f"   mean gen/quad = {np.nanmean(d/dq):.4f}"
                  + (f"   ({nref} бинов без опоры)" if nref else ""), flush=True)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{ch}: радиационная поправка из генератора против квадратуры "
                 f"(rad {Nr}, norad {Nn})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"delta_{ch}.png")
    fig.savefig(out, bbox_inches="tight")
    json.dump(res, open(os.path.join(OUT_DIR, f"delta_{ch}.json"), "w"),
              indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pi0")
