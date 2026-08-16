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
from validate_channel import parse_card, get_model, read_events, MESON_M  # noqa

M_P = 0.93827208816
E0 = 10.6
BLUE, GREEN = "#0072B2", "#009E73"
OUT_DIR = os.path.expanduser("~/Library/CloudStorage/OneDrive-JeffersonLab/"
                             "Jlab_OneDrive/Work/exclurad_py")


def observables(ev, ch):
    """(Q2, W, -t, phi_deg) per event -- the four profile axes."""
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


def quad_delta(ch, card, axis, edges, rng, nq):
    """delta from the quadrature, per bin of `axis`, others integrated."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pts = []
        while len(pts) < nq:
            Q2 = rng.uniform(*card["Q2"])
            W2 = rng.uniform(*card["W2"])
            t = rng.uniform(*card["t"])
            ph = rng.uniform(0.0, 2 * np.pi)
            val = {"Q2": Q2, "W": np.sqrt(max(W2, 0.0)), "mt": -t,
                   "phi": np.degrees(ph)}[axis]
            if lo <= val < hi:
                pts.append((Q2, W2, t, ph))
        jobs = max(1, (os.cpu_count() or 2) - 1)
        chunks = [pts[i::jobs] for i in range(jobs)]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            res = list(ex.map(_quad_chunk, [(ch, card, c) for c in chunks]))
        so = sum(r[0] for r in res)
        sb = sum(r[1] for r in res)
        out.append(so / sb if sb > 0 else np.nan)
    return np.array(out)


def main(ch):
    card = parse_card(ch)
    rad = sorted(glob.glob(os.environ["RAD_GLOB"]))
    nor = sorted(glob.glob(os.environ["NORAD_GLOB"]))
    if not rad or not nor:
        sys.exit("need RAD_GLOB and NORAD_GLOB")
    ev_r, _ = read_events(rad, ch)
    ev_n, _ = read_events(nor, ch)
    Nr, Nn = len(ev_r), len(ev_n)
    s_obs = float(os.environ.get("SIG_OBS", "1"))
    s_born = float(os.environ.get("SIG_BORN", "1"))
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
          ("phi", r"$\phi$ [deg]", np.linspace(0, 360, 13))]

    rng = np.random.default_rng(20260816)
    nq = int(os.environ.get("NQUAD", "60"))
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
        dq = quad_delta(ch, card, name, edges, rng, nq)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(ctr, dq, "*-", ms=11, color=GREEN, mec="k", mew=.4,
                label="квадратура exclurad", zorder=5)
        ax.errorbar(ctr[ok], d[ok], yerr=e[ok], fmt="o", ms=5, color=BLUE,
                    capsize=2, label="генератор rad/norad")
        ax.set_xlabel(lab, fontsize=9)
        ax.set_ylabel(r"$\delta = \sigma_{obs}/\sigma_{Born}$", fontsize=9)
        ax.grid(alpha=.3, lw=.5)
        ax.tick_params(labelsize=8)
        res[name] = dict(centres=ctr.tolist(), gen=d.tolist(),
                         err=e.tolist(), quad=dq.tolist())
        with np.errstate(invalid="ignore"):
            pull = (d - dq) / e
        good = np.isfinite(pull)
        if good.any():
            print(f"[{ch}] {name}: max |pull| = {np.abs(pull[good]).max():.1f}"
                  f"   mean gen/quad = {np.nanmean(d/dq):.4f}", flush=True)
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
