"""Overnight channel validator: sample vs EXACT theory, one channel per call.

    python3 production/validate_channel.py <pi0|eta|rho|phi|jpsi>

Three measurements, all against the same exact quadrature (rc_factor), which
is the validated reference -- generators are never compared to each other:

  1. THEORY tail fractions: sigma_obs-weighted frac(v > x) over the card's
     box, from rc_factor alone (no sampler, no table, no majorant).
     Parallelised over points.
  2. SAMPLE fractions, raw and REWEIGHTED: each event carries
     w = eta_exact / min(eta_table, cap), which makes every weighted
     distribution follow sigma_obs exactly whatever the proposal did.
  3. LUND integrity: PDG masses, decay closure, radiated-photon bookkeeping.

Verdict per threshold: PASS if |weighted - theory| < max(3 sigma_stat, 1.5 pp).
Writes JSON next to the log for the morning report.

Env: SAMPLE_GLOB, ETA_PROPOSAL_CAP (must match generation), NTHEORY (dflt 400).
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

M_P = 0.93827208816
E0 = 10.6
THR = (0.005, 0.010, 0.050, 0.100, 0.300)
NIGHT = os.environ.get(
    "NIGHT_DIR",
    "/private/tmp/claude-501/-Users-vpk/8f359e6f-199b-4043-a168-f687a79578de/"
    "scratchpad/night")

MESON_PID = {"pi0": 111, "eta": 221, "rho": 113, "phi": 333, "jpsi": 443}
MESON_M = {"pi0": 0.1349768, "eta": 0.547862, "rho": 0.77526,
           "phi": 1.019461, "jpsi": 3.096900}
DAUGHTERS = {"pi0": (22, 22), "eta": (22, 22), "rho": (211, -211),
             "phi": (321, -321), "jpsi": (-11, 11)}


def parse_card(ch):
    """Positional run card -- single source of truth for box/vcut/model."""
    vals = []
    with open(os.path.expanduser(f"~/exclurad_py/cards/rad_{ch}.txt")) as fh:
        for line in fh:
            body = line.split("!")[0].strip()
            if body:
                vals.append(body)
    return dict(model=vals[0], ebeam=float(vals[1]),
                W2=tuple(map(float, vals[4].split())),
                Q2=tuple(map(float, vals[5].split())),
                t=tuple(map(float, vals[6].split())),
                vcut=float(vals[7]), t_nucl=float(vals[8]),
                seed=int(vals[10]))


def get_model(card, ch, t_nucl=None):
    from exclurad_py import models
    if ch in ("pi0", "eta"):
        return models.get(card["model"],
                          t_nucl=card["t_nucl"] if t_nucl is None else t_nucl)
    return models.get(card["model"])


def read_events(paths, ch):
    """(Q2, W2, t, phi, v, integrity dict) from LUND, channel-aware.

    detected hadron: proton for pi0/eta (baryon_detected), meson for VM.
    v = MX^2(e', detected) - mU^2.
    """
    mpid = MESON_PID[ch]
    m_meson = MESON_M[ch]
    ps = ch in ("pi0", "eta")
    mU2 = m_meson ** 2 if ps else M_P ** 2
    rows = []
    integ = dict(events=0, malformed=0, bad_mass=0, decay_open=0,
                 with_rad_photon=0)
    for path in paths:
        with open(path) as fh:
            while True:
                line = fh.readline()
                if not line:
                    break
                if line.startswith("#"):
                    continue
                f = line.split()
                if len(f) < 10:
                    continue
                try:
                    npart = int(f[0])
                except ValueError:
                    continue
                el = pr = ms = None
                dsum = np.zeros(4)
                nd = 0
                rad = False
                for i in range(npart):
                    q = fh.readline().split()
                    if len(q) < 11:
                        continue
                    pid, st, par = int(q[3]), int(q[2]), int(q[4])
                    p4 = np.array([float(q[6]), float(q[7]),
                                   float(q[8]), float(q[9])])
                    if pid == 11 and st == 1 and par == 0 and el is None:
                        el = p4
                    elif pid == 2212 and st == 1 and par == 0:
                        pr = p4
                    if pid == mpid:
                        ms = p4
                        mfile = float(q[10])
                        # 1e-4 tolerance: phi is WRITTEN as 1.019412 (the
                        # diffrad_vm transcription value, vector.py:290), a
                        # deliberate 49 keV convention -- not an error this
                        # check should fire on.  Flagged for a daytime
                        # decision: PDG vs diffrad_vm compatibility.
                        if abs(mfile - m_meson) > 1e-4:
                            integ["bad_mass"] += 1
                    if par != 0:
                        dsum += p4
                        nd += 1
                    if pid == 22 and st == 1 and par == 0:
                        rad = True
                integ["events"] += 1
                if rad:
                    integ["with_rad_photon"] += 1
                det = pr if ps else ms
                if el is None or det is None:
                    integ["malformed"] += 1
                    continue
                if ms is not None and nd == 2:
                    if np.abs(dsum - ms).max() > 1e-6:
                        integ["decay_open"] += 1
                nu = E0 - el[3]
                pmag = np.linalg.norm(el[:3])
                Q2 = 2.0 * E0 * el[3] * (1.0 - el[2] / pmag)
                W2 = M_P * M_P + 2.0 * M_P * nu - Q2
                if ps:
                    t = 2.0 * M_P * (M_P - det[3])           # target transfer
                else:
                    q4 = np.array([0, 0, E0, E0]) - el
                    d4 = q4 - det
                    t = d4[3] ** 2 - d4[:3] @ d4[:3]         # photon transfer
                phi = np.arctan2(det[1], det[0])
                X = np.array([0, 0, E0, E0 + M_P]) - el - det
                v = X[3] ** 2 - X[:3] @ X[:3] - mU2
                rows.append((Q2, W2, t, phi, v))
    return np.array(rows), integ


# ---- parallel exact-eta workers (top level: picklable under spawn) -------- #
def _theory_chunk(args):
    ch, card, pts = args
    from exclurad_py.core.rc import rc_factor
    sf = get_model(card, ch)
    tk = getattr(sf.topology, "t_kind", "photon")
    mU2 = MESON_M[ch] ** 2 if ch in ("pi0", "eta") else M_P ** 2
    mx2_tot = (card["vcut"] + mU2) if card["vcut"] > 0 else None
    tot = 0.0
    above = np.zeros(len(THR))
    n_ok = 0
    for Q2, W2, t, ph in pts:
        if hasattr(sf, "t_nucl"):
            sf.t_nucl = float(t)
        try:
            s_tot = rc_factor(Ebeam=card["ebeam"], Q2=Q2, W2=W2, t=t, phi=ph,
                              sfmodel=sf, h=+1, mx2_cut=mx2_tot,
                              t_kind=tk)["sigma_obs"]
            if not np.isfinite(s_tot) or s_tot <= 0:
                continue
            s_x = [rc_factor(Ebeam=card["ebeam"], Q2=Q2, W2=W2, t=t, phi=ph,
                             sfmodel=sf, h=+1, mx2_cut=x + mU2,
                             t_kind=tk)["sigma_obs"] for x in THR]
        except Exception:
            continue
        if not all(np.isfinite(s) for s in s_x):
            continue
        n_ok += 1
        tot += s_tot
        for j, s in enumerate(s_x):
            above[j] += max(s_tot - s, 0.0)
    return tot, above, n_ok


def _eta_chunk(args):
    ch, card, rows = args
    from exclurad_py.core.rc import rc_factor
    sf = get_model(card, ch)
    tk = getattr(sf.topology, "t_kind", "photon")
    mU2 = MESON_M[ch] ** 2 if ch in ("pi0", "eta") else M_P ** 2
    mx2 = (card["vcut"] + mU2) if card["vcut"] > 0 else None
    out = []
    for Q2, W2, t, phi in rows:
        if hasattr(sf, "t_nucl"):
            sf.t_nucl = float(t)
        try:
            e = rc_factor(Ebeam=card["ebeam"], Q2=Q2, W2=W2, t=t, phi=phi,
                          sfmodel=sf, h=+1, mx2_cut=mx2, t_kind=tk)["eta"]
            out.append(e if np.isfinite(e) and e > 0 else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def main(ch):
    card = parse_card(ch)
    os.makedirs(NIGHT, exist_ok=True)
    jobs = max(1, (os.cpu_count() or 2) - 1)
    rng = np.random.default_rng(20260815)
    res = dict(channel=ch, card={k: v for k, v in card.items()})

    # ---- 1. theory ----
    n_th = int(os.environ.get("NTHEORY", "400"))
    pts = [(rng.uniform(*card["Q2"]), rng.uniform(*card["W2"]),
            rng.uniform(*card["t"]), rng.uniform(0, 2 * np.pi))
           for _ in range(n_th)]
    chunks = [pts[i::jobs] for i in range(jobs)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        parts = list(ex.map(_theory_chunk, [(ch, card, c) for c in chunks]))
    tot = sum(p[0] for p in parts)
    above = np.sum([p[1] for p in parts], axis=0)
    n_ok = sum(p[2] for p in parts)
    theory = {x: 100.0 * a / tot for x, a in zip(THR, above)}
    # jackknife over worker chunks: the sigma_obs weights are heavy-tailed,
    # so the theory column carries real MC error that must enter the verdict
    th_err = {}
    good = [p for p in parts if p[0] > 0]
    for j, x in enumerate(THR):
        jk = np.array([100.0 * (above[j] - p[1][j]) / (tot - p[0])
                       for p in good])
        th_err[x] = float(np.sqrt(max(len(jk) - 1, 1) * jk.var()))
    print(f"[{ch}] theory: {n_ok}/{n_th} physical points; "
          f"MC err " + " ".join(f"{th_err[x]:.2f}" for x in THR), flush=True)
    res["theory"] = theory
    res["theory_err"] = {str(k): v for k, v in th_err.items()}
    res["theory_npoints"] = n_ok

    # ---- 2. sample, raw + weighted ----
    pat = os.environ.get("SAMPLE_GLOB", os.path.join(
        os.path.dirname(NIGHT), f"mctest/{ch}/*_rad_v1/lund/*.lund"))
    files = sorted(glob.glob(pat))
    if not files:
        print(f"[{ch}] NO SAMPLE at {pat}", flush=True)
        res["verdict"] = "NO SAMPLE"
        json.dump(res, open(f"{NIGHT}/{ch}.json", "w"), indent=1)
        return
    ev, integ = read_events(files, ch)
    res["integrity"] = integ
    print(f"[{ch}] sample: {len(ev)} events, integrity {integ}", flush=True)

    from exclurad_py.generator import Box
    from exclurad_py.generator.tables import RCTable
    sf = get_model(card, ch)
    mU2 = MESON_M[ch] ** 2 if ch in ("pi0", "eta") else M_P ** 2
    mx2 = (card["vcut"] + mU2) if card["vcut"] > 0 else None
    tab = RCTable(sf, card["ebeam"],
                  Box(Q2=card["Q2"], W2=card["W2"], t=card["t"]),
                  mx2_cut=mx2, helicity=+1)
    tab.build(verbose=False)
    eta_tab = tab(ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3])
    cap = float(os.environ.get("ETA_PROPOSAL_CAP", "0"))
    if cap > 0:
        eta_tab = np.minimum(eta_tab, cap)

    rows = ev[:, :4].tolist()
    chunks = [rows[i::jobs] for i in range(jobs)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        parts = list(ex.map(_eta_chunk, [(ch, card, c) for c in chunks]))
    eta_ex = np.full(len(rows), np.nan)
    for j, r in enumerate(parts):
        eta_ex[j::jobs] = r

    ok = np.isfinite(eta_ex) & np.isfinite(eta_tab) & (eta_tab > 0)
    w = eta_ex[ok] / eta_tab[ok]
    v = ev[ok, 4]
    res["n_used"] = int(ok.sum())
    res["w_stats"] = dict(median=float(np.median(w)), mean=float(w.mean()),
                          max=float(w.max()))

    # ---- 3. verdict ----
    print(f"\n[{ch}]  {'v >':>8} {'raw':>8} {'weighted':>9} {'+-':>6} "
          f"{'THEORY':>8}  verdict", flush=True)
    verdicts = {}
    for x in THR:
        sel = v > x
        raw = 100.0 * sel.mean()
        wt = 100.0 * w[sel].sum() / w.sum()
        # binomial-with-weights error
        err = 100.0 * np.sqrt(np.sum(w[sel] ** 2)) / w.sum()
        th = theory[x]
        tot_err = np.hypot(err, th_err[x])
        ok_x = abs(wt - th) < max(3 * tot_err, 1.5)
        verdicts[x] = dict(raw=raw, weighted=wt, err=err, theory=th,
                           ok=bool(ok_x))
        print(f"[{ch}]  {x:8.3f} {raw:7.2f}% {wt:8.2f}% {err:5.2f}% "
              f"{th:7.2f}%  {'PASS' if ok_x else 'FAIL'}", flush=True)
    res["fractions"] = {str(k): d for k, d in verdicts.items()}
    npass = sum(d["ok"] for d in verdicts.values())
    res["verdict"] = f"{npass}/{len(THR)} PASS"
    clean = (integ["malformed"] == 0 and integ["bad_mass"] == 0
             and integ["decay_open"] == 0)
    res["lund_clean"] = bool(clean)
    print(f"\n[{ch}] VERDICT: {res['verdict']}, LUND "
          f"{'clean' if clean else 'PROBLEMS'}", flush=True)
    json.dump(res, open(f"{NIGHT}/{ch}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pi0")
