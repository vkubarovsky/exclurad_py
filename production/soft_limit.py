"""Soft-limit closure: radiative events with v -> 0 must look like Born.

A radiative sample restricted to vanishing v is, physically, the Born
sample.  So the azimuthal harmonics of the rad sample below a small v cut
have to converge on the norad sample's, up to the radiative modulation
itself (a few per cent).  Nothing else in the validation suite tests the
generator's phi handling directly -- the acceptance gate looks at v
fractions, and delta_profiles compares against the quadrature, which
shares the same convention and so cannot see a convention error at all.

This check found its first bug on 2026-08-17.  A G1 sample generated
WITHOUT EXACT_ACCEPT gave cos(phi) = +83.2% below v = 0.002 where its own
Born sample gave +5.7%, and cos(2phi) came out with the opposite sign.
The cause was the eta table: without stage-2 exact accept, events are
accepted against the INTERPOLATED eta, and in that narrow forward box the
interpolation is poor (the builder warns, rms 1.028 where eta spans
0.87-36).  A phi-dependent error of that size skews the sample.  The same
test on the production-scheme samples converges properly: norad +28.30%,
rad below 0.002 +31.99%.

    RAD_GLOB=... NORAD_GLOB=... python3 production/soft_limit.py <channel>
Env: EBEAM (default 10.6), VSOFT (default 0.002), TOL (default 8, in
percentage points on each harmonic), NPRED (default 1500; 0 disables the
weighted prediction).

SHARPENED 2026-08-18 on VPK's point.  "rad(v -> 0) == norad" is not what
the physics says: a soft radiative event still carries the virtual and
soft correction, and that correction is itself phi-dependent -- its own
cos(phi) harmonic runs around +12%.  So the honest statement is

    rad(v < vsoft)  ==  norad weighted by eta(v < vsoft)

evaluated at each Born event's own kinematics.  Comparing against the bare
Born forced a tolerance of 8 percentage points just to absorb the
correction we were supposed to be testing; against the weighted prediction
the residual is the generator's error alone.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_channel import MESON_PID, MESON_M  # noqa: E402

M_P = 0.93827208816


def read(files, mpid, nmax=400000):
    """(e', p, meson) four-vectors [px,py,pz,E] per event.

    LUND particle lines carry 14 fields; event headers carry 10 (this code)
    or 15 (exclugen), so "not 14 fields" identifies a header in either.
    """
    out = []
    for fn in files:
        el = pr = ms = None
        for line in open(fn):
            f = line.split()
            if len(f) != 14:
                el = pr = ms = None
                continue
            pid, st = int(f[3]), int(f[2])
            p = (float(f[6]), float(f[7]), float(f[8]), float(f[9]))
            if pid == 11 and st == 1:
                el = p
            elif pid == 2212:
                pr = p
            elif pid == mpid:
                ms = p
            if el and pr and ms:
                out.append(el + pr + ms)
                el = pr = ms = None
                if len(out) >= nmax:
                    return np.array(out)
    return np.array(out) if out else np.zeros((0, 12))


def kin(ev, E0, m_meson, ps):
    """(v, phi_Trento of the meson in degrees)."""
    el, pr, ms = ev[:, 0:4], ev[:, 4:8], ev[:, 8:12]
    det = pr if ps else ms
    mU2 = m_meson ** 2 if ps else M_P ** 2
    X = np.array([0.0, 0.0, E0, E0 + M_P]) - el - det
    v = X[:, 3] ** 2 - np.einsum("ij,ij->i", X[:, :3], X[:, :3]) - mU2
    beam = np.array([0.0, 0.0, E0])
    q = beam[None, :] - el[:, :3]
    qh = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    a = np.cross(qh, beam[None, :])
    b = np.cross(qh, ms[:, :3])
    den = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    cp = np.einsum("ij,ij->i", a, b) / den
    sp = np.einsum("ij,ij->i", np.cross(beam[None, :], ms[:, :3]), qh) / den
    return v, np.degrees(np.arctan2(sp, cp)) % 360.0


def kin_full(ev, E0, m_meson, ps):
    """(Q2, W2, t) per event -- what the quadrature needs to predict eta."""
    el, pr, ms = ev[:, 0:4], ev[:, 4:8], ev[:, 8:12]
    det = pr if ps else ms
    nu = E0 - el[:, 3]
    pmag = np.linalg.norm(el[:, :3], axis=1)
    Q2 = 2.0 * E0 * el[:, 3] * (1.0 - el[:, 2] / np.maximum(pmag, 1e-12))
    W2 = M_P ** 2 + 2.0 * M_P * nu - Q2
    if ps:
        t = 2.0 * M_P * (M_P - det[:, 3])
    else:
        q4 = np.array([0.0, 0.0, E0, E0])[None, :] - el
        d4 = q4 - det
        t = d4[:, 3] ** 2 - np.einsum("ij,ij->i", d4[:, :3], d4[:, :3])
    return np.stack([Q2, W2, t], axis=1)


def all_harmonics(phi_deg, edges, kmax=4, weights=None):
    """cos(k phi) and sin(k phi) amplitudes in percent, plus the stat error.

    Only cos(phi) and cos(2phi) may be nonzero in an unpolarised Born
    sample: the cross section decomposition stops at 2phi, and the sin
    terms carry the beam helicity, which every production card averages
    over (ihel = 0).  Everything else is a null test.

    Measured on the 2026-08-18 pi0 sample (100k Born): cos1 +14.04 +- 0.45,
    cos2 -59.57 +- 0.45, and all six of sin1, sin2, cos3, sin3, cos4, sin4
    within 2.5 sigma of zero.  For the phi channel all eight vanish, as
    they must -- sigma_TT and sigma_LT are zero in diffrad_vm.

    The radiative sample is where this earns its keep.  Splitting pi0's rad
    events by inelasticity: at v < 0.002 the harmonics sit on the Born
    values, by v = 0.05-0.20 cos(phi) has collapsed from +14 to +1 (3.3
    sigma) and cos(2phi) has grown from -60 to -85 (6.4 sigma).  That is
    the radiative correction to the azimuthal moments, and it is large.
    cos(3phi) stays at zero throughout, which is not automatic: the
    reconstructed phi of a radiative event is not the vertex phi, and that
    mixing could have generated one.
    """
    ctr = 0.5 * (edges[:-1] + edges[1:])
    p = np.radians(ctr)
    h, _ = np.histogram(phi_deg, bins=edges, weights=weights)
    n_eff = len(phi_deg) if weights is None else \
        weights.sum() ** 2 / np.maximum((weights ** 2).sum(), 1e-30)
    if h.sum() <= 0 or n_eff < 200:
        return {}
    y = h / h.sum()
    err = 2 * 100 / np.sqrt(2 * n_eff)
    out = {}
    for k in range(1, kmax + 1):
        for fn, nm in ((np.cos, "cos"), (np.sin, "sin")):
            out[f"{nm}{k}"] = (2 * np.mean(y * fn(k * p)) / y.mean() * 100, err)
    return out


def harmonics(phi_deg, edges):
    ctr = 0.5 * (edges[:-1] + edges[1:])
    p = np.radians(ctr)
    h, _ = np.histogram(phi_deg, bins=edges)
    n = h.sum()
    if n == 0:
        return np.nan, np.nan, np.nan
    y = h / n
    c1 = 2 * np.mean(y * np.cos(p)) / y.mean() * 100
    c2 = 2 * np.mean(y * np.cos(2 * p)) / y.mean() * 100
    # binomial error on each harmonic, propagated through the same weights
    err = 2 * np.sqrt(np.sum((np.cos(p) / y.mean()) ** 2 * y * (1 - y) / n)) \
        / len(y) * 100
    return c1, c2, err


def main(ch):
    E0 = float(os.environ.get("EBEAM", "10.6"))
    vsoft = float(os.environ.get("VSOFT", "0.002"))
    tol = float(os.environ.get("TOL", "8"))
    ps = ch in ("pi0", "eta")
    mpid, m_meson = MESON_PID[ch], MESON_M[ch]
    rad = sorted(glob.glob(os.environ["RAD_GLOB"]))
    nor = sorted(glob.glob(os.environ["NORAD_GLOB"]))
    if not rad or not nor:
        sys.exit("need RAD_GLOB and NORAD_GLOB")
    edges = np.linspace(0, 360, 13)

    vn, pn = kin(read(nor, mpid), E0, m_meson, ps)
    vr, pr_ = kin(read(rad, mpid), E0, m_meson, ps)
    b1, b2, be = harmonics(pn, edges)
    print(f"[{ch}] E={E0} ГэВ, мягкий порог v < {vsoft}, допуск {tol} п.п.")
    print(f"[{ch}] {'образец':<16} {'n':>8} {'cos phi':>10} {'cos 2phi':>10}")
    print(f"[{ch}] {'norad':<16} {len(pn):8d} {b1:+9.2f}% {b2:+9.2f}%")
    ok = True
    for th in (vsoft, 10 * vsoft, 50 * vsoft):
        sel = vr < th
        if sel.sum() < 500:
            print(f"[{ch}] rad v<{th:<10.4g} мало событий ({sel.sum()})")
            continue
        c1, c2, _ = harmonics(pr_[sel], edges)
        tag = "  <-- мягкий" if th == vsoft else ""
        print(f"[{ch}] {'rad v<'+format(th,'.4g'):<16} {sel.sum():8d} "
              f"{c1:+9.2f}% {c2:+9.2f}%{tag}")
        if th == vsoft:
            d1, d2 = abs(c1 - b1), abs(c2 - b2)
            ok = (d1 < tol) and (d2 < tol)
            print(f"[{ch}] отклонение в мягком пределе: cos {d1:.2f} п.п., "
                  f"cos2 {d2:.2f} п.п.")
    # ---- weighted prediction: norad * eta(v < vsoft), per-event ---------- #
    npred = int(os.environ.get("NPRED", "1500"))
    if npred > 0:
        from exclurad_py.core.rc import rc_factor
        from validate_channel import parse_card, get_model
        card = parse_card(ch)
        sf = get_model(card, ch)
        tk = getattr(sf.topology, "t_kind", "photon")
        mU2 = m_meson ** 2 if ps else M_P ** 2
        rng = np.random.default_rng(20260818)
        sub = rng.choice(len(pn), size=min(npred, len(pn)), replace=False)
        w = np.zeros(len(sub))
        kq = kin_full(read(nor, mpid), E0, m_meson, ps)
        for j, i in enumerate(sub):
            Q2, W2, t = kq[i]
            if hasattr(sf, "t_nucl"):
                sf.t_nucl = float(t)
            try:
                w[j] = rc_factor(Ebeam=E0, Q2=Q2, W2=W2, t=t,
                                 phi=np.radians(pn[i]), sfmodel=sf, h=+1,
                                 mx2_cut=vsoft + mU2)["eta"]
            except Exception:
                w[j] = np.nan
        good = np.isfinite(w) & (w > 0)
        if good.sum() > 200:
            hp = all_harmonics(pn[sub][good], np.linspace(0, 360, 25),
                               weights=w[good])
            hs = all_harmonics(pr_[vr < vsoft], np.linspace(0, 360, 25))
            print(f"[{ch}] предсказание norad x eta(v<{vsoft}) на "
                  f"{good.sum()} точках, <eta> = {w[good].mean():.4f}")
            print(f"[{ch}] {'гармоника':>10} {'предсказано':>16} "
                  f"{'rad мягкие':>16} {'разница':>14}")
            for k in ("cos1", "cos2"):
                a, ea = hp[k]
                b, eb = hs[k]
                d, sd = abs(b - a), np.hypot(ea, eb)
                print(f"[{ch}] {k+'phi':>10} {a:+8.2f}±{ea:4.2f}% "
                      f"{b:+8.2f}±{eb:4.2f}% {d:7.2f} п.п. ({d/sd:.1f}s)")

    # ---- absolute scale ------------------------------------------------- #
    # The harmonics above are relative amplitudes; multiplied by the total
    # cross section they become the actual azimuthal moments in nb.  With
    # dsigma/dphi = (sigma_tot / 2pi) (1 + a1 cos phi + a2 cos 2phi), the
    # cos(k phi) moment is sigma_tot * a_k / 2.
    sig = {}
    for tag, pat in (("norad", os.environ["NORAD_GLOB"]),
                     ("rad", os.environ["RAD_GLOB"])):
        vals = []
        for f in glob.glob(os.path.join(
                os.path.dirname(os.path.dirname(pat)), "inputs",
                "config_*.json")):
            try:
                import json
                r = json.load(open(f)).get("report", {})
                if r.get("sigma_nb"):
                    vals.append(float(r["sigma_nb"]))
            except Exception:
                pass
        if vals:
            sig[tag] = float(np.mean(vals))
    if sig:
        hB = all_harmonics(pn, np.linspace(0, 360, 25))
        print(f"[{ch}] абсолютная нормировка (sigma по отчёту генератора):")
        for tag in ("norad", "rad"):
            if tag in sig:
                print(f"[{ch}]   sigma_{tag:<5} = {sig[tag]:.6g} нб "
                      f"(по коробке карточки)")
        if "norad" in sig and hB:
            st = sig["norad"]
            print(f"[{ch}]   sigma_U (phi-усреднённое) = {st:.6g} нб")
            for k, nm in (("cos1", "LT-момент"), ("cos2", "TT-момент")):
                a, ea = hB[k]
                print(f"[{ch}]   {nm:<12} = {st*a/200:+.6g} ± "
                      f"{st*ea/200:.2g} нб   ({a:+.2f}% от sigma_U)")
        if "rad" in sig and "norad" in sig:
            print(f"[{ch}]   sigma_rad / sigma_born = "
                  f"{sig['rad']/sig['norad']:.4f}  <- поправка по коробке")

    # full harmonic content: cos1/cos2 are physics, the other six are null
    fine = np.linspace(0, 360, 25)
    hb = all_harmonics(pn, fine)
    hr = all_harmonics(pr_[vr < vsoft], fine)
    if hb and hr:
        print(f"[{ch}] {'гармоника':>10} {'norad':>16} {'rad мягкие':>16}")
        nulls = []
        for k in sorted(hb, key=lambda s: (s[3:], s[:3])):
            a, ea = hb[k]
            b, eb = hr[k]
            mark = ""
            if k not in ("cos1", "cos2"):
                nulls.append(abs(a) / ea)
                mark = "  нуль-тест" if abs(a) < 3 * ea else "  <-- НЕ НОЛЬ"
            print(f"[{ch}] {k+'phi':>10} {a:+8.2f}±{ea:4.2f}% "
                  f"{b:+8.2f}±{eb:4.2f}%{mark}")
        bad = sum(1 for z in nulls if z >= 3)
        print(f"[{ch}] нулевых гармоник вне 3 сигм: {bad}/{len(nulls)}")
        ok = ok and bad == 0

    print(f"[{ch}] VERDICT: {'PASS' if ok else 'FAIL'} — мягкий предел "
          f"{'сходится к Борну' if ok else 'НЕ сходится к Борну'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "pi0"))
