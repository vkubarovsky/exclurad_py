"""Доказательство поправки: RC(phi) из событий против точной квадратуры.

    RAD_GLOB=... NORAD_GLOB=... python3 production/rc_proof_phi.py pi0 [-o f.png]

Это центральная проверка генератора, и устроена она так, чтобы её нельзя
было пройти случайно.

Слева -- радиационная поправка как функция азимута, полученная ДВУМЯ
независимыми путями:

  * из событий:  delta(phi) = N_rad(phi)/N_born(phi), просто отношение
    заполнений двух выборок, без всякой теории;
  * из квадратуры: rc_factor в тех же точках, прямое интегрирование
    формул Бардина-Шумейко, без всякого розыгрыша.

Совпадение этих двух кривых означает, что плотность генерируемых событий
воспроизводит точный расчёт -- то есть анализ, набивший гистограмму по
нашим событиям, получит правильную поправку сам собой, без перевзвешивания.

Азимут выбран не случайно: по нему поправка меняется сильнее всего (у pi0
от 1.25 до 0.85), и именно азимутальные моменты извлекают из данных.
Ошибка в конвенции phi или в тензорной свёртке видна здесь и почти не
видна в проинтегрированных величинах.

Справа -- отклонение в сигмах и chi2/N.  Ошибка КВАДРАТУРЫ тоже учтена:
она усредняется по конечному числу точек и своя ошибка у неё есть.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_channel import expand_glob, parse_card, read_events  # noqa
from delta_profiles import quad_at_points  # noqa

BLUE, VERM = "#0072B2", "#D55E00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", nargs="?", default="pi0")
    ap.add_argument("-o", "--out", default="rc_proof_phi.png")
    ap.add_argument("-n", "--nbins", type=int, default=18)
    a = ap.parse_args()
    ch = a.channel
    card = parse_card(ch)

    ev_r, _ = read_events(expand_glob(os.environ["RAD_GLOB"]), ch)
    ev_n, _ = read_events(expand_glob(os.environ["NORAD_GLOB"]), ch)
    phi_r = np.degrees(ev_r[:, 3]) % 360.0
    phi_n = np.degrees(ev_n[:, 3]) % 360.0
    print("[%s] rad %d, born %d" % (ch, len(ev_r), len(ev_n)), flush=True)

    edges = np.linspace(0, 360, a.nbins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    hr, _ = np.histogram(phi_r, bins=edges)
    hn, _ = np.histogram(phi_n, bins=edges)

    # Отношение заполнений с общей нормировкой: абсолютных сечений здесь
    # нет, и не нужно -- сравнивается ФОРМА зависимости от азимута, а она
    # и есть то, что путается при ошибке в конвенции.
    d_ev = (hr / hr.sum()) / (hn / hn.sum())
    e_ev = d_ev * np.sqrt(1.0 / hr + 1.0 / hn)

    # Опора: квадратура в точках самих борновских событий, то есть с той
    # же мерой, с какой усредняет выборка.
    nq = int(os.environ.get("NQUAD", "20000"))
    rng = np.random.default_rng(20260823)
    sub = rng.choice(len(ev_n), size=min(nq, len(ev_n)), replace=False)
    print("[%s] опорных точек: %d" % (ch, len(sub)), flush=True)
    dref = quad_at_points(ch, card, ev_n[sub, :4].tolist())
    phi_q = np.degrees(ev_n[sub, 3]) % 360.0
    idx = np.digitize(phi_q, edges) - 1
    d_q = np.full(a.nbins, np.nan)
    e_q = np.full(a.nbins, np.nan)
    for j in range(a.nbins):
        s = (idx == j) & np.isfinite(dref)
        if s.sum() > 10:
            d_q[j] = dref[s].mean()
            e_q[j] = dref[s].std() / np.sqrt(s.sum())
    d_q_n = d_q / np.nanmean(d_q)
    e_q_n = e_q / np.nanmean(d_q)
    d_ev_n = d_ev / np.nanmean(d_ev)
    e_ev_n = e_ev / np.nanmean(d_ev)

    ok = np.isfinite(d_q_n) & (hr > 20) & (hn > 20)
    pull = (d_ev_n - d_q_n) / np.sqrt(e_ev_n ** 2 + np.nan_to_num(e_q_n) ** 2)
    chi2 = float(np.nansum(pull[ok] ** 2) / max(ok.sum() - 1, 1))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=140,
                           gridspec_kw=dict(width_ratios=[2, 1]))
    ax[0].errorbar(ctr[ok], d_q_n[ok], yerr=e_q_n[ok], fmt="-", lw=2.0,
                   color=VERM, label="exact quadrature (EXCLURAD)")
    ax[0].errorbar(ctr[ok], d_ev_n[ok], yerr=e_ev_n[ok], fmt="o", ms=6,
                   color=BLUE, capsize=2.5, label="from events, $N_{rad}/N_{Born}$")
    ax[0].set_xlabel(r"$\phi_h$ (Trento), deg", fontsize=11)
    ax[0].set_ylabel(r"$\delta(\phi)$, normalised to its mean", fontsize=11)
    ax[0].set_xlim(0, 360)
    ax[0].grid(alpha=.3, lw=.5)
    ax[0].legend(frameon=False, fontsize=10)
    ax[0].set_title("%s: radiative correction over the azimuth" % ch, fontsize=11)

    ax[1].axhspan(-2, 2, color="0.88")
    ax[1].axhline(0, color="0.4", lw=.8)
    ax[1].plot(ctr[ok], pull[ok], "o", ms=6, color="black")
    ax[1].set_xlabel(r"$\phi_h$, deg", fontsize=11)
    ax[1].set_ylabel(r"deviation, $\sigma$", fontsize=11)
    ax[1].set_xlim(0, 360)
    ax[1].set_ylim(-5, 5)
    ax[1].grid(alpha=.3, lw=.5)
    ax[1].set_title(r"$\chi^2/N = %.2f$ over %d bins" % (chi2, int(ok.sum())),
                    fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")

    print("\n%6s %12s %12s %8s" % ("phi", "события", "квадратура", "σ"))
    for j in range(a.nbins):
        if ok[j]:
            print("%6.0f %7.4f±%.4f %7.4f±%.4f %8.1f"
                  % (ctr[j], d_ev_n[j], e_ev_n[j], d_q_n[j], e_q_n[j], pull[j]))
    print("\nchi2/N = %.2f по %d бинам, max |σ| = %.1f"
          % (chi2, int(ok.sum()), np.nanmax(np.abs(pull[ok]))))
    print("записано:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
