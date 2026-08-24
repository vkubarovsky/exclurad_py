"""Что меняет Брейт-Вигнер: rho и phi, розыгрыш массы включён против выключен.

    python3 production/bw_look.py <кат.BW=0> <кат.BW=1> -c rho [-o p.png]

Смотреть только на пик бессмысленно: он и так обязан появиться.  Важно, что
масса тянет за собой -- передачу t, порог по W и, главное, дочерние треки,
потому что в детекторе видны именно они, а не мезон.  У rho с шириной 150
МэВ импульсы пи+пи- разъезжаются заметно; у phi ширина 4 МэВ, и там всё
должно остаться на месте.  Оба ответа одинаково содержательны -- второй
показывает, что розыгрыш не тащит за собой ничего лишнего.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_lund import read, MESON                      # noqa: E402

BLUE, VERM = "#0072B2", "#D55E00"
M_P = 0.93827208816


def load(run_dir, channel):
    f = glob.glob(os.path.join(run_dir, channel, "*", "lund", "*.lund"))
    if not f:
        sys.exit("нет lund в " + run_dir)
    return read(f[0], channel), f[0]


def mass_of(p4):
    p4 = np.asarray(p4)
    return np.sqrt(np.maximum(p4[..., 3] ** 2 - np.sum(p4[..., :3] ** 2, -1), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_off")
    ap.add_argument("dir_on")
    ap.add_argument("-c", "--channel", default="rho")
    ap.add_argument("-o", "--out", default="bw_look.png")
    a = ap.parse_args()
    ch = a.channel
    lab, dec = MESON[ch][2], MESON[ch][3]
    d0, f0 = load(a.dir_off, ch)
    d1, f1 = load(a.dir_on, ch)
    print("BW=0: %s (%d)" % (f0, d0["N"]))
    print("BW=1: %s (%d)" % (f1, d1["N"]))

    def dau(d, i):
        """Импульс и полярный угол i-й дочерней частицы в лаборатории."""
        p = np.asarray(d["p_da"] if i == 0 else d["p_db"])
        pm = np.sqrt(np.sum(p[:, 1:] ** 2, 1))
        th = np.degrees(np.arccos(np.clip(p[:, 3] / np.maximum(pm, 1e-12),
                                          -1, 1)))
        return pm, th

    p0, t0 = dau(d0, 0)
    p1, t1 = dau(d1, 0)

    def opening(d):
        """Угол разлёта дочерних в лаборатории -- он прямо задан массой."""
        a, b = np.asarray(d["p_da"]), np.asarray(d["p_db"])
        na = np.linalg.norm(a[:, 1:], axis=1)
        nb = np.linalg.norm(b[:, 1:], axis=1)
        c = np.einsum("ij,ij->i", a[:, 1:], b[:, 1:]) / np.maximum(na * nb, 1e-12)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    panels = [
        ("масса мезона", r"$M_{%s}$ (ГэВ)" % lab, d0["Mrec"], d1["Mrec"], None),
        ("передача", r"$-t$ (ГэВ$^2$)", -np.asarray(d0["t"]),
         -np.asarray(d1["t"]), None),
        ("энергия в с.ц.м.", r"$W$ (ГэВ)", d0["W"], d1["W"], None),
        ("импульс $h^+$", r"$|p|$ (ГэВ)", p0, p1, None),
        ("угол $h^+$", r"$\theta$ (град)", t0, t1, None),
        # Недостающая масса для Борна -- ровно масса протона в каждом
        # событии, смотреть там нечего.  Угол разлёта содержателен: он
        # задаётся массой мезона напрямую.
        ("угол разлёта пары", r"$\theta_{h^+h^-}$ (град)",
         opening(d0), opening(d1), None),
    ]

    fig = plt.figure(figsize=(16.5, 7.4), dpi=135)
    outer = fig.add_gridspec(2, 3, hspace=.40, wspace=.26)
    for k, (title, xl, x0, x1, _) in enumerate(panels):
        row, col = divmod(k, 3)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[2.4, 1],
                                            hspace=.06)
        # sharex обязателен: нижняя панель рисует только бины, где населены
        # ОБЕ гистограммы, и при фиксированной массе это один-единственный
        # бин.  Без связи осей её подписи (единственные видимые) оказываются
        # от совсем другой шкалы, чем гистограмма над ними.
        at = fig.add_subplot(inner[0])
        ab = fig.add_subplot(inner[1], sharex=at)
        x0 = np.asarray(x0, float)
        x1 = np.asarray(x1, float)
        # Диапазон -- по КАЖДОЙ выборке отдельно, потом объединение.
        # При фиксированной массе выборка это дельта-функция, и общие
        # процентили схлопнули бы окно на ней, срезав весь хвост
        # Брейта-Вигнера -- то есть спрятав ровно то, что показываем.
        q0 = np.percentile(x0, [0.2, 99.8])
        q1 = np.percentile(x1, [0.2, 99.8])
        lo, hi = min(q0[0], q1[0]), max(q0[1], q1[1])
        if hi <= lo:
            lo, hi = lo - 1e-3, hi + 1e-3
        e = np.linspace(lo, hi, 41)
        c = .5 * (e[1:] + e[:-1])
        h0, _ = np.histogram(x0, bins=e)
        h1, _ = np.histogram(x1, bins=e)
        at.step(c, h0, where="mid", lw=1.4, color=BLUE,
                label="масса фиксирована")
        at.step(c, h1, where="mid", lw=1.6, color=VERM,
                label="Брейт-Вигнер")
        at.set_ylabel("событий", fontsize=8)
        at.tick_params(labelsize=7, labelbottom=False)
        at.grid(alpha=.25, lw=.4)
        at.set_title(title, fontsize=10)
        ok = (h0 > 10) & (h1 > 10)
        r = np.full(len(c), np.nan)
        er = np.full(len(c), np.nan)
        r[ok] = h1[ok] / h0[ok]
        er[ok] = r[ok] * np.sqrt(1.0 / h1[ok] + 1.0 / h0[ok])
        ab.axhline(1.0, color="0.4", lw=.8)
        ab.errorbar(c[ok], r[ok], yerr=er[ok], fmt="o", ms=2.6, lw=.8,
                    color="black")
        fin = np.isfinite(r)
        if fin.any():
            m = np.nanmax(np.abs(r[fin] - 1))
            ab.set_ylim(1 - 1.25 * m - .02, 1 + 1.25 * m + .02)
        at.set_xlim(lo, hi)
        ab.set_xlabel(xl, fontsize=9)
        ab.set_ylabel("BW / фикс.", fontsize=7)
        ab.tick_params(labelsize=7)
        ab.grid(alpha=.25, lw=.4)
        if k == 0:
            at.legend(frameon=False, fontsize=8.5)
    fig.suptitle(r"$%s$: что меняет розыгрыш массы (%d событий, Борн, $%s$)"
                 % (lab, d1["N"], dec[2] if dec else ""),
                 fontsize=12, fontweight="bold")
    fig.savefig(a.out, bbox_inches="tight")
    print("\n%-22s %10s %10s %8s" % ("величина", "фикс.", "BW", "сдвиг"))
    for title, _, x0, x1, _ in panels:
        m0, m1 = np.mean(x0), np.mean(x1)
        print("%-22s %10.4f %10.4f %7.2f%%"
              % (title, m0, m1, 100 * (m1 / m0 - 1) if m0 else float("nan")))
    print("\nзаписано:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
