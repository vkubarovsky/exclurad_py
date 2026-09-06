"""Отчёт по боевому прогону: rad против norad, двенадцать величин.

    python3 production/report_run.py <каталог прогона> [-c pi0] [-o отчёт.png]

Каталог -- тот, что создаёт run_prod_v1.sh: README.md, card_*.txt и круги
cycNN внутри.  Читаются ВСЕ круги сразу: они делались разными зёрнами и
складываются.

Что рисуется.  Верх каждой клетки -- наложение двух распределений,
нормированных на единичную площадь: борновское и радиационное.  Низ --
их отношение с ошибкой.  Отношение и есть то, ради чего всё считается:
это радиационная поправка как функция каждой переменной, измеренная прямо
на событиях, без всякой квадратуры.

Отдельно печатается сводка: версия кода из шапки LUND, число событий,
число кругов, доля событий с излучённым фотоном, и -- главное -- доля
событий, которые НЕ пройдут эксклюзивный рез, посчитанный по номинальной
энергии пучка.  С включённым радиатором она не ноль (около 0.8 %) и у
борновских тоже: радиатор действует до взаимодействия, каким бы оно ни
было.  Это не изъян -- внешняя потеря увеличивает восстановленную
неупругость, и события уходят из окна наружу.  Аналитику это надо знать
до того, как он удивится.
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
from validate_lund import read, MESON  # noqa: E402

M_P = 0.93827208816
BORN_C, RAD_C = "#0072B2", "#D55E00"

# (ключ, подпись, бинов, диапазон, логарифм по Y, только в rad)
PANELS = [
    ("Q2",     r"$Q^2$ (GeV$^2$)",              50, None,          False, False),
    ("xB",     r"$x_B$",                        50, None,          False, False),
    ("mt",     r"$-t$ (GeV$^2$)",               60, None,          True,  False),
    ("y",      r"$y$",                          50, (0.0, 1.0),    False, False),
    ("W",      r"$W$ (GeV)",                    50, None,          False, False),
    ("nu",     r"$\nu$ (GeV)",                  50, None,          False, False),
    ("pT2",    r"$p_T^2$ of the meson (GeV$^2$)",     50, None,          True,  False),
    ("phi",    r"$\phi_h$ (Trento, deg)",      36, (0.0, 360.0),  False, False),
    ("v_rec",  r"$v$ at nominal $E$ (GeV$^2$)", 50, (-0.02, 0.30), True, False),
    ("MX",     r"$M_X$ of the missing system (GeV)",     50, None,          True,  False),
    ("E_miss", r"missing energy (GeV)",    50, None,          True,  False),
    ("Egam",   r"$E_\gamma$, rad only (GeV)", 50, None,          True,  True),
]


def gather(run_dir, channel, mode):
    """Все круги одного режима в один набор."""
    pat = os.path.join(run_dir, "cyc*", channel, "*_%s_v1" % mode,
                       "lund", "*.lund")
    files = sorted(glob.glob(pat))
    if not files:
        return None, 0, [], []
    parts, cycles = [], set()
    for f in files:
        d = read(f, channel)
        if d["N"]:
            parts.append(d)
        cycles.add(f.split("/cyc")[1].split("/")[0])
    if not parts:
        return None, 0, [], []
    out = {}
    for k in ("Q2", "xB", "t", "y", "W", "Mmiss", "pT2", "Mrec", "nu",
              "E_miss", "Egam", "v"):
        out[k] = np.concatenate([p[k] for p in parts])
    out["mt"] = -out["t"]
    out["N"] = sum(p["N"] for p in parts)
    out["n_ph"] = sum(int(p["has_ph"].sum()) for p in parts)
    return out, len(files), sorted(cycles), files


def raw_recon(files, channel, E0=10.6):
    """v и phi так, как их считает АНАЛИЗ: из e' и зарегистрированного
    адрона, при НОМИНАЛЬНОЙ энергии пучка.

    Считать это из готовых величин validate_lund нельзя: его Mmiss -- масса
    системы (отдача + фотон), то есть истинная, а анализ фотона не видит и
    об индивидуальной потере энергии в мишени не знает.  Разница между
    истинной и восстановленной неупругостью и есть измеряемый эффект.
    """
    det_pid = 2212 if channel in ("pi0", "eta") else MESON[channel][0]
    mU2 = (MESON[channel][1] ** 2 if channel in ("pi0", "eta") else M_P ** 2)
    vs, phis = [], []
    for path in files:
        el = det = None
        for line in open(path):
            q = line.split()
            if len(q) != 14:
                el = det = None
                continue
            pid, st = int(q[3]), int(q[2])
            p4 = np.array([float(q[6]), float(q[7]), float(q[8]), float(q[9])])
            if pid == 11 and st == 1:
                el = p4
            elif pid == det_pid and st == 1:
                det = p4
            if el is not None and det is not None:
                X = np.array([0.0, 0.0, E0, E0 + M_P]) - el - det
                vs.append(X[3] ** 2 - X[:3] @ X[:3] - mU2)
                beam = np.array([0.0, 0.0, E0])
                qv = beam - el[:3]
                nq = np.linalg.norm(qv)
                if nq > 1e-12:
                    qh = qv / nq
                    # плоскость мезона: для pi0/eta мезон недостающий, берём
                    # его как X минус фотон -- проще взять сам X
                    a = np.cross(qh, beam)
                    b = np.cross(qh, X[:3])
                    den = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
                    cp = float(a @ b) / den
                    sp = float(np.cross(beam, X[:3]) @ qh) / den
                    phis.append(np.degrees(np.arctan2(sp, cp)) % 360.0)
                el = det = None
    return np.array(vs), np.array(phis)


def add_recon(d, files, channel):
    """Величины в том виде, в каком их видит анализ.

    M_X здесь -- масса НЕДОСТАЮЩЕЙ системы (для pi0/eta это сам мезон), а
    не то, что даёт validate_lund: его Mmiss -- масса системы "отдача плюс
    фотон", то есть около массы протона, и аналитику она не нужна.
    """
    d["v_rec"], d["phi"] = raw_recon(files, channel)
    mU2 = (MESON[channel][1] ** 2 if channel in ("pi0", "eta") else M_P ** 2)
    d["MX"] = np.sqrt(np.maximum(d["v_rec"] + mU2, 0.0))
    return d


def panel(ax_top, ax_bot, born, rad, key, label, nb, rng, logy, rad_only):
    r = rad.get(key)
    b = None if rad_only else born.get(key)
    if r is None or len(r) == 0:
        ax_top.set_visible(False); ax_bot.set_visible(False)
        return
    if b is not None and len(b) == 0:
        b = None
    if rng is None:
        src = r if b is None else np.concatenate([b, r])
        lo, hi = np.percentile(src, 0.2), np.percentile(src, 99.8)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            # вырожденная величина (масса мезона -- дельта-функция):
            # рисовать нечего, панель честно гасим
            ax_top.set_visible(False); ax_bot.set_visible(False)
            return
        rng = (lo, hi)
    # Ось у обеих панелей ЯВНО из rng.  Нижняя рисует только бины с >20
    # событиями; на редкой выборке она пуста, остаётся с дефолтными
    # пределами +-0.05, и именно её подписи видны -- у верхней они скрыты.
    # На дымовом прогоне amp2026c семь панелей так показали ось "+-0.05"
    # под правильными гистограммами.  В августе каждая нижняя панель была
    # населена, и дефект прятался.
    ax_top.set_xlim(rng[0], rng[1])
    ax_bot.set_xlim(rng[0], rng[1])
    e = np.linspace(rng[0], rng[1], nb + 1)
    hr, _ = np.histogram(r, bins=e)
    fr = hr / max(hr.sum(), 1)
    c = 0.5 * (e[:-1] + e[1:])
    if b is not None:
        hb, _ = np.histogram(b, bins=e)
        fb = hb / max(hb.sum(), 1)
        ax_top.step(c, fb, where="mid", color=BORN_C, lw=1.4, label="Born")
    ax_top.step(c, fr, where="mid", color=RAD_C, lw=1.4, label="radiative")
    ax_top.set_ylabel("fraction", fontsize=8)
    ax_top.tick_params(labelsize=7, labelbottom=False)
    ax_top.grid(alpha=.25, lw=.4)
    if logy:
        ax_top.set_yscale("log")
    if b is None:
        ax_bot.set_visible(False)
        ax_top.set_xlabel(label, fontsize=8)
        ax_top.tick_params(labelbottom=True, labelsize=7)
        return

    ok = (hb > 20) & (hr > 20)
    ratio = np.full(len(c), np.nan)
    err = np.full(len(c), np.nan)
    ratio[ok] = fr[ok] / fb[ok]
    err[ok] = ratio[ok] * np.sqrt(1.0 / hr[ok] + 1.0 / hb[ok])
    ax_bot.axhline(1.0, color="0.5", lw=.8)
    ax_bot.errorbar(c[ok], ratio[ok], yerr=err[ok], fmt="o", ms=2.5,
                    color="black", lw=.8, capsize=1.5)
    ax_bot.set_xlabel(label, fontsize=8)
    ax_bot.set_ylabel("rad/Born", fontsize=7)
    ax_bot.tick_params(labelsize=7)
    ax_bot.grid(alpha=.25, lw=.4)
    if np.isfinite(ratio[ok]).any():
        # Границы по квантилям, а не по медиане плюс-минус: у величин, где
        # борновское распределение почти пусто в хвосте (v по номинальной
        # энергии), отношение улетает на порядки и рамка вида
        # "медиана +- 0.35" показывает пустоту около двадцати двух.
        lo, hi = np.nanpercentile(ratio[ok], [5, 95])
        pad = max(0.05, 0.25 * (hi - lo))
        ax_bot.set_ylim(max(0.0, lo - pad), hi + pad)


def version_of(run_dir, channel):
    f = sorted(glob.glob(os.path.join(run_dir, "cyc*", channel, "*",
                                      "lund", "*.lund")))
    if not f:
        return "неизвестно"
    tag = commit = "?"
    for line in open(f[0]):
        if not line.startswith("#"):
            break
        if "version_tag:" in line:
            tag = line.split(":", 1)[1].strip()
        elif "git_commit:" in line:
            commit = line.split(":", 1)[1].strip()
    return "%s (%s)" % (tag, commit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-c", "--channel", default="pi0")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    ch = a.channel
    out = a.out or os.path.join(a.run_dir, "report_%s.png" % ch)

    born, nb_files, cyc_b, fb = gather(a.run_dir, ch, "born")
    rad, nr_files, cyc_r, fr = gather(a.run_dir, ch, "rad")
    if born is None or rad is None:
        sys.exit("нет данных: born %s, rad %s"
                 % ("есть" if born else "нет", "есть" if rad else "нет"))
    born = add_recon(born, fb, ch)
    rad = add_recon(rad, fr, ch)

    ver = version_of(a.run_dir, ch)
    vcut = 0.20
    over = float(np.mean(rad["v_rec"] > vcut)) if len(rad["v_rec"]) else 0.0
    over_b = float(np.mean(born["v_rec"] > vcut)) if len(born["v_rec"]) else 0.0

    print("=" * 66)
    print("Отчёт по прогону: %s" % a.run_dir)
    print("Канал %s, версия кода %s" % (ch, ver))
    print("-" * 66)
    print("  Борн:           %9d событий, %3d файлов, кругов %d"
          % (born["N"], nb_files, len(cyc_b)))
    print("  С излучением:   %9d событий, %3d файлов, кругов %d"
          % (rad["N"], nr_files, len(cyc_r)))
    print("  с фотоном:      %6.2f %% радиационных" % (100 * rad["n_ph"] / max(rad["N"], 1)))
    print("  выше реза v>%.2f по НОМИНАЛЬНОЙ энергии:" % vcut)
    print("      Борн  %6.3f %%      с излучением %6.3f %%"
          % (100 * over_b, 100 * over))
    print("      Обе доли ненулевые, и это ПРАВИЛЬНО: радиатор работает и в")
    print("      борновском режиме -- электрон теряет энергию в мишени ДО")
    print("      взаимодействия, каким бы оно ни было.  Ноль был бы только")
    print("      при itarg=0.  Разница между строками -- добавка от")
    print("      внутреннего излучения поверх внешнего.")
    print("=" * 66)

    # Каждой величине -- пара осей, между парами широкий зазор: в первой
    # редакции подписи осей затирались следующим рядом и были не видны.
    fig = plt.figure(figsize=(18, 13), dpi=130)
    outer = fig.add_gridspec(3, 4, hspace=.42, wspace=.30)
    for i, (key, label, nb, rng, logy, ronly) in enumerate(PANELS):
        row, col = divmod(i, 4)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[2.4, 1], hspace=.05)
        panel(fig.add_subplot(inner[0]), fig.add_subplot(inner[1]),
              born, rad, key, label, nb, rng, logy, ronly)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="upper right",
               frameon=False, fontsize=10)
    fig.suptitle("%s, version %s — Born %d, radiative %d events"
                 % (ch, ver, born["N"], rad["N"]),
                 fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    print("записано:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
