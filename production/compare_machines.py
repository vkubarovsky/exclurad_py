"""Сверка прогонов, сделанных на разных машинах.

    # на каждой машине:
    python3 production/compare_machines.py dump <каталог прогона> <имя.npz> [-c pi0]
    # на любой из них, собрав файлы:
    python3 production/compare_machines.py plot метка=a.npz метка=b.npz ... [-o p.png]

Один код, разные зёрна, разное железо -- распределения обязаны совпадать в
пределах статистики.  Это единственная проверка, которая ловит порчу от
самой машины: другой процессор, другая версия питона, другая сборка numpy.

Гоняются ГИСТОГРАММЫ, а не события: выборка на машину -- сотни мегабайт, а
её гистограммы -- десятки килобайт.  Границы бинов фиксированы в коде, а не
подбираются по данным, иначе наборы окажутся на разных сетках и сравнивать
будет нечего.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_lund import read, MESON  # noqa: E402

M_P = 0.93827208816

# Границы ФИКСИРОВАНЫ: наборы разных машин должны лечь на одну сетку, иначе
# сравнивать нечего.  Но фиксированы они ПО КАНАЛУ, а не на все сразу: эти
# числа были подобраны под пи0, и у J/psi, живущего при Q2 < 0.25 и
# W около 4.2, гистограмма по Q2 от 0.5 до 5.0 вышла бы пустой.
#
# Берутся из КАРТОЧКИ канала -- она одна и та же на всех машинах, поэтому
# сетка остаётся общей, и при этом сама подстраивается под коробку.
BINS_PI0 = {
    "Q2":    (0.5, 5.0, 45),
    "xB":    (0.03, 0.62, 45),
    "mt":    (0.0, 3.0, 45),
    "y":     (0.1, 0.95, 45),
    "W":     (1.8, 4.0, 45),
    "nu":    (1.5, 9.5, 45),
    "pT2":   (0.0, 3.0, 45),
    "phi":   (0.0, 360.0, 36),
    "v_rec": (-0.02, 0.30, 40),
    "Egam":  (0.0, 5.0, 40),
}
BINS = dict(BINS_PI0)          # заполняется set_bins() под нужный канал


def set_bins(channel, card=None):
    """Сетка под коробку канала.  Без карточки остаётся пи0-шная."""
    global BINS
    if card is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        card = os.path.join(here, "cards", "rad_%s.txt" % channel)
    if not os.path.exists(card):
        print("  (карточки %s нет -- сетка остаётся пи0-шной)" % card)
        return
    v = [ln.split("!")[0].strip()
         for ln in open(card) if ln.split("!")[0].strip()]
    W2 = [float(x) for x in v[4].split()]
    Q2 = [float(x) for x in v[5].split()]
    t = [abs(float(x)) for x in v[6].split()]
    vcut = float(v[7].replace("d", "e"))
    y = [float(x) for x in v[13].split()] if len(v) > 13 else [0.1, 0.95]
    xB = [float(x) for x in v[14].split()] if len(v) > 14 else [0.0, 0.62]
    E0 = float(v[1])
    nu = [(W2[0] - 0.8803 + Q2[0]) / (2 * 0.93827),
          (W2[1] - 0.8803 + Q2[1]) / (2 * 0.93827)]
    B = dict(BINS_PI0)
    B["Q2"] = (Q2[0], Q2[1], 45)
    B["W"] = (W2[0] ** 0.5, W2[1] ** 0.5, 45)
    B["mt"] = (min(t), max(t), 45)
    B["y"] = (y[0], y[1], 45)
    B["xB"] = (xB[0], xB[1], 45)
    B["nu"] = (max(nu[0], 0.0), min(nu[1], E0), 45)
    B["pT2"] = (0.0, max(t), 45)
    # v_rec: до ката плюс хвост от радиатора; при vcut = 0 -- до предела
    hi = 0.30 if vcut <= 0 else 1.5 * vcut
    B["v_rec"] = (-0.02, hi, 40)
    B["Egam"] = (0.0, E0 / 2.0, 40)
    BINS = B
LABEL = {
    "Q2": r"$Q^2$ (GeV$^2$)", "xB": r"$x_B$", "mt": r"$-t$ (GeV$^2$)",
    "y": r"$y$", "W": r"$W$ (GeV)", "nu": r"$\nu$ (GeV)",
    "pT2": r"$p_T^2$ (GeV$^2$)", "phi": r"$\phi_h$ (deg)",
    "v_rec": r"$v$ at nominal $E$", "Egam": r"$E_\gamma$ (GeV)",
}


def edges(key):
    lo, hi, n = BINS[key]
    return np.linspace(lo, hi, n + 1)


def recon(files, channel, E0=10.6):
    """Неупругость и азимут при НОМИНАЛЬНОЙ энергии пучка -- как их видит
    анализ, знающий только e' и зарегистрированный адрон.

    Свой разборщик здесь был и искал зарегистрированный адрон как частицу со
    СТАТУСОМ 1.  У векторных мезонов при idecay=1 мезон записан документацией
    со статусом 0, а живут в файле его дочерние, поэтому для rho, phi и J/psi
    не находилось ничего: панели по азимуту и по неупругости выходили пустыми
    и просто не рисовались.  Азимут при этом -- та переменная, по которой
    радиационная поправка меняется сильнее всего.

    Теперь чтение идёт через validate_lund.read, который уже собирает мезон
    из дочерних; заодно исчез второй разборщик LUND.
    """
    det_is_proton = channel in ("pi0", "eta")
    mU2 = (MESON[channel][1] ** 2 if det_is_proton else M_P ** 2)
    vs, phis = [], []
    for path in files:
        d = read(path, channel)
        if not d["N"]:
            continue
        el = np.asarray(d["p_scat"])                 # (E, px, py, pz)
        det = (np.asarray(d["p_rec"]) if det_is_proton
               else np.asarray(d["p_da"]) + np.asarray(d["p_db"]))
        # X = beam + target - e' - detected, всё при НОМИНАЛЬНОЙ энергии
        Xe = E0 + M_P - el[:, 0] - det[:, 0]
        Xp = -el[:, 1:] - det[:, 1:]
        Xp[:, 2] += E0
        vs.extend(Xe ** 2 - np.einsum("ij,ij->i", Xp, Xp) - mU2)

        beam = np.array([0.0, 0.0, E0])
        qv = beam - el[:, 1:]
        nq = np.linalg.norm(qv, axis=1)
        ok = nq > 1e-12
        qh = qv[ok] / nq[ok, None]
        aa = np.cross(qh, beam)
        bb = np.cross(qh, Xp[ok])
        den = np.maximum(np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1),
                         1e-12)
        num = np.einsum("ij,ij->i", np.cross(beam, Xp[ok]), qh) / den
        cos = np.einsum("ij,ij->i", aa, bb) / den
        phis.extend(np.degrees(np.arctan2(num, cos)) % 360.0)
    return np.array(vs), np.array(phis)


def do_dump(run_dir, out, channel):
    files = sorted(glob.glob(os.path.join(
        run_dir, "cyc*", channel, "*_rad_v1", "lund", "*.lund")))
    if not files:
        sys.exit("нет rad-файлов в " + run_dir)
    acc = {k: np.zeros(BINS[k][2], dtype=np.int64) for k in BINS}
    n_tot = 0
    for f in files:
        d = read(f, channel)
        if not d["N"]:
            continue
        n_tot += d["N"]
        d["mt"] = -d["t"]
        for k in ("Q2", "xB", "mt", "y", "W", "nu", "pT2", "Egam"):
            if k in d and len(d[k]):
                h, _ = np.histogram(d[k], bins=edges(k))
                acc[k] += h
    v, ph = recon(files, channel)
    for k, arr in (("v_rec", v), ("phi", ph)):
        if len(arr):
            h, _ = np.histogram(arr, bins=edges(k))
            acc[k] += h
    np.savez(out, n=n_tot, nfiles=len(files), **acc)
    print("записано %s: %d событий из %d файлов" % (out, n_tot, len(files)))


def do_plot(items, outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = []
    for it in items:
        lab, _, path = it.partition("=")
        d = np.load(path)
        data.append((lab or os.path.basename(path), d))
    ref_lab, ref = data[0]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666"]

    fig = plt.figure(figsize=(18, 11), dpi=130)
    outer = fig.add_gridspec(3, 4, hspace=.42, wspace=.30)
    worst = {}
    for i, key in enumerate(BINS):
        row, col = divmod(i, 4)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=.05)
        at = fig.add_subplot(inner[0]); ab = fig.add_subplot(inner[1])
        e = edges(key); c = 0.5 * (e[:-1] + e[1:])
        hr = ref[key].astype(float)
        if hr.sum() == 0:
            at.set_visible(False); ab.set_visible(False); continue
        fr = hr / hr.sum()
        chi_all = []
        for j, (lab, d) in enumerate(data):
            h = d[key].astype(float)
            if h.sum() == 0:
                continue
            f = h / h.sum()
            at.fill_between(c, 0, f, step="mid", lw=0,
                            color=colors[j % len(colors)], alpha=0.15)
            at.step(c, f, where="mid", lw=1.4, color=colors[j % len(colors)],
                    label=lab)
            if j == 0:
                continue
            ok = (h > 20) & (hr > 20)
            ef = np.sqrt(h) / h.sum()
            er = np.sqrt(hr) / hr.sum()
            pull = np.zeros(len(c))
            pull[ok] = (f[ok] - fr[ok]) / np.sqrt(ef[ok] ** 2 + er[ok] ** 2)
            ab.plot(c[ok], pull[ok], "o", ms=2.5,
                    color=colors[j % len(colors)])
            if ok.sum() > 1:
                chi_all.append(float(np.sum(pull[ok] ** 2) / ok.sum()))
        at.set_ylabel("fraction", fontsize=8)
        at.tick_params(labelsize=7, labelbottom=False)
        at.grid(alpha=.25, lw=.4)
        ab.axhspan(-2, 2, color="0.88", zorder=0)
        ab.axhline(0, color="0.4", lw=.7)
        ab.set_ylim(-5, 5)
        ab.set_xlabel(LABEL.get(key, key), fontsize=8)
        ab.set_ylabel(r"$\sigma$ vs first", fontsize=7)
        ab.tick_params(labelsize=7)
        ab.grid(alpha=.25, lw=.4)
        if chi_all:
            worst[key] = max(chi_all)
            ab.text(.03, .82, r"$\chi^2/N$ up to %.2f" % max(chi_all), fontsize=7,
                    transform=ab.transAxes)
    fig.axes[0].legend(frameon=False, fontsize=9)
    tot = ", ".join("%s %d" % (l, int(d["n"])) for l, d in data)
    fig.suptitle("Machine cross-check (reference: %s) — %s" % (ref_lab, tot),
                 fontsize=12, fontweight="bold")
    fig.savefig(outfile, bbox_inches="tight")
    print("\n%-10s %s" % ("величина", "худший χ²/N среди машин"))
    for k, v in sorted(worst.items(), key=lambda x: -x[1]):
        print("  %-8s %6.2f" % (k, v))
    print("\nзаписано:", outfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "plot"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("-c", "--channel", default="pi0")
    ap.add_argument("-o", "--out", default="compare_machines.png")
    # Карточку можно назвать явно.  Иначе она ищется рядом со скриптом, а
    # скрипт бывает скопирован в сторону -- и тогда сетка молча остаётся
    # пи0-шной, что для J/psi означает пустую картинку с видом картинки.
    ap.add_argument("-k", "--card", default=None,
                    help="карточка канала; по умолчанию ../cards/rad_<ch>.txt")
    a = ap.parse_args()
    set_bins(a.channel, a.card)
    if a.mode == "dump":
        do_dump(a.args[0], a.args[1], a.channel)
    else:
        do_plot(a.args, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
