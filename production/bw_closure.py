"""Замыкание по Брейту-Вигнеру: спектр масс из событий против предсказания.

    python3 production/bw_closure.py cards/born_rho.txt <файл.lund> [-o p.png]

Зачем отдельная проверка.  Розыгрыш массы сам по себе уже сверен с
diffrad_vm.f90 -- но в генераторе масса не остаётся при себе: она входит в
кинематику, в границы по t и в сечение, и событие после этого проходит
отбор.  Поэтому в выборке масса распределена НЕ как Брейт-Вигнер, а как

    p(m)  ~  BW(m) / N(W2)  *  Int dV sigma_born(V; m) ,

и средняя масса rho уезжает с 0.894 (чистый BW) на ~0.85.  Такой сдвиг
одинаково хорошо объясняется и правильным отбором, и ошибкой в проводке
массы, так что верить ему на слово нельзя -- его надо предсказать.

Что здесь независимо, а что нет.  Интеграл берётся равномерным розыгрышем
по коробке, без мажоранты, без accept-reject, без цикла генератора -- то
есть проверяется именно РОЗЫГРЫШ.  Сечение считается тем же born_xsec, что
и в генераторе: его физика проверяется в другом месте, здесь общая.  Такая
проверка ловит ровно тот класс ошибок, которого стоит бояться при проводке:
масса разыграна, но не дошла до сечения; границы по t считаются при
номинальной массе; мажоранта не пересчитана и перекос съеден отбором.

Нормировка N(W2) -- не деталь.  Потолок розыгрыша min(sqrt(W2) - m_p,
m0 + 5*Gamma) зависит от события, и в углу малых W2 он режет резонанс.
Усечённый BW нормирован на СВОЙ отрезок, поэтому события из этого угла
входят в предсказание с большим весом.  Без этого множителя предсказание
разъезжается с выборкой на краях -- и разъезд выглядел бы как ошибка кода.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exclurad_py import models                                   # noqa: E402
from exclurad_py.runcard import parse                            # noqa: E402
from exclurad_py.generator.events import Box, Generator          # noqa: E402
from exclurad_py.generator.breit_wigner import (                 # noqa: E402
    DELTA_WIDTH, M_DAUGHTER, RUNNING_WIDTH, WIDTH, _running_width)

BLUE, VERM, GREY = "#0072B2", "#D55E00", "#666666"
DAU_PID = {"rho": 211, "phi": 321, "omega": 211, "jpsi": 11}


def bw_density(channel, m, m0):
    """Плотность по m, которую даёт sample_mass -- в аналитическом виде.

    Замена m^2 = m0^2 + m0*G*tan(theta) при равномерном theta даёт
    p(m^2) ~ m0*G*bw_con, а отбраковка на бегущую ширину домножает это на
    bw_run/bw_con * (G(m)/G) * (m/m0).  В произведении:

        p(m^2) ~ m * G(m) / [ (m^2-m0^2)^2 + m0^2 G(m)^2 ],   p(m) = 2m p(m^2)

    то есть релятивистский Брейт-Вигнер с бегущей шириной.  Для постоянной
    ширины остаётся обычный вид.  Что эта формула и вправду описывает
    розыгрыш -- проверяется здесь же (--selftest).
    """
    gam = WIDTH[channel]
    m = np.asarray(m, dtype=float)
    if channel not in RUNNING_WIDTH:
        # Постоянная ширина: только якобиан замены, отбраковки нет.
        #   p(m^2) ~ m0*G*bw_con   ->   p(m) = 2m * p(m^2)
        # Множителя m/m0 здесь НЕТ.  Он появляется лишь вместе с
        # отбраковкой на бегущую ширину, и если поставить его и сюда, у phi
        # набегает 2% перекоса по массе -- незаметных на 400 тысячах
        # событий и очевидных на полутора миллионах.
        return 2.0 * m * m0 * gam / ((m * m - m0 * m0) ** 2 + m0 * m0 * gam * gam)
    m_dau = M_DAUGHTER[channel]
    p0 = np.sqrt(max(0.0, 0.25 * m0 * m0 - m_dau * m_dau))
    p = np.sqrt(np.maximum(0.0, 0.25 * m * m - m_dau * m_dau))
    g = gam * (p / p0) ** 3 * (m0 / m)
    # Бегущая ширина: якобиан m0*G*bw_con, умноженный на вероятность
    # принятия bw_run/bw_con * (G(m)/G) * (m/m0).  В произведении m0 и G
    # сокращаются и остаётся релятивистский вид с m^2*G(m) в числителе.
    return 2.0 * m * m * g / ((m * m - m0 * m0) ** 2 + m0 * m0 * g * g)


def bw_binned(channel, edges, m0, k=21):
    """Интеграл плотности по каждому бину, а не значение в центре.

    У phi ширина 4.3 МэВ, а бин выходит около 0.9 МэВ: кривизна внутри
    бина уже не мала, и подмена интеграла центральным значением даёт
    перекос, который на 400 тысячах событий не виден, а на полутора
    миллионах даёт chi2/N около 2.4 -- то есть выглядит как ошибка
    розыгрыша, хотя это ошибка сравнения.
    """
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        x = np.linspace(edges[i], edges[i + 1], k)
        out[i] = np.trapezoid(bw_density(channel, x, m0), x)
    return out


def read_masses(path, channel):
    """Инвариантная масса пары дочерних частиц, событие за событием."""
    pid = DAU_PID[channel]
    out, cur = [], []
    for ln in open(path):
        if ln.startswith("#"):
            continue
        q = ln.split()
        if len(q) != 14:
            if len(cur) == 2:
                p = cur[0] + cur[1]
                out.append(np.sqrt(max(p[3] ** 2 - p[:3] @ p[:3], 0.0)))
            cur = []
            continue
        if abs(int(q[3])) == pid and int(q[2]) == 1:
            cur.append(np.array([float(q[6]), float(q[7]),
                                 float(q[8]), float(q[9])]))
    if len(cur) == 2:
        p = cur[0] + cur[1]
        out.append(np.sqrt(max(p[3] ** 2 - p[:3] @ p[:3], 0.0)))
    return np.array(out)


def predict(cfg, channel, m_grid, npt, seed):
    """p(m) ~ BW(m) * <sigma(m)/N(W2)> по равномерной коробке."""
    kw = dict(t_nucl=cfg["t_nucl"]) if channel in ("pi0", "eta") else {}
    sf = models.get(cfg["model"], **kw)
    sf.Ebeam = float(cfg["ebeam"])
    mx2 = cfg["vcut"] + sf.mU ** 2 if cfg["vcut"] > 0 else None
    g = Generator(sf, Ebeam=cfg["ebeam"],
                  box=Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"]),
                  mx2_cut=mx2, helicity=cfg["ihel"], seed=seed,
                  born_only=True,
                  cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"]),
                  itarg=cfg["itarg"], vz=cfg["vz"])
    m0, gam = float(sf.mV), WIDTH[channel]
    m_min = 2.0 * M_DAUGHTER[channel]
    m_hard = m0 + 5.0 * gam

    rng = np.random.default_rng(seed)
    Q2 = rng.uniform(*cfg["Q2"], npt)
    W2 = rng.uniform(*cfg["W2"], npt)
    t = rng.uniform(*cfg["t"], npt)
    phi = rng.uniform(0.0, 2.0 * np.pi, npt)

    # Нормировка усечённого BW для каждой точки: N(W2) на своём отрезке.
    mm = np.linspace(m_min, m_hard, 400)
    dens = bw_density(channel, mm, m0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1])
                                           * np.diff(mm))])
    m_cap = np.minimum(np.sqrt(W2) - sf.mU, m_hard)
    norm = np.interp(m_cap, mm, cum)

    keep = (norm > 0) & np.array([g._pass_yxb(a, b) and g._pass_electron_acc(a, b)
                                  for a, b in zip(Q2, W2)])
    idx = np.flatnonzero(keep)
    n_use = max(len(idx), 1)
    pred = np.zeros(len(m_grid))
    perr = np.zeros(len(m_grid))
    for i, m in enumerate(m_grid):
        acc = acc2 = 0.0
        for j in idx:
            if m > m_cap[j]:
                continue
            g._mV = float(m)
            g._set_event_t(t[j])
            b = g._born(Q2[j], W2[j], t[j], phi[j], +1)
            if b is not None and np.isfinite(b) and b > 0:
                w = b / norm[j]
                acc += w
                acc2 += w * w
        # Ошибка интеграла -- не украшение: при 1500 точках она того же
        # порядка, что и статистика выборки, и без неё расхождение
        # выглядит значимым там, где значимости нет.
        pred[i] = acc / n_use
        var = max(acc2 / n_use - pred[i] ** 2, 0.0) / n_use
        perr[i] = np.sqrt(var)
    return pred, perr, m0


def selftest(channel, m0, n=400000, seed=3):
    """Совпадает ли аналитическая плотность с самим розыгрышем."""
    from exclurad_py.generator.breit_wigner import sample_mass
    rng = np.random.default_rng(seed)
    m_hi = m0 + 5.0 * WIDTH[channel]
    s = np.array([sample_mass(channel, m0, 99.0, rng) for _ in range(n)])
    lo = 2.0 * M_DAUGHTER[channel]
    e = np.linspace(lo, m_hi, 61)
    h, _ = np.histogram(s, bins=e)
    p = bw_binned(channel, e, m0)
    p = p / p.sum() * h.sum()
    ok = h > 20
    chi = float(np.sum((h[ok] - p[ok]) ** 2 / p[ok]) / ok.sum())
    print("[selftest] %s: формула против розыгрыша, chi2/N = %.2f по %d бинам"
          % (channel, chi, int(ok.sum())))
    return chi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("lund")
    ap.add_argument("-o", "--out", default="bw_closure.png")
    ap.add_argument("-n", "--npt", type=int, default=1500)
    ap.add_argument("-b", "--nbins", type=int, default=30)
    a = ap.parse_args()

    cfg = parse(a.card)
    channel = cfg["model"].split(".")[0]
    if WIDTH.get(channel, 0.0) < DELTA_WIDTH:
        sys.exit("%s: ширина ничтожна, проверять нечего" % channel)

    chi_self = selftest(channel, float(models.get(cfg["model"]).mV))

    m = read_masses(a.lund, channel)
    print("событий: %d, <M> = %.5f, rms = %.5f" % (len(m), m.mean(), m.std()))
    lo, hi = m.min(), m.max()
    edges = np.linspace(lo, hi, a.nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    h, _ = np.histogram(m, bins=edges)

    pred, perr, m0 = predict(cfg, channel, ctr, a.npt, cfg["seed"] + 1)
    bw = bw_binned(channel, edges, m0)
    pred, perr = pred * bw, perr * bw
    if pred.sum() <= 0:
        sys.exit("предсказание нулевое -- коробка не даёт сечения")
    scale = h.sum() / pred.sum()
    pred, perr = pred * scale, perr * scale

    ok = (h > 15) & (pred > 15)
    pull = np.zeros(len(ctr))
    pull[ok] = (h[ok] - pred[ok]) / np.sqrt(h[ok] + perr[ok] ** 2)
    chi2 = float(np.sum(pull[ok] ** 2) / max(ok.sum() - 1, 1))

    pure = bw_binned(channel, edges, m0)
    pure = pure / pure.sum() * h.sum()

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=140,
                           gridspec_kw=dict(width_ratios=[2, 1]))
    ax[0].errorbar(ctr, h, yerr=np.sqrt(h), fmt="o", ms=5, color=BLUE,
                   capsize=2.5, label="generator events")
    ax[0].fill_between(ctr, pred - perr, pred + perr, color=VERM, alpha=.25,
                       lw=0)
    ax[0].plot(ctr, pred, "-", lw=2.0, color=VERM,
               label=r"prediction: BW$/N(W^2)\times\int\sigma_{Born}$")
    ax[0].plot(ctr, pure, "--", lw=1.4, color=GREY,
               label="pure Breit-Wigner (reference)")
    ax[0].axvline(m0, color="0.5", lw=.8, ls=":")
    ax[0].set_xlabel(r"$M_{%s}$ (GeV)" % channel, fontsize=11)
    ax[0].set_ylabel("events per bin", fontsize=11)
    ax[0].grid(alpha=.3, lw=.5)
    ax[0].legend(frameon=False, fontsize=9.5)
    ax[0].set_title(r"%s: mass as an event property, $\langle M\rangle$ = %.4f"
                    % (channel, m.mean()), fontsize=11)

    ax[1].axhspan(-2, 2, color="0.88")
    ax[1].axhline(0, color="0.4", lw=.8)
    ax[1].plot(ctr[ok], pull[ok], "o", ms=5, color="black")
    ax[1].set_xlabel(r"$M$ (GeV)", fontsize=11)
    ax[1].set_ylabel(r"deviation, $\sigma$", fontsize=11)
    ax[1].set_ylim(-5, 5)
    ax[1].grid(alpha=.3, lw=.5)
    ax[1].set_title(r"$\chi^2/N$ = %.2f over %d bins" % (chi2, int(ok.sum())),
                    fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")

    print("\nchi2/N = %.2f по %d бинам, max |σ| = %.1f"
          % (chi2, int(ok.sum()), np.max(np.abs(pull[ok]))))
    print("самопроверка формулы: chi2/N = %.2f" % chi_self)
    print("записано:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
