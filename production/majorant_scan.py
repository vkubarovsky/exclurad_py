"""Подбор сетки мажоранты: во что обходится её грубость.

    python3 production/majorant_scan.py cards/rad_jpsi.txt [шейпы через ;]

Эффективность разыгрывания -- это отношение среднего сечения к среднему
значению мажоранты.  Мажоранта кусочно-постоянна и берёт в каждой клетке
максимум, поэтому чем сильнее сечение меняется ВНУТРИ клетки, тем дороже
обходится каждое событие.  У J/psi коробка по t тянется от -1e-4 до -7, и
при сетке 8x8x8x12 эффективность выходит 0.38 % -- из тысячи проб выживают
четыре.

Считается без единого разыгранного события: интеграл мажоранты берётся
прямо по её клеткам, интеграл сечения -- равномерным Монте-Карло по
коробке.  Борновское сечение стоит микросекунды, поэтому перебрать десяток
сеток дешевле, чем один раз ошибиться и оплатить это боевым счётом.

Запас (MAJ_SAFETY) входит множителем: при 4.0 эффективность не может
превысить 0.25 просто по определению.  Здесь он показан отдельно от формы,
потому что лечатся они по-разному: форма -- измельчением, запас -- только
тогда, когда мажоранта перестала пробиваться.
"""
import contextlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exclurad_py import models                                # noqa: E402
from exclurad_py.runcard import parse                         # noqa: E402
from exclurad_py.generator.events import Box, Generator       # noqa: E402


@contextlib.contextmanager
def _env(**kw):
    """Временно выставить переменные окружения и вернуть как было.

    Без возврата вызывающий получает MAJ_SAFETY=1.0 на весь процесс, то есть
    мажоранту без запаса, и не догадывается об этом: Generator читает
    переменную в конструкторе, а поставил её не он.
    """
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build(cfg, shape, safety, seed=12345):
    ch = cfg["model"].split(".")[0]
    kw = dict(t_nucl=cfg["t_nucl"]) if ch in ("pi0", "eta") else {}
    sf = models.get(cfg["model"], **kw)
    sf.Ebeam = float(cfg["ebeam"])
    mx2 = cfg["vcut"] + sf.mU ** 2 if cfg["vcut"] > 0 else None
    with _env(MAJ_SHAPE=",".join(str(x) for x in shape), MAJ_SAFETY=safety):
        g = Generator(sf, Ebeam=cfg["ebeam"],
                      box=Box(Q2=cfg["Q2"], W2=cfg["W2"], t=cfg["t"]),
                      mx2_cut=mx2, helicity=cfg["ihel"], seed=seed,
                      born_only=True,             # eta ~ 1: мешает, не помогает
                      cuts=dict(y=cfg["y"], xB=cfg["xB"], iacc=cfg["iacc"]),
                      itarg=0, vz=(0.0, 0.0))
        t0 = time.time()
        g.setup()
    return g, time.time() - t0


def integral_M(g):
    """Интеграл мажоранты по коробке -- прямо по её клеткам."""
    e = g.maj["edges"]
    vol = 1.0
    for k, ax in (("Q2", 0), ("W2", 1), ("t", 2), ("phi", 3)):
        d = np.abs(np.diff(e[k]))
        vol = vol * d.reshape([-1 if i == ax else 1 for i in range(4)])
    return float((g.maj["M"] * vol).sum())


def integral_sigma(g, cfg, npt, seed=999):
    """Интеграл сечения равномерным МК, с теми же катами, что у генератора."""
    rng = np.random.default_rng(seed)
    Q2 = rng.uniform(*cfg["Q2"], npt)
    W2 = rng.uniform(*cfg["W2"], npt)
    t = rng.uniform(*cfg["t"], npt)
    ph = rng.uniform(0.0, 2.0 * np.pi, npt)
    volume = ((cfg["Q2"][1] - cfg["Q2"][0]) * (cfg["W2"][1] - cfg["W2"][0])
              * abs(cfg["t"][1] - cfg["t"][0]) * 2.0 * np.pi)
    s = 0.0
    n = 0
    for j in range(npt):
        if not (g._pass_yxb(Q2[j], W2[j]) and g._pass_electron_acc(Q2[j], W2[j])):
            continue
        g._set_event_t(t[j])
        b = g._born(Q2[j], W2[j], t[j], ph[j], +1)
        if b is not None and np.isfinite(b) and b > 0:
            s += b
        n += 1
    return s / npt * volume, n


def main():
    card = sys.argv[1] if len(sys.argv) > 1 else "cards/rad_jpsi.txt"
    shapes = sys.argv[2] if len(sys.argv) > 2 else \
        "8,8,8,12;12,8,12,12;16,8,16,12;20,8,24,12;24,8,32,16"
    npt = int(os.environ.get("NPT", "40000"))
    cfg = parse(card)
    print("карточка: %s\n" % card, flush=True)

    ref = None
    print("%-16s %8s %10s %12s %10s" %
          ("сетка", "клеток", "сборка, с", "инт. мажор.", "эффект."))
    for sh in shapes.split(";"):
        shape = tuple(int(x) for x in sh.split(","))
        g, dt = build(cfg, shape, 1.0)
        if ref is None:
            ref, nkeep = integral_sigma(g, cfg, npt)
            print("(интеграл сечения %.6g по %d точкам, годных %d)\n"
                  % (ref, npt, nkeep), flush=True)
        IM = integral_M(g)
        print("%-16s %8d %10.1f %12.6g %9.3f%%"
              % (sh, int(np.prod(shape)), dt, IM, 100.0 * ref / IM), flush=True)
    print("\nПри MAJ_SAFETY=S эффективность делится на S: 4.0 -> потолок 25 %.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
