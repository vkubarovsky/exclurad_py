"""How long will this card take?  Ask before starting, not after.

    python3 production/estimate_rate.py <card> [n_probe] [n_jobs]

Runs the card through THE SAME code path as a real run -- exclurad_py.run
with nev replaced by a small number -- and reports ms per written event and
the projected wall time for the card's own nev.

Why it exists, and why it insists on the real path: on 2026-08-18 the J/psi
rate was measured by building a Generator by hand, and the card's cuts (y,
xB and the iacc acceptance preset) were left out of that construction.  The
answer came back 860 ms per event.  The real cost with those cuts was
36957 ms -- a factor of 43 -- and the run was launched on the wrong number
and spent eight hours producing 60% of one part before anyone noticed.

Anything that reconstructs the configuration by hand can omit a field.
This calls run() on a copy of the card with only nev changed, so the only
difference from production is how many events come out.
"""
import os
import re
import shutil
import sys
import tempfile
import time


def main(card, n_probe=25, n_jobs=14):
    from exclurad_py.runcard import parse
    from exclurad_py.run import run

    cfg = parse(card)
    nev_real = int(cfg["nev"])

    # copy the card, change ONLY nev and the output root
    tmp = tempfile.mkdtemp(prefix="rate_")
    probe = os.path.join(tmp, os.path.basename(card))
    body = 0
    out = []
    for line in open(card).read().splitlines():
        if line.split("!")[0].strip():
            body += 1
            if body == 3:                      # nev
                line = re.sub(r"^\s*\S+", str(n_probe), line, count=1)
            elif body == 13:                   # output root
                line = re.sub(r"^\s*\S+", tmp, line, count=1)
        out.append(line)
    open(probe, "w").write("\n".join(out) + "\n")

    print(f"карточка {os.path.basename(card)}: {cfg['model']}, "
          f"nev={nev_real}, iacc={cfg.get('iacc')}, "
          f"y={cfg.get('y')}, xB={cfg.get('xB')}", flush=True)
    print(f"пробный прогон на {n_probe} событий тем же кодом...", flush=True)

    # TWO probes: the marginal cost is the slope, which drops the one-off
    # setup (table build plus majorant, a minute or two) exactly.  A single
    # probe divides that setup over its handful of events and reports a
    # per-event cost several times too high -- the pi0 card came out at
    # 16273 ms/event that way against a true 2000.
    def _probe(nev):
        body = 0
        out = []
        for line in open(card).read().splitlines():
            if line.split("!")[0].strip():
                body += 1
                if body == 3:
                    line = re.sub(r"^\s*\S+", str(nev), line, count=1)
                elif body == 13:
                    line = re.sub(r"^\s*\S+", tmp, line, count=1)
            out.append(line)
        open(probe, "w").write("\n".join(out) + "\n")
        t = time.time()
        r = run(probe, verbose=False)
        return time.time() - t, max(r.get("n", 0), 1), r

    # Three runs, not two.  The first builds the eta table and the majorant
    # into the temp output directory; the next two find them cached.  Without
    # that warm-up the second probe is CHEAPER than the first and the slope
    # comes out negative -- the pi0 card reported -5040 ms/event and a
    # projection of minus ten hours.
    n_small = max(3, n_probe // 5)
    _probe(n_small)                                  # warm-up, discarded
    t_small, k_small, _ = _probe(n_small)
    t_big, k_big, res = _probe(n_probe)
    rep = res.get("report", {})
    per = (t_big - t_small) / max(k_big - k_small, 1)
    if per <= 0:
        per = t_big / max(k_big, 1)
        print("  (наклон непригоден, беру полное время последней пробы)")
    setup = max(t_small - k_small * per, 0.0)
    n = k_big
    print(f"  настройка (таблица + мажоранта): {setup:.0f} с, одноразово")
    print(f"  {1000*per:.0f} мс на записанное событие (без настройки)")
    for k, lab in (("trials", "проб"), ("accepted", "принято ступенью 1"),
                   ("accept2", "принято ступенью 2"),
                   ("cut_yxb", "срезано y/xB"),
                   ("cut_eacceptance", "срезано электроном (рано)"),
                   ("cut_acceptance", "срезано приёмкой (поздно)")):
        if k in rep:
            print(f"  {lab:<28} {rep[k]}")
    if res.get("health"):
        for m in res["health"]:
            print(f"  ЗДОРОВЬЕ: {m}")

    tot = setup + nev_real * per / n_jobs
    h, m = divmod(int(tot // 60), 60)
    print(f"\n  прогноз на {nev_real} событий при {n_jobs} ядрах: "
          f"{h} ч {m:02d} мин")
    if tot > 4 * 3600:
        print("  *** дольше четырёх часов -- стоит пересмотреть до запуска ***")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0], int(a[1]) if len(a) > 1 else 25,
                  int(a[2]) if len(a) > 2 else 14))
