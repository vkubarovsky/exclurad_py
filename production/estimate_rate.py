"""How long will this card take?  Ask before starting, not after.

    python3 production/estimate_rate.py <card> [n_probe] [n_jobs]

Runs the card through THE SAME code path as a real run -- exclurad_py.run
with nev replaced by a small number -- and reports ms per written event and
the projected wall time for the card's own nev.

The projection comes from a PARALLEL probe: n_jobs workers at once, timed
as a whole.  A single process on an idle machine divided by the core count
is not the same number and is not close to it (see the comment at the call
site: 1313 ms against 4540).

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

    # ---- and now the number that actually decides the wall clock -------- #
    # The slope above is ONE process on an otherwise idle machine, and
    # dividing it by n_jobs assumes n_jobs equal, independent cores.  Neither
    # holds: Apple silicon mixes performance and efficiency cores, and the
    # workers contend for memory bandwidth.  Measured on the mac mini for the
    # pi0 G1 card on 2026-08-20 -- 1313 ms alone on an idle machine, 2613 ms
    # for one process on a loaded one, 4540 ms each with fourteen running.
    # The serial estimate said 15 hours for a pass that would have taken 60.
    #
    # So run n_jobs of them at once and time the aggregate.  The table and
    # the majorant are already cached in tmp by the probes above, so this
    # measures generation only.
    #
    # It measures the machine AS IT IS NOW: run it on an idle one, or the
    # answer describes the competition rather than the card.  The ratio to
    # the serial slope is printed for exactly that reason -- around x3.5 is
    # the honest cost of fourteen-way parallelism, much more than that means
    # something else is eating the machine.
    par = _parallel_probe(probe, tmp, n_jobs, max(6, n_probe // 5))
    shutil.rmtree(tmp, ignore_errors=True)
    if par is None:
        print("\n  параллельная проба не удалась, беру последовательную оценку")
        tot = setup + nev_real * per / n_jobs
    else:
        thr, per_par = par
        print(f"\n  под нагрузкой ({n_jobs} процессов): {1000*per_par:.0f} мс "
              f"на событие в каждом, {thr:.2f} событий/с на машину")
        print(f"  замедление против одиночного процесса: "
              f"x{per_par/per:.1f}")
        tot = setup + nev_real / thr
    h, m = divmod(int(tot // 60), 60)
    print(f"\n  прогноз на {nev_real} событий при {n_jobs} ядрах: "
          f"{h} ч {m:02d} мин")
    if tot > 4 * 3600:
        print("  *** дольше четырёх часов -- стоит пересмотреть до запуска ***")
    return 0


def _parallel_probe(card, tmp, n_jobs, n_each):
    """Aggregate throughput with n_jobs workers running at once.

    Uses the production entry point -- run(_part=(i, n, seed)) -- so the
    workers divide the card's nev exactly as a real pass does.
    """
    import subprocess
    body = 0
    out = []
    for line in open(card).read().splitlines():
        if line.split("!")[0].strip():
            body += 1
            if body == 3:
                line = re.sub(r"^\s*\S+", str(n_each * n_jobs), line, count=1)
        out.append(line)
    par_card = os.path.join(tmp, "parallel_" + os.path.basename(card))
    open(par_card, "w").write("\n".join(out) + "\n")
    code = ("import sys;from exclurad_py.run import run;"
            "run(sys.argv[1],verbose=False,"
            "_part=(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])))")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    t = time.time()
    procs = [subprocess.Popen([sys.executable, "-c", code, par_card,
                               str(i), str(n_jobs), str(4200 + 97 * i)],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, env=env)
             for i in range(1, n_jobs + 1)]
    rc = [p.wait() for p in procs]
    dt = time.time() - t
    n_ok = sum(1 for r in rc if r == 0)
    if n_ok == 0 or dt <= 0:
        return None
    n_tot = n_each * n_ok
    return n_tot / dt, dt * n_ok / n_tot


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0], int(a[1]) if len(a) > 1 else 25,
                  int(a[2]) if len(a) > 2 else 14))
