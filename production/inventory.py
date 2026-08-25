"""Опись выборки: что есть на этой машине и в каком оно состоянии.

    python3 production/inventory.py <каталог прогона> [ещё каталоги ...]
    python3 production/inventory.py --clean <каталог>      удалить пустые файлы

Зачем.  Выборка растёт на четырёх машинах, кругами, месяцами, и рано или
поздно возникает вопрос «что у нас вообще есть».  Отвечать на него ручным
подсчётом файлов нельзя: файл нулевого размера выглядит как файл, круг может
быть недоигран, а часть -- снята надзором на записи.  Здесь всё это названо
поимённо.

Событий НЕ пересчитываем по строкам: это гигабайты чтения.  Число написано в
шапке каждого файла (`run.n_events`), её и читаем -- один заголовок вместо
всего файла.  Пустой файл шапки не имеет и попадает в отдельную графу.
"""
import argparse
import glob
import json
import os
import sys


def head_events(path):
    """Число событий из шапки LUND; None, если шапки нет (файл пуст/битый)."""
    try:
        with open(path) as fh:
            for _ in range(80):
                line = fh.readline()
                if not line:
                    break
                if line.startswith("# run.n_events:"):
                    return int(line.split(":")[1])
                if not line.startswith("#"):
                    break
    except OSError:
        pass
    return None


def scan(run_dir):
    ch = None
    cycles = sorted(glob.glob(os.path.join(run_dir, "cyc*")))
    modes = {}
    empties = []
    for cyc in cycles:
        for mode in ("born", "rad"):
            files = glob.glob(os.path.join(cyc, "*", "*_%s_v1" % mode,
                                           "lund", "*.lund"))
            if not files:
                continue
            ch = ch or os.path.basename(os.path.dirname(
                os.path.dirname(os.path.dirname(files[0]))))
            d = modes.setdefault(mode, dict(files=0, empty=0, events=0,
                                            bytes=0, cycles=set()))
            d["cycles"].add(os.path.basename(cyc))
            for f in files:
                sz = os.path.getsize(f)
                n = head_events(f) if sz else None
                if n is None:
                    d["empty"] += 1
                    empties.append(f)
                else:
                    d["files"] += 1
                    d["events"] += n
                    d["bytes"] += sz
    # здоровье и версия -- из описаний частей
    health_bad, tags = 0, set()
    for cfg in glob.glob(os.path.join(run_dir, "cyc*", "*", "*", "inputs",
                                      "config*.json")):
        try:
            d = json.load(open(cfg))
        except (OSError, ValueError):
            continue
        h = d.get("health")
        if h and h != "clean" and h != []:
            health_bad += 1
        t = (d.get("stamp") or {}).get("version_tag")
        if t:
            tags.add(t)
    # снятия надзором -- из общего лога рядом с каталогом
    kills = 0
    log = run_dir.rstrip("/") + ".log"
    if os.path.exists(log):
        kills = sum(1 for l in open(log, errors="ignore")
                    if "ЗАВИСАНИЕ" in l or "ПРЕДЕЛ" in l)
    return ch, cycles, modes, empties, health_bad, tags, kills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--clean", action="store_true",
                    help="удалить файлы без шапки (пустые/битые)")
    a = ap.parse_args()
    grand = {}
    for run_dir in a.dirs:
        ch, cycles, modes, empties, hb, tags, kills = scan(run_dir)
        print("\n=== %s   канал %s" % (run_dir, ch or "?"))
        print("    кругов на диске: %d   тег(и) кода: %s"
              % (len(cycles), ", ".join(sorted(tags)) or "?"))
        for mode in ("rad", "born"):
            d = modes.get(mode)
            if not d:
                continue
            print("    %-5s файлов %-5d событий %-10d  %5.1f ГБ  кругов %d%s"
                  % (mode, d["files"], d["events"], d["bytes"] / 2 ** 30,
                     len(d["cycles"]),
                     "   ПУСТЫХ %d" % d["empty"] if d["empty"] else ""))
            g = grand.setdefault((ch, mode), [0, 0, 0])
            g[0] += d["files"]
            g[1] += d["events"]
            g[2] += d["empty"]
        if hb:
            print("    частей с не-clean health: %d" % hb)
        if kills:
            print("    снятий надзором в логе: %d" % kills)
        if empties and a.clean:
            for f in empties:
                os.remove(f)
            print("    УДАЛЕНО пустых файлов: %d" % len(empties))
        elif empties:
            print("    пустых файлов: %d  (удалить: --clean)" % len(empties))
    if len(a.dirs) > 1 or grand:
        print("\n=== итого")
        for (ch, mode), g in sorted(grand.items()):
            print("    %-6s %-5s файлов %-5d событий %d%s"
                  % (ch, mode, g[0], g[1],
                     "   пустых %d" % g[2] if g[2] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
