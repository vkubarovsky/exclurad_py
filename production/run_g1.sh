#!/bin/bash
# Прогон G1: узкие коробки, целиком внутри физической области.
#   $1 = корень вывода, $2 = база зерна, $3 = каталог репозитория
# Каждая машина считает все пять каналов; союз двух машин -- 300К rad.
set -u
OUT=$1; BASE=$2; REPO=$3
cd $REPO; export PYTHONPATH=$REPO
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1
J=$(sysctl -n hw.ncpu); NEV_RAD=150000; NEV_NORAD=750000
mkdir -p $OUT $OUT/logs
echo "=== G1: ядер $J, база зерна $BASE, вывод $OUT"
for ch in pi0 eta rho phi jpsi; do
  for mode in norad rad; do
    if [ $mode = norad ]; then ib=1; nev=$NEV_NORAD; off=5300; else ib=0; nev=$NEV_RAD; off=7700; fi
    python3 - "$ch" "$ib" "$nev" "$OUT/$mode" "$OUT" <<'PY'
import re, pathlib, sys
ch, ib, nev, out, root = sys.argv[1:6]
k = 0; res = []
for line in pathlib.Path(f"cards/g1_{ch}.txt").read_text().splitlines():
    if line.split("!")[0].strip():
        k += 1
        if k == 3:    line = re.sub(r'^\s*\S+', f"{nev:<34}", line, count=1)
        elif k == 4:  line = re.sub(r'^\s*\S+', f"{ib:<34}", line, count=1)
        elif k == 13: line = re.sub(r'^\s*\S+', f"{out:<34}", line, count=1)
    res.append(line)
pathlib.Path(f"{root}/card_{ch}_{ib}.txt").write_text("\n".join(res) + "\n")
PY
    echo "@@@ $ch $mode начато $(date +%H:%M)"
    for i in $(seq 1 $J); do
      python3 -u $OUT/../part_runner.py $OUT/card_${ch}_${ib}.txt $i $J $((BASE+off*i)) \
        > $OUT/logs/${mode}_${ch}_$i.log 2>&1 &
      # Ждём etatable.npz -- его пишет первая часть.  majorant.npz не
      # появляется никогда: его сохраняет run_parallel, а части зовут
      # run(_part=...), который кэш только читает.  У born-прогона таблицы
      # нет вовсе -- ждать нечего.
      if [ $i -eq 1 ] && [ "$mode" = rad ]; then
        for _ in $(seq 1 900); do
          ls $OUT/$mode/$ch/*/etatable.npz >/dev/null 2>&1 && break
          kill -0 $! 2>/dev/null || break
          sleep 3
        done
        sleep 5
      fi
    done
    wait
    n=$(ls $OUT/$mode/$ch/*/lund/*.lund 2>/dev/null | wc -l | tr -d ' ')
    echo "@@@ $ch $mode готово $(date +%H:%M): $n файлов; $(grep -h 'health:' $OUT/logs/${mode}_${ch}_1.log | tail -1)"
  done
done
echo "@@@ G1 ЗАВЕРШЁН $(date +%H:%M)"
