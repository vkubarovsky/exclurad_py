#!/bin/bash
# Запуск на втором маке (M3).  Самодостаточно: клонирует репозиторий,
# собирает карточки и считает указанные каналы.
#
#   ./run_on_m3.sh rho phi jpsi        # каналы через пробел
#   ./run_on_m3.sh                     # по умолчанию rho phi jpsi
#
# Результаты лягут в ~/exclurad_run/out, лог в ~/exclurad_run/run.log
set -u
CH="${@:-rho phi jpsi}"
ROOT=~/exclurad_run
REPO=$ROOT/exclurad_py
BRANCH=feature/exclurad_py
NEV_RAD=50000
NEV_NORAD=500000

echo "=== каналы: $CH"
command -v python3 >/dev/null || { echo "нет python3"; exit 1; }
python3 -c "import numpy" 2>/dev/null || { echo "нет numpy: pip3 install numpy"; exit 1; }

mkdir -p $ROOT/out $ROOT/logs
if [ -d $REPO/.git ]; then
  git -C $REPO fetch origin && git -C $REPO checkout $BRANCH && git -C $REPO pull --ff-only
else
  git clone -b $BRANCH git@github.com:vkubarovsky/exclurad_py.git $REPO || exit 1
fi
echo "=== версия: $(git -C $REPO log --oneline -1)"

# Производительные ядра, а не все.  На Apple Silicon hw.ncpu считает и
# энергоэффективные, которые для этой задачи втрое медленнее; блок
# заканчивается по последнему воркеру, так что четыре медленных растягивают
# весь блок.  На M4 Pro это 10 из 14, на базовом M3 -- 4 из 8.
JOBS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)
echo "=== $(sysctl -n machdep.cpu.brand_string): ядер всего $(sysctl -n hw.ncpu), беру производительные: $JOBS"

cd $REPO
export PYTHONPATH=$REPO
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1

# part_runner: одна часть общего прогона, как на мини
cat > $ROOT/part_runner.py <<'PY'
import sys
from exclurad_py.run import run
card, ipart, npart, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
run(card, verbose=(ipart == 1), _part=(ipart, npart, seed))
PY

mkcard () {   # $1=канал $2=режим $3=iborn $4=nev
  python3 - "$1" "$2" "$3" "$4" "$REPO" "$ROOT" <<'PY'
import sys, re, pathlib
ch, m, ib, nev, repo, root = sys.argv[1:7]
out = []; k = 0
for line in (pathlib.Path(repo) / f"cards/rad_{ch}.txt").read_text().splitlines():
    if line.split("!")[0].strip():
        k += 1
        if k == 3: line = re.sub(r'^\s*\S+', nev, line, count=1)
        elif k == 4: line = re.sub(r'^\s*\S+', ib, line, count=1)
        elif k == 13: line = re.sub(r'^\s*\S+', f"{root}/out/{m}", line, count=1)
    out.append(line)
pathlib.Path(f"{root}/card_{m}_{ch}.txt").write_text("\n".join(out) + "\n")
PY
}

for ch in $CH; do
  for mode in norad rad; do
    if [ "$mode" = norad ]; then ib=1; nev=$NEV_NORAD; off=5300; else ib=0; nev=$NEV_RAD; off=7700; fi
    mkcard $ch $mode $ib $nev
    echo "@@@ $ch $mode начато $(date +%H:%M)"
    for i in $(seq 1 $JOBS); do
      python3 -u $ROOT/part_runner.py $ROOT/card_${mode}_${ch}.txt $i $JOBS $((20260901+off*i)) \
        > $ROOT/logs/${mode}_${ch}_$i.log 2>&1 &
      # первая часть строит таблицу и мажоранту; ждём именно её, а не таймер
      if [ $i -eq 1 ]; then
        while kill -0 $! 2>/dev/null && ! ls $ROOT/out/$mode/$ch/*/majorant.npz >/dev/null 2>&1; do sleep 3; done
        sleep 3
      fi
    done
    wait
    n=$(ls $ROOT/out/$mode/$ch/*/lund/*.lund 2>/dev/null | wc -l | tr -d ' ')
    echo "@@@ $ch $mode готово $(date +%H:%M): $n файлов; $(grep -h 'health:' $ROOT/logs/${mode}_${ch}_1.log | tail -1)"
  done
  RAD="$ROOT/out/rad/$ch/*_rad_v1/lund/*.lund"
  NOR="$ROOT/out/norad/$ch/*_born_v1/lund/*.lund"
  echo "@@@ $ch: мягкий предел"
  RAD_GLOB="$RAD" NORAD_GLOB="$NOR" NPRED=1200 python3 -u production/soft_limit.py $ch
  echo "@@@ $ch: профили delta"
  RAD_GLOB="$RAD" NORAD_GLOB="$NOR" NQUAD=3000 python3 -u production/delta_profiles.py $ch
done
echo "@@@ ГОТОВО $(date +%H:%M)"
