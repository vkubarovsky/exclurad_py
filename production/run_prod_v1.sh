#!/bin/bash
# Боевой прогон prod-pi0-v1: пи-ноль с до-вершинным внешним излучением.
#
#   run_prod_v1.sh <репозиторий> <корень> <база зерна> <ядер> <рубеж-epoch> <метка>
#
# Один скрипт на три машины.  Каждый круг -- своё зерно и свой каталог,
# поэтому образцы разных машин и разных кругов СКЛАДЫВАЮТСЯ.
#
# Устройство каталога (единое везде):
#   <корень>/prod_pi0_v1/
#       README.md          что это, каким кодом, какой коробкой
#       card_rad.txt       карточки, по одной на режим
#       card_born.txt
#       cyc01/ cyc02/ ...  круги: lund/ и logs/ внутри
#       STOP               создать -- петля закончит круг и выйдет
#
# ОСТАНОВКА -- ТОЛЬКО ФАЙЛОМ.  Убивать надзирателя нельзя: он живёт в tmux,
# и pkill по имени валит сессию, а та по SIGHUP -- всю группу вместе с
# воркерами.  Так за неделю трижды пропал идущий счёт.  `touch STOP` --
# петля доработает круг и выйдет сама.
set -u
REPO=$1; OUT_ROOT=$2; BASE=$3; J=$4; DEADLINE=$5; TAG=$6
NEV_RAD=100000
NEV_BORN=500000

OUT=$OUT_ROOT/prod_pi0_v1
STOP=$OUT/STOP
mkdir -p "$OUT"
rm -f "$STOP"

cd "$REPO"; export PYTHONPATH=$REPO
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0

VER=$(python3 -c "from exclurad_py.provenance import git_hash, git_tag; print(git_hash(), git_tag())" 2>/dev/null || echo "unknown unknown")

if [ ! -f "$OUT/README.md" ]; then
  cat > "$OUT/README.md" <<EOF
# prod-pi0-v1 -- пи-ноль с внешним излучением в мишени

Машина: $TAG ($(hostname)), ядер $J
Код: $VER
Начато: $(date '+%Y-%m-%d %H:%M %Z')

## Что это

Боевая выборка exclusive пи-ноль на протоне при 10.6 ГэВ, с ПОЛНОЙ
радиационной поправкой (внутренней) и с ДО-ВЕРШИННЫМ ВНЕШНИМ излучением в
пятисантиметровой мишени жидкого водорода.

Внешнее излучение выходной ноги здесь НЕ моделируется намеренно: его
считает Geant4 в GEMC, и повторять его тут значило бы посчитать дважды.
Входную ногу может дать только генератор -- Geant4 стартует с вершины и
энергию пучка получает готовой.

## Коробка

Q2 0.5-5.0 ГэВ^2, W2 3.24-16.0 ГэВ^2, |t| до 3.0 ГэВ^2, v_cut 0.20 ГэВ^2.
Вершина равномерно по -7.5..-2.5 см, itarg=1, X0(LH2) = 890.4 см.

## Проверки, на которых стоит эта версия

* радиатор сверен с exclugen: доля событий, выбитых за обрезание
  эксклюзивности, 0.800 +- 0.026 % против 0.873 +- 0.112 % у exclugen;
* при itarg=0 выше обрезания НОЛЬ событий из сорока тысяч;
* целостность LUND: четырёхимпульс сходится до 1.25e-8 ГэВ, распад до 1e-10;
* воспроизводимость: тот же засев даёт побайтово тот же файл;
* согласие с точной квадратурой в коробке G1: все двадцать клеток <= 3.0 сигмы.

## Оговорка про обрезание

v_cut накладывается в ИСТИННЫХ переменных, при собственной энергии события.
Анализ будет резать по ВОССТАНОВЛЕННОЙ недостающей массе, считая энергию
пучка номинальной, и около 0.8 % событий этот рез не пройдут.  Так и должно
быть: внешняя потеря только УВЕЛИЧИВАЕТ восстановленную неупругость
(v_вос - v_ист = 2 dE (E_X - p_Xz) >= 0), поэтому события уходят из окна
наружу и никогда внутрь -- выборка полна.

## Как остановить

    touch $OUT/STOP

Петля закончит текущий круг и выйдет.  Не убивайте процессы.
EOF
  echo "@@@ $TAG README записан, код $VER"
fi

mkcard () {   # $1=iborn $2=nev $3=имя
  python3 - "$1" "$2" "$OUT/$3" <<'PY'
import re, pathlib, sys
ib, nev, dst = sys.argv[1:4]
k = 0; out = []
for line in pathlib.Path("cards/rad_pi0.txt").read_text().splitlines():
    if line.split("!")[0].strip():
        k += 1
        if   k == 3:  line = re.sub(r'^\s*\S+', "%-34s" % nev, line, count=1)
        elif k == 4:  line = re.sub(r'^\s*\S+', "%-34s" % ib, line, count=1)
        elif k == 13: line = re.sub(r'^\s*\S+', "%-34s" % "PLACEHOLDER", line, count=1)
    out.append(line)
pathlib.Path(dst).write_text("\n".join(out) + "\n")
PY
}
mkcard 0 $NEV_RAD card_rad.txt
mkcard 1 $NEV_BORN card_born.txt

# нумерация продолжается с последнего круга: перезапуск ничего не затирает
n=$(ls -d "$OUT"/cyc* 2>/dev/null | sed 's/.*cyc//' | sort -n | tail -1)
n=$((10#${n:-0}))

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  [ -f "$STOP" ] && { echo "@@@ $TAG STOP -- выхожу, кругов сделано $n"; break; }
  n=$((n+1))
  W=$OUT/cyc$(printf %02d $n)
  mkdir -p "$W/logs"
  # ~2.6 с/событие на ядро для радиационных, борновские в десять раз дешевле
  need=$(( (NEV_RAD * 26 / 10 + NEV_BORN * 26 / 100) / J + 900 ))
  left=$(( DEADLINE - $(date +%s) ))
  if [ $left -lt $need ]; then
    echo "@@@ $TAG круг $n не влезает ($need нужно, $left осталось) -- стоп"
    rmdir "$W/logs" "$W" 2>/dev/null; n=$((n-1)); break
  fi

  for mode in born rad; do
    [ $mode = born ] && { ib=1; off=5300; } || { ib=0; off=7700; }
    sed "s|PLACEHOLDER|$W|" "$OUT/card_$mode.txt" > "$W/card_$mode.txt"
    echo "@@@ $TAG круг $n $mode начато $(date '+%d.%m %H:%M')"
    for i in $(seq 1 $J); do
      python3 -u "$OUT_ROOT/part_runner.py" "$W/card_$mode.txt" $i $J \
              $((BASE + n*310000 + off*i)) > "$W/logs/${mode}_$i.log" 2>&1 &
      # Первая часть строит и кэширует таблицу eta; ждём ИМЕННО её файл.
      # majorant.npz не появится никогда -- его пишет run_parallel, а части
      # зовут run(_part=...), который кэш только читает.  У born таблицы нет.
      if [ $i -eq 1 ] && [ $mode = rad ]; then
        for _ in $(seq 1 900); do
          ls "$W"/pi0/*/etatable.npz >/dev/null 2>&1 && break
          kill -0 $! 2>/dev/null || break
          sleep 3
        done; sleep 5
      fi
    done
    wait
    nf=$(ls "$W"/pi0/*/lund/*.lund 2>/dev/null | wc -l | tr -d ' ')
    echo "@@@ $TAG круг $n $mode готово $(date '+%d.%m %H:%M'): $nf файлов; $(grep -h 'health:' "$W/logs/${mode}_1.log" | tail -1)"
    # Круг без файлов -- поломка, а не "мало посчиталось".  22.08 петля так
    # прокрутила 826 пустых кругов за два с половиной часа: карточки были
    # новые, разборщик старый, каждый воркер падал за секунду.
    if [ "$nf" -eq 0 ]; then
      echo "@@@ $TAG ОСТАНОВ: круг $n режим $mode не дал ни одного файла"
      tail -5 "$W/logs/${mode}_1.log" 2>/dev/null | sed 's/^/    /'
      exit 1
    fi
  done
done
echo "@@@ $TAG ПЕТЛЯ ЗАВЕРШЕНА, кругов $n, $(date '+%d.%m %H:%M')"
