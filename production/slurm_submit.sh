#!/bin/bash
# Запуск счёта массивом задач slurm на ферме JLab.
#
#   slurm_submit.sh <каталог вывода> <всего событий> <частей> <iborn> <зерно> [канал] [лимит]
#
# Жил в каталоге ДАННЫХ, а не в репозитории, и был намертво привязан к пи0.
# И то и другое неправильно: инструмент, лежащий рядом с выборкой, не имеет
# версии и не переезжает вместе с кодом, а привязка к каналу означает копию
# скрипта на каждый канал -- и в копии всегда недостаёт одной защиты.
#
# Два шага, и это не прихоть: таблицу eta строит первая часть и кладёт в
# кэш, остальные её читают.  Пустить массив разом нельзя -- задачи
# передерутся за один файл и построят его столько раз, сколько их есть.
# Поэтому часть 1 идёт отдельной задачей, а массив 2..N ждёт её через
# --dependency=afterok.
#
# Скрипт задачи собирается ЗАКРЫТЫМ heredoc с подстановкой через sed.  Если
# собирать открытым, да ещё через ssh, слои экранирования съедают
# $SLURM_ARRAY_TASK_ID: 23.08 он попал в файл как \$SLURM_... и первая же
# задача упала с syntax error, а зависимый массив был отменён целиком.
#
# Имя скрипта задачи включает режим.  Раньше оба режима писали один task.sh,
# и второй запуск перезаписывал первый; спасало только то, что slurm снимает
# копию скрипта при постановке.  Проверено -- работает, но полагаться на это
# нельзя: достаточно один раз поставить задачи в другом порядке.
set -eu
OUT=$1; NEV=$2; NPART=$3; IBORN=$4; SEED=$5; CH=${6:-pi0}; TLIM=${7:-01:00:00}
# Раздел -- через окружение.  priority ограничен четырьмя часами, и в них не
# влезает круг: 18.09 все четыре поляризованные задачи упали по TIMEOUT ровно
# на 04:00:20, УЖЕ построив таблицу eta, а зависимые массивы 2..N были сняты
# следом.  production даёт двое суток.
PARTITION=${PARTITION:-priority}
CODE=${CODE:-/volatile/clas12/vpk/exclurad_py}
MODE=$([ "$IBORN" = 1 ] && echo born || echo rad)
MEM=${MEM:-512M}
# Карточка: по умолчанию по каналу, но CARD в окружении ПЕРЕКРЫВАЕТ.  Без
# этого поляризованную карточку (она отличается от канальной только ihel и
# ptarg) нельзя подать вообще -- ровно та же дыра, из-за которой сентябрьский
# "amp2609" прогон оказался vpk2021.  Нет файла -- падаем громко.
CARD=${CARD:-$CODE/cards/rad_$CH.txt}
[ -f "$CARD" ] || { echo "НЕТ КАРТОЧКИ: $CARD"; exit 1; }
mkdir -p "$OUT/logs"

# --- карточка
python3 - "$IBORN" "$NEV" "$OUT" "$CARD" <<'PY'
import re, pathlib, sys
ib, nev, out, card = sys.argv[1:5]
k = 0; res = []
for line in pathlib.Path(card).read_text().splitlines():
    if line.split("!")[0].strip():
        k += 1
        if   k == 3:  line = re.sub(r"^\s*\S+", "%-34s" % nev, line, count=1)
        elif k == 4:  line = re.sub(r"^\s*\S+", "%-34s" % ib, line, count=1)
        elif k == 13: line = re.sub(r"^\s*\S+", "%-34s" % out, line, count=1)
    res.append(line)
pathlib.Path("%s/card_%s.txt" % (out, ib)).write_text("\n".join(res) + "\n")
PY

# --- запускалка одной части
cat > "$OUT/part_runner.py" <<'PYRUN'
import sys
from exclurad_py.run import run
card, ipart, npart, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
run(card, verbose=(ipart == 1), _part=(ipart, npart, seed))
PYRUN

# --- скрипт задачи: heredoc ЗАКРЫТЫЙ, подстановка ниже
TASKSH="$OUT/task_$MODE.sh"
cat > "$TASKSH" <<'TASK'
#!/bin/bash
#SBATCH --account=clas12
#SBATCH --partition=@PARTITION@
#SBATCH --time=@TLIM@
#SBATCH --cpus-per-task=1
# Memory: measured, not guessed.  320 completed jobs of 21.09 used
# median 278 MB, max 320 MB (MaxRSS) -- the eta table is LOADED from
# ETA_TABLE, never built, so nothing here is large.  2G was inherited from
# the old two-stage script whose first job built the table, and gave 13.6%
# efficiency: ~1.7 GB x 100 jobs of farm memory held idle, which is what the
# scicomp memory-efficiency warning of 22.09 was about.  512M is 1.6x the
# observed maximum.  Override with MEM= if a card ever needs more.
#SBATCH --mem=@MEM@
#SBATCH --output=@OUT@/logs/@MODE@_%a.out
cd @CODE@
export PYTHONPATH=@CODE@
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
export ETA_TABLE="@ETA_TABLE@"
python3 -u @OUT@/part_runner.py @OUT@/card_@IBORN@.txt \
        "$SLURM_ARRAY_TASK_ID" @NPART@ \
        "$(( @SEED@ + 7700 * SLURM_ARRAY_TASK_ID ))"
TASK
sed -i.bak -e "s|@TLIM@|$TLIM|g" -e "s|@PARTITION@|$PARTITION|g" -e "s|@OUT@|$OUT|g" -e "s|@CODE@|$CODE|g" \
           -e "s|@IBORN@|$IBORN|g" -e "s|@NPART@|$NPART|g" -e "s|@SEED@|$SEED|g" \
           -e "s|@MODE@|$MODE|g" -e "s|@ETA_TABLE@|${ETA_TABLE:-}|g" -e "s|@MEM@|$MEM|g" "$TASKSH"
rm -f "$TASKSH.bak"
chmod +x "$TASKSH"

# --- проверка синтаксиса ДО отправки: упавшая первая задача уносит массив
bash -n "$TASKSH" || { echo "ОШИБКА: скрипт задачи не разбирается"; exit 1; }
echo "скрипт задачи проверен: $TASKSH"

# Два режима.  Без готовой таблицы -- как прежде: часть 1 строит её, массив
# 2..N ждёт через afterok.  Ждёт он не файла, а ЗАВЕРШЕНИЯ части 1, то есть
# и всех её событий: 20.09 116 задач простояли десять часов за одной.
# С ETA_TABLE (готовая таблица, ключ проверяет RCTable.load) строить нечего:
# весь массив 1..N уходит сразу, без зависимости.
if [ -n "${ETA_TABLE:-}" ]; then
  [ -f "$ETA_TABLE" ] || { echo "НЕТ ТАБЛИЦЫ: $ETA_TABLE"; exit 1; }
  # ARRAY_FROM>1: досдать хвост массива, части 1..ARRAY_FROM-1 уже идут или
  # готовы (20.09: часть 1 считала, 2..30 пересдавались без зависимости).
  A0=${ARRAY_FROM:-1}
  J1=$(sbatch --parsable --array="$A0"-"$NPART" --job-name="${CH}_arr_$MODE" "$TASKSH")
  echo "массив $A0-$NPART (таблица $ETA_TABLE): $J1"
else
  J1=$(sbatch --parsable --array=1-1 --job-name="${CH}_tab_$MODE" "$TASKSH")
  echo "задача 1 (строит таблицу): $J1"
  if [ "$NPART" -gt 1 ]; then
    J2=$(sbatch --parsable --dependency=afterok:"$J1" --array=2-"$NPART" \
                --job-name="${CH}_arr_$MODE" "$TASKSH")
    echo "массив 2-$NPART: $J2"
  fi
fi

# Снимать ТОЛЬКО ожидающие: scancel по базовому номеру массива уносит и
# считающие задачи.  23.08 так пропала работа шестнадцати задач.
#     scancel --state=PENDING -u $USER
