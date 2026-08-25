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
CODE=${CODE:-/volatile/clas12/vpk/exclurad_py}
MODE=$([ "$IBORN" = 1 ] && echo born || echo rad)
[ -f "$CODE/cards/rad_$CH.txt" ] || { echo "нет карточки $CODE/cards/rad_$CH.txt"; exit 1; }
mkdir -p "$OUT/logs"

# --- карточка
python3 - "$IBORN" "$NEV" "$OUT" "$CODE/cards/rad_$CH.txt" <<'PY'
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
#SBATCH --partition=priority
#SBATCH --time=@TLIM@
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=@OUT@/logs/@MODE@_%a.out
cd @CODE@
export PYTHONPATH=@CODE@
# Версия снята при ПОСТАНОВКЕ, на узле входа, где git работает.  На счётном
# узле git может не отработать, и часть запишется безымянной.
export EXCLURAD_COMMIT=@COMMIT@ EXCLURAD_TAG=@TAGV@ EXCLURAD_BRANCH=@BRANCH@
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1
export MAJ_SAFETY=@SAFETY@
@SHAPELINE@
python3 -u @OUT@/part_runner.py @OUT@/card_@IBORN@.txt \
        "$SLURM_ARRAY_TASK_ID" @NPART@ \
        "$(( @SEED@ + 7700 * SLURM_ARRAY_TASK_ID ))"
TASK
# Сетка и запас мажоранты -- те же, что в run_prod_v1.sh.  Здесь стоял
# MAJ_SAFETY=4.0 жёстко, и phi через slurm считался вдвое медленнее, чем
# показал замер; настройки, померенные один раз, обязаны действовать всюду.
case "$CH" in
  jpsi) SAFETY=1.6; SHAPE=16,8,16,12 ;;
  phi)  SAFETY=1.6; SHAPE= ;;
  *)    SAFETY=4.0; SHAPE= ;;
esac
SHAPELINE=$([ -n "$SHAPE" ] && echo "export MAJ_SHAPE=$SHAPE" || echo "unset MAJ_SHAPE 2>/dev/null || true")
VERC=$(cd "$CODE" && python3 -c "from exclurad_py.provenance import git_hash; print(git_hash())" 2>/dev/null || echo unknown)
VERT=$(cd "$CODE" && python3 -c "from exclurad_py.provenance import git_tag; print(git_tag())" 2>/dev/null || echo unknown)
VERB=$(git -C "$CODE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
echo "версия при постановке: $VERT ($VERC, $VERB)"
sed -i.bak -e "s|@SAFETY@|$SAFETY|g" -e "s|@SHAPELINE@|$SHAPELINE|g" \
           -e "s|@COMMIT@|$VERC|g" -e "s|@TAGV@|$VERT|g" -e "s|@BRANCH@|$VERB|g" \
           -e "s|@TLIM@|$TLIM|g" -e "s|@OUT@|$OUT|g" -e "s|@CODE@|$CODE|g" \
           -e "s|@IBORN@|$IBORN|g" -e "s|@NPART@|$NPART|g" -e "s|@SEED@|$SEED|g" \
           -e "s|@MODE@|$MODE|g" "$TASKSH"
rm -f "$TASKSH.bak"
chmod +x "$TASKSH"

# --- проверка синтаксиса ДО отправки: упавшая первая задача уносит массив
bash -n "$TASKSH" || { echo "ОШИБКА: скрипт задачи не разбирается"; exit 1; }
echo "скрипт задачи проверен: $TASKSH"

J1=$(sbatch --parsable --array=1-1 --job-name="${CH}_tab_$MODE" "$TASKSH")
echo "задача 1 (строит таблицу): $J1"
if [ "$NPART" -gt 1 ]; then
  J2=$(sbatch --parsable --dependency=afterok:"$J1" --array=2-"$NPART" \
              --job-name="${CH}_arr_$MODE" "$TASKSH")
  echo "массив 2-$NPART: $J2"
fi

# Снимать ТОЛЬКО ожидающие: scancel по базовому номеру массива уносит и
# считающие задачи.  23.08 так пропала работа шестнадцати задач.
#     scancel --state=PENDING -u $USER
