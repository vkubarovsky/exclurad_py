#!/bin/bash
# Stage A of the two-stage chain: generate one chunk of events.
#
#   sbatch --array=1-200 --job-name=gen_rad  --time=04:00:00 gen_array.sbatch rad
#   sbatch --array=1-200 --job-name=gen_born --time=00:30:00 gen_array.sbatch born
#
# Split from the simulation on purpose.  One job doing generation + GEMC +
# reconstruction wants 12 hours; the farm schedules by backfill (bf_window
# 2880 min) and the queue is full of 2-5 hour jobs, so a 12-hour request waits
# for a rare big gap.  Two jobs of 4 and 5 hours slot into the common ones.
# It also decouples the failures: a GEMC crash no longer throws away two hours
# of generated events.
#
# Writes one clean LUND (no '#' since 2026-09-24) plus its .info sidecar and
# config.json, which carries sigma_nb.
set -u
MODE=${1:?born or rad}
R=/volatile/clas12/vpk/prod_pi0_rga_fa18_inb_20260924
V=/volatile/clas12/vpk/exclurad_py
I=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
W=$R/$MODE/gen/w_$I
NEV=${NEV:-5000}
# seeds: disjoint between modes and across tasks, and far from the September
# campaign's 3-9 x 10^7
SEEDBASE=$([ "$MODE" = born ] && echo 610000000 || echo 620000000)
SEED=$(( SEEDBASE + SLURM_ARRAY_TASK_ID * 10000 ))

export TMPDIR=/tmp
export PYTHONPATH=$V OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
# built once in tables/ and shared: 200 jobs each rebuilding the eta table is
# what cost ten hours of idle farm on 2026-09-20
[ "$MODE" = rad ] && export ETA_TABLE=$R/tables/etatable.npz
export MAJ_TABLE=$R/tables/majorant.npz

rm -rf "$W"; mkdir -p "$W"
sed -e "s|OUTROOT|$W|" -e "s|SEED|$SEED|" -e "s|^5000$|$NEV|" \
    "$R/cards/rga_$MODE.txt" > "$W/card.txt"

T0=$(date +%s)
cd "$V"
python3 -u -m exclurad_py.run "$W/card.txt" > "$W/gen.log" 2>&1
RC=$?
T1=$(date +%s)
echo "GEN seconds: $((T1-T0))  rc=$RC"
grep -E "setup |generate |health:" "$W/gen.log" || true

SRC=$(ls "$W"/*/*_${MODE}_v1/lund/*.lund 2>/dev/null | head -1)
if [ $RC -ne 0 ] || [ -z "$SRC" ]; then
  echo "FATAL: generation produced no LUND"; cp "$W/gen.log" "$R/$MODE/logs/gen_fail_$I.log"; exit 1
fi
N=$(grep -vc '^#' "$SRC")
echo "events written: $(grep -c '^ *[0-9]* *1 *1 ' "$SRC" 2>/dev/null || echo '?')  lines: $N"

# The LUND is already clean: exclurad_py stopped writing '#' comments on
# 2026-09-24 precisely because gemc -- and the OSG portal, which runs the same
# gemc -- exits silently on the first one.  Nothing to strip.
NAME=excluradpy_pi0_${MODE}_fa18_inb_$I
cp "$SRC" "$R/$MODE/lund_osg/$NAME.lund"
ln -sf "$R/$MODE/lund_osg/$NAME.lund" "$R/$MODE/chunks/lund_$I.dat"

# Provenance travels beside the file, not inside it: the .info sidecar the
# generator writes, and its own config.json, which carries sigma_nb -- the
# normalisation, without which the sample cannot be turned into barns.  The
# previous version of this script deleted config.json with the work directory.
mkdir -p "$R/$MODE/prov"
cp "$SRC.info" "$R/$MODE/lund_osg/$NAME.lund.info" 2>/dev/null || echo "WARNING: no .info sidecar"
cp "$(dirname "$(dirname "$SRC")")/inputs/config.json" "$R/$MODE/prov/${MODE}_$I.config.json" 2>/dev/null \
  || echo "WARNING: no config.json -- sigma_nb lost for chunk $I"
cp "$W/card.txt" "$R/$MODE/prov/${MODE}_$I.card.txt"
grep -o '"sigma_nb":[^,]*' "$R/$MODE/prov/${MODE}_$I.config.json" 2>/dev/null | head -1

rm -rf "$W"
echo "DONE $MODE $I"
