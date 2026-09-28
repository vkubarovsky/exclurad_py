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
# ONE production directory for all three stages.  Until 2026-09-28 this
# script said 20260924 while sim_array.sbatch said t2_20260925: stage A
# wrote where stage B did not look.  Set PROD_DIR once and export it.
R=${PROD_DIR:-/volatile/clas12/vpk/prod_pi0_rga_fa18_inb_t2_20260925}
# The code checkout the jobs run.  Moved off /volatile on 2026-09-28 (VPK):
# /volatile is scratch and gets cleaned, so the generator must not live
# there.  $HOME works for anyone, which matters for Story.
V=${EXRU:-$HOME/exclurad_py}
I=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
W=$R/$MODE/gen/w_$I
NEV=${NEV:-5000}
# seeds: disjoint between modes and across tasks, and far from the September
# campaign's 3-9 x 10^7
# Overridable, and it MUST be overridden per campaign.  The base was baked
# in, so the 2026-09-28 reference would have reused the 2026-09-25 seeds
# exactly: a different box makes different events, but two samples whose
# random streams coincide cannot be combined or compared as independent.
SEEDBASE=${SEEDBASE:-$([ "$MODE" = born ] && echo 610000000 || echo 620000000)}
SEED=$(( SEEDBASE + SLURM_ARRAY_TASK_ID * 10000 ))

export TMPDIR=/tmp
export PYTHONPATH=$V OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
# built once in tables/ and shared: 200 jobs each rebuilding the eta table is
# what cost ten hours of idle farm on 2026-09-20
[ "$MODE" = rad ] && export ETA_TABLE=$R/tables/etatable.npz
# ONE MAJORANT PER MODE.  _maj_key carries born_only, so a born job cannot
# use the rad majorant -- and the cache does not say so: unlike the eta
# table, which raises on a key mismatch, the majorant falls through
# silently and rebuilds.  Measured: 25 s in each of 200 born jobs.
export MAJ_TABLE=$R/tables/majorant_$MODE.npz

rm -rf "$W"; mkdir -p "$W"
# Substitute by FIELD, not by pattern.  's|^5000$|$NEV|' looked right and did
# nothing: the card line is "5000    ! nev", so the end anchor never matched
# and NEV was silently ignored -- a 50-event test generated 5000.
awk -v out="$W" -v seed="$SEED" -v nev="$NEV" '
  { line = $0; sub(/!.*/, "", line) }
  line ~ /[^ \t]/ {
      k++
      if (k == 3)  { print nev  "   ! nev";     next }
      if (k == 11) { print seed "   ! seed";    next }
      if (k == 13) { print out  "   ! outroot"; next }
  }
  { print }' "$R/cards/rga_$MODE.txt" > "$W/card.txt"
grep -qE "^$NEV" "$W/card.txt" || { echo "FATAL: nev not substituted"; head -20 "$W/card.txt"; exit 1; }

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
# NO polarity in the name.  The generator knows nothing about the torus --
# one LUND feeds both the inbending and the outbending simulation, and
# sim_array.sbatch puts the polarity into ITS output names, where it is
# true.  Until 2026-09-28 this said "inb", so every outbending sample would
# have been built from a file whose name claimed otherwise.
NAME=excluradpy_pi0_${MODE}_fa18_$I
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
