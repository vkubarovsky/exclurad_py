#!/bin/bash
#
# HOW TO RUN A CAMPAIGN WITH THESE: docs/farm_practice.md
# It carries the sizing, the chaining, and every trap that has cost
# a run here.  Read it before writing a chain of your own.
#
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

# THE INTERPRETER, which is not the same thing as the code checkout above.
# numba became the default on 2026-10-02 (d23dc4e), but ifarm's system
# python3 does not have it: a job run with python3 warns loudly and then
# runs the numpy path CORRECTLY and 2.3x slower.  This venv is the
# production stack with numba added and nothing else changed --
# --system-site-packages keeps the system numpy 1.23.5 and scipy 1.9.3,
# which is what every byte-identity test was done against.  Changing numpy
# here would quietly invalidate all of them.
PY=${EXRU_PY:-/work/clas12/vpk/venv-prod-numba}
[ -x "$PY/bin/python" ] || { echo "FATAL: no interpreter at $PY/bin/python"; exit 1; }

export TMPDIR=/tmp
export PYTHONPATH=$V OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
# REQUIRE numba rather than accept it.  Unset, the code falls back to numpy
# with a warning -- correct, but a 200-job campaign would then run 2.3x
# slower than planned and the only trace would be a line in each job log.
# =1 makes the wrong interpreter a startup failure instead.
export RC_NUMBA=1
# Private compile cache: each job compiles once into its own work directory,
# which is deleted with it.  The alternative is 200 jobs sharing one cache
# tree on Lustre, which costs 5 s less and fails as a physics problem.
export NUMBA_CACHE_DIR=$W
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
"$PY/bin/python" -u -m exclurad_py.run "$W/card.txt" > "$W/gen.log" 2>&1
RC=$?
T1=$(date +%s)
echo "GEN seconds: $((T1-T0))  rc=$RC"
# WARNING and *** must be in this filter.  They were not, and gen.log is
# deleted with $W, so the eta-interpolation warning that the 2026-09-28
# million actually emitted was invisible in all 200 job logs -- it surfaced
# only because another session ran the same tables by hand a day later.
# A warning that no finished run can be asked about is not a warning.
grep -E "setup |generate |health:|WARNING|\*\*\*|backend" "$W/gen.log" || true

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
# GEN_TAG names the sample; four channels now share this script and a
# file called excluradpy_pi0_* in the eta directory is a trap for whoever
# merges them later.  Default keeps the old RG-A pi0 name.
NAME=excluradpy_${GEN_TAG:-pi0_fa18}_${MODE}_$I
# MAKE THE OUTPUT DIRECTORIES.  This script used to assume they existed,
# which they did in the RG-A directory because earlier campaigns had made
# them.  In a FRESH production directory they do not, and on 2026-09-29 that
# cost 100 tasks and 200k events: cp failed with "No such file or directory",
# `ln -sf` happily created a symlink to the file cp had not written, the
# chunk directory filled with 50 DANGLING LINKS that count like real files,
# the script printed DONE, and `rm -rf $W` deleted the events.
mkdir -p "$R/$MODE/lund_osg" "$R/$MODE/chunks" "$R/$MODE/logs" "$R/$MODE/prov"
cp "$SRC" "$R/$MODE/lund_osg/$NAME.lund" || {
  echo "FATAL: could not copy the LUND to $R/$MODE/lund_osg -- keeping $W"; exit 1; }
ln -sf "$R/$MODE/lund_osg/$NAME.lund" "$R/$MODE/chunks/lund_$I.dat"
# ln -sf succeeds against a missing target, so the link proves nothing: test
# what it points AT.  -e follows the link; a dangling one fails here.
[ -e "$R/$MODE/chunks/lund_$I.dat" ] || {
  echo "FATAL: chunk lund_$I.dat is a dangling link -- keeping $W"; exit 1; }

# Provenance travels beside the file, not inside it: the .info sidecar the
# generator writes, and its own config.json, which carries sigma_nb -- the
# normalisation, without which the sample cannot be turned into barns.  The
# previous version of this script deleted config.json with the work directory.
#
# THE COPY IS RENAMED, AND THAT MATTERS WHEN YOU GO LOOKING.  It lands as
# $R/$MODE/prov/${MODE}_$I.config.json, NOT as config.json, so
# `find <campaign> -name config.json` finds nothing and reads as "the record
# was discarded".  It was not: prov/ is the surviving copy and it holds the
# whole `report` block -- sigma_nb, and also b_viol / max_b_viol / accepted,
# the stage-2 numbers that say whether the sample is exactly unweighted.
# Measured this way on 2026-10-07 across two 160-job campaigns: amp2614
# 46/1201080 = 0.0038 % worst x1.69, amp2613 316/2145731 = 0.0147 % worst
# x5.41.  Search for '*.config.json'.
#
# THOSE ARE FOUR CHANNELS AVERAGED TOGETHER, which is not what you want when
# you are asking about one.  Each 160-job campaign is 40 jobs each of pi0p,
# pi0n, etap, etan, and the eta arms carry almost no violations and pull the
# mean down.  Broken out for amp2614 (2026-10-20):
#     pi0 p   28 / 320868 = 0.0087 %   worst x1.49
#     pi0 n   17 / 334381 = 0.0051 %   worst x1.69
#     eta p    0 / 279533 = 0.0000 %   worst x1.00
#     eta n    1 / 266298 = 0.0004 %   worst x1.04
# So the pi0 rate is 2.3x the aggregate.  Quote the channel, not the average.
mkdir -p "$R/$MODE/prov"
cp "$SRC.info" "$R/$MODE/lund_osg/$NAME.lund.info" 2>/dev/null || echo "WARNING: no .info sidecar"
cp "$(dirname "$(dirname "$SRC")")/inputs/config.json" "$R/$MODE/prov/${MODE}_$I.config.json" 2>/dev/null \
  || echo "WARNING: no config.json -- sigma_nb lost for chunk $I"
cp "$W/card.txt" "$R/$MODE/prov/${MODE}_$I.card.txt"
grep -o '"sigma_nb":[^,]*' "$R/$MODE/prov/${MODE}_$I.config.json" 2>/dev/null | head -1

# Only now, with a chunk that exists and holds the events, is the work
# directory expendable.  Every exit above keeps it, so post-mortem and resume
# are possible without regenerating.
CHUNKN=$(grep -c '^ *[0-9]* *1 *1 ' "$R/$MODE/chunks/lund_$I.dat" 2>/dev/null || echo 0)
if [ "$CHUNKN" -lt 1 ]; then
  echo "FATAL: chunk lund_$I.dat holds $CHUNKN events -- keeping $W"; exit 1
fi
rm -rf "$W"
echo "DONE $MODE $I  ($CHUNKN events in chunk)"
