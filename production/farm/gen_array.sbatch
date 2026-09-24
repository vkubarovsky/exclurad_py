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
# Writes BOTH the LUND with its provenance header (kept, and what OSG would be
# given) and a stripped copy for GEMC, which exits silently on a '#' line.
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

# One LUND, stripped of '#'.  Both GEMC and the OSG portal run the same gemc,
# which exits silently on a comment line, so the OSG-facing file has to be the
# stripped one too -- keeping a second, header-bearing copy would be 800 kB per
# chunk to preserve 1.8 kB of provenance.
grep -v '^#' "$SRC" > "$R/$MODE/lund_osg/excluradpy_pi0_${MODE}_fa18_inb_$I.lund"
ln -sf "$R/$MODE/lund_osg/excluradpy_pi0_${MODE}_fa18_inb_$I.lund" "$R/$MODE/chunks/lund_$I.dat"

# The provenance travels separately, because nothing downstream carries it: the
# stripped LUND has lost it and a hipo file records only RUN::config (run number
# and fields), nothing about the generator.  Three small files per chunk:
#   header      - the '#' block: model, git commit, rc_quad, spins, host, seed
#   config.json - the generator's own report, and sigma_nb, the NORMALISATION
#                 without which the sample cannot be turned into barns
#   card        - the run card, verbatim
mkdir -p "$R/$MODE/prov"
grep '^#' "$SRC" > "$R/$MODE/prov/${MODE}_$I.header.txt"
cp "$(dirname "$(dirname "$SRC")")/inputs/config.json" "$R/$MODE/prov/${MODE}_$I.config.json" 2>/dev/null \
  || echo "WARNING: no config.json -- sigma_nb lost for chunk $I"
cp "$W/card.txt" "$R/$MODE/prov/${MODE}_$I.card.txt"
grep -E "sigma_nb" "$R/$MODE/prov/${MODE}_$I.config.json" 2>/dev/null | head -1

rm -rf "$W"
echo "DONE $MODE $I"
