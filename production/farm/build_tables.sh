#!/bin/bash
# Stage 0: build the shared eta table and the two majorants, ONCE, for a box.
#
#   export PROD_DIR=/volatile/clas12/vpk/prod_pi0_rga_fa18_inb_<date>
#   bash production/farm/build_tables.sh
#
# Run it on ifarm (an interactive node is fine -- it is minutes, not hours)
# BEFORE submitting gen_array.sbatch, and run it again whenever the BOX
# changes.  It writes, into $PROD_DIR/tables/:
#
#     etatable.npz         eta on a (5,5,5,8) grid          rad only
#     majorant_rad.npz     majorant + STRATIFIED stage-2 bound B
#     majorant_born.npz    majorant only (born has no stage 2)
#     build.log            what was built, and the bound's spread
#
# Why this script exists at all.  Nothing built these before: the 2026-09-25
# tables came from an ad-hoc snippet that lived only on the farm, and it used
# calibrate()'s 300 PROPOSAL-DRAWN probes for the stage-2 bound -- the lottery
# that made 198 of 200 chunks exceed their bound, worst x3.35.  The bound is a
# property of the BOX; it is built here, once, by probing every cell uniformly.
#
# WHEN TO RERUN.  The eta table refuses a stale cache loudly (RCTable.load
# raises on a key mismatch).  The majorant does NOT: it falls through and
# rebuilds inside the job, silently, 200 times.  So do not rely on being told.
# Rebuild whenever any of these changes: model, ebeam, W2/Q2/t box, vcut,
# ihel, ptarg, t_nucl, fermi.  NOT for: nev, seed, outroot, theta_e, y, xB,
# iacc, vz, itarg.
#
# Cards must already be in $PROD_DIR/cards/ as rga_born.txt and rga_rad.txt --
# the names gen_array.sbatch reads.
set -eu
R=${PROD_DIR:?set PROD_DIR to the production directory}
V=${EXRU:-$HOME/exclurad_py}
NW=${NW:-32}           # probe workers
PER=${PER:-400}        # probes per bound cell

mkdir -p "$R/tables"
cd "$V"
export PYTHONPATH=$V OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
export ETA_TABLE=$R/tables/etatable.npz

L=$R/tables/build.log
{
  echo "=== $(date '+%F %T')  box from $R/cards/"
  git -C "$V" describe --tags --always 2>/dev/null || true
  grep -E "! (W2|Q2|t|vcut|model|ebeam|ihel|ptarg|t_nucl|fermi)" "$R/cards/rga_rad.txt" || true

  # rad first: it builds the eta table that both the bound and the rad jobs read
  echo "--- rad: eta table + majorant + stratified bound ---"
  python3 production/build_bound.py "$R/cards/rga_rad.txt" \
          "$R/tables/majorant_rad.npz" "$NW" "$PER"

  echo "--- born: majorant only ---"
  python3 production/build_bound.py "$R/cards/rga_born.txt" \
          "$R/tables/majorant_born.npz" 1 1
} 2>&1 | tee -a "$L"

ls -la "$R/tables/"
# A bound whose median sits far above ~1.6 means the probes missed; a B of
# 1.0 everywhere means they never ran.  Read the two lines above before
# submitting 200 jobs against this file.
