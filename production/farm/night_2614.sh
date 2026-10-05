#!/bin/bash
# Campaign 2026-10-05: amp2614 (run Mom_B12), LUND only.
#
#   8 samples x 100 k events: pi0/eta x proton/neutron x rad/born,
#   unpolarised target (ptarg 0.0, ihel 0), vcut = 0.
#
# Run it on an ifarm login node:
#
#     bash production/farm/night_2613.sh cards     # write the 8 cards
#     bash production/farm/night_2613.sh tables    # build eta table + bound, per channel
#     bash production/farm/night_2613.sh submit    # sbatch the 8 arrays
#     bash production/farm/night_2613.sh status
#
# Why one production directory PER CHANNEL.  build_tables.sh and
# gen_array.sbatch both read $PROD_DIR/cards/rga_<mode>.txt and write
# $PROD_DIR/tables/ -- one box per directory is the interface.  Four channels
# are four boxes (different model, and the neutron cards have their own t
# range), so they get four directories under one campaign root.
#
# The stage-2 bound is built by production/build_bound.py, STRATIFIED: every
# cell is probed the same number of times.  calibrate()'s proposal-drawn
# probes never reach the tail cells, which is how the 2026-09-25 campaign
# exceeded its bound in 198 of 200 chunks.  amp2614 needs this more than
# amp2609 did: eta at (W2 16, Q2 1.15, t -0.3, h +1) is 194 against 14.4.
set -eu
R=${PROD_DIR:-/volatile/clas12/vpk/night_amp2614_20261005}
V=${EXRU:-$HOME/exclurad_py_htilde}
PY=${EXRU_PY:-/work/clas12/vpk/venv-prod-numba}
NEV_TOT=${NEV_TOT:-100000}
NJOB=${NJOB:-20}
NEV=$(( NEV_TOT / NJOB ))
NW=${NW:-32}           # probe workers for the bound
PER=${PER:-400}        # probes per bound cell

# channel : source card : model : seed base
SPECS="pi0p:rad_pi0_rga.txt:pi0.amp2614:641000000
etap:rad_eta_rga.txt:eta.amp2614:642000000
pi0n:rad_pi0n_rgb_none.txt:pi0n.amp2614:643000000
etan:rad_etan_rgb_none.txt:etan.amp2614:644000000"

cards() {
  for spec in $SPECS; do
    IFS=: read -r ch card model _ <<< "$spec"
    mkdir -p "$R/$ch/cards"
    for mode in rad born; do
      ib=$([ "$mode" = born ] && echo 1 || echo 0)
      # Substitute by FIELD INDEX, the way gen_array.sbatch does.  The cards
      # are positional (exclurad_py/runcard.py FIELDS); matching on the value
      # silently misses, as 's|^5000$|...|' once did against "5000  ! nev".
      awk -v model="$model" -v nev="$NEV" -v ib="$ib" '
        { line = $0; sub(/!.*/, "", line) }
        line ~ /[^ \t]/ {
            k++
            if (k == 1) { print model "   ! model"; next }
            if (k == 3) { print nev   "   ! nev";   next }
            if (k == 4) { print ib    "   ! iborn"; next }
        }
        { print }' "$V/cards/$card" > "$R/$ch/cards/rga_$mode.txt"
      # the three fields that decide the sample must be what we think
      head -40 "$R/$ch/cards/rga_$mode.txt" | grep -qE "^$model " \
        || { echo "FATAL: model not substituted in $ch/$mode"; exit 1; }
    done
    echo "  $ch: cards written ($model, $NEV ev x $NJOB jobs = $NEV_TOT)"
  done
}

tables() {
  export PYTHONPATH=$V OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
  export RC_NUMBA=1
  cd "$V"
  for spec in $SPECS; do
    IFS=: read -r ch _ _ _ <<< "$spec"
    mkdir -p "$R/$ch/tables"
    export ETA_TABLE=$R/$ch/tables/etatable.npz
    L=$R/$ch/tables/build.log
    {
      echo "=== $(date '+%F %T')  $ch"
      git -C "$V" log --oneline -1
      grep -E "! (model|ebeam|W2|Q2|t range|vcut|ihel|ptarg|t_nucl|fermi)" \
           "$R/$ch/cards/rga_rad.txt" || true
      echo "--- rad: eta table + majorant + stratified bound ---"
      "$PY/bin/python" production/build_bound.py "$R/$ch/cards/rga_rad.txt" \
              "$R/$ch/tables/majorant_rad.npz" "$NW" "$PER"
      echo "--- born: majorant only ---"
      "$PY/bin/python" production/build_bound.py "$R/$ch/cards/rga_born.txt" \
              "$R/$ch/tables/majorant_born.npz" 1 1
    } 2>&1 | tee -a "$L"
  done
}

submit() {
  for spec in $SPECS; do
    IFS=: read -r ch _ _ sb <<< "$spec"
    for mode in rad born; do
      mkdir -p "$R/$ch/$mode/logs"
      if [ "$mode" = born ]; then tm=01:00:00; off=5000000; else tm=06:00:00; off=0; fi
      seedbase=$(( sb + off ))
      sbatch --array=1-$NJOB --job-name=g13_${ch}_${mode} --time=$tm \
             --account=clas12 --partition=production --mem=4G --cpus-per-task=1 \
             --output="$R/$ch/$mode/logs/job_%a.log" \
             --export="ALL,PROD_DIR=$R/$ch,EXRU=$V,EXRU_PY=$PY,NEV=$NEV,SEEDBASE=$seedbase,GEN_TAG=$ch" \
             "$V/production/farm/gen_array.sbatch" $mode
    done
  done
  squeue -u "$USER" -o "%.10i %.20j %.8T %.6D %R" | head -20
}

status() {
  squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R" | head -25
  echo
  for spec in $SPECS; do
    IFS=: read -r ch _ _ _ <<< "$spec"
    for mode in rad born; do
      d=$R/$ch/$mode/lund_osg
      n=$(ls "$d"/*.lund 2>/dev/null | wc -l)
      ev=$(cat "$d"/*.lund 2>/dev/null | grep -c '^ *[0-9]* *1 *1 ' || echo 0)
      printf "  %-5s %-5s  %2d/%d chunks  %8s events\n" "$ch" "$mode" "$n" "$NJOB" "$ev"
    done
  done
}

case "${1:-}" in
  cards) cards ;; tables) tables ;; submit) submit ;; status) status ;;
  *) echo "usage: $0 cards|tables|submit|status"; exit 1 ;;
esac
