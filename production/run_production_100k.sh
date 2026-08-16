#!/bin/bash
# PRODUCTION: 100k RAD per channel in the FROZEN conventions (VPK 15.08):
#   scheme A (EXPO_TAIL), stage-2 exact accept, frozen kf, PDG masses.
# Each channel is validated (acceptance gate) right after generation;
# a channel that fails 5/5 STOPS the run for that channel and says so.
#
# Usage:  ./production/run_production_100k.sh [channels...]   (default: all 5)
set -u
CH="${@:-pi0 eta rho phi jpsi}"
SC=${NIGHT_DIR_BASE:-/Volumes/wd_14tb/mc/exclurad_py}
cd ~/exclurad_py
export ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1
export THEORY_EXPO=1 SAMPLE_EXACT=1 NTHEORY=1500

# stale eta-table caches (pre gilman-v3) must not survive into production
find $SC -name "etatable.npz" -not -newer exclurad_py/generator/tables.py \
     -print -delete 2>/dev/null

for ch in $CH; do
  echo "########## PROD $ch generate $(date +%F' '%H:%M) ##########"
  PYTHONPATH=~/exclurad_py python3 -u -m exclurad_py.run cards/rad_$ch.txt \
      --jobs 12 2>&1 | tail -3
  echo "########## PROD $ch acceptance $(date +%F' '%H:%M) ##########"
  NIGHT_DIR=$SC/validation_$(date +%Y%m%d) \
  SAMPLE_GLOB="$SC/$ch/*_rad_v1/lund/*.lund" \
    PYTHONPATH=~/exclurad_py python3 -u production/validate_channel.py $ch \
    | tee /tmp/prod_${ch}_verdict.txt
  if ! grep -q "5/5 PASS" /tmp/prod_${ch}_verdict.txt; then
    echo "########## PROD $ch: ACCEPTANCE FAILED -- samples NOT released"
    echo "DO NOT USE: failed acceptance $(date +%F)" \
        > $SC/$ch/$(ls -t $SC/$ch | head -1)/DO_NOT_USE.md
  fi
done
echo "########## PRODUCTION SWEEP DONE $(date +%F' '%H:%M) ##########"
