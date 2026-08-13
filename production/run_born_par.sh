#!/bin/bash
# Explicit --jobs, one channel at a time internally parallel: 3 x 4 = 12 cores.
cd ~/exclurad_py
for ch in pi0 eta jpsi; do
  rm -rf /Volumes/wd_14tb/mc/exclurad_py/$ch/20260813_10.6_born_v1
  ( PYTHONPATH=~/exclurad_py python3 -u -m exclurad_py.run cards/born_$ch.txt --jobs 4 \
      > production/born_$ch.log 2>&1 ; echo "DONE $ch" ) &
done
wait
echo "ALL PARALLEL BORN DONE"
