#!/bin/bash
cd ~/exclurad_py
for ch in pi0 eta rho phi jpsi; do
  echo "############ $ch $(date +%H:%M) ############"
  rm -rf /Volumes/wd_14tb/mc/exclurad_py/$ch/*_rad_v1
  PYTHONPATH=~/exclurad_py python3 -u -m exclurad_py.run cards/rad_$ch.txt --jobs 6 \
    2>&1 | grep -E "eta table|majorant|interp|events in|health|WARNING"
done
echo "ALL_RAD_DONE $(date +%H:%M)"
