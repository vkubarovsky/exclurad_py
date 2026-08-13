#!/bin/bash
# Clean regeneration of all five Born samples with the current code.
cd ~/exclurad_py
for ch in pi0 eta rho phi jpsi; do
  rm -rf /Volumes/wd_14tb/mc/exclurad_py/$ch/20260813_10.6_born_v1
done
for ch in pi0 eta rho phi jpsi; do
  echo "############ $ch ############"
  PYTHONPATH=~/exclurad_py python3 -u -m exclurad_py.run cards/born_$ch.txt --jobs 6 \
    2>&1 | grep -E "calibrated|majorant|events in|health|WARNING"
done
echo "ALL CLEAN BORN DONE"
