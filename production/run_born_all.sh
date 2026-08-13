#!/bin/bash
# Born for all five channels from matched run cards.
cd ~/exclurad_py
for ch in pi0 eta rho phi jpsi; do
  echo "############ $ch ############"
  PYTHONPATH=~/exclurad_py python3 -u -m exclurad_py.run cards/born_$ch.txt || echo "FAILED: $ch"
done
echo "ALL BORN DONE"
