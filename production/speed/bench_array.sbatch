#!/bin/bash
# Compute-node A/B for the speed branch.  Each task times BOTH arms on the
# SAME node, back to back, so the node-to-node spread (1824-2870 ms/event in
# the campaign) cancels in the per-task ratio.  Each task also cmp's the two
# LUND files: same seed, so they must be byte-identical.
set -u
D=/volatile/clas12/vpk/speed_work
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# THE CAMPAIGN TABLES, read-only, not ones I built.  Mine differed (5437 vs
# 5138 bytes) and a benchmark on a different table measures a different
# generator.
T=/volatile/clas12/vpk/prod_pi0_rga_t25_20260928/tables
export ETA_TABLE=$T/etatable.npz MAJ_TABLE=$T/majorant_rad.npz
# The full production set.  Dropping any one silently measures something
# else: without ETA_PROPOSAL_CAP the proposal bound is 7.3x looser AND the
# majorant rebuilds instead of loading, with nothing in the log to say so.
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
I=$(printf "%03d" "${SLURM_ARRAY_TASK_ID:-0}")
W=$D/bench/$I
rm -rf "$W"; mkdir -p "$W"
NEV=${NEV:-25}
for TAG in base spd; do
  sed -e "s|^5000  |$NEV    |" -e "s|^\./exclurad_out|$W/$TAG|" \
      -e "s|^20260923 |$((70000000 + ${SLURM_ARRAY_TASK_ID:-0})) |" \
      "$D/$TAG/cards/rad_pi0_rga.txt" > "$W/card_$TAG.txt"
  cd "$D/$TAG"
  python3 -u -m exclurad_py.run "$W/card_$TAG.txt" > "$W/log_$TAG" 2>&1
  MS=$(grep -oE "[0-9.]+ ms/event" "$W/log_$TAG" | grep -oE "^[0-9.]+" | tail -1)
  SETUP=$(grep -oE "setup [0-9]+s" "$W/log_$TAG" | grep -oE "[0-9]+" | tail -1)
  # GUARD: slow setup means the majorant rebuilt, i.e. the environment is not
  # production's, and the number must not be reported.
  if [ "${SETUP:-99}" -gt 1 ]; then
    echo "GUARD $I $TAG FAIL setup=${SETUP}s majorant-not-cached"
  fi
  echo "RESULT $I $TAG ${MS:-NA} setup=${SETUP:-NA}s $(hostname)"
done
B=$(find "$W/base" -name "*.lund" 2>/dev/null | head -1)
S=$(find "$W/spd" -name "*.lund" 2>/dev/null | head -1)
if [ -n "$B" ] && [ -n "$S" ] && cmp -s "$B" "$S"; then
  echo "BYTES $I identical"
else
  echo "BYTES $I DIFFER"
fi
