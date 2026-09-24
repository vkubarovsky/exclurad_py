#!/bin/bash
# run_prod_v2.sh -- barrier-free production loop.
#
#   run_prod_v2.sh <repo> <card> <out> <nparts> <seedbase> <host-tag> [mode]
#
# Every part is its own loop: generate one file of NEV events (default 5000),
# move it into place, take a new seed, repeat -- until <out>/STOP exists.  No
# cycle, no waiting for the slowest sibling: v1 spent 11-15% of its time with
# most parts idle at the end of every cycle.  A part that fails five times in
# a row stops itself and says so; the others go on.
#
# Requirements that are not optional:
#   ETA_TABLE   the parked eta table for THIS card's spins (RCTable.load
#               refuses a mismatch); no table is built here, ever
#   <out>       on the machine's LOCAL disk -- never a network or external
#               volume (20.09: a mount wedged; a volume permission vanished)
#
# File names carry host and part, so files from many machines can sit in one
# directory without collision; a later merge_lund renumbers them if wanted:
#   <out>/rad/<chan>_rad_<host>_p<part>_<seq>.lund
set -u
REPO=$1; CARD=$2; OUT=$3; J=$4; SEEDBASE=$5; HOST=$6; MODE=${7:-rad}
NEV=${NEV:-5000}
CHAN=$(awk 'NR==1{print $1}' "$CARD" | sed 's/\..*//')
case $CHAN in pi0) TAG=pi0_p;; eta) TAG=eta_p;; pi0n) TAG=pi0_n;; etan) TAG=eta_n;; *) TAG=$CHAN;; esac
[ -n "${ETA_TABLE:-}" ] && [ -f "$ETA_TABLE" ] || { echo "ETA_TABLE missing or not a file: '${ETA_TABLE:-}'"; exit 1; }
[ -f "$CARD" ] || { echo "no card: $CARD"; exit 1; }
case "$OUT" in /Volumes/*|/Users/*/mnt/*|/net/*|/mnt/*) echo "REFUSING to write to a mounted/external path: $OUT"; exit 1;; esac
mkdir -p "$OUT/$MODE" "$OUT/logs" "$OUT/work"
cp "$CARD" "$OUT/card_template.txt"
export PYTHONPATH=$REPO OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1
# Квадратура sigma_F берётся из quad_default() -- (20,12,16,12) с 24.09.26,
# было (30,16,24,16).  2.5x скорости за 0.04% на eta (740 точек боевого
# бокса, веса по борновскому сечению).  RC_QUAD=n_v,n_core,n_tail,n_phi
# перекрывает; значение пишется в шапку LUND как run.rc_quad.
IBORN=$([ "$MODE" = born ] && echo 1 || echo 0)
log(){ echo "$(date '+%d.%m %H:%M') $*" >> "$OUT/run.log"; }
log "start: $TAG $MODE, $J parts, NEV=$NEV, seedbase $SEEDBASE, host $HOST, table $ETA_TABLE, code $(cd $REPO && git describe --tags 2>/dev/null)"

part_loop() {
  local i=$1 seq=0 fails=0
  local W="$OUT/work/p$i"
  while [ ! -f "$OUT/STOP" ]; do
    seq=$((seq+1))
    local seed=$((SEEDBASE + i*100000 + seq))
    rm -rf "$W"; mkdir -p "$W"
    # card: line 3 = nev, 4 = iborn, 11 = seed, 13 = output root (field order of runcard.py)
    awk -v nev="$NEV" -v ib="$IBORN" -v sd="$seed" -v out="$W" '
      { l=$0; sub(/!.*/, "", l) }
      NR==3  { print nev; next } NR==4 { print ib; next }
      NR==11 { print sd;  next } NR==13 { print out; next }
      { print l }' "$OUT/card_template.txt" > "$W/card.txt"
    local t0=$(date +%s)
    if python3 -c "from exclurad_py.run import run; run('$W/card.txt', verbose=False)" > "$W/part.log" 2>&1; then
      local f; f=$(ls "$W"/*/*_${MODE}_v1/lund/*.lund 2>/dev/null | head -1)
      if [ -n "$f" ]; then
        local dest; dest=$(printf "%s/%s/%s_%s_%s_p%02d_%04d.lund" "$OUT" "$MODE" "$TAG" "$MODE" "$HOST" "$i" "$seq")
        mv "$f" "$dest" && cp "$W/card.txt" "$OUT/logs/$(basename "$dest" .lund).card"
        log "p$i #$seq ok $(( $(date +%s) - t0 ))s seed $seed -> $(basename "$dest")"
        fails=0
      else
        fails=$((fails+1)); log "p$i #$seq NO FILE (fail $fails)"; cp "$W/part.log" "$OUT/logs/p${i}_${seq}.fail.log"
      fi
    else
      fails=$((fails+1)); log "p$i #$seq FAILED (fail $fails): $(tail -1 "$W/part.log" | cut -c1-120)"; cp "$W/part.log" "$OUT/logs/p${i}_${seq}.fail.log"
    fi
    [ $fails -ge 5 ] && { log "p$i: five failures in a row -- part stops"; return 1; }
    [ $fails -gt 0 ] && sleep 60
  done
  log "p$i: STOP seen after $seq files"
}

for i in $(seq 1 $J); do part_loop $i & sleep 2; done
wait
log "all parts ended"
