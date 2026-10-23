#!/bin/bash
# Campaign conductor.  Runs ON ifarm, detached, and does not need anyone
# watching.  VPK asked for this on 2026-10-09 and the reason became concrete
# on 2026-10-13: 778 tasks finished on the 10th, nobody noticed, and the
# queue sat empty for two days because the only thing that could have
# noticed was a chat session that does not exist between messages.
#
#   start:  PROD_DIR=... EXRU=... setsid nohup bash conductor.sh >> $LOG 2>&1 </dev/null &
#   watch:  cat $PROD_DIR/STATUS.txt
#   stop:   touch $PROD_DIR/CONDUCTOR_STOP
#
# Generalised from the J/psi conductor on 2026-10-15.  The channel is read
# from the CARD, exactly as sim_array.sbatch reads it, so the two can never
# disagree about which files they are naming.  Everything else is unchanged.
#
# Idempotent: it decides what to do from what is ON DISK, not from memory, so
# it can be killed and restarted at any point and picks up correctly.
#
# It never starts a new CHANNEL.  One conductor runs one campaign and stops.
#
# SEVERAL CONDUCTORS MAY RUN AT ONCE, one per campaign, so the farm does not
# drain between channels.  That needs the channel in the slurm job name: until
# 2026-10-23 the name was fill_rad_inb_45nA, identical for every campaign, and
# `queued` counts by exact name -- so a pi0 conductor would have seen phi's
# jobs, concluded its own were already submitted, and quietly submitted
# nothing for as long as the other campaign ran.
set -u
R=${PROD_DIR:?set PROD_DIR to the production directory}
EX=${EXRU:-$HOME/exclurad_py}
S=$R/STATUS.txt
STOP=$R/CONDUCTOR_STOP
LOG=${CONDUCTOR_LOG:-$R/conductor.log}
N=${NCHUNK:-200}                        # chunks per set
BGIN=${BGIN:-45nA_10604MeV}             # inbending  background current
BGOUT=${BGOUT:-50nA_10604MeV}           # outbending background current
export PROD_DIR=$R EXRU=$EX

# The channel comes from the card, not from an environment variable one has
# to remember: sim_array.sbatch derives it the same way, and a mismatch here
# would make `have` count zero for ever and resubmit 200 finished tasks.
CHAN=$(awk '{sub(/!.*/,"")} /[^ \t]/{print $1; exit}' "$R/cards/rga_rad.txt" | cut -d. -f1)
[ -n "$CHAN" ] || { echo "cannot read channel from $R/cards/rga_rad.txt"; exit 1; }

say(){ echo "$(date '+%F %H:%M') $*" >> "$LOG"; }

have(){ ls "$R/$1/dst_$2"/*_bg$3_*.hipo 2>/dev/null | wc -l; }
# Indices that still need simulating: the DST is absent AND the generated
# chunk is there to simulate.  The chunk test is not decoration -- the
# conductor is started while generation is still running, so without it the
# first pass would submit 200 tasks that die one second in with "FATAL: no
# chunk lund_NNN.dat", and the gap would then look filled to `queued`.
# -e, not -s: the chunk is a SYMLINK into lund_osg, and -s on a dangling
# link is false for the right reason while -f would be true for the wrong one.
missing(){ for i in $(seq -w 1 $N); do
             [ -e "$R/$1/chunks/lund_$i.dat" ] || continue
             [ -s "$R/$1/dst_$2/excluradpy_${CHAN}_${1}_fa18_${2}_bg${3}_$i.hipo" ] || echo $((10#$i))
           done | paste -sd,; }
queued(){ squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^$1$"; }

submit(){ # mode pol bgcur jobname timelimit indices
  local m=$1 p=$2 bg=$3 jn=$4 tl=$5 idx=$6
  [ -z "$idx" ] && return 0
  cd "$EX" || return 1
  local id
  id=$(sbatch --parsable --export=ALL,BGCUR="$bg" --array="$idx" \
       --job-name="$jn" --time="$tl" production/farm/sim_array.sbatch "$m" "$p" 2>&1)
  say "submitted $jn [$idx] -> $id"
}

status(){
  { echo "$CHAN campaign -- conductor status"
    date '+written %F %H:%M'
    echo
    echo "PROD_DIR $R"
    echo
    printf "%-26s %7s %7s\n" "set" "done" "of"
    for m in rad born; do for p in inb outb; do
      bg=$([ "$p" = inb ] && echo "${BGIN%%_*}" || echo "${BGOUT%%_*}")
      printf "  %-24s %7s %7s\n" "$m $p  $bg" "$(have "$m" "$p" "$bg")" "$N"
    done; done
    for m in rad born; do for p in inb outb; do
      printf "  %-24s %7s %7s\n" "$m $p  no background" "$(have "$m" "$p" none)" "$N"
    done; done
    echo
    echo "queue:"
    squeue -u "$USER" -h -o "  %j %T" 2>/dev/null | sort | uniq -c | sed 's/^/ /' || echo "  (empty)"
    echo
    tail -6 "$LOG" 2>/dev/null | sed 's/^/  /'
  } > "$S".tmp && mv "$S".tmp "$S"
}

say "conductor started for channel $CHAN, pid $$, prod $R"
while :; do
  [ -f "$STOP" ] && { say "STOP file present, exiting"; status; exit 0; }

  # ---- 1. the with-background sets: fill any gaps -------------------------
  for m in rad born; do for p in inb outb; do
    bg=$([ "$p" = inb ] && echo "$BGIN" || echo "$BGOUT")
    tag=${bg%%_*}
    jn="fill_${CHAN}_${m}_${p}_${tag}"
    [ "$(queued "$jn")" -gt 0 ] && continue
    miss=$(missing "$m" "$p" "$tag")
    [ -n "$miss" ] && submit "$m" "$p" "$bg" "$jn" 14:00:00 "$miss"
  done; done

  # ---- 2. the no-background sets, gated on one clean task -----------------
  # One task proves the configuration before 799 follow it.  The gate earned
  # itself on 2026-10-14: MS was unset in the no-background branch under
  # `set -u` and every task died at line 198 -- after a six-hour GEMC pass.
  if [ "$(have rad inb none)" -ge 1 ]; then
    for m in rad born; do for p in inb outb; do
      jn="nobg_${CHAN}_${m}_${p}"
      [ "$(queued "$jn")" -gt 0 ] && continue
      miss=$(missing "$m" "$p" none)
      # 14 h, NOT 10.  The J/psi no-background jobs fitted in 10 h (GEMC
      # 15.7-18.0 ks, recon 8.2-10.9 ks, total 6.7-8.0 h), and dropping the
      # background does NOT make reconstruction cheap -- it was still 8+ ks
      # there.  phi's GEMC is 1.7x slower than J/psi's, 28.1-28.9 ks
      # inbending, measured on chunk 001 of this campaign: with recon that is
      # ~10.4 h and every one of 800 tasks would have been killed at the limit
      # after burning ten hours.  SLURM refuses to raise a running job's
      # limit (Access/permission denied), so the only fix is to submit it
      # right.  Kaons are the reason: they decay in flight and GEMC tracks
      # the secondaries.
      [ -n "$miss" ] && submit "$m" "$p" none "$jn" 14:00:00 "$miss"
    done; done
  else
    [ "$(queued nobg_test)" -eq 0 ] && [ "$(queued nobg_rad_inb)" -eq 0 ] && \
      submit rad inb none nobg_test 14:00:00 1
  fi

  # ---- 3. done? -----------------------------------------------------------
  tot=0
  for m in rad born; do for p in inb outb; do
    bg=$([ "$p" = inb ] && echo "${BGIN%%_*}" || echo "${BGOUT%%_*}")
    tot=$(( tot + $(have "$m" "$p" "$bg") + $(have "$m" "$p" none) ))
  done; done
  if [ "$tot" -ge $((8 * N)) ]; then
    say "ALL $((8*N)) $CHAN DSTs present.  Campaign complete; stopping.  A new channel is a new conductor and a new decision."
    status; exit 0
  fi

  status
  sleep ${POLL:-1800}
done
