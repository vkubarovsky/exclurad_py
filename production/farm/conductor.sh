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
# THE CHUNK NAME IS printf %03d, NOT seq -w.
#
# This was `for i in $(seq -w 1 $N)`, and seq -w pads to the width of N: with
# N=400 that is three digits and matches lund_001.dat, with N=2000 it is FOUR
# and looks for lund_0001.dat, which does not exist.  gen_array.sbatch names
# chunks with printf %03d -- three digits up to 999 and four from 1000 -- so
# the two agreed only while N was under 1000.
#
# Found on 2026-10-26 starting the 10-million J/psi campaign at NCHUNK=2000:
# the conductor came up, logged "conductor started", found no chunk for any of
# indices 1-999, submitted NOTHING and did not even rewrite STATUS.txt.  It
# would have sat there silently for ever.  A conductor that finds no work is
# indistinguishable from a conductor with no work to do, which is the whole
# problem with the old loop.
# The index list goes to `sbatch --array=`, which takes a BOUNDED string.  It
# was `paste -sd,`, one index per comma: with 2000 missing chunks that is an
# 8900-character argument and slurm refuses the whole submission with
#   sbatch: error: Batch job submission failed: Pathname of a file, directory
#           or other parameter too long
# Found on 2026-10-27 with the campaign stalled at 2186 of 16000 DSTs: 36
# submissions had succeeded and 140 had failed, every no-background set among
# them, and the conductor had been looping on those four failures ever since.
# Nothing was wrong with the farm -- the work was never submitted.
#
# So emit slurm RANGE syntax instead.  The gaps are large contiguous blocks,
# so 1,2,3,...,2000 collapses to 1-2000: nine characters instead of 8900.
# Singletons stay single, which is what the a==p test is for.
#
# MAXSUB caps one submission.  MaxSubmit on this account is 10000 tasks and the
# conductor drives eight sets, so an uncapped 2000 per set would ask for 16000
# and fail again -- differently, and just as silently.  The conductor loops, so
# a cap costs nothing: each pass tops the set back up.
MAXSUB=${MAXSUB:-1000}
missing(){ for i in $(seq 1 $N | while read x; do printf "%03d\n" "$x"; done); do
             [ -e "$R/$1/chunks/lund_$i.dat" ] || continue
             [ -s "$R/$1/dst_$2/excluradpy_${CHAN}_${1}_fa18_${2}_bg${3}_$i.hipo" ] || echo $((10#$i))
           done | head -n "$MAXSUB" | awk '
             NR==1          { s=$1; p=$1; next }
             $1==p+1        { p=$1; next }
                            { out = out sep (s==p ? s : s "-" p); sep=","; s=$1; p=$1 }
             END            { if (NR) print out sep (s==p ? s : s "-" p) }'; }
queued(){ squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^$1$"; }

submit(){ # mode pol bgcur jobname timelimit indices
  local m=$1 p=$2 bg=$3 jn=$4 tl=$5 idx=$6
  [ -z "$idx" ] && return 0
  cd "$EX" || return 1
  # sbatch's output went straight into the log line whatever it was, so a
  # refusal was recorded as "submitted ... -> sbatch: error: ...".  STATUS.txt
  # and the log then both read as though the work were queued, which is how 140
  # consecutive failures went unnoticed.  A job id is all digits; anything else
  # is a failure and has to say so.
  local id rc
  id=$(sbatch --parsable --export=ALL,BGCUR="$bg" --array="$idx" \
       --job-name="$jn" --time="$tl" production/farm/sim_array.sbatch "$m" "$p" 2>&1)
  rc=$?
  case $id in
    *[!0-9]*|"") say "SUBMIT FAILED $jn rc=$rc array=$idx -- $id"; return 1 ;;
  esac
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
  # The no-background sets go LAST.  VPK, 2026-10-27: they are a systematic
  # check, the with-background sets are the measurement, and on 2026-10-27 the
  # conductor put 4000 no-background tasks in the queue ahead of a
  # with-background campaign that was 14 % done.  So they are gated on all four
  # with-background sets being FULL, not merely on the one proving task.
  #
  # The proving-task gate is kept inside that: one clean task before 999 follow
  # it.  It earned itself on 2026-10-14, when MS was unset in the
  # no-background branch under `set -u` and every task died at line 198 --
  # after a full six-hour GEMC pass.
  bgfull=1
  for m in rad born; do for p in inb outb; do
    bg=$([ "$p" = inb ] && echo "${BGIN%%_*}" || echo "${BGOUT%%_*}")
    [ "$(have "$m" "$p" "$bg")" -ge "$N" ] || bgfull=0
  done; done

  if [ "$bgfull" -eq 0 ]; then
    :   # with-background still running; no-background waits its turn
  elif [ "$(have rad inb none)" -ge 1 ]; then
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
    # REGENERATE THE MANIFEST ON THE WAY OUT.
    #
    # make_manifest.py derives everything from the directory, which is why it
    # cannot drift from its contents -- but only at the moment it is run.  The
    # phi and pi0 manifests were written on 2026-10-23, before their GEMC, and
    # then said "DST files 0, ntuples: none" for three days while the
    # directories filled with 1600 and 3200 of each.  Anyone reading them
    # concluded the campaign had never run.  A derived document that is not
    # re-derived is exactly as stale as a typed one, which is the thing the
    # generator was written to prevent.
    #
    # The completion of a campaign is the one moment when the directory is
    # final, so that is where it belongs.  Failure here must not change the
    # exit: the campaign IS complete and the conductor must still say so.
    if [ -x "$(command -v python3)" ] && [ -f "$EX/production/analysis/make_manifest.py" ]; then
      if python3 "$EX/production/analysis/make_manifest.py" "$R" --write >/dev/null 2>&1; then
        say "MANIFEST.txt regenerated"
      else
        say "WARNING: could not regenerate MANIFEST.txt -- it still describes the directory as it was BEFORE this campaign ran"
      fi
    fi
    status; exit 0
  fi

  status
  sleep ${POLL:-1800}
done
