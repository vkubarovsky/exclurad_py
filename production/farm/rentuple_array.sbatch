#!/bin/bash
#SBATCH --job-name=rentuple
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --output=/farm_out/vpk/rentuple_%A_%a.out
#SBATCH --error=/farm_out/vpk/rentuple_%A_%a.out
#   the farm's cli_filter refuses --output anywhere but /farm_out or /dev/null
#
# RE-READ AN EXISTING DST INTO THE CURRENT NTUPLE RECORD.  No gemc, no
# reconstruction -- the DST is primary data and is never touched; the .f32 is
# derived and is rebuilt from it.
#
# Why this exists: on 2026-10-22 the readers grew the vertex fields (vector
# 73 -> 80 floats, pseudoscalar 50 -> 55).  Every campaign simulated before
# that carries DSTs whose MC::Lund and REC::Particle vz were simply never
# read.  Re-running gemc for them would cost thousands of farm hours; reading
# the DST again costs about a minute per file.
#
# Verified before this script was written, on
# prod_jpsi_f18in_20261009/rad_pre_tlimits/.../001.hipo: the first 73 fields of
# the rebuilt ntuple are BYTE-IDENTICAL to the 73-field file made on 10
# October.  So the rebuild adds the vertex and changes nothing else.
#
# ONE JOB DOES A BATCH OF FILES, NOT ONE.  Re-reading a DST takes 12-19
# seconds, and on 2026-10-22 this ran as 5199 one-file array tasks.  The farm
# administrators wrote the next morning: a large volume of jobs each under two
# minutes loads the scheduler out of proportion to the work, because the fixed
# per-job cost is then most of the job.  They are right.  Array task I now
# processes lines (I-1)*STRIDE+1 .. I*STRIDE of the list serially, so the same
# 5199 files go as 52 jobs of fifteen to twenty minutes.
#
#   usage:  PROD_DIR=<campaign root> bash rentuple_array.sbatch --list
#           (prints the list and the --array range to use)
#
#           PROD_DIR=<campaign root> sbatch --array=1-NJOB rentuple_array.sbatch
#
#           STRIDE=<n>   files per job, default 100
#
set -u
R=${PROD_DIR:?set PROD_DIR to the campaign root}
EX=${EXRU:-/home/vpk/exclurad_py}
EX_PY=${EXRU_PY:-/work/clas12/vpk/venv-prod-numba}/bin/python
[ -x "$EX_PY" ] || EX_PY=python3
LIST=$R/rentuple.list
STRIDE=${STRIDE:-100}                     # files per array task; see the header

# ---------------------------------------------------------------- the list
# One line per DST: <dst path> <mode> <pol>.  Modes are taken from the
# directory layout rather than a hardcoded (rad born) pair, so a campaign with
# arms set aside -- born_vz0, rad_pre_tlimits, born_itarg1 -- gets all of them.
# One line per DST: <dst path> <mode> <pol> <out dir>.
#
# TWO LAYOUTS.  Campaigns from 2026-09-28 on put the polarity in the directory
# name -- dst_inb / out_inb -- because one LUND feeds both torus settings.  The
# ones before that have a bare dst/ and out/, with the polarity only in the
# file name.  Globbing dst_* missed those entirely, so the September campaigns
# could not be rebuilt at all.  The output directory is therefore derived from
# the input one by replacing the leading "dst" with "out", which gives
# out_inb for the new layout and out for the old, instead of being rebuilt
# from a polarity that the old layout does not carry.
if [ "${1:-}" = "--list" ]; then
  : > $LIST
  for d in $R/*/dst*; do
    [ -d "$d" ] || continue
    mode=$(basename $(dirname $d))
    b=$(basename $d); sfx=${b#dst}            # "_inb", or empty
    pol=${sfx#_}; pol=${pol:-none}
    outdir=$(dirname $d)/out$sfx
    for f in $d/*.hipo; do [ -f "$f" ] && echo "$f $mode $pol $outdir" >> $LIST; done
  done
  N=$(wc -l < $LIST)
  echo "$LIST: $N DSTs"
  awk '{print $2, $3}' $LIST | sort | uniq -c
  echo "submit with:  PROD_DIR=$R sbatch --array=1-$(( (N + STRIDE - 1) / STRIDE )) \\"
  echo "                  production/farm/rentuple_array.sbatch     # STRIDE=$STRIDE"
  exit 0
fi

I=${SLURM_ARRAY_TASK_ID:?no array id}
N=$(wc -l < $LIST)
LO=$(( (I - 1) * STRIDE + 1 ))
HI=$(( I * STRIDE )); [ "$HI" -gt "$N" ] && HI=$N
[ "$LO" -le "$N" ] || { echo "task $I: lines $LO.. are past the end of the list ($N) -- nothing to do"; exit 0; }
echo "task $I: lines $LO-$HI of $N  (STRIDE=$STRIDE)"
T0=$SECONDS

# ONE FILE.  Returns non-zero instead of exiting, so one bad DST does not take
# the other forty-nine in the batch with it; the failures are counted and named
# in the summary below, and the job exits non-zero if there were any.
do_one() {
  local L=$1
  read -r DST MODE POL OUTDIR < <(sed -n "${L}p" $LIST)
  [ -n "${DST:-}" ]    || { echo "FATAL: line $L of $LIST is empty"; return 1; }
  [ -f "$DST" ]        || { echo "FATAL: $DST does not exist"; return 1; }
  [ -n "${OUTDIR:-}" ] || { echo "FATAL: line $L has no output directory -- rebuild the list with --list"; return 1; }
  TAG=$(basename "$DST" .hipo)
  OUT=$OUTDIR/$TAG.f32
  echo "--- line $L: $TAG   mode $MODE  pol $POL"

  # ------------------------------------------------------------- the channel
  # From the card that produced the arm, as everywhere else.  An arm set aside
  # (born_vz0, rad_pre_tlimits) has no card of its own, so fall back to the base
  # mode's card: rad_pre_tlimits -> rga_rad.txt, born_vz0 -> rga_born.txt.
  BASE=$MODE; case "$MODE" in rad*) BASE=rad ;; born*) BASE=born ;; esac
  CARD=$R/cards/rga_$BASE.txt
  MODEL=$(awk '{sub(/!.*/,"")} /[^ \t]/{print $1; exit}' "$CARD" 2>/dev/null)
  CHAN=$(echo "$MODEL" | cut -d. -f1); CHAN=${CHAN:-pi0}
  case "$MODEL" in pi0n.*|etan.*) HPID=2112 ;; *) HPID=2212 ;; esac
  HPID=${HPID_OVERRIDE:-$HPID}

  # WHICH READER, AND FROM WHERE.  sim_array.sbatch prefers $R/<reader>.groovy so
  # an operator can drop a custom one into a campaign.  That is exactly wrong
  # here: the campaign roots hold the reader AS IT WAS when the campaign ran --
  # prod_phi_f18in_20261015/vm_to_ntuple.groovy is dated 15 October, 73 fields --
  # and using it would rebuild the old record and report success.  So the reader
  # comes from the checkout, always, and $R is not consulted.
  case "$CHAN" in
    jpsi|phi|rho)      NTG=$EX/production/gemc_check/vm_to_ntuple.groovy   ; NTARG5=$CHAN ;;
    pi0|pi0n|eta|etan) NTG=$EX/production/gemc_check/hipo_to_ntuple.groovy ; NTARG5=$HPID ;;
    *)                 NTG=$EX/production/gemc_check/hipo_to_ntuple.groovy ; NTARG5=$HPID ;;
  esac
  [ -f "$NTG" ] || { echo "FATAL: reader $NTG missing"; return 1; }
  NFLD=$(awk '/final int NFLD/{print $NF+0; exit}' "$NTG")
  [ "${NFLD:-0}" -gt 0 ] || { echo "FATAL: cannot read NFLD from $NTG"; return 1; }
  echo "channel $CHAN  model ${MODEL:-?}  hpid $HPID  reader $(basename $NTG)  NFLD $NFLD"

  # How many records the existing file holds, so the rebuild can be compared.
  #
  # ONLY THIS CHANNEL'S TWO LENGTHS, longer first.  The first version of this
  # tried 80 73 55 50 in turn and took whatever divided first, which is wrong
  # because a size can divide a length belonging to the other channel: an old
  # pi0 file of 5000 records x 50 fields is 250000 floats, and 250000/80 = 3125
  # exactly, so it was read as 3125 records of 80 and every one of the 800 files
  # in prod_pi0_rga_t25_20260928 was refused with "rebuilt 5000 against the old
  # file's 3125".  The rebuild had been right; the comparison was not.
  # Longer first matches campaign_lib: a new pi0 file of 5000 x 55 = 275000
  # floats also divides 50 (5500), and 55 is the true one.
  case "$CHAN" in
    jpsi|phi|rho) TRY="80 73" ;;
    *)            TRY="55 50" ;;
  esac
  # A HEADERED FILE IS ASKED, NOT MEASURED.  Since 2026-10-23 an .f32 may carry
  # a 64-byte header stating its record length; f32_format reads it and the size
  # arithmetic is the fallback for legacy files only.  Guessing is how an old pi0
  # file of 5000 x 50 came out as 3125 x 80 and 800 rebuilds were refused.
  OLDSZ=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  OLDN=0; OLDF=""
  if [ "$OLDSZ" -gt 0 ]; then
    read -r OLDF OLDN < <("$EX_PY" "$EX/production/analysis/f32_format.py" --fields "$OUT" 2>/dev/null) || { OLDF=""; OLDN=0; }
    if [ -z "${OLDF:-}" ]; then
      for f in $TRY; do
        if [ $((OLDSZ % (f*4))) -eq 0 ]; then OLDN=$((OLDSZ / (f*4))); OLDF=$f; break; fi
      done
      echo "existing f32: $OLDSZ bytes = $OLDN records x ${OLDF:-?} fields (legacy, tried $TRY)"
    else
      echo "existing f32: $OLDSZ bytes, header says $OLDN records x $OLDF fields"
    fi
  fi

  # UNIQUE PER RUNNER, not just per task index.  Driven by hand on two
  # interactive nodes at once -- jpsi on one, phi on the other -- every task
  # fell back to "local_$I", both nodes see the same /volatile, and the `rm -rf`
  # below wiped the other node's work in progress: 989 files came back "the
  # reader wrote nothing".  Nothing was lost, because the old file is only
  # replaced after the checks, but the work was.
  W=/volatile/clas12/vpk/rentuple_work/${SLURM_ARRAY_JOB_ID:-$(hostname -s)_$(basename $R)}_$L
  rm -rf $W; mkdir -p $W "$OUTDIR"; cd $W
  export TMPDIR=$W                          # /tmp is noexec on the compute nodes
  unset JDK_JAVA_OPTIONS                    # java 21 from coatjava breaks run-groovy
  export PATH=/group/clas12/packages/groovy/4.0.3/bin:$PATH
  /group/clas12/packages/coatjava/9.0.2/bin/run-groovy "$NTG" \
      "$DST" 1000000 $W/new.f32 0 "$NTARG5" 2>&1 | grep -E "DONE|FATAL|Exception" || true

  # ------------------------------------------------------------- the checking
  # Not the exit status of a pipeline that ends in grep.  Three things must hold
  # before the old file is replaced, and the old file survives if any fails.
  NEWSZ=$(stat -c %s $W/new.f32 2>/dev/null || echo 0)
  if [ "$NEWSZ" -eq 0 ]; then
    echo "FATAL: the reader wrote nothing -- keeping $OUT as it was"; return 1; fi
  # The reader now writes a header, so the record bytes are the file minus it,
  # and f32_format validates the count rather than this arithmetic doing it.
  NEWN=$("$EX_PY" "$EX/production/analysis/f32_format.py" --records $W/new.f32 2>/dev/null || echo "")
  if [ -z "$NEWN" ]; then
    if [ $((NEWSZ % (NFLD*4))) -ne 0 ]; then
      echo "FATAL: $NEWSZ bytes is not a whole number of $NFLD-field records"; return 1; fi
    NEWN=$((NEWSZ / (NFLD*4)))
  fi
  if [ "$OLDN" -gt 0 ] && [ "$NEWN" -ne "$OLDN" ]; then
    echo "FATAL: rebuilt $NEWN records against the old file's $OLDN."
    echo "       Same DST, same reader logic -- a different count means the"
    echo "       channel or the hadron pid is wrong, not that the DST changed."
    echo "       keeping $OUT as it was"; return 1; fi
  echo "rebuilt $NEWN records x $NFLD fields = $NEWSZ bytes"

  mv -f $W/new.f32 "$OUT"
  cd /; rm -rf $W
  ls -la "$OUT"
  return 0
}

NOK=0; NBAD=0; BAD=""
for L in $(seq $LO $HI); do
  if do_one "$L"; then NOK=$((NOK+1)); else NBAD=$((NBAD+1)); BAD="$BAD $L"; fi
done
echo "=== task $I done in $((SECONDS-T0))s: $NOK rebuilt, $NBAD failed"
[ "$NBAD" -eq 0 ] || { echo "    failed lines:$BAD"; exit 1; }
