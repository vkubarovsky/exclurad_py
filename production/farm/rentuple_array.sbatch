#!/bin/bash
#SBATCH --job-name=rentuple
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --output=/volatile/clas12/vpk/rentuple_logs/%A_%a.out
#SBATCH --error=/volatile/clas12/vpk/rentuple_logs/%A_%a.out
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
#   usage:  PROD_DIR=<campaign root> sbatch --array=1-N rentuple_array.sbatch
#           (N = number of lines in $PROD_DIR/rentuple.list, written by the
#            --list pass below)
#
#           bash rentuple_array.sbatch --list     # build the file list only
#
set -u
R=${PROD_DIR:?set PROD_DIR to the campaign root}
EX=${EXRU:-/home/vpk/exclurad_py}
LIST=$R/rentuple.list

# ---------------------------------------------------------------- the list
# One line per DST: <dst path> <mode> <pol>.  Modes are taken from the
# directory layout rather than a hardcoded (rad born) pair, so a campaign with
# arms set aside -- born_vz0, rad_pre_tlimits, born_itarg1 -- gets all of them.
if [ "${1:-}" = "--list" ]; then
  : > $LIST
  for d in $R/*/dst_*; do
    [ -d "$d" ] || continue
    mode=$(basename $(dirname $d)); pol=$(basename $d); pol=${pol#dst_}
    for f in $d/*.hipo; do [ -f "$f" ] && echo "$f $mode $pol" >> $LIST; done
  done
  echo "$LIST: $(wc -l < $LIST) DSTs"
  awk '{print $2}' $LIST | sort | uniq -c
  exit 0
fi

I=${SLURM_ARRAY_TASK_ID:?no array id}
read -r DST MODE POL < <(sed -n "${I}p" $LIST)
[ -n "${DST:-}" ] || { echo "FATAL: line $I of $LIST is empty"; exit 1; }
[ -f "$DST" ]     || { echo "FATAL: $DST does not exist"; exit 1; }
TAG=$(basename "$DST" .hipo)
OUT=$R/$MODE/out_$POL/$TAG.f32
echo "task $I: $TAG   mode $MODE  pol $POL"

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
[ -f "$NTG" ] || { echo "FATAL: reader $NTG missing"; exit 1; }
NFLD=$(awk '/final int NFLD/{print $NF+0; exit}' "$NTG")
[ "${NFLD:-0}" -gt 0 ] || { echo "FATAL: cannot read NFLD from $NTG"; exit 1; }
echo "channel $CHAN  model ${MODEL:-?}  hpid $HPID  reader $(basename $NTG)  NFLD $NFLD"

# how many records the existing file holds, so the rebuild can be compared
OLDSZ=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
OLDN=0
if [ "$OLDSZ" -gt 0 ]; then
  for f in 80 73 55 50; do
    if [ $((OLDSZ % (f*4))) -eq 0 ]; then OLDN=$((OLDSZ / (f*4))); OLDF=$f; break; fi
  done
  echo "existing f32: $OLDSZ bytes = $OLDN records x ${OLDF:-?} fields"
fi

W=/volatile/clas12/vpk/rentuple_work/${SLURM_ARRAY_JOB_ID:-local}_$I
rm -rf $W; mkdir -p $W "$R/$MODE/out_$POL"; cd $W
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
  echo "FATAL: the reader wrote nothing -- keeping $OUT as it was"; exit 1; fi
if [ $((NEWSZ % (NFLD*4))) -ne 0 ]; then
  echo "FATAL: $NEWSZ bytes is not a whole number of $NFLD-field records"; exit 1; fi
NEWN=$((NEWSZ / (NFLD*4)))
if [ "$OLDN" -gt 0 ] && [ "$NEWN" -ne "$OLDN" ]; then
  echo "FATAL: rebuilt $NEWN records against the old file's $OLDN."
  echo "       Same DST, same reader logic -- a different count means the"
  echo "       channel or the hadron pid is wrong, not that the DST changed."
  echo "       keeping $OUT as it was"; exit 1; fi
echo "rebuilt $NEWN records x $NFLD fields = $NEWSZ bytes"

mv -f $W/new.f32 "$OUT"
cd /; rm -rf $W
ls -la "$OUT"
