#!/bin/bash
# Catch-up pass: the GEMC output already exists, so redo only what failed.
#
# The 2026-09-25 born array ran GEMC for 3.3 h per chunk and then died in
# bg-merger with
#     No native library found for os.name=Linux, os.arch=x86_64
# The identical command with the identical modules works on ifarm2402, so it
# is the compute node, not the environment: sqlite-jdbc extracts its .so into
# java.io.tmpdir (/tmp) and dlopens it, and on the node that fails -- /tmp is
# either noexec or full after GEMC has spent three hours there.  Point the
# driver at the job's own directory on /volatile instead.
#
# Two other things this fixes:
#   - the merged.hipo guard was `[ -s ... ]`, which a 336 kB header passes;
#     it now demands tens of MB, the way the gemc.hipo guard already did.
#   - gemc.hipo is kept until the DST exists, so a failure here never costs
#     three hours of GEMC again.
set -u
MODE=${1:?born or rad}
POL=${2:?inb or outb}
# Must match sim_array.sbatch exactly: this resumes ITS work directory, so a
# mismatched polarity would merge the wrong background into a surviving
# gemc.hipo and nothing downstream would notice.
case "$POL" in
  inb)  BGPOL=tor-1.00_sol-1.00 ;;
  outb) BGPOL=tor+1.00_sol-1.00 ;;
  *)    echo "FATAL: polarity must be inb or outb, got '$POL'"; exit 1 ;;
esac
# ONE production directory for all three stages.  Until 2026-09-28 this
# script said 20260924 while sim_array.sbatch said t2_20260925: stage A
# wrote where stage B did not look.  Set PROD_DIR once and export it.
R=${PROD_DIR:-/volatile/clas12/vpk/prod_pi0_rga_fa18_inb_t2_20260925}
I=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
W=$R/$MODE/work_$POL/w_$I
BGDIR=/cache/clas12/rg-a/production/background/rga_fall2018/$BGPOL/50nA_10604MeV/10k
BG=$(printf "%s/%05d.hipo" "$BGDIR" $(( (SLURM_ARRAY_TASK_ID - 1) % 100 + 1 )))
DET='DC,FTOF,ECAL,HTCC,LTCC,BST,BMT,CND,CTOF,FTCAL,FTHODO'
CFG=/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/prod
YAML=$CFG/coatjava/10.0.7/rga_fall2018.yaml
OUTTAG=excluradpy_pi0_${MODE}_fa18_${POL}_bg50nA_$I

[ -s "$W/gemc.hipo" ] || { echo "FATAL: no gemc.hipo in $W -- nothing to catch up"; exit 1; }
SZ=$(stat -c %s "$W/gemc.hipo")
[ "$SZ" -lt 1000000 ] && { echo "FATAL: gemc.hipo is only $SZ bytes"; exit 1; }
# Same tape guard as sim_array: /cache is tape-backed and an evicted file
# can be a stub that passes -f and then feeds bg-merger nothing.
if [ ! -s "$BG" ] || [ "$(stat -c %s "$BG")" -lt 1000000 ]; then
  echo "FATAL: background $BG is missing or not staged ($(stat -c %s "$BG" 2>/dev/null || echo absent) bytes)."
  echo "       Stage it first:  jcache get /mss/clas12/rg-a/production/background/rga_fall2018/$BGPOL/50nA_10604MeV/10k/*.hipo"
  exit 1
fi
echo "reusing gemc.hipo: $SZ bytes, background $(basename "$BG")"

export TMPDIR=$W
source /etc/profile.d/modules.sh
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module load sqlite/5.13
export RCDB_CONNECTION=mysql://null
module load coatjava/10.0.7
# Java 21 reads JDK_JAVA_OPTIONS; bg-merger and recon-util invoke java directly
# with fixed flags, so this is the only way to reach the JVM.
export JDK_JAVA_OPTIONS="-Djava.io.tmpdir=$W -Dorg.sqlite.tmpdir=$W"
which recon-util bg-merger

cd "$W"
T1=$(date +%s)
rm -f merged.hipo
bg-merger -b "$BG" -i gemc.hipo -o merged.hipo -d "$DET" > bg.log 2>&1
T2=$(date +%s); MS=$(stat -c %s merged.hipo 2>/dev/null || echo 0)
echo "BGMERGE seconds: $((T2-T1))  size: $MS"
if [ "$MS" -lt 10000000 ]; then echo "FATAL: merged.hipo is $MS bytes"; tail -25 bg.log; exit 1; fi

rm -f recon.hipo
recon-util -y "$YAML" -i merged.hipo -o recon.hipo > recon.log 2>&1
T3=$(date +%s); RS=$(stat -c %s recon.hipo 2>/dev/null || echo 0)
echo "RECON seconds: $((T3-T2))  size: $RS"
if [ "$RS" -lt 10000000 ]; then echo "FATAL: recon.hipo is $RS bytes"; tail -25 recon.log; exit 1; fi

module purge
export PATH=/group/clas12/packages/groovy/4.0.3/bin:$PATH
unset JDK_JAVA_OPTIONS
/group/clas12/packages/coatjava/9.0.2/bin/run-groovy "$R/hipo_to_ntuple.groovy" \
    recon.hipo 1000000 "$R/$MODE/out_$POL/$OUTTAG.f32" 2>&1 | grep -E "DONE|FATAL|Exception" || true
/group/clas12/packages/coatjava/9.0.2/bin/hipo-utils -filter \
    -b "RUN::*,REC::*,RECFT::*,MC::*" -merge \
    -o "$R/$MODE/dst_$POL/$OUTTAG.hipo" recon.hipo > /dev/null 2>&1
T4=$(date +%s); echo "NTUPLE+DST seconds: $((T4-T3))"
ls -la "$R/$MODE/dst_$POL/$OUTTAG.hipo" "$R/$MODE/out_$POL/$OUTTAG.f32" 2>/dev/null
# only now is gemc.hipo expendable
if [ -s "$R/$MODE/dst_$POL/$OUTTAG.hipo" ]; then cd "$R"; rm -rf "$W"; fi
echo "DONE $MODE $I"
