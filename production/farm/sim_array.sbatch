#!/bin/bash
# Defaults, so a bare `sbatch sim_array.sbatch` cannot run with the farm's
# 512 MB default memory: on 2026-10-08 a J/psi test submitted without --mem
# had GEMC and recon-util (Xmx1536m) OOM-killed in all three jobs.  Flags on
# the sbatch command line still override these.
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#
# HOW TO RUN A CAMPAIGN WITH THESE: docs/farm_practice.md
# It carries the sizing, the chaining, and every trap that has cost
# a run here.  Read it before writing a chain of your own.
#
# Stage B: GEMC, background merge, reconstruction, ntuple and DST, for one chunk.
#
#   sbatch --array=1-200 --job-name=sim_rad_inb  --time=10:00:00 sim_array.sbatch rad inb
#   sbatch --array=1-200 --job-name=sim_rad_outb --time=10:00:00 sim_array.sbatch rad outb
#
# POLARITY is the second argument, added 2026-09-28 (VPK: inbending AND
# outbending).  Only three things depend on it -- the torus scale, the
# background directory, and the output names -- and the gcard and yaml do
# NOT: the yaml names only the SYMMETRIC field maps, and recon takes the
# sign from RUN::config, which gemc writes from -SCALE_FIELD.  Both
# polarities share one generated LUND, because the generator knows nothing
# about the torus.
#
# Configuration: set by RGCFG/GEMCV/COATV/BGTREE/BGCUR/RGTAG below; defaults rga_fall2018,
# gemc 5.13 + coatjava 10.0.7 -- the pass-2 pair from clas12-config setup.json.
# Background: 50 nA over the eleven detectors the official production merges;
# bg-merger's own default is DC,FTOF only, which would leave the calorimeter
# clean -- no background under the pi0 photons, exactly where it is needed.
#
# THREE FIXES after the 2026-09-25 born array (104 of 200 jobs lost):
#
# 1. sqlite-jdbc could not load its native library on the compute nodes:
#        No native library found for os.name=Linux, os.arch=x86_64
#    The identical command with the identical modules works on ifarm2402, so
#    it is the node.  The driver extracts its .so into java.io.tmpdir (/tmp)
#    and dlopens it; on the node that fails -- /tmp is either noexec or full
#    after GEMC has spent three hours there.  Both cures are the same: give
#    the JVM a temp directory on /volatile.  bg-merger and recon-util call
#    java directly with fixed flags, so JDK_JAVA_OPTIONS (Java 9+) is the
#    only way in.
#
# 2. The merged.hipo guard was `[ -s merged.hipo ]`.  A header-only hipo is
#    336 kB and passes it, so the job fed garbage to recon-util and reported
#    a reconstruction failure for what was a merge failure.  Both guards now
#    demand tens of MB, the way the gemc.hipo one already did.
#
# 3. gemc.hipo is kept until the DST exists.  Three hours of GEMC must never
#    again be thrown away by a failure in a step that takes minutes; on a
#    failure the work directory survives and post_array.sbatch resumes from it.
set -u
MODE=${1:?born or rad}
POL=${2:?inb or outb}
case "$POL" in
  inb)  TORUS=-1.00; BGPOL=tor-1.00_sol-1.00 ;;
  outb) TORUS=+1.00; BGPOL=tor+1.00_sol-1.00 ;;
  *)    echo "FATAL: polarity must be inb or outb, got '$POL'"; exit 1 ;;
esac
# ONE production directory for all three stages.  Until 2026-09-28 this
# script said 20260924 while sim_array.sbatch said t2_20260925: stage A
# wrote where stage B did not look.  Set PROD_DIR once and export it.
R=${PROD_DIR:-/volatile/clas12/vpk/prod_pi0_rga_fa18_inb_t2_20260925}
I=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
# The 50 nA files live on TAPE and are evicted from /cache.  On 2026-09-28
# both polarities' directories were empty and `jcache get` had to stage 100
# files each, which is why the guard below exists: it turns a cold cache
# into one line instead of 200 jobs dying inside gemc three hours in.
# RUN GROUP.  Everything below defaults to RG-A fall 2018 exactly as before,
# so an existing submission is unchanged.  RG-B needs six of them and NOTHING
# else, which is the point of naming them: the run group is not one setting,
# it is a gcard, a yaml, two module versions, a background tree and a beam
# energy, and picking five of the six right is how a sample ends up passed
# through the wrong detector without any error at all.
#
#   RG-B spring 2019 (deuterium, inbending only -- rgb_fall2019 is torus +1.00):
#     export RGCFG=rgb_spring2019 GEMCV=5.11 COATV=11.1.0 \
#            BGTREE=rg-b/production/background/rgb_spring2019 \
#            BGCUR=50nA_10599MeV RGTAG=sp19
#
# The gemc and coatjava versions are an APPROVED PAIR from clas12-config
# setup.json, not two independent choices: 5.13+10.0.7 for RG-A pass 2,
# 5.11+11.1.0 for RG-B.  Do not mix halves of two pairs.
RGCFG=${RGCFG:-rga_fall2018}
GEMCV=${GEMCV:-5.13}
COATV=${COATV:-10.0.7}
BGTREE=${BGTREE:-rg-a/production/background/rga_fall2018}
BGCUR=${BGCUR:-50nA_10604MeV}
RGTAG=${RGTAG:-fa18}

# THE WORK DIRECTORY MUST CARRY THE BACKGROUND SETTING.  Until 2026-10-11 it
# was $R/$MODE/work_$POL/w_$I -- mode, polarity and task index only -- so two
# jobs that differ ONLY in the background (45 nA against 50 nA, or merging
# against not) share one scratch directory and the second one's `rm -rf "$W"`
# deletes the first one's gemc.hipo mid-run.  That is exactly what happened:
# 12739796_1 (rad inb, no background) ran 09:36-12:25 and 12742865_1 (rad inb,
# 45 nA) started at 10:45 into the same directory; the first job then spent
# 10107 s in GEMC and ended on "gemc.hipo is 0 bytes".  The victim was the
# no-background job and the 45 nA output is intact -- but with the whole
# no-background campaign still to run, every one of its 800 tasks would have
# collided with its with-background twin.  OUTTAG already carried the
# background tag, so the OUTPUTS never collided; only the scratch did, which
# is why it looked like a GEMC failure.
W=$R/$MODE/work_${POL}_bg${BGCUR%%_*}/w_$I
# The sqlite module tracks the gemc version, not coatjava's.
SQLV=${SQLV:-$GEMCV}
# BGCUR=none runs WITHOUT background merging: gemc.hipo goes straight to
# recon-util.  VPK's campaign asks for both, with and without, for every
# polarity, so this is a first-class mode and not a debug switch.
NOBG=0; [ "$BGCUR" = "none" ] && NOBG=1
BGDIR=/cache/clas12/$BGTREE/$BGPOL/$BGCUR/10k
# 100 background files for 200 tasks: each is used by two tasks, whose
# generated events are independent, so no correlation is introduced
BG=$(printf "%s/%05d.hipo" "$BGDIR" $(( (SLURM_ARRAY_TASK_ID - 1) % 100 + 1 )))
DET='DC,FTOF,ECAL,HTCC,LTCC,BST,BMT,CND,CTOF,FTCAL,FTHODO'
CFG=/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/prod
GC=$CFG/gemc/$GEMCV/$RGCFG.gcard
YAML=$CFG/coatjava/$COATV/$RGCFG.yaml
# The channel comes from the card's model line (pi0.amp2614 -> pi0,
# jpsi.diffrad_vm -> jpsi).  It used to be the literal "pi0", so a J/psi
# campaign wrote files called excluradpy_pi0_*.
CHAN=$(awk '{sub(/!.*/,"")} /[^ \t]/{print $1; exit}' "$R/cards/rga_$MODE.txt" 2>/dev/null | cut -d. -f1)
CHAN=${CHAN:-pi0}
OUTTAG=excluradpy_${CHAN}_${MODE}_${RGTAG}_${POL}_bg${BGCUR%%_*}_$I   # bgnone when off
for f in "$GC" "$YAML"; do
  [ -f "$f" ] || { echo "FATAL: $f does not exist -- wrong run group or version"; exit 1; }
done

mkdir -p "$R/$MODE/out_$POL" "$R/$MODE/dst_$POL"
rm -rf "$W"; mkdir -p "$W"; cd "$W"
export TMPDIR=$W                         # not /tmp: see fix 1
source /etc/profile.d/modules.sh
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module use /cvmfs/oasis.opensciencegrid.org/jlab/geant4/modules
module load gemc/$GEMCV
module load sqlite/$SQLV
export RCDB_CONNECTION=mysql://null
module load coatjava/$COATV
export JDK_JAVA_OPTIONS="-Djava.io.tmpdir=$W -Dorg.sqlite.tmpdir=$W"
which gemc recon-util bg-merger

cp "$R/$MODE/chunks/lund_$I.dat" lund.dat || { echo "FATAL: no chunk lund_$I.dat"; exit 1; }
# Not just "does it exist".  /cache is tape-backed: an evicted file can be
# a zero-length or stub entry that passes -f and then feeds gemc nothing.
# Stage the whole directory with `jcache get` before submitting.
if [ "$NOBG" = 1 ]; then
  echo "background merging: OFF (BGCUR=none)"
elif [ ! -s "$BG" ] || [ "$(stat -c %s "$BG")" -lt 1000000 ]; then
  echo "FATAL: background $BG is missing or not staged ($(stat -c %s "$BG" 2>/dev/null || echo absent) bytes)."
  echo "       Stage it first:  jcache get /mss/clas12/$BGTREE/$BGPOL/$BGCUR/10k/*.hipo"
  exit 1
fi
echo "chunk $(grep -c '^ *[0-9]* *1 *1 ' lund.dat) events, torus $TORUS, background $([ "$NOBG" = 1 ] && echo NONE || basename "$BG")"

T0=$(date +%s)
gemc -USE_GUI=0 -NGENP=100 -N=100000 -INPUT_GEN_FILE="lund, lund.dat" "$GC" \
     -SCALE_FIELD="binary_torus, $TORUS" -SCALE_FIELD="binary_solenoid, -1.00" \
     -OUTPUT="hipo, gemc.hipo" -INTEGRATEDRAW="*" > gemc.log 2>&1
T1=$(date +%s); GS=$(stat -c %s gemc.hipo 2>/dev/null || echo 0)
echo "GEMC seconds: $((T1-T0))  size: $GS"
# a gemc.hipo of a few kB means gemc read nothing -- almost always a '#' in the LUND
if [ "$GS" -lt 1000000 ]; then echo "FATAL: gemc.hipo is $GS bytes, gemc read no events"; tail -20 gemc.log; exit 1; fi

rm -f merged.hipo
if [ "$NOBG" = 1 ]; then
  # No merge: recon reads gemc.hipo directly.  A symlink rather than a copy,
  # because a 5000-event gemc.hipo is ~100 MB and the work directory is on
  # /volatile.
  ln -sf gemc.hipo merged.hipo
  T2=$(date +%s)
  # MS MUST BE SET HERE TOO.  The recon guard below compares RS against
  # MS/20, and with `set -u` an unset MS is a fatal error, not a zero: the
  # first no-background task died at that line after a full GEMC pass, 6 h
  # in, and the conductor resubmitted it twice before the watcher stopped it
  # on 2026-10-14.  merged.hipo IS gemc.hipo here, so its size is gemc's.
  MS=$(stat -c %s gemc.hipo 2>/dev/null || echo 0)
  echo "BGMERGE skipped (no background), merged.hipo -> gemc.hipo, $MS bytes"
else
bg-merger -b "$BG" -i gemc.hipo -o merged.hipo -d "$DET" > bg.log 2>&1
T2=$(date +%s); MS=$(stat -c %s merged.hipo 2>/dev/null || echo 0)
echo "BGMERGE seconds: $((T2-T1))  size: $MS"
# SCALE-FREE, because 10 MB is a 5000-event number and a 100-event test chunk
# legitimately merges to 3.5 MB -- the guard fired on a perfectly good file on
# 2026-09-28 and the chain looked broken when it was not.  What is actually
# being caught is a header-only hipo, which is ~336 kB WHATEVER the chunk size,
# so compare against gemc.hipo: the merge adds background hits, so the result
# must be strictly larger than its own input, which already passed its guard.
GS=$(stat -c %s gemc.hipo)
if [ "$MS" -le "$GS" ]; then
  echo "FATAL: merged.hipo is $MS bytes, not larger than gemc.hipo at $GS -- the merge added nothing"
  tail -25 bg.log; exit 1
fi
fi

rm -f recon.hipo
recon-util -y "$YAML" -i merged.hipo -o recon.hipo > recon.log 2>&1
T3=$(date +%s); RS=$(stat -c %s recon.hipo 2>/dev/null || echo 0)
echo "RECON seconds: $((T3-T2))  size: $RS"
# Same reasoning.  NOTE reconstruction GROWS the file -- measured 3.3 -> 7.3 MB,
# because the REC:: banks are added on top of the hits rather than replacing
# them.  The threshold is still a fraction of the input, deliberately: it is
# there to catch output that is empty, not to model the size.
if [ "$RS" -lt $((MS / 20)) ]; then
  echo "FATAL: recon.hipo is $RS bytes against merged.hipo's $MS -- reconstruction produced almost nothing"
  tail -25 recon.log; exit 1
fi

module purge                              # java 21 from coatjava breaks run-groovy
unset JDK_JAVA_OPTIONS
export PATH=/group/clas12/packages/groovy/4.0.3/bin:$PATH
# The groovy lives in the production directory, copied there at setup.  On
# 2026-09-28 it was missing, run-groovy threw FileNotFoundException, the `||
# true` swallowed it and the job printed DONE with no ntuple at all -- 200
# jobs would have "succeeded" and produced nothing to analyse.  So: take it
# from the checkout when the production directory has no copy, and check the
# result rather than trusting the exit status of a pipeline ending in grep.
# THE HADRON, and this is not a detail.  hipo_to_ntuple.groovy takes the
# hadron's pid as its FIFTH argument and DEFAULTS TO 2212.  Nothing passed it,
# so every sample was analysed as if the detected hadron were a proton.
#
# On a proton target that is right.  On a NEUTRON target the LUND carries two
# baryons -- the struck neutron, which is the physics, and the spectator
# PROTON -- so the default picks the spectator, and nothing complains: the
# groovy finds a proton in every event, badGen stays 0, the ntuple is full,
# and every row is the wrong particle.  Measured on 2026-09-28: 100 of 100.
#
# So it is derived from the CARD rather than from an environment variable one
# has to remember.  The model name already says which nucleon was struck --
# pi0n.* and etan.* are the neutron models -- and it is the same file that
# decided the physics, so the two cannot drift apart.  HPID still overrides.
CARD=$R/cards/rga_$MODE.txt
if [ -z "${HPID:-}" ] && [ -f "$CARD" ]; then
  MODEL=$(awk '{sub(/!.*/,"")} /[^ \t]/{print $1; exit}' "$CARD")
  case "$MODEL" in
    pi0n.*|etan.*) HPID=2112 ;;
    *)             HPID=2212 ;;
  esac
fi
HPID=${HPID:-2212}
echo "hadron pid: $HPID (model ${MODEL:-unknown})"
# WHICH NTUPLE SCRIPT.  $R wins, so an operator can drop a custom one in the
# production directory; otherwise it is chosen BY CHANNEL.  The pi0 script
# requires the pi0's two photons and writes NOTHING for a J/psi event, and
# the size guard below then kills the task -- after gemc and recon have
# already run, which is the most expensive place to fail.  Leaving that to a
# manual copy step is how it gets forgotten.
NTG="$R/hipo_to_ntuple.groovy"
# jpsi_to_ntuple.groovy is J/psi SPECIFIC, not vector-generic: it finds the
# meson row by pid 443 and the daughters by pid +-11 with parent = that row
# (lines 57-66).  For phi -> K+K- (333, +-321) and rho -> pi+pi- (113,
# +-211) it finds neither, writes nothing, and the size guard below kills
# the task -- after gemc and recon, the expensive place.  Routing phi and
# rho to it, as the first version of this block did on 2026-10-09, would
# have reproduced exactly the failure the routing was added to prevent.
# Refuse at the start instead, which costs a minute rather than a chunk.
case "$CHAN" in
  jpsi)     DEFNT=jpsi_to_ntuple.groovy ;;
  pi0|pi0n|eta|etan) DEFNT=hipo_to_ntuple.groovy ;;
  phi|rho)  [ -f "$R/hipo_to_ntuple.groovy" ] || {
              echo "FATAL: no ntuple reader for channel '$CHAN'."
              echo "       jpsi_to_ntuple.groovy is J/psi specific (pid 443, daughters +-11);"
              echo "       hipo_to_ntuple.groovy is pi0/eta specific (two photons from the pi0 row)."
              echo "       Generalise one of them, or put a reader at \$PROD_DIR/hipo_to_ntuple.groovy."
              exit 1; }
            DEFNT=hipo_to_ntuple.groovy ;;
  *)        DEFNT=hipo_to_ntuple.groovy ;;
esac
# NOT $0.  slurm runs a COPY of this script from its spool directory, so $0
# points there and not at the checkout -- the fallback silently resolved to
# nothing and 50 tasks died after GEMC and reconstruction had already run.
# $EXRU is the checkout the job was told to use, and gen_array uses it too.
[ -f "$NTG" ] || NTG="${EXRU:-$HOME/exclurad_py}/production/gemc_check/$DEFNT"
[ -f "$NTG" ] || { echo "FATAL: $DEFNT found neither in $R nor in the checkout"; exit 1; }
echo "ntuple script: $NTG (channel $CHAN)"
/group/clas12/packages/coatjava/9.0.2/bin/run-groovy "$NTG" \
    recon.hipo 1000000 "$R/$MODE/out_$POL/$OUTTAG.f32" 0 "$HPID" 2>&1 | grep -E "DONE|FATAL|Exception" || true
NS=$(stat -c %s "$R/$MODE/out_$POL/$OUTTAG.f32" 2>/dev/null || echo 0)
if [ "$NS" -lt 1000 ]; then
  echo "FATAL: ntuple $OUTTAG.f32 is $NS bytes -- keeping gemc.hipo for post_array"; exit 1
fi
/group/clas12/packages/coatjava/9.0.2/bin/hipo-utils -filter \
    -b "RUN::*,REC::*,RECFT::*,MC::*" -merge \
    -o "$R/$MODE/dst_$POL/$OUTTAG.hipo" recon.hipo > /dev/null 2>&1
T4=$(date +%s); echo "NTUPLE+DST seconds: $((T4-T3))"
echo "TOTAL seconds: $((T4-T0))"
ls -la "$R/$MODE/dst_$POL/$OUTTAG.hipo" "$R/$MODE/out_$POL/$OUTTAG.f32" 2>/dev/null
# only a finished DST makes three hours of GEMC expendable
if [ -s "$R/$MODE/dst_$POL/$OUTTAG.hipo" ]; then cd "$R"; rm -rf "$W"; fi
echo "DONE $MODE $POL $I"
