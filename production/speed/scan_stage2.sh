#!/bin/bash
# Scan the two stage-2 efficiency levers.
#
# WHY: the generator spends its time in the EXACT rc_factor, and production
# calls it ~2.5 times per written event because stage 2 discards ~60 %.  The
# floor is 1.0.  Neither lever changes the physics -- stage 2 corrects
# stage 1 exactly, and the sample is exactly unweighted for ANY B above the
# true maximum -- so both are pure sampling efficiency.  What they can break
# is the BOUND, which is why b_viol is reported for every configuration.
#
#   TABLE_SHAPE  the eta proposal grid: a better proposal tightens the
#                exact/proposal ratio r
#   B_SHAPE / B_PROBES  the stage-2 bound grid: more cells and more probes
#                mean each cell gets its own bound instead of falling back
#                on the global maximum set by the single worst cell
#
# Usage:  bash production/speed/scan_stage2.sh [NEV]
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
. "$HERE/production/speed/lund_outroot.sh"
NEV=${1:-200}
OUT=$LUND_OUTROOT/scan_stage2
mkdir -p "$OUT"
CARD=$OUT/card.txt
sed -e "s|^5000  |$NEV    |" -e "s|^\./exclurad_out|$OUT/run|" \
    "$HERE/cards/rad_pi0_rga.txt" > "$CARD"

run_one () {  # name TABLE_SHAPE B_SHAPE B_PROBES
  local name=$1 ts=$2 bs=$3 bp=$4
  local d=$OUT/$name; rm -rf "$d"; mkdir -p "$d"
  ( cd "$HERE"
    export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
    export ETA_TABLE=$d/eta.npz MAJ_TABLE=$d/maj.npz
    [ -n "$ts" ] && export TABLE_SHAPE=$ts || unset TABLE_SHAPE
    [ -n "$bs" ] && export B_SHAPE=$bs     || unset B_SHAPE
    [ -n "$bp" ] && export B_PROBES=$bp    || unset B_PROBES
    sed "s|$OUT/run|$d/run|" "$CARD" > "$d/card.txt"
    /usr/bin/time -p python3 -m exclurad_py.run "$d/card.txt" > "$d/log" 2>&1
  )
  python3 - "$d" "$name" "$ts" "$bs" "$bp" <<'PY'
import json,sys,glob,re
d,name,ts,bs,bp = sys.argv[1:6]
try:
    cj = glob.glob(f"{d}/run/**/config.json", recursive=True)[0]
    r  = json.load(open(cj)); r = r.get("report", r)
    log = open(f"{d}/log").read()
    ms  = re.search(r"([\d.]+) ms/event", log)
    setup = re.search(r"setup (\d+)s", log)
    interp = re.search(r"'rms': ([\d.]+)", log)
    a1, a2 = r.get("accepted", 0), r.get("accept2", 0)
    print(f"{name:14s} ts={ts or 'dflt':9s} bs={bs or 'dflt':9s} bp={bp or 'dflt':5s} "
          f"| {float(ms.group(1)) if ms else -1:8.1f} ms/ev "
          f"| stage2 {100*a2/max(a1,1):5.1f}% | RC/ev {a1/max(a2,1):5.2f} "
          f"| b_viol {r.get('b_viol',-1):3d} max {r.get('max_b_viol',0):.2f} "
          f"| build {setup.group(1) if setup else '?':>3s}s | rms {interp.group(1) if interp else '?'}")
except Exception as e:
    print(f"{name:14s} FAILED: {e}")
PY
}

echo "NEV=$NEV   output: $OUT"
run_one baseline      ""         ""        ""
run_one table_x2      "7,7,7,11" ""        ""
run_one table_x4      "10,10,10,16" ""     ""
run_one bshape_fine   ""         "6,6,6,8" "2000"
run_one bprobes_only  ""         ""        "3000"
run_one both          "10,10,10,16" "6,6,6,8" "2000"
