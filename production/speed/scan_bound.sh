#!/bin/bash
# Follow-up scan: the stage-2 BOUND alone, with the eta table held fixed.
#
# The first scan said the eta table is not the lever -- halving its
# interpolation error bought 3 % -- and that the bound calibration is.  This
# isolates the bound: one shared default table, built once, so no
# configuration pays a build and nothing differs but B_SHAPE / B_PROBES.
#
# RC/event is the metric to read, not ms/event: it is a count, so it does
# not care what else the machine is doing.  b_viol is the thing that can go
# wrong -- a tighter bound is a bound more often exceeded.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
. "$HERE/production/speed/lund_outroot.sh"
NEV=${1:-300}
OUT=$LUND_OUTROOT/scan_bound; mkdir -p "$OUT"
SHARED=$OUT/shared_eta.npz

run_one () {  # name B_SHAPE B_PROBES
  local name=$1 bs=$2 bp=$3
  local d=$OUT/$name; rm -rf "$d"; mkdir -p "$d"
  [ -f "$SHARED" ] && cp "$SHARED" "$d/eta.npz"
  ( cd "$HERE"
    export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
    export ETA_TABLE=$d/eta.npz MAJ_TABLE=$d/maj.npz
    [ -n "$bs" ] && export B_SHAPE=$bs  || unset B_SHAPE
    [ -n "$bp" ] && export B_PROBES=$bp || unset B_PROBES
    sed -e "s|^5000  |$NEV    |" -e "s|^\./exclurad_out|$d/run|" \
        "$HERE/cards/rad_pi0_rga.txt" > "$d/card.txt"
    python3 -m exclurad_py.run "$d/card.txt" > "$d/log" 2>&1 )
  [ -f "$SHARED" ] || cp "$d/eta.npz" "$SHARED"
  python3 - "$d" "$name" "$bs" "$bp" <<'PY'
import json,sys,glob,re
d,name,bs,bp = sys.argv[1:5]
try:
    r = json.load(open(glob.glob(f"{d}/run/**/config.json", recursive=True)[0]))
    r = r.get("report", r); log = open(f"{d}/log").read()
    ms = re.search(r"([\d.]+) ms/event", log)
    B  = re.search(r"global ([\d.]+)", log)
    a1, a2 = r.get("accepted",0), r.get("accept2",0)
    print(f"{name:14s} bs={bs or 'dflt':9s} bp={bp or '300':6s} | "
          f"{float(ms.group(1)) if ms else -1:7.1f} ms/ev | stage2 {100*a2/max(a1,1):5.1f}% | "
          f"RC/ev {a1/max(a2,1):5.2f} | b_viol {r.get('b_viol',-1):3d} "
          f"max {r.get('max_b_viol',0):.2f} | Bglobal {B.group(1) if B else '?'}")
except Exception as e:
    print(f"{name:14s} FAILED: {e}")
PY
}
echo "NEV=$NEV  (eta table shared and fixed; only the bound changes)"
run_one p300_dflt   ""        "300"
run_one p1000       ""        "1000"
run_one p3000       ""        "3000"
run_one p10000      ""        "10000"
run_one b666_p3000  "6,6,6,8" "3000"
run_one b888_p10000 "8,8,8,10" "10000"
