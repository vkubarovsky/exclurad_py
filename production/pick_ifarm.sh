#!/bin/bash
# Which login node to use right now.  Echoes a hostname; use it.
#
#   H=$(bash production/pick_ifarm.sh) && ssh $H '...'
#
# ifarm2401 and ifarm2402 are both AMD EPYC 9554 and were verified on
# 2026-10-03 to produce BYTE-IDENTICAL output for the same card and seed,
# so either is safe for a byte comparison and the only question is which is
# less busy.  That matters: measured the same minute, 2401 was at 5 % and
# 2402 at 66 %, and the same 300-event run took 4m29 against 6m10.
#
# Note `ifarm` without a number resolves to 2401, so a mix of `ifarm` and
# `ifarm2402` in scripts lands on both -- harmless here only BECAUSE they
# agree byte for byte, which was checked rather than assumed.
#
# TIMING IS A DIFFERENT QUESTION.  Never measure speed on a login node:
# it reads about 30 % fast against compute nodes and its load swings by an
# order of magnitude.  See docs/benchmarking.md.
best=""; bestload=""
for h in ifarm2401 ifarm2402; do
  l=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$h" 'read a _ < /proc/loadavg; echo $a' 2>/dev/null)
  [ -z "$l" ] && continue
  if [ -z "$bestload" ] || awk "BEGIN{exit !($l < $bestload)}"; then best=$h; bestload=$l; fi
done
[ -z "$best" ] && { echo "ifarm2402"; exit 0; }   # both unreachable: say something
echo "$best"
