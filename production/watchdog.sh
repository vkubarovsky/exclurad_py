#!/bin/bash
# watchdog.sh <out-root> [interval-s]  -- run in tmux ON THE MACHINE; appends one
# line per tick to <out>/watchdog.log: workers, files in the last hour, disk, load,
# swap.  Marks ALERT when there are no workers, or no new file for 3 h.
OUT=$1; T=${2:-900}; ALERT_H=3
while true; do
  w=$(pgrep -f "exclurad_py.run import run" 2>/dev/null | wc -l | tr -d ' ')
  new=$(find "$OUT" -name '*.lund' -mmin -60 2>/dev/null | wc -l | tr -d ' ')
  old=$(find "$OUT" -name '*.lund' -mmin -$((ALERT_H*60)) 2>/dev/null | wc -l | tr -d ' ')
  tot=$(find "$OUT" -name '*.lund' 2>/dev/null | wc -l | tr -d ' ')
  disk=$(df -h "$OUT" | tail -1 | awk '{print $4}')
  load=$(uptime | sed 's/.*load[ a-z]*: *//' | cut -d, -f1 | tr -d ' ')
  swap=$( (sysctl -n vm.swapusage 2>/dev/null | grep -oE 'used = [0-9.]+M' | cut -d= -f2) || free -m 2>/dev/null | awk '/Swap/{print $3"M"}')
  a=""; [ "$w" -eq 0 ] && a=" ALERT:no-workers"; [ "$old" -eq 0 ] && [ -f "$OUT/run.log" ] && a="$a ALERT:no-file-${ALERT_H}h"; [ -f "$OUT/STOP" ] && a="$a STOP"
  echo "$(date '+%d.%m %H:%M') workers=$w files_1h=$new total=$tot disk=$disk load=$load swap=${swap:-?}$a" >> "$OUT/watchdog.log"
  sleep $T
done
