#!/bin/bash
# puller.sh -- run on the mini in a tmux server that has permission to write
# /Volumes/wd_14tb.  Every INTERVAL pulls each machine's local output into
#   <dest>/incoming_<host>/   (rsync -a, additive; short-lived processes only)
# Config lines: host:remote-path  (host "local" = the mini itself)
DEST=$1; CONF=$2; T=${3:-1800}
while true; do
  while IFS=: read -r host rpath; do
    [ -z "$host" ] && continue
    d="$DEST/incoming_$host"; mkdir -p "$d"
    if [ "$host" = local ]; then rsync -a --exclude work --exclude '*.tmp' "$rpath/" "$d/" 2>>"$DEST/puller.log"
    else rsync -a --exclude work --exclude '*.tmp' -e "ssh -o ConnectTimeout=60 -o BatchMode=yes" "$host:$rpath/" "$d/" 2>>"$DEST/puller.log"; fi
    echo "$(date '+%d.%m %H:%M') $host: rc=$? files=$(find "$d" -name '*.lund' | wc -l | tr -d ' ')" >> "$DEST/puller.log"
  done < "$CONF"
  sleep $T
done
