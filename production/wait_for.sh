#!/bin/bash
# Wait for work to FINISH.  Never wait for a word in a log.
#
#   bash production/wait_for.sh pid   <pid> [timeout_s] [poll_s]
#   bash production/wait_for.sh file  <path> [timeout_s] [poll_s]
#   bash production/wait_for.sh slurm <name-or-jobid> [timeout_s] [poll_s]
#   bash production/wait_for.sh ssh   <host> <remote-cmd> [timeout_s]
#
# WHY THIS FILE EXISTS.  On 2026-10-05 four processes were found on the mini
# that had been waiting six and seven days for something that had already
# happened or could never happen:
#
#     ssh ifarm2402 'until grep -q "RESULT\|ERROR" nb39b.log; do sleep 20; done'
#
# The job it watched died on 29 September with
#     AttributeError: module 'llvmlite.binding' has no attribute 'get_host_cpu_name'
# Python writes "AttributeError", not "ERROR", and never writes "RESULT" when it
# raises -- so the watcher slept through exactly the case it was written for.  A
# second one was a bare `tail -22` on a directory nothing was writing to any more.
#
# THE RULE.  A watcher must key on something the work writes UNCONDITIONALLY --
# its own exit -- and must have a timeout.  A word that only appears on success
# turns every failure into an infinite sleep, and silence is the most common
# failure mode there is.
#
# Three correct shapes, in order of preference:
#   1. the process itself:      wait $pid               (local, exact)
#   2. a sentinel it always writes:  trap 'echo "__DONE rc=$?" >> log' EXIT
#   3. the scheduler:           squeue -h -j $id | wc -l
# and in every case a deadline after which the watcher says WHY it gave up.
set -u
KIND=${1:?pid|file|slurm|ssh}

case "$KIND" in
pid)
  P=${2:?pid}; T=${3:-86400}; S=${4:-10}; t=0
  while kill -0 "$P" 2>/dev/null; do
    [ "$t" -ge "$T" ] && { echo "TIMEOUT after ${T}s: pid $P still alive"; exit 2; }
    sleep "$S"; t=$((t+S))
  done
  echo "DONE: pid $P exited after ${t}s"
  ;;

file)
  # A sentinel file, which the job must write on EVERY exit path:
  #     trap 'echo "__DONE rc=$?" >> "$F"' EXIT
  F=${2:?path}; T=${3:-86400}; S=${4:-10}; t=0
  while [ ! -s "$F" ]; do
    [ "$t" -ge "$T" ] && { echo "TIMEOUT after ${T}s: $F never appeared"; exit 2; }
    sleep "$S"; t=$((t+S))
  done
  echo "DONE after ${t}s: $(tail -1 "$F")"
  ;;

slurm)
  J=${2:?name or jobid}; T=${3:-86400}; S=${4:-60}; t=0
  q() { squeue -u "$USER" -h -r -n "$J" 2>/dev/null | wc -l; }
  [ "$(q)" -eq 0 ] && q() { squeue -u "$USER" -h -r -j "$J" 2>/dev/null | wc -l; }
  while [ "$(q)" -gt 0 ]; do
    [ "$t" -ge "$T" ] && { echo "TIMEOUT after ${T}s: $(q) elements of $J still queued"; exit 2; }
    sleep "$S"; t=$((t+S))
  done
  echo "DONE: no elements of $J left after ${t}s"
  ;;

ssh)
  # Run the remote command to completion and bring its exit code back.  The ssh
  # exit status IS the remote one, so nothing needs parsing and a failure cannot
  # be mistaken for silence.  -o BatchMode=yes so a prompt fails instead of hanging.
  H=${2:?host}; C=${3:?command}; T=${4:-86400}
  ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=4 "$H" "timeout $T bash -lc $(printf %q "$C")"
  rc=$?
  [ $rc -eq 124 ] && { echo "TIMEOUT after ${T}s on $H"; exit 2; }
  echo "DONE: remote rc=$rc"; exit $rc
  ;;

*) echo "usage: $0 pid|file|slurm|ssh ..."; exit 1 ;;
esac
