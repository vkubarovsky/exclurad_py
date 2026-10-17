#!/bin/bash
# Prepare, smoke-test and submit the five radiative pi0 generator campaigns
# that feed the RGA/RGK OSG detector productions.
set -euo pipefail

ACTION=${1:-}
CAMPAIGN=${2:-}
EXRU=${EXRU:-/work/clas12/storyf/exclurad_py_amp2614_50a5074_rc_clean}
EXRU_PY=${EXRU_PY:-/work/clas12/vpk/venv-prod-numba}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-}
MAX_ACTIVE=${MAX_ACTIVE:-40}
TRANCHE=${TRANCHE:-1}
GEN_TIME=${GEN_TIME:-04:00:00}
EVENTS_PER_JOB=${EVENTS_PER_JOB:-10000}
JOBS_PER_TRANCHE=${JOBS_PER_TRANCHE:-10000}
BOUND_WORKERS=${BOUND_WORKERS:-32}
BOUND_PROBES_PER_CELL=${BOUND_PROBES_PER_CELL:-1600}
BOUND_MARGIN=${BOUND_MARGIN:-2.0}
PILOT_JOBS=${PILOT_JOBS:-2}
PILOT_EVENTS_PER_JOB=${PILOT_EVENTS_PER_JOB:-10000}
PILOT_MAX_ACTIVE=${PILOT_MAX_ACTIVE:-2}
PILOT_TIME=${PILOT_TIME:-04:00:00}
B_VIOL_FRACTION_MAX=${B_VIOL_FRACTION_MAX:-0.001}

CAMPAIGNS=(rga_fa18_10604 rgk_fa18_6535 rgk_fa18_7546 rgk_sp24_6395 rgk_sp24_8477)

usage() {
  cat <<'EOF'
Usage: generator_campaign_matrix.sh <prepare|smoke|pilot|audit|submit|status> <campaign|all>

Campaigns:
  rga_fa18_10604   one generator sample shared by RGA inbending/outbending
  rgk_fa18_6535
  rgk_fa18_7546
  rgk_sp24_6395
  rgk_sp24_8477

Defaults:
  stage-2 bound: 1,600 probes/cell, multiplicative margin 2.0
  pilot: 2 jobs x 10,000 events; production submission requires a passed audit
  stage-2 b_viol tolerance: 0.1% of stage-1 accepts, matching generator health()
  10,000 events/job x 10,000 jobs = 100M events per campaign and tranche
  at most 40 active Slurm tasks per submitted campaign

Required:
  Set CAMPAIGN_ROOT to a new, empty /volatile campaign directory.  There is no
  implicit historical default, so an old table or submission cannot be reused
  accidentally.

Examples:
  bash production/farm/generator_campaign_matrix.sh prepare all
  bash production/farm/generator_campaign_matrix.sh smoke all
  bash production/farm/generator_campaign_matrix.sh pilot rga_fa18_10604
  bash production/farm/generator_campaign_matrix.sh audit rga_fa18_10604
  bash production/farm/generator_campaign_matrix.sh status all
  MAX_ACTIVE=40 bash production/farm/generator_campaign_matrix.sh submit rga_fa18_10604
  TRANCHE=2 MAX_ACTIVE=40 bash production/farm/generator_campaign_matrix.sh submit rga_fa18_10604
EOF
}

load_campaign() {
  local key=$1
  case "$key" in
    rga_fa18_10604)
      CARD_REL=cards/rad_pi0_rga_fa18_10604_amp2614.txt
      GEN_TAG=pi0_rga_fa18_10604_amp2614
      SEEDBASE=700000000
      ;;
    rgk_fa18_6535)
      CARD_REL=cards/rad_pi0_rgk_fa18_6535_amp2614.txt
      GEN_TAG=pi0_rgk_fa18_6535_amp2614
      SEEDBASE=950000000
      ;;
    rgk_fa18_7546)
      CARD_REL=cards/rad_pi0_rgk_fa18_7546_amp2614.txt
      GEN_TAG=pi0_rgk_fa18_7546_amp2614
      SEEDBASE=1200000000
      ;;
    rgk_sp24_6395)
      CARD_REL=cards/rad_pi0_rgk_sp24_6395_amp2614.txt
      GEN_TAG=pi0_rgk_sp24_6395_amp2614
      SEEDBASE=1450000000
      ;;
    rgk_sp24_8477)
      CARD_REL=cards/rad_pi0_rgk_sp24_8477_amp2614.txt
      GEN_TAG=pi0_rgk_sp24_8477_amp2614
      SEEDBASE=1700000000
      ;;
    *) echo "FATAL: unknown campaign '$key'" >&2; exit 2 ;;
  esac
  PROD_DIR=$CAMPAIGN_ROOT/$key
  CARD_SRC=$EXRU/$CARD_REL
  CARD_DST=$PROD_DIR/cards/generator_rad.txt
}

install_immutable() {
  local src=$1 dst=$2
  if [ -e "$dst" ]; then
    cmp -s "$src" "$dst" || {
      echo "FATAL: $dst exists and differs from $src; refusing to overwrite it" >&2
      exit 3
    }
  else
    cp -a "$src" "$dst"
  fi
}

require_preflight() {
  [ -d "$EXRU/.git" ] || git -C "$EXRU" rev-parse --git-dir >/dev/null 2>&1 || {
    echo "FATAL: EXRU is not a Git checkout: $EXRU" >&2; exit 4; }
  [ -x "$EXRU_PY/bin/python" ] || {
    echo "FATAL: production Python is absent: $EXRU_PY/bin/python" >&2; exit 4; }
  [ -f "$CARD_SRC" ] || { echo "FATAL: missing card $CARD_SRC" >&2; exit 4; }
}

require_clean_checkout() {
  [ "$(git -C "$EXRU" status --porcelain | wc -l)" -eq 0 ] || {
    echo "FATAL: generator checkout is dirty; commit the validated branch first" >&2
    exit 4
  }
}

validate_positive_integer() {
  local name=$1 value=$2
  case "$value" in
    ''|*[!0-9]*|0) echo "FATAL: $name must be a positive integer, got '$value'" >&2; exit 4 ;;
  esac
}

validate_bound_cache() {
  local path=$1
  PYTHONPATH="$EXRU${PYTHONPATH:+:$PYTHONPATH}" \
    "$EXRU_PY/bin/python" - "$path" "$BOUND_MARGIN" "$BOUND_PROBES_PER_CELL" <<'PY'
import sys
import numpy as np
from production.audit_generator_pilot import bound_arrays

path, expected_margin, expected_probes = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
try:
    normalization, metadata = bound_arrays(path)
except Exception as exc:
    raise SystemExit(f"FATAL: invalid bound cache {path}: {exc}") from exc
margin = metadata["margin"]
probes = metadata["probes_per_cell"]
if not np.isclose(margin, expected_margin, rtol=0.0, atol=1e-12):
    raise SystemExit(
        f"FATAL: {path} margin={margin:g}, requested {expected_margin:g}")
if probes != expected_probes:
    raise SystemExit(
        f"FATAL: {path} probes_per_cell={probes}, requested {expected_probes}")
print(
    f"validated bound cache: margin={margin:g}, probes_per_cell={probes}, "
    f"integral(M*B)={normalization:.12g} nb")
PY
}

prepare_one() {
  local key=$1
  load_campaign "$key"
  require_preflight
  require_clean_checkout
  validate_positive_integer BOUND_WORKERS "$BOUND_WORKERS"
  validate_positive_integer BOUND_PROBES_PER_CELL "$BOUND_PROBES_PER_CELL"
  mkdir -p "$PROD_DIR/cards" "$PROD_DIR/logs"
  install_immutable "$CARD_SRC" "$CARD_DST"

  if [ -s "$PROD_DIR/tables/etatable.npz" ] && [ -s "$PROD_DIR/tables/majorant_rad.npz" ]; then
    echo "$key: reusing existing radiative tables in $PROD_DIR/tables"
  else
    PROD_DIR="$PROD_DIR" EXRU="$EXRU" EXRU_PY="$EXRU_PY" \
      CARD_RAD="$CARD_DST" BUILD_BORN=0 NW="$BOUND_WORKERS" \
      PER="$BOUND_PROBES_PER_CELL" B_MARGIN="$BOUND_MARGIN" \
      bash "$EXRU/production/farm/build_tables.sh"
  fi
  validate_bound_cache "$PROD_DIR/tables/majorant_rad.npz"

  local revision dirty
  revision=$(git -C "$EXRU" rev-parse HEAD)
  dirty=$(git -C "$EXRU" status --porcelain | wc -l)
  {
    echo "campaign=$key"
    echo "generator=pi0.amp2614"
    echo "generator_revision=$revision"
    echo "generator_dirty_paths=$dirty"
    echo "events_per_job=$EVENTS_PER_JOB"
    echo "jobs_per_tranche=$JOBS_PER_TRANCHE"
    echo "bound_workers=$BOUND_WORKERS"
    echo "bound_probes_per_cell=$BOUND_PROBES_PER_CELL"
    echo "bound_margin=$BOUND_MARGIN"
    echo "pilot_jobs=$PILOT_JOBS"
    echo "pilot_events_per_job=$PILOT_EVENTS_PER_JOB"
    echo "max_b_viol_fraction=$B_VIOL_FRACTION_MAX"
    echo "seedbase=$SEEDBASE"
    echo "card=$CARD_DST"
    sha256sum "$CARD_DST" "$PROD_DIR/tables/etatable.npz" "$PROD_DIR/tables/majorant_rad.npz"
  } > "$PROD_DIR/GENERATOR_CAMPAIGN.txt.pending"
  if [ -e "$PROD_DIR/GENERATOR_CAMPAIGN.txt" ]; then
    cmp -s "$PROD_DIR/GENERATOR_CAMPAIGN.txt.pending" "$PROD_DIR/GENERATOR_CAMPAIGN.txt" || {
      echo "FATAL: campaign record changed; inspect the two files before replacing either" >&2
      exit 5
    }
    rm -f "$PROD_DIR/GENERATOR_CAMPAIGN.txt.pending"
  else
    mv "$PROD_DIR/GENERATOR_CAMPAIGN.txt.pending" "$PROD_DIR/GENERATOR_CAMPAIGN.txt"
  fi
  echo "$key: prepared at $PROD_DIR"
}

prepare_validation_root() {
  local name=$1 root=$PROD_DIR/$1
  mkdir -p "$root/cards" "$root/tables" "$root/logs"
  install_immutable "$CARD_DST" "$root/cards/generator_rad.txt"
  install_immutable "$PROD_DIR/tables/etatable.npz" "$root/tables/etatable.npz"
  install_immutable "$PROD_DIR/tables/majorant_rad.npz" "$root/tables/majorant_rad.npz"
  echo "$root"
}

smoke_one() {
  local key=$1 smoke job
  load_campaign "$key"
  require_preflight
  require_clean_checkout
  [ -s "$PROD_DIR/tables/majorant_rad.npz" ] || {
    echo "FATAL: prepare $key before submitting its smoke test" >&2; exit 6; }
  smoke=$(prepare_validation_root smoke)
  job=$(sbatch --parsable --array=1 --job-name="smk_${key}" --time=00:20:00 \
    --output="$smoke/logs/slurm-%A_%a.out" --error="$smoke/logs/slurm-%A_%a.err" \
    --export="ALL,PROD_DIR=$smoke,EXRU=$EXRU,EXRU_PY=$EXRU_PY,CARD_FILE=$smoke/cards/generator_rad.txt,GEN_TAG=$GEN_TAG,SEEDBASE=$SEEDBASE,NEV=50,TASK_OFFSET=0,ID_WIDTH=5,B_VIOL_FRACTION_MAX=$B_VIOL_FRACTION_MAX" \
    "$EXRU/production/farm/gen_array.sbatch" rad)
  printf 'smoke job=%s submitted_utc=%s\n' "$job" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$PROD_DIR/SLURM_JOB_IDS.txt"
  echo "$key: smoke job=$job"
}

pilot_one() {
  local key=$1 pilot job last record pilot_seedbase
  load_campaign "$key"
  require_preflight
  require_clean_checkout
  validate_positive_integer PILOT_JOBS "$PILOT_JOBS"
  validate_positive_integer PILOT_EVENTS_PER_JOB "$PILOT_EVENTS_PER_JOB"
  validate_positive_integer PILOT_MAX_ACTIVE "$PILOT_MAX_ACTIVE"
  [ -s "$PROD_DIR/smoke/rad/chunks/lund_00001.dat" ] || {
    echo "FATAL: successful smoke output is absent; run smoke $key first" >&2; exit 7; }
  record=$PROD_DIR/PILOT_SUBMITTED.txt
  [ ! -e "$record" ] || {
    echo "FATAL: pilot was already submitted; see $record" >&2; exit 7; }
  pilot=$(prepare_validation_root pilot)
  last=$(( PILOT_JOBS - 1 ))
  pilot_seedbase=$(( SEEDBASE + 50000000 ))
  job=$(sbatch --parsable --array="0-${last}%${PILOT_MAX_ACTIVE}" \
    --job-name="pilot_${key}" --time="$PILOT_TIME" \
    --output="$pilot/logs/slurm-%A_%a.out" \
    --error="$pilot/logs/slurm-%A_%a.err" \
    --export="ALL,PROD_DIR=$pilot,EXRU=$EXRU,EXRU_PY=$EXRU_PY,CARD_FILE=$pilot/cards/generator_rad.txt,GEN_TAG=$GEN_TAG,SEEDBASE=$pilot_seedbase,NEV=$PILOT_EVENTS_PER_JOB,TASK_OFFSET=1,ID_WIDTH=5,B_VIOL_FRACTION_MAX=$B_VIOL_FRACTION_MAX" \
    "$EXRU/production/farm/gen_array.sbatch" rad)
  {
    echo "job=$job"
    echo "jobs=$PILOT_JOBS"
    echo "events_per_job=$PILOT_EVENTS_PER_JOB"
    echo "max_active=$PILOT_MAX_ACTIVE"
    echo "seedbase=$pilot_seedbase"
    echo "generator_revision=$(git -C "$EXRU" rev-parse HEAD)"
    echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$record"
  printf 'pilot job=%s submitted_utc=%s\n' \
    "$job" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$PROD_DIR/SLURM_JOB_IDS.txt"
  echo "$key: pilot job=$job"
}

audit_one() {
  local key=$1 pilot output pass pending revision
  load_campaign "$key"
  require_preflight
  require_clean_checkout
  pilot=$PROD_DIR/pilot
  output=$pilot/PILOT_AUDIT.json
  pass=$PROD_DIR/PILOT_PASSED.txt
  pending=$pass.pending
  revision=$(git -C "$EXRU" rev-parse HEAD)
  "$EXRU_PY/bin/python" "$EXRU/production/audit_generator_pilot.py" \
    "$pilot" --expected-runs "$PILOT_JOBS" \
    --expected-events "$PILOT_EVENTS_PER_JOB" \
    --expected-revision "$revision" \
    --max-b-viol-fraction "$B_VIOL_FRACTION_MAX" --output "$output"
  {
    echo "generator_revision=$revision"
    echo "pilot_jobs=$PILOT_JOBS"
    echo "pilot_events_per_job=$PILOT_EVENTS_PER_JOB"
    echo "max_b_viol_fraction=$B_VIOL_FRACTION_MAX"
    sha256sum "$output" "$pilot"/rad/prov/rad_*.config.json \
      "$pilot/tables/majorant_rad.npz"
  } > "$pending"
  if [ -e "$pass" ]; then
    cmp -s "$pending" "$pass" || {
      echo "FATAL: a different pilot pass record already exists: $pass" >&2
      exit 8
    }
    rm -f "$pending"
  else
    mv "$pending" "$pass"
  fi
  echo "$key: pilot audit passed; production gate=$pass"
}

submit_one() {
  local key=$1 offset last expected smoke_lund record job pass revision
  load_campaign "$key"
  require_preflight
  smoke_lund=$PROD_DIR/smoke/rad/chunks/lund_00001.dat
  [ -s "$smoke_lund" ] || {
    echo "FATAL: successful smoke output is absent: $smoke_lund" >&2; exit 7; }
  require_clean_checkout
  revision=$(git -C "$EXRU" rev-parse HEAD)
  pass=$PROD_DIR/PILOT_PASSED.txt
  [ -s "$pass" ] || {
    echo "FATAL: pilot audit has not passed; missing $pass" >&2; exit 7; }
  grep -Fx "generator_revision=$revision" "$pass" >/dev/null || {
    echo "FATAL: pilot pass record does not certify current revision $revision" >&2
    exit 7
  }
  case "$TRANCHE:$MAX_ACTIVE:$EVENTS_PER_JOB:$JOBS_PER_TRANCHE" in
    *[!0-9:]*|0:*|*::*) echo "FATAL: tranche and job controls must be positive integers" >&2; exit 7 ;;
  esac
  [ "$MAX_ACTIVE" -gt 0 ] && [ "$EVENTS_PER_JOB" -gt 0 ] && [ "$JOBS_PER_TRANCHE" -gt 0 ] || {
    echo "FATAL: job controls must be greater than zero" >&2; exit 7; }
  record=$PROD_DIR/SUBMITTED_TRANCHE_${TRANCHE}.txt
  [ ! -e "$record" ] || {
    echo "FATAL: tranche $TRANCHE was already submitted; see $record" >&2
    exit 7
  }
  # Slurm's MaxArraySize caps the largest legal index.  A zero-based array
  # keeps 10,000 tasks within indices 0-9999; TASK_OFFSET maps those to stable
  # one-based chunk ids 00001-10000 (and 10001-20000 in tranche two).
  offset=$(( (TRANCHE - 1) * JOBS_PER_TRANCHE + 1 ))
  last=$(( JOBS_PER_TRANCHE - 1 ))
  expected=$(( EVENTS_PER_JOB * JOBS_PER_TRANCHE ))
  echo "$key: submitting tranche $TRANCHE, offset $offset, $expected events"
  job=$(sbatch --parsable --array="0-${last}%${MAX_ACTIVE}" \
    --job-name="gen_${key}_t${TRANCHE}" --time="$GEN_TIME" \
    --output="$PROD_DIR/logs/slurm-t${TRANCHE}-%A_%a.out" \
    --error="$PROD_DIR/logs/slurm-t${TRANCHE}-%A_%a.err" \
    --export="ALL,PROD_DIR=$PROD_DIR,EXRU=$EXRU,EXRU_PY=$EXRU_PY,CARD_FILE=$CARD_DST,GEN_TAG=$GEN_TAG,SEEDBASE=$SEEDBASE,NEV=$EVENTS_PER_JOB,TASK_OFFSET=$offset,ID_WIDTH=5,B_VIOL_FRACTION_MAX=$B_VIOL_FRACTION_MAX" \
    "$EXRU/production/farm/gen_array.sbatch" rad)
  {
    echo "job=$job"
    echo "tranche=$TRANCHE"
    echo "task_offset=$offset"
    echo "jobs=$JOBS_PER_TRANCHE"
    echo "events_per_job=$EVENTS_PER_JOB"
    echo "expected_events=$expected"
    echo "max_active=$MAX_ACTIVE"
    echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$record"
  printf 'production tranche=%s job=%s submitted_utc=%s\n' \
    "$TRANCHE" "$job" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$PROD_DIR/SLURM_JOB_IDS.txt"
  echo "$key: production tranche=$TRANCHE job=$job"
}

status_one() {
  local key=$1 nlund nprov nlog npilot gate
  load_campaign "$key"
  nlund=0; nprov=0; nlog=0; npilot=0; gate=no
  [ ! -d "$PROD_DIR/rad/lund_osg" ] || nlund=$(find "$PROD_DIR/rad/lund_osg" -maxdepth 1 -type f -name '*.lund' | wc -l)
  [ ! -d "$PROD_DIR/rad/prov" ] || nprov=$(find "$PROD_DIR/rad/prov" -maxdepth 1 -type f -name '*.config.json' | wc -l)
  [ ! -d "$PROD_DIR/logs" ] || nlog=$(find "$PROD_DIR/logs" -maxdepth 1 -type f -name 'slurm-*.out' | wc -l)
  [ ! -d "$PROD_DIR/pilot/rad/prov" ] || npilot=$(find "$PROD_DIR/pilot/rad/prov" -maxdepth 1 -type f -name '*.config.json' | wc -l)
  [ ! -s "$PROD_DIR/PILOT_PASSED.txt" ] || gate=yes
  printf '%-18s  lund=%5d  provenance=%5d  pilot=%2d  gate=%-3s  logs=%5d  root=%s\n' \
    "$key" "$nlund" "$nprov" "$npilot" "$gate" "$nlog" "$PROD_DIR"
}

[ -n "$ACTION" ] && [ -n "$CAMPAIGN" ] || { usage; exit 2; }
[ -n "$CAMPAIGN_ROOT" ] || {
  echo "FATAL: set CAMPAIGN_ROOT to a new campaign root" >&2
  exit 2
}
case "$ACTION" in prepare|smoke|pilot|audit|submit|status) ;; *) usage; exit 2 ;; esac

if [ "$ACTION" = submit ] && [ "$CAMPAIGN" = all ]; then
  echo "FATAL: submit one production campaign at a time; a 10,000-task array fills the Farm submission quota" >&2
  exit 2
fi

if [ "$CAMPAIGN" = all ]; then
  for key in "${CAMPAIGNS[@]}"; do
    "${ACTION}_one" "$key"
  done
else
  "${ACTION}_one" "$CAMPAIGN"
fi
