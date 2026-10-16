#!/bin/bash
# Prepare, smoke-test and submit the five radiative pi0 generator campaigns
# that feed the RGA/RGK OSG detector productions.
set -euo pipefail

ACTION=${1:-}
CAMPAIGN=${2:-}
EXRU=${EXRU:-/work/clas12/storyf/exclurad_py_amp2614_50a5074_rc_clean}
EXRU_PY=${EXRU_PY:-/work/clas12/vpk/venv-prod-numba}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-/volatile/clas12/storyf/exclugen_pi0_radiative_amp2614_50a5074_20261016}
MAX_ACTIVE=${MAX_ACTIVE:-40}
TRANCHE=${TRANCHE:-1}
GEN_TIME=${GEN_TIME:-04:00:00}
EVENTS_PER_JOB=${EVENTS_PER_JOB:-10000}
JOBS_PER_TRANCHE=${JOBS_PER_TRANCHE:-10000}

CAMPAIGNS=(rga_fa18_10604 rgk_fa18_6535 rgk_fa18_7546 rgk_sp24_6395 rgk_sp24_8477)

usage() {
  cat <<'EOF'
Usage: generator_campaign_matrix.sh <prepare|smoke|submit|status> <campaign|all>

Campaigns:
  rga_fa18_10604   one generator sample shared by RGA inbending/outbending
  rgk_fa18_6535
  rgk_fa18_7546
  rgk_sp24_6395
  rgk_sp24_8477

Defaults:
  10,000 events/job x 10,000 jobs = 100M events per campaign and tranche
  at most 40 active Slurm tasks per submitted campaign

Examples:
  bash production/farm/generator_campaign_matrix.sh prepare all
  bash production/farm/generator_campaign_matrix.sh smoke all
  bash production/farm/generator_campaign_matrix.sh status all
  MAX_ACTIVE=40 bash production/farm/generator_campaign_matrix.sh submit all
  TRANCHE=2 MAX_ACTIVE=40 bash production/farm/generator_campaign_matrix.sh submit all
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

prepare_one() {
  local key=$1
  load_campaign "$key"
  require_preflight
  require_clean_checkout
  mkdir -p "$PROD_DIR/cards" "$PROD_DIR/logs"
  install_immutable "$CARD_SRC" "$CARD_DST"

  if [ -s "$PROD_DIR/tables/etatable.npz" ] && [ -s "$PROD_DIR/tables/majorant_rad.npz" ]; then
    echo "$key: reusing existing radiative tables in $PROD_DIR/tables"
  else
    PROD_DIR="$PROD_DIR" EXRU="$EXRU" EXRU_PY="$EXRU_PY" \
      CARD_RAD="$CARD_DST" BUILD_BORN=0 \
      bash "$EXRU/production/farm/build_tables.sh"
  fi

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

prepare_smoke_root() {
  local smoke=$PROD_DIR/smoke
  mkdir -p "$smoke/cards" "$smoke/tables" "$smoke/logs"
  install_immutable "$CARD_DST" "$smoke/cards/generator_rad.txt"
  install_immutable "$PROD_DIR/tables/etatable.npz" "$smoke/tables/etatable.npz"
  install_immutable "$PROD_DIR/tables/majorant_rad.npz" "$smoke/tables/majorant_rad.npz"
  echo "$smoke"
}

smoke_one() {
  local key=$1 smoke job
  load_campaign "$key"
  require_preflight
  require_clean_checkout
  [ -s "$PROD_DIR/tables/majorant_rad.npz" ] || {
    echo "FATAL: prepare $key before submitting its smoke test" >&2; exit 6; }
  smoke=$(prepare_smoke_root)
  job=$(sbatch --parsable --array=1 --job-name="smk_${key}" --time=00:20:00 \
    --output="$smoke/logs/slurm-%A_%a.out" --error="$smoke/logs/slurm-%A_%a.err" \
    --export="ALL,PROD_DIR=$smoke,EXRU=$EXRU,EXRU_PY=$EXRU_PY,CARD_FILE=$smoke/cards/generator_rad.txt,GEN_TAG=$GEN_TAG,SEEDBASE=$SEEDBASE,NEV=50,TASK_OFFSET=0,ID_WIDTH=5" \
    "$EXRU/production/farm/gen_array.sbatch" rad)
  printf 'smoke job=%s submitted_utc=%s\n' "$job" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$PROD_DIR/SLURM_JOB_IDS.txt"
  echo "$key: smoke job=$job"
}

submit_one() {
  local key=$1 offset last expected smoke_lund record job
  load_campaign "$key"
  require_preflight
  smoke_lund=$PROD_DIR/smoke/rad/chunks/lund_00001.dat
  [ -s "$smoke_lund" ] || {
    echo "FATAL: successful smoke output is absent: $smoke_lund" >&2; exit 7; }
  require_clean_checkout
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
    --export="ALL,PROD_DIR=$PROD_DIR,EXRU=$EXRU,EXRU_PY=$EXRU_PY,CARD_FILE=$CARD_DST,GEN_TAG=$GEN_TAG,SEEDBASE=$SEEDBASE,NEV=$EVENTS_PER_JOB,TASK_OFFSET=$offset,ID_WIDTH=5" \
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
  local key=$1 nlund nprov nlog
  load_campaign "$key"
  nlund=0; nprov=0; nlog=0
  [ ! -d "$PROD_DIR/rad/lund_osg" ] || nlund=$(find "$PROD_DIR/rad/lund_osg" -maxdepth 1 -type f -name '*.lund' | wc -l)
  [ ! -d "$PROD_DIR/rad/prov" ] || nprov=$(find "$PROD_DIR/rad/prov" -maxdepth 1 -type f -name '*.config.json' | wc -l)
  [ ! -d "$PROD_DIR/logs" ] || nlog=$(find "$PROD_DIR/logs" -maxdepth 1 -type f -name 'slurm-*.out' | wc -l)
  printf '%-18s  lund=%5d  provenance=%5d  slurm_logs=%5d  root=%s\n' \
    "$key" "$nlund" "$nprov" "$nlog" "$PROD_DIR"
}

[ -n "$ACTION" ] && [ -n "$CAMPAIGN" ] || { usage; exit 2; }
case "$ACTION" in prepare|smoke|submit|status) ;; *) usage; exit 2 ;; esac

if [ "$CAMPAIGN" = all ]; then
  for key in "${CAMPAIGNS[@]}"; do
    "${ACTION}_one" "$key"
  done
else
  "${ACTION}_one" "$CAMPAIGN"
fi
