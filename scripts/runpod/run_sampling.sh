#!/usr/bin/env bash
#
# run_sampling.sh: sample every dev query N times from the untrained, normal-reward and count-once-reward policies,
# compute the behavior metrics, upload the database to a private Hugging Face dataset repo (verified), and log a
# W&B report. Runs unattended on a one-GPU pod in the volume's data center (pod_bootstrap.sh with
# DRBM25_ENTRY=scripts/runpod/run_sampling.sh), resuming from /workspace/drbm25/samples where it left off.
#
# Stages: settings, java and uv, uv sync (vllm + analysis extras), pilot (SAMPLE sources at PILOT_N samples on the
# first PILOT_LIMIT queries, into samples/pilot), sample (all queries at SAMPLE_N into samples/db), metrics, hub,
# report. A failed report does not fail the run: the database is already on the Hub.
#
# Environment: HF_TOKEN and WANDB_API_KEY (RunPod secrets), WANDB_PROJECT (default drbm25), SAMPLE_N (32),
# PILOT_N (8), PILOT_LIMIT (64), SAMPLE_HUB_REPO (objones25/drbm25-rewrites), PRICE_PER_HOUR (optional: records
# spend in spend.json), DRBM25_CLEAR_FAILED=1 to archive an earlier samples/FAILED, NO_TERMINATE=1 or
# TERMINATE_ON_FINISH=no to keep the pod. Every exit records spend, copies logs and terminates the pod.
#
# Exit status: 0 success, 1 failure (samples/FAILED written; the training FAILED marker is never touched).
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly REPO_ROOT=$SCRIPT_DIR/../..
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

readonly SAMPLE_SOURCES=(original untrained trained-normal trained-count_once)
readonly SAMPLE_TIMEOUT_S=36000  # per source: a hang, not slowness, ends a sampler (the watchdog bounds the pod)
readonly METRICS_TIMEOUT_S=14400
readonly HUB_TIMEOUT_S=7200
readonly REPORT_TIMEOUT_S=7200

stage() {
  RUNPOD_STAGE=$1
  _runpod_log "=== stage: $1"
}

sampling_paths() {
  runpod_paths
  SAMPLES_DIR=$DRBM25_ROOT/samples
  # A failed sampling run must never block a training launch: its marker lives under samples/.
  DRBM25_FAILED_FILE=$SAMPLES_DIR/FAILED
}

sampling_sync_deps() {
  local py
  py=$(<.python-version)
  uv python install "$py"
  uv sync --frozen --extra vllm --extra analysis
}

require_sampling_settings() {
  local name
  for name in HF_TOKEN WANDB_API_KEY; do
    if [[ -z ${!name-} || ${!name} == *RUNPOD_SECRET* ]]; then
      mark_failed "$name is not set (it comes from a RunPod secret)"
      return 1
    fi
  done
  export WANDB_PROJECT=${WANDB_PROJECT:-drbm25}
  SAMPLE_N=${SAMPLE_N:-32}
  PILOT_N=${PILOT_N:-8}
  PILOT_LIMIT=${PILOT_LIMIT:-64}
  SAMPLE_HUB_REPO=${SAMPLE_HUB_REPO:-objones25/drbm25-rewrites}
}

# sample_source SOURCE N OUT_DIR [extra args...]
sample_source() {
  local source=$1 n=$2 out=$3
  shift 3
  _uv_timeout "$SAMPLE_TIMEOUT_S" python -m drbm25.eval.sample --config "$RUNPOD_CONFIG" --source "$source" \
    --n "$n" --out-dir "$out" "${RUNPOD_SET_ARGS[@]}" "$@"
}

main() {
  local source logfile start_unix
  sampling_paths
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  check_failed_marker || exit 1
  cd "$REPO_ROOT"
  mkdir -p "$DRBM25_LOG_DIR" "$SAMPLES_DIR"
  logfile=$DRBM25_LOG_DIR/sampling-$(date '+%Y%m%dT%H%M%S').log
  install_exit_traps

  stage settings
  require_sampling_settings
  printf 'run_sampling.sh: logging to %s\n' "$logfile" >&2
  exec >>"$logfile" 2>&1
  if [[ -n ${PRICE_PER_HOUR-} ]]; then
    start_unix=$(launch_start_unix)
    spend_begin "$start_unix" "$PRICE_PER_HOUR"
  fi

  stage "java and uv"
  install_java21
  install_uv
  stage "uv sync"
  sampling_sync_deps
  runpod_set_args

  stage pilot
  for source in "${SAMPLE_SOURCES[@]}"; do
    sample_source "$source" "$PILOT_N" "$SAMPLES_DIR/pilot" --limit "$PILOT_LIMIT"
  done
  stage sample
  for source in "${SAMPLE_SOURCES[@]}"; do
    sample_source "$source" "$SAMPLE_N" "$SAMPLES_DIR/db"
  done
  stage metrics
  _uv_timeout "$METRICS_TIMEOUT_S" python -m drbm25.analysis.pipeline --db "$SAMPLES_DIR/db" \
    --index-dir "$DRBM25_INDEX_DIR" --out "$SAMPLES_DIR/metrics"
  stage hub
  _uv_timeout "$HUB_TIMEOUT_S" python -m drbm25.analysis.hub --db "$SAMPLES_DIR/db" --metrics "$SAMPLES_DIR/metrics" \
    --repo "$SAMPLE_HUB_REPO"
  stage report
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! _uv_timeout "$REPORT_TIMEOUT_S" python -m drbm25.analysis.report --metrics "$SAMPLES_DIR/metrics" \
    --db "$SAMPLES_DIR/db" --config "$RUNPOD_CONFIG" "${RUNPOD_SET_ARGS[@]}"; then
    _runpod_log "WARNING: the W&B report failed; the database and metrics are on the Hub ($SAMPLE_HUB_REPO) and in $SAMPLES_DIR"
  fi
  stage finished
  _runpod_log "finished: database $SAMPLES_DIR/db (Hub: $SAMPLE_HUB_REPO), metrics $SAMPLES_DIR/metrics"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
