#!/usr/bin/env bash
#
# run_all.sh: the one-shot RunPod launcher for the two-run experiment (normal vs count-once BM25
# reward). Run it alone on a fresh 2x A100 80GB pod whose network volume is mounted at /workspace.
#
# Usage: see usage() below, or run with --help.
# Requires: bash 3.2+, curl, tar, GNU timeout, CUDA GPUs (Java, uv and Python 3.12 are installed from pinned
# downloads). Must be run from a clone of the repository.
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly REPO_ROOT=$SCRIPT_DIR/../..
# Seconds kept free before the deadline for the final evaluation (four scoring runs over the dev
# queries), the report with its 10,000-resample bootstrap, the log copy and the terminate call.
# One hour is an unmeasured, deliberately generous estimate (about $2.4-4.4 of a $160 budget).
readonly DEFAULT_RESERVE_S=3600

usage() {
  cat <<EOF2
Usage: $SCRIPT_NAME [--no-terminate] [-h|--help]

Runs the whole pipeline on a RunPod pod, resuming from /workspace/drbm25 where it left off.
Run it inside tmux (or under nohup) so a closed SSH or web terminal cannot interrupt it; output
goes to /workspace/drbm25/logs/run_all-<time>.log after the prompts.
Prompts: HF_TOKEN, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY, RUNPOD_API_KEY, TERMINATE_ON_FINISH
(hidden for secrets; skipped when set in the environment or in /workspace/drbm25/.env), and on every
launch BUDGET_USD (the dollars still left to spend, shown with what earlier launches spent) and
PRICE_PER_HOUR, which are never saved. Then Java 21 and uv, uv sync, dataset and index, model snapshot,
GPU smoke test, alternating training chunks of both runs within the budget, evaluation, report.

When it ends, for any reason, it copies logs to /workspace/drbm25/logs and terminates the pod
through the RunPod API, unless --no-terminate is given or TERMINATE_ON_FINISH=no.
RESERVE_S (default $DEFAULT_RESERVE_S) is the number of seconds before the budget deadline at
which training stops so evaluation and the report still fit.

Unattended (DRBM25_NONINTERACTIVE=1, or stdin is not a terminal; set by pod_bootstrap.sh when launch.sh created
the pod): nothing is ever asked; a missing setting is a failure (FAILED marker, pod terminated), and a FAILED
marker from an earlier launch terminates the pod too. DRBM25_CLEAR_FAILED=1 moves that marker into logs/ and
runs. DRBM25_POD_START_UNIX (the container's start) is used as the start of this launch's spend.

Exit status: 0 success, 1 failure (a FAILED marker is left in /workspace/drbm25; an interactive launch that finds one stops without terminating the pod), 2 usage error.
EOF2
}

stage() {
  RUNPOD_STAGE=$1
  _runpod_log "=== stage: $1"
}

main() {
  local no_terminate=0 start_unix total_steps chunk_steps deadline logfile now
  while (($# > 0)); do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      --no-terminate) no_terminate=1 ;;
      *)
        printf '%s: unknown option: %s\n' "$SCRIPT_NAME" "$1" >&2
        exit 2
        ;;
    esac
    shift
  done

  # shellcheck source=lib.sh
  source "$SCRIPT_DIR/lib.sh"
  start_unix=$(launch_start_unix)
  if ((no_terminate)); then export NO_TERMINATE=1; fi
  runpod_paths
  # A leftover FAILED marker stops the launch before any prompt. Interactively the pod is NOT terminated;
  # unattended (pod_bootstrap.sh) it is, since nobody is there to fix the cause.
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  check_failed_marker || exit 1
  cd "$REPO_ROOT"
  mkdir -p "$DRBM25_LOG_DIR"
  logfile=$DRBM25_LOG_DIR/run_all-$(date '+%Y%m%dT%H%M%S').log

  # From here on, every exit (success, error, Ctrl-C, closed terminal, SIGTERM) records spend, copies
  # logs and terminates the pod.
  install_exit_traps

  stage settings
  collect_settings
  # Straight to a file, not through tee: a closed terminal cannot break the log, and on_exit writes here.
  printf '%s: logging to %s (follow it with: tail -f %s)\n' "$SCRIPT_NAME" "$logfile" "$logfile" >&2
  exec >>"$logfile" 2>&1
  spend_begin "$start_unix" "$PRICE_PER_HOUR"
  _runpod_log "spent by earlier launches: \$$SPEND_BASE_USD; this launch's remaining budget: \$$BUDGET_USD at \$$PRICE_PER_HOUR/h"

  # Java, uv and the venv live on the pod's own disk (Java's tarball on the volume), so these run on
  # every launch; they are quick when already in place.
  stage "java and uv"
  install_java21
  install_uv
  stage "uv sync"
  sync_deps

  runpod_set_args
  stage "data and index"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! step_done data; then
    prepare_data_stage
    mark_done data
  fi
  stage "model snapshot"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! step_done model; then
    download_model
    mark_done model
  fi
  # Per pod: a new pod may have a different GPU or a fresh venv.
  stage "gpu smoke"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! step_done "smoke-${RUNPOD_POD_ID:-local}"; then
    smoke_stage
    mark_done "smoke-${RUNPOD_POD_ID:-local}"
  fi

  stage training
  total_steps=$(config_run_int "$RUNPOD_CONFIG" total_steps)
  chunk_steps=$(config_run_int "$RUNPOD_CONFIG" chunk_steps)
  RESERVE_S=${RESERVE_S:-$DEFAULT_RESERVE_S}
  deadline=$(compute_deadline_unix "$start_unix" "$BUDGET_USD" "$PRICE_PER_HOUR")
  DEADLINE_UNIX=$deadline
  export DEADLINE_UNIX RESERVE_S
  now=$(date +%s)
  _runpod_log "budget \$$BUDGET_USD at \$$PRICE_PER_HOUR/h: deadline in $(((deadline - now) / 60)) min, reserve $((RESERVE_S / 60)) min for eval and report"
  train_and_report "$total_steps" "$chunk_steps"
  stage finished
  _runpod_log "finished; report: $DRBM25_RUNS_DIR/eval/report.md"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
