#!/usr/bin/env bash
#
# lib.sh: functions for scripts/runpod/run_all.sh, the one-shot RunPod launcher. Source it; do not
# execute it. Sourcing also sources scripts/expanse/common.sh (exit_code, latest_step, JDK pin,
# sha256 and config helpers are shared with the Expanse scripts) and so turns on set -Eeuo pipefail.
#
# Requires: bash 3.2+, curl, tar, python3 with pip, and on the pod: uv, Java 21, CUDA GPUs.
# audit-bash disable-file=B101 # errexit is enabled by the sourced common.sh
# audit-bash disable-file=B403 # library of functions sourced by run_all.sh and the tests
# bash-target: 3.2
#
# Interface (every function returns a status; the ones that print data say so):
#   runpod_paths               export DRBM25_ROOT and the paths under it (default /workspace/drbm25)
#   prompt_var NAME PROMPT [--secret] [--optional] [--default V]   (never asks when unattended)
#   noninteractive             true when unattended: DRBM25_NONINTERACTIVE=1, or stdin is not a terminal
#                              and no DRBM25_TTY is given
#   write_env_file PATH NAME...  write NAME=value lines (mode 600) for the set variables
#   load_env_file PATH         export NAME=value lines for allowed names not already set
#   collect_settings           prompts for what is missing, validates, rewrites the .env
#   step_done MARKER / mark_done MARKER / mark_failed REASON
#   terminate_pod              DELETE the pod through the RunPod API, 3 attempts (user key, then pod key)
#   install_java21, install_uv, sync_deps, prepare_data_stage, download_model
#   smoke_stage                GPU tests, then the memory check at micro batch 8, 4, 2 (first that
#                              passes is saved to $DRBM25_ROOT/micro_batch and used by every chunk)
#   compute_deadline_unix START_UNIX BUDGET_USD PRICE_PER_HOUR   print the deadline (unix seconds)
#   run_alternating_chunks UNTIL CHUNK   status 0 done, 3 budget stop, 1 failure
#   eval_stage, train_and_report UNTIL CHUNK
#   spend_begin / spend_record / spend_total   dollars spent, kept in spend.json on the volume
#   refuse_if_failed, check_failed_marker, install_exit_traps
#   launch_start_unix          print when billing for this launch began (DRBM25_POD_START_UNIX or now)
#   on_exit STATUS             record spend, copy logs, then terminate the pod unless told not to
#
# Unattended runs (set by scripts/runpod/pod_bootstrap.sh): DRBM25_NONINTERACTIVE=1, DRBM25_CLEAR_FAILED=1,
# DRBM25_POD_START_UNIX.
# Test hooks (environment): DRBM25_ROOT, DRBM25_TTY (file read instead of /dev/tty),
# DRBM25_SLEEP (command used for the terminate backoff), DRBM25_CONFIG.

# shellcheck source=../expanse/common.sh
# shellcheck disable=SC2312 # status is checked explicitly here, not left to set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../expanse/common.sh"

# uv: the astral-sh/uv GitHub release tarball for x86_64 Linux, pinned by URL and SHA-256, installed into
# $DRBM25_ROOT/bin (no pip, so PEP 668 "externally-managed" Pythons do not matter, and nothing is piped
# to a shell). 0.13.7 was uploaded to PyPI on 2026-11-03, more than two weeks before 2026-11-20. The
# digest is the content of the release's published uv-x86_64-unknown-linux-gnu.tar.gz.sha256 and equals
# the SHA-256 of the tarball downloaded on 2026-11-20 when the pin was written.
readonly UV_VERSION=0.13.7
readonly UV_URL='https://github.com/astral-sh/uv/releases/download/0.13.7/uv-x86_64-unknown-linux-gnu.tar.gz'
readonly UV_SHA256='c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83'
readonly RUNPOD_CONSOLE_URL=https://console.runpod.io/pods
readonly RUNPOD_CONFIG_DEFAULT=configs/runpod_a100_full.yaml
# BUDGET_USD and PRICE_PER_HOUR are deliberately not stored: a relaunch must ask for the budget that
# is left (see spend_total), or it could silently reuse the whole budget again.
# Only these names are read from the .env, so a damaged file cannot set PATH or LD_PRELOAD.
readonly RUNPOD_SETTING_NAMES=(HF_TOKEN WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY RUNPOD_API_KEY TERMINATE_ON_FINISH VOLUME_GB)
readonly TERMINATE_ATTEMPTS=3
# Sleeps between the 3 attempts (after attempt 1 and after attempt 2).
readonly TERMINATE_BACKOFF_S=(5 15)
readonly KILL_GRACE_S=10          # on_exit: TERM, then KILL this much later, for a timed command
readonly KILL_AFTER_S=60           # SIGKILL this long after the TERM that `timeout` sends
readonly TRAIN_TIMEOUT_MARGIN_S=1800
readonly TRAIN_TIMEOUT_MIN_S=900
readonly EVAL_TIMEOUT_S=7200       # per evaluation command and for the report
readonly SMOKE_TIMEOUT_S=3600      # per smoke command
readonly DISK_MIN_KB=94371840      # 90 GiB before a run's first save: one checkpoint (~74 GB) + best/ (~6.8 GB)
readonly DISK_BEST_KB=7340032      # 7 GiB: room for a run's first best/ (3B actor in bf16, ~6.8 GB)
# PPO micro batch sizes tried by the smoke stage, largest first. 8 is the A100 80GB ceiling (per-rank
# mini-batch 16 / 2 GPUs); the memory check falls back to 4, then 2.
readonly MICRO_BATCH_CANDIDATES=(8 4 2)

_runpod_log() {
  local now
  now=$(date '+%Y-%m-%dT%H:%M:%S')
  printf '%s runpod: %s\n' "$now" "$*" >&2
}

runpod_paths() {
  DRBM25_ROOT=${DRBM25_ROOT:-/workspace/drbm25}
  export DRBM25_ROOT
  export HF_HOME="$DRBM25_ROOT/hf"
  export DRBM25_INDEX_DIR="$DRBM25_ROOT/indexes"
  export DRBM25_DATA_DIR="$DRBM25_ROOT/data"
  export DRBM25_RUNS_DIR="$DRBM25_ROOT/runs"
  export DRBM25_JDK_DIR="$DRBM25_ROOT/jdk"
  export WANDB_DIR="$DRBM25_ROOT/wandb"
  export UV_CACHE_DIR="$DRBM25_ROOT/uv-cache"
  # uv provides its own Python (on the volume), so the image's system Python and torch are irrelevant.
  export UV_PYTHON_INSTALL_DIR="$DRBM25_ROOT/python"
  export UV_PYTHON_PREFERENCE=only-managed
  # The cache is on the network volume and the venv is not: hardlinks cannot cross filesystems.
  export UV_LINK_MODE=copy
  DRBM25_DONE_DIR="$DRBM25_ROOT/.done"
  DRBM25_LOG_DIR="$DRBM25_ROOT/logs"
  DRBM25_ENV_FILE="$DRBM25_ROOT/.env"
  DRBM25_SPEND_FILE="$DRBM25_ROOT/spend.json"
  DRBM25_FAILED_FILE="$DRBM25_ROOT/FAILED"
  DRBM25_MICRO_BATCH_FILE="$DRBM25_ROOT/micro_batch"
  RUNPOD_CONFIG=${DRBM25_CONFIG:-$RUNPOD_CONFIG_DEFAULT}
}

# --set overrides: the config holds /workspace/drbm25 paths; these keep tests and DRBM25_ROOT consistent.
# The micro batch size chosen by the smoke stage (if any) goes to every training chunk, so it is the
# same across relaunches and lands in run_manifest.json.
runpod_set_args() {
  local micro
  drbm25_set_args
  RUNPOD_SET_ARGS=("${DRBM25_SET_ARGS[@]}")
  if [[ -n ${WANDB_PROJECT-} ]]; then RUNPOD_SET_ARGS+=(--set "run.wandb_project=$WANDB_PROJECT"); fi
  if [[ -n ${WANDB_ENTITY-} ]]; then RUNPOD_SET_ARGS+=(--set "run.wandb_entity=$WANDB_ENTITY"); fi
  if [[ -f $DRBM25_MICRO_BATCH_FILE ]]; then
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    micro=$(_saved_micro_batch) || return 1
    RUNPOD_SET_ARGS+=(--set "ppo.micro_batch_size=$micro")
  fi
}

# Prints the saved micro batch size; fails on anything but one of MICRO_BATCH_CANDIDATES.
_saved_micro_batch() {
  local micro candidate
  micro=$(<"$DRBM25_MICRO_BATCH_FILE")
  for candidate in "${MICRO_BATCH_CANDIDATES[@]}"; do
    if [[ $micro == "$candidate" ]]; then
      printf '%s\n' "$micro"
      return 0
    fi
  done
  _common_log "error: $DRBM25_MICRO_BATCH_FILE holds '$micro', expected one of ${MICRO_BATCH_CANDIDATES[*]}; before training has started, delete it to re-run the memory check from 8"
  return 1
}

_uv() {
  # --no-sync: the environment was synced once by sync_deps; `uv run` must not change it.
  uv run --no-sync "$@"
}

# ---------------------------------------------------------------- secrets

# Runs "$@" with xtrace off, so a `bash -x` run never prints a secret, and restores it afterwards.
_quiet() {
  local xtrace_was_on=0 rc=0
  if [[ $- == *x* ]]; then xtrace_was_on=1; fi
  set +x
  "$@" || rc=$?
  if ((xtrace_was_on)); then set -x; fi
  return "$rc"
}

# Unattended: nobody can answer a prompt. DRBM25_NONINTERACTIVE=1 forces it, =0 rules it out; unset, it
# follows stdin (a pod's start command has none), unless a test supplies answers through DRBM25_TTY.
noninteractive() {
  case ${DRBM25_NONINTERACTIVE-} in
    1) return 0 ;;
    0) return 1 ;;
  esac
  [[ ! -t 0 && -z ${DRBM25_TTY-} ]]
}

# Errors are returned explicitly: the body runs inside `||` (via _quiet), where set -e is off.
_prompt_var_body() {
  local name=$1 prompt=$2
  shift 2
  local secret=0 optional=0 default='' tty=${DRBM25_TTY:-/dev/tty} answer=''
  while (($# > 0)); do
    case $1 in
      --secret) secret=1 ;;
      --optional) optional=1 ;;
      --default)
        (($# >= 2)) || return 2
        default=$2
        shift
        ;;
      *)
        _common_log "error: prompt_var: unknown option $1"
        return 2
        ;;
    esac
    shift
  done
  if [[ -n ${!name-} ]] || { ((optional)) && [[ -n ${!name+set} ]]; }; then
    return 0
  fi
  # shellcheck disable=SC2310 # a predicate: its status is the answer
  if noninteractive; then
    _common_log "error: $name is not set and this run is unattended (DRBM25_NONINTERACTIVE or no terminal), so nobody can be asked; set it in the environment"
    return 1
  fi
  if [[ ! -r $tty ]]; then
    _common_log "error: $name is not set and there is no terminal to ask; set it in the environment or in ${DRBM25_ENV_FILE:-the .env file}"
    return 1
  fi
  printf '%s (%s)%s: ' "$prompt" "$name" "${default:+ [$default]}" >&2
  if ((secret)); then
    IFS= read -rs answer <"$tty" || [[ -n $answer ]] || answer=''
    printf '\n' >&2
  else
    IFS= read -r answer <"$tty" || [[ -n $answer ]] || answer=''
  fi
  if [[ -z $answer ]]; then answer=$default; fi
  if [[ -z $answer ]] && ((!optional)); then
    _common_log "error: $name is required"
    return 1
  fi
  printf -v "$name" '%s' "$answer"
  export "${name?}"
}

prompt_var() {
  _quiet _prompt_var_body "$@"
}

_write_env_file_body() {
  local path=$1 name value content='' dir tmp
  shift
  for name in "$@"; do
    if [[ ! $name =~ ^[A-Z_]+$ ]]; then
      _common_log "error: write_env_file: bad variable name '$name'"
      return 1
    fi
    # Unset names are skipped; set-but-empty (an optional answer) is kept so a re-run does not ask again.
    if [[ -z ${!name+set} ]]; then continue; fi
    value=${!name}
    if [[ $value == *$'\n'* ]]; then
      _common_log "error: $name holds a newline and cannot be stored in the .env"
      return 1
    fi
    content+="$name=$value"$'\n'
  done
  dir=$(dirname -- "$path")
  mkdir -p -- "$dir" || return 1
  # umask 077 makes the file mode 600 from the moment it exists; the rename then replaces the old one whole.
  tmp=$(umask 077 && mktemp "$dir/.env.XXXXXX") || return 1
  if ! (umask 077 && printf '%s' "$content" >"$tmp"); then
    rm -f -- "$tmp"
    return 1
  fi
  chmod 600 "$tmp" && mv -f -- "$tmp" "$path"
}

write_env_file() {
  _quiet _write_env_file_body "$@"
}

_load_env_file_body() {
  local path=$1 line name value allowed
  [[ -f $path ]] || return 0
  while IFS= read -r line || [[ -n $line ]]; do
    # Parsed, never evaluated: a value such as $(cmd) stays text.
    [[ $line =~ ^([A-Z_]+)=(.*)$ ]] || continue
    name=${BASH_REMATCH[1]}
    value=${BASH_REMATCH[2]}
    for allowed in "${RUNPOD_SETTING_NAMES[@]}"; do
      if [[ $name == "$allowed" && -z ${!name-} ]]; then
        printf -v "$name" '%s' "$value"
        export "${name?}"
      fi
    done
  done <"$path"
}

load_env_file() {
  _quiet _load_env_file_body "$@"
}

_require_positive_number() {
  local name=$1 value=${!1-}
  if [[ ! $value =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ $value =~ ^0+(\.0+)?$ ]]; then
    _common_log "error: $name must be a positive number (for example 90 or 2.78), got '$value'"
    return 1
  fi
}

# A pod created by launch.sh gets its keys as `{{ RUNPOD_SECRET_<name> }}` references that RunPod replaces at
# boot. A reference that reaches the script unreplaced (a deleted or misnamed secret) must stop the run here,
# not fail hours later as a bad token.
_require_substituted_secrets() {
  local name
  for name in HF_TOKEN WANDB_API_KEY RUNPOD_API_KEY; do
    if [[ ${!name-} == *RUNPOD_SECRET_* ]]; then
      # The value is only a reference, never the secret, so it is safe to print.
      _common_log "error: $name holds the unreplaced RunPod secret reference '${!name}'; create that secret (scripts/runpod/launch.sh does) and relaunch"
      return 1
    fi
  done
}

collect_settings() {
  load_env_file "$DRBM25_ENV_FILE"
  prompt_var HF_TOKEN "Hugging Face token (hidden)" --secret
  prompt_var WANDB_API_KEY "Weights & Biases API key (hidden)" --secret
  prompt_var WANDB_PROJECT "W&B project" --default drbm25
  prompt_var WANDB_ENTITY "W&B entity (optional, Enter to skip)" --optional
  prompt_var RUNPOD_API_KEY "RunPod API key with pods:write, used only to terminate this pod (hidden)" --secret
  local spent
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  spent=$(spend_total) || return 1
  prompt_var BUDGET_USD "REMAINING dollar budget for this launch (already spent by earlier launches: \$$spent; for example 90)"
  prompt_var PRICE_PER_HOUR "This pod's price per hour in dollars (for example 2.78)"
  prompt_var TERMINATE_ON_FINISH "Terminate the pod when finished (yes/no)" --default yes
  prompt_var VOLUME_GB "Size of the network volume in GB (optional; lets the disk check use it, since df may show pooled space)" --optional
  _require_substituted_secrets
  _require_positive_number BUDGET_USD
  _require_positive_number PRICE_PER_HOUR
  if [[ -n $VOLUME_GB && ! $VOLUME_GB =~ ^[1-9][0-9]*$ ]]; then
    _common_log "error: VOLUME_GB must be a whole number of GB (for example 400), got '$VOLUME_GB'"
    return 1
  fi
  case $TERMINATE_ON_FINISH in
    yes | no) ;;
    *)
      _common_log "error: TERMINATE_ON_FINISH must be yes or no, got '$TERMINATE_ON_FINISH'"
      return 1
      ;;
  esac
  write_env_file "$DRBM25_ENV_FILE" "${RUNPOD_SETTING_NAMES[@]}"
}

# ---------------------------------------------------------------- markers

_marker_path() {
  if [[ ! $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    _common_log "error: bad marker name '$1'"
    return 1
  fi
  printf '%s/%s\n' "$DRBM25_DONE_DIR" "$1"
}

step_done() {
  local path
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  path=$(_marker_path "$1") || return 1
  [[ -f $path ]]
}

mark_done() {
  local path
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  path=$(_marker_path "$1") || return 1
  mkdir -p -- "$DRBM25_DONE_DIR"
  date '+%Y-%m-%dT%H:%M:%S' >"$path"
}

# A marker for a failure that must not be retried blindly: the next launch stops on it at once
# instead of spending budget on the same error.
mark_failed() {
  # Best effort: a full or read-only volume must not stop the caller from cleaning up.
  # shellcheck disable=SC2312 # status is checked explicitly here, not left to set -e
  if ! { mkdir -p -- "$DRBM25_ROOT" && printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$1" >"$DRBM25_FAILED_FILE"; }; then
    _runpod_log "WARNING: could not write $DRBM25_FAILED_FILE"
  fi
  _runpod_log "error: $1 (FAILED marker: $DRBM25_FAILED_FILE; delete it after fixing the cause)"
}

# A launch that finds the marker must not start (and must not terminate: the user is here to fix it).
refuse_if_failed() {
  if [[ -f $DRBM25_FAILED_FILE ]]; then
    _common_log "error: an earlier launch failed: $(<"$DRBM25_FAILED_FILE")"
    _common_log "error: fix the cause, then delete $DRBM25_FAILED_FILE and run again"
    _common_log "error: nothing was started and this pod was NOT terminated: it is still billing; terminate it at $RUNPOD_CONSOLE_URL if you stop here"
    return 1
  fi
}

# What a launch does with a FAILED marker. Returns 0 to go on, 1 to stop.
# DRBM25_CLEAR_FAILED=1 (launch.sh --clear-failed, after the cause is fixed) moves the marker into logs/.
# Unattended, nobody is there to fix anything, so the pod is terminated (unless TERMINATE_ON_FINISH=no)
# instead of sitting idle and billing; interactively the launch stops and leaves the pod to the user.
check_failed_marker() {
  local archived
  [[ -f $DRBM25_FAILED_FILE ]] || return 0
  if [[ ${DRBM25_CLEAR_FAILED-} == 1 ]]; then
    archived=$DRBM25_LOG_DIR/FAILED-$(date '+%Y%m%dT%H%M%S')
    mkdir -p -- "$DRBM25_LOG_DIR" && mv -f -- "$DRBM25_FAILED_FILE" "$archived" || return 1
    _runpod_log "DRBM25_CLEAR_FAILED=1: moved the FAILED marker ($(<"$archived")) to $archived"
    return 0
  fi
  # shellcheck disable=SC2310 # a predicate: its status is the answer
  if ! noninteractive; then
    refuse_if_failed
    return 1
  fi
  _common_log "error: an earlier launch failed: $(<"$DRBM25_FAILED_FILE")"
  _common_log "error: fix the cause, then relaunch with: bash scripts/runpod/launch.sh --clear-failed"
  if [[ ${TERMINATE_ON_FINISH-} == no ]]; then
    _common_log "error: TERMINATE_ON_FINISH=no, so this pod is left running (and billing)"
  else
    # shellcheck disable=SC2310 # best effort: the launch stops either way
    terminate_pod || true
  fi
  return 1
}

# ---------------------------------------------------------------- spend and time

# Test hook: DRBM25_NOW_UNIX pins the clock.
_now_unix() {
  if [[ -n ${DRBM25_NOW_UNIX-} ]]; then printf '%s\n' "$DRBM25_NOW_UNIX"; else date +%s; fi
}

# When billing for this launch began: DRBM25_POD_START_UNIX (pod_bootstrap.sh records the container start,
# so boot and clone time count), or now. A value that is not a plain number or lies in the future is ignored.
launch_start_unix() {
  local now start=${DRBM25_POD_START_UNIX-}
  now=$(_now_unix)
  if [[ $start =~ ^[0-9]+$ ]] && ((start <= now)); then
    printf '%s\n' "$start"
  else
    printf '%s\n' "$now"
  fi
}

# Dollars spent by earlier launches (spend.json on the volume); 0 when there is none.
spend_total() {
  local value
  if [[ ! -f $DRBM25_SPEND_FILE ]]; then
    printf '0\n'
    return 0
  fi
  value=$(sed -n 's/.*"spent_usd": *\([0-9][0-9]*\(\.[0-9]*\)\{0,1\}\).*/\1/p' "$DRBM25_SPEND_FILE")
  if [[ ! $value =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    _common_log "error: $DRBM25_SPEND_FILE is malformed (no spent_usd number); fix or delete it"
    return 1
  fi
  printf '%s\n' "$value"
}

# Start counting this launch's spend: START_UNIX is when billing for this launch began.
spend_begin() {
  SPEND_START_UNIX=$1
  SPEND_PRICE=$2
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  SPEND_BASE_USD=$(spend_total) || return 1
}

# Writes base + elapsed hours x price. Called at the end of every chunk and from on_exit.
spend_record() {
  local now spent tmp
  if [[ -z ${SPEND_PRICE-} ]]; then return 0; fi
  now=$(_now_unix)
  spent=$(awk -v b="$SPEND_BASE_USD" -v e="$((now - SPEND_START_UNIX))" -v p="$SPEND_PRICE" 'BEGIN { printf "%.4f", b + e * p / 3600 }')
  if ! { mkdir -p -- "$DRBM25_ROOT" && tmp=$(mktemp "$DRBM25_ROOT/.spend.XXXXXX") &&
    printf '{"spent_usd": %s, "updated_unix": %s}\n' "$spent" "$now" >"$tmp" &&
    mv -f -- "$tmp" "$DRBM25_SPEND_FILE"; }; then
    _runpod_log "WARNING: could not record spend in $DRBM25_SPEND_FILE"
    return 1
  fi
}

_timeout_bin() {
  if [[ -n ${DRBM25_TIMEOUT-} ]]; then
    printf '%s\n' "$DRBM25_TIMEOUT"
  elif command -v timeout >/dev/null 2>&1; then
    printf 'timeout\n'
  elif command -v gtimeout >/dev/null 2>&1; then
    printf 'gtimeout\n'
  else
    return 1
  fi
}

# Runs "$@" for at most SECS seconds (GNU timeout) in the background and waits for it, because a
# trapped signal (Ctrl-C, closed terminal, TERM) only interrupts `wait`, not a foreground command.
# TIMEOUT_PID is the timeout process (a process group leader); on_exit kills its group.
# TIMEOUT_FIRED is 1 when the time limit, not something else, ended the command.
_with_timeout() {
  local secs=$1 bin rc=0 started elapsed
  shift
  TIMEOUT_FIRED=0
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  bin=$(_timeout_bin) || {
    _common_log "error: GNU timeout not found (on macOS install coreutils for gtimeout)"
    return 1
  }
  started=$(_now_unix)
  "$bin" --kill-after="$KILL_AFTER_S" "$secs" "$@" &
  TIMEOUT_PID=$!
  wait "$TIMEOUT_PID" || rc=$?
  TIMEOUT_PID=''
  elapsed=$(($(_now_unix) - started))
  # 124: the limit was hit. 137: SIGKILL, either sent by timeout after TERM was ignored (then the
  # limit has passed) or by something else such as the out-of-memory killer (then it has not).
  if ((rc == 124 || (rc == 137 && elapsed >= secs))); then
    TIMEOUT_FIRED=1
    _runpod_log "error: command timed out after ${secs}s (hang?): $1 $2 ..."
  elif ((rc == 137)); then
    _runpod_log "error: command was killed with SIGKILL after ${elapsed}s, before the ${secs}s limit (out-of-memory kill?): $1 $2 ..."
  fi
  return "$rc"
}

_uv_timeout() {
  local secs=$1
  shift
  _with_timeout "$secs" uv run --no-sync "$@"
}

# Fails (FAILED marker) when the volume cannot hold what the next chunk writes beside what exists: a new
# checkpoint next to the last one, and a new best/ next to the current one, so 1.2 x (last checkpoint +
# best/, or DISK_BEST_KB before the first best/); 90 GiB before the first checkpoint. `df -Pk` and
# `du -sk` are POSIX.
_check_disk() {
  local mode=$1 run_dir=$2 have=$3 need_kb=$DISK_MIN_KB free_kb ckpt used_kb best_kb=$DISK_BEST_KB
  if ((have > 0)); then
    ckpt=$run_dir/$(printf 'step_%06d' "$have")
    if [[ -d $ckpt ]]; then
      used_kb=$(du -sk "$ckpt" | awk '{print $1}')
      if [[ -d $run_dir/best ]]; then best_kb=$(du -sk "$run_dir/best" | awk '{print $1}'); fi
      need_kb=$(((used_kb + best_kb) * 12 / 10))
    fi
  fi
  free_kb=$(df -Pk "$DRBM25_ROOT" | awk 'NR == 2 {print $4}')
  # df on a network volume may show pooled space, so a user-given volume size also bounds free space.
  if [[ -n ${VOLUME_GB-} ]]; then
    used_kb=$(du -sk "$DRBM25_ROOT" | awk '{print $1}')
    if [[ $free_kb =~ ^[0-9]+$ ]] && ((VOLUME_GB * 1048576 - used_kb < free_kb)); then
      free_kb=$((VOLUME_GB * 1048576 - used_kb))
      if ((free_kb < need_kb)); then
        mark_failed "not enough disk before training $mode: VOLUME_GB=$VOLUME_GB minus ${used_kb} KB used leaves ${free_kb} KB, need ${need_kb} KB; use a bigger volume"
        return 1
      fi
    fi
  fi
  if [[ ! $free_kb =~ ^[0-9]+$ ]]; then
    mark_failed "cannot read free disk space of $DRBM25_ROOT (df printed '$free_kb')"
    return 1
  fi
  if ((free_kb < need_kb)); then
    mark_failed "not enough disk before training $mode: ${free_kb} KB free on the volume, need ${need_kb} KB (checkpoints are tens of GB each); free space or use a bigger volume"
    return 1
  fi
}

# ---------------------------------------------------------------- terminate

# One DELETE with KEY (header on stdin, -H @- curl 7.55+, so the key is never in ps output).
# RunPod docs, "Terminate a pod": DELETE /v2/pods/{id}, 204 on success; the key needs pods:write.
_delete_pod_with() {
  curl --fail --silent --show-error --max-time 30 -X DELETE \
    -H @- "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" >/dev/null <<<"Authorization: Bearer $1"
}

# Keys, in order: RUNPOD_API_KEY (the user's pods:write key) unless it is missing or an unreplaced
# {{ RUNPOD_SECRET_* }} reference, then DRBM25_POD_RUNPOD_API_KEY, the pod-scoped key RunPod injects
# (https://docs.runpod.io/pods/templates/environment-variables), saved by pod_bootstrap.sh. Each attempt tries
# both, so a rejected user key falls back at once.
_terminate_pod_body() {
  local sleep_cmd=${DRBM25_SLEEP:-sleep} attempt user=${RUNPOD_API_KEY-} pod=${DRBM25_POD_RUNPOD_API_KEY-}
  if [[ -z ${RUNPOD_POD_ID-} ]]; then
    _runpod_log "WARNING: RUNPOD_POD_ID is not set (RunPod sets it inside pods), so this pod was NOT terminated. Terminate it yourself at $RUNPOD_CONSOLE_URL or billing continues."
    return 1
  fi
  if [[ ! $RUNPOD_POD_ID =~ ^[A-Za-z0-9_-]+$ ]]; then
    _runpod_log "WARNING: RUNPOD_POD_ID has unexpected characters; not calling the API. Terminate the pod at $RUNPOD_CONSOLE_URL"
    return 1
  fi
  if [[ $user == *RUNPOD_SECRET_* ]]; then
    _runpod_log "WARNING: RUNPOD_API_KEY is an unreplaced RunPod secret reference; using the pod-scoped key"
    user=''
  fi
  if [[ -z $user && -z $pod ]]; then
    _runpod_log "WARNING: RUNPOD_API_KEY is not set, so pod $RUNPOD_POD_ID was NOT terminated. Terminate it at $RUNPOD_CONSOLE_URL"
    return 1
  fi
  for ((attempt = 1; attempt <= TERMINATE_ATTEMPTS; attempt++)); do
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    if [[ -n $user ]] && _delete_pod_with "$user"; then
      _runpod_log "terminated pod $RUNPOD_POD_ID"
      return 0
    fi
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    if [[ -n $pod ]] && _delete_pod_with "$pod"; then
      _runpod_log "terminated pod $RUNPOD_POD_ID (with the pod-scoped key)"
      return 0
    fi
    _runpod_log "terminate attempt $attempt/$TERMINATE_ATTEMPTS failed for pod $RUNPOD_POD_ID"
    if ((attempt < TERMINATE_ATTEMPTS)); then
      "$sleep_cmd" "${TERMINATE_BACKOFF_S[attempt - 1]}"
    fi
  done
  _runpod_log "ERROR: could not terminate pod $RUNPOD_POD_ID after $TERMINATE_ATTEMPTS attempts. Terminate it now at $RUNPOD_CONSOLE_URL (billing continues until you do)."
  return 1
}

terminate_pod() {
  _quiet _terminate_pod_body
}

# ---------------------------------------------------------------- environment stages

_install_jdk_tarball() (
  local target=$DRBM25_JDK_DIR/$JDK_RELEASE staging tarball machine
  machine=$(uname -m)
  [[ $machine == x86_64 ]] || {
    _common_log "error: the pinned JDK is for x86_64, this machine is $machine"
    exit 1
  }
  mkdir -p "$DRBM25_JDK_DIR"
  staging=$(mktemp -d "$DRBM25_JDK_DIR/.install.XXXXXX")
  trap 'rm -rf -- "${staging:?}"' EXIT
  tarball=$staging/jdk.tar.gz
  _runpod_log "downloading $JDK_URL"
  # Download, verify the pinned digest, then unpack; the download is never piped to a shell or tar.
  curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    --max-time 900 --output "$tarball" "$JDK_URL"
  verify_sha256 "$tarball" "$JDK_SHA256"
  tar -xzf "$tarball" -C "$staging"
  [[ -x $staging/$JDK_RELEASE/bin/java ]] || {
    _common_log "error: tarball did not contain $JDK_RELEASE/bin/java"
    exit 1
  }
  rm -rf -- "${target:?}"
  mv -- "$staging/$JDK_RELEASE" "$target"
)

# Java lives on the network volume, so a new pod only re-links it.
install_java21() {
  local major=''
  jdk_env
  # shellcheck disable=SC2310 # failure here only means "no usable java yet"
  major=$(java_major 2>/dev/null) || true
  if [[ $major != 21 ]]; then
    _runpod_log "Java is '${major:-missing}', installing the pinned Temurin 21 into $DRBM25_JDK_DIR"
    if [[ ! -x $DRBM25_JDK_DIR/$JDK_RELEASE/bin/java ]]; then
      # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
      _install_jdk_tarball || return 1
    fi
    ln -sfn "$JDK_RELEASE" "$DRBM25_JDK_DIR/current"
    jdk_env
  fi
  require_java21
}

_install_uv_tarball() (
  local bindir=$1 staging tarball machine
  machine=$(uname -m)
  [[ $machine == x86_64 ]] || {
    _common_log "error: the pinned uv release is for x86_64, this machine is $machine"
    exit 1
  }
  mkdir -p "$bindir"
  staging=$(mktemp -d "$bindir/.install.XXXXXX")
  trap 'rm -rf -- "${staging:?}"' EXIT
  tarball=$staging/uv.tar.gz
  _runpod_log "downloading $UV_URL"
  curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    --max-time 300 --output "$tarball" "$UV_URL"
  verify_sha256 "$tarball" "$UV_SHA256"
  tar -xzf "$tarball" -C "$staging"
  [[ -x $staging/uv-x86_64-unknown-linux-gnu/uv ]] || {
    _common_log "error: tarball did not contain uv-x86_64-unknown-linux-gnu/uv"
    exit 1
  }
  mv -f -- "$staging/uv-x86_64-unknown-linux-gnu/uv" "$bindir/uv"
)

# uv lives in $DRBM25_ROOT/bin on the volume, so a new pod only puts it on PATH.
install_uv() {
  local bindir=$DRBM25_ROOT/bin
  export PATH="$bindir:$PATH"
  # shellcheck disable=SC2312 # status is checked explicitly here, not left to set -e
  if [[ -x $bindir/uv && $("$bindir/uv" --version) == "uv $UV_VERSION"* ]]; then
    return 0
  fi
  _install_uv_tarball "$bindir"
}

# Python 3.12 comes from uv (version from .python-version), installed once onto the volume.
sync_deps() {
  local py
  if [[ ! -f .python-version ]]; then
    _common_log "error: no .python-version in $PWD; run from the repository root"
    return 1
  fi
  py=$(<.python-version)
  uv python install "$py"
  uv sync --frozen --extra vllm
}

prepare_data_stage() {
  runpod_set_args
  mkdir -p "$DRBM25_INDEX_DIR" "$DRBM25_DATA_DIR" "$DRBM25_RUNS_DIR" "$HF_HOME" "$WANDB_DIR"
  _uv python scripts/prepare_data.py --config "$RUNPOD_CONFIG" "${RUNPOD_SET_ARGS[@]}"
}

# Same snapshot-then-prove-it's-cached step as prepare_login.sh; the revision comes from the config.
download_model() {
  _uv python - "$RUNPOD_CONFIG" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from drbm25.config import load_config

cfg = load_config(Path(sys.argv[1]))
path = snapshot_download(cfg.model.name, revision=cfg.model.revision)
offline = snapshot_download(cfg.model.name, revision=cfg.model.revision, local_files_only=True)
assert path == offline, (path, offline)
print(f"model cached: {cfg.model.name}@{cfg.model.revision} -> {path}")
PY
}

# Prints "yes" when either run has a complete training checkpoint, else "no".
_training_started() {
  local mode have
  for mode in normal count_once; do
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    have=$(latest_step "$DRBM25_RUNS_DIR/$mode") || return 1
    if ((have > 0)); then
      printf 'yes\n'
      return 0
    fi
  done
  printf 'no\n'
}

smoke_stage() {
  local micro saved='' rc tried='' started
  local candidates=("${MICRO_BATCH_CANDIDATES[@]}")
  runpod_set_args
  _runpod_log "GPU smoke: the gpu-marked tests (vLLM engine), then the 2-GPU memory check"
  # --no-cov: the repo's coverage gate would fail a two-file run.
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! _uv_timeout "$SMOKE_TIMEOUT_S" pytest -p no:randomly --no-cov -m gpu tests/rollout/test_vllm_engine.py tests/ppo/test_models_full.py; then
    _runpod_log "error: GPU pytest failed"
    return 1
  fi
  # A size saved by an earlier pod caps the search: a run never moves to a larger micro batch.
  if [[ -f $DRBM25_MICRO_BATCH_FILE ]]; then
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    saved=$(_saved_micro_batch) || return 1
  fi
  # Once either run has a checkpoint the micro batch is fixed, so both runs cover every step at the
  # same size; a new pod that cannot fit it stops instead of silently stepping down.
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  started=$(_training_started) || return 1
  if [[ $started == yes ]]; then
    if [[ -z $saved ]]; then
      mark_failed "training checkpoints exist in $DRBM25_RUNS_DIR but $DRBM25_MICRO_BATCH_FILE is missing, so the micro batch the runs trained at is unknown; restore that file"
      return 1
    fi
    candidates=("$saved")
  fi
  for micro in "${candidates[@]}"; do
    if [[ -n $saved ]] && ((micro > saved)); then continue; fi
    tried+="$micro "
    rm -rf -- "${DRBM25_ROOT:?}/memcheck"
    rc=0
    # The --set after RUNPOD_SET_ARGS wins over a saved size there (the last --set of a key wins).
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    _uv_timeout "$SMOKE_TIMEOUT_S" torchrun --nproc_per_node 2 scripts/memory_check.py --config "$RUNPOD_CONFIG" \
      "${RUNPOD_SET_ARGS[@]}" --set "ppo.micro_batch_size=$micro" --ckpt-dir "$DRBM25_ROOT/memcheck" || rc=$?
    if ((rc == 0)); then
      printf '%s\n' "$micro" >"$DRBM25_MICRO_BATCH_FILE.tmp"
      mv -f -- "$DRBM25_MICRO_BATCH_FILE.tmp" "$DRBM25_MICRO_BATCH_FILE"
      _runpod_log "memory check passed at micro batch $micro (saved to $DRBM25_MICRO_BATCH_FILE)"
      rm -rf -- "${DRBM25_ROOT:?}/memcheck"
      return 0
    fi
    if ((TIMEOUT_FIRED)); then
      _runpod_log "error: memory check timed out at micro batch $micro (a hang, not memory)"
      return 1
    fi
    # Exit 1: an error (an OOM included) or under 2 GiB headroom per GPU; 137: killed (out of memory).
    if [[ $started == yes ]]; then
      mark_failed "memory check failed at micro batch $micro (status $rc) on this pod, and training checkpoints already exist at that size; not stepping down (the two runs would differ in micro batch). Use a pod with more GPU memory"
      return 1
    fi
    _runpod_log "memory check failed at micro batch $micro (status $rc); trying a smaller one"
  done
  mark_failed "memory check failed at every micro batch tried (${tried% }); see the log above"
  return 1
}

# ---------------------------------------------------------------- budget

compute_deadline_unix() {
  local start=$1 budget=$2 price=$3
  [[ $start =~ ^[0-9]+$ ]] || {
    _common_log "error: start time must be unix seconds, got '$start'"
    return 1
  }
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  BUDGET_USD=$budget _require_positive_number BUDGET_USD || return 1
  # The message names "price" in lower case on purpose: it reads as a sentence for the person at the prompt.
  if [[ ! $price =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ $price =~ ^0+(\.0+)?$ ]]; then
    _common_log "error: price per hour must be a positive number, got '$price'"
    return 1
  fi
  _uv python -c '
import sys
from drbm25.budget import deadline_from_budget
start, budget, price = (float(a) for a in sys.argv[1:4])
print(int(deadline_from_budget(budget_usd=budget, price_per_hour=price, start_unix=start)))
' "$start" "$budget" "$price"
}

# ---------------------------------------------------------------- training

_is_positive_int() {
  [[ $1 =~ ^[0-9]+$ && ! $1 =~ ^0+$ ]]
}

# Wall-clock seconds of the last chunk that ran (0 before the first). The trainer adds it to its
# budget check, because its own step timing misses torchrun start, model loads, vLLM init,
# validation and the checkpoint save.
LAST_CHUNK_S=0

# One torchrun for MODE up to STEP. Status 0 ok, 3 budget stop, 1 failure (FAILED written).
_train_chunk() {
  local mode=$1 step=$2 torchrun_rc=0 code='' have after limit run_dir started
  run_dir=$DRBM25_RUNS_DIR/$mode
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  have=$(latest_step "$run_dir") || return 1
  if ((have >= step)); then
    _runpod_log "$mode already at step $have (>= $step); skipping this chunk"
    return 0
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _check_disk "$mode" "$run_dir" "$have" || return 1
  # The trainer stops itself at the deadline, so a chunk running past it plus a margin is hung.
  limit=$((DEADLINE_UNIX - $(_now_unix) + TRAIN_TIMEOUT_MARGIN_S))
  if ((limit < TRAIN_TIMEOUT_MIN_S)); then limit=$TRAIN_TIMEOUT_MIN_S; fi
  # A leftover exit_code (0 or 3 from the previous chunk) must never be read for this one: rank 0
  # only overwrites it after load_config, and torchrun can fail before that.
  rm -f -- "$run_dir/exit_code"
  _runpod_log "training $mode: step $have -> $step (timeout ${limit}s, previous chunk ${LAST_CHUNK_S}s)"
  started=$(_now_unix)
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _uv_timeout "$limit" torchrun --nproc_per_node 2 --standalone -m drbm25.train \
    --config "$RUNPOD_CONFIG" --reward-mode "$mode" --run-name "$mode" --until-step "$step" \
    --deadline-unix "$DEADLINE_UNIX" --reserve-s "$RESERVE_S" --prev-chunk-s "$LAST_CHUNK_S" \
    "${RUNPOD_SET_ARGS[@]}" || torchrun_rc=$?
  LAST_CHUNK_S=$(($(_now_unix) - started))
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  spend_record || true
  if ((TIMEOUT_FIRED)); then
    mark_failed "$mode training timed out after ${limit}s (hang?); see $run_dir/train_rank0.log"
    return 1
  fi
  if ((torchrun_rc == 137)); then
    mark_failed "$mode training was killed with SIGKILL before the ${limit}s limit (out of memory?); see $run_dir/train_rank0.log"
    return 1
  fi
  # torchrun collapses worker exit codes; rank 0 writes the real one to exit_code.
  # shellcheck disable=SC2310 # a missing or malformed file means "no usable exit_code"
  code=$(run_exit_code "$run_dir") || code=''
  case $code in
    0)
      # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
      after=$(latest_step "$run_dir") || return 1
      if ((after < step)); then
        mark_failed "$mode wrote exit_code 0 but its latest complete checkpoint is step $after, not step $step"
        return 1
      fi
      return 0
      ;;
    3)
      _runpod_log "$mode stopped on the budget deadline (exit_code 3)"
      return 3
      ;;
    1) mark_failed "$mode run did not finish (exit_code 1, torchrun exited $torchrun_rc): killed or out of memory; see $run_dir/train_rank0.log" ;;
    2) mark_failed "$mode raised an exception (exit_code 2); see $run_dir/train_rank0.log" ;;
    '') mark_failed "$mode has no usable exit_code file in $run_dir (torchrun exited $torchrun_rc)" ;;
    *) mark_failed "$mode wrote unexpected exit_code $code" ;;
  esac
  return 1
}

_spike_check() {
  local mode metrics
  for mode in normal count_once; do
    metrics=$DRBM25_RUNS_DIR/$mode/metrics.jsonl
    if [[ -f $metrics ]]; then
      # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
      _uv python scripts/spike_monitor.py "$metrics" >&2 ||
        _runpod_log "WARNING: spike_monitor failed on $metrics (not fatal)"
    fi
  done
}

# Alternates normal and count_once in chunks of CHUNK steps so both stop within one chunk of each other.
run_alternating_chunks() {
  local until_step=${1:?UNTIL} chunk=${2:?CHUNK} step=0 mode rc
  require_env DEADLINE_UNIX RESERVE_S
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! _is_positive_int "$until_step" || ! _is_positive_int "$chunk"; then
    _common_log "error: UNTIL and CHUNK must be positive integers, got '$until_step' and '$chunk'"
    return 1
  fi
  if [[ -f $DRBM25_FAILED_FILE ]]; then
    _common_log "error: an earlier launch failed: $(<"$DRBM25_FAILED_FILE")"
    _common_log "error: fix the cause, then delete $DRBM25_FAILED_FILE to retry"
    return 1
  fi
  runpod_set_args
  # At most ceil(UNTIL/CHUNK) iterations: step grows by CHUNK >= 1 until it reaches UNTIL.
  while ((step < until_step)); do
    step=$((step + chunk))
    if ((step > until_step)); then step=$until_step; fi
    for mode in normal count_once; do
      rc=0
      # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
      _train_chunk "$mode" "$step" || rc=$?
      case $rc in
        0) ;;
        3) return 3 ;;
        *) return 1 ;;
      esac
    done
    _spike_check
  done
}

# ---------------------------------------------------------------- evaluation

_run_eval() {
  local source=$1 out=$2 checkpoint=${3-}
  local args=(--config "$RUNPOD_CONFIG" --source "$source" --out "$out")
  if [[ -n $checkpoint ]]; then args+=(--checkpoint "$checkpoint"); fi
  _uv_timeout "$EVAL_TIMEOUT_S" python -m drbm25.eval.run_eval "${args[@]}" "${RUNPOD_SET_ARGS[@]}"
}

# Always re-run: evaluation is deterministic (greedy) and cheap next to the cost of a missed report.
# Each trained row is the run's best/ policy (selected on the validation split carved from train);
# the test set is all of dev.
eval_stage() {
  local normal_dir=$DRBM25_RUNS_DIR/normal count_once_dir=$DRBM25_RUNS_DIR/count_once
  local out_dir=$DRBM25_RUNS_DIR/eval run_dir report_rc
  runpod_set_args
  for run_dir in "$normal_dir" "$count_once_dir"; do
    if [[ ! -f $run_dir/best/meta.json ]]; then
      _common_log "error: no best/ policy in $run_dir (training never reached a validation step?)"
      return 1
    fi
  done
  mkdir -p "$out_dir"
  _runpod_log "evaluating the best/ policy of both runs"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _run_eval original "$out_dir/original.jsonl" || return 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _run_eval untrained "$out_dir/untrained.jsonl" || return 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _run_eval checkpoint "$out_dir/trained-normal.jsonl" "$normal_dir/best" || return 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _run_eval checkpoint "$out_dir/trained-count_once.jsonl" "$count_once_dir/best" || return 1
  # report.md is written before the W&B upload; a stale one must not pass for this launch's.
  rm -f -- "$out_dir/report.md"
  report_rc=0
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _uv_timeout "$EVAL_TIMEOUT_S" python -m drbm25.eval.report \
    --inputs "$out_dir/original.jsonl" "$out_dir/untrained.jsonl" \
    "$out_dir/trained-normal.jsonl" "$out_dir/trained-count_once.jsonl" \
    --out "$out_dir/report.md" --wandb --config "$RUNPOD_CONFIG" "${RUNPOD_SET_ARGS[@]}" || report_rc=$?
  if ((report_rc != 0)); then
    if [[ ! -f $out_dir/report.md ]]; then
      _runpod_log "error: the report failed (status $report_rc) before writing $out_dir/report.md"
      return 1
    fi
    # The results are on the volume; a W&B outage must not mark the whole experiment FAILED.
    _runpod_log "WARNING: the report was written but its W&B upload failed (status $report_rc); results are in $out_dir (sync them to W&B by hand)"
  fi
  _runpod_log "report: $out_dir/report.md"
}

# Train, then evaluate whatever both runs reached. A budget stop (3) is a normal end; a failure is not.
train_and_report() {
  local rc=0
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  run_alternating_chunks "$1" "$2" || rc=$?
  case $rc in
    0) ;;
    3) _runpod_log "budget stop: evaluating the best/ policy each run has so far" ;;
    *) return 1 ;;
  esac
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! eval_stage; then
    mark_failed "evaluation or report failed; checkpoints are intact, re-run run_all.sh"
    return 1
  fi
}

# ---------------------------------------------------------------- exit

_copy_logs() {
  local run_dir name file
  for run_dir in "$DRBM25_RUNS_DIR"/*/; do
    [[ -d $run_dir ]] || continue
    name=$(basename -- "$run_dir")
    mkdir -p -- "$DRBM25_LOG_DIR/$name"
    for file in "$run_dir"train_rank*.log "$run_dir"exit_code "$run_dir"metrics.jsonl "$run_dir"report.md; do
      [[ -f $file ]] || continue
      cp -f -- "$file" "$DRBM25_LOG_DIR/$name/" || _runpod_log "WARNING: could not copy $file"
    done
  done
}

# Stops the command `_with_timeout` is waiting on: TERM to its process group, a short grace, then KILL.
# GNU timeout leads its own group, so this reaches torchrun and its workers. Best effort.
_kill_timed_command() {
  local pid=${TIMEOUT_PID-} i
  if [[ -z $pid ]]; then return 0; fi
  _runpod_log "stopping the running command (process group $pid)"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for ((i = 0; i < KILL_GRACE_S; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    "${DRBM25_SLEEP:-sleep}" 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  TIMEOUT_PID=''
}

# Traps: any exit, including a closed terminal (HUP), INT and TERM, copies logs and terminates the pod.
# SIGPIPE is ignored so a dead pipe or terminal cannot kill the script halfway through on_exit.
install_exit_traps() {
  trap 'on_exit $?' EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap '' PIPE
}

# EXIT trap body. Everything in it is best effort and terminate_pod runs last, whatever failed before
# (a full volume, a missing directory, an unset variable). It always returns 0 so the script keeps the
# exit status it was already ending with.
on_exit() {
  local status=${1:-0}
  set +e +u
  # A second Ctrl-C, hangup or TERM must not cut on_exit short; children (curl) inherit the ignore.
  trap '' HUP INT TERM
  _kill_timed_command
  # audit-bash disable=B102 # this only removes an inherited ERR trap; set -E comes from common.sh
  trap - ERR
  if ((status != 0)) && [[ ! -f ${DRBM25_FAILED_FILE-} ]]; then
    mark_failed "run_all.sh exited with status $status during stage '${RUNPOD_STAGE:-unknown}'"
  fi
  spend_record
  _copy_logs
  if [[ ${NO_TERMINATE-} == 1 || ${TERMINATE_ON_FINISH-} == no ]]; then
    _runpod_log "not terminating the pod (--no-terminate or TERMINATE_ON_FINISH=no): billing continues"
    return 0
  fi
  terminate_pod
  return 0
}
