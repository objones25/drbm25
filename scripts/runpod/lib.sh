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
#   prompt_var NAME PROMPT [--secret] [--optional] [--default V]
#   write_env_file PATH NAME...  write NAME=value lines (mode 600) for the set variables
#   load_env_file PATH         export NAME=value lines for allowed names not already set
#   collect_settings           prompts for what is missing, validates, rewrites the .env
#   step_done MARKER / mark_done MARKER / mark_failed REASON
#   terminate_pod              DELETE the pod through the RunPod API, 3 attempts
#   install_java21, install_uv, sync_deps, prepare_data_stage, download_model, smoke_stage
#   compute_deadline_unix START_UNIX BUDGET_USD PRICE_PER_HOUR   print the deadline (unix seconds)
#   run_alternating_chunks UNTIL CHUNK   status 0 done, 3 budget stop, 1 failure
#   eval_stage, train_and_report UNTIL CHUNK
#   on_exit STATUS             copy logs, then terminate the pod unless told not to
#
# Test hooks (environment): DRBM25_ROOT, DRBM25_TTY (file read instead of /dev/tty),
# DRBM25_SLEEP (command used for the terminate backoff), DRBM25_CONFIG.

# shellcheck source=../expanse/common.sh
# shellcheck disable=SC2312 # status is checked explicitly here, not left to set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../expanse/common.sh"

# uv from PyPI: 0.13.7 was published 2026-11-03 (PyPI upload time), over two weeks before the
# 2026-11-20 pin date; 0.13.8 and 0.13.9 are newer than that cut-off.
readonly UV_VERSION=0.13.7
readonly RUNPOD_CONSOLE_URL=https://console.runpod.io/pods
readonly RUNPOD_CONFIG_DEFAULT=configs/runpod_a100_full.yaml
# Only these names are read from the .env, so a damaged file cannot set PATH or LD_PRELOAD.
readonly RUNPOD_SETTING_NAMES=(HF_TOKEN WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY RUNPOD_API_KEY BUDGET_USD PRICE_PER_HOUR TERMINATE_ON_FINISH)
readonly TERMINATE_ATTEMPTS=3
readonly TERMINATE_BACKOFF_S=(5 15 45)

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
  # The cache is on the network volume and the venv is not: hardlinks cannot cross filesystems.
  export UV_LINK_MODE=copy
  DRBM25_DONE_DIR="$DRBM25_ROOT/.done"
  DRBM25_LOG_DIR="$DRBM25_ROOT/logs"
  DRBM25_ENV_FILE="$DRBM25_ROOT/.env"
  DRBM25_FAILED_FILE="$DRBM25_ROOT/FAILED"
  RUNPOD_CONFIG=${DRBM25_CONFIG:-$RUNPOD_CONFIG_DEFAULT}
}

# --set overrides: the config holds /workspace/drbm25 paths; these keep tests and DRBM25_ROOT consistent.
runpod_set_args() {
  drbm25_set_args
  RUNPOD_SET_ARGS=("${DRBM25_SET_ARGS[@]}")
  if [[ -n ${WANDB_PROJECT-} ]]; then RUNPOD_SET_ARGS+=(--set "run.wandb_project=$WANDB_PROJECT"); fi
  if [[ -n ${WANDB_ENTITY-} ]]; then RUNPOD_SET_ARGS+=(--set "run.wandb_entity=$WANDB_ENTITY"); fi
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
  if [[ ! -r $tty ]]; then
    _common_log "error: $name is not set and there is no terminal to ask; set it in the environment or in $DRBM25_ENV_FILE"
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

collect_settings() {
  load_env_file "$DRBM25_ENV_FILE"
  prompt_var HF_TOKEN "Hugging Face token (hidden)" --secret
  prompt_var WANDB_API_KEY "Weights & Biases API key (hidden)" --secret
  prompt_var WANDB_PROJECT "W&B project" --default drbm25
  prompt_var WANDB_ENTITY "W&B entity (optional, Enter to skip)" --optional
  prompt_var RUNPOD_API_KEY "RunPod API key with pods:write, used only to terminate this pod (hidden)" --secret
  prompt_var BUDGET_USD "Dollar budget for this pod (for example 90)"
  prompt_var PRICE_PER_HOUR "This pod's price per hour in dollars (for example 2.78)"
  prompt_var TERMINATE_ON_FINISH "Terminate the pod when finished (yes/no)" --default yes
  _require_positive_number BUDGET_USD
  _require_positive_number PRICE_PER_HOUR
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
  mkdir -p -- "$DRBM25_ROOT"
  # shellcheck disable=SC2312 # status is checked explicitly here, not left to set -e
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$1" >"$DRBM25_FAILED_FILE"
  _runpod_log "error: $1 (wrote $DRBM25_FAILED_FILE; delete it after fixing the cause)"
}

# ---------------------------------------------------------------- terminate

_terminate_pod_body() {
  local sleep_cmd=${DRBM25_SLEEP:-sleep} attempt
  if [[ -z ${RUNPOD_POD_ID-} ]]; then
    _runpod_log "WARNING: RUNPOD_POD_ID is not set (RunPod sets it inside pods), so this pod was NOT terminated. Terminate it yourself at $RUNPOD_CONSOLE_URL or billing continues."
    return 1
  fi
  if [[ ! $RUNPOD_POD_ID =~ ^[A-Za-z0-9_-]+$ ]]; then
    _runpod_log "WARNING: RUNPOD_POD_ID has unexpected characters; not calling the API. Terminate the pod at $RUNPOD_CONSOLE_URL"
    return 1
  fi
  if [[ -z ${RUNPOD_API_KEY-} ]]; then
    _runpod_log "WARNING: RUNPOD_API_KEY is not set, so pod $RUNPOD_POD_ID was NOT terminated. Terminate it at $RUNPOD_CONSOLE_URL"
    return 1
  fi
  for ((attempt = 1; attempt <= TERMINATE_ATTEMPTS; attempt++)); do
    # The header is read from stdin (-H @-, curl 7.55+) so the key never appears in ps output.
    # RunPod docs, "Terminate a pod": DELETE /v2/pods/{id}, 204 on success; the key needs pods:write.
    if curl --fail --silent --show-error --max-time 30 -X DELETE \
      -H @- "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" >/dev/null <<<"Authorization: Bearer $RUNPOD_API_KEY"; then
      _runpod_log "terminated pod $RUNPOD_POD_ID"
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

# uv comes from PyPI at a pinned version (pip install uv==X), not from a downloaded installer script.
install_uv() {
  local user_bin
  # shellcheck disable=SC2312 # status is checked explicitly here, not left to set -e
  if command -v uv >/dev/null 2>&1 && [[ $(uv --version) == "uv $UV_VERSION"* ]]; then
    return 0
  fi
  _runpod_log "installing uv==$UV_VERSION with pip"
  if ! python3 -m pip install --quiet --disable-pip-version-check "uv==$UV_VERSION"; then
    _common_log "error: pip could not install uv==$UV_VERSION (on an externally-managed Python, install uv $UV_VERSION by hand)"
    return 1
  fi
  user_bin=$(python3 -m site --user-base)/bin
  export PATH="$user_bin:$PATH"
  command -v uv >/dev/null 2>&1 || {
    _common_log "error: uv is installed but not on PATH"
    return 1
  }
}

sync_deps() {
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

smoke_stage() {
  runpod_set_args
  _runpod_log "GPU smoke: vLLM and full-model tests, then the 2-GPU memory check"
  # --no-cov: the repo's coverage gate would fail a two-file run.
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! _uv pytest -p no:randomly --no-cov -m gpu tests/rollout/test_vllm_engine.py tests/ppo/test_models_full.py; then
    _runpod_log "error: GPU pytest failed"
    return 1
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! _uv torchrun --nproc_per_node 2 scripts/memory_check.py --config "$RUNPOD_CONFIG" \
    "${RUNPOD_SET_ARGS[@]}" --ckpt-dir "$DRBM25_ROOT/memcheck"; then
    _runpod_log "error: memory check failed (exit 1 means an error or under 2 GiB headroom per GPU)"
    return 1
  fi
  rm -rf -- "${DRBM25_ROOT:?}/memcheck"
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

# One torchrun for MODE up to STEP. Status 0 ok, 3 budget stop, 1 failure (FAILED written).
_train_chunk() {
  local mode=$1 step=$2 torchrun_rc=0 code='' have run_dir
  run_dir=$DRBM25_RUNS_DIR/$mode
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  have=$(latest_step "$run_dir") || return 1
  if ((have >= step)); then
    _runpod_log "$mode already at step $have (>= $step); skipping this chunk"
    return 0
  fi
  _runpod_log "training $mode: step $have -> $step"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _uv torchrun --nproc_per_node 2 --standalone -m drbm25.train \
    --config "$RUNPOD_CONFIG" --reward-mode "$mode" --run-name "$mode" --until-step "$step" \
    --deadline-unix "$DEADLINE_UNIX" --reserve-s "$RESERVE_S" "${RUNPOD_SET_ARGS[@]}" || torchrun_rc=$?
  # torchrun collapses worker exit codes; rank 0 writes the real one to exit_code.
  # shellcheck disable=SC2310 # a missing or malformed file means "no usable exit_code"
  code=$(run_exit_code "$run_dir") || code=''
  case $code in
    0) return 0 ;;
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
  _uv python -m drbm25.eval.run_eval "${args[@]}" "${RUNPOD_SET_ARGS[@]}"
}

# Always re-run: evaluation is deterministic (greedy) and cheap next to the cost of a missed report.
eval_stage() {
  local normal_dir=$DRBM25_RUNS_DIR/normal count_once_dir=$DRBM25_RUNS_DIR/count_once
  local out_dir=$DRBM25_RUNS_DIR/eval step step_name
  runpod_set_args
  mkdir -p "$out_dir"
  # Both rows of the report must come from the same step; common_step raises when none is shared.
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  step=$(_uv python -c '
import sys
from pathlib import Path
from drbm25.eval.report import common_step
print(common_step(Path(sys.argv[1]), Path(sys.argv[2])))
' "$normal_dir" "$count_once_dir") || return 1
  if [[ ! $step =~ ^[0-9]+$ ]]; then
    _common_log "error: common_step printed '$step', expected a step number"
    return 1
  fi
  step_name=$(printf 'step_%06d' "$step")
  _runpod_log "evaluating step $step of both runs"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _run_eval original "$out_dir/original.jsonl" || return 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _run_eval untrained "$out_dir/untrained.jsonl" || return 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _run_eval checkpoint "$out_dir/trained-normal.jsonl" "$normal_dir/$step_name" || return 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _run_eval checkpoint "$out_dir/trained-count_once.jsonl" "$count_once_dir/$step_name" || return 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _uv python -m drbm25.eval.report \
    --inputs "$out_dir/original.jsonl" "$out_dir/untrained.jsonl" \
    "$out_dir/trained-normal.jsonl" "$out_dir/trained-count_once.jsonl" \
    --out "$out_dir/report.md" --wandb --config "$RUNPOD_CONFIG" || return 1
  _runpod_log "report: $out_dir/report.md"
}

# Train, then evaluate whatever both runs reached. A budget stop (3) is a normal end; a failure is not.
train_and_report() {
  local rc=0
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  run_alternating_chunks "$1" "$2" || rc=$?
  case $rc in
    0) ;;
    3) _runpod_log "budget stop: evaluating the last step both runs reached" ;;
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

# EXIT trap body. Always attempts to terminate (also after a failure) unless told not to, and always
# returns 0 so the script keeps the exit status it was already ending with.
on_exit() {
  local status=${1:-0}
  if ((status != 0)) && [[ ! -f $DRBM25_FAILED_FILE ]]; then
    mark_failed "run_all.sh exited with status $status during stage '${RUNPOD_STAGE:-unknown}'"
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _copy_logs || _runpod_log "WARNING: copying logs failed"
  if [[ ${NO_TERMINATE-} == 1 || ${TERMINATE_ON_FINISH-} == no ]]; then
    _runpod_log "not terminating the pod (--no-terminate or TERMINATE_ON_FINISH=no): billing continues"
    return 0
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  terminate_pod || true
}
