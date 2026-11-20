#!/usr/bin/env bash
#
# common.sh: helpers shared by the Expanse scripts. Source it; do not execute it.
#
# Requires: bash 4.0+ (Expanse login and compute nodes; also fine on macOS bash 3.2 and 5.x).
# audit-bash disable-file=B403 # library of functions sourced by the other scripts
# bash-target: 3.2
#
# Interface:
#   require_env VAR...      die naming the first unset or empty variable
#   drbm25_paths            export DRBM25_ROOT, HF_HOME, UV_*, DRBM25_{INDEX,DATA,RUNS,JDK}_DIR
#   drbm25_set_args         fill DRBM25_SET_ARGS with --set overrides for the config paths
#   offline_env             offline Hugging Face and W&B unless DRBM25_ONLINE=1
#   jdk_env                 put the user-space JDK 21 on PATH/JAVA_HOME if one is installed
#   latest_step RUN_DIR     print the step named by RUN_DIR/latest (0 if none)
#   run_exit_code RUN_DIR   print RUN_DIR/exit_code (written by rank 0 of drbm25.train)
#   run_is_complete RUN_DIR TOTAL_STEPS   true if exit_code is 0 and latest step >= TOTAL_STEPS
#   java_major              print the major version of the `java` on PATH
#   require_java21          die unless `java` on PATH is version 21
#   config_total_steps CFG  print run.total_steps from a drbm25 YAML config
#   verify_sha256 FILE HEX  die unless FILE's SHA-256 equals HEX

set -Eeuo pipefail

_common_log() {
  printf '%s: %s\n' "${0##*/}" "$*" >&2
}

_common_die() {
  _common_log "error: $*"
  exit 1
}

require_env() {
  if (($# == 0)); then
    _common_log "require_env needs at least one variable name"
    exit 2
  fi
  local name
  for name in "$@"; do
    # ${!name-} is the indirect expansion; the default keeps set -u quiet when unset.
    if [[ -z ${!name-} ]]; then
      _common_die "required environment variable $name is not set"
    fi
  done
}

drbm25_paths() {
  require_env EXPANSE_ACCOUNT USER
  # The account id is never hard-coded; it comes from the caller's environment.
  DRBM25_ROOT=${DRBM25_ROOT:-/expanse/lustre/projects/$EXPANSE_ACCOUNT/$USER/drbm25}
  export DRBM25_ROOT
  export HF_HOME="$DRBM25_ROOT/hf"
  export DRBM25_INDEX_DIR="$DRBM25_ROOT/indexes"
  export DRBM25_DATA_DIR="$DRBM25_ROOT/data"
  export DRBM25_RUNS_DIR="$DRBM25_ROOT/runs"
  export DRBM25_JDK_DIR="$DRBM25_ROOT/jdk"
  # Offline W&B runs land here; `wandb sync` is run on the login node from this directory.
  export WANDB_DIR="$DRBM25_ROOT/wandb"
  # Venv and uv cache live on Lustre, not in the 100 GB home quota.
  export UV_PROJECT_ENVIRONMENT="$DRBM25_ROOT/venv"
  export UV_CACHE_DIR="$DRBM25_ROOT/uv-cache"
}

# Path overrides for configs/expanse_v100_lora.yaml, which holds placeholder paths.
drbm25_set_args() {
  require_env DRBM25_INDEX_DIR DRBM25_DATA_DIR DRBM25_RUNS_DIR
  # shellcheck disable=SC2034 # read by the scripts that source this file
  DRBM25_SET_ARGS=(
    --set "retrieval.index_dir=$DRBM25_INDEX_DIR"
    --set "data.data_dir=$DRBM25_DATA_DIR"
    --set "run.output_dir=$DRBM25_RUNS_DIR"
  )
}

offline_env() {
  case ${DRBM25_ONLINE-} in
    '')
      export HF_HUB_OFFLINE=1
      export WANDB_MODE=offline
      ;;
    1)
      # Clear inherited values so an earlier offline_env cannot leave the job offline.
      unset HF_HUB_OFFLINE WANDB_MODE
      ;;
    *) _common_die "DRBM25_ONLINE must be unset or 1, got '$DRBM25_ONLINE'" ;;
  esac
}

jdk_env() {
  require_env DRBM25_JDK_DIR
  local home=$DRBM25_JDK_DIR/current
  if [[ -d $home/bin ]]; then
    export JAVA_HOME=$home
    export PATH="$home/bin:$PATH"
  fi
}

latest_step() {
  local run_dir=$1 pointer name
  pointer=$run_dir/latest
  if [[ ! -f $pointer ]]; then
    printf '0\n'
    return 0
  fi
  name=$(<"$pointer")
  if [[ ! $name =~ ^step_([0-9]{6})$ ]]; then
    _common_log "error: malformed latest pointer in $pointer: '$name'"
    return 1
  fi
  # 10# forces base 10: step_000050 would otherwise be read as invalid octal.
  printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
}

run_exit_code() {
  local file=$1/exit_code code
  if [[ ! -f $file ]]; then
    _common_log "error: no exit_code file at $file"
    return 1
  fi
  code=$(<"$file")
  if [[ ! $code =~ ^[0-9]+$ ]]; then
    _common_log "error: exit_code file $file does not hold a number: '$code'"
    return 1
  fi
  printf '%s\n' "$code"
}

run_is_complete() {
  local run_dir=$1 total=$2 code step
  [[ -f $run_dir/exit_code ]] || return 1
  # Plain assignments (not in a condition) keep set -e live: a malformed file aborts loudly.
  code=$(run_exit_code "$run_dir")
  [[ $code == 0 ]] || return 1
  step=$(latest_step "$run_dir")
  ((step >= total))
}

java_major() {
  local first
  if ! command -v java >/dev/null 2>&1; then
    _common_log "error: java not found on PATH"
    return 1
  fi
  # `java -version` prints to stderr. sed -n 1p reads all input, so pipefail never sees SIGPIPE.
  first=$(java -version 2>&1 | sed -n 1p)
  if [[ $first =~ \"([0-9]+)(\.([0-9]+))? ]]; then
    # Java 8 and older report "1.8.0_x": the major version is the second number.
    if [[ ${BASH_REMATCH[1]} == 1 && -n ${BASH_REMATCH[3]} ]]; then
      printf '%s\n' "${BASH_REMATCH[3]}"
    else
      printf '%s\n' "${BASH_REMATCH[1]}"
    fi
    return 0
  fi
  _common_log "error: cannot parse java version from: $first"
  return 1
}

require_java21() {
  local major
  major=$(java_major)
  if [[ $major != 21 ]]; then
    _common_die "pyserini needs Java 21 but found Java $major; run scripts/expanse/prepare_login.sh or load a JDK 21 module"
  fi
}

# Reads run.total_steps with sed so the sbatch scripts need no Python before the job starts.
config_total_steps() {
  local config=$1 matches
  matches=$(sed -n 's/^  total_steps: *\([0-9][0-9]*\) *\(#.*\)\{0,1\}$/\1/p' "$config")
  if [[ ! $matches =~ ^[0-9]+$ ]]; then
    _common_die "cannot find exactly one 'run.total_steps' in $config (indented two spaces under run:)"
  fi
  printf '%s\n' "$matches"
}

verify_sha256() {
  local file=$1 expected=$2 actual
  [[ $expected =~ ^[0-9a-f]{64}$ ]] || _common_die "expected digest must be 64 hex characters (lowercase), got '$expected'"
  [[ -f $file ]] || _common_die "cannot verify $file: no such file"
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum -- "$file")
  else
    actual=$(shasum -a 256 -- "$file")
  fi
  actual=${actual%% *}
  if [[ $actual != "$expected" ]]; then
    _common_die "SHA-256 mismatch for $file: expected $expected, got $actual"
  fi
}
