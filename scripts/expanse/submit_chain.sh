#!/usr/bin/env bash
#
# submit_chain.sh: submit N chained train.sbatch jobs for one reward mode.
#
# Usage: see usage() below, or run with --help.
# Requires: bash 3.2+, Slurm's sbatch, EXPANSE_ACCOUNT in the environment.
# bash-target: 3.2

set -Eeuo pipefail

readonly SCRIPT_NAME=${0##*/}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly MAX_JOBS=20 # each job is up to 47.5 h; 20 is ~40 days, far past the allocation's needs

usage() {
  cat <<EOF2
Usage: $SCRIPT_NAME REWARD_MODE N [-- TRAIN_ARGS...]

Submits N copies of train.sbatch for REWARD_MODE (normal or count_once). Job k+1 starts
after job k ends, however it ended (--dependency=afterany), and resumes from the run's
latest checkpoint. Once the run has reached total_steps the remaining jobs exit 0
immediately. Prints the N job ids, one per line, on stdout.

Arguments after -- go to every job's drbm25.train call (for example a pilot:
  $SCRIPT_NAME normal 1 -- --run-name pilot_normal --until-step 150).

Run the two reward modes as two independent chains so they train concurrently:
  $SCRIPT_NAME normal 3
  $SCRIPT_NAME count_once 3

Run it from the repository root. Needs EXPANSE_ACCOUNT in the environment.
A job that fails for a real reason (not the time limit) still lets the next job start and
fail again; cancel the rest of the chain with scancel <ids>.
Exit status: 0 success, 1 runtime error, 2 usage error.
EOF2
}

usage_error() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  printf "Try '%s --help'.\n" "$SCRIPT_NAME" >&2
  exit 2
}

main() {
  # audit-bash disable=B307 # false positive: $# is the argument count, not outside data
  if (($# > 0)) && [[ $1 == -h || $1 == --help ]]; then
    usage
    exit 0
  fi
  (($# >= 2)) || usage_error "REWARD_MODE and N are required"
  local mode=$1 count=$2
  shift 2
  case $mode in
    normal | count_once) ;;
    *) usage_error "unknown reward mode '$mode' (expected normal or count_once)" ;;
  esac
  [[ $count =~ ^[0-9]{1,3}$ ]] || usage_error "N must be a positive integer, got '$count'"
  count=$((10#$count)) # 10#: "08" is decimal, not an invalid octal number
  if ((count < 1 || count > MAX_JOBS)); then
    usage_error "N must be between 1 and $MAX_JOBS, got $count"
  fi
  if (($# > 0)); then
    [[ $1 == -- ]] || usage_error "unexpected argument '$1' (extra train arguments go after --)"
    shift
  fi

  # shellcheck source=common.sh
  source "$SCRIPT_DIR/common.sh"
  require_env EXPANSE_ACCOUNT
  # Slurm runs train.sbatch with this as its working directory (SLURM_SUBMIT_DIR).
  local repo_root
  repo_root=$(cd "$SCRIPT_DIR/../.." && pwd)

  local previous='' job_id i
  local -a dependency
  for ((i = 1; i <= count; i++)); do
    dependency=()
    if [[ -n $previous ]]; then
      dependency=("--dependency=afterany:$previous")
    fi
    # --parsable prints "id" or "id;cluster"; keep the id. The account cannot be set with
    # #SBATCH inside train.sbatch because directives do not expand variables.
    job_id=$(sbatch --parsable --account="$EXPANSE_ACCOUNT" --chdir="$repo_root" ${dependency[@]+"${dependency[@]}"} \
      "$SCRIPT_DIR/train.sbatch" "$mode" "$@")
    job_id=${job_id%%;*}
    [[ $job_id =~ ^[0-9]+$ ]] || _common_die "sbatch returned an unexpected job id: '$job_id'"
    printf '%s\n' "$job_id"
    previous=$job_id
  done
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
