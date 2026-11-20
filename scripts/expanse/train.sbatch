#!/usr/bin/env bash
#SBATCH --job-name=drbm25-train
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -t 47:30:00
#SBATCH --constraint="lustre"
#SBATCH --no-requeue
#SBATCH --output=%x.%j.out
# The account cannot be set here (directives do not expand variables). submit_chain.sh passes
# --account="$EXPANSE_ACCOUNT"; for a direct run use:
#   sbatch --account="$EXPANSE_ACCOUNT" scripts/expanse/train.sbatch REWARD_MODE [TRAIN_ARGS...]
#
# train.sbatch: one resumable PPO training job (2 GPUs) for REWARD_MODE (normal|count_once).
#
# Resumes from the run's latest checkpoint. Exit status 0 means the run reached total_steps
# (or the --until-step in TRAIN_ARGS); any other status means failure or the time limit.
# Run it from the repository root. Extra arguments are passed to drbm25.train, e.g. a pilot:
#   --run-name pilot_normal --until-step 150 --set ppo.actor_lr=1e-5
# Requires: bash, the env made by prepare_login.sh, EXPANSE_ACCOUNT in the environment.
# audit-bash disable-file=B403 # straight-line Slurm job script
# bash-target: 4.0

set -Eeuo pipefail

readonly SCRIPT_NAME=train.sbatch
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
readonly REPO_ROOT
readonly CONFIG=configs/expanse_v100_lora.yaml

log() {
  local now
  now=$(date '+%Y-%m-%dT%H:%M:%S')
  printf '%s %s: %s\n' "$now" "$SCRIPT_NAME" "$*" >&2
}
die() {
  log "error: $*"
  exit 1
}

[[ -f $REPO_ROOT/scripts/expanse/common.sh ]] || die "submit from the repository root (no scripts/expanse/common.sh in $REPO_ROOT)"
# shellcheck source=common.sh
source "$REPO_ROOT/scripts/expanse/common.sh"

if (($# < 1)); then
  log "usage: sbatch --account=\"\$EXPANSE_ACCOUNT\" train.sbatch REWARD_MODE [TRAIN_ARGS...]"
  exit 2
fi
reward_mode=$1
shift
case $reward_mode in
  normal | count_once) ;;
  *)
    log "error: unknown reward mode '$reward_mode' (expected normal or count_once)"
    exit 2
    ;;
esac

# --run-name in TRAIN_ARGS (pilots) moves the run directory; default is the reward mode.
run_name=$reward_mode
train_args=()
while (($# > 0)); do
  case $1 in
    --run-name)
      (($# >= 2)) || {
        log "error: --run-name needs a value"
        exit 2
      }
      run_name=$2
      shift
      ;;
    --run-name=*) run_name=${1#*=} ;;
    *) train_args+=("$1") ;;
  esac
  shift
done

if [[ ! $run_name =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ || $run_name == *..* ]]; then
  log "error: run name '$run_name' must be letters, digits, '_', '.', '-' and contain no '..'"
  exit 2
fi

drbm25_paths
offline_env
jdk_env
require_java21
drbm25_set_args

cd "$REPO_ROOT"
total_steps=$(config_total_steps "$CONFIG")
run_dir=$DRBM25_RUNS_DIR/$run_name

# afterany chains start every job; later ones must do nothing once the run is finished.
# A malformed marker file makes run_is_complete return false, so the job trains (and rewrites it).
# shellcheck disable=SC2310 # intentional: any failure inside means "not complete"
if run_is_complete "$run_dir" "$total_steps"; then
  log "run $run_name already complete ($total_steps steps); nothing to do"
  exit 0
fi

torchrun=${DRBM25_TORCHRUN:-$UV_PROJECT_ENVIRONMENT/bin/torchrun}
[[ -x $torchrun ]] || die "no torchrun at $torchrun; run scripts/expanse/prepare_login.sh on the login node"

start_step=$(latest_step "$run_dir")
log "job ${SLURM_JOB_ID:-local}: $run_name ($reward_mode) from step $start_step, HF offline=${HF_HUB_OFFLINE-0}"
mkdir -p "$DRBM25_RUNS_DIR" "$WANDB_DIR"

# torchrun's exit status collapses to 1, so the status is taken from rank 0's exit_code file.
torchrun_rc=0
"$torchrun" --nproc_per_node 2 --standalone -m drbm25.train \
  --config "$CONFIG" --reward-mode "$reward_mode" --run-name "$run_name" \
  "${DRBM25_SET_ARGS[@]}" ${train_args[@]+"${train_args[@]}"} || torchrun_rc=$?

# shellcheck disable=SC2310 # the || die is the error handling; run_exit_code logs its own reason
code=$(run_exit_code "$run_dir") || die "torchrun exited $torchrun_rc and left no usable exit_code file in $run_dir"
case $code in
  0)
    log "training finished: exit_code 0"
    exit 0
    ;;
  3) die "drbm25.train stopped on its budget deadline (exit_code 3), which this job never sets; treating as an error" ;;
  *) die "drbm25.train failed: exit_code $code (torchrun exited $torchrun_rc); see $run_dir/train_rank0.log" ;;
esac
