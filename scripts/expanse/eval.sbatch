#!/usr/bin/env bash
#SBATCH --job-name=drbm25-eval
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH -t 04:00:00
#SBATCH --constraint="lustre"
#SBATCH --no-requeue
#SBATCH --output=%x.%j.out
# The account cannot be set here (directives do not expand variables). Submit with:
#   sbatch --account="$EXPANSE_ACCOUNT" scripts/expanse/eval.sbatch   (from the repo root)
#
# eval.sbatch: Exp 1 / Exp 2 scoring after both training runs have finished.
# Scores the original queries, the untrained model, and the "normal" and "count_once" runs at
# the largest step both runs have a complete checkpoint for (drbm25.eval.report.common_step),
# then writes the report. Everything lands in $DRBM25_ROOT/runs/eval/:
#   original.jsonl untrained.jsonl trained-normal.jsonl trained-count_once.jsonl report.md
# The report is built offline here; add --wandb from the login node (see README).
# Requires: bash, the env made by prepare_login.sh, EXPANSE_ACCOUNT in the environment.
# audit-bash disable-file=B403 # straight-line Slurm job script
# bash-target: 4.0

set -Eeuo pipefail

readonly SCRIPT_NAME=eval.sbatch
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

drbm25_paths
offline_env
jdk_env
require_java21
drbm25_set_args
cd "$REPO_ROOT"
python=${DRBM25_PYTHON:-$UV_PROJECT_ENVIRONMENT/bin/python}
[[ -x $python ]] || die "no python at $python; run scripts/expanse/prepare_login.sh on the login node"

normal_dir=$DRBM25_RUNS_DIR/normal
count_once_dir=$DRBM25_RUNS_DIR/count_once
out_dir=$DRBM25_RUNS_DIR/eval
mkdir -p "$out_dir"

# The step both runs reached. A run that is still behind caps the comparison, which is the
# point: both rows of the report must come from the same step. Exits non-zero when none is shared.
step=$("$python" -c '
import sys
from pathlib import Path
from drbm25.eval.report import common_step
print(common_step(Path(sys.argv[1]), Path(sys.argv[2])))
' "$normal_dir" "$count_once_dir")
[[ $step =~ ^[0-9]+$ ]] || die "common_step printed '$step', expected a step number"
step_name=$(printf 'step_%06d' "$step")
total_steps=$(config_total_steps "$CONFIG")
if ((step < total_steps)); then
  log "warning: evaluating step $step, but total_steps is $total_steps (training not finished?)"
fi
log "evaluating step $step ($step_name) of both runs"

run_eval() {
  local source=$1 out=$2
  shift 2
  "$python" -m drbm25.eval.run_eval --config "$CONFIG" --source "$source" --out "$out" \
    "${DRBM25_SET_ARGS[@]}" "$@"
}

run_eval original "$out_dir/original.jsonl"
run_eval untrained "$out_dir/untrained.jsonl"
run_eval checkpoint "$out_dir/trained-normal.jsonl" --checkpoint "$normal_dir/$step_name"
run_eval checkpoint "$out_dir/trained-count_once.jsonl" --checkpoint "$count_once_dir/$step_name"

"$python" -m drbm25.eval.report \
  --inputs "$out_dir/original.jsonl" "$out_dir/untrained.jsonl" \
  "$out_dir/trained-normal.jsonl" "$out_dir/trained-count_once.jsonl" \
  --out "$out_dir/report.md"
log "report: $out_dir/report.md"
