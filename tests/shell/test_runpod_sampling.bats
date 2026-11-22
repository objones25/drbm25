#!/usr/bin/env bats
# Tests for scripts/runpod/run_sampling.sh. The script is sourced in a child bash, the steps that need the
# network, Java or a GPU are replaced by functions that log their arguments, then main runs.

bats_require_minimum_version 1.5.0

setup() {
  SCRIPT="${BATS_TEST_DIRNAME}/../../scripts/runpod/run_sampling.sh"
  export HOME=$BATS_TEST_TMPDIR
  cd "$BATS_TEST_TMPDIR" || return 1
  export CALLS=$BATS_TEST_TMPDIR/calls.log
  : >"$CALLS"
  export DRBM25_ROOT=$BATS_TEST_TMPDIR/ws/drbm25
  mkdir -p "$DRBM25_ROOT"
}

# Runs main with stubbed steps. FAIL_ON (a substring of a _uv_timeout command line) makes that command fail.
run_main() {
  env -i PATH="$PATH" HOME="$HOME" CALLS="$CALLS" DRBM25_ROOT="$DRBM25_ROOT" DRBM25_NONINTERACTIVE=1 \
    HF_TOKEN=hf_x WANDB_API_KEY=wb_x FAIL_ON="${FAIL_ON-}" "$@" bash -c '
      source "$1"
      install_java21() { echo "install_java21" >>"$CALLS"; }
      install_uv() { echo "install_uv" >>"$CALLS"; }
      sampling_sync_deps() { echo "sync" >>"$CALLS"; }
      terminate_pod() { echo "terminate" >>"$CALLS"; }
      _uv_timeout() {
        shift
        echo "uv $*" >>"$CALLS"
        if [[ -n $FAIL_ON && "$*" == *"$FAIL_ON"* ]]; then return 1; fi
      }
      main
    ' _ "$SCRIPT"
}

uv_calls() { grep '^uv ' "$CALLS" | sed 's/^uv //'; }

@test "run_sampling runs pilot, full sampling, metrics, hub upload and report in order, then terminates" {
  run -0 run_main
  local calls=() line
  while IFS= read -r line; do calls+=("$line"); done < <(uv_calls)
  [[ ${calls[0]} == "python -m drbm25.eval.sample "*"--source original --n 8 --out-dir $DRBM25_ROOT/samples/pilot "*"--limit 64" ]]
  [[ ${calls[3]} == *"--source trained-count_once --n 8 --out-dir $DRBM25_ROOT/samples/pilot "*"--limit 64" ]]
  [[ ${calls[4]} == *"--source original --n 32 --out-dir $DRBM25_ROOT/samples/db "* ]]
  [[ ${calls[5]} == *"--source untrained --n 32 "* ]]
  [[ ${calls[6]} == *"--source trained-normal --n 32 "* ]]
  [[ ${calls[7]} == *"--source trained-count_once --n 32 "* ]]
  [[ ${calls[4]} != *"--limit"* ]]
  [[ ${calls[8]} == "python -m drbm25.analysis.pipeline --db $DRBM25_ROOT/samples/db --index-dir $DRBM25_ROOT/indexes --out $DRBM25_ROOT/samples/metrics"* ]]
  [[ ${calls[9]} == "python -m drbm25.analysis.hub --db $DRBM25_ROOT/samples/db --metrics $DRBM25_ROOT/samples/metrics --repo objones25/drbm25-rewrites" ]]
  [[ ${calls[10]} == "python -m drbm25.analysis.report --metrics $DRBM25_ROOT/samples/metrics --db $DRBM25_ROOT/samples/db "* ]]
  [[ $(tail -1 "$CALLS") == terminate ]]
  [[ ! -f $DRBM25_ROOT/samples/FAILED ]]
}

@test "run_sampling passes the index, data and output dirs to the sampler" {
  run -0 run_main
  uv_calls | head -1 | grep -q -- "--set retrieval.index_dir=$DRBM25_ROOT/indexes --set data.data_dir=$DRBM25_ROOT/data"
}

@test "SAMPLE_N, PILOT_N, PILOT_LIMIT and SAMPLE_HUB_REPO override the defaults" {
  run -0 run_main SAMPLE_N=4 PILOT_N=2 PILOT_LIMIT=5 SAMPLE_HUB_REPO=objones25/other
  uv_calls | sed -n 1p | grep -q -- '--n 2 .*--limit 5'
  uv_calls | sed -n 5p | grep -q -- '--n 4 '
  uv_calls | grep -q -- '--repo objones25/other'
}

@test "a failing sampler leaves samples/FAILED (not the training FAILED marker) and still terminates" {
  run -1 run_main FAIL_ON="--source untrained --n 32"
  [[ -f $DRBM25_ROOT/samples/FAILED ]]
  [[ ! -f $DRBM25_ROOT/FAILED ]]
  grep -q "stage 'sample'" "$DRBM25_ROOT/samples/FAILED"
  ! uv_calls | grep -q 'drbm25.analysis.pipeline'
  [[ $(tail -1 "$CALLS") == terminate ]]
}

@test "a failed W&B report does not fail the run: the database is already uploaded" {
  run -0 run_main FAIL_ON="drbm25.analysis.report"
  [[ ! -f $DRBM25_ROOT/samples/FAILED ]]
  [[ $(tail -1 "$CALLS") == terminate ]]
}

@test "a failed hub upload fails the run before the report" {
  run -1 run_main FAIL_ON="drbm25.analysis.hub"
  [[ -f $DRBM25_ROOT/samples/FAILED ]]
  ! uv_calls | grep -q 'drbm25.analysis.report'
}

@test "run_sampling stops when HF_TOKEN is missing" {
  run -1 run_main HF_TOKEN=
  [[ -f $DRBM25_ROOT/samples/FAILED ]]
  [[ -z $(uv_calls) ]]
}

@test "an earlier samples/FAILED stops the run unless DRBM25_CLEAR_FAILED=1" {
  mkdir -p "$DRBM25_ROOT/samples"
  echo old >"$DRBM25_ROOT/samples/FAILED"
  run -1 run_main
  [[ -z $(uv_calls) ]]
  run -0 run_main DRBM25_CLEAR_FAILED=1
  [[ ! -f $DRBM25_ROOT/samples/FAILED ]]
  ls "$DRBM25_ROOT"/logs/FAILED-*
}

@test "NO_TERMINATE=1 keeps the pod" {
  run -0 run_main NO_TERMINATE=1
  ! grep -qx terminate "$CALLS"
}
