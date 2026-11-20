#!/usr/bin/env bats
# Tests for scripts/runpod/lib.sh and run_all.sh. Run: bats tests/shell/
# lib.sh is sourced in a child bash (`bash -c`), never into bats itself. curl, uv, sleep and the
# train/eval commands are PATH shims that log every call to $STUB_LOG.

bats_require_minimum_version 1.5.0

setup() {
  LIB="${BATS_TEST_DIRNAME}/../../scripts/runpod/lib.sh"
  RUN_ALL="${BATS_TEST_DIRNAME}/../../scripts/runpod/run_all.sh"
  export HOME=$BATS_TEST_TMPDIR
  cd "$BATS_TEST_TMPDIR" || return 1

  export STUB_LOG=$BATS_TEST_TMPDIR/calls.log
  export STUB_DIR=$BATS_TEST_TMPDIR/stub-state
  : >"$STUB_LOG"
  mkdir -p "$STUB_DIR" "$BATS_TEST_TMPDIR/bin"
  make_stubs

  export DRBM25_ROOT=$BATS_TEST_TMPDIR/ws/drbm25
  export DRBM25_SLEEP=sleep_stub
  export DRBM25_TTY=$BATS_TEST_TMPDIR/tty
  : >"$DRBM25_TTY"
  mkdir -p "$DRBM25_ROOT"
}

make_stubs() {
  local bin=$BATS_TEST_TMPDIR/bin
  # curl fails the first $STUB_CURL_FAILS calls, then succeeds.
  cat >"$bin/curl" <<'EOF'
#!/usr/bin/env bash
echo "curl $*" >>"$STUB_LOG"
n=$(cat "$STUB_DIR/curl_n" 2>/dev/null || echo 0)
echo $((n + 1)) >"$STUB_DIR/curl_n"
if ((n < ${STUB_CURL_FAILS:-0})); then exit 22; fi
exit 0
EOF
  cat >"$bin/sleep_stub" <<'EOF'
#!/usr/bin/env bash
echo "sleep $*" >>"$STUB_LOG"
EOF
  # uv run --no-sync CMD...: torchrun writes exit_code from $STUB_DIR/exit_<run>_<step> (default 0).
  cat >"$bin/uv" <<'EOF'
#!/usr/bin/env bash
echo "uv $*" >>"$STUB_LOG"
[[ $1 == run && $2 == --no-sync ]] && shift 2 || exit 0
cmd=$1
shift
case $cmd in
  torchrun)
    if [[ $* == *memory_check.py* ]]; then exit "${STUB_MEM_RC:-0}"; fi
    run='' step='' out=''
    while (($#)); do
      case $1 in
        --run-name) run=$2 ;;
        --until-step) step=$2 ;;
        run.output_dir=*) out=${1#run.output_dir=} ;;
      esac
      shift
    done
    mkdir -p "$out/$run"
    code=0
    [[ -f $STUB_DIR/exit_${run}_$step ]] && code=$(cat "$STUB_DIR/exit_${run}_$step")
    if [[ $code != missing ]]; then echo "$code" >"$out/$run/exit_code"; fi
    echo '{"step": 1}' >>"$out/$run/metrics.jsonl"
    if [[ $code == 0 && -z ${STUB_NO_LATEST-} ]]; then printf 'step_%06d\n' "$step" >"$out/$run/latest"; fi
    exit 1 # torchrun collapses worker codes
    ;;
  pytest) exit "${STUB_PYTEST_RC:-0}" ;;
  python)
    if [[ $* == *common_step* ]]; then echo "${STUB_COMMON_STEP:-100}"; fi
    if [[ $* == *deadline_from_budget* ]]; then echo 1234567890; fi
    if [[ $* == *spike_monitor.py* ]]; then exit "${STUB_SPIKE_RC:-0}"; fi
    exit 0
    ;;
esac
EOF
  # GNU timeout shim: STUB_TIMEOUT_RC makes it report a timeout without running the command.
  cat >"$bin/timeout" <<'EOF'
#!/usr/bin/env bash
echo "timeout $*" >>"$STUB_LOG"
if [[ -n ${STUB_TIMEOUT_RC-} ]]; then exit "$STUB_TIMEOUT_RC"; fi
while [[ $1 == --* ]]; do shift; done
shift
exec "$@"
EOF
  cat >"$bin/df" <<'EOF'
#!/usr/bin/env bash
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "vol 1 1 ${STUB_DF_KB:-999999999999} 1% /workspace"
EOF
  chmod +x "$bin"/*
  export PATH=$bin:$PATH
}

# Run a snippet with lib.sh sourced. Extra args are KEY=VALUE environment entries.
in_lib() {
  env -i PATH="$PATH" HOME="$HOME" STUB_LOG="$STUB_LOG" STUB_DIR="$STUB_DIR" \
    DRBM25_ROOT="$DRBM25_ROOT" DRBM25_SLEEP="$DRBM25_SLEEP" DRBM25_TTY="$DRBM25_TTY" \
    DRBM25_NOW_UNIX="${DRBM25_NOW_UNIX-}" \
    "$@" bash -c 'source "$1"; shift; runpod_paths; eval "$1"' _ "$LIB" "$SNIPPET"
}

calls() { cat "$STUB_LOG"; }

train_calls() { grep '^uv ' "$STUB_LOG" | grep -o 'torchrun.*--run-name [a-z_]* .*--until-step [0-9]*' | sed 's/.*--run-name \([a-z_]*\) .*--until-step \([0-9]*\)/\1@\2/'; }

# ---------------------------------------------------------------- env file

@test "write_env_file writes mode 600, NAME=value lines, and echoes no value" {
  SNIPPET='write_env_file "$DRBM25_ROOT/.env" HF_TOKEN WANDB_PROJECT'
  run -0 --separate-stderr in_lib HF_TOKEN=hf_secret123 WANDB_PROJECT=drbm25
  [[ $output != *hf_secret123* ]]
  [[ $stderr != *hf_secret123* ]]
  [[ $(stat -c %a "$DRBM25_ROOT/.env" 2>/dev/null || stat -f %Lp "$DRBM25_ROOT/.env") == 600 ]]
  grep -qx 'HF_TOKEN=hf_secret123' "$DRBM25_ROOT/.env"
  grep -qx 'WANDB_PROJECT=drbm25' "$DRBM25_ROOT/.env"
}

@test "write_env_file skips unset variables and keeps set-but-empty ones as NAME=" {
  SNIPPET='write_env_file "$DRBM25_ROOT/.env" HF_TOKEN WANDB_ENTITY WANDB_PROJECT'
  run -0 in_lib HF_TOKEN=abc WANDB_ENTITY=
  grep -qx 'WANDB_ENTITY=' "$DRBM25_ROOT/.env"
  ! grep -q WANDB_PROJECT "$DRBM25_ROOT/.env"
}

@test "write_env_file rejects a value holding a newline and writes nothing" {
  SNIPPET='write_env_file "$DRBM25_ROOT/.env" HF_TOKEN'
  run -1 --separate-stderr in_lib "HF_TOKEN=$(printf 'a\nb')"
  [[ $stderr == *HF_TOKEN* ]]
  [[ ! -e $DRBM25_ROOT/.env ]]
}

@test "load_env_file loads NAME=value lines and never executes code in values" {
  printf 'HF_TOKEN=abc\nlower=nope\nBAD LINE\nWANDB_PROJECT=$(touch %s/pwned)\n# comment\n' "$BATS_TEST_TMPDIR" >"$DRBM25_ROOT/.env"
  SNIPPET='load_env_file "$DRBM25_ROOT/.env"; printf "%s|%s|%s\n" "$HF_TOKEN" "${lower-unset}" "$WANDB_PROJECT"'
  run -0 in_lib
  [[ $output == 'abc|unset|$(touch '*'/pwned)' ]]
  [[ ! -e $BATS_TEST_TMPDIR/pwned ]]
}

@test "load_env_file ignores names outside the settings list, such as PATH" {
  printf 'PATH=/nonexistent\nHF_TOKEN=abc\n' >"$DRBM25_ROOT/.env"
  SNIPPET='load_env_file "$DRBM25_ROOT/.env"; command -v bash >/dev/null && printf "%s" "$HF_TOKEN"'
  run -0 in_lib
  [[ $output == abc ]]
}

@test "load_env_file does not override a variable already in the environment" {
  printf 'HF_TOKEN=from_file\n' >"$DRBM25_ROOT/.env"
  SNIPPET='load_env_file "$DRBM25_ROOT/.env"; printf "%s" "$HF_TOKEN"'
  run -0 in_lib HF_TOKEN=from_env
  [[ $output == from_env ]]
}

@test "load_env_file on a missing file is a no-op" {
  SNIPPET='load_env_file "$DRBM25_ROOT/none.env"'
  run -0 in_lib
}

# ---------------------------------------------------------------- prompts

@test "prompt_var keeps a variable that is already set and reads nothing" {
  SNIPPET='prompt_var BUDGET_USD "Budget"; printf "%s" "$BUDGET_USD"'
  printf 'ignored\n' >"$DRBM25_TTY"
  run -0 in_lib BUDGET_USD=90
  [[ $output == 90 ]]
}

@test "prompt_var reads a missing variable from the tty file" {
  SNIPPET='prompt_var BUDGET_USD "Budget"; printf "%s" "$BUDGET_USD"'
  printf '75\n' >"$DRBM25_TTY"
  run -0 --separate-stderr in_lib
  [[ $output == 75 ]]
  [[ $stderr == *Budget* ]]
}

@test "prompt_var --secret does not echo the value" {
  SNIPPET='prompt_var HF_TOKEN "HF token" --secret; [[ $HF_TOKEN == hf_topsecret ]]'
  printf 'hf_topsecret\n' >"$DRBM25_TTY"
  run -0 --separate-stderr in_lib
  [[ $output != *hf_topsecret* ]]
  [[ $stderr != *hf_topsecret* ]]
}

@test "prompt_var uses --default on an empty answer" {
  SNIPPET='prompt_var TERMINATE_ON_FINISH "Terminate" --default yes; printf "%s" "$TERMINATE_ON_FINISH"'
  printf '\n' >"$DRBM25_TTY"
  run -0 --separate-stderr in_lib
  [[ $output == yes ]]
}

@test "prompt_var --optional accepts an empty answer" {
  SNIPPET='prompt_var WANDB_ENTITY "Entity" --optional; printf "[%s]" "$WANDB_ENTITY"'
  printf '\n' >"$DRBM25_TTY"
  run -0 --separate-stderr in_lib
  [[ $output == '[]' ]]
}

@test "prompt_var fails on an empty required answer, naming the variable" {
  SNIPPET='prompt_var BUDGET_USD "Budget"'
  printf '\n' >"$DRBM25_TTY"
  run -1 --separate-stderr in_lib
  [[ $stderr == *BUDGET_USD* ]]
}

@test "prompt_var fails with a hint when there is no tty input at all" {
  SNIPPET='prompt_var BUDGET_USD "Budget"'
  export DRBM25_TTY=$BATS_TEST_TMPDIR/does-not-exist
  run -1 --separate-stderr in_lib
  [[ $stderr == *BUDGET_USD* ]]
}

@test "collect_settings on a re-run with an existing .env prompts only for missing values" {
  cat >"$DRBM25_ROOT/.env" <<'EOF'
HF_TOKEN=hf_a
RUNPOD_API_KEY=rp_a
WANDB_PROJECT=drbm25
TERMINATE_ON_FINISH=yes
WANDB_ENTITY=
EOF
  SNIPPET='collect_settings; printf "%s" "$WANDB_API_KEY"'
  printf 'wb_new\n' >"$DRBM25_TTY"
  run -0 --separate-stderr in_lib BUDGET_USD=90 PRICE_PER_HOUR=2.78
  [[ $output == wb_new ]]
  [[ $stderr == *WANDB_API_KEY* ]]
  [[ $stderr != *HF_TOKEN* ]]
  [[ $stderr != *BUDGET_USD* ]]
  grep -qx 'WANDB_API_KEY=wb_new' "$DRBM25_ROOT/.env"
  ! grep -q 'BUDGET_USD\|PRICE_PER_HOUR' "$DRBM25_ROOT/.env"
  [[ $(stat -c %a "$DRBM25_ROOT/.env" 2>/dev/null || stat -f %Lp "$DRBM25_ROOT/.env") == 600 ]]
}

@test "collect_settings never reuses a saved budget or price from the .env" {
  printf 'HF_TOKEN=a\nWANDB_API_KEY=b\nRUNPOD_API_KEY=c\nWANDB_PROJECT=p\nWANDB_ENTITY=\nBUDGET_USD=90\nPRICE_PER_HOUR=2.78\nTERMINATE_ON_FINISH=yes\n' >"$DRBM25_ROOT/.env"
  SNIPPET='collect_settings'
  run -1 --separate-stderr in_lib
  [[ $stderr == *BUDGET_USD* ]]
}

@test "collect_settings shows the dollars already spent on earlier launches at the budget prompt" {
  printf '{"spent_usd": 41.5000, "updated_unix": 1}\n' >"$DRBM25_ROOT/spend.json"
  SNIPPET='collect_settings'
  printf '50\n' >"$DRBM25_TTY"
  run -1 --separate-stderr in_lib HF_TOKEN=a WANDB_API_KEY=b RUNPOD_API_KEY=c WANDB_PROJECT=p WANDB_ENTITY=
  [[ $stderr == *41.5* ]]
}

@test "collect_settings rejects a non-numeric budget and an invalid terminate flag" {
  SNIPPET='collect_settings'
  run -1 --separate-stderr in_lib HF_TOKEN=a WANDB_API_KEY=b RUNPOD_API_KEY=c WANDB_PROJECT=p BUDGET_USD=lots PRICE_PER_HOUR=2.78
  [[ $stderr == *BUDGET_USD* ]]
  run -1 --separate-stderr in_lib HF_TOKEN=a WANDB_API_KEY=b RUNPOD_API_KEY=c WANDB_PROJECT=p BUDGET_USD=90 PRICE_PER_HOUR=2.78 TERMINATE_ON_FINISH=maybe
  [[ $stderr == *TERMINATE_ON_FINISH* ]]
}

# ---------------------------------------------------------------- markers

@test "step_done is false until mark_done, which creates the marker under .done" {
  SNIPPET='if step_done data; then echo early; fi; mark_done data; step_done data && echo done'
  run -0 in_lib
  [[ $output == done ]]
  [[ -f $DRBM25_ROOT/.done/data ]]
}

@test "mark_done rejects a marker name with a path separator" {
  SNIPPET='mark_done ../evil'
  run -1 in_lib
  [[ ! -e $DRBM25_ROOT/evil ]]
}

# ---------------------------------------------------------------- terminate_pod

@test "terminate_pod sends DELETE to the pod URL and returns 0 on the first success" {
  SNIPPET='terminate_pod'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key
  grep -q 'curl --fail --silent --show-error --max-time 30 -X DELETE' "$STUB_LOG"
  grep -q 'https://api.runpod.io/v2/pods/pod123' "$STUB_LOG"
  [[ $(grep -c '^curl' "$STUB_LOG") == 1 ]]
  [[ $output != *rp_key* ]]
}

@test "terminate_pod keeps the API key out of curl's arguments" {
  SNIPPET='terminate_pod'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key
  ! grep -q rp_key "$STUB_LOG"
  grep -q -- '-H @-' "$STUB_LOG"
}

@test "terminate_pod succeeds on the 2nd attempt after a 5 s backoff" {
  SNIPPET='terminate_pod'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key STUB_CURL_FAILS=1
  [[ $(grep -c '^curl' "$STUB_LOG") == 2 ]]
  [[ $(grep '^sleep' "$STUB_LOG") == 'sleep 5' ]]
}

@test "terminate_pod retries 3 times, then returns 1 and prints the pod id and console URL" {
  SNIPPET='terminate_pod'
  STUB_CURL_FAILS=99
  export STUB_CURL_FAILS
  run -1 --separate-stderr in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key STUB_CURL_FAILS=99
  [[ $(grep -c '^curl' "$STUB_LOG") == 3 ]]
  [[ $(grep '^sleep' "$STUB_LOG" | tr '\n' ',') == 'sleep 5,sleep 15,' ]]
  [[ $stderr == *pod123* ]]
  [[ $stderr == *https://console.runpod.io/pods* ]]
  [[ $stderr != *rp_key* ]]
}

@test "terminate_pod without RUNPOD_POD_ID warns loudly and does not call curl" {
  SNIPPET='terminate_pod'
  run -1 --separate-stderr in_lib RUNPOD_API_KEY=rp_key
  [[ $stderr == *RUNPOD_POD_ID* ]]
  [[ ! -s $STUB_LOG ]]
}

@test "terminate_pod without RUNPOD_API_KEY fails naming the pod and does not call curl" {
  SNIPPET='terminate_pod'
  run -1 --separate-stderr in_lib RUNPOD_POD_ID=pod123
  [[ $stderr == *RUNPOD_API_KEY* ]]
  [[ $stderr == *pod123* ]]
  [[ ! -s $STUB_LOG ]]
}

@test "terminate_pod does not leak the key under xtrace" {
  SNIPPET='set -x; terminate_pod'
  run -0 --separate-stderr in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key
  [[ $stderr != *rp_key* ]]
}

# ---------------------------------------------------------------- chunks

chunk_env() { printf '%s\n' "DEADLINE_UNIX=2000000000" "RESERVE_S=3600"; }

@test "run_alternating_chunks 100 50 trains normal@50, count_once@50, normal@100, count_once@100" {
  SNIPPET='runpod_paths; run_alternating_chunks 100 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $(train_calls | tr '\n' ' ') == 'normal@50 count_once@50 normal@100 count_once@100 ' ]]
}

@test "run_alternating_chunks passes the config, reward mode, deadline and reserve to drbm25.train" {
  SNIPPET='runpod_paths; run_alternating_chunks 50 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  grep -q -- '--nproc_per_node 2' "$STUB_LOG"
  grep -q -- '-m drbm25.train --config configs/runpod_a100_full.yaml --reward-mode normal --run-name normal' "$STUB_LOG"
  grep -q -- '--reward-mode count_once --run-name count_once' "$STUB_LOG"
  grep -q -- '--deadline-unix 2000000000 --reserve-s 3600' "$STUB_LOG"
  grep -q -- "run.output_dir=$DRBM25_ROOT/runs" "$STUB_LOG"
}

@test "run_alternating_chunks ends on UNTIL when it is not a multiple of CHUNK" {
  SNIPPET='runpod_paths; run_alternating_chunks 120 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $(train_calls | tr '\n' ' ') == 'normal@50 count_once@50 normal@100 count_once@100 normal@120 count_once@120 ' ]]
}

@test "run_alternating_chunks runs spike_monitor on both metrics files after each chunk" {
  SNIPPET='runpod_paths; run_alternating_chunks 100 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $(grep -c 'spike_monitor.py .*runs/normal/metrics.jsonl' "$STUB_LOG") == 2 ]]
  [[ $(grep -c 'spike_monitor.py .*runs/count_once/metrics.jsonl' "$STUB_LOG") == 2 ]]
}

@test "a failing spike_monitor is logged as a warning and is not fatal" {
  SNIPPET='runpod_paths; run_alternating_chunks 50 50'
  run -0 --separate-stderr in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_SPIKE_RC=1
  [[ $stderr == *"spike_monitor"* ]]
}

@test "a budget stop (exit_code 3) ends the loop with status 3 and skips the other run" {
  echo 3 >"$STUB_DIR/exit_normal_50"
  SNIPPET='runpod_paths; rc=0; run_alternating_chunks 100 50 || rc=$?; echo "rc=$rc"'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $output == *rc=3* ]]
  [[ $(train_calls | tr '\n' ' ') == 'normal@50 ' ]]
}

@test "a failed run (exit_code 2) returns 1 and writes the FAILED marker" {
  echo 2 >"$STUB_DIR/exit_count_once_50"
  SNIPPET='runpod_paths; rc=0; run_alternating_chunks 100 50 || rc=$?; echo "rc=$rc"'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $output == *rc=1* ]]
  [[ $(train_calls | tr '\n' ' ') == 'normal@50 count_once@50 ' ]]
  [[ -f $DRBM25_ROOT/FAILED ]]
  grep -q 'count_once' "$DRBM25_ROOT/FAILED"
}

@test "a missing exit_code file is a failure" {
  echo missing >"$STUB_DIR/exit_normal_50"
  SNIPPET='runpod_paths; rc=0; run_alternating_chunks 100 50 || rc=$?; echo "rc=$rc"'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $output == *rc=1* ]]
  [[ -f $DRBM25_ROOT/FAILED ]]
}

@test "exit_code 1 (killed or unfinished) is a failure" {
  echo 1 >"$STUB_DIR/exit_normal_50"
  SNIPPET='runpod_paths; rc=0; run_alternating_chunks 100 50 || rc=$?; echo "rc=$rc"'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $output == *rc=1* ]]
}

@test "an existing FAILED marker stops run_alternating_chunks before any training" {
  echo "earlier failure" >"$DRBM25_ROOT/FAILED"
  SNIPPET='runpod_paths; run_alternating_chunks 100 50'
  run -1 --separate-stderr in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $stderr == *FAILED* ]]
  [[ -z $(train_calls) ]]
}

@test "run_alternating_chunks rejects non-positive or non-numeric arguments" {
  SNIPPET='runpod_paths; run_alternating_chunks 100 0'
  run -1 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  SNIPPET='runpod_paths; run_alternating_chunks abc 50'
  run -1 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
}

# ---------------------------------------------------------------- deadline

@test "compute_deadline_unix calls drbm25.budget.deadline_from_budget with the budget and price" {
  SNIPPET='compute_deadline_unix 1000 90 2.78'
  run -0 in_lib
  [[ $output == 1234567890 ]]
  grep -q 'deadline_from_budget' "$STUB_LOG"
  grep -q ' 1000 90 2.78' "$STUB_LOG"
}

@test "compute_deadline_unix rejects a non-numeric price" {
  SNIPPET='compute_deadline_unix 1000 90 cheap'
  run -1 --separate-stderr in_lib
  [[ $stderr == *price* ]]
}

# ---------------------------------------------------------------- smoke

@test "smoke_stage runs the gpu pytest and the memory check" {
  SNIPPET='runpod_paths; smoke_stage'
  run -0 in_lib
  grep -q 'pytest .*-m gpu tests/rollout/test_vllm_engine.py tests/ppo/test_models_full.py' "$STUB_LOG"
  grep -q 'torchrun --nproc_per_node 2 scripts/memory_check.py --config configs/runpod_a100_full.yaml' "$STUB_LOG"
}

@test "smoke_stage fails when the gpu tests fail" {
  SNIPPET='runpod_paths; smoke_stage'
  run -1 --separate-stderr in_lib STUB_PYTEST_RC=1
  [[ $stderr == *pytest* ]]
}

@test "smoke_stage fails when the memory check fails (headroom under 2 GiB)" {
  SNIPPET='runpod_paths; smoke_stage'
  run -1 --separate-stderr in_lib STUB_MEM_RC=1
  [[ $stderr == *memory* ]]
}

# ---------------------------------------------------------------- train_and_report

@test "train_and_report evaluates original, untrained and both checkpoints at the common step, then reports" {
  SNIPPET='runpod_paths; train_and_report 100 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_COMMON_STEP=100
  grep -q 'run_eval --config configs/runpod_a100_full.yaml --source original' "$STUB_LOG"
  grep -q -- '--source untrained' "$STUB_LOG"
  grep -q -- "--source checkpoint --out $DRBM25_ROOT/runs/eval/trained-normal.jsonl --checkpoint $DRBM25_ROOT/runs/normal/step_000100" "$STUB_LOG"
  grep -q -- "--checkpoint $DRBM25_ROOT/runs/count_once/step_000100" "$STUB_LOG"
  grep -q -- 'drbm25.eval.report --inputs' "$STUB_LOG"
  grep -q -- '--wandb --config configs/runpod_a100_full.yaml' "$STUB_LOG"
}

@test "a budget exit stops the loop and still runs eval on what exists" {
  echo 3 >"$STUB_DIR/exit_count_once_50"
  SNIPPET='runpod_paths; train_and_report 100 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_COMMON_STEP=50
  [[ $(train_calls | tr '\n' ' ') == 'normal@50 count_once@50 ' ]]
  grep -q -- '--source original' "$STUB_LOG"
  grep -q -- "runs/normal/step_000050" "$STUB_LOG"
  grep -q -- 'drbm25.eval.report' "$STUB_LOG"
  [[ ! -e $DRBM25_ROOT/FAILED ]]
}

@test "a training failure returns 1 and runs no eval" {
  echo 2 >"$STUB_DIR/exit_normal_50"
  SNIPPET='runpod_paths; train_and_report 100 50'
  run -1 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  ! grep -q 'run_eval' "$STUB_LOG"
  [[ -f $DRBM25_ROOT/FAILED ]]
}

# ---------------------------------------------------------------- on_exit

on_exit_env() { printf '%s\n' RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key; }

@test "on_exit terminates the pod after a failure and writes the FAILED marker" {
  SNIPPET='runpod_paths; on_exit 1'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key TERMINATE_ON_FINISH=yes
  grep -q 'curl .*-X DELETE .*pods/pod123' "$STUB_LOG"
  [[ -f $DRBM25_ROOT/FAILED ]]
}

@test "on_exit terminates after success too" {
  SNIPPET='runpod_paths; on_exit 0'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key
  grep -q 'curl .*-X DELETE' "$STUB_LOG"
  [[ ! -e $DRBM25_ROOT/FAILED ]]
}

@test "on_exit does not terminate with NO_TERMINATE=1" {
  SNIPPET='runpod_paths; on_exit 1'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key NO_TERMINATE=1
  ! grep -q '^curl' "$STUB_LOG"
}

@test "on_exit does not terminate with TERMINATE_ON_FINISH=no" {
  SNIPPET='runpod_paths; on_exit 1'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key TERMINATE_ON_FINISH=no
  ! grep -q '^curl' "$STUB_LOG"
}

@test "on_exit copies run logs into logs/ before terminating" {
  mkdir -p "$DRBM25_ROOT/runs/normal"
  echo log >"$DRBM25_ROOT/runs/normal/train_rank0.log"
  echo 0 >"$DRBM25_ROOT/runs/normal/exit_code"
  SNIPPET='runpod_paths; on_exit 0'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key NO_TERMINATE=1
  [[ -f $DRBM25_ROOT/logs/normal/train_rank0.log ]]
  [[ -f $DRBM25_ROOT/logs/normal/exit_code ]]
}

@test "on_exit still returns 0 when terminate fails, so the original status is kept" {
  SNIPPET='runpod_paths; on_exit 1'
  run -0 --separate-stderr in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key STUB_CURL_FAILS=99
  [[ $stderr == *pod123* ]]
}

# ---------------------------------------------------------------- run_all.sh

@test "run_all.sh --help exits 0 and documents --no-terminate" {
  run -0 bash "$RUN_ALL" --help
  [[ $output == *--no-terminate* ]]
}

@test "run_all.sh rejects an unknown option with status 2" {
  run -2 --separate-stderr bash "$RUN_ALL" --bogus
  [[ $stderr == *--bogus* ]]
}

# ---------------------------------------------------------------- fix round 1

@test "on_exit still terminates the pod when the volume is unwritable (C1)" {
  SNIPPET='set -Ee; trap "false" ERR; chmod -R a-w "$DRBM25_ROOT"; rc=0; on_exit 1 || rc=$?; chmod -R u+w "$DRBM25_ROOT"; echo "rc=$rc"'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key
  grep -q 'curl .*-X DELETE .*pods/pod123' "$STUB_LOG"
}

@test "on_exit terminates even when the paths were never set up (C1)" {
  SNIPPET='unset DRBM25_ROOT; set -u; bash_rc=0; DRBM25_FAILED_FILE=/nonexistent/dir/FAILED DRBM25_LOG_DIR=/nonexistent/logs DRBM25_RUNS_DIR=/nonexistent/runs on_exit 1'
  run -0 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key
  grep -q 'curl .*-X DELETE' "$STUB_LOG"
}

@test "runpod config keeps one checkpoint per run (C2)" {
  grep -qE '^  keep_ckpts: 1( |$)' "${BATS_TEST_DIRNAME}/../../configs/runpod_a100_full.yaml"
}

@test "a chunk is refused with FAILED when the volume has under 90 GB free and no checkpoint exists (C2)" {
  SNIPPET='rc=0; run_alternating_chunks 50 50 || rc=$?; echo "rc=$rc"'
  run -0 --separate-stderr in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_DF_KB=94371839
  [[ $output == *rc=1* ]]
  [[ -z $(train_calls) ]]
  grep -qi 'disk' "$DRBM25_ROOT/FAILED"
}

@test "a chunk runs when the volume has exactly 90 GB free and no checkpoint exists (C2)" {
  SNIPPET='run_alternating_chunks 50 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_DF_KB=94371840
  [[ $(train_calls | tr '\n' ' ') == 'normal@50 count_once@50 ' ]]
}

@test "the disk preflight needs 1.2x the last checkpoint size once one exists (C2)" {
  local mode
  for mode in normal count_once; do
    mkdir -p "$DRBM25_ROOT/runs/$mode/step_000050"
    head -c 2097152 /dev/zero >"$DRBM25_ROOT/runs/$mode/step_000050/blob"
    echo step_000050 >"$DRBM25_ROOT/runs/$mode/latest"
  done
  SNIPPET='rc=0; run_alternating_chunks 100 50 || rc=$?; echo "rc=$rc"'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_DF_KB=2400
  [[ $output == *rc=1* ]]
  [[ -z $(train_calls) ]]
  grep -q 'normal' "$DRBM25_ROOT/FAILED"
  rm -f "$DRBM25_ROOT/FAILED"
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_DF_KB=2500
  [[ $(train_calls | head -1) == normal@100 ]]
}

@test "the report carries the W&B project and entity (W&B)" {
  SNIPPET='train_and_report 50 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_COMMON_STEP=50 WANDB_PROJECT=myproj WANDB_ENTITY=myteam
  grep 'drbm25.eval.report' "$STUB_LOG" | grep -q -- '--set run.wandb_project=myproj'
  grep 'drbm25.eval.report' "$STUB_LOG" | grep -q -- '--set run.wandb_entity=myteam'
}

@test "spend_record stores spent-so-far = earlier spend + elapsed hours x price (I1)" {
  printf '{"spent_usd": 10.0000, "updated_unix": 1}\n' >"$DRBM25_ROOT/spend.json"
  SNIPPET='spend_begin 1000 2.5; spend_record'
  run -0 in_lib DRBM25_NOW_UNIX=4600
  grep -q '"spent_usd": 12.5000' "$DRBM25_ROOT/spend.json"
}

@test "spend_record accumulates across launches without double counting (I1)" {
  SNIPPET='spend_begin 0 2; spend_record; spend_record'
  run -0 in_lib DRBM25_NOW_UNIX=1800
  grep -q '"spent_usd": 1.0000' "$DRBM25_ROOT/spend.json"
  SNIPPET='spend_total'
  run -0 in_lib
  [[ $output == 1.0000 ]]
}

@test "spend_total is 0 when no spend file exists, and fails on a malformed one (I1)" {
  SNIPPET='spend_total'
  run -0 in_lib
  [[ $output == 0 ]]
  echo garbage >"$DRBM25_ROOT/spend.json"
  run -1 --separate-stderr in_lib
  [[ $stderr == *spend.json* ]]
}

@test "spend_record before spend_begin is a no-op (I1)" {
  SNIPPET='spend_record'
  run -0 in_lib
  [[ ! -e $DRBM25_ROOT/spend.json ]]
}

@test "each finished chunk and on_exit record spend (I1)" {
  SNIPPET='spend_begin 0 3; run_alternating_chunks 50 50'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 DRBM25_NOW_UNIX=3600
  grep -q '"spent_usd": 3.0000' "$DRBM25_ROOT/spend.json"
  rm "$DRBM25_ROOT/spend.json"
  SNIPPET='spend_begin 0 3; on_exit 0'
  run -0 in_lib DRBM25_NOW_UNIX=7200 NO_TERMINATE=1
  grep -q '"spent_usd": 6.0000' "$DRBM25_ROOT/spend.json"
}

@test "a stale exit_code 0 from an earlier chunk is not read as success (I2)" {
  mkdir -p "$DRBM25_ROOT/runs/normal"
  echo 0 >"$DRBM25_ROOT/runs/normal/exit_code"
  echo missing >"$STUB_DIR/exit_normal_50"
  SNIPPET='rc=0; run_alternating_chunks 50 50 || rc=$?; echo "rc=$rc"'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600
  [[ $output == *rc=1* ]]
  [[ -f $DRBM25_ROOT/FAILED ]]
}

@test "exit_code 0 without a checkpoint at the requested step is a failure (I2)" {
  SNIPPET='rc=0; run_alternating_chunks 50 50 || rc=$?; echo "rc=$rc"'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=3600 STUB_NO_LATEST=1
  [[ $output == *rc=1* ]]
  grep -q 'step 50' "$DRBM25_ROOT/FAILED"
}

@test "install_exit_traps: SIGHUP runs on_exit, which terminates the pod (I3)" {
  SNIPPET='install_exit_traps; kill -HUP $$'
  run -129 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key
  grep -q 'curl .*-X DELETE .*pods/pod123' "$STUB_LOG"
}

@test "install_exit_traps: SIGTERM and SIGINT also terminate the pod (I3)" {
  SNIPPET='install_exit_traps; kill -TERM $$'
  run -143 in_lib RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key
  grep -q 'curl .*-X DELETE' "$STUB_LOG"
}

@test "run_all.sh logs straight to a file and tells the user to use tmux (I3)" {
  ! grep -q 'tee' "$RUN_ALL"
  grep -q 'tmux' "${BATS_TEST_DIRNAME}/../../README.md"
}

@test "training runs under timeout, limited to the time to the deadline plus a margin (I4)" {
  SNIPPET='run_alternating_chunks 50 50'
  run -0 in_lib DEADLINE_UNIX=10000 RESERVE_S=600 DRBM25_NOW_UNIX=1000
  grep -q 'timeout --kill-after=[0-9]* 10800 uv run --no-sync torchrun' "$STUB_LOG"
}

@test "a hung training chunk (timeout status 124) becomes FAILED (I4)" {
  SNIPPET='rc=0; run_alternating_chunks 50 50 || rc=$?; echo "rc=$rc"'
  run -0 in_lib DEADLINE_UNIX=2000000000 RESERVE_S=600 STUB_TIMEOUT_RC=124
  [[ $output == *rc=1* ]]
  grep -qi 'timed out' "$DRBM25_ROOT/FAILED"
}

@test "smoke and eval commands also run under timeout and fail when it fires (I4)" {
  SNIPPET='smoke_stage'
  run -0 in_lib
  [[ $(grep -c '^timeout' "$STUB_LOG") == 2 ]]
  : >"$STUB_LOG"
  run -1 --separate-stderr in_lib STUB_TIMEOUT_RC=124
  [[ $stderr == *"timed out"* ]]
  : >"$STUB_LOG"
  SNIPPET='eval_stage'
  run -0 in_lib STUB_COMMON_STEP=50
  [[ $(grep -c '^timeout .*run_eval' "$STUB_LOG") == 4 ]]
  grep -q '^timeout .*eval.report' "$STUB_LOG"
}

@test "refuse_if_failed returns 1, names the marker and says the pod was not terminated" {
  echo "earlier failure" >"$DRBM25_ROOT/FAILED"
  SNIPPET='refuse_if_failed'
  run -1 --separate-stderr in_lib
  [[ $stderr == *FAILED* ]]
  [[ $stderr == *"NOT terminated"* ]]
  rm "$DRBM25_ROOT/FAILED"
  run -0 in_lib
}

@test "run_all.sh stops before prompting and does not terminate when FAILED exists" {
  echo "earlier failure" >"$DRBM25_ROOT/FAILED"
  run -1 --separate-stderr env DRBM25_ROOT="$DRBM25_ROOT" DRBM25_TTY="$DRBM25_TTY" RUNPOD_POD_ID=pod123 RUNPOD_API_KEY=rp_key bash "$RUN_ALL"
  [[ $stderr == *FAILED* ]]
  ! grep -q '^curl' "$STUB_LOG"
}
