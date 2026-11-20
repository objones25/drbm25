#!/usr/bin/env bats
# Tests for submit_chain.sh and train.sbatch with sbatch/torchrun/java stubbed.

bats_require_minimum_version 1.5.0

setup() {
  REPO="${BATS_TEST_DIRNAME}/../.."
  cd "$BATS_TEST_TMPDIR" || return 1
  # bash sources ~/.bashrc when stdin looks like a remote-shell socket; an empty HOME keeps tests deterministic.
  export HOME=$BATS_TEST_TMPDIR
  mkdir -p bin root/venv/bin
  # sbatch stub: log the arguments, print an increasing job id like `sbatch --parsable`.
  cat >bin/sbatch <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$SBATCH_LOG"
n=$(wc -l <"$SBATCH_LOG")
printf '%s\n' "$((1000 + n))"
STUB
  # java stub for require_java21.
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" "openjdk version \\"21.0.13\\"" >&2\n' >bin/java
  # torchrun stub: records its argv, then writes the exit_code file like rank 0 would.
  cat >root/venv/bin/torchrun <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$TORCHRUN_LOG"
run_dir="$DRBM25_RUNS_DIR/${STUB_RUN:-normal}"
mkdir -p "$run_dir"
[[ -z ${STUB_CODE-} ]] || printf '%s\n' "$STUB_CODE" >"$run_dir/exit_code"
exit "${STUB_RC:-0}"
STUB
  chmod +x bin/sbatch bin/java root/venv/bin/torchrun
  export SBATCH_LOG="$PWD/sbatch.log" TORCHRUN_LOG="$PWD/torchrun.log"
  : >"$SBATCH_LOG"
  export PATH="$PWD/bin:$PATH"
  export EXPANSE_ACCOUNT=abc123 DRBM25_ROOT="$PWD/root" SLURM_SUBMIT_DIR="$REPO" SLURM_JOB_ID=77
}

# ---- submit_chain.sh ----

@test "submit_chain chains N jobs with afterany and prints their ids" {
  run -0 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal 3
  [[ ${lines[0]} == 1001 ]]
  [[ ${lines[1]} == 1002 ]]
  [[ ${lines[2]} == 1003 ]]
  mapfile -t calls <"$SBATCH_LOG"
  [[ ${#calls[@]} -eq 3 ]]
  [[ ${calls[0]} != *dependency* ]]
  [[ ${calls[1]} == *"--dependency=afterany:1001"* ]]
  [[ ${calls[2]} == *"--dependency=afterany:1002"* ]]
  [[ ${calls[0]} == *"--account=abc123"* ]]
  [[ ${calls[0]} == *"train.sbatch normal"* ]]
}

@test "submit_chain strips the cluster suffix of --parsable ids" {
  printf '#!/usr/bin/env bash\nprintf "42;expanse\\n"\n' >bin/sbatch
  run -0 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" count_once 1
  [[ $output == 42 ]]
}

@test "submit_chain rejects an unknown reward mode before submitting" {
  run -2 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" bogus 3
  [[ $stderr == *"reward mode"* ]]
  [[ ! -s $SBATCH_LOG ]]
}

@test "submit_chain rejects a non-numeric or out-of-range count" {
  run -2 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal abc
  run -2 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal 0
  run -2 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal 99
  [[ ! -s $SBATCH_LOG ]]
}

@test "submit_chain requires EXPANSE_ACCOUNT" {
  unset EXPANSE_ACCOUNT
  run -1 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal 1
  [[ $stderr == *"EXPANSE_ACCOUNT"* ]]
}

@test "submit_chain forwards extra args after -- to every job" {
  run -0 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal 2 -- --until-step 150
  mapfile -t calls <"$SBATCH_LOG"
  [[ ${calls[0]} == *"train.sbatch normal --until-step 150" ]]
  [[ ${calls[1]} == *"train.sbatch normal --until-step 150" ]]
}

# ---- train.sbatch ----

run_train() {
  run --separate-stderr bash "$REPO/scripts/expanse/train.sbatch" "$@"
}

@test "train.sbatch exits 0 when exit_code says 0" {
  STUB_CODE=0 run_train normal
  [[ $status -eq 0 ]]
  [[ $(<"$TORCHRUN_LOG") == *"--nproc_per_node 2"* ]]
  [[ $(<"$TORCHRUN_LOG") == *"--reward-mode normal"* ]]
  [[ $(<"$TORCHRUN_LOG") == *"--run-name normal"* ]]
  [[ $(<"$TORCHRUN_LOG") == *"--set run.output_dir=$DRBM25_ROOT/runs"* ]]
}

@test "train.sbatch treats a budget stop (3) as an error with a clear message" {
  STUB_CODE=3 STUB_RC=1 run_train normal
  [[ $status -eq 1 ]]
  [[ $stderr == *"budget"* ]]
}

@test "train.sbatch fails when exit_code says 1, even if torchrun returned 0" {
  STUB_CODE=1 STUB_RC=0 run_train normal
  [[ $status -eq 1 ]]
}

@test "train.sbatch fails when torchrun never wrote exit_code" {
  STUB_RC=1 run_train normal
  [[ $status -eq 1 ]]
  [[ $stderr == *"exit_code"* ]]
}

@test "train.sbatch no-ops without launching torchrun when the run is complete" {
  mkdir -p root/runs/normal
  printf '0\n' >root/runs/normal/exit_code
  printf 'step_001751\n' >root/runs/normal/latest
  run_train normal
  [[ $status -eq 0 ]]
  [[ $stderr == *"already complete"* ]]
  [[ ! -e $TORCHRUN_LOG ]]
}

@test "train.sbatch rejects an unknown reward mode" {
  run_train bogus
  [[ $status -eq 2 ]]
  [[ ! -e $TORCHRUN_LOG ]]
}

@test "train.sbatch forwards extra arguments to drbm25.train" {
  STUB_CODE=0 run_train normal --until-step 150 --set ppo.actor_lr=1e-5
  [[ $status -eq 0 ]]
  [[ $(<"$TORCHRUN_LOG") == *"--until-step 150 --set ppo.actor_lr=1e-5"* ]]
}

@test "train.sbatch fails on Java 17" {
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" "openjdk version \\"17.0.1\\"" >&2\n' >bin/java
  STUB_CODE=0 run_train normal
  [[ $status -eq 1 ]]
  [[ $stderr == *"21"* ]]
  [[ ! -e $TORCHRUN_LOG ]]
}

@test "train.sbatch sets offline mode unless DRBM25_ONLINE=1" {
  printf '#!/usr/bin/env bash\nprintf "%%s %%s\\n" "${HF_HUB_OFFLINE-unset}" "${WANDB_MODE-unset}" >>"$TORCHRUN_LOG"\nmkdir -p "$DRBM25_RUNS_DIR/normal"; echo 0 >"$DRBM25_RUNS_DIR/normal/exit_code"\n' >root/venv/bin/torchrun
  run_train normal
  [[ $(<"$TORCHRUN_LOG") == "1 offline" ]]
  : >"$TORCHRUN_LOG"
  DRBM25_ONLINE=1 run_train normal
  [[ $(<"$TORCHRUN_LOG") == "unset unset" ]]
}

# ---- eval.sbatch ----

setup_eval_python() {
  # python stub: `-c` prints the common step; `-m drbm25.eval.*` records its arguments.
  cat >bin/fakepython <<'STUB'
#!/usr/bin/env bash
if [[ $1 == -c ]]; then
  if [[ -n ${STUB_COMMON_FAIL-} ]]; then echo "no common step" >&2; exit 1; fi
  printf '%s\n' "${STUB_COMMON_STEP:-1751}"
else
  printf '%s\n' "$*" >>"$PYTHON_LOG"
fi
STUB
  chmod +x bin/fakepython
  export DRBM25_PYTHON="$PWD/bin/fakepython" PYTHON_LOG="$PWD/python.log"
  : >"$PYTHON_LOG"
}

@test "eval.sbatch scores all four sources at the common step and builds the report" {
  setup_eval_python
  run --separate-stderr bash "$REPO/scripts/expanse/eval.sbatch"
  [[ $status -eq 0 ]]
  log=$(<"$PYTHON_LOG")
  [[ $log == *"--source original --out $DRBM25_ROOT/runs/eval/original.jsonl"* ]]
  [[ $log == *"--source untrained --out $DRBM25_ROOT/runs/eval/untrained.jsonl"* ]]
  [[ $log == *"--source checkpoint --out $DRBM25_ROOT/runs/eval/trained-normal.jsonl"*"--checkpoint $DRBM25_ROOT/runs/normal/step_001751"* ]]
  [[ $log == *"--checkpoint $DRBM25_ROOT/runs/count_once/step_001751"* ]]
  [[ $log == *"drbm25.eval.report --inputs"*"--out $DRBM25_ROOT/runs/eval/report.md"* ]]
}

@test "eval.sbatch uses the step common_step returns, zero-padded" {
  setup_eval_python
  STUB_COMMON_STEP=150 run --separate-stderr bash "$REPO/scripts/expanse/eval.sbatch"
  [[ $status -eq 0 ]]
  [[ $(<"$PYTHON_LOG") == *"runs/normal/step_000150"* ]]
  [[ $stderr == *"warning"* ]]
}

@test "eval.sbatch stops before scoring when the runs share no checkpoint" {
  setup_eval_python
  STUB_COMMON_FAIL=1 run --separate-stderr bash "$REPO/scripts/expanse/eval.sbatch"
  [[ $status -ne 0 ]]
  [[ ! -s $PYTHON_LOG ]]
}

# ---- prepare_login.sh / probe.sbatch ----

@test "prepare_login --help documents the steps and exits 0" {
  run -0 "$REPO/scripts/expanse/prepare_login.sh" --help
  [[ $output == *"JDK 21"* ]]
}

@test "prepare_login rejects an unknown argument with exit 2" {
  run -2 --separate-stderr "$REPO/scripts/expanse/prepare_login.sh" --bogus
  [[ $stderr == *"--bogus"* ]]
}

@test "prepare_login requires EXPANSE_ACCOUNT" {
  unset EXPANSE_ACCOUNT
  run -1 --separate-stderr "$REPO/scripts/expanse/prepare_login.sh"
  [[ $stderr == *"EXPANSE_ACCOUNT"* ]]
}

@test "prepare_login pins an https JDK URL and a 64-hex SHA-256" {
  run -0 grep -E "^readonly JDK_SHA256='[0-9a-f]{64}'$" "$REPO/scripts/expanse/prepare_login.sh"
  run -0 grep -E "^readonly JDK_URL='https://" "$REPO/scripts/expanse/prepare_login.sh"
}

@test "probe.sbatch requires EXPANSE_ACCOUNT" {
  unset EXPANSE_ACCOUNT
  run -1 --separate-stderr bash "$REPO/scripts/expanse/probe.sbatch"
  [[ $stderr == *"EXPANSE_ACCOUNT"* ]]
}

@test "submit_chain reads a zero-padded count as decimal, not octal" {
  run -0 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal 08
  [[ ${#lines[@]} -eq 8 ]]
}

@test "submit_chain rejects a count with too many digits" {
  run -2 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal 99999999999999999999
}

@test "train.sbatch honours --run-name in the extra args for the run dir and the exit_code lookup" {
  STUB_RUN=pilot_normal STUB_CODE=0 run_train normal --run-name pilot_normal --until-step 150
  [[ $status -eq 0 ]]
  log=$(<"$TORCHRUN_LOG")
  [[ $log == *"--run-name pilot_normal"* ]]
  [[ $log != *"--run-name normal"* ]]
  [[ $log == *"--until-step 150"* ]]
}

@test "train.sbatch rejects --run-name without a value" {
  run_train normal --run-name
  [[ $status -eq 2 ]]
}

@test "train.sbatch rejects a --run-name that could escape the runs directory" {
  run_train normal --run-name ../evil
  [[ $status -eq 2 ]]
  [[ ! -e $TORCHRUN_LOG ]]
}

# ---- run.sh (login-shell-safe wrapper) ----

@test "run.sh fails with a message naming EXPANSE_ACCOUNT, and never runs the command" {
  unset EXPANSE_ACCOUNT
  run -1 --separate-stderr "$REPO/scripts/expanse/run.sh" touch ran.flag
  [[ $stderr == *"EXPANSE_ACCOUNT"* ]]
  [[ ! -e ran.flag ]]
}

@test "run.sh without a command is a usage error" {
  run -2 --separate-stderr "$REPO/scripts/expanse/run.sh"
  [[ $stderr == *"COMMAND"* ]]
}

@test "run.sh --help exits 0" {
  run -0 "$REPO/scripts/expanse/run.sh" --help
  [[ $output == *"Usage:"* ]]
}

@test "run.sh runs the command with the Lustre paths and an online (login-node) environment" {
  HF_HUB_OFFLINE=1 WANDB_MODE=offline run -0 "$REPO/scripts/expanse/run.sh" \
    bash -c 'printf "%s\n" "$WANDB_DIR" "$HF_HOME" "${HF_HUB_OFFLINE-unset}" "${WANDB_MODE-unset}"'
  [[ ${lines[0]} == "$DRBM25_ROOT/wandb" ]]
  [[ ${lines[1]} == "$DRBM25_ROOT/hf" ]]
  [[ ${lines[2]} == unset ]]
  [[ ${lines[3]} == unset ]]
}

@test "run.sh --offline sets the offline variables" {
  run -0 "$REPO/scripts/expanse/run.sh" --offline bash -c 'printf "%s %s\n" "$HF_HUB_OFFLINE" "$WANDB_MODE"'
  [[ $output == "1 offline" ]]
}

@test "run.sh passes the command's exit status and arguments through unchanged" {
  run -7 "$REPO/scripts/expanse/run.sh" bash -c 'exit 7'
  run -0 "$REPO/scripts/expanse/run.sh" bash -c 'printf "[%s]" "$1"' _ 'a b'
  [[ $output == "[a b]" ]]
}

@test "run.sh puts a user-space JDK on PATH when one is installed" {
  mkdir -p root/jdk/current/bin
  run -0 "$REPO/scripts/expanse/run.sh" bash -c 'printf "%s\n" "$JAVA_HOME"'
  [[ $output == "$DRBM25_ROOT/jdk/current" ]]
}

# ---- train.sbatch failure guard ----

@test "train.sbatch writes FAILED with the job id and reason when exit_code says 2" {
  STUB_CODE=2 STUB_RC=1 run_train normal
  [[ $status -eq 1 ]]
  [[ -f $DRBM25_ROOT/runs/normal/FAILED ]]
  failed=$(<"$DRBM25_ROOT/runs/normal/FAILED")
  [[ $failed == *"job 77"* ]]
  [[ $failed == *"exit_code 2"* ]]
}

@test "train.sbatch writes FAILED on a budget stop (3), which Expanse never expects" {
  STUB_CODE=3 STUB_RC=1 run_train normal
  [[ $status -eq 1 ]]
  [[ $(<"$DRBM25_ROOT/runs/normal/FAILED") == *"exit_code 3"* ]]
}

@test "train.sbatch writes FAILED when no exit_code file exists after torchrun" {
  STUB_RC=1 run_train normal
  [[ $status -eq 1 ]]
  [[ -f $DRBM25_ROOT/runs/normal/FAILED ]]
}

@test "train.sbatch leaves no FAILED marker for exit_code 1 (killed), so the chain resumes" {
  STUB_CODE=1 STUB_RC=1 run_train normal
  [[ $status -eq 1 ]]
  [[ ! -e $DRBM25_ROOT/runs/normal/FAILED ]]
}

@test "train.sbatch exits non-zero at once when FAILED exists, telling the user how to retry" {
  mkdir -p root/runs/normal
  printf 'job 5: exit_code 2\n' >root/runs/normal/FAILED
  run_train normal
  [[ $status -eq 1 ]]
  [[ $stderr == *"FAILED"* ]]
  [[ $stderr == *"delete"* ]]
  [[ $stderr == *"job 5: exit_code 2"* ]]
  [[ ! -e $TORCHRUN_LOG ]]
}

@test "train.sbatch FAILED guard is per run name, so a pilot is not blocked by the real run" {
  mkdir -p root/runs/normal
  printf 'x\n' >root/runs/normal/FAILED
  STUB_RUN=pilot STUB_CODE=0 run_train normal --run-name pilot
  [[ $status -eq 0 ]]
}

@test "submit_chain exports the repo root so jobs do not depend on where sbatch was run from" {
  run -0 --separate-stderr "$REPO/scripts/expanse/submit_chain.sh" normal 1
  [[ $(<"$SBATCH_LOG") == *"--export=ALL,DRBM25_REPO_ROOT="*"/deep-retrieval-CS279"* ]]
}

@test "train.sbatch prefers DRBM25_REPO_ROOT over SLURM_SUBMIT_DIR" {
  STUB_CODE=0 SLURM_SUBMIT_DIR=/nonexistent DRBM25_REPO_ROOT="$(cd "$REPO" && pwd)" run_train normal
  [[ $status -eq 0 ]]
}
