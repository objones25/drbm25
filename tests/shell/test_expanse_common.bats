#!/usr/bin/env bats
# Tests for scripts/expanse/common.sh. Run: bats tests/shell/
# common.sh is sourced in a child bash (`bash -c`), never into bats itself.

bats_require_minimum_version 1.5.0

setup() {
  COMMON="${BATS_TEST_DIRNAME}/../../scripts/expanse/common.sh"
  cd "$BATS_TEST_TMPDIR" || return 1
  # bash sources ~/.bashrc when stdin looks like a remote-shell socket; an empty HOME keeps tests deterministic.
  export HOME=$BATS_TEST_TMPDIR
}

# Run a snippet with common.sh sourced, in a clean environment.
in_common() {
  env -i PATH="$PATH" HOME="$HOME" "$@" bash -c 'source "$1"; shift; eval "$1"' _ "$COMMON" "$SNIPPET"
}

@test "require_env passes when every variable is set" {
  SNIPPET='require_env FOO BAR'
  run -0 in_common FOO=1 BAR=2
}

@test "require_env dies naming the missing variable" {
  SNIPPET='require_env FOO BAR'
  run -1 --separate-stderr in_common FOO=1
  [[ $stderr == *"BAR"* ]]
  [[ $stderr != *"FOO"* ]]
}

@test "require_env treats an empty variable as missing" {
  SNIPPET='require_env FOO'
  run -1 --separate-stderr in_common FOO=
  [[ $stderr == *"FOO"* ]]
}

@test "require_env with no names is a programmer error" {
  SNIPPET='require_env'
  run -2 --separate-stderr in_common
  [[ $stderr == *"at least one"* ]]
}

@test "drbm25_paths exports the Lustre layout" {
  SNIPPET='drbm25_paths; printf "%s\n" "$DRBM25_ROOT" "$HF_HOME" "$DRBM25_INDEX_DIR" "$DRBM25_DATA_DIR" "$DRBM25_RUNS_DIR" "$DRBM25_JDK_DIR"'
  run -0 in_common EXPANSE_ACCOUNT=abc123 USER=u
  [[ ${lines[0]} == /expanse/lustre/projects/abc123/u/drbm25 ]]
  [[ ${lines[1]} == /expanse/lustre/projects/abc123/u/drbm25/hf ]]
  [[ ${lines[2]} == /expanse/lustre/projects/abc123/u/drbm25/indexes ]]
  [[ ${lines[3]} == /expanse/lustre/projects/abc123/u/drbm25/data ]]
  [[ ${lines[4]} == /expanse/lustre/projects/abc123/u/drbm25/runs ]]
  [[ ${lines[5]} == /expanse/lustre/projects/abc123/u/drbm25/jdk ]]
}

@test "drbm25_paths fails without EXPANSE_ACCOUNT" {
  SNIPPET='drbm25_paths'
  run -1 --separate-stderr in_common USER=u
  [[ $stderr == *"EXPANSE_ACCOUNT"* ]]
}

@test "drbm25_paths honours an explicit DRBM25_ROOT" {
  SNIPPET='drbm25_paths; printf "%s\n" "$DRBM25_ROOT" "$HF_HOME"'
  run -0 in_common EXPANSE_ACCOUNT=abc123 USER=u DRBM25_ROOT=/x/y
  [[ ${lines[0]} == /x/y ]]
  [[ ${lines[1]} == /x/y/hf ]]
}

@test "drbm25_set_args builds the three path overrides" {
  SNIPPET='drbm25_paths; drbm25_set_args; printf "%s\n" "${DRBM25_SET_ARGS[@]}"'
  run -0 in_common EXPANSE_ACCOUNT=abc123 USER=u
  [[ ${lines[0]} == --set ]]
  [[ ${lines[1]} == retrieval.index_dir=/expanse/lustre/projects/abc123/u/drbm25/indexes ]]
  [[ ${lines[3]} == data.data_dir=/expanse/lustre/projects/abc123/u/drbm25/data ]]
  [[ ${lines[5]} == run.output_dir=/expanse/lustre/projects/abc123/u/drbm25/runs ]]
  [[ ${#lines[@]} -eq 6 ]]
}

@test "offline_env sets both variables by default" {
  SNIPPET='offline_env; printf "%s\n" "$HF_HUB_OFFLINE" "$WANDB_MODE"'
  run -0 in_common
  [[ ${lines[0]} == 1 ]]
  [[ ${lines[1]} == offline ]]
}

@test "offline_env with DRBM25_ONLINE=1 sets neither" {
  SNIPPET='offline_env; printf "[%s][%s]\n" "${HF_HUB_OFFLINE-unset}" "${WANDB_MODE-unset}"'
  run -0 in_common DRBM25_ONLINE=1
  [[ $output == "[unset][unset]" ]]
}

@test "offline_env with DRBM25_ONLINE=1 clears inherited offline values" {
  SNIPPET='offline_env; printf "[%s][%s]\n" "${HF_HUB_OFFLINE-unset}" "${WANDB_MODE-unset}"'
  run -0 in_common DRBM25_ONLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline
  [[ $output == "[unset][unset]" ]]
}

@test "offline_env rejects a DRBM25_ONLINE value other than 1" {
  SNIPPET='offline_env'
  run -1 --separate-stderr in_common DRBM25_ONLINE=yes
  [[ $stderr == *"DRBM25_ONLINE"* ]]
}

@test "latest_step reads the step number from the latest pointer" {
  mkdir -p run/step_000150
  printf 'step_000150\n' >run/latest
  SNIPPET='latest_step run'
  run -0 in_common
  [[ $output == 150 ]]
}

@test "latest_step prints 0 for a run without checkpoints" {
  mkdir -p run
  SNIPPET='latest_step run'
  run -0 in_common
  [[ $output == 0 ]]
}

@test "latest_step rejects a malformed pointer" {
  mkdir -p run
  printf 'garbage\n' >run/latest
  SNIPPET='latest_step run'
  run -1 --separate-stderr in_common
  [[ $stderr == *"malformed"* ]]
}

@test "run_exit_code reads the file and fails closed when missing or bad" {
  mkdir -p run
  printf '3\n' >run/exit_code
  SNIPPET='run_exit_code run'
  run -0 in_common
  [[ $output == 3 ]]
  rm run/exit_code
  run -1 --separate-stderr in_common
  [[ $stderr == *"exit_code"* ]]
  printf 'abc\n' >run/exit_code
  run -1 --separate-stderr in_common
  [[ $stderr == *"exit_code"* ]]
}

@test "run_is_complete needs exit_code 0 and latest step >= total" {
  mkdir -p run
  printf '0\n' >run/exit_code
  printf 'step_001751\n' >run/latest
  SNIPPET='run_is_complete run 1751'
  run -0 in_common
  printf 'step_001700\n' >run/latest
  run -1 in_common
  printf 'step_001751\n' >run/latest
  printf '3\n' >run/exit_code
  run -1 in_common
}

@test "run_is_complete is false for a run that never started" {
  SNIPPET='run_is_complete nonexistent 1751'
  run -1 in_common
}

@test "jdk_env puts the user-space JDK first on PATH when present" {
  mkdir -p root/jdk/current/bin
  # pwd -P: macOS's temp dir is a symlink, and the snippet's $PWD is the physical path.
  real_tmp=$(pwd -P)
  SNIPPET='DRBM25_JDK_DIR=$PWD/root/jdk; jdk_env; printf "%s\n" "$JAVA_HOME" "${PATH%%:*}"'
  run -0 in_common
  [[ ${lines[0]} == "$real_tmp/root/jdk/current" ]]
  [[ ${lines[1]} == "$real_tmp/root/jdk/current/bin" ]]
}

@test "jdk_env leaves the environment alone when no user-space JDK exists" {
  SNIPPET='DRBM25_JDK_DIR=$PWD/none; jdk_env; printf "[%s]\n" "${JAVA_HOME-unset}"'
  run -0 in_common
  [[ $output == "[unset]" ]]
}

@test "drbm25_paths points W&B offline runs at Lustre" {
  SNIPPET='drbm25_paths; printf "%s\n" "$WANDB_DIR"'
  run -0 in_common EXPANSE_ACCOUNT=abc123 USER=u
  [[ $output == /expanse/lustre/projects/abc123/u/drbm25/wandb ]]
}

@test "drbm25_paths puts the venv and uv cache on Lustre" {
  SNIPPET='drbm25_paths; printf "%s\n" "$UV_PROJECT_ENVIRONMENT" "$UV_CACHE_DIR"'
  run -0 in_common EXPANSE_ACCOUNT=abc123 USER=u
  [[ ${lines[0]} == /expanse/lustre/projects/abc123/u/drbm25/venv ]]
  [[ ${lines[1]} == /expanse/lustre/projects/abc123/u/drbm25/uv-cache ]]
}

# A fake `java` whose -version output (on stderr, like the real one) is $1.
fake_java() {
  mkdir -p fakebin
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" %q >&2\n' "$1" >fakebin/java
  chmod +x fakebin/java
}

@test "java_major reads the major version from java -version" {
  fake_java 'openjdk version "21.0.13" 2026-01-20 LTS'
  SNIPPET='PATH=$PWD/fakebin:$PATH; java_major'
  run -0 in_common
  [[ $output == 21 ]]
}

@test "java_major understands the legacy 1.8 version string" {
  fake_java 'java version "1.8.0_292"'
  SNIPPET='PATH=$PWD/fakebin:$PATH; java_major'
  run -0 in_common
  [[ $output == 8 ]]
}

@test "java_major fails when java is not installed" {
  SNIPPET='PATH=/nonexistent; java_major'
  run -1 --separate-stderr in_common
  [[ $stderr == *"java"* ]]
}

@test "require_java21 rejects Java 17 naming the version it found" {
  fake_java 'openjdk version "17.0.9" 2023-10-17'
  SNIPPET='PATH=$PWD/fakebin:$PATH; require_java21'
  run -1 --separate-stderr in_common
  [[ $stderr == *"17"* ]]
  [[ $stderr == *"21"* ]]
}

@test "require_java21 accepts Java 21" {
  fake_java 'openjdk version "21.0.13" 2026-01-20 LTS'
  SNIPPET='PATH=$PWD/fakebin:$PATH; require_java21'
  run -0 in_common
}

@test "config_total_steps reads run.total_steps from the real Expanse config" {
  SNIPPET="config_total_steps '${BATS_TEST_DIRNAME}/../../configs/expanse_v100_lora.yaml'"
  run -0 in_common
  [[ $output == 1751 ]]
}

@test "config_total_steps fails when the key is missing" {
  printf 'run:\n  seed: 1\n' >cfg.yaml
  SNIPPET='config_total_steps cfg.yaml'
  run -1 --separate-stderr in_common
  [[ $stderr == *"total_steps"* ]]
}

@test "verify_sha256 accepts a matching digest" {
  printf 'hello\n' >f.bin
  SNIPPET='verify_sha256 f.bin 5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03'
  run -0 in_common
}

@test "verify_sha256 rejects a mismatching digest and names the file" {
  printf 'hello\n' >f.bin
  SNIPPET='verify_sha256 f.bin 0000000000000000000000000000000000000000000000000000000000000000'
  run -1 --separate-stderr in_common
  [[ $stderr == *"f.bin"* ]]
  [[ $stderr == *"mismatch"* ]]
}

@test "verify_sha256 rejects a malformed expected digest" {
  printf 'hello\n' >f.bin
  SNIPPET='verify_sha256 f.bin notahash'
  run -1 --separate-stderr in_common
  [[ $stderr == *"64 hex"* ]]
}

@test "verify_sha256 fails on a missing file" {
  SNIPPET='verify_sha256 nope.bin 5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03'
  run -1 --separate-stderr in_common
  [[ $stderr == *"nope.bin"* ]]
}
