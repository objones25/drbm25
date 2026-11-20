#!/usr/bin/env bash
#
# pod_bootstrap.sh: the start command of a pod created by scripts/runpod/launch.sh. launch.sh sends this
# file's text as the pod's command (entrypoint ["/bin/bash", "-c"]), so it runs before any clone exists and
# must not source anything from the repository.
#
# Order, chosen so every failure still reaches the terminate call:
#   1. save RunPod's pod-scoped RUNPOD_API_KEY, install the EXIT/signal traps;
#   2. open the log ($DRBM25_ROOT/logs/bootstrap-<time>.log, or $TMPDIR when the volume is not writable);
#   3. start the watchdog: an independent process that terminates the pod at DRBM25_HARD_DEADLINE_UNIX
#      (launch time + budget / price + 1 h, from launch.sh) whatever the run is doing;
#   4. start the image's services (SSH, Jupyter), fetch DRBM25_REF of DRBM25_REPO_URL into $DRBM25_ROOT/repo
#      (reusing an earlier clone), and run run_all.sh unattended (DRBM25_NONINTERACTIVE=1, stdin /dev/null).
# On exit (success, failure or signal), unless TERMINATE_ON_FINISH=no, it terminates the pod (run_all.sh has
# normally done so already). If that fails it hands over to the watchdog, which retries with backoff for about
# 35 minutes and then stops this process; the bootstrap then exits, ending the container, rather than waiting
# on the image's services forever.
#
# Terminate keys, in order: DRBM25_RUNPOD_API_KEY (the user's pods:write key, a RunPod secret) unless it is
# missing or an unreplaced {{ RUNPOD_SECRET_* }} reference, then the pod-scoped RUNPOD_API_KEY that RunPod
# injects (https://docs.runpod.io/pods/templates/environment-variables), kept as DRBM25_POD_RUNPOD_API_KEY.
#
# Environment (set by launch.sh in the pod body): DRBM25_REPO_URL, DRBM25_REF, DRBM25_RUNPOD_API_KEY,
# DRBM25_HARD_DEADLINE_UNIX, TERMINATE_ON_FINISH; RUNPOD_POD_ID and RUNPOD_API_KEY are set by RunPod.
# Test hooks: DRBM25_ROOT (default /workspace/drbm25), DRBM25_START_SH (default /start.sh), DRBM25_SLEEP
# (command used for terminate backoff).
# Requires: bash, git (installed with apt-get if missing), curl; setsid when present. Runs on the pod (Ubuntu 24.04).
# audit-bash disable-file=B403 # run as `bash -c "<text>"` by the pod, never sourced
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly ROOT=${DRBM25_ROOT:-/workspace/drbm25}
# The official runpod/pytorch image starts SSH and Jupyter from /start.sh; RunPod's "Build a custom Pod
# template" guide (https://docs.runpod.io/pods/templates/create-custom-template, "Option 2", FROM
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404) runs `/start.sh &` before its own application.
readonly START_SH=${DRBM25_START_SH:-/start.sh}
readonly FETCH_TIMEOUT_S=900
readonly BOOTSTRAP_TERMINATE_ROUNDS=3 # without a watchdog to hand over to
readonly WATCHDOG_TERMINATE_ROUNDS=8  # backoff 30, 60, ..., 600 s: about 35 minutes

LOGFILE=''
WATCHDOG_PID=''
SERVICES_PID=''
RUN_ALL_PID=''

log() {
  local now
  now=$(date '+%Y-%m-%dT%H:%M:%S')
  printf '%s pod_bootstrap: %s\n' "$now" "$*" >&2
}

# ---------------------------------------------------------------- terminate (also copied into the watchdog)

# DELETE https://api.runpod.io/v2/pods/{id} (https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod) with
# KEY read from stdin, so no key is in the process list.
_delete_pod_with() {
  curl --fail --silent --show-error --max-time 30 -X DELETE -H @- \
    "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" >/dev/null <<<"Authorization: Bearer $1"
}

# One round: the user's key, then the pod-scoped key.
terminate_once() {
  local user=${DRBM25_RUNPOD_API_KEY-} pod=${DRBM25_POD_RUNPOD_API_KEY-}
  if [[ ! ${RUNPOD_POD_ID-} =~ ^[A-Za-z0-9_-]+$ ]]; then
    log "ERROR: RUNPOD_POD_ID is missing or malformed: cannot terminate; do it at https://console.runpod.io/pods"
    return 1
  fi
  if [[ -n $user && $user != *RUNPOD_SECRET_* ]]; then
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    if _delete_pod_with "$user"; then
      log "terminated pod $RUNPOD_POD_ID"
      return 0
    fi
    log "terminating with the user's RunPod key failed"
  else
    log "the user's RunPod key is missing or an unreplaced secret reference"
  fi
  if [[ -n $pod ]]; then
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    if _delete_pod_with "$pod"; then
      log "terminated pod $RUNPOD_POD_ID (with the pod-scoped key)"
      return 0
    fi
    log "terminating with the pod-scoped key failed"
  fi
  return 1
}

# Up to ROUNDS rounds, waiting 30 s, 60 s, ... (at most 600 s) between them.
terminate_retry() {
  local rounds=$1 round delay=30
  for ((round = 1; round <= rounds; round++)); do
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    if terminate_once; then return 0; fi
    if ((round == rounds)); then break; fi
    log "terminate round $round of $rounds failed; next in ${delay}s"
    "${DRBM25_SLEEP:-sleep}" "$delay"
    delay=$((delay * 2))
    if ((delay > 600)); then delay=600; fi
  done
  return 1
}

# ---------------------------------------------------------------- watchdog (runs as its own process)

# watchdog_main DEADLINE_UNIX BOOTSTRAP_PID ROUNDS: sleeps until the deadline, or until SIGUSR1 (the bootstrap
# could not terminate), then terminates the pod in ROUNDS rounds. If that fails too, it sends SIGTERM to the
# bootstrap (PID 1) so the container exits. SIGTERM ends it quietly (the run finished normally).
watchdog_main() {
  local deadline=$1 parent=$2 rounds=$3 now polls
  WD_GO=0
  WD_SLEEP_PID=''
  trap 'WD_GO=1; if [[ -n $WD_SLEEP_PID ]]; then kill "$WD_SLEEP_PID" 2>/dev/null; fi' USR1
  trap 'if [[ -n $WD_SLEEP_PID ]]; then kill "$WD_SLEEP_PID" 2>/dev/null; fi; exit 0' TERM
  # At most one poll a minute, so 100000 polls outlast any deadline launch.sh computes.
  for ((polls = 0; polls < 100000; polls++)); do
    if ((WD_GO)); then break; fi
    now=$(date +%s)
    if ((now >= deadline)); then break; fi
    if ((deadline - now < 60)); then sleep $((deadline - now)) & else sleep 60 & fi
    WD_SLEEP_PID=$!
    wait "$WD_SLEEP_PID" 2>/dev/null || true
    WD_SLEEP_PID=''
  done
  if ((WD_GO)); then
    log "watchdog: the bootstrap could not terminate the pod; retrying for about 35 minutes"
  else
    log "watchdog: hard deadline $deadline reached; terminating the pod whatever the run is doing"
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if terminate_retry "$rounds"; then exit 0; fi
  log "watchdog: ERROR: could not terminate pod ${RUNPOD_POD_ID-}; stopping the bootstrap (pid $parent) so the container exits. Check https://console.runpod.io/pods"
  kill -TERM "$parent" 2>/dev/null || true
  exit 1
}

# The watchdog runs from a file holding copies of the functions above, under setsid (its own session) when
# available, so nothing run_all.sh does to its own process groups reaches it.
start_watchdog() {
  local deadline=${DRBM25_HARD_DEADLINE_UNIX-} file
  if [[ ! $deadline =~ ^[0-9]+$ ]]; then
    log "WARNING: DRBM25_HARD_DEADLINE_UNIX is not set: no watchdog; only run_all.sh's own budget stop applies"
    return 0
  fi
  if ! file=$(mktemp "${TMPDIR:-/tmp}/drbm25-watchdog.XXXXXX"); then
    log "WARNING: cannot write the watchdog script: no watchdog"
    return 0
  fi
  {
    printf 'set -u\n'
    declare -f log _delete_pod_with terminate_once terminate_retry watchdog_main
    printf 'watchdog_main "$@"\n'
  } >"$file"
  if command -v setsid >/dev/null 2>&1; then
    setsid bash "$file" "$deadline" "$$" "$WATCHDOG_TERMINATE_ROUNDS" </dev/null &
  else
    bash "$file" "$deadline" "$$" "$WATCHDOG_TERMINATE_ROUNDS" </dev/null &
  fi
  WATCHDOG_PID=$!
  log "watchdog started (pid $WATCHDOG_PID): terminates the pod at unix time $deadline"
}

stop_watchdog() {
  if [[ -z $WATCHDOG_PID ]]; then return 0; fi
  kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
  wait "$WATCHDOG_PID" 2>/dev/null || true
  WATCHDOG_PID=''
}

# ---------------------------------------------------------------- steps

open_log() {
  local stamp
  stamp=$(date '+%Y%m%dT%H%M%S')
  LOGFILE=$ROOT/logs/bootstrap-$stamp.log
  if ! { mkdir -p "$ROOT/logs" && : >>"$LOGFILE"; } 2>/dev/null; then
    LOGFILE=${TMPDIR:-/tmp}/drbm25-bootstrap-$stamp.log
    log "WARNING: $ROOT/logs is not writable (did the network volume mount at /workspace?); logging to $LOGFILE"
    printf '%s\n' "WARNING: $ROOT/logs is not writable (did the network volume mount at /workspace?)" >>"$LOGFILE"
  fi
  log "logging to $LOGFILE"
  exec >>"$LOGFILE" 2>&1
}

# Fetches DRBM25_REF (a branch, tag or full commit sha) with depth 1 and checks it out detached. An existing
# clone keeps its untracked and ignored files (.venv); tracked files are reset to the ref.
fetch_repo() {
  local repo=$ROOT/repo timeout_cmd=() sha
  if ! command -v git >/dev/null 2>&1; then
    log "git not found; installing it with apt-get"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git || return 1
  fi
  # audit-bash disable=B204 # runs on the pod (GNU coreutils); used only when present
  if command -v timeout >/dev/null 2>&1; then timeout_cmd=(timeout "$FETCH_TIMEOUT_S"); fi
  if [[ ! -d $repo/.git ]]; then git init -q "$repo" || return 1; fi
  git -C "$repo" remote remove origin 2>/dev/null || true
  git -C "$repo" remote add origin "$DRBM25_REPO_URL" || return 1
  ${timeout_cmd[@]+"${timeout_cmd[@]}"} git -C "$repo" fetch --quiet --depth 1 origin "$DRBM25_REF" || return 1
  git -C "$repo" checkout --quiet --force --detach FETCH_HEAD || return 1
  sha=$(git -C "$repo" rev-parse HEAD) || return 1
  log "checked out $DRBM25_REF ($sha)"
}

# EXIT trap. Never waits without a bound: on the image's services only when the user asked to keep the pod.
on_exit() {
  local status=$1
  set +e +u
  trap '' TERM INT HUP
  if [[ -n $RUN_ALL_PID ]] && kill -0 "$RUN_ALL_PID" 2>/dev/null; then
    log "stopping run_all.sh (pid $RUN_ALL_PID) so it records spend and copies logs"
    kill -TERM "$RUN_ALL_PID" 2>/dev/null
    wait "$RUN_ALL_PID" 2>/dev/null
  fi
  if [[ ${TERMINATE_ON_FINISH-} == no ]]; then
    stop_watchdog
    log "TERMINATE_ON_FINISH=no: leaving the pod running (it keeps billing); bootstrap status $status"
    if [[ -n $SERVICES_PID ]]; then wait "$SERVICES_PID"; fi
    return 0
  fi
  log "bootstrap ending with status $status; terminating the pod"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if terminate_once; then
    stop_watchdog
    return 0
  fi
  log "ERROR: could not terminate pod ${RUNPOD_POD_ID-} (log: $LOGFILE)"
  if [[ -n $WATCHDOG_PID ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill -USR1 "$WATCHDOG_PID" 2>/dev/null
    # Bounded: the watchdog gives up after WATCHDOG_TERMINATE_ROUNDS rounds.
    wait "$WATCHDOG_PID" 2>/dev/null
  else
    terminate_retry "$BOOTSTRAP_TERMINATE_ROUNDS"
  fi
  log "exiting so the container stops; if the pod still shows in https://console.runpod.io/pods, terminate it there"
  exit 1
}

main() {
  local pod_start_unix rc=0
  pod_start_unix=$(date +%s)
  # Saved before anything sets RUNPOD_API_KEY to the user's key for run_all.sh.
  export DRBM25_POD_RUNPOD_API_KEY=${DRBM25_POD_RUNPOD_API_KEY:-${RUNPOD_API_KEY-}}
  trap 'on_exit $?' EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'exit 129' HUP
  open_log
  start_watchdog
  if [[ -x $START_SH ]]; then
    "$START_SH" &
    SERVICES_PID=$!
  else
    log "WARNING: $START_SH not found: SSH and Jupyter are not started; follow the run with launch.sh --logs"
  fi

  if [[ -z ${DRBM25_REPO_URL-} || -z ${DRBM25_REF-} ]]; then
    log "error: DRBM25_REPO_URL and DRBM25_REF must be set (launch.sh sets them)"
    return 1
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! fetch_repo; then
    log "error: could not fetch $DRBM25_REF from $DRBM25_REPO_URL"
    return 1
  fi

  log "starting run_all.sh unattended"
  # The user's key (pods:write) comes in as DRBM25_RUNPOD_API_KEY so RunPod's pod-scoped RUNPOD_API_KEY cannot
  # shadow it; run_all.sh gets it as RUNPOD_API_KEY and the pod-scoped one as its fallback. Started in the
  # background and waited on, so a signal to this process interrupts the wait at once.
  # Exported, not passed as env arguments: an argument list is visible in ps.
  (
    cd "$ROOT/repo" || exit 1
    export RUNPOD_API_KEY="${DRBM25_RUNPOD_API_KEY-}" DRBM25_NONINTERACTIVE=1 DRBM25_POD_START_UNIX="$pod_start_unix"
    exec bash scripts/runpod/run_all.sh
  ) </dev/null &
  RUN_ALL_PID=$!
  wait "$RUN_ALL_PID" || rc=$?
  RUN_ALL_PID=''
  log "run_all.sh exited with status $rc"
  return "$rc"
}

main "$@"
