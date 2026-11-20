#!/usr/bin/env bash
#
# pod_bootstrap.sh: the start command of a pod created by scripts/runpod/launch.sh. launch.sh sends this
# file's text as the pod's command (entrypoint ["/bin/bash", "-c"]), so it runs before any clone exists and
# must not source anything from the repository.
#
# It starts the image's own services (SSH, Jupyter) in the background, fetches DRBM25_REF of DRBM25_REPO_URL
# into $DRBM25_ROOT/repo (reusing an earlier clone on the network volume), then runs run_all.sh unattended:
# DRBM25_NONINTERACTIVE=1, stdin from /dev/null, output to $DRBM25_ROOT/logs/bootstrap-<time>.log.
# run_all.sh terminates the pod when it ends. If the fetch fails, this script terminates the pod itself,
# so a broken launch does not sit idle and billing (unless TERMINATE_ON_FINISH=no).
#
# Environment (set by launch.sh in the pod body): DRBM25_REPO_URL, DRBM25_REF, DRBM25_RUNPOD_API_KEY (exported
# to run_all.sh as RUNPOD_API_KEY), TERMINATE_ON_FINISH; RUNPOD_POD_ID is set by RunPod.
# Test hooks: DRBM25_ROOT (default /workspace/drbm25), DRBM25_START_SH (default /start.sh).
# Requires: bash, git (installed with apt-get if missing), curl. Runs on the pod (Ubuntu 24.04).
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
readonly TERMINATE_ATTEMPTS=3

log() {
  local now
  now=$(date '+%Y-%m-%dT%H:%M:%S')
  printf '%s pod_bootstrap: %s\n' "$now" "$*" >&2
}

# DELETE https://api.runpod.io/v2/pods/{id} (https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod),
# key from stdin so it is not in the process list. Same call as terminate_pod in lib.sh, which is not
# available here when the clone failed.
terminate_self() {
  local attempt
  if [[ ${TERMINATE_ON_FINISH-} == no ]]; then
    log "TERMINATE_ON_FINISH=no: leaving the pod running (it keeps billing)"
    return 0
  fi
  if [[ ! ${RUNPOD_POD_ID-} =~ ^[A-Za-z0-9_-]+$ || -z ${DRBM25_RUNPOD_API_KEY-} ]]; then
    log "ERROR: cannot terminate this pod (RUNPOD_POD_ID or the API key is missing); terminate it at https://console.runpod.io/pods"
    return 1
  fi
  for ((attempt = 1; attempt <= TERMINATE_ATTEMPTS; attempt++)); do
    if curl --fail --silent --show-error --max-time 30 -X DELETE -H @- \
      "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" >/dev/null <<<"Authorization: Bearer $DRBM25_RUNPOD_API_KEY"; then
      log "terminated pod $RUNPOD_POD_ID"
      return 0
    fi
    sleep $((attempt * 5))
  done
  log "ERROR: could not terminate pod $RUNPOD_POD_ID; terminate it at https://console.runpod.io/pods"
  return 1
}

# Fetches DRBM25_REF (a branch, tag or full commit sha) with depth 1 and checks it out detached. An existing
# clone keeps its untracked and ignored files (.venv); tracked files are reset to the ref.
fetch_repo() {
  local repo=$ROOT/repo timeout_cmd=()
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
  local sha
  sha=$(git -C "$repo" rev-parse HEAD) || return 1
  log "checked out $DRBM25_REF ($sha)"
}

main() {
  local pod_start_unix logfile services_pid='' rc=0
  pod_start_unix=$(date +%s)
  if [[ -x $START_SH ]]; then
    "$START_SH" &
    services_pid=$!
  else
    log "WARNING: $START_SH not found: SSH and Jupyter are not started; follow the run with launch.sh --logs"
  fi
  mkdir -p "$ROOT/logs"
  logfile=$ROOT/logs/bootstrap-$(date '+%Y%m%dT%H%M%S').log
  log "logging to $logfile"
  exec >>"$logfile" 2>&1

  if [[ -z ${DRBM25_REPO_URL-} || -z ${DRBM25_REF-} ]]; then
    log "error: DRBM25_REPO_URL and DRBM25_REF must be set (launch.sh sets them)"
    # shellcheck disable=SC2310 # best effort: the bootstrap fails either way
    terminate_self || true
    return 1
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! fetch_repo; then
    log "error: could not fetch $DRBM25_REF from $DRBM25_REPO_URL"
    # shellcheck disable=SC2310 # best effort: the bootstrap fails either way
    terminate_self || true
    return 1
  fi

  log "starting run_all.sh unattended"
  # RunPod also sets a pod-scoped RUNPOD_API_KEY; the user's key (pods:write) comes in under its own name so
  # it cannot be shadowed, and is what run_all.sh's terminate call uses.
  (
    cd "$ROOT/repo" &&
      RUNPOD_API_KEY=${DRBM25_RUNPOD_API_KEY-} DRBM25_NONINTERACTIVE=1 DRBM25_POD_START_UNIX=$pod_start_unix \
        bash scripts/runpod/run_all.sh </dev/null
  ) || rc=$?
  log "run_all.sh exited with status $rc"
  # run_all.sh has normally terminated the pod by now. If not (TERMINATE_ON_FINISH=no, or the terminate call
  # failed), keep the image's services up so the pod can still be reached.
  if [[ -n $services_pid ]]; then wait "$services_pid" || true; fi
  return "$rc"
}

main "$@"
