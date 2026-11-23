#!/usr/bin/env bash
#
# launch.sh: start the two-run experiment on a new RunPod pod from your own machine, unattended.
#
# Run it from a checkout of this repository. It finds the cheapest 2x A100 80GB that is in stock, shows the
# price and asks for `yes`, stores your keys as RunPod secrets, reuses (or creates) the network volume
# `drbm25`, and creates a pod that clones the repository and runs scripts/runpod/run_all.sh on its own
# (scripts/runpod/pod_bootstrap.sh is its start command). Then it exits: your machine can disconnect.
#
# Usage: see usage() below, or run with --help.
# Requires: bash 3.2+ (macOS /bin/bash is fine), curl 7.55+, jq 1.6+ (macOS 15 ships /usr/bin/jq), git.
# bash-target: 3.2
#
# RunPod REST API v2 (base https://api.runpod.io, bearer key; https://docs.runpod.io/api-reference-v2/overview),
# endpoints used, each checked against its reference page on 2026-11-20:
#   GET    /v2/catalog/gpus     https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types
#   GET    /v2/catalog/datacenters  https://docs.runpod.io/api-reference-v2/catalog/list-data-centers
#   GET    /v2/network-volumes  https://docs.runpod.io/api-reference-v2/network-volumes/list-network-volumes
#   POST   /v2/network-volumes  https://docs.runpod.io/api-reference-v2/network-volumes/create-a-network-volume
#   GET    /v2/account/secrets  https://docs.runpod.io/api-reference-v2/account/list-secrets
#   POST   /v2/account/secrets  https://docs.runpod.io/api-reference-v2/account/create-a-secret
#   PATCH  /v2/account/secrets/{id}  https://docs.runpod.io/api-reference-v2/account/update-a-secret
#   GET    /v2/pods             https://docs.runpod.io/api-reference-v2/pods/list-pods
#   POST   /v2/pods             https://docs.runpod.io/api-reference-v2/pods/create-a-pod
#   POST   /v2/pods/{id}/exec   https://docs.runpod.io/api-reference-v2/pods/run-a-command-in-a-pod
#   DELETE /v2/pods/{id}        https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod
# Hugging Face Hub API (base https://huggingface.co, bearer HF_TOKEN), the endpoints huggingface_hub 0.36.2 calls
# (read from its installed source on 2026-11-20):
#   GET /api/whoami-v2                       HfApi.whoami; the token role is .auth.accessToken.role
#   GET /{repo_id}/resolve/{rev}/{filename}  hf_hub_download (constants.HUGGINGFACE_CO_URL_TEMPLATE); a small
#                                            JSON file is served directly (no LFS redirect is followed here)
# Test hooks (environment): DRBM25_TTY (file read instead of /dev/tty), DRBM25_SLEEP (command used to wait).

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly REPO_ROOT=$SCRIPT_DIR/../..

readonly API_BASE=https://api.runpod.io
readonly HF_API_BASE=https://huggingface.co
# Repo names drbm25.publish creates: <user>/<prefix>-<model slug>-<reward mode> (DEFAULT_REPO_PREFIX there).
readonly HF_REPO_PREFIX=drbm25
# A token may write the user's model repos when its role is write, or when it is fine-grained and one of its
# scoped entries names the user with repo.write (the whoami-v2 shape from secondary sources; drbm25.publish
# check_write_access applies the same rule). Prints ok:<user> or error:<reason>.
# shellcheck disable=SC2016 # jq program, not shell
readonly HF_WRITE_JQ='(.name // "") as $n | (.auth.accessToken.role // "") as $r |
  if $n == "" then "error:HF_TOKEN: whoami returned no user name"
  elif $r == "write" then "ok:" + $n
  elif $r == "read" then "error:HF_TOKEN is read-only; the pod could not upload the models. Create a token with the write role at https://huggingface.co/settings/tokens"
  elif $r == "fineGrained" then
    (if any(.auth.accessToken.fineGrained.scoped[]?; .entity.name == $n and ((.permissions // []) | index("repo.write")) != null)
     then "ok:" + $n
     else "error:HF_TOKEN is a fine-grained token without repo.write on user " + $n + "; grant it or use a write token" end)
  else "error:HF_TOKEN: cannot tell whether this token may write (role \($r)); use a write token" end'
readonly DEFAULT_REPO_URL=https://github.com/objones25/drbm25
readonly POD_NAME=drbm25
readonly VOLUME_NAME=drbm25
readonly DEFAULT_VOLUME_GB=400
# The official "Runpod Pytorch 2.8.0" image (CUDA 12.8.1), the example image of the create-a-pod reference.
readonly IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
readonly MIN_CUDA=12.8 # the image's CUDA; gpu.minCudaVersion on create, minCudaVersion on the catalog
# GPU type ids from https://docs.runpod.io/references/gpu-types (as of 2026-11-20).
readonly GPU_ID_SXM='NVIDIA A100-SXM4-80GB'
readonly GPU_ID_PCIE='NVIDIA A100 80GB PCIe'
# 96 GB Blackwell, opt-in only (--gpu pro6000): untested with this code, so `any` never picks it; the smoke stage
# (gpu tests + memory check) is what vets it on the pod.
readonly GPU_ID_PRO6000='NVIDIA RTX PRO 6000 Blackwell Server Edition'
readonly GPU_COUNT=2
# Container disk (ephemeral): the image, apt and pip/compile caches. Venv, uv cache, models and runs are on
# the volume. An estimate, not measured on a pod.
readonly CONTAINER_DISK_GB=80
# Standard network volume storage, first 1 TB: https://docs.runpod.io/storage/network-volumes (as of
# 2026-11-20). The volume is created with type STANDARD so this is the rate that applies.
readonly VOLUME_USD_PER_GB_MONTH=0.07
# Secret names may not start with "RUNPOD" (create-a-secret reference); pods read them as
# {{ RUNPOD_SECRET_<name> }} (https://docs.runpod.io/pods/templates/secrets).
readonly SECRET_HF=drbm25_hf_token
readonly SECRET_WANDB=drbm25_wandb_api_key
readonly SECRET_RUNPOD=drbm25_runpod_api_key
readonly API_ATTEMPTS=4
readonly API_BACKOFF_S=(2 5 15) # waits after attempts 1-3 for a 5xx or network error
readonly RETRY_AFTER_MAX_S=60
readonly CURL_MAX_TIME_S=60
readonly LOG_LINES=40
readonly WATCH_LOG_LINES=10
readonly WATCH_MAX_POLLS=2000 # 2000 x 10 min is about 14 days, longer than any budget buys
readonly CONSOLE_URL=https://console.runpod.io/pods
# The pod's watchdog terminates it at launch + budget / price + this margin, whatever the run is doing.
# Evaluation, report and uploads are budgeted inside run_all.sh's RESERVE_S (5400 s) before the budget deadline, so this
# margin is slack for an overrun; keep it >= lib.sh PUBLISH_TIMEOUT_S (a bats test checks this).
readonly HARD_DEADLINE_MARGIN_S=3600
# Run inside the pod by --status: per run "<mode> <sha256 of best/meta.json, or none> <stale|ok>" (stale: the run
# has MARKER_STALE, its Hub marker could not be withdrawn before training).
# shellcheck disable=SC2016 # expanded by the pod's shell, not here
readonly VOLUME_SHA_SCRIPT='cd /workspace/drbm25/runs 2>/dev/null || exit 0
for m in normal count_once; do
  s=ok; [ -f "$m/MARKER_STALE" ] && s=stale
  if [ -f "$m/best/meta.json" ]; then echo "$m $(sha256sum "$m/best/meta.json" | cut -d " " -f 1) $s"; else echo "$m none $s"; fi
done'
# Run inside the pod by --logs and --watch (sh -c, fixed text; the line count comes in through env).
# shellcheck disable=SC2016 # expanded by the pod's shell, not here
readonly LOG_SCRIPT='cd /workspace/drbm25 2>/dev/null || { echo "no /workspace/drbm25 yet (the pod is still starting)"; exit 0; }
if [ -f FAILED ]; then echo "FAILED marker: $(cat FAILED)"; fi
if [ -f PUBLISH_FAILED ]; then echo "PUBLISH_FAILED marker: $(cat PUBLISH_FAILED)"; fi
for m in normal count_once; do if [ -f "runs/$m/UPLOADED.json" ]; then echo "runs/$m/UPLOADED.json: uploaded and verified"; fi; if [ -f "runs/$m/MARKER_STALE" ]; then echo "runs/$m/MARKER_STALE: $(cat "runs/$m/MARKER_STALE")"; fi; done
if [ -f spend.json ]; then echo "spend so far: $(cat spend.json)"; fi
for f in "$(ls -t logs/bootstrap-*.log 2>/dev/null | head -n 1)" "$(ls -t logs/run_all-*.log 2>/dev/null | head -n 1)"; do
  [ -n "$f" ] || continue
  echo "== /workspace/drbm25/$f"
  tail -n "$DRBM25_TAIL_LINES" "$f"
done'

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [launch options]
       $SCRIPT_NAME --status | --logs | --watch [--interval-min N] | --stop   [--pod ID]

Launch (the default): find the cheapest 2x A100 80GB in stock on RunPod's secure cloud, print the hourly price
and data center, ask for 'yes', then store HF_TOKEN, WANDB_API_KEY and RUNPOD_API_KEY as RunPod secrets,
reuse the network volume '$VOLUME_NAME' (or create one in the chosen data center) and create a pod that
clones the repository and runs scripts/runpod/run_all.sh unattended. The pod terminates itself when the run
ends or fails; the volume and its data are kept. Run this again to resume from the latest checkpoint.

Asked for (hidden for keys) unless set in the environment: RUNPOD_API_KEY (needs pods:write and access to
network volumes, secrets and the catalog), HF_TOKEN, WANDB_API_KEY, WANDB_PROJECT (default drbm25),
WANDB_ENTITY (optional), BUDGET_USD (the dollars still left to spend; earlier spend is in the pod log).
HF_TOKEN must be allowed to write model repos (checked before anything is created): when the run ends the pod
uploads both runs' best policy, verified, to private repos $HF_REPO_PREFIX-<model>-<reward mode> under your account.

Launch options:
  --yes              do not ask for confirmation (for scripts); the price is still printed
  --gpu sxm|pcie|any|pro6000   A100 80GB variant (default any: the cheapest A100 in stock), or pro6000 for
                     2x RTX PRO 6000 96GB (opt-in; never chosen by any)
  --cloud secure|community|any   only secure works: network volumes need the Secure Cloud (default secure)
  --volume-gb N      size of a NEW volume in GB (default $DEFAULT_VOLUME_GB); an existing one is reused as is
  --ref REF          branch, tag or full commit sha to run (default: this checkout's HEAD, which must be pushed)
  --repo-url URL     https URL of the public repository (default $DEFAULT_REPO_URL)
  --clear-failed     the pod moves an earlier FAILED marker into logs/ and runs (after you fixed the cause)

Other modes:
  --status           print the $POD_NAME pod's id, status, price and data center, and whether each run's
                     Hugging Face repo holds UPLOADED.json for the best/ on the volume (yes, no, or unknown without a pod)
  --logs             print the FAILED marker, spend and the last $LOG_LINES lines of the newest logs (via the exec API)
  --watch            poll status and logs every --interval-min minutes (default 10) until the pod is gone
  --stop             terminate the pod after you type 'yes' (the volume and its data are kept);
                     --stop --yes terminates without asking
  --pod ID           act on this pod id instead of the one named $POD_NAME
  -h, --help         show this help

Exit status: 0 success, 1 failure (nothing billable is created unless it says so), 2 usage error.
EOF
}

usage_error() {
  printf '%s: %s (see --help)\n' "$SCRIPT_NAME" "$1" >&2
  exit 2
}

say() {
  printf '%s\n' "$*" >&2
}

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

cleanup() {
  if [[ -n ${TMP_DIR-} ]]; then rm -rf -- "${TMP_DIR:?}" || true; fi
}

# ---------------------------------------------------------------- API

_retry_after() {
  local value
  value=$(awk 'tolower($1) == "retry-after:" { gsub(/\r/, "", $2); print $2; exit }' "$TMP_DIR/headers" 2>/dev/null) || value=''
  if [[ ! $value =~ ^[0-9]+$ ]]; then value=${API_BACKOFF_S[0]}; fi
  if ((value > RETRY_AFTER_MAX_S)); then value=$RETRY_AFTER_MAX_S; fi
  printf '%s\n' "$value"
}

# api OUT METHOD PATH [BODY_FILE]: one RunPod API call. The response body goes to OUT, the HTTP status to
# API_STATUS (000: no response). Returns 0 on 2xx. The key is read from AUTH_FILE (-H @file, curl 7.55+) and
# the body from a file, so neither is in curl's argv. A 429 is retried after Retry-After (docs: "Rate
# limits"); a 5xx or network error is retried for GET, PATCH and DELETE, never for POST, which may already
# have created the resource.
# API_BASE_URL and API_AUTH_FILE (set as locals by hf_api) point the same logic at the Hugging Face API.
api() {
  local out=$1 method=$2 path=$3 body=${4-} attempt status wait_s
  local base=${API_BASE_URL:-$API_BASE} auth=${API_AUTH_FILE:-$AUTH_FILE}
  local -a args
  for ((attempt = 1; attempt <= API_ATTEMPTS; attempt++)); do
    args=(--silent --show-error --proto '=https' --tlsv1.2 --connect-timeout 15 --max-time "$CURL_MAX_TIME_S"
      -X "$method" -H "@$auth" -H 'Accept: application/json' -o "$out" -D "$TMP_DIR/headers" -w '%{http_code}')
    if [[ -n $body ]]; then args+=(-H 'Content-Type: application/json' --data-binary "@$body"); fi
    : >"$out"
    # audit-bash disable=B202 # args always holds the base options, never empty
    status=$(curl "${args[@]}" "$base$path") || status=000
    API_STATUS=$status
    case $status in
      2??) return 0 ;;
      429) wait_s=$(_retry_after) ;;
      5?? | 000)
        if [[ $method == POST ]]; then return 1; fi
        wait_s=${API_BACKOFF_S[attempt - 1]-0}
        ;;
      *) return 1 ;;
    esac
    if ((attempt == API_ATTEMPTS)); then break; fi
    say "API $method $base$path: HTTP $status; retrying in ${wait_s}s (attempt $attempt of $API_ATTEMPTS)"
    "${DRBM25_SLEEP:-sleep}" "$wait_s"
  done
  return 1
}

# Dies with what failed, the HTTP status and RunPod's problem detail (never a request body).
api_fail() {
  local what=$1 out=$2 detail=''
  if [[ -s $out ]]; then
    detail=$(jq -r '[.detail // .title // empty] + (.errors // []) | join("; ")' "$out" 2>/dev/null) || detail=''
  fi
  case $API_STATUS in
    000) die "$what: no response from $API_BASE (network error or timeout)" ;;
    401 | 403) die "$what: HTTP $API_STATUS ${detail:+($detail) }- check that RUNPOD_API_KEY is valid and allowed to do this" ;;
    *) die "$what: HTTP $API_STATUS${detail:+: $detail}" ;;
  esac
}

_require_single_page() {
  local more
  more=$(jq -r '.pagination.hasNextPage // false' "$1")
  if [[ $more == true ]]; then
    die "$2: more than one page (over 1000 entries) is not supported"
  fi
}

# hf_api OUT PATH: one GET to the Hugging Face API, with HF_TOKEN's header from HF_AUTH_FILE (mode 600).
hf_api() {
  # shellcheck disable=SC2034 # read by api() through dynamic scope
  local API_BASE_URL=$HF_API_BASE API_AUTH_FILE=$HF_AUTH_FILE
  api "$1" GET "$2"
}

# Sets HF_USER from whoami-v2. With --write, dies unless the token can write the user's model repos, so a
# launch never pays for a run whose models cannot be uploaded.
hf_whoami() {
  local out=$TMP_DIR/hf-whoami.json verdict
  # printf is a builtin: the token reaches the file without appearing in any process's argv.
  printf 'Authorization: Bearer %s\n' "$HF_TOKEN" >"$HF_AUTH_FILE"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! hf_api "$out" /api/whoami-v2; then
    case $API_STATUS in
      401) die "HF_TOKEN is invalid (Hugging Face answered 401); create one at https://huggingface.co/settings/tokens" ;;
      000) die "cannot reach $HF_API_BASE to check HF_TOKEN (network error or timeout); nothing was created" ;;
      *) die "checking HF_TOKEN: Hugging Face answered HTTP $API_STATUS; nothing was created" ;;
    esac
  fi
  if [[ ${1-} == --write ]]; then
    verdict=$(jq -r "$HF_WRITE_JQ" "$out") || die "unexpected answer from $HF_API_BASE/api/whoami-v2"
  else
    verdict=$(jq -r '"ok:" + (.name // "")' "$out") || die "unexpected answer from $HF_API_BASE/api/whoami-v2"
  fi
  case $verdict in
    ok:?*) HF_USER=${verdict#ok:} ;;
    error:*) die "${verdict#error:}" ;;
    *) die "HF_TOKEN: whoami returned no user name" ;;
  esac
  [[ $HF_USER =~ ^[A-Za-z0-9._-]+$ ]] || die "unexpected Hugging Face user name from whoami: '$HF_USER'"
}

# Prints the model slug drbm25.publish (model_slug) derives from the RunPod config: the model name after its
# org, lower case, without -instruct, plus -lora for train_mode lora (Qwen/Qwen2.5-3B-Instruct, full: qwen2.5-3b).
runpod_model_slug() {
  local config=$REPO_ROOT/$RUNPOD_CONFIG_DEFAULT name mode
  [[ -f $config ]] || die "cannot read $config"
  name=$(awk '/^model:/ { m = 1; next } m && /^[^ ]/ { m = 0 } m && $1 == "name:" { gsub(/"/, "", $2); print $2; exit }' "$config")
  mode=$(awk '/^model:/ { m = 1; next } m && /^[^ ]/ { m = 0 } m && $1 == "train_mode:" { print $2; exit }' "$config")
  [[ -n $name && -n $mode ]] || die "no model.name or model.train_mode in $config"
  name=$(printf '%s' "${name##*/}" | tr '[:upper:]' '[:lower:]')
  name=${name%-instruct}
  if [[ $mode == lora ]]; then name+=-lora; fi
  printf '%s\n' "$name"
}

# Sets VOLUME_SHA_normal and VOLUME_SHA_count_once to the sha256 of each run's best/meta.json on the volume ("none"
# when a run has no best/), and VOLUME_STALE_<mode> to stale or ok (MARKER_STALE present), read through the exec API of the running pod POD_ID. Leaves them empty (unknown) when
# there is no running pod or the exec call fails; never dies.
read_volume_best_shas() {
  local out=$TMP_DIR/exec-vol.json body=$TMP_DIR/exec-vol-body.json mode sha stale stdout
  VOLUME_SHA_normal='' VOLUME_SHA_count_once='' VOLUME_STALE_normal='' VOLUME_STALE_count_once=''
  if [[ -z ${POD_ID-} || ${POD_STATUS-} != RUNNING ]]; then return 0; fi
  jq -n --arg script "$VOLUME_SHA_SCRIPT" '{command: ["sh", "-c", $script], timeoutSeconds: 20}' >"$body"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! api "$out" POST "/v2/pods/$POD_ID/exec" "$body"; then
    say "cannot read the volume through pod $POD_ID (HTTP $API_STATUS); the upload state is unknown"
    return 0
  fi
  stdout=$(jq -r '.stdout // ""' "$out") || stdout=''
  while read -r mode sha stale; do
    [[ $sha =~ ^[A-Za-z0-9]+$ && $stale =~ ^(stale|ok)$ ]] || continue
    case $mode in
      normal) VOLUME_SHA_normal=$sha VOLUME_STALE_normal=$stale ;;
      count_once) VOLUME_SHA_count_once=$sha VOLUME_STALE_count_once=$stale ;;
    esac
  done <<<"$stdout"
}

# Prints, per run, the Hub's UPLOADED.json (committed by drbm25.publish only after every file was verified; its
# best step always shown) and, when the volume can be read through a running pod, whether it was written for the
# volume's current best/ (sha256 of best/meta.json). Then: safe to delete the volume yes (both markers match the
# volume), no (a marker is missing or stale), or unknown (the volume cannot be read: no running pod).
hf_upload_status() {
  local slug mode repo out=$TMP_DIR/hf-marker.json step hub_sha volume_sha stale missing=0 matched=0
  slug=$(runpod_model_slug)
  read_volume_best_shas
  printf 'Hugging Face uploads (private repos of %s):\n' "$HF_USER"
  for mode in normal count_once; do
    repo=$HF_USER/$HF_REPO_PREFIX-$slug-$mode
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    if ! hf_api "$out" "/$repo/resolve/main/UPLOADED.json"; then
      case $API_STATUS in
        401 | 404)
          printf '  %s: not uploaded (no verified UPLOADED.json)\n' "$repo"
          missing=$((missing + 1))
          continue
          ;;
        *) die "reading $repo/UPLOADED.json on $HF_API_BASE: HTTP $API_STATUS" ;;
      esac
    fi
    step=$(jq -r '.best_step // "?"' "$out") || die "unexpected UPLOADED.json in $repo"
    hub_sha=$(jq -r '.files["provenance/best_meta.json"].sha256 // ""' "$out") || die "unexpected UPLOADED.json in $repo"
    if [[ $mode == normal ]]; then
      volume_sha=$VOLUME_SHA_normal stale=$VOLUME_STALE_normal
    else
      volume_sha=$VOLUME_SHA_count_once stale=$VOLUME_STALE_count_once
    fi
    if [[ $stale == stale ]]; then
      printf '  %s: no (stale marker could not be withdrawn before training; UPLOADED.json says best step %s)\n' "$repo" "$step"
      missing=$((missing + 1))
    elif [[ -z $volume_sha ]]; then
      printf '  %s: UPLOADED.json for best step %s (cannot compare with the volume: no running pod)\n' "$repo" "$step"
    elif [[ -n $hub_sha && $volume_sha == "$hub_sha" ]]; then
      printf '  %s: UPLOADED.json for best step %s, matches the volume'"'"'s best/\n' "$repo" "$step"
      matched=$((matched + 1))
    else
      printf '  %s: UPLOADED.json for best step %s, STALE: the volume'"'"'s best/ is different; publish again\n' "$repo" "$step"
      missing=$((missing + 1))
    fi
  done
  if ((matched == 2)); then
    printf 'safe to delete the volume: yes (both runs are on the Hub, verified, for the best/ the volume holds)\n'
  elif ((missing > 0)); then
    printf 'safe to delete the volume: no (keep it; publish again: see PUBLISH_FAILED in --logs)\n'
  else
    printf 'safe to delete the volume: unknown (the volume can only be compared while a pod runs; markers are withdrawn at the start of every launch that trains, and the last run_all log says "safe to delete the volume" when the pod checked it)\n'
  fi
}

# ---------------------------------------------------------------- settings

_require_command() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required but not found${2:+ ($2)}"
}

# The ref must exist on the public repository, or the pod's fetch fails after billing has started. Default:
# this checkout's HEAD sha, which must be the tip of a branch or tag there (git ls-remote lists only tips).
resolve_ref() {
  local refs found
  _require_command git
  if [[ -z $OPT_REF ]]; then
    OPT_REF=$(git -C "$REPO_ROOT" rev-parse HEAD) || die "cannot read this checkout's HEAD; pass --ref"
  fi
  # No terminal prompts for credentials, and give up on a transfer slower than 1 KB/s for 30 s.
  refs=$(GIT_TERMINAL_PROMPT=0 GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=30 git ls-remote "$OPT_REPO_URL") ||
    die "cannot list the refs of $OPT_REPO_URL (git ls-remote failed); check the URL and your network"
  found=$(printf '%s\n' "$refs" | awk -v r="$OPT_REF" '
    $1 == r || $2 == "refs/heads/" r || $2 == "refs/tags/" r { print "yes"; exit }')
  if [[ $found != yes ]]; then
    die "ref $OPT_REF is not a branch, tag or branch-tip commit on $OPT_REPO_URL: push it first (git push), or pass --ref <pushed branch>"
  fi
}

ask_api_key() {
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  prompt_var RUNPOD_API_KEY "RunPod API key (pods:write, network volumes, secrets; hidden)" --secret || exit 1
  # printf is a builtin: the key reaches the file without appearing in any process's argv.
  printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" >"$AUTH_FILE"
}

collect_launch_settings() {
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  prompt_var HF_TOKEN "Hugging Face token (hidden)" --secret || exit 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  prompt_var WANDB_API_KEY "Weights & Biases API key (hidden)" --secret || exit 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  prompt_var WANDB_PROJECT "W&B project" --default drbm25 || exit 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  prompt_var WANDB_ENTITY "W&B entity (optional, Enter to skip)" --optional || exit 1
  export WANDB_ENTITY=${WANDB_ENTITY-}
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  prompt_var BUDGET_USD "REMAINING dollar budget (total minus what earlier launches spent; see --logs)" || exit 1
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  _require_positive_number BUDGET_USD || exit 1
  hf_whoami --write
  say "HF_TOKEN can write model repos of $HF_USER (the pod uploads both runs there when it finishes)"
}

# yes or no from the terminal; anything but 'yes' is no. --yes answers for the user.
confirm() {
  local question=$1 answer='' tty=${DRBM25_TTY:-/dev/tty}
  if ((OPT_YES)); then return 0; fi
  # shellcheck disable=SC2310 # a predicate: its status is the answer
  if noninteractive || [[ ! -r $tty ]]; then
    die "confirmation needed but there is no terminal to ask; pass --yes to confirm in advance"
  fi
  printf '%s Type yes to continue: ' "$question" >&2
  IFS= read -r answer <"$tty" || true
  [[ $answer == yes ]]
}

# ---------------------------------------------------------------- reads

# Prints the drbm25 pods (or --pod) as TSV: id status cost dataCenterId.
list_pods() {
  local out=$TMP_DIR/pods.json
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  api "$out" GET /v2/pods || api_fail "listing pods" "$out"
  _require_single_page "$out" "listing pods"
  jq -r --arg name "$POD_NAME" --arg id "$OPT_POD" \
    '.pods[] | select(if $id == "" then .name == $name else .id == $id end)
      | [.id, .status, (.cost // "?" | tostring), (.dataCenterId // "?")] | @tsv' "$out"
}

# Sets POD_ID, POD_STATUS, POD_COST; returns 1 when there is no such pod.
find_pod() {
  local rows count
  rows=$(list_pods)
  count=$(printf '%s' "$rows" | awk 'END { print NR }')
  if ((count == 0)); then return 1; fi
  if ((count > 1)); then
    die "more than one pod named $POD_NAME: ${rows//$'\n'/; } - pass --pod ID"
  fi
  IFS=$'\t' read -r POD_ID POD_STATUS POD_COST _ <<<"$rows"
  [[ $POD_ID =~ ^[A-Za-z0-9_-]+$ ]] || die "unexpected pod id from the API: '$POD_ID'"
}

# Sets VOLUME_ID, VOLUME_DC, VOLUME_SIZE (empty when there is no volume named drbm25).
find_volume() {
  local out=$TMP_DIR/volumes.json rows count
  VOLUME_ID='' VOLUME_DC='' VOLUME_SIZE=''
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  api "$out" GET /v2/network-volumes || api_fail "listing network volumes" "$out"
  _require_single_page "$out" "listing network volumes"
  rows=$(jq -r --arg name "$VOLUME_NAME" \
    '.networkVolumes[] | select(.name == $name) | [.id, .dataCenter, (.size | tostring)] | @tsv' "$out")
  count=$(printf '%s' "$rows" | awk 'END { print NR }')
  if ((count == 0)); then return 0; fi
  if ((count > 1)); then
    die "more than one network volume named $VOLUME_NAME (${rows//$'\n'/; }); keep one (rename or delete the others in the console) and run again"
  fi
  IFS=$'\t' read -r VOLUME_ID VOLUME_DC VOLUME_SIZE <<<"$rows"
  [[ $VOLUME_ID =~ ^[A-Za-z0-9_-]+$ ]] || die "unexpected volume id from the API: '$VOLUME_ID'"
}

# Sets OFFER_PRICE (USD/h for all GPUs), OFFER_GPU, OFFER_DC, OFFER_AVAIL: the cheapest stocked offer in one of
# ALLOWED_DCS (a JSON array of data center ids, or null for any). Prices are list prices per GPU per cloud
# (catalog docs); stock is per data center.
find_offer() {
  local allowed=$1 out=$TMP_DIR/gpus.json ids row
  case $OPT_GPU in
    sxm) ids=$(jq -cn --arg a "$GPU_ID_SXM" '[$a]') ;;
    pcie) ids=$(jq -cn --arg a "$GPU_ID_PCIE" '[$a]') ;;
    any) ids=$(jq -cn --arg a "$GPU_ID_SXM" --arg b "$GPU_ID_PCIE" '[$a, $b]') ;;
    pro6000) ids=$(jq -cn --arg a "$GPU_ID_PRO6000" '[$a]') ;;
  esac
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  api "$out" GET "/v2/catalog/gpus?include=AVAILABILITY&product=POD&count=$GPU_COUNT&cloud=SECURE&minCudaVersion=$MIN_CUDA" ||
    api_fail "reading GPU stock and prices" "$out"
  # Price rounded to cents; the 1e-6 keeps float noise (2.19 x 200 = 438.00000000000006) from rounding up.
  row=$(jq -r --argjson ids "$ids" --argjson n "$GPU_COUNT" --argjson dcs "$allowed" '
    [ .gpus[] | select(.id as $id | $ids | index($id)) | select(.secure == true and .price.secure != null)
      | . as $g | ((($g.price.secure * $n * 100) - 0.000001 | ceil) / 100) as $price
      | ($g.dataCenters // [])[] | select(.availability != "NONE") | .id as $d | select($dcs == null or ($dcs | index($d)))
      | {price: $price, gpu: $g.id, dc: .id, avail: .availability} ]
    | sort_by(.price, ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}[.avail] // 3))
    | .[0] // empty | [(.price | tostring), .gpu, .dc, .avail] | @tsv' "$out")
  OFFER_PRICE='' OFFER_GPU='' OFFER_DC='' OFFER_AVAIL=''
  if [[ -n $row ]]; then IFS=$'\t' read -r OFFER_PRICE OFFER_GPU OFFER_DC OFFER_AVAIL <<<"$row"; fi
}

# Prints a JSON array of the data centers that offer STANDARD network volumes (networkVolumeTypes, from the
# data center catalog). A new volume is STANDARD so its documented price is the one quoted.
standard_volume_dcs() {
  local out=$TMP_DIR/datacenters.json
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  api "$out" GET /v2/catalog/datacenters || api_fail "reading the data center catalog" "$out"
  jq -c '[.dataCenters[] | select((.networkVolumeTypes // []) | index("STANDARD")) | .id]' "$out"
}

# Prints "DC: TYPES" for every data center, for the message when no STANDARD one has stock.
_volume_types_summary() {
  jq -r '[.dataCenters[] | .id + ": " + (((.networkVolumeTypes // []) | join("+")) | if . == "" then "none" else . end)] | join(", ")' \
    "$TMP_DIR/datacenters.json"
}

# ---------------------------------------------------------------- writes

# Creates or rotates secret NAME with the value of exported variable VAR. The value goes into a mode-600 body
# file through jq's env (not its argv) and the file is removed right after the call.
upsert_secret() {
  local name=$1 var=$2 out=$TMP_DIR/secret-resp.json body=$TMP_DIR/secret.json id
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  api "$out" GET "/v2/account/secrets?name=$name" || api_fail "looking up secret $name" "$out"
  id=$(jq -r --arg n "$name" '[.secrets[]? | select((.name | ascii_downcase) == ($n | ascii_downcase))][0].id // empty' "$out")
  if [[ -n $id ]]; then
    [[ $id =~ ^[A-Za-z0-9_-]+$ ]] || die "unexpected secret id from the API: '$id'"
    jq -n --arg var "$var" '{value: env[$var]}' >"$body"
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    api "$out" PATCH "/v2/account/secrets/$id" "$body" || { rm -f -- "$body"; api_fail "updating secret $name" "$out"; }
    say "updated RunPod secret $name"
  else
    jq -n --arg var "$var" --arg n "$name" '{name: $n, value: env[$var], description: "drbm25 launcher"}' >"$body"
    # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
    api "$out" POST /v2/account/secrets "$body" || { rm -f -- "$body"; api_fail "creating secret $name" "$out"; }
    say "created RunPod secret $name"
  fi
  rm -f -- "$body"
}

create_volume() {
  local out=$TMP_DIR/volume-resp.json body=$TMP_DIR/volume.json
  jq -n --arg name "$VOLUME_NAME" --argjson size "$OPT_VOLUME_GB" --arg dc "$OFFER_DC" \
    '{name: $name, size: $size, dataCenter: $dc, type: "STANDARD"}' >"$body"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  api "$out" POST /v2/network-volumes "$body" ||
    api_fail "creating the network volume (nothing billable was created; check the console's Storage page if unsure)" "$out"
  VOLUME_ID=$(jq -r '.id // empty' "$out")
  VOLUME_SIZE=$(jq -r '.size // empty' "$out")
  VOLUME_DC=$OFFER_DC
  [[ $VOLUME_ID =~ ^[A-Za-z0-9_-]+$ ]] || die "unexpected volume id from the API: '$VOLUME_ID'"
  say "created network volume $VOLUME_NAME ($VOLUME_ID, $VOLUME_SIZE GB) in $VOLUME_DC"
}

# The create-a-pod body. entrypoint+cmd override the image's command explicitly (BaseContainerConfig); the
# command is pod_bootstrap.sh, which starts the image's /start.sh services first. Keys are secret references.
pod_body() {
  jq -n \
    --arg name "$POD_NAME" --arg image "$IMAGE" --arg gpu "$OFFER_GPU" --argjson count "$GPU_COUNT" \
    --arg min_cuda "$MIN_CUDA" --arg dc "$OFFER_DC" --arg volume "$VOLUME_ID" \
    --argjson disk "$CONTAINER_DISK_GB" --rawfile bootstrap "$SCRIPT_DIR/pod_bootstrap.sh" \
    --arg hf "{{ RUNPOD_SECRET_$SECRET_HF }}" --arg wandb "{{ RUNPOD_SECRET_$SECRET_WANDB }}" \
    --arg runpod "{{ RUNPOD_SECRET_$SECRET_RUNPOD }}" \
    --arg project "$WANDB_PROJECT" --arg entity "$WANDB_ENTITY" --arg budget "$BUDGET_USD" \
    --arg price "$OFFER_PRICE" --arg volume_gb "$VOLUME_SIZE" --arg repo "$OPT_REPO_URL" --arg ref "$OPT_REF" \
    --arg clear "$OPT_CLEAR_FAILED" --arg hard_deadline "$HARD_DEADLINE_UNIX" \
    --arg budget_deadline "$BUDGET_DEADLINE_UNIX" --arg launch_id "$LAUNCH_ID" '{
      name: $name, image: $image, cloud: "SECURE", dataCenterIds: [$dc],
      gpu: {id: $gpu, count: $count, minCudaVersion: $min_cuda},
      disk: $disk, ports: ["8888/http", "22/tcp"], startSsh: true, startJupyter: true,
      mounts: {network: [{volumeId: $volume, path: "/workspace"}]},
      entrypoint: ["/bin/bash", "-c"], cmd: [$bootstrap],
      env: {
        HF_TOKEN: $hf, WANDB_API_KEY: $wandb, DRBM25_RUNPOD_API_KEY: $runpod,
        WANDB_PROJECT: $project, WANDB_ENTITY: $entity, BUDGET_USD: $budget, PRICE_PER_HOUR: $price,
        TERMINATE_ON_FINISH: "yes", VOLUME_GB: $volume_gb, DRBM25_REPO_URL: $repo, DRBM25_REF: $ref,
        DRBM25_NONINTERACTIVE: "1", DRBM25_CLEAR_FAILED: $clear, DRBM25_HARD_DEADLINE_UNIX: $hard_deadline,
        DRBM25_BUDGET_DEADLINE_UNIX: $budget_deadline, DRBM25_LAUNCH_ID: $launch_id
      }
    }'
}

create_pod() {
  local out=$TMP_DIR/pod-resp.json body=$TMP_DIR/pod.json cost now suffix
  now=$(date +%s)
  # Absolute deadlines, so a container that RunPod restarts gets no fresh budget: the budget deadline is
  # launch time + the hours the confirmed budget buys at the quoted price (run_all.sh's deadline), and the
  # watchdog's hard deadline is a margin after it.
  BUDGET_DEADLINE_UNIX=$(awk -v t="$now" -v b="$BUDGET_USD" -v p="$OFFER_PRICE" 'BEGIN { printf "%d", t + b / p * 3600 }')
  HARD_DEADLINE_UNIX=$((BUDGET_DEADLINE_UNIX + HARD_DEADLINE_MARGIN_S))
  # Names this launch's done marker on the volume (pod_bootstrap.sh): time plus 8 random hex digits.
  suffix=$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n') || die "cannot read /dev/urandom"
  LAUNCH_ID=$(date -u '+%Y%m%dT%H%M%SZ')-$suffix
  [[ $LAUNCH_ID =~ ^[0-9TZ]+-[0-9a-f]{8}$ ]] || die "could not make a launch id (got '$LAUNCH_ID')"
  pod_body >"$body"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! api "$out" POST /v2/pods "$body"; then
    case $API_STATUS in
      5?? | 000) api_fail "creating the pod (it may have been created anyway: check with $SCRIPT_NAME --status before running again)" "$out" ;;
      *) api_fail "creating the pod (the volume is kept; run again later, stock changes often)" "$out" ;;
    esac
  fi
  POD_ID=$(jq -r '.id // empty' "$out")
  cost=$(jq -r '.cost // empty' "$out")
  [[ $POD_ID =~ ^[A-Za-z0-9_-]+$ ]] || die "unexpected pod id from the API: '$POD_ID' (check $CONSOLE_URL)"
  if [[ $cost =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v c="$cost" -v p="$OFFER_PRICE" 'BEGIN { exit !(c > p + 0.005) }'; then
    say "WARNING: the pod's hourly cost is \$$cost, above the quoted \$$OFFER_PRICE that the run's budget uses."
    say "WARNING: stop it with $SCRIPT_NAME --stop if that is not acceptable."
  fi
}

# ---------------------------------------------------------------- modes

do_launch() {
  local hours monthly volume_line allowed_dcs types
  resolve_ref
  collect_launch_settings
  # shellcheck disable=SC2310 # find_pod returns 1 for "no pod", which is what is wanted here
  if find_pod; then
    die "pod $POD_ID ($POD_NAME, $POD_STATUS) already exists; two pods must not share the volume. See $SCRIPT_NAME --status, or --stop it first"
  fi
  find_volume
  if [[ -n $VOLUME_ID ]]; then
    allowed_dcs=$(jq -cn --arg dc "$VOLUME_DC" '[$dc]')
  else
    allowed_dcs=$(standard_volume_dcs)
  fi
  find_offer "$allowed_dcs"
  if [[ -z $OFFER_GPU ]]; then
    if [[ -n $VOLUME_ID ]]; then
      die "no ${GPU_COUNT}x GPU of --gpu $OPT_GPU is in stock in $VOLUME_DC, where volume $VOLUME_NAME ($VOLUME_ID) lives. A volume is tied to its data center, so try again later (or move the data yourself); no second volume is created"
    fi
    types=$(_volume_types_summary)
    die "no ${GPU_COUNT}x GPU of --gpu $OPT_GPU is in stock on the secure cloud in a data center that offers STANDARD network volumes right now (volume types per data center: $types). HIGH_PERFORMANCE volumes cost more and their price is not published in the docs, so this launcher does not create them: try again later, or create a volume named $VOLUME_NAME yourself in the console and run again"
  fi

  hours=$(awk -v b="$BUDGET_USD" -v p="$OFFER_PRICE" 'BEGIN { printf "%.1f", b / p }')
  if [[ -n $VOLUME_ID ]]; then
    volume_line="reuse $VOLUME_NAME ($VOLUME_ID, $VOLUME_SIZE GB) in $VOLUME_DC"
  else
    monthly=$(awk -v g="$OPT_VOLUME_GB" -v r="$VOLUME_USD_PER_GB_MONTH" 'BEGIN { printf "%.2f", g * r }')
    volume_line="CREATE $VOLUME_NAME: $OPT_VOLUME_GB GB standard in $OFFER_DC, about \$$monthly/month (\$$VOLUME_USD_PER_GB_MONTH/GB/month) until you delete it"
  fi
  say ""
  say "Plan:"
  say "  pod:     ${GPU_COUNT}x $OFFER_GPU, secure cloud, data center $OFFER_DC (stock $OFFER_AVAIL)"
  say "  price:   \$$OFFER_PRICE/h (list price) while the pod exists, plus ${CONTAINER_DISK_GB} GB container disk"
  say "  budget:  \$$BUDGET_USD remaining buys about $hours h; the run stops, evaluates, reports and terminates the pod"
  say "           (a watchdog in the pod terminates it anyway $((HARD_DEADLINE_MARGIN_S / 3600)) h after that)"
  say "  volume:  $volume_line"
  say "  code:    $OPT_REPO_URL at $OPT_REF"
  say ""
  # shellcheck disable=SC2310 # a predicate: its status is the answer
  if ! confirm "Create these billable resources?"; then
    die "not confirmed: nothing was created"
  fi

  upsert_secret "$SECRET_HF" HF_TOKEN
  upsert_secret "$SECRET_WANDB" WANDB_API_KEY
  upsert_secret "$SECRET_RUNPOD" RUNPOD_API_KEY
  if [[ -z $VOLUME_ID ]]; then create_volume; fi
  create_pod

  say ""
  say "Created pod $POD_ID in $OFFER_DC at \$$OFFER_PRICE/h. It clones the repository and runs on its own;"
  say "you can disconnect. It terminates itself when the run ends or fails (the volume is kept)."
  say "  console:  $CONSOLE_URL"
  say "  progress: bash scripts/runpod/launch.sh --logs   (or --watch, --status; --stop to terminate)"
  say "  on the pod (web terminal or SSH): tail -f /workspace/drbm25/logs/bootstrap-*.log, then run_all-*.log"
}

do_status() {
  local rows count
  rows=$(list_pods)
  POD_ID='' POD_STATUS=''
  if [[ -z $rows ]]; then
    printf 'no %s pod exists (the network volume and its data are kept)\n' "$POD_NAME"
  else
    printf '%s\n' "$rows" | awk -F '\t' '{ printf "pod %s  status %s  $%s/h  data center %s\n", $1, $2, $3, $4 }'
    # One pod: its volume can be read for the upload check (find_pod dies on several).
    count=$(printf '%s\n' "$rows" | awk 'END { print NR }')
    if ((count == 1)); then
      IFS=$'\t' read -r POD_ID POD_STATUS _ <<<"$rows"
      [[ $POD_ID =~ ^[A-Za-z0-9_-]+$ ]] || die "unexpected pod id from the API: '$POD_ID'"
    fi
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  prompt_var HF_TOKEN "Hugging Face token, to check the uploads (hidden)" --secret || exit 1
  hf_whoami
  hf_upload_status
}

# Prints the newest log lines through the exec API. Returns 1 when the pod cannot run commands yet.
show_logs() {
  local lines=$1 out=$TMP_DIR/exec-resp.json body=$TMP_DIR/exec.json
  jq -n --arg script "$LOG_SCRIPT" --arg n "$lines" \
    '{command: ["sh", "-c", $script], timeoutSeconds: 20, env: {DRBM25_TAIL_LINES: $n}}' >"$body"
  # POST, so a 5xx is not retried: exec of a read-only command is harmless to repeat, but one call is enough.
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  if ! api "$out" POST "/v2/pods/$POD_ID/exec" "$body"; then
    if [[ $API_STATUS == 409 ]]; then
      say "pod $POD_ID is not running a container yet (HTTP 409); try again in a minute"
      return 1
    fi
    api_fail "reading logs from pod $POD_ID" "$out"
  fi
  jq -r '.stdout // ""' "$out"
  jq -r 'if (.stderr // "") != "" then "stderr: " + .stderr else empty end' "$out" >&2
}

do_logs() {
  # shellcheck disable=SC2310 # find_pod returns 1 for "no pod"
  find_pod || die "no $POD_NAME pod exists; the logs stay on the volume under /workspace/drbm25/logs"
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  show_logs "$LOG_LINES" || exit 1
}

do_watch() {
  local poll now
  for ((poll = 1; poll <= WATCH_MAX_POLLS; poll++)); do
    # shellcheck disable=SC2310 # find_pod returns 1 for "no pod"
    if ! find_pod; then
      say "no $POD_NAME pod exists any more: the run has ended (its logs and report are on the volume)"
      return 0
    fi
    now=$(date '+%Y-%m-%d %H:%M')
    say "$now pod $POD_ID: $POD_STATUS (\$$POD_COST/h)"
    # shellcheck disable=SC2310 # a failed log read must not end the watch
    if [[ $POD_STATUS == RUNNING ]]; then show_logs "$WATCH_LOG_LINES" || true; fi
    "${DRBM25_SLEEP:-sleep}" $((OPT_INTERVAL_MIN * 60))
  done
  die "stopped watching after $WATCH_MAX_POLLS polls; the pod is still there"
}

do_stop() {
  # shellcheck disable=SC2310 # find_pod returns 1 for "no pod"
  find_pod || die "no $POD_NAME pod exists; nothing to stop"
  # shellcheck disable=SC2310 # a predicate: its status is the answer
  if ! confirm "Terminate pod $POD_ID ($POD_STATUS, \$$POD_COST/h)? The run stops at once; the network volume and its data are kept."; then
    die "not confirmed: pod $POD_ID is still running"
  fi
  # shellcheck disable=SC2310 # status is checked explicitly here, not left to set -e
  api "$TMP_DIR/delete-resp.json" DELETE "/v2/pods/$POD_ID" || api_fail "terminating pod $POD_ID (check $CONSOLE_URL)" "$TMP_DIR/delete-resp.json"
  say "terminated pod $POD_ID; resume later with: bash scripts/runpod/launch.sh"
}

# ---------------------------------------------------------------- main

parse_args() {
  OPT_MODE=launch OPT_YES=0 OPT_GPU=any OPT_VOLUME_GB=$DEFAULT_VOLUME_GB OPT_REF='' OPT_REPO_URL=$DEFAULT_REPO_URL
  OPT_CLEAR_FAILED=0 OPT_POD='' OPT_INTERVAL_MIN=10
  local cloud=secure
  while (($# > 0)); do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      --yes) OPT_YES=1 ;;
      --clear-failed) OPT_CLEAR_FAILED=1 ;;
      --status | --logs | --watch | --stop)
        [[ $OPT_MODE == launch ]] || usage_error "only one of --status, --logs, --watch, --stop"
        OPT_MODE=${1#--}
        ;;
      --gpu | --cloud | --volume-gb | --ref | --repo-url | --pod | --interval-min)
        (($# >= 2)) || usage_error "$1 needs a value"
        case $1 in
          --gpu) OPT_GPU=$2 ;;
          --cloud) cloud=$2 ;;
          --volume-gb) OPT_VOLUME_GB=$2 ;;
          --ref) OPT_REF=$2 ;;
          --repo-url) OPT_REPO_URL=$2 ;;
          --pod) OPT_POD=$2 ;;
          --interval-min) OPT_INTERVAL_MIN=$2 ;;
        esac
        shift
        ;;
      *) usage_error "unknown option: $1" ;;
    esac
    shift
  done
  case $OPT_GPU in sxm | pcie | any | pro6000) ;; *) usage_error "--gpu must be sxm, pcie, any or pro6000, got '$OPT_GPU'" ;; esac
  case $cloud in
    secure | any) ;;
    # https://docs.runpod.io/storage/network-volumes: "Network volumes are only available for Pods in the
    # Secure Cloud." Without the volume a run cannot resume, so community pods are not offered.
    community) usage_error "--cloud community cannot be used: network volumes are only available for Pods in the Secure Cloud, and the run keeps its data and checkpoints on one" ;;
    *) usage_error "--cloud must be secure, community or any, got '$cloud'" ;;
  esac
  # 10-4096 GB: the create-a-network-volume reference.
  if [[ ! $OPT_VOLUME_GB =~ ^[1-9][0-9]*$ ]] || ((OPT_VOLUME_GB < 10 || OPT_VOLUME_GB > 4096)); then
    usage_error "--volume-gb must be a whole number from 10 to 4096, got '$OPT_VOLUME_GB'"
  fi
  if [[ ! $OPT_INTERVAL_MIN =~ ^[1-9][0-9]*$ ]]; then usage_error "--interval-min must be a positive whole number"; fi
  if [[ -n $OPT_REF ]] && { [[ ! $OPT_REF =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || [[ $OPT_REF == *..* ]]; }; then
    usage_error "--ref must be a branch, tag or commit sha (letters, digits, . _ / -), got '$OPT_REF'"
  fi
  if [[ ! $OPT_REPO_URL =~ ^https://[A-Za-z0-9.-]+/[A-Za-z0-9._/-]+$ ]]; then
    usage_error "--repo-url must be an https URL, got '$OPT_REPO_URL'"
  fi
  if [[ -n $OPT_POD && ! $OPT_POD =~ ^[A-Za-z0-9_-]+$ ]]; then usage_error "--pod must be a pod id"; fi
}

main() {
  parse_args "$@"
  # lib.sh: prompt_var (hidden prompts, environment first), noninteractive, _require_positive_number.
  # shellcheck source=lib.sh
  source "$SCRIPT_DIR/lib.sh"
  _require_command curl
  _require_command jq "macOS 15 and later ship /usr/bin/jq; otherwise brew install jq"
  # Every file this script writes holds or may hold a secret: mode 600 from the start, removed on exit.
  umask 077
  TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/drbm25-launch.XXXXXX") || die "cannot create a temporary directory"
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  AUTH_FILE=$TMP_DIR/auth
  HF_AUTH_FILE=$TMP_DIR/hf_auth
  ask_api_key
  case $OPT_MODE in
    launch) do_launch ;;
    status) do_status ;;
    logs) do_logs ;;
    watch) do_watch ;;
    stop) do_stop ;;
  esac
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
