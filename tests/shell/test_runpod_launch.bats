#!/usr/bin/env bats
# Tests for scripts/runpod/launch.sh (laptop launcher) and scripts/runpod/pod_bootstrap.sh (the pod's start
# command). Run: bats tests/shell/
# curl is a PATH shim: it never reaches the network. It records each call's method and URL, its argv, and a
# copy of each request body, and answers from response files under $STUB_DIR/resp (see make_curl_stub).

bats_require_minimum_version 1.5.0

readonly HF=hf_SECRET_hf_value
readonly WB=wb_SECRET_wb_value
readonly RP=rp_SECRET_rp_value

setup() {
  LAUNCH="${BATS_TEST_DIRNAME}/../../scripts/runpod/launch.sh"
  BOOTSTRAP="${BATS_TEST_DIRNAME}/../../scripts/runpod/pod_bootstrap.sh"
  export HOME=$BATS_TEST_TMPDIR
  cd "$BATS_TEST_TMPDIR" || return 1
  export STUB_DIR=$BATS_TEST_TMPDIR/stub
  mkdir -p "$STUB_DIR/resp" "$STUB_DIR/bodies" "$BATS_TEST_TMPDIR/bin"
  : >"$STUB_DIR/calls.log"
  : >"$STUB_DIR/argv.log"
  make_curl_stub
  make_git_stub
  cat >"$BATS_TEST_TMPDIR/bin/sleep_stub" <<'EOF'
#!/usr/bin/env bash
echo "sleep $*" >>"$STUB_DIR/calls.log"
EOF
  chmod +x "$BATS_TEST_TMPDIR/bin/"*
  export PATH=$BATS_TEST_TMPDIR/bin:$PATH
  export DRBM25_SLEEP=sleep_stub
  export DRBM25_TTY=$BATS_TEST_TMPDIR/tty
  : >"$DRBM25_TTY"
  export TMPDIR=$BATS_TEST_TMPDIR/tmp
  mkdir -p "$TMPDIR"
  export RUNPOD_API_KEY=$RP HF_TOKEN=$HF WANDB_API_KEY=$WB WANDB_PROJECT=drbm25 WANDB_ENTITY= BUDGET_USD=150
  default_responses
}

make_curl_stub() {
  cat >"$BATS_TEST_TMPDIR/bin/curl" <<'EOF'
#!/usr/bin/env bash
# Records the call, then answers from $STUB_DIR/resp/<METHOD>_<path with / as _>.{json,status}. A .status file
# holds one status per line, used in order (the last one repeats); defaults: 200, or 201 for POST, 204 DELETE.
printf '%s\n' "$*" >>"$STUB_DIR/argv.log"
method=GET out='' hdr='' body='' auth='' url='' fail=0
while (($#)); do
  case $1 in
    --fail) fail=1 ;;
    -X) method=$2; shift ;;
    -o) out=$2; shift ;;
    -D) hdr=$2; shift ;;
    -w) shift ;;
    --data-binary) body=${2#@}; shift ;;
    -H)
      # -H @-: the header comes on stdin (pod_bootstrap.sh, lib.sh); -H @file: launch.sh's auth file.
      if [[ $2 == @- ]]; then cat >>"$STUB_DIR/auth_stdin.log"; elif [[ $2 == @* ]]; then auth=${2#@}; fi
      shift
      ;;
    https://*) url=$1 ;;
  esac
  shift
done
path=${url#https://api.runpod.io}
path=${path#https://huggingface.co}
path=${path%%\?*}
key=${method}$(printf '%s' "$path" | tr '/' '_')
n=$(cat "$STUB_DIR/n_$key" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" >"$STUB_DIR/n_$key"
printf '%s %s\n' "$method" "$url" >>"$STUB_DIR/calls.log"
if [[ -n $auth ]]; then
  ls -l "$auth" | cut -c1-10 >>"$STUB_DIR/auth_modes"
  echo "$auth" >>"$STUB_DIR/auth_paths"
fi
if [[ -n $body ]]; then
  total=$(ls "$STUB_DIR/bodies" | wc -l | tr -d ' ')
  cp "$body" "$STUB_DIR/bodies/$((total + 1))-$key.json"
fi
case $method in POST) status=201 ;; DELETE) status=204 ;; *) status=200 ;; esac
if [[ -f $STUB_DIR/resp/$key.status ]]; then
  status=$(sed -n "${n}p" "$STUB_DIR/resp/$key.status")
  [[ -n $status ]] || status=$(sed -n '$p' "$STUB_DIR/resp/$key.status")
fi
if [[ $status == 000 ]]; then printf '000'; exit 7; fi
[[ -n $hdr ]] && printf 'HTTP/1.1 %s\r\nRetry-After: 3\r\n\r\n' "$status" >"$hdr"
if [[ -n $out ]]; then
  if [[ -f $STUB_DIR/resp/$key.json ]]; then cp "$STUB_DIR/resp/$key.json" "$out"; else : >"$out"; fi
fi
printf '%s' "$status"
if ((fail && status >= 400)); then exit 22; fi
EOF
}

# git ls-remote answers from $STUB_DIR/ls_remote (no network); every other git command is the real git.
make_git_stub() {
  local real_git
  real_git=$(command -v git)
  printf '%s\trefs/heads/feat/test\n' 0123456789abcdef0123456789abcdef01234567 >"$STUB_DIR/ls_remote"
  cat >"$BATS_TEST_TMPDIR/bin/git" <<EOF
#!/usr/bin/env bash
if [[ \$1 == ls-remote ]]; then
  echo "git \$*" >>"\$STUB_DIR/calls.log"
  cat "\$STUB_DIR/ls_remote"
  exit 0
fi
exec "$real_git" "\$@"
EOF
}

resp() { # KEY JSON: the body returned for KEY (for example GET_v2_pods)
  printf '%s\n' "$2" >"$STUB_DIR/resp/$1.json"
}

resp_status() { # KEY STATUS...: statuses returned for KEY, in order
  printf '%s\n' "${@:2}" >"$STUB_DIR/resp/$1.status"
}

default_responses() {
  resp GET_api_whoami-v2 '{"type":"user","name":"hfuser","auth":{"type":"access_token","accessToken":{"displayName":"t","role":"write"}}}'
  resp GET_v2_catalog_gpus '{"gpus":[
    {"id":"NVIDIA A100-SXM4-80GB","name":"A100 SXM","memory":80,"secure":true,"community":true,
     "price":{"secure":2.19,"community":1.39},"availability":"LOW",
     "dataCenters":[{"id":"US-MD-1","name":"US Maryland 1","availability":"LOW"},
                    {"id":"EU-RO-1","name":"EU Romania 1","availability":"HIGH"}]},
    {"id":"NVIDIA A100 80GB PCIe","name":"A100 PCIe","memory":80,"secure":true,"community":true,
     "price":{"secure":2.09,"community":1.19},"availability":"MEDIUM",
     "dataCenters":[{"id":"US-KS-2","name":"US Kansas 2","availability":"MEDIUM"}]},
    {"id":"NVIDIA GeForce RTX 4090","name":"RTX 4090","memory":24,"secure":true,"community":true,
     "price":{"secure":0.44,"community":0.31},"availability":"HIGH",
     "dataCenters":[{"id":"US-MD-1","name":"US Maryland 1","availability":"HIGH"}]}]}'
  resp GET_v2_network-volumes '{"networkVolumes":[{"id":"vol1","name":"drbm25","size":400,"dataCenter":"US-MD-1","type":"STANDARD"}],"pagination":{"nextCursor":null,"hasNextPage":false}}'
  resp GET_v2_pods '{"pods":[],"pagination":{"nextCursor":null,"hasNextPage":false}}'
  resp GET_v2_account_secrets '{"secrets":[]}'
  resp POST_v2_account_secrets '{"id":"sec9","name":"x"}'
  resp POST_v2_network-volumes '{"id":"volNEW","name":"drbm25","size":400,"dataCenter":"US-KS-2","type":"STANDARD"}'
  resp POST_v2_pods '{"id":"pod123","name":"drbm25","status":"PROVISIONING","cost":4.38,"dataCenterId":"US-MD-1"}'
  resp GET_v2_catalog_datacenters '{"dataCenters":[
    {"id":"US-MD-1","name":"US Maryland 1","networkVolumeTypes":["STANDARD","HIGH_PERFORMANCE"]},
    {"id":"EU-RO-1","name":"EU Romania 1","networkVolumeTypes":["STANDARD"]},
    {"id":"US-KS-2","name":"US Kansas 2","networkVolumeTypes":["STANDARD"]}]}'
}

launch() {
  bash "$LAUNCH" --ref feat/test "$@"
}

calls() { cat "$STUB_DIR/calls.log"; }

body_of() { # KEY: the last recorded request body for KEY
  local f
  f=$(ls "$STUB_DIR/bodies" | grep -- "-$1.json\$" | sort -n | tail -n 1)
  [[ -n $f ]] || return 1
  cat "$STUB_DIR/bodies/$f"
}

no_billable_calls() {
  ! grep -q '^POST https://api.runpod.io/v2/pods$' "$STUB_DIR/calls.log" &&
    ! grep -q '^POST https://api.runpod.io/v2/network-volumes' "$STUB_DIR/calls.log"
}

# ---------------------------------------------------------------- Hugging Face token and uploads

@test "a read-only HF token stops the launch before anything billable or any secret is written" {
  resp GET_api_whoami-v2 '{"type":"user","name":"hfuser","auth":{"type":"access_token","accessToken":{"role":"read"}}}'
  run -1 --separate-stderr launch --yes
  [[ $stderr == *read-only* ]]
  grep -q '^GET https://huggingface.co/api/whoami-v2$' "$STUB_DIR/calls.log"
  ! grep -q '^\(POST\|PATCH\|DELETE\)' "$STUB_DIR/calls.log"
}

@test "a fine-grained HF token without repo.write on the user stops the launch" {
  resp GET_api_whoami-v2 '{"name":"hfuser","auth":{"accessToken":{"role":"fineGrained","fineGrained":{"scoped":[{"entity":{"type":"user","name":"hfuser"},"permissions":["repo.content.read"]}]}}}}'
  run -1 --separate-stderr launch --yes
  [[ $stderr == *repo.write* ]]
  no_billable_calls
}

@test "a fine-grained HF token with repo.write on the user is accepted" {
  resp GET_api_whoami-v2 '{"name":"hfuser","auth":{"accessToken":{"role":"fineGrained","fineGrained":{"scoped":[{"entity":{"type":"user","name":"hfuser"},"permissions":["repo.content.read","repo.write"]}]}}}}'
  run -0 launch --yes
  grep -q '^GET https://huggingface.co/api/whoami-v2$' "$STUB_DIR/calls.log"
  grep -q '^POST https://api.runpod.io/v2/pods$' "$STUB_DIR/calls.log"
}

@test "an invalid HF token (401) stops the launch before anything billable" {
  resp_status GET_api_whoami-v2 401
  run -1 --separate-stderr launch --yes
  [[ $stderr == *HF_TOKEN* ]]
  no_billable_calls
}

@test "the HF token check happens before the pod is created, with the token only in a header file" {
  run -0 launch --yes
  grep -q '^GET https://huggingface.co/api/whoami-v2$' "$STUB_DIR/calls.log"
  [[ $(grep -n 'huggingface.co/api/whoami-v2' "$STUB_DIR/calls.log" | cut -d: -f1) -lt $(grep -n '^POST https://api.runpod.io/v2/pods$' "$STUB_DIR/calls.log" | cut -d: -f1) ]]
  ! grep -q -e "$HF" "$STUB_DIR/argv.log"
}

hub_marker() { # MODE BEST_STEP META_SHA: the Hub's UPLOADED.json for MODE
  resp "GET_hfuser_drbm25-qwen2.5-3b-$1_resolve_main_UPLOADED.json" \
    "{\"repo_id\":\"hfuser/drbm25-qwen2.5-3b-$1\",\"reward_mode\":\"$1\",\"best_step\":$2,\"files\":{\"provenance/best_meta.json\":{\"sha256\":\"$3\",\"size\":10}}}"
}

running_pod() { # STDOUT: what the pod's exec of the volume check prints
  resp GET_v2_pods '{"pods":[{"id":"pod7","name":"drbm25","status":"RUNNING","cost":4.38,"dataCenterId":"US-MD-1"}],"pagination":{"hasNextPage":false}}'
  resp POST_v2_pods_pod7_exec "$(jq -cn --arg o "$1" '{stdout: $o, stderr: ""}')"
}

@test "--status says safe to delete only when both Hub markers match the volume's current best/ (read through the pod)" {
  hub_marker normal 400 aaa
  hub_marker count_once 350 bbb
  running_pod $'normal aaa ok\ncount_once bbb ok\n'
  run -0 bash "$LAUNCH" --status
  [[ $output == *"hfuser/drbm25-qwen2.5-3b-normal: UPLOADED.json for best step 400, matches the volume"* ]]
  [[ $output == *"hfuser/drbm25-qwen2.5-3b-count_once: UPLOADED.json for best step 350, matches the volume"* ]]
  [[ $output == *"safe to delete the volume: yes"* ]]
}

@test "--status says no when a Hub marker is for an older best/ than the volume holds" {
  hub_marker normal 400 aaa
  hub_marker count_once 350 bbb
  running_pod $'normal aaa ok\ncount_once NEWER ok\n'
  run -0 bash "$LAUNCH" --status
  [[ $output == *"count_once: UPLOADED.json for best step 350, STALE"* ]]
  [[ $output == *"safe to delete the volume: no"* ]]
}

@test "--status without a pod cannot read the volume and says unknown, never yes, with each marker's best step" {
  hub_marker normal 400 aaa
  hub_marker count_once 350 bbb
  run -0 bash "$LAUNCH" --status
  [[ $output == *"no drbm25 pod"* ]]
  [[ $output == *"normal: UPLOADED.json for best step 400"* ]]
  [[ $output == *"count_once: UPLOADED.json for best step 350"* ]]
  [[ $output == *"safe to delete the volume: unknown"* ]]
  [[ $output != *"safe to delete the volume: yes"* ]]
}

@test "--status says no when the volume holds MARKER_STALE for a run (its Hub marker could not be withdrawn)" {
  hub_marker normal 400 aaa
  hub_marker count_once 350 bbb
  running_pod $'normal aaa stale\ncount_once bbb ok\n'
  run -0 bash "$LAUNCH" --status
  [[ $output == *"hfuser/drbm25-qwen2.5-3b-normal: no (stale marker could not be withdrawn"* ]]
  [[ $output == *"safe to delete the volume: no"* ]]
  [[ $output != *"safe to delete the volume: yes"* ]]
}

@test "--status says no when a run has no Hub marker (404), even without a pod" {
  resp_status GET_hfuser_drbm25-qwen2.5-3b-normal_resolve_main_UPLOADED.json 404
  hub_marker count_once 350 bbb
  run -0 bash "$LAUNCH" --status
  [[ $output == *"hfuser/drbm25-qwen2.5-3b-normal: not uploaded"* ]]
  [[ $output == *"safe to delete the volume: no"* ]]
}

@test "--status with a pod whose exec fails (409) says unknown" {
  hub_marker normal 400 aaa
  hub_marker count_once 350 bbb
  running_pod ''
  resp_status POST_v2_pods_pod7_exec 409
  run -0 bash "$LAUNCH" --status
  [[ $output == *"safe to delete the volume: unknown"* ]]
}

@test "the watchdog margin is at least the publish timeout, so it never fires inside a budgeted upload" {
  local margin publish
  margin=$(sed -n 's/^readonly HARD_DEADLINE_MARGIN_S=\([0-9]*\).*/\1/p' "$LAUNCH")
  publish=$(sed -n 's/^readonly PUBLISH_TIMEOUT_S=\([0-9]*\).*/\1/p' "${BATS_TEST_DIRNAME}/../../scripts/runpod/lib.sh")
  [[ -n $margin && -n $publish ]]
  ((margin >= publish))
}

# ---------------------------------------------------------------- usage

@test "launch.sh --help exits 0 and documents --yes, --status, --logs, --watch and --stop" {
  run -0 bash "$LAUNCH" --help
  [[ $output == *--yes* && $output == *--status* && $output == *--logs* && $output == *--watch* && $output == *--stop* ]]
  [[ $output == *"--stop --yes"* ]]
}

@test "launch.sh rejects an unknown option with status 2" {
  run -2 bash "$LAUNCH" --bogus
  [[ $output == *--bogus* ]]
}

@test "launch.sh rejects --cloud community: network volumes need the secure cloud" {
  run -2 launch --cloud community --yes
  [[ $output == *"Secure Cloud"* ]]
  no_billable_calls
}

@test "launch.sh rejects a non-numeric --volume-gb with status 2" {
  run -2 launch --volume-gb lots --yes
  no_billable_calls
}

# ---------------------------------------------------------------- the main path

@test "with the volume in US-MD-1 it picks the cheapest stocked 2x A100 there and prints price and data center" {
  run -0 launch --yes
  [[ $output == *'$4.38/h'* ]]
  [[ $output == *US-MD-1* ]]
  [[ $output == *"NVIDIA A100-SXM4-80GB"* ]]
  run -0 body_of POST_v2_pods
  [[ $(jq -r '.gpu.id' <<<"$output") == "NVIDIA A100-SXM4-80GB" ]]
  [[ $(jq -r '.gpu.count' <<<"$output") == 2 ]]
  [[ $(jq -r '.cloud' <<<"$output") == SECURE ]]
  [[ $(jq -c '.dataCenterIds' <<<"$output") == '["US-MD-1"]' ]]
  [[ $(jq -c '.mounts.network' <<<"$output") == '[{"volumeId":"vol1","path":"/workspace"}]' ]]
  [[ $(jq -r '.env.PRICE_PER_HOUR' <<<"$output") == 4.38 ]]
  [[ $(jq -r '.env.VOLUME_GB' <<<"$output") == 400 ]]
  ! grep -q '^POST https://api.runpod.io/v2/network-volumes' "$STUB_DIR/calls.log"
}

@test "the availability query asks for 2 GPUs, pods, the secure cloud and CUDA 12.8 or newer" {
  run -0 launch --yes
  grep -q '^GET https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD&count=2&cloud=SECURE&minCudaVersion=12.8$' "$STUB_DIR/calls.log"
}

@test "the pod body references RunPod secrets and never holds a secret value" {
  run -0 launch --yes
  run -0 body_of POST_v2_pods
  [[ $output != *"$HF"* && $output != *"$WB"* && $output != *"$RP"* ]]
  [[ $(jq -r '.env.HF_TOKEN' <<<"$output") == '{{ RUNPOD_SECRET_drbm25_hf_token }}' ]]
  [[ $(jq -r '.env.WANDB_API_KEY' <<<"$output") == '{{ RUNPOD_SECRET_drbm25_wandb_api_key }}' ]]
  [[ $(jq -r '.env.DRBM25_RUNPOD_API_KEY' <<<"$output") == '{{ RUNPOD_SECRET_drbm25_runpod_api_key }}' ]]
}

@test "the pod env carries the non-secret settings for an unattended run" {
  run -0 launch --yes
  run -0 body_of POST_v2_pods
  [[ $(jq -r '.env.WANDB_PROJECT' <<<"$output") == drbm25 ]]
  [[ $(jq -r '.env.WANDB_ENTITY' <<<"$output") == '' ]]
  [[ $(jq -r '.env.BUDGET_USD' <<<"$output") == 150 ]]
  [[ $(jq -r '.env.TERMINATE_ON_FINISH' <<<"$output") == yes ]]
  [[ $(jq -r '.env.DRBM25_NONINTERACTIVE' <<<"$output") == 1 ]]
  [[ $(jq -r '.env.DRBM25_REPO_URL' <<<"$output") == https://github.com/objones25/drbm25 ]]
  [[ $(jq -r '.env.DRBM25_REF' <<<"$output") == feat/test ]]
  [[ $(jq -r '.env.DRBM25_CLEAR_FAILED' <<<"$output") == 0 ]]
}

@test "the pod runs the official PyTorch image with ports 22 and 8888 and pod_bootstrap.sh as its command" {
  run -0 launch --yes
  run -0 body_of POST_v2_pods
  [[ $(jq -r '.image' <<<"$output") == runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 ]]
  [[ $(jq -c '.ports' <<<"$output") == '["8888/http","22/tcp"]' ]]
  [[ $(jq -c '.entrypoint' <<<"$output") == '["/bin/bash","-c"]' ]]
  [[ $(jq -r '.cmd[0]' <<<"$output") == "$(cat "$BOOTSTRAP")" ]]
  (($(jq -r '.disk' <<<"$output") >= 50))
}

@test "secrets go to the secrets API: POST when missing, PATCH when the name exists" {
  resp GET_v2_account_secrets '{"secrets":[{"id":"sec1","name":"DRBM25_HF_TOKEN"}]}'
  run -0 launch --yes
  grep -q '^PATCH https://api.runpod.io/v2/account/secrets/sec1$' "$STUB_DIR/calls.log"
  [[ $(grep -c '^POST https://api.runpod.io/v2/account/secrets$' "$STUB_DIR/calls.log") == 2 ]]
  run -0 body_of PATCH_v2_account_secrets_sec1
  [[ $(jq -r '.value' <<<"$output") == "$HF" ]]
  cat "$STUB_DIR"/bodies/*-POST_v2_account_secrets.json | jq -r '.name + "=" + .value' >"$BATS_TEST_TMPDIR/posted"
  grep -qx "drbm25_wandb_api_key=$WB" "$BATS_TEST_TMPDIR/posted"
  grep -qx "drbm25_runpod_api_key=$RP" "$BATS_TEST_TMPDIR/posted"
}

@test "no secret ever appears in curl's arguments" {
  run -0 launch --yes
  [[ -s $STUB_DIR/argv.log ]]
  ! grep -q -e "$HF" -e "$WB" -e "$RP" "$STUB_DIR/argv.log"
}

@test "the auth header file is mode 600 and deleted when launch.sh exits" {
  run -0 launch --yes
  [[ $(sort -u "$STUB_DIR/auth_modes") == '-rw-------' ]]
  while IFS= read -r f; do [[ ! -e $f ]]; done <"$STUB_DIR/auth_paths"
  [[ -z $(ls -A "$TMPDIR") ]]
}

@test "no secret appears in launch.sh's output" {
  run -0 launch --yes
  [[ $output != *"$HF"* && $output != *"$WB"* && $output != *"$RP"* ]]
}

@test "it prints the pod id, the console URL and how to follow the logs" {
  run -0 launch --yes
  [[ $output == *pod123* ]]
  [[ $output == *https://console.runpod.io/pods* ]]
  [[ $output == *--logs* ]]
  [[ $output == */workspace/drbm25/logs* ]]
}

@test "a pod cost above the quoted price is a loud warning" {
  resp POST_v2_pods '{"id":"pod123","status":"PROVISIONING","cost":5.10,"dataCenterId":"US-MD-1"}'
  run -0 launch --yes
  [[ $output == *WARNING* && $output == *5.1* ]]
}

# ---------------------------------------------------------------- confirmation

@test "without --yes, an answer other than yes creates nothing billable and changes no secret" {
  printf 'y\n' >"$DRBM25_TTY"
  run -1 launch
  [[ $output == *'$4.38/h'* ]]
  no_billable_calls
  ! grep -q '^POST https://api.runpod.io/v2/account/secrets' "$STUB_DIR/calls.log"
  ! grep -q '^PATCH' "$STUB_DIR/calls.log"
}

@test "without --yes, typing yes creates the pod" {
  printf 'yes\n' >"$DRBM25_TTY"
  run -0 launch
  grep -q '^POST https://api.runpod.io/v2/pods$' "$STUB_DIR/calls.log"
}

@test "without --yes and with no terminal to ask, nothing billable is created" {
  run -1 env -u DRBM25_TTY bash "$LAUNCH" --ref feat/test </dev/null
  no_billable_calls
}

# ---------------------------------------------------------------- volume logic

@test "an existing volume whose data center has no stock stops the launch and names the data center" {
  resp GET_v2_network-volumes '{"networkVolumes":[{"id":"vol1","name":"drbm25","size":400,"dataCenter":"CA-MTL-1","type":"STANDARD"}],"pagination":{"hasNextPage":false}}'
  run -1 launch --yes
  [[ $output == *CA-MTL-1* ]]
  no_billable_calls
}

@test "--gpu pcie with the volume in a data center that has only SXM stops instead of creating a second volume" {
  run -1 launch --gpu pcie --yes
  [[ $output == *US-MD-1* ]]
  no_billable_calls
}

@test "with no volume it creates one of --volume-gb in the cheapest data center, after showing size and monthly cost" {
  resp GET_v2_network-volumes '{"networkVolumes":[],"pagination":{"hasNextPage":false}}'
  printf 'yes\n' >"$DRBM25_TTY"
  run -0 launch --volume-gb 500
  [[ $output == *500* && $output == *'$35.00/month'* ]]
  [[ $output == *US-KS-2* && $output == *'$4.18/h'* ]]
  run -0 body_of POST_v2_network-volumes
  [[ $(jq -c . <<<"$output") == '{"name":"drbm25","size":500,"dataCenter":"US-KS-2","type":"STANDARD"}' ]]
  run -0 body_of POST_v2_pods
  [[ $(jq -r '.mounts.network[0].volumeId' <<<"$output") == volNEW ]]
  [[ $(jq -r '.gpu.id' <<<"$output") == "NVIDIA A100 80GB PCIe" ]]
  [[ $(jq -r '.env.VOLUME_GB' <<<"$output") == 400 ]]
}

@test "the volume is created before the pod" {
  resp GET_v2_network-volumes '{"networkVolumes":[],"pagination":{"hasNextPage":false}}'
  run -0 launch --yes
  run -0 grep -n '^POST https://api.runpod.io/v2/\(network-volumes\|pods\)$' "$STUB_DIR/calls.log"
  [[ ${lines[0]} == *network-volumes* && ${lines[1]} == *pods* ]]
}

@test "two volumes named drbm25 stop the launch" {
  resp GET_v2_network-volumes '{"networkVolumes":[{"id":"a","name":"drbm25","size":400,"dataCenter":"US-MD-1"},{"id":"b","name":"drbm25","size":400,"dataCenter":"US-KS-2"}],"pagination":{"hasNextPage":false}}'
  run -1 launch --yes
  [[ $output == *a* && $output == *b* ]]
  no_billable_calls
}

@test "an existing drbm25 pod stops the launch so two pods never share the volume" {
  resp GET_v2_pods '{"pods":[{"id":"podOLD","name":"drbm25","status":"RUNNING","cost":4.38}],"pagination":{"hasNextPage":false}}'
  run -1 launch --yes
  [[ $output == *podOLD* ]]
  no_billable_calls
}

@test "--clear-failed sets DRBM25_CLEAR_FAILED=1 in the pod env" {
  run -0 launch --yes --clear-failed
  run -0 body_of POST_v2_pods
  [[ $(jq -r '.env.DRBM25_CLEAR_FAILED' <<<"$output") == 1 ]]
}

# ---------------------------------------------------------------- settings

@test "a missing HF_TOKEN with no terminal fails before any API write" {
  run -1 env -u HF_TOKEN -u DRBM25_TTY bash "$LAUNCH" --ref feat/test --yes </dev/null
  [[ $output == *HF_TOKEN* ]]
  ! grep -q '^\(POST\|PATCH\|DELETE\)' "$STUB_DIR/calls.log"
}

@test "secrets are read from the terminal hidden and never echoed" {
  printf '%s\n' "$HF" >"$DRBM25_TTY"
  run -0 env -u HF_TOKEN bash "$LAUNCH" --ref feat/test --yes
  [[ $output == *HF_TOKEN* ]]
  [[ $output != *"$HF"* ]]
  run -0 body_of POST_v2_account_secrets
}

@test "a non-numeric budget is rejected before any API write" {
  run -1 env BUDGET_USD=lots bash "$LAUNCH" --ref feat/test --yes
  [[ $output == *BUDGET_USD* ]]
  ! grep -q '^\(POST\|PATCH\|DELETE\)' "$STUB_DIR/calls.log"
}

@test "a --ref with shell metacharacters is rejected" {
  run -2 bash "$LAUNCH" --ref 'main;rm -rf /' --yes
  no_billable_calls
}

# ---------------------------------------------------------------- API errors

@test "a 429 on a read is retried after Retry-After" {
  resp_status GET_v2_network-volumes 429 200
  run -0 launch --yes
  [[ $(grep -c '^GET https://api.runpod.io/v2/network-volumes' "$STUB_DIR/calls.log") == 2 ]]
  grep -q '^sleep 3$' "$STUB_DIR/calls.log"
}

@test "a read that keeps failing gives up after a bounded number of attempts" {
  resp_status GET_v2_network-volumes 503
  run -1 launch --yes
  (($(grep -c '^GET https://api.runpod.io/v2/network-volumes' "$STUB_DIR/calls.log") <= 5))
  [[ $output == *503* ]]
  no_billable_calls
}

@test "a 5xx on pod create is not retried (it may have been created) and says to check --status" {
  resp_status POST_v2_pods 502
  resp POST_v2_pods '{"title":"Bad Gateway","status":502,"detail":"upstream"}'
  run -1 launch --yes
  [[ $(grep -c '^POST https://api.runpod.io/v2/pods$' "$STUB_DIR/calls.log") == 1 ]]
  [[ $output == *--status* ]]
}

@test "a 400 on pod create shows RunPod's detail" {
  resp_status POST_v2_pods 400
  resp POST_v2_pods '{"title":"Bad Request","status":400,"detail":"no capacity for this gpu"}'
  run -1 launch --yes
  [[ $output == *"no capacity for this gpu"* ]]
}

# ---------------------------------------------------------------- status, logs, stop, watch

@test "--status prints the drbm25 pod's id, status and hourly cost" {
  resp GET_v2_pods '{"pods":[{"id":"pod7","name":"drbm25","status":"RUNNING","cost":4.38,"dataCenterId":"US-MD-1"},{"id":"other","name":"else","status":"RUNNING","cost":1}],"pagination":{"hasNextPage":false}}'
  run -0 bash "$LAUNCH" --status
  [[ $output == *pod7* && $output == *RUNNING* && $output == *4.38* ]]
  [[ $output != *other* ]]
}

@test "--status with no pod says so and exits 0" {
  run -0 bash "$LAUNCH" --status
  [[ $output == *"no drbm25 pod"* ]]
}

@test "--logs runs a fixed tail command in the pod through the exec API and prints its output" {
  resp GET_v2_pods '{"pods":[{"id":"pod7","name":"drbm25","status":"RUNNING","cost":4.38}],"pagination":{"hasNextPage":false}}'
  resp POST_v2_pods_pod7_exec '{"stdout":"== logs/run_all-1.log\nstage training\n","stderr":"","exitCode":0,"timedOut":false,"truncated":false}'
  resp_status POST_v2_pods_pod7_exec 200
  run -0 bash "$LAUNCH" --logs
  [[ $output == *"stage training"* ]]
  run -0 body_of POST_v2_pods_pod7_exec
  [[ $(jq -r '.command[0]' <<<"$output") == sh ]]
  [[ $(jq -r '.env.DRBM25_TAIL_LINES' <<<"$output") == 40 ]]
}

@test "--stop terminates the pod after yes and never deletes the volume" {
  resp GET_v2_pods '{"pods":[{"id":"pod7","name":"drbm25","status":"RUNNING","cost":4.38}],"pagination":{"hasNextPage":false}}'
  printf 'yes\n' >"$DRBM25_TTY"
  run -0 bash "$LAUNCH" --stop
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod7$' "$STUB_DIR/calls.log"
  ! grep -q 'network-volumes/' "$STUB_DIR/calls.log"
}

@test "--stop without yes terminates nothing" {
  resp GET_v2_pods '{"pods":[{"id":"pod7","name":"drbm25","status":"RUNNING","cost":4.38}],"pagination":{"hasNextPage":false}}'
  printf 'no\n' >"$DRBM25_TTY"
  run -1 bash "$LAUNCH" --stop
  ! grep -q '^DELETE' "$STUB_DIR/calls.log"
}

@test "--watch polls until the pod is gone, printing log lines while it runs" {
  resp GET_v2_pods '{"pods":[{"id":"pod7","name":"drbm25","status":"RUNNING","cost":4.38}],"pagination":{"hasNextPage":false}}'
  resp POST_v2_pods_pod7_exec '{"stdout":"stage training\n","stderr":"","exitCode":0,"timedOut":false,"truncated":false}'
  resp_status POST_v2_pods_pod7_exec 200
  # After the first sleep the listing finds no pod: the run finished and terminated it.
  cat >"$BATS_TEST_TMPDIR/bin/sleep_stub" <<'EOF'
#!/usr/bin/env bash
echo "sleep $*" >>"$STUB_DIR/calls.log"
printf '{"pods":[],"pagination":{"hasNextPage":false}}\n' >"$STUB_DIR/resp/GET_v2_pods.json"
EOF
  run -0 bash "$LAUNCH" --watch --interval-min 5
  [[ $output == *"stage training"* ]]
  [[ $output == *"no drbm25 pod"* ]]
  grep -q '^sleep 300$' "$STUB_DIR/calls.log"
}

# ---------------------------------------------------------------- pod_bootstrap.sh

make_fixture_repo() { # a git repo whose run_all.sh records how it was started
  local src=$BATS_TEST_TMPDIR/src
  mkdir -p "$src/scripts/runpod"
  cat >"$src/scripts/runpod/run_all.sh" <<'EOF'
#!/usr/bin/env bash
{
  echo "version=$(cat VERSION)"
  echo "noninteractive=$DRBM25_NONINTERACTIVE"
  echo "runpod_key=$RUNPOD_API_KEY"
  echo "pod_start=$DRBM25_POD_START_UNIX"
  if [[ -t 0 ]]; then echo stdin=tty; else echo "stdin=$(cat)"; fi
} >>"$STUB_DIR/run_all.out"
EOF
  echo v1 >"$src/VERSION"
  git -C "$src" init -q -b main
  git -C "$src" add -A
  git -C "$src" -c user.name=t -c user.email=t@t commit -q -m v1
  FIXTURE_REPO=$src
}

bootstrap() {
  env DRBM25_ROOT="$BATS_TEST_TMPDIR/ws/drbm25" DRBM25_START_SH="$BATS_TEST_TMPDIR/start.sh" \
    DRBM25_REPO_URL="file://$FIXTURE_REPO" DRBM25_REF=main DRBM25_RUNPOD_API_KEY=$RP \
    RUNPOD_POD_ID=pod123 "$@" bash -c "$(cat "$BOOTSTRAP")"
}

@test "pod_bootstrap.sh starts the image's services, clones the ref and runs run_all.sh unattended" {
  make_fixture_repo
  printf '#!/usr/bin/env bash\necho started >>"$STUB_DIR/start.out"\n' >"$BATS_TEST_TMPDIR/start.sh"
  chmod +x "$BATS_TEST_TMPDIR/start.sh"
  run -0 bootstrap
  grep -qx started "$STUB_DIR/start.out"
  grep -qx version=v1 "$STUB_DIR/run_all.out"
  grep -qx noninteractive=1 "$STUB_DIR/run_all.out"
  grep -qx "runpod_key=$RP" "$STUB_DIR/run_all.out"
  grep -qx 'stdin=' "$STUB_DIR/run_all.out"
  grep -q '^pod_start=[0-9][0-9]*$' "$STUB_DIR/run_all.out"
  ls "$BATS_TEST_TMPDIR"/ws/drbm25/logs/bootstrap-*.log
}

@test "pod_bootstrap.sh reuses an existing clone: fetches the ref and checks it out" {
  make_fixture_repo
  run -0 bootstrap
  echo v2 >"$FIXTURE_REPO/VERSION"
  git -C "$FIXTURE_REPO" -c user.name=t -c user.email=t@t commit -q -am v2
  run -0 bootstrap
  [[ $(sed -n 's/^version=//p' "$STUB_DIR/run_all.out" | tr '\n' ' ') == 'v1 v2 ' ]]
}

@test "pod_bootstrap.sh accepts a commit sha as the ref" {
  make_fixture_repo
  local sha
  sha=$(git -C "$FIXTURE_REPO" rev-parse HEAD)
  run -0 bootstrap DRBM25_REF="$sha"
  grep -qx version=v1 "$STUB_DIR/run_all.out"
}

@test "pod_bootstrap.sh terminates the pod when the clone fails, keeping the key out of argv" {
  make_fixture_repo
  run -1 bootstrap DRBM25_REPO_URL="file://$BATS_TEST_TMPDIR/nope"
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod123$' "$STUB_DIR/calls.log"
  ! grep -q "$RP" "$STUB_DIR/argv.log"
  [[ ! -f $STUB_DIR/run_all.out ]]
}

@test "pod_bootstrap.sh does not terminate on a failed clone when TERMINATE_ON_FINISH=no" {
  make_fixture_repo
  run -1 bootstrap DRBM25_REPO_URL="file://$BATS_TEST_TMPDIR/nope" TERMINATE_ON_FINISH=no
  ! grep -q '^DELETE' "$STUB_DIR/calls.log"
}

@test "pod_bootstrap.sh terminates the pod when DRBM25_REF is missing" {
  make_fixture_repo
  run -1 bootstrap DRBM25_REF=
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod123$' "$STUB_DIR/calls.log"
}

# ---------------------------------------------------------------- fix round 1

@test "--stop --yes terminates without asking" {
  resp GET_v2_pods '{"pods":[{"id":"pod7","name":"drbm25","status":"RUNNING","cost":4.38}],"pagination":{"hasNextPage":false}}'
  run -0 env -u DRBM25_TTY bash "$LAUNCH" --stop --yes </dev/null
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod7$' "$STUB_DIR/calls.log"
}

@test "the pod env carries a hard deadline: now + budget / price hours + 1 h" {
  local before after lo hi got
  before=$(date +%s)
  run -0 launch --yes
  after=$(date +%s)
  run -0 body_of POST_v2_pods
  got=$(jq -r '.["env"]["DRBM25_HARD_DEADLINE_UNIX"]' <<<"$output")
  # 150 / 4.38 h = 123287.67 s, plus 3600 s
  lo=$((before + 123287 + 3600 - 2))
  hi=$((after + 123288 + 3600 + 2))
  ((got >= lo && got <= hi))
}

@test "a new volume only goes to a data center offering STANDARD volumes, checked before confirming" {
  resp GET_v2_network-volumes '{"networkVolumes":[],"pagination":{"hasNextPage":false}}'
  resp GET_v2_catalog_datacenters '{"dataCenters":[
    {"id":"US-MD-1","networkVolumeTypes":["STANDARD"]},{"id":"EU-RO-1","networkVolumeTypes":["STANDARD"]},
    {"id":"US-KS-2","networkVolumeTypes":["HIGH_PERFORMANCE"]}]}'
  run -0 launch --yes
  [[ $output != *US-KS-2* ]]
  run -0 body_of POST_v2_network-volumes
  [[ $(jq -r '.dataCenter' <<<"$output") == EU-RO-1 ]]
}

@test "with no stocked data center offering STANDARD volumes it stops before confirming or writing secrets" {
  resp GET_v2_network-volumes '{"networkVolumes":[],"pagination":{"hasNextPage":false}}'
  resp GET_v2_catalog_datacenters '{"dataCenters":[{"id":"US-MD-1","networkVolumeTypes":["HIGH_PERFORMANCE"]},
    {"id":"EU-RO-1","networkVolumeTypes":[]},{"id":"US-KS-2","networkVolumeTypes":["HIGH_PERFORMANCE"]}]}'
  printf 'yes\n' >"$DRBM25_TTY"
  run -1 launch
  [[ $output == *STANDARD* && $output == *HIGH_PERFORMANCE* ]]
  [[ $output != *"Type yes"* ]]
  ! grep -q '^\(POST\|PATCH\|DELETE\)' "$STUB_DIR/calls.log"
}

@test "a ref that is not on origin (git ls-remote) stops the launch before any API write" {
  : >"$STUB_DIR/ls_remote"
  run -1 launch --yes
  [[ $output == *feat/test* && $output == *push* ]]
  grep -q '^git ls-remote https://github.com/objones25/drbm25' "$STUB_DIR/calls.log"
  ! grep -q '^\(POST\|PATCH\|DELETE\)' "$STUB_DIR/calls.log"
}

@test "the default ref is HEAD's sha when it is a branch tip on origin" {
  local sha
  sha=$(git -C "$BATS_TEST_DIRNAME/../.." rev-parse HEAD)
  printf '%s\trefs/heads/some-branch\n' "$sha" >"$STUB_DIR/ls_remote"
  run -0 bash "$LAUNCH" --yes
  run -0 body_of POST_v2_pods
  [[ $(jq -r '.["env"]["DRBM25_REF"]' <<<"$output") == "$sha" ]]
}

@test "the default ref is refused when HEAD is not a branch or tag tip on origin" {
  run -1 bash "$LAUNCH" --yes
  [[ $output == *push* ]]
  no_billable_calls
}

bootstrap_bounded() { # like bootstrap, killed after 60 s so a hang fails the test instead of blocking it
  perl -e 'alarm 60; exec @ARGV' env DRBM25_ROOT="${BOOT_ROOT:-$BATS_TEST_TMPDIR/ws/drbm25}" \
    DRBM25_START_SH="$BATS_TEST_TMPDIR/start.sh" DRBM25_REPO_URL="file://$FIXTURE_REPO" DRBM25_REF=main \
    DRBM25_RUNPOD_API_KEY=$RP RUNPOD_API_KEY=pod_scoped_key RUNPOD_POD_ID=pod123 "$@" bash -c "$(cat "$BOOTSTRAP")"
}

@test "bootstrap: an unreplaced user secret reference falls back to the pod-scoped key for DELETE" {
  make_fixture_repo
  run -1 bootstrap_bounded DRBM25_RUNPOD_API_KEY='{{ RUNPOD_SECRET_drbm25_runpod_api_key }}' \
    DRBM25_REPO_URL="file://$BATS_TEST_TMPDIR/nope"
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod123$' "$STUB_DIR/calls.log"
  grep -qx 'Authorization: Bearer pod_scoped_key' "$STUB_DIR/auth_stdin.log"
  ! grep -q RUNPOD_SECRET "$STUB_DIR/auth_stdin.log"
  ! grep -q pod_scoped_key "$STUB_DIR/argv.log"
}

@test "bootstrap: when the user key's DELETE fails it retries with the pod-scoped key" {
  make_fixture_repo
  resp_status DELETE_v2_pods_pod123 401 204
  run -1 bootstrap_bounded DRBM25_REPO_URL="file://$BATS_TEST_TMPDIR/nope"
  [[ $(sed -n 1p "$STUB_DIR/auth_stdin.log") == "Authorization: Bearer $RP" ]]
  [[ $(sed -n 2p "$STUB_DIR/auth_stdin.log") == 'Authorization: Bearer pod_scoped_key' ]]
}

@test "bootstrap: the run's environment carries the pod-scoped key for run_all.sh's terminate fallback" {
  make_fixture_repo
  printf 'echo "pod_key=$DRBM25_POD_RUNPOD_API_KEY" >>"$STUB_DIR/run_all.out"\n' >>"$FIXTURE_REPO/scripts/runpod/run_all.sh"
  git -C "$FIXTURE_REPO" -c user.name=t -c user.email=t@t commit -q -am pod-key
  run -0 bootstrap_bounded
  grep -qx 'pod_key=pod_scoped_key' "$STUB_DIR/run_all.out"
}

@test "bootstrap: the watchdog terminates the pod at the hard deadline while run_all.sh is still running" {
  make_fixture_repo
  # run_all.sh "trains" until it sees the DELETE (at most 20 s).
  cat >"$FIXTURE_REPO/scripts/runpod/run_all.sh" <<'EOF'
#!/usr/bin/env bash
for i in $(seq 1 40); do
  if grep -q '^DELETE' "$STUB_DIR/calls.log"; then echo saw_delete >>"$STUB_DIR/run_all.out"; exit 0; fi
  sleep 0.5
done
echo no_delete >>"$STUB_DIR/run_all.out"
EOF
  git -C "$FIXTURE_REPO" -c user.name=t -c user.email=t@t commit -q -am slow
  run bootstrap_bounded DRBM25_HARD_DEADLINE_UNIX=$(($(date +%s) - 1))
  grep -qx saw_delete "$STUB_DIR/run_all.out"
  grep -q 'watchdog: hard deadline' "$BATS_TEST_TMPDIR"/ws/drbm25/logs/bootstrap-*.log
}

@test "bootstrap: the watchdog stays quiet before the deadline and is stopped when the run ends" {
  make_fixture_repo
  run -0 bootstrap_bounded DRBM25_HARD_DEADLINE_UNIX=$(($(date +%s) + 3600)) TERMINATE_ON_FINISH=no
  ! grep -q '^DELETE' "$STUB_DIR/calls.log"
  local pid i
  pid=$(sed -n 's/.*watchdog started (pid \([0-9]*\).*/\1/p' "$BATS_TEST_TMPDIR"/ws/drbm25/logs/bootstrap-*.log)
  [[ -n $pid ]]
  for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.3; done
  ! kill -0 "$pid" 2>/dev/null
}

@test "bootstrap: an unwritable volume still reaches terminate, logging to TMPDIR" {
  make_fixture_repo
  : >"$BATS_TEST_TMPDIR/not-a-dir"
  BOOT_ROOT=$BATS_TEST_TMPDIR/not-a-dir/drbm25 run -1 bootstrap_bounded
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod123$' "$STUB_DIR/calls.log"
  ls "$TMPDIR"/drbm25-bootstrap-*.log
  grep -q 'not writable' "$TMPDIR"/drbm25-bootstrap-*.log
}

@test "bootstrap: when terminate keeps failing it retries a bounded number of times and exits, never waiting forever" {
  make_fixture_repo
  resp_status DELETE_v2_pods_pod123 500
  # The image's services would run forever; the bootstrap must not wait on them.
  printf '#!/usr/bin/env bash\necho $$ >"$STUB_DIR/start.pid"\nexec sleep 120\n' >"$BATS_TEST_TMPDIR/start.sh"
  chmod +x "$BATS_TEST_TMPDIR/start.sh"
  # DRBM25_WATCHDOG_ROUNDS shortens the watchdog's 48 h of retries to 12 rounds (sleep is a stub here).
  run bootstrap_bounded DRBM25_HARD_DEADLINE_UNIX=$(($(date +%s) + 3600)) DRBM25_WATCHDOG_ROUNDS=12
  kill "$(cat "$STUB_DIR/start.pid")" 2>/dev/null || true
  ((status != 0 && status != 142)) # 142: killed by the 60 s alarm, i.e. a hang
  (($(grep -c '^DELETE' "$STUB_DIR/calls.log") > 2))
  grep -q 'ERROR: could not terminate' "$BATS_TEST_TMPDIR"/ws/drbm25/logs/bootstrap-*.log
}

@test "bootstrap: with neither the volume nor TMPDIR writable, terminate is still called" {
  make_fixture_repo
  : >"$BATS_TEST_TMPDIR/not-a-dir"
  BOOT_ROOT=$BATS_TEST_TMPDIR/not-a-dir/drbm25 TMPDIR=$BATS_TEST_TMPDIR/not-a-dir/tmp run -1 bootstrap_bounded
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod123$' "$STUB_DIR/calls.log"
}

# ---------------------------------------------------------------- fix round 2

@test "the pod env carries absolute deadlines and a launch id" {
  local before after budget hard id
  before=$(date +%s)
  run -0 launch --yes
  after=$(date +%s)
  run -0 body_of POST_v2_pods
  budget=$(jq -r '.["env"]["DRBM25_BUDGET_DEADLINE_UNIX"]' <<<"$output")
  hard=$(jq -r '.["env"]["DRBM25_HARD_DEADLINE_UNIX"]' <<<"$output")
  id=$(jq -r '.["env"]["DRBM25_LAUNCH_ID"]' <<<"$output")
  # 150 / 4.38 h = 123287.67 s
  ((budget >= before + 123287 - 1 && budget <= after + 123288 + 1))
  ((hard == budget + 3600))
  [[ $id =~ ^[A-Za-z0-9_-]{8,}$ ]]
}

@test "two launches get different launch ids" {
  run -0 launch --yes
  local first
  first=$(body_of POST_v2_pods | jq -r '.["env"]["DRBM25_LAUNCH_ID"]')
  run -0 launch --yes
  [[ $(body_of POST_v2_pods | jq -r '.["env"]["DRBM25_LAUNCH_ID"]') != "$first" ]]
}

@test "bootstrap writes the launch's done marker when run_all.sh returns" {
  make_fixture_repo
  run -0 bootstrap_bounded DRBM25_LAUNCH_ID=L2
  [[ -f $BATS_TEST_TMPDIR/ws/drbm25/.launch-L2-done ]]
}

@test "a restarted bootstrap after its launch finished does not run run_all.sh again, and terminates" {
  make_fixture_repo
  mkdir -p "$BATS_TEST_TMPDIR/ws/drbm25"
  date >"$BATS_TEST_TMPDIR/ws/drbm25/.launch-L1-done"
  run bootstrap_bounded DRBM25_LAUNCH_ID=L1
  [[ ! -f $STUB_DIR/run_all.out ]]
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod123$' "$STUB_DIR/calls.log"
}

@test "another launch's done marker does not stop this launch" {
  make_fixture_repo
  mkdir -p "$BATS_TEST_TMPDIR/ws/drbm25"
  date >"$BATS_TEST_TMPDIR/ws/drbm25/.launch-OLD-done"
  run -0 bootstrap_bounded DRBM25_LAUNCH_ID=NEW
  grep -qx version=v1 "$STUB_DIR/run_all.out"
}

@test "a restarted bootstrap past the budget deadline minus the reserve does not run run_all.sh, and terminates" {
  make_fixture_repo
  run bootstrap_bounded DRBM25_LAUNCH_ID=L3 DRBM25_BUDGET_DEADLINE_UNIX=$(($(date +%s) + 600))
  [[ ! -f $STUB_DIR/run_all.out ]]
  grep -q '^DELETE https://api.runpod.io/v2/pods/pod123$' "$STUB_DIR/calls.log"
}

@test "bootstrap runs run_all.sh when the budget deadline is more than the reserve away" {
  make_fixture_repo
  run -0 bootstrap_bounded DRBM25_LAUNCH_ID=L4 DRBM25_BUDGET_DEADLINE_UNIX=$(($(date +%s) + 7200))
  grep -qx version=v1 "$STUB_DIR/run_all.out"
}

@test "the watchdog keeps retrying past 8 rounds until terminate succeeds" {
  make_fixture_repo
  # The bootstrap's own attempt (user key, pod key) and 10 watchdog rounds fail; the 11th round's first call works.
  local i
  for i in $(seq 1 22); do echo 500; done >"$STUB_DIR/resp/DELETE_v2_pods_pod123.status"
  echo 204 >>"$STUB_DIR/resp/DELETE_v2_pods_pod123.status"
  run bootstrap_bounded DRBM25_REPO_URL="file://$BATS_TEST_TMPDIR/nope" DRBM25_HARD_DEADLINE_UNIX=$(($(date +%s) + 3600))
  [[ $(grep -c '^DELETE' "$STUB_DIR/calls.log") == 23 ]]
  grep -q 'terminated pod pod123' "$BATS_TEST_TMPDIR"/ws/drbm25/logs/bootstrap-*.log
}

@test "the watchdog waits 10 minutes between late retries" {
  make_fixture_repo
  resp_status DELETE_v2_pods_pod123 500
  run bootstrap_bounded DRBM25_REPO_URL="file://$BATS_TEST_TMPDIR/nope" DRBM25_HARD_DEADLINE_UNIX=$(($(date +%s) + 3600)) \
    DRBM25_WATCHDOG_ROUNDS=10
  grep -q '^sleep 600$' "$STUB_DIR/calls.log"
  ! grep -q '^sleep \(6[0-9][1-9]\|[7-9][0-9][0-9]\|[0-9]\{4,\}\)$' "$STUB_DIR/calls.log"
}
