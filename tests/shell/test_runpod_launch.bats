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
method=GET out='' hdr='' body='' auth='' url=''
while (($#)); do
  case $1 in
    -X) method=$2; shift ;;
    -o) out=$2; shift ;;
    -D) hdr=$2; shift ;;
    -w) shift ;;
    --data-binary) body=${2#@}; shift ;;
    -H) [[ $2 == @* ]] && auth=${2#@}; shift ;;
    https://*) url=$1 ;;
  esac
  shift
done
path=${url#https://api.runpod.io}
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
EOF
}

resp() { # KEY JSON: the body returned for KEY (for example GET_v2_pods)
  printf '%s\n' "$2" >"$STUB_DIR/resp/$1.json"
}

resp_status() { # KEY STATUS...: statuses returned for KEY, in order
  printf '%s\n' "${@:2}" >"$STUB_DIR/resp/$1.status"
}

default_responses() {
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

# ---------------------------------------------------------------- usage

@test "launch.sh --help exits 0 and documents --yes, --status, --logs, --watch and --stop" {
  run -0 bash "$LAUNCH" --help
  [[ $output == *--yes* && $output == *--status* && $output == *--logs* && $output == *--watch* && $output == *--stop* ]]
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
