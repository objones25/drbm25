#!/usr/bin/env bash
#
# prepare_login.sh: one-time setup on an Expanse LOGIN node (compute nodes may have no
# internet). Idempotent: every step is skipped or re-verified when its output exists.
#
# Usage: see usage() below, or run with --help.
# Requires: bash 4.0+ (Expanse), uv on PATH, curl, tar, EXPANSE_ACCOUNT in the environment.
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly REPO_ROOT=$SCRIPT_DIR/../..
readonly CONFIG=configs/expanse_v100_lora.yaml

# JDK_RELEASE, JDK_URL and JDK_SHA256 (Eclipse Temurin 21, Linux x64) come from common.sh.
readonly DOWNLOAD_TIMEOUT_S=900

log() {
  local now
  now=$(date '+%Y-%m-%dT%H:%M:%S')
  printf '%s %s: %s\n' "$now" "$SCRIPT_NAME" "$*" >&2
}
die() {
  log "error: $*"
  exit 1
}

usage() {
  cat <<EOF2
Usage: $SCRIPT_NAME [-h|--help]

Run once on an Expanse login node, from any directory. Needs
EXPANSE_ACCOUNT in the environment (and HF_TOKEN if the model or dataset needs it).
Everything goes under /expanse/lustre/projects/\$EXPANSE_ACCOUNT/\$USER/drbm25
(override with DRBM25_ROOT). Steps:
  1. uv sync --frozen            venv on Lustre
  2. JDK 21                      installed to \$DRBM25_ROOT/jdk unless 'java' is already 21
  3. scripts/prepare_data.py     dataset download and the 800K-document BM25 index
  4. model snapshot download     into the Lustre Hugging Face cache (HF_HOME)
Exit status: 0 success, 1 runtime error, 2 usage error.
EOF2
}

staging=''

cleanup() {
  # Failure-tolerant: leaves nothing behind when the download or checksum step fails or is interrupted.
  if [[ -n $staging ]]; then rm -rf -- "${staging:?}" || true; fi
}
trap cleanup EXIT

install_jdk() {
  local target=$DRBM25_JDK_DIR/$JDK_RELEASE tarball machine
  if [[ -x $target/bin/java ]]; then
    log "JDK already unpacked at $target"
  else
    machine=$(uname -m)
    [[ $machine == x86_64 ]] || die "the pinned JDK is for x86_64, this machine is $machine"
    mkdir -p "$DRBM25_JDK_DIR"
    staging=$(mktemp -d "$DRBM25_JDK_DIR/.install.XXXXXX")
    tarball=$staging/jdk.tar.gz
    log "downloading $JDK_URL"
    # Download, verify the digest, and only then unpack; the tarball is never piped to a shell or tar.
    curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
      --max-time "$DOWNLOAD_TIMEOUT_S" --output "$tarball" "$JDK_URL"
    verify_sha256 "$tarball" "$JDK_SHA256"
    tar -xzf "$tarball" -C "$staging"
    [[ -x $staging/$JDK_RELEASE/bin/java ]] || die "tarball did not contain $JDK_RELEASE/bin/java"
    mv -- "$staging/$JDK_RELEASE" "$target"
    rm -rf -- "${staging:?}"
    staging=''
  fi
  ln -sfn "$JDK_RELEASE" "$DRBM25_JDK_DIR/current"
}

ensure_java21() {
  local major='' java_path
  jdk_env
  # A missing or unparsable java is expected on a fresh account; anything but 21 triggers the install.
  # shellcheck disable=SC2310 # failure here only means "no usable java yet"
  major=$(java_major 2>/dev/null) || true
  if [[ $major == 21 ]]; then
    java_path=$(command -v java)
    log "Java 21 already available: $java_path"
    return 0
  fi
  log "Java is '${major:-missing}', installing the pinned Temurin 21 into $DRBM25_JDK_DIR"
  install_jdk
  jdk_env
  require_java21
}

download_model() {
  # Resolves model.name and model.revision from the config, downloads, then proves the cache is
  # complete by resolving it again with local_files_only (what the offline jobs will do).
  uv run --frozen python - "$CONFIG" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from drbm25.config import load_config

cfg = load_config(Path(sys.argv[1]))
path = snapshot_download(cfg.model.name, revision=cfg.model.revision)
offline = snapshot_download(cfg.model.name, revision=cfg.model.revision, local_files_only=True)
assert path == offline, (path, offline)
print(f"model cached: {cfg.model.name}@{cfg.model.revision} -> {path}")
PY
}

main() {
  if (($# > 0)); then
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      *)
        printf '%s: unexpected argument: %s\n' "$SCRIPT_NAME" "$1" >&2
        exit 2
        ;;
    esac
  fi

  # shellcheck source=common.sh
  source "$SCRIPT_DIR/common.sh"
  drbm25_paths
  drbm25_set_args
  # The login node has internet; downloads must not be blocked by an inherited offline setting.
  DRBM25_ONLINE=1 offline_env
  command -v uv >/dev/null 2>&1 || die "uv not found on PATH; install it first (https://docs.astral.sh/uv/)"
  cd "$REPO_ROOT"

  mkdir -p "$DRBM25_ROOT" "$HF_HOME" "$DRBM25_INDEX_DIR" "$DRBM25_DATA_DIR" "$DRBM25_RUNS_DIR" "$WANDB_DIR"

  log "1/4 uv sync --frozen (venv: $UV_PROJECT_ENVIRONMENT)"
  uv sync --frozen
  log "2/4 Java 21"
  ensure_java21
  log "3/4 dataset and BM25 index"
  uv run --frozen python scripts/prepare_data.py --config "$CONFIG" "${DRBM25_SET_ARGS[@]}"
  log "4/4 model snapshot"
  download_model
  log "done. Next: sbatch --account=\"\$EXPANSE_ACCOUNT\" scripts/expanse/probe.sbatch"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
