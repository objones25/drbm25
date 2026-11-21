#!/usr/bin/env bash
#
# run.sh: run one command with the Expanse environment (Lustre paths, HF_HOME, WANDB_DIR, JDK 21).
# Use this instead of sourcing common.sh in your login shell: common.sh turns on `set -e -u` and
# its helpers call `exit`, which would end an interactive SSH session on the first error.
#
# Usage: see usage() below, or run with --help.
# Requires: bash 3.2+, EXPANSE_ACCOUNT in the environment.
# bash-target: 3.2

set -Eeuo pipefail

readonly SCRIPT_NAME=${0##*/}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

usage() {
  cat <<EOF2
Usage: $SCRIPT_NAME [--offline] COMMAND [ARG...]

Runs COMMAND (replacing this process, so its exit status is yours) after setting up the paths from
common.sh (DRBM25_ROOT, HF_HOME, WANDB_DIR, UV_PROJECT_ENVIRONMENT, ...) and putting a user-space
JDK 21 on PATH if prepare_login.sh installed one. Needs EXPANSE_ACCOUNT.
Default is the login-node setting (online). --offline sets HF_HUB_OFFLINE=1 and WANDB_MODE=offline.

Example: $SCRIPT_NAME uv run --frozen wandb sync RUN_DIR...
Exit status: the command's, or 1 runtime error, 2 usage error.
EOF2
}

main() {
  local offline=0
  while (($# > 0)); do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      --offline) offline=1 ;;
      --)
        shift
        break
        ;;
      -?*)
        printf '%s: unknown option: %s\n' "$SCRIPT_NAME" "$1" >&2
        exit 2
        ;;
      *) break ;;
    esac
    shift
  done
  if (($# == 0)); then
    printf '%s: COMMAND is required\n' "$SCRIPT_NAME" >&2
    exit 2
  fi

  # shellcheck source=common.sh
  source "$SCRIPT_DIR/common.sh"
  drbm25_paths
  if ((offline)); then
    offline_env
  else
    DRBM25_ONLINE=1 offline_env
  fi
  jdk_env
  exec "$@"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
