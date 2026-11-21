#!/usr/bin/env bash
#SBATCH --job-name=drbm25-probe
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH -t 00:30:00
#SBATCH --constraint="lustre"
#SBATCH --no-requeue
#SBATCH --output=%x.%j.out
# The account cannot be set here (directives do not expand variables). Submit with:
#   sbatch --account="$EXPANSE_ACCOUNT" scripts/expanse/probe.sbatch   (from the repo root)
#
# probe.sbatch: the first job on Expanse. Records what the compute nodes actually offer and
# writes $DRBM25_ROOT/probes/probe_<jobid>.json (also printed to the job log):
#   internet_ok, nvidia_smi, torch_version, torch_arch_list, device_capability, java_version,
#   modules_java_cuda, hf_generate_ok, vllm_generate_ok (null if not installed), notes.
# Run prepare_login.sh first: this job is offline for Hugging Face and needs the cached model.
# For the optional vLLM check, run once on the login node:
#   UV_PROJECT_ENVIRONMENT="$DRBM25_ROOT/venv-vllm" uv sync --frozen --extra vllm
# Requires: bash, the env made by prepare_login.sh, EXPANSE_ACCOUNT in the environment.
# audit-bash disable-file=B403 # straight-line Slurm job script
# bash-target: 4.0

set -Eeuo pipefail

readonly SCRIPT_NAME=probe.sbatch
# submit_chain.sh sets DRBM25_REPO_ROOT. SLURM_SUBMIT_DIR is where sbatch was invoked (not --chdir),
# so a direct `sbatch` must be run from the repository root.
REPO_ROOT=${DRBM25_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
readonly REPO_ROOT
readonly CONFIG=configs/expanse_v100_lora.yaml
readonly INTERNET_URL=https://huggingface.co

log() {
  local now
  now=$(date '+%Y-%m-%dT%H:%M:%S')
  printf '%s %s: %s\n' "$now" "$SCRIPT_NAME" "$*" >&2
}
die() {
  log "error: $*"
  exit 1
}

[[ -f $REPO_ROOT/scripts/expanse/common.sh ]] || die "submit from the repository root (no scripts/expanse/common.sh in $REPO_ROOT)"
# shellcheck source=common.sh
source "$REPO_ROOT/scripts/expanse/common.sh"

drbm25_paths
offline_env # HF stays offline whatever the internet check says: the probe tests the offline path
jdk_env
cd "$REPO_ROOT"
python=${DRBM25_PYTHON:-$UV_PROJECT_ENVIRONMENT/bin/python}
[[ -x $python ]] || die "no python at $python; run scripts/expanse/prepare_login.sh on the login node"
probe_dir=$DRBM25_ROOT/probes
mkdir -p "$probe_dir"
jobid=${SLURM_JOB_ID:-local}

# Each probe is allowed to fail: the point is to record the failure, not to stop the job.
log "nvidia-smi"
nvidia_smi=$(nvidia-smi 2>&1) || nvidia_smi="nvidia-smi failed: $nvidia_smi"
printf '%s\n' "$nvidia_smi"

log "internet check: $INTERNET_URL"
http_code=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' "$INTERNET_URL" 2>&1) || http_code="curl failed: $http_code"
printf 'http_code=%s\n' "$http_code"

log "java -version"
java_version=$(java -version 2>&1) || java_version="java failed: $java_version"
printf '%s\n' "$java_version"

# `module` is a shell function from Lmod; load its init file if the batch shell lacks it.
# /etc/profile.d/modules.sh is the usual location (unverified on Expanse: check `module avail`).
if ! declare -F module >/dev/null 2>&1; then
  # shellcheck disable=SC1091 # system file, absent off-cluster
  source /etc/profile.d/modules.sh 2>/dev/null || true
fi
if declare -F module >/dev/null 2>&1; then
  modules_text=$(module avail java cuda 2>&1) || modules_text="module avail failed: $modules_text"
else
  modules_text="the 'module' command is not available in this shell"
fi
printf '%s\n' "$modules_text"

vllm_python=''
vllm_note=''
case $http_code in
  2?? | 3??)
    # Internet works: try to build the optional vLLM env now (a no-op if it already exists).
    if [[ ! -x $DRBM25_ROOT/venv-vllm/bin/python ]] && command -v uv >/dev/null 2>&1; then
      log "uv sync --frozen --extra vllm (separate env)"
      if ! UV_PROJECT_ENVIRONMENT=$DRBM25_ROOT/venv-vllm uv sync --frozen --extra vllm; then
        vllm_note="uv sync --extra vllm failed on the compute node"
      fi
    fi
    ;;
  *) ;;
esac
if [[ -x $DRBM25_ROOT/venv-vllm/bin/python ]]; then
  vllm_python=$DRBM25_ROOT/venv-vllm/bin/python
fi

log "torch and generation checks"
export PROBE_NVIDIA_SMI=$nvidia_smi PROBE_HTTP_CODE=$http_code PROBE_JAVA_VERSION=$java_version
export PROBE_MODULES=$modules_text PROBE_VLLM_PYTHON=$vllm_python PROBE_OUT=$probe_dir/probe_$jobid.json
export PROBE_CONFIG=$CONFIG PROBE_VLLM_NOTE=$vllm_note
"$python" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from drbm25.config import load_config

PROMPTS = ["The capital of France is", "Solar panel efficiency depends on", "def fibonacci(n):"]
notes: list[str] = []
http_code = os.environ["PROBE_HTTP_CODE"]
cfg = load_config(Path(os.environ["PROBE_CONFIG"]))
n_gpus = torch.cuda.device_count()
notes.append(f"visible GPUs: {n_gpus}")
if os.environ["PROBE_VLLM_NOTE"]:
    notes.append(os.environ["PROBE_VLLM_NOTE"])


def hf_generate() -> bool:
    tok = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda:0")
    batch = tok(PROMPTS, return_tensors="pt", padding=True).to("cuda:0")
    out = model.generate(**batch, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
    notes.append(f"hf sample: {texts[0]!r}")
    return len(texts) == len(PROMPTS) and all(t.strip() for t in texts)


# Recording the failure is the purpose of the probe, so exceptions are caught here and reported.
try:
    hf_ok = hf_generate()
except Exception as exc:
    hf_ok = False
    notes.append(f"hf generate failed: {type(exc).__name__}: {exc}")

VLLM_SNIPPET = """
import sys
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], revision=sys.argv[2], dtype="float16", gpu_memory_utilization=0.5, max_model_len=512)
outs = llm.generate(sys.argv[3:], SamplingParams(temperature=0.0, max_tokens=16))
assert len(outs) == len(sys.argv[3:]) and all(o.outputs[0].text.strip() for o in outs)
print("vllm ok")
"""
vllm_python = os.environ["PROBE_VLLM_PYTHON"]
vllm_ok: bool | None = None
has_vllm = bool(vllm_python) and subprocess.run(
    [vllm_python, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('vllm') else 1)"],
    check=False,
).returncode == 0
if has_vllm:
    proc = subprocess.run(
        [vllm_python, "-c", VLLM_SNIPPET, cfg.model.name, cfg.model.revision, *PROMPTS],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    vllm_ok = proc.returncode == 0 and "vllm ok" in proc.stdout
    if not vllm_ok:
        notes.append(f"vllm failed (exit {proc.returncode}): {proc.stderr[-400:]}")
else:
    notes.append("vLLM not checked: no vllm in $DRBM25_ROOT/venv-vllm (see the header of probe.sbatch)")

report = {
    "internet_ok": http_code.isdigit() and 200 <= int(http_code) < 400,
    "nvidia_smi": os.environ["PROBE_NVIDIA_SMI"],
    "torch_version": torch.__version__,
    "torch_arch_list": torch.cuda.get_arch_list(),
    "device_capability": list(torch.cuda.get_device_capability(0)) if n_gpus else None,
    "java_version": os.environ["PROBE_JAVA_VERSION"],
    "modules_java_cuda": os.environ["PROBE_MODULES"],
    "hf_generate_ok": hf_ok,
    "vllm_generate_ok": vllm_ok,
    "notes": notes,
}
text = json.dumps(report, indent=2)
Path(os.environ["PROBE_OUT"]).write_text(text + "\n")
print(text)
PY
log "wrote $PROBE_OUT"
