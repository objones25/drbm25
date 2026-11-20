# drbm25

PPO training of an LLM query rewriter with a BM25 reward (CS279 DeepRetrieval reproduction).

## Setup

```bash
java -version          # Java 21 is required by pyserini 2.x
uv sync                # creates .venv from uv.lock (Python 3.12)
uv sync --extra vllm   # Linux only: adds vllm==0.19.0
uv run pytest          # default run deselects gpu and integration tests
uv run ruff check .
uv run pyright
```

## Running on SDSC Expanse (2x V100 32GB)

Scripts live in `scripts/expanse/`; the profile is `configs/expanse_v100_lora.yaml` (Qwen2.5-1.5B,
LoRA, fp16). The profile holds placeholder paths: the scripts pass `--set` overrides for
`retrieval.index_dir`, `data.data_dir` and `run.output_dir`, so do not edit the YAML.
Compute nodes are assumed to have no internet (unverified until the probe runs): jobs set
`HF_HUB_OFFLINE=1` and `WANDB_MODE=offline` unless `DRBM25_ONLINE=1` is exported at submit time.

Everything below is a sequence of commands to run in a login-node shell. Run `sbatch` and `scripts/expanse/...` commands from the repository root.

1. Log in and set the two required variables. `EXPANSE_ACCOUNT` is your allocation id (it is never
   hard-coded). Export `HF_TOKEN` only if a download needs it, and `WANDB_API_KEY` for W&B.
   The code can stay in `$HOME` (100 GB quota); the venv, HF cache, index, data and runs go to
   `/expanse/lustre/projects/$EXPANSE_ACCOUNT/$USER/drbm25` (override with `DRBM25_ROOT`).

   ```bash
   ssh "$USER@<expanse-login-host>"   # host name: see the SDSC Expanse user guide
   export EXPANSE_ACCOUNT=<your-allocation-id>
   ```

2. One-time preparation on the login node (needs `uv` on `PATH`). It runs `uv sync --frozen`, installs
   Temurin JDK 21 under `$DRBM25_ROOT/jdk` if `java` is not already 21 (pinned URL, SHA-256 checked),
   builds the 800K-document index with `scripts/prepare_data.py`, and downloads the model snapshot.
   It is safe to re-run. Check `module avail java` first if you would rather use a site JDK 21 module.

   ```bash
   scripts/expanse/prepare_login.sh
   ```

3. Probe job (`gpu-debug`, 2 GPUs, 30 min). It records internet reachability, `nvidia-smi`, torch arch
   list and capability, Java, `module avail java cuda`, and a 3-prompt fp16 HF generation, and writes
   `$DRBM25_ROOT/probes/probe_<jobid>.json`. The vLLM check runs only if
   `scripts/expanse/run.sh bash -c 'UV_PROJECT_ENVIRONMENT="$DRBM25_ROOT/venv-vllm" uv sync --frozen --extra vllm'`
   has been done (the probe tries it itself only when the compute node has internet). Module names are not known in advance:
   use what `module avail` prints.

   ```bash
   sbatch --account="$EXPANSE_ACCOUNT" scripts/expanse/probe.sbatch
   scripts/expanse/run.sh bash -c 'cat "$DRBM25_ROOT"/probes/probe_*.json'
   ```

   If `internet_ok` is true you may submit with `DRBM25_ONLINE=1` exported to get live W&B and no offline HF.

4. Pilot (<= 150 steps under both reward modes per candidate LoRA setting; the pick is made on validation
   nDCG, then frozen for both real runs). Extra arguments after `--` go to `drbm25.train`; `--run-name`
   keeps the pilot out of the real run directories.

   ```bash
   scripts/expanse/submit_chain.sh normal 1 -- --run-name pilot_normal --until-step 150 --set ppo.actor_lr=2e-5
   scripts/expanse/submit_chain.sh count_once 1 -- --run-name pilot_count_once --until-step 150 --set ppo.actor_lr=2e-5
   ```

   If you change a setting, put it in `configs/expanse_v100_lora.yaml` before step 5 so both runs share it.

5. The two real runs, as two independent chains so they train concurrently. Each job is `gpu-shared`,
   2 GPUs, at most 47.5 h; job k+1 starts after job k ends (`--dependency=afterany`) and resumes from the
   run's latest checkpoint. Jobs after the run reaches `total_steps` exit 0 immediately. Three jobs is
   the planned allocation; a failed job does not stop the chain, so check `squeue` and cancel with
   `scancel` if a run fails for a non-time-limit reason.

   ```bash
   scripts/expanse/submit_chain.sh normal 3
   scripts/expanse/submit_chain.sh count_once 3
   squeue -u "$USER"
   ```

   Each job's status is read from `$DRBM25_ROOT/runs/<run>/exit_code` (written by rank 0, because
   `torchrun` collapses worker exit codes to 1): 0 done, 3 budget stop, 2 failed with an exception,
   1 started but never finished (killed, e.g. out of memory). `train.sbatch` exits 0 only for 0.
   On 2, on 3 (these jobs never set a deadline, so a stop means something is wrong) or when no
   `exit_code` exists, it writes `$DRBM25_ROOT/runs/<run>/FAILED` (job id and reason) and every later job
   in the chain exits non-zero immediately instead of repeating the failure. Inspect `train_rank0.log`
   in the run directory, fix the cause, delete `FAILED`, then submit again. After 1 (killed) the next
   job resumes from the latest checkpoint; a job ended by the time limit does the same. A worker
   other than rank 0 that raises can leave 1 rather than 2, so check `squeue` and the logs anyway.
   Job output: `drbm25-train.<jobid>.out` in the repository directory.

6. Sync offline W&B runs from the login node (needs internet and `WANDB_API_KEY`). Do not `source`
   `common.sh` in your login shell (it turns on `set -eu` and its helpers call `exit`); `run.sh` sets
   the same environment for one command in its own process:

   ```bash
   scripts/expanse/run.sh bash -c 'uv run --frozen wandb sync "$WANDB_DIR"/wandb/offline-run-*'
   ```

7. Evaluation, once both chains have finished. `eval.sbatch` scores the original queries, the untrained
   model and both trained runs at the largest step both runs have a complete checkpoint for
   (`drbm25.eval.report.common_step`) and writes `$DRBM25_ROOT/runs/eval/report.md` and the four
   `.jsonl` files. To also log the tables to W&B, rerun the report from the login node with `--wandb`
   (the report does not need Java).

   ```bash
   sbatch --account="$EXPANSE_ACCOUNT" scripts/expanse/eval.sbatch
   scripts/expanse/run.sh bash -c 'uv run --frozen python -m drbm25.eval.report --wandb \
     --config configs/expanse_v100_lora.yaml \
     --inputs "$DRBM25_RUNS_DIR"/eval/{original,untrained,trained-normal,trained-count_once}.jsonl \
     --out "$DRBM25_RUNS_DIR"/eval/report.md'
   ```

Shell tests: `bats tests/shell/` (needs bats >= 1.5 and bash >= 4 for `mapfile`) and
`shellcheck scripts/expanse/*.sh scripts/expanse/*.sbatch`.

## A100 layout (2x A100 80GB, full fine-tune)

`configs/runpod_a100_full.yaml` (Qwen2.5-3B, `model.train_mode: full`, `rollout.backend: vllm`) builds
an FSDP2-sharded fp32 actor and critic, a frozen bf16 reference, and one colocated vLLM 0.19 engine
per rank. It needs `uv sync --extra vllm`, CUDA, and torchrun (`--nproc_per_node 2`); checkpoints are
sharded (`step_NNNNNN/dcp/`, written by every rank). Before a long run, measure peak memory per GPU
(sampled through generation) over two PPO steps, a validation pass, a sharded save and a resume
(synthetic queries, no index or dataset needed; exits 1 when the headroom is under 2 GiB):

```bash
uv run torchrun --nproc_per_node 2 scripts/memory_check.py --config configs/runpod_a100_full.yaml
uv run pytest -m gpu tests/rollout/test_vllm_engine.py   # vLLM greedy == HF, weight sync
```

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: vLLM's sleep mode refuses it.
