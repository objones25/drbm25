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

## RunPod one-shot launcher (A100 fallback)

`scripts/runpod/run_all.sh` runs the whole experiment on a fresh 2x A100 80GB pod, alone: it prompts for the
keys, installs everything, trains the `normal` and `count_once` runs in alternating chunks within a dollar
budget, evaluates, writes the report and terminates the pod (also after a fatal error, a closed terminal or Ctrl-C).

Prices (RunPod API, as of 2026-11-19): 2x A100 SXM 80GB community $2.78/h, secure $4.38/h; 2x A100 PCIe 80GB
community $2.38/h, secure $4.18/h. $100 buys about 36-42 h on community cloud and 23-24 h on secure. Community
stock was listed as low.

Template and stock: the official "Runpod Pytorch 2.8.0" image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (host CUDA
   12.8-13.4 allowed). Check GPU stock first: as of 2026-11-20 the RunPod API showed 2x A100 SXM 80GB available only on
   SECURE cloud in US-MD-1, with availability LOW; community cloud and PCIe showed none. A network volume is tied to its
   data center, so confirm that the data center currently has 2x A100 80GB in stock BEFORE creating the volume.
1. Create a network volume of at least **400 GB** in a data center that has A100 80GB in stock. Size estimate
   (not measured on a pod yet): a full fine-tune checkpoint holds fp32 parameters plus two Adam moments for the
   actor and the critic, about 74 GB; `run.keep_ckpts` is 1, but a save needs the old checkpoint beside the new one,
   so each run peaks at 2 checkpoints: 2 runs x 2 x 74 GB = about 296 GB, plus the model cache, dataset, index, venv
   cache and logs. The launcher refuses to start a chunk with less than 1.2x the last checkpoint (90 GiB before the
   first one) free. Network volume storage is billed per GB-month: check the current rate in the RunPod console
   before choosing the size. After the first checkpoint, confirm the real size on the pod with
   `du -sh /workspace/drbm25/runs/*/step_*` and `df -h /workspace`.
2. Deploy a pod with 2x A100 80GB and that volume mounted at `/workspace`. Create a RunPod API key with
   `pods:write` (it is used only to terminate the pod).
3. In the pod, run inside `tmux` so a closed SSH or web terminal cannot interrupt the run:
   ```bash
   cd /workspace && git clone <repo-url> repo && cd repo && git checkout feat/ppo-bm25
   tmux new -s drbm25      # reattach later with: tmux attach -t drbm25
   bash scripts/runpod/run_all.sh
   ```
   (`nohup bash scripts/runpod/run_all.sh` also works after the prompts have been answered via environment
   variables.) It asks, with hidden input for secrets, for `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_PROJECT` (default
   `drbm25`), `WANDB_ENTITY` (optional), `RUNPOD_API_KEY` and `TERMINATE_ON_FINISH` (default `yes`); anything already
   set in the environment is not asked again, and these answers go to `/workspace/drbm25/.env` (mode 600, never logged
   or committed; on a re-run only missing values are asked). On every launch it also asks for `BUDGET_USD`, the
   dollars still left to spend, shown next to what earlier launches already spent, and `PRICE_PER_HOUR` (this pod's
   price from the list above). Those two are never saved. After the prompts all output goes to
   `/workspace/drbm25/logs/run_all-<time>.log`; follow it with `tail -f`.
4. Stages, in order: Java 21 (pinned Temurin tarball, SHA-256 checked), uv 0.13.7 (pinned GitHub release tarball for x86_64 Linux,
   SHA-256 checked, installed into `/workspace/drbm25/bin`; no pip, so Ubuntu 24.04's externally-managed Python does not
   matter), Python 3.12 from `uv python install` (onto the volume, `UV_PYTHON_PREFERENCE=only-managed`, so the image's
   system Python and preinstalled torch are never used), `uv sync --frozen --extra vllm`, dataset and index, model snapshot at the config's pinned revision, GPU smoke test (`pytest
   -m gpu` on the vLLM and full-model tests, then `scripts/memory_check.py`, which fails with under 2 GiB headroom),
   training, evaluation, report. Data and model stages leave markers in `/workspace/drbm25/.done/` and are skipped on
   a re-run; Java, uv, `uv sync` and training run every launch (training resumes from the latest checkpoint), and the
   smoke test runs once per pod. The W&B project and entity you entered are applied to training, evaluation and the
   report.
5. Budget: the deadline is `start + BUDGET_USD / PRICE_PER_HOUR` hours (`drbm25.budget.deadline_from_budget`),
   measured from when the script starts. Spend (elapsed time x price) is written to `/workspace/drbm25/spend.json`
   after every chunk and on exit, so a relaunch shows the total already spent; give it the remaining budget, not the
   original one. Time before the script starts (pod boot, your typing) is not counted. Training stops at the next
   checkpoint boundary once less than `RESERVE_S` (default 3600 s, an unmeasured estimate for four evaluation runs,
   the report and the terminate call) is left; override with `RESERVE_S=...`. Both runs advance in chunks of
   `run.chunk_steps` (50), so they stop within one chunk of each other and the evaluation uses the last step both
   reached.
6. Hangs: each training chunk runs under GNU `timeout` (time to the deadline plus 30 min), evaluation commands under
   2 h and each smoke command under 1 h. A timeout is a failure: `FAILED` is written and the pod is terminated. A `SIGKILL` before the limit (exit 137, for example
   the out-of-memory killer) is reported as that, not as a timeout.

Ctrl-C, a closed terminal (SIGHUP) or SIGTERM stops the running command (its process group gets TERM, then KILL after
10 s), records spend, copies logs and terminates the pod; further signals during that cleanup are ignored.

`df` on a RunPod network volume may report pooled rather than per-volume space. Set `VOLUME_GB` to the size you created
(the launcher asks, optional, and saves it in `.env`) and the preflight also requires
`VOLUME_GB - used` (from `du -sk /workspace/drbm25`) to cover the next checkpoint. On the pod, compare
`du -sh /workspace` with the volume size you created.

Terminating the pod releases GPU billing; the network volume keeps all data. Use `--no-terminate` or
`TERMINATE_ON_FINISH=no` to keep the pod. If the terminate call fails after 3 attempts (waits of 5 s and 15 s) the
script prints the pod id and `https://console.runpod.io/pods`: terminate it there yourself.

Resuming on a new pod: attach the same volume at `/workspace`, clone again and re-run `bash scripts/runpod/run_all.sh`
inside tmux. A failure leaves `/workspace/drbm25/FAILED` (and terminates the pod). A later launch that finds the file
stops immediately, before prompting, and does NOT terminate the pod (it is still billing): fix the cause, delete the
file and run again, or terminate the pod yourself.

Where things are (all under `/workspace/drbm25`): `runs/{normal,count_once}/` (sharded checkpoints, `metrics.jsonl`,
`train_rank*.log`, `exit_code`), `runs/eval/` (per-query jsonl and `report.md`), `logs/` (the launcher log and
copies of the run logs, made on exit), `spend.json`, `hf/` (model cache), `indexes/`, `data/`, `.env`.

Tests: `bats tests/shell/test_runpod_lib.bats` (curl, uv, timeout, df and sleep are PATH shims) and
`shellcheck scripts/runpod/*.sh`.
