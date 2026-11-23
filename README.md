# drbm25

PPO training of an LLM query rewriter with a BM25 reward: a CS291A class project that reproduces and extends the MS MARCO-Science
BM25 result of DeepRetrieval (Jiang et al., 2025, [arXiv:2503.00223](https://arxiv.org/abs/2503.00223)).

## Results

Our baselines match the paper's (raw query nDCG@10 38.8 in both; untrained model 25.8 vs 25.9); our trained models reach
39.7 and 39.6 against the paper's 41.1. We followed the released code, which trains with an nDCG cutoff of 10 where the paper's
text says 3000; see the report's "Comparison with the paper" for this and other candidate causes, none tested. For the reproduction and an
ablation, two Qwen2.5-3B-Instruct policies were trained for 1,751 PPO steps each, one with the standard BM25 reward and
one with a count-once reward that collapses repeated query terms. On the 2,305 MS MARCO-Science dev queries
the two reach nDCG@10 0.397 and 0.396 (raw query 0.388, untrained model 0.258); the difference is −0.0006, 95% CI
[−0.0058, +0.0047]. Repetition did not pay under standard BM25 after the first steps (when the normal run drifted back into it, standard BM25
scored those rewrites lower than count-once BM25), so the two rewards trained nearly the same thing, but the policies
still differ in determinism, reasoning template and wording.

- Write-up: [report/REPORT.md](report/REPORT.md); slides: `report/drbm25_results.pptx` (30-minute talk).
- Training runs (W&B project `drbm25`): `hunfj08k` (normal), `yld3obpy` (count-once), `yk2tdsy3` (eval report).
- Models: `objones25/drbm25-qwen2.5-3b-normal` and `objones25/drbm25-qwen2.5-3b-count_once` (private).
- Sampled rewrites with reasoning and metrics: dataset `objones25/drbm25-rewrites` (private).

## Setup

```bash
java -version          # Java 21 is required by pyserini 2.x
uv sync                # creates .venv from uv.lock (Python 3.12)
uv sync --extra vllm   # Linux only: adds vllm==0.19.0
uv sync --extra analysis  # behavior analysis: pyarrow, scipy, scikit-learn, matplotlib, sentence-transformers, wandb-workspaces
uv run pytest          # default run deselects gpu and integration tests
uv run ruff check .
uv run pyright
```

## End-to-end smoke test (CPU, no downloads)

`tests/test_e2e_smoke.py` (marks `java` and `slow`; part of the default `uv run pytest`) runs the whole
pipeline on `configs/smoke_cpu.yaml` (tiny local model, the 1,002-document toy index): it trains 4 steps
under each reward mode in two chunks (`--until-step 2`, then `4`, as the RunPod launcher does), scores the
original queries, the untrained policy and each run's `best/` with `run_eval`, builds the report, and
checks the four table rows, nDCG in [0, 1], best step, validation score and training steps columns, `run_manifest.json`
seeds, and that both runs saw the same data order. It needs Java 21 on `PATH` (on the dev Mac:
`JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home PATH=/opt/homebrew/opt/openjdk@21/bin:$PATH`).
It checks that the stages connect, not that training helps.

## What a training run leaves behind

Each run directory `runs/<run>/` holds:

- `step_NNNNNN/` and `latest`: the one resume checkpoint (`run.keep_ckpts: 1`; model, optimizers, scaler,
  RNG, data cursor). A save writes the new one beside the old one, then deletes the old one.
- `best/`: the policy with the highest validation score so far, which is what evaluation scores. Every
  `run.val_every` (50) steps and at `run.total_steps`, the actor rewrites the validation queries
  greedily; when nDCG@10 under the run's own reward mode (`val/ndcg_normal` or `val/ndcg_count_once`)
  strictly beats the stored best, `best/` is replaced atomically (ties keep the earlier step).
  `best/model.safetensors` holds the actor only (LoRA: the actor adapter; full fine-tune: the actor in
  bf16 with HF names, about 6.8 GB at 3B); `best/meta.json` records the step, both validation scores,
  the selection metric, the reward mode, the config sha256, the seeds and the git sha and dirty flag. A
  resume keeps the stored best. With 2 ranks, validation is a collective: whenever each rank generates
  on its own (the LoRA layout, or `rollout.backend: vllm` on either layout, which is the A100 config)
  each rank rewrites every other validation query and the nDCG sums and counts are all-reduced, so two
  vLLM engines never have to produce bit-identical greedy text; only the full layout with the `hf`
  backend has every rank rewrite all of them (its FSDP forwards inside `generate` are collectives).
  Every rank gets the same reduced scores and decision.
- `run_manifest.json`: every seed with its derivation (`run.seed`: model construction, data order, per-rank
  sampling and vLLM seeds `run.seed + rank`; `data.split_seed`: the train/validation split), the resolved
  config and its sha256, git sha and dirty flag, dataset and model revisions, index checksum and size,
  library versions, CUDA, driver and GPUs, world size, host, start time, and one `resumed_at` entry per
  resume. A resume with a different `run.seed` or `data.split_seed` is refused. The same dict is the
  W&B run's `config.meta`.

`run.dist_timeout_min` (60 in every profile) is the `torch.distributed` collective timeout passed to
`init_process_group`: a rank waiting longer than that in a collective (for example while rank 0 writes
a checkpoint) is killed by NCCL's watchdog.

Splits: validation = `data.val_size` (500) train queries picked by `data.split_seed` (1234); training uses
the other 21,916; the test set is all 2,305 dev queries (`run_eval` never sees validation queries).

## Reading the report

`python -m drbm25.eval.report` writes `report.md` from the per-source eval jsonl files:

- nDCG@10 table: one row per source under both scorers. Trained rows also show the `best/` policy's step
  and validation score (selection metric), and **training steps**: how far the run had trained when it
  was evaluated (its latest complete checkpoint, recorded as `trained_steps` in the eval jsonl header).
  The two runs alternate in chunks, so after a budget stop `count_once` can be one chunk behind `normal`;
  the report then prints "Training steps differ: ..." under the table. Each row is still that run's best
  validated policy, but the runs did not train equally long: read a gap between them with that in mind
  (and compare the best steps, which may be well before either run's last step).
- Paired bootstrap 95% CIs of the per-query differences.
- Query statistics per source: `mean_n_terms`, `mean_n_unique_terms`, `repeat_ratio`, `any_repeat_rate`,
  `format_error_rate`, `mean_think_chars`. Term statistics cover rewrites without a format error.
  `repeat_ratio` is pooled over terms: repeated term occurrences over all term occurrences of the source.
  The training metric `rewrite/repeat_ratio` (W&B, `metrics.jsonl`) is instead the per-query mean of
  1 - unique/total, which weights short and long queries equally, so the two need not agree.
  `mean_think_chars` is the mean length in characters of the stripped reasoning before `</think>`, over
  rewrites without a format error (`-` for the original queries, which have none).
- Reproducibility: bootstrap resamples and seed; each trained run's config sha256, seeds and training git sha.

With `--wandb` (and `--config`) the tables are also logged to a W&B run (`job_type` report), and the input
jsonl files plus `report.md` are logged as one artifact, `eval-results` (type `evaluation`).

## Publishing results

```bash
uv run python -m drbm25.publish --run-dir RUN_DIR [--eval-jsonl F] [--report F] [--namespace N] \
  [--repo-prefix drbm25] [--private | --public] [--code-url URL] [--dry-run | --invalidate]
```

The RunPod launcher runs this for both runs after the report; on Expanse run it from the login node (step 8
above). It needs `HF_TOKEN` in the environment with write access, checked with `whoami` before anything is
created or uploaded. Each run goes to its own repo, private unless `--public`:
`<your user>/drbm25-<model>-<reward mode>`, for example `drbm25-qwen2.5-3b-normal` (RunPod, full fine-tune) or
`drbm25-qwen2.5-1.5b-lora-count_once` (Expanse, LoRA). What is uploaded:

- the selected policy from `best/`: for a full fine-tune `model.safetensors` plus the base model's `config.json`,
  `generation_config.json` and tokenizer files; for LoRA a PEFT adapter (`adapter_model.safetensors`,
  `adapter_config.json` naming the base model and its pinned revision) plus the tokenizer files;
- `provenance/best_meta.json`, `provenance/run_manifest.json`, the run's test-set evaluation
  `provenance/trained-<mode>.jsonl` and `provenance/report.md`;
- `README.md`, a model card with the base model and revision, reward mode, best and trained steps, validation and
  test nDCG@10 under both scorers, seeds, config sha256 and the code commit;
- the base model's `LICENSE`. The two base models are licensed differently (their model cards at the pinned
  revisions, read 2026-11-20): Qwen2.5-1.5B-Instruct is `apache-2.0`, Qwen2.5-3B-Instruct is the Qwen Research
  License (non-commercial). The card copies the base card's license fields, and for the research license a
  `NOTICE` file carries the attribution its section 3c requires.

Resume checkpoints (optimizer state) are never uploaded. `load_run` refuses an eval file that scores a different
best step, or a report whose row for the run disagrees with the eval file.

Fail closed: a publish first deletes `RUN_DIR/UPLOADED.json` and the repo's `UPLOADED.json`, before anything is
staged or uploaded, so a publish that fails at any later point leaves no marker. `--invalidate` does only that;
`run_all.sh` runs it for every run that will train further, before training, because training can replace
`best/`; the local marker is always removed. If the Hub marker cannot be withdrawn (Hugging Face outage, token),
the run gets `runs/<mode>/MARKER_STALE` with the reason, a `WARNING` is logged and training goes on (an outage
must not cost training budget). `MARKER_STALE` is removed only after a verified new marker is written; while it
exists the run does not count as uploaded (`--status` prints `no (stale marker could not be withdrawn ...)`).

Verification: after the commit the repo is listed again and every file is compared with the local copy (LFS files
by sha256 and size, small files by their git blob id and size). The listed LFS sha256 is the hash the uploading
client declared, so every LFS file (the weights) is then also downloaded back at the verified revision
(`hf_hub_download(..., force_download=True)` into a temp dir under `RUN_DIR`, deleted right after) and its sha256
recomputed. Before the first download it checks that 1.2 x the largest weight file is free under `RUN_DIR`, and
fails otherwise (on RunPod: `PUBLISH_FAILED`, the pod is still terminated). The CLI turns SIGTERM (the stage's
`timeout`, a pod shutdown) into a clean exit (status 143) so the temp copy is removed, and every publish starts by
deleting `.tmp-download-*` dirs an earlier killed publish left (logged). Only if all match is `RUN_DIR/UPLOADED.json` written (repo id, the verified revision, best step, every
file's sha256 and size, time) and a copy without the revision and time committed to the repo. A re-run uploads no
file whose hash is unchanged (it only withdraws and re-commits the marker, and downloads the weights back again).
Network errors (connection, timeout, HTTP 5xx/429) are retried up to 4 attempts with waits of 2, 5 and 15 s.

When is it safe to delete the RunPod volume? `bash scripts/runpod/launch.sh --status` prints each run's Hub marker
with its best step and one of:
- `safe to delete the volume: yes`: both repos hold `UPLOADED.json` and, read through the running pod, each one
  was written for the `best/meta.json` the volume holds now;
- `no`: a marker is missing, it is for an older `best/` than the volume holds, or the run has `MARKER_STALE`
  (publish again);
- `unknown`: no running pod, so the volume cannot be compared. Markers are withdrawn at the start of every launch
  that trains, and the pod's own check is in the last `run_all-*.log` ("safe to delete the volume", logged only when
  both `runs/*/UPLOADED.json` match the current `best/meta.json`). Start a pod (or read that log) to be sure.
If `PUBLISH_FAILED` exists, keep the volume and publish again. The upload covers the selected policies and results,
not the resume checkpoints: after deleting the volume a run cannot be resumed.

Before making a repo public (`--public`, or later on the Hub), read `provenance/run_manifest.json`: it records the
machine's hostname, GPU and driver, and the resolved config including absolute paths (for example
`/workspace/drbm25/...` or your Expanse project directory).

Loading a published model (private repos need a token with read access):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# full fine-tune
model = AutoModelForCausalLM.from_pretrained("<user>/drbm25-qwen2.5-3b-normal")
# LoRA: the base at the revision in adapter_config.json, then the adapter
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
model = PeftModel.from_pretrained(base, "<user>/drbm25-qwen2.5-1.5b-lora-normal")
tokenizer = AutoTokenizer.from_pretrained("<user>/drbm25-qwen2.5-1.5b-lora-normal")
```

## Rewrite sampling and behavior analysis

Samples every dev query N times from each policy and measures how the policies behave, not only how they
score. Four modules, run in this order (all take `--help`):

```bash
# 1. Sample: one row per response (reasoning, rewrite, analyzed terms, top-10 docs, nDCG under both scorers).
#    SOURCE is original | untrained | trained-normal | trained-count_once. Shards are resumable.
uv run python -m drbm25.eval.sample --config CONFIG --source SOURCE --n 32 --out-dir samples/db \
  [--model REPO] [--limit N] [--shard-size 128] [--base-seed 1234] [--set key=value ...]
# 2. Metrics: per-sample, per-query and cross-policy tables, paired query-level comparisons, vocabulary statistics.
uv run python -m drbm25.analysis.pipeline --db samples/db --index-dir INDEX --out samples/metrics \
  [--no-embed] [--embed-device DEV] [--n-resamples 10000] [--c2st-permutations 200]
# 3. Upload to a private Hugging Face dataset repo; every file is downloaded back and hash-checked.
uv run python -m drbm25.analysis.hub --db samples/db --metrics samples/metrics --repo OWNER/NAME
# 4. W&B run with tables and figures, plus a W&B Report.
uv run python -m drbm25.analysis.report --metrics samples/metrics --db samples/db --config CONFIG [--no-report]
```

Sampling uses the training prompt, tokenizer and rollout settings (temperature 0.6, top-p 1, no top-k), and the
same seed per query for every source; each row records the seed its sample used. Comparisons treat the query as
the unit: per-query means first, then a paired bootstrap over queries and a Wilcoxon test; nDCG of normal vs
count-once is the primary comparison and every other metric is Benjamini-Yekutieli corrected.

On RunPod, `scripts/runpod/run_sampling.sh` runs all four steps on one GPU against the `drbm25` volume: a pilot
(`PILOT_N` samples, default 8, on the first `PILOT_LIMIT` queries, default 64), the metrics on the pilot as a
smoke test, the full sampling (`SAMPLE_N`, default 32), metrics, upload (`SAMPLE_HUB_REPO`, default
`objones25/drbm25-rewrites`) and report. A failed report does not fail the run. Its failure marker is
`/workspace/drbm25/samples/FAILED`, so it never blocks a training launch. `launch.sh` has no mode for it yet:
create the pod the way `launch.sh` does (same image, volume and secrets, `pod_bootstrap.sh` as the command) and
set `DRBM25_ENTRY=scripts/runpod/run_sampling.sh`; the bootstrap only accepts a script directly under
`scripts/runpod/`. The 2026-11-22 run took about 2.3 hours on one RTX PRO 4500 (about $1.70).

To rebuild the write-up and slides from the W&B runs and the dataset:

```bash
uv run python scripts/report_data.py --wandb WANDB_DIR --metrics METRICS_DIR --db DB_DIR  # report/data, report/figures
uv run python scripts/deck_figures.py                                                    # report/figures/deck_*.png
(cd report/deck && npm ci && node build.js)                                             # report/drbm25_results.pptx
```

`WANDB_DIR` holds `<run_id>_history.json` for `hunfj08k` and `yld3obpy` (from `wandb.Api().run(...).scan_history()`)
and the eval tables of `yk2tdsy3`; `METRICS_DIR` and `DB_DIR` are the `metrics/` and `db/` folders of the dataset.

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
   model and the `best/` policy of each trained run on all 2,305 dev queries, and writes
   `$DRBM25_ROOT/runs/eval/report.md` and the four `.jsonl` files. It stops if a run has no `best/`, and
   warns if a run's latest checkpoint is short of `total_steps`. The report shows each run's best step and
   validation score, and a reproducibility block (bootstrap resamples and seed, `--bootstrap-seed`, default
   0; each run's config sha256 and seeds). To also log the tables to W&B, rerun the report from the login
   node with `--wandb` (the report does not need Java).

   ```bash
   sbatch --account="$EXPANSE_ACCOUNT" scripts/expanse/eval.sbatch
   scripts/expanse/run.sh bash -c 'uv run --frozen python -m drbm25.eval.report --wandb \
     --config configs/expanse_v100_lora.yaml \
     --inputs "$DRBM25_RUNS_DIR"/eval/{original,untrained,trained-normal,trained-count_once}.jsonl \
     --out "$DRBM25_RUNS_DIR"/eval/report.md'
   ```

8. Publish both runs to the Hugging Face Hub from the login node (compute nodes may be offline; see
   "Publishing results"). `HF_TOKEN` must be in your environment and allowed to write model repos:

   ```bash
   scripts/expanse/run.sh bash -c 'for m in normal count_once; do
     uv run --frozen python -m drbm25.publish --run-dir "$DRBM25_RUNS_DIR/$m" || exit 1; done'
   ```

Shell tests: `bats tests/shell/` (needs bats >= 1.5 and bash >= 4 for `mapfile`) and
`shellcheck scripts/expanse/*.sh scripts/expanse/*.sbatch`.

## A100 layout (2x A100 80GB, full fine-tune)

`configs/runpod_a100_full.yaml` (Qwen2.5-3B, `model.train_mode: full`, `rollout.backend: vllm`) builds
an FSDP2-sharded fp32 actor and critic, a frozen bf16 reference, and one colocated vLLM 0.19 engine
per rank. It needs `uv sync --extra vllm`, CUDA, and torchrun (`--nproc_per_node 2`); checkpoints are
sharded (`step_NNNNNN/dcp/`, written by every rank). `ppo.micro_batch_size` is 8, the ceiling (per-rank
mini-batch 16 / 2 = 8; PPO batch 64 and mini-batch 16 are unchanged). The controller's estimate, not
measured on a pod: about 46 GiB per GPU of sharded fp32 actor and critic states, 2.9 GiB of reference,
about 1.3 GiB of activations per 606-token sequence, so micro batch 8 leaves about 19 GiB of headroom
while vLLM sleeps. The RunPod launcher checks it and falls back to 4, then 2. Before a long run, measure peak memory per GPU
(sampled through generation) over two PPO steps, a validation pass, a sharded save and a resume
(synthetic queries, no index or dataset needed; exits 1 when the headroom is under 2 GiB):

```bash
uv run torchrun --nproc_per_node 2 scripts/memory_check.py --config configs/runpod_a100_full.yaml
uv run pytest -m gpu tests/rollout/test_vllm_engine.py   # vLLM greedy == HF, weight sync
```

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: vLLM's sleep mode refuses it.

## RunPod launcher (A100 fallback)

One command on your own machine starts the whole experiment on a new 2x A100 80GB pod and returns; the pod
clones this (public) repository and runs `scripts/runpod/run_all.sh` unattended. It trains the `normal` and
`count_once` runs in alternating chunks within a dollar budget, evaluates, writes the report and terminates
itself (also after a fatal error). The network volume `drbm25` keeps everything between pods.

### Prerequisites

- bash 3.2+ (macOS `/bin/bash` is fine), curl 7.55+, `jq` (macOS 15 and later ship `/usr/bin/jq`; otherwise
  `brew install jq`), git, and a checkout of this repository.
- The code the pod runs must be on the public repository. `--ref` defaults to this checkout's HEAD commit, which
  must be the tip of a branch or tag on `https://github.com/objones25/drbm25` (push first); or pass
  `--ref <pushed branch or tag>`. `launch.sh` checks this with `git ls-remote` before it creates anything.
- A RunPod API key allowed to read the catalog, list and create network volumes, pods and secrets, and
  terminate pods (`pods:write`). The pod uses the same key, stored as a RunPod secret, to terminate itself.
- A Hugging Face token with the `write` role (or a fine-grained token with `repo.write` on your user), and a
  Weights & Biases API key. `launch.sh` checks the token with `whoami` before it creates anything and refuses a
  read-only one, because the pod uploads the trained models to your account when it finishes.

### Launch

```bash
bash scripts/runpod/launch.sh
```

It asks (hidden for keys; anything already set in the environment is not asked) for `RUNPOD_API_KEY`,
`HF_TOKEN`, `WANDB_API_KEY`, `WANDB_PROJECT` (default `drbm25`), `WANDB_ENTITY` (optional) and `BUDGET_USD`,
the dollars still left to spend. Then it:

1. refuses if a pod named `drbm25` already exists (two pods must never share the volume);
2. reads live stock and list prices for 2x A100 80GB, SXM and PCIe, on the secure cloud with CUDA 12.8 or
   newer, and picks the cheapest offer in stock (`--gpu sxm|pcie|any`, default `any`). `--gpu pro6000` instead
   asks for 2x RTX PRO 6000 96GB (Blackwell), for when no A100 pair is in stock where the volume lives; `any`
   never picks it. Community cloud is not offered: RunPod network volumes only work on Secure Cloud pods;
3. reuses the network volume named `drbm25` if there is one. A volume is tied to its data center, so the pod
   must go there: if that data center has no stock, it says so and stops (it never creates a second volume).
   With no volume it offers to create one (`--volume-gb`, default 400; standard tier) in the chosen data center,
   choosing only among data centers whose catalog entry lists STANDARD volumes (checked before you confirm; if
   none with stock does, it stops and lists each data center's volume types, since the high-performance price
   is not published in the docs);
4. prints the plan: GPU, data center, the exact hourly price, what the budget buys, the volume (and its
   monthly cost if new) and the git ref, and creates nothing billable until you type `yes` (`--yes` skips the
   question for scripts; the plan is still printed);
5. stores `HF_TOKEN`, `WANDB_API_KEY` and `RUNPOD_API_KEY` as RunPod secrets (`drbm25_hf_token`,
   `drbm25_wandb_api_key`, `drbm25_runpod_api_key`; created or rotated). The pod's environment holds only
   `{{ RUNPOD_SECRET_<name> }}` references, never the values; keys never appear in curl's arguments (the
   header and bodies are mode-600 temporary files, deleted on exit);
6. creates the pod: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (the official "Runpod Pytorch
   2.8.0"), the volume at `/workspace`, 80 GB container disk, ports 22 and 8888. Its start command is
   `scripts/runpod/pod_bootstrap.sh`: it starts the image's `/start.sh` (SSH, Jupyter) in the background, fetches
   the ref into `/workspace/drbm25/repo` (reusing an earlier clone), then runs `run_all.sh` with stdin from
   `/dev/null`. The budget, the quoted price, `TERMINATE_ON_FINISH=yes`, the volume size, the W&B settings, a
   launch id and two absolute deadlines go in as plain environment variables. The deadlines are
   `DRBM25_BUDGET_DEADLINE_UNIX` = launch time + budget / price, and `DRBM25_HARD_DEADLINE_UNIX`, 1 h later;
7. prints the pod id, the console URL (`https://console.runpod.io/pods`) and how to follow progress, and exits.

Each RunPod endpoint used is listed with its documentation page at the top of `scripts/runpod/launch.sh`.

The pod is terminated through the RunPod API (the documented way to stop GPU billing) by three parties:
- `run_all.sh`, when it ends, fails or is signalled.
- `pod_bootstrap.sh`, on exit: after `run_all.sh` returns, after a skipped run, or after any failure. Failures
  include a failed clone and a volume that is not writable (it then logs to `/tmp`). Its exit trap is in place
  before any step that can fail.
- A watchdog process, started first, at `DRBM25_HARD_DEADLINE_UNIX`, whatever the run is doing. That is 1 h
  after the budget deadline, and the evaluation, report and uploads are budgeted inside `RESERVE_S` before the
  budget deadline, so the watchdog only cuts in when they overrun their estimate by more than an hour (the margin
  is kept at least as long as one upload's timeout, `PUBLISH_TIMEOUT_S`, 30 min). A cut-off upload leaves no
  marker; the data stays on the volume.

Every terminate call tries your key (the `drbm25_runpod_api_key` secret) first, then the pod-scoped
`RUNPOD_API_KEY` that RunPod injects into every pod. This covers a secret that was deleted or never replaced.
If the calls keep failing, the watchdog retries every 10 minutes for about 48 hours and logs each attempt, while
the bootstrap waits for it. The scripts do not assume that the container exiting ends billing. If a launch went
wrong, check `--status` and the console.

Restarts: if RunPod restarts the container, the bootstrap runs again with the same environment. It does not
run `run_all.sh` again when this launch already finished (`/workspace/drbm25/.launch-<id>-done`, written when
`run_all.sh` returns). It also does not run it when less than `RESERVE_S` remains before the budget deadline.
In both cases it terminates the pod instead. Otherwise `run_all.sh` resumes from the latest checkpoint against
the same absolute budget deadline.

### Cost

Prices (RunPod API, as of 2026-11-19): 2x A100 SXM 80GB community $2.78/h, secure $4.38/h; 2x A100 PCIe 80GB
community $2.38/h, secure $4.18/h. Only the secure prices apply: network volumes require the Secure Cloud
(https://docs.runpod.io/storage/network-volumes, as of 2026-11-20). `launch.sh` prints the live list
price before you confirm and passes it to the run as `PRICE_PER_HOUR`; if the created pod reports a higher hourly
cost, it prints a warning. Standard network volume storage was $0.07/GB/month for the first TB (RunPod
docs, as of 2026-11-20): 400 GB is about $28/month, billed until you delete the volume.

Suggested budget: **$160** (the project's cap). At $4.18-4.38/h it buys 36.5-38.3 h, one hour of which is
reserved for evaluation, the report and the uploads (`RESERVE_S`, 1.5 h). Whether 1,751 steps fit is not known in advance: the
trainer times the first 10 PPO steps and logs `budget/projected_steps` (to `metrics.jsonl`, the log and W&B),
computed as steps done + (deadline - reserve - now) / mean step time. Read it after the first 10 steps of the
first chunk. Each run's projection counts only its own step time, but both runs share the clock in alternating
chunks, so expect each run to reach roughly half of the projected steps beyond those already done (an inference
from the alternation, not a measurement).

### Watch, stop, resume

```bash
bash scripts/runpod/launch.sh --status
bash scripts/runpod/launch.sh --logs
bash scripts/runpod/launch.sh --watch --interval-min 10
bash scripts/runpod/launch.sh --stop
bash scripts/runpod/launch.sh --stop --yes
```

`--status` prints the pod's id, status, hourly cost and data center, then asks the Hub whether each run's repo
holds `UPLOADED.json` (with its best step) and prints `safe to delete the volume: yes` only when both markers
match the volume's current `best/`, read through the running pod; without a pod it prints `unknown` (see
"Publishing results"). `--logs` runs a fixed read-only command
in the pod through RunPod's exec API and prints the FAILED marker (if any), `spend.json` and the last 40 lines of
the newest `logs/bootstrap-*.log` and `logs/run_all-*.log` (plus `PUBLISH_FAILED` and each run's `UPLOADED.json`). `--watch` repeats status and the last 10 log lines
until the pod is gone. `--stop` terminates the pod after you type `yes` (`--stop --yes` without asking); the
volume and its data are kept, but
the run is cut off at once, so `spend.json` misses the time since the last finished chunk. In the RunPod web
terminal or over SSH: `tail -f /workspace/drbm25/logs/bootstrap-*.log` (then `run_all-*.log`).

Resume: run `bash scripts/runpod/launch.sh` again. It reuses the volume, the pod fetches the ref into the existing
clone, finished stages are skipped and training resumes from the latest checkpoint. Give it the budget that is
left: `--logs` (while a pod exists) or the last run's log shows the total in `spend.json`.

Failures: any failure writes `/workspace/drbm25/FAILED` and terminates the pod. Read the cause with `--logs`
before the pod is gone, or in the logs on the volume. A later launch that finds the marker terminates its pod at
once instead of running (it would only repeat the failure). Fix the cause, then relaunch with
`bash scripts/runpod/launch.sh --clear-failed`: the pod moves the marker to `logs/FAILED-<time>` and runs. If
the clone itself fails (bad `--ref`, network), `pod_bootstrap.sh` terminates the pod. An unattended run never
asks anything: a missing setting, or a secret reference RunPod did not replace (a deleted secret), is a failure.

### What the run does

1. Settings: the environment from `launch.sh`, else `/workspace/drbm25/.env` (mode 600; secrets included,
   never logged or committed). `BUDGET_USD` and `PRICE_PER_HOUR` are never saved. After the settings all output
   goes to `/workspace/drbm25/logs/run_all-<time>.log`.
2. Stages, in order: Java 21 (pinned Temurin tarball, SHA-256 checked), uv 0.13.7 (pinned GitHub release tarball for x86_64 Linux,
   SHA-256 checked, installed into `/workspace/drbm25/bin`; no pip, so Ubuntu 24.04's externally-managed Python does not
   matter), Python 3.12 from `uv python install` (onto the volume, `UV_PYTHON_PREFERENCE=only-managed`, so the image's
   system Python and preinstalled torch are never used), `uv sync --frozen --extra vllm`, dataset and index, model snapshot at the config's pinned revision, GPU smoke test (`pytest
   -m gpu` over `tests/rollout/test_vllm_engine.py` and `tests/ppo/test_models_full.py`; the gpu-marked tests are
   the vLLM engine tests, `test_models_full.py` has none and runs on CPU/gloo in the normal suite; then
   `scripts/memory_check.py` at micro batch 8; if it fails, by
   exiting 1 for an error, an out-of-memory error or under 2 GiB headroom, or by being killed, it is retried at 4,
   then 2; the first size that passes is saved to `/workspace/drbm25/micro_batch` and passed as
   `--set ppo.micro_batch_size=N` to every training chunk, so it is recorded in each run's `run_manifest.json`;
   if 2 fails too, `FAILED` is written and the pod is terminated; a later pod never retries a size larger than
   the saved one, and once either run has a checkpoint the saved size is the only one checked: if it fails on
   a new pod, `FAILED` is written instead of stepping down, so both runs train every step at the same micro
   batch), training, evaluation of each run's `best/` policy, report, and publishing both runs to the Hugging
   Face Hub (`python -m drbm25.publish`, at most 30 min per run including the download-back check; before
   training, each run that will train has its upload markers withdrawn, see "Publishing results"; a failure writes `/workspace/drbm25/PUBLISH_FAILED`,
   logs an `ERROR` and still ends the launch normally and terminates the pod, so a relaunch only needs to publish
   again). Data and model stages leave markers in `/workspace/drbm25/.done/` and are skipped on
   a re-run; Java, uv, `uv sync` and training run every launch (training resumes from the latest checkpoint), and the
   smoke test runs once per pod. The W&B project and entity you entered are applied to training, evaluation and the
   report.
3. Budget: for a pod made by `launch.sh` the deadline is the absolute `DRBM25_BUDGET_DEADLINE_UNIX` (launch
   time + `BUDGET_USD / PRICE_PER_HOUR` hours), so a restarted container gets no fresh budget. In the manual flow
   it is `start + BUDGET_USD / PRICE_PER_HOUR` hours from when `run_all.sh` starts
   (`drbm25.budget.deadline_from_budget`). Spend is counted from the container's start (`DRBM25_POD_START_UNIX`)
   on top of what `spend.json` already holds, so a restart adds only its own time; the gap between the last
   record and a restart is not counted. Spend (elapsed time x price) is written to `/workspace/drbm25/spend.json`
   after every chunk and on exit, so a relaunch shows the total already spent; give it the remaining budget, not the
   original one. Training stops at the next
   checkpoint boundary once one more chunk would end less than `RESERVE_S` (default 5400 s: an unmeasured 1 h
   for four evaluation runs and the report plus 30 min for the two uploads, and the terminate call) before the deadline; override with
   `RESERVE_S=...`. The chunk estimate is the larger of the trainer's own (mean step time x `run.ckpt_every`)
   and the previous chunk's wall-clock, which the launcher measures and passes as `--prev-chunk-s` (it includes
   torchrun start, model loads, vLLM init, validation and the checkpoint save). Both runs advance in chunks of
   `run.chunk_steps` (50), so they stop within one chunk of each other: after a budget stop `count_once` can be
   one chunk behind `normal`. The evaluation scores each run's `best/` policy so far, and the report's
   training steps column shows how far each run got (see "Reading the report").
   The report's W&B upload is best effort: if it fails after `report.md` is written, the launcher logs a
   `WARNING` and still ends successfully (the results are on the volume; sync them by hand).
4. Hangs: each training chunk runs under GNU `timeout` (time to the deadline plus 30 min), evaluation commands under
   2 h and each smoke command under 1 h. A timeout is a failure: `FAILED` is written and the pod is terminated. A `SIGKILL` before the limit (exit 137, for example
   the out-of-memory killer) is reported as that, not as a timeout.

Volume size: at least **400 GB** (not measured on a pod yet). A full fine-tune checkpoint holds fp32 parameters
plus two Adam moments for the actor and the critic, about 74 GB, and `best/` holds the 3B actor in bf16, about
6.8 GB (6.2 GB of parameters plus the tied LM head stored under its own name). Each run keeps one checkpoint and
one `best/`: 74 + 6.8 GB. A save writes the new checkpoint beside the old one, so the run being saved peaks at
2 x 74 + 6.8 = about 155 GB while the other holds about 81 GB (the runs train in turn): about 236 GB, plus the
model cache, dataset, index, venv cache and logs. The run refuses to start a chunk unless 1.2 x (last checkpoint
+ `best/`) is free (7 GiB stands in for a `best/` not written yet; 90 GiB before the first checkpoint). `df` on
a RunPod network volume may report pooled rather than per-volume space, so `VOLUME_GB` (set by `launch.sh` from
the volume's size) also bounds free space as `VOLUME_GB - used` (from `du -sk /workspace/drbm25`). After the first
checkpoint, confirm the real sizes on the pod with `du -sh /workspace/drbm25/runs/*/step_* /workspace/drbm25/runs/*/best`.

Terminating the pod releases GPU billing; the network volume keeps all data. If the run's terminate call fails
after 3 attempts (waits of 5 s and 15 s) it logs the pod id and `https://console.runpod.io/pods`: `--status`
shows the pod is still there; stop it with `--stop`.

### Manual fallback (inside a pod)

Deploy a 2x A100 80GB secure-cloud pod with the volume at `/workspace` yourself, then in its terminal:

```bash
mkdir -p /workspace/drbm25 && cd /workspace/drbm25 && git clone https://github.com/objones25/drbm25 repo && cd repo
tmux new -s drbm25
bash scripts/runpod/run_all.sh
```

Interactively it prompts for the same settings (hidden for secrets) plus `RUNPOD_API_KEY`, `PRICE_PER_HOUR` and
`TERMINATE_ON_FINISH` (default `yes`), and saves all but the budget and price to `/workspace/drbm25/.env`. Ctrl-C,
a closed terminal (SIGHUP) or SIGTERM stops the running command (its process group gets TERM, then KILL after
10 s), records spend, copies logs and terminates the pod; `--no-terminate` or `TERMINATE_ON_FINISH=no` keeps it.
An interactive launch that finds a `FAILED` marker stops before prompting and does NOT terminate the pod: fix the
cause, delete the file and run again, or terminate the pod yourself.

Where things are (all under `/workspace/drbm25`): `runs/{normal,count_once}/` (the sharded resume checkpoint,
`best/`, `run_manifest.json`, `metrics.jsonl`, `train_rank*.log`, `exit_code`, `UPLOADED.json` once published),
`runs/eval/` (per-query jsonl and `report.md`), `PUBLISH_FAILED` (when the upload failed), `repo/` (the clone the pod runs), `logs/` (`bootstrap-*.log`, `run_all-*.log`, copies of the
run logs made on exit, archived `FAILED-*` markers), `micro_batch` (the
micro batch size the smoke stage chose), `spend.json`, `hf/` (model cache), `indexes/`, `data/`, `.env`.

Tests: `bats tests/shell/test_runpod_lib.bats tests/shell/test_runpod_launch.bats` (curl, uv, timeout, df and
sleep are PATH shims; nothing calls the real RunPod API) and `shellcheck scripts/runpod/*.sh`.
