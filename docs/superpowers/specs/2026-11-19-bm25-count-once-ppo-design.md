# Design: PPO query rewriting with normal vs count-once BM25

Date: 2026-11-19. Status: draft for approval.
Team: CS 291A (UCSB, Fall 2026) DeepRetrieval reproduction project.

## 1. Intent

DeepRetrieval (Jiang et al. 2025, arXiv 2503.00223) trains an LLM with PPO to
rewrite queries, using retrieval quality as the reward. Its BM25 prompt asks for
Boolean queries, but Pyserini's default BM25 drops "and"/"or" as stopwords and
multiplies a term's score by how often it repeats in the query (Anserini
`BagOfWordsQueryGenerator` boosts each unique term by its count; confirmed in the
anserini 0.22.1 bytecode). Our question: **how much of the BM25 gain comes from
repeating words?**

We answer it with two experiments on the MS MARCO science subset (MS-S):

1. **Untrained vs untrained.** Score the original queries and the untrained model's
   rewrites under both normal BM25 and count-once BM25.
2. **Trained vs trained.** Train the same model twice, once rewarded by normal BM25
   and once by count-once BM25. Score both trained models under both BM25 versions.

Questions to answer (slide 26): does RL improve retrieval as much when repeats
don't count; does the normal-BM25 model rely on repeats (its score drops under
count-once); do the two models write different kinds of queries.

Success: both training runs finish under identical settings, and we produce the
2x2 (+ baseline) table of dev nDCG@10 with paired confidence intervals, plus query
statistics that show whether the models differ in repetition.

## 2. Goals, non-goals, anti-goals

Priorities (from the team lead)
- **Expanse V100 is the primary path** (a teammate wants cluster practice). It is
  built and verified first. RunPod A100 is the fallback the lead can run alone if
  the team can't be organized or Expanse access (SSH, allocation) doesn't come through.
- Deadline: about 3 weeks from 2026-11-19 (≈ 2026-12-10).
- RunPod budget: ≤ $100 total, ideally.
- Expanse access is not yet working (SSH trouble, no confirmed allocation), so the
  RunPod path must be runnable end to end by one person without Expanse.

Goals
- Our own PPO training code (no verl), faithful to the paper's algorithm.
- Two hardware profiles from one codebase:
  - **Expanse**: 2x V100 32GB, Qwen2.5-1.5B-Instruct, PPO with LoRA.
  - **RunPod**: 2x A100 80GB, Qwen2.5-3B-Instruct (the paper's model), PPO with full
    fine-tuning, matching the paper's setup.
  - The model is a config value; nothing hard-codes a model name or size.
- uv for packages, Hugging Face Hub for models/data, W&B for metrics.
- Tests for every piece of logic that decides a number we report.

Non-goals
- Other DeepRetrieval tasks (PubMed, QA, dense, SQL), GRPO, multi-node training.
- Matching the paper's absolute numbers. We compare our two runs to each other.
- Qwen3.5-2B. It is a vision-language model whose Gated DeltaNet layers have no fast
  V100 kernels, and it needs ~34 GB per fully trained copy vs 28 GB for Qwen2.5-1.5B
  (section 5). It stays out unless the team asks for it later.

Anti-goals (would make the project a failure even if code runs)
- The two runs differ in anything other than the BM25 scorer used for reward
  (model, seed, data order, hyperparameters, steps, code version).
- A silent fallback turns a bug into a number: e.g. a search error scored as 0, a
  parse failure scored as a valid query, NaN loss skipped without stopping.
- Scoring with a BM25 that is not what we claim (wrong k1/b, wrong analyzer, the
  full 8.8M MS MARCO index instead of the 800K collection).

## 3. Constraints and verified facts

| Fact | Source |
|---|---|
| Upstream algorithm is PPO + GAE with a full critic; KL in reward, `kl_coef=0.001`, fixed; `use_kl_loss=False` | upstream `verl/trainer/config/ppo_trainer.yaml:139-145`, `scripts/train/msmarco.sh:32` |
| batch 64, PPO mini-batch 16, 1 PPO epoch, clip 0.2, value clip 0.5, entropy 0.001, grad clip 1.0, γ=λ=1 | `msmarco.sh:14-30`, yaml defaults |
| actor lr 1e-6, critic lr 1e-5, constant schedule, no warmup | `msmarco.sh:18,30`; paper App. B.2 |
| max prompt 256, max response 350 tokens | `msmarco.sh:16-17`; paper App. B.2 (350 for classic retrieval) |
| temperature: paper says 0.6; `msmarco.sh` leaves 1.0; from-scratch `msmarco_beir.sh` uses 0.6 | paper App. B.2; `msmarco_beir.sh:24` |
| Reward: bad format −2; good format & nDCG@10=0 → 0; else 0.1 + nDCG@10. Search k=10 in train and eval | `verl/utils/reward_score/msmarco.py:199-262` |
| Format: response (after the prompt's `<think>`) must contain exactly one each of `<think>`(from prompt),`</think>`,`<answer>`,`</answer>` in order; `<answer>` is JSON with key `query` | `msmarco.py:61-151` |
| Prompt: hand-written ChatML string, no chat template; ends with `Let me rewrite the query with reasoning. \n<think>\n` | `data_preprocess/msmarco_sub.py:34-60` |
| MS-S: 22,416 train, 2,305 dev queries; `{"question","docs_id"}` jsonl; upstream val = first 101 dev rows (we validate on 500 held-out train queries instead, §6.2); test = all dev | HF `DeepRetrieval/datasets` `raw_data/msmarco/msmarco_science/`; `msmarco_sub.py:65-91` |
| Corpus: 800K passages (`collection_800k.tsv`) = all relevant pids of health/science/tech + random fill, seed 42 | HF `raw_data/msmarco/readme.md`, `get_sub_collection.py` |
| BM25 k1=0.9, b=0.4 (Pyserini default; upstream never calls `set_bm25`) | pyserini 0.22.1 `_searcher.py:345` |
| Count-once is buildable: analyze with Pyserini's analyzer, dedupe terms, build a Lucene BooleanQuery of SHOULD TermQuery clauses, pass to `LuceneSearcher.search` | pyserini `_searcher.py:104,139-149`, `querybuilder.py:40-53` |
| vLLM docs list V100 (CC 7.0) through v0.19.0; v0.20.0 requires CC ≥ 7.5. On CC 7.0, v0.19 selects `TRITON_ATTN` (FlashAttention requires ≥ 8.0); Triton's README lists only CC 8.0+ | docs.vllm.ai/en/v0.19.0 vs v0.20.0 install pages; vLLM v0.19.0 `vllm/platforms/cuda.py`, `v1/attention/backends/{flash_attn,triton_attn}.py`; triton README |
| PyTorch dropped Volta from CUDA 12.8 wheels in 2.11; torch 2.10 still has it; vLLM 0.19.0 pins `torch==2.10.0` | dev-discuss.pytorch.org "Dropping Volta support from CUDA-12.8 binaries for release 2.11"; PyPI vllm 0.19.0 metadata |
| V100 has no bf16 (vLLM raises for bf16 below CC 8.0) and no FlashAttention-2 | vLLM v0.19.0 `platforms/cuda.py:check_if_supports_dtype` |
| Expanse `gpu-shared`: ≤ 3 GPUs, 48 h wall time; `gpu-debug`: 30 min, ≤ 2 GPUs; Lustre jobs need `--constraint="lustre"`; local NVMe at `/scratch/$USER/job_$SLURM_JOB_ID`; home 100 GB, not for running jobs | sdsc.edu Expanse user guide |
| Unknown: whether Expanse compute nodes have internet; which CUDA driver Expanse runs; Java module availability | not stated in the user guide |

## 4. Approaches considered

For the V100 profile (full PPO with critic does not fit: 55.6 GB of optimizer
state for 1.5B before activations, vs 64 GB total):

- **A. PPO with LoRA (chosen by the team).** Same algorithm as the paper. Actor
  and critic are two LoRA adapters on one frozen fp16 base; the reference policy
  is the base with adapters disabled (free). Fits easily; lets 3B fit too.
  Tradeoff: LoRA needs a higher learning rate than the paper's 1e-6, so those two
  numbers differ from the paper.
- B. GRPO, full fine-tuning: no critic, ~14 GB/GPU sharded. Rejected: departs from
  the paper's PPO and needs several samples per query, slow without vLLM.
- C. Paper PPO, full fine-tuning with CPU offload / 8-bit Adam. Rejected: very slow on V100.

For the A100 profile, the team chose **paper-faithful PPO with full fine-tuning**
of Qwen2.5-3B-Instruct (actor and critic), FSDP over 2 GPUs, bf16, vLLM rollouts.

## 5. Memory sizing (computed)

Full fine-tune with AdamW and fp32 master weights ≈ 18 bytes/parameter.

| Model | Params | One fully trained copy | Actor + critic | fp16/bf16 weights |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.544B (matches 3.087 GB bf16 file) | 27.8 GB | 55.6 GB | 3.1 GB |
| Qwen3.5-2B | ~1.9B text (2.27B incl. vision, from 4.55 GB file) | 34.2 GB | 68.4 GB | 3.8 GB |
| Qwen2.5-3B-Instruct | 3.09B | 55.5 GB | 111 GB | 6.2 GB |

- **V100 LoRA profile, per GPU:** fp16 base 3.1 GB + LoRA adapters/optimizer < 1 GB
  + activations with gradient checkpointing (micro-batch 4 × ~610 tokens) + fp32
  logits for a micro-batch (~1.2 GB at 4×512×151,936×4 B) + generation KV cache
  (28 KiB/token → 32 seqs × 610 tokens ≈ 0.55 GB). Expected well under 32 GB.
- **A100 full profile, per GPU:** actor and critic states sharded by FSDP
  (111 GB / 2 = 55.5 GB) + reference model sharded (3.1 GB) + colocated vLLM
  (~0.2 × 80 = 16 GB) ≈ 75 GB before activations. **Tight.** Upstream ran this
  exact configuration on 2× A100 80GB with `gpu_memory_utilization=0.2`, gradient
  checkpointing, and reference-model offload. Our first A100 step measures it;
  mitigations in order: reference model on CPU, vLLM sleep mode during training,
  lower vLLM memory fraction.

## 6. Design

### 6.1 Package layout (uv project, `src/` layout)

The tree as built (2026-11-20; `git ls-files src scripts configs`):

```
pyproject.toml, uv.lock, .python-version
configs/            expanse_v100_lora.yaml, runpod_a100_full.yaml, smoke_cpu.yaml
src/drbm25/
  config.py         frozen dataclasses; YAML loaded with yaml.safe_load; validated at load
  data.py           MS-S download and parsing; typed Example; seeded train/val split and data order
  prepare.py        download MS-S + build the BM25 index once, atomically, idempotently
  prompt.py         the upstream sparse prompt, verbatim
  reward.py         pure: parse response -> ParsedRewrite | FormatError; nDCG@k; reward
  checkpoint.py     atomic resume checkpoints (step_*/latest) and the selected policy (best/)
  manifest.py       run_manifest.json: seeds, resolved config + sha256, versions, hardware
  budget.py         dollar budget -> deadline; the chunk stop rule
  distributed.py    process-group setup, manual gradient averaging (no DDP wrapper)
  tiny.py           tiny random Qwen2 + tokenizer for tests and CPU smoke runs
  train.py          resumable PPO training CLI (python -m drbm25.train)
  bm25/
    index.py        build the Lucene index for the 800K collection (pyserini)
    searcher.py     ScoringMode = Literal["normal","count_once"]; Bm25Searcher.search(q, mode, k)
  rollout/
    base.py         RolloutEngine protocol, RolloutBatch, shared prompt/response handling
    hf_generate.py  batched HF generate (V100; CPU runs)
    vllm_engine.py  colocated vLLM, weight sync after each update (A100)
  ppo/
    gae.py          token-level KL-in-reward + GAE (gamma = lambda = 1), pure tensor functions
    losses.py       clipped policy loss, clipped value loss, entropy, pure
    models.py       PolicyModels protocol; LoRA actor/critic on a shared base (V100)
    models_full.py  FSDP2 actor/critic/reference, full fine-tune (A100)
    trainer.py      one PPO step: rollout -> reward -> ref/old logprobs, values -> GAE -> update
  eval/
    records.py      QueryRecord and the per-source jsonl format (with a provenance header)
    run_eval.py     Exp 1 and Exp 2 scoring of one source; writes per-query jsonl
    stats.py        paired bootstrap CI; query statistics
    report.py       report.md (+ optional W&B tables and artifact) from the jsonl files
  obs/              logging and W&B plumbing: setup.py, json_formatter.py, context.py,
                    redact.py, metrics.py, wandb_run.py, runs.py
scripts/
  prepare_data.py, memory_check.py, spike_monitor.py
  audit_tests.py, audit_observability.py, lint_pytorch.py, diff_guard.py, mutate_check.py
  expanse/          common.sh, run.sh, prepare_login.sh (login node), probe.sbatch
                    (gpu-debug), train.sbatch, submit_chain.sh, eval.sbatch
  runpod/           launch.sh (laptop launcher), pod_bootstrap.sh (pod start command),
                    run_all.sh (in-pod pipeline), lib.sh
tests/
```

### 6.2 Data and prompt
- Download `DeepRetrieval/datasets` subsets with `huggingface_hub` (MS-S
  train/dev jsonl, `collection_800k.tsv`), pinned to a dataset revision hash.
- Prompt: upstream sparse prompt verbatim (including its stray quote), raw ChatML
  string, no chat template, so our untrained baseline matches upstream's input.
  The prompt ends inside `<think>`.
- Splits (changed 2026-11-20, user decision: "val set should be separate from test
  set. Maybe split train into val + train."): validation = 500 train queries drawn
  by `numpy.random.default_rng(data.split_seed).permutation` (`data.split_seed` =
  1234, separate from `run.seed`); training = the other 21,916 train queries; test
  (final evaluation) = all 2,305 dev rows, untouched, so the Exp 1 baseline stays
  valid. Validation and test share no query (by qid and by text; the 22,416 train
  and 2,305 dev query texts are all distinct). Upstream used the first 101 dev rows
  for validation; we do not, because validation now selects the reported
  checkpoint and must not see the test set. Training data order fixed by seed and
  identical across the two runs.

### 6.3 BM25 scoring (the experimental variable)
- One Lucene index of the 800K collection, built once with Pyserini
  (`JsonCollection`, default English analyzer, `-storeRaw`), checksum recorded.
- `normal`: pass the query string to `LuceneSearcher.search` (Anserini bag-of-words,
  repeated terms boosted by count). k1=0.9, b=0.4 set explicitly, not left to defaults.
- `count_once`: analyze the query with the same analyzer as the index, keep unique
  terms in first-seen order, build a BooleanQuery of SHOULD TermQuery clauses with
  boost 1, call `search(query_obj, k)`. Must not use `querybuilder.get_term_query`
  (it re-analyzes and keeps only the first token).
- Same index, analyzer, k1, b, k for both modes; the only difference is qtf → 1.
- An empty query after analysis is a legal input: zero hits, nDCG 0 (counted in a
  metric). Lucene `TooManyClauses` (> 1024 unique terms) raises a specific error,
  is counted, and the sample gets reward 0. Any other search exception crashes.
- Searches for a batch run in a bounded thread pool (JVM is thread-safe for search);
  each worker has a timeout.
- Training reward uses k=10 (upstream code). Evaluation retrieves k=10 for nDCG@10.

### 6.4 Reward
- `parse_response(text) -> ParsedRewrite | FormatError` mirrors upstream's checks
  exactly (tag counts and order, last `<answer>` block, JSON with string `query`).
  FormatError carries the reason (logged as a category count).
- `reward = -2` on FormatError; `0` if nDCG@10 = 0; else `0.1 + nDCG@10`. Binary
  relevance, ideal DCG over min(|relevant|, 10). Asserted range [−2, 1.1].
- Fixes vs upstream, on purpose: no `processed_str` unbound-variable path; a search
  error is not silently scored 0.

### 6.5 PPO (one algorithm, two memory layouts)
Shared core (pure functions, unit tested):
- Per-token KL penalty in the reward: `r_t = −β (logπ_old − logπ_ref)`, plus the
  scalar task reward on the last response token. β = 0.001, fixed.
- GAE with γ=λ=1 → advantage = return − value; advantages whitened per batch (as verl does).
- Policy loss: clipped ratio, ε = 0.2, token-mean over valid response tokens.
- Value loss: clipped, 0.5 range. Entropy bonus 0.001. Grad clip 1.0.
- Per step: 64 prompts, one sample each (n=1), 4 mini-batches of 16, 1 PPO epoch.
- Temperature 0.6, top-p 1.0 (paper text; the team can override — see open items).

V100 / LoRA layout (`configs/expanse_v100_lora.yaml`):
- Model `Qwen/Qwen2.5-1.5B-Instruct`, fp16 frozen base, SDPA attention, gradient checkpointing.
- Two adapters on one base: `actor` (LoRA on all linear layers) and `critic`
  (LoRA + scalar value head on the final hidden state). Reference = adapters disabled.
- LoRA params and value head in fp32; autocast fp16; `torch.amp.GradScaler`.
- 2 processes via `torchrun`, data parallel: each rank generates and trains on 32 of
  the 64 prompts; DDP all-reduces only the trainable parameters.
- Rollout: HF `generate` with the actor adapter. vLLM is opt-in after the V100 smoke test.
- Starting values (to tune in a short pilot, then frozen for both runs): rank 32,
  alpha 64, dropout 0, actor lr 2e-5, critic lr 1e-4.

A100 / full layout (`configs/runpod_a100_full.yaml`):
- Model `Qwen/Qwen2.5-3B-Instruct`. Separate actor and critic (critic = backbone +
  value head), FSDP2 over 2 GPUs, fp32 master weights, bf16 compute, fp32 reduce.
  Frozen bf16 reference model, sharded or offloaded.
- Rollout: one colocated vLLM engine per rank (TP=1, bf16), weights pushed from the
  actor after each update; generation is data-parallel over ranks.
- Paper learning rates: actor 1e-6, critic 1e-5.

### 6.6 Run protocol
- Each training run = fixed number of PPO steps (`total_steps`), same for both runs.
  Target 1,751 steps: upstream's 5 epochs over its 22,416 train rows
  (5 × 22,416 / 64 = 1,751.25). We train on 21,916 rows (500 are held out for
  validation, §6.2), so 1,751 steps of 64 are ≈ 5.11 epochs of our train split.
  The real cap comes from the budget (below).
- **Equal progress under any interruption.** The two runs advance in alternating
  chunks of `chunk_steps` (default 50): run normal to step 50, checkpoint, run
  count-once to step 50, checkpoint, and so on. If money or time runs out, both runs
  stop within one chunk of each other; evaluation then scores each run's best
  validated policy so far. On Expanse, where two 2-GPU jobs can run side by side, the two runs go as
  two concurrent jobs instead, and alternation isn't needed.
- **Budget guard (RunPod).** The launcher takes the dollar budget and the pod's
  hourly price, and turns them into a wall-clock deadline. The trainer stops at the
  next checkpoint boundary once one more chunk would end inside the reserve needed
  for final evaluation and upload. The chunk estimate is the larger of the trainer's
  own (mean PPO step time × steps per chunk) and the previous chunk's wall-clock,
  which the launcher measures and passes as `--prev-chunk-s`, so per-chunk overhead
  (torchrun start, model loads, vLLM init, validation, the ~74 GB sharded save) is
  counted. Once either run has a checkpoint, the launcher keeps the micro-batch
  size fixed: a new pod that cannot fit it fails loudly instead of stepping down, so
  both runs train every step at the same size. The report's W&B upload is best
  effort: a failure after `report.md` is written is a warning, not a failed run. Live prices
  (RunPod API, 2026-11-19): 2× A100 SXM 80GB community $2.78/h, secure $4.38/h;
  2× A100 PCIe 80GB community $2.38/h, secure $4.18/h. $100 ≈ 36–42 h on community
  cloud, ≈ 23–24 h on secure. Community availability was listed as LOW.
- **Step-time calibration.** The first 10 PPO steps of the first chunk are timed,
  and the launcher logs the projected steps reachable within budget, so the lead
  sees early whether 1,751 steps is realistic. On 3B full fine-tuning it likely isn't
  in $100 (unmeasured; to be confirmed by calibration).
- Seeds: one seed per run (budget), same seed for both runs. Uncertainty comes from
  paired bootstrap over the 2,305 dev queries, not over seeds — stated as a limitation.
- Validation every 50 steps and at the final step on the 500-query validation split
  (§6.2), greedy, scored with **both** BM25 modes regardless of which one is the
  reward.
- **Best-checkpoint selection** (user request 2026-11-20: "1 checkpoint per run,
  keep the best one"). The selection metric is validation nDCG@10 under the run's
  own reward mode (`val/ndcg_normal` for the normal run, `val/ndcg_count_once` for
  the count-once run). When it strictly beats the stored best, `best/` in the run
  directory is replaced atomically (temp dir, move-aside, rename); ties keep the
  earlier step. `best/` holds the policy only (LoRA: the actor adapter; full FT:
  the gathered actor in bf16 with HF names, about 6.8 GB at 3B) and `meta.json`
  (step, both validation scores, selection metric, reward mode, config sha256,
  seeds, git sha and dirty flag). With 2 ranks, whenever generation is per-rank
  and independent (the LoRA layout, and the vLLM backend on either layout, i.e.
  the A100 run) validation is sharded across ranks (rank r takes rows r::2) and
  the nDCG sums and counts are all-reduced; separate vLLM engines are not
  guaranteed bit-identical greedy output, so validating every row on every rank
  and comparing would crash the run. Only the full layout with the `hf` backend
  validates every row on every rank (its FSDP forwards inside `generate` are
  collectives). All ranks get identical reduced scores and reach the same
  decision (checked on the reduced values). The
  process group's collective timeout is explicit (`run.dist_timeout_min`, 60). A
  resume restores the best-so-far score from `best/meta.json` without
  re-evaluating. Exp 2 reports each run's `best/` policy, so the two runs may be
  compared at different steps; the report shows each run's best step and its
  validation score.
- Checkpoint every 50 steps (LoRA adapters + optimizer + scaler + RNG + data cursor;
  full-FT sharded state on A100), keeping one resume checkpoint per run
  (`keep_ckpts: 1`); `best/` is separate and never pruned. Resume is bit-for-bit on
  data order, so a job can span several 48 h Expanse allocations.
- **Reproducibility manifest.** Every training invocation writes
  `run_manifest.json` (rank 0): every seed and its derivation (`run.seed` for model
  construction, data order and per-rank sampling and vLLM seeds = `run.seed + rank`;
  `data.split_seed` for the split), the resolved config and its sha256, git sha and
  dirty flag, dataset / model revisions, index checksum and size, library versions,
  CUDA, driver, GPUs, world size, host and start time. A resume appends to
  `resumed_at` and is refused if a config seed changed. The same dict is the W&B
  `config.meta`.
- Final evaluation uses greedy decoding (deterministic), max 350 new tokens.

### 6.7 Evaluation outputs
- Exp 1: rows = {original query, untrained rewrite}; columns = {normal, count-once} nDCG@10.
- Exp 2: rows = {trained-normal, trained-count-once}, each run's `best/` policy
  (§6.6); same columns, plus each run's best step, its validation score and its
  training steps (the run's latest complete checkpoint when evaluated, recorded in
  the eval jsonl header as `trained_steps`). A budget stop can leave count-once one
  chunk behind normal; the report then notes that the runs trained for different
  numbers of steps.
- Test set = all 2,305 dev queries for every row (never used for selection).
- Per-query jsonl with query text, rewrite, analyzed terms, both scores and the
  `<think>` length in characters; its first line is a header with the source,
  checkpoint, eval config sha256 and seed, training steps, and for `best/` the step,
  validation scores, training config sha256 and seeds.
- Paired bootstrap 95% CI (10,000 resamples, seed `--bootstrap-seed`, default 0,
  recorded in the report) for each difference the questions ask about.
- Query statistics per model: analyzed length, unique terms, **repeat ratio**
  (1 − unique/total analyzed terms, pooled over the source's terms: repeated
  occurrences over all occurrences; the training metric `rewrite/repeat_ratio` is
  the per-query mean instead, so the two need not agree, and the report says so),
  share of rewrites with any repeat, format-error rate, mean `<think>` length in
  characters (stripped reasoning before `</think>`, rewrites without a format
  error only). Term statistics cover rewrites without a format error.

### 6.8 Observability
- stdlib logging, JSON lines to stdout and file, tagged with run id, rank, step.
  Only rank 0 logs to W&B.
- W&B: full resolved config + git SHA + index checksum + dataset revision in
  `run.config` (secrets never). Each run is a separate W&B run (`group` = experiment,
  `job_type` = reward mode); resumed runs reattach to the same W&B run id, stored in
  the checkpoint. The PPO step counter is the x-axis via `define_metric`.
- Every step:
  - Reward and task: reward mean/std/min/max; format-error rate split by reason;
    zero-nDCG rate; train nDCG@10 under the reward mode **and** the other mode;
    rewrite length; repeat ratio; share of rewrites with any repeat; search time.
  - PPO health: policy loss, value loss, entropy, KL to reference (mean and max
    per token), approx KL between old and new policy, clip fraction, value clip
    fraction, explained variance of the critic, advantage mean/std (before whitening),
    returns mean, value mean.
  - Optimization: learning rate per param group (actor, critic), global grad norm
    before clipping for actor and critic, share of steps clipped, GradScaler scale
    and skipped-step count (V100), update-to-weight ratio (‖Δθ‖/‖θ‖) per group.
  - System: time per phase (generate, reference/old logprobs, values, search,
    update, weight sync), tokens/s for generation and training, peak and reserved
    GPU memory per rank, host RAM.
- Every `deep_log_every` steps (default 10): per-layer statistics, gathered
  with hooks only on those steps so normal steps pay nothing:
  - per-layer grad norm and weight norm (or LoRA A/B norms on V100) for each
    decoder block's attention and MLP;
  - activation statistics of each decoder block's output (mean, std, max |x|,
    fraction of non-finite values), captured with forward hooks on one micro-batch;
  - histograms of per-token KL, advantages and response lengths.
- Leading indicators of instability (transformer-architecture skill,
  `references/stability.md`): final-layer output norm, max attention logit on the
  deep-log micro-batch (flag above 1e4), and the spike score of the policy loss and
  grad norm. These are logged and fed through the skill's `spike_monitor.py`
  after each chunk.
- Generation sanity: pad token ≠ EOS token (Qwen2.5 uses `<|endoftext|>` for padding
  and `<|im_end|>` for EOS; asserted at startup), left padding for generation, and
  the share of responses that hit `max_new_tokens` without emitting EOS.
- Every 50 steps: validation nDCG@10 under both modes, and a W&B table of 8 sample
  rewrites (query, rewrite, analyzed terms, reward under both modes).
- Final evaluation summaries and per-query jsonl are logged as W&B artifacts:
  `report --wandb` logs the tables and one `eval-results` artifact (type
  `evaluation`) holding every input jsonl and `report.md`.
  LoRA adapters (small) are logged as artifacts; full-FT checkpoints stay on disk.
- Alerts: `run.alert` on NaN abort, budget stop, and run completion, so the lead
  is notified without watching.
- `WANDB_MODE=offline` when compute nodes lack internet; `wandb sync` from the login node.
- Final eval summaries and per-query jsonl logged as W&B artifacts. Checkpoints stay on disk.

### 6.9 Failure handling (negative space)
- Crash immediately on: NaN/Inf in loss, logprobs, values, or advantages; reward
  outside [−2, 1.1]; shape/mask mismatches between response tokens and logprobs;
  GradScaler scale collapsing below a floor for N consecutive steps; ratio
  exploding (approx KL > a bound) for N consecutive steps.
- Bounded everything: total steps, generation tokens, search thread pool and
  timeout, download retries (3, with backoff), resume attempts.
- Config validation at load: batch divisible by mini-batch × world size;
  precision legal for the GPU (no bf16 on CC < 8.0); reward mode ∈ ScoringMode.
- Startup checks: `torch.cuda.get_arch_list()` contains the running GPU's arch;
  Java version; index document count = 800,000.

### 6.10 Environment and platforms
- uv project, Python pinned in `.python-version`, one `uv.lock`.
- Target pins (verified against the registry during planning):
  `torch==2.10.0` (last release with Volta in its CUDA 12.8 wheels),
  `vllm==0.19.0` (pins torch 2.10.0; last release whose docs list V100) as an
  optional extra, `transformers`, `peft`, `pyserini`, `wandb`, `huggingface_hub`.
  One lock serves both platforms.
- Every library call used is checked during implementation against the installed
  version with `inspect.signature` / `inspect.getsource`, plus Context7 docs.
- Java (JDK version required by the pinned Pyserini) installed in user space if no module exists.
- Expanse: code in home; venv, HF cache, index, checkpoints on Lustre project
  storage; model/data/index prepared on the login node; `HF_HUB_OFFLINE=1` in jobs.
  2 GPUs in `gpu-shared`, ≤ 48 h per job, chained via resume.
- Expanse is designed to work **offline on compute nodes**. Whether those nodes have
  internet is unknown (not stated in the user guide). Login-node prep downloads the
  model, data and index. Jobs set `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline`, then
  `wandb sync` runs from the login node. The first job on the cluster is a
  30-minute `gpu-debug` probe that records: internet reachability, `nvidia-smi`
  driver/CUDA, `torch.cuda.get_arch_list()`, Java availability, and whether vLLM
  0.19 can generate in fp16 on a V100. If the probe shows internet, online W&B is enabled.

RunPod (amended 2026-11-20): the lead runs `scripts/runpod/launch.sh` on their own machine. It checks
live 2x A100 80GB stock and price, asks for `yes`, stores the keys as RunPod secrets, reuses or creates
the `drbm25` network volume and creates a pod whose start command (`pod_bootstrap.sh`) clones the
public repository and runs `scripts/runpod/run_all.sh` unattended (`DRBM25_NONINTERACTIVE=1`: no
prompts; a missing setting is a failure that terminates the pod). The README has the runbook.
`run_all.sh` can still be run by hand inside a pod:
1. **Prompts** (interactively, hidden input for secrets; or reads them from env)
   for: `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` (optional),
   `RUNPOD_API_KEY` (needs `pods:write`, used only to terminate the pod),
   dollar budget, hourly price, and whether to terminate on finish (default yes).
2. Writes them to `/workspace/drbm25/.env` with mode 600. On a re-run it reuses
   the file and asks only for what's missing. The `.env` is never logged, never
   committed, and never sent to W&B.
3. Installs system deps (Java), uv, and `uv sync --frozen` from the committed lock.
4. Downloads the model and data into `/workspace` (the mounted network volume),
   then builds the Lucene index there if absent. Each step is skipped when its
   output already exists and its checksum matches, so re-runs are cheap.
5. Runs the GPU smoke test (one PPO step, memory report) and stops on failure.
6. Runs both training runs (alternating chunks), resuming automatically from the
   latest complete checkpoint of each run on `/workspace`.
7. Runs the final evaluation for Exp 1 and Exp 2 and uploads summaries to W&B.
8. Terminates the pod through the RunPod API (`DELETE /v2/pods/$RUNPOD_POD_ID`).
   Termination releases compute and stops billing. The network volume is only
   detached and keeps its data (RunPod docs, "Terminate a pod"). The script also
   terminates on a fatal error, after writing logs to the volume, so a crash at 3am
   doesn't keep billing. A `--no-terminate` flag is there for debugging.
- Checkpoints on `/workspace/drbm25/runs/<run>/step_<N>/`: written to a temp
  directory, fsynced, then renamed, with a `latest` pointer updated last. A partial
  write is never mistaken for a checkpoint. The last 2 are kept per run.
- Resume: rerunning `run_all.sh` on any new pod with the same volume continues
  from `latest` for each run, including the W&B run id, optimizer, scaler, RNG and
  data cursor.
- HF token and W&B key come from environment variables (`HF_TOKEN`,
  `WANDB_API_KEY`); never written to files or logs.

### 6.11 Testing
- Test-driven: every unit of logic gets a failing test first, seen failing for the
  right reason, then the code. Bugs found later get a reproducing test first.
- The pytest-expert gates apply: strict config, `audit_tests.py` clean, ruff,
  random order, `diff_guard.py` against the pre-change commit, and a mutation check
  on the reward, nDCG, count-once query builder, GAE and loss functions.
- The pytorch skill's linter (`lint_pytorch.py`) runs on all training code, and its
  `diagnose.py overfit` check runs on the tiny PPO model.
- The observability skill's audit (`audit_observability.py`) must exit 0 on `src/`.
- Shell scripts (RunPod launcher, sbatch files) get ShellCheck, plus bats tests for
  the prompt/.env/skip-if-done/terminate logic, with the RunPod API stubbed.
- pytest; unit tests run on CPU in seconds:
  - reward parsing: every upstream format case, plus property tests (Hypothesis)
    that `parse_response` never raises on arbitrary text.
  - nDCG: hand-computed cases (rank 1 → 1.0, rank 3 → 0.5, none → 0) and bounds.
  - BM25: tiny toy index (slide 23's "solar efficiency" example) proving that
    `normal` flips the ranking with "solar solar solar efficiency" and
    `count_once` does not; count-once on a query without repeats equals normal.
  - GAE/KL/losses vs hand-computed values; clipping behavior at the boundaries.
  - One PPO step on a tiny randomly initialized Qwen2 config on CPU: loss finite,
    LoRA params change, base params don't, reference logprobs unchanged.
  - Config validation rejects bad combinations.
- GPU tests (marked, run on the cluster): memory fits for each profile, one real
  PPO step, resume reproduces the same next batch.
- The test audit tools from the pytest-expert skill run in CI-less local mode.

## 7. Calls made by the designer (team can override)
- Temperature 0.6 (paper text) instead of upstream script's default 1.0.
- LoRA rank 32, alpha 64, actor lr 2e-5, critic lr 1e-4, pending a short pilot.
- Greedy decoding at evaluation; k=10 retrieval for training and evaluation.
- Search errors other than TooManyClauses crash rather than scoring 0.
- Paired bootstrap CIs over queries, since only one seed per run is affordable.
- Same codebase serves both platforms; one lockfile; vLLM optional on V100.
- Format-error rewrites score nDCG 0 at evaluation and stay in the average, so
  every row covers the same 2,305 queries. Upstream's eval excluded failures.
- LoRA pilot: ≤ 150 steps under **both** reward modes for each candidate setting.
  The pick is the setting with the best mean validation-nDCG gain across the two
  modes, with no NaN or scaler collapse. If no candidate improves validation nDCG
  over the untrained model, the team is told before the real runs start.
- Alternating 50-step chunks on RunPod so both runs always have equal steps.
- RunPod pods are terminated (not stopped) at the end. Stopping would keep billing
  for the container disk, and everything we keep is on the network volume.
- Expanse jobs assume no internet on compute nodes until the probe shows otherwise.

## 8. Left to the builder
- Exact library versions beyond the pins above (resolve and verify at planning).
- Thread-pool size and search timeout values.
- Value head initialization details.
- File formats for checkpoints and per-query outputs.

## 9. Open items for the team
- Expanse SSH access and allocation (lead is confirming with the professor). The
  V100 code is built and tested on CPU plus the RunPod GPUs in the meantime. The
  first Expanse job is the `gpu-debug` probe.
- Whether 3B full fine-tuning reaches enough steps within $100. The calibration
  numbers decide; if not, the options are fewer steps or Qwen2.5-1.5B full FT on A100.
- Parameter and KV-cache numbers in section 5 were rechecked with the
  transformer-architecture skill's `config_budget.py --from-hf-config` on the
  official configs: 1.54B / 3.09B total, 28 / 36 KiB KV cache per token. They match.
