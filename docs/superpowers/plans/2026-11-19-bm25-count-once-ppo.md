# PPO Query Rewriting: Normal vs Count-Once BM25 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Our own PPO trainer that rewrites MS-S queries for BM25, trained twice
(reward = normal BM25 vs count-once BM25), plus the evaluation for Experiments 1
and 2. It runs on Expanse 2× V100 (LoRA, primary) and RunPod 2× A100 (full
fine-tune, fallback).

**Architecture:** Pure, CPU-tested core (reward, nDCG, GAE, losses, budget, data
order) under a thin orchestration layer (`PPOTrainer`). Two memory layouts sit
behind one `PolicyModels` protocol (LoRA on a shared base; FSDP2 full fine-tune),
and two rollout engines behind one `RolloutEngine` protocol (HF `generate`;
colocated vLLM). BM25 is Pyserini over one 800K Lucene index; `count_once` builds
a BooleanQuery of de-duplicated analyzed terms. Each platform has its own shell
entry point (sbatch on Expanse, `run_all.sh` on RunPod) that calls the same Python CLIs.

**Tech Stack:** Python 3.12, uv, torch 2.10.0, transformers 4.57.6, peft 0.21.2,
pyserini 2.0.0 (Java 21), vllm 0.19.0 (Linux extra), wandb 0.31.0, pytest 9.2,
Hypothesis, ruff, ShellCheck, bats.

**Spec:** `docs/superpowers/specs/2026-11-19-bm25-count-once-ppo-design.md` (read it first).

## Global Constraints

- `.python-version` = `3.12`; `[tool.uv] exclude-newer = "2026-11-05T00:00:00Z"`, so
  every resolved package is at least two weeks old.
- Pins: `torch==2.10.0`, `transformers==4.57.6`, `peft==0.21.2`, `pyserini==2.0.0`,
  `wandb==0.31.0`, extra `vllm = ["vllm==0.19.0; sys_platform == 'linux'"]`.
  Reason: vLLM 0.19 needs `transformers<5`, pyserini ≥ 2.1 needs `transformers>=5`;
  torch 2.10 is the last release whose CUDA 12.8 wheels include sm_70 (V100).
- Java 21 is required by pyserini 2.x (pyserini README).
- Never use bf16 on compute capability < 8.0; V100 uses fp16 + `torch.amp.GradScaler`.
- `attn_implementation="sdpa"` everywhere (no FlashAttention-2).
- BM25: k1 = 0.9, b = 0.4, set explicitly; same index, analyzer, k1, b, k for both modes.
- Reward: −2 format error; 0 if nDCG@10 = 0; else 0.1 + nDCG@10. Range [−2, 1.1].
- PPO: batch 64, mini-batch 16, 1 PPO epoch, clip 0.2, value clip 0.5, entropy 0.001,
  KL-in-reward β = 0.001 (fixed), γ = λ = 1, grad clip 1.0, n = 1 sample per prompt,
  temperature 0.6, top-p 1.0, max prompt 256 tokens, max response 350 tokens.
- LR: full FT actor 1e-6 / critic 1e-5; LoRA starting point actor 2e-5 / critic 1e-4,
  rank 32, alpha 64, dropout 0 (pilot may change; then frozen for both runs).
- Model names are config values only: `Qwen/Qwen2.5-1.5B-Instruct` (V100),
  `Qwen/Qwen2.5-3B-Instruct` (A100).
- Secrets only from env vars `HF_TOKEN`, `WANDB_API_KEY`, `RUNPOD_API_KEY`; never in
  logs, W&B config, or git.
- Every library call is checked against the installed version with
  `inspect.signature` / `inspect.getsource` before use (Context7 docs as backup).
- Gates before any task is done: `uv run pytest`, `uv run ruff check .`,
  `uv run python scripts/audit_tests.py tests/`,
  `uv run python scripts/lint_pytorch.py src/`,
  `uv run python scripts/audit_observability.py src/` (all exit 0),
  `shellcheck scripts/**/*.sh` for tasks that touch shell.
- Conventional Commits; work on branch `feat/ppo-bm25` (from `docs/design-spec`).
  Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A rewrite that analyzes to zero terms (`"AND OR ( )"`) → zero hits, nDCG 0,
   reward 0 in both modes, counted in `search/empty_query_rate`; no crash. (Task 4)
2. A response cut off at 350 tokens without `</answer>` → `FormatError(TAG_COUNT)`,
   reward −2, counted in `rollout/hit_max_tokens_rate`. (Tasks 3, 10)
3. A crash during a checkpoint write leaves a `.tmp-*` directory → resume ignores it
   and loads the previous complete checkpoint. (Task 11)
4. A rewrite with > 1024 unique analyzed terms → `TooManyClausesError`, reward 0,
   counted in `search/too_many_clauses_rate`; any other search exception propagates. (Task 4)
5. RunPod terminate call fails (network/API 5xx) → 3 retries with backoff, then a
   loud error naming the pod id and the console URL, exit code 1. (Task 17)

---

## File map

```
pyproject.toml, uv.lock, .python-version, ruff.toml
configs/{smoke_cpu,expanse_v100_lora,runpod_a100_full}.yaml
src/drbm25/
  __init__.py
  config.py          typed config, YAML load, validation
  prompt.py          upstream sparse prompt, verbatim
  data.py            HF download, Example, splits, deterministic data order
  reward.py          parse_response, ndcg_at_k, compute_reward
  budget.py          deadline + stop decision (pure)
  checkpoint.py      atomic save/load/latest/prune
  distributed.py     DistInfo, init/teardown, allreduce_mean_grads
  bm25/index.py      tsv→jsonl, build Lucene index, IndexInfo
  bm25/searcher.py   Bm25Searcher (normal | count_once), errors
  ppo/gae.py         KL-penalised rewards, GAE, whitening
  ppo/losses.py      policy/value/entropy losses, masked_mean
  ppo/models.py      PolicyModels protocol, token_logprobs, LoraPolicyModels
  ppo/models_full.py FullPolicyModels (FSDP2)
  ppo/trainer.py     PPOTrainer.step
  rollout/base.py    RolloutBatch, RolloutEngine protocol
  rollout/hf_generate.py, rollout/vllm_engine.py
  obs/               obslog package copied from observability-expert assets
  obs/metrics.py     per-layer grads, activation probe, update ratio, qk bound
  train.py           CLI entry (torchrun)
  eval/run_eval.py, eval/stats.py, eval/report.py
scripts/
  audit_tests.py, diff_guard.py, mutate_check.py      (from pytest-expert)
  lint_pytorch.py                                     (from pytorch skill)
  audit_observability.py                              (from observability-expert)
  prepare_data.py                                     download + index build CLI
  expanse/{common.sh,probe.sbatch,prepare_login.sh,train.sbatch,submit_chain.sh,eval.sbatch}
  runpod/{lib.sh,run_all.sh}
tests/  (mirrors src/; tests/shell/*.bats; tests/golden/)
README.md
```

---

### Task 1: Project scaffold and test gates

**Files:**
- Create: `pyproject.toml`, `.python-version`, `ruff.toml`, `src/drbm25/__init__.py`,
  `tests/conftest.py`, `tests/test_package.py`, `scripts/audit_tests.py`,
  `scripts/diff_guard.py`, `scripts/mutate_check.py`, `scripts/lint_pytorch.py`,
  `scripts/audit_observability.py`, `scripts/spike_monitor.py` (from transformer-architecture), `README.md`

**Interfaces:**
- Produces: importable package `drbm25` with `__version__: str = "0.1.0"`; pytest
  markers `java`, `gpu`, `integration`, `slow` registered (strict); `gpu` and
  `integration` deselected by default via `addopts = "-m 'not gpu and not integration'"`.

- [ ] **Step 1:** `git checkout -b feat/ppo-bm25`. `uv init --lib --name drbm25 --python 3.12`,
  then set `exclude-newer` and the pins from Global Constraints with `uv add`
  (`uv add --optional vllm "vllm==0.19.0; sys_platform == 'linux'"`; dev group:
  `pytest pytest-cov hypothesis ruff pytest-randomly pytest-xdist pyyaml`).
- [ ] **Step 2:** Merge `~/.claude/skills/pytest-expert/assets/pytest-config.toml`
  into `pyproject.toml` (strict markers, branch coverage, floor 80), copy
  `assets/conftest.py` + `assets/conftest_ml.py` content into `tests/conftest.py`,
  adopt `assets/ruff-pytest.toml` as `ruff.toml`. Copy the five scripts listed above
  from the three skills into `scripts/`.
- [ ] **Step 3: Write the failing test** `tests/test_package.py::test_package_exposes_version`
  — `assert drbm25.__version__ == "0.1.0"`. Run `uv run pytest tests/test_package.py -q`;
  expected FAIL (`AttributeError`).
- [ ] **Step 4:** Add `__version__ = "0.1.0"`. Run again: PASS.
- [ ] **Step 5: Verify the stack resolves and torch is introspectable.**
  Run: `uv run python -c "import torch, transformers, peft, pyserini, wandb; print(torch.__version__, transformers.__version__, peft.__version__)"`
  Expected: `2.10.0 4.57.6 0.21.2`. Run `uv run python ~/.claude/skills/pytorch/scripts/check_env.py`
  and record its output in the commit body.
- [ ] **Step 6:** All gates (Global Constraints) exit 0. README gets a 10-line
  "Setup" section (`uv sync`, Java 21 check `java -version`).
- [ ] **Step 7: Commit** `chore: scaffold uv project with test and lint gates`.

---

### Task 2: Typed config and validation

**Files:**
- Create: `src/drbm25/config.py`, `configs/smoke_cpu.yaml`,
  `configs/expanse_v100_lora.yaml`, `configs/runpod_a100_full.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  ```python
  ScoringMode = Literal["normal", "count_once"]
  Precision = Literal["fp16", "bf16", "fp32"]
  TrainMode = Literal["lora", "full"]
  RolloutBackend = Literal["hf", "vllm"]
  class ConfigError(ValueError): ...
  @dataclass(frozen=True) ModelCfg(name: str, revision: str, precision: Precision,
      train_mode: TrainMode, gradient_checkpointing: bool)
  @dataclass(frozen=True) LoraCfg(rank: int, alpha: int, dropout: float, target_modules: tuple[str, ...])
  @dataclass(frozen=True) PPOCfg(batch_size: int, mini_batch_size: int, micro_batch_size: int,
      ppo_epochs: int, clip_ratio: float, value_clip: float, entropy_coef: float,
      kl_coef: float, gamma: float, lam: float, grad_clip: float, actor_lr: float, critic_lr: float)
  @dataclass(frozen=True) RolloutCfg(backend: RolloutBackend, temperature: float, top_p: float,
      max_prompt_tokens: int, max_new_tokens: int, generation_batch_size: int,
      vllm_gpu_memory_utilization: float)
  @dataclass(frozen=True) RetrievalCfg(index_dir: Path, k1: float, b: float, train_k: int,
      eval_k: int, search_threads: int, search_timeout_s: float)
  @dataclass(frozen=True) DataCfg(hf_dataset: str, hf_revision: str, data_dir: Path, val_size: int)
  @dataclass(frozen=True) RunCfg(output_dir: Path, total_steps: int, chunk_steps: int,
      val_every: int, ckpt_every: int, keep_ckpts: int, deep_log_every: int, seed: int,
      wandb_project: str, wandb_entity: str | None, wandb_group: str)
  @dataclass(frozen=True) Config(model, lora: LoraCfg | None, ppo, rollout, retrieval, data, run)
  def load_config(path: Path, overrides: Mapping[str, str] = MappingProxyType({})) -> Config
  def validate_for_runtime(cfg: Config, *, device_capability: tuple[int, int] | None, world_size: int) -> None
  ```

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`:
  - `test_loads_expanse_profile_values`: `cfg.model.name == "Qwen/Qwen2.5-1.5B-Instruct"`,
    `cfg.model.precision == "fp16"`, `cfg.model.train_mode == "lora"`, `cfg.lora.rank == 32`,
    `cfg.ppo.batch_size == 64`, `cfg.ppo.kl_coef == pytest.approx(0.001)`,
    `cfg.rollout.temperature == pytest.approx(0.6)`, `cfg.retrieval.k1 == pytest.approx(0.9)`,
    `cfg.retrieval.b == pytest.approx(0.4)`, `cfg.run.total_steps == 1751`.
  - `test_loads_runpod_profile_values`: 3B, `bf16`, `full`, `lora is None`,
    `actor_lr == approx(1e-6)`, `critic_lr == approx(1e-5)`, `rollout.backend == "vllm"`.
  - `test_override_changes_nested_field`: `load_config(p, {"run.total_steps": "20"}).run.total_steps == 20`.
  - `test_rejects_unknown_key` → `pytest.raises(ConfigError, match=r"unknown key 'ppo\.batchsize'")`.
  - `test_rejects_bf16_on_volta`: `validate_for_runtime(a100_cfg, device_capability=(7, 0), world_size=2)`
    raises `ConfigError` matching `"bf16 requires compute capability >= 8.0"`.
  - `test_rejects_batch_not_divisible`: batch 64, world 3 → `ConfigError` matching `"batch_size 64"`.
  - `test_rejects_lora_mode_without_lora_section` and `test_rejects_full_mode_with_lora_section`.
  - `test_rejects_reward_range_breaking_values`: `kl_coef < 0` → `ConfigError`.
- [ ] **Step 2:** Run `uv run pytest tests/test_config.py -q`; expected FAIL (ImportError).
- [ ] **Step 3: Implement** `config.py` and the three YAML files.
  Validation rules: `batch_size % world_size == 0`; `mini_batch_size % world_size == 0`;
  `(mini_batch_size // world_size) % micro_batch_size == 0`; `batch_size % mini_batch_size == 0`;
  bf16 needs capability ≥ (8, 0) when a capability is given; `train_mode == "lora"` ⇔
  `lora is not None`; positive LRs, `0 < clip_ratio < 1`, `kl_coef >= 0`. YAML via `yaml.safe_load`.
  Smoke config: model name `"tiny-local"` (tests and the CLI build the Task 9 tiny
  Qwen2 in-process when they see this name; nothing is downloaded), and sets `total_steps: 4`, `batch_size: 4`, `mini_batch_size: 2`,
  `micro_batch_size: 1`, `max_new_tokens: 16`, `precision: fp32`, `backend: hf`.
- [ ] **Step 4:** Tests PASS; gates green.
- [ ] **Step 5: Commit** `feat(config): typed config profiles with runtime validation`.

---

### Task 3: Reward parsing, nDCG, reward

**Files:** Create `src/drbm25/reward.py`; Test `tests/test_reward.py`

**Interfaces:**
- Produces:
  ```python
  class FormatErrorReason(StrEnum): THINK_REOPENED, TAG_COUNT, TAG_ORDER, BAD_JSON, NO_QUERY_KEY, QUERY_NOT_STRING
  @dataclass(frozen=True) ParsedRewrite(query: str)
  @dataclass(frozen=True) FormatError(reason: FormatErrorReason)
  REWARD_FORMAT_ERROR: Final = -2.0; FORMAT_BONUS: Final = 0.1
  def parse_response(response_text: str) -> ParsedRewrite | FormatError
  def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float
  def compute_reward(parsed: ParsedRewrite | FormatError, ndcg: float | None) -> float
  ```
  `response_text` is the generated text only; the prompt already opened `<think>`.
  Upstream equivalence (`msmarco.py:83-151`): the response must contain zero `<think>`,
  exactly one each of `</think>`, `<answer>`, `</answer>`, in that order; the answer body
  must be JSON with a string `query`.

- [ ] **Step 1: Write failing tests:**
  - `test_parses_valid_response`: `"reason\n</think>\n<answer>\n{\"query\": \"polymer synthesis\"}\n</answer>"`
    → `ParsedRewrite("polymer synthesis")`.
  - parametrized `test_rejects_malformed_response` rows (text → reason):
    second `<think>` → `THINK_REOPENED`; missing `</answer>` (truncated) → `TAG_COUNT`;
    two `<answer>` blocks → `TAG_COUNT`; `<answer>` before `</think>` → `TAG_ORDER`;
    `{query: x}` → `BAD_JSON`; `{"q": "x"}` → `NO_QUERY_KEY`; `{"query": ["x"]}` → `QUERY_NOT_STRING`.
  - `test_parse_never_raises` (Hypothesis `st.text()`): result is `ParsedRewrite | FormatError`.
  - parametrized `test_ndcg_hand_values`: `(["a"], {"a"}, 10) → 1.0`;
    `(["x","y","a"], {"a"}, 10) → 0.5`; `(["x"]*10 + ["a"], {"a"}, 10) → 0.0`;
    `(["a","x","b"], {"a","b"}, 10) → approx(0.919720, abs=1e-6)`; `([], {"a"}, 10) → 0.0`.
  - `test_ndcg_bounds` (Hypothesis): `0.0 <= ndcg <= 1.0`.
  - `test_ndcg_rejects_empty_relevant_set` → `pytest.raises(ValueError, match="relevant_ids is empty")`.
  - parametrized `test_reward_cases`: `(FormatError(BAD_JSON), None) → -2.0`;
    `(ParsedRewrite("q"), 0.0) → 0.0`; `(ParsedRewrite("q"), 0.5) → approx(0.6)`.
  - `test_reward_requires_ndcg_for_parsed` → `ValueError`.
- [ ] **Step 2:** Run; FAIL (ImportError).
- [ ] **Step 3: Implement.** Binary gains, DCG `Σ 1/log2(rank+1)`, IDCG over
  `min(|relevant|, k)` ideal positions. Assert `-2.0 <= reward <= 1.1` before returning.
- [ ] **Step 4:** PASS. Run `uv run python scripts/mutate_check.py src/drbm25/reward.py --function ndcg_at_k -- uv run pytest -x -q -p no:randomly --no-cov tests/test_reward.py`
  and the same for `parse_response` and `compute_reward`; kill or justify every survivor.
- [ ] **Step 5: Commit** `feat(reward): upstream-equivalent format check, nDCG@k, reward`.

---

### Task 4: BM25 index and searcher (normal | count_once)

**Files:** Create `src/drbm25/bm25/{__init__,index,searcher}.py`;
Test `tests/bm25/test_index.py`, `tests/bm25/test_searcher.py`, `tests/bm25/conftest.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True) IndexInfo(n_docs: int, checksum: str)   # sha256 over sorted index file names+sizes
  def tsv_to_jsonl(collection_tsv: Path, out_dir: Path) -> int     # {"id","contents"} lines, returns count
  def build_index(jsonl_dir: Path, index_dir: Path, *, threads: int) -> IndexInfo
  def read_index_info(index_dir: Path) -> IndexInfo
  class SearchError(RuntimeError); class TooManyClausesError(SearchError)
  @dataclass(frozen=True) SearchResult(doc_ids: tuple[str, ...], n_terms: int, n_unique_terms: int)
  class Bm25Searcher:
      def __init__(self, index_dir: Path, *, k1: float, b: float, threads: int, timeout_s: float) -> None
      def analyze(self, text: str) -> list[str]
      def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult
      def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]
  ```
- `normal`: `LuceneSearcher.search(query_string, k)` after `set_bm25(k1, b)`.
  `count_once`: analyze with the index's analyzer, keep first occurrence of each term,
  build `BooleanQuery` of `SHOULD` `TermQuery("contents", term)` clauses with
  pyserini's `querybuilder.get_boolean_query_builder()`, `JTermQuery`, `JTerm`,
  `JBooleanClauseOccur`; pass the built query object to `search`. Never use
  `querybuilder.get_term_query` (it re-analyzes and keeps only `[0]`). Empty term list →
  `SearchResult((), 0, 0)` without calling Lucene, in both modes. Lucene's
  `IndexSearcher.TooManyClauses` (surfaced as a `JavaException`) → `TooManyClausesError`.
  `batch_search` uses a `ThreadPoolExecutor(max_workers=threads)` with per-future
  `timeout_s`; a timeout raises `SearchError`.

- [ ] **Step 1: Java check.** `java -version` must print 21. If not on the dev Mac,
  install with `brew install openjdk@21` (ask the user first) and set `JAVA_HOME`.
- [ ] **Step 2: Verify the pyserini API** for 2.0.0 with `inspect.signature` on
  `LuceneSearcher.search`, `LuceneSearcher.set_bm25`, `pyserini.search.lucene.querybuilder.*`,
  `pyserini.analysis.Analyzer`, `pyserini.analysis.get_lucene_analyzer`, and
  `pyserini.index.lucene` CLI args. Paste the signatures in the task log; adjust names below if they differ.
- [ ] **Step 3: Write failing tests.** Fixture `toy_index` (session-scoped, `tmp_path_factory`):
  1,000 docs of exactly 50 tokens of filler words `w0..w199` (seeded), where "solar"
  appears in 100 docs and "efficiency" in 50, plus Doc A (`solar` ×3 + 47 filler) and
  Doc B (`solar` ×1, `efficiency` ×1 + 48 filler), mirroring slide 23. Mark tests `@pytest.mark.java`.
  - `test_index_counts_docs`: `read_index_info(idx).n_docs == 1002`.
  - `test_analyze_drops_boolean_operators_and_stems`:
    `analyze("solar AND (wind OR efficiency)") == ["solar", "wind", "effici"]`.
  - `test_normal_mode_lets_repetition_flip_ranking`: query `"solar solar solar efficiency"`,
    mode normal → `rank(A) < rank(B)`.
  - `test_count_once_ignores_repetition`: same query, count_once → `rank(B) < rank(A)`.
  - `test_modes_agree_without_repeats`: query `"solar efficiency"` → identical `doc_ids` top 10.
  - `test_empty_query_returns_no_hits` (parametrized over both modes): `"AND OR ( )"` →
    `SearchResult((), 0, 0)`.
  - `test_too_many_unique_terms_raises`: 1,100 distinct made-up words, count_once →
    `pytest.raises(TooManyClausesError)`; same for normal.
  - `test_batch_search_preserves_order`: results line up with input order for 3 queries.
  - `test_reports_term_counts`: `"solar solar efficiency"` → `n_terms == 3`, `n_unique_terms == 2`.
- [ ] **Step 4:** Run `uv run pytest tests/bm25 -q`; FAIL (ImportError).
- [ ] **Step 5: Implement** `index.py` (call `python -m pyserini.index.lucene --collection JsonCollection
  --generator DefaultLuceneDocumentGenerator --threads N --storePositions --storeDocvectors --storeRaw`
  via `subprocess.run([...], check=True)` with an argument list) and `searcher.py`.
- [ ] **Step 6:** PASS; mutation check on `searcher.py --function search`; gates green.
- [ ] **Step 7: Commit** `feat(bm25): Lucene index build and normal/count-once searcher`.

---

### Task 5: Prompt, data download, splits, data order

**Files:** Create `src/drbm25/prompt.py`, `src/drbm25/data.py`, `tests/golden/prompt_polymers.txt`;
Test `tests/test_prompt.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  ```python
  def build_prompt(query: str) -> str
  @dataclass(frozen=True) Example(qid: str, query: str, relevant_ids: frozenset[str])
  @dataclass(frozen=True) DataPaths(train_jsonl: Path, dev_jsonl: Path, collection_tsv: Path)
  def download_msmarco_science(cfg: DataCfg) -> DataPaths
  def load_examples(path: Path) -> list[Example]          # qid = f"{path.stem}-{line_no}"
  def split_dev(dev: Sequence[Example], *, val_size: int) -> tuple[list[Example], list[Example]]  # (val, test)
  def batch_indices(cursor: int, batch_size: int, n_examples: int, seed: int) -> list[int]
  ```
- HF files (dataset `DeepRetrieval/datasets`, pinned `hf_revision`):
  `raw_data/msmarco/msmarco_science/train.jsonl`, `.../dev.jsonl`, and the 800K
  collection file named in `raw_data/msmarco/readme.md` (verify the exact path with
  `huggingface_hub.list_repo_files` in Step 1 and record it in `DataCfg` defaults).
  Download with `hf_hub_download(..., revision=cfg.hf_revision, token=os.environ.get("HF_TOKEN"))`,
  3 attempts with exponential backoff on `requests`/`httpx` connection errors only.
- `batch_indices`: example order for epoch `e` is a permutation drawn from
  `numpy.random.default_rng([seed, e])`; global cursor `c` maps to epoch `c // n` and
  offset `c % n`; a batch may span the epoch boundary.

- [ ] **Step 1:** List repo files and pin `hf_revision` to the current commit sha.
- [ ] **Step 2: Golden prompt.** Copy lines 16–60 of
  `upstream/DeepRetrieval/code/data_preprocess/msmarco_sub.py` (instruction constant +
  prompt function only) into a scratch script in the session scratchpad, run it with
  `python -I` on `{"input": "how are polymers made"}` in sparse mode, and save stdout
  byte-for-byte as `tests/golden/prompt_polymers.txt`.
- [ ] **Step 3: Write failing tests:**
  - `test_prompt_matches_upstream_bytes`: `build_prompt("how are polymers made") == golden`.
  - `test_prompt_ends_inside_think`: endswith `"Let me rewrite the query with reasoning. \n<think>\n"`.
  - `test_load_examples_parses_fixture` on a 3-line fixture jsonl (`question`, `docs_id`).
  - `test_load_examples_rejects_row_without_docs` → `ValueError` naming the line number.
  - `test_split_dev_takes_first_101_as_val_and_all_as_test` (fixture of 120 rows).
  - `test_batch_indices_deterministic_and_identical_across_calls`.
  - `test_batch_indices_cover_epoch_exactly_once` (n=10, batch=5, cursors 0 and 5).
  - `test_batch_indices_span_epoch_boundary` (n=10, batch=4, cursor 8 → 2 from epoch 0, 2 from epoch 1).
  - `test_download_uses_pinned_revision` with `monkeypatch` replacing `hf_hub_download`
    (fake records kwargs; assert `revision == cfg.hf_revision`).
  - `@pytest.mark.integration test_real_msmarco_science_sizes`: train 22,416, dev 2,305.
- [ ] **Step 4:** FAIL → implement → PASS (unit); run the integration test once locally.
- [ ] **Step 5: Commit** `feat(data): verbatim prompt, MS-S download, splits, seeded data order`.

---

### Task 6: Real index + Experiment 1 original-query baseline (CPU)

This is the first real result and needs no GPU.

**Files:** Create `scripts/prepare_data.py`; Test `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: Task 4 `tsv_to_jsonl`, `build_index`; Task 5 `download_msmarco_science`.
- Produces: `prepare(cfg: Config, *, expected_docs: int = 800_000) -> IndexInfo` and
  CLI `uv run python scripts/prepare_data.py --config <yaml>` → downloads,
  converts, builds the index if `read_index_info` fails or `n_docs != 800000`, writes
  `<index_dir>/INDEX_INFO.json`; idempotent (second run does no work, prints `index ok`).

- [ ] **Step 1: Failing test** `test_prepare_is_idempotent` on the toy collection (tiny tsv
  fixture, `expected_docs` overridable): second call returns without calling `build_index`
  (spy via `monkeypatch`, `autospec=True`).
- [ ] **Step 2:** Implement; PASS.
- [ ] **Step 3: Real run (marked integration).** Build the 800K index locally and assert
  `n_docs == 800000`. Record wall time and index size in the commit body.
- [ ] **Step 4:** Commit `feat(data): idempotent data and index preparation CLI`.

(The original-query rows of Exp 1 are produced in Task 14 with this index.)

---

### Task 7: GAE, KL-penalised rewards, PPO losses

**Files:** Create `src/drbm25/ppo/{__init__,gae,losses}.py`; Test `tests/ppo/test_gae.py`, `tests/ppo/test_losses.py`

**Interfaces:**
- Produces (all tensors `[B, T]` over response tokens, `mask` float 0/1):
  ```python
  def masked_mean(x: Tensor, mask: Tensor) -> Tensor
  def masked_whiten(x: Tensor, mask: Tensor, *, eps: float = 1e-8) -> Tensor
  def kl_penalized_rewards(task_reward: Tensor, logprobs_old: Tensor, logprobs_ref: Tensor,
                           mask: Tensor, *, kl_coef: float) -> Tensor   # task reward on last valid token
  def gae(rewards: Tensor, values: Tensor, mask: Tensor, *, gamma: float, lam: float) -> tuple[Tensor, Tensor]  # (advantages, returns)
  def policy_loss(logprobs_new: Tensor, logprobs_old: Tensor, advantages: Tensor, mask: Tensor,
                  *, clip_ratio: float) -> tuple[Tensor, dict[str, Tensor]]   # keys: clipfrac, approx_kl
  def value_loss(values_new: Tensor, values_old: Tensor, returns: Tensor, mask: Tensor,
                 *, clip_range: float) -> tuple[Tensor, dict[str, Tensor]]    # keys: vf_clipfrac
  def entropy_from_logits(logits: Tensor, mask: Tensor) -> Tensor
  ```
  Conventions follow verl: KL estimator `old − ref`; value loss `0.5·max((v−R)², (v_clip−R)²)`;
  token-mean over all valid tokens in the micro-batch.

- [ ] **Step 1: Failing tests** (float64, exact values):
  - `test_kl_rewards_hand_values`: task 1.0, old `[-1,-1,-1]`, ref `[-1.5,-1,-0.5]`, β 0.1 →
    `[-0.05, 0.0, 1.05]`.
  - `test_kl_rewards_put_task_reward_on_last_valid_token`: mask `[1,1,0]` → reward on index 1, index 2 is 0.
  - `test_gae_gamma_lambda_one_hand_values`: rewards `[-0.05,0,1.05]`, values `[0.2,0.3,0.4]` →
    advantages `[0.8,0.75,0.65]`, returns `[1.0,1.05,1.05]`.
  - `test_gae_zero_beyond_mask`: masked positions have advantage 0 and do not leak value.
  - `test_whiten_has_zero_mean_unit_std_on_mask`.
  - `test_policy_loss_clips_positive_advantage`: logp_new−logp_old = 0.5, A = 1, ε 0.2 →
    loss `approx(-1.2)`, clipfrac 1.0.
  - `test_policy_loss_clips_negative_advantage`: ratio 0.5, A = −1 → loss `approx(0.8)`.
  - `test_policy_loss_unclipped_inside_trust_region`: ratio 1.1, A = 1 → `approx(-1.1)`, clipfrac 0.
  - `test_value_loss_hand_value`: v_old 0, v_new 1.0, R 0.2, clip 0.5 → `approx(0.32)`.
  - `test_entropy_uniform_logits_equals_log_vocab`: V = 8 → `approx(log(8))`.
  - `test_masked_mean_rejects_empty_mask` → `ValueError("mask has no valid tokens")`.
- [ ] **Step 2:** FAIL → implement → PASS. Mutation check each function; gates green.
- [ ] **Step 3: Commit** `feat(ppo): KL-in-reward, GAE, clipped policy and value losses`.

---

### Task 8: Logging, W&B wrapper, budget

**Files:** Copy `~/.claude/skills/observability-expert/assets/obslog/*` →
`src/drbm25/obs/`; Create `src/drbm25/budget.py`; Test `tests/obs/test_wandb_run.py`, `tests/test_budget.py`

**Interfaces:**
- Produces:
  ```python
  # obs (adapted from obslog): configure(level: str, json_path: Path | None) -> None; bind_context(**fields)
  def open_run(cfg: Config, *, run_name: str, reward_mode: ScoringMode, resume_id: str | None,
               extra_config: Mapping[str, object]) -> ContextManager[wandb.Run]
      # project/entity/group from cfg.run; job_type = reward_mode; define_metric("train/step"),
      # all other metrics step_metric="train/step"; mode "offline" when WANDB_MODE=offline;
      # resume="must" when resume_id given else "never"; config = asdict(cfg) + extra_config
      # (git sha, index checksum, dataset revision); asserts no key contains "token"/"key"/"secret".
  def deadline_from_budget(*, budget_usd: float, price_per_hour: float, start_unix: float) -> float
  def should_stop(*, now_unix: float, deadline_unix: float, mean_chunk_s: float, reserve_s: float) -> bool
  ```
- [ ] **Step 1: Failing tests:**
  - `test_open_run_offline_records_config_and_job_type` (`WANDB_MODE=offline`, `WANDB_DIR=tmp_path`
    via monkeypatch): `run.config["ppo"]["batch_size"] == 64`, `run.job_type == "normal"`.
  - `test_open_run_rejects_secret_like_config_key` → `ValueError` matching `"secret-like key"`.
  - `test_open_run_marks_failed_on_exception`: exception inside the block propagates.
  - `test_deadline_from_budget`: $100 at $2.78/h from t=0 → `approx(129496.4, abs=0.1)` s.
  - `test_should_stop_when_next_chunk_would_cross_reserve`: now 0, deadline 1000, chunk 600,
    reserve 500 → True; chunk 400 → False.
  - `test_budget_rejects_nonpositive_price` → `ValueError`.
- [ ] **Step 2:** FAIL → implement → PASS. Run `uv run python scripts/audit_observability.py src/` (exit 0)
  and the obslog's own tests copied into `tests/obs/`.
- [ ] **Step 3: Commit** `feat(obs): JSON logging, W&B run wrapper, budget guard`.

---

### Task 9: Policy models (protocol + LoRA layout)

**Files:** Create `src/drbm25/ppo/models.py`, `src/drbm25/distributed.py`; Test `tests/ppo/test_models_lora.py`, `tests/conftest.py` (tiny model fixture)

**Interfaces:**
- Produces:
  ```python
  def token_logprobs(logits: Tensor, input_ids: Tensor) -> Tensor   # [B, L-1]: log p(x_{t+1} | x_<=t), fp32
  class PolicyModels(Protocol):
      def actor_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor
      def ref_logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor        # no grad
      def critic_values(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor     # [B, L]
      def actor_parameters(self) -> list[nn.Parameter]
      def critic_parameters(self) -> list[nn.Parameter]
      def generation_module(self) -> nn.Module                                          # actor active
      def trainable_state_dict(self) -> dict[str, Tensor]
      def load_trainable_state_dict(self, state: Mapping[str, Tensor]) -> None
      def decoder_layers(self) -> Sequence[nn.Module]                                   # for probes
  class LoraPolicyModels:   # implements PolicyModels
      def __init__(self, base: PreTrainedModel, lora: LoraCfg, *, gradient_checkpointing: bool) -> None
  @dataclass(frozen=True) DistInfo(rank: int, world_size: int, local_rank: int, device: torch.device)
  def init_distributed() -> DistInfo          # env:// when WORLD_SIZE set, else single process
  def allreduce_mean_grads(params: Sequence[nn.Parameter], dist: DistInfo) -> None
  ```
- LoRA layout: one `PeftModel` with adapters `"actor"` and `"critic"` (LoRA on
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`), plus
  `value_head = nn.Linear(hidden, 1)` in fp32, zero-initialised weight and bias.
  `critic_values` runs the base with the critic adapter, takes the final hidden state
  (`output_hidden_states=True`, last entry), applies the head in fp32. `ref_logits`
  runs under `disable_adapter()` and `torch.no_grad()`. No DDP wrapper (adapter switching
  leaves the other adapter unused); gradients are averaged with `allreduce_mean_grads`
  after backward. Verify with `inspect` that peft 0.21.2 keeps adapter weights in fp32
  over an fp16 base (`autocast_adapter_dtype`), and that gradient checkpointing needs
  `enable_input_require_grads()`.
- Test fixture `tiny_qwen2` (in `tests/conftest.py`): `Qwen2Config(vocab_size=128,
  hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
  num_key_value_heads=2, tie_word_embeddings=True)`, seeded, fp32, CPU, `attn_implementation="sdpa"`.

- [ ] **Step 1: Failing tests:**
  - `test_token_logprobs_matches_manual_gather` (B=2, L=5, V=7).
  - `test_ref_logits_equal_base_logits_at_init_and_after_actor_update`: after one optimizer
    step on actor params, `ref_logits` unchanged (`torch.testing.assert_close`).
  - `test_actor_equals_ref_at_init` (LoRA B matrices are zero at init).
  - `test_actor_step_changes_only_actor_params`: base and critic params bit-identical, actor LoRA changed.
  - `test_critic_step_changes_only_critic_params_and_value_head`.
  - `test_critic_values_shape_and_zero_at_init`: `[B, L]`, all zeros.
  - `test_trainable_state_roundtrip`: save → new instance → load → identical logits and values.
  - `test_allreduce_single_process_is_identity`.
  - `test_lora_overfits_one_batch` (pytorch skill protocol): 100 SGD steps on a fixed
    token target drive actor CE from > 4.0 to < 0.5.
- [ ] **Step 2:** FAIL → implement → PASS. Run `lint_pytorch.py src/drbm25/ppo/`.
- [ ] **Step 3: Commit** `feat(models): PolicyModels protocol and LoRA actor/critic on shared base`.

---

### Task 10: Rollout engine (HF generate)

**Files:** Create `src/drbm25/rollout/{__init__,base,hf_generate}.py`; Test `tests/rollout/test_hf_generate.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True) RolloutBatch(prompt_ids: Tensor, prompt_mask: Tensor,      # [B, P] left-padded
                                       response_ids: Tensor, response_mask: Tensor,  # [B, R] right-padded; mask includes EOS
                                       response_text: tuple[str, ...], hit_max_tokens: Tensor)  # [B] bool
  class RolloutEngine(Protocol):
      def generate(self, prompts: Sequence[str], *, greedy: bool) -> RolloutBatch
      def sync_weights(self, models: PolicyModels) -> None
  class HFGenerateEngine:
      def __init__(self, models: PolicyModels, tokenizer: PreTrainedTokenizerBase, cfg: RolloutCfg, device: torch.device) -> None
  ```
  Left padding; prompts longer than `max_prompt_tokens` raise `ValueError` (MS-S prompts
  are ~200 tokens; a long one is a data bug); generation in chunks of
  `generation_batch_size`; `sync_weights` is a no-op (same module). Startup asserts
  `tokenizer.pad_token_id != tokenizer.eos_token_id` and `padding_side == "left"`.
  Decoding: `do_sample=not greedy`, `temperature`, `top_p`, `max_new_tokens`.

- [ ] **Step 1: Failing tests** (tiny model + a tiny word-level tokenizer fixture with
  distinct pad/eos; build with `tokenizers` `WordLevel`, wrap in `PreTrainedTokenizerFast`):
  - `test_shapes_and_left_padding` for prompts of different lengths.
  - `test_response_mask_stops_after_eos` (force EOS by setting its logit bias via a
    `LogitsProcessor` stub that emits EOS at step 3) → mask sum == 3 for that row.
  - `test_hit_max_tokens_flag` when no EOS is produced.
  - `test_greedy_is_deterministic` (two calls equal).
  - `test_rejects_overlong_prompt` → `ValueError` matching `"max_prompt_tokens"`.
  - `test_rejects_pad_equal_eos_tokenizer` → `ValueError` matching `"pad_token_id"`.
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3: Commit** `feat(rollout): batched HF generate engine with response masks`.

---

### Task 11: Checkpoints and resume state

**Files:** Create `src/drbm25/checkpoint.py`; Test `tests/test_checkpoint.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True) TrainState(step: int, data_cursor: int, wandb_run_id: str | None)
  @dataclass(frozen=True) LoadedCheckpoint(path: Path, train_state: TrainState, model: dict[str, Tensor],
                                           optimizers: dict[str, dict], scaler: dict | None, rng: dict[str, object])
  def save_checkpoint(run_dir: Path, *, train_state: TrainState, model: Mapping[str, Tensor],
                      optimizers: Mapping[str, dict], scaler: dict | None, keep: int) -> Path
  def latest_checkpoint(run_dir: Path) -> Path | None
  def load_checkpoint(path: Path) -> LoadedCheckpoint
  def capture_rng() -> dict[str, object]; def restore_rng(state: Mapping[str, object]) -> None
  ```
  Layout `run_dir/step_000050/{model.safetensors, optim.pt, meta.json}`; write into
  `run_dir/.tmp-step_000050-<pid>/`, fsync files, `os.replace` to final name, then write
  `run_dir/latest` (text file with the dir name) via temp + `os.replace`. Prune to `keep`
  newest complete checkpoints. `torch.load(..., weights_only=True)`; RNG state stored in
  `meta.json`/tensors so `weights_only=True` works (python `random` state as JSON list,
  numpy as dict, torch CPU/CUDA states as uint8 tensors).

- [ ] **Step 1: Failing tests:**
  - `test_roundtrip_restores_everything` (model tensors, optimizer state of a real AdamW, TrainState).
  - `test_latest_points_to_newest`.
  - `test_ignores_incomplete_tmp_dir` (Review Focus 3): create `.tmp-step_000100-1/` with
    partial files → `latest_checkpoint` returns step 50.
  - `test_prunes_to_keep`: save steps 50,100,150 with keep=2 → only 100,150 remain.
  - `test_restore_rng_reproduces_next_draws` (torch, numpy, random).
  - `test_load_rejects_missing_meta` → `FileNotFoundError` naming the path.
- [ ] **Step 2:** FAIL → implement → PASS; mutation check `latest_checkpoint`.
- [ ] **Step 3: Commit** `feat(checkpoint): atomic checkpoints with RNG and data cursor`.

---

### Task 12: Model and training observability metrics

**Files:** Create `src/drbm25/obs/metrics.py`; Test `tests/obs/test_metrics.py`

**Interfaces:**
- Produces:
  ```python
  def grad_norm(params: Sequence[nn.Parameter]) -> float                       # global L2, before clipping
  def per_layer_grad_and_weight_norms(layers: Sequence[nn.Module], prefix: str) -> dict[str, float]
      # keys f"{prefix}/layer_{i:02d}/{attn|mlp}/{grad_norm|weight_norm}"
  class ActivationProbe:   # context manager; forward hooks on each decoder layer output
      def __init__(self, layers: Sequence[nn.Module], prefix: str) -> None
      def stats(self) -> dict[str, float]   # f"{prefix}/layer_{i:02d}/{mean|std|absmax|nonfinite_frac}"
  def snapshot(params: Sequence[nn.Parameter]) -> list[Tensor]
  def update_ratio(before: Sequence[Tensor], params: Sequence[nn.Parameter]) -> float   # ||Δθ|| / ||θ_before||
  def qk_logit_bound(layer: nn.Module, hidden: Tensor) -> float
      # max_{i,j} ||q_i||·||k_j|| / sqrt(head_dim) from q_proj/k_proj outputs; an upper bound
      # on |attention logit| that RoPE (a rotation) cannot exceed. Flag > 1e4.
  def explained_variance(values: Tensor, returns: Tensor, mask: Tensor) -> float
  ```
- [ ] **Step 1: Failing tests:**
  - `test_grad_norm_hand_value` (two params with grads `[3]`, `[4]` → 5.0).
  - `test_per_layer_keys_and_values` on tiny model after a backward.
  - `test_activation_probe_removes_hooks_on_exit` (`len(layer._forward_hooks) == 0` after).
  - `test_activation_probe_counts_nonfinite` (inject inf in a hook input).
  - `test_update_ratio_hand_value` (θ `[3,4]` → `[3,4.5]`: 0.5/5 = 0.1).
  - `test_qk_bound_is_upper_bound_of_true_logits` on tiny model: compute true max |q·k|/√d
    with RoPE applied via the model's own rotary module; assert `bound >= true`.
  - `test_explained_variance_perfect_is_one_and_constant_is_zero`.
- [ ] **Step 2:** FAIL → implement → PASS; `audit_observability.py` exit 0.
- [ ] **Step 3: Commit** `feat(obs): per-layer grads, activation probe, update ratio, QK bound`.

---

### Task 13: PPO trainer step and train CLI (single node, CPU-testable)

**Files:** Create `src/drbm25/ppo/trainer.py`, `src/drbm25/train.py`; Test `tests/ppo/test_trainer.py`, `tests/test_train_cli.py`

**Interfaces:**
- Consumes: Tasks 2–12.
- Produces:
  ```python
  class NonFiniteError(RuntimeError); class TrainingHealthError(RuntimeError)
  class PPOTrainer:
      def __init__(self, cfg: Config, *, models: PolicyModels, engine: RolloutEngine,
                   searcher: Bm25Searcher, reward_mode: ScoringMode, dist: DistInfo,
                   scaler: torch.amp.GradScaler | None) -> None
      def step(self, examples: Sequence[Example], *, step: int) -> dict[str, float]
      def validate(self, examples: Sequence[Example]) -> dict[str, float]   # greedy; nDCG@10 both modes
      def optimizer_states(self) -> dict[str, dict]; def load_optimizer_states(self, s: Mapping[str, dict]) -> None
  # CLI
  python -m drbm25.train --config PATH --reward-mode {normal,count_once} --run-name NAME
      [--until-step N] [--deadline-unix T] [--set key=value ...]
  # exit codes: 0 reached until-step/total_steps; 3 stopped by budget deadline; 1 error
  ```
- `step` order: generate (actor) → parse → `batch_search` in `reward_mode` (reward) and
  in the other mode (metric only) → rewards → concat prompt+response ids → old logprobs
  (actor, no grad), ref logprobs, values → KL rewards → GAE → whiten advantages →
  for each mini-batch: micro-batches with grad accumulation (actor loss
  `pg − entropy_coef·entropy`; critic `value_loss`), `allreduce_mean_grads`, scaler unscale,
  clip 1.0, step → `engine.sync_weights(models)`.
- Guards (raise immediately): non-finite loss / logprobs / values / advantages →
  `NonFiniteError` naming the tensor and step; reward outside [−2, 1.1] → `AssertionError`;
  GradScaler scale < 1.0 for 5 consecutive steps or approx_kl > 1.0 for 5 consecutive
  steps → `TrainingHealthError`.
- Metrics keys (W&B names, every step): `reward/{mean,std,min,max}`,
  `reward/format_error_rate`, `reward/format_error/{reason}`, `reward/zero_ndcg_rate`,
  `ndcg/train_{normal,count_once}`, `rewrite/{n_terms,n_unique_terms,repeat_ratio,any_repeat_rate}`,
  `search/{empty_query_rate,too_many_clauses_rate,seconds}`,
  `ppo/{policy_loss,value_loss,entropy,kl_ref_mean,kl_ref_max,approx_kl,clipfrac,vf_clipfrac,explained_var,adv_mean_raw,adv_std_raw,returns_mean,values_mean}`,
  `optim/{actor_lr,critic_lr,actor_grad_norm,critic_grad_norm,actor_clipped,critic_clipped,actor_update_ratio,critic_update_ratio,scaler_scale,scaler_skipped}`,
  `rollout/{response_len_mean,hit_max_tokens_rate,tokens_per_s}`,
  `time/{generate,logprobs,values,search,update,sync}`, `sys/{gpu_mem_peak_gb,gpu_mem_reserved_gb,host_rss_gb}`.
  Every `deep_log_every` steps add Task 12's per-layer, activation and `stab/qk_logit_bound_max`,
  `stab/final_norm_out` keys; every `val_every` steps `val/ndcg_{normal,count_once}` and a
  `samples` table (8 rows).
- Startup checks (fail before any training): on CUDA, `f"sm_{major}{minor}"` is in
  `torch.cuda.get_arch_list()`; `java -version` reports 21; `read_index_info(index_dir).n_docs`
  equals 800,000 (smoke config: the toy count); tokenizer pad ≠ EOS.
- Every step's metrics are also appended to `output_dir/run_name/metrics.jsonl` (rank 0),
  which `scripts/spike_monitor.py --window 200` reads after each chunk/job.
- `run.alert(...)` on `NonFiniteError`, `TrainingHealthError`, budget stop, and completion.
  On LoRA runs, the final adapters are logged as a W&B artifact (`type="model"`).
- CLI loop: `init_distributed` → load config → `validate_for_runtime` → build models,
  engine, searcher → resume from `latest_checkpoint(output_dir/run_name)` (restores model,
  optimizers, scaler, RNG, cursor, W&B id) → `open_run` → loop steps until
  `min(until_step, total_steps)`; checkpoint every `ckpt_every` and at exit; time the
  first 10 steps and log `budget/projected_steps` when `--deadline-unix` is given;
  `should_stop` checked at checkpoint boundaries → exit 3.

- [ ] **Step 1: Failing trainer tests** (tiny model, toy index from Task 4, smoke config):
  - `test_step_returns_finite_metrics_with_expected_keys` (subset check on the key list above).
  - `test_step_changes_actor_and_critic_not_base`.
  - `test_reward_uses_requested_mode_and_logs_other`: monkeypatch searcher so the two
    modes return different docs; reward matches `reward_mode`, `ndcg/train_*` both logged.
  - `test_nan_logprob_raises_nonfinite_error` (inject NaN via a model wrapper) →
    `pytest.raises(NonFiniteError, match="logprobs_old at step 0")`.
  - `test_ppo_learns_toy_reward`: fake searcher/reward that gives 1.0 when the response
    contains token id 7, else 0; 30 steps at lr 1e-2 raise mean reward from < 0.2 to > 0.6.
- [ ] **Step 2: Failing CLI tests:**
  - `test_cli_runs_two_steps_and_writes_checkpoint` (subprocess, smoke config, `WANDB_MODE=offline`).
  - `test_cli_resume_continues_from_checkpoint_with_same_next_batch`: run to step 2, then
    to step 4; batch indices logged at step 3 equal those of an uninterrupted 4-step run.
  - `test_cli_two_process_gloo` (marked `slow`): `torchrun --nproc_per_node 2` on CPU,
    2 steps; both ranks finish, rank 0 alone writes the checkpoint.
  - `test_cli_exits_3_when_deadline_passed` (`--deadline-unix` = now).
  - `test_cli_fails_fast_on_wrong_index_size` → exit 1, log names expected and actual counts.
  - `test_cli_writes_metrics_jsonl_readable_by_spike_monitor` (run `scripts/spike_monitor.py`
    on it; exit 0).
- [ ] **Step 3:** FAIL → implement → PASS; `lint_pytorch.py`, `audit_observability.py`,
  pytorch `diagnose.py` overfit on the tiny actor; mutation check `PPOTrainer.step` guards.
- [ ] **Step 4: Commit** `feat(ppo): PPO trainer step and resumable train CLI`.

---

### Task 14: Evaluation, statistics, report (Exp 1 and Exp 2)

**Files:** Create `src/drbm25/eval/{__init__,run_eval,stats,report}.py`; Test `tests/eval/*.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True) QueryRecord(qid: str, source: str, query_text: str, rewrite_text: str | None,
      format_error: str | None, terms: tuple[str, ...], ndcg_normal: float, ndcg_count_once: float)
  def score_original(examples: Sequence[Example], searcher: Bm25Searcher, *, k: int) -> list[QueryRecord]
  def score_rewrites(examples: Sequence[Example], engine: RolloutEngine, searcher: Bm25Searcher, *, k: int,
                     source: str) -> list[QueryRecord]     # greedy; format error → nDCG 0 in both modes
  @dataclass(frozen=True) PairedCI(mean_diff: float, low: float, high: float)
  def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, *, n_resamples: int = 10_000, seed: int = 0,
                          alpha: float = 0.05) -> PairedCI
  def query_stats(records: Sequence[QueryRecord]) -> dict[str, float]
      # mean n_terms, mean unique, repeat_ratio, any_repeat_rate, format_error_rate
  # CLIs
  python -m drbm25.eval.run_eval --config P --source {original,untrained,checkpoint} [--checkpoint DIR] --out FILE.jsonl
  python -m drbm25.eval.report --inputs A.jsonl B.jsonl ... --out report.md [--wandb]
  ```
  `report` builds the 2×2 + baseline table (rows = source, columns = nDCG@10 normal /
  count-once), the CIs for: trained-normal vs trained-count_once under each scorer; each
  trained model's normal − count_once gap; untrained vs original. It refuses inputs whose
  qid sets differ (`ValueError`). For checkpoints it evaluates the largest step present
  in **both** runs (`common_step(run_dir_a, run_dir_b) -> int`).

- [ ] **Step 1: Failing tests:**
  - `test_format_error_scores_zero_in_both_modes`.
  - `test_score_original_uses_query_text` (toy index; relevant doc ranked 1 → nDCG 1.0).
  - `test_bootstrap_identical_arrays_gives_zero_interval`.
  - `test_bootstrap_detects_constant_shift` (b = a + 0.1 → CI ≈ [0.1, 0.1]).
  - `test_bootstrap_seeded_reproducible`.
  - `test_query_stats_hand_values` (records with terms `("a","a","b")`, `("c",)` →
    repeat_ratio approx(1/4), any_repeat_rate 0.5).
  - `test_report_rejects_mismatched_qids` → `ValueError` matching `"qid sets differ"`.
  - `test_common_step_picks_max_shared` (dirs with steps {50,100,150} and {50,100} → 100).
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3: Real Exp 1 baseline (CPU).** Run `run_eval --source original` on the 2,305
  dev queries with the real index; commit the jsonl summary numbers (not the jsonl) in
  `docs/results.md`.
- [ ] **Step 4: Commit** `feat(eval): Exp 1/2 scoring, paired bootstrap, report`.

---

### Task 15: A100 layout — FSDP2 full fine-tune + vLLM rollout

**Files:** Create `src/drbm25/ppo/models_full.py`, `src/drbm25/rollout/vllm_engine.py`;
Modify `src/drbm25/train.py` (choose layout by `cfg.model.train_mode`, engine by `cfg.rollout.backend`),
`src/drbm25/checkpoint.py` (sharded save via `torch.distributed.checkpoint` when `train_mode == "full"`);
Test `tests/ppo/test_models_full.py`, `tests/rollout/test_vllm_engine.py` (gpu)

**Interfaces:**
- Produces `FullPolicyModels` (implements `PolicyModels`): actor `AutoModelForCausalLM`
  loaded fp32; critic = `AutoModel` backbone + fp32 `value_head`; both wrapped per
  decoder layer then root with FSDP2 `fully_shard(..., mp_policy=MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32))`;
  ref model bf16, `fully_shard` without optimizer (or CPU-offloaded via `CPUOffloadPolicy`
  if the memory test fails). `actor_full_state_dict_bf16() -> dict[str, Tensor]` gathers on
  all ranks for vLLM sync.
- `VLLMEngine(model_name: str, revision: str, cfg: RolloutCfg, dist: DistInfo)`: one
  `vllm.LLM` per rank, `tensor_parallel_size=1`, `dtype` from config (`half` on V100,
  `bfloat16` on A100), `gpu_memory_utilization=cfg.vllm_gpu_memory_utilization`,
  `enable_sleep_mode=True` if available in 0.19 (verify), prompts sharded by rank.
  `sync_weights` loads the gathered actor weights into the engine's model
  (verify the 0.19 path with `inspect`: `llm.llm_engine.model_executor` /
  `collective_rpc("load_weights")`; TRL's colocate mode is the reference pattern).
  Returns the same `RolloutBatch` as Task 10 (re-tokenized `response_ids` from vLLM token ids).
- [ ] **Step 1: Verify APIs** with `inspect` on torch 2.10 (`torch.distributed.fsdp.fully_shard`,
  `MixedPrecisionPolicy`, `CPUOffloadPolicy`, `torch.distributed.checkpoint.state_dict.get_model_state_dict`)
  and vllm 0.19 (`LLM.__init__`, `SamplingParams`, weight-loading entry point). Record findings.
- [ ] **Step 2: Failing CPU tests** (gloo, 2 processes via `torch.multiprocessing.spawn`, tiny model, fp32):
  - `test_full_models_match_lora_protocol` (runtime `isinstance` against a
    `runtime_checkable` `PolicyModels`).
  - `test_fsdp_actor_step_matches_single_process_step` (same seed, same batch →
    `assert_close` on gathered params, rtol 1e-5).
  - `test_sharded_checkpoint_roundtrip`.
- [ ] **Step 3: GPU tests** (`@pytest.mark.gpu`, run on the A100 pod in Task 17's smoke stage):
  `test_vllm_generate_matches_hf_greedy_on_3_prompts` (token-identical greedy output for
  the first 20 tokens) and `test_vllm_sync_weights_changes_output_after_update`.
- [ ] **Step 4:** FAIL → implement → PASS (CPU now; GPU tests on the pod).
- [ ] **Step 5: Commit** `feat(a100): FSDP2 full fine-tune layout and colocated vLLM engine`.

---

### Task 16: Expanse scripts (V100, primary)

**Files:** Create `scripts/expanse/{common.sh,probe.sbatch,prepare_login.sh,train.sbatch,submit_chain.sh,eval.sbatch}`,
`tests/shell/test_expanse_common.bats`; Modify `README.md` (Expanse section)

**Interfaces:**
- `common.sh`: `require_env VAR...` (dies naming the missing var), `drbm25_paths`
  (exports `DRBM25_ROOT=/expanse/lustre/projects/$EXPANSE_ACCOUNT/$USER/drbm25`, HF cache,
  index, runs dirs), `offline_env` (`HF_HUB_OFFLINE=1`, `WANDB_MODE=offline` unless
  `DRBM25_ONLINE=1`).
- `probe.sbatch` (`gpu-debug`, 2 GPUs, 30 min, `--constraint="lustre"`): prints
  `nvidia-smi`, `curl -sS --max-time 10 -o /dev/null -w '%{http_code}' https://huggingface.co`,
  `torch.cuda.get_arch_list()` and capability, `java -version`, `module avail java cuda 2>&1`,
  then a 3-prompt fp16 generation with HF and (if `uv sync --extra vllm` succeeded)
  with vLLM 0.19; writes `probe_<jobid>.json`.
- `prepare_login.sh`: `uv sync --frozen`, JDK 21 into `$DRBM25_ROOT/jdk` if `java -version`
  isn't 21 (Temurin tarball, sha256-verified), `prepare_data.py`, model snapshot download.
- `train.sbatch REWARD_MODE`: `gpu-shared`, `--gpus=2`, `--cpus-per-task=10`, `--mem=180G`,
  `-t 47:30:00`, `--constraint="lustre"`, `--account=$EXPANSE_ACCOUNT`; runs
  `torchrun --nproc_per_node 2 -m drbm25.train --config configs/expanse_v100_lora.yaml
  --reward-mode $1 --run-name $1`; exit 0 means done.
- `submit_chain.sh REWARD_MODE N`: submits N `train.sbatch` jobs, each `--dependency=afterany:<prev>`
  (each resumes from `latest`); prints job ids. Both reward modes are submitted as
  two independent chains so they run concurrently.
- [ ] **Step 1: Failing bats tests** for `common.sh`: `require_env` fails with the var
  name; `drbm25_paths` exports expected paths given `EXPANSE_ACCOUNT=abc123`, `USER=u`;
  `offline_env` sets both vars and respects `DRBM25_ONLINE=1`.
- [ ] **Step 2:** FAIL → implement → PASS; `shellcheck` clean on all six files
  (bash-scripting-specialist conventions: `set -euo pipefail`, quoted expansions).
- [ ] **Step 3:** README Expanse runbook: login, `prepare_login.sh`, `sbatch probe.sbatch`,
  pilot, `submit_chain.sh normal 3` + `submit_chain.sh count_once 3`, `wandb sync`, eval.
- [ ] **Step 4: Commit** `feat(expanse): probe, prepare, chained train and eval jobs`.

---

### Task 17: RunPod one-shot launcher (A100 fallback)

**Files:** Create `scripts/runpod/lib.sh`, `scripts/runpod/run_all.sh`, `tests/shell/test_runpod_lib.bats`;
Modify `README.md` (RunPod section)

**Interfaces (`lib.sh` functions, each unit-tested):**
- `prompt_var NAME "Prompt" [--secret]`: if `$NAME` is set (env or `.env`), keep it; else
  read from tty (`read -rs` for secrets).
- `write_env_file PATH NAME...`: writes `NAME=value` lines with `umask 077` (mode 600);
  never echoes values.
- `load_env_file PATH`: sources only `^[A-Z_]+=` lines.
- `step_done MARKER` / `mark_done MARKER`: marker files in `/workspace/drbm25/.done/`.
- `terminate_pod`: `curl --fail --max-time 30 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY"
  "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID"` (RunPod docs, "Terminate a pod":
  `DELETE /v2/pods/{id}`, 204 on success, key needs `pods:write`); 3 attempts with 5/15/45 s backoff; on final failure print the pod id and
  `https://console.runpod.io/pods` and return 1 (Review Focus 5).
- `run_alternating_chunks UNTIL CHUNK`: for s in CHUNK, 2·CHUNK, …, UNTIL: train normal
  `--until-step s`, then count_once `--until-step s`, then `spike_monitor.py` on both
  `metrics.jsonl` files (warnings logged, not fatal); stop the loop on exit code 3 (budget).
- `run_all.sh`: `trap on_exit EXIT` where `on_exit` copies logs to `/workspace/drbm25/logs/`
  and calls `terminate_pod` unless `--no-terminate` or `TERMINATE_ON_FINISH=no`.
  Sequence: prompts (HF_TOKEN, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY optional,
  RUNPOD_API_KEY, BUDGET_USD, PRICE_PER_HOUR, TERMINATE_ON_FINISH default yes) →
  `.env` → apt Java 21 + uv → `uv sync --frozen --extra vllm` → `prepare_data.py` →
  model snapshot → GPU smoke (`pytest -m gpu tests/rollout/test_vllm_engine.py` + 1 PPO step) →
  deadline = `deadline_from_budget(...)` → `run_alternating_chunks 1751 50` →
  `eval` for original, untrained, both checkpoints at `common_step` → `report --wandb`.
  Each stage guarded by `step_done`.
- [ ] **Step 1: Failing bats tests** (stub `curl`, `uv`, `python` with PATH shims that log calls):
  - `.env` written with mode 600 and contains no value echoed to stdout;
  - re-run with an existing `.env` prompts only for missing vars;
  - `terminate_pod` retries 3 times then returns 1 and prints the pod id;
  - `terminate_pod` succeeds on the 2nd attempt → returns 0;
  - `run_alternating_chunks 100 50` calls train in order normal@50, count_once@50,
    normal@100, count_once@100;
  - a budget exit (stubbed exit 3) stops the loop and still runs eval;
  - `on_exit` terminates after a failure, and does not with `--no-terminate`.
- [ ] **Step 2:** FAIL → implement → PASS; `shellcheck` clean.
- [ ] **Step 3:** README RunPod runbook: create a network volume (≥ 100 GB) in a data
  center with A100 80GB stock, deploy a 2× A100 pod with it mounted at `/workspace`,
  `git clone`, `bash scripts/runpod/run_all.sh`. State the live prices from the spec
  and that terminate releases billing while the volume keeps the data.
- [ ] **Step 4: Commit** `feat(runpod): one-shot launcher with resume, budget stop and self-terminate`.

---

### Task 18: End-to-end smoke and final review gate

**Files:** Create `tests/test_e2e_smoke.py`; Modify `README.md` (results + reproduce section)

- [ ] **Step 1: Failing test** `test_e2e_tiny_pipeline` (marked `slow`, CPU): toy index →
  `train` 4 steps for each reward mode with alternating `--until-step 2,4` → `run_eval` for
  original, untrained, both checkpoints → `report` produces a table with 4 rows and the
  CI columns; asserts row names and that every nDCG is in [0, 1].
- [ ] **Step 2:** Make it pass (fix integration seams only; no new features).
- [ ] **Step 3:** Run every gate plus `uv run python scripts/diff_guard.py --base docs/design-spec`,
  `uv run pytest -p randomly -n auto`; all exit 0.
- [ ] **Step 4: Commit** `test: end-to-end tiny pipeline`.

---

## Runbook after the code (operational, not TDD)

1. **Expanse probe** (as soon as SSH works): `sbatch scripts/expanse/probe.sbatch`.
   Decide from its JSON: online vs offline W&B; HF vs vLLM rollout on V100; cu128 vs cu126
   torch if the driver is too old for CUDA 12.8 (add a `[tool.uv.sources]` index switch then).
2. **V100 GPU tests**: `pytest -m gpu` in a `gpu-debug` job: memory for batch 64 fits,
   one real PPO step, resume reproduces the next batch.
3. **LoRA pilot**: ≤ 150 steps × both reward modes × candidates
   {(2e-5, 1e-4), (5e-5, 2e-4)} for (actor, critic) lr; choose by mean validation-nDCG gain
   across the two modes; freeze in `configs/expanse_v100_lora.yaml`; commit.
4. **Full runs**: `submit_chain.sh normal 3` and `submit_chain.sh count_once 3`.
5. **Eval + report**: `eval.sbatch` for original, untrained, both checkpoints at the common
   step; `report`; results into `docs/results.md`.
6. **Fallback**: if Expanse isn't usable by ~2026-11-29, run `run_all.sh` on RunPod.
