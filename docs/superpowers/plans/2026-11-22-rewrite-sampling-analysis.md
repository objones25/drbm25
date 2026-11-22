# Rewrite Sampling Database + Behavior Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample every dev query 32 times from the untrained, normal-reward and count-once-reward policies, store every response (reasoning + rewrite + retrieval + scores) in a Parquet database uploaded to a private Hugging Face dataset repo, compute behavior metrics, and publish a W&B report comparing the policies.

**Architecture:** A vLLM sampler (`drbm25.eval.sample`) writes resumable Parquet shards per source; pure metric modules under `drbm25.analysis` turn the shards into per-sample and per-query metric tables plus query-level statistical comparisons; `drbm25.analysis.report` logs tables, figures and a programmatic W&B Report; `drbm25.analysis.hub` uploads and verifies the database. A pod script (`scripts/runpod/run_sampling.sh`) runs the whole thing on one GPU in EU-RO-1 against the existing volume, started through `pod_bootstrap.sh` with a new `DRBM25_ENTRY`.

**Tech Stack:** vLLM 0.19 (existing), pyserini/Lucene (existing), pyarrow, scipy, scikit-learn, matplotlib, sentence-transformers, wandb + wandb-workspaces, huggingface_hub (existing). New packages go in a new optional extra `analysis`, exact versions pinned, `uv.lock` updated.

**Spec:** the design approved in conversation on 2026-11-22 (summarized in Global Constraints below; research findings from three agents: IR rewrite metrics, emergent-behavior metrics, repeated-sampling statistics).

## Global Constraints

- Sources: `original` (raw query, 1 row per query, no model), `untrained` (`Qwen/Qwen2.5-3B-Instruct` at the config's pinned revision), `trained-normal` (`objones25/drbm25-qwen2.5-3b-normal`), `trained-count_once` (`objones25/drbm25-qwen2.5-3b-count_once`). Hub revisions are resolved to a commit sha at run time and recorded.
- N = 32 samples per query per model source; a pilot with N = 8 and `--limit` runs first on the pod.
- Sampling = training rollout settings: `sampling_kwargs(cfg.rollout, greedy=False, eos_id)` (temperature 0.6, top_p 1.0, top_k off, max_new_tokens 350) with `n=N`, `logprobs=0`, and a per-query seed `seed = base_seed * 1_000_003 + query_index` shared by all sources (common random numbers). Prompts are `build_prompt(query)` tokenized with the training tokenizer and sent as token ids, as `VLLMEngine.generate` does.
- Parsing and scoring reuse `drbm25.reward.parse_response` and `drbm25.ppo.trainer.score_responses` unchanged; retrieval k = `cfg.retrieval.eval_k` (10).
- Unit of statistical analysis is the query, never the sample. Per-query means first, then paired query-level bootstrap (scipy `bootstrap(paired=True, method="BCa")`, 10,000 resamples, seed 0), Wilcoxon signed-rank, Benjamini-Yekutieli over the secondary-metric family; nDCG@10 is the primary comparison outside the family.
- Database: Parquet, one row per sample, written as shards `db/<source>/shard-<k>.parquet`; private HF dataset repo `objones25/drbm25-rewrites`, uploaded files verified by downloading back and comparing sha256.
- W&B project `drbm25`, entity from the API key default (WANDB_ENTITY may be unset). Per-sample table above 200,000 rows is not logged as a Table: the full database goes in as an Artifact; Tables hold per-query aggregates and a browsable sample of rows.
- Pod: one GPU, EU-RO-1 (volume `x1entqgdav` holds index, data, venv, JDK), RTX PRO 4500 32 GB or L4 24 GB, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; the pod terminates itself on exit (lib.sh exit traps) and has a watchdog hard deadline.
- Never print or log secrets; HF_TOKEN / WANDB_API_KEY come from RunPod secrets.

## Review Focus

1. A response with a format error (no `</think>`, bad JSON) must still be a row: `rewrite=None`, `format_error` set, nDCG 0.0, `think_text` the text before `</think>` if exactly one exists else None. Test in Task 3.
2. A rewrite that analyzes to zero terms (e.g. `"the of"`) must not divide by zero in retention/Jaccard/IDF/Vendi. Set metrics use one rule everywhere: two empty sets -> 1.0, exactly one empty -> 0.0; IDF summaries of an empty term list are NaN and counted. Tests in Tasks 5 and 6.
3. A sampler crash mid-source must resume without duplicating rows: existing complete shards are skipped, a partial shard (no `_SUCCESS` sidecar) is rewritten. Test in Task 4.
4. Queries over Lucene's clause limit (`SearchResult=None` from `score_responses`) keep a row with empty doc lists and `search_error` set. Test in Task 3.
5. A policy whose 32 samples are all identical must give distinct fraction 1/32, Vendi 1.0, nDCG std 0.0, best-of-k = mean. Test in Task 6.

---

### Task 1: Dependencies and the `think_text` helper

**Files:**
- Modify: `pyproject.toml` (new extra `analysis`), `uv.lock`
- Modify: `src/drbm25/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Produces: `think_text(response_text: str) -> str | None` in `drbm25.reward`: the stripped text before the single `</think>`, or None when the response does not contain exactly one `</think>`.

- [ ] **Step 1:** Tests `test_think_text_returns_stripped_reasoning` (`"  plan here \n</think>\n<answer>{...}</answer>"` -> `"plan here"`), `test_think_text_none_without_close_tag`, `test_think_text_none_with_two_close_tags`.
- [ ] **Step 2:** Run `uv run pytest tests/test_reward.py -k think_text -q`; expect NameError/ImportError on `think_text`.
- [ ] **Step 3:** Implement `think_text`; refactor `think_length_chars` to use it (behavior unchanged: still raises on != 1 tag).
- [ ] **Step 4:** Add extra `analysis = [pyarrow==X, scipy==X, scikit-learn==X, matplotlib==X, sentence-transformers==X, wandb-workspaces==X]` with exact versions resolved by `uv add --optional analysis ...` then pinned to `==`; check each package's PyPI page links to its official repo. `uv sync --extra analysis`, `uv run pytest -q` (JAVA_HOME=JDK 21) green.
- [ ] **Step 5:** Commit `feat(reward): think_text helper; add analysis extra`.

### Task 2: Sample row schema and Parquet shard IO

**Files:**
- Create: `src/drbm25/eval/sample_db.py`
- Test: `tests/eval/test_sample_db.py`

**Interfaces:**
- Produces:
  - `SampleRow` frozen dataclass: `source: str, qid: str, query_index: int, sample_index: int, query_text: str, response_text: str | None, think_text: str | None, rewrite: str | None, format_error: str | None, terms: tuple[str, ...], n_response_tokens: int | None, hit_max_tokens: bool | None, sum_logprob: float | None, doc_ids_normal: tuple[str, ...], scores_normal: tuple[float, ...], doc_ids_count_once: tuple[str, ...], scores_count_once: tuple[float, ...], ndcg_normal: float, ndcg_count_once: float, search_error: str | None, model_id: str | None, model_revision: str | None, seed: int | None`.
  - `write_shard(path: Path, rows: Sequence[SampleRow]) -> None` (writes `path` then `path.with_suffix(".parquet._SUCCESS")`), `shard_complete(path: Path) -> bool`, `read_db(root: Path) -> list[SampleRow]` (all complete shards under `root/*/shard-*.parquet`, sorted by source, query_index, sample_index).
- [ ] **Step 1:** Tests: `test_write_then_read_round_trips_all_fields` (incl. None values and empty tuples), `test_read_db_ignores_shard_without_success_marker`, `test_read_db_sorts_rows`, `test_write_shard_rejects_empty`.
- [ ] **Step 2:** Run; expect ImportError.
- [ ] **Step 3:** Implement with an explicit `pyarrow` schema (lists as `list<string>` / `list<float64>`); write to a temp file in the same dir and `os.replace`, then the marker.
- [ ] **Step 4:** Run tests; pass.
- [ ] **Step 5:** Commit `feat(eval): sample database rows and Parquet shards`.

### Task 3: Turn generated responses into rows

**Files:**
- Create: `src/drbm25/eval/sample.py` (library part)
- Test: `tests/eval/test_sample.py`

**Interfaces:**
- Consumes: `score_responses` (trainer), `think_text`, `SampleRow`, `Example` (data), `Searcher` protocol.
- Produces:
  - `rows_for_responses(examples: Sequence[Example], query_indices: Sequence[int], responses: Sequence[Sequence[GeneratedText]], searcher: Searcher, *, source: str, k: int, model_id: str | None, model_revision: str | None, seeds: Sequence[int | None]) -> list[SampleRow]` where `GeneratedText` is a frozen dataclass `(text: str, n_tokens: int, hit_max_tokens: bool, sum_logprob: float | None)`; responses[i] holds the N samples of examples[i].
  - `original_rows(examples, query_indices, searcher, *, k) -> list[SampleRow]` (source `original`, sample_index 0, response/think/logprob fields None, rewrite = query).
- [ ] **Step 1:** Tests with the existing fake searcher pattern from `tests/ppo` (deterministic doc ids per query): `test_rows_carry_reasoning_rewrite_and_both_modes`, `test_format_error_row_scores_zero_and_keeps_think_text` (Review Focus 1), `test_clause_overflow_row_has_empty_docs_and_search_error` (Review Focus 4, fake raises `TooManyClausesError`), `test_original_rows_score_raw_query`, `test_rows_are_ordered_by_query_then_sample`.
- [ ] **Step 2:** Run; fail on import.
- [ ] **Step 3:** Implement: flatten responses, one `score_responses` call per batch, rebuild rows; `terms` = `searcher.analyze(rewrite)` for parsed rows, `()` otherwise; `search_error` formatted like `QueryRecord` (`"too_many_clauses:<mode>[,<mode>]"`).
- [ ] **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat(eval): build sample rows from generated responses`.

### Task 4: vLLM sampler and CLI with resume

**Files:**
- Modify: `src/drbm25/eval/sample.py` (engine + CLI)
- Test: `tests/eval/test_sample.py`

**Interfaces:**
- Produces: `generate_samples(llm: Any, tokenizer, prompts: Sequence[str], seeds: Sequence[int], cfg: RolloutCfg, *, n: int, make_sampling_params: Callable[..., Any]) -> list[list[GeneratedText]]`; CLI `python -m drbm25.eval.sample --config PATH --source {original,untrained,trained-normal,trained-count_once} --n 32 --out-dir DIR [--model REPO] [--limit N] [--shard-size 128] [--base-seed 1234] [--set k=v ...]`.
- [ ] **Step 1:** Tests with a fake `LLM` (pattern of `FakeLLM` in `tests/rollout/test_vllm_engine.py`): `test_generate_samples_requests_n_samples_with_training_sampling_params` (asserts `n`, `temperature==0.6`, `top_k==0`, `logprobs==0`, `seed`, `stop_token_ids==[eos]`), `test_generate_samples_decodes_with_training_tokenizer`, `test_sum_logprob_is_sum_of_sampled_token_logprobs`, `test_cli_resumes_skipping_complete_shards` (Review Focus 3: pre-write shard 0 complete and shard 1 without marker; fake LLM records which prompts it saw: only shard 1 and later), `test_cli_original_source_needs_no_model`.
- [ ] **Step 2:** Run; fail.
- [ ] **Step 3:** Implement. `vllm.LLM(model=..., revision=sha, tokenizer_revision=sha, dtype="bfloat16", seed=base_seed, gpu_memory_utilization=0.85, max_model_len=max_prompt_tokens+max_new_tokens, generation_config="vllm")` (no external launcher, no sleep mode). Resolve `revision` for Hub repos with `HfApi().model_info(repo).sha`; untrained uses `cfg.model.name`/`cfg.model.revision`. Shards of `--shard-size` queries; skip a shard when `shard_complete`. Log progress (queries/s, samples/s) per shard.
- [ ] **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat(eval): vLLM sampler CLI with resumable shards`.

### Task 5: Text metrics

**Files:**
- Create: `src/drbm25/analysis/__init__.py`, `src/drbm25/analysis/text.py`
- Test: `tests/analysis/test_text.py`

**Interfaces:**
- Produces (all pure, on analyzed term tuples):
  - `TermDiff` frozen dataclass `(retention: float, jaccard: float, n_added: int, n_dropped: int, n_kept: int)`; `term_diff(original: Sequence[str], rewrite: Sequence[str]) -> TermDiff` on term sets. Empty-set rule (Review Focus 2): jaccard = 1.0 if both empty, 0.0 if exactly one empty; retention = 1.0 if original empty.
  - `Repetition` `(n_terms: int, n_unique: int, duplicate_rate: float, max_multiplicity: int, has_repeat: bool)`; `repetition(terms) -> Repetition` (duplicate_rate = 1 - unique/total; 0.0 for empty).
  - `rewrite_class(diff: TermDiff) -> Literal["same", "expansion", "compression", "substitution", "mixed"]`: same = no added, no dropped; expansion = added>0, dropped=0; compression = added=0, dropped>0; substitution = added>0, dropped>0, n_added == n_dropped; mixed otherwise.
  - `ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]`; `distinct_n(samples: Sequence[Sequence[str]], n: int) -> float` (unique n-grams / total n-grams over the samples; 0.0 when no n-grams).
  - `reasoning_overlap(think_terms: Sequence[str], rewrite_terms: Sequence[str]) -> float` = fraction of unique rewrite terms present in think terms (1.0 when rewrite empty).
- [ ] **Step 1:** Tests: exact values on small cases for each function, each empty case (Review Focus 2), `rewrite_class` for each label, and Hypothesis properties `term_diff(x, x).jaccard == 1.0`, `0 <= jaccard <= 1`, `0 <= retention <= 1`, `0 <= distinct_n <= 1`.
- [ ] **Step 2:** Run; fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat(analysis): term diff, repetition, n-gram and overlap metrics`.

### Task 6: Ranking and within-query diversity metrics

**Files:**
- Create: `src/drbm25/analysis/ranking.py`, `src/drbm25/analysis/diversity.py`
- Test: `tests/analysis/test_ranking.py`, `tests/analysis/test_diversity.py`

**Interfaces:**
- Produces:
  - `rbo_ext(a: Sequence[str], b: Sequence[str], *, p: float = 0.9) -> float`: extrapolated RBO for two prefix-truncated rankings of equal depth (Webber, Moffat, Zobel 2010, eq. 30 for equal lengths): `rbo_ext = X_k/k * p^k + (1-p)/p * sum_{d=1..k} X_d/d * p^d`, X_d = |a[:d] ∩ b[:d]|. Both empty -> 1.0; one empty -> 0.0. Verify the formula against the paper before coding; cross-check the `rbo` package value on 3 cases in a test fixture computed offline and hard-coded.
  - `jaccard_at_k(a, b, k) -> float` (same empty rule as Task 5).
  - `distinct_fraction(rewrites: Sequence[str | None]) -> float` (unique non-None / count, format errors counted as one value `None`).
  - `vendi_jaccard(samples: Sequence[Sequence[str]]) -> float`: K_ij = Jaccard of term sets (empty-set rule), `exp(-sum λ log λ)` over eigenvalues λ of K/n (λ <= 0 dropped).
  - `expected_best_of_k(scores: Sequence[float], k: int) -> float`: sorted ascending s_(1..n), `sum_{j=k..n} s_(j) * C(j-1, k-1) / C(n, k)`; raises if k > n.
- [ ] **Step 1:** Tests: RBO of identical lists = 1.0, disjoint = 0.0, known offline values; jaccard_at_k cases; Review Focus 5 (all-identical samples: distinct 1/32, vendi 1.0, best-of-k == mean); Vendi of n mutually disjoint samples = n; `expected_best_of_k(scores, 1) == mean`, `expected_best_of_k(scores, n) == max`, and a Monte Carlo check (seeded, 20,000 draws without replacement) within 0.005 for k=4, n=32.
- [ ] **Step 2:** Run; fail.
- [ ] **Step 3:** Implement with numpy (`numpy.linalg.eigvalsh`, `math.comb`).
- [ ] **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat(analysis): RBO, Jaccard@k, Vendi, distinct fraction, expected best-of-k`.

### Task 7: Policy-level lexical comparison

**Files:**
- Create: `src/drbm25/analysis/lexical.py`
- Test: `tests/analysis/test_lexical.py`

**Interfaces:**
- Produces:
  - `fightin_words(counts_a: Mapping[str, int], counts_b: Mapping[str, int], prior: Mapping[str, float]) -> dict[str, float]`: z-scores of the log-odds difference with informative Dirichlet prior (Monroe, Colaresi, Quinn 2008, eq. 16-22): δ_w = log((y_aw+α_w)/(n_a+α_0-y_aw-α_w)) - log((y_bw+α_w)/(n_b+α_0-y_bw-α_w)), σ² ≈ 1/(y_aw+α_w) + 1/(y_bw+α_w), z = δ/σ. Prior = pooled counts of all model sources scaled to sum α_0 = 1000 (document in the docstring). Verify formula against the paper before coding.
  - `js_divergence(counts_a, counts_b) -> float` (base 2, add-0.5 smoothing over the union vocabulary; scipy `jensenshannon` squared).
  - `c2st(texts_a: Sequence[str], texts_b: Sequence[str], groups_a: Sequence[str], groups_b: Sequence[str], *, seed: int = 0) -> C2STResult(accuracy: float, p_value: float, top_features_a: list[str], top_features_b: list[str])`: TF-IDF (word 1-2 grams) + LogisticRegression, `GroupKFold(5)` by query id, `permutation_test_score(n_permutations=200)`.
- [ ] **Step 1:** Tests: fightin_words sign (word only in A gets z>0), symmetry (swap a/b flips signs); js of identical = 0, of disjoint = 1; c2st on separable synthetic text (A uses "alpha", B "beta") accuracy > 0.95, p < 0.05; on identical distributions accuracy within 0.4-0.6; groups never split across folds (assert via a spy on the splitter or by construction).
- [ ] **Step 2:** Run; fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Pass.
- [ ] **Step 5:** Commit `feat(analysis): Fightin' Words, JS divergence, classifier two-sample test`.

### Task 8: Query-level statistics

**Files:**
- Create: `src/drbm25/analysis/stats.py`
- Test: `tests/analysis/test_stats.py`

**Interfaces:**
- Produces:
  - `PairedComparison` `(metric: str, a: str, b: str, mean_a: float, mean_b: float, mean_diff: float, ci_low: float, ci_high: float, wilcoxon_p: float, n_queries: int)`; `paired_compare(per_query_a: Mapping[str, float], per_query_b: Mapping[str, float], *, metric: str, a: str, b: str, n_resamples: int = 10_000, seed: int = 0) -> PairedComparison` (inner join on qid, diff = b - a as in the existing report; scipy `bootstrap((da, db), lambda x, y, axis: np.mean(y - x, axis=axis), paired=True, method="BCa")`; Wilcoxon `zero_method="wilcox"`; NaNs dropped per qid and counted).
  - `variance_components(per_sample: Mapping[str, Sequence[float]]) -> tuple[float, float, float]` -> (tau2 between-query, sigma2 within-query, icc) as in Miller 2024 sec. 5 (`tau2 = var(query means) - sigma2/K`).
  - `fdr_by(p_values: Sequence[float]) -> list[float]` (scipy `false_discovery_control(method="by")`).
- [ ] **Step 1:** Tests: identical inputs -> diff 0 and CI containing 0; a constant shift of +0.1 -> CI around 0.1; inner join drops unmatched qids; `variance_components` on synthetic data with known tau2=1, sigma2=4 (seeded, 2000 queries x 32) recovers both within 10%; `fdr_by` matches scipy on a fixed vector.
- [ ] **Step 2-4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat(analysis): paired query-level bootstrap, variance components, BY FDR`.

### Task 9: Index statistics and embedding similarity

**Files:**
- Create: `src/drbm25/analysis/corpus.py`
- Test: `tests/analysis/test_corpus.py`

**Interfaces:**
- Produces:
  - `IdfTable` with `idf(term: str) -> float` = `log(N / df)` (df=0 -> log(N/1) and counted as OOV); built by `IdfTable.from_index(index_dir: Path)` via pyserini `LuceneIndexReader` (`stats()["documents"]`, `get_term_counts(term, analyzer=None)`), cached per term.
  - `idf_summary(terms: Sequence[str], table) -> tuple[float, float]` (mean, max; NaN when empty, and callers report NaN count).
  - `embed_similarity(originals: Sequence[str], rewrites: Sequence[str | None], *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 512, device: str | None = None) -> list[float | None]` (cosine; None for None rewrites); encode unique strings once. Confirm the model id on the sentence-transformers docs before pinning; record the model revision.
- [ ] **Step 1:** Tests: `IdfTable` against the existing toy index fixture (`tests/bm25` `toy_index`): rare term idf > common term idf, unknown term handled; `embed_similarity` with an injected fake encoder (constructor arg `encode: Callable[[list[str]], np.ndarray]`) returns 1.0 for identical strings and None for None.
- [ ] **Step 2-4:** fail, implement, pass (`@pytest.mark.java` for the index test).
- [ ] **Step 5:** Commit `feat(analysis): IDF table from the Lucene index and embedding similarity`.

### Task 10: Metrics pipeline CLI

**Files:**
- Create: `src/drbm25/analysis/pipeline.py`
- Test: `tests/analysis/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 2, 5-9.
- Produces: CLI `python -m drbm25.analysis.pipeline --db DIR --index-dir DIR --out DIR [--embed-model NAME] [--no-embed]` writing `out/sample_metrics.parquet` (one row per sample: keys + all per-sample metrics), `out/query_metrics.parquet` (one row per (source, qid): means/stds and within-query diversity metrics), `out/comparisons.json` (list of `PairedComparison` for every metric x pair among `untrained`, `trained-normal`, `trained-count_once`, plus `original` where defined, with `q_by` added for the secondary family), `out/lexical.json` (Fightin' Words top 30 per direction for unigram/bigram/trigram, JS divergences, C2ST results for normal vs count_once on rewrites and on reasoning), `out/variance.json` (variance components per source for nDCG).
  - Per-sample metrics: term_diff vs the original query's analyzed terms, repetition, rewrite_class, rewrite length (terms, chars), think length (chars, words), reasoning_overlap, mean/max IDF of rewrite and of added terms, RBO and Jaccard@10 vs the original query's doc list (normal mode), embedding similarity, n_response_tokens, mean logprob.
  - Per-query metrics: means and std of nDCG (both modes), distinct_fraction, vendi_jaccard, expected_best_of_k for k in (1, 2, 4, 8, 16), mean pairwise RBO between samples (first 8 samples, normal mode), RBO between policies' modal rewrites.
- [ ] **Step 1:** Test on a tiny synthetic db (2 queries x 3 sources x 4 samples, fake index stats and `--no-embed`): output files exist, row counts (24 sample rows + 2 original; 8 query rows), a hand-computed retention and best-of-k value, comparisons contain nDCG primary without `q_by` and secondary metrics with `q_by`.
- [ ] **Step 2-4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat(analysis): metrics pipeline over the sample database`.

### Task 11: W&B logging and report

**Files:**
- Create: `src/drbm25/analysis/report.py`
- Test: `tests/analysis/test_report.py`

**Interfaces:**
- Produces: CLI `python -m drbm25.analysis.report --metrics DIR --db DIR --config PATH [--project drbm25] [--browse-rows 5000] [--no-report]`: one W&B run `rewrite-analysis` (job_type `analysis`) logging: summary scalars (per-source means with CIs), `query_metrics` Table, `comparisons` Table, `fightin_words` Tables, a `browse` Table of `--browse-rows` sampled rows (seed 0, stratified by source) with query, think_text, rewrite, nDCG both modes, matplotlib figures as `wandb.Image` (nDCG ECDF of per-query means by source; distributions of rewrite length, think length, retention, embedding similarity, duplicate rate; best-of-k curves; rewrite_class shares; Fightin' Words bar charts); artifact `rewrite-db` (type `dataset`) with the db and metrics dirs. Then a W&B Report (`wandb_workspaces.reports.v2`) titled "Query rewrite behavior: untrained vs normal vs count-once" with sections Summary, Retrieval quality, Rewrite behavior, Diversity, Vocabulary, Reasoning, Methods, built from panels of the logged keys. Report failures log a warning and leave the run intact (the run is the source of truth).
- [ ] **Step 1:** Tests with `WANDB_MODE=offline` in `tmp_path`: run logs the expected keys (read back from the offline run dir via the summary file), browse table has the requested row count; report building is behind a function `build_report(entity, project, run_path) -> str` tested with a fake `wr` module injected (asserts sections and the run path), real API not called in tests.
- [ ] **Step 2-4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat(analysis): W&B tables, figures and programmatic report`.

### Task 12: Hugging Face dataset upload

**Files:**
- Create: `src/drbm25/analysis/hub.py`
- Test: `tests/analysis/test_hub.py`

**Interfaces:**
- Produces: CLI `python -m drbm25.analysis.hub --db DIR --metrics DIR --repo objones25/drbm25-rewrites`: create the private dataset repo (`create_repo(repo_type="dataset", private=True, exist_ok=True)`), write a dataset card README (schema table from `SampleRow`, sources with model ids and revisions, sampling settings, N, license note: Qwen base model license, data derived from MS MARCO via DeepRetrieval), upload with `upload_folder`, then verify: list files via `HfApi.list_repo_tree(recursive=True)` and download each back with `hf_hub_download(force_download=True)` into a temp dir, comparing sha256 with the local file; write `db/UPLOADED.json` (repo, commit sha, file hashes) only after every file matched; exit 1 otherwise.
- [ ] **Step 1:** Tests with a fake `HfApi` and fake download function (dependency injection): success writes UPLOADED.json; a hash mismatch exits 1 and writes nothing; the card lists every source present.
- [ ] **Step 2-4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat(analysis): upload and verify the rewrite database on the Hugging Face Hub`.

### Task 13: Pod pipeline

**Files:**
- Create: `scripts/runpod/run_sampling.sh`
- Modify: `scripts/runpod/pod_bootstrap.sh` (entry point from `DRBM25_ENTRY`, default `scripts/runpod/run_all.sh`, must match `^scripts/runpod/[a-z_]+\.sh$`)
- Test: `tests/shell/test_runpod_sampling.bats`, `tests/shell/test_pod_bootstrap.bats` (or the existing bootstrap test file)

**Interfaces:**
- Consumes: lib.sh `runpod_paths`, `install_java21` (sets libjsig LD_PRELOAD), `install_uv`, `install_exit_traps`, `terminate_pod`, `_uv_timeout`, `stage`.
- Produces: `run_sampling.sh` stages: settings (require HF_TOKEN, WANDB_API_KEY; `SAMPLE_N` default 32, `PILOT_N` default 8, `PILOT_LIMIT` default 64), java and uv, uv sync (`--extra vllm --extra analysis`), pilot (`original` + each model source with `--n $PILOT_N --limit $PILOT_LIMIT` into `samples/pilot`), full (`original`, `untrained`, `trained-normal`, `trained-count_once` with `--n $SAMPLE_N` into `samples/db`), metrics, hub upload, report, finished. Root `$DRBM25_ROOT/samples`. FAILED marker `samples/FAILED` on failure; pod terminated on every exit unless `--no-terminate`.
- [ ] **Step 1:** bats tests with the existing `uv` stub pattern: stages run in order with the expected CLI args; a failing sampler writes `samples/FAILED` and still calls terminate; `DRBM25_ENTRY` with a value outside the pattern is rejected by bootstrap and falls back to failure (not to an arbitrary path); default entry is still run_all.sh.
- [ ] **Step 2-4:** fail, implement, pass; shellcheck clean.
- [ ] **Step 5:** Commit `feat(runpod): sampling pipeline script and bootstrap entry point`.

### Task 14: Rehearsal, pilot and full run

- [ ] **Step 1:** Full local gates: `uv run pytest -q`, `bats tests/shell`, `shellcheck`, ruff, pyright.
- [ ] **Step 2:** CPU rehearsal pod (cpu3c, EU-RO-1, volume): `uv sync --extra vllm --extra analysis`; run `original` sampler on `--limit 64`, metrics pipeline (`--no-embed` and with embed on CPU), hub upload to the real repo path under a `rehearsal/` prefix is NOT done; instead hub upload with `--repo objones25/drbm25-rewrites-rehearsal`, then delete that repo after checking (ask the user before deleting anything on the Hub). Delete the pod.
- [ ] **Step 3:** State the GPU type and hourly price to the user and create the GPU pod via MCP with `DRBM25_ENTRY=scripts/runpod/run_sampling.sh`, secrets from RunPod (`drbm25_hf_token`, `drbm25_wandb_api_key`, `drbm25_runpod_api_key`), watchdog deadline now + 12 h. Watch the pilot; report throughput and the projected full-run time/cost.
- [ ] **Step 4:** After the run: confirm `samples/db/UPLOADED.json`, the W&B report URL, and summarize findings to the user.
