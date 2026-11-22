"""Sample every query N times from a policy, parse and score each response, and store the rows.

CLI: ``python -m drbm25.eval.sample --config PATH --source SOURCE --n N --out-dir DIR [--model REPO]
[--limit N] [--shard-size Q] [--base-seed S] [--set key=value ...]``. Rows go to ``DIR/<source>/shard-<k>.parquet``
(``drbm25.eval.sample_db``); a rerun skips complete shards, so a crashed run resumes.

Responses are parsed and scored with the training parser and scorer (``parse_response`` /
``score_responses``), and sampled with the training rollout settings (``sampling_kwargs``: temperature, top-p 1,
no top-k, the training max_new_tokens) from the training prompt and tokenizer, so the database holds what the
policies actually do. Each query's request seed is ``base_seed * 1_000_003 + query_index * 1024`` for every source
(common random numbers across policies); vLLM seeds sample j with that + j, so seeds never overlap between queries
(n <= 1024), and each row stores the seed its sample used. ``sum_logprob`` is vLLM's log-probability of each
sampled token under the raw (temperature 1) model distribution, summed over the response.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, get_args

from huggingface_hub import HfApi
from transformers import PreTrainedTokenizerBase

from drbm25.bm25.searcher import Bm25Searcher, SearchResult
from drbm25.config import Config, RolloutCfg, ScoringMode, load_config
from drbm25.data import DEV_FILE, Example, load_examples
from drbm25.eval.records import (
    SOURCE_ORIGINAL,
    SOURCE_TRAINED_COUNT_ONCE,
    SOURCE_TRAINED_NORMAL,
    SOURCE_UNTRAINED,
    search_error_label,
)
from drbm25.eval.sample_db import SampleRow, shard_complete, write_shard
from drbm25.obs import configure
from drbm25.ppo.trainer import Searcher, score_responses, search_all
from drbm25.prompt import build_prompt
from drbm25.reward import FormatError, ParsedRewrite, ndcg_at_k, think_text
from drbm25.rollout.base import encode_prompts, special_ids
from drbm25.rollout.vllm_engine import sampling_kwargs
from drbm25.train import build_tokenizer, check_index, check_java

logger = logging.getLogger(__name__)

_MODES: Final[tuple[ScoringMode, ...]] = get_args(ScoringMode)
SAMPLE_SOURCES: Final = (SOURCE_ORIGINAL, SOURCE_UNTRAINED, SOURCE_TRAINED_NORMAL, SOURCE_TRAINED_COUNT_ONCE)
# The verified uploads of the two training runs (runs/<mode>/UPLOADED.json on the volume).
DEFAULT_MODELS: Final = {
    SOURCE_TRAINED_NORMAL: "objones25/drbm25-qwen2.5-3b-normal",
    SOURCE_TRAINED_COUNT_ONCE: "objones25/drbm25-qwen2.5-3b-count_once",
}
SEED_STRIDE: Final = 1_000_003
# vLLM seeds sample j of an n-sample request with seed + j, so query seeds are spaced wider than any n.
SEED_QUERY_STRIDE: Final = 1024
VLLM_GPU_MEMORY_UTILIZATION: Final = 0.85  # inference only: nothing else shares the GPU


@dataclass(frozen=True)
class GeneratedText:
    """One sampled response: decoded text, its token count, whether it hit the length limit, and the sum of
    its sampled tokens' log-probabilities (None when the engine did not return them)."""

    text: str
    n_tokens: int
    hit_max_tokens: bool
    sum_logprob: float | None


def _docs(result: SearchResult | None) -> tuple[tuple[str, ...], tuple[float, ...]]:
    return ((), ()) if result is None else (tuple(result.doc_ids), tuple(result.scores))


def rows_for_responses(
    examples: Sequence[Example],
    query_indices: Sequence[int],
    responses: Sequence[Sequence[GeneratedText]],
    searcher: Searcher,
    *,
    source: str,
    k: int,
    model_id: str | None,
    model_revision: str | None,
    seeds: Sequence[int | None],
) -> list[SampleRow]:
    """Rows for ``responses[i]`` (the samples of ``examples[i]``), ordered by query then sample."""
    if not len(examples) == len(query_indices) == len(responses) == len(seeds):
        raise ValueError(
            f"{len(examples)} examples, {len(query_indices)} query indices, {len(responses)} responses and "
            f"{len(seeds)} seeds must have the same length"
        )
    flat = [(i, j, gen) for i, samples in enumerate(responses) for j, gen in enumerate(samples)]
    if not flat:
        raise ValueError("no responses to score")
    scored = score_responses(searcher, [g.text for _, _, g in flat], [examples[i] for i, _, _ in flat], k=k)
    rows = []
    for pos, (i, j, gen) in enumerate(flat):
        parsed = scored.parsed[pos]
        found: dict[ScoringMode, SearchResult | None] = {mode: scored.results[mode][pos] for mode in _MODES}
        if isinstance(parsed, ParsedRewrite):
            rewrite, format_error = parsed.query, None
            terms, search_error = tuple(searcher.analyze(parsed.query)), search_error_label(found)
        else:
            if not isinstance(parsed, FormatError):
                raise AssertionError(f"unexpected parse result {parsed!r}")
            rewrite, format_error, terms, search_error = None, parsed.reason.value, (), None
        docs_normal, scores_normal = _docs(found["normal"])
        docs_once, scores_once = _docs(found["count_once"])
        rows.append(
            SampleRow(
                source=source,
                qid=examples[i].qid,
                query_index=query_indices[i],
                sample_index=j,
                query_text=examples[i].query,
                response_text=gen.text,
                think_text=think_text(gen.text),
                rewrite=rewrite,
                format_error=format_error,
                terms=terms,
                n_response_tokens=gen.n_tokens,
                hit_max_tokens=gen.hit_max_tokens,
                sum_logprob=gen.sum_logprob,
                doc_ids_normal=docs_normal,
                scores_normal=scores_normal,
                doc_ids_count_once=docs_once,
                scores_count_once=scores_once,
                ndcg_normal=scored.ndcg["normal"][pos],
                ndcg_count_once=scored.ndcg["count_once"][pos],
                search_error=search_error,
                model_id=model_id,
                model_revision=model_revision,
                seed=None if (query_seed := seeds[i]) is None else query_seed + j,  # the seed vLLM used
            )
        )
    return rows


def original_rows(
    examples: Sequence[Example], query_indices: Sequence[int], searcher: Searcher, *, k: int
) -> list[SampleRow]:
    """One row per example scoring its own query text (source ``original``)."""
    if len(examples) != len(query_indices) or not examples:
        raise ValueError(
            f"need matching, non-empty examples and query indices, got {len(examples)}/{len(query_indices)}"
        )
    queries = [e.query for e in examples]
    found_all = {mode: search_all(searcher, queries, mode=mode, k=k) for mode in _MODES}
    rows = []
    for i, example in enumerate(examples):
        found: dict[ScoringMode, SearchResult | None] = {mode: found_all[mode][i] for mode in _MODES}
        docs_normal, scores_normal = _docs(found["normal"])
        docs_once, scores_once = _docs(found["count_once"])
        rows.append(
            SampleRow(
                source=SOURCE_ORIGINAL,
                qid=example.qid,
                query_index=query_indices[i],
                sample_index=0,
                query_text=example.query,
                response_text=None,
                think_text=None,
                rewrite=example.query,
                format_error=None,
                terms=tuple(searcher.analyze(example.query)),
                n_response_tokens=None,
                hit_max_tokens=None,
                sum_logprob=None,
                doc_ids_normal=docs_normal,
                scores_normal=scores_normal,
                doc_ids_count_once=docs_once,
                scores_count_once=scores_once,
                ndcg_normal=0.0 if found["normal"] is None else ndcg_at_k(docs_normal, example.relevant_ids, k),
                ndcg_count_once=0.0 if found["count_once"] is None else ndcg_at_k(docs_once, example.relevant_ids, k),
                search_error=search_error_label(found),
                model_id=None,
                model_revision=None,
                seed=None,
            )
        )
    return rows


# ---------------------------------------------------------------- generation


def _completion_sample(completion: Any, eos: int, tokenizer: PreTrainedTokenizerBase) -> GeneratedText:
    tokens = list(completion.token_ids)
    end = tokens.index(eos) + 1 if eos in tokens else len(tokens)
    kept = tokens[:end]
    sum_logprob = None
    if completion.logprobs is not None:
        sum_logprob = float(sum(step[tok].logprob for step, tok in zip(completion.logprobs[:end], kept, strict=True)))
    return GeneratedText(
        text=tokenizer.decode(kept, skip_special_tokens=True),
        n_tokens=len(kept),
        hit_max_tokens=eos not in tokens,
        sum_logprob=sum_logprob,
    )


def generate_samples(
    llm: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    seeds: Sequence[int],
    cfg: RolloutCfg,
    *,
    n: int,
    make_sampling_params: Callable[..., Any],
) -> list[list[GeneratedText]]:
    """``n`` samples per prompt with the training rollout sampling settings, one seed per prompt."""
    if len(prompts) != len(seeds):
        raise ValueError(f"{len(prompts)} prompts but {len(seeds)} seeds")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    ids = special_ids(tokenizer)
    import torch

    prompt_ids, prompt_mask = encode_prompts(
        tokenizer, prompts, torch.device("cpu"), max_prompt_tokens=cfg.max_prompt_tokens
    )
    requests = [{"prompt_token_ids": row[m.bool()].tolist()} for row, m in zip(prompt_ids, prompt_mask, strict=True)]
    base = sampling_kwargs(cfg, greedy=False, eos_id=ids.eos)
    params = [make_sampling_params(**{**base, "n": n, "logprobs": 0, "seed": seed}) for seed in seeds]
    outputs = llm.generate(requests, sampling_params=params, use_tqdm=False)
    if len(outputs) != len(requests):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(requests)} prompts")
    result = []
    for out in outputs:
        if len(out.outputs) != n:
            raise RuntimeError(f"vLLM returned {len(out.outputs)} completions, expected n={n}")
        result.append([_completion_sample(c, ids.eos, tokenizer) for c in out.outputs])
    return result


def _build_llm(model: str, revision: str, cfg: Config, seed: int) -> tuple[Any, Callable[..., Any]]:
    """A standalone vLLM engine for ``model`` at ``revision`` and its SamplingParams class.

    The engine runs in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``), as in training: the JVM (BM25) is
    already running here, and training ran vLLM in-process next to it; a forked engine process would not."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    vllm = importlib.import_module("vllm")
    llm = vllm.LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=cfg.rollout.max_prompt_tokens + cfg.rollout.max_new_tokens,
        generation_config="vllm",
    )
    return llm, vllm.SamplingParams


def _resolve_revision(model: str, revision: str | None) -> str:
    """A pinned revision as given, else the Hub repo's current commit sha (recorded in every row)."""
    if revision is not None:
        return revision
    sha = HfApi().model_info(model).sha
    if not sha:
        raise RuntimeError(f"could not resolve a commit sha for {model}")
    return sha


def _model_for(source: str, override: str | None, cfg: Config) -> tuple[str, str | None]:
    if source == SOURCE_UNTRAINED:
        return (override, None) if override else (cfg.model.name, cfg.model.revision)
    return override or DEFAULT_MODELS[source], None


# ---------------------------------------------------------------- CLI


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample dev queries N times from a policy into Parquet shards.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", choices=SAMPLE_SOURCES, required=True)
    parser.add_argument("--n", type=int, required=True, help="samples per query (ignored for original)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=None, help="model repo or path (default: per source)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=128, help="queries per shard")
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ns = parser.parse_args(argv)
    if ns.n < 1 or ns.shard_size < 1:
        parser.error("--n and --shard-size must be >= 1")
    if ns.n > SEED_QUERY_STRIDE:
        parser.error(f"--n must be <= {SEED_QUERY_STRIDE} (the per-query seed stride)")
    overrides: dict[str, str] = {}
    for item in ns.overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--set expects key=value, got {item!r}")
        overrides[key] = value
    ns.overrides = overrides
    return ns


def main(argv: Sequence[str] | None = None) -> int:
    ns = _parse_args(argv)
    configure(level="INFO", fmt="json")
    cfg = load_config(ns.config, ns.overrides)
    check_java()
    check_index(cfg)
    r = cfg.retrieval
    searcher = Bm25Searcher(r.index_dir, k1=r.k1, b=r.b, threads=r.search_threads, timeout_s=r.search_timeout_s)
    examples = load_examples(cfg.data.data_dir / DEV_FILE)[: ns.limit]
    out = ns.out_dir / ns.source
    shards = [(start, examples[start : start + ns.shard_size]) for start in range(0, len(examples), ns.shard_size)]
    pending = [(k, start, chunk) for k, (start, chunk) in enumerate(shards) if not shard_complete(_shard(out, k))]
    logger.info("%s: %d queries in %d shards, %d to do", ns.source, len(examples), len(shards), len(pending))
    if not pending:
        return 0
    if ns.source == SOURCE_ORIGINAL:
        for k, start, chunk in pending:
            indices = list(range(start, start + len(chunk)))
            write_shard(_shard(out, k), original_rows(chunk, indices, searcher, k=r.eval_k))
        return 0
    model, pinned = _model_for(ns.source, ns.model, cfg)
    revision = _resolve_revision(model, pinned)
    logger.info("sampling %s from %s@%s, n=%d", ns.source, model, revision, ns.n)
    llm, make_sampling_params = _build_llm(model, revision, cfg, ns.base_seed)
    tokenizer = build_tokenizer(cfg)
    for k, start, chunk in pending:
        t0 = time.perf_counter()
        indices = list(range(start, start + len(chunk)))
        seeds = [ns.base_seed * SEED_STRIDE + i * SEED_QUERY_STRIDE for i in indices]
        gens = generate_samples(
            llm,
            tokenizer,
            [build_prompt(e.query) for e in chunk],
            seeds,
            cfg.rollout,
            n=ns.n,
            make_sampling_params=make_sampling_params,
        )
        rows = rows_for_responses(
            chunk,
            indices,
            gens,
            searcher,
            source=ns.source,
            k=r.eval_k,
            model_id=model,
            model_revision=revision,
            seeds=seeds,
        )
        write_shard(_shard(out, k), rows)
        elapsed = time.perf_counter() - t0
        logger.info(
            "shard %d: %d queries, %d samples in %.1fs (%.1f samples/s)",
            k,
            len(chunk),
            len(rows),
            elapsed,
            len(rows) / elapsed,
        )
    return 0


def _shard(out: Path, k: int) -> Path:
    return out / f"shard-{k:05d}.parquet"


if __name__ == "__main__":
    raise SystemExit(main())
