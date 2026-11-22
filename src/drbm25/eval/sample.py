"""Sample every query N times from a policy, parse and score each response, and store the rows.

Library part: turning generated texts into :class:`~drbm25.eval.sample_db.SampleRow` rows with the training
parser and scorer (``parse_response`` / ``score_responses``), so the database holds exactly what the policies
were rewarded for.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, get_args

from drbm25.bm25.searcher import SearchResult
from drbm25.config import ScoringMode
from drbm25.data import Example
from drbm25.eval.records import SOURCE_ORIGINAL, search_error_label
from drbm25.eval.sample_db import SampleRow
from drbm25.ppo.trainer import Searcher, score_responses, search_all
from drbm25.reward import FormatError, ParsedRewrite, ndcg_at_k, think_text

logger = logging.getLogger(__name__)

_MODES: Final[tuple[ScoringMode, ...]] = get_args(ScoringMode)


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
                seed=seeds[i],
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
