from collections.abc import Sequence

import pytest

from drbm25.bm25.searcher import SearchResult, TooManyClausesError
from drbm25.config import ScoringMode
from drbm25.data import Example
from drbm25.eval.sample import GeneratedText, original_rows, rows_for_responses

K = 10


class _WordSearcher:
    """Doc ids are the query's words (``d:<word>``); ``boom`` overflows in count_once mode."""

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult:
        if "boom" in query.split() and mode == "count_once":
            raise TooManyClausesError("too many")
        words = query.split()[:k]
        docs = tuple(dict.fromkeys(f"d:{w}" for w in words))
        return SearchResult(docs, len(words), len(set(words)), tuple(float(len(docs) - i) for i in range(len(docs))))

    def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]:
        return [self.search(q, mode=mode, k=k) for q in queries]

    def analyze(self, text: str) -> list[str]:
        return text.lower().split()


def _answer(query: str, think: str = "reason") -> str:
    return f'{think}\n</think>\n<answer>{{"query": "{query}"}}</answer>'


def _gen(text: str, n_tokens: int = 5, logprob: float | None = -1.0) -> GeneratedText:
    return GeneratedText(text=text, n_tokens=n_tokens, hit_max_tokens=False, sum_logprob=logprob)


EX = [Example("q1", "Heart attack", frozenset({"d:aspirin"})), Example("q2", "copper uses", frozenset({"d:wire"}))]


def _rows(responses: list[list[GeneratedText]]):
    return rows_for_responses(
        EX,
        [0, 1],
        responses,
        _WordSearcher(),
        source="trained-normal",
        k=K,
        model_id="m",
        model_revision="sha",
        seeds=[11, 12],
    )


def test_rows_carry_reasoning_rewrite_and_both_modes() -> None:
    rows = _rows([[_gen(_answer("aspirin heart", "think hard"))], [_gen(_answer("wire copper"))]])

    first = rows[0]
    assert (first.qid, first.query_index, first.sample_index, first.seed) == ("q1", 0, 0, 11)
    assert first.think_text == "think hard"
    assert first.rewrite == "aspirin heart"
    assert first.terms == ("aspirin", "heart")
    assert first.doc_ids_normal == ("d:aspirin", "d:heart")
    assert first.scores_count_once == (2.0, 1.0)
    assert first.ndcg_normal == pytest.approx(1.0)
    assert first.ndcg_count_once == pytest.approx(1.0)
    assert (first.model_id, first.model_revision, first.n_response_tokens, first.sum_logprob) == ("m", "sha", 5, -1.0)
    assert first.format_error is None
    assert first.search_error is None


def test_format_error_row_scores_zero_and_keeps_think_text() -> None:
    rows = _rows([[_gen("half a thought </think> no answer")], [_gen(_answer("wire"))]])

    bad = rows[0]
    assert bad.format_error == "tag_count"
    assert bad.think_text == "half a thought"
    assert bad.rewrite is None
    assert bad.terms == ()
    assert bad.doc_ids_normal == ()
    assert bad.doc_ids_count_once == ()
    assert (bad.ndcg_normal, bad.ndcg_count_once) == (0.0, 0.0)
    assert bad.response_text == "half a thought </think> no answer"


def test_clause_overflow_row_has_empty_docs_and_search_error() -> None:
    rows = _rows([[_gen(_answer("aspirin boom"))], [_gen(_answer("wire"))]])

    over = rows[0]
    assert over.search_error == "too_many_clauses:count_once"
    assert over.doc_ids_count_once == ()
    assert over.ndcg_count_once == 0.0
    assert over.doc_ids_normal == ("d:aspirin", "d:boom")


def test_rows_are_ordered_by_query_then_sample() -> None:
    rows = _rows(
        [[_gen(_answer("a")), _gen(_answer("b"))], [_gen(_answer("c")), _gen(_answer("d"))]],
    )

    assert [(r.query_index, r.sample_index, r.rewrite) for r in rows] == [
        (0, 0, "a"),
        (0, 1, "b"),
        (1, 0, "c"),
        (1, 1, "d"),
    ]


def test_original_rows_score_raw_query() -> None:
    rows = original_rows([Example("q9", "aspirin heart", frozenset({"d:heart"}))], [4], _WordSearcher(), k=K)

    (row,) = rows
    assert (row.source, row.qid, row.query_index, row.sample_index) == ("original", "q9", 4, 0)
    assert row.rewrite == "aspirin heart"
    assert row.terms == ("aspirin", "heart")
    assert row.response_text is None
    assert row.think_text is None
    assert row.sum_logprob is None
    assert row.ndcg_normal == pytest.approx(0.6309, abs=1e-4)


def test_rows_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="responses"):
        _rows([[_gen(_answer("a"))]])
