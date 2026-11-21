from pathlib import Path

import pytest

from drbm25.bm25.searcher import Bm25Searcher, SearchResult, TooManyClausesError
from drbm25.config import ScoringMode

pytestmark = pytest.mark.java

MODES: tuple[ScoringMode, ...] = ("normal", "count_once")
REPEATED_QUERY = "solar solar solar efficiency"
DOC_A = "docA"  # solar x3, built in conftest.toy_index
DOC_B = "docB"  # solar x1 + efficiency x1


@pytest.fixture(scope="module")
def searcher(toy_index: Path) -> Bm25Searcher:
    return Bm25Searcher(toy_index, k1=0.9, b=0.4, threads=2, timeout_s=30.0)


def _rank(result: SearchResult, doc_id: str) -> int:
    assert doc_id in result.doc_ids, f"{doc_id} not retrieved: {result.doc_ids}"
    return result.doc_ids.index(doc_id)


def test_analyze_drops_boolean_operators_and_stems(searcher: Bm25Searcher) -> None:
    assert searcher.analyze("solar AND (wind OR efficiency)") == ["solar", "wind", "effici"]


def test_normal_mode_lets_repetition_flip_ranking(searcher: Bm25Searcher) -> None:
    result = searcher.search(REPEATED_QUERY, mode="normal", k=10)
    assert _rank(result, DOC_A) < _rank(result, DOC_B)


def test_count_once_ignores_repetition(searcher: Bm25Searcher) -> None:
    result = searcher.search(REPEATED_QUERY, mode="count_once", k=10)
    assert _rank(result, DOC_B) < _rank(result, DOC_A)


def test_count_once_matches_deduplicated_normal_query(searcher: Bm25Searcher) -> None:
    once = searcher.search(REPEATED_QUERY, mode="count_once", k=10)
    dedup = searcher.search("solar efficiency", mode="normal", k=10)
    assert (once.doc_ids, once.scores) == (dedup.doc_ids, dedup.scores)


def test_modes_agree_without_repeats(searcher: Bm25Searcher) -> None:
    normal = searcher.search("solar efficiency", mode="normal", k=10)
    once = searcher.search("solar efficiency", mode="count_once", k=10)
    assert len(normal.doc_ids) == 10
    assert normal.doc_ids == once.doc_ids
    assert normal.scores == once.scores
    assert normal.scores[0] > 0.0


def test_scores_are_descending(searcher: Bm25Searcher) -> None:
    result = searcher.search(REPEATED_QUERY, mode="normal", k=10)
    assert list(result.scores) == sorted(result.scores, reverse=True)
    assert len(result.scores) == len(result.doc_ids) == 10


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("mode", MODES)
def test_k_limits_hits(searcher: Bm25Searcher, mode: ScoringMode, k: int) -> None:
    assert len(searcher.search("solar", mode=mode, k=k).doc_ids) == k


@pytest.mark.parametrize("mode", MODES)
def test_empty_query_returns_no_hits(searcher: Bm25Searcher, mode: ScoringMode) -> None:
    assert searcher.search("AND OR ( )", mode=mode, k=10) == SearchResult((), 0, 0, ())


@pytest.mark.parametrize("mode", MODES)
def test_too_many_unique_terms_raises(searcher: Bm25Searcher, mode: ScoringMode) -> None:
    query = " ".join(f"zq{i}x" for i in range(1100))
    with pytest.raises(TooManyClausesError):
        searcher.search(query, mode=mode, k=10)


@pytest.mark.parametrize("mode", MODES)
def test_reports_term_counts(searcher: Bm25Searcher, mode: ScoringMode) -> None:
    result = searcher.search("solar solar efficiency", mode=mode, k=10)
    assert (result.n_terms, result.n_unique_terms) == (3, 2)


@pytest.mark.parametrize("mode", MODES)
def test_batch_search_preserves_order(searcher: Bm25Searcher, mode: ScoringMode) -> None:
    queries = ["solar", "efficiency", REPEATED_QUERY]
    batch = searcher.batch_search(queries, mode=mode, k=5)
    assert batch == [searcher.search(q, mode=mode, k=5) for q in queries]
    assert batch[0] != batch[1]


def test_batch_search_surfaces_too_many_clauses(searcher: Bm25Searcher) -> None:
    query = " ".join(f"zq{i}x" for i in range(1100))
    with pytest.raises(TooManyClausesError):
        searcher.batch_search(["solar", query], mode="count_once", k=5)


def test_rejects_unknown_mode(searcher: Bm25Searcher) -> None:
    with pytest.raises(ValueError, match="mode"):
        searcher.search("solar", mode="bogus", k=10)  # type: ignore[arg-type]  # invalid Literal on purpose


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_non_positive_k(searcher: Bm25Searcher, k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        searcher.search("solar", mode="normal", k=k)
