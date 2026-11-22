import pytest
from hypothesis import given
from hypothesis import strategies as st

from drbm25.analysis.text import (
    TermDiff,
    distinct_n,
    ngrams,
    reasoning_overlap,
    repetition,
    rewrite_class,
    term_diff,
)

TERMS = st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), max_size=8)


def test_term_diff_counts_kept_added_dropped_on_sets() -> None:
    diff = term_diff(["heart", "attack", "aspirin", "aspirin"], ["aspirin", "myocardi", "infarct"])

    assert diff == TermDiff(
        retention=pytest.approx(1 / 3), jaccard=pytest.approx(1 / 4), n_added=2, n_dropped=2, n_kept=1
    )


def test_term_diff_both_empty_is_identical() -> None:
    assert term_diff([], []) == TermDiff(retention=1.0, jaccard=1.0, n_added=0, n_dropped=0, n_kept=0)


def test_term_diff_empty_rewrite_keeps_nothing() -> None:
    diff = term_diff(["a", "b"], [])

    assert (diff.retention, diff.jaccard, diff.n_dropped) == (0.0, 0.0, 2)


def test_term_diff_empty_original_has_full_retention() -> None:
    diff = term_diff([], ["a"])

    assert (diff.retention, diff.jaccard, diff.n_added) == (1.0, 0.0, 1)


@given(TERMS, TERMS)
def test_term_diff_bounds(a: list[str], b: list[str]) -> None:
    diff = term_diff(a, b)
    assert 0.0 <= diff.jaccard <= 1.0
    assert 0.0 <= diff.retention <= 1.0


@given(TERMS)
def test_term_diff_of_itself_is_identical(a: list[str]) -> None:
    assert term_diff(a, a).jaccard == 1.0


def test_repetition_profile() -> None:
    rep = repetition(["solar", "solar", "solar", "panel"])

    assert (rep.n_terms, rep.n_unique, rep.max_multiplicity, rep.has_repeat) == (4, 2, 3, True)
    assert rep.duplicate_rate == pytest.approx(0.5)


def test_repetition_of_empty_terms() -> None:
    rep = repetition([])

    assert (rep.n_terms, rep.n_unique, rep.duplicate_rate, rep.max_multiplicity, rep.has_repeat) == (
        0,
        0,
        0.0,
        0,
        False,
    )


@pytest.mark.parametrize(
    ("original", "rewrite", "label"),
    [
        (["a", "b"], ["b", "a"], "same"),
        (["a"], ["a", "b"], "expansion"),
        (["a", "b"], ["a"], "compression"),
        (["a", "b"], ["a", "c"], "substitution"),
        (["a", "b"], ["c", "d", "e"], "mixed"),
    ],
)
def test_rewrite_class(original: list[str], rewrite: list[str], label: str) -> None:
    assert rewrite_class(term_diff(original, rewrite)) == label


def test_ngrams() -> None:
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert ngrams(["a"], 2) == []


def test_ngrams_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError, match="n must be"):
        ngrams(["a"], 0)


def test_distinct_n_over_samples() -> None:
    assert distinct_n([["a", "b"], ["a", "b"], ["a", "c"]], 1) == pytest.approx(3 / 6)
    assert distinct_n([["a", "b"], ["a", "b"]], 2) == pytest.approx(1 / 2)


def test_distinct_n_without_ngrams_is_zero() -> None:
    assert distinct_n([["a"], []], 2) == 0.0


@given(st.lists(TERMS, max_size=5), st.integers(min_value=1, max_value=3))
def test_distinct_n_bounds(samples: list[list[str]], n: int) -> None:
    assert 0.0 <= distinct_n(samples, n) <= 1.0


def test_reasoning_overlap_is_share_of_rewrite_terms_in_reasoning() -> None:
    assert reasoning_overlap(["aspirin", "heart", "risk"], ["aspirin", "infarct", "aspirin"]) == pytest.approx(0.5)


def test_reasoning_overlap_of_empty_rewrite_is_one() -> None:
    assert reasoning_overlap(["x"], []) == 1.0
