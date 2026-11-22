import pytest

from drbm25.analysis.ranking import jaccard_at_k, rbo_ext

# Reference values from the rbo package 0.1.3 (RankingSimilarity(a, b).rbo_ext(p=0.9)), computed offline.
RBO_CASES = [
    ("abcdefghij", "abcdefghij", 1.0),
    ("abcde", "bacde", 0.9),
    ("abcde", "fghij", 0.0),
    ("abcdefghij", "cbadefhgji", 0.8426250167142857),
    ("abcd", "adbc", 0.928),
]


@pytest.mark.parametrize(("a", "b", "expected"), RBO_CASES)
def test_rbo_ext_matches_reference_values(a: str, b: str, expected: float) -> None:
    assert rbo_ext(list(a), list(b), p=0.9) == pytest.approx(expected, abs=1e-12)


def test_rbo_ext_empty_rule() -> None:
    assert rbo_ext([], []) == 1.0
    assert rbo_ext(["a"], []) == 0.0


# Uneven lengths (a query that matched fewer than k documents), same reference package.
UNEVEN_CASES = [
    ("abcdefghij", "abc", 1.0),
    ("abc", "xbayz", 0.585),
    ("abcdefgh", "hgfe", 0.2947294928571429),
]


@pytest.mark.parametrize(("a", "b", "expected"), UNEVEN_CASES)
def test_rbo_ext_handles_uneven_lengths(a: str, b: str, expected: float) -> None:
    assert rbo_ext(list(a), list(b), p=0.9) == pytest.approx(expected, abs=1e-12)
    assert rbo_ext(list(b), list(a), p=0.9) == pytest.approx(expected, abs=1e-12)


def test_rbo_ext_rejects_bad_p() -> None:
    with pytest.raises(ValueError, match="p must be"):
        rbo_ext(["a"], ["a"], p=1.0)


def test_jaccard_at_k() -> None:
    assert jaccard_at_k(["a", "b", "c"], ["b", "c", "d"], 2) == pytest.approx(1 / 3)
    assert jaccard_at_k([], [], 10) == 1.0
    assert jaccard_at_k(["a"], [], 10) == 0.0
