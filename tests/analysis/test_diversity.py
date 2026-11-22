import random
import statistics

import pytest

from drbm25.analysis.diversity import distinct_fraction, expected_best_of_k, vendi_jaccard


def test_identical_samples_have_minimal_diversity() -> None:
    samples = [["aspirin", "heart"]] * 32

    assert distinct_fraction(["aspirin heart"] * 32) == pytest.approx(1 / 32)
    assert vendi_jaccard(samples) == pytest.approx(1.0)


def test_disjoint_samples_have_vendi_equal_to_their_count() -> None:
    assert vendi_jaccard([["a"], ["b"], ["c"], ["d"]]) == pytest.approx(4.0)


def test_vendi_handles_empty_term_lists() -> None:
    # two empty samples are identical to each other and share nothing with ["a"]
    assert vendi_jaccard([[], [], ["a"]]) == pytest.approx(vendi_jaccard([["b"], ["b"], ["a"]]))


def test_distinct_fraction_counts_format_errors_as_one_value() -> None:
    assert distinct_fraction(["x", None, None, "y"]) == pytest.approx(3 / 4)


def test_best_of_k_extremes() -> None:
    scores = [0.1, 0.5, 0.2, 0.9, 0.0]

    assert expected_best_of_k(scores, 1) == pytest.approx(statistics.mean(scores))
    assert expected_best_of_k(scores, 5) == pytest.approx(0.9)


def test_best_of_k_of_constant_scores_is_the_constant() -> None:
    assert expected_best_of_k([0.4] * 32, 8) == pytest.approx(0.4)


def test_best_of_k_matches_monte_carlo() -> None:
    rng = random.Random(0)
    scores = [rng.random() for _ in range(32)]
    draws = [max(rng.sample(scores, 4)) for _ in range(20_000)]

    assert expected_best_of_k(scores, 4) == pytest.approx(statistics.mean(draws), abs=0.005)


def test_best_of_k_rejects_k_above_n() -> None:
    with pytest.raises(ValueError, match="k"):
        expected_best_of_k([0.1, 0.2], 3)
