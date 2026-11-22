import math
import random

import pytest

from drbm25.analysis.lexical import c2st, fightin_words, js_divergence, scaled_prior


def test_fightin_words_hand_value() -> None:
    # delta_x = log(11/1) - log(1/11); var = 1/11 + 1/1
    z = fightin_words({"x": 10, "y": 0}, {"x": 0, "y": 10}, {"x": 1.0, "y": 1.0})

    assert z["x"] == pytest.approx(2 * math.log(11) / math.sqrt(12 / 11))
    assert z["y"] == pytest.approx(-z["x"])


def test_fightin_words_word_only_in_a_scores_positive() -> None:
    # "the" has the same rate in both (100 of 200 tokens); "aspirin" appears only in A
    z = fightin_words(
        {"aspirin": 100, "the": 100}, {"the": 100, "solar": 100}, {"aspirin": 1.0, "the": 2.0, "solar": 1.0}
    )

    assert z["aspirin"] > 1.96
    assert abs(z["the"]) < 1.96


def test_fightin_words_is_antisymmetric() -> None:
    a, b = {"p": 7, "q": 2, "r": 1}, {"p": 1, "q": 5, "r": 3}
    prior = scaled_prior({"p": 8, "q": 7, "r": 4}, alpha0=10.0)

    za, zb = fightin_words(a, b, prior), fightin_words(b, a, prior)

    assert all(za[w] == pytest.approx(-zb[w]) for w in prior)


def test_scaled_prior_sums_to_alpha0() -> None:
    prior = scaled_prior({"a": 3, "b": 1}, alpha0=1000.0)

    assert prior == {"a": pytest.approx(750.0), "b": pytest.approx(250.0)}


def test_js_divergence_bounds() -> None:
    assert js_divergence({"a": 5, "b": 5}, {"a": 5, "b": 5}) == pytest.approx(0.0)
    assert js_divergence({"a": 1_000_000}, {"b": 1_000_000}) == pytest.approx(1.0, abs=1e-4)  # add-0.5 smoothing


def _texts(word: str, n: int, rng: random.Random) -> list[str]:
    filler = ["solar", "panel", "heart", "copper", "water", "boiling"]
    return [" ".join([word, *rng.sample(filler, 3)]) for _ in range(n)]


def test_c2st_separates_distinguishable_policies() -> None:
    rng = random.Random(0)
    groups = [f"q{i}" for i in range(60)]

    res = c2st(_texts("alpha", 60, rng), _texts("beta", 60, rng), groups, groups, n_permutations=50, seed=0)

    assert res.accuracy > 0.95
    assert res.p_value < 0.05
    assert "alpha" in res.top_features_a
    assert "beta" in res.top_features_b


def test_c2st_is_near_chance_for_identical_policies() -> None:
    rng = random.Random(1)
    groups = [f"q{i}" for i in range(60)]

    res = c2st(_texts("same", 60, rng), _texts("same", 60, rng), groups, groups, n_permutations=50, seed=0)

    assert 0.3 <= res.accuracy <= 0.7
    assert res.p_value > 0.05


def test_c2st_rejects_mismatched_groups() -> None:
    with pytest.raises(ValueError, match="groups"):
        c2st(["a b"], ["c d"], ["q1", "q2"], ["q1"])
