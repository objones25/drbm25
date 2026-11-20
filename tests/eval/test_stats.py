"""paired_bootstrap_ci and query_stats."""

import numpy as np
import pytest

from drbm25.eval.records import QueryRecord
from drbm25.eval.stats import PairedCI, paired_bootstrap_ci, query_stats


def _record(terms: tuple[str, ...], *, format_error: str | None = None, qid: str = "q") -> QueryRecord:
    return QueryRecord(
        qid=qid,
        source="s",
        query_text="q",
        rewrite_text=None if format_error is None else "r",
        format_error=format_error,
        terms=terms,
        ndcg_normal=0.0,
        ndcg_count_once=0.0,
    )


def test_bootstrap_identical_arrays_gives_zero_interval() -> None:
    a = np.array([0.1, 0.5, 0.9, 0.0])
    assert paired_bootstrap_ci(a, a.copy(), n_resamples=200) == PairedCI(0.0, 0.0, 0.0)


def test_bootstrap_detects_constant_shift() -> None:
    a = np.random.default_rng(1).random(50)
    ci = paired_bootstrap_ci(a, a + 0.1, n_resamples=500)
    assert ci.mean_diff == pytest.approx(0.1)
    assert ci.low == pytest.approx(0.1)
    assert ci.high == pytest.approx(0.1)


def test_bootstrap_seeded_reproducible_and_seed_matters() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.random(30), rng.random(30)
    first = paired_bootstrap_ci(a, b, n_resamples=300, seed=7)
    assert paired_bootstrap_ci(a, b, n_resamples=300, seed=7) == first
    assert paired_bootstrap_ci(a, b, n_resamples=300, seed=8) != first


def test_bootstrap_interval_contains_mean_and_is_ordered() -> None:
    rng = np.random.default_rng(4)
    a, b = rng.random(200), rng.random(200)
    ci = paired_bootstrap_ci(a, b, n_resamples=2000)
    assert ci.mean_diff == pytest.approx(float(np.mean(b - a)))
    assert ci.low < ci.mean_diff < ci.high


def test_bootstrap_interval_width_matches_normal_theory() -> None:
    # d has mean 0 and sd 1; the standard error of the mean is 1/sqrt(n), so a 95% interval is ~ +-1.96/sqrt(n).
    n = 400
    d = np.random.default_rng(5).standard_normal(n)
    d = (d - d.mean()) / d.std()
    ci = paired_bootstrap_ci(np.zeros(n), d, n_resamples=4000)
    assert ci.high - ci.low == pytest.approx(2 * 1.96 / np.sqrt(n), rel=0.1)


def test_bootstrap_alpha_widens_interval() -> None:
    rng = np.random.default_rng(6)
    a, b = rng.random(100), rng.random(100)
    narrow = paired_bootstrap_ci(a, b, n_resamples=1000, alpha=0.5)
    wide = paired_bootstrap_ci(a, b, n_resamples=1000, alpha=0.01)
    assert wide.high - wide.low > narrow.high - narrow.low


@pytest.mark.parametrize(
    ("a", "b"),
    [(np.zeros(3), np.zeros(4)), (np.zeros(0), np.zeros(0))],
    ids=["different-length", "empty"],
)
def test_bootstrap_rejects_bad_shapes(a: np.ndarray, b: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"length|empty"):
        paired_bootstrap_ci(a, b, n_resamples=10)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_bootstrap_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        paired_bootstrap_ci(np.zeros(3), np.ones(3), n_resamples=10, alpha=alpha)


def test_bootstrap_defaults_are_10000_resamples_and_seed_0() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.random(40), rng.random(40)
    assert paired_bootstrap_ci(a, b) == paired_bootstrap_ci(a, b, n_resamples=10_000, seed=0)


def test_bootstrap_accepts_one_resample_and_rejects_zero() -> None:
    assert paired_bootstrap_ci(np.zeros(3), np.ones(3), n_resamples=1) == PairedCI(1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="n_resamples"):
        paired_bootstrap_ci(np.zeros(3), np.ones(3), n_resamples=0)


def test_bootstrap_can_draw_the_first_query() -> None:
    ci = paired_bootstrap_ci(np.zeros(2), np.array([1.0, 0.0]), n_resamples=400)
    assert (ci.low, ci.high) == (0.0, 1.0)


def test_bootstrap_uses_alpha_over_two_in_each_tail() -> None:
    # Resampled means are 0, 0.5, 1 with probability 1/4, 1/2, 1/4: the 0.3 and 0.7 quantiles are both 0.5.
    ci = paired_bootstrap_ci(np.zeros(2), np.array([1.0, 0.0]), n_resamples=4000, alpha=0.6)
    assert (ci.low, ci.high) == (0.5, 0.5)


def test_query_stats_hand_values() -> None:
    stats = query_stats([_record(("a", "a", "b")), _record(("c",))])
    assert stats["mean_n_terms"] == pytest.approx(2.0)
    assert stats["mean_n_unique_terms"] == pytest.approx(1.5)
    assert stats["repeat_ratio"] == pytest.approx(1 / 4)
    assert stats["any_repeat_rate"] == pytest.approx(0.5)
    assert stats["format_error_rate"] == pytest.approx(0.0)


def test_query_stats_format_errors_counted_but_not_in_term_means() -> None:
    stats = query_stats([_record(("a", "a")), _record((), format_error="tag_count")])
    assert stats["format_error_rate"] == pytest.approx(0.5)
    assert stats["mean_n_terms"] == pytest.approx(2.0)
    assert stats["repeat_ratio"] == pytest.approx(0.5)
    assert stats["any_repeat_rate"] == pytest.approx(1.0)


def test_query_stats_all_format_errors() -> None:
    stats = query_stats([_record((), format_error="bad_json")])
    assert stats == {
        "mean_n_terms": 0.0,
        "mean_n_unique_terms": 0.0,
        "repeat_ratio": 0.0,
        "any_repeat_rate": 0.0,
        "format_error_rate": 1.0,
    }


def test_query_stats_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no records"):
        query_stats([])
