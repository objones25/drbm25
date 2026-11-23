import numpy as np
import pytest
from scipy.stats import false_discovery_control

from drbm25.analysis.stats import fdr_by, paired_compare, variance_components


def _per_query(values: np.ndarray) -> dict[str, float]:
    return {f"q{i}": float(v) for i, v in enumerate(values)}


def test_paired_compare_identical_inputs_has_zero_difference() -> None:
    rng = np.random.default_rng(0)
    a = _per_query(rng.random(200))

    cmp = paired_compare(a, dict(a), metric="ndcg", a="x", b="y", n_resamples=500)

    assert cmp.mean_diff == pytest.approx(0.0)
    assert cmp.ci_low <= 0.0 <= cmp.ci_high
    assert cmp.n_queries == 200


def test_paired_compare_recovers_a_constant_shift() -> None:
    rng = np.random.default_rng(1)
    base = rng.random(300)
    noise = rng.normal(0, 0.01, 300)

    cmp = paired_compare(_per_query(base), _per_query(base + 0.1 + noise), metric="m", a="x", b="y", n_resamples=1000)

    assert cmp.mean_diff == pytest.approx(0.1, abs=0.005)
    assert cmp.ci_low > 0.09
    assert cmp.ci_high < 0.11
    assert cmp.wilcoxon_p < 1e-10
    assert (cmp.mean_a, cmp.mean_b) == (pytest.approx(base.mean()), pytest.approx((base + 0.1 + noise).mean()))


def test_paired_compare_inner_joins_and_drops_nan() -> None:
    a = {"q1": 0.1, "q2": 0.2, "q3": float("nan"), "q4": 0.4}
    b = {"q1": 0.2, "q2": 0.4, "q3": 0.3, "q9": 1.0}

    cmp = paired_compare(a, b, metric="m", a="x", b="y", n_resamples=200)

    assert cmp.n_queries == 2
    assert cmp.n_dropped_nan == 1
    assert cmp.mean_diff == pytest.approx(0.15)


def test_paired_compare_without_shared_queries_fails() -> None:
    with pytest.raises(ValueError, match="no queries"):
        paired_compare({"q1": 1.0}, {"q2": 1.0}, metric="m", a="x", b="y")


def test_variance_components_recovers_known_values() -> None:
    rng = np.random.default_rng(2)
    means = rng.normal(0.0, 1.0, 2000)  # tau2 = 1
    per_sample = {f"q{i}": list(m + rng.normal(0.0, 2.0, 32)) for i, m in enumerate(means)}  # sigma2 = 4

    tau2, sigma2, icc = variance_components(per_sample)

    assert tau2 == pytest.approx(1.0, rel=0.1)
    assert sigma2 == pytest.approx(4.0, rel=0.1)
    assert icc == pytest.approx(0.2, rel=0.15)


def test_variance_components_needs_two_samples_per_query() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        variance_components({"q1": [0.1], "q2": [0.2, 0.3]})


def test_fdr_by_matches_scipy() -> None:
    ps = [0.001, 0.01, 0.02, 0.04, 0.3, 0.8]

    assert fdr_by(ps) == pytest.approx(list(false_discovery_control(ps, method="by")))
