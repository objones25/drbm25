from __future__ import annotations

import pytest

from drbm25.budget import deadline_from_budget, should_stop


def test_deadline_from_budget() -> None:
    d = deadline_from_budget(budget_usd=100.0, price_per_hour=2.78, start_unix=0.0)
    assert d == pytest.approx(129496.4, abs=0.1)


def test_deadline_adds_to_start() -> None:
    assert deadline_from_budget(budget_usd=1.0, price_per_hour=1.0, start_unix=1000.0) == pytest.approx(4600.0)


def test_should_stop_when_next_chunk_would_cross_reserve() -> None:
    assert should_stop(now_unix=0, deadline_unix=1000, mean_chunk_s=600, reserve_s=500) is True
    assert should_stop(now_unix=0, deadline_unix=1000, mean_chunk_s=400, reserve_s=500) is False


def test_should_stop_boundary_is_not_stop() -> None:
    assert should_stop(now_unix=0, deadline_unix=1000, mean_chunk_s=500, reserve_s=500) is False


@pytest.mark.parametrize(
    ("budget", "price"), [(100.0, 0.0), (100.0, -1.0), (0.0, 2.0), (-5.0, 2.0)]
)
def test_budget_rejects_nonpositive_inputs(budget: float, price: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        deadline_from_budget(budget_usd=budget, price_per_hour=price, start_unix=0.0)


def test_should_stop_rejects_negative_reserve() -> None:
    with pytest.raises(ValueError, match="reserve_s"):
        should_stop(now_unix=0, deadline_unix=1, mean_chunk_s=1, reserve_s=-1)
