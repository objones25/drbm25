"""Dollar budget to wall-clock deadline, and the stop rule for chunked training."""

from __future__ import annotations

_SECONDS_PER_HOUR = 3600.0


def deadline_from_budget(*, budget_usd: float, price_per_hour: float, start_unix: float) -> float:
    """Unix time at which `budget_usd` is spent at `price_per_hour` since `start_unix`."""
    if budget_usd <= 0:
        raise ValueError(f"budget_usd must be positive, got {budget_usd}")
    if price_per_hour <= 0:
        raise ValueError(f"price_per_hour must be positive, got {price_per_hour}")
    return start_unix + budget_usd / price_per_hour * _SECONDS_PER_HOUR


def should_stop(*, now_unix: float, deadline_unix: float, mean_chunk_s: float, reserve_s: float) -> bool:
    """True when one more chunk would end later than `reserve_s` before the deadline."""
    if reserve_s < 0:
        raise ValueError(f"reserve_s must be non-negative, got {reserve_s}")
    return now_unix + mean_chunk_s > deadline_unix - reserve_s
