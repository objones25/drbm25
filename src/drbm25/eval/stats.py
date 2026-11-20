"""Paired bootstrap confidence intervals and per-source query statistics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from drbm25.eval.records import QueryRecord


@dataclass(frozen=True)
class PairedCI:
    mean_diff: float  # mean of b - a over the queries
    low: float
    high: float


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, *, n_resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> PairedCI:
    """Percentile CI of mean(b - a): resample query indices with replacement, ``n_resamples`` times."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"a and b must be 1-D arrays of the same length, got shapes {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("a and b are empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    diff = b.astype(np.float64) - a.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.fromiter(
        (diff[rng.integers(0, diff.size, size=diff.size)].mean() for _ in range(n_resamples)),
        dtype=np.float64,
        count=n_resamples,
    )
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return PairedCI(mean_diff=float(diff.mean()), low=float(low), high=float(high))


def query_stats(records: Sequence[QueryRecord]) -> dict[str, float]:
    """How much the scored queries repeat terms.

    Term statistics cover the records without a format error (those have no terms);
    ``format_error_rate`` covers all records. ``repeat_ratio`` is repeated term occurrences
    over all term occurrences; ``any_repeat_rate`` is the share of queries with a repeated term.
    """
    if not records:
        raise ValueError("no records")
    valid = [r for r in records if r.format_error is None]
    total_terms = sum(len(r.terms) for r in valid)
    total_unique = sum(len(set(r.terms)) for r in valid)
    return {
        "mean_n_terms": total_terms / len(valid) if valid else 0.0,
        "mean_n_unique_terms": total_unique / len(valid) if valid else 0.0,
        "repeat_ratio": (total_terms - total_unique) / total_terms if total_terms else 0.0,
        "any_repeat_rate": sum(len(set(r.terms)) < len(r.terms) for r in valid) / len(valid) if valid else 0.0,
        "format_error_rate": (len(records) - len(valid)) / len(records),
    }
