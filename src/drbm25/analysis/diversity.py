"""Within-query diversity and best-of-k over a query's N samples."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from drbm25.analysis.text import set_jaccard


def distinct_fraction(rewrites: Sequence[str | None]) -> float:
    """Distinct rewrites over samples; every format error (None) counts as the same value."""
    if not rewrites:
        raise ValueError("no rewrites")
    return len(set(rewrites)) / len(rewrites)


def vendi_jaccard(samples: Sequence[Sequence[str]]) -> float:
    """Vendi score (Friedman and Dieng, TMLR 2023) with the term-set Jaccard similarity as kernel:
    ``exp(-sum λ log λ)`` over the eigenvalues λ of K / n, the effective number of distinct samples."""
    n = len(samples)
    if n == 0:
        raise ValueError("no samples")
    sets = [frozenset(s) for s in samples]
    kernel = np.array([[set_jaccard(x, y) for y in sets] for x in sets], dtype=np.float64)
    eig = np.linalg.eigvalsh(kernel / n)
    eig = eig[eig > 1e-12]
    return float(math.exp(-np.sum(eig * np.log(eig))))


def expected_best_of_k(scores: Sequence[float], k: int) -> float:
    """Expected maximum of k scores drawn without replacement from ``scores``:
    ``sum_{j=k..n} s_(j) * C(j-1, k-1) / C(n, k)`` over the ascending order statistics s_(j)."""
    n = len(scores)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    ordered = sorted(scores)
    total = math.comb(n, k)
    return sum(ordered[j - 1] * math.comb(j - 1, k - 1) for j in range(k, n + 1)) / total
