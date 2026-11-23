"""Similarity of two ranked document lists."""

from __future__ import annotations

from collections.abc import Sequence

from drbm25.analysis.text import set_jaccard


def rbo_ext(a: Sequence[str], b: Sequence[str], *, p: float = 0.9) -> float:
    """Extrapolated rank-biased overlap of two truncated rankings (Webber, Moffat and Zobel, ACM TOIS 2010,
    eq. 32, which covers rankings of different lengths, e.g. a query that matched fewer than k documents).

    With S the shorter ranking (length s), L the longer (length l) and ``X_d = |S[:min(d, s)] & L[:d]|``:
    ``(1-p)/p * (sum_{d=1..l} X_d/d p^d + sum_{d=s+1..l} X_s (d-s)/(s d) p^d) + ((X_l - X_s)/l + X_s/s) p^l``.
    Two empty rankings are identical (1.0); one empty ranking shares nothing (0.0).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    n_short, n_long = len(short), len(long_)
    if n_long == 0:
        return 1.0
    if n_short == 0:
        return 0.0
    overlaps = []  # X_d for d = 1..n_long
    seen_short: set[str] = set()
    seen_long: set[str] = set()
    for d in range(1, n_long + 1):
        if d <= n_short:
            seen_short.add(short[d - 1])
        seen_long.add(long_[d - 1])
        overlaps.append(len(seen_short & seen_long))
    x_s, x_l = overlaps[n_short - 1], overlaps[n_long - 1]
    head = sum(overlaps[d - 1] / d * p**d for d in range(1, n_long + 1))
    tail = sum(x_s * (d - n_short) / (n_short * d) * p**d for d in range(n_short + 1, n_long + 1))
    return (1 - p) / p * (head + tail) + ((x_l - x_s) / n_long + x_s / n_short) * p**n_long


def jaccard_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    return set_jaccard(set(a[:k]), set(b[:k]))
