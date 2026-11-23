"""Per-rewrite text metrics on analyzed BM25 terms.

Set metrics share one rule for empty inputs: two empty sets are identical (1.0), exactly one empty set shares
nothing with the other (0.0).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

RewriteClass = Literal["same", "expansion", "compression", "substitution", "mixed"]


@dataclass(frozen=True)
class TermDiff:
    """Term-set comparison of a rewrite against the original query."""

    retention: float  # share of the original's unique terms kept in the rewrite
    jaccard: float
    n_added: int
    n_dropped: int
    n_kept: int


@dataclass(frozen=True)
class Repetition:
    n_terms: int
    n_unique: int
    duplicate_rate: float  # 1 - unique/total; 0.0 for no terms
    max_multiplicity: int
    has_repeat: bool


def set_jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def term_diff(original: Sequence[str], rewrite: Sequence[str]) -> TermDiff:
    orig, new = set(original), set(rewrite)
    kept = orig & new
    return TermDiff(
        retention=1.0 if not orig else len(kept) / len(orig),
        jaccard=set_jaccard(orig, new),
        n_added=len(new - orig),
        n_dropped=len(orig - new),
        n_kept=len(kept),
    )


def repetition(terms: Sequence[str]) -> Repetition:
    counts = Counter(terms)
    total, unique = len(terms), len(counts)
    top = max(counts.values(), default=0)
    return Repetition(
        n_terms=total,
        n_unique=unique,
        duplicate_rate=0.0 if total == 0 else 1.0 - unique / total,
        max_multiplicity=top,
        has_repeat=top > 1,
    )


def rewrite_class(diff: TermDiff) -> RewriteClass:
    """same: no term added or dropped; expansion: only added; compression: only dropped; substitution: as many
    added as dropped; mixed: both, in different numbers."""
    if diff.n_added == 0 and diff.n_dropped == 0:
        return "same"
    if diff.n_dropped == 0:
        return "expansion"
    if diff.n_added == 0:
        return "compression"
    return "substitution" if diff.n_added == diff.n_dropped else "mixed"


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(samples: Sequence[Sequence[str]], n: int) -> float:
    """Unique n-grams over all n-grams pooled across ``samples``; 0.0 when there are none."""
    grams = [g for sample in samples for g in ngrams(sample, n)]
    return 0.0 if not grams else len(set(grams)) / len(grams)


def reasoning_overlap(think_terms: Sequence[str], rewrite_terms: Sequence[str]) -> float:
    """Share of the rewrite's unique terms that also appear in the reasoning; 1.0 for an empty rewrite."""
    rewrite = set(rewrite_terms)
    return 1.0 if not rewrite else len(rewrite & set(think_terms)) / len(rewrite)
