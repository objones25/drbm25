"""Policy-level vocabulary comparison: Fightin' Words, Jensen-Shannon divergence, classifier two-sample test."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, permutation_test_score


def scaled_prior(pooled_counts: Mapping[str, int], *, alpha0: float) -> dict[str, float]:
    """An informative Dirichlet prior: pooled counts rescaled to sum to ``alpha0``."""
    total = sum(pooled_counts.values())
    if total <= 0 or alpha0 <= 0:
        raise ValueError("pooled counts and alpha0 must be positive")
    return {w: alpha0 * c / total for w, c in pooled_counts.items()}


def fightin_words(
    counts_a: Mapping[str, int], counts_b: Mapping[str, int], prior: Mapping[str, float]
) -> dict[str, float]:
    """Z-scores of the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi and Quinn, Political
    Analysis 2008), per word of ``prior``; positive means A uses the word more. As in their reference
    implementation (jmhessel/fightingwords), with prior a_w summing to a0:
    ``delta = log((y_a+a_w)/(n_a+a0-y_a-a_w)) - log((y_b+a_w)/(n_b+a0-y_b-a_w))``,
    ``var = 1/(y_a+a_w) + 1/(y_b+a_w)``, ``z = delta / sqrt(var)``."""
    alpha0 = sum(prior.values())
    n_a, n_b = sum(counts_a.values()), sum(counts_b.values())
    scores = {}
    for word, alpha in prior.items():
        if alpha <= 0:
            raise ValueError(f"prior for {word!r} must be positive, got {alpha}")
        y_a, y_b = counts_a.get(word, 0), counts_b.get(word, 0)
        delta = math.log((y_a + alpha) / (n_a + alpha0 - y_a - alpha)) - math.log(
            (y_b + alpha) / (n_b + alpha0 - y_b - alpha)
        )
        scores[word] = delta / math.sqrt(1.0 / (y_a + alpha) + 1.0 / (y_b + alpha))
    return scores


def js_divergence(counts_a: Mapping[str, int], counts_b: Mapping[str, int]) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) of two count distributions, add-0.5 smoothed over their
    union vocabulary."""
    vocab = sorted(set(counts_a) | set(counts_b))
    if not vocab:
        raise ValueError("no words")
    p = np.array([counts_a.get(w, 0) + 0.5 for w in vocab], dtype=np.float64)
    q = np.array([counts_b.get(w, 0) + 0.5 for w in vocab], dtype=np.float64)
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)


@dataclass(frozen=True)
class C2STResult:
    accuracy: float  # mean grouped cross-validated accuracy
    p_value: float  # label permutation within query groups
    top_features_a: list[str]
    top_features_b: list[str]


def c2st(
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    groups_a: Sequence[str],
    groups_b: Sequence[str],
    *,
    n_permutations: int = 200,
    n_top: int = 20,
    seed: int = 0,
) -> C2STResult:
    """Classifier two-sample test (Lopez-Paz and Oquab 2017): can TF-IDF (word 1-2 grams) + logistic
    regression tell A's texts from B's? Folds never split a query (GroupKFold by query id) and the permutation
    test shuffles labels within each query, so the test is paired by query."""
    if len(texts_a) != len(groups_a) or len(texts_b) != len(groups_b):
        raise ValueError("each text needs exactly one query id in groups")
    texts = [*texts_a, *texts_b]
    labels = np.array([1] * len(texts_a) + [0] * len(texts_b))
    groups = np.array([*groups_a, *groups_b])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")  # noqa: S106 -- a regex, not a password
    features = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    n_splits = min(5, len(set(groups)))
    accuracy, _, p_value = permutation_test_score(
        model,
        features,
        labels,
        groups=groups,
        cv=GroupKFold(n_splits=n_splits),
        n_permutations=n_permutations,
        random_state=seed,
        scoring="accuracy",
    )
    model.fit(features, labels)
    names = vectorizer.get_feature_names_out()
    order = np.argsort(model.coef_[0])
    return C2STResult(
        accuracy=float(accuracy),
        p_value=float(p_value),
        top_features_a=[str(names[i]) for i in order[::-1][:n_top]],
        top_features_b=[str(names[i]) for i in order[:n_top]],
    )
