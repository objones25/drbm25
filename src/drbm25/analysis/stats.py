"""Query-level statistics: the query is the unit of analysis, never the individual sample.

Each metric is first averaged over a query's samples; policies are then compared on those per-query values
with the paired query-resampling bootstrap the training report uses (``drbm25.eval.stats``) and a Wilcoxon
signed-rank test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import false_discovery_control, wilcoxon

from drbm25.eval.stats import paired_bootstrap_ci


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    a: str
    b: str
    mean_a: float
    mean_b: float
    mean_diff: float  # mean over queries of b - a
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    n_queries: int
    n_dropped_nan: int  # shared queries skipped because either value was NaN


def paired_compare(
    per_query_a: Mapping[str, float],
    per_query_b: Mapping[str, float],
    *,
    metric: str,
    a: str,
    b: str,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedComparison:
    """Compare per-query values of policy ``a`` and ``b`` over the queries both have (NaNs dropped)."""
    shared = sorted(set(per_query_a) & set(per_query_b))
    keep = [q for q in shared if not (math.isnan(per_query_a[q]) or math.isnan(per_query_b[q]))]
    if not keep:
        raise ValueError(f"no queries with values for both {a} and {b} ({metric})")
    xa = np.array([per_query_a[q] for q in keep], dtype=np.float64)
    xb = np.array([per_query_b[q] for q in keep], dtype=np.float64)
    ci = paired_bootstrap_ci(xa, xb, n_resamples=n_resamples, seed=seed)
    diff = xb - xa
    p_value = 1.0 if np.all(diff == 0) else float(wilcoxon(xa, xb, zero_method="wilcox").pvalue)  # pyright: ignore[reportAttributeAccessIssue] -- scipy result object is untyped
    return PairedComparison(
        metric=metric,
        a=a,
        b=b,
        mean_a=float(xa.mean()),
        mean_b=float(xb.mean()),
        mean_diff=ci.mean_diff,
        ci_low=ci.low,
        ci_high=ci.high,
        wilcoxon_p=p_value,
        n_queries=len(keep),
        n_dropped_nan=len(shared) - len(keep),
    )


def variance_components(per_sample: Mapping[str, Sequence[float]]) -> tuple[float, float, float]:
    """Between-query variance tau2, within-query (sampling) variance sigma2 and the ICC tau2 / (tau2 + sigma2),
    following Miller (2024, "Adding Error Bars to Evals", sec. 5): sigma2 is the mean per-query sample variance
    and tau2 = Var(query means) - sigma2 / K, with K the mean number of samples per query."""
    groups = [np.asarray(v, dtype=np.float64) for v in per_sample.values()]
    if not groups or any(g.size < 2 for g in groups):
        raise ValueError("every query needs at least 2 samples")
    sigma2 = float(np.mean([g.var(ddof=1) for g in groups]))
    k = float(np.mean([g.size for g in groups]))
    tau2 = float(np.var([g.mean() for g in groups], ddof=1) - sigma2 / k)
    total = tau2 + sigma2
    return tau2, sigma2, (tau2 / total if total > 0 else float("nan"))


def fdr_by(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Yekutieli adjusted p-values (valid under any dependence between the tests)."""
    return [float(q) for q in false_discovery_control(list(p_values), method="by")]
