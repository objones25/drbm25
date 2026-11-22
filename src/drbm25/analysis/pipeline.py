"""Metrics over the rewrite sampling database.

CLI: ``python -m drbm25.analysis.pipeline --db DIR --index-dir DIR --out DIR [--no-embed] [--embed-device DEV]``.

Writes to ``--out``:

* ``sample_metrics.parquet``: one row per sample (keys + per-sample metrics). Term metrics compare the rewrite's
  analyzed terms with the original query's; RBO and Jaccard@10 compare its normal-mode top 10 with the original
  query's. Format errors have no term metrics (null) and score 0.0.
* ``query_metrics.parquet``: one row per (source, query): nDCG mean and spread, format-error rate, distinct
  rewrites, Vendi diversity, expected best-of-k nDCG, mean pairwise RBO between samples, and the per-query mean
  of every numeric per-sample metric (``mean_<name>``).
* ``cross_policy.parquet``: per query, the normal and count-once policies' modal rewrites and the RBO and
  Jaccard@10 of their result lists.
* ``comparisons.json``: paired query-level comparisons of every per-query metric between the model sources
  (and against ``original`` where the metric exists for it). nDCG of trained-normal vs trained-count_once is the
  primary comparison; every other comparison gets a Benjamini-Yekutieli adjusted p-value ``q_by``.
* ``lexical.json``: Fightin' Words (top 30 each way), Jensen-Shannon divergence for rewrite 1-3 grams and
  reasoning unigrams per policy pair, and classifier two-sample tests (normal vs count-once, sample 0 of each
  query) on rewrites and on reasoning.
* ``variance.json``: between-/within-query variance of nDCG per model source.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from drbm25.analysis.corpus import (
    EMBED_MODEL,
    EMBED_REVISION,
    Encoder,
    IdfTable,
    embed_similarity,
    idf_summary,
    sentence_transformer,
)
from drbm25.analysis.diversity import distinct_fraction, expected_best_of_k, vendi_jaccard
from drbm25.analysis.lexical import c2st, fightin_words, js_divergence, scaled_prior
from drbm25.analysis.ranking import jaccard_at_k, rbo_ext
from drbm25.analysis.stats import fdr_by, paired_compare, variance_components
from drbm25.analysis.text import ngrams, reasoning_overlap, repetition, rewrite_class, term_diff
from drbm25.eval.records import SOURCE_ORIGINAL, SOURCE_TRAINED_COUNT_ONCE, SOURCE_TRAINED_NORMAL, SOURCE_UNTRAINED
from drbm25.eval.sample_db import SampleRow, read_db
from drbm25.obs import configure

logger = logging.getLogger(__name__)

MODEL_SOURCES: Final = (SOURCE_UNTRAINED, SOURCE_TRAINED_NORMAL, SOURCE_TRAINED_COUNT_ONCE)
PRIMARY_PAIR: Final = (SOURCE_TRAINED_NORMAL, SOURCE_TRAINED_COUNT_ONCE)
PRIMARY_METRICS: Final = ("ndcg_normal_mean", "ndcg_count_once_mean")
BEST_OF_K: Final = (1, 2, 4, 8, 16)
PAIRWISE_RBO_SAMPLES: Final = 8
RANK_DEPTH: Final = 10
FIGHTIN_TOP: Final = 30
PRIOR_ALPHA0: Final = 1000.0

Analyzer = Callable[[str], list[str]]


# ---------------------------------------------------------------- per sample


def _nan_to_none(value: float) -> float | None:
    return None if math.isnan(value) else value


def sample_metrics(
    rows: Sequence[SampleRow], original: dict[str, SampleRow], *, analyze: Analyzer, idf_table: IdfTable
) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        orig = original[r.qid]
        parsed = r.rewrite is not None and r.format_error is None
        m: dict[str, Any] = {
            "source": r.source,
            "qid": r.qid,
            "query_index": r.query_index,
            "sample_index": r.sample_index,
            "ndcg_normal": r.ndcg_normal,
            "ndcg_count_once": r.ndcg_count_once,
            "is_format_error": int(r.format_error is not None),
            "hit_max_tokens": None if r.hit_max_tokens is None else int(r.hit_max_tokens),
            "n_response_tokens": r.n_response_tokens,
            "mean_logprob": (
                r.sum_logprob / r.n_response_tokens if r.sum_logprob is not None and r.n_response_tokens else None
            ),
            "think_chars": None if r.think_text is None else len(r.think_text),
            "think_words": None if r.think_text is None else len(r.think_text.split()),
        }
        names = (
            "retention", "jaccard", "n_added", "n_dropped", "n_kept", "rewrite_class", "n_terms", "n_unique",
            "duplicate_rate", "max_multiplicity", "has_repeat", "rewrite_chars", "idf_mean", "idf_max",
            "added_idf_mean", "rbo_vs_original", "jaccard10_vs_original", "reasoning_overlap",
        )  # fmt: skip
        m.update(dict.fromkeys(names))
        if parsed:
            diff = term_diff(orig.terms, r.terms)
            rep = repetition(r.terms)
            idf_mean, idf_max = idf_summary(r.terms, idf_table)
            added = [t for t in dict.fromkeys(r.terms) if t not in set(orig.terms)]
            added_mean, _ = idf_summary(added, idf_table)
            m.update(
                retention=diff.retention,
                jaccard=diff.jaccard,
                n_added=diff.n_added,
                n_dropped=diff.n_dropped,
                n_kept=diff.n_kept,
                rewrite_class=rewrite_class(diff),
                n_terms=rep.n_terms,
                n_unique=rep.n_unique,
                duplicate_rate=rep.duplicate_rate,
                max_multiplicity=rep.max_multiplicity,
                has_repeat=int(rep.has_repeat),
                rewrite_chars=len(r.rewrite or ""),
                idf_mean=_nan_to_none(idf_mean),
                idf_max=_nan_to_none(idf_max),
                added_idf_mean=_nan_to_none(added_mean),
                rbo_vs_original=rbo_ext(r.doc_ids_normal[:RANK_DEPTH], orig.doc_ids_normal[:RANK_DEPTH]),
                jaccard10_vs_original=jaccard_at_k(r.doc_ids_normal, orig.doc_ids_normal, RANK_DEPTH),
            )
            if r.think_text is not None:
                m["reasoning_overlap"] = reasoning_overlap(analyze(r.think_text), r.terms)
        out.append(m)
    return out


# ---------------------------------------------------------------- per query

_NOT_AVERAGED: Final = frozenset({"source", "qid", "query_index", "sample_index", "rewrite_class"})


def _mean_or_none(values: Sequence[float | int | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return mean(present) if present else None


def _modal(rewrites: Sequence[str | None]) -> str | None:
    counts = Counter(rewrites)
    return max(counts, key=lambda k: (counts[k], k is not None, k or ""))


def query_metrics(rows: Sequence[SampleRow], per_sample: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.source, r.qid)].append(i)
    out = []
    for (source, qid), idx in sorted(groups.items()):
        rs = [rows[i] for i in idx]
        ms = [per_sample[i] for i in idx]
        normal = [r.ndcg_normal for r in rs]
        q: dict[str, Any] = {
            "source": source,
            "qid": qid,
            "n_samples": len(rs),
            "ndcg_normal_mean": mean(normal),
            "ndcg_normal_std": pstdev(normal),
            "ndcg_count_once_mean": mean(r.ndcg_count_once for r in rs),
            "ndcg_count_once_std": pstdev([r.ndcg_count_once for r in rs]),
            "distinct_fraction": distinct_fraction([r.rewrite for r in rs]),
            "vendi_jaccard": vendi_jaccard([r.terms for r in rs]),
        }
        for k in BEST_OF_K:
            q[f"best_of_{k}"] = expected_best_of_k(normal, k) if k <= len(normal) else None
        firsts = [r.doc_ids_normal[:RANK_DEPTH] for r in rs[:PAIRWISE_RBO_SAMPLES]]
        pairs = list(combinations(firsts, 2))
        q["pairwise_rbo"] = mean(rbo_ext(a, b) for a, b in pairs) if pairs else None
        for name in ms[0]:
            if name not in _NOT_AVERAGED:
                q[f"mean_{name}"] = _mean_or_none([m[name] for m in ms])
        out.append(q)
    return out


def cross_policy(rows: Sequence[SampleRow]) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, list[SampleRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.source in PRIMARY_PAIR:
            by_query[r.qid][r.source].append(r)
    out = []
    for qid, per_source in sorted(by_query.items()):
        if set(per_source) != set(PRIMARY_PAIR):
            continue
        modal = {}
        docs = {}
        for source in PRIMARY_PAIR:
            rewrite = _modal([r.rewrite for r in per_source[source]])
            modal[source] = rewrite
            docs[source] = next(r.doc_ids_normal[:RANK_DEPTH] for r in per_source[source] if r.rewrite == rewrite)
        a, b = PRIMARY_PAIR
        out.append(
            {
                "qid": qid,
                "modal_normal": modal[a],
                "modal_count_once": modal[b],
                "same_modal_rewrite": int(modal[a] == modal[b]),
                "rbo_modal": rbo_ext(docs[a], docs[b]),
                "jaccard10_modal": jaccard_at_k(docs[a], docs[b], RANK_DEPTH),
            }
        )
    return out


# ---------------------------------------------------------------- comparisons


def comparisons(per_query: Sequence[dict[str, Any]], *, n_resamples: int) -> list[dict[str, Any]]:
    table: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for q in per_query:
        for name, value in q.items():
            if name in ("source", "qid", "n_samples") or value is None:
                continue
            table[name][q["source"]][q["qid"]] = float(value)
    pairs = [*combinations(MODEL_SOURCES, 2), *((SOURCE_ORIGINAL, s) for s in MODEL_SOURCES)]
    results = []
    for metric in sorted(table):
        for a, b in pairs:
            if not table[metric].get(a) or not table[metric].get(b):
                continue
            if a == SOURCE_ORIGINAL and metric.endswith("_std"):
                continue  # one sample per query: no spread
            try:
                cmp = paired_compare(
                    table[metric][a], table[metric][b], metric=metric, a=a, b=b, n_resamples=n_resamples
                )
            except ValueError:
                continue  # no shared non-NaN queries
            primary = (a, b) == PRIMARY_PAIR and metric in PRIMARY_METRICS
            results.append({**asdict(cmp), "primary": primary, "q_by": None})
    secondary = [r for r in results if not r["primary"]]
    if secondary:
        for r, q in zip(secondary, fdr_by([r["wilcoxon_p"] for r in secondary]), strict=True):
            r["q_by"] = q
    return results


# ---------------------------------------------------------------- lexical


def _counts(texts: Sequence[Sequence[str]], n: int) -> Counter[str]:
    return Counter(" ".join(g) for t in texts for g in ngrams(t, n))


def lexical(rows: Sequence[SampleRow], *, analyze: Analyzer, c2st_permutations: int) -> dict[str, Any]:
    terms: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    thinks: dict[str, list[list[str]]] = defaultdict(list)
    for r in rows:
        if r.source in MODEL_SOURCES and r.rewrite is not None and r.format_error is None:
            terms[r.source].append(r.terms)
            if r.think_text is not None:
                thinks[r.source].append(analyze(r.think_text))
    corpora = {
        **{f"rewrite_{n}": {s: _counts(terms[s], n) for s in MODEL_SOURCES} for n in (1, 2, 3)},
        "think_1": {s: _counts(thinks[s], 1) for s in MODEL_SOURCES},
    }
    pairs: dict[str, Any] = {}
    for a, b in combinations(MODEL_SOURCES, 2):
        fw: dict[str, Any] = {}
        js: dict[str, float | None] = {}
        for name, counts in corpora.items():
            pooled = sum((counts[s] for s in MODEL_SOURCES), Counter())
            if not counts[a] or not counts[b] or not pooled:
                fw[name] = {"top_a": [], "top_b": [], "note": "no n-grams of this order in one of the policies"}
                js[name] = None
                continue
            z = fightin_words(counts[a], counts[b], scaled_prior(pooled, alpha0=PRIOR_ALPHA0))
            ranked = sorted(z.items(), key=lambda kv: kv[1])
            fw[name] = {"top_a": ranked[::-1][:FIGHTIN_TOP], "top_b": ranked[:FIGHTIN_TOP]}
            js[name] = js_divergence(counts[a], counts[b])
        pairs[f"{a}|{b}"] = {"fightin_words": fw, "js_divergence": js}
    first = {
        s: [
            r
            for r in rows
            if r.source == s and r.sample_index == 0 and r.rewrite is not None and r.format_error is None
        ]
        for s in PRIMARY_PAIR
    }
    a, b = PRIMARY_PAIR
    tests: dict[str, Any] = {}
    for kind, text in (("rewrite", lambda r: r.rewrite or ""), ("think", lambda r: r.think_text or "")):
        res = c2st(
            [text(r) for r in first[a]],
            [text(r) for r in first[b]],
            [r.qid for r in first[a]],
            [r.qid for r in first[b]],
            n_permutations=c2st_permutations,
        )
        tests[kind] = asdict(res)
    return {"pairs": pairs, "c2st": tests, "fightin_prior_alpha0": PRIOR_ALPHA0}


# ---------------------------------------------------------------- driver


def _write_parquet(path: Path, records: Sequence[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(list(records)), path)


def run_pipeline(
    db: Path,
    out: Path,
    *,
    analyze: Analyzer,
    idf_table: IdfTable,
    encode: Encoder | None,
    n_resamples: int = 10_000,
    c2st_permutations: int = 200,
    embed: bool = False,
) -> None:
    """Compute every output file from the database at ``db`` into ``out``. Embedding similarity is added when
    ``embed`` is set or an ``encode`` function is given."""
    rows = read_db(db)
    original = {r.qid: r for r in rows if r.source == SOURCE_ORIGINAL}
    missing = sorted({r.qid for r in rows} - set(original))
    if not original or missing:
        raise ValueError(f"the database needs original rows for every query; missing {len(missing) or 'all'}")
    out.mkdir(parents=True, exist_ok=True)
    logger.info("computing per-sample metrics for %d rows", len(rows))
    per_sample = sample_metrics(rows, original, analyze=analyze, idf_table=idf_table)
    if embed or encode is not None:
        sims = embed_similarity(
            [r.query_text for r in rows], [r.rewrite if r.format_error is None else None for r in rows], encode=encode
        )
        for m, s in zip(per_sample, sims, strict=True):
            m["embed_sim"] = s
    _write_parquet(out / "sample_metrics.parquet", per_sample)
    per_query = query_metrics(rows, per_sample)
    _write_parquet(out / "query_metrics.parquet", per_query)
    _write_parquet(out / "cross_policy.parquet", cross_policy(rows))
    logger.info("comparing policies on %d query rows", len(per_query))
    (out / "comparisons.json").write_text(json.dumps(comparisons(per_query, n_resamples=n_resamples), indent=2) + "\n")
    (out / "lexical.json").write_text(
        json.dumps(lexical(rows, analyze=analyze, c2st_permutations=c2st_permutations), indent=2) + "\n"
    )
    variance = {}
    for source in MODEL_SOURCES:
        per = defaultdict(list)
        for r in rows:
            if r.source == source:
                per[r.qid].append(r.ndcg_normal)
        if per and all(len(v) >= 2 for v in per.values()):
            tau2, sigma2, icc = variance_components(per)
            variance[source] = {"tau2": tau2, "sigma2": sigma2, "icc": icc, "k": mean(len(v) for v in per.values())}
    (out / "variance.json").write_text(json.dumps(variance, indent=2) + "\n")
    logger.info("wrote metrics to %s", out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute behavior metrics over the rewrite sampling database.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-embed", action="store_true", help=f"skip embedding similarity ({EMBED_MODEL})")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--c2st-permutations", type=int, default=200)
    ns = parser.parse_args(argv)
    configure(level="INFO", fmt="json")
    from drbm25.bm25.searcher import Bm25Searcher  # importing pyserini starts the JVM

    searcher = Bm25Searcher(ns.index_dir, k1=0.9, b=0.4, threads=1, timeout_s=60.0)  # only its analyzer is used
    encode = None
    if not ns.no_embed:
        encode = sentence_transformer(EMBED_MODEL, EMBED_REVISION, 512, ns.embed_device)
    run_pipeline(
        ns.db,
        ns.out,
        analyze=searcher.analyze,
        idf_table=IdfTable.from_index(ns.index_dir),
        encode=encode,
        n_resamples=ns.n_resamples,
        c2st_permutations=ns.c2st_permutations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
