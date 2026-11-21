"""BM25 search in two scoring modes over the same index, analyzer, k1, b and k.

normal:     Anserini's default bag-of-words query; a term repeated n times in the
            query gets boost n.
count_once: the same analyzed terms, de-duplicated, as a BooleanQuery of SHOULD
            TermQuery clauses with boost 1; repetition has no effect.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from drbm25.config import ScoringMode

logger = logging.getLogger(__name__)


def _import_pyserini_search() -> None:
    """Import pyserini.search.lucene without an OpenAI key.

    pyserini 2.0.0's search package imports pyserini.encode._openai, which builds an
    openai.OpenAI client at import time; the resolved openai 3.x raises when the key is
    empty. Nothing here calls OpenAI, so a placeholder is set only for the import and
    removed afterwards. A real key, if present, is left untouched.
    """
    key = "OPENAI_API_KEY"
    placeholder = key not in os.environ
    if placeholder:
        os.environ[key] = "unused-placeholder-for-pyserini-import"
    try:
        import pyserini.search.lucene  # noqa: F401
    finally:
        if placeholder:
            del os.environ[key]


_import_pyserini_search()

# jnius must be imported after pyserini, which puts the Anserini jars on the classpath
# before the JVM starts.
# Types for these come from typings/jnius (pyjnius ships no stubs).
from jnius import JavaException, detach  # noqa: E402
from pyserini.analysis import Analyzer  # noqa: E402
from pyserini.search.lucene import LuceneSearcher, querybuilder  # noqa: E402

FIELD = "contents"
_TOO_MANY_CLAUSES = "org.apache.lucene.search.IndexSearcher$TooManyClauses"
_MODES: tuple[str, ...] = get_args(ScoringMode)


class SearchError(RuntimeError):
    pass


class TooManyClausesError(SearchError):
    pass


@dataclass(frozen=True)
class SearchResult:
    doc_ids: tuple[str, ...]
    n_terms: int  # analyzed query terms, with repeats
    n_unique_terms: int
    scores: tuple[float, ...]  # BM25 score per doc id, descending

    def __post_init__(self) -> None:
        if len(self.doc_ids) != len(self.scores):
            raise AssertionError(f"{len(self.doc_ids)} doc ids but {len(self.scores)} scores")
        if not 0 <= self.n_unique_terms <= self.n_terms:
            raise AssertionError(f"n_unique_terms={self.n_unique_terms} outside [0, n_terms={self.n_terms}]")


class Bm25Searcher:
    def __init__(self, index_dir: Path, *, k1: float, b: float, threads: int, timeout_s: float) -> None:
        if not index_dir.is_dir():
            raise FileNotFoundError(f"index directory not found: {index_dir}")
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self._searcher = LuceneSearcher(str(index_dir))
        # Sets the similarity on the underlying IndexSearcher, so it applies to string
        # queries and to the BooleanQuery objects built for count_once alike.
        self._searcher.set_bm25(k1, b)
        # The searcher's own analyzer: the one its query generator uses in normal mode
        # (DefaultEnglishAnalyzer: Porter stemming, Lucene English stopwords).
        self._analyzer = Analyzer(self._searcher.object.get_analyzer())
        self._threads = threads
        self._timeout_s = timeout_s

    def analyze(self, text: str) -> list[str]:
        return [str(token) for token in self._analyzer.analyze(text)]

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        terms = self.analyze(query)
        unique_terms = list(dict.fromkeys(terms))
        if not unique_terms:
            return SearchResult((), 0, 0, ())
        try:
            # normal: Anserini re-analyzes the string with the same analyzer and boosts
            # each term by its count. count_once: TermQuery objects are built from the
            # analyzed terms directly so they are not stemmed a second time.
            lucene_query: Any = query if mode == "normal" else self._count_once_query(unique_terms)
            hits = self._searcher.search(lucene_query, k)
        except JavaException as exc:
            if exc.classname == _TOO_MANY_CLAUSES:
                raise TooManyClausesError(
                    f"{len(unique_terms)} unique query terms exceed Lucene's clause limit"
                ) from exc
            raise SearchError(f"Lucene search failed: {exc.classname}") from exc
        return SearchResult(
            doc_ids=tuple(str(hit.docid) for hit in hits),
            n_terms=len(terms),
            n_unique_terms=len(unique_terms),
            scores=tuple(float(hit.score) for hit in hits),
        )

    def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]:
        executor = ThreadPoolExecutor(max_workers=self._threads)
        try:
            futures: list[Future[SearchResult]] = [
                executor.submit(self._search_in_worker, query, mode, k) for query in queries
            ]
            try:
                return [future.result(timeout=self._timeout_s) for future in futures]
            except FutureTimeoutError as exc:
                raise SearchError(f"a search exceeded {self._timeout_s}s") from exc
        finally:
            # Do not wait: a timed-out Java call cannot be interrupted.
            executor.shutdown(wait=False, cancel_futures=True)

    def _search_in_worker(self, query: str, mode: ScoringMode, k: int) -> SearchResult:
        try:
            return self.search(query, mode=mode, k=k)
        finally:
            # pyjnius attaches each Python thread to the JVM; detach before the thread exits.
            detach()

    @staticmethod
    def _count_once_query(unique_terms: list[str]) -> Any:
        builder = querybuilder.get_boolean_query_builder()
        should = querybuilder.JBooleanClauseOccur.should.value
        for term in unique_terms:
            builder.add(querybuilder.JTermQuery(querybuilder.JTerm(FIELD, term)), should)
        return builder.build()
