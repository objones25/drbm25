"""Collection statistics from the Lucene index (IDF) and semantic similarity between original and rewrite."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

EMBED_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION: Final = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pinned Hub commit (2026-11-22)

Encoder = Callable[[list[str]], np.ndarray]


@dataclass
class IdfTable:
    """``idf(t) = log(N / df(t))`` over the index; a term absent from the index counts as df = 1."""

    n_docs: int
    df: dict[str, int] = field(default_factory=dict)
    reader: Any = None  # pyserini LuceneIndexReader, for terms not yet in ``df``
    n_unknown: int = 0

    @classmethod
    def from_index(cls, index_dir: Path) -> IdfTable:
        from pyserini.index.lucene import LuceneIndexReader

        reader = LuceneIndexReader(str(index_dir))
        return cls(n_docs=int(reader.stats()["documents"]), reader=reader)

    def idf(self, term: str) -> float:
        if term not in self.df:
            if self.reader is None:
                raise KeyError(f"term {term!r} not in the table and no index reader to look it up")
            df, _ = self.reader.get_term_counts(term, analyzer=None)  # term is already analyzed
            if df == 0:
                self.n_unknown += 1
            self.df[term] = int(df)
        return math.log(self.n_docs / max(self.df[term], 1))


def idf_summary(terms: Sequence[str], table: IdfTable) -> tuple[float, float]:
    """Mean and max IDF of ``terms`` (with repeats); NaN for no terms."""
    if not terms:
        return math.nan, math.nan
    values = [table.idf(t) for t in terms]
    return sum(values) / len(values), max(values)


def sentence_transformer(model_name: str, revision: str, batch_size: int, device: str | None) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True))

    return encode


def embed_similarity(
    originals: Sequence[str],
    rewrites: Sequence[str | None],
    *,
    encode: Encoder | None = None,
    model_name: str = EMBED_MODEL,
    revision: str = EMBED_REVISION,
    batch_size: int = 512,
    device: str | None = None,
) -> list[float | None]:
    """Cosine similarity of each original query and its rewrite (None for a missing rewrite). Every distinct
    string is encoded once."""
    if len(originals) != len(rewrites):
        raise ValueError(f"{len(originals)} originals but {len(rewrites)} rewrites")
    encoder = encode or sentence_transformer(model_name, revision, batch_size, device)
    unique = sorted({*originals, *(r for r in rewrites if r is not None)})
    vectors = np.asarray(encoder(unique), dtype=np.float64)
    if vectors.shape[0] != len(unique):
        raise RuntimeError(f"encoder returned {vectors.shape[0]} vectors for {len(unique)} texts")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1.0, norms)
    index = {text: i for i, text in enumerate(unique)}
    return [
        None if r is None else float(vectors[index[o]] @ vectors[index[r]])
        for o, r in zip(originals, rewrites, strict=True)
    ]
