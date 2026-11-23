"""The rewrite sampling database: one row per sampled response, stored as Parquet shards.

Layout: ``<root>/<source>/shard-<k>.parquet``, each next to a ``shard-<k>.parquet._SUCCESS`` marker written
only after the shard is complete, so a crash mid-write leaves a shard that ``read_db`` ignores and the sampler
rewrites.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

_SUCCESS_SUFFIX: Final = "._SUCCESS"


@dataclass(frozen=True)
class SampleRow:
    """One sampled response, parsed and scored under both BM25 modes.

    ``original`` rows hold the raw query (``rewrite`` = the query, response fields None). A format error keeps
    the raw response, sets ``format_error``, has no rewrite or terms and scores 0.0. ``search_error`` names the
    modes whose search overflowed Lucene's clause limit (empty doc lists, score 0.0).
    """

    source: str
    qid: str
    query_index: int
    sample_index: int
    query_text: str
    response_text: str | None
    think_text: str | None
    rewrite: str | None
    format_error: str | None
    terms: tuple[str, ...]
    n_response_tokens: int | None
    hit_max_tokens: bool | None
    sum_logprob: float | None
    doc_ids_normal: tuple[str, ...]
    scores_normal: tuple[float, ...]
    doc_ids_count_once: tuple[str, ...]
    scores_count_once: tuple[float, ...]
    ndcg_normal: float
    ndcg_count_once: float
    search_error: str | None
    model_id: str | None
    model_revision: str | None
    seed: int | None


SCHEMA: Final = pa.schema(
    [
        ("source", pa.string()),
        ("qid", pa.string()),
        ("query_index", pa.int64()),
        ("sample_index", pa.int64()),
        ("query_text", pa.string()),
        ("response_text", pa.string()),
        ("think_text", pa.string()),
        ("rewrite", pa.string()),
        ("format_error", pa.string()),
        ("terms", pa.list_(pa.string())),
        ("n_response_tokens", pa.int64()),
        ("hit_max_tokens", pa.bool_()),
        ("sum_logprob", pa.float64()),
        ("doc_ids_normal", pa.list_(pa.string())),
        ("scores_normal", pa.list_(pa.float64())),
        ("doc_ids_count_once", pa.list_(pa.string())),
        ("scores_count_once", pa.list_(pa.float64())),
        ("ndcg_normal", pa.float64()),
        ("ndcg_count_once", pa.float64()),
        ("search_error", pa.string()),
        ("model_id", pa.string()),
        ("model_revision", pa.string()),
        ("seed", pa.int64()),
    ]
)
_TUPLE_FIELDS: Final = frozenset(
    {"terms", "doc_ids_normal", "scores_normal", "doc_ids_count_once", "scores_count_once"}
)
if [f.name for f in dataclasses.fields(SampleRow)] != SCHEMA.names:
    raise AssertionError("SampleRow fields and the Parquet SCHEMA disagree")


def _marker(path: Path) -> Path:
    return path.with_name(path.name + _SUCCESS_SUFFIX)


def write_shard(path: Path, rows: Sequence[SampleRow]) -> None:
    """Write ``rows`` to ``path`` atomically, then its success marker."""
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _marker(path).unlink(missing_ok=True)
    columns = {name: [getattr(r, name) for r in rows] for name in SCHEMA.names}
    table = pa.Table.from_pydict(
        {k: [list(v) for v in vals] if k in _TUPLE_FIELDS else vals for k, vals in columns.items()}, schema=SCHEMA
    )
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    _marker(path).write_text("ok\n", encoding="utf-8")


def shard_complete(path: Path) -> bool:
    return path.is_file() and _marker(path).is_file()


def read_db(root: Path) -> list[SampleRow]:
    """Every row of every complete shard under ``root``, sorted by source, query and sample."""
    rows: list[SampleRow] = []
    for path in sorted(root.glob("*/shard-*.parquet")):
        if not shard_complete(path):
            continue
        for record in pq.read_table(path, schema=SCHEMA).to_pylist():
            values: dict[str, Any] = {k: tuple(v) if k in _TUPLE_FIELDS else v for k, v in record.items()}
            rows.append(SampleRow(**values))
    rows.sort(key=lambda r: (r.source, r.query_index, r.sample_index))
    return rows
