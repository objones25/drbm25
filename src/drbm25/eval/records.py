"""One scored query, and its JSON-lines file format (one file per source).

A file may start with one header line, ``{"header": {...}}``: where the scores came from (the
checkpoint, and for a ``best/`` policy its step, validation scores, config sha256 and seeds).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

SOURCE_ORIGINAL: Final = "original"
SOURCE_UNTRAINED: Final = "untrained"
SOURCE_TRAINED_NORMAL: Final = "trained-normal"
SOURCE_TRAINED_COUNT_ONCE: Final = "trained-count_once"
# Report row order.
SOURCES: Final = (SOURCE_ORIGINAL, SOURCE_UNTRAINED, SOURCE_TRAINED_NORMAL, SOURCE_TRAINED_COUNT_ONCE)
TRAINED_SOURCES: Final = (SOURCE_TRAINED_NORMAL, SOURCE_TRAINED_COUNT_ONCE)


@dataclass(frozen=True)
class QueryRecord:
    """One query scored under both BM25 modes.

    ``terms`` are the analyzed terms (with repeats) of the scored query: the rewrite when there
    is one, else the original. A format error keeps the raw response in ``rewrite_text``, sets
    ``format_error`` to the reason and scores 0.0 in both modes. ``search_error`` names the
    modes whose search overflowed Lucene's clause limit (those score 0.0), as
    ``"too_many_clauses:<mode>[,<mode>]"``.
    """

    qid: str
    source: str
    query_text: str
    rewrite_text: str | None
    format_error: str | None
    terms: tuple[str, ...]
    ndcg_normal: float
    ndcg_count_once: float
    search_error: str | None = None


_HEADER_KEY: Final = "header"


def write_records(path: Path, records: Sequence[QueryRecord], *, header: Mapping[str, Any] | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        if header is not None:
            f.write(json.dumps({_HEADER_KEY: dict(header)}) + "\n")
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def read_records(path: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if line_no == 1 and _HEADER_KEY in row:
                continue
            try:
                records.append(QueryRecord(**{**row, "terms": tuple(row["terms"])}))
            except (TypeError, KeyError) as exc:
                raise ValueError(f"{path}: line {line_no} is not a QueryRecord: {exc}") from exc
    return records


def read_header(path: Path) -> dict[str, Any] | None:
    """The file's header, or None for a file written without one."""
    with path.open(encoding="utf-8") as f:
        first = f.readline()
    if not first:
        return None
    row = json.loads(first)
    header = row.get(_HEADER_KEY)
    if header is not None and not isinstance(header, dict):
        raise ValueError(f"{path}: header must be a JSON object")
    return header
