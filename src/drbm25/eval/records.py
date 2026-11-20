"""One scored query, and its JSON-lines file format (one file per source)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

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


def write_records(path: Path, records: Sequence[QueryRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def read_records(path: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            try:
                records.append(QueryRecord(**{**row, "terms": tuple(row["terms"])}))
            except (TypeError, KeyError) as exc:
                raise ValueError(f"{path}: line {line_no} is not a QueryRecord: {exc}") from exc
    return records
