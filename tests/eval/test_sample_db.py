from pathlib import Path
from typing import Any

import pytest

from drbm25.eval.sample_db import SampleRow, read_db, shard_complete, write_shard


def _row(source: str = "trained-normal", query_index: int = 0, sample_index: int = 0, **kw: Any) -> SampleRow:
    base: dict[str, Any] = {
        "source": source,
        "qid": f"q{query_index}",
        "query_index": query_index,
        "sample_index": sample_index,
        "query_text": "heart attack aspirin",
        "response_text": 'think</think>\n<answer>{"query": "aspirin myocardial infarction"}</answer>',
        "think_text": "think",
        "rewrite": "aspirin myocardial infarction",
        "format_error": None,
        "terms": ("aspirin", "myocardi", "infarct"),
        "n_response_tokens": 17,
        "hit_max_tokens": False,
        "sum_logprob": -3.25,
        "doc_ids_normal": ("d1", "d2"),
        "scores_normal": (12.5, 11.0),
        "doc_ids_count_once": ("d2", "d1"),
        "scores_count_once": (10.0, 9.5),
        "ndcg_normal": 0.5,
        "ndcg_count_once": 0.25,
        "search_error": None,
        "model_id": "objones25/drbm25-qwen2.5-3b-normal",
        "model_revision": "abc123",
        "seed": 7,
    }
    base.update(kw)
    return SampleRow(**base)


def test_write_then_read_round_trips_all_fields(tmp_path: Path) -> None:
    rows = [
        _row(),
        _row(
            sample_index=1,
            response_text="no tags",
            think_text=None,
            rewrite=None,
            format_error="tag_count",
            terms=(),
            sum_logprob=None,
            doc_ids_normal=(),
            scores_normal=(),
            doc_ids_count_once=(),
            scores_count_once=(),
            ndcg_normal=0.0,
            ndcg_count_once=0.0,
        ),
    ]
    write_shard(tmp_path / "trained-normal" / "shard-00000.parquet", rows)

    assert read_db(tmp_path) == rows


def test_read_db_ignores_shard_without_success_marker(tmp_path: Path) -> None:
    done = tmp_path / "untrained" / "shard-00000.parquet"
    write_shard(done, [_row("untrained")])
    partial = tmp_path / "untrained" / "shard-00001.parquet"
    write_shard(partial, [_row("untrained", query_index=1)])
    partial.with_name(partial.name + "._SUCCESS").unlink()

    assert shard_complete(done) is True
    assert shard_complete(partial) is False
    assert [r.query_index for r in read_db(tmp_path)] == [0]


def test_read_db_sorts_rows_by_source_query_and_sample(tmp_path: Path) -> None:
    write_shard(tmp_path / "untrained" / "shard-00000.parquet", [_row("untrained", 1, 1), _row("untrained", 0, 1)])
    write_shard(tmp_path / "original" / "shard-00000.parquet", [_row("original", 1, 0), _row("original", 0, 0)])

    keys = [(r.source, r.query_index, r.sample_index) for r in read_db(tmp_path)]

    assert keys == [("original", 0, 0), ("original", 1, 0), ("untrained", 0, 1), ("untrained", 1, 1)]


def test_write_shard_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        write_shard(tmp_path / "x" / "shard-00000.parquet", [])
