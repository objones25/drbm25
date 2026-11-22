"""Shared fixtures: a tiny synthetic sample database and its metrics."""

from collections.abc import Callable
from pathlib import Path

import pytest

from drbm25.analysis.corpus import IdfTable
from drbm25.analysis.pipeline import run_pipeline
from drbm25.eval.sample_db import SampleRow, write_shard

ORIG = {"q0": "heart attack aspirin", "q1": "copper uses"}
# per source, per query: 4 rewrites
REWRITES = {
    "untrained": {
        "q0": ["heart heart attack", "heart attack", None, "aspirin aspirin"],
        "q1": ["copper", "copper wire", "copper", "uses"],
    },
    "trained-normal": {
        "q0": ["aspirin infarct", "aspirin infarct", "aspirin heart", "aspirin"],
        "q1": ["copper wire", "copper wire", "copper metal", "wire"],
    },
    "trained-count_once": {
        "q0": ["aspirin infarct", "aspirin risk", "aspirin", "heart"],
        "q1": ["copper wire", "copper alloy", "metal", "wire"],
    },
}
NDCG = {"untrained": 0.1, "trained-normal": 0.4, "trained-count_once": 0.5}


def _row(source: str, qi: int, si: int, rewrite: str | None, ndcg: float) -> SampleRow:
    qid = f"q{qi}"
    terms = tuple(rewrite.split()) if rewrite else ()
    docs = tuple(f"d:{t}" for t in dict.fromkeys(terms))
    return SampleRow(
        source=source,
        qid=qid,
        query_index=qi,
        sample_index=si,
        query_text=ORIG[qid],
        response_text=None if source == "original" else f"think about {rewrite}</think>...",
        think_text=None if source == "original" else f"think about {rewrite}",
        rewrite=rewrite,
        format_error=None if rewrite else "tag_count",
        terms=terms,
        n_response_tokens=None if source == "original" else 10,
        hit_max_tokens=None if source == "original" else False,
        sum_logprob=None if source == "original" else -5.0,
        doc_ids_normal=docs,
        scores_normal=tuple(1.0 for _ in docs),
        doc_ids_count_once=docs,
        scores_count_once=tuple(1.0 for _ in docs),
        ndcg_normal=ndcg,
        ndcg_count_once=ndcg,
        search_error=None,
        model_id=None,
        model_revision=None,
        seed=None,
    )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    root = tmp_path / "db"
    write_shard(
        root / "original" / "shard-00000.parquet", [_row("original", i, 0, ORIG[f"q{i}"], 0.3) for i in range(2)]
    )
    for source, per_q in REWRITES.items():
        rows = []
        for i in range(2):
            for j, rw in enumerate(per_q[f"q{i}"]):
                rows.append(_row(source, i, j, rw, 0.0 if rw is None else NDCG[source] + 0.1 * j))
        write_shard(root / source / "shard-00000.parquet", rows)
    return root


def run_small_pipeline(db: Path, out: Path) -> None:
    table = IdfTable(n_docs=1000, df={})
    table.df.update(
        {
            t: 10
            for t in [
                "heart",
                "attack",
                "aspirin",
                "copper",
                "uses",
                "wire",
                "metal",
                "infarct",
                "risk",
                "alloy",
                "think",
                "about",
            ]
        }
    )
    run_pipeline(
        db,
        out,
        analyze=lambda s: s.lower().split(),
        idf_table=table,
        encode=None,
        n_resamples=200,
        c2st_permutations=20,
    )


@pytest.fixture
def run_pipeline_small() -> Callable[[Path, Path], None]:
    return run_small_pipeline


@pytest.fixture
def metrics_dir(db: Path, tmp_path: Path) -> Path:
    out = tmp_path / "metrics"
    run_small_pipeline(db, out)
    return out
