import json
from pathlib import Path

import pyarrow.parquet as pq
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


def _run(db: Path, out: Path) -> None:
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


def test_pipeline_writes_all_outputs(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "metrics"
    _run(db, out)

    for name in [
        "sample_metrics.parquet",
        "query_metrics.parquet",
        "cross_policy.parquet",
        "comparisons.json",
        "lexical.json",
        "variance.json",
    ]:
        assert (out / name).is_file(), name
    assert pq.read_table(out / "sample_metrics.parquet").num_rows == 2 + 3 * 2 * 4
    assert pq.read_table(out / "query_metrics.parquet").num_rows == 2 + 3 * 2


def test_pipeline_per_sample_term_diff_against_original(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "metrics"
    _run(db, out)

    rows = pq.read_table(out / "sample_metrics.parquet").to_pylist()
    first = next(r for r in rows if r["source"] == "trained-normal" and r["qid"] == "q0" and r["sample_index"] == 0)
    assert first["retention"] == pytest.approx(1 / 3)  # {aspirin} of {heart, attack, aspirin}
    assert first["n_added"] == 1
    assert first["rewrite_class"] == "mixed"
    bad = next(r for r in rows if r["source"] == "untrained" and r["qid"] == "q0" and r["sample_index"] == 2)
    assert bad["is_format_error"] == 1
    assert bad["retention"] is None


def test_pipeline_per_query_best_of_k_and_diversity(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "metrics"
    _run(db, out)

    rows = pq.read_table(out / "query_metrics.parquet").to_pylist()
    q = next(r for r in rows if r["source"] == "trained-normal" and r["qid"] == "q0")
    assert q["ndcg_normal_mean"] == pytest.approx(0.4 + 0.15)
    assert q["best_of_1"] == pytest.approx(0.55)
    assert q["best_of_4"] == pytest.approx(0.7)
    assert q["distinct_fraction"] == pytest.approx(3 / 4)


def test_pipeline_comparisons_mark_primary_and_adjust_the_rest(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "metrics"
    _run(db, out)

    comps = json.loads((out / "comparisons.json").read_text())
    primary = [c for c in comps if c["primary"]]
    assert {(c["metric"], c["a"], c["b"]) for c in primary} == {
        ("ndcg_normal_mean", "trained-normal", "trained-count_once"),
        ("ndcg_count_once_mean", "trained-normal", "trained-count_once"),
    }
    assert all(c["q_by"] is None for c in primary)
    secondary = [c for c in comps if not c["primary"]]
    assert secondary
    assert all(c["q_by"] is not None for c in secondary)
    assert any(c["a"] == "original" for c in secondary)


def test_pipeline_lexical_has_fightin_words_and_c2st(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "metrics"
    _run(db, out)

    lex = json.loads((out / "lexical.json").read_text())
    pair = lex["pairs"]["trained-normal|trained-count_once"]
    assert set(pair["fightin_words"]) == {"rewrite_1", "rewrite_2", "rewrite_3", "think_1"}
    assert "js_divergence" in pair
    assert set(lex["c2st"]) == {"rewrite", "think"}


def test_pipeline_requires_original_rows(db: Path, tmp_path: Path) -> None:
    for p in (db / "original").iterdir():
        p.unlink()
    with pytest.raises(ValueError, match="original"):
        _run(db, tmp_path / "metrics")
