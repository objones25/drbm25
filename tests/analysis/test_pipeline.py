import json
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest


def test_pipeline_writes_all_outputs(
    db: Path, tmp_path: Path, run_pipeline_small: Callable[[Path, Path], None]
) -> None:
    out = tmp_path / "metrics"
    run_pipeline_small(db, out)

    expected = [
        "sample_metrics.parquet",
        "query_metrics.parquet",
        "cross_policy.parquet",
        "comparisons.json",
        "lexical.json",
        "variance.json",
    ]
    assert [n for n in expected if not (out / n).is_file()] == []
    assert pq.read_table(out / "sample_metrics.parquet").num_rows == 2 + 3 * 2 * 4
    assert pq.read_table(out / "query_metrics.parquet").num_rows == 2 + 3 * 2


def test_pipeline_per_sample_term_diff_against_original(
    db: Path, tmp_path: Path, run_pipeline_small: Callable[[Path, Path], None]
) -> None:
    out = tmp_path / "metrics"
    run_pipeline_small(db, out)

    rows = pq.read_table(out / "sample_metrics.parquet").to_pylist()
    first = next(r for r in rows if r["source"] == "trained-normal" and r["qid"] == "q0" and r["sample_index"] == 0)
    assert first["retention"] == pytest.approx(1 / 3)  # {aspirin} of {heart, attack, aspirin}
    assert first["n_added"] == 1
    assert first["rewrite_class"] == "mixed"
    bad = next(r for r in rows if r["source"] == "untrained" and r["qid"] == "q0" and r["sample_index"] == 2)
    assert bad["is_format_error"] == 1
    assert bad["retention"] is None


def test_pipeline_per_query_best_of_k_and_diversity(
    db: Path, tmp_path: Path, run_pipeline_small: Callable[[Path, Path], None]
) -> None:
    out = tmp_path / "metrics"
    run_pipeline_small(db, out)

    rows = pq.read_table(out / "query_metrics.parquet").to_pylist()
    q = next(r for r in rows if r["source"] == "trained-normal" and r["qid"] == "q0")
    assert q["ndcg_normal_mean"] == pytest.approx(0.4 + 0.15)
    assert q["best_of_1"] == pytest.approx(0.55)
    assert q["best_of_4"] == pytest.approx(0.7)
    assert q["distinct_fraction"] == pytest.approx(3 / 4)


def test_pipeline_comparisons_mark_primary_and_adjust_the_rest(
    db: Path, tmp_path: Path, run_pipeline_small: Callable[[Path, Path], None]
) -> None:
    out = tmp_path / "metrics"
    run_pipeline_small(db, out)

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


def test_pipeline_lexical_has_fightin_words_and_c2st(
    db: Path, tmp_path: Path, run_pipeline_small: Callable[[Path, Path], None]
) -> None:
    out = tmp_path / "metrics"
    run_pipeline_small(db, out)

    lex = json.loads((out / "lexical.json").read_text())
    pair = lex["pairs"]["trained-normal|trained-count_once"]
    assert set(pair["fightin_words"]) == {"rewrite_1", "rewrite_2", "rewrite_3", "think_1"}
    assert "js_divergence" in pair
    assert set(lex["c2st"]) == {"rewrite", "think"}


def test_pipeline_requires_original_rows(
    db: Path, tmp_path: Path, run_pipeline_small: Callable[[Path, Path], None]
) -> None:
    for p in (db / "original").iterdir():
        p.unlink()
    with pytest.raises(ValueError, match="original"):
        run_pipeline_small(db, tmp_path / "metrics")
