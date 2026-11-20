"""The report: input checks, common_step, tables, CIs, W&B logging."""

import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import drbm25.eval.report as report_module
from drbm25.checkpoint import TrainState, save_checkpoint
from drbm25.eval.records import QueryRecord, write_records
from drbm25.eval.report import build_report, common_step, main

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"


def _records(
    source: str, normal: list[float], once: list[float], *, qids: list[str] | None = None
) -> list[QueryRecord]:
    ids = qids or [f"q{i}" for i in range(len(normal))]
    return [
        QueryRecord(
            qid=qid,
            source=source,
            query_text="t",
            rewrite_text=None,
            format_error=None,
            terms=("a", "a"),
            ndcg_normal=n,
            ndcg_count_once=c,
        )
        for qid, n, c in zip(ids, normal, once, strict=True)
    ]


def _all_sources() -> dict[str, list[QueryRecord]]:
    return {
        "original": _records("original", [0.2, 0.4, 0.6], [0.2, 0.4, 0.6]),
        "untrained": _records("untrained", [0.0, 0.2, 0.4], [0.0, 0.2, 0.4]),
        "trained-normal": _records("trained-normal", [0.5, 0.7, 0.9], [0.3, 0.5, 0.7]),
        "trained-count_once": _records("trained-count_once", [0.4, 0.6, 0.8], [0.5, 0.7, 0.9]),
    }


def test_report_rejects_mismatched_qids() -> None:
    sources = _all_sources()
    sources["untrained"] = _records("untrained", [0.0, 0.1, 0.2], [0.0, 0.1, 0.2], qids=["q0", "q1", "other"])
    with pytest.raises(ValueError, match="qid sets differ"):
        build_report(sources)


def test_report_rejects_duplicate_qids_within_a_source() -> None:
    sources = {"original": _records("original", [0.1, 0.2], [0.1, 0.2], qids=["q0", "q0"])}
    with pytest.raises(ValueError, match="duplicate qid"):
        build_report(sources)


def test_report_rejects_unknown_source_and_empty_input() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        build_report({"mystery": _records("mystery", [0.1], [0.1])})
    with pytest.raises(ValueError, match="no inputs"):
        build_report({})


def test_report_main_rejects_mismatched_qids_from_files(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_records(a, _records("original", [0.1], [0.1], qids=["x"]))
    write_records(b, _records("untrained", [0.1], [0.1], qids=["y"]))
    with pytest.raises(ValueError, match="qid sets differ"):
        main(["--inputs", str(a), str(b), "--out", str(tmp_path / "r.md")])


def test_report_main_rejects_file_mixing_sources(tmp_path: Path) -> None:
    mixed = tmp_path / "m.jsonl"
    write_records(
        mixed, [*_records("original", [0.1], [0.1], qids=["x"]), *_records("untrained", [0.1], [0.1], qids=["y"])]
    )
    with pytest.raises(ValueError, match="more than one source"):
        main(["--inputs", str(mixed), "--out", str(tmp_path / "r.md")])


def test_report_table_has_hand_computed_means() -> None:
    text = build_report(_all_sources())
    assert "| original | 0.4000 | 0.4000 |" in text
    assert "| untrained | 0.2000 | 0.2000 |" in text
    assert "| trained-normal | 0.7000 | 0.5000 |" in text
    assert "| trained-count_once | 0.6000 | 0.7000 |" in text


def test_report_only_lists_present_sources() -> None:
    text = build_report({"original": _records("original", [0.5], [0.25])})
    assert "| original | 0.5000 | 0.2500 |" in text
    assert "trained" not in text
    assert "untrained" not in text
    assert "Paired bootstrap" not in text  # no comparison has both of its sources


def test_report_ci_rows_have_hand_computed_differences() -> None:
    text = build_report(_all_sources(), n_resamples=200)
    # (comparison, scorer, mean of b - a); every per-query difference is constant, so the CI is a point.
    assert "| trained-count_once - trained-normal | nDCG@10 normal | -0.1000 | -0.1000 | -0.1000 |" in text
    assert "| trained-count_once - trained-normal | nDCG@10 count_once | +0.2000 | +0.2000 | +0.2000 |" in text
    assert "| trained-normal: normal - count_once | nDCG@10 | +0.2000 | +0.2000 | +0.2000 |" in text
    assert "| trained-count_once: normal - count_once | nDCG@10 | -0.1000 | -0.1000 | -0.1000 |" in text
    assert "| untrained - original | nDCG@10 normal | -0.2000 | -0.2000 | -0.2000 |" in text
    assert "| untrained - original | nDCG@10 count_once | -0.2000 | -0.2000 | -0.2000 |" in text


def test_report_skips_comparisons_whose_sources_are_missing() -> None:
    text = build_report({"trained-normal": _all_sources()["trained-normal"]}, n_resamples=50)
    assert "trained-normal: normal - count_once" in text
    assert "untrained - original" not in text
    assert "trained-count_once - trained-normal" not in text


def test_report_includes_query_stats_per_source() -> None:
    text = build_report(_all_sources())
    assert "| original | 2.00 | 1.00 | 0.500 | 1.000 | 0.000 |" in text


def test_report_aligns_queries_by_qid_not_by_position() -> None:
    a = _records("original", [0.1, 0.9], [0.1, 0.9], qids=["x", "y"])
    b = _records("untrained", [0.9, 0.1], [0.9, 0.1], qids=["y", "x"])
    text = build_report({"original": a, "untrained": b}, n_resamples=50)
    # by qid: untrained x = 0.1, y = 0.9, identical to original, so the difference is exactly zero
    assert "| untrained - original | nDCG@10 normal | +0.0000 | +0.0000 | +0.0000 |" in text


def test_report_main_writes_markdown(tmp_path: Path) -> None:
    paths = []
    for name, recs in _all_sources().items():
        path = tmp_path / f"{name}.jsonl"
        write_records(path, recs)
        paths.append(str(path))
    out = tmp_path / "report.md"
    assert main(["--inputs", *paths, "--out", out.as_posix(), "--n-resamples", "100"]) == 0
    assert "| trained-normal | 0.7000 | 0.5000 |" in out.read_text()


# ------------------------------------------------------------------ common_step
def _ckpt_run(root: Path, name: str, steps: list[int]) -> Path:
    run_dir = root / name
    for step in steps:
        save_checkpoint(
            run_dir,
            train_state=TrainState(step=step, data_cursor=0, wandb_run_id=None, reward_mode="normal"),
            model={},
            optimizers={},
            scaler=None,
            keep=10,
        )
    return run_dir


def test_common_step_picks_max_shared(tmp_path: Path) -> None:
    a = _ckpt_run(tmp_path, "a", [50, 100, 150])
    b = _ckpt_run(tmp_path, "b", [50, 100])
    assert common_step(a, b) == 100
    assert common_step(b, a) == 100


def test_common_step_without_overlap_raises(tmp_path: Path) -> None:
    a = _ckpt_run(tmp_path, "a", [50])
    b = _ckpt_run(tmp_path, "b", [100])
    with pytest.raises(ValueError, match="no common step"):
        common_step(a, b)


def test_common_step_ignores_incomplete_checkpoint(tmp_path: Path) -> None:
    a = _ckpt_run(tmp_path, "a", [50, 100])
    b = _ckpt_run(tmp_path, "b", [50, 100])
    (b / "step_000100" / "meta.json").unlink()
    assert common_step(a, b) == 50


# ------------------------------------------------------------------ W&B
class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []

    def log(self, data: Mapping[str, Any]) -> None:
        self.logged.append(dict(data))


def test_wandb_flag_logs_tables_in_a_report_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _FakeRun()
    opened: dict[str, Any] = {}

    @contextmanager
    def fake_open_run(cfg: object, **kwargs: Any) -> Generator[_FakeRun]:
        opened.update(kwargs)
        yield run

    monkeypatch.setattr(report_module, "open_run", fake_open_run)
    paths = []
    for name, recs in _all_sources().items():
        path = tmp_path / f"{name}.jsonl"
        write_records(path, recs)
        paths.append(str(path))
    argv = ["--inputs", *paths, "--out", str(tmp_path / "r.md"), "--n-resamples", "50"]
    assert main([*argv, "--wandb", "--config", str(SMOKE)]) == 0
    assert opened["job_type"] == "report"
    (row,) = run.logged
    assert set(row) == {"report/ndcg", "report/paired_ci", "report/query_stats"}
    assert row["report/ndcg"].columns == ["source", "ndcg10_normal", "ndcg10_count_once"]
    assert len(row["report/ndcg"].data) == 4


def test_wandb_report_applies_set_overrides_to_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    @contextmanager
    def fake_open_run(cfg: Any, **kwargs: Any) -> Generator[_FakeRun]:
        seen.append(cfg)
        yield _FakeRun()

    monkeypatch.setattr(report_module, "open_run", fake_open_run)
    paths = []
    for name, recs in _all_sources().items():
        path = tmp_path / f"{name}.jsonl"
        write_records(path, recs)
        paths.append(str(path))
    argv = ["--inputs", *paths, "--out", str(tmp_path / "r.md"), "--n-resamples", "50", "--wandb", "--config", str(SMOKE)]
    argv += ["--set", "run.wandb_project=my-proj", "--set", "run.wandb_entity=my-team"]
    assert main(argv) == 0
    (cfg,) = seen
    assert cfg.run.wandb_project == "my-proj"
    assert cfg.run.wandb_entity == "my-team"


def test_report_rejects_a_malformed_set_override(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    write_records(path, next(iter(_all_sources().values())))
    with pytest.raises(SystemExit):
        main(["--inputs", str(path), "--out", str(tmp_path / "r.md"), "--set", "nokey"])


def test_wandb_flag_needs_config(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    write_records(path, _records("original", [0.1], [0.1]))
    with pytest.raises(SystemExit) as exc:
        main(["--inputs", str(path), "--out", str(tmp_path / "r.md"), "--wandb"])
    assert exc.value.code == 2


def test_jsonl_round_trip_keeps_every_field(tmp_path: Path) -> None:
    rec = QueryRecord("q", "s", "text", "rw", "bad_json", ("a",), 0.25, 0.5, search_error="too_many_clauses:normal")
    path = tmp_path / "r.jsonl"
    write_records(path, [rec])
    assert json.loads(path.read_text())["terms"] == ["a"]
    from drbm25.eval.records import read_records

    assert read_records(path) == [rec]
