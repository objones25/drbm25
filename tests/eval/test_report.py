"""The report: input checks, tables (with each run's best step), CIs, provenance, W&B logging."""

import json
import math
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import drbm25.eval.report as report_module
from drbm25.eval.records import QueryRecord, read_header, read_records, write_records
from drbm25.eval.report import build_report, build_report_data, main, render_markdown
from drbm25.eval.stats import paired_bootstrap_ci

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"


def _records(
    source: str,
    normal: list[float],
    once: list[float],
    *,
    qids: list[str] | None = None,
    think_chars: int | None = None,
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
            think_chars=think_chars,
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


def test_report_query_stats_show_mean_think_length_in_characters() -> None:
    sources = _all_sources()
    sources["untrained"] = _records("untrained", [0.0, 0.2, 0.4], [0.0, 0.2, 0.4], think_chars=120)
    text = build_report(sources)
    assert "| repeat_ratio | any_repeat_rate | format_error_rate | mean_think_chars |" in text
    assert "| untrained | 2.00 | 1.00 | 0.500 | 1.000 | 0.000 | 120.0 |" in text
    assert "| original | 2.00 | 1.00 | 0.500 | 1.000 | 0.000 | - |" in text  # no rewrite, no think


def test_report_states_how_repeat_ratio_differs_from_the_training_metric() -> None:
    text = build_report(_all_sources())
    assert "repeat_ratio here is pooled over terms" in text
    assert "rewrite/repeat_ratio" in text  # the per-query mean logged during training


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


# ------------------------------------------------------------------ best step and provenance
def _best_header(source: str, step: int, normal: float, count_once: float, metric: str) -> dict[str, Any]:
    return {
        "format_version": 1,
        "source": source,
        "checkpoint": f"/runs/{source}/best",
        "eval_config_sha256": "e" * 64,
        "eval_seed": 1234,
        "best": {
            "step": step,
            "val_ndcg_normal": normal,
            "val_ndcg_count_once": count_once,
            "selection_metric": metric,
            "config_sha256": "c" * 64,
            "seeds": {"run_seed": 1234, "split_seed": 1234},
            "git": {"sha": "a" * 40, "dirty": False},
        },
    }


def _headers() -> dict[str, dict[str, Any]]:
    return {
        "trained-normal": _best_header("trained-normal", 1450, 0.4211, 0.3012, "val/ndcg_normal"),
        "trained-count_once": _best_header("trained-count_once", 900, 0.3518, 0.3907, "val/ndcg_count_once"),
    }


def test_report_table_shows_each_runs_best_step_and_validation_score() -> None:
    text = render_markdown(build_report_data(_all_sources(), headers=_headers(), n_resamples=50))
    assert "| source | nDCG@10 normal | nDCG@10 count_once | best step | val score (selection metric) |" in text
    assert "| original | 0.4000 | 0.4000 | - | - |" in text
    assert "| trained-normal | 0.7000 | 0.5000 | 1450 | 0.4211 (val/ndcg_normal) |" in text
    assert "| trained-count_once | 0.6000 | 0.7000 | 900 | 0.3907 (val/ndcg_count_once) |" in text


def _headers_with_trained_steps(normal: int, count_once: int) -> dict[str, dict[str, Any]]:
    headers = _headers()
    headers["trained-normal"]["trained_steps"] = normal
    headers["trained-count_once"]["trained_steps"] = count_once
    return headers


def test_report_table_shows_each_runs_trained_steps() -> None:
    headers = _headers_with_trained_steps(1500, 1500)
    text = render_markdown(build_report_data(_all_sources(), headers=headers, n_resamples=50))
    assert "| best step | val score (selection metric) | training steps |" in text
    assert "| original | 0.4000 | 0.4000 | - | - | - |" in text
    assert "| trained-normal | 0.7000 | 0.5000 | 1450 | 0.4211 (val/ndcg_normal) | 1500 |" in text
    assert "| trained-count_once | 0.6000 | 0.7000 | 900 | 0.3907 (val/ndcg_count_once) | 1500 |" in text
    assert "Training steps differ" not in text


def test_report_flags_unequal_trained_steps() -> None:
    headers = _headers_with_trained_steps(1500, 1450)
    text = render_markdown(build_report_data(_all_sources(), headers=headers, n_resamples=50))
    assert "Training steps differ: trained-normal 1500, trained-count_once 1450" in text


def test_report_records_the_bootstrap_seed_and_each_sources_seeds() -> None:
    text = render_markdown(build_report_data(_all_sources(), headers=_headers(), n_resamples=50, seed=7))
    assert "Bootstrap: 50 resamples, seed 7" in text
    assert "| trained-normal | " + "c" * 64 + " | run_seed=1234, split_seed=1234 | " + "a" * 40 + " |" in text


def test_report_bootstrap_seed_drives_the_confidence_intervals() -> None:
    sources = _all_sources()
    report = build_report_data(sources, n_resamples=50, seed=7)
    name, scorer, ci = report.paired_ci[0]
    a = [r.ndcg_normal for r in sources["trained-normal"]]
    b = [r.ndcg_normal for r in sources["trained-count_once"]]
    import numpy as np

    assert (name, scorer) == ("trained-count_once - trained-normal", "nDCG@10 normal")
    assert ci == paired_bootstrap_ci(np.array(a), np.array(b), n_resamples=50, seed=7)


def test_cli_reads_headers_and_records_the_bootstrap_seed(tmp_path: Path) -> None:
    headers = _headers()
    paths = []
    for name, recs in _all_sources().items():
        path = tmp_path / f"{name}.jsonl"
        write_records(path, recs, header=headers.get(name))
        paths.append(str(path))
    out = tmp_path / "r.md"
    assert main(["--inputs", *paths, "--out", str(out), "--n-resamples", "50", "--bootstrap-seed", "7"]) == 0
    text = out.read_text()
    assert "| trained-normal | 0.7000 | 0.5000 | 1450 | 0.4211 (val/ndcg_normal) |" in text
    assert "Bootstrap: 50 resamples, seed 7" in text


def test_cli_bootstrap_seed_defaults_to_0(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    write_records(path, _records("original", [0.1], [0.1]))
    out = tmp_path / "r.md"
    assert main(["--inputs", str(path), "--out", str(out), "--n-resamples", "10"]) == 0
    assert "Bootstrap: 10 resamples, seed 0" in out.read_text()


def test_records_header_round_trips_and_is_not_a_record(tmp_path: Path) -> None:
    recs = _records("trained-normal", [0.5], [0.3])
    path = tmp_path / "r.jsonl"
    write_records(path, recs, header=_headers()["trained-normal"])
    assert read_header(path) == _headers()["trained-normal"]
    assert read_records(path) == recs
    plain = tmp_path / "plain.jsonl"
    write_records(plain, recs)
    assert read_header(plain) is None
    assert read_records(plain) == recs


# ------------------------------------------------------------------ W&B
class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []
        self.artifacts: list[Any] = []

    def log(self, data: Mapping[str, Any]) -> None:
        self.logged.append(dict(data))

    def log_artifact(self, artifact: Any, **kwargs: Any) -> Any:
        del kwargs
        self.artifacts.append(artifact)
        return artifact


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
    assert opened["extra_config"]["bootstrap_seed"] == 0
    (row,) = run.logged
    assert set(row) == {"report/ndcg", "report/paired_ci", "report/query_stats"}
    assert row["report/ndcg"].columns == ["source", "ndcg10_normal", "ndcg10_count_once"]
    assert len(row["report/ndcg"].data) == 4


def test_wandb_flag_logs_the_eval_jsonl_files_and_report_as_an_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _FakeRun()

    @contextmanager
    def fake_open_run(cfg: object, **kwargs: Any) -> Generator[_FakeRun]:
        del cfg, kwargs
        yield run

    monkeypatch.setattr(report_module, "open_run", fake_open_run)
    paths = []
    for name, recs in _all_sources().items():
        path = tmp_path / f"{name}.jsonl"
        write_records(path, recs)
        paths.append(str(path))
    argv = ["--inputs", *paths, "--out", str(tmp_path / "report.md"), "--n-resamples", "50"]
    assert main([*argv, "--wandb", "--config", str(SMOKE)]) == 0
    (artifact,) = run.artifacts
    assert artifact.type == "evaluation"
    assert sorted(artifact.manifest.entries) == sorted([*(f"{n}.jsonl" for n in _all_sources()), "report.md"])


def test_wandb_query_stats_table_has_no_nan_for_a_missing_think_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _FakeRun()

    @contextmanager
    def fake_open_run(cfg: object, **kwargs: Any) -> Generator[_FakeRun]:
        del cfg, kwargs
        yield run

    monkeypatch.setattr(report_module, "open_run", fake_open_run)
    path = tmp_path / "original.jsonl"
    write_records(path, _records("original", [0.1], [0.1]))
    argv = ["--inputs", str(path), "--out", str(tmp_path / "r.md"), "--n-resamples", "10"]
    assert main([*argv, "--wandb", "--config", str(SMOKE)]) == 0
    (row,) = run.logged
    table = row["report/query_stats"]
    think = table.data[0][table.columns.index("mean_think_chars")]
    assert think is None
    assert not any(isinstance(v, float) and math.isnan(v) for v in table.data[0][1:])


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
    argv = [
        "--inputs",
        *paths,
        "--out",
        str(tmp_path / "r.md"),
        "--n-resamples",
        "50",
        "--wandb",
        "--config",
        str(SMOKE),
    ]
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
    rec = QueryRecord(
        "q", "s", "text", "rw", "bad_json", ("a",), 0.25, 0.5, search_error="too_many_clauses:normal", think_chars=42
    )
    path = tmp_path / "r.jsonl"
    write_records(path, [rec])
    assert json.loads(path.read_text())["terms"] == ["a"]
    assert read_records(path) == [rec]
