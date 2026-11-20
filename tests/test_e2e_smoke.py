"""End to end on CPU: toy index -> train both reward modes in chunks -> eval five sources -> report."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from drbm25.data import DEV_FILE, TRAIN_FILE
from drbm25.eval import report as report_module
from drbm25.eval import run_eval as run_eval_module
from drbm25.eval.records import SOURCE_ORIGINAL, SOURCE_UNTRAINED, read_records
from drbm25.obs.setup import shutdown
from drbm25.train import EXIT_OK
from drbm25.train import main as train_main

pytestmark = [pytest.mark.java, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"
MODES = ("normal", "count_once")


def _write_data(data_dir: Path) -> None:
    for name, n in ((TRAIN_FILE, 20), (DEV_FILE, 6)):
        path = data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.dumps({"question": f"solar efficiency {i}", "docs_id": ["docB"]}) for i in range(n)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sets(workdir: Path, toy_index: Path) -> list[str]:
    sets = {
        "retrieval.index_dir": str(toy_index),
        "data.data_dir": str(workdir / "data"),
        "data.val_size": "4",
        "run.output_dir": str(workdir / "out"),
    }
    return [arg for key, value in sets.items() for arg in ("--set", f"{key}={value}")]


def _in_own_process(main: Any, argv: list[str]) -> int:
    """Logging is configured once per process, so undo it after each CLI call."""
    try:
        return main(argv)
    finally:
        shutdown()


def _batch_indices(run_dir: Path) -> dict[int, list[int]]:
    lines = (run_dir / "train_rank0.log").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return {r["step"]: r["batch_indices"] for r in records if r.get("message") == "batch"}


def _train_in_chunks(sets: list[str]) -> None:
    """Two chunks per reward mode (``--until-step 2`` then ``4``), as the RunPod launcher does."""
    for mode in MODES:
        for until in ("2", "4"):
            argv = ["--config", str(SMOKE), "--reward-mode", mode, "--run-name", mode, "--until-step", until, *sets]
            assert _in_own_process(train_main, argv) == EXIT_OK


def _evaluate(jobs: dict[str, list[str]], evals: Path, sets: list[str]) -> None:
    for name, args in jobs.items():
        argv = ["--config", str(SMOKE), *args, "--out", str(evals / f"{name}.jsonl"), *sets]
        assert _in_own_process(run_eval_module.main, argv) == 0


def _check_rows(rows: list[list[str]]) -> None:
    assert [r[0] for r in rows] == [SOURCE_ORIGINAL, SOURCE_UNTRAINED, "trained-normal", "trained-count_once"]
    for row in rows:
        assert 0.0 <= float(row[1]) <= 1.0
        assert 0.0 <= float(row[2]) <= 1.0
    for row in rows[:2]:
        assert row[3:] == ["-", "-"]
    for row in rows[2:]:
        assert row[3] in {"2", "4"}
        assert re.match(r"-?\d+\.\d{4} \(val/ndcg_(normal|count_once)\)$", row[4])


def _check_records(evals: Path, names: list[str]) -> None:
    qids = [r.qid for r in read_records(evals / "original.jsonl")]
    for name in names:
        records = read_records(evals / f"{name}.jsonl")
        assert [r.qid for r in records] == qids
        assert all(0.0 <= r.ndcg_normal <= 1.0 and 0.0 <= r.ndcg_count_once <= 1.0 for r in records)


def _check_provenance(out: Path) -> None:
    for mode in MODES:
        manifest = json.loads((out / mode / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["seeds"]["run_seed"] == 1234
        assert manifest["seeds"]["split_seed"] == 1234
        assert (out / mode / "best" / "meta.json").is_file()


@pytest.fixture
def offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.mark.usefixtures("offline_env")
def test_e2e_tiny_pipeline(tmp_path: Path, toy_index: Path) -> None:
    _write_data(tmp_path / "data")
    sets = _sets(tmp_path, toy_index)
    out = tmp_path / "out"
    _train_in_chunks(sets)

    # Five sources on the same dev queries; trained ones come from each run's best/ policy.
    evals = tmp_path / "evals"
    jobs = {
        "original": ["--source", SOURCE_ORIGINAL],
        "untrained": ["--source", SOURCE_UNTRAINED],
        "normal": ["--source", "checkpoint", "--checkpoint", str(out / "normal" / "best")],
        "count_once": ["--source", "checkpoint", "--checkpoint", str(out / "count_once" / "best")],
    }
    _evaluate(jobs, evals, sets)
    report_path = tmp_path / "report.md"
    inputs = [str(evals / f"{name}.jsonl") for name in jobs]
    argv = ["--inputs", *inputs, "--out", str(report_path), "--n-resamples", "200", "--bootstrap-seed", "7"]
    assert _in_own_process(report_module.main, argv) == 0

    text = report_path.read_text(encoding="utf-8")
    assert "Queries: 6" in text
    assert "| best step | val score (selection metric) |" in text
    assert "Bootstrap: 200 resamples, seed 7" in text
    table = [line for line in text.splitlines() if re.match(r"\| (original|untrained|trained-\w+) \|", line)]
    assert len(table) >= 4
    _check_rows([[c.strip() for c in line.strip("|").split("|")] for line in table[:4]])
    _check_records(evals, list(jobs))
    _check_provenance(out)

    # Both modes saw the same data in the same order.
    normal, count_once = (_batch_indices(out / m) for m in MODES)
    assert sorted(normal) == [0, 1, 2, 3]
    assert normal == count_once
