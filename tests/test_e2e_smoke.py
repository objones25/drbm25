"""End to end on CPU: toy index -> train both reward modes in chunks -> eval five sources -> report."""

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from drbm25.bm25.searcher import Bm25Searcher
from drbm25.checkpoint import load_best, read_best_meta
from drbm25.config import ScoringMode, load_config
from drbm25.data import DEV_FILE, TRAIN_FILE
from drbm25.eval import report as report_module
from drbm25.eval import run_eval as run_eval_module
from drbm25.eval.records import SOURCE_ORIGINAL, SOURCE_UNTRAINED, read_header, read_records
from drbm25.obs.setup import shutdown
from drbm25.ppo.models import LoraPolicyModels
from drbm25.train import EXIT_OK, build_base, check_layout
from drbm25.train import main as train_main

pytestmark = [pytest.mark.java, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"
MODES: tuple[ScoringMode, ...] = ("normal", "count_once")


# (query, relevant doc) over the toy index: varied so that per-query nDCGs differ (some 0, some > 0).
QUERIES = (
    ("solar efficiency", "docB"),
    ("efficiency", "docB"),
    ("solar", "docA"),
    ("zzqx", "docB"),  # matches no document: always 0
    ("solar solar solar efficiency", "docB"),
    ("efficiency solar", "docA"),
)


def _write_data(data_dir: Path) -> None:
    for name, n in ((TRAIN_FILE, 20), (DEV_FILE, len(QUERIES))):
        path = data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            json.dumps({"question": QUERIES[i % len(QUERIES)][0], "docs_id": [QUERIES[i % len(QUERIES)][1]]})
            for i in range(n)
        ]
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


def _check_rows(rows: list[list[str]], out: Path) -> None:
    assert [r[0] for r in rows] == [SOURCE_ORIGINAL, SOURCE_UNTRAINED, "trained-normal", "trained-count_once"]
    for row in rows:
        assert 0.0 <= float(row[1]) <= 1.0
        assert 0.0 <= float(row[2]) <= 1.0
    for row in rows[:2]:
        assert row[3:] == ["-", "-"]
    for row, mode in zip(rows[2:], MODES, strict=True):
        meta = read_best_meta(out / mode / "best")
        assert meta.reward_mode == mode
        assert row[3] == str(meta.step)
        assert row[4] == f"{meta.score:.4f} (val/ndcg_{mode})"


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


def _ndcg10(doc_ids: tuple[str, ...], relevant: str) -> float:
    """nDCG@10 with one relevant document (ideal DCG is 1), written out by hand."""
    return next((1 / math.log2(rank + 2) for rank, d in enumerate(doc_ids[:10]) if d == relevant), 0.0)


def _check_original_matches_direct_search(toy_index: Path, normal: float, count_once: float) -> None:
    searcher = Bm25Searcher(toy_index, k1=0.9, b=0.4, threads=2, timeout_s=30.0)
    reported_by_mode: dict[ScoringMode, float] = {"normal": normal, "count_once": count_once}
    for mode, reported in reported_by_mode.items():
        scores = [_ndcg10(searcher.search(q, mode=mode, k=10).doc_ids, doc) for q, doc in QUERIES]
        assert any(x == 0.0 for x in scores)
        assert any(x > 0.0 for x in scores)
        assert reported == pytest.approx(sum(scores) / len(scores), abs=5e-5)  # the report prints 4 decimals


def _check_trained_from_best(tmp_path: Path, evals: Path, out: Path, sets: list[str]) -> None:
    """Each trained row is scored from that run's best/ policy, and best/ differs from the untrained policy."""
    overrides = {kv.partition("=")[0]: kv.partition("=")[2] for kv in sets[1::2]}
    cfg = load_config(SMOKE, overrides)
    torch.manual_seed(cfg.run.seed)  # as run_eval does before building the model
    device = torch.device("cpu")
    base, _ = build_base(cfg, device)
    untrained = LoraPolicyModels(base, check_layout(cfg, device), gradient_checkpointing=False).actor_state_dict()
    for mode in MODES:
        best = out / mode / "best"
        meta, weights = load_best(best)
        assert weights.keys() == untrained.keys()
        assert any(not torch.equal(weights[k], untrained[k]) for k in weights)
        header = read_header(evals / f"{mode}.jsonl")
        assert header is not None
        assert header["source"] == f"trained-{mode}"
        assert header["checkpoint"] == str(best)
        assert header["best"]["step"] == meta.step
        assert header["best"]["selection_metric"] == f"val/ndcg_{mode}"
        assert header["best"]["val_ndcg_normal"] == meta.val_ndcg_normal
        assert header["best"]["val_ndcg_count_once"] == meta.val_ndcg_count_once
        assert header["best"]["config_sha256"] == meta.config_sha256


def _check_cis(text: str) -> None:
    rows = [
        [c.strip() for c in line.strip("|").split("|")]
        for line in text.splitlines()
        if re.match(r"\| (trained|untrained)[^|]* - [^|]*\|", line)
    ]
    assert len(rows) >= 4
    triples = [(float(r[3]), float(r[2]), float(r[4])) for r in rows]  # low, mean, high
    assert all(math.isfinite(x) for t in triples for x in t)
    assert all(low <= mean <= high for low, mean, high in triples)
    assert any(low < high for low, _, high in triples)


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
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in table[:4]]
    _check_rows(rows, out)
    _check_original_matches_direct_search(toy_index, float(rows[0][1]), float(rows[0][2]))
    _check_trained_from_best(tmp_path, evals, out, sets)
    _check_cis(text)
    _check_records(evals, list(jobs))
    _check_provenance(out)

    # Both modes saw the same data in the same order.
    normal, count_once = (_batch_indices(out / m) for m in MODES)
    assert sorted(normal) == [0, 1, 2, 3]
    assert normal == count_once
