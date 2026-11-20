"""``python -m drbm25.train``: checkpoints, resume, exit codes, startup checks, metrics.jsonl."""

import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import wandb
import yaml

import drbm25.train as train_module
from drbm25.bm25.index import read_index_info
from drbm25.checkpoint import load_best, load_sharded_actor
from drbm25.config import load_config
from drbm25.data import DEV_FILE, TRAIN_FILE, Example, load_examples, split_train
from drbm25.distributed import DistInfo
from drbm25.obs.runs import STEP_METRIC
from drbm25.obs.setup import shutdown
from drbm25.ppo.trainer import NonFiniteError, PPOTrainer
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer
from drbm25.train import (
    EXIT_BUDGET,
    EXIT_ERROR,
    EXIT_OK,
    StartupCheckError,
    check_index,
    check_java,
    check_tokenizer,
    main,
)

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"
N_TRAIN = 20


def _write_data(data_dir: Path) -> None:
    for name, n in ((TRAIN_FILE, N_TRAIN), (DEV_FILE, 6)):
        path = data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.dumps({"question": f"solar efficiency {i}", "docs_id": ["docB"]}) for i in range(n)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def workdir(tmp_path: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_data(tmp_path / "data")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return tmp_path


def _argv(workdir: Path, toy_index: Path, run_name: str, *extra: str) -> list[str]:
    sets = {
        "retrieval.index_dir": str(toy_index),
        "data.data_dir": str(workdir / "data"),
        "data.val_size": "4",
        "run.output_dir": str(workdir / "out"),
    }
    argv = ["--config", str(SMOKE), "--reward-mode", "normal", "--run-name", run_name]
    for key, value in sets.items():
        argv += ["--set", f"{key}={value}"]
    return argv + list(extra)


def _main(argv: list[str]) -> int:
    """``main`` as if in its own process: logging is configured once per process, so undo it after."""
    try:
        return main(argv)
    finally:
        shutdown()  # also stops the queue listener, so every record is flushed to the file


def _log_records(run_dir: Path, rank: int = 0) -> list[dict[str, Any]]:
    lines = (run_dir / f"train_rank{rank}.log").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _batches(run_dir: Path) -> dict[int, list[int]]:
    return {r["step"]: r["batch_indices"] for r in _log_records(run_dir) if r.get("message") == "batch"}


def _subprocess_env(workdir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        WANDB_MODE="offline",
        WANDB_DIR=str(workdir),
        WANDB_SILENT="true",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(REPO / "src"),
        GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
    )
    return env


# --------------------------------------------------------------------------- end to end
@pytest.mark.java
@pytest.mark.slow
def test_cli_runs_two_steps_and_writes_checkpoint(workdir: Path, toy_index: Path) -> None:
    proc = subprocess.run(  # noqa: S603 -- this interpreter, fixed module
        [sys.executable, "-m", "drbm25.train", *_argv(workdir, toy_index, "two", "--until-step", "2")],
        env=_subprocess_env(workdir),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == EXIT_OK, proc.stdout[-4000:] + proc.stderr[-4000:]
    run_dir = workdir / "out" / "two"
    assert (run_dir / "latest").read_text().strip() == "step_000002"
    meta = json.loads((run_dir / "step_000002" / "meta.json").read_text())
    assert meta["train_state"]["step"] == 2
    assert meta["train_state"]["data_cursor"] == 8
    assert meta["train_state"]["wandb_run_id"]
    assert meta["train_state"]["reward_mode"] == "normal"
    assert (run_dir / "exit_code").read_text().strip() == "0"


@pytest.mark.java
@pytest.mark.slow
def test_cli_resume_with_other_reward_mode_fails(workdir: Path, toy_index: Path) -> None:
    assert _main(_argv(workdir, toy_index, "mix", "--until-step", "2")) == EXIT_OK
    argv = _argv(workdir, toy_index, "mix", "--until-step", "4")
    argv[argv.index("--reward-mode") + 1] = "count_once"
    assert _main(argv) == EXIT_ERROR


@pytest.mark.java
@pytest.mark.slow
def test_cli_resume_continues_from_checkpoint_with_same_next_batch(workdir: Path, toy_index: Path) -> None:
    assert _main(_argv(workdir, toy_index, "chunked", "--until-step", "2")) == EXIT_OK
    assert not (workdir / "out" / "chunked" / "final_adapters.safetensors").exists()  # chunk end, not the end
    assert _main(_argv(workdir, toy_index, "chunked", "--until-step", "4")) == EXIT_OK
    assert _main(_argv(workdir, toy_index, "straight", "--until-step", "4")) == EXIT_OK
    chunked, straight = _batches(workdir / "out" / "chunked"), _batches(workdir / "out" / "straight")
    assert sorted(chunked) == [0, 1, 2, 3]  # the second invocation resumed at step 2, not 0
    assert chunked[2] == straight[2]
    assert chunked[2] != chunked[0]
    records = _log_records(workdir / "out" / "chunked")
    assert any(str(r["message"]).startswith("resumed from") for r in records)
    # val_every = 2: validation after the 2nd and 4th steps (step indices 1 and 3).
    assert [r["step"] for r in records if str(r["message"]).startswith("validation")] == [1, 3]
    rows = [json.loads(line) for line in (workdir / "out" / "chunked" / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2, 3]
    # Model, optimizer and RNG restore: after resuming, steps 2-3 match the uninterrupted run.
    straight_rows = [
        json.loads(line) for line in (workdir / "out" / "straight" / "metrics.jsonl").read_text().splitlines()
    ]
    for got, want in zip(rows[2:], straight_rows[2:], strict=True):
        assert _deterministic(got) == pytest.approx(_deterministic(want), nan_ok=True)
    ids = {
        json.loads((workdir / "out" / "chunked" / f"step_00000{s}" / "meta.json").read_text())["train_state"][
            "wandb_run_id"
        ]
        for s in (2, 4)
    }
    assert len(ids) == 1  # the resumed run reattached to the same W&B run id
    assert (workdir / "out" / "chunked" / "final_adapters.safetensors").is_file()  # total_steps reached


_NONDETERMINISTIC_PREFIXES = ("time/", "sys/", "budget/")
_NONDETERMINISTIC_KEYS = {"rollout/tokens_per_s", "search/seconds"}


def _deterministic(row: dict[str, float]) -> dict[str, float]:
    """The metrics that depend only on data, weights, optimizer state and RNG (not wall clock)."""
    return {
        k: v for k, v in row.items() if not k.startswith(_NONDETERMINISTIC_PREFIXES) and k not in _NONDETERMINISTIC_KEYS
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.java
@pytest.mark.slow
def test_cli_two_process_gloo(workdir: Path, toy_index: Path) -> None:
    argv = _argv(workdir, toy_index, "ddp", "--until-step", "2", "--set", "ppo.mini_batch_size=2")
    proc = subprocess.run(  # noqa: S603 -- this interpreter, fixed module
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "2",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            str(_free_port()),
            "-m",
            "drbm25.train",
            *argv,
        ],
        env=_subprocess_env(workdir),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == EXIT_OK, proc.stdout[-4000:] + proc.stderr[-4000:]
    run_dir = workdir / "out" / "ddp"
    assert (run_dir / "step_000002" / "meta.json").is_file()
    steps = {rank: [r["step"] for r in _log_records(run_dir, rank) if r.get("message") == "batch"] for rank in (0, 1)}
    assert steps == {0: [0, 1], 1: [0, 1]}
    # Rank 0 alone writes checkpoints: each save logs its path on the writing rank only.
    saves = {
        rank: [r for r in _log_records(run_dir, rank) if str(r["message"]).startswith("checkpoint")] for rank in (0, 1)
    }
    assert len(saves[0]) == 1
    assert saves[1] == []
    # Validation is sharded across the two ranks and all-reduced: both log the same scores.
    validations = {
        rank: [r["message"] for r in _log_records(run_dir, rank) if str(r["message"]).startswith("validation")]
        for rank in (0, 1)
    }
    assert len(validations[0]) == 1
    assert validations[0] == validations[1]
    assert (run_dir / "best" / "meta.json").is_file()


@pytest.mark.java
@pytest.mark.slow
def test_cli_exits_3_when_deadline_passed(workdir: Path, toy_index: Path) -> None:
    code = _main(_argv(workdir, toy_index, "late", "--deadline-unix", str(time.time())))
    assert code == EXIT_BUDGET
    run_dir = workdir / "out" / "late"
    assert (run_dir / "exit_code").read_text().strip() == "3"
    assert not (run_dir / "metrics.jsonl").exists()  # stopped before the first step
    assert any(str(r["message"]).startswith("budget deadline reached") for r in _log_records(run_dir))


@pytest.mark.java
@pytest.mark.slow
def test_cli_stops_at_checkpoint_boundary_when_budget_runs_out(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # should_stop's arithmetic is tested in test_budget; here: it is consulted at startup and at
    # each checkpoint boundary, with the observed chunk time, and a True stops the run with code 3.
    calls: list[dict[str, float]] = []

    def stop_on_second_call(**kwargs: float) -> bool:
        calls.append(kwargs)
        return len(calls) == 2

    monkeypatch.setattr(train_module, "should_stop", stop_on_second_call)
    deadline = time.time() + 3600
    code = _main(_argv(workdir, toy_index, "budget", "--deadline-unix", str(deadline), "--reserve-s", "60"))
    assert code == EXIT_BUDGET
    assert len(calls) == 2
    assert calls[0]["mean_chunk_s"] == pytest.approx(0.0)  # startup: nothing timed yet
    assert calls[1]["mean_chunk_s"] > 0.0
    assert calls[1]["deadline_unix"] == pytest.approx(deadline)
    assert calls[1]["reserve_s"] == pytest.approx(60.0)
    run_dir = workdir / "out" / "budget"
    assert (run_dir / "latest").read_text().strip() == "step_000002"  # ckpt_every = 2
    assert (run_dir / "exit_code").read_text().strip() == "3"
    rows = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == 2
    assert json.loads(rows[-1])["budget/projected_steps"] > 2


@pytest.mark.java
def test_cli_sweeps_stale_checkpoint_temps_at_startup(workdir: Path, toy_index: Path) -> None:
    run_dir = workdir / "out" / "stale"
    (run_dir / ".tmp-step_000002-999").mkdir(parents=True)
    (run_dir / ".tmp-step_000002-999" / "model.safetensors").write_bytes(b"partial")
    (run_dir / ".tmp-latest-999").write_text("step_000002\n")
    assert _main(_argv(workdir, toy_index, "stale", "--deadline-unix", "0")) == EXIT_BUDGET
    assert list(run_dir.glob(".tmp-*")) == []
    removed = [r["message"] for r in _log_records(run_dir) if str(r["message"]).startswith("removed stale")]
    assert len(removed) == 2


@pytest.mark.java
def test_cli_fails_fast_on_wrong_index_size(workdir: Path, toy_index: Path) -> None:
    code = _main(_argv(workdir, toy_index, "badindex", "--set", "retrieval.expected_docs=5"))
    assert code == EXIT_ERROR
    run_dir = workdir / "out" / "badindex"
    errors = [r for r in _log_records(run_dir) if r["level"] == "ERROR"]
    assert len(errors) == 1
    assert "expected 5 docs, found 1002" in json.dumps(errors[0])
    assert list(run_dir.glob("step_*")) == []
    assert (run_dir / "exit_code").read_text().strip() == "2"  # failed with an exception, not killed


@pytest.mark.java
@pytest.mark.slow
def test_cli_writes_metrics_jsonl_readable_by_spike_monitor(workdir: Path, toy_index: Path) -> None:
    assert _main(_argv(workdir, toy_index, "spikes", "--until-step", "3")) == EXIT_OK
    metrics = workdir / "out" / "spikes" / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    # Step 3 is not a ckpt_every boundary, so this checkpoint is the one written at exit.
    assert (workdir / "out" / "spikes" / "latest").read_text().strip() == "step_000003"
    assert all(isinstance(v, int | float) for row in rows for v in row.values())
    proc = subprocess.run(  # noqa: S603 -- this interpreter, a vendored script
        [sys.executable, str(REPO / "scripts" / "spike_monitor.py"), str(metrics), "--window", "200"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "loss.spike_score_pct" in proc.stdout
    assert "grad_norm.spike_score_pct" in proc.stdout


@pytest.fixture
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    sent: list[tuple[str, str]] = []

    def record(run: object, title: str, text: str, level: str) -> None:
        assert run is not None  # rank 0 always has the W&B run
        sent.append((level, title))

    monkeypatch.setattr(train_module, "_alert", record)
    return sent


@pytest.mark.java
def test_cli_alerts_on_completion_and_budget_stop(
    workdir: Path, toy_index: Path, alerts: list[tuple[str, str]]
) -> None:
    assert _main(_argv(workdir, toy_index, "done", "--until-step", "1")) == EXIT_OK
    assert _main(_argv(workdir, toy_index, "stop", "--deadline-unix", "0")) == EXIT_BUDGET
    assert alerts == [("INFO", "done: finished"), ("WARN", "stop: budget stop")]


@pytest.mark.java
def test_cli_alerts_and_exits_1_on_nonfinite_error(
    workdir: Path, toy_index: Path, alerts: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_: object, **__: object) -> dict[str, float]:
        raise NonFiniteError("non-finite logprobs_old at step 0")

    monkeypatch.setattr(PPOTrainer, "step", explode)
    assert _main(_argv(workdir, toy_index, "nan")) == EXIT_ERROR
    assert alerts == [("ERROR", "nan: training aborted")]
    assert (workdir / "out" / "nan" / "exit_code").read_text().strip() == "2"
    assert list((workdir / "out" / "nan").glob("step_*")) == []  # no checkpoint of a broken state


@pytest.mark.java
def test_exit_code_reads_1_while_the_run_is_in_progress(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A previous chunk's 0 must not survive a run that is killed before main's finally.
    run_dir = workdir / "out" / "killed"
    run_dir.mkdir(parents=True)
    (run_dir / "exit_code").write_text("0\n")
    during: list[str] = []

    def fake_train(*_: object) -> int:
        during.append((run_dir / "exit_code").read_text().strip())
        return EXIT_OK

    monkeypatch.setattr(train_module, "_train", fake_train)
    assert _main(_argv(workdir, toy_index, "killed")) == EXIT_OK
    assert during == ["1"]
    assert (run_dir / "exit_code").read_text().strip() == "0"


@pytest.mark.java
def test_exit_code_file_reads_2_when_main_fails_with_an_exception(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1 = started but never finished (killed); 2 = ended by an exception. Chain launchers rely on the split.
    def boom(*_: object) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(train_module, "_train", boom)
    assert _main(_argv(workdir, toy_index, "failed")) == EXIT_ERROR
    assert (workdir / "out" / "failed" / "exit_code").read_text().strip() == "2"


def test_cli_joins_the_process_group_with_the_configured_timeout(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[timedelta] = []

    def fake_init(*, timeout: timedelta) -> DistInfo:
        seen.append(timeout)
        return DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))

    monkeypatch.setattr(train_module, "init_distributed", fake_init)
    monkeypatch.setattr(train_module, "_train", lambda *_: EXIT_OK)
    assert _main(_argv(workdir, toy_index, "pg", "--set", "run.dist_timeout_min=7")) == EXIT_OK
    assert seen == [timedelta(minutes=7)]


class _FakeRun:
    id = "fake-run"

    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []

    def log(self, data: dict[str, Any]) -> None:
        self.logged.append(data)

    def alert(self, **_: object) -> None:
        pass


@pytest.mark.java
def test_cli_logs_histograms_to_wandb_on_deep_log_steps_only(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()

    @contextmanager
    def fake_open_run(*_: object, **__: object) -> Generator[_FakeRun]:
        yield fake

    monkeypatch.setattr(train_module, "open_run", fake_open_run)
    assert _main(_argv(workdir, toy_index, "hist", "--until-step", "2")) == EXIT_OK
    steps = [entry for entry in fake.logged if "ppo/policy_loss" in entry]
    assert [entry[STEP_METRIC] for entry in steps] == [0, 1]
    hist_keys = ("hist/kl_ref", "hist/advantages", "hist/response_len")
    assert all(isinstance(steps[0][k], wandb.Histogram) for k in hist_keys)  # step 0: deep_log_every = 10
    assert [k for k in steps[1] if k.startswith("hist/")] == []
    rows = (workdir / "out" / "hist" / "metrics.jsonl").read_text()
    assert "hist/" not in rows  # histograms go to W&B only


def _full_config(workdir: Path, **model: str) -> Path:
    raw = yaml.safe_load(SMOKE.read_text())
    raw["model"]["train_mode"] = "full"
    raw["model"].update(model)
    raw["lora"] = None
    path = workdir / "full.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _first_error(run_dir: Path) -> str:
    return json.dumps(next(r for r in _log_records(run_dir) if r["level"] == "ERROR"))


@pytest.mark.java
def test_cli_full_layout_needs_torchrun(workdir: Path, toy_index: Path) -> None:
    argv = _argv(workdir, toy_index, "full")
    argv[argv.index(str(SMOKE))] = str(_full_config(workdir))
    assert _main(argv) == EXIT_ERROR
    assert "launch with torchrun" in _first_error(workdir / "out" / "full")


@pytest.mark.java
def test_cli_full_layout_rejects_fp16(workdir: Path, toy_index: Path) -> None:
    argv = _argv(workdir, toy_index, "full16")
    argv[argv.index(str(SMOKE))] = str(_full_config(workdir, precision="fp16"))
    assert _main(argv) == EXIT_ERROR
    assert "bf16 or fp32" in _first_error(workdir / "out" / "full16")


@pytest.mark.java
def test_cli_vllm_backend_needs_cuda(workdir: Path, toy_index: Path) -> None:
    assert _main(_argv(workdir, toy_index, "cpuvllm", "--set", "rollout.backend=vllm")) == EXIT_ERROR
    assert "needs CUDA" in _first_error(workdir / "out" / "cpuvllm")


def _torchrun(workdir: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- this interpreter, fixed module
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "2",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            str(_free_port()),
            "-m",
            "drbm25.train",
            *argv,
        ],
        env=_subprocess_env(workdir),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.mark.java
@pytest.mark.slow
def test_cli_full_layout_two_ranks_writes_sharded_checkpoints_and_resumes(workdir: Path, toy_index: Path) -> None:
    full = str(_full_config(workdir))

    def argv(until: int) -> list[str]:
        out = _argv(workdir, toy_index, "fsdp", "--until-step", str(until), "--set", "ppo.mini_batch_size=2")
        out[out.index(str(SMOKE))] = full
        return out

    first = _torchrun(workdir, argv(2))
    assert first.returncode == EXIT_OK, first.stdout[-4000:] + first.stderr[-4000:]
    run_dir = workdir / "out" / "fsdp"
    meta = json.loads((run_dir / "step_000002" / "meta.json").read_text())
    assert meta["layout"] == "sharded"
    assert meta["train_state"]["reward_mode"] == "normal"
    assert sorted(p.name for p in (run_dir / "step_000002" / "dcp").iterdir()) == [
        ".metadata",
        "__0_0.distcp",
        "__1_0.distcp",
    ]
    second = _torchrun(workdir, argv(3))
    assert second.returncode == EXIT_OK, second.stdout[-4000:] + second.stderr[-4000:]
    assert (run_dir / "latest").read_text().strip() == "step_000003"
    assert (run_dir / "exit_code").read_text().strip() == "0"
    records = {rank: _log_records(run_dir, rank) for rank in (0, 1)}
    steps = {rank: [r["step"] for r in recs if r.get("message") == "batch"] for rank, recs in records.items()}
    assert steps == {0: [0, 1, 2], 1: [0, 1, 2]}
    resumed = {rank: any(str(r["message"]).startswith("resumed from") for r in recs) for rank, recs in records.items()}
    assert resumed == {0: True, 1: True}
    # val_every = 2: every rank generates the validation rollouts after step index 1 (FSDP collectives).
    validated = {
        rank: [r["step"] for r in recs if str(r["message"]).startswith("validation")] for rank, recs in records.items()
    }
    assert validated == {0: [1], 1: [1]}
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert all(math.isfinite(row["ppo/policy_loss"]) for row in rows)
    # best/: the full actor (HF names) gathered from both ranks at the validated step, written once.
    best_meta, best_weights = load_best(run_dir / "best")
    assert (best_meta.step, best_meta.layout, best_meta.reward_mode) == (2, "full", "normal")
    actor_at_2 = build_tiny_qwen2(seed=0)
    load_sharded_actor(run_dir / "step_000002", actor_at_2)
    expected = actor_at_2.state_dict()
    assert sorted(best_weights) == sorted(expected)
    for key, tensor in expected.items():
        assert torch.equal(best_weights[key], tensor), key


@pytest.mark.java
def test_cli_validates_on_held_out_train_queries_and_never_trains_on_them(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained: list[str] = []
    validated: list[str] = []
    real_step, real_validate = PPOTrainer.step, PPOTrainer.validate_with_samples

    def record_step(self: PPOTrainer, examples: list[Example], *, step: int) -> dict[str, float]:
        trained.extend(e.qid for e in examples)
        return real_step(self, examples, step=step)

    def record_validate(self: PPOTrainer, examples: list[Example]) -> Any:
        validated.extend(e.qid for e in examples)
        return real_validate(self, examples)

    monkeypatch.setattr(PPOTrainer, "step", record_step)
    monkeypatch.setattr(PPOTrainer, "validate_with_samples", record_validate)
    assert _main(_argv(workdir, toy_index, "heldout", "--until-step", "4")) == EXIT_OK
    _, val = split_train(load_examples(workdir / "data" / TRAIN_FILE), val_size=4, seed=1234)
    assert sorted(set(validated)) == sorted(e.qid for e in val)
    assert all(q.startswith("train-") for q in validated)  # never dev (the test set)
    assert set(trained).isdisjoint(validated)
    assert len(set(trained)) == N_TRAIN - 4  # 4 steps x 4 = 16 rows: one full epoch of the rest


# --------------------------------------------------------------------------- best/ selection
def _scripted_validation(
    monkeypatch: pytest.MonkeyPatch, normal: dict[int, float], count_once: dict[int, float]
) -> list[int]:
    """Replace validation with fixed scores keyed by completed steps; returns the steps validated."""
    seen: list[int] = []

    def fake_validate(run: object, trainer: object, val: object, step: int) -> dict[str, float]:
        done = step + 1
        seen.append(done)
        return {"val/ndcg_normal": normal[done], "val/ndcg_count_once": count_once[done]}

    monkeypatch.setattr(train_module, "_validate", fake_validate)
    return seen


def _best(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "best" / "meta.json").read_text())


@pytest.mark.java
def test_cli_keeps_the_step_with_the_highest_validation_score(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A normal-reward run selects on val/ndcg_normal and ignores the count_once score.
    _scripted_validation(monkeypatch, {1: 0.2, 2: 0.5, 3: 0.4}, {1: 0.9, 2: 0.1, 3: 0.95})
    argv = _argv(workdir, toy_index, "best", "--until-step", "3", "--set", "run.val_every=1")
    assert _main(argv) == EXIT_OK
    meta = _best(workdir / "out" / "best")
    assert meta["step"] == 2
    assert meta["val_ndcg_normal"] == pytest.approx(0.5)
    assert meta["val_ndcg_count_once"] == pytest.approx(0.1)
    assert meta["selection_metric"] == "val/ndcg_normal"
    assert meta["reward_mode"] == "normal"
    assert meta["layout"] == "lora"
    manifest = json.loads((workdir / "out" / "best" / "run_manifest.json").read_text())
    assert meta["config_sha256"] == manifest["config_sha256"]
    assert meta["seeds"] == manifest["seeds"]
    assert meta["git"] == manifest["git"]  # the code version behind the selected policy


@pytest.mark.java
def test_cli_best_tie_keeps_the_earlier_step(workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted_validation(monkeypatch, {1: 0.2, 2: 0.5, 3: 0.5}, {1: 0.0, 2: 0.0, 3: 0.0})
    assert _main(_argv(workdir, toy_index, "tie", "--until-step", "3", "--set", "run.val_every=1")) == EXIT_OK
    assert _best(workdir / "out" / "tie")["step"] == 2


@pytest.mark.java
def test_cli_count_once_run_selects_on_its_own_metric(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_validation(monkeypatch, {1: 0.9, 2: 0.1, 3: 0.2}, {1: 0.1, 2: 0.3, 3: 0.2})
    argv = _argv(workdir, toy_index, "co", "--until-step", "3", "--set", "run.val_every=1")
    argv[argv.index("--reward-mode") + 1] = "count_once"
    assert _main(argv) == EXIT_OK
    meta = _best(workdir / "out" / "co")
    assert (meta["step"], meta["selection_metric"]) == (2, "val/ndcg_count_once")


@pytest.mark.java
def test_cli_validates_at_the_final_step_even_off_the_val_every_grid(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _scripted_validation(monkeypatch, {2: 0.1, 3: 0.3}, {2: 0.0, 3: 0.0})
    argv = _argv(workdir, toy_index, "final", "--set", "run.total_steps=3")
    assert _main(argv) == EXIT_OK
    assert seen == [2, 3]  # val_every = 2, total_steps = 3
    assert _best(workdir / "out" / "final")["step"] == 3


@pytest.mark.java
def test_cli_resume_restores_best_so_far_and_a_lower_score_does_not_replace_it(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_validation(monkeypatch, {1: 0.2, 2: 0.5}, {1: 0.0, 2: 0.0})
    assert _main(_argv(workdir, toy_index, "rb", "--until-step", "2", "--set", "run.val_every=1")) == EXIT_OK
    weights_before = (workdir / "out" / "rb" / "best" / "model.safetensors").read_bytes()
    seen = _scripted_validation(monkeypatch, {3: 0.4, 4: 0.45}, {3: 0.0, 4: 0.0})
    assert _main(_argv(workdir, toy_index, "rb", "--until-step", "4", "--set", "run.val_every=1")) == EXIT_OK
    assert seen == [3, 4]  # resumed at step 2: no re-evaluation of the stored best
    meta = _best(workdir / "out" / "rb")
    assert (meta["step"], meta["val_ndcg_normal"]) == (2, pytest.approx(0.5))
    assert (workdir / "out" / "rb" / "best" / "model.safetensors").read_bytes() == weights_before
    assert any(str(r["message"]).startswith("best so far") for r in _log_records(workdir / "out" / "rb"))


@pytest.mark.java
def test_cli_best_weights_are_the_actor_adapter_at_the_best_step(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safetensors.torch import load_file

    _scripted_validation(monkeypatch, {1: 0.2, 2: 0.5, 3: 0.4}, {1: 0.0, 2: 0.0, 3: 0.0})
    argv = _argv(
        workdir, toy_index, "w", "--until-step", "3", "--set", "run.val_every=1", "--set", "run.ckpt_every=1",
        "--set", "run.keep_ckpts=5",
    )
    assert _main(argv) == EXIT_OK
    run_dir = workdir / "out" / "w"
    best = load_file(str(run_dir / "best" / "model.safetensors"))
    at_step = load_file(str(run_dir / "step_000002" / "model.safetensors"))
    assert len(best) == 28  # tiny model: 2 layers x 7 targets x (A, B)
    assert all(".actor." in k for k in best)
    assert sorted(best) == sorted(k for k in at_step if ".lora_" in k and ".actor." in k)
    for key, tensor in best.items():
        assert torch.equal(tensor, at_step[key]), key
    later = load_file(str(run_dir / "step_000003" / "model.safetensors"))
    assert any(not torch.equal(best[k], later[k]) for k in best)  # step 3 trained further


# --------------------------------------------------------------------------- reproducibility manifest
@pytest.mark.java
def test_cli_writes_the_manifest_and_sends_it_to_wandb_meta(
    workdir: Path, toy_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    @contextmanager
    def fake_open_run(*_: object, **kwargs: Any) -> Generator[_FakeRun]:
        captured.update(kwargs["extra_config"])
        yield _FakeRun()

    monkeypatch.setattr(train_module, "open_run", fake_open_run)
    assert _main(_argv(workdir, toy_index, "man", "--until-step", "1")) == EXIT_OK
    manifest = json.loads((workdir / "out" / "man" / "run_manifest.json").read_text())
    assert manifest["seeds"]["run_seed"] == 1234
    assert manifest["seeds"]["split_seed"] == 1234
    assert manifest["seeds"]["sampling_seeds"] == [1234]
    assert manifest["config"]["data"]["val_size"] == 4
    assert manifest["index"]["checksum"] == read_index_info(toy_index).checksum
    assert manifest["effective_micro_batch_size"] == 1
    assert manifest["versions"]["torch"].startswith("2.10.0")
    assert manifest["resumed_at"] == []
    assert captured == manifest


@pytest.mark.java
def test_cli_resume_appends_to_the_manifest_and_refuses_a_changed_seed(workdir: Path, toy_index: Path) -> None:
    assert _main(_argv(workdir, toy_index, "seeds", "--until-step", "1")) == EXIT_OK
    run_dir = workdir / "out" / "seeds"
    first = json.loads((run_dir / "run_manifest.json").read_text())
    assert _main(_argv(workdir, toy_index, "seeds", "--until-step", "2")) == EXIT_OK
    second = json.loads((run_dir / "run_manifest.json").read_text())
    assert second["seeds"] == first["seeds"]
    assert second["start_time_utc"] == first["start_time_utc"]
    assert [r["step"] for r in second["resumed_at"]] == [1]
    assert _main(_argv(workdir, toy_index, "seeds", "--until-step", "3", "--set", "run.seed=99")) == EXIT_ERROR
    assert (run_dir / "exit_code").read_text().strip() == "2"
    assert "seed" in _first_error(run_dir)
    assert json.loads((run_dir / "run_manifest.json").read_text()) == second


# --------------------------------------------------------------------------- startup checks
def test_check_java_rejects_missing_binary(tmp_path: Path) -> None:
    with pytest.raises(StartupCheckError, match="cannot run"):
        check_java(str(tmp_path / "no-java"))


def test_check_java_rejects_other_major_version(tmp_path: Path) -> None:
    fake = tmp_path / "java"
    fake.write_text("#!/bin/sh\necho 'openjdk version \"18.0.1\" 2022-04-19' >&2\n")
    fake.chmod(0o755)
    with pytest.raises(StartupCheckError, match=r"Java 21 is required.*reported 18"):
        check_java(str(fake))


def test_check_java_accepts_21(tmp_path: Path) -> None:
    fake = tmp_path / "java"
    fake.write_text("#!/bin/sh\necho 'openjdk version \"21.0.9\" 2025-10-21' >&2\n")
    fake.chmod(0o755)
    assert check_java(str(fake)) is None


def test_check_java_rejects_failing_binary(tmp_path: Path) -> None:
    fake = tmp_path / "java"
    fake.write_text("#!/bin/sh\necho 'openjdk version \"21.0.9\"' >&2\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(StartupCheckError, match="cannot run"):
        check_java(str(fake))


@pytest.mark.java
def test_check_index_returns_the_index_checksum(toy_index: Path) -> None:
    cfg = load_config(SMOKE, {"retrieval.index_dir": str(toy_index)})
    assert check_index(cfg) == read_index_info(toy_index).checksum


def test_check_tokenizer_rejects_pad_equal_to_eos() -> None:
    tokenizer = build_tiny_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    with pytest.raises(StartupCheckError, match="must be set and differ from eos_token_id"):
        check_tokenizer(tokenizer)


def test_cli_rejects_bad_run_name(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--config", str(SMOKE), "--reward-mode", "normal", "--run-name", "../escape"])
    assert excinfo.value.code == 2
    assert "run name must match" in capsys.readouterr().err
