"""open_run: the project's W&B wrapper. Runs offline; no network."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import wandb
from wandb.sdk.wandb_run import Run

from drbm25.config import Config
from drbm25.obs.runs import open_run


@pytest.fixture(autouse=True)
def offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")


def test_open_run_offline_records_config_and_job_type(cfg: Config) -> None:
    with open_run(cfg, run_name="t", reward_mode="normal", resume_id=None, extra_config={"git_sha": "abc123"}) as run:
        assert run.config["ppo"]["batch_size"] == cfg.ppo.batch_size
        assert run.config["meta"]["git_sha"] == "abc123"
        assert run.config["retrieval"]["index_dir"] == str(cfg.retrieval.index_dir)
        assert run.job_type == "normal"
        assert run.group == cfg.run.wandb_group
        assert run.project == cfg.run.wandb_project
        assert run.name == "t"
    assert wandb.run is None


def test_open_run_uses_count_once_as_job_type(cfg: Config) -> None:
    with open_run(cfg, run_name="t", reward_mode="count_once", resume_id=None, extra_config={}) as run:
        assert run.job_type == "count_once"


@pytest.mark.parametrize(
    ("extra", "path"),
    [
        ({"hf_token": "x"}, "meta.hf_token"),
        ({"nested": {"api_key": "x"}}, "meta.nested.api_key"),
        ({"Client_SECRET": "x"}, "meta.Client_SECRET"),
        ({"password": "x"}, "meta.password"),
    ],
)
def test_open_run_rejects_secret_like_config_key(cfg: Config, extra: dict[str, object], path: str) -> None:
    with (
        pytest.raises(ValueError, match=rf"secret-like key '{path}'"),
        open_run(cfg, run_name="t", reward_mode="normal", resume_id=None, extra_config=extra),
    ):
        pytest.fail("block must not run")


def test_open_run_accepts_token_count_fields(cfg: Config) -> None:
    """max_prompt_tokens/max_new_tokens are real Config fields, not credentials."""
    with open_run(cfg, run_name="t", reward_mode="normal", resume_id=None, extra_config={"n_tokens": 3}) as run:
        assert run.config["rollout"]["max_new_tokens"] == cfg.rollout.max_new_tokens


def test_open_run_marks_failed_on_exception(cfg: Config) -> None:
    with (
        pytest.raises(RuntimeError, match="boom"),
        open_run(cfg, run_name="t", reward_mode="normal", resume_id=None, extra_config={}),
    ):
        raise RuntimeError("boom")
    assert wandb.run is None


def test_open_run_resume_id_must_exist_offline_run_uses_it(cfg: Config) -> None:
    """With resume_id the run id is that id, and the run is started with resume='must'."""
    with open_run(cfg, run_name="t", reward_mode="normal", resume_id="abc12345", extra_config={}) as run:
        assert run.id == "abc12345"
        assert run.settings.resume == "must"


def test_open_run_without_resume_id_uses_never(cfg: Config) -> None:
    with open_run(cfg, run_name="t", reward_mode="normal", resume_id=None, extra_config={}) as run:
        assert run.settings.resume == "never"


def test_open_run_defines_train_step_as_x_axis(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []
    original = Run.define_metric

    def spy(self: Run, name: str, step_metric: str | None = None, **kw: Any) -> Any:
        calls.append((name, step_metric))
        return original(self, name, step_metric=step_metric, **kw)

    monkeypatch.setattr(Run, "define_metric", spy)
    with open_run(cfg, run_name="t", reward_mode="normal", resume_id=None, extra_config={}) as run:
        run.log({"train/step": 3, "train/loss": 0.5})
        assert run.summary["train/loss"] == pytest.approx(0.5)
    assert calls == [("train/step", None), ("*", "train/step")]


def test_open_run_with_entity_none_ok(cfg: Config) -> None:
    cfg2 = replace(cfg, run=replace(cfg.run, wandb_entity=None))
    with open_run(cfg2, run_name="t", reward_mode="normal", resume_id=None, extra_config={}) as run:
        assert run.id
