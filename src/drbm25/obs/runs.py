"""Project W&B run: project/entity/group from Config, one x-axis, no secrets in config.

Built on `start_run` (offline fallback, failed-run marking, log-context binding).
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import wandb

from drbm25.config import Config, ScoringMode
from drbm25.obs.redact import is_secret_name, name_words
from drbm25.obs.wandb_run import start_run

STEP_METRIC = "train/step"


def _is_secret_key(name: str) -> bool:
    # is_secret_name matches words, so max_new_tokens (a real Config field) passes;
    # a bare "key" word is added because it is not in its word list on its own.
    return is_secret_name(name) or "key" in name_words(name)


def _plain_mapping(mapping: Mapping[object, object], path: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, item in mapping.items():
        sub = f"{path}.{key}" if path else str(key)
        if _is_secret_key(str(key)):
            raise ValueError(f"refusing to put secret-like key '{sub}' in W&B config")
        out[str(key)] = _plain(item, sub)
    return out


def _plain(value: object, path: str) -> object:
    """Convert to W&B-safe plain values (Path -> str, tuple -> list); reject secret-like keys."""
    if isinstance(value, Mapping):
        return _plain_mapping(value, path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_plain(v, path) for v in value]
    return value


def build_run_config(cfg: Config, extra_config: Mapping[str, object]) -> dict[str, object]:
    """`asdict(cfg)` plus `extra_config` under "meta", validated and made JSON-friendly."""
    return _plain_mapping({**asdict(cfg), "meta": dict(extra_config)}, "")


@contextmanager
def open_run(
    cfg: Config,
    *,
    run_name: str,
    reward_mode: ScoringMode,
    resume_id: str | None,
    extra_config: Mapping[str, object],
    job_type: str | None = None,
) -> Generator[wandb.Run]:
    """Open the W&B run for one training job; an exception in the block marks it failed.

    The W&B job_type is `reward_mode` unless `job_type` is given (the report passes "report"). `WANDB_MODE` (e.g. offline) is honoured. With `resume_id` the run is resumed with
    resume="must", otherwise started with resume="never".
    """
    config = build_run_config(cfg, extra_config)
    with start_run(
        project=cfg.run.wandb_project,
        entity=cfg.run.wandb_entity,
        group=cfg.run.wandb_group,
        job_type=reward_mode if job_type is None else job_type,
        name=run_name,
        config=config,
        run_id=resume_id,
        resume="must" if resume_id is not None else "never",
    ) as run:
        run.define_metric(STEP_METRIC)
        run.define_metric("*", step_metric=STEP_METRIC)
        yield run
