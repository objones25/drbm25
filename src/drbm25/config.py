"""Typed, frozen run configuration loaded from YAML, with dotted-key overrides.

`load_config` parses and type-checks (unknown keys, wrong types and bad enum
values raise `ConfigError`). `validate_for_runtime` checks the cross-field and
hardware rules that need to know the device and world size.
"""

from __future__ import annotations

import math
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml

ScoringMode = Literal["normal", "count_once"]
Precision = Literal["fp16", "bf16", "fp32"]
TrainMode = Literal["lora", "full"]
RolloutBackend = Literal["hf", "vllm"]

_BF16_MIN_CAPABILITY = (8, 0)


class ConfigError(ValueError):
    """The configuration is malformed or inconsistent."""


@dataclass(frozen=True)
class ModelCfg:
    name: str
    revision: str
    precision: Precision
    train_mode: TrainMode
    gradient_checkpointing: bool


@dataclass(frozen=True)
class LoraCfg:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class PPOCfg:
    batch_size: int
    mini_batch_size: int
    micro_batch_size: int
    ppo_epochs: int
    clip_ratio: float
    value_clip: float
    entropy_coef: float
    kl_coef: float
    gamma: float
    lam: float
    grad_clip: float
    actor_lr: float
    critic_lr: float


@dataclass(frozen=True)
class RolloutCfg:
    backend: RolloutBackend
    temperature: float
    top_p: float
    max_prompt_tokens: int
    max_new_tokens: int
    generation_batch_size: int
    vllm_gpu_memory_utilization: float


@dataclass(frozen=True)
class RetrievalCfg:
    index_dir: Path
    k1: float
    b: float
    train_k: int
    eval_k: int
    search_threads: int
    search_timeout_s: float


@dataclass(frozen=True)
class DataCfg:
    hf_dataset: str
    hf_revision: str
    data_dir: Path
    val_size: int


@dataclass(frozen=True)
class RunCfg:
    output_dir: Path
    total_steps: int
    chunk_steps: int
    val_every: int
    ckpt_every: int
    keep_ckpts: int
    deep_log_every: int
    seed: int
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str


@dataclass(frozen=True)
class Config:
    model: ModelCfg
    lora: LoraCfg | None
    ppo: PPOCfg
    rollout: RolloutCfg
    retrieval: RetrievalCfg
    data: DataCfg
    run: RunCfg


def _is_union(tp: Any) -> bool:
    return get_origin(tp) in (typing.Union, types.UnionType)


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Convert a parsed YAML value to type `tp`, strictly (no bool-as-int, no str-as-number)."""
    if _is_union(tp):
        args = get_args(tp)
        if value is None and type(None) in args:
            return None
        (inner,) = (a for a in args if a is not type(None))
        return _coerce(value, inner, where)
    if is_dataclass(tp):
        return _build(tp, value, where)
    if get_origin(tp) is Literal:
        if value not in get_args(tp):
            raise ConfigError(f"{where}: {value!r} is not one of {list(get_args(tp))}")
        return value
    if get_origin(tp) is tuple:
        if not isinstance(value, list | tuple) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{where}: expected a list of strings, got {value!r}")
        return tuple(value)
    if tp is bool:
        ok = isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif tp is float:
        ok = isinstance(value, int | float) and not isinstance(value, bool)
    elif tp is str or tp is Path:
        ok = isinstance(value, str)
    else:
        raise TypeError(f"{where}: unsupported config field type {tp!r}")
    if not ok:
        raise ConfigError(f"{where}: expected {getattr(tp, '__name__', tp)}, got {value!r}")
    return Path(value) if tp is Path else float(value) if tp is float else value


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {data!r}")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    for key in data:
        if key not in names:
            raise ConfigError(f"unknown key '{_join(where, key)}'")
    kwargs = {}
    for name in names:
        tp = hints[name]
        if name not in data:
            if _is_union(tp) and type(None) in get_args(tp):
                kwargs[name] = None
                continue
            raise ConfigError(f"missing key '{_join(where, name)}'")
        kwargs[name] = _coerce(data[name], tp, _join(where, name))
    return cls(**kwargs)


def _join(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _parse_override(text: str, tp: Any, where: str) -> Any:
    """Turn an override string into a native value for type `tp`."""
    if _is_union(tp):
        if text == "null":
            return None
        (tp,) = (a for a in get_args(tp) if a is not type(None))
    try:
        if tp is bool:
            if text not in ("true", "false"):
                raise ValueError(text)
            return text == "true"
        if tp is int:
            return int(text)
        if tp is float:
            return float(text)
        if get_origin(tp) is tuple:
            return [t for t in text.split(",") if t]
    except ValueError:
        raise ConfigError(f"{where}: cannot parse {text!r} as {getattr(tp, '__name__', tp)}") from None
    return text  # str, Path, Literal: validated by _coerce


def _apply_override(raw: dict[str, Any], dotted: str, text: str) -> None:
    node: dict[str, Any] = raw
    cls: type = Config
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        hints = get_type_hints(cls)
        if part not in hints:
            raise ConfigError(f"unknown key '{dotted}'")
        tp = hints[part]
        if i == len(parts) - 1:
            if is_dataclass(tp) or (_is_union(tp) and any(is_dataclass(a) for a in get_args(tp))):
                raise ConfigError(f"'{dotted}' is a section; override one of its fields")
            node[part] = _parse_override(text, tp, dotted)
            return
        if _is_union(tp):
            (tp,) = (a for a in get_args(tp) if a is not type(None))
        if not is_dataclass(tp):
            raise ConfigError(f"unknown key '{dotted}'")
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node, cls = child, tp


def load_config(path: Path, overrides: Mapping[str, str] = MappingProxyType({})) -> Config:
    """Load a YAML profile, apply `{"section.field": "value"}` overrides, return a typed Config."""
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    for dotted, text in overrides.items():
        _apply_override(raw, dotted, text)
    return _build(Config, raw, "")


def _require_finite_floats(section: Any, where: str = "") -> None:
    """NaN and inf pass ordinary `<`/`<=` range checks, so reject them explicitly."""
    for f in fields(section):
        value = getattr(section, f.name)
        if is_dataclass(value):
            _require_finite_floats(value, _join(where, f.name))
        elif isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"{_join(where, f.name)} must be finite, got {value}")


def validate_for_runtime(cfg: Config, *, device_capability: tuple[int, int] | None, world_size: int) -> None:
    """Check cross-field and hardware rules; raise ConfigError on the first violation."""
    if world_size < 1:
        raise ConfigError(f"world_size must be >= 1, got {world_size}")
    p = cfg.ppo
    _require_finite_floats(cfg)
    for name in ("batch_size", "mini_batch_size", "micro_batch_size"):
        if getattr(p, name) < 1:
            raise ConfigError(f"{name} must be >= 1, got {getattr(p, name)}")
    if cfg.lora is not None and not cfg.lora.target_modules:
        raise ConfigError("lora.target_modules must not be empty")
    if cfg.model.precision == "bf16" and device_capability is not None and device_capability < _BF16_MIN_CAPABILITY:
        raise ConfigError(
            f"bf16 requires compute capability >= 8.0, got {device_capability[0]}.{device_capability[1]}; "
            "use fp16 on V100"
        )
    if (cfg.model.train_mode == "lora") != (cfg.lora is not None):
        raise ConfigError(f"model.train_mode '{cfg.model.train_mode}' needs lora section present iff mode is 'lora'")
    if p.batch_size % world_size:
        raise ConfigError(f"batch_size {p.batch_size} is not divisible by world_size {world_size}")
    if p.mini_batch_size % world_size:
        raise ConfigError(f"mini_batch_size {p.mini_batch_size} is not divisible by world_size {world_size}")
    if p.batch_size % p.mini_batch_size:
        raise ConfigError(f"batch_size {p.batch_size} is not a multiple of mini_batch_size {p.mini_batch_size}")
    per_rank = p.mini_batch_size // world_size
    if per_rank % p.micro_batch_size:
        raise ConfigError(
            f"micro_batch_size {p.micro_batch_size} does not divide per-rank mini_batch_size {per_rank}"
        )
    if p.actor_lr <= 0:
        raise ConfigError(f"actor_lr must be > 0, got {p.actor_lr}")
    if p.critic_lr <= 0:
        raise ConfigError(f"critic_lr must be > 0, got {p.critic_lr}")
    if not 0 < p.clip_ratio < 1:
        raise ConfigError(f"clip_ratio must be in (0, 1), got {p.clip_ratio}")
    if p.kl_coef < 0:
        raise ConfigError(f"kl_coef must be >= 0, got {p.kl_coef}")
