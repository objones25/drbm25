"""Atomic checkpoints and resume state for long PPO runs.

Layout::

    run_dir/
      latest                      text file naming the newest step dir
      step_000050/
        model.safetensors         model tensors
        optim.pt                  optimizer + scaler state + torch RNG tensors (weights_only)
        meta.json                 train state, python/numpy RNG, format_version (written last)

A checkpoint is built in ``.tmp-step_NNNNNN-<pid>/``, every file is fsynced, the directory is
renamed into place with ``os.replace`` and the parent directory is fsynced; only then is
``latest`` swapped (temp file + ``os.replace``). A directory is *complete* when it has
``meta.json``, so a crash at any point leaves the previous checkpoint usable.

``save_checkpoint`` is rank-agnostic: in a multi-process run the caller must call it on rank 0
only. ``numpy.random.Generator`` objects are not captured (they are not global state); keep the
data order reproducible through ``TrainState.data_cursor`` and a seed instead.
"""

import json
import os
import random
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

FORMAT_VERSION = 1
_MODEL_FILE = "model.safetensors"
_OPTIM_FILE = "optim.pt"
_META_FILE = "meta.json"
_LATEST_FILE = "latest"
_STEP_PREFIX = "step_"
_TMP_PREFIX = ".tmp-"


@dataclass(frozen=True)
class TrainState:
    step: int
    data_cursor: int
    wandb_run_id: str | None


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    train_state: TrainState
    model: dict[str, Tensor]
    optimizers: dict[str, dict[str, Any]]
    scaler: dict[str, Any] | None
    rng: dict[str, object]


def capture_rng() -> dict[str, object]:
    """Snapshot python, numpy (legacy global) and torch CPU/CUDA RNG states.

    Keys: ``python`` (JSON-compatible list), ``numpy`` (JSON-compatible dict), ``torch_cpu``
    (uint8 tensor), ``torch_cuda`` (list of uint8 tensors, one per device; empty without CUDA).
    """
    version, internal, gauss_next = random.getstate()
    name, keys, pos, has_gauss, cached = cast(
        tuple[str, np.ndarray, int, int, float], np.random.get_state()
    )
    return {
        "python": [version, list(internal), gauss_next],
        "numpy": {
            "bit_generator": name,
            "keys": keys.tolist(),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, object]) -> None:
    """Inverse of :func:`capture_rng`. Raises ``KeyError`` on a missing key."""
    py = cast(list[Any], state["python"])
    np_state = cast(dict[str, Any], state["numpy"])
    torch_cpu = cast(Tensor, state["torch_cpu"])
    torch_cuda = cast(list[Tensor], state["torch_cuda"])
    if torch_cuda and not torch.cuda.is_available():
        raise RuntimeError("checkpoint holds CUDA RNG states but CUDA is not available")
    random.setstate((py[0], tuple(py[1]), py[2]))
    np.random.set_state(
        (
            np_state["bit_generator"],
            np.array(np_state["keys"], dtype=np.uint32),
            np_state["pos"],
            np_state["has_gauss"],
            np_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(torch_cpu)
    if torch_cuda:
        torch.cuda.set_rng_state_all(torch_cuda)


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    # Directory fsync is POSIX-only; elsewhere opening a directory fails and there is nothing to do.
    if os.name != "posix":
        return
    _fsync_path(path)


def _step_name(step: int) -> str:
    return f"{_STEP_PREFIX}{step:06d}"


def _is_complete(path: Path) -> bool:
    return (path / _META_FILE).is_file()


def _complete_step_dirs(run_dir: Path) -> list[Path]:
    """Complete ``step_*`` dirs, oldest first (names sort by step)."""
    return sorted(p for p in run_dir.glob(f"{_STEP_PREFIX}*") if p.is_dir() and _is_complete(p))


def _portable(tensor: Tensor) -> Tensor:
    # clone() gives each tensor its own storage; safetensors refuses shared storage (tied weights, views).
    return tensor.detach().to("cpu").contiguous().clone()


def _write_checkpoint_files(
    tmp: Path,
    *,
    train_state: TrainState,
    model: Mapping[str, Tensor],
    optimizers: Mapping[str, dict[str, Any]],
    scaler: dict[str, Any] | None,
) -> None:
    rng = capture_rng()
    save_file({k: _portable(v) for k, v in model.items()}, str(tmp / _MODEL_FILE))
    torch.save(
        {
            "optimizers": dict(optimizers),
            "scaler": scaler,
            "rng_torch_cpu": rng["torch_cpu"],
            "rng_torch_cuda": rng["torch_cuda"],
        },
        tmp / _OPTIM_FILE,
    )
    meta = {
        "format_version": FORMAT_VERSION,
        "train_state": {
            "step": train_state.step,
            "data_cursor": train_state.data_cursor,
            "wandb_run_id": train_state.wandb_run_id,
        },
        "rng_python": rng["python"],
        "rng_numpy": rng["numpy"],
    }
    # meta.json is written last: its presence marks the directory complete.
    (tmp / _META_FILE).write_text(json.dumps(meta))
    for name in (_MODEL_FILE, _OPTIM_FILE, _META_FILE):
        _fsync_path(tmp / name)
    _fsync_dir(tmp)


def _remove_dir(path: Path) -> None:
    """Delete a checkpoint dir so a crash mid-delete never leaves a dir that looks complete.

    ``rmtree`` may remove ``meta.json`` last, so the dir is first renamed (atomic) to a
    ``.tmp-del-*`` name that every reader ignores, then deleted.
    """
    aside = path.parent / f"{_TMP_PREFIX}del-{path.name}-{os.getpid()}"
    if aside.exists():
        shutil.rmtree(aside)
    os.replace(path, aside)
    shutil.rmtree(aside)


def _write_latest(run_dir: Path, name: str) -> None:
    tmp = run_dir / f"{_TMP_PREFIX}{_LATEST_FILE}-{os.getpid()}"
    tmp.write_text(name + "\n")
    _fsync_path(tmp)
    os.replace(tmp, run_dir / _LATEST_FILE)
    _fsync_dir(run_dir)


def save_checkpoint(
    run_dir: Path,
    *,
    train_state: TrainState,
    model: Mapping[str, Tensor],
    optimizers: Mapping[str, dict[str, Any]],
    scaler: dict[str, Any] | None,
    keep: int,
) -> Path:
    """Write ``run_dir/step_NNNNNN`` atomically, point ``latest`` at it, prune old ones.

    Rank-agnostic: call on rank 0 only. Saving a step that already exists replaces it.
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    run_dir.mkdir(parents=True, exist_ok=True)
    name = _step_name(train_state.step)
    final = run_dir / name
    tmp = run_dir / f"{_TMP_PREFIX}{name}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        _write_checkpoint_files(
            tmp, train_state=train_state, model=model, optimizers=optimizers, scaler=scaler
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # os.replace cannot overwrite a non-empty directory, so an existing step is moved aside
    # (atomic) first; a crash before the swap leaves no half-deleted dir that looks complete.
    old = run_dir / f"{_TMP_PREFIX}old-{name}-{os.getpid()}"
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    _fsync_dir(run_dir)
    _write_latest(run_dir, name)
    if old.exists():
        shutil.rmtree(old)
    _prune(run_dir, keep=keep, protect=name)
    return final


def _prune(run_dir: Path, *, keep: int, protect: str) -> None:
    """Keep the ``protect`` dir plus the ``keep - 1`` newest other complete dirs (total <= keep)."""
    others = [p for p in _complete_step_dirs(run_dir) if p.name != protect]
    for old in others[: max(len(others) - (keep - 1), 0)]:
        _remove_dir(old)


def complete_steps(run_dir: Path) -> list[int]:
    """Steps that have a complete checkpoint in ``run_dir``, ascending."""
    if not run_dir.is_dir():
        return []
    return [int(p.name[len(_STEP_PREFIX) :]) for p in _complete_step_dirs(run_dir)]


def latest_checkpoint(run_dir: Path) -> Path | None:
    """The dir named by ``latest`` if complete, else the newest complete ``step_*`` dir."""
    pointer = run_dir / _LATEST_FILE
    if pointer.is_file():
        target = run_dir / pointer.read_text().strip()
        if target.name.startswith(_STEP_PREFIX) and target.is_dir() and _is_complete(target):
            return target
    if not run_dir.is_dir():
        return None
    complete = _complete_step_dirs(run_dir)
    return complete[-1] if complete else None


def load_checkpoint(path: Path) -> LoadedCheckpoint:
    """Load a checkpoint dir. Raises ``FileNotFoundError`` if incomplete, ``ValueError`` on unknown format."""
    meta_path = path / _META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"checkpoint {path} has no {_META_FILE}; it is missing or incomplete")
    meta = json.loads(meta_path.read_text())
    version = meta.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version {version!r}; expected {FORMAT_VERSION}")
    ts = meta["train_state"]
    optim = torch.load(path / _OPTIM_FILE, map_location="cpu", weights_only=True)
    return LoadedCheckpoint(
        path=path,
        train_state=TrainState(
            step=ts["step"], data_cursor=ts["data_cursor"], wandb_run_id=ts["wandb_run_id"]
        ),
        model=load_file(str(path / _MODEL_FILE), device="cpu"),
        optimizers=optim["optimizers"],
        scaler=optim["scaler"],
        rng={
            "python": meta["rng_python"],
            "numpy": meta["rng_numpy"],
            "torch_cpu": optim["rng_torch_cpu"],
            "torch_cuda": optim["rng_torch_cuda"],
        },
    )
