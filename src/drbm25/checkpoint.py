"""Atomic checkpoints and resume state for long PPO runs.

Layout::

    run_dir/
      latest                      text file naming the newest step dir
      step_000050/
        model.safetensors         model tensors
        optim.pt                  optimizer + scaler state + torch RNG tensors (weights_only)
        meta.json                 train state, python/numpy RNG, format_version (written last)

The full fine-tune layout (``model.train_mode: full``) saves a *sharded* checkpoint instead::

      step_000050/
        dcp/                      torch.distributed.checkpoint files, one per rank + .metadata
        rng.pt                    rank 0's torch RNG tensors (weights_only)
        meta.json                 as above, plus "layout": "sharded" (written last)

A checkpoint is built in ``.tmp-step_NNNNNN-<pid>/``, every file is fsynced, the directory is
renamed into place with ``os.replace`` and the parent directory is fsynced; only then is
``latest`` swapped (temp file + ``os.replace``). A directory is *complete* when it has
``meta.json``, so a crash at any point leaves the previous checkpoint usable.

Sharded checkpoints use the same temp-dir / ``meta.json``-last / move-aside / ``latest`` protocol;
every rank writes its DCP shard into the shared temp dir and rank 0 alone publishes it.

The policy selected on the validation set is kept apart from the resume checkpoints, so pruning
(``keep``) never touches it::

      best/
        model.safetensors         actor weights only: LoRA actor adapter (lora layout) or the full
                                  actor with HF names in the compute dtype (full layout)
        meta.json                 kind "best", step, both val nDCG@10s, selection metric, reward mode,
                                  layout, config sha256, seeds, format_version (written last)

``save_best`` builds ``.tmp-best-<pid>/``, moves the old ``best/`` aside to ``.tmp-old-best-<pid>/``
and renames the new one into place. A crash between those two renames leaves no ``best/``; the
previous one is then still complete under ``.tmp-old-best-*`` and ``recover_best`` (run at
training startup, before stale temps are swept) puts it back.

``save_checkpoint`` is rank-agnostic: in a multi-process run the caller must call it on rank 0
only. ``numpy.random.Generator`` objects are not captured (they are not global state); keep the
data order reproducible through ``TrainState.data_cursor`` and a seed instead.
"""

import json
import os
import random
import shutil
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

import numpy as np
import torch
import torch.distributed as tdist
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from drbm25.config import ScoringMode
from drbm25.distributed import DistInfo

FORMAT_VERSION = 1
_MODEL_FILE = "model.safetensors"
_OPTIM_FILE = "optim.pt"
_META_FILE = "meta.json"
_LATEST_FILE = "latest"
_DCP_DIR = "dcp"
_RNG_FILE = "rng.pt"
_LAYOUT_SHARDED = "sharded"
_ACTOR_MODEL_PREFIX = "actor.model."  # FullPolicyModels.sharded_state()["actor"]["model"], flattened by DCP
_STEP_PREFIX = "step_"
_TMP_PREFIX = ".tmp-"
BEST_DIR = "best"
_BEST_KIND = "best"
_OLD_BEST_PREFIX = f"{_TMP_PREFIX}old-{BEST_DIR}-"
BestLayout = Literal["lora", "full"]


@dataclass(frozen=True)
class TrainState:
    step: int
    data_cursor: int
    wandb_run_id: str | None
    reward_mode: ScoringMode


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
    _write_meta(tmp, train_state, rng, layout=None)
    for name in (_MODEL_FILE, _OPTIM_FILE, _META_FILE):
        _fsync_path(tmp / name)
    _fsync_dir(tmp)


def _write_meta(tmp: Path, train_state: TrainState, rng: Mapping[str, object], *, layout: str | None) -> None:
    """Write ``meta.json`` (call last: its presence marks the directory complete)."""
    meta: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "train_state": {
            "step": train_state.step,
            "data_cursor": train_state.data_cursor,
            "wandb_run_id": train_state.wandb_run_id,
            "reward_mode": train_state.reward_mode,
        },
        "rng_python": rng["python"],
        "rng_numpy": rng["numpy"],
    }
    if layout is not None:
        meta["layout"] = layout
    (tmp / _META_FILE).write_text(json.dumps(meta))


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
    return _publish(run_dir, tmp, name, keep=keep)


def _publish(run_dir: Path, tmp: Path, name: str, *, keep: int) -> Path:
    """Rename the complete ``tmp`` dir to ``run_dir/name``, point ``latest`` at it, prune."""
    final = run_dir / name
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
    meta, train_state = _read_meta(path)
    if meta.get("layout") == _LAYOUT_SHARDED:
        raise ValueError(f"checkpoint {path} is a sharded (full fine-tune) checkpoint; use load_sharded_checkpoint")
    optim = torch.load(path / _OPTIM_FILE, map_location="cpu", weights_only=True)
    return LoadedCheckpoint(
        path=path,
        train_state=train_state,
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



def _read_meta(path: Path) -> tuple[dict[str, Any], TrainState]:
    meta_path = path / _META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"checkpoint {path} has no {_META_FILE}; it is missing or incomplete")
    meta = json.loads(meta_path.read_text())
    version = meta.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version {version!r}; expected {FORMAT_VERSION}")
    ts = meta["train_state"]
    if "reward_mode" not in ts:
        raise ValueError(f"checkpoint {path} has no train_state.reward_mode; it predates the field and cannot be used")
    if ts["reward_mode"] not in get_args(ScoringMode):
        raise ValueError(f"checkpoint {path}: train_state.reward_mode {ts['reward_mode']!r} is not a scoring mode")
    state = TrainState(
        step=ts["step"], data_cursor=ts["data_cursor"], wandb_run_id=ts["wandb_run_id"], reward_mode=ts["reward_mode"]
    )
    return meta, state


def _barrier(dist: DistInfo) -> None:
    if dist.world_size > 1:
        tdist.barrier()


def save_sharded_checkpoint(
    run_dir: Path, *, train_state: TrainState, state: dict[str, Any], keep: int, dist: DistInfo
) -> Path:
    """Write ``run_dir/step_NNNNNN`` from a DCP state dict; a collective, call on EVERY rank.

    ``state`` is a ``torch.distributed.checkpoint`` state dict (e.g.
    ``FullPolicyModels.sharded_state``). Every rank writes its shard into one shared temp dir
    (named without a pid, since ranks have different pids); rank 0 then writes ``rng.pt`` and
    ``meta.json`` and publishes the dir exactly as :func:`save_checkpoint` does. Returns the
    final path on every rank.
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    name = _step_name(train_state.step)
    tmp = run_dir / f"{_TMP_PREFIX}{name}-sharded"
    if dist.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
    _barrier(dist)
    # FileSystemWriter fsyncs every data file (sync_files=True is the default).
    dcp_save(state, checkpoint_id=tmp / _DCP_DIR)
    _barrier(dist)
    if dist.rank == 0:
        try:
            rng = capture_rng()
            torch.save({"torch_cpu": rng["torch_cpu"], "torch_cuda": rng["torch_cuda"]}, tmp / _RNG_FILE)
            _write_meta(tmp, train_state, rng, layout=_LAYOUT_SHARDED)
            for path in (*sorted((tmp / _DCP_DIR).iterdir()), tmp / _RNG_FILE, tmp / _META_FILE):
                _fsync_path(path)
            _fsync_dir(tmp / _DCP_DIR)
            _fsync_dir(tmp)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        _publish(run_dir, tmp, name, keep=keep)
    _barrier(dist)
    return run_dir / name


def load_sharded_checkpoint(path: Path, state: dict[str, Any]) -> LoadedCheckpoint:
    """Fill ``state`` (same structure as saved, e.g. a fresh ``sharded_state``) in place; a collective.

    The returned ``LoadedCheckpoint`` has empty ``model`` / ``optimizers`` (they are in ``state``),
    ``scaler=None``, and rank 0's RNG as saved.
    """
    meta, train_state = _read_meta(path)
    if meta.get("layout") != _LAYOUT_SHARDED:
        raise ValueError(f"checkpoint {path} is not a sharded checkpoint; use load_checkpoint")
    dcp_load(state, checkpoint_id=path / _DCP_DIR)
    rng = torch.load(path / _RNG_FILE, map_location="cpu", weights_only=True)
    return LoadedCheckpoint(
        path=path,
        train_state=train_state,
        model={},
        optimizers={},
        scaler=None,
        rng={
            "python": meta["rng_python"],
            "numpy": meta["rng_numpy"],
            "torch_cpu": rng["torch_cpu"],
            "torch_cuda": rng["torch_cuda"],
        },
    )


CheckpointLayout = Literal["single", "sharded"]


def checkpoint_layout(path: Path) -> CheckpointLayout:
    """``"sharded"`` for a full fine-tune (DCP) checkpoint, ``"single"`` for a rank-0 LoRA one."""
    meta, _ = _read_meta(path)
    return "sharded" if meta.get("layout") == _LAYOUT_SHARDED else "single"


def load_sharded_actor(path: Path, model: nn.Module) -> TrainState:
    """Load the actor weights of a sharded checkpoint into a plain HF ``model`` (in place), in ONE process.

    No process group is needed (``dcp.load(..., no_dist=True)``); DCP reassembles the full tensors
    from every rank's shard file. The saved keys are ``actor.model.<name>`` with ``name`` exactly
    as in the plain model's ``state_dict()`` (``FullPolicyModels`` shards the HF model in place),
    so any missing or unexpected name raises ``KeyError``. Optimizer and critic state are not read.
    """
    meta, train_state = _read_meta(path)
    if meta.get("layout") != _LAYOUT_SHARDED:
        raise ValueError(f"checkpoint {path} is not a sharded (full fine-tune) checkpoint")
    saved = {
        key[len(_ACTOR_MODEL_PREFIX) :]
        for key in FileSystemReader(path / _DCP_DIR).read_metadata().state_dict_metadata
        if key.startswith(_ACTOR_MODEL_PREFIX)
    }
    own = model.state_dict()  # detached views of the live parameters: DCP loads into them in place
    missing, unexpected = sorted(own.keys() - saved), sorted(saved - own.keys())
    if missing or unexpected:
        raise KeyError(f"actor weights in {path} do not match the model: missing={missing} unexpected={unexpected}")
    with warnings.catch_warnings():
        # torch 2.10 warns on every single-process load, even with no_dist=True; that is the intent here.
        warnings.filterwarnings("ignore", message="torch.distributed is disabled", category=UserWarning)
        dcp_load({"actor": {"model": own}}, checkpoint_id=path / _DCP_DIR, no_dist=True)
    return train_state



# ---------------------------------------------------------------------------- best/
@dataclass(frozen=True)
class BestMeta:
    """What ``best/meta.json`` records about the selected policy."""

    step: int
    val_ndcg_normal: float
    val_ndcg_count_once: float
    reward_mode: ScoringMode
    layout: BestLayout
    config_sha256: str
    seeds: dict[str, Any]

    @property
    def selection_metric(self) -> str:
        """The run's own reward as a validation metric: ``val/ndcg_<reward_mode>``."""
        return f"val/ndcg_{self.reward_mode}"

    @property
    def score(self) -> float:
        return self.val_ndcg_normal if self.reward_mode == "normal" else self.val_ndcg_count_once


def _best_meta_json(meta: BestMeta) -> dict[str, object]:
    return {
        "kind": _BEST_KIND,
        "format_version": FORMAT_VERSION,
        "step": meta.step,
        "val_ndcg_normal": meta.val_ndcg_normal,
        "val_ndcg_count_once": meta.val_ndcg_count_once,
        "selection_metric": meta.selection_metric,
        "reward_mode": meta.reward_mode,
        "layout": meta.layout,
        "config_sha256": meta.config_sha256,
        "seeds": meta.seeds,
    }


def is_best_dir(path: Path) -> bool:
    """True for a complete ``best/`` dir (its ``meta.json`` says ``kind: best``)."""
    meta_path = path / _META_FILE
    if not meta_path.is_file():
        return False
    return json.loads(meta_path.read_text()).get("kind") == _BEST_KIND


def read_best_meta(path: Path) -> BestMeta:
    """The metadata of a ``best/`` dir (ValueError if ``path`` is not one)."""
    if not is_best_dir(path):
        raise ValueError(f"{path} is not a best/ dir (no meta.json with kind 'best')")
    raw = json.loads((path / _META_FILE).read_text())
    if raw.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported best format_version {raw.get('format_version')!r}")
    if raw["reward_mode"] not in get_args(ScoringMode) or raw["layout"] not in get_args(BestLayout):
        raise ValueError(f"{path}: bad reward_mode {raw['reward_mode']!r} or layout {raw['layout']!r}")
    return BestMeta(
        step=raw["step"],
        val_ndcg_normal=raw["val_ndcg_normal"],
        val_ndcg_count_once=raw["val_ndcg_count_once"],
        reward_mode=raw["reward_mode"],
        layout=raw["layout"],
        config_sha256=raw["config_sha256"],
        seeds=raw["seeds"],
    )


def best_meta(run_dir: Path) -> BestMeta | None:
    """``run_dir/best``'s metadata, or None when the run has no best policy yet."""
    path = run_dir / BEST_DIR
    return read_best_meta(path) if is_best_dir(path) else None


def load_best(path: Path) -> tuple[BestMeta, dict[str, Tensor]]:
    """The metadata and actor weights (on CPU) of a ``best/`` dir."""
    meta = read_best_meta(path)
    return meta, load_file(str(path / _MODEL_FILE), device="cpu")


def save_best(run_dir: Path, *, weights: Mapping[str, Tensor], meta: BestMeta) -> Path:
    """Atomically replace ``run_dir/best`` with ``weights`` and ``meta``. Rank-agnostic: call on rank 0 only."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / f"{_TMP_PREFIX}{BEST_DIR}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        save_file({k: _portable(v) for k, v in weights.items()}, str(tmp / _MODEL_FILE))
        (tmp / _META_FILE).write_text(json.dumps(_best_meta_json(meta)))
        for name in (_MODEL_FILE, _META_FILE):
            _fsync_path(tmp / name)
        _fsync_dir(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    final = run_dir / BEST_DIR
    old = run_dir / f"{_OLD_BEST_PREFIX}{os.getpid()}"
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    _fsync_dir(run_dir)
    if old.exists():
        shutil.rmtree(old)
    return final


def recover_best(run_dir: Path) -> bool:
    """Put back a best/ that a crash left moved aside; True if one was restored.

    Call before stale ``.tmp-*`` dirs are swept. With a complete ``best/`` in place this does nothing.
    """
    if is_best_dir(run_dir / BEST_DIR) or not run_dir.is_dir():
        return False
    aside = sorted(p for p in run_dir.glob(f"{_OLD_BEST_PREFIX}*") if is_best_dir(p))
    if not aside:
        return False
    final = run_dir / BEST_DIR
    if final.exists():  # an incomplete leftover: never a usable best
        shutil.rmtree(final)
    os.replace(aside[-1], final)
    _fsync_dir(run_dir)
    return True
