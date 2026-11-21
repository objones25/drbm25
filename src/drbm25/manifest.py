"""Reproducibility manifest: ``<run_dir>/run_manifest.json``.

Written by rank 0 at the start of every training invocation. The first start records every seed
that affects results (with how each is derived), the resolved config and its sha256, the git
commit, dataset / model / index identity, library versions and hardware. A resume keeps all of
that and appends one entry to ``resumed_at``; a resume whose config seeds differ is refused.
The same dict goes to W&B as ``config["meta"]``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

import torch

from drbm25.config import Config, ScoringMode

logger = logging.getLogger(__name__)

MANIFEST_FILE: Final = "run_manifest.json"
FORMAT_VERSION: Final = 1
# Seeds that come straight from the config: a resume must not change any of them.
_CONFIG_SEEDS: Final = ("run_seed", "split_seed", "data_order_seed", "model_init_seed")
_PACKAGES: Final = ("torch", "transformers", "peft", "vllm", "pyserini", "wandb", "numpy", "safetensors")
_SUBPROCESS_TIMEOUT_S: Final = 10.0
_SEED_DERIVATION: Final = {
    "run_seed": "config run.seed",
    "split_seed": "config data.split_seed: numpy default_rng(split_seed).permutation picks the data.val_size "
    "validation rows of train",
    "data_order_seed": "= run.seed: epoch e of the remaining train rows is default_rng([run.seed, e]).permutation "
    "(drbm25.data.batch_indices)",
    "model_init_seed": "= run.seed: python/numpy/torch seeded with run.seed on every rank before the models are "
    "built (LoRA A matrices, value head, tiny model)",
    "sampling_seeds": "run.seed + rank: python/numpy/torch reseeded per rank after construction (rollout sampling)",
    "vllm_seeds": "run.seed + rank: the seed of each rank's vLLM engine (null on the hf backend)",
    "resume_reseed": "on resume rank 0 restores its saved RNG state; other ranks reseed with "
    "run.seed + rank + 1000 * step",
    "bootstrap_seed": "drbm25.eval.report --bootstrap-seed (default 0), recorded in report.md",
}


@dataclass(frozen=True)
class Environment:
    """Code and machine facts; ``None`` where a fact is unavailable (no git checkout, no GPU)."""

    git_sha: str | None
    git_dirty: bool | None
    versions: dict[str, str | None]
    cuda: str | None
    driver: str | None
    gpus: tuple[str, ...]
    hostname: str


def seeds(cfg: Config, *, world_size: int) -> dict[str, Any]:
    """Every seed that affects a run's results, by name (derivations in the manifest)."""
    seed = cfg.run.seed
    per_rank = [seed + rank for rank in range(world_size)]
    return {
        "run_seed": seed,
        "split_seed": cfg.data.split_seed,
        "data_order_seed": seed,
        "model_init_seed": seed,
        "sampling_seeds": per_rank,
        "vllm_seeds": per_rank if cfg.rollout.backend == "vllm" else None,
    }


def _plain_config(cfg: Config) -> dict[str, Any]:
    plain = json.loads(json.dumps(asdict(cfg), default=str))
    if not isinstance(plain, dict):
        raise TypeError("a Config must serialise to a JSON object")
    return plain


def config_sha256(cfg: Config) -> str:
    """sha256 of the resolved config as canonical JSON (sorted keys, Paths as strings)."""
    canonical = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_manifest(
    cfg: Config,
    *,
    run_name: str,
    reward_mode: ScoringMode,
    world_size: int,
    index_checksum: str,
    env: Environment,
    started_utc: datetime,
) -> dict[str, Any]:
    if started_utc.tzinfo is None:
        raise ValueError("started_utc must be timezone-aware")
    return {
        "format_version": FORMAT_VERSION,
        "run_name": run_name,
        "reward_mode": reward_mode,
        "seeds": seeds(cfg, world_size=world_size),
        "seed_derivation": dict(_SEED_DERIVATION),
        "config": _plain_config(cfg),
        "config_sha256": config_sha256(cfg),
        "effective_micro_batch_size": cfg.ppo.micro_batch_size,
        "git": {"sha": env.git_sha, "dirty": env.git_dirty},
        "dataset": {"hf_dataset": cfg.data.hf_dataset, "hf_revision": cfg.data.hf_revision},
        "model": {"name": cfg.model.name, "revision": cfg.model.revision},
        # check_index has verified that the index holds exactly expected_docs documents.
        "index": {"checksum": index_checksum, "n_docs": cfg.retrieval.expected_docs},
        "versions": dict(env.versions),
        "hardware": {"cuda": env.cuda, "driver": env.driver, "gpus": list(env.gpus)},
        "world_size": world_size,
        "hostname": env.hostname,
        "start_time_utc": started_utc.isoformat(),
        "resumed_at": [],
    }


def check_seeds_unchanged(existing: dict[str, Any] | None, current: dict[str, Any]) -> None:
    """Raise ValueError when a resume would change a config seed (``current`` = :func:`seeds`)."""
    if existing is None:
        return
    changed = {k: (existing["seeds"][k], current[k]) for k in _CONFIG_SEEDS if existing["seeds"][k] != current[k]}
    if changed:
        raise ValueError(f"this run was started with other seeds; refusing to resume with {changed} (old, new)")


def update_manifest(existing: dict[str, Any] | None, new: dict[str, Any], *, step: int) -> dict[str, Any]:
    """``new`` on a first start; on a resume, ``existing`` plus one ``resumed_at`` entry. Inputs are not mutated."""
    if existing is None:
        return copy.deepcopy(new)
    check_seeds_unchanged(existing, new["seeds"])
    merged = copy.deepcopy(existing)
    merged["resumed_at"].append(
        {
            "time_utc": new["start_time_utc"],
            "step": step,
            "config_sha256": new["config_sha256"],
            "git": copy.deepcopy(new["git"]),
            "hostname": new["hostname"],
            "world_size": new["world_size"],
            "effective_micro_batch_size": new["effective_micro_batch_size"],
        }
    )
    return merged


def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / MANIFEST_FILE
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: not a format_version {FORMAT_VERSION} run manifest")
    return manifest


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Write atomically (temp file + ``os.replace``) so a crash never leaves half a manifest."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / f".tmp-{MANIFEST_FILE}-{os.getpid()}"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, run_dir / MANIFEST_FILE)


# ---------------------------------------------------------------------------- environment (I/O)
def _run(argv: list[str]) -> str | None:
    """stdout of a fixed command, or None (logged) when it is unavailable on this machine."""
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("manifest: %s unavailable (%s); recorded as null", argv[0], exc)
        return None
    return proc.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None  # vllm is a Linux-only extra: absent is a fact worth recording, not an error


def collect_environment() -> Environment:
    sha = _run(["git", "rev-parse", "HEAD"])
    # Tracked changes only: untracked files (notes, downloads) do not change the code that ran.
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"]) if sha is not None else None
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) if torch.cuda.is_available() else None
    gpus = tuple(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if torch.cuda.is_available() else ()
    return Environment(
        git_sha=sha,
        git_dirty=None if status is None else bool(status),
        versions={"python": platform.python_version(), **{name: _package_version(name) for name in _PACKAGES}},
        cuda=torch.version.cuda,
        driver=None if driver is None else driver.splitlines()[0],
        gpus=gpus,
        hostname=socket.gethostname(),
    )
