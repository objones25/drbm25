"""MS MARCO-Science data: download, parsing, the train/val split, and the seeded data order.

The test set is all of dev; the validation set is a seeded sample of train (see `split_train`)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from drbm25.config import DataCfg

# Facts about the pinned dataset revision (cfg.hf_revision), not run configuration.
TRAIN_FILE = "raw_data/msmarco/msmarco_science/train.jsonl"
DEV_FILE = "raw_data/msmarco/msmarco_science/dev.jsonl"
COLLECTION_FILE = "raw_data/msmarco/collection_800k.tsv"

_DOWNLOAD_ATTEMPTS = 3
_BACKOFF_BASE_S = 1.0
_CONNECTION_ERRORS = (requests.ConnectionError, requests.Timeout)


def _is_connection_failure(error: Exception) -> bool:
    """True for connection errors, including huggingface_hub's wrapping of them.

    With nothing cached, hf_hub_download (0.36.2, `_raise_on_head_call_error`) re-raises a failed
    HEAD request as `LocalEntryNotFoundError ... from head_call_error`. Only that case, with a
    connection error as the cause, is retryable; a genuinely missing file is not.
    """
    if isinstance(error, _CONNECTION_ERRORS):
        return True
    return isinstance(error, LocalEntryNotFoundError) and isinstance(error.__cause__, _CONNECTION_ERRORS)


@dataclass(frozen=True)
class Example:
    qid: str
    query: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class DataPaths:
    train_jsonl: Path
    dev_jsonl: Path
    collection_tsv: Path


def _download_with_retry(cfg: DataCfg, filename: str, *, sleep: Callable[[float], None] = time.sleep) -> str:
    """Download one file at the pinned revision; retry only connection-type errors."""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            return hf_hub_download(
                repo_id=cfg.hf_dataset,
                filename=filename,
                repo_type="dataset",
                revision=cfg.hf_revision,
                local_dir=cfg.data_dir,
                token=os.environ.get("HF_TOKEN"),
            )
        except (*_CONNECTION_ERRORS, LocalEntryNotFoundError) as error:
            if not _is_connection_failure(error) or attempt == _DOWNLOAD_ATTEMPTS:
                raise
            sleep(_BACKOFF_BASE_S * 2 ** (attempt - 1))
    raise AssertionError("unreachable: the loop returns or raises")


def download_msmarco_science(cfg: DataCfg) -> DataPaths:
    return DataPaths(
        train_jsonl=Path(_download_with_retry(cfg, TRAIN_FILE)),
        dev_jsonl=Path(_download_with_retry(cfg, DEV_FILE)),
        collection_tsv=Path(_download_with_retry(cfg, COLLECTION_FILE)),
    )


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            docs = row.get("docs_id")
            if not docs:
                raise ValueError(f"{path}: line {line_no} has no docs_id")
            examples.append(
                Example(
                    qid=f"{path.stem}-{line_no}",
                    query=row["question"],
                    relevant_ids=frozenset(str(d) for d in docs),
                )
            )
    return examples


def split_train(train: Sequence[Example], *, val_size: int, seed: int) -> tuple[list[Example], list[Example]]:
    """Return (train_rest, val): `val_size` rows drawn by `default_rng(seed).permutation`, the rest kept.

    Both lists keep the order of `train`, so `batch_indices` over `train_rest` depends only on the
    split and its own seed. The split depends only on `seed` and the file, never on the reward
    mode or `run.seed`, so both runs validate on the same queries.
    """
    if not 0 < val_size < len(train):
        raise ValueError(f"val_size must be in 1..{len(train) - 1} (train has {len(train)} rows), got {val_size}")
    chosen = set(np.random.default_rng(seed).permutation(len(train))[:val_size].tolist())
    rest = [e for i, e in enumerate(train) if i not in chosen]
    val = [e for i, e in enumerate(train) if i in chosen]
    return rest, val


def batch_indices(cursor: int, batch_size: int, n_examples: int, seed: int) -> list[int]:
    """Example indices for the batch starting at global `cursor`.

    Epoch e uses the permutation from default_rng([seed, e]); the cursor maps to
    epoch cursor // n and offset cursor % n, and a batch may span an epoch boundary.
    """
    if batch_size <= 0 or n_examples <= 0 or cursor < 0:
        raise ValueError(f"need batch_size > 0, n_examples > 0, cursor >= 0; got {batch_size}, {n_examples}, {cursor}")
    indices: list[int] = []
    perms: dict[int, np.ndarray] = {}
    for position in range(cursor, cursor + batch_size):
        epoch, offset = divmod(position, n_examples)
        if epoch not in perms:
            perms[epoch] = np.random.default_rng([seed, epoch]).permutation(n_examples)
        indices.append(int(perms[epoch][offset]))
    return indices
