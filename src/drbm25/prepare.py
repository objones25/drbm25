"""Download MS MARCO-Science and build the BM25 index, once, atomically, and idempotently.

CLI: `python -m drbm25.prepare --config PATH [--set key=value ...]`
(`scripts/prepare_data.py` is a thin wrapper around `main`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from drbm25.bm25.index import INDEX_INFO_NAME, IndexInfo, build_index, read_index_info, tsv_to_jsonl
from drbm25.config import Config, load_config
from drbm25.data import download_msmarco_science

logger = logging.getLogger(__name__)

EXPECTED_DOCS = 800_000
_MAX_INDEX_THREADS = 16


def prepare(cfg: Config, *, expected_docs: int = EXPECTED_DOCS) -> IndexInfo:
    """Return the index at `cfg.retrieval.index_dir`, building it first if it does not exist.

    An existing index with the wrong document count is an error, never silently rebuilt: delete
    it deliberately. The build happens in a sibling temp dir and is moved into place with
    `os.replace`, so an interrupted build never leaves something that looks complete.
    """
    index_dir = cfg.retrieval.index_dir
    try:
        existing = read_index_info(index_dir)
    except FileNotFoundError:
        logger.info("no index at %s, building", index_dir)
    else:
        if existing.n_docs != expected_docs:
            raise ValueError(
                f"index {index_dir} has the wrong size: expected {expected_docs} docs, found {existing.n_docs}; "
                "delete it to rebuild"
            )
        logger.info("index ok: %s (%d docs)", index_dir, existing.n_docs)
        return existing

    paths = download_msmarco_science(cfg.data)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    threads = min(os.cpu_count() or 1, _MAX_INDEX_THREADS)
    with tempfile.TemporaryDirectory(dir=index_dir.parent, prefix=f".{index_dir.name}.build-") as tmp:
        tmp_dir = Path(tmp)
        n_docs = tsv_to_jsonl(paths.collection_tsv, tmp_dir / "jsonl")
        if n_docs != expected_docs:
            raise ValueError(f"{paths.collection_tsv}: expected {expected_docs} docs, found {n_docs}")
        info = build_index(tmp_dir / "jsonl", tmp_dir / "index", threads=threads)
        if info.n_docs != expected_docs:
            raise ValueError(f"built index: expected {expected_docs} docs, found {info.n_docs}")
        os.replace(tmp_dir / "index", index_dir)
    _write_index_info(index_dir, info, hf_revision=cfg.data.hf_revision)
    return info


def _write_index_info(index_dir: Path, info: IndexInfo, *, hf_revision: str) -> None:
    record = {
        "n_docs": info.n_docs,
        "checksum": info.checksum,
        "hf_revision": hf_revision,
        "built_at": datetime.now(UTC).isoformat(),
    }
    tmp_path = index_dir / f"{INDEX_INFO_NAME}.tmp"
    tmp_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, index_dir / INDEX_INFO_NAME)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MS MARCO-Science and build the BM25 index.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    overrides: dict[str, str] = {}
    for item in args.overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--set expects key=value, got {item!r}")
        overrides[key] = value
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, overrides)
    info = prepare(cfg)
    print(f"index ok: {cfg.retrieval.index_dir} n_docs={info.n_docs} checksum={info.checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
