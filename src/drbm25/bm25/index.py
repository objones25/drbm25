"""Build a Lucene index from a TSV collection with pyserini's default English analyzer."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

JSONL_NAME = "collection.jsonl"


@dataclass(frozen=True)
class IndexInfo:
    n_docs: int
    checksum: str  # sha256 over sorted index file names and sizes


def tsv_to_jsonl(collection_tsv: Path, out_dir: Path) -> int:
    """Convert `id<TAB>contents` lines to pyserini JsonCollection lines; return the doc count.

    Duplicate ids are rejected: nDCG would otherwise see one relevant id twice in a ranking.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with collection_tsv.open(encoding="utf-8") as src, (out_dir / JSONL_NAME).open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            doc_id, sep, contents = line.rstrip("\n").partition("\t")
            if not sep or not doc_id:
                raise ValueError(f"{collection_tsv}: line {line_no} is not 'id<TAB>contents'")
            if doc_id in seen:
                raise ValueError(f"{collection_tsv}: duplicate doc id {doc_id!r} on line {line_no}")
            seen.add(doc_id)
            dst.write(json.dumps({"id": doc_id, "contents": contents}) + "\n")
    logger.info("wrote %d docs to %s", len(seen), out_dir / JSONL_NAME)
    return len(seen)


def build_index(jsonl_dir: Path, index_dir: Path, *, threads: int) -> IndexInfo:
    """Index `jsonl_dir` into `index_dir` with Anserini's IndexCollection (needs Java 21)."""
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(jsonl_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    logger.info("building index: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603  # fixed argument list, no shell
    info = read_index_info(index_dir)
    logger.info("index %s: %d docs, checksum %s", index_dir, info.n_docs, info.checksum)
    return info


def read_index_info(index_dir: Path) -> IndexInfo:
    if not index_dir.is_dir():
        raise FileNotFoundError(f"index directory not found: {index_dir}")
    # Imported here because importing pyserini starts the JVM.
    from pyserini.index.lucene import LuceneIndexReader

    n_docs = int(LuceneIndexReader(str(index_dir)).stats()["documents"])
    digest = hashlib.sha256()
    for path in sorted(p for p in index_dir.iterdir() if p.is_file()):
        digest.update(f"{path.name}\t{path.stat().st_size}\n".encode())
    return IndexInfo(n_docs=n_docs, checksum=digest.hexdigest())
