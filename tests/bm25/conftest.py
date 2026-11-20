"""Session-scoped toy Lucene index mirroring slide 23 (repetition flips the ranking)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from drbm25.bm25.index import build_index, tsv_to_jsonl

N_FILLER_DOCS = 1000
DOC_LEN = 50
N_SOLAR_DOCS = 100
N_EFFICIENCY_DOCS = 50
TOY_SEED = 279
DOC_A = "docA"  # solar x3
DOC_B = "docB"  # solar x1, efficiency x1


def _filler(rng: random.Random, n: int) -> list[str]:
    return [f"w{rng.randrange(200)}" for _ in range(n)]


def _toy_rows() -> list[tuple[str, str]]:
    rng = random.Random(TOY_SEED)
    # Disjoint doc sets so no filler doc ties with Doc B on both terms.
    special = rng.sample(range(N_FILLER_DOCS), N_SOLAR_DOCS + N_EFFICIENCY_DOCS)
    solar_docs = set(special[:N_SOLAR_DOCS])
    efficiency_docs = set(special[N_SOLAR_DOCS:])
    rows: list[tuple[str, str]] = []
    for i in range(N_FILLER_DOCS):
        if i in solar_docs:
            tokens = ["solar", *_filler(rng, DOC_LEN - 1)]
        elif i in efficiency_docs:
            tokens = ["efficiency", *_filler(rng, DOC_LEN - 1)]
        else:
            tokens = _filler(rng, DOC_LEN)
        rows.append((f"d{i}", " ".join(tokens)))
    rows.append((DOC_A, " ".join(["solar"] * 3 + _filler(rng, DOC_LEN - 3))))
    rows.append((DOC_B, " ".join(["solar", "efficiency", *_filler(rng, DOC_LEN - 2)])))
    return rows


@pytest.fixture(scope="session")
def toy_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("toy")
    tsv = root / "collection.tsv"
    tsv.write_text("".join(f"{doc_id}\t{text}\n" for doc_id, text in _toy_rows()), encoding="utf-8")
    jsonl_dir = root / "jsonl"
    tsv_to_jsonl(tsv, jsonl_dir)
    index_dir = root / "index"
    build_index(jsonl_dir, index_dir, threads=2)
    return index_dir
