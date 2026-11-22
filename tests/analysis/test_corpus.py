import math
from pathlib import Path

import numpy as np
import pytest

from drbm25.analysis.corpus import IdfTable, embed_similarity, idf_summary


@pytest.mark.java
def test_idf_table_ranks_rare_terms_above_common_ones(toy_index: Path) -> None:
    table = IdfTable.from_index(toy_index)

    # analyzed forms: "solar" appears in more toy docs than "effici" (efficiency)
    assert table.n_docs == 1002
    assert table.idf("effici") > table.idf("solar") > 0.0


@pytest.mark.java
def test_idf_table_treats_unknown_term_as_df_one(toy_index: Path) -> None:
    table = IdfTable.from_index(toy_index)

    assert table.idf("zzzunseenzzz") == pytest.approx(math.log(1002))
    assert table.n_unknown == 1


def test_idf_summary_mean_and_max() -> None:
    table = IdfTable(n_docs=100, df={"a": 10, "b": 1})

    mean, top = idf_summary(["a", "b", "a"], table)

    assert mean == pytest.approx((2 * math.log(10) + math.log(100)) / 3)
    assert top == pytest.approx(math.log(100))


def test_idf_summary_of_no_terms_is_nan() -> None:
    mean, top = idf_summary([], IdfTable(n_docs=10, df={}))

    assert math.isnan(mean)
    assert math.isnan(top)


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Deterministic 3-d vectors: identical strings get identical vectors."""
    return np.array([[len(t), t.count("a") + 1.0, t.count("e") + 1.0] for t in texts], dtype=np.float64)


def test_embed_similarity_cosine_with_none_rewrites() -> None:
    sims = embed_similarity(["heart attack", "copper"], ["heart attack", None], encode=_fake_encode)

    assert sims[0] == pytest.approx(1.0)
    assert sims[1] is None


def test_embed_similarity_encodes_each_unique_string_once() -> None:
    seen: list[str] = []

    def encode(texts: list[str]) -> np.ndarray:
        seen.extend(texts)
        return _fake_encode(texts)

    embed_similarity(["q", "q", "q"], ["r", "r", "s"], encode=encode)

    assert sorted(seen) == ["q", "r", "s"]
