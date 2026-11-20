import json
from pathlib import Path
from typing import Any

import pytest
from requests.exceptions import ConnectionError as ConnDown
from requests.exceptions import Timeout as ConnSlow

import drbm25.data as data
from drbm25.config import DataCfg
from drbm25.data import (
    Example,
    batch_indices,
    download_msmarco_science,
    load_examples,
    split_dev,
)

SHA = "44eb79992e8c5ed4784dbe6559f8b097d7d03855"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _examples(n: int) -> list[Example]:
    return [Example(qid=f"dev-{i}", query=f"q{i}", relevant_ids=frozenset({str(i)})) for i in range(n)]


def test_load_examples_parses_fixture(tmp_path: Path):
    path = _write_jsonl(
        tmp_path / "dev.jsonl",
        [
            {"question": "what is dna", "docs_id": ["1"]},
            {"question": "what is rna", "docs_id": ["2", "3"]},
            {"question": "what is atp", "docs_id": [4]},
        ],
    )
    assert load_examples(path) == [
        Example("dev-1", "what is dna", frozenset({"1"})),
        Example("dev-2", "what is rna", frozenset({"2", "3"})),
        Example("dev-3", "what is atp", frozenset({"4"})),
    ]


def test_load_examples_rejects_row_without_docs(tmp_path: Path):
    path = _write_jsonl(
        tmp_path / "dev.jsonl",
        [{"question": "ok", "docs_id": ["1"]}, {"question": "bad", "docs_id": []}],
    )
    with pytest.raises(ValueError, match="line 2"):
        load_examples(path)


def test_split_dev_takes_first_101_as_val_and_all_as_test():
    dev = _examples(120)
    val, test = split_dev(dev, val_size=101)
    assert val == dev[:101]
    assert test == dev


def test_split_dev_rejects_val_size_larger_than_dev():
    with pytest.raises(ValueError, match="val_size"):
        split_dev(_examples(5), val_size=6)


def test_batch_indices_deterministic_and_identical_across_calls():
    first = batch_indices(cursor=3, batch_size=4, n_examples=10, seed=7)
    assert first == batch_indices(cursor=3, batch_size=4, n_examples=10, seed=7)
    assert first != batch_indices(cursor=3, batch_size=4, n_examples=10, seed=8)


def test_batch_indices_cover_epoch_exactly_once():
    both = batch_indices(0, 5, 10, seed=1) + batch_indices(5, 5, 10, seed=1)
    assert sorted(both) == list(range(10))


def test_batch_indices_span_epoch_boundary():
    spanning = batch_indices(8, 4, 10, seed=1)
    epoch0_tail = batch_indices(8, 2, 10, seed=1)
    epoch1_head = batch_indices(10, 2, 10, seed=1)
    assert spanning == epoch0_tail + epoch1_head
    assert sorted(spanning[:2] + batch_indices(0, 8, 10, seed=1)) == list(range(10))


def test_batch_indices_rejects_nonpositive_sizes():
    with pytest.raises(ValueError, match="batch_size"):
        batch_indices(0, 0, 10, seed=1)
    with pytest.raises(ValueError, match="n_examples"):
        batch_indices(0, 4, 0, seed=1)


def _cfg(tmp_path: Path) -> DataCfg:
    return DataCfg(hf_dataset="DeepRetrieval/datasets", hf_revision=SHA, data_dir=tmp_path / "d", val_size=101)


def test_download_uses_pinned_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / Path(kwargs["filename"]).name)

    monkeypatch.setattr(data, "hf_hub_download", fake)
    monkeypatch.setenv("HF_TOKEN", "t")
    paths = download_msmarco_science(_cfg(tmp_path))
    assert [c["filename"] for c in calls] == [
        "raw_data/msmarco/msmarco_science/train.jsonl",
        "raw_data/msmarco/msmarco_science/dev.jsonl",
        "raw_data/msmarco/collection_800k.tsv",
    ]
    assert {c["revision"] for c in calls} == {SHA}
    assert {c["repo_id"] for c in calls} == {"DeepRetrieval/datasets"}
    assert {c["repo_type"] for c in calls} == {"dataset"}
    assert {c["token"] for c in calls} == {"t"}
    assert paths.train_jsonl.name == "train.jsonl"
    assert paths.dev_jsonl.name == "dev.jsonl"
    assert paths.collection_tsv.name == "collection_800k.tsv"


def test_download_retries_connection_errors_with_backoff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    attempts: list[int] = []

    def down(**kwargs: Any) -> str:
        raise ConnDown("down")

    def ok(**kwargs: Any) -> str:
        return str(tmp_path / "f")

    steps = iter([down, down, ok])

    def flaky(**kwargs: Any) -> str:
        attempts.append(1)
        return next(steps)(**kwargs)

    sleeps: list[float] = []
    monkeypatch.setattr(data, "hf_hub_download", flaky)
    data._download_with_retry(_cfg(tmp_path), "raw_data/x", sleep=sleeps.append)
    assert len(attempts) == 3
    assert sleeps == [1.0, 2.0]


def test_download_gives_up_after_three_connection_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    attempts: list[int] = []

    def down(**kwargs: Any) -> str:
        attempts.append(1)
        raise ConnSlow("slow")

    monkeypatch.setattr(data, "hf_hub_download", down)
    with pytest.raises(ConnSlow):
        data._download_with_retry(_cfg(tmp_path), "raw_data/x", sleep=lambda _s: None)
    assert len(attempts) == 3


def test_download_does_not_retry_other_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    attempts: list[int] = []

    def bad(**kwargs: Any) -> str:
        attempts.append(1)
        raise PermissionError("nope")

    monkeypatch.setattr(data, "hf_hub_download", bad)
    with pytest.raises(PermissionError):
        data._download_with_retry(_cfg(tmp_path), "raw_data/x", sleep=lambda _s: None)
    assert len(attempts) == 1


@pytest.mark.integration
def test_real_msmarco_science_sizes(tmp_path: Path):
    cfg = _cfg(tmp_path)
    train = Path(data._download_with_retry(cfg, data.TRAIN_FILE))
    dev = Path(data._download_with_retry(cfg, data.DEV_FILE))
    assert len(load_examples(train)) == 22416
    assert len(load_examples(dev)) == 2305
