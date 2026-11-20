import json
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from requests.exceptions import ConnectionError as ConnDown
from requests.exceptions import Timeout as ConnSlow

import drbm25.data as data
from drbm25.config import DataCfg
from drbm25.data import (
    Example,
    batch_indices,
    download_msmarco_science,
    load_examples,
    split_train,
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


def _train_examples(n: int) -> list[Example]:
    return [Example(qid=f"train-{i}", query=f"train q{i}", relevant_ids=frozenset({str(i)})) for i in range(n)]


def test_split_train_val_has_val_size_rows_and_partitions_train_exactly():
    train = _train_examples(50)
    rest, val = split_train(train, val_size=12, seed=1234)
    assert len(val) == 12
    assert len(rest) == 38
    assert sorted(rest + val, key=lambda e: int(e.qid.split("-")[1])) == train
    assert {e.qid for e in rest}.isdisjoint({e.qid for e in val})
    assert {e.query for e in rest}.isdisjoint({e.query for e in val})


def test_split_train_keeps_train_order_in_the_remaining_rows():
    # batch_indices permutes indices of the remaining list, so its order must be the file order.
    rest, _ = split_train(_train_examples(30), val_size=10, seed=7)
    positions = [int(e.qid.split("-")[1]) for e in rest]
    assert positions == sorted(positions)


def test_split_train_is_a_seeded_sample_not_the_first_rows():
    train = _train_examples(100)
    rest, val = split_train(train, val_size=10, seed=1234)
    assert split_train(train, val_size=10, seed=1234) == (rest, val)
    assert split_train(train, val_size=10, seed=4321)[1] != val
    assert val != train[:10]


def test_split_train_val_is_disjoint_from_the_dev_test_set():
    train, dev = _train_examples(40), _examples(15)
    rest, val = split_train(train, val_size=8, seed=1234)
    test = list(dev)  # the test set is all of dev, untouched by the split
    assert {e.qid for e in val}.isdisjoint({e.qid for e in test})
    assert {e.query for e in val}.isdisjoint({e.query for e in test})
    assert {e.qid for e in rest}.isdisjoint({e.qid for e in test})


@pytest.mark.parametrize("val_size", [0, -1, 20, 21])
def test_split_train_rejects_val_size_outside_1_to_n_minus_1(val_size: int):
    with pytest.raises(ValueError, match="val_size"):
        split_train(_train_examples(20), val_size=val_size, seed=1)


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
    return DataCfg(
        hf_dataset="DeepRetrieval/datasets", hf_revision=SHA, data_dir=tmp_path / "d", val_size=500, split_seed=1234
    )


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


def _local_entry_not_found(cause: BaseException | None) -> LocalEntryNotFoundError:
    """Mimic huggingface_hub 0.36.2 `_raise_on_head_call_error`: `raise ... from head_call_error`."""
    err = LocalEntryNotFoundError("cannot locate file")
    err.__cause__ = cause
    return err


def test_download_retries_when_hf_wraps_connection_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    attempts: list[int] = []

    def wrapped_down(**kwargs: Any) -> str:
        raise _local_entry_not_found(ConnDown("down"))

    def ok(**kwargs: Any) -> str:
        return str(tmp_path / "f")

    steps = iter([wrapped_down, wrapped_down, ok])

    def flaky(**kwargs: Any) -> str:
        attempts.append(1)
        return next(steps)(**kwargs)

    monkeypatch.setattr(data, "hf_hub_download", flaky)
    data._download_with_retry(_cfg(tmp_path), "raw_data/x", sleep=lambda _s: None)
    assert len(attempts) == 3


@pytest.mark.parametrize(
    "error",
    [
        _local_entry_not_found(None),
        _local_entry_not_found(PermissionError("nope")),
        EntryNotFoundError("missing file"),
    ],
    ids=["no-cause", "non-connection-cause", "entry-not-found"],
)
def test_download_does_not_retry_missing_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception):
    attempts: list[int] = []

    def missing(**kwargs: Any) -> str:
        attempts.append(1)
        raise error

    monkeypatch.setattr(data, "hf_hub_download", missing)
    with pytest.raises(type(error)):
        data._download_with_retry(_cfg(tmp_path), "raw_data/x", sleep=lambda _s: None)
    assert len(attempts) == 1


@pytest.mark.integration
def test_real_msmarco_science_sizes(tmp_path: Path):
    cfg = _cfg(tmp_path)
    train = Path(data._download_with_retry(cfg, data.TRAIN_FILE))
    dev = Path(data._download_with_retry(cfg, data.DEV_FILE))
    train_examples, dev_examples = load_examples(train), load_examples(dev)
    assert len(train_examples) == 22416
    assert len(dev_examples) == 2305
    rest, val = split_train(train_examples, val_size=cfg.val_size, seed=cfg.split_seed)
    assert (len(rest), len(val)) == (21916, 500)
    test_queries = {e.query for e in dev_examples}
    assert {e.query for e in val}.isdisjoint(test_queries)
    assert {e.query for e in val}.isdisjoint({e.query for e in rest})
