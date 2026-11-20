import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

import drbm25.prepare as prepare_mod
from drbm25.bm25.index import IndexInfo, build_index, read_index_info
from drbm25.config import Config, load_config
from drbm25.data import DataPaths
from drbm25.prepare import prepare

REPO = Path(__file__).resolve().parents[1]
TOY_DOCS = 3


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return load_config(
        REPO / "configs" / "smoke_cpu.yaml",
        {"retrieval.index_dir": str(tmp_path / "index"), "data.data_dir": str(tmp_path / "data")},
    )


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    tsv = tmp_path / "collection.tsv"
    tsv.write_text("1\tsolar panels\n2\twind turbines\n3\tsolar efficiency\n", encoding="utf-8")
    paths = DataPaths(train_jsonl=tmp_path / "t.jsonl", dev_jsonl=tmp_path / "d.jsonl", collection_tsv=tsv)
    spy = create_autospec(prepare_mod.download_msmarco_science, return_value=paths)
    monkeypatch.setattr(prepare_mod, "download_msmarco_science", spy)
    return spy


@pytest.mark.java
def test_prepare_is_idempotent(cfg: Config, fake_download: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    first = prepare(cfg, expected_docs=TOY_DOCS)
    assert first.n_docs == TOY_DOCS

    build_spy = create_autospec(build_index)
    monkeypatch.setattr(prepare_mod, "build_index", build_spy)
    fake_download.reset_mock()
    second = prepare(cfg, expected_docs=TOY_DOCS)

    assert second == first
    build_spy.assert_not_called()
    fake_download.assert_not_called()


@pytest.mark.java
def test_prepare_writes_index_info_json(cfg: Config, fake_download: Any) -> None:
    info = prepare(cfg, expected_docs=TOY_DOCS)
    written = json.loads((cfg.retrieval.index_dir / "INDEX_INFO.json").read_text(encoding="utf-8"))
    assert written["n_docs"] == TOY_DOCS
    assert written["checksum"] == info.checksum
    assert written["hf_revision"] == cfg.data.hf_revision
    assert written["built_at"].endswith("+00:00")


@pytest.mark.java
def test_prepare_refuses_index_with_wrong_doc_count(cfg: Config, fake_download: Any) -> None:
    prepare(cfg, expected_docs=TOY_DOCS)
    with pytest.raises(ValueError, match=r"expected 800000 docs, found 3"):
        prepare(cfg, expected_docs=800_000)


@pytest.mark.java
def test_prepare_failed_build_leaves_no_index(cfg: Config, fake_download: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> IndexInfo:
        raise RuntimeError("indexer crashed")

    monkeypatch.setattr(prepare_mod, "build_index", boom)
    with pytest.raises(RuntimeError, match="indexer crashed"):
        prepare(cfg, expected_docs=TOY_DOCS)
    assert not cfg.retrieval.index_dir.exists()


def _info_path(cfg: Config) -> Path:
    return cfg.retrieval.index_dir / "INDEX_INFO.json"


@pytest.mark.java
def test_prepare_crash_before_move_leaves_no_index_dir(
    cfg: Config, fake_download: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("crash while writing info")

    monkeypatch.setattr(prepare_mod, "_write_index_info", boom)
    with pytest.raises(RuntimeError, match="crash while writing info"):
        prepare(cfg, expected_docs=TOY_DOCS)
    assert not cfg.retrieval.index_dir.exists()


@pytest.mark.java
def test_prepare_index_dir_always_has_info(cfg: Config, fake_download: Any) -> None:
    prepare(cfg, expected_docs=TOY_DOCS)
    assert _info_path(cfg).is_file()
    assert not list(cfg.retrieval.index_dir.glob("*.tmp"))


@pytest.mark.java
def test_prepare_rejects_index_missing_info(cfg: Config, fake_download: Any) -> None:
    prepare(cfg, expected_docs=TOY_DOCS)
    _info_path(cfg).unlink()
    with pytest.raises(ValueError, match=r"no INDEX_INFO\.json"):
        prepare(cfg, expected_docs=TOY_DOCS)


@pytest.mark.java
@pytest.mark.parametrize(
    ("field", "bad_value"), [("n_docs", 99), ("checksum", "deadbeef"), ("hf_revision", "other-rev")]
)
def test_prepare_rejects_mismatched_info(cfg: Config, fake_download: Any, field: str, bad_value: object) -> None:
    prepare(cfg, expected_docs=TOY_DOCS)
    record = json.loads(_info_path(cfg).read_text(encoding="utf-8"))
    good = record[field]
    record[field] = bad_value
    _info_path(cfg).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{field} mismatch: recorded {bad_value!r}, expected {good!r}"):
        prepare(cfg, expected_docs=TOY_DOCS)


@pytest.mark.integration
@pytest.mark.java
def test_prepare_real_800k_index() -> None:
    """Builds the real index (downloads ~800K passages). Run with `-m integration` and
    `--no-cov`; set DRBM25_REAL_INDEX_DIR/DRBM25_REAL_DATA_DIR to reuse locations."""
    cfg = load_config(
        REPO / "configs" / "runpod_a100_full.yaml",
        {
            "retrieval.index_dir": os.environ.get("DRBM25_REAL_INDEX_DIR", str(REPO / "indexes" / "msmarco_800k")),
            "data.data_dir": os.environ.get("DRBM25_REAL_DATA_DIR", str(REPO / "data" / "msmarco")),
        },
    )
    start = time.monotonic()
    info = prepare(cfg)
    assert info.n_docs == 800_000
    assert read_index_info(cfg.retrieval.index_dir) == info
    logging.getLogger(__name__).info("prepare wall time: %.1fs", time.monotonic() - start)


def test_prepare_rejects_collection_with_wrong_doc_count(cfg: Config, fake_download: Any) -> None:
    with pytest.raises(ValueError, match=r"expected 800000 docs, found 3"):
        prepare(cfg)
    assert not cfg.retrieval.index_dir.exists()


def test_main_prints_index_ok(cfg: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    spy = create_autospec(prepare, return_value=IndexInfo(n_docs=7, checksum="abc"))
    monkeypatch.setattr(prepare_mod, "prepare", spy)
    code = prepare_mod.main(
        [
            "--config",
            str(REPO / "configs" / "smoke_cpu.yaml"),
            "--set",
            f"retrieval.index_dir={cfg.retrieval.index_dir}",
        ]
    )
    assert code == 0
    assert "index ok" in capsys.readouterr().out
    assert spy.call_args.args[0].retrieval.index_dir == cfg.retrieval.index_dir


def test_main_rejects_malformed_set(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        prepare_mod.main(["--config", str(REPO / "configs" / "smoke_cpu.yaml"), "--set", "no-equals"])
    assert exc.value.code == 2
    assert "key=value" in capsys.readouterr().err
