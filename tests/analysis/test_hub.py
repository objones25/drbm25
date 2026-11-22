import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import CommitOperationAdd

from drbm25.analysis.hub import dataset_card, upload_database
from drbm25.eval.sample_db import read_db

WHOAMI = {"name": "objones25", "auth": {"accessToken": {"role": "write"}}}


@dataclass
class _Commit:
    oid: str


@dataclass
class FakeHub:
    """An in-memory dataset repo; ``corrupt`` names a file whose stored bytes differ from what was uploaded."""

    files: dict[str, bytes] = field(default_factory=dict)
    created: list[dict[str, Any]] = field(default_factory=list)
    corrupt: str | None = None
    commits: int = 0

    def whoami(self) -> dict[str, Any]:
        return WHOAMI

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.created.append({"repo_id": repo_id, **kwargs})

    def create_commit(self, repo_id: str, operations: list[CommitOperationAdd], **kwargs: Any) -> _Commit:
        assert repo_id == "objones25/drbm25-rewrites"
        assert kwargs["repo_type"] == "dataset"
        for op in operations:
            assert isinstance(op.path_or_fileobj, str)  # the uploader passes local paths
            data = Path(op.path_or_fileobj).read_bytes()
            self.files[op.path_in_repo] = data + (b"x" if op.path_in_repo == self.corrupt else b"")
        self.commits += 1
        return _Commit(oid=f"sha{self.commits}")

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: Any) -> str:
        assert repo_id == "objones25/drbm25-rewrites"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["force_download"] is True
        dest = Path(kwargs["local_dir"]) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[filename])
        return str(dest)


def test_upload_writes_marker_only_after_every_file_verifies(db: Path, metrics_dir: Path) -> None:
    hub = FakeHub()

    marker = upload_database(db, metrics_dir, repo_id="objones25/drbm25-rewrites", api=hub)

    assert hub.created == [
        {"repo_id": "objones25/drbm25-rewrites", "repo_type": "dataset", "private": True, "exist_ok": True}
    ]
    assert "README.md" in hub.files
    assert "db/trained-normal/shard-00000.parquet" in hub.files
    assert "metrics/comparisons.json" in hub.files
    assert not any(p.endswith("._SUCCESS") for p in hub.files)
    saved = json.loads((db / "UPLOADED.json").read_text())
    assert saved == marker
    assert saved["repo_id"] == "objones25/drbm25-rewrites"
    assert saved["commit"] == "sha1"
    assert set(saved["files"]) == set(hub.files)


def test_upload_mismatch_fails_and_writes_no_marker(db: Path, metrics_dir: Path) -> None:
    hub = FakeHub(corrupt="metrics/lexical.json")

    with pytest.raises(RuntimeError, match=r"metrics/lexical\.json"):
        upload_database(db, metrics_dir, repo_id="objones25/drbm25-rewrites", api=hub)

    assert not (db / "UPLOADED.json").exists()


def test_upload_rejects_another_users_namespace(db: Path, metrics_dir: Path) -> None:
    with pytest.raises(ValueError, match="objones25"):
        upload_database(db, metrics_dir, repo_id="someone-else/x", api=FakeHub())


def test_dataset_card_lists_sources_models_and_schema(db: Path) -> None:
    card = dataset_card(read_db(db), repo_id="objones25/drbm25-rewrites")

    wanted = ("untrained", "trained-normal", "trained-count_once", "original", "think_text", "ndcg_count_once")
    assert [t for t in wanted if t not in card] == []
    assert card.startswith("---\n")
    assert "license:" in card


def test_upload_ignores_a_stale_marker_from_an_earlier_upload(db: Path, metrics_dir: Path) -> None:
    (db / "UPLOADED.json").write_text("{}")
    hub = FakeHub()

    upload_database(db, metrics_dir, repo_id="objones25/drbm25-rewrites", api=hub)

    assert "db/UPLOADED.json" not in hub.files


def test_upload_fails_on_empty_database(tmp_path: Path, metrics_dir: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no rows"):
        upload_database(tmp_path / "empty", metrics_dir, repo_id="objones25/drbm25-rewrites", api=FakeHub())
