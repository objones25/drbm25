import json
import re
from pathlib import Path

import pytest

from drbm25.bm25.index import read_index_info, tsv_to_jsonl


def _read_jsonl(out_dir: Path) -> list[dict[str, str]]:
    files = sorted(out_dir.glob("*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def test_tsv_to_jsonl_writes_id_and_contents(tmp_path: Path) -> None:
    tsv = tmp_path / "c.tsv"
    tsv.write_text("1\tsolar panels\n2\tefficiency\twith tab\n", encoding="utf-8")
    n = tsv_to_jsonl(tsv, tmp_path / "out")
    assert n == 2
    assert _read_jsonl(tmp_path / "out") == [
        {"id": "1", "contents": "solar panels"},
        {"id": "2", "contents": "efficiency\twith tab"},
    ]


def test_tsv_to_jsonl_rejects_duplicate_ids(tmp_path: Path) -> None:
    tsv = tmp_path / "c.tsv"
    tsv.write_text("1\ta\n2\tb\n1\tc\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate doc id '1' on line 3"):
        tsv_to_jsonl(tsv, tmp_path / "out")


def test_tsv_to_jsonl_rejects_line_without_tab(tmp_path: Path) -> None:
    tsv = tmp_path / "c.tsv"
    tsv.write_text("1\ta\nno-tab-here\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2"):
        tsv_to_jsonl(tsv, tmp_path / "out")


@pytest.mark.java
def test_index_counts_docs(toy_index: Path) -> None:
    assert read_index_info(toy_index).n_docs == 1002


@pytest.mark.java
def test_index_checksum_is_stable_sha256(toy_index: Path) -> None:
    first = read_index_info(toy_index)
    assert re.fullmatch(r"[0-9a-f]{64}", first.checksum)
    assert read_index_info(toy_index) == first


def test_read_index_info_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_index_info(tmp_path / "nope")
