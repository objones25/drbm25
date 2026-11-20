from __future__ import annotations

from pathlib import Path

import pytest

from drbm25.config import Config, load_config

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return load_config(
        REPO / "configs" / "smoke_cpu.yaml",
        {"retrieval.index_dir": str(tmp_path / "index"), "data.data_dir": str(tmp_path / "data")},
    )
