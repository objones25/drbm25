"""The autouse `_restore_logging` fixture keeps configure() from leaking between tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from drbm25.obs import setup

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"

_TWO_TESTS = '''
import logging
from drbm25.obs import setup


def test_a_calls_configure():
    setup.configure(level="INFO", force=True)
    logging.getLogger("leak.a").warning("hello")
    assert setup._configured is True


def test_b_sees_clean_state_and_caplog_works(caplog):
    assert setup._configured is False
    handlers = list(logging.getLogger().handlers)
    assert [h for h in handlers if type(h).__name__ == "ContextQueueHandler"] == []
    assert [h for h in handlers if getattr(getattr(h, "stream", None), "closed", False)] == []
    with caplog.at_level(logging.INFO, logger="leak.b"):
        logging.getLogger("leak.b").info("captured")
    assert [r.getMessage() for r in caplog.records] == ["captured"]
'''


_COUNT_TESTS = '''
import logging
import pytest


@pytest.mark.parametrize("i", range(5))
def test_n(i):
    print("ROOT_HANDLERS", len(logging.getLogger().handlers))
'''


def test_shutdown_makes_configure_effective_again() -> None:
    first = setup.configure(level="INFO", force=True)
    assert first["root"]["handlers"] == ["queue"]
    assert setup._configured is True
    assert setup.configure(level="DEBUG") == {}  # second call is a no-op while configured
    setup.shutdown()
    assert setup._configured is False
    again = setup.configure(level="DEBUG")
    assert again["root"]["handlers"] == ["queue"]
    assert again["handlers"]["stdout"]["level"] == "DEBUG"
    assert setup._configured is True


def test_root_handler_count_does_not_grow_across_tests(tmp_path: Path) -> None:
    """Five tests in a fresh pytest process; the root handler count must stay constant."""
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_count.py").write_text(_COUNT_TESTS)
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no external input
        [sys.executable, "-m", "pytest", "-p", "no:randomly", "-p", "no:cacheprovider", "-q", "-s", "--no-header", "--rootdir", str(tmp_path), str(tmp_path / "test_count.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    counts = [int(line.split()[1]) for line in proc.stdout.splitlines() if "ROOT_HANDLERS" in line]
    assert len(counts) == 5
    assert len(set(counts)) == 1, counts


def test_configure_in_one_test_does_not_leak_into_the_next(tmp_path: Path) -> None:
    """Run two ordered tests in a fresh pytest process using this repo's conftest."""
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_two.py").write_text(_TWO_TESTS)
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no external input
        [sys.executable, "-m", "pytest", "-p", "no:randomly", "-p", "no:cacheprovider", "-q", "--no-header", "-W", "error", "--rootdir", str(tmp_path), str(tmp_path / "test_two.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "2 passed" in proc.stdout
