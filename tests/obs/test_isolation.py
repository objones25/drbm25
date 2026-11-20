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


def test_shutdown_makes_configure_effective_again() -> None:
    setup.configure(level="INFO", force=True)
    assert setup._configured is True
    setup.shutdown()
    assert setup._configured is False
    assert setup.configure(level="INFO") != {}  # not a no-op


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
