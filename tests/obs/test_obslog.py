"""Tests for the vendored logging and W&B plumbing in drbm25.obs.

The W&B tests run in offline mode and need no account. They are skipped if
wandb is not installed.
"""

from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from drbm25.obs.context import ContextFilter, log_context, new_request_id
from drbm25.obs.json_formatter import JSONFormatter
from drbm25.obs.redact import REDACTED, RedactingFilter
from drbm25.obs.setup import build_config, configure

SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def _reset_logging():
    """Each test gets a clean root logger; logging is global state."""
    root = logging.getLogger()  # obs: allow LOG003 -- the fixture resets root on purpose
    saved = list(root.handlers), root.level
    for handler in list(root.handlers):
        root.removeHandler(handler)
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved[0]:
        root.addHandler(handler)
    root.setLevel(saved[1])


class _ExtraRecord(logging.LogRecord):
    """LogRecord whose `extra=` fields the checker may read as attributes."""

    if TYPE_CHECKING:
        def __getattr__(self, name: str) -> Any: ...


def make_record(**extra) -> _ExtraRecord:
    record = _ExtraRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="order %s received", args=(1001,), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# --- JSON formatter -------------------------------------------------------

def test_json_formatter_emits_one_object_per_record():
    out = json.loads(JSONFormatter().format(make_record()))
    assert out["message"] == "order 1001 received"
    assert out["level"] == "INFO"
    assert out["timestamp"].endswith("+00:00")


def test_extra_fields_are_promoted_to_top_level():
    out = json.loads(JSONFormatter().format(make_record(order_id=1001, tier="gold")))
    assert out["order_id"] == 1001
    assert out["tier"] == "gold"


def test_static_fields_appear_on_every_record():
    fmt = JSONFormatter(static={"service": "checkout", "env": "prod"})
    out = json.loads(fmt.format(make_record()))
    assert out["service"] == "checkout"
    assert out["env"] == "prod"


def test_exception_is_structured_not_a_blob():
    try:
        raise ValueError("payment declined")
    except ValueError:  # obs: allow LOG007 -- capturing exc_info is the point
        record = make_record()
        record.exc_info = sys.exc_info()
    out = json.loads(JSONFormatter().format(record))
    assert out["exception"]["type"] == "ValueError"
    assert out["exception"]["message"] == "payment declined"
    assert "Traceback" in out["exception"]["traceback"]
    # The traceback must not be glued onto the message.
    assert "Traceback" not in out["message"]


def test_unserializable_value_does_not_kill_the_record():
    out = json.loads(JSONFormatter().format(make_record(path=Path("/data/x"))))
    assert out["path"] == "/data/x"


# --- context --------------------------------------------------------------

def test_context_filter_attaches_ambient_fields():
    record = make_record()
    with log_context(request_id="abc123"):
        ContextFilter().filter(record)
    assert record.request_id == "abc123"


def test_context_is_isolated_per_thread():
    """Two concurrent requests must not see each other's ids."""
    seen: dict[str, str] = {}

    def handle(name: str) -> None:
        with log_context(request_id=name):
            record = make_record()
            ContextFilter().filter(record)
            seen[name] = record.request_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(handle, ["r1", "r2", "r3", "r4"]))
    assert seen == {"r1": "r1", "r2": "r2", "r3": "r3", "r4": "r4"}


def test_context_resets_on_block_exit():
    record = make_record()
    with log_context(request_id="inner"):
        pass
    ContextFilter().filter(record)
    assert not hasattr(record, "request_id")


def test_new_request_id_is_unique():
    assert len({new_request_id() for _ in range(500)}) == 500


# --- redaction ------------------------------------------------------------

def test_secret_named_field_is_redacted():
    record = make_record(api_key="sk-live-1234567890", user="ana")
    RedactingFilter().filter(record)
    assert record.api_key == REDACTED
    assert record.user == "ana"


def test_nested_dict_secrets_are_redacted():
    record = make_record(payload={"user": "ana", "password": "hunter2"})
    RedactingFilter().filter(record)
    assert record.payload == {"user": "ana", "password": REDACTED}


def test_ml_config_names_are_not_mistaken_for_secrets():
    """Substring matching used to redact max_tokens, tokenizer, author, passes."""
    benign = dict(max_tokens=512, tokenizer="gpt2", tokens_per_sec=1e4,
                  author="ana", num_passes=2, pad_token_id=0)
    secrets = {"hf_token": "hf_x", "apiKey": "k", "db_password": "p"}
    record = make_record(**benign, **secrets)
    RedactingFilter().filter(record)
    for key, value in benign.items():
        assert getattr(record, key) == value, key
    assert record.hf_token == record.apiKey == record.db_password == REDACTED


def test_message_scanning_is_opt_in():
    record = make_record()
    record.msg, record.args = "using Bearer abcdef1234567890", ()
    RedactingFilter(scan_message=True).filter(record)
    assert REDACTED in record.getMessage()


# --- setup / dictConfig ---------------------------------------------------

def test_all_handlers_live_on_root():
    config = build_config()
    assert config["root"]["handlers"]
    assert all("handlers" not in spec for spec in config["loggers"].values())


def test_disable_existing_loggers_is_off():
    assert build_config()["disable_existing_loggers"] is False


def test_console_goes_to_stdout_not_stderr():
    assert build_config()["handlers"]["stdout"]["stream"] == "ext://sys.stdout"


def test_configure_is_idempotent():
    first = configure(level="INFO", force=True)
    second = configure(level="DEBUG")
    assert first
    assert second == {}
    assert len(logging.getLogger().handlers) == 1  # obs: allow LOG003


@pytest.mark.skipif(sys.version_info < (3, 12), reason="queue auto-wiring is 3.12+")
def test_build_config_works_without_configure():
    """Django's LOGGING = build_config(...) and gunicorn's logconfig_dict call
    dictConfig themselves and never reach configure(). The queue listener must
    still start, or every record sits in the queue and nothing is printed."""
    code = (
        "import logging, logging.config\n"
        "from drbm25.obs.setup import build_config\n"
        "logging.config.dictConfig(build_config(level='INFO'))\n"
        "logging.getLogger('app').warning('visible')\n"
    )
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no external input
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": str(SRC), "PATH": ""}, timeout=60,
    )
    assert "visible" in proc.stdout, proc.stderr


@pytest.mark.skipif(sys.version_info < (3, 12), reason="queue auto-wiring is 3.12+")
def test_queue_handler_preserves_exc_info_and_context(capsys):
    """The whole point of the custom QueueHandler: structured output survives."""
    configure(level="INFO", fmt="json", use_queue=True, force=True)
    logger = logging.getLogger("queue-test")
    with log_context(request_id="ctx-1"):
        try:
            raise RuntimeError("kaboom")
        except RuntimeError:
            logger.exception("failed while charging")
    handler = logging.getHandlerByName("queue")
    assert isinstance(handler, logging.handlers.QueueHandler)
    assert handler.listener is not None
    handler.listener.stop()  # flush before reading

    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")][-1]
    out = json.loads(line)
    assert out["request_id"] == "ctx-1"
    assert out["exception"]["type"] == "RuntimeError"
    assert "Traceback" not in out["message"]


def test_stock_queue_handler_would_lose_the_structure(capsys):
    """Documents why ContextQueueHandler exists -- this is the failure mode."""
    import logging.handlers
    import queue as queue_mod

    q: queue_mod.Queue = queue_mod.Queue(-1)
    sink = logging.StreamHandler(sys.stdout)
    sink.setFormatter(JSONFormatter())
    listener = logging.handlers.QueueListener(q, sink)
    listener.start()
    logger = logging.getLogger("stock-queue")
    logger.propagate = False
    logger.addHandler(logging.handlers.QueueHandler(q))
    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        logger.exception("failed")
    listener.stop()

    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert "exception" not in out          # exc_info was cleared
    assert "Traceback" in out["message"]   # ...and flattened into the message


# --- W&B (offline) --------------------------------------------------------

wandb = pytest.importorskip("wandb")


@pytest.fixture()
def offline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    return tmp_path


def test_start_run_finishes_the_run(offline_env):
    from drbm25.obs.wandb_run import start_run

    with start_run(project="test", config={"lr": 0.1}, dir=str(offline_env)) as run:
        run.log({"loss": 1.0})
        run_id = run.id
    assert wandb.run is None, "the run was left open"
    assert run_id


def test_failure_inside_the_block_marks_the_run_failed(offline_env):
    from drbm25.obs.wandb_run import start_run

    def diverge() -> None:
        with start_run(project="test", config={}, dir=str(offline_env)) as run:
            run.log({"loss": float("nan")})
            raise FloatingPointError("diverged")

    with pytest.raises(FloatingPointError, match="diverged"):
        diverge()
    assert wandb.run is None


def test_config_keys_are_immutable_without_allow_val_change(offline_env):
    from drbm25.obs.wandb_run import start_run

    with start_run(project="test", config={"lr": 0.1}, dir=str(offline_env)) as run:
        with pytest.raises(wandb.errors.Error, match="Attempted to change value"):
            run.config["lr"] = 0.2
        run.config.update({"lr": 0.2}, allow_val_change=True)
        assert run.config["lr"] == pytest.approx(0.2)


def test_summary_goal_changes_the_shape_of_the_summary_value(offline_env):
    """define_metric(summary="min") makes summary[name] a sub-dict, not a float."""
    from drbm25.obs.wandb_run import start_run

    with start_run(
        project="test", config={}, dir=str(offline_env),
        summary_goals={"val/loss": "min"},
    ) as run:
        run.log({"val/loss": 0.5})
        run.log({"val/loss": 0.2})
        run.log({"val/loss": 0.9})
        assert dict(run.summary["val/loss"]) == {"min": pytest.approx(0.2)}


def test_out_of_order_steps_are_dropped(offline_env):
    """Why WB006 exists: logging backwards loses data with no exception."""
    from drbm25.obs.wandb_run import start_run

    with start_run(project="test", config={}, dir=str(offline_env)) as run:
        run.log({"loss": 1.0}, step=0)  # obs: allow WB006
        run.log({"loss": 0.5}, step=5)  # obs: allow WB006
        assert run.step == 5

        run.log({"loss": 99.0}, step=1)  # obs: allow WB006 -- discarded, no exception
        assert run.step == 5, "a backwards step must not rewind the run"

        # A step commits when the next one starts, so this flushes step 5.
        run.log({"loss": 0.25}, step=6)  # obs: allow WB006
        assert run.summary["loss"] == pytest.approx(0.5), "the step-1 write should have vanished"


def test_redact_config_strips_credentials():
    from drbm25.obs.wandb_run import redact_config

    out = redact_config({"lr": 0.1, "hf_token": "hf_abcdefg", "api_key": "sk-x"})
    assert out["lr"] == pytest.approx(0.1)
    assert out["hf_token"] == REDACTED
    assert out["api_key"] == REDACTED


def test_resolve_mode_falls_back_to_offline(monkeypatch):
    from drbm25.obs import wandb_run

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(wandb, "login", lambda **_kw: False)
    assert wandb_run.resolve_mode() == "offline"
    assert wandb_run.resolve_mode("disabled") == "disabled"


def test_resolve_mode_asks_wandb_about_credentials(monkeypatch):
    """wandb 0.31 logs in against api.forge.coreweave.com, so a netrc grep for
    api.wandb.ai misses a fresh login and silently runs offline."""
    from drbm25.obs import wandb_run

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(wandb, "login", lambda **_kw: True)
    assert wandb_run.resolve_mode() == "online"


def test_start_run_falls_back_offline_when_server_unreachable(
    offline_env, monkeypatch
):
    """With credentials but no network, wandb.init raises CommError after
    init_timeout instead of degrading; start_run must degrade for it."""
    from drbm25.obs import wandb_run

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(wandb_run, "resolve_mode", lambda _requested=None: "online")
    real_init = wandb.init

    def refuse_online(**_kwargs):
        raise wandb.errors.CommError("connection refused")

    by_mode = {"online": refuse_online, "offline": real_init}

    def fake_init(**kwargs):
        return by_mode[kwargs["mode"]](**kwargs)

    monkeypatch.setattr(wandb, "init", fake_init)
    with wandb_run.start_run(project="t", config={}, dir=str(offline_env)) as run:
        assert run.offline


def test_start_run_does_not_hide_a_bad_key(offline_env, monkeypatch):
    """AuthenticationError subclasses CommError; a revoked key must raise,
    not silently degrade to offline as an unreachable server does."""
    from drbm25.obs import wandb_run

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(wandb_run, "resolve_mode", lambda _requested=None: "online")

    def fake_init(**kwargs):
        raise wandb.errors.AuthenticationError("invalid API key")

    monkeypatch.setattr(wandb, "init", fake_init)
    with pytest.raises(wandb.errors.AuthenticationError), wandb_run.start_run(
        project="t", config={}, dir=str(offline_env)
    ):
        pass


def test_caught_exception_without_context_manager_leaves_run_open(offline_env):
    """Why WB003 exists: an uncaught crash is marked failed by wandb's own
    excepthook, but an exception caught further up never reaches it."""
    run = wandb.init(project="t", config={}, dir=str(offline_env))  # obs: allow WB003
    try:
        raise RuntimeError("step failed")
    except RuntimeError:
        logging.getLogger("t").exception("handled upstream")
    assert wandb.run is run, "still open: it will be recorded as finished, exit 0"
    run.finish(exit_code=1)
