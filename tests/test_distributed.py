"""init_distributed and allreduce_mean_grads: single process in-proc, two gloo processes via subprocess."""

import socket
import subprocess
import sys
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import drbm25.distributed as distributed_module
from drbm25.distributed import DistInfo, allreduce_mean_grads, init_distributed

TIMEOUT = timedelta(minutes=60)
DIST_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture
def no_dist_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in DIST_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.usefixtures("no_dist_env")
def test_init_distributed_without_env_is_single_process() -> None:
    info = init_distributed(timeout=TIMEOUT)
    expected_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    assert info == DistInfo(rank=0, world_size=1, local_rank=0, device=expected_device)
    assert not torch.distributed.is_initialized()


@pytest.mark.usefixtures("no_dist_env")
@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"WORLD_SIZE": "2"}, "RANK"),
        ({"WORLD_SIZE": "2", "RANK": "2", "LOCAL_RANK": "0"}, "RANK"),
        ({"WORLD_SIZE": "0", "RANK": "0", "LOCAL_RANK": "0"}, "WORLD_SIZE"),
        ({"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "-1"}, "LOCAL_RANK"),
    ],
)
def test_init_distributed_rejects_bad_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], match: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=match):
        init_distributed(timeout=TIMEOUT)
    assert not torch.distributed.is_initialized()


@pytest.mark.usefixtures("no_dist_env")
def test_init_distributed_passes_the_timeout_to_the_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rank 0 can spend minutes on a save while others wait in a collective; the timeout must be explicit.
    seen: dict[str, Any] = {}

    def fake_init(backend: str, **kwargs: Any) -> None:
        seen.update(kwargs, backend=backend)

    monkeypatch.setattr(distributed_module.tdist, "init_process_group", fake_init)
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: False)
    for key, value in {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}.items():
        monkeypatch.setenv(key, value)
    info = init_distributed(timeout=timedelta(minutes=45))
    assert (info.rank, info.world_size) == (1, 2)
    assert seen["backend"] == "gloo"
    assert seen["timeout"] == timedelta(minutes=45)


def test_allreduce_single_process_is_identity() -> None:
    a = nn.Parameter(torch.ones(3))
    b = nn.Parameter(torch.ones(2, 2))
    a.grad = torch.tensor([1.0, -2.0, 3.5])
    info = DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))
    allreduce_mean_grads([a, b], info)
    torch.testing.assert_close(a.grad, torch.tensor([1.0, -2.0, 3.5]), rtol=0, atol=0)
    assert b.grad is None


_WORKER = textwrap.dedent(
    """
    import sys
    import torch
    from torch import nn
    import torch.distributed as tdist
    from drbm25.distributed import allreduce_mean_grads, init_distributed

    from datetime import timedelta

    info = init_distributed(timeout=timedelta(minutes=5))
    p = nn.Parameter(torch.zeros(2))
    q = nn.Parameter(torch.zeros(1))  # never gets a grad on any rank: skipped
    p.grad = torch.full((2,), float(info.rank + 1))
    allreduce_mean_grads([p, q], info)
    assert q.grad is None
    ok = torch.equal(p.grad, torch.full((2,), 1.5))
    mismatch = nn.Parameter(torch.zeros(1))
    if info.rank == 0:
        mismatch.grad = torch.ones(1)
    try:
        allreduce_mean_grads([mismatch], info)
        raised = False
    except RuntimeError:
        raised = True
    tdist.destroy_process_group()
    print(f"rank={info.rank} world={info.world_size} ok={ok} raised={raised}")
    sys.exit(0 if ok and raised else 1)
    """
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.slow
def test_allreduce_two_gloo_processes_averages_and_detects_missing_grads(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    port = _free_port()
    src = str(Path(__file__).resolve().parents[1] / "src")
    procs = []
    for rank in range(2):
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": src,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
        }
        procs.append(
            subprocess.Popen(  # noqa: S603 -- fixed argv: this interpreter and a script we wrote
                [sys.executable, str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        )
    outputs = [p.communicate(timeout=120) for p in procs]
    for rank, (proc, (out, err)) in enumerate(zip(procs, outputs, strict=True)):
        assert proc.returncode == 0, f"rank {rank} failed:\n{out}\n{err}"
        assert f"rank={rank} world=2 ok=True raised=True" in out
