"""Process-group setup and manual gradient averaging (no DDP wrapper).

DDP is not used because the LoRA layout switches adapters between forwards, so each
backward leaves one adapter unused, which DDP's reducer treats as an error unless
``find_unused_parameters`` is set. Averaging gradients by hand after ``backward()``
is simpler and exact.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as tdist
from torch import nn


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def _env_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise ValueError(f"WORLD_SIZE is set but {name} is not; launch with torchrun or set {name}")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def init_distributed() -> DistInfo:
    """Join the ``env://`` process group when ``WORLD_SIZE`` is set, else run single-process.

    Reads ``RANK``, ``WORLD_SIZE`` and ``LOCAL_RANK`` (as set by torchrun; ``MASTER_ADDR``
    and ``MASTER_PORT`` are read by ``init_process_group`` itself). The backend is NCCL when
    CUDA is available and gloo otherwise; the device is ``cuda:LOCAL_RANK`` or CPU.
    """
    has_cuda = torch.cuda.is_available()
    if "WORLD_SIZE" not in os.environ:
        single = torch.device("cuda", 0) if has_cuda else torch.device("cpu")  # lint: ignore[CUDA_HARDCODE] chosen once
        return DistInfo(0, 1, 0, single)

    world_size = _env_int("WORLD_SIZE")
    rank = _env_int("RANK")
    local_rank = _env_int("LOCAL_RANK")
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, {world_size}), got {rank}")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")

    if has_cuda:
        device = torch.device("cuda", local_rank)  # lint: ignore[CUDA_HARDCODE] device chosen here, once
        torch.cuda.set_device(device)
        tdist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size, device_id=device)
    else:
        device = torch.device("cpu")
        tdist.init_process_group("gloo", init_method="env://", rank=rank, world_size=world_size)
    return DistInfo(rank, world_size, local_rank, device)


def allreduce_mean_grads(params: Sequence[nn.Parameter], dist: DistInfo) -> None:
    """Replace each ``p.grad`` by its mean over ranks; params whose grad is None are skipped.

    Every rank must call this with the same ``params`` in the same order, and a given
    param must have a grad on all ranks or on none: otherwise the collectives would pair
    up different tensors (silent corruption or a hang). That is a programmer error, so
    the grad-presence pattern is all-reduced first and a mismatch raises ``RuntimeError``
    on every rank before any gradient is touched.
    """
    if dist.world_size == 1:
        return
    present = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=dist.device)
    counts = present.clone()
    tdist.all_reduce(counts, op=tdist.ReduceOp.SUM)
    if bool(((counts != 0) & (counts != dist.world_size)).any()):
        bad = [i for i, c in enumerate(counts.tolist()) if c not in (0, dist.world_size)]
        raise RuntimeError(f"params at indices {bad} have a grad on some ranks but not others")
    for p in params:
        if p.grad is None:
            continue
        tdist.all_reduce(p.grad, op=tdist.ReduceOp.SUM)
        p.grad.div_(dist.world_size)
