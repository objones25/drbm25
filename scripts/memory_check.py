"""Peak GPU memory per rank for one PPO step of the full A100 layout (FSDP2 + colocated vLLM).

Run on the pod (Task 17 smoke stage), from the repo root:

    torchrun --nproc_per_node 2 scripts/memory_check.py --config configs/runpod_a100_full.yaml \
        [--set key=value ...] [--steps N]

Builds the layout exactly as ``drbm25.train`` does (vLLM first, then the sharded actor, critic
and reference), then runs ``--steps`` PPO steps (default 1) on synthetic queries with a fake
BM25 searcher, so no index or dataset is needed. Each rank prints one line per phase:

    rank R | phase | torch peak GiB | torch reserved GiB | device used GiB | device total GiB

"torch peak" is ``torch.cuda.max_memory_allocated`` (training tensors only: vLLM's sleep-mode
allocations go through its own CuMem pool). "device used" is ``total - free`` from
``torch.cuda.mem_get_info`` and includes vLLM. The A100 budget (spec §5) is 80 GiB per GPU; if
"device used" after the step is within a few GiB of it, apply the spec's mitigations in order.
Exit code 0 when the step finished, 1 on any error (an OOM included).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as tdist

from drbm25.bm25.searcher import SearchResult
from drbm25.config import ScoringMode, load_config, validate_for_runtime
from drbm25.data import Example
from drbm25.distributed import DistInfo, init_distributed
from drbm25.ppo.trainer import PPOTrainer
from drbm25.train import build_layout, check_backend, check_full_layout

_GIB = 1024**3


class _FakeSearcher:
    """Half the queries rank the relevant doc first, so rewards and advantages are not all equal."""

    def search(self, query: str, *, mode: ScoringMode, k: int) -> SearchResult:
        del mode, k
        words = query.split()
        doc = "rel" if len(words) % 2 == 0 else "other"
        return SearchResult((doc,), len(words), len(set(words)), (1.0,))

    def batch_search(self, queries: Sequence[str], *, mode: ScoringMode, k: int) -> list[SearchResult]:
        return [self.search(q, mode=mode, k=k) for q in queries]

    def analyze(self, text: str) -> list[str]:
        return text.split()


def _report(dist: DistInfo, phase: str) -> None:
    free, total = torch.cuda.mem_get_info(dist.device)
    print(
        f"rank {dist.rank} | {phase:<14} | torch peak {torch.cuda.max_memory_allocated(dist.device) / _GIB:6.2f} GiB"
        f" | torch reserved {torch.cuda.memory_reserved(dist.device) / _GIB:6.2f} GiB"
        f" | device used {(total - free) / _GIB:6.2f} GiB | device total {total / _GIB:6.2f} GiB",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=1)
    ns = parser.parse_args(argv)
    overrides = dict(item.split("=", 1) for item in ns.overrides)
    dist = init_distributed()
    if dist.device.type != "cuda":
        print("memory_check needs CUDA", file=sys.stderr)
        return 1
    try:
        cfg = load_config(ns.config, overrides)
        check_full_layout(cfg)
        check_backend(cfg, dist.device)
        validate_for_runtime(
            cfg, device_capability=torch.cuda.get_device_capability(dist.device), world_size=dist.world_size
        )
        _report(dist, "start")
        layout = build_layout(cfg, dist)
        _report(dist, "built")
        trainer = PPOTrainer(
            cfg,
            models=layout.models,
            engine=layout.engine,
            searcher=_FakeSearcher(),
            reward_mode="normal",
            dist=dist,
            scaler=None,
        )
        local = cfg.ppo.batch_size // dist.world_size
        examples = [
            Example(
                qid=f"q{i}", query=f"effect of drug {i} on blood pressure in adults", relevant_ids=frozenset({"rel"})
            )
            for i in range(local)
        ]
        for step in range(ns.steps):
            torch.cuda.reset_peak_memory_stats(dist.device)
            metrics = trainer.step(examples, step=step)
            _report(dist, f"after step {step}")
            if dist.rank == 0:
                print(
                    f"step {step}: response_len_mean {metrics['rollout/response_len_mean']:.1f}, "
                    f"time/generate {metrics['time/generate']:.1f}s, time/update {metrics['time/update']:.1f}s, "
                    f"time/sync {metrics['time/sync']:.1f}s, sys/gpu_mem_peak_gb {metrics['sys/gpu_mem_peak_gb']:.2f}",
                    flush=True,
                )
    finally:
        if tdist.is_initialized():
            tdist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
