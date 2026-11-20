"""Peak GPU memory per rank for the full A100 layout (FSDP2 + colocated vLLM), through every phase.

Run on the pod (Task 17 smoke stage), from the repo root:

    torchrun --nproc_per_node 2 scripts/memory_check.py --config configs/runpod_a100_full.yaml \
        [--set key=value ...] [--steps 2] [--ckpt-dir /workspace/drbm25/memcheck]

Builds the layout exactly as ``drbm25.train`` does (vLLM first, then the sharded actor, critic
and reference) and runs, on synthetic queries with a fake BM25 searcher (no index or dataset):
``--steps`` PPO steps, one validation pass (greedy generation), one sharded DCP checkpoint save,
and a resume (load of that checkpoint into the live models and optimizers).

A background thread samples ``torch.cuda.mem_get_info`` every 50 ms, so the peak includes the
vLLM generation phase (vLLM allocates through its own CuMem pool, which
``torch.cuda.max_memory_allocated`` does not see). Each rank prints one line per phase:

    rank R | phase | torch peak | device peak (sampled) | device total | bound | headroom

``bound`` = torch peak + vLLM budget (``gpu_memory_utilization`` x total), an upper estimate of
what the phase can need when vLLM is awake. Exit code 0 when every phase finished and the
sampled device peak stayed below the total; 1 on any error (an OOM included) or when the
headroom is under 2 GiB, the point at which to apply the spec's mitigations in order (reference
model on CPU, then a lower ``rollout.vllm_gpu_memory_utilization``).
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as tdist

from drbm25.bm25.searcher import SearchResult
from drbm25.checkpoint import TrainState, latest_checkpoint, load_sharded_checkpoint, save_sharded_checkpoint
from drbm25.config import Config, ScoringMode, dist_timeout, load_config, validate_for_runtime
from drbm25.data import Example
from drbm25.distributed import DistInfo, init_distributed
from drbm25.ppo.trainer import PPOTrainer
from drbm25.train import check_backend, check_full_layout, init_layout

_GIB = 1024**3
_SAMPLE_S = 0.05
_HEADROOM_MIN_GIB = 2.0


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


class _DevicePeak:
    """Highest ``total - free`` seen by a sampling thread since the last ``reset``."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _used(self) -> int:
        free, total = torch.cuda.mem_get_info(self._device)
        return total - free

    def _run(self) -> None:
        torch.cuda.set_device(self._device)
        while not self._stop.wait(_SAMPLE_S):
            self._peak = max(self._peak, self._used())

    def __enter__(self) -> _DevicePeak:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def reset(self) -> None:
        self._peak = self._used()

    def peak(self) -> int:
        return max(self._peak, self._used())


class _Reporter:
    def __init__(self, dist: DistInfo, cfg: Config, sampler: _DevicePeak) -> None:
        self._dist = dist
        self._vllm_budget = cfg.rollout.vllm_gpu_memory_utilization if cfg.rollout.backend == "vllm" else 0.0
        self._sampler = sampler
        self.min_headroom_gib = float("inf")

    @contextmanager
    def phase(self, name: str) -> Generator[None]:
        torch.cuda.reset_peak_memory_stats(self._dist.device)
        self._sampler.reset()
        yield
        torch.cuda.synchronize(self._dist.device)
        total = torch.cuda.mem_get_info(self._dist.device)[1]
        torch_peak = torch.cuda.max_memory_allocated(self._dist.device)
        device_peak = self._sampler.peak()
        bound = torch_peak + self._vllm_budget * total
        headroom = (total - device_peak) / _GIB
        self.min_headroom_gib = min(self.min_headroom_gib, headroom)
        print(
            f"rank {self._dist.rank} | {name:<16} | torch peak {torch_peak / _GIB:6.2f} GiB"
            f" | device peak {device_peak / _GIB:6.2f} GiB | device total {total / _GIB:6.2f} GiB"
            f" | bound {bound / _GIB:6.2f} GiB | headroom {headroom:6.2f} GiB",
            flush=True,
        )


def _examples(n: int) -> list[Example]:
    return [
        Example(qid=f"q{i}", query=f"effect of drug {i} on blood pressure in adults", relevant_ids=frozenset({"rel"}))
        for i in range(n)
    ]


def _run(cfg: Config, dist: DistInfo, steps: int, ckpt_dir: Path, report: _Reporter) -> None:
    with report.phase("build"):
        layout = init_layout(cfg, dist)
    if layout.full is None:
        raise ValueError("memory_check measures the full layout: use a config with model.train_mode 'full'")
    trainer = PPOTrainer(
        cfg, models=layout.models, engine=layout.engine, searcher=_FakeSearcher(), reward_mode="normal", dist=dist,
        scaler=None,
    )  # fmt: skip
    with report.phase("initial sync"):
        layout.engine.sync_weights(layout.models)
    examples = _examples(cfg.ppo.batch_size // dist.world_size)
    for step in range(steps):
        with report.phase(f"step {step}"):
            metrics = trainer.step(examples, step=step)
        if dist.rank == 0:
            print(
                f"step {step}: response_len_mean {metrics['rollout/response_len_mean']:.1f}, "
                f"time/generate {metrics['time/generate']:.1f}s, time/update {metrics['time/update']:.1f}s, "
                f"time/sync {metrics['time/sync']:.1f}s",
                flush=True,
            )
    with report.phase("validation"):
        trainer.validate_with_samples(_examples(cfg.data.val_size))
    state = TrainState(step=steps, data_cursor=steps * cfg.ppo.batch_size, wandb_run_id=None, reward_mode="normal")
    with report.phase("checkpoint save"):
        save_sharded_checkpoint(
            ckpt_dir, train_state=state, state=layout.full.sharded_state(trainer.optimizers()), keep=1, dist=dist
        )
    path = latest_checkpoint(ckpt_dir)
    if path is None:
        raise RuntimeError(f"no complete checkpoint in {ckpt_dir} after the save")
    with report.phase("resume load"):
        template = layout.full.sharded_state(trainer.optimizers())
        load_sharded_checkpoint(path, template)
        layout.full.load_sharded_state(template, trainer.optimizers())
        layout.engine.sync_weights(layout.models)
    with report.phase(f"step {steps} (resumed)"):
        trainer.step(examples, step=steps)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("runs/memory_check"))
    ns = parser.parse_args(argv)
    overrides = dict(item.split("=", 1) for item in ns.overrides)
    cfg = load_config(ns.config, overrides)
    dist = init_distributed(timeout=dist_timeout(cfg))
    if dist.device.type != "cuda":
        print("memory_check needs CUDA", file=sys.stderr)
        return 1
    try:
        check_full_layout(cfg)
        check_backend(cfg, dist.device)
        validate_for_runtime(
            cfg, device_capability=torch.cuda.get_device_capability(dist.device), world_size=dist.world_size
        )
        with _DevicePeak(dist.device) as sampler:
            report = _Reporter(dist, cfg, sampler)
            _run(cfg, dist, ns.steps, ns.ckpt_dir, report)
        ok = report.min_headroom_gib >= _HEADROOM_MIN_GIB
        print(
            f"rank {dist.rank} | minimum headroom {report.min_headroom_gib:.2f} GiB "
            f"({'OK' if ok else f'under {_HEADROOM_MIN_GIB} GiB: apply the spec mitigations'})",
            flush=True,
        )
        return 0 if ok else 1
    finally:
        if tdist.is_initialized():
            tdist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
