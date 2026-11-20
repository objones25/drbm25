"""Resumable PPO training CLI.

    python -m drbm25.train --config PATH --reward-mode {normal,count_once} --run-name NAME
        [--until-step N] [--deadline-unix T] [--reserve-s S] [--set key=value ...]

Exit codes: 0 reached ``min(until_step, total_steps)``; 3 stopped by the budget deadline;
1 any error. torchrun turns any non-zero worker exit into its own exit code 1, so rank 0 also
writes the code to ``output_dir/run_name/exit_code``; launchers read that file to tell a
budget stop (3) from a crash (1).

Run directory ``output_dir/run_name/``: ``step_NNNNNN/`` checkpoints (rank 0), ``metrics.jsonl``
(rank 0, one JSON object of floats per step, read by ``scripts/spike_monitor.py``),
``train_rank{R}.log`` (JSON log lines per rank, including each step's global batch indices),
``exit_code``. Only the LoRA layout (``model.train_mode: lora``) with the HF rollout backend is
built here; the full fine-tune / vLLM layout is added by the A100 task.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, get_args

import numpy as np
import torch
import torch.distributed as tdist
from safetensors.torch import save_file
from torch import nn
from torch.amp.grad_scaler import GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

import wandb
from drbm25.bm25.index import read_index_info
from drbm25.bm25.searcher import Bm25Searcher
from drbm25.budget import should_stop
from drbm25.checkpoint import (
    TrainState,
    latest_checkpoint,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from drbm25.config import Config, ConfigError, LoraCfg, ScoringMode, load_config, validate_for_runtime
from drbm25.data import DEV_FILE, TRAIN_FILE, Example, batch_indices, load_examples, split_dev
from drbm25.distributed import DistInfo, init_distributed
from drbm25.obs import configure
from drbm25.obs.context import bind_log_context
from drbm25.obs.runs import STEP_METRIC, open_run
from drbm25.obs.wandb_run import log_checkpoint
from drbm25.ppo.models import LoraPolicyModels
from drbm25.ppo.trainer import NonFiniteError, PPOTrainer, TrainingHealthError
from drbm25.rollout.hf_generate import HFGenerateEngine
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer

logger = logging.getLogger(__name__)

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
EXIT_BUDGET: Final = 3
TINY_MODEL: Final = "tiny-local"
REQUIRED_JAVA_MAJOR: Final = 21
CALIBRATION_STEPS: Final = 10
_JAVA_TIMEOUT_S: Final = 30.0
_GIT_TIMEOUT_S: Final = 10.0
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_JAVA_VERSION = re.compile(r'version "(\d+)')
_TMP_PREFIX = ".tmp-"
_DTYPES: Final = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class StartupCheckError(RuntimeError):
    """The environment cannot run this config (wrong GPU arch, Java, index size, tokenizer)."""


@dataclass(frozen=True)
class _Args:
    config: Path
    reward_mode: ScoringMode
    run_name: str
    until_step: int | None
    deadline_unix: float | None
    reserve_s: float
    overrides: dict[str, str]


# ---------------------------------------------------------------------------- arguments
def _run_name(text: str) -> str:
    if not _RUN_NAME.fullmatch(text):
        raise argparse.ArgumentTypeError(f"run name must match {_RUN_NAME.pattern}, got {text!r}")
    return text


def _parse_args(argv: Sequence[str] | None) -> _Args:
    parser = argparse.ArgumentParser(prog="python -m drbm25.train", description="Resumable PPO training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=get_args(ScoringMode), required=True)
    parser.add_argument("--run-name", type=_run_name, required=True)
    parser.add_argument("--until-step", type=int, default=None, help="stop after this many total steps")
    parser.add_argument("--deadline-unix", type=float, default=None, help="budget deadline (unix seconds)")
    parser.add_argument("--reserve-s", type=float, default=0.0, help="seconds kept free before the deadline")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ns = parser.parse_args(argv)
    overrides: dict[str, str] = {}
    for item in ns.overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--set expects key=value, got {item!r}")
        overrides[key] = value
    if ns.until_step is not None and ns.until_step < 0:
        parser.error(f"--until-step must be >= 0, got {ns.until_step}")
    if ns.reserve_s < 0:
        parser.error(f"--reserve-s must be >= 0, got {ns.reserve_s}")
    return _Args(ns.config, ns.reward_mode, ns.run_name, ns.until_step, ns.deadline_unix, ns.reserve_s, overrides)


# ---------------------------------------------------------------------------- startup checks
def _device_capability(device: torch.device) -> tuple[int, int] | None:
    return torch.cuda.get_device_capability(device) if device.type == "cuda" else None


def check_cuda_arch(device: torch.device) -> None:
    """The installed torch must ship kernels for this GPU (V100 = sm_70 needs a cu12.8-or-older wheel)."""
    capability = _device_capability(device)
    if capability is None:
        return
    arch = f"sm_{capability[0]}{capability[1]}"
    archs = torch.cuda.get_arch_list()
    if arch not in archs:
        raise StartupCheckError(f"GPU arch {arch} is not in this torch build's arch list {archs}")


def check_java(java: str = "java") -> None:
    """``java -version`` must report major version 21 (pyserini 2.x)."""
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            [java, "-version"], capture_output=True, text=True, timeout=_JAVA_TIMEOUT_S, check=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StartupCheckError(f"cannot run '{java} -version': {exc}") from exc
    found = _JAVA_VERSION.search(proc.stderr + proc.stdout)
    major = int(found.group(1)) if found else None
    if major != REQUIRED_JAVA_MAJOR:
        raise StartupCheckError(f"Java {REQUIRED_JAVA_MAJOR} is required, '{java} -version' reported {major}")


def check_index(cfg: Config) -> str:
    """The index must hold exactly ``retrieval.expected_docs`` documents; returns its checksum."""
    info = read_index_info(cfg.retrieval.index_dir)
    if info.n_docs != cfg.retrieval.expected_docs:
        raise StartupCheckError(
            f"index {cfg.retrieval.index_dir} has the wrong size: expected {cfg.retrieval.expected_docs} docs, "
            f"found {info.n_docs}"
        )
    return info.checksum


def check_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise StartupCheckError(
            f"tokenizer pad_token_id {tokenizer.pad_token_id} must be set and differ from eos_token_id "
            f"{tokenizer.eos_token_id}"
        )


# ---------------------------------------------------------------------------- building blocks
def build_base(cfg: Config, device: torch.device) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if cfg.model.name == TINY_MODEL:
        tiny = build_tiny_qwen2(seed=cfg.run.seed)
        nn.Module.to(tiny, device)  # transformers' decorated .to() confuses pyright
        return tiny, build_tiny_tokenizer()
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        dtype=_DTYPES[cfg.model.precision],
        attn_implementation="sdpa",
    )
    nn.Module.to(base, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision, padding_side="left")
    return base, tokenizer


def check_layout(cfg: Config, device: torch.device) -> LoraCfg:
    if cfg.model.train_mode != "lora" or cfg.lora is None:
        raise ConfigError("this CLI builds the LoRA layout only; train_mode 'full' needs the FSDP2 layout (A100 task)")
    if cfg.rollout.backend != "hf":
        raise ConfigError("this CLI builds the HF rollout engine only; backend 'vllm' needs the A100 task")
    if cfg.model.precision == "fp16" and device.type != "cuda":
        raise ConfigError("precision fp16 needs CUDA (GradScaler); use fp32 on CPU")
    return cfg.lora


def _sweep_stale_tmp(run_dir: Path) -> None:
    """Remove ``.tmp-*`` leftovers of crashed checkpoint writes (rank 0, before any save)."""
    for path in sorted(run_dir.glob(f"{_TMP_PREFIX}*")):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        logger.info("removed stale checkpoint temp %s", path)


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git from PATH is intended
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("git SHA unavailable; W&B config records null", exc_info=True)
        return None
    return proc.stdout.strip()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def _rank0_run(
    cfg: Config, args: _Args, dist: DistInfo, resume_id: str | None, extra: dict[str, object]
) -> Iterator[wandb.Run | None]:
    if dist.rank != 0:
        yield None
        return
    with open_run(
        cfg, run_name=args.run_name, reward_mode=args.reward_mode, resume_id=resume_id, extra_config=extra
    ) as run:
        yield run


def _alert(run: wandb.Run | None, title: str, text: str, level: Literal["INFO", "WARN", "ERROR"]) -> None:
    if run is not None:
        run.alert(title=title, text=text, level=level)


def _broadcast_flag(flag: bool, dist: DistInfo) -> bool:
    """Rank 0's decision, on every rank (each rank's clock could disagree at the boundary)."""
    if dist.world_size == 1:
        return flag
    tensor = torch.tensor([int(flag)], device=dist.device)
    tdist.broadcast(tensor, src=0)
    return bool(tensor.item())


# ---------------------------------------------------------------------------- the run
@dataclass
class _Budget:
    deadline_unix: float | None
    reserve_s: float
    ckpt_every: int
    step_seconds: list[float]

    def mean_chunk_s(self) -> float:
        if not self.step_seconds:
            return 0.0
        return sum(self.step_seconds) / len(self.step_seconds) * self.ckpt_every

    def stop(self) -> bool:
        if self.deadline_unix is None:
            return False
        return should_stop(
            now_unix=time.time(),
            deadline_unix=self.deadline_unix,
            mean_chunk_s=self.mean_chunk_s(),
            reserve_s=self.reserve_s,
        )

    def projected_steps(self, steps_done: int) -> float:
        if self.deadline_unix is None or not self.step_seconds:
            raise ValueError("projection needs a deadline and at least one timed step")
        mean_step = sum(self.step_seconds) / len(self.step_seconds)
        return steps_done + max(self.deadline_unix - self.reserve_s - time.time(), 0.0) / mean_step


def _train(args: _Args, cfg: Config, dist: DistInfo, run_dir: Path) -> int:
    lora = check_layout(cfg, dist.device)
    validate_for_runtime(cfg, device_capability=_device_capability(dist.device), world_size=dist.world_size)
    check_cuda_arch(dist.device)
    check_java()
    index_checksum = check_index(cfg)
    if dist.rank == 0:
        _sweep_stale_tmp(run_dir)
    if dist.world_size > 1:
        tdist.barrier()

    data_dir = cfg.data.data_dir
    train = load_examples(data_dir / TRAIN_FILE)
    val, _ = split_dev(load_examples(data_dir / DEV_FILE), val_size=cfg.data.val_size)
    _seed(cfg.run.seed + dist.rank)

    base, tokenizer = build_base(cfg, dist.device)
    check_tokenizer(tokenizer)
    models = LoraPolicyModels(base, lora, gradient_checkpointing=cfg.model.gradient_checkpointing)
    engine = HFGenerateEngine(models, tokenizer, cfg.rollout, dist.device)
    r = cfg.retrieval
    searcher = Bm25Searcher(r.index_dir, k1=r.k1, b=r.b, threads=r.search_threads, timeout_s=r.search_timeout_s)
    scaler = GradScaler("cuda") if cfg.model.precision == "fp16" else None
    trainer = PPOTrainer(
        cfg, models=models, engine=engine, searcher=searcher, reward_mode=args.reward_mode, dist=dist, scaler=scaler
    )

    state = TrainState(step=0, data_cursor=0, wandb_run_id=None, reward_mode=args.reward_mode)
    last_saved: int | None = None
    ckpt_path = latest_checkpoint(run_dir)
    if ckpt_path is not None:
        ckpt = load_checkpoint(ckpt_path)
        if ckpt.train_state.reward_mode != args.reward_mode:
            raise ValueError(
                f"{ckpt_path} was trained with reward_mode {ckpt.train_state.reward_mode!r}, not {args.reward_mode!r}"
            )
        models.load_trainable_state_dict(ckpt.model)
        trainer.load_optimizer_states(ckpt.optimizers)
        if scaler is not None:
            if ckpt.scaler is None:
                raise ValueError(f"{ckpt_path} has no GradScaler state but this run uses fp16")
            scaler.load_state_dict(ckpt.scaler)
        if dist.rank == 0:
            restore_rng(ckpt.rng)  # saved from rank 0; other ranks reseed from the step below
        else:
            _seed(cfg.run.seed + dist.rank + 1000 * ckpt.train_state.step)
        state = ckpt.train_state
        last_saved = state.step
        logger.info("resumed from %s at step %d (cursor %d)", ckpt_path, state.step, state.data_cursor)

    until = cfg.run.total_steps if args.until_step is None else min(args.until_step, cfg.run.total_steps)
    local = cfg.ppo.batch_size // dist.world_size
    budget = _Budget(args.deadline_unix, args.reserve_s, cfg.run.ckpt_every, [])
    extra: dict[str, object] = {
        "git_sha": _git_sha(),
        "index_checksum": index_checksum,
        "dataset_revision": cfg.data.hf_revision,
        "world_size": dist.world_size,
        "reward_mode": args.reward_mode,
    }

    def save() -> None:
        nonlocal last_saved
        if dist.rank == 0:
            path = save_checkpoint(
                run_dir,
                train_state=state,
                model=models.trainable_state_dict(),
                optimizers=trainer.optimizer_states(),
                scaler=None if scaler is None else scaler.state_dict(),
                keep=cfg.run.keep_ckpts,
            )
            logger.info("checkpoint %s", path)
        last_saved = state.step

    with _rank0_run(cfg, args, dist, state.wandb_run_id, extra) as run:
        if run is not None:
            state = TrainState(state.step, state.data_cursor, run.id, state.reward_mode)
        exit_code = EXIT_BUDGET if _broadcast_flag(budget.stop(), dist) else EXIT_OK
        try:
            while exit_code == EXIT_OK and state.step < until:
                step = state.step
                bind_log_context(step=step)
                indices = batch_indices(state.data_cursor, cfg.ppo.batch_size, len(train), cfg.run.seed)
                logger.info("batch", extra={"batch_indices": indices})
                started = time.perf_counter()
                metrics = trainer.step(
                    [train[i] for i in indices[dist.rank * local : (dist.rank + 1) * local]], step=step
                )
                state = TrainState(step + 1, state.data_cursor + cfg.ppo.batch_size, state.wandb_run_id, state.reward_mode)
                if len(budget.step_seconds) < CALIBRATION_STEPS:
                    # Running estimate on each of the first CALIBRATION_STEPS steps; logged once at the end.
                    budget.step_seconds.append(time.perf_counter() - started)
                    if args.deadline_unix is not None:
                        metrics["budget/projected_steps"] = budget.projected_steps(state.step)
                        if len(budget.step_seconds) == CALIBRATION_STEPS:
                            logger.info("projected steps within budget: %.0f", metrics["budget/projected_steps"])
                if dist.rank == 0:
                    _log_step(run, run_dir, step, metrics, trainer.histograms())
                    if state.step % cfg.run.val_every == 0:
                        _validate(run, trainer, val, step)
                if state.step % cfg.run.ckpt_every == 0:
                    save()
                    if _broadcast_flag(budget.stop(), dist):
                        exit_code = EXIT_BUDGET
        except (NonFiniteError, TrainingHealthError) as exc:
            _alert(run, f"{args.run_name}: training aborted", f"{type(exc).__name__}: {exc}", "ERROR")
            raise
        if last_saved != state.step:
            save()
        if exit_code == EXIT_BUDGET:
            logger.warning("budget deadline reached; stopping at step %d", state.step)
            _alert(run, f"{args.run_name}: budget stop", f"stopped at step {state.step}", "WARN")
        else:
            logger.info("reached step %d", state.step)
            if run is not None and state.step >= cfg.run.total_steps:
                adapters = run_dir / "final_adapters.safetensors"
                save_file(
                    {k: v.detach().cpu().contiguous() for k, v in models.trainable_state_dict().items()}, adapters
                )
                log_checkpoint(run, adapters, name=f"{args.run_name}-adapters", artifact_type="model")
            _alert(run, f"{args.run_name}: finished", f"reached step {state.step}", "INFO")
    return exit_code


def _log_step(
    run: wandb.Run | None,
    run_dir: Path,
    step: int,
    metrics: dict[str, float],
    histograms: dict[str, list[float]],
) -> None:
    """Append the scalar row to metrics.jsonl; send scalars and (deep-log steps) histograms to W&B."""
    # loss / grad_norm are the column names scripts/spike_monitor.py analyses; histograms stay out
    # of metrics.jsonl, which must hold numbers only.
    row = {"step": step, "loss": metrics["ppo/policy_loss"], "grad_norm": metrics["optim/actor_grad_norm"], **metrics}
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    if run is not None:
        hists = {key: wandb.Histogram(values) for key, values in histograms.items()}
        run.log({**metrics, **hists, STEP_METRIC: step})


def _validate(run: wandb.Run | None, trainer: PPOTrainer, val: Sequence[Example], step: int) -> None:
    metrics, samples = trainer.validate_with_samples(val)
    logger.info(
        "validation ndcg normal=%.4f count_once=%.4f", metrics["val/ndcg_normal"], metrics["val/ndcg_count_once"]
    )
    if run is None:
        return
    table = wandb.Table(columns=["query", "rewrite", "analyzed_terms", "reward_normal", "reward_count_once"])
    for s in samples:
        table.add_data(s.query, s.rewrite, s.analyzed_terms, s.reward_normal, s.reward_count_once)
    run.log({**metrics, "samples": table, STEP_METRIC: step})


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dist = init_distributed()
    code = EXIT_ERROR
    run_dir: Path | None = None
    try:
        cfg = load_config(args.config, args.overrides)
        run_dir = cfg.run.output_dir / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if dist.rank == 0:
            # Until the real code is written at the end, a killed process (OOM, Slurm limit,
            # torchrun teardown) must not leave the previous chunk's 0 or 3 behind.
            (run_dir / "exit_code").write_text(f"{EXIT_ERROR}\n")
        configure(level="INFO", fmt="json", log_file=str(run_dir / f"train_rank{dist.rank}.log"))
        bind_log_context(run_name=args.run_name, reward_mode=args.reward_mode, rank=dist.rank)
        code = _train(args, cfg, dist, run_dir)
    except Exception:
        # Entry point: every failure becomes exit code 1, with its traceback in the log.
        logger.exception("training failed")
        code = EXIT_ERROR
    finally:
        if run_dir is not None and dist.rank == 0:
            (run_dir / "exit_code").write_text(f"{code}\n")
        if tdist.is_initialized():
            tdist.destroy_process_group()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
