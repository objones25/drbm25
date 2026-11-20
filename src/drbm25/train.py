"""Resumable PPO training CLI.

    python -m drbm25.train --config PATH --reward-mode {normal,count_once} --run-name NAME
        [--until-step N] [--deadline-unix T] [--reserve-s S] [--set key=value ...]

Exit codes: 0 reached ``min(until_step, total_steps)``; 3 stopped by the budget deadline;
1 any error. torchrun turns any non-zero worker exit into its own exit code 1, so rank 0 also
writes the code to ``output_dir/run_name/exit_code``; launchers read that file to tell a
budget stop (3) from a failure. The file holds 0 done, 3 budget stop, 2 ended by an exception,
and 1 when the run started and never finished (killed by OOM, a Slurm time limit, teardown).

Run directory ``output_dir/run_name/``: ``step_NNNNNN/`` resume checkpoints (``run.keep_ckpts``
of them), ``best/`` (the policy with the highest validation score so far, see below),
``run_manifest.json`` (every seed, the resolved config and its sha256, versions; see
``drbm25.manifest``), ``metrics.jsonl`` (rank 0, one JSON object of floats per step, read by
``scripts/spike_monitor.py``), ``train_rank{R}.log`` (JSON log lines per rank, including each
step's global batch indices), ``exit_code``.

Best policy: every ``run.val_every`` steps and at ``run.total_steps`` the actor is validated
(greedy, nDCG@10 under both modes) on ``data.val_size`` train queries held out of training. When
``val/ndcg_<reward_mode>`` strictly beats the stored best, ``best/`` is replaced atomically (ties
keep the earlier step). A resume restores the best-so-far score from ``best/meta.json``.

Layouts (``model.train_mode``): ``lora`` (V100) -- LoRA actor/critic adapters on one frozen
base, checkpoints written by rank 0; ``full`` (A100) -- FSDP2-sharded fp32 actor and critic plus
a frozen reference (``FullPolicyModels``), sharded DCP checkpoints written by every rank; needs
torchrun and precision bf16 or fp32. Rollout engines (``rollout.backend``): ``hf`` -- transformers
``generate`` on the live actor (``synced_gpus`` on the full layout, for CPU runs); ``vllm`` -- one
colocated vLLM engine per rank (CUDA and a real model only), built before the training models.
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
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, get_args

import numpy as np
import torch
import torch.distributed as tdist
import wandb
from safetensors.torch import save_file
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from drbm25.bm25.index import read_index_info
from drbm25.bm25.searcher import Bm25Searcher
from drbm25.budget import should_stop
from drbm25.checkpoint import (
    BestMeta,
    LoadedCheckpoint,
    TrainState,
    best_meta,
    latest_checkpoint,
    load_checkpoint,
    load_sharded_checkpoint,
    recover_best,
    restore_rng,
    save_best,
    save_checkpoint,
    save_sharded_checkpoint,
)
from drbm25.config import (
    Config,
    ConfigError,
    LoraCfg,
    ScoringMode,
    dist_timeout,
    load_config,
    validate_for_runtime,
)
from drbm25.data import TRAIN_FILE, Example, batch_indices, load_examples, split_train
from drbm25.distributed import DistInfo, assert_params_equal, broadcast_params, init_distributed
from drbm25.manifest import (
    build_manifest,
    check_seeds_unchanged,
    collect_environment,
    config_sha256,
    read_manifest,
    seeds,
    update_manifest,
    write_manifest,
)
from drbm25.obs import configure
from drbm25.obs.context import bind_log_context
from drbm25.obs.runs import STEP_METRIC, open_run
from drbm25.obs.wandb_run import log_checkpoint
from drbm25.ppo.models import LoraPolicyModels, PolicyModels
from drbm25.ppo.models_full import FullPolicyModels
from drbm25.ppo.trainer import NonFiniteError, PPOTrainer, TrainingHealthError
from drbm25.rollout.base import RolloutEngine
from drbm25.rollout.hf_generate import HFGenerateEngine
from drbm25.rollout.vllm_engine import VLLMDtype, VLLMEngine
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer

logger = logging.getLogger(__name__)

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
EXIT_BUDGET: Final = 3
# Only ever written to the exit_code file, never returned: the process still exits 1 on an exception.
# The file reads 1 for a run that started and never finished (killed), 2 for one that raised.
EXIT_CODE_FILE_FAILED: Final = 2
TINY_MODEL: Final = "tiny-local"
REQUIRED_JAVA_MAJOR: Final = 21
CALIBRATION_STEPS: Final = 10
_JAVA_TIMEOUT_S: Final = 30.0
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_JAVA_VERSION = re.compile(r'version "(\d+)')
_TMP_PREFIX = ".tmp-"
_DTYPES: Final = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_VLLM_DTYPES: Final[dict[str, VLLMDtype]] = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


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
def precision_dtype(cfg: Config) -> torch.dtype:
    return _DTYPES[cfg.model.precision]


def build_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    if cfg.model.name == TINY_MODEL:
        return build_tiny_tokenizer()
    return AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision, padding_side="left")


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
    return base, build_tokenizer(cfg)


def build_full_models(cfg: Config, dist: DistInfo) -> FullPolicyModels:
    """Actor and critic in fp32, reference in the compute dtype, loaded on CPU and sharded onto ``dist.device``.

    ``fully_shard`` moves each unit to the device as it shards it, so the GPU never holds a full
    fp32 copy; host memory holds all three models briefly (about 31 GB per rank at 3B).
    """
    compute = _DTYPES[cfg.model.precision]
    if cfg.model.name == TINY_MODEL:
        actor, critic, ref = (build_tiny_qwen2(seed=cfg.run.seed) for _ in range(3))
        backbone: PreTrainedModel = critic.model
    else:
        common = {"revision": cfg.model.revision, "attn_implementation": "sdpa"}
        actor = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32, **common)
        backbone = AutoModel.from_pretrained(cfg.model.name, dtype=torch.float32, **common)
        ref = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=compute, **common)
    mesh = init_device_mesh(dist.device.type, (dist.world_size,))
    return FullPolicyModels(
        actor,
        backbone,
        ref,
        mesh=mesh,
        compute_dtype=compute,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )


def check_layout(cfg: Config, device: torch.device) -> LoraCfg:
    """The LoRA layout's requirements; returns its LoRA settings (also used by evaluation)."""
    if cfg.model.train_mode != "lora" or cfg.lora is None:
        raise ConfigError("this builds the LoRA layout only; train_mode 'full' checkpoints need the FSDP2 layout")
    if cfg.model.precision == "fp16" and device.type != "cuda":
        raise ConfigError("precision fp16 needs CUDA (GradScaler); use fp32 on CPU")
    return cfg.lora


def check_full_layout(cfg: Config) -> None:
    if cfg.model.precision == "fp16":
        raise ConfigError("train_mode 'full' supports precision bf16 or fp32; fp16 would need a sharded GradScaler")
    if not tdist.is_initialized():
        raise ConfigError("train_mode 'full' shards with FSDP2 and needs a process group: launch with torchrun")


def check_backend(cfg: Config, device: torch.device) -> None:
    if cfg.rollout.backend != "vllm":
        return
    if device.type != "cuda":
        raise ConfigError("rollout.backend 'vllm' needs CUDA; use backend 'hf' on CPU")
    if cfg.model.name == TINY_MODEL:
        raise ConfigError(f"rollout.backend 'vllm' needs a Hugging Face model, not {TINY_MODEL!r}")


def _sweep_stale_tmp(run_dir: Path) -> None:
    """Remove ``.tmp-*`` leftovers of crashed checkpoint writes (rank 0, before any save)."""
    for path in sorted(run_dir.glob(f"{_TMP_PREFIX}*")):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        logger.info("removed stale checkpoint temp %s", path)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def _rank0_run(
    cfg: Config, args: _Args, dist: DistInfo, resume_id: str | None, extra: dict[str, object]
) -> Generator[wandb.Run | None]:
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


@dataclass(frozen=True)
class Layout:
    models: PolicyModels
    engine: RolloutEngine
    full: FullPolicyModels | None  # set on the full layout (sharded checkpoints)


def init_layout(cfg: Config, dist: DistInfo) -> Layout:
    """Build the layout with the SAME seed on every rank, then seed each rank differently for sampling.

    LoRA ``A`` matrices are initialised from the global RNG; built after a per-rank seed they would
    differ across ranks, and averaging gradients over different weights makes the ranks drift.
    """
    _seed(cfg.run.seed)
    layout = build_layout(cfg, dist)
    sync_replicated_params(layout, dist)
    _seed(cfg.run.seed + dist.rank)
    return layout


def sync_replicated_params(layout: Layout, dist: DistInfo) -> None:
    """LoRA layout: broadcast rank 0's trainable params and assert all ranks match (a collective).

    Belt and braces after construction and after a resume load. The full layout is skipped: FSDP2
    parameters are shards of a full copy that is built identically on every rank.
    """
    if layout.full is not None:
        return
    params = [*layout.models.actor_parameters(), *layout.models.critic_parameters()]
    broadcast_params(params, dist)
    assert_params_equal(params, dist)


def build_layout(cfg: Config, dist: DistInfo) -> Layout:
    full = cfg.model.train_mode == "full"
    tokenizer = build_tokenizer(cfg)
    check_tokenizer(tokenizer)
    vllm: VLLMEngine | None = None
    if cfg.rollout.backend == "vllm":
        # First: vLLM checks that gpu_memory_utilization x total memory is still free.
        vllm = VLLMEngine.create(
            cfg.model.name,
            cfg.model.revision,
            cfg.rollout,
            dist,
            tokenizer=tokenizer,
            dtype=_VLLM_DTYPES[cfg.model.precision],
            seed=cfg.run.seed + dist.rank,
        )
    if full:
        full_models = build_full_models(cfg, dist)
        models: PolicyModels = full_models
    else:
        full_models = None
        base, _ = build_base(cfg, dist.device)
        lora = check_layout(cfg, dist.device)
        models = LoraPolicyModels(base, lora, gradient_checkpointing=cfg.model.gradient_checkpointing)
    engine: RolloutEngine = (
        vllm if vllm is not None else HFGenerateEngine(models, tokenizer, cfg.rollout, dist.device, synced_gpus=full)
    )
    return Layout(models=models, engine=engine, full=full_models)


def _train(args: _Args, cfg: Config, dist: DistInfo, run_dir: Path) -> int:
    if cfg.model.train_mode == "full":
        check_full_layout(cfg)
    else:
        check_layout(cfg, dist.device)
    check_backend(cfg, dist.device)
    validate_for_runtime(cfg, device_capability=_device_capability(dist.device), world_size=dist.world_size)
    check_cuda_arch(dist.device)
    check_java()
    index_checksum = check_index(cfg)
    previous_manifest = read_manifest(run_dir)
    check_seeds_unchanged(previous_manifest, seeds(cfg, world_size=dist.world_size))  # every rank: fail together
    if dist.rank == 0:
        if recover_best(run_dir):  # before the sweep, which would delete the moved-aside best
            logger.warning("restored best/ that a crash had left moved aside")
        _sweep_stale_tmp(run_dir)
    if dist.world_size > 1:
        tdist.barrier()

    data_dir = cfg.data.data_dir
    # Validation (best-checkpoint selection) uses train queries held out of training; dev is the test set only.
    train, val = split_train(load_examples(data_dir / TRAIN_FILE), val_size=cfg.data.val_size, seed=cfg.data.split_seed)
    layout = init_layout(cfg, dist)  # seeds: shared for construction, then seed + rank
    models, engine = layout.models, layout.engine
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
        ckpt = _load_resume_state(ckpt_path, layout, trainer)
        if ckpt.train_state.reward_mode != args.reward_mode:
            raise ValueError(
                f"{ckpt_path} was trained with reward_mode {ckpt.train_state.reward_mode!r}, not {args.reward_mode!r}"
            )
        if scaler is not None:
            if ckpt.scaler is None:
                raise ValueError(f"{ckpt_path} has no GradScaler state but this run uses fp16")
            scaler.load_state_dict(ckpt.scaler)
        sync_replicated_params(layout, dist)
        if dist.rank == 0:
            restore_rng(ckpt.rng)  # saved from rank 0; other ranks reseed from the step below
        else:
            _seed(cfg.run.seed + dist.rank + 1000 * ckpt.train_state.step)
        state = ckpt.train_state
        last_saved = state.step
        logger.info("resumed from %s at step %d (cursor %d)", ckpt_path, state.step, state.data_cursor)
    best = best_meta(run_dir)
    if best is not None:
        if best.reward_mode != args.reward_mode:
            raise ValueError(f"{run_dir}/best was selected for reward_mode {best.reward_mode!r}, not {args.reward_mode!r}")
        logger.info("best so far: step %d, %s = %.4f", best.step, best.selection_metric, best.score)

    # vLLM starts from the checkpoint on disk; after a resume the actor differs. A no-op for HF.
    engine.sync_weights(models)

    until = cfg.run.total_steps if args.until_step is None else min(args.until_step, cfg.run.total_steps)
    local = cfg.ppo.batch_size // dist.world_size
    budget = _Budget(args.deadline_unix, args.reserve_s, cfg.run.ckpt_every, [])
    env = collect_environment()
    selection = _Selection(
        run_dir=run_dir,
        reward_mode=args.reward_mode,
        config_sha256=config_sha256(cfg),
        seeds=seeds(cfg, world_size=dist.world_size),
        git_sha=env.git_sha,
        git_dirty=env.git_dirty,
        best=best,
    )
    extra: dict[str, object] = {}
    if dist.rank == 0:
        manifest = build_manifest(
            cfg,
            run_name=args.run_name,
            reward_mode=args.reward_mode,
            world_size=dist.world_size,
            index_checksum=index_checksum,
            env=env,
            started_utc=datetime.now(UTC),
        )
        manifest = update_manifest(previous_manifest, manifest, step=state.step)
        write_manifest(run_dir, manifest)
        extra = manifest  # the same dict goes to W&B config["meta"]

    def save() -> None:
        nonlocal last_saved
        if layout.full is not None:
            path = save_sharded_checkpoint(
                run_dir,
                train_state=state,
                state=layout.full.sharded_state(trainer.optimizers()),
                keep=cfg.run.keep_ckpts,
                dist=dist,
            )
            if dist.rank == 0:
                logger.info("checkpoint %s", path)
        elif dist.rank == 0:
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
                if state.step % cfg.run.val_every == 0 or state.step == cfg.run.total_steps:
                    # Every rank validates (a collective: LoRA shards the queries, FSDP needs every rank
                    # in each forward) and gets the same scores; run is None off rank 0, so only rank 0 logs.
                    scores = _validate(run, trainer, val, step)
                    selection.update(scores, step=state.step, layout=layout, dist=dist)
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
            if layout.full is not None and state.step >= cfg.run.total_steps:
                logger.info(
                    "full fine-tune: the final sharded checkpoint step_%06d is the trained model "
                    "(score it with drbm25.eval.run_eval --source checkpoint; weights stay on disk, no W&B upload)",
                    state.step,
                )
            elif run is not None and state.step >= cfg.run.total_steps:
                adapters = run_dir / "final_adapters.safetensors"
                save_file(
                    {k: v.detach().cpu().contiguous() for k, v in models.trainable_state_dict().items()}, adapters
                )
                log_checkpoint(run, adapters, name=f"{args.run_name}-adapters", artifact_type="model")
            _alert(run, f"{args.run_name}: finished", f"reached step {state.step}", "INFO")
    return exit_code


@dataclass
class _Selection:
    """Best-policy tracking on the validation split; every rank tracks the same ``best``."""

    run_dir: Path
    reward_mode: ScoringMode
    config_sha256: str
    seeds: dict[str, object]
    git_sha: str | None
    git_dirty: bool | None
    best: BestMeta | None

    def update(self, scores: dict[str, float], *, step: int, layout: Layout, dist: DistInfo) -> None:
        """Replace ``best/`` when the run's metric strictly beats the stored best; a collective on every rank.

        ``scores`` (``val/ndcg_normal``, ``val/ndcg_count_once``) are the same on every rank, so each
        rank reaches the same decision without a broadcast; ``_check_same_on_all_ranks`` proves it.
        Rank 0 writes; on the full layout every rank joins the gather of the actor.
        """
        candidate = BestMeta(
            step=step,
            val_ndcg_normal=scores["val/ndcg_normal"],
            val_ndcg_count_once=scores["val/ndcg_count_once"],
            reward_mode=self.reward_mode,
            layout="full" if layout.full is not None else "lora",
            config_sha256=self.config_sha256,
            seeds=self.seeds,
            git_sha=self.git_sha,
            git_dirty=self.git_dirty,
        )
        improved = self.best is None or candidate.score > self.best.score
        _check_same_on_all_ranks([candidate.score, float(improved)], dist)
        if not improved:
            return
        weights: dict[str, torch.Tensor] = {}
        if layout.full is not None:
            # A collective; to host memory so the GPU holds one gathered tensor at a time.
            weights = layout.full.merged_actor_state_dict(device=torch.device("cpu"))
        elif dist.rank == 0:
            if not isinstance(layout.models, LoraPolicyModels):
                raise TypeError(f"the lora layout needs LoraPolicyModels, got {type(layout.models).__name__}")
            weights = layout.models.actor_state_dict()
        if dist.rank == 0:
            save_best(self.run_dir, weights=weights, meta=candidate)
            logger.info("new best: step %d, %s = %.4f", step, candidate.selection_metric, candidate.score)
        self.best = candidate
        if dist.world_size > 1:
            tdist.barrier()


def _check_same_on_all_ranks(values: list[float], dist: DistInfo) -> None:
    """Raise RuntimeError on every rank unless all ranks hold the same ``values``; a collective."""
    if dist.world_size == 1:
        return
    high = torch.tensor(values, dtype=torch.float64, device=dist.device)
    low = high.clone()
    tdist.all_reduce(high, op=tdist.ReduceOp.MAX)
    tdist.all_reduce(low, op=tdist.ReduceOp.MIN)
    if not torch.equal(high, low):
        raise RuntimeError(f"ranks disagree on the best-policy decision: max {high.tolist()} vs min {low.tolist()}")


def _load_resume_state(path: Path, layout: Layout, trainer: PPOTrainer) -> LoadedCheckpoint:
    """Load model and optimizer state from ``path`` into the live objects (every rank)."""
    if layout.full is None:
        ckpt = load_checkpoint(path)
        layout.models.load_trainable_state_dict(ckpt.model)
        trainer.load_optimizer_states(ckpt.optimizers)
        return ckpt
    template = layout.full.sharded_state(trainer.optimizers())
    ckpt = load_sharded_checkpoint(path, template)
    layout.full.load_sharded_state(template, trainer.optimizers())
    return ckpt


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


def _validate(run: wandb.Run | None, trainer: PPOTrainer, val: Sequence[Example], step: int) -> dict[str, float]:
    """Greedy validation under both modes; logs to W&B on rank 0 and returns the ``val/*`` metrics."""
    metrics, samples = trainer.validate_with_samples(val)
    logger.info(
        "validation ndcg normal=%.4f count_once=%.4f", metrics["val/ndcg_normal"], metrics["val/ndcg_count_once"]
    )
    if run is None:
        return metrics
    table = wandb.Table(columns=["query", "rewrite", "analyzed_terms", "reward_normal", "reward_count_once"])
    for s in samples:
        table.add_data(s.query, s.rewrite, s.analyzed_terms, s.reward_normal, s.reward_count_once)
    run.log({**metrics, "samples": table, STEP_METRIC: step})
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    code = EXIT_ERROR
    file_code = EXIT_ERROR  # what the exit_code file gets; stays 1 if the process dies before finally
    run_dir: Path | None = None
    dist: DistInfo | None = None
    try:
        # The config comes first: it sets the process group's collective timeout. A config error
        # therefore fails on every rank before any process group exists, and no exit_code is written.
        cfg = load_config(args.config, args.overrides)
        dist = init_distributed(timeout=dist_timeout(cfg))
        run_dir = cfg.run.output_dir / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if dist.rank == 0:
            # Until the real code is written at the end, a killed process (OOM, Slurm limit,
            # torchrun teardown) must not leave the previous chunk's 0 or 3 behind.
            (run_dir / "exit_code").write_text(f"{EXIT_ERROR}\n")
        configure(level="INFO", fmt="json", log_file=str(run_dir / f"train_rank{dist.rank}.log"))
        bind_log_context(run_name=args.run_name, reward_mode=args.reward_mode, rank=dist.rank)
        code = _train(args, cfg, dist, run_dir)
        file_code = code
    except Exception:
        # Entry point: every failure becomes exit code 1, with its traceback in the log.
        logger.exception("training failed")
        code = EXIT_ERROR
        file_code = EXIT_CODE_FILE_FAILED
    finally:
        if run_dir is not None and dist is not None and dist.rank == 0:
            (run_dir / "exit_code").write_text(f"{file_code}\n")
        if tdist.is_initialized():
            tdist.destroy_process_group()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
